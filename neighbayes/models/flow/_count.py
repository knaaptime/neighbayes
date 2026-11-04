r"""The count flow families: negative binomial, and Poisson.

Each is one class (see
:class:`~neighbayes.models._mixins._designs.FlowCountDesigns`): ``separable=True``
selects the separable filter and ``T`` a panel, the cross-section being one
period.  The filter acts on the log-mean,

.. math::

    \eta_t = A(\rho)^{-1}X_t\beta + c + \tau_t,

so the count likelihood carries no ``|A|`` Jacobian.  The SAR negative
binomial's effects are levels after the filter on its separable form: fixed
pair effects ``c`` (``pair_effects="fixed"``), origin and destination levels
``γ_i + κ_j`` (``origin_effects``, ``destination_effects``) and period effects
``τ``.  The aspatial negative binomial's pair effects are random and partially
pooled.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._backends.sampler_helpers import select_backend
from ..._lazy_deps import pm, xr
from ..._ops import kron_solve_matrix
from .._base._nb import nb_alpha_rv
from .._mixins._count_panel import CountPanelFEMixin
from .._mixins._designs import FlowCountDesigns
from ._flow import FlowModel

_POISSON_DISPERSION_WARN = 2.0

_COUNT_FE_DOC = """
    Effects
    -------
    A panel (``T``) takes ``pair_effects="random"`` and ``time_effects=True``.
    The within transform of the Gaussian panels does not carry to a log link,
    so the effects are parameters, ``η_t = X_tβ + C + τ_t``: pair effects (one
    per origin–destination pair), drawn jointly with β and held as a pair
    index rather than ``n²`` dummy columns.

    Pair effects are **partially pooled**: ``C_ij ~ N(μ, σ²)`` with σ learned
    (half-t(3, ``pair_effect_sd_scale``) prior, default scale 1).  A pair seen
    in few periods is shrunk toward the common mean in proportion to how little
    its data say.  Pair effects absorb nothing: the intercept and
    time-invariant columns (intra indicator, log distance) stay in the design
    and anchor the level.  ``mundlak=True`` adds the pair means of the
    time-varying columns, absorbing a correlation between the pair effects and
    ``X``.

    Period effects join the coefficient block under a fixed prior
    (``time_effect_mu``, default 0 beside pair effects and ``log mean y``
    without; ``time_effect_sigma``, default 2.5): ``T − 1`` of them (first
    period the baseline) beside pair effects, all ``T`` without (the intercept
    is then absorbed).  ``alpha_fixed`` holds α at a value; a large one (10-20
    times the typical mean count) makes the model an essentially Poisson one
    that keeps the exact Pólya–Gamma sampler (see
    :mod:`neighbayes.models._base._nb`).

    ``fit(store_effects=None)`` keeps every draw of the pair effects when
    they fit in about 500 MB and otherwise only their posterior mean and sd
    (``idata["pair_effect_summary"]``); ``posterior_predictive`` needs the
    draws.  Effects run on the NumPy Gibbs backend.
"""

_SAR_NB_EFFECTS_DOC = r"""
    Effects
    -------
    The within transform of the Gaussian panels does not carry to a log link,
    so the effects are parameters, added to the log-mean after the filter and
    sampled with the exact likelihood (``separable=True``):

    .. math::

        \eta_{t,ij} = (A^{-1}X_t\beta)_{ij} + c_{ij} + \tau_t
        \quad\text{or}\quad
        (A^{-1}X_t\beta)_{ij} + \gamma_i + \kappa_j + \tau_t.

    ``pair_effects="fixed"`` (a panel) gives each origin–destination pair a
    level under a flat prior: the fixed-effects static panel, identified by
    the within-pair variation alone.  Under a flat prior a level inside the
    filter is a reparameterization of one after it, ``A^{-1}c`` ranging over
    every array as ``c`` does.  The pair levels absorb every time-invariant
    column (the intercept, log distance, the intra indicator, time-invariant
    attributes), so ``ρ`` is identified by the time-varying columns.  A pair
    with no count in any (live) period has no proper posterior and leaves
    the likelihood; its level is NaN and it predicts zero counts.  The
    posterior integrates each level out rather than profiling it, and the
    NB2 log-mean is information-orthogonal to α, so the posterior of α
    escapes the leading, order-``1/T`` incidental-parameter bias of the
    dummy-variable estimator (Lancaster 2002).  Pair effects are fixed only:
    random ones inside the filter tie ρ to the spatial pattern of the pair
    levels, which moved ρ by ten posterior sd in simulation.

    ``origin_effects`` and ``destination_effects`` (``"fixed"``, or
    ``"random"``; cross-section or panel) add a level per origin and per
    destination: the production-, attraction- or doubly constrained gravity
    model with balancing factors (Griffith & Fischer 2016).  Fixed levels take
    a flat prior, the first set carrying the overall level and a later one
    holding its first level at 0.  Random levels are ``N(μ, τ²)`` with ``τ``
    learned under a half-t(3, ``*_effect_sd_scale``) prior (default 1) and
    ``*_effect_mu`` defaulting to ``log mean(y)`` shared between the random
    levels present.  For a row-standardized ``W`` the filter maps a column
    constant along a row (an origin attribute, the intercept) to one again,
    so the origin levels absorb the intercept and the time-invariant origin
    attributes, and the destination levels the destination attributes.  One
    set of levels leaves the other side's ρ well identified; with both, ρ is
    identified weakly.

    ``time_effects=True`` adds period levels ``N(time_effect_mu,
    time_effect_sigma²)`` (``time_effect_sigma`` 2.5): beside other effects
    the first period is the baseline and ``time_effect_mu`` defaults to 0;
    alone, every period has a level, centred on ``log mean(y)``, and the
    intercept is absorbed.  Absorbed columns are dropped from the design and
    their impacts come back as NaN.  The posterior holds ``pair_effect``
    (dims origin, destination), ``origin_effect``, ``destination_effect``,
    ``time_effect`` and the random sets' ``*_sd``.  ``fit(store_effects=None)``
    keeps every draw of the pair effects when they fit in about 500 MB and
    otherwise only their posterior mean and sd
    (``idata["pair_effect_summary"]``); ``posterior_predictive`` needs the
    draws.  NUTS takes the random origin and destination effects and the
    period effects, not the fixed ones.

    The unrestricted panel takes ``time_effects=True`` only, its period
    effects joining the coefficient block of its Pólya–Gamma sampler (NumPy).
    ``alpha_fixed`` holds α at a value; a large one (10-20 times the typical
    mean count) makes the model an essentially Poisson one (see
    :mod:`neighbayes.models._base._nb`).
"""


def _absorbed_by_levels(X, n: int, T: int, kinds) -> np.ndarray:
    """The columns of a time-first flow design in the span of the level sets.

    Each kind of level (``"origin"``, ``"destination"``, ``"pair"``,
    ``"period"``) is constant over some axes of the ``T × n × n`` flow arrays;
    a column is absorbed when sweeping out each set's means in turn leaves no
    residual (exact on the balanced flow array).
    """
    from ...samplers.negbin_reduced._flow_exact import _LEVEL_AXES

    B = np.asarray(X, dtype=np.float64).T.reshape(-1, T, n, n)
    rest = B
    for kind in kinds:
        rest = rest - rest.mean(axis=_LEVEL_AXES[kind], keepdims=True)
    tol = 1e-10 * np.maximum(np.abs(B).max(axis=(1, 2, 3)), 1.0)
    return np.abs(rest).max(axis=(1, 2, 3)) <= tol


class _FlowCount(CountPanelFEMixin):
    """Count flows: pair and period effects, the filtered log-mean, prediction.

    What depends on the filter (``_flow_filter``) branches on it: the
    log-mean ``η``, the Gibbs filter, the PyMC model and the effects.
    """

    _effect_index = "pair"
    _robust_nuts = False

    def _attach_log_abs_det_or_warn(self, idata) -> None:
        """Record the ``log|A|`` diagnostic without risking the finished posterior.

        It runs after the chains; a failure (out of memory, a solver or XLA
        runtime error, a singular or invalid system) warns and leaves
        ``sample_stats`` without ``log_abs_det``.
        """
        try:
            self._attach_flow_log_abs_det(idata)
        except (MemoryError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"The log|A| diagnostic failed ({type(exc).__name__}: {exc}); the "
                "posterior is returned without sample_stats['log_abs_det'].",
                UserWarning,
                stacklevel=3,
            )

    def _finish_count_init(
        self,
        y_arr: np.ndarray,
        effects,
        mundlak=False,
        origin_effects=None,
        destination_effects=None,
    ) -> None:
        self._y_int_vec = np.asarray(y_arr).reshape(-1).astype(np.int64)
        self._level_kinds: tuple[tuple[str, str], ...] = ()
        if mundlak and self._pair_effects == "fixed":
            raise ValueError(
                f"{type(self).__name__} takes no mundlak=True: its pair effects are "
                "fixed and absorb the pair means."
            )
        levels = (
            (effects in (1, 3) and self._pair_effects == "fixed")
            or origin_effects is not None
            or destination_effects is not None
            or (
                effects in (2, 3)
                and self._takes_levels
                and self._flow_filter == "separable"
            )
        )
        if levels:
            # Levels after the filter replace the count effects.
            self._init_count_fe(0, self._N, self._T, False)
            self._init_levels(effects, origin_effects, destination_effects)
            self.effects = effects
        else:
            self._init_count_fe(effects, self._N, self._T, mundlak)
            self.effects = self._count_effects

    def _init_levels(self, effects, origin_effects, destination_effects) -> None:
        """Levels after the filter: validate, and drop the columns they absorb."""
        for arg, value in (
            ("origin_effects", origin_effects),
            ("destination_effects", destination_effects),
        ):
            if value not in (None, "random", "fixed"):
                raise ValueError(
                    f"{arg} must be None, 'fixed' or 'random'; got {value!r}."
                )
        name = type(self).__name__
        pairs = effects in (1, 3)
        od = tuple(
            (side, how)
            for side, how in (
                ("origin", origin_effects),
                ("destination", destination_effects),
            )
            if how is not None
        )
        if self._flow_filter != "separable":
            raise NotImplementedError(
                f"{name} takes fixed pair effects and origin and destination effects "
                "with separable=True."
            )
        if pairs and od:
            raise ValueError(
                f"{name}: fixed pair effects absorb origin and destination effects; "
                "pass one or the other."
            )
        kinds = ((("pair", "fixed"),) if pairs else od) + (
            (("period", "fixed"),) if effects in (2, 3) else ()
        )
        what = (
            "fixed pair effects"
            if pairs
            else "origin and destination effects"
            if od
            else "period effects"
        )
        absorbed = _absorbed_by_levels(
            self._X, self._n, self._T, [kind for kind, _ in kinds]
        )
        keep = np.flatnonzero(~absorbed)
        if not keep.size:
            raise ValueError(
                f"No covariate is left after the {what}, so the spatial parameters "
                "are not identified."
            )
        slopes = [
            n_
            for n_, a, col in zip(self._feature_names, absorbed, self._X.T)
            if a and not np.allclose(col, col[0])
        ]
        if slopes:
            warnings.warn(
                f"{slopes} are absorbed by the {what}; their effects are not "
                "identified (NaN).",
                UserWarning,
                stacklevel=5,
            )
        prior = self._beta_keep
        self._beta_keep = keep if prior is None else prior[keep]
        self._X = self._X[:, keep]
        self._feature_names = [self._feature_names[j] for j in keep]
        self._level_kinds = kinds

    def _flow_levels(self, cell_mask=None):
        """The model's levels after the filter as :class:`FlowLevels`, or ``None``.

        The first set carries the overall level ``log mean(y)``: a fixed one
        starts there, random ones share it as their prior mean, period levels
        are centred on it.  Fixed sets after the first hold their first level at
        0 (period levels at ``time_effect_mu``); pair, origin and destination
        levels take a flat prior, period levels ``N(time_effect_mu,
        time_effect_sigma²)``.  ``log mean(y)`` reads the live cells of
        ``cell_mask`` (flat bool) only.
        """
        if not self._level_kinds:
            return None
        from ...samplers.negbin_reduced._flow_exact import FlowLevels, LevelSet

        p = self.priors
        y = self._y_int_vec if cell_mask is None else self._y_int_vec[cell_mask]
        level = float(np.log(max(float(np.mean(y)), 1e-3)))
        n_random = sum(how == "random" for _, how in self._level_kinds)
        lead_random = self._level_kinds[0][1] == "random"
        sets = []
        for i, (kind, how) in enumerate(self._level_kinds):
            if kind == "period":
                mu = float(p.get("time_effect_mu", level if i == 0 else 0.0))
                sets.append(
                    LevelSet(kind, False, mu, float(p.get("time_effect_sigma", 2.5)))
                )
            elif how == "random":
                mu = level / n_random if lead_random else 0.0
                sets.append(
                    LevelSet(
                        kind,
                        True,
                        float(p.get(f"{kind}_effect_mu", mu)),
                        float(p.get(f"{kind}_effect_sd_scale", 1.0)),
                    )
                )
            else:
                sets.append(LevelSet(kind, False, level if i == 0 else 0.0))
        return FlowLevels(tuple(sets))

    @property
    def _level_index(self) -> dict:
        """Each stacked flow's unit of every kind of level (rows are origins)."""
        r = np.arange(self._N * self._T)
        o, d = (r // self._n) % self._n, r % self._n
        return {"origin": o, "destination": d, "pair": (o, d), "period": r // self._N}

    @staticmethod
    def _level_name(kind: str) -> str:
        """A level set's posterior name (period levels are the time effects)."""
        return "time_effect" if kind == "period" else f"{kind}_effect"

    @property
    def _level_periods(self) -> list[str]:
        """The stored period levels: every period alone, all but the baseline beside others."""
        first = 0 if self._level_kinds[0][0] == "period" else 1
        return [f"t{t}" for t in range(first, self._T)]

    def _fe_coords(self) -> dict:
        coords = super()._fe_coords()
        for kind, _ in self._level_kinds:
            if kind == "period":
                coords["time"] = self._level_periods
            else:
                coords.update(
                    origin=list(range(self._n)), destination=list(range(self._n))
                )
        return coords

    def _pymc_fe_terms(self, zero: bool = False, fe_priors: Optional[dict] = None):
        """The effects' ``(inner, outer)`` parts, the levels joining ``outer``."""
        inner, outer = super()._pymc_fe_terms(zero, fe_priors)
        lv = self._flow_levels()
        if lv is None or zero:
            return inner, outer
        flat = [st.kind for st in lv.sets if st.flat]
        if flat:
            raise NotImplementedError(
                f"{type(self).__name__}'s fixed {' and '.join(flat)} effects run on "
                "the Gibbs sampler (sampler='gibbs')."
            )
        index = self._level_index
        for i, st in enumerate(lv.sets):
            name = self._level_name(st.kind)
            if st.random:
                # Non-centred, as the pair effects: c = μ + σ·z.
                sd = pm.HalfStudentT(f"{name}_sd", nu=3.0, sigma=st.scale)
                raw = pm.Normal(f"{name}_raw", 0.0, 1.0, dims=st.kind)
                level = pm.Deterministic(name, st.mu + sd * raw, dims=st.kind)
            else:
                level = pm.Normal(name, st.mu, st.scale, dims="time")
                if lv.baseline(i):
                    # Beside another set the first period is held at its mean.
                    level = pt.concatenate([pt.as_tensor_variable([st.mu]), level])
            outer = outer + level[index[st.kind]]
        return inner, outer

    def _fe_draws(self, n_draws: int):
        """Per-draw ``(inner, outer)`` effect parts, the levels joining ``outer``.

        A unit dropped for want of counts (NaN level) predicts zero counts.
        """
        inner, outer = super()._fe_draws(n_draws)
        lv = self._flow_levels()
        if lv is None:
            return inner, outer
        post = self._idata.posterior
        index = self._level_index
        for i, st in enumerate(lv.sets):
            name = self._level_name(st.kind)
            if name not in post.data_vars:
                raise RuntimeError(
                    f"posterior_predictive needs per-draw {name} draws, which this "
                    "fit stored only as a summary; refit with store_effects=True."
                )
            lev = post[name].values
            lev = lev.reshape((-1,) + lev.shape[2:])[:n_draws]
            if st.kind == "period" and lv.baseline(i):
                lev = np.concatenate([np.full((len(lev), 1), st.mu), lev], axis=1)
            idx = index[st.kind]
            part = lev[(slice(None),) + (idx if isinstance(idx, tuple) else (idx,))]
            part = np.where(np.isnan(part), -np.inf, part)
            outer = part if outer is None else outer + part
        return inner, outer

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords.update(self._fe_coords())
        return coords

    def _compute_jacobian_log_det(self, posterior) -> None:
        # The filter acts on the latent log-mean; the count density on the
        # observed counts is already the complete pointwise likelihood.
        return None

    @property
    def _count_spatial(self) -> bool:
        return self._flow_filter is not None

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        if self._flow_filter == "unrestricted":
            return self._effects_unrestricted(draws)
        if self._flow_filter == "separable":
            return self._effects_separable(draws)
        return self._effects_aspatial(draws)

    def _eta_reduced(self, rho: dict, beta: np.ndarray, inner=None) -> np.ndarray:
        """``(I_T ⊗ A(ρ))⁻¹ (Xβ + inner)`` for one draw, time-first stacked."""
        Xb = self._X @ beta if inner is None else self._X @ beta + inner
        if self._flow_filter is None:
            return Xb
        n, N, T = self._n, self._N, self._T
        Xb = Xb.reshape(T, N).T
        if self._flow_filter == "separable":
            I_n = sp.eye(n, format="csr", dtype=np.float64)
            Ld = (I_n - rho["rho_d"] * self._W_sparse).tocsr()
            Lo = (I_n - rho["rho_o"] * self._W_sparse).tocsr()
            return kron_solve_matrix(Lo, Ld, Xb, n).T.reshape(-1)
        return self._solve_A(rho["rho_d"], rho["rho_o"], rho["rho_w"], Xb).T.reshape(-1)

    def _count_filter(self):
        """The filter the unrestricted panel's count Gibbs sampler moves ρ through."""
        from ...samplers.count_panel._filters import FlowUnrestrictedFilter

        return FlowUnrestrictedFilter(
            self._Wd, self._Wo, self._Ww, self._W_sparse, self._T,
            self.priors.get("rho_lower", -0.999), self.priors.get("rho_upper", 0.999),
            self.nonnegative_dependence,
        )  # fmt: skip

    def _build_pymc_model(self) -> pm.Model:
        """The NB2 count model; no ``|A|`` Jacobian, the filter acting on the mean."""
        pv = self._flow_count_priors()
        n, N, T = self._n, self._N, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            rho = self._pymc_rho()
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            inner, outer = self._pymc_fe_terms()
            Xb = pt.dot(X_t, beta) + inner
            if self._flow_filter is None:
                eta = Xb
            else:
                Xb_mat = pt.reshape(Xb, (T, N)).T
                if self._flow_filter == "separable":
                    from ..._ops import KroneckerFlowSolveMatrixOp

                    solve = KroneckerFlowSolveMatrixOp(self._W_sparse, n)
                    eta_mat = solve(rho[0], rho[1], Xb_mat)
                else:
                    from ..._ops import SparseFlowSolveMatrixOp

                    solve = SparseFlowSolveMatrixOp(self._Wd, self._Wo, self._Ww)
                    eta_mat = solve(*rho, Xb_mat)
                eta = pt.reshape(eta_mat.T, (N * T,))
            lam = pm.Deterministic("lambda", pt.exp(eta + outer))
            alpha = nb_alpha_rv(pv["alpha_fixed"], pv["alpha_nu"], pv["alpha_sigma"])
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)
            # No |A| Jacobian: y is not transformed, the filter acts on the mean.
            # (Copying the Gaussian Jacobian biases ρ toward the negative-logdet
            # region.)
        return model

    def _pymc_rho(self) -> tuple:
        """The spatial parameters of the PyMC count model, by filter."""
        if self._flow_filter is None:
            return ()
        if self._flow_filter == "separable":
            lower = self.priors.get("rho_lower", -0.999)
            upper = self.priors.get("rho_upper", 0.999)
            rho_d = pm.Uniform("rho_d", lower=lower, upper=upper)
            rho_o = pm.Uniform("rho_o", lower=lower, upper=upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)
            return rho_d, rho_o
        if self.nonnegative_dependence:
            rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
            return tuple(
                pm.Deterministic(k, rho_simplex[i])
                for i, k in enumerate(("rho_d", "rho_o", "rho_w"))
            )
        lower = self.priors.get("rho_lower", -1.0)
        upper = self.priors.get("rho_upper", 1.0)
        rho = tuple(
            pm.Uniform(k, lower=lower, upper=upper) for k in ("rho_d", "rho_o", "rho_w")
        )
        slack = 1.0 - rho[0] - rho[1] - rho[2]
        pm.Potential("stability", pt.switch(slack > 0.0, 0.0, -1e6 * slack**2))
        return rho

    def _rho_draws(self) -> dict:
        post = self._idata.posterior
        return {
            k: post[k].values.reshape(-1)
            for k in ("rho_d", "rho_o", "rho_w")
            if k in post.data_vars
        }

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts, time-first stacked.

        Returns an array of shape ``(n_draws, n² · T)``.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
        rho = self._rho_draws()
        alpha = post["alpha"].values.reshape(-1) if "alpha" in post.data_vars else None
        inner, outer = self._fe_draws(total)
        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N * self._T), dtype=np.float64)
        for g in range(total):
            eta = self._eta_reduced(
                {k: float(v[g]) for k, v in rho.items()},
                beta_draws[g],
                None if inner is None else inner[g],
            )
            if outer is not None:
                eta = eta + outer[g]
            out[g] = self._count_draw(
                rng, eta, None if alpha is None else float(alpha[g])
            )
        return out

    def _fit_count_aspatial(
        self,
        *,
        sampler,
        backend=None,
        draws,
        tune,
        chains,
        random_seed,
        progressbar,
        n_jobs,
        idata_kwargs,
        store_effects,
        model_type,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Gibbs with ρ fixed at 0 (pair effects integrated out), or NUTS."""
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "nuts":
            return FlowModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                backend=backend,
                **sample_kwargs,
            )
        self._resolve_gibbs_backend(backend, jax=False)
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        from ...samplers.count_panel._filters import NoFilter

        pv = self._flow_count_priors()
        return self._run_count_panel_gibbs(
            NoFilter(),
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            store_effects=store_effects,
            model_type=model_type,
        )


class NegBinFlow(FlowCountDesigns, _FlowCount, FlowModel):
    """Aspatial negative binomial (NB2) origin-destination gravity model.

    ``log μ_t = X_t β``.  ``fit()`` runs Pólya–Gamma Gibbs on NumPy (with a
    panel's pair effects integrated out); ``sampler="nuts"`` runs PyMC.
    """

    __doc__ = __doc__ + _COUNT_FE_DOC

    _DESIGNS = {"static": {}}
    _spatial_prefix = None

    def _fit_static(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        backend: str | None = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior: Gibbs (default, ρ fixed at 0, on NumPy) or
        ``"nuts"`` (``backend`` as in :meth:`SARNegBinFlow.fit`)."""
        return self._fit_count_aspatial(
            sampler=sampler,
            backend=backend,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            idata_kwargs=idata_kwargs,
            store_effects=store_effects,
            model_type="nb_flow_panel",
            **sample_kwargs,
        )


class SARNegBinFlow(FlowCountDesigns, _FlowCount, FlowModel):
    r"""Reduced-form SAR negative binomial (NB2) origin-destination flow model.

    .. math::

        y_{ij,t} \sim \operatorname{NegBin}(\mu_{ij,t}, \alpha), \qquad
        \log \mu_t = A(\rho)^{-1}X_t\beta + c + \tau_t.

    ``separable=True`` pins ``ρ_w = -ρ_d ρ_o``, whose ``n × n`` structured
    sweep scales (and runs on JAX): it slices the exact likelihood, and takes
    the fixed pair effects ``c``, origin and destination effects and period
    effects ``τ`` (below).  The unrestricted flow's Pólya–Gamma JAX kernel
    covers the cross-section without effects.
    """

    __doc__ = __doc__ + _SAR_NB_EFFECTS_DOC

    _DESIGNS = {
        "static": {"_flow_filter": "unrestricted"},
        "separable": {"_flow_filter": "separable"},
    }
    _pair_effects = "fixed"
    _takes_levels = True

    def _fit_static(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        backend: str | None = None,
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior of the unrestricted NB2 SAR flow.

        ``sampler="gibbs"`` (default) runs the reduced-form Pólya–Gamma Gibbs
        sampler with per-period Kronecker solves — the recommended path for NB
        models.  ``sampler="nuts"`` uses the PyMC count path (exact likelihood,
        much slower).  The count likelihood carries no ``|A|`` change-of-variables
        term on either path.  With ``attach_log_abs_det`` (default) the per-draw
        spatial-filter Jacobian ``T·log|A(ρ)|`` is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics (never folded into
        ``log_likelihood``); set it ``False`` to skip the per-draw resolvent cost
        at very large ``N``.

        ``backend="auto"`` (default) takes JAX when it is installed and the
        configuration has a JAX kernel, else NumPy; NUTS takes ``"pymc"`` (its
        default), ``"nutpie"``, ``"numpyro"`` or ``"blackjax"``.  The unrestricted JAX kernel
        covers the cross-section (``T = 1``) without effects; panels and period
        effects (the unrestricted panel's only effects) run on NumPy.

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` on either sampler; off by default, as in PyMC.
        """
        log_lik = bool((idata_kwargs or {}).get("log_likelihood", False))
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "gibbs":
            gibbs_backend = self._resolve_gibbs_backend(
                backend, jax=not self._count_effects and self._T == 1
            )
            if self._count_effects:
                pv = self._flow_count_priors()
                idata = self._run_count_panel_gibbs(
                    self._count_filter(),
                    beta_mu=pv["beta_mu"],
                    beta_sigma=pv["beta_sigma"],
                    alpha_sigma=pv["alpha_sigma"],
                    alpha_nu=pv["alpha_nu"],
                    alpha_fixed=pv["alpha_fixed"],
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    log_likelihood=log_lik,
                    store_effects=None,
                    model_type="nb_sar_flow_panel_fe",
                )
            else:
                idata = self._fit_flow_gibbs(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    gibbs_backend=gibbs_backend,
                    log_likelihood=log_lik,
                )
        else:
            idata = FlowModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                backend=backend,
                **sample_kwargs,
            )
        if attach_log_abs_det:
            self._attach_log_abs_det_or_warn(idata)
        return idata

    def _fit_separable(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        backend: str | None = None,
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_effects: Optional[bool] = None,
        cell_mask: Optional[np.ndarray] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior of the separable NB2 SAR flow.

        ``sampler="gibbs"`` (default) slices the exact NB2 likelihood with
        per-period Kronecker solves, the effects added after the filter (see
        the class docstring); ``sampler="nuts"`` uses the PyMC count path
        (much slower, and without fixed effects).  With
        ``attach_log_abs_det`` (default) the per-draw Jacobian ``T·log|A(ρ)|``
        (using the separability relation ``ρ_w = −ρ_d ρ_o``) is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics — not folded into the
        count model's ``log_likelihood``.

        ``backend="jax"`` runs the same structured sweep compiled with JAX,
        chains on threads; ``"numpy"`` runs it on the host; ``"auto"`` (default)
        takes JAX when it is installed, else NumPy.

        ``cell_mask`` (Gibbs only) removes cells from the count likelihood: a
        bool array of live cells shaped ``(T, n, n)``, ``(T·n·n,)`` or
        ``(n, n)`` (broadcast over periods), such as all but the intraregional
        diagonal.  A fixed level with no live count is dropped (NaN).  The
        data-derived default priors read the live cells only (explicit
        ``beta_mu``, ``*_effect_mu`` overrides pass through), so the masked
        counts do not reach the posterior.

        ``store_effects`` (default: when they fit in about 500 MB) keeps every
        draw of the pair effects; otherwise only their posterior mean and sd
        (``idata["pair_effect_summary"]``), and ``posterior_predictive``
        needs the draws.

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period; 0 off the
        live cells) for ``az.loo`` on either sampler; off by default, as in
        PyMC.
        """
        log_lik = bool((idata_kwargs or {}).get("log_likelihood", False))
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "gibbs":
            idata = self._fit_flow_gibbs(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                n_jobs=n_jobs,
                gibbs_backend=self._resolve_gibbs_backend(backend, jax=True),
                log_likelihood=log_lik,
                store_effects=store_effects,
                cell_mask=cell_mask,
            )
        else:
            if cell_mask is not None:
                raise ValueError("cell_mask is supported only by the Gibbs sampler.")
            idata = FlowModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                backend=backend,
                **sample_kwargs,
            )
        if attach_log_abs_det:
            self._attach_log_abs_det_or_warn(idata)
        return idata

    def _fit_flow_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        log_likelihood: bool = False,
        store_effects: Optional[bool] = None,
        cell_mask: Optional[np.ndarray] = None,
    ) -> xr.DataTree:
        """The posterior without count effects, by reduced-form Gibbs (:mod:`._nb_gibbs`).

        Levels after the filter (:meth:`_flow_levels`) ride along on the
        separable flow's exact sampler.
        """
        from ...samplers.negbin_reduced._flow_exact import normalize_cell_mask
        from ._nb_gibbs import run_negbin_flow_gibbs

        separable = self._flow_filter == "separable"
        cell_mask = normalize_cell_mask(cell_mask, self._T, self._n)
        levels = self._flow_levels(cell_mask)
        keep_pairs = True
        if levels is not None and any(st.kind == "pair" for st in levels.sets):
            keep_pairs = self._store_group_draws(
                store_effects, chains, draws, n_groups=self._N
            )
        return run_negbin_flow_gibbs(
            self,
            separable=separable,
            model_type="nb_sar_flow_sep" if separable else "nb_sar_flow",
            omega_size=self._N * self._T,
            T=self._T,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            gibbs_backend=gibbs_backend,
            log_likelihood=log_likelihood,
            levels=levels,
            keep_pair_draws=keep_pairs,
            cell_mask=cell_mask,
        )


class _PoissonFlowMixin:
    """``fit`` of the Poisson flows: auxiliary-mixture Gibbs.

    The sampler of Frühwirth-Schnatter & Wagner (2006), there being no exact
    Pólya–Gamma representation of the Poisson.

    **Use it for counts that are close to Poisson.**  The augmentation draws
    are exact, but the finite normal mixture standing in for the
    ``−log Gamma`` error has quadratic log-tails where the true density's are
    linear.  When counts sit far from their fitted means — overdispersion the
    model does not capture — their working residuals land deep in those tails
    and pull the fit toward the large counts.  Against an exact intercept-only
    posterior: no bias on Poisson data, about one posterior sd at Pearson
    dispersion ≈ 4, seven at ≈ 13.  ``fit`` reports the dispersion in
    ``idata.attrs["pearson_dispersion"]`` and warns above
    ``_POISSON_DISPERSION_WARN``.  For overdispersed data use the NB flow
    models; Pólya–Gamma is exact there.  ``priors={"alpha_fixed": a}`` on an NB
    model with ``a`` 10-20 times the typical mean is an essentially Poisson
    model on that exact sampler (mixing degrades only for ``a`` ≳ 1000).

    Gibbs-only: unlike the NB classes there is no ``sampler="nuts"`` path yet.
    """

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        sampler: str = "gibbs",
        backend: str | None = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup sweeps, and number of chains.
        random_seed : int, optional
            Seed for reproducibility.
        sampler : {"gibbs"}, default "gibbs"
            Only the auxiliary-mixture Gibbs sampler is implemented.
        backend : {"auto", "numpy"}, optional
            The sampler runs on NumPy only.
        progressbar : bool, default True
            Show a progress bar.
        n_jobs : int, default -1
            Chain-level parallelism.
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow) for ``az.loo``.
            Off by default, as in PyMC.
        """
        if sampler != "gibbs":
            raise NotImplementedError(
                f"{type(self).__name__} supports sampler='gibbs' only; "
                f"got {sampler!r}.  There is no NUTS path for the "
                "auxiliary-mixture Poisson flow models yet."
            )
        self._resolve_gibbs_backend(
            select_backend(sampler, backend, sample_kwargs), jax=False
        )
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        idata = self._fit_gibbs(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
        )
        self._check_dispersion(idata)
        return idata

    def _build_pymc_model(self) -> pm.Model:
        raise NotImplementedError(
            f"{type(self).__name__} builds no PyMC graph: it samples by the "
            "auxiliary-mixture Gibbs sampler only."
        )

    def _poisson_eta(self, rho_d, rho_o, rho_w, beta) -> np.ndarray:
        """``A(ρ)⁻¹Xβ`` for one parameter value (``rho_w`` implied when separable)."""
        return self._eta_reduced({"rho_d": rho_d, "rho_o": rho_o, "rho_w": rho_w}, beta)

    def _check_dispersion(self, idata) -> float:
        """Pearson dispersion at the posterior mean; warn when well above 1."""
        post = idata.posterior
        mean = {k: float(post[k].mean()) for k in ("rho_d", "rho_o", "rho_w")}
        beta = post["beta"].mean(("chain", "draw")).values
        mu = np.exp(np.clip(self._poisson_eta(**mean, beta=beta), -50.0, 50.0))
        y = self._y_int_vec.astype(np.float64)
        phi = float(np.sum((y - mu) ** 2 / mu) / max(y.size - beta.size, 1))
        idata.attrs["pearson_dispersion"] = phi
        if phi > _POISSON_DISPERSION_WARN:
            warnings.warn(
                f"Pearson dispersion {phi:.2f}: the counts are overdispersed "
                "relative to this Poisson fit, and the auxiliary-mixture sampler "
                "is biased toward large counts under overdispersion (about one "
                "posterior sd at dispersion 4).  Use the negative binomial flow "
                "model instead.",
                UserWarning,
                stacklevel=3,
            )
        return phi


class SARPoissonFlow(FlowCountDesigns, _PoissonFlowMixin, _FlowCount, FlowModel):
    """Reduced-form SAR Poisson origin-destination flow model (cross-section).

    Sampled by auxiliary-mixture Gibbs, for counts close to Poisson: under
    overdispersion the sampler is biased (see the warning ``fit`` raises, and
    use :class:`SARNegBinFlow` instead).  ``separable=True``
    (``ρ_w = -ρ_d ρ_o``) is the recommended form.

    .. warning::
       The unrestricted 3-ρ parameterization mixes poorly at moderate ``n``:
       ρ_d, ρ_o and ρ_w trade off along a ridge that one-at-a-time slice
       updates cannot traverse.  Measured at n=36 (N=1296): ESS ≈ 4 of 2000
       draws with rhat ≈ 1.4, as for the unrestricted :class:`SARNegBinFlow`
       on the same data.
    """

    _DESIGNS = {
        "static": {"_flow_filter": "unrestricted"},
        "separable": {"_flow_filter": "separable"},
    }
    _has_panel = False

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        rho = self._rho_draws()
        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N), dtype=np.float64)
        for g in range(total):
            eta = self._eta_reduced(
                {k: float(v[g]) for k, v in rho.items()}, beta_draws[g]
            )
            out[g] = rng.poisson(np.exp(np.clip(eta, -50.0, 50.0))).astype(np.float64)
        return out

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample via reduced-form auxiliary-mixture Gibbs."""
        from ._poisson_gibbs import run_poisson_flow_gibbs

        separable = self._flow_filter == "separable"
        return run_poisson_flow_gibbs(
            self,
            separable=separable,
            model_type="pois_sar_flow_sep" if separable else "pois_sar_flow",
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=log_likelihood,
        )
