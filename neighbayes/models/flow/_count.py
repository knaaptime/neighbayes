r"""The count flow families: negative binomial, and Poisson.

Each is one class (see
:class:`~neighbayes.models._mixins._designs.FlowCountDesigns`): ``separable=True``
selects the separable filter and ``T`` a panel, the cross-section being one
period.  The filter acts on the log-mean,

.. math::

    \eta_t = A(\rho)^{-1}(X_t\beta + C) + \tau_t,

so the count likelihood carries no ``|A|`` Jacobian.  A negative binomial
panel's pair effects ``C`` are random and partially pooled, and its period
effects ``τ`` fixed.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
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
    A panel (``T``) takes ``pair_effects="random"`` and ``time_effects=True``.  The within transform of the Gaussian panels does not carry to
    a log link, so the effects are parameters: pair effects (one per
    origin–destination pair), integrated out of every ρ update and drawn
    jointly with β, held as a pair index rather than ``n²`` dummy columns.  They
    enter inside the spatial filter, with the covariates,
    ``η_t = A⁻¹(X_tβ + C) + τ_t``: random effects propagate as ``X`` does.
    Under pooling that is a different prior from effects added after the
    filter, not a reparameterization (inside, ``A⁻¹C`` has covariance
    ``σ²(AᵀA)⁻¹``).  The separable panel's structured sweep cannot integrate
    them out at scale, so it holds them while ρ moves and draws them by
    conjugate gradients.

    Pair effects are **partially pooled**: ``C_ij ~ N(μ, σ²)`` with σ learned
    (half-t(3, ``pair_effect_sd_scale``) prior, default scale 1).  A pair seen
    in few periods is shrunk toward the common mean in proportion to how little
    its data say; with weekly sparse flows a fixed, wide prior instead leaves
    the dispersion α and ρ with an incidental-parameter bias and lets ρ drift
    to a degenerate mode near 1.  Pair effects absorb nothing: the intercept
    and time-invariant columns (intra indicator, log distance) stay in the
    design and anchor the level.  ``mundlak=True`` adds the pair means of the
    time-varying columns inside the filter, absorbing a correlation between the
    pair effects and ``X``.

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
    draws.  Effects run on the NumPy Gibbs backend, and on JAX for the
    separable panels (the structured sweep that scales).
"""


class _FlowCount(CountPanelFEMixin):
    """Count flows: pair and period effects, the filtered log-mean, prediction.

    What depends on the filter (``_flow_filter``) branches on it: the
    log-mean ``η``, the Gibbs filter, the PyMC model and the effects.
    """

    _effect_index = "pair"
    _robust_nuts = False

    def _finish_count_init(self, y_arr: np.ndarray, effects, mundlak=False) -> None:
        self._y_int_vec = np.asarray(y_arr).reshape(-1).astype(np.int64)
        self._init_count_fe(effects, self._N, self._T, mundlak)
        self.effects = self._count_effects

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
        """The filter the count Gibbs sampler moves ρ through."""
        from ...samplers.count_panel._filters import (
            FlowSeparableFilter,
            FlowUnrestrictedFilter,
            NoFilter,
        )

        lower = self.priors.get("rho_lower", -0.999)
        upper = self.priors.get("rho_upper", 0.999)
        if self._flow_filter == "unrestricted":
            return FlowUnrestrictedFilter(
                self._Wd, self._Wo, self._Ww, self._W_sparse, self._T, lower, upper,
                self.nonnegative_dependence,
            )  # fmt: skip
        if self._flow_filter == "separable":
            return FlowSeparableFilter(self._W_sparse, self._T, lower, upper)
        return NoFilter()

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
        \log \mu_t = A(\rho)^{-1}(X_t\beta + C) + \tau_t.

    ``separable=True`` pins ``ρ_w = -ρ_d ρ_o``, whose ``n × n`` structured
    sweep scales (and runs on JAX); the unrestricted flow's JAX kernel covers
    the cross-section without effects.
    """

    __doc__ = __doc__ + _COUNT_FE_DOC

    _DESIGNS = {
        "static": {"_flow_filter": "unrestricted"},
        "separable": {"_flow_filter": "separable"},
    }

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
        store_effects: Optional[bool] = None,
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
        covers the cross-section (``T = 1``) without effects; panels and pair or
        period effects run on NumPy.  With pair or period effects the sweep
        also integrates out the pair effects; see the class docstring.

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
                    store_effects=store_effects,
                    model_type="nb_sar_flow_panel_fe",
                )
            else:
                idata = self._fit_pooled_gibbs(
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
            self._attach_flow_log_abs_det(idata)
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

        ``sampler="gibbs"`` (default) runs the reduced-form Pólya–Gamma Gibbs
        sampler with per-period Kronecker solves; ``sampler="nuts"`` uses the
        PyMC count path (exact likelihood, much slower).  With
        ``attach_log_abs_det`` (default) the per-draw Jacobian ``T·log|A(ρ)|``
        (using the separability relation ``ρ_w = −ρ_d ρ_o``) is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics — not folded into the
        count model's ``log_likelihood``.

        ``backend="jax"`` runs the same structured sweep compiled with JAX,
        chains on threads; ``"numpy"`` runs it on the host; ``"auto"`` (default)
        takes JAX when it is installed, else NumPy.  With pair or period
        effects the structured sweep also integrates out the effects, on either backend (see the class
        docstring).

        ``cell_mask`` (Gibbs effects path only) masks cells out of the count
        likelihood: a bool array of live cells shaped ``(T, n, n)``,
        ``(T·n·n,)`` or ``(n, n)`` (broadcast over periods).  Masked cells are
        drawn-then-discarded — their Pólya–Gamma variates keep the random
        stream mask-invariant — and pairs whose every period is masked stay
        prior-only and drop out of the learned pair-sd deviance, so masking
        the intraunit diagonal leaves the posterior of every other cell
        bit-identical.  The default priors are data-derived, so under a mask
        they re-derive from the live cells alone (the Gelman intercept
        centring and the effect levels use ``log mean(y_live)``); explicit
        ``beta_mu``/``pair_effect_mu``/``time_effect_mu`` overrides pass
        through untouched.  Raises on the pooled or NUTS paths.

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` on either sampler; off by default, as in PyMC.
        """
        log_lik = bool((idata_kwargs or {}).get("log_likelihood", False))
        backend = select_backend(sampler, backend, sample_kwargs)
        if sampler == "gibbs":
            gibbs_backend = self._resolve_gibbs_backend(backend, jax=True)
            if self._count_effects:
                idata = self._fit_gibbs_fe(
                    gibbs_backend=gibbs_backend,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    log_likelihood=log_lik,
                    store_effects=store_effects,
                    cell_mask=cell_mask,
                )
            else:
                if cell_mask is not None:
                    raise ValueError(
                        "cell_mask is supported by the effects Gibbs path; "
                        "fit a panel with pair or period effects."
                    )
                idata = self._fit_pooled_gibbs(
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
            if cell_mask is not None:
                raise ValueError(
                    "cell_mask is supported only by the Gibbs effects path"
                )
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
            self._attach_flow_log_abs_det(idata)
        return idata

    def _fit_pooled_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """The posterior without effects, by reduced-form Pólya–Gamma Gibbs."""
        from ._nb_gibbs import run_negbin_flow_gibbs

        separable = self._flow_filter == "separable"
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
        )

    def _fit_gibbs_fe(
        self,
        *,
        gibbs_backend: str = "numpy",
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_effects: Optional[bool],
        cell_mask: Optional[np.ndarray] = None,
    ) -> xr.DataTree:
        """Structured sweep with pair and period effects integrated out."""

        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.negbin_reduced._flow_structured import classify_flow_design
        from ...samplers.negbin_reduced._flow_structured_fe import (
            _normalize_cell_mask,
            run_chain_separable_structured_fe,
        )

        pv = self._flow_count_priors()
        fp = self._count_fe_priors()
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)
        cell_mask = _normalize_cell_mask(cell_mask, self._T, self._n)
        if cell_mask is not None:
            # Both default priors are data-derived — the Gelman intercept
            # centring and the effect levels use log mean y — and masked
            # cells are discarded from the likelihood, so re-derive them
            # from the live cells.  Rebuilding through the same helpers
            # keeps every explicit prior override in force.
            from .._base._shared import gelman_default_beta_prior

            y_live = self._y_int_vec.astype(np.float64)[cell_mask]
            X_arr = np.asarray(self._X, dtype=np.float64)
            k = X_arr.shape[1]
            names = list(self._feature_names) or [f"x{j}" for j in range(k)]
            g_mu, g_sd = gelman_default_beta_prior(y_live, X_arr, names, link="log")
            pv = dict(pv)
            pv["beta_mu"] = np.broadcast_to(
                np.asarray(self.priors.get("beta_mu", g_mu), dtype=np.float64), (k,)
            ).copy()
            pv["beta_sigma"] = np.broadcast_to(
                np.asarray(self.priors.get("beta_sigma", g_sd), dtype=np.float64), (k,)
            ).copy()
            fp = self._effect_priors(
                float(np.log(max(float(np.mean(y_live)), 1e-3))), self._X, ""
            )
        priors = SimpleNamespace(
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )
        keep_groups = self._store_group_draws(store_effects, chains, draws)
        y = self._y_int_vec.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        struct = classify_flow_design(X, self._n, self._T)
        W_csc = self._W_sparse.tocsc()
        k = X.shape[1]
        pair = self._pair_args(fp) if self._fe_groups else None
        periods = (
            (fp["time_effect_mu"], fp["time_effect_sigma"], self._n_tau)
            if self._fe_periods
            else None
        )

        def _init(rng):
            return SimpleNamespace(
                beta=rng.normal(0.0, 0.1, size=k),
                rho_d=rng.uniform(-0.1, 0.1),
                rho_o=rng.uniform(-0.1, 0.1),
                alpha=1.0 if pv["alpha_fixed"] is None else pv["alpha_fixed"],
            )

        if gibbs_backend == "jax":
            from ...samplers._utils._seeds import seed_sequence_to_int
            from ...samplers.negbin_reduced._flow_structured_fe_jax import (
                run_chains_jax_flow_structured_fe,
            )

            int_seeds = [
                seed_sequence_to_int(s) for s in spawn_chain_seeds(random_seed, chains)
            ]
            results = run_chains_jax_flow_structured_fe(
                y,
                W_csc,
                self._n,
                priors,
                [_init(np.random.default_rng(s)) for s in int_seeds],
                draws,
                tune,
                struct=struct,
                pair_effects=pair,
                period_effects=periods,
                keep_group_draws=keep_groups,
                jax_seeds=int_seeds,
                store_log_lik=log_likelihood,
                cell_mask=cell_mask,
            )
            return self._assemble_count_idata(
                results, ["rho_d", "rho_o", "rho_w"], keep_groups, log_likelihood
            )

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            rng = np.random.default_rng(seed)
            init = _init(rng)
            return run_chain_separable_structured_fe(
                y,
                X,
                W_csc,
                self._n,
                priors,
                init,
                draws,
                tune,
                T=self._T,
                pair_effects=pair,
                period_effects=periods,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
                rng=rng,
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                store_group_draws=keep_groups,
                struct=struct,
                cell_mask=cell_mask,
            )

        seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type="nb_sar_flow_sep_panel_fe",
            timeout=None,
        )
        return self._assemble_count_idata(
            results, ["rho_d", "rho_o", "rho_w"], keep_groups, log_likelihood
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
