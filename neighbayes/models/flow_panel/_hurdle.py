r"""Separable SAR hurdle NB flow models (panel and cross-section).

.. math::

    \eta_{z,t} &= (L_{z,o} \otimes L_{z,d})^{-1} X_{z,t}\beta_z
        + \tau_{z,t} + C_z, \qquad
    P(y > 0) = \mathrm{logit}^{-1}(\eta_z), \\
    \eta^{\mathrm{c}}_t &= (L_o^{\rho} \otimes L_d^{\rho})^{-1} X_t\beta
        + \tau_t + C, \qquad
    y \mid y > 0 \sim \mathrm{NB2}(e^{\eta^{\mathrm{c}}}, \alpha)
    \text{ truncated at } 0,

with :math:`L_{z,k} = I - \rho_{z,k} W_z` and
:math:`L_k^{\rho} = I - \rho_k W`.  ``effects`` sets pair and period effects
on **both** parts.  The zero part is fit to the observed ``d = 1(y > 0)``,
so unlike the flow ZINB it is identified on sparse flows; the price is that a
zero is not split into "closed" and "open but quiet".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._backends.sampler_helpers import select_backend
from ..._lazy_deps import pm, xr
from ..._ops import kron_solve_matrix
from ...graph import _weights_to_csr
from .._base._nb import nb_alpha_rv
from .._design import FacadeMeta, flow
from .._mixins._count_panel import pop_mundlak, require_counts
from .._mixins._hurdle import HurdleMixin
from ..flow._flow import FlowModel, _Separable
from ..priors import FlowPanelHurdlePriors
from ._panel import (
    _pop_effects,
    _SeparableCountFlowPanel,
)


class SARHurdleNegBinFlowSeparablePanelRE(
    HurdleMixin, _SeparableCountFlowPanel, _Separable, FlowModel
):
    r"""Reduced-form separable SAR hurdle NB flow panel.

    .. math::

        \eta_{z,t} = (L_{z,o} \otimes L_{z,d})^{-1} X_{z,t}\beta_z
        + \tau_{z,t} + C_z,
        \qquad
        \eta^{\mathrm{c}}_t = (L_o^{\rho} \otimes L_d^{\rho})^{-1} X_t\beta
        + \tau_t + C,

    ``P(y > 0) = logit⁻¹(η_z)`` and ``y | y > 0`` a zero-truncated NB2 with
    log-mean ``η^c``.  ``effects`` sets pair and period effects on **both**
    parts: in the zero part a pair effect absorbs the pair's baseline chance of
    any flow, so ``β_z`` is identified from the weeks in which a pair switches
    between zero and positive; in the count part it absorbs the pair's usual
    volume.  The parts share no parameters.  ``ρ_{z,d}``, ``ρ_{z,o}``, ``β_z``
    are the posterior's ``rho_d_zero``, ``rho_o_zero``, ``beta_zero``.

    Sampling: ``fit()`` runs the structured ``n × n`` Pólya–Gamma sweep
    (:mod:`neighbayes.samplers.hurdle._flow_structured`), on NumPy or
    ``backend="jax"``; ``sampler="nuts"`` fits the same model in PyMC.

    Parameters
    ----------
    y, X, W, T, col_names, k, priors, shared_od_covariates
        As :class:`SARNegBinFlowSeparablePanelRE`.
    X_zero : array-like or pandas.DataFrame, optional
        Zero-part design, time-first stacked ``(n²·T, p)``.  Default: ``X``
        (then the zero part's flow impacts are available).
    W_zero : libpysal.graph.Graph or scipy.sparse matrix, optional
        ``n × n`` zero-part weights.  Default: ``W``.
    effects : int, default 0
        Effects in both parts: ``0`` pooled, ``1`` pair, ``2`` period,
        ``3`` both.
    priors : dict, optional
        See :class:`~neighbayes.models.priors.FlowPanelHurdlePriors`; the count
        priors are centred on the positive flows.
    """

    _priors_cls = FlowPanelHurdlePriors

    def __init__(self, y, X, W, X_zero=None, W_zero=None, **kwargs):
        effects = _pop_effects(kwargs)
        mundlak = pop_mundlak(kwargs)
        y_arr = require_counts(y, type(self).__name__)
        if not np.any(np.asarray(y_arr) > 0):
            raise ValueError(
                f"{type(self).__name__} needs at least one positive count."
            )
        super().__init__(y_arr.astype(np.float64), X, W, effects=0, **kwargs)
        # Binary design, before the effects drop absorbed columns.
        self._sel_is_X = X_zero is None
        if X_zero is None:
            self._Z = np.asarray(self._X, dtype=np.float64).copy()
            self._sel_feature_names = list(self._feature_names)
        else:
            Z_arr = np.asarray(X_zero, dtype=np.float64)
            Z_arr = Z_arr[:, None] if Z_arr.ndim == 1 else Z_arr
            if Z_arr.shape[0] != self._N * self._T:
                raise ValueError(
                    f"Z must have n²·T = {self._N * self._T} rows, "
                    f"got {Z_arr.shape[0]}."
                )
            self._sel_feature_names = (
                [str(c) for c in X_zero.columns]
                if isinstance(X_zero, pd.DataFrame)
                else [f"z{j}" for j in range(Z_arr.shape[1])]
            )
            self._Z = Z_arr
        if W_zero is None:
            self._W_zero_sparse, self._same_W = self._W_sparse, True
        else:
            self._W_zero_sparse, self._same_W = _weights_to_csr(W_zero), False
            if self._W_zero_sparse.shape != self._W_sparse.shape:
                raise ValueError("W_zero must be n × n, like W.")
        self._is_sel_row_std = True
        self._effects_equation = "count"
        self._finish_count_init(y_arr, effects, mundlak)
        self._binary_effects_design(self._N, self._T)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords["coefficient_zero"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def _hurdle_flow_priors(self):
        from types import SimpleNamespace

        beta_mu, beta_sigma = self._hurdle_beta_prior(self._X, self._feature_names)
        gamma_mu, gamma_sigma = self._hurdle_gamma_prior(
            self._Z, self._sel_feature_names
        )
        a_sigma, a_nu, a_fixed = self._hurdle_alpha()
        return SimpleNamespace(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            alpha_sigma=a_sigma,
            alpha_nu=a_nu,
            alpha_fixed=a_fixed,
            rho_bounds=(
                float(self.priors.get("rho_lower", -0.999)),
                float(self.priors.get("rho_upper", 0.999)),
            ),
            lam_bounds=(
                float(self.priors.get("rho_lower_zero", -0.999)),
                float(self.priors.get("rho_upper_zero", 0.999)),
            ),
        )

    def _fe_args(self, fp: dict):
        pair = self._pair_args(fp) if self._fe_groups else None
        periods = (
            (fp["time_effect_mu"], fp["time_effect_sigma"], self._n_tau)
            if self._fe_periods
            else None
        )
        return pair, periods

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def fit(
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
        """Sample the posterior: structured Gibbs (default) or ``sampler="nuts"``.

        ``backend="auto"`` (default) takes JAX when it is installed, else
        NumPy; ``"numpy"`` or ``"jax"`` pins one, and NUTS takes ``"pymc"``
        (its default), ``"nutpie"``, ``"numpyro"`` or ``"blackjax"``.
        ``store_effects`` keeps every draw of both halves' pair effects
        (default: when they fit in about 500 MB).  ``idata_kwargs=
        {"log_likelihood": True}`` stores the pointwise hurdle log-likelihood.
        """
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
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        return self._fit_gibbs_hurdle(
            backend=self._resolve_gibbs_backend(backend, jax=True),
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            store_effects=store_effects,
        )

    def _fit_gibbs_hurdle(
        self,
        *,
        backend: str,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_effects: Optional[bool],
    ) -> xr.DataTree:
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.hurdle._flow_structured import run_chain_hurdle_flow_structured

        pr = self._hurdle_flow_priors()
        pair, periods = self._fe_args(self._count_fe_priors())
        sel_pair, sel_periods = self._fe_args(self._sel_fe_priors())
        keep_groups = self._store_group_draws(store_effects, chains, draws)
        y = self._y_int_vec.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        Z = np.asarray(self._Z, dtype=np.float64)
        W_csc, W_zero_csc = self._W_sparse.tocsc(), self._W_zero_sparse.tocsc()
        scalars = ["rho_d", "rho_o", "rho_w", "rho_d_zero", "rho_o_zero", "rho_w_zero"]
        extra = {"beta_zero": ("coefficient_zero", self._sel_feature_names)}
        fe_kwargs = dict(
            pair_effects=pair,
            period_effects=periods,
            sel_pair_effects=sel_pair,
            sel_period_effects=sel_periods,
        )

        if backend == "jax":
            from ...samplers.hurdle._flow_structured_jax import (
                run_chains_jax_hurdle_flow_structured,
            )

            int_seeds = [
                seed_sequence_to_int(s) for s in spawn_chain_seeds(random_seed, chains)
            ]
            results = run_chains_jax_hurdle_flow_structured(
                y, X, Z, W_csc, W_zero_csc, self._n, self._T, pr, draws, tune,
                keep_group_draws=keep_groups, jax_seeds=int_seeds,
                store_log_lik=log_likelihood, **fe_kwargs,
            )  # fmt: skip
        else:

            def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
                return run_chain_hurdle_flow_structured(
                    y, X, Z, W_csc, W_zero_csc, self._n, self._T, pr, draws, tune,
                    rho_bounds=pr.rho_bounds, lam_bounds=pr.lam_bounds,
                    rng=np.random.default_rng(seed), chain_id=chain_id,
                    progress_manager=progress_manager, store_log_lik=log_likelihood,
                    store_group_draws=keep_groups, **fe_kwargs,
                )  # fmt: skip

            seeds = (
                spawn_chain_seeds(random_seed, chains)
                if random_seed is not None
                else None
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
                model_type=type(self).__name__,
                timeout=None,
            )
        return self._assemble_count_idata(
            results, scalars, keep_groups, log_likelihood, extra=extra, sel_effects=True
        )

    def _build_pymc_model(self) -> pm.Model:
        """The same hurdle and priors as the Gibbs path, in PyMC."""
        from ..._ops import KroneckerFlowSolveMatrixOp

        pr = self._hurdle_flow_priors()
        n, N, T = self._n, self._N, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", *pr.rho_bounds)
            rho_o = pm.Uniform("rho_o", *pr.rho_bounds)
            pm.Deterministic("rho_w", -rho_d * rho_o)
            lam_d = pm.Uniform("rho_d_zero", *pr.lam_bounds)
            lam_o = pm.Uniform("rho_o_zero", *pr.lam_bounds)
            pm.Deterministic("rho_w_zero", -lam_d * lam_o)
            beta = pm.Normal(
                "beta", mu=pr.beta_mu, sigma=pr.beta_sigma, dims="coefficient"
            )
            gamma = pm.Normal(
                "beta_zero",
                mu=pr.gamma_mu,
                sigma=pr.gamma_sigma,
                dims="coefficient_zero",
            )
            inner_c, outer_c = self._pymc_fe_terms()
            inner_b, outer_b = self._pymc_fe_terms(True, self._sel_fe_priors())
            Xb = pt.reshape(pt.dot(X_t, beta) + inner_c, (T, N)).T
            Zg = pt.reshape(pt.dot(Z_t, gamma) + inner_b, (T, N)).T
            eta_c = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_sparse, n)(rho_d, rho_o, Xb).T,
                (N * T,),
            )
            eta_b = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_zero_sparse, n)(lam_d, lam_o, Zg).T,
                (N * T,),
            )
            eta_c = eta_c + outer_c
            eta_b = eta_b + outer_b
            alpha = nb_alpha_rv(pr.alpha_fixed, pr.alpha_nu, pr.alpha_sigma)
            pm.HurdleNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_b),
                mu=pt.exp(eta_c),
                alpha=alpha,
                observed=self._y_int_vec,
            )
        return model

    # ------------------------------------------------------------------
    # Per-draw predictors and impacts
    # ------------------------------------------------------------------

    def _kron_eta(self, W, rd: float, ro: float, b: np.ndarray) -> np.ndarray:
        n, N, T = self._n, self._N, self._T
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rd * W).tocsr()
        Lo = (I_n - ro * W).tocsr()
        return kron_solve_matrix(Lo, Ld, b.reshape(T, N).T, n).T.reshape(-1)

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_b, η_c)`` at draw ``g`` (pair-effect posterior means if summarized)."""
        f = self._flat_draw
        inner_b, outer_b = self._effects_at(g, True)
        inner_c, outer_c = self._effects_at(g, False)
        eta_b = self._kron_eta(
            self._W_zero_sparse,
            float(f("rho_d_zero")[g]),
            float(f("rho_o_zero")[g]),
            self._Z @ f("beta_zero")[g] + inner_b,
        )
        eta_c = self._kron_eta(
            self._W_sparse,
            float(f("rho_d")[g]),
            float(f("rho_o")[g]),
            self._X @ f("beta")[g] + inner_c,
        )
        return eta_b + outer_b, eta_c + outer_c

    def _gamma_layout(self, posterior) -> np.ndarray:
        """Posterior ``beta_zero`` in the full flow-design layout (absorbed columns NaN)."""
        gamma = posterior["beta_zero"].values.reshape(-1, len(self._sel_feature_names))
        n_mundlak = len(self._sel_mundlak_names)
        if n_mundlak:
            gamma = gamma[:, : gamma.shape[1] - n_mundlak]
        if self._sel_keep is None:
            return gamma
        full = np.full((gamma.shape[0], len(self._sel_design_feature_names)), np.nan)
        full[:, self._sel_keep] = gamma
        return full

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        post = self._idata.posterior
        if self._effects_equation == "zero":
            return self._compute_flow_effects_kron(
                post["rho_d_zero"].values.reshape(-1),
                post["rho_o_zero"].values.reshape(-1),
                self._gamma_layout(post),
                draws=draws,
            )
        return self._compute_flow_effects_kron(
            post["rho_d"].values.reshape(-1),
            post["rho_o"].values.reshape(-1),
            self._beta_layout(post),
            draws=draws,
        )

    def impacts(
        self,
        equation: str = "count",
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ):
        """Origin, destination, intraunit and network impacts for one part.

        ``equation="count"`` (default) on the count part's log-mean through
        ``ρ``; ``"zero"`` on the zero part's log-odds of any flow through
        ``rho_zero`` and ``beta_zero``, available when the zero part's design
        is the flow design (``X_zero=None``).  Other arguments as
        :meth:`FlowModel.impacts`.
        """
        if equation not in ("count", "zero"):
            raise ValueError(f"equation must be 'count' or 'zero', got {equation!r}")
        if equation == "zero" and not self._sel_is_X:
            raise NotImplementedError(
                "Zero-part flow impacts need the flow design layout; they are "
                "available when the model is built with X_zero=None (X_zero = X)."
            )
        self._effects_equation = equation
        try:
            return FlowModel.impacts(
                self,
                draws=draws,
                return_posterior_samples=return_posterior_samples,
                ci=ci,
                mode=mode,
            )
        finally:
            self._effects_equation = "count"


class SARHurdleNegBinFlowSeparable(SARHurdleNegBinFlowSeparablePanelRE):
    """Cross-sectional separable SAR hurdle NB flow model (one period).

    :class:`SARHurdleNegBinFlowSeparablePanelRE` with ``T = 1`` and no
    effects; see there for the model, sampling and arguments.  ``y`` is the
    ``n²`` flow vector (or the ``n × n`` matrix).
    """

    def __init__(self, y, X, W, **kwargs):
        if _pop_effects(kwargs):
            raise ValueError(
                "SARHurdleNegBinFlowSeparable is a cross-section; pair and period effects need "
                "SARHurdleNegBinFlowSeparablePanelRE."
            )
        kwargs.pop("T", None)
        super().__init__(y, X, W, T=1, **kwargs)


class SARHurdleNegBinFlow(metaclass=FacadeMeta):
    """Hurdle negative binomial flow model with SAR dependence.

    Implemented as the separable flow only: pass ``separable=True``.  Pass
    ``T`` for a flow panel, whose pair effects are random and time effects
    fixed.

    Parameters
    ----------
    separable : bool
        Must be ``True``.
    T : int, optional
        Periods; omit (or ``1``) for a cross-section.
    pair_effects : {None, "random"}, default None
    time_effects : bool, default False
    **kwargs
        ``y``, ``X``, ``W``, the zero part's ``X_zero`` and ``W_zero``, ``priors``,
        and ``logdet_method``.
    """

    _dispatch = classmethod(flow)
