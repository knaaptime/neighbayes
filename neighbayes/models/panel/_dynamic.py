"""Dynamic spatial panel model classes inspired by legacy dynamic panel routines.

This module provides Bayesian dynamic spatial panel models that include a
lagged dependent variable :math:`\\phi y_{i,t-1}` alongside the spatial lag.
Estimation is performed by direct posterior sampling of the conditional
likelihood — *not* by Arellano-Bond / system-GMM moment conditions, which
are not implemented in this package.

Notes
-----
The classical bias result is due to Nickell (1981): when unit fixed
effects are removed by within-demeaning of a panel that contains a
lagged dependent variable, the demeaned lag becomes correlated with the
demeaned error of order :math:`O(1/T)`, biasing the OLS / quasi-MLE
estimator of :math:`\\phi` toward zero.  Bayesian inference suffers
from the same identification pathology because the *likelihood* (not
the estimator) is the source of the bias.  This package therefore
forbids ``model=1`` (unit FE) for dynamic specifications and the
constructor raises ``ValueError``.  Users who require unit FE with a
lagged dependent variable should use a GMM package (e.g. ``linearmodels``).

References
----------
Nickell, S. (1981). Biases in dynamic models with fixed effects.
*Econometrica*, 49(6), 1417–1426.
"""

from __future__ import annotations

from functools import cached_property
from typing import Union

import numpy as np
import pytensor.tensor as pt
from pytensor import sparse as pts

from ..._backends.sampler_helpers import use_jax_likelihood
from ..._lazy_deps import pm
from ..._logdet import get_cached_logdet_fn
from .._base._shared import (
    _pointwise_gaussian_loglik,
    _write_log_likelihood_to_idata,
    gelman_default_beta_prior,
)
from ..panel_base import SpatialPanelModel, _resolve_effects
from ..priors import (
    PanelOLSDynamicPriors,
    PanelSARDynamicPriors,
    PanelSDEMDynamicPriors,
    PanelSDMRDynamicPriors,
    PanelSDMUDynamicPriors,
    PanelSEMDynamicPriors,
    PanelSLXDynamicPriors,
)


class _DynamicPanelMixin:
    """Shared helpers for dynamic panel classes based on lagged stacked arrays.

    Dynamic panel models include a lagged dependent variable (φ·y_{t-1}).
    When unit fixed effects are applied (model=1), the within-transformation
    creates correlation between the demeaned lagged dependent variable and
    the demeaned error term, biasing the estimator toward zero (Nickell, 1981).
    This is known as the **Nickell bias** or **dynamic panel bias**.

    To avoid this, dynamic panel models in this package require model=0
    (pooled), model=2 (time FE), or model=3 (two-way FE).  Setting model=1
    (unit FE) raises a ValueError at construction time.
    """

    # Fixed-T bias here is Nickell's, not Lee & Yu's; see SpatialPanelModel.
    _lee_yu = False

    _y_dyn: np.ndarray
    _y_lag: np.ndarray
    _Wy_dyn: np.ndarray
    _Wy_lag: np.ndarray
    _X_dyn: np.ndarray
    _WX_dyn: np.ndarray
    _Z_dyn: np.ndarray
    _n_time_eff: int

    def _dyn_beta_prior(self, design: np.ndarray):
        """Gelman et al. (2008) ``beta`` prior on ``design``, scaled to ``y_dyn``."""
        names = list(self._model_coords()["coefficient"])
        mu, sd = gelman_default_beta_prior(self._y_dyn, design, names)
        return self.priors.get("beta_mu", mu), self.priors.get("beta_sigma", sd)

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data

        if isinstance(self, SARPanelDynamic):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelDynamic):
            beta_draws = _get_posterior_draws(idata, "beta")
            ni = self._nonintercept_indices
            direct_samples = beta_draws[:, ni].copy()
            indirect_samples = np.zeros_like(direct_samples)
            total_samples = direct_samples.copy()

        elif isinstance(self, (SDMRPanelDynamic, SDMUPanelDynamic)):
            self._prepare_dynamic_design()
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")
            beta1_draws = beta_draws[:, : self._X_dyn.shape[1]]
            beta2_draws = beta_draws[:, self._X_dyn.shape[1] :]
            mean_diag_M = self._batch_mean_diag(rho_draws)
            mean_diag_MW = self._batch_mean_diag_MW(rho_draws)
            mean_row_sum_M = self._batch_mean_row_sum(rho_draws)
            mean_row_sum_MW = self._batch_mean_row_sum_MW(rho_draws)
            wx_idx = self._wx_column_indices
            direct_samples = (
                mean_diag_M[:, None] * beta1_draws[:, wx_idx]
                + mean_diag_MW[:, None] * beta2_draws
            )
            total_samples = (
                mean_row_sum_M[:, None] * beta1_draws[:, wx_idx]
                + mean_row_sum_MW[:, None] * beta2_draws
            )
            indirect_samples = total_samples - direct_samples

        else:
            # SLX/SDEM/DLM-like: no rho, effects are linear in beta
            self._prepare_dynamic_design()
            beta_draws = _get_posterior_draws(idata, "beta")
            k = self._X_dyn.shape[1]
            kw = self._WX_dyn.shape[1]
            beta1_draws = beta_draws[:, :k]

            if kw == 0:
                # No WX columns: direct = beta, indirect = 0, total = beta
                direct_samples = beta1_draws.copy()
                indirect_samples = np.zeros_like(beta1_draws)
                total_samples = beta1_draws.copy()
            else:
                beta2_draws = beta_draws[:, k : k + kw]
                mean_diag_w = float(self._W_sparse.diagonal().mean())
                mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])
                wx_idx = self._wx_column_indices
                direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws
                total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws
                indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples

    def __init__(self, *args, effects: Union[str, int] = 0, **kwargs):
        effects_int = _resolve_effects(effects) if isinstance(effects, str) else effects
        if effects_int == 1:
            raise ValueError(
                "effects='unit' (unit fixed effects) is not supported for dynamic "
                "panel models due to the Nickell bias (Nickell, 1981).  "
                "When unit fixed effects are removed via within-demeaning, "
                "the demeaned lagged dependent variable (y_{i,t-1} - ȳ_{i}) "
                "becomes mechanically correlated with the demeaned error "
                "(ε_{i,t} - ε̄_{i}), because ȳ_{i} contains ε_{i,t-1}.  "
                "This correlation biases the autoregressive coefficient φ "
                "toward zero, and the bias only vanishes as T → ∞.  "
                "For short panels (small T), the bias is severe and renders "
                "the estimator inconsistent.  Use effects='pooled', "
                "effects='time', or effects='two_way' instead.  "
                "For unit-specific heterogeneity with a lagged dependent "
                "variable, consider the Arellano-Bond GMM estimator or "
                "a random-effects specification (effects='pooled' with unit priors)."
            )
        super().__init__(*args, effects=effects, **kwargs)

    def _prepare_dynamic_design(self) -> None:
        if hasattr(self, "_Z_dyn"):
            return

        if self._T < 2:
            raise ValueError("Dynamic panel models require T >= 2.")

        T, N = self._T, self._N
        k = self._X.shape[1]
        kw = self._WX.shape[1]

        y2 = self._y.reshape(T, N)
        wy2 = self._Wy.reshape(T, N)
        X3 = self._X.reshape(T, N, k)

        self._y_dyn = y2[1:, :].reshape(-1)
        self._y_lag = y2[:-1, :].reshape(-1)
        self._Wy_dyn = wy2[1:, :].reshape(-1)
        self._Wy_lag = wy2[:-1, :].reshape(-1)
        self._X_dyn = X3[1:, :, :].reshape(-1, k)

        if kw > 0:
            WX3 = self._WX.reshape(T, N, kw)
            self._WX_dyn = WX3[1:, :, :].reshape(-1, kw)
            self._Z_dyn = np.hstack([self._X_dyn, self._WX_dyn])
        else:
            self._WX_dyn = np.empty((self._X_dyn.shape[0], 0), dtype=float)
            self._Z_dyn = self._X_dyn

        self._n_time_eff = T - 1

    @cached_property
    def _W_dense_dyn(self) -> np.ndarray:
        """Dense (N*(T-1)) x (N*(T-1)) block-diagonal W for dynamic period.

        .. warning::

           This property materializes an ``(N*(T-1)) × (N*(T-1))`` dense matrix
           and grows as ``O(N² (T-1)²)``.  Prefer :meth:`_batch_sparse_lag`
           with ``T_eff=self._n_time_eff`` for batch log-likelihood paths.
           This property is retained for diagnostics requiring the full matrix.
        """
        import warnings

        n = self._N * self._n_time_eff
        nbytes = n * n * 8
        if nbytes > 100 * 1024 * 1024:
            warnings.warn(
                f"Materialising a {n}×{n} dense dynamic-panel weight matrix "
                f"({nbytes / 1024**2:.0f} MB).  Use _batch_sparse_lag with "
                "T_eff=self._n_time_eff instead.",
                ResourceWarning,
                stacklevel=3,
            )
        Wn = (
            self._W_sparse.toarray()
            if hasattr(self._W_sparse, "toarray")
            else np.asarray(self._W_sparse)
        )
        return np.kron(np.eye(self._n_time_eff), Wn)

    @cached_property
    def _W_sparse_dyn(self):
        """Sparse (N*(T-1))×(N*(T-1)) Kronecker block weight ``I_{T-1} ⊗ W_n``."""
        import scipy.sparse as sp

        # Force ``csr_matrix`` (not ``csr_array``) for pytensor.sparse compatibility.
        return sp.csr_matrix(
            sp.kron(
                sp.eye(self._n_time_eff, format="csr"), self._W_sparse, format="csr"
            )
        )

    @cached_property
    def _W_pt_sparse_dyn(self):
        """PyTensor sparse variable wrapping :attr:`_W_sparse_dyn`."""
        import scipy.sparse as sp
        from pytensor import sparse as pts

        return pts.as_sparse_variable(sp.csc_matrix(self._W_sparse_dyn))

    @cached_property
    def _dynamic_logdet_cache(self) -> dict:
        """Per-model store of dynamic logdet callables, keyed by rho bounds."""
        return {}

    def _dynamic_logdet_fn(self, lower: float, upper: float):
        """Build and cache dynamic-panel logdet callable keyed by bounds."""
        method = self._logdet_bounds.method
        key = (method, float(lower), float(upper), int(self._n_time_eff))
        fn = self._dynamic_logdet_cache.get(key)
        if fn is None:
            fn = get_cached_logdet_fn(
                self._W_for_logdet,
                method=method,
                rho_min=lower,
                rho_max=upper,
                T=self._n_time_eff,
            )
            self._dynamic_logdet_cache[key] = fn
        return fn

    def _reconstruct_dynamic_log_likelihood(
        self,
        *,
        spatial_param: str,
        nuts_sampler: str,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for dynamic panel models."""
        if not hasattr(self, "_idata"):
            return
        if spatial_param not in {"rho", "lam"}:
            return

        # SEM/SDEM on JAX backends build an observed CustomDist and already
        # have complete log_likelihood from PyMC.
        if spatial_param == "lam" and use_jax_likelihood(nuts_sampler):
            return

        self._prepare_dynamic_design()
        idata = self._idata
        n_obs = int(self._y_dyn.shape[0])

        beta_raw = idata.posterior["beta"].values
        beta_draws = beta_raw.reshape(-1, beta_raw.shape[-1])
        x_cols = int(self._X_dyn.shape[1])
        z_cols = int(self._X_dyn.shape[1] + self._WX_dyn.shape[1])
        if beta_draws.shape[1] == x_cols:
            design = self._X_dyn
        elif beta_draws.shape[1] == z_cols:
            design = np.hstack([self._X_dyn, self._WX_dyn])
        else:
            raise ValueError(
                "beta draw width does not match dynamic design: "
                f"{beta_draws.shape[1]} not in {{{x_cols}, {z_cols}}}."
            )

        phi_draws = idata.posterior["phi"].values.reshape(-1)
        sigma_draws = idata.posterior["sigma"].values.reshape(-1)
        nu_draws = np.full(phi_draws.shape[0], self._nu) if self.robust else None
        spatial_draws = idata.posterior[spatial_param].values.reshape(-1)

        if spatial_param == "rho":
            mu = (
                phi_draws[:, None] * self._y_lag[None, :]
                + spatial_draws[:, None] * self._Wy_dyn[None, :]
                + beta_draws @ design.T
            )
            if "theta" in idata.posterior:
                theta_draws = idata.posterior["theta"].values.reshape(-1)
                mu = mu + theta_draws[:, None] * self._Wy_lag[None, :]
            elif design.shape[1] == z_cols:
                mu = mu - (
                    spatial_draws[:, None] * phi_draws[:, None] * self._Wy_lag[None, :]
                )
            eps = self._y_dyn[None, :] - mu
        else:
            resid = (
                self._y_dyn[None, :]
                - phi_draws[:, None] * self._y_lag[None, :]
                - beta_draws @ design.T
            )
            w_resid = self._batch_sparse_lag(resid, T_eff=self._n_time_eff)
            eps = resid - spatial_draws[:, None] * w_resid

        ll_data = _pointwise_gaussian_loglik(eps, sigma_draws, nu_draws)
        # _logdet_numpy_vec_fn is scaled by the full panel length T, but the
        # dynamic design only spans the T-1 effective periods — rescale so the
        # Jacobian is T_eff * logdet(I - rho W), matching the sampled model.
        jacobian = self._logdet_numpy_vec_fn(spatial_draws) * (
            self._n_time_eff / self._T
        )
        ll_total = ll_data + jacobian[:, None] / n_obs

        n_chains = idata.posterior.sizes["chain"]
        n_draws_per_chain = idata.posterior.sizes["draw"]
        _write_log_likelihood_to_idata(
            idata,
            ll_total.reshape(n_chains, n_draws_per_chain, n_obs),
        )

    def _fit_dynamic(
        self,
        spatial_param: str,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Shared NUTS fit for spatial dynamic panels.

        Dynamic panel builders use ``pm.Normal("obs", observed=y)``, which
        auto-captures the Gaussian log-likelihood, plus a ``pm.Potential``
        Jacobian term that is not captured.  When ``log_likelihood=True`` is
        requested, the Jacobian-corrected log-likelihood is reconstructed
        post-sampling via :meth:`_reconstruct_dynamic_log_likelihood`.
        """
        idata_kwargs = idata_kwargs or {}
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
        progressbar = sample_kwargs.pop("progressbar", True)

        _, compute_log_likelihood = self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=compute_log_likelihood,
            sample_kwargs=sample_kwargs,
        )

        if compute_log_likelihood:
            self._reconstruct_dynamic_log_likelihood(
                spatial_param=spatial_param,
                nuts_sampler=nuts_sampler,
            )

        return self._idata


class OLSPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelOLSDynamicPriors
    _has_wx_in_beta = True

    """Dynamic panel regression without contemporaneous spatial dependence.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1} + x_{it}'\\beta
        + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2),

    where the :math:`(\\sum_j w_{ij} x_{jt})'\\theta` block is present
    only when the base design marks covariates as laggable. The admissible
    panel transformations are pooled, time effects, and two-way effects;
    ``model=1`` is rejected by :class:`_DynamicPanelMixin` because unit
    fixed effects with a lagged dependent variable induce Nickell bias.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)`` in unit-major order.
        Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Used to construct the
        ``WX`` block. Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode; must be
        ``>= 2``.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected because it
        induces Nickell bias.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        Accepted for API consistency; unused (no contemporaneous
        spatial lag on ``y``).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(self._Z_dyn)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        with pm.Model(coords=self._model_coords()) as model:
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            mu = phi * self._y_lag + pt.dot(self._Z_dyn, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return phi * self._y_lag + self._Z_dyn @ beta


class SDMRPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSDMRDynamicPriors
    _has_wx_in_beta = True

    """Dynamic restricted spatial Durbin panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1}
        + \\rho \\sum_j w_{ij} y_{jt}
        - \\rho \\phi \\sum_j w_{ij} y_{j,t-1}
        + x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\varepsilon_{it},

    where the restriction ties the lagged spatial spillover term to
    :math:`-\\rho \\phi`. As with all dynamic classes here, ``model=1`` is
    disallowed because of Nickell bias.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(self._Z_dyn)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        logdet_fn = self._dynamic_logdet_fn(rho_lower, rho_upper)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            mu = (
                phi * self._y_lag
                + rho * self._Wy_dyn
                - rho * phi * self._Wy_lag
                + pt.dot(self._Z_dyn, beta)
            )
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach Jacobian-corrected log-likelihood."""
        return self._fit_dynamic(
            "rho",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return (
            phi * self._y_lag
            + rho * self._Wy_dyn
            - rho * phi * self._Wy_lag
            + self._Z_dyn @ beta
        )


class SDMUPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSDMUDynamicPriors
    _has_wx_in_beta = True

    """Dynamic unrestricted spatial Durbin panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1}
        + \\rho \\sum_j w_{ij} y_{jt} + \\psi \\sum_j w_{ij} y_{j,t-1}
        + x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\varepsilon_{it},

    where :math:`\\psi` is an unrestricted coefficient on the lagged
    spatial outcome. As with the other dynamic specifications, admissible
    panel transforms are pooled, time effects, and two-way effects.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardized.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode;
        ``T >= 2`` is required.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected (Nickell bias).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``theta_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\psi` (lagged-spatial coefficient).
        - ``theta_upper`` (float, default 0.95): Upper bound of
          Uniform prior on :math:`\\psi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        bounds = self._logdet_bounds
        rho_lower = bounds.rho_min
        rho_upper = bounds.rho_max
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        theta_lower = self.priors.get("theta_lower", -0.95)
        theta_upper = self.priors.get("theta_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(self._Z_dyn)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        logdet_fn = self._dynamic_logdet_fn(rho_lower, rho_upper)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            theta = pm.Uniform("theta", lower=theta_lower, upper=theta_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            mu = (
                phi * self._y_lag
                + rho * self._Wy_dyn
                + theta * self._Wy_lag
                + pt.dot(self._Z_dyn, beta)
            )
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach Jacobian-corrected log-likelihood."""
        return self._fit_dynamic(
            "rho",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        phi = float(self._posterior_mean("phi"))
        theta = float(self._posterior_mean("theta"))
        beta = self._posterior_mean("beta")
        return (
            phi * self._y_lag
            + rho * self._Wy_dyn
            + theta * self._Wy_lag
            + self._Z_dyn @ beta
        )


class SARPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSARDynamicPriors

    """Dynamic spatial-lag panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1} + \\rho \\sum_j w_{ij} y_{jt}
        + x_{it}'\\beta + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    This is the dynamic analogue of :class:`SARPanelFE`: it adds a lagged
    dependent variable but no Durbin block. The Jacobian contribution is
    based on :math:`|I - \\rho W|^{T-1}` for the contemporaneous spatial
    lag.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardized.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode;
        ``T >= 2`` is required.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected (Nickell bias).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        rho_lower = self.priors.get("rho_lower", -0.95)
        rho_upper = self.priors.get("rho_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(self._X_dyn)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        logdet_fn = self._dynamic_logdet_fn(rho_lower, rho_upper)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            mu = phi * self._y_lag + rho * self._Wy_dyn + pt.dot(self._X_dyn, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach Jacobian-corrected log-likelihood."""
        return self._fit_dynamic(
            "rho",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        rho = float(self._posterior_mean("rho"))
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return phi * self._y_lag + rho * self._Wy_dyn + self._X_dyn @ beta


class SEMPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSEMDynamicPriors

    """Dynamic spatial-error panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1} + x_{it}'\\beta + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    The likelihood uses the filtered residuals
    :math:`(I - \\lambda W)(y_t - \\phi y_{t-1} - X_t \\beta)` and the
    associated Jacobian :math:`|I - \\lambda W|^{T-1}`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardized.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode;
        ``T >= 2`` is required.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected (Nickell bias).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)(y_t - \\phi y_{t-1} - X_t \\beta) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        self._prepare_dynamic_design()

        lam_lower = self.priors.get("lam_lower", -0.95)
        lam_upper = self.priors.get("lam_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(self._X_dyn)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        logdet_fn = self._dynamic_logdet_fn(lam_lower, lam_upper)

        W_pt = self._W_pt_sparse_dyn
        n_obs = int(self._y_dyn.shape[0])
        # ``logdet_fn`` already includes the T_eff multiplier; distribute the
        # full panel Jacobian uniformly across the n_obs entries.
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if jax_logp:
                ylag_const = pt.as_tensor_variable(self._y_lag)
                X_const = pt.as_tensor_variable(self._X_dyn)

                def _eps(value, lam_, phi_, beta_):
                    resid = value - phi_ * ylag_const - pt.dot(X_const, beta_)
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu

                    def sempanel_dyn_logp(value, lam_, phi_, beta_, sigma_, nu_):
                        eps = _eps(value, lam_, phi_, beta_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        phi,
                        beta,
                        sigma,
                        nu,
                        logp=sempanel_dyn_logp,
                        observed=self._y_dyn,
                    )
                else:

                    def sempanel_dyn_logp(value, lam_, phi_, beta_, sigma_):
                        eps = _eps(value, lam_, phi_, beta_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        phi,
                        beta,
                        sigma,
                        logp=sempanel_dyn_logp,
                        observed=self._y_dyn,
                    )
            else:
                resid = self._y_dyn - phi * self._y_lag - pt.dot(self._X_dyn, beta)
                eps = resid - lam * pts.structured_dot(W_pt, resid[:, None]).flatten()
                if self.robust:
                    nu = self._nu
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                    ).sum()
                else:
                    logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
                pm.Potential("eps_loglik", logp_eps)
                pm.Potential("jacobian", logdet_fn(lam))

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
        return self._fit_dynamic(
            "lam",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        return phi * self._y_lag + self._X_dyn @ beta


class SDEMPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSDEMDynamicPriors
    _has_wx_in_beta = True

    """Dynamic spatial Durbin error panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1} + x_{it}'\\beta
        + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    This is the dynamic analogue of :class:`SDEMPanelFE`: lagged outcome
    persistence enters through :math:`\\phi`, while spatial dependence
    remains in the disturbance.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be
        row-standardized.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode;
        ``T >= 2`` is required.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected (Nickell bias).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)\\bigl(y_t - \\phi y_{t-1} - X_t \\beta - (W X_t)\\theta\\bigr) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        return self._feature_names

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        self._prepare_dynamic_design()

        Z = np.hstack([self._X_dyn, self._WX_dyn])

        lam_lower = self.priors.get("lam_lower", -0.95)
        lam_upper = self.priors.get("lam_upper", 0.95)
        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(Z)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        logdet_fn = self._dynamic_logdet_fn(lam_lower, lam_upper)

        W_pt = self._W_pt_sparse_dyn
        n_obs = int(self._y_dyn.shape[0])
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            if jax_logp:
                ylag_const = pt.as_tensor_variable(self._y_lag)
                Z_const = pt.as_tensor_variable(Z)

                def _eps(value, lam_, phi_, beta_):
                    resid = value - phi_ * ylag_const - pt.dot(Z_const, beta_)
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu

                    def sdempanel_dyn_logp(value, lam_, phi_, beta_, sigma_, nu_):
                        eps = _eps(value, lam_, phi_, beta_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        phi,
                        beta,
                        sigma,
                        nu,
                        logp=sdempanel_dyn_logp,
                        observed=self._y_dyn,
                    )
                else:

                    def sdempanel_dyn_logp(value, lam_, phi_, beta_, sigma_):
                        eps = _eps(value, lam_, phi_, beta_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        phi,
                        beta,
                        sigma,
                        logp=sdempanel_dyn_logp,
                        observed=self._y_dyn,
                    )
            else:
                resid = self._y_dyn - phi * self._y_lag - pt.dot(Z, beta)
                eps = resid - lam * pts.structured_dot(W_pt, resid[:, None]).flatten()
                if self.robust:
                    nu = self._nu
                    logp_eps = pm.logp(
                        pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), eps
                    ).sum()
                else:
                    logp_eps = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), eps).sum()
                pm.Potential("eps_loglik", logp_eps)
                pm.Potential("jacobian", logdet_fn(lam))

        return model

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        idata_kwargs: dict | None = None,
        **sample_kwargs,
    ):
        """Sample posterior and attach pointwise log-likelihood for IC metrics."""
        return self._fit_dynamic(
            "lam",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            idata_kwargs=idata_kwargs,
            **sample_kwargs,
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X_dyn, self._WX_dyn])
        return phi * self._y_lag + Z @ beta


class SLXPanelDynamic(_DynamicPanelMixin, SpatialPanelModel):
    _priors_cls = PanelSLXDynamicPriors
    _has_wx_in_beta = True

    """Dynamic SLX panel regression.

    Implements

    .. math::

        y_{it} = \\phi y_{i,t-1} + x_{it}'\\beta
        + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    This is the dynamic analogue of :class:`SLXPanelFE`. There is no
    contemporaneous spatial lag on :math:`y`, so no Jacobian adjustment is
    required.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula. Requires ``data``, ``unit_col``,
        ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Used to construct the
        ``WX`` block. Should be row-standardized.
    unit_col, time_col : str, optional
        Column names identifying the unit and time period in ``data``.
        Required in formula mode.
    N, T : int, optional
        Cross-sectional and time dimensions. Required in matrix mode;
        ``T >= 2`` is required.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``2`` time FE,
        ``3`` two-way FE. ``1`` (unit FE) is rejected (Nickell bias).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``phi_lower`` (float, default -0.95): Lower bound of
          Uniform prior on :math:`\\phi`.
        - ``phi_upper`` (float, default 0.95): Upper bound of Uniform
          prior on :math:`\\phi`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        Accepted for API consistency; unused (no spatial Jacobian).
    robust : bool, default False
        If True, replace the Normal error with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_t \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    def _beta_names(self) -> list[str]:
        if self._wx_feature_names:
            return self._feature_names + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        return self._feature_names

    def _build_pymc_model(self) -> pm.Model:
        self._prepare_dynamic_design()

        Z = np.hstack([self._X_dyn, self._WX_dyn])

        phi_lower = self.priors.get("phi_lower", -0.95)
        phi_upper = self.priors.get("phi_upper", 0.95)
        beta_mu, beta_sigma = self._dyn_beta_prior(Z)
        sigma2_alpha = self.priors.get("sigma2_alpha", 2.0)
        sigma2_beta = self.priors.get("sigma2_beta", float(np.var(self._y_dyn)))

        with pm.Model(coords=self._model_coords()) as model:
            phi = pm.Uniform("phi", lower=phi_lower, upper=phi_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))

            mu = phi * self._y_lag + pt.dot(Z, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y_dyn)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y_dyn)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        self._prepare_dynamic_design()
        phi = float(self._posterior_mean("phi"))
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X_dyn, self._WX_dyn])
        return phi * self._y_lag + Z @ beta
