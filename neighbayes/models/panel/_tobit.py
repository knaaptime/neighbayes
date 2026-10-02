"""Spatial panel Tobit models.

Implements left-censored (default at 0) panel spatial Tobit variants:

- ``SARPanelTobit``: spatial lag panel Tobit
- ``SEMPanelTobit``: spatial error panel Tobit

Notes
-----
The Tobit specification splits the observation vector into two pieces:

* **Uncensored** observations enter the likelihood directly through the
  Gaussian density evaluated at the observed value.
* **Censored** observations have unknown latent values
  :math:`y^*_i < c` (with :math:`c` the censoring threshold).  These are
  augmented in the PyMC model via a half-Normal *gap* parameter
  ``y_cens_gap`` such that
  :math:`y^*_i = c - \\text{gap}_i` with :math:`\\text{gap}_i \\geq 0`,
  which trades the analytic ``Φ((c-μ)/σ)`` factor of a marginal Tobit
  likelihood for tractable posterior sampling on the joint
  :math:`(\\theta, y^*_{\\text{cens}})` space (Albert & Chib, 1993;
  Chib, 1992).

These classes force ``model=0`` (pooled transform) because within
transformations are not compatible with the censoring augmentation on
the *observed* scale: subtracting unit means would mix censored and
uncensored values inside the Gaussian likelihood.

References
----------
Chib, S. (1992). Bayes inference in the Tobit censored regression
model. *Journal of Econometrics*, 51(1–2), 79–99.

Albert, J.H. & Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *Journal of the American Statistical
Association*, 88(422), 669–679.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor import sparse as pts

from ..._backends.sampler_helpers import use_jax_likelihood
from ..._lazy_deps import pm, xr
from .._base._shared import _tobit_pointwise_loglik, _write_log_likelihood_to_idata
from ..panel_base import SpatialPanelModel
from ..priors import PanelSARTobitPriors, PanelSEMTobitPriors


class _PanelTobitBase(SpatialPanelModel):
    """Shared helpers for panel Tobit classes."""

    # Censored likelihood: the Lee & Yu dimensions are Gaussian-only.
    _lee_yu = False

    def __init__(self, *args, censoring: float = 0.0, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled — Tobit uses no FE transform
        self.censoring = float(censoring)
        super().__init__(*args, **kwargs)
        self._censored_mask = self._y <= self.censoring + 1e-12
        self._censored_idx = np.where(self._censored_mask)[0]

    def _tobit_beta_prior(self):
        """Gelman et al. (2008) ``beta`` prior, scaled to the observed ``y``."""
        return self._resolved_beta_prior(
            self._X, list(self._model_coords()["coefficient"])
        )

    def _tobit_sigma(self):
        """``σ² ~ IG(sigma2_alpha, sigma2_beta)`` (default ``IG(2, Var y)``); σ recorded."""
        sigma2 = pm.InverseGamma(
            "sigma2",
            alpha=float(self.priors.get("sigma2_alpha", 2.0)),
            beta=float(self.priors.get("sigma2_beta", np.var(self._y))),
        )
        return pm.Deterministic("sigma", pt.sqrt(sigma2))

    @staticmethod
    def _censored_gap(n_cens: int):
        """Gap ``c − y*`` of each censored latent value, with a flat prior on (0, ∞).

        The regression density on ``y*`` is the only distribution the latent
        values get, so integrating them out leaves the censored likelihood
        ``P(y* ≤ c)`` exactly.  A proper prior on the gap would be a second,
        spurious density on the same data.
        """
        return pm.HalfFlat("y_cens_gap", shape=n_cens)

    def _latent_y_tensor(self) -> pt.TensorVariable:
        y_lat = pt.as_tensor_variable(self._y.astype(np.float64))
        n_cens = int(self._censored_idx.size)
        if n_cens > 0:
            y_cens_gap = self._censored_gap(n_cens)
            y_cens = self.censoring - y_cens_gap
            y_lat = pt.set_subtensor(y_lat[self._censored_idx], y_cens)
        return y_lat

    def _posterior_latent_y_mean(self) -> np.ndarray:
        y_lat = self._y.copy().astype(float)
        if self._censored_idx.size > 0 and "y_cens_gap" in self._idata.posterior:
            gap_hat = (
                self._idata.posterior["y_cens_gap"].mean(("chain", "draw")).to_numpy()
            )
            y_lat[self._censored_idx] = self.censoring - np.asarray(
                gap_hat, dtype=float
            )
        return y_lat

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw."""
        raise NotImplementedError(
            "Spatial effects not yet implemented for panel Tobit models."
        )


class SARPanelTobit(_PanelTobitBase):
    _priors_cls = PanelSARTobitPriors
    "Bayesian spatial lag panel Tobit model.\n\n    .. math::\n        y^* = \\rho W y^* + X\\beta + \\varepsilon,\\quad \\varepsilon \\sim N(0,\\sigma^2 I)\n\n    with observed outcome\n\n    .. math::\n        y = \\max(c, y^*)\n\n    Parameters\n    ----------\n    formula : str, optional\n        Wilkinson-style formula. Requires ``data``, ``unit_col``,\n        ``time_col``.\n    data : pandas.DataFrame, optional\n        Long-format panel data when using formula mode.\n    y : array-like, optional\n        Stacked observed outcome of shape ``(N*T,)``. Required in\n        matrix mode. Values at or below ``censoring`` are treated as\n        left-censored.\n    X : array-like or pandas.DataFrame, optional\n        Stacked design matrix. Required in matrix mode.\n    W : libpysal.graph.Graph or scipy.sparse matrix\n        Spatial weights of shape ``(N, N)``. Should be\n        row-standardized.\n    unit_col, time_col : str, optional\n        Column names identifying the unit and time period in ``data``.\n        Required in formula mode.\n    N, T : int, optional\n        Cross-sectional and time dimensions. Required in matrix mode.\n    censoring : float, default 0.0\n        Left-censoring threshold ``c``. Observations with\n        ``y <= censoring`` are treated as censored and the latent\n        ``y*`` is sampled below ``c`` with a flat prior on the gap.\n    priors : dict, optional\n        Override default priors. Supported keys:\n\n        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform\n          prior on :math:`\\rho`.\n        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform\n          prior on :math:`\\rho`.\n        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal\n          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.\n        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):\n          InverseGamma prior on :math:`\\sigma^2`.\n\n        - ``nu`` (float, default 4.0): Fixed Student-t degrees of\n          freedom (only used when ``robust=True``).\n\n    logdet_method : str, optional\n        How to compute :math:`\\log|I - \\rho W|`; auto-selected when\n        ``None`` (default).\n    robust : bool, default False\n        If True, replace the Normal error with Student-t. See\n        *Robust regression* below.\n\n    Notes\n    -----\n    The base-class ``model`` argument is not exposed; pooled mean\n    structure (``model=0``) is used.\n\n    **Robust regression**\n\n    When ``robust=True``, the error distribution is changed from Normal\n    to Student-t.  For uncensored observations the density becomes:\n\n    .. math::\n\n        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =\n        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)\n\n    and for censored observations:\n\n    .. math::\n\n        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)\n\n    where :math:`T_\\nu` is the Student-t CDF and :math:`\\nu` is a **fixed**\n    hyperparameter set by ``priors={'nu': value}`` (default 4).\n    "

    def _build_pymc_model(self) -> pm.Model:
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        beta_mu, beta_sigma = self._tobit_beta_prior()
        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._tobit_sigma()
            y_lat = self._latent_y_tensor()
            resid = (
                y_lat
                - rho * pts.structured_dot(W_pt, y_lat[:, None]).flatten()
                - pt.dot(self._X, beta)
            )
            if self.robust:
                nu = self._nu
                logp_resid = pm.logp(
                    pm.StudentT.dist(nu=nu, mu=0.0, sigma=sigma), resid
                ).sum()
            else:
                logp_resid = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), resid).sum()
            pm.Potential("resid_loglik", logp_resid)
            pm.Potential("jacobian", logdet_fn(rho))
        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        y_lat = self._posterior_latent_y_mean()
        return rho * self._sparse_panel_lag(y_lat) + self._X @ beta

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SAR panel Tobit model expresses both the residual log-likelihood
        and the Jacobian via ``pm.Potential``, so PyMC captures no pointwise
        ``log_likelihood``.  We rebuild it here from the Tobit censoring
        formula with latent mean :math:`\\mu = \\rho W y^* + X\\beta`; the
        per-period Jacobian is scaled by ``T``.
        """
        if "log_likelihood" in idata.children and "obs" in idata.log_likelihood:
            return idata
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = rho.shape
        s = c * d
        n = self._y.shape[0]
        rho_f = rho.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        y_lat = self._posterior_latent_y_mean()
        Wy_lat = self._sparse_panel_lag(y_lat)
        mu = rho_f[:, None] * Wy_lat[None, :] + beta_f @ self._X.T
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        # ``_logdet_numpy_vec_fn`` is built with ``T=self._T`` and already
        # returns T·log|I_N - ρW_N|; scaling by T again squared the Jacobian.
        jac = self._logdet_numpy_vec_fn(rho_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata


class SEMPanelTobit(_PanelTobitBase):
    _priors_cls = PanelSEMTobitPriors
    "Bayesian spatial error panel Tobit model.\n\n    .. math::\n        y^* = X\\beta + u,\\quad u = \\lambda W u + \\varepsilon,\n        \\quad \\varepsilon \\sim N(0,\\sigma^2 I)\n\n    with observed outcome ``y = max(c, y*)``.\n\n    Parameters\n    ----------\n    formula : str, optional\n        Wilkinson-style formula. Requires ``data``, ``unit_col``,\n        ``time_col``.\n    data : pandas.DataFrame, optional\n        Long-format panel data when using formula mode.\n    y : array-like, optional\n        Stacked observed outcome of shape ``(N*T,)``. Required in\n        matrix mode. Values at or below ``censoring`` are treated as\n        left-censored.\n    X : array-like or pandas.DataFrame, optional\n        Stacked design matrix. Required in matrix mode.\n    W : libpysal.graph.Graph or scipy.sparse matrix\n        Spatial weights of shape ``(N, N)``. Should be\n        row-standardized.\n    unit_col, time_col : str, optional\n        Column names identifying the unit and time period in ``data``.\n        Required in formula mode.\n    N, T : int, optional\n        Cross-sectional and time dimensions. Required in matrix mode.\n    censoring : float, default 0.0\n        Left-censoring threshold ``c``.\n    priors : dict, optional\n        Override default priors. Supported keys:\n\n        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform\n          prior on :math:`\\lambda`.\n        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform\n          prior on :math:`\\lambda`.\n        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal\n          prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.\n        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):\n          InverseGamma prior on :math:`\\sigma^2`.\n\n        - ``nu`` (float, default 4.0): Fixed Student-t degrees of\n          freedom (only used when ``robust=True``).\n\n    logdet_method : str, optional\n        How to compute :math:`\\log|I - \\lambda W|`; auto-selected\n        when ``None`` (default).\n    robust : bool, default False\n        If True, replace the Normal innovation with Student-t. See\n        *Robust regression* below.\n\n    Notes\n    -----\n    The base-class ``model`` argument is not exposed; pooled mean\n    structure (``model=0``) is used.\n\n    **Robust regression**\n\n    When ``robust=True``, the error distribution is changed from Normal\n    to Student-t.  For uncensored observations the density becomes:\n\n    .. math::\n\n        f(y^*_i \\mid \\mu_i, \\sigma, \\nu) =\n        \\frac{1}{\\sigma} \\, t_\\nu\\!\\left(\\frac{y^*_i - \\mu_i}{\\sigma}\\right)\n\n    and for censored observations:\n\n    .. math::\n\n        P(y^*_i \\le c) = T_\\nu\\!\\left(\\frac{c - \\mu_i}{\\sigma}\\right)\n\n    where :math:`T_\\nu` is the Student-t CDF and :math:`\\nu` is a **fixed**\n    hyperparameter set by ``priors={'nu': value}`` (default 4).\n    "

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        beta_mu, beta_sigma = self._tobit_beta_prior()
        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs
        jax_logp = use_jax_likelihood(nuts_sampler)
        n_cens = int(self._censored_idx.size)
        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._tobit_sigma()
            if jax_logp:
                if n_cens > 0:
                    y_cens_gap = self._censored_gap(n_cens)
                else:
                    y_cens_gap = None
                X_const = pt.as_tensor_variable(self._X)
                cens_idx_const = (
                    pt.as_tensor_variable(self._censored_idx) if n_cens > 0 else None
                )
                censoring_const = self.censoring

                def _eps(value, lam_, beta_, gap_):
                    y_lat = value
                    if gap_ is not None:
                        y_cens = censoring_const - gap_
                        y_lat = pt.set_subtensor(y_lat[cens_idx_const], y_cens)
                    resid = y_lat - pt.dot(X_const, beta_)
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu
                    if n_cens > 0:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_, nu_):
                            eps = _eps(value, lam_, beta_, gap_)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            y_cens_gap,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                    else:

                        def sempanel_tobit_logp(value, lam_, beta_, sigma_, nu_):
                            eps = _eps(value, lam_, beta_, None)
                            log_dens = pm.logp(
                                pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                            )
                            return log_dens + logdet_fn(lam_) * inv_n

                        pm.CustomDist(
                            "obs",
                            lam,
                            beta,
                            sigma,
                            nu,
                            logp=sempanel_tobit_logp,
                            observed=self._y,
                        )
                elif n_cens > 0:

                    def sempanel_tobit_logp(value, lam_, beta_, sigma_, gap_):
                        eps = _eps(value, lam_, beta_, gap_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        y_cens_gap,
                        logp=sempanel_tobit_logp,
                        observed=self._y,
                    )
                else:

                    def sempanel_tobit_logp(value, lam_, beta_, sigma_):
                        eps = _eps(value, lam_, beta_, None)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        logp=sempanel_tobit_logp,
                        observed=self._y,
                    )
            else:
                y_lat = self._latent_y_tensor()
                resid = y_lat - pt.dot(self._X, beta)
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

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        beta = self._posterior_mean("beta")
        return self._X @ beta

    def _postprocess_idata(self, idata: xr.DataTree) -> xr.DataTree:
        """Attach the complete pointwise Tobit log-likelihood for IC metrics.

        The SEM panel Tobit model expresses both the error log-likelihood and
        the Jacobian via ``pm.Potential`` (NumPy path); on JAX backends the
        ``obs`` CustomDist already carries a complete ``log_likelihood`` and we
        return early.  Otherwise we rebuild it from the Tobit censoring formula
        with latent mean :math:`\\mu = X\\beta` (the spatial filter is absorbed
        into the ``T``-scaled Jacobian).
        """
        if "log_likelihood" in idata.children and "obs" in idata.log_likelihood:
            return idata
        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        c, d = lam.shape
        s = c * d
        n = self._y.shape[0]
        lam_f = lam.reshape(s)
        beta_f = beta.reshape(s, beta.shape[-1])
        sigma_f = sigma.reshape(s)
        mu = beta_f @ self._X.T
        nu_f = np.full(s, self._nu) if self.robust else None
        ll = _tobit_pointwise_loglik(
            self._y, mu, sigma_f, self._censored_mask, self.censoring, nu_f
        )
        # ``_logdet_numpy_vec_fn`` is built with ``T=self._T`` and already
        # returns T·log|I_N - λW_N|; scaling by T again squared the Jacobian.
        jac = self._logdet_numpy_vec_fn(lam_f)
        ll = (ll + jac[:, None] / n).reshape(c, d, n)
        _write_log_likelihood_to_idata(idata, ll)
        return idata
