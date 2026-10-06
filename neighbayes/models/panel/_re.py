"""Bayesian spatial panel models with unit random effects.

Analogues of the legacy ``prandom`` (non-spatial GLS random effects) and the
LeSage/Pace spatial panel routines, cast as hierarchical Bayesian models.

Model structure for all three classes
--------------------------------------
.. math::
    y_{it} = \\mu_{it} + \\alpha_i + \\varepsilon_{it}

where :math:`\\mu_{it}` is the spatial or non-spatial mean depending on the model.

    \\alpha_i \\sim N(0, \\sigma_\\alpha^2), \\quad
    \\varepsilon_{it} \\sim N(0, \\sigma^2)

Priors and parameterization
---------------------------
:math:`\\sigma_\\alpha` gets a half-t prior (half-Cauchy by default) scaled to
``sd(y)``, and :math:`\\sigma^2` the package's ``Inv-Gamma(2, Var(y))``; the
NUTS and Gibbs paths share both (see :func:`_re_prior_values`).  NUTS samples
the effects non-centered by default, :math:`\\alpha_i = \\sigma_\\alpha z_i`
with :math:`z_i \\sim N(0, 1)`: the same prior and posterior as the centered
form, without the funnel between :math:`\\sigma_\\alpha` and
:math:`\\alpha` that makes NUTS diverge when the effects are weakly
identified.  ``centered=True`` restores the centered form, which mixes better
when every unit has many observations.

Data convention
---------------
Observations must be stacked time-first (time period changes slowest),
so that observation index ``i`` belongs to unit ``i % N``.  This matches
the convention used by all other panel classes in this package.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor import sparse as pts

from ..._backends.sampler_helpers import use_jax_likelihood
from ..._lazy_deps import pm, xr
from ...samplers._registry import register
from ..panel_base import SpatialPanelModel
from ..priors import (
    PanelOLSREPriors,
    PanelSARREPriors,
    PanelSDEMREPriors,
    PanelSEMREPriors,
)


def _re_prior_values(model, Z: np.ndarray) -> dict:
    """Resolved prior hyperparameters, shared by the NUTS and Gibbs paths.

    ``Z`` is the design ``beta`` multiplies (``[X, WX]`` for SDEM); the
    ``beta`` default is the data-scaled Gelman et al. (2008) prior on it.
    ``sigma_alpha_scale`` defaults to ``sd(y)``, so the half-t prior on
    :math:`\\sigma_\\alpha` means the same thing whatever units ``y`` is in.
    """
    names = list(model._model_coords()["coefficient"])
    beta_mu, beta_sigma = model._gelman_default_beta_prior(Z, names)
    p = model.priors
    return {
        "beta_mu": p.get("beta_mu", beta_mu),
        "beta_sigma": p.get("beta_sigma", beta_sigma),
        "sigma2_alpha": float(p.get("sigma2_alpha", 2.0)),
        "sigma2_beta": float(p.get("sigma2_beta", np.var(model._y))),
        "sigma_alpha_nu": float(p.get("sigma_alpha_nu", 1.0)),
        "sigma_alpha_scale": float(p.get("sigma_alpha_scale", np.std(model._y))),
    }


def _random_effects(pv: dict, centered: bool):
    """``sigma_alpha`` and the unit effects ``alpha``, inside a model context.

    The half-t prior is ``HalfCauchy`` when ``nu == 1``.  Non-centered,
    ``alpha`` is a ``Deterministic`` of the standardized ``alpha_z``, so the
    posterior carries ``alpha`` under either parameterization.
    """
    nu, scale = pv["sigma_alpha_nu"], pv["sigma_alpha_scale"]
    if nu == 1.0:
        sigma_alpha = pm.HalfCauchy("sigma_alpha", beta=scale)
    else:
        sigma_alpha = pm.HalfStudentT("sigma_alpha", nu=nu, sigma=scale)
    if centered:
        alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, dims="unit")
    else:
        alpha_z = pm.Normal("alpha_z", mu=0.0, sigma=1.0, dims="unit")
        alpha = pm.Deterministic("alpha", sigma_alpha * alpha_z, dims="unit")
    return sigma_alpha, alpha


def _re_gibbs_priors(pv: dict, rho_lower: float, rho_upper: float):
    """The Gibbs sampler's priors, built from the same resolved values."""
    from ...samplers.panel import REGibbsPriors

    return REGibbsPriors(
        beta_mu=pv["beta_mu"],
        beta_sigma=pv["beta_sigma"],
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        sigma2_alpha=pv["sigma2_alpha"],
        sigma2_beta=pv["sigma2_beta"],
        sigma_alpha_nu=pv["sigma_alpha_nu"],
        sigma_alpha_scale=pv["sigma_alpha_scale"],
    )


class _RandomEffectsMixin:
    """Adds the ``centered`` switch for the unit effects' parameterization."""

    def __init__(self, *, centered: bool = False, **kwargs):
        self.centered = bool(centered)
        super().__init__(**kwargs)


class OLSPanelRE(_RandomEffectsMixin, SpatialPanelModel):
    """Bayesian random effects panel regression (non-spatial).

    .. math::
        y_{it} = X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Accepts a
        :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix; legacy ``libpysal.weights.W`` is not accepted (use
        ``w.sparse``). Should be row-standardized. Unused in the RE
        likelihood but required by the base class for consistency
        (e.g. computing spatial lags for SDM/SDEM variants).
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, data-scaled (see
          :class:`~neighbayes.models.priors.BasePriors`).
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_nu`` (float, default 1.0): degrees of freedom of the
          half-t prior on :math:`\\sigma_\\alpha`; 1 is the half-Cauchy.
        - ``sigma_alpha_scale`` (float, default sd(y)): scale of that prior.
          See :class:`~neighbayes.models.priors.PanelREMixinPriors`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    centered : bool, default False
        Parameterization of the unit effects under NUTS.  The default samples
        them non-centered, :math:`\\alpha_i = \\sigma_\\alpha z_i`, which
        avoids the funnel between :math:`\\sigma_\\alpha` and
        :math:`\\alpha` when the effects are weakly identified (few periods,
        small :math:`\\sigma_\\alpha`).  ``True`` samples :math:`\\alpha`
        directly, which mixes better when every unit has many observations.
        The prior and the posterior are the same either way.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    Data are **not** demeaned — the random effects absorb the unit-level
    mean structure probabilistically.  This is the Bayesian analogue of
    the classical GLS random-effects estimator in ``prandom.m``.

    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelOLSREPriors
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    def __init__(self, **kwargs):
        kwargs.pop("model", None)  # RE always uses raw (pooled) data
        kwargs["effects"] = 0  # pooled — no FE transform
        super().__init__(**kwargs)
        # obs i → unit i % N  (time-first stacking)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for non-spatial random effects panel.

        Returns
        -------
        pymc.Model
        """
        pv = _re_prior_values(self, self._X)
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        sigma2_alpha, sigma2_beta = pv["sigma2_alpha"], pv["sigma2_beta"]

        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha, alpha = _random_effects(pv, self.centered)

            mu = pt.dot(self._X, beta) + alpha[unit_idx]
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)

        return model

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values including unit random effects.

        Returns
        -------
        np.ndarray
        """
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SARPanelRE(_RandomEffectsMixin, SpatialPanelModel):
    """Bayesian spatial lag panel model with unit random effects.

    .. math::
        y_{it} = \\rho (Wy)_{it} + X_{it}\\beta + \\alpha_i + \\varepsilon_{it}

    where :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)` are unit-level
    random effects and :math:`\\varepsilon_{it} \\sim N(0, \\sigma^2)`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, data-scaled (see
          :class:`~neighbayes.models.priors.BasePriors`).
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_nu`` (float, default 1.0): degrees of freedom of the
          half-t prior on :math:`\\sigma_\\alpha`; 1 is the half-Cauchy.
        - ``sigma_alpha_scale`` (float, default sd(y)): scale of that prior.
          See :class:`~neighbayes.models.priors.PanelREMixinPriors`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    centered : bool, default False
        Parameterization of the unit effects under NUTS.  The default samples
        them non-centered, :math:`\\alpha_i = \\sigma_\\alpha z_i`, which
        avoids the funnel between :math:`\\sigma_\\alpha` and
        :math:`\\alpha` when the effects are weakly identified (few periods,
        small :math:`\\sigma_\\alpha`).  ``True`` samples :math:`\\alpha`
        directly, which mixes better when every unit has many observations.
        The prior and the posterior are the same either way.
    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`; auto-selected
        (``"eigenvalue"`` for ``N <= 2000`` else ``"chebyshev"``) when
        ``None`` (default).
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelSARREPriors
    _jacobian_param: str | None = "rho"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_re")

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self) -> pm.Model:
        """Construct the PyMC model for SAR panel with random effects.

        Returns
        -------
        pymc.Model
        """
        rho_lower = self.priors.get("rho_lower", -1.0)
        rho_upper = self.priors.get("rho_upper", 1.0)
        pv = _re_prior_values(self, self._X)
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        sigma2_alpha, sigma2_beta = pv["sigma2_alpha"], pv["sigma2_beta"]

        logdet_fn = self._logdet_pytensor_fn
        unit_idx = self._unit_idx

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha, alpha = _random_effects(pv, self.centered)

            mu = rho * self._Wy + pt.dot(self._X, beta) + alpha[unit_idx]
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=self._y)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=self._y)
            pm.Potential("jacobian", logdet_fn(rho))

        return model

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        log_likelihood: bool = False,
    ) -> "xr.DataTree":
        """Sample posterior via 5-block RE Gibbs (β, σ², α, σ_α², ρ).

        NumPy-only; there is no JAX kernel for the RE sampler.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers. ``-1`` uses all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.

        Returns
        -------
        xr.DataTree
        """
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        from ...samplers.panel import GaussianSARREGibbs

        priors = _re_gibbs_priors(
            _re_prior_values(self, self._X),
            self._logdet_bounds.rho_min,
            self._logdet_bounds.rho_max,
        )

        gibbs = GaussianSARREGibbs(
            y=self._y,
            X=self._X,
            W_sparse=self._W_sparse_NT,
            Wy=self._Wy,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=list(self._feature_names),
            N=self._N,
            T=self._T,
            unit_idx=self._unit_idx,
            W_eigs=self._logdet_eigs,
            logdet_method=self._logdet_bounds.method,
        )

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            log_likelihood=log_likelihood,
        )
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values.

        Returns
        -------
        np.ndarray
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return rho * self._Wy + self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SEMPanelRE(_RandomEffectsMixin, SpatialPanelModel):
    """Bayesian spatial error panel model with unit random effects.

    .. math::
        y_{it} = X_{it}\\beta + \\alpha_i + u_{it}, \\quad
        u_{it} = \\lambda (Wu)_{it} + \\varepsilon_{it}

    Equivalently the spatially-filtered residual is i.i.d.:

    .. math::
        \\varepsilon_{it} = (I - \\lambda W)(y - X\\beta - \\alpha)_{it}
        \\sim N(0, \\sigma^2)

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`\\beta`, data-scaled (see
          :class:`~neighbayes.models.priors.BasePriors`).
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_nu`` (float, default 1.0): degrees of freedom of the
          half-t prior on :math:`\\sigma_\\alpha`; 1 is the half-Cauchy.
        - ``sigma_alpha_scale`` (float, default sd(y)): scale of that prior.
          See :class:`~neighbayes.models.priors.PanelREMixinPriors`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    centered : bool, default False
        Parameterization of the unit effects under NUTS.  The default samples
        them non-centered, :math:`\\alpha_i = \\sigma_\\alpha z_i`, which
        avoids the funnel between :math:`\\sigma_\\alpha` and
        :math:`\\alpha` when the effects are weakly identified (few periods,
        small :math:`\\sigma_\\alpha`).  ``True`` samples :math:`\\alpha`
        directly, which mixes better when every unit has many observations.
        The prior and the posterior are the same either way.
    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.

    **Identification of λ in SEM-RE models**

    The spatial error parameter :math:`\\lambda` is **weakly identified**
    when random effects :math:`\\alpha_i` are present.  The random effects
    absorb spatial correlation across units, making it difficult for the
    data to distinguish between :math:`\\lambda` (spatial error dependence)
    and :math:`\\sigma_\\alpha^2` (between-unit variance).  Both Gibbs and
    NUTS samplers will tend to estimate :math:`\\lambda` near zero even when
    the true value is moderate, because the posterior genuinely
    concentrates there.  This is a model identification issue, not a
    sampler bug.

    Possible remedies include:

    - Use fixed effects (``SEMPanelFE``) instead of random effects
    - Use the spatial Durbin error model (``SDEMPanelRE``), which adds WX terms
    - Use longer panels (:math:`T \\to \\infty`) which provide more
      information to separate :math:`\\lambda` from :math:`\\alpha`

    **Robust regression**

    When ``robust=True``, the spatially-filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} = (I - \\lambda W)(y - X\\beta - \\alpha_i) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelSEMREPriors
    _jacobian_param: str | None = "lam"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_re")

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SEM panel with random effects.

        Parameters
        ----------
        nuts_sampler :
            Resolved sampler.  When JAX-backed, the likelihood is registered
            via :class:`pymc.CustomDist` so PyMC's JAX path captures
            ``log_likelihood`` natively; otherwise the
            :func:`pymc.Potential` formulation is used.

        Returns
        -------
        pymc.Model
        """
        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        pv = _re_prior_values(self, self._X)
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        sigma2_alpha, sigma2_beta = pv["sigma2_alpha"], pv["sigma2_beta"]

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        unit_idx = self._unit_idx

        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs  # _logdet_pytensor_fn already includes T multiplier
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha, alpha = _random_effects(pv, self.centered)

            if jax_logp:
                X_const = pt.as_tensor_variable(self._X)
                y_const = pt.as_tensor_variable(self._y)
                unit_idx_const = pt.as_tensor_variable(unit_idx)

                def _eps(lam_, beta_, alpha_):
                    resid = y_const - pt.dot(X_const, beta_) - alpha_[unit_idx_const]
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu

                    def sempanel_re_logp(value, lam_, beta_, sigma_, alpha_, nu_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        nu,
                        logp=sempanel_re_logp,
                        observed=self._y,
                    )
                else:

                    def sempanel_re_logp(value, lam_, beta_, sigma_, alpha_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        logp=sempanel_re_logp,
                        observed=self._y,
                    )
            else:
                # epsilon = (I - lam*W)(y - X@beta - alpha_expanded)
                #         = resid - lam * W @ resid
                resid = self._y - pt.dot(self._X, beta) - alpha[unit_idx]
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

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        log_likelihood: bool = False,
    ) -> "xr.DataTree":
        """Sample posterior via 5-block RE Gibbs (β, σ², α, σ_α², λ).

        NumPy-only; there is no JAX kernel for the RE sampler.

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers. ``-1`` uses all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.

        Returns
        -------
        xr.DataTree
        """
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        from ...samplers.panel import GaussianSEMREGibbs

        priors = _re_gibbs_priors(
            _re_prior_values(self, self._X),
            self._logdet_bounds.rho_min,
            self._logdet_bounds.rho_max,
        )

        gibbs = GaussianSEMREGibbs(
            y=self._y,
            X=self._X,
            W_sparse=self._W_sparse_NT,
            priors=priors,
            logdet_fn=self._logdet_numpy_fn,
            logdet_vec_fn=self._logdet_numpy_vec_fn,
            feature_names=list(self._feature_names),
            N=self._N,
            T=self._T,
            unit_idx=self._unit_idx,
            W_eigs=self._logdet_eigs,
            logdet_method=self._logdet_bounds.method,
        )

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            log_likelihood=log_likelihood,
        )
        return self._idata

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean fitted values (on the observed y scale).

        Returns
        -------
        np.ndarray
        """
        beta = self._posterior_mean("beta")
        alpha = self._posterior_mean("alpha")
        return self._X @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior samples of direct, indirect, and total effects."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        ni = self._nonintercept_indices

        if isinstance(self, SARPanelRE):
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)
            direct_samples = mean_diag[:, None] * beta_draws
            total_samples = mean_row_sum[:, None] * beta_draws
            indirect_samples = total_samples - direct_samples

        elif isinstance(self, SEMPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        elif isinstance(self, OLSPanelRE):
            beta_draws = _get_posterior_draws(idata, "beta")[:, ni]
            direct_samples = beta_draws.copy()
            indirect_samples = np.zeros_like(beta_draws)
            total_samples = beta_draws.copy()

        return direct_samples, indirect_samples, total_samples


class SDEMPanelRE(_RandomEffectsMixin, SpatialPanelModel):
    """Bayesian spatial Durbin error panel model with unit random effects.

    .. math::
        y_{it} = X_{it}\\beta + (WX)_{it}\\theta + \\alpha_i + u_{it}, \\quad
        u_{it} = \\lambda (Wu)_{it} + \\varepsilon_{it}

    Combines the SDEM mean structure (covariates plus their spatial lags)
    with random unit effects :math:`\\alpha_i \\sim N(0, \\sigma_\\alpha^2)`
    and a spatially-correlated error term governed by :math:`\\lambda`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode.
    y : array-like, optional
        Stacked response of shape ``(N*T,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)``. Used to construct the
        ``WX`` block and the spatial filter on the disturbance.
        Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode.
    T : int, optional
        Number of time periods. Required in matrix mode.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu``, ``beta_sigma`` (array, default Gelman 2008): Normal
          prior on :math:`[\\beta, \\theta]`, data-scaled (see
          :class:`~neighbayes.models.priors.BasePriors`).
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma prior alpha for sigma2
        - ``sigma2_beta`` (float, default var(y)): InverseGamma prior beta for sigma2
          for :math:`\\sigma`.
        - ``sigma_alpha_nu`` (float, default 1.0): degrees of freedom of the
          half-t prior on :math:`\\sigma_\\alpha`; 1 is the half-Cauchy.
        - ``sigma_alpha_scale`` (float, default sd(y)): scale of that prior.
          See :class:`~neighbayes.models.priors.PanelREMixinPriors`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    centered : bool, default False
        Parameterization of the unit effects under NUTS.  The default samples
        them non-centered, :math:`\\alpha_i = \\sigma_\\alpha z_i`, which
        avoids the funnel between :math:`\\sigma_\\alpha` and
        :math:`\\alpha` when the effects are weakly identified (few periods,
        small :math:`\\sigma_\\alpha`).  ``True`` samples :math:`\\alpha`
        directly, which mixes better when every unit has many observations.
        The prior and the posterior are the same either way.
    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`; auto-selected
        when ``None`` (default).
    robust : bool, default False
        If True, replace the Normal innovation with Student-t.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. At least one column must be
        lagged; if no WX columns remain a :class:`ValueError` is
        raised. Pass a subset to restrict which variables receive a
        spatial lag.

    Notes
    -----
    The base-class ``model`` argument is not exposed; pooled mean
    structure (``model=0``) is used because unit heterogeneity is
    captured by the random effect rather than by within-unit demeaning.
    """

    _has_wx_in_beta = True
    _jacobian_param: str | None = "lam"
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    _priors_cls = PanelSDEMREPriors

    def __init__(self, **kwargs):
        kwargs.pop("model", None)
        kwargs["effects"] = 0  # pooled
        super().__init__(**kwargs)
        if not self._wx_column_indices:
            raise ValueError(
                "SDEMPanelRE requires at least one WX column. Pass "
                "`w_vars=[...]` to choose which regressors receive a spatial "
                "lag, or fit an SEMPanelRE model instead."
            )
        self._unit_idx = np.arange(self._N * self._T) % self._N

    def _model_coords(self) -> dict:
        coords = super()._model_coords()
        coords["unit"] = list(range(self._N))
        return coords

    def _build_pymc_model(self, nuts_sampler: str = "pymc") -> pm.Model:
        """Construct the PyMC model for SDEM panel with random effects.

        Parameters
        ----------
        nuts_sampler :
            Resolved sampler.  When JAX-backed, the likelihood is registered
            via :class:`pymc.CustomDist` so PyMC's JAX path captures
            ``log_likelihood`` natively; otherwise the
            :func:`pymc.Potential` formulation is used.
        """
        Z = np.hstack([self._X, self._WX])

        lam_lower = self.priors.get("lam_lower", -1.0)
        lam_upper = self.priors.get("lam_upper", 1.0)
        pv = _re_prior_values(self, Z)
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        sigma2_alpha, sigma2_beta = pv["sigma2_alpha"], pv["sigma2_beta"]

        logdet_fn = self._logdet_pytensor_fn
        W_pt = self._W_pt_sparse
        unit_idx = self._unit_idx

        n_obs = int(self._y.shape[0])
        inv_n = 1.0 / n_obs  # _logdet_pytensor_fn already includes T multiplier
        jax_logp = use_jax_likelihood(nuts_sampler)

        with pm.Model(coords=self._model_coords()) as model:
            lam = pm.Uniform("lam", lower=lam_lower, upper=lam_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma2 = pm.InverseGamma("sigma2", alpha=sigma2_alpha, beta=sigma2_beta)
            sigma = pm.Deterministic("sigma", pt.sqrt(sigma2))
            sigma_alpha, alpha = _random_effects(pv, self.centered)

            if jax_logp:
                Z_const = pt.as_tensor_variable(Z)
                y_const = pt.as_tensor_variable(self._y)
                unit_idx_const = pt.as_tensor_variable(unit_idx)

                def _eps(lam_, beta_, alpha_):
                    resid = y_const - pt.dot(Z_const, beta_) - alpha_[unit_idx_const]
                    return (
                        resid
                        - lam_ * pts.structured_dot(W_pt, resid[:, None]).flatten()
                    )

                if self.robust:
                    nu = self._nu

                    def sdempanel_re_logp(value, lam_, beta_, sigma_, alpha_, nu_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(
                            pm.StudentT.dist(nu=nu_, mu=0.0, sigma=sigma_), eps
                        )
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        nu,
                        logp=sdempanel_re_logp,
                        observed=self._y,
                    )
                else:

                    def sdempanel_re_logp(value, lam_, beta_, sigma_, alpha_):
                        eps = _eps(lam_, beta_, alpha_)
                        log_dens = pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma_), eps)
                        return log_dens + logdet_fn(lam_) * inv_n

                    pm.CustomDist(
                        "obs",
                        lam,
                        beta,
                        sigma,
                        alpha,
                        logp=sdempanel_re_logp,
                        observed=self._y,
                    )
            else:
                resid = self._y - pt.dot(Z, beta) - alpha[unit_idx]
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
        alpha = self._posterior_mean("alpha")
        Z = np.hstack([self._X, self._WX])
        return Z @ beta + alpha[self._unit_idx]

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior samples of direct/indirect/total effects (SDEM form)."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")
        k = self._X.shape[1]
        kw = self._WX.shape[1]
        beta1_draws = beta_draws[:, :k]
        beta2_draws = beta_draws[:, k : k + kw]

        mean_diag_w = float(self._W_sparse.diagonal().mean())
        mean_row_sum_w = float(self._W_sparse.sum() / self._W_sparse.shape[0])

        wx_idx = self._wx_column_indices
        direct_samples = beta1_draws[:, wx_idx] + mean_diag_w * beta2_draws
        total_samples = beta1_draws[:, wx_idx] + mean_row_sum_w * beta2_draws
        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples


# ---------------------------------------------------------------------------
# Gibbs registry entry
# ---------------------------------------------------------------------------


def _run_gaussian_re(
    model,
    *,
    draws,
    tune,
    chains,
    random_seed,
    thin,
    n_jobs,
    progressbar,
    backend,
    log_likelihood=False,
):
    """Registry runner for Gaussian panel random-effects Gibbs.

    Thin adapter over the per-class ``_fit_gibbs`` (SAR/SEM RE own their own
    5-block sampler).  RE Gibbs is NumPy-only — there is no JAX kernel and no
    ``slice_width`` option — so ``backend`` is always
    ``"numpy"`` and no family options are threaded.
    """
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        log_likelihood=log_likelihood,
    )


register(
    "gaussian",
    "panel_re",
    run=_run_gaussian_re,
    backends={"numpy"},
    skips_log_likelihood=True,
)
