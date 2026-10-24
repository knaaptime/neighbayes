r"""The Gaussian flow families: :class:`OLSFlow`, :class:`SARFlow`, :class:`SEMFlow`.

Each is one class (see :class:`~neighbayes.models._mixins._designs.FlowDesigns`).
For every period :math:`t` of ``T`` (``T=1`` is the cross-section) the
:math:`N = n^2` flows follow

.. math::

    y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta
        + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_N),

with :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n` and
:math:`W_w = W \otimes W` (SARFlow); the same filter on the disturbance
(SEMFlow, parameters ``lam_*``); or no filter (OLSFlow).  ``separable=True``
pins :math:`\rho_w = -\rho_d\rho_o`, so the filter factors as
:math:`L_o \otimes L_d` and its log-determinant is
:math:`n\log|L_d| + n\log|L_o|`.  Panel fixed effects follow Lee and Yu (2010)
(see :class:`~neighbayes.models.flow.FlowModel`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._backends.sampler_helpers import select_backend
from ..._lazy_deps import pm, xr
from ..._ops import kron_solve_matrix
from ...graph import flow_lags
from .._mixins._designs import FlowDesigns
from ._flow import FlowModel, _Aspatial

_RESOLVENT_ONLY = (
    "{name} with the unrestricted filter samples by the resolvent-gradient "
    "sampler only; pass separable=True for a PyMC (NUTS) model."
)


class _ResolventFit:
    """``fit`` of the unrestricted filter: the resolvent sampler, or NUTS."""

    def _fit_static(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str | None = None,
        backend: str | None = None,
        step_size: float = 5e-4,
        n_probes: int = 48,
        n_quad: int = 8,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Parameters
        ----------
        sampler : {"gibbs", "nuts", None}, default None
            ``"gibbs"`` (the default) runs the resolvent-gradient sampler, MALA
            on the spatial parameters within conjugate Gibbs for ``β, σ²``,
            whose accuracy improves with the flow sample size ``N``;
            ``"nuts"`` runs PyMC.  ``robust=True`` takes NUTS.
        backend : str, optional
            For Gibbs, what runs the resolvent estimator of the
            log-determinant (the model's ``logdet_method``): ``"auto"``
            (default; JAX when it is installed, else NumPy), ``"jax"`` or
            ``"numpy"``; the exact trace series runs on NumPy.  For NUTS,
            ``"pymc"`` (the default), ``"nutpie"``, ``"numpyro"`` or
            ``"blackjax"``.
        step_size, n_probes, n_quad : float/int
            Resolvent sampler parameters (Gibbs path only).
        n_jobs : int, default -1
            Parallel workers for the Gibbs path (``-1`` = all CPUs).
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow-period) for ``az.loo``,
            on either sampler.  Off by default, as in PyMC.
        """
        if sampler is None:
            # The resolvent sampler is Gaussian; Student-t errors need NUTS.
            sampler = "nuts" if self.robust else "gibbs"
        elif sampler == "gibbs" and self.robust:
            raise NotImplementedError(
                "robust=True has no Gibbs sampler for flows; use sampler='nuts'."
            )
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
        self._pymc_model = None
        self._idata = self._sample_resolvent(
            draws=draws,
            tune=tune,
            chains=chains,
            step_size=step_size,
            n_probes=n_probes,
            logdet_method=self._flow_logdet_choice(backend, sample_kwargs),
            n_quad=n_quad,
            coord_names=list(self._feature_names) or None,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            compute_log_likelihood=bool(
                (idata_kwargs or {}).get("log_likelihood", False)
            ),
        )
        return self._idata

    def _resolvent_kwargs(self) -> dict:
        return dict(
            T=self._T,
            nonnegative_dependence=self.nonnegative_dependence,
            fe_dims=self._flow_fe_dims,
            priors=self._flow_gaussian_priors(),
        )


def _observe(model, mu, sigma, y_t) -> None:
    """The Normal (or, ``robust=True``, fixed-ν Student-t) observation model."""
    if model.robust:
        pm.StudentT("obs", nu=model._nu, mu=mu, sigma=sigma, observed=y_t)
    else:
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)


def _require_separable_logdet(model) -> None:
    if model._separable_logdet_fn is None:
        raise RuntimeError(
            f"{type(model).__name__}(separable=True) requires precomputed logdet "
            "data; initialize with a separable logdet_method (None/auto, "
            "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
        )


_COMMON_PARAMS = """
    Parameters
    ----------
    y : array-like
        The flows: ``(n, n)`` or ``(n²,)`` for a cross-section; ``(T, n, n)``,
        ``(T, n²)`` or ``(n² T,)`` for a panel, periods stacked time-first.
    X : np.ndarray or pandas.DataFrame, shape (n² T, p)
        Origin-destination design, typically from
        :func:`~neighbayes.graph.flow_design_matrix`; DataFrame columns are
        kept as feature names.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized regional weights on *n* units.
    T : int, default 1
        Number of periods; ``1`` is the cross-section.
    pair_effects : {None, "fixed"}, default None
        Fixed origin-destination pair effects (panels only), concentrated out
        by the within transform; columns fixed within pairs are absorbed.
    time_effects : bool, default False
        Fixed period effects (panels only).
    col_names : list of str, optional
        Column labels for ``X``.
    k : int, optional
        Number of destination/origin variable pairs, inferred from
        ``dest_*``/``orig_*`` column names when omitted.
    shared_od_covariates : bool, optional
        Whether the destination and origin design blocks are identical;
        detected when ``None``.  Sets the default ``impacts(mode="auto")``.
    robust : bool, default False
        Student-t errors with ``ν`` fixed at ``priors["nu"]`` (default 4,
        LeSage's ``rval``), sampled by NUTS."""


class OLSFlow(FlowDesigns, _Aspatial, FlowModel):
    r"""Non-spatial Bayesian origin-destination gravity model.

    The conventional log-linear gravity specification of
    :cite:t:`thomas-agnan2014SpatialEconometric` (eq. 83.2) with no spatial
    lag terms, :math:`y_t = X_t\beta + \varepsilon_t`, sampled by NUTS.  It
    is the null model of the flow LM tests (``spatial_diagnostics()``).
    """

    __doc__ = (
        __doc__
        + _COMMON_PARAMS
        + """
    priors : dict, optional
        ``beta_mu``, ``beta_sigma`` (default Gelman et al. 2008, scaled to
        ``sd(y)`` and each column's sd), ``sigma2_alpha``, ``sigma2_beta``
        (InverseGamma on ``σ²``, default 2 and ``Var(y)``) and ``nu``.
    """
    )

    _DESIGNS = {"static": {}}
    _robust_nuts = True

    def _build_pymc_model_static(self) -> pm.Model:
        pv = self._flow_gaussian_priors()
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)
            _observe(self, pt.dot(X_t, beta), sigma, y_t)
        return model


class SARFlow(FlowDesigns, _ResolventFit, FlowModel):
    r"""Bayesian SAR origin-destination flow model.

    .. math::

        y_t = \rho_d W_d y_t + \rho_o W_o y_t + \rho_w W_w y_t + X_t\beta
            + \varepsilon_t

    The unrestricted flow's ``fit()`` runs the resolvent-gradient sampler;
    ``sampler="nuts"`` runs PyMC with the exact trace-series Jacobian.
    ``separable=True`` pins :math:`\rho_w = -\rho_d\rho_o` and samples by NUTS.
    """

    __doc__ = (
        __doc__
        + _COMMON_PARAMS
        + """
    separable : bool, default False
        Pin ``ρ_w = -ρ_d ρ_o``.
    logdet_method : str, optional
        The unrestricted filter takes ``"trace_series"`` (exact, from trace
        moments of ``W``) or ``"resolvent"`` (the stochastic resolvent
        estimator, run on ``fit(backend=...)``); ``None`` takes the first for
        an undirected ``W`` and the second for a directed one.  The separable
        filter takes the single-``W`` methods (``"exact"``, ``"cheb"``,
        ``"aaa"``, ``"spline"``, ``"stochastic"``) with ``logdet_decomp``.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the separable filter's exact log-determinant values.
    nonnegative_dependence : bool, default True
        The unrestricted filter's prior: ``ρ_d, ρ_o, ρ_w ≥ 0`` with sum at
        most 1 (``pm.Dirichlet("rho_simplex", a=ones(4))``); if False,
        independent ``Uniform(rho_lower, rho_upper)`` priors with a
        quadratic-wall stability potential.  The separable filter always takes
        Uniform priors on ``ρ_d`` and ``ρ_o``.
    priors : dict, optional
        ``beta_mu``, ``beta_sigma``, ``sigma2_alpha``, ``sigma2_beta`` (as
        :class:`OLSFlow`), ``rho_lower``, ``rho_upper`` (default ±1, ±0.999
        separable) and ``nu``.
    """
    )

    _DESIGNS = {
        "static": {"_flow_filter": "unrestricted"},
        "separable": {"_flow_filter": "separable"},
    }
    _robust_nuts = True

    # -- unrestricted ------------------------------------------------------

    def _sample_resolvent(self, **kwargs) -> xr.DataTree:
        from ...samplers.gaussian._flow_resolvent import sample_flow_resolvent

        return sample_flow_resolvent(
            self._W_sparse, self._y, self._X, **self._resolvent_kwargs(), **kwargs
        )

    def _build_pymc_model_static(self) -> pm.Model:
        from ..._ops import FlowLogdetOp

        pv = self._flow_gaussian_priors()
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.nonnegative_dependence:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential("stability", pt.switch(slack > 0.0, 0.0, -1e6 * slack**2))

            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)
            # A y = X β + ε, so y | ρ is Normal about the lagged terms plus X β,
            # with the change of variables carried by log|A| per period.
            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            _observe(self, mu, sigma, y_t)
            logdet, _ = FlowLogdetOp(self._trace_logdet())(rho_d, rho_o, rho_w)
            pm.Potential(
                "jacobian", self._flow_jacobian(logdet, 1 - rho_d - rho_o - rho_w, pt)
            )
        return model

    def _compute_jacobian_log_det_static(self, posterior) -> np.ndarray:
        return self._jacobian_unrestricted(posterior)

    def _compute_impacts_posterior_static(self, draws: Optional[int] = None) -> dict:
        return self._effects_unrestricted(draws)

    # -- separable ---------------------------------------------------------

    def _build_pymc_model_separable(self) -> pm.Model:
        _require_separable_logdet(self)
        pv = self._flow_gaussian_priors()
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)
            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            _observe(self, mu, sigma, y_t)
            pm.Potential(
                "jacobian",
                self._flow_jacobian(
                    self._separable_logdet_fn(rho_d, rho_o),
                    (1 - rho_d) * (1 - rho_o),
                    pt,
                ),
            )
        return model

    def _compute_jacobian_log_det_separable(self, posterior) -> np.ndarray:
        return self._jacobian_separable(posterior)

    def _compute_impacts_posterior_separable(self, draws: Optional[int] = None) -> dict:
        return self._effects_separable(draws)


class SEMFlow(FlowDesigns, _ResolventFit, FlowModel):
    r"""Bayesian spatial-error origin-destination flow model.

    .. math::

        y_t = X_t\beta + u_t, \qquad
        u_t = \lambda_d W_d u_t + \lambda_o W_o u_t + \lambda_w W_w u_t
            + \varepsilon_t

    The mean is :math:`X_t\beta`, so the effects are the closed form of
    :class:`OLSFlow` (Thomas-Agnan and LeSage 2014, Table 83.1).  The
    unrestricted flow's ``fit()`` runs the resolvent-gradient sampler (GLS
    Gibbs for ``β, σ²``); ``separable=True`` pins
    :math:`\lambda_w = -\lambda_d\lambda_o` and samples by NUTS.
    """

    __doc__ = (
        __doc__
        + _COMMON_PARAMS
        + """
    separable : bool, default False
        Pin ``λ_w = -λ_d λ_o``.  Only the separable flow takes ``robust``.
    logdet_method, logdet_decomp, nonnegative_dependence
        As :class:`SARFlow`, for ``λ``.
    priors : dict, optional
        As :class:`SARFlow`, with ``lam_lower``, ``lam_upper``.
    """
    )

    _DESIGNS = {
        "static": {"_flow_filter": "unrestricted"},
        "separable": {"_flow_filter": "separable", "_robust_nuts": True},
    }
    _spatial_prefix = "lam"

    def _init_static(self) -> None:
        # Lags of the (demeaned) design: constants, precomputed once.
        self._Wd_X, self._Wo_X, self._Ww_X = flow_lags(
            self._W_sparse, self._X.astype(np.float64), T=self._T
        )

    _init_separable = _init_static

    def _compute_impacts_posterior(self, draws: Optional[int] = None) -> dict:
        return self._effects_aspatial(draws)

    # -- unrestricted ------------------------------------------------------

    def _sample_resolvent(self, **kwargs) -> xr.DataTree:
        from ...samplers.gaussian._flow_resolvent import sample_sem_flow_resolvent

        return sample_sem_flow_resolvent(
            self._W_sparse, self._y, self._X, **self._resolvent_kwargs(), **kwargs
        )

    def _build_pymc_model_static(self) -> pm.Model:
        raise NotImplementedError(_RESOLVENT_ONLY.format(name="SEMFlow"))

    def _simulate_y_rep_static(
        self, lam_d, lam_o, lam_w, beta, sigma, rng
    ) -> np.ndarray:
        """``y_rep,t = X_t β + B^{-1} ε_t``."""
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        eps = rng.normal(scale=float(sigma), size=(self._N, self._T))
        return Xb + self._solve_A(lam_d, lam_o, lam_w, eps).T.reshape(-1)

    # -- separable ---------------------------------------------------------

    def _build_pymc_model_separable(self) -> pm.Model:
        _require_separable_logdet(self)
        pv = self._flow_gaussian_priors()
        lam_lower = self.priors.get("lam_lower", -0.999)
        lam_upper = self.priors.get("lam_upper", 0.999)
        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
            lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
            lam_w = pm.Deterministic("lam_w", -lam_d * lam_o)
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)
            # B(y - X β) = ε, written as y | λ about its filtered lags.
            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            _observe(self, mu, sigma, y_t)
            pm.Potential(
                "jacobian",
                self._flow_jacobian(
                    self._separable_logdet_fn(lam_d, lam_o),
                    (1 - lam_d) * (1 - lam_o),
                    pt,
                ),
            )
        return model

    def _compute_jacobian_log_det_separable(self, posterior) -> np.ndarray:
        return self._jacobian_separable(posterior)

    def _simulate_y_rep_separable(
        self, lam_d, lam_o, lam_w, beta, sigma, rng
    ) -> np.ndarray:
        """``y_rep,t = X_t β + (L_o ⊗ L_d)^{-1} ε_t`` by two ``n × n`` solves."""
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - lam_d * self._W_sparse).tocsr()
        Lo = (I_n - lam_o * self._W_sparse).tocsr()
        eps = rng.normal(scale=float(sigma), size=(self._N, self._T))
        return Xb + kron_solve_matrix(Lo, Ld, eps, n).T.reshape(-1)
