"""Spatial panel model classes."""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..panel_base import SpatialPanelModel
from ..priors import (
    PanelOLSPriors,
    PanelSARPriors,
    PanelSDEMPriors,
    PanelSDMPriors,
    PanelSEMPriors,
    PanelSLXPriors,
)


class OLSPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian pooled and fixed-effects linear panel regression.

    Implements the Gaussian panel model

    .. math::

        y_{it} = x_{it}'\\beta + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2),

    where the included effects depend on ``model``: ``0`` pooled,
    ``1`` unit effects, ``2`` time effects, ``3`` two-way effects. The
    within transformation is handled by
    :class:`~neighbayes.models.panel_base.SpatialPanelModel` before the
    likelihood is evaluated.

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
        Stacked design matrix of shape ``(N*T, k)``. Required in
        matrix mode. DataFrame columns are preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)`` (preferred) or
        ``(N*T, N*T)`` block-diagonal. Accepted for API consistency
        with the other panel models but does not enter the OLS
        likelihood; required if downstream Bayesian LM diagnostics
        will be run.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    This is the aspatial baseline for panel LM diagnostics and panel model
    comparison. The spatial weights object ``W`` is accepted for API
    consistency but does not enter the likelihood.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelOLSPriors
    _jacobian_param: str | None = None
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    # _fitted_mean_from_posterior inherited from SharedSpatialMethods
    # (X @ beta, no rho, no WX)

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        OLS panel has no spatial structure: Direct = beta, Indirect = 0.
        """
        return self._sem_effects()


class SARPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial-lag panel regression.

    Implements

    .. math::

        y_{it} = \\rho \\sum_j w_{ij} y_{jt} + x_{it}'\\beta + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2),

    with the same pooled, unit-effect, time-effect, or two-way panel
    transformation selected by ``model`` as in :class:`OLSPanelFE`.

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
        Spatial weights of shape ``(N, N)`` (preferred) or
        ``(N*T, N*T)``. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix; legacy ``libpysal.weights.W`` is
        not accepted (use ``w.sparse``). Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.

    Notes
    -----
    The likelihood combines the Gaussian observation density with the
    spatial Jacobian term associated with :math:`I - \\rho W`.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelSARPriors
    _jacobian_param: str | None = "rho"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sar"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters."""
        # Delegate to SharedSpatialMethods default (handles intercept_dropped).
        return super()._fitted_mean_from_posterior()

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SAR panel impacts use the same eigenvalue-based formulas as
        cross-sectional SAR, applied per draw.
        """
        return self._sar_effects()


class SEMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial-error panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\alpha_i + \\tau_t + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    Spatial dependence enters through the disturbance, while the panel
    transformation selected by ``model`` absorbs pooled, unit, time, or
    two-way effects before likelihood evaluation.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`\\beta`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.

    Notes
    -----
    Direct effects equal :math:`\\beta`; indirect effects are zero
    because spatial dependence enters only through the disturbance.

    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)\\bigl(y_t - X_t \\beta - \\alpha\\bigr) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = PanelSEMPriors
    _jacobian_param: str | None = "lam"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sem"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients."""
        return super()._fitted_mean_from_posterior()

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SEM panel has no spatial multiplier on X: Direct = beta, Indirect = 0.
        """
        return self._sem_effects()


class SDMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial Durbin panel regression.

    Implements

    .. math::

        y_{it} = \\rho \\sum_j w_{ij} y_{jt} + x_{it}'\\beta
        + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    The coefficient vector sampled by the model stacks the local and
    lagged-regressor blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\rho`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = "rho"

    _priors_cls = PanelSDMPriors
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sdm"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        return super()._fitted_mean_from_posterior()

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SDM panel impacts use the same eigenvalue-based formulas as
        cross-sectional SDM, applied per draw.
        """
        return self._sdm_effects()


class SDEMPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian spatial Durbin error panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + u_{it},
        \\qquad u_{it} = \\lambda \\sum_j w_{ij} u_{jt} + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    The sampled coefficient vector stacks the local and lagged-covariate
    blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Should
        be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -1.0): Lower bound of Uniform
          prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of Uniform
          prior on :math:`\\lambda`.
        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal innovation with Student-t. See
        *Robust regression* below.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the spatially filtered error distribution is
    changed from Normal to Student-t, yielding a model that is robust to
    heavy-tailed outliers:

    .. math::

        \\varepsilon_t = (I - \\lambda W)\\bigl(y_t - X_t \\beta - (W X_t)\\theta - \\alpha\\bigr) \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = "lam"

    _priors_cls = PanelSDEMPriors
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "panel_fe")
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sdem"

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        return super()._fitted_mean_from_posterior()

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SDEM panel impacts match SLX form (no rho multiplier).
        """
        return self._slx_effects()


class SLXPanelFE(GaussianLikelihoodMixin, SpatialPanelModel):
    """Bayesian SLX panel regression.

    Implements

    .. math::

        y_{it} = x_{it}'\\beta + \\Bigl(\\sum_j w_{ij} x_{jt}\\Bigr)'\\theta
        + \\alpha_i + \\tau_t + \\varepsilon_{it},
        \\qquad \\varepsilon_{it} \\sim \\mathcal{N}(0, \\sigma^2).

    There is no contemporaneous spatial lag on :math:`y`, so no Jacobian
    adjustment is required. The coefficient vector stacks the local and
    lagged-covariate blocks as :math:`[\\beta, \\theta]`.

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
        Spatial weights of shape ``(N, N)`` or ``(N*T, N*T)``. Used
        to construct the ``WX`` block. Should be row-standardized.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    model : int, default 0
        Fixed-effects specification: ``0`` pooled, ``1`` unit FE,
        ``2`` time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (array, default Gelman 2008): Normal prior mean
          for :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (array, default Gelman 2008): Normal prior std
          for :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): InverseGamma shape for
          :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): InverseGamma
          scale for :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : {"exact", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is approximated over rho: evaluated exactly at
        every rho, interpolated (Chebyshev, AAA rational, or cubic spline)
        through exact values, or expanded stochastically with no
        factorization.  ``None`` (default) selects from the size and symmetry
        of ``W``: ``"exact"`` for ``n <= 500``; ``"aaa"`` up to
        ``n = 60000`` (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the
        factorization's fill-in would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "eig", "chol", "lu"}, default "auto"
        What supplies the exact values: a dense eigendecomposition, a sparse
        Cholesky (undirected ``W``; scikit-sparse CHOLMOD) or a sparse LU.
        ``"auto"`` takes the first of ``"eig"`` (``n <= 500``,
        ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``), ``"chol"`` (undirected ``W``) and
        ``"lu"`` that the method is implemented on.  Ignored by
        ``"stochastic"``.  A pair with no implementation (``"spline"`` on
        ``"chol"``, ``"exact"`` on ``"lu"``) raises.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See
        *Robust regression* below.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged.

    Notes
    -----
    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon_{it} \\sim t_\\nu(0, \\sigma^2)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _has_wx_in_beta: bool = True
    _jacobian_param: str | None = None
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    _priors_cls = PanelSLXPriors

    # _fitted_mean_from_posterior inherited from SharedSpatialMethods
    # ([X, WX] @ beta, no rho)

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        SLX panel impacts are linear in beta (no rho multiplier).
        """
        return self._slx_effects()
