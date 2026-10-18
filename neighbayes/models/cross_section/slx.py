"""Spatial Lag X (SLX) Model.

y = X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

No spatial lag on y, so no Jacobian adjustment is needed and NUTS
converges without difficulty.
"""

from __future__ import annotations

import numpy as np

from .._design import gaussian
from .._mixins import GaussianLikelihoodMixin, GaussianPredictionMixin
from ..base import SpatialModel
from ..priors import SLXPriors


class SLX(GaussianPredictionMixin, GaussianLikelihoodMixin, SpatialModel):
    """Bayesian SLX (Spatial Lag X) model.

    Adds spatial lags of the regressors :math:`X` to a standard linear
    model, without any spatial dependence in :math:`y`:

    .. math::
        y = X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The sampled coefficient vector stacks the local and lagged-regressor
    blocks as :math:`[\\beta, \\theta]`. There is no Jacobian term
    because there is no spatial lag of :math:`y`.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. Intercept is included by default; suppress with
        ``"y ~ x - 1"``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``. Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode. DataFrame columns are
        preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``. Used to construct the
        ``WX`` block. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix; legacy ``libpysal.weights.W`` is
        not accepted (use ``w.sparse`` or
        ``libpysal.graph.Graph.from_W(w)``). Should be row-standardized.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` (array or float, optional): Normal prior mean for
          coefficients :math:`[\beta, \theta]`. Default is Gelman et al.
          (2008) weakly-informative mean (``mean(y)`` for intercept/constant
          columns, 0 for slopes).
        - ``beta_sigma`` (array or float, optional): Normal prior std for
          coefficients. Default is Gelman et al. (2008) weakly-informative
          scale: ``2.5 * sd(y)`` for intercept/constant columns and
          ``2.5 * sd(y) / sd(x_j)`` for each slope.
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
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
        If True, replace the Normal error with Student-t. See *Robust
        regression* below.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``durbin_vars=["income", "density"]``. SLX requires at least one
        WX column; if filtering eliminates all of them a ValueError is
        raised.

    Notes
    -----
    Direct effects equal :math:`\\beta` and indirect effects equal
    :math:`\\theta`. There is no global spillover multiplier because
    :math:`y` has no spatial autoregression.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _dispatch = classmethod(gaussian)

    _priors_cls = SLXPriors
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ("WX",)
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = True
    _gibbs_class: str | None = None
    _model_type: str = "slx"
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SLX model (no :math:`\\rho`) the impact for a lagged covariate
        is :math:`\\text{Direct}_k = \\beta_{1k} + \\beta_{2k}\\,
        \\overline{\\text{diag}}(W)` and :math:`\\text{Total}_k = \\beta_{1k}
        + \\beta_{2k}\\,\\overline{\\text{rowsum}}(W)`; effects are reported for
        the spatially lagged covariates only.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each of
            shape ``(G, k_wx)``.
        """
        return self._slx_effects()

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values.
        """
        k = self._X.shape[1]
        beta = self._posterior_mean("beta")
        Z = np.hstack([self._X, self._WX])
        if beta.shape[0] != 2 * k:
            raise ValueError("Unexpected beta dimension for SLX fitted mean.")
        return Z @ beta
