"""Spatial Durbin Model (SDM).

y = rho * W @ y + X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines a spatial lag on y (SAR) with spatially lagged covariates (SLX).
Jacobian log|I - rho*W| is required as in the SAR model.
"""

from __future__ import annotations

import numpy as np

from .._design import gaussian
from .._mixins import GaussianLikelihoodMixin, GaussianPredictionMixin
from ..base import SpatialModel
from ..priors import SDMPriors


class SDM(GaussianPredictionMixin, GaussianLikelihoodMixin, SpatialModel):
    """Bayesian Spatial Durbin Model.

    Combines a spatial lag of :math:`y` with spatial lags of the
    regressors :math:`X`:

    .. math::
        y = \\rho Wy + X\\beta + WX\\theta + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The sampled coefficient vector stacks the local and lagged-regressor
    blocks as :math:`[\\beta, \\theta]`. The likelihood includes the
    spatial Jacobian :math:`\\log|I - \\rho W|`.

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
        Spatial weights of shape ``(n, n)``. Accepts a
        :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
        matrix. The legacy :class:`libpysal.weights.W` object is **not**
        accepted; pass ``w.sparse`` or ``libpysal.graph.Graph.from_W(w)``.
        Should be row-standardized; a :class:`UserWarning` is raised
        otherwise.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -1.0): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 1.0): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu``, ``beta_sigma`` (float or array, default Gelman et al.
          2008): Normal prior on :math:`[\\beta, \\theta]`, scaled to ``sd(y)`` and each column's sd.
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
    logdet_adapt : bool, default True
        Halfway through warmup, let the Gibbs sampler fit the log-determinant
        to where the chains have found the posterior rather than to the prior
        interval.  An interpolant on Cholesky or LU values (``"cheb"``,
        ``"aaa"``) is refit on that range, padded by ten warmup standard
        deviations; the window becomes the sampler's support, is recorded in
        ``idata.attrs["logdet_refit_window"]``, and a warning is raised if the
        retained draws reach an edge it introduced.  ``"aaa"`` also adds exact
        factorizations when the posterior lies near a singularity
        (``idata.attrs["logdet_aaa_nodes"]``).  ``"stochastic"`` grows its
        Hutchinson probes, from 50 to at most 200, until the bias they leave
        in the posterior mean of the spatial parameter is expected to stay
        under 0.0225 posterior sd (``idata.attrs["logdet_probes"]``).
        ``False`` keeps the fit on the prior interval.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See *Robust
        regression* below.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``durbin_vars=["income", "density"]``. SDM requires at least one
        WX column; if filtering eliminates all of them a ValueError is
        raised.

    Notes
    -----
    Direct, indirect and total effects of :math:`X` on :math:`y`
    incorporate both the local and lagged-X blocks via the spatial
    multiplier :math:`(I - \\rho W)^{-1}` and are reported by
    :meth:`impacts`.

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

    _priors_cls = SDMPriors
    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy", "WX")
    _jacobian_param: str | None = "rho"
    _has_wx_in_beta: bool = True
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sdm"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "cross_section")

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SDM model the spatial multiplier acts on both :math:`X` and
        :math:`WX`: :math:`S_k = (I-\\rho W)^{-1}(\\beta_{1k} I + \\beta_{2k} W)`.
        The two diagonal traces ride the resolvent identities
        :math:`\\text{tr}(S)/n = 1 - (\\rho/n) g` and
        :math:`\\text{tr}(SW)/n = -g/n` (``g = d/drho log|I - rho W|``), so
        no O(n³) eigendecomposition is triggered.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each of
            shape ``(G, k_wx)``.
        """
        return self._sdm_effects()

    # _fitted_mean_from_posterior inherited from SharedSpatialMethods
    # (rho * Wy + [X, WX] @ beta via _jacobian_param + _has_wx_in_beta)
