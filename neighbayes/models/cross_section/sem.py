"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..base import SpatialModel
from ..priors import SEMPriors


class SEM(GaussianLikelihoodMixin, SpatialModel):
    """Bayesian Spatial Error Model.

    Spatial dependence enters through the disturbance via the
    autoregressive parameter :math:`\\lambda`:

    .. math::
        y = X\\beta + u, \\quad u = \\lambda Wu + \\varepsilon,
        \\quad \\varepsilon \\sim N(0, \\sigma^2 I).

    The likelihood includes the spatial Jacobian
    :math:`\\log|I - \\lambda W|` so that posterior inference on
    :math:`\\lambda` is exact.

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

        - ``lam_lower`` (float, default -1.0): Lower bound of the
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 1.0): Upper bound of the
          Uniform prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\\beta`.
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`. ``None`` (default)
        auto-selects by size: ``"eigenvalue"`` for ``n <= 500``; for
        ``500 < n <= 60000``, ``"cheb_cholesky"`` (exact, sparse Cholesky
        at Chebyshev nodes) when ``W`` is symmetric else ``"aaa"`` (AAA
        rational approximation); ``"cheb_stochastic"`` for ``n > 60000``.
        Explicit opt-ins: ``"chebyshev"`` (Barry-Pace) and ``"slq"``
        (stochastic Lanczos quadrature).
    logdet_refit : bool, default True
        Rebuild the log-determinant interpolant halfway through warmup, on
        the range the chains have found rather than the interval implied by
        the prior.  A warmup posterior is typically one to two orders of
        magnitude narrower than the prior, which needs far fewer interpolation
        nodes and drives the approximation error over the posterior's support
        down to the factorization's roundoff floor.  Applies to
        ``"cheb_cholesky"``, ``"lu_cheb"``, ``"aaa"`` and ``"chol_aaa"``;
        ignored otherwise.

        The interpolant is only valid on its interval, so the refit window
        becomes the sampler's support.  The window is padded by
        ``logdet_refit_pad_sd`` warmup standard deviations, recorded in
        ``idata.attrs["logdet_refit_window"]``, and a warning is raised if
        the retained draws ever reach an edge the refit introduced.
    logdet_refit_pad_sd : float, default 10.0
        Padding for the refit window, in warmup posterior standard
        deviations.  At the default the truncated tail is ~1e-23 under
        normality, and the padding costs a node or two at most.
    logdet_aaa_check : bool, default True
        For the AAA methods (``"aaa"``, ``"chol_aaa"``), set the number of
        exact factorizations from where the posterior lies.  Warmup starts on a
        14-node fit; halfway through, nodes are added only if the warmup
        posterior lies closer to a singularity of the Jacobian than four times
        the distance at which the fit's poles resolve it.  The count used is
        recorded in ``idata.attrs["logdet_aaa_nodes"]``.  Off, or on a path
        without a warmup midpoint, the count is fixed by the prior interval:
        14 nodes within ``|ρ| ≤ 0.9``, 18 otherwise.
    robust : bool, default False
        If True, replace the Normal disturbance with Student-t. See
        *Robust regression* below.

    Notes
    -----
    Because spatial dependence enters only through the disturbance,
    direct effects equal :math:`\\beta` and indirect effects are zero.

    **Robust regression**

    When ``robust=True``, the spatially-filtered innovation is
    Student-t:

    .. math::

        \\varepsilon = (I - \\lambda W)(y - X\\beta) \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = SEMPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "lam"
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sem"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "cross_section")

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        For the SEM model the spatial multiplier does not apply to :math:`X`
        directly, so :math:`\\text{Direct}_k = \\beta_k`,
        :math:`\\text{Indirect}_k = 0`, and :math:`\\text{Total}_k = \\beta_k`.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)``, each of
            shape ``(G, k)``.
        """
        return self._sem_effects()
