"""Spatial Durbin Model (SDM).

y = rho * W @ y + X @ beta1 + W @ X @ beta2 + epsilon,  epsilon ~ N(0, sigma^2 I)

Combines a spatial lag on y (SAR) with spatially lagged covariates (SLX).
Jacobian log|I - rho*W| is required as in the SAR model.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin
from ..base import SpatialModel
from ..priors import SDMPriors


class SDM(GaussianLikelihoodMixin, SpatialModel):
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
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`[\\beta, \\theta]`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`[\\beta, \\theta]`.
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
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
    logdet_probe_check : bool, default True
        For ``"cheb_stochastic"``, set the number of Hutchinson probes from
        where the posterior lies.  Warmup starts on 50 probes; halfway through,
        the probes' own spread prices the bias they leave in the posterior mean
        of the spatial parameter, and the pool grows, to at most 200, until
        that bias is expected to stay under 0.0225 posterior sd.  The count
        used is recorded in ``idata.attrs["logdet_probes"]``, with a warning
        if the cap is reached first.  Probes cost setup time only; the cost of
        each draw does not depend on how many there are.
    robust : bool, default False
        If True, replace the Normal error with Student-t. See *Robust
        regression* below.
    w_vars : list of str, optional
        Names of X columns to spatially lag. By default all
        non-constant columns are lagged. Pass a subset to restrict
        which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``. SDM requires at least one
        WX column; if filtering eliminates all of them a ValueError is
        raised.

    Notes
    -----
    Direct, indirect and total effects of :math:`X` on :math:`y`
    incorporate both the local and lagged-X blocks via the spatial
    multiplier :math:`(I - \\rho W)^{-1}` and are reported by
    :meth:`spatial_effects`.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.
    """

    _priors_cls = SDMPriors
    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy", "WX")
    _jacobian_param: str | None = "rho"
    _has_wx_in_beta: bool = True
    _gibbs_class: str | None = "GaussianSARGibbs"
    _model_type: str = "sdm"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "cross_section")

    def _compute_spatial_effects_posterior(
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
