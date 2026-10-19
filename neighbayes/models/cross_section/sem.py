"""Spatial Error Model (SEM).

y = X @ beta + u,  u = lambda * W @ u + epsilon,  epsilon ~ N(0, sigma^2 I)

Equivalently: (I - lambda*W)(y - X@beta) = epsilon
Likelihood: epsilon ~ N(0, sigma^2 I), plus Jacobian log|I - lambda*W|.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin, GaussianPredictionMixin
from .._mixins._designs import GaussianDesigns
from ..base import SpatialModel
from ..panel._dynamic import _SEMDynamic
from ..panel._re import _SEMRandomEffects
from ..priors import PanelSEMDynamicPriors, PanelSEMREPriors, SEMPriors


class SEM(
    GaussianDesigns,
    _SEMRandomEffects,
    _SEMDynamic,
    GaussianPredictionMixin,
    GaussianLikelihoodMixin,
    SpatialModel,
):
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
        - ``beta_mu``, ``beta_sigma`` (float or array, default Gelman et al.
          2008): Normal prior on :math:`\\beta`, scaled to ``sd(y)`` and each column's sd.
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
        If True, replace the Normal disturbance with Student-t. See
        *Robust regression* below.

    time_col, unit_col : str, optional
        Formula-mode panel: the period and unit columns of ``data``.  Given
        time, the model is a panel, stacked by period, then unit.
    N, T : int, optional
        Matrix-mode panel: the numbers of units and periods; ``T=1`` is the
        cross-section.
    unit_effects : {None, "fixed", "random"}, default None
        Panel unit effects: ``"fixed"`` applies the within transformation
        (with Lee and Yu's 2010 likelihood dimensions), ``"random"`` adds a normal
        effect per unit, ``unit_effect`` with sd ``unit_effect_sd`` (priors
        ``unit_effect_sd_nu`` and ``unit_effect_sd_scale``, half-Cauchy on ``sd(y)``
        by default).
    time_effects : bool, default False
        Fixed time effects (panels).
    dynamic : bool, default False
        Add the time lag :math:`\\phi y_{t-1}` (priors ``phi_lower``,
        ``phi_upper``; the ``lam`` bounds default to ``±0.95``).  A dynamic
        panel takes no unit effects (Nickell 1981).
    centered : bool, default False
        With random unit effects, sample them centered under NUTS rather than
        as :math:`\\alpha_i = \\sigma_\\alpha z_i`; better when every unit has
        many observations.

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

    _DESIGNS = {
        "static": {},
        "random": {
            "_priors_cls": PanelSEMREPriors,
            "_gibbs_class": None,
            "_gibbs_key": ("gaussian", "panel_re"),
        },
        "dynamic": {
            "_priors_cls": PanelSEMDynamicPriors,
            "_jacobian_param": None,
            "_gibbs_class": None,
            "_gibbs_key": None,
            "_likelihood": "",
            "_lee_yu": False,
        },
    }

    _priors_cls = SEMPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "lam"
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = "GaussianSEMGibbs"
    _model_type: str = "sem"
    _likelihood: str = "gaussian"
    _gibbs_key: tuple[str, str] | None = ("gaussian", "cross_section")

    def _compute_impacts_posterior_static(
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
