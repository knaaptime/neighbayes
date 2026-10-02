"""Bayesian OLS (non-spatial) cross-sectional regression model.

y = X @ beta + epsilon,  epsilon ~ N(0, sigma^2 I)

This model contains no spatial structure of its own.  It is the natural
baseline from which Bayesian spatial specification tests are run to
determine which spatial model — SAR, SEM, SLX, etc. — is most appropriate.
W is optional at construction time.
"""

from __future__ import annotations

import numpy as np

from .._mixins import GaussianLikelihoodMixin, GaussianPredictionMixin
from ..base import SpatialModel
from ..priors import OLSPriors


class OLS(GaussianPredictionMixin, GaussianLikelihoodMixin, SpatialModel):
    """Bayesian ordinary least squares cross-sectional regression.

    .. math::
        y = X\\beta + \\varepsilon, \\quad \\varepsilon \\sim N(0, \\sigma^2 I)

    This model places diffuse Normal priors on the coefficient vector
    :math:`\\beta` and a HalfNormal prior on the noise standard deviation
    :math:`\\sigma`.

    ``W`` is **optional**.  If supplied, Bayesian LM tests can be run on
    the OLS posterior to guide model selection.

    **Robust regression**

    When ``robust=True``, the error distribution is changed from Normal
    to Student-t, yielding a model that is robust to heavy-tailed outliers:

    .. math::

        \\varepsilon \\sim t_\\nu(0, \\sigma^2 I)

    where :math:`\\nu` is a **fixed** hyperparameter set by ``priors={"nu": value}``
    (default 4, LeSage's ``rval``); larger values approach the Normal.  Values
    must exceed 2 so the variance exists.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + income"``.
        If provided, ``data`` must also be supplied.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable of shape ``(n,)``.  Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Predictor matrix.  Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights matrix of shape ``(n, n)``.  Not used during
        estimation; required for Bayesian LM specification tests.
    priors : dict, optional
        Override default priors.  Supported keys:

        - ``beta_mu`` (array or float, optional): Prior mean for
          :math:`\beta`. Default is Gelman et al. (2008) weakly-informative
          mean (``mean(y)`` for intercept/constant columns, 0 for slopes).
        - ``beta_sigma`` (array or float, optional): Prior std for
          :math:`\beta`. Default is Gelman et al. (2008) weakly-informative
          scale: ``2.5 * sd(y)`` for intercept/constant columns and
          ``2.5 * sd(y) / sd(x_j)`` for each slope.
        - ``sigma2_alpha`` (float, default 2.0): Shape of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``sigma2_beta`` (float, default ``Var(y)``): Scale of the
          InverseGamma prior on :math:`\\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal.
    """

    _priors_cls = OLSPriors
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = "ols"
    _likelihood: str = "gaussian"  # NUTS-only (no _gibbs_key)

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Not applicable — OLS has no spatial lag structure.

        Raises
        ------
        NotImplementedError
            Always raised; use Bayesian LM diagnostics instead.
        """
        raise NotImplementedError(
            "OLS has no spatial lag structure and therefore no spatial effects. "
            "Use Bayesian LM diagnostics to assess which spatial model "
            "is appropriate, then refit with SAR, SEM, SLX, SDM, or SDEM."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean coefficients.

        Returns
        -------
        np.ndarray
            Posterior-mean fitted values ``X @ E[beta | data]``.
        """
        beta = self._posterior_mean("beta")
        return self._X @ beta
