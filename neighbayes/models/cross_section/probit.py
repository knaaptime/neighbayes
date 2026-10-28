"""Bayesian (non-spatial) probit regression.

.. math::

    y_i \\sim \\mathrm{Bernoulli}(\\Phi(x_i'\\beta))

The probit counterpart of :class:`neighbayes.models.Logit`: no spatial
structure, NUTS sampler.  ``W`` is optional and used only by Bayesian LM
spatial-specification diagnostics applied after fitting.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from scipy.special import ndtr

from ..._lazy_deps import pm
from .._mixins._designs import BinaryDesigns
from ..base import SpatialModel
from ..priors import ProbitPriors


class Probit(BinaryDesigns, SpatialModel):
    r"""Bayesian (non-spatial) probit regression.

    .. math::
        y_i = \mathbb 1\{x_i'\beta + \varepsilon_i > 0\}, \qquad
        \varepsilon_i \sim N(0, 1),

    so that :math:`P(y_i = 1) = \Phi(x_i'\beta)`.  Sampled with NUTS.  ``W``
    is optional and is used only by Bayesian LM diagnostics to assess whether
    a spatial probit (:class:`~neighbayes.models.SARProbit`,
    :class:`~neighbayes.models.SEMProbit`) is warranted.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable of shape ``(n,)``.  Required in matrix mode.
        Must contain only 0/1 values.
    X : array-like or pandas.DataFrame, optional
        Design matrix.  Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights matrix.  Not used during estimation; required for
        post-hoc Bayesian LM spatial diagnostics.
    priors : dict or ProbitPriors, optional
        ``beta_mu``, ``beta_sigma`` (default Gelman et al. 2008 on the probit
        scale: intercept at ``Φ⁻¹(mean(y))``, scale ``2.5 / 1.6``; slopes
        ``(2.5 / 1.6) / sd(x_j)``).
    """

    _priors_cls = ProbitPriors
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = "probit"
    _link: str = "probit"

    def _build_pymc_model(self) -> pm.Model:
        X = self._X
        beta_mu, beta_sigma = self._resolved_beta_prior(link="probit")

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            eta = pt.dot(X, beta)
            pm.Bernoulli("obs", p=pm.math.invprobit(eta), observed=self._y)
        return model

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Probit has no spatial structure and therefore no impacts. "
            "Use Bayesian LM diagnostics to assess whether a spatial probit "
            "(e.g. SARProbit) is appropriate."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean response probabilities ``P(y=1 | X, E[beta])``."""
        return ndtr(self._X @ self._posterior_mean("beta"))
