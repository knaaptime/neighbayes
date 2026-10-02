"""Bayesian (non-spatial) logistic regression.

.. math::

    y_i \\sim \\mathrm{Bernoulli}(\\mathrm{logit}^{-1}(x_i'\\beta))

The natural binary-response analogue of :class:`neighbayes.models.OLS`:
no spatial structure, NUTS sampler.  ``W`` is optional and used only
by Bayesian LM spatial-specification diagnostics applied after fitting.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from ..._lazy_deps import pm
from ..base import SpatialModel
from ..priors import SARLogitPriors


class Logit(SpatialModel):
    r"""Bayesian (non-spatial) logistic regression.

    .. math::
        y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(x_i'\beta))

    Sampled with NUTS.  ``W`` is optional and is used only by Bayesian
    LM diagnostics to assess whether a spatial binary-response model
    (e.g. :class:`neighbayes.models.SARLogit`) is warranted.

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
    priors : dict or SARLogitPriors, optional
        Override default priors.  Supported keys:

        - ``beta_mu`` (array or float, optional): Prior mean for
          :math:`\beta`.  Default uses the Gelman et al. (2008)
          weakly-informative mean (``mean(y)`` for intercept-like
          columns, 0 for slopes).
        - ``beta_sigma`` (array or float, optional): Prior std for
          :math:`\beta`.  Default uses the Gelman et al. (2008)
          weakly-informative scale.
    """

    _priors_cls = SARLogitPriors
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = "logit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError("robust=True is not supported for Logit.")

        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("Logit requires binary y with values in {0, 1}.")

    def _build_pymc_model(self) -> pm.Model:
        X = self._X
        names = list(self._feature_names)
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            X, names, link="logit"
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            eta = pt.dot(X, beta)
            pm.Bernoulli("obs", logit_p=eta, observed=self._y)
        return model

    # ------------------------------------------------------------------
    # Spatial-effects hooks (not applicable for non-spatial model)
    # ------------------------------------------------------------------

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "Logit has no spatial structure and therefore no spatial effects. "
            "Use Bayesian LM diagnostics to assess whether a spatial binary "
            "model (e.g. SARLogit) is appropriate."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean response probabilities ``P(y=1 | X, E[beta])``."""
        beta = self._posterior_mean("beta")
        eta = self._X @ beta
        return 1.0 / (1.0 + np.exp(-eta))
