"""Bayesian (non-spatial) Negative Binomial regression.

.. math::

    y_i \\sim \\mathrm{NegBin}(\\mu_i, \\alpha), \\quad
    \\mu_i = \\exp(x_i'\\beta)

NB2 parameterization: ``Var(y_i) = mu_i + mu_i^2 / alpha``.  The natural
count-response analogue of :class:`neighbayes.models.OLS`: no spatial
structure, NUTS sampler.  ``W`` is optional and used only by Bayesian LM
spatial-specification diagnostics applied after fitting.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt

from ..._lazy_deps import pm
from ..base import SpatialModel
from ..priors import NegBinPriors


class NegBin(SpatialModel):
    r"""Bayesian (non-spatial) Negative Binomial regression.

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
        \mu_i = \exp(x_i'\beta)

    NB2 parameterization: :math:`\mathrm{Var}(y_i) = \mu_i + \mu_i^2 / \alpha`.
    Sampled with NUTS.  ``W`` is optional and is used only by Bayesian
    LM diagnostics to assess whether a spatial count model
    (e.g. :class:`neighbayes.models.SARNegBinStructural`) is warranted.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Non-negative integer dependent variable of shape ``(n,)``.
        Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Design matrix.  Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights matrix.  Not used during estimation; required for
        post-hoc Bayesian LM spatial diagnostics.
    priors : dict or NegBinPriors, optional
        Override default priors.  Supported keys:

        - ``beta_mu`` (array or float, optional): Prior mean for
          :math:`\beta`.  Default uses the Gelman et al. (2008)
          weakly-informative mean.
        - ``beta_sigma`` (array or float, optional): Prior std for
          :math:`\beta`.  Default uses the Gelman et al. (2008)
          weakly-informative scale.
        - ``alpha_sigma`` (float, default 2.5): Scale of the
          :math:`\mathrm{Half\text{-}Student\text{-}t}` prior on
          :math:`\alpha`.
        - ``alpha_nu`` (float, default 3.0): Degrees of freedom of the
          :math:`\mathrm{Half\text{-}Student\text{-}t}` prior on
          :math:`\alpha`.
    """

    _priors_cls = NegBinPriors
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = "negbin"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError("robust=True is not supported for NegBin.")

        y_round = np.round(self._y).astype(np.int64)
        if not np.allclose(self._y, y_round):
            raise ValueError("NegBin requires integer-valued observations.")
        if np.any(y_round < 0):
            raise ValueError("NegBin requires non-negative integer observations.")
        self._y_int = y_round
        self._y = y_round.astype(np.float64)

    def _build_pymc_model(self) -> pm.Model:
        X = self._X
        names = list(self._feature_names)
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            X, names, link="log"
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)
            mu = pt.exp(pt.dot(X, beta))
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=self._y_int)
        return model

    # ------------------------------------------------------------------
    # Spatial-effects hooks (not applicable for non-spatial model)
    # ------------------------------------------------------------------

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "NegBin has no spatial structure and therefore no "
            "spatial effects. Use Bayesian LM diagnostics to assess whether "
            "a spatial count model (e.g. SARNegBinStructural) is appropriate."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Posterior-mean response mean ``exp(X @ E[beta])``."""
        beta = self._posterior_mean("beta")
        return np.exp(self._X @ beta)
