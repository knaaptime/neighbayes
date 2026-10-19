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
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._mixins._designs import CountDesigns
from ..base import SpatialModel
from ..panel._count import _CountPanel
from ..priors import NegBinPriors, PanelCountPriors


class NegBin(CountDesigns, _CountPanel, SpatialModel):
    r"""Bayesian (non-spatial) Negative Binomial regression.

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
        \mu_i = \exp(x_i'\beta)

    NB2 parameterization: :math:`\mathrm{Var}(y_i) = \mu_i + \mu_i^2 / \alpha`.
    Sampled with NUTS.  ``W`` is optional and is used only by Bayesian
    LM diagnostics to assess whether a spatial count model
    (e.g. :class:`neighbayes.models.SARNegBin`) is warranted.

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
        post-hoc Bayesian LM spatial diagnostics, and by a panel.
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

        A panel's are those of
        :class:`~neighbayes.models.priors.PanelCountPriors`, with
        ``alpha_fixed`` and the effects' ``unit_effect_mu``,
        ``unit_effect_sd_scale``, ``time_effect_mu`` and
        ``time_effect_sigma``.
    time_col, unit_col : str, optional
        Formula-mode panel: the period and unit columns of ``data``.  Given
        time, the model is a panel (:mod:`neighbayes.models.panel._count`),
        stacked by period, then unit, and sampled by Pólya–Gamma Gibbs with
        the unit effects integrated out.
    N, T : int, optional
        Matrix-mode panel: the numbers of units and periods.
    unit_effects : {None, "random"}, default None
        A panel's random unit effects, ``unit_effect``.
    time_effects : bool, default False
        A panel's fixed time effects, ``time_effect``.
    mundlak : bool, default False
        Add the unit means of the time-varying columns to a panel's design,
        absorbing a correlation between the unit effects and ``X``.  Needs
        unit effects.
    """

    _DESIGNS = {
        "static": {},
        "panel": {"_priors_cls": PanelCountPriors, "_gibbs_key": ("count", "panel")},
    }

    _priors_cls = NegBinPriors
    _model_type: str = "negbin"
    _count_spatial = False

    def _build_pymc_model_static(self) -> pm.Model:
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
            alpha = nb_alpha_rv(nb_alpha_fixed(self.priors), alpha_nu, alpha_sigma)
            mu = pt.exp(pt.dot(X, beta))
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=self._y_int)
        return model

    # ------------------------------------------------------------------
    # Impacts hooks (not applicable for non-spatial model)
    # ------------------------------------------------------------------

    def _compute_impacts_posterior_static(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "NegBin has no spatial structure and therefore no "
            "impacts. Use Bayesian LM diagnostics to assess whether "
            "a spatial count model (e.g. SARNegBin) is appropriate."
        )

    def _fitted_mean_from_posterior_static(self) -> np.ndarray:
        """Posterior-mean response mean ``exp(X @ E[beta])``."""
        beta = self._posterior_mean("beta")
        return np.exp(self._X @ beta)
