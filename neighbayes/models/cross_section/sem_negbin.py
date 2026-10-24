r"""Latent SEM negative binomial.

.. math::

    y_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
    \eta = X\beta + (I - \lambda W)^{-1}\varepsilon, \qquad
    \varepsilon_i = \log\gamma_i,\; \gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)

The error process carries the negative binomial's own gamma disturbance, so
:math:`\lambda = 0` is :class:`~neighbayes.models.NegBin`, and the GLM suite's
``LM-Error`` is this model's score test.  A SEM whose error process carried no
disturbance would be the aspatial model, so the unmarked SEM negative binomial
is the latent one (ADR 0003).
"""

from __future__ import annotations

import numpy as np

from .._mixins._designs import CountDesigns
from ..base import SpatialModel
from ..priors import SEMNegBinPriors
from ._latent_negbin import LatentNegBinMixin


class SEMNegBin(CountDesigns, LatentNegBinMixin, SpatialModel):
    r"""Bayesian latent SEM negative binomial, sampled by an exact Gibbs sweep.

    .. math::

        y_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
        (I - \lambda W)(\eta - X\beta) = \varepsilon, \qquad
        \varepsilon_i = \log\gamma_i,\;
        \gamma_i \sim \mathrm{Gamma}(\alpha, \alpha).

    The sweep draws the log-means, :math:`\lambda`, :math:`\beta` and the
    dispersion :math:`\alpha`, and interweaves moves that hold the
    disturbances instead, which keep :math:`\lambda` and :math:`\alpha`
    mixing when the counts are small (:mod:`neighbayes.samplers.negbin_latent`);
    on JAX by default (``backend="numpy"`` for the compiled NumPy sweep).
    There is no NUTS path.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Non-negative integer counts of shape ``(n,)`` (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Design matrix (matrix mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    priors : dict or SEMNegBinPriors, optional
        ``lam_lower``, ``lam_upper``: the Uniform prior on :math:`\lambda`
        (default the stability bounds of ``W``); ``beta_mu``, ``beta_sigma``
        (default Gelman et al. 2008 on the log scale); ``alpha_sigma``,
        ``alpha_nu`` (Half-Student-t on α) and ``alpha_fixed``, as for
        :class:`~neighbayes.models.NegBin`.
    logdet_method, logdet_decomp
        As for :class:`~neighbayes.models.SEM`.

    Notes
    -----
    Impacts are :math:`\beta` on the log-mean (the SEM has no spillovers);
    the fitted means simulate the disturbances, whose propagation makes the
    expected count differ from :math:`e^{x_i^\top\beta}`.
    """

    _DESIGNS = {"static": {}}
    _priors_cls = SEMNegBinPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _jacobian_param: str | None = "lam"
    _model_type: str = "sem_negbin"
    _gibbs_key: tuple[str, str] | None = ("count_latent", "cross_section")
    _latent_process = "sem"

    # The one design is the latent SEM.
    _fit_gibbs_static = LatentNegBinMixin._fit_gibbs_latent
    _build_pymc_model_static = LatentNegBinMixin._build_pymc_model_latent
    _fitted_mean_from_posterior_static = (
        LatentNegBinMixin._fitted_mean_from_posterior_latent
    )

    def _compute_impacts_posterior_static(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct effects ``beta`` on the log-mean; no indirect effects."""
        return self._sem_effects()
