r"""Latent spatial Durbin error logit.

.. math::

    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    y^* = X\beta + WX\theta + (I - \lambda W)^{-1}\varepsilon, \qquad
    \varepsilon_i \sim \mathrm{Logistic}(0, 1)

The latent SEM logit with the covariates' spillover as separate coefficients:
``WX theta`` carries the spillover and :math:`\lambda` the clustering of the
disturbance, the pattern the GLM tests show when the spillover and error terms
are both significant out of the proportion one :math:`\rho` implies.
"""

from __future__ import annotations

import numpy as np

from .._mixins._designs import BinaryDesigns
from ..base import SpatialModel
from ..priors import SEMLogitPriors
from ._latent_binary import LatentLogitMixin


class SDEMLogit(BinaryDesigns, LatentLogitMixin, SpatialModel):
    r"""Bayesian latent SDEM logit, sampled by the latent logit's exact Gibbs sweep.

    .. math::

        y_i = \mathbb 1\{y^*_i > 0\}, \qquad
        (I - \lambda W)(y^* - X\beta - WX\theta) = \varepsilon, \qquad
        \varepsilon_i \sim \mathrm{Logistic}(0, 1).

    It is :class:`SEMLogit` on the design ``[X, WX]``; ``beta`` carries both
    blocks, the ``WX`` coefficients labelled ``W*{name}``.

    Parameters
    ----------
    formula, data, y, X, W, logdet_method, logdet_decomp
        As for :class:`SEMLogit`.
    priors : dict or SEMLogitPriors, optional
        As for :class:`SEMLogit`; ``beta_mu`` and ``beta_sigma`` cover both
        blocks of ``beta``.
    durbin_vars : list of str, optional
        The covariates whose spatial lags enter; all but the intercept by
        default.

    Notes
    -----
    Impacts are on the latent index, as for the Gaussian SDEM: ``direct =
    beta`` and ``indirect = theta`` times the mean row sum of ``W`` (one for a
    row-standardized ``W``).
    """

    _priors_cls = SEMLogitPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ("WX",)
    _jacobian_param: str | None = "lam"
    _has_wx_in_beta: bool = True
    _model_type: str = "sdem_logit"
    _gibbs_key: tuple[str, str] | None = ("binary_latent", "cross_section")
    _latent_process = "sem"

    # The one design is the latent SDEM.
    _fit_gibbs_static = LatentLogitMixin._fit_gibbs_latent
    _build_pymc_model_static = LatentLogitMixin._build_pymc_model_latent
    _fitted_probabilities_static = LatentLogitMixin._fitted_probabilities_latent
    _fitted_mean_from_posterior_static = (
        LatentLogitMixin._fitted_mean_from_posterior_latent
    )

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct ``beta``, indirect ``theta`` on the latent index (as SLX/SDEM)."""
        return self._slx_effects()
