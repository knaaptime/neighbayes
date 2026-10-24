r"""Latent spatial Durbin error negative binomial.

.. math::

    y_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
    \eta = X\beta + WX\theta + (I - \lambda W)^{-1}\varepsilon, \qquad
    \varepsilon_i = \log\gamma_i,\; \gamma_i \sim \mathrm{Gamma}(\alpha, \alpha)

The latent SEM negative binomial with the covariates' spillover as separate
coefficients: ``WX theta`` carries the spillover and :math:`\lambda` the
clustering of the disturbance, the pattern the GLM tests show when the
spillover and error terms are both significant out of the proportion one
:math:`\rho` implies.
"""

from __future__ import annotations

import numpy as np

from .._mixins._designs import CountDesigns
from ..base import SpatialModel
from ..priors import SEMNegBinPriors
from ._latent_negbin import LatentNegBinMixin


class SDEMNegBin(CountDesigns, LatentNegBinMixin, SpatialModel):
    r"""Bayesian latent SDEM negative binomial, sampled as :class:`SEMNegBin` is.

    .. math::

        y_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
        (I - \lambda W)(\eta - X\beta - WX\theta) = \varepsilon, \qquad
        \varepsilon_i = \log\gamma_i,\;
        \gamma_i \sim \mathrm{Gamma}(\alpha, \alpha).

    It is :class:`SEMNegBin` on the design ``[X, WX]``; ``beta`` carries both
    blocks, the ``WX`` coefficients labelled ``W*{name}``.

    Parameters
    ----------
    formula, data, y, X, W, logdet_method, logdet_decomp
        As for :class:`SEMNegBin`.
    priors : dict or SEMNegBinPriors, optional
        As for :class:`SEMNegBin`; ``beta_mu`` and ``beta_sigma`` cover both
        blocks of ``beta``.
    durbin_vars : list of str, optional
        The covariates whose spatial lags enter; all but the intercept by
        default.

    Notes
    -----
    Impacts are on the log-mean, as for the Gaussian SDEM: ``direct = beta``
    and ``indirect = theta`` times the mean row sum of ``W`` (one for a
    row-standardized ``W``).
    """

    _DESIGNS = {"static": {}}
    _priors_cls = SEMNegBinPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ("WX",)
    _jacobian_param: str | None = "lam"
    _has_wx_in_beta: bool = True
    _model_type: str = "sdem_negbin"
    _gibbs_key: tuple[str, str] | None = ("count_latent", "cross_section")
    _latent_process = "sem"

    # The one design is the latent SDEM.
    _fit_gibbs_static = LatentNegBinMixin._fit_gibbs_latent
    _build_pymc_model_static = LatentNegBinMixin._build_pymc_model_latent
    _fitted_mean_from_posterior_static = (
        LatentNegBinMixin._fitted_mean_from_posterior_latent
    )

    def _compute_impacts_posterior_static(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct ``beta``, indirect ``theta`` on the log-mean (as SLX/SDEM)."""
        return self._slx_effects()
