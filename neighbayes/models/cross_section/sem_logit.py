r"""Latent SEM logit.

.. math::

    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    y^* = X\beta + (I - \lambda W)^{-1}\varepsilon, \qquad
    \varepsilon_i \sim \mathrm{Logistic}(0, 1)

The error process carries the logit's own disturbance, so :math:`\lambda = 0`
is the logit, and the GLM suite's ``LM-Error`` is this model's score test.  A
SEM whose error process carried no disturbance would be the aspatial logit, so
the unmarked SEM logit is the latent one (ADR 0003).
"""

from __future__ import annotations

import numpy as np

from .._mixins._designs import BinaryDesigns
from ..base import SpatialModel
from ..priors import SEMLogitPriors
from ._latent_binary import LatentLogitMixin


class SEMLogit(BinaryDesigns, LatentLogitMixin, SpatialModel):
    r"""Bayesian latent SEM logit, sampled by an exact Gibbs sweep.

    .. math::

        y_i = \mathbb 1\{y^*_i > 0\}, \qquad
        (I - \lambda W)(y^* - X\beta) = \varepsilon, \qquad
        \varepsilon_i \sim \mathrm{Logistic}(0, 1).

    The sweep draws the latent utilities (Albert & Chib 1993), the Kolmogorov
    scales that make the logistic a normal scale mixture (Holmes & Held 2006;
    ``augax``), :math:`\lambda` with :math:`\beta` integrated out, then
    :math:`\beta`, on JAX by default (``backend="numpy"`` for the compiled
    NumPy sweep).  There is no NUTS path, the likelihood being an orthant
    probability.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable (0/1) of shape ``(n,)`` (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Design matrix (matrix mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    priors : dict or SEMLogitPriors, optional
        ``lam_lower``, ``lam_upper`` (default -0.999, 0.999): the Uniform prior
        on :math:`\lambda`; ``beta_mu``, ``beta_sigma`` (default Gelman et al.
        2008 on the logit scale).
    logdet_method, logdet_decomp
        As for :class:`~neighbayes.models.SEM`.

    Notes
    -----
    Impacts are :math:`\beta` on the latent index (the SEM has no
    spillovers); :meth:`fitted_probabilities` simulates the utilities, whose
    marginals the multiplier makes non-logistic.
    """

    _priors_cls = SEMLogitPriors
    _spatial_params: tuple[str, ...] = ("lam",)
    _jacobian_param: str | None = "lam"
    _model_type: str = "sem_logit"
    _gibbs_key: tuple[str, str] | None = ("binary_latent", "cross_section")
    _latent_process = "sem"

    # The one design is the latent SEM.
    _fit_gibbs_static = LatentLogitMixin._fit_gibbs_latent
    _build_pymc_model_static = LatentLogitMixin._build_pymc_model_latent
    _fitted_probabilities_static = LatentLogitMixin._fitted_probabilities_latent
    _fitted_mean_from_posterior_static = (
        LatentLogitMixin._fitted_mean_from_posterior_latent
    )

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct effects ``beta`` on the latent index; no indirect effects."""
        return self._sem_effects()
