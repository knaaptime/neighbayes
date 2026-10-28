r"""SAR probit: the latent SAR, or the reduced form with ``latent=False``.

.. math::

    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    y^* = (I - \rho W)^{-1}(X\beta + \varepsilon), \qquad
    \varepsilon_i \sim N(0, 1)

The textbook SAR probit (McMillen 1992; LeSage and Pace 2009, §10.1): the
multiplier propagates the probit's own disturbance, so :math:`\rho = 0` is the
probit and the probit's normalization fixes the disturbance's scale, with no
parameter pinned.  The unmarked SAR probit is this latent SAR (ADR 0003);
``latent=False`` selects the reduced form, :math:`y_i \sim
\mathrm{Bernoulli}(\Phi(\eta_i))` with :math:`\eta = (I - \rho W)^{-1}X\beta`,
which propagates the covariates only (:mod:`._reduced_binary`).
"""

from __future__ import annotations

from .._mixins._designs import BinaryDesigns
from ..base import SpatialModel
from ..priors import SARProbitPriors
from ._latent_binary import LatentProbitMixin
from ._reduced_binary import ReducedBinaryMixin


class SARProbit(BinaryDesigns, ReducedBinaryMixin, LatentProbitMixin, SpatialModel):
    r"""Bayesian SAR probit: the latent SAR, or the reduced form with ``latent=False``.

    .. math::

        y_i = \mathbb 1\{y^*_i > 0\}, \qquad
        (I - \rho W) y^* = X\beta + \varepsilon, \qquad
        \varepsilon_i \sim N(0, 1).

    The sweep draws the latent utilities (Albert & Chib 1993), :math:`\rho`
    with :math:`\beta` integrated out, then :math:`\beta`, on JAX by default,
    the utilities by colour class, or with ``backend="numpy"`` one at a time.
    There is no NUTS path, the likelihood being an orthant probability.

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
    priors : dict or SARProbitPriors, optional
        ``rho_lower``, ``rho_upper`` (default -0.999, 0.999): the Uniform
        prior on :math:`\rho`; ``beta_mu``, ``beta_sigma`` (default Gelman et
        al. 2008 on the probit scale: intercept at ``Φ⁻¹(mean(y))``, scale
        ``2.5 / 1.6``; slopes ``(2.5 / 1.6) / sd(x_j)``).
    logdet_method, logdet_decomp
        As for :class:`~neighbayes.models.SAR`; the reduced form has no
        log-determinant.
    latent : bool, default True
        ``False`` fits the reduced form, :math:`y_i \sim
        \mathrm{Bernoulli}(\Phi(\eta_i))`, :math:`\eta = (I - \rho
        W)^{-1}X\beta`, by the reduced SAR logit's sampler with Albert–Chib
        utilities in place of its Pólya–Gamma step.

    Notes
    -----
    Impacts are on the latent index (``scale="index"``), as for the Gaussian
    SAR, or on :math:`P(y_i = 1)` (``scale="probability"``), whose derivative
    carries the density of the propagated disturbance at the threshold;
    :meth:`fitted_probabilities` simulates the utilities, whose marginal
    variances the multiplier makes unequal.
    """

    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sar_probit"
    _gibbs_key: tuple[str, str] | None = ("binary", "cross_section")
    _priors_cls = SARProbitPriors
    _DESIGNS = {
        "static": {},
        "latent": {"_gibbs_key": ("binary_latent", "cross_section")},
    }
    _latent_default = True
    _latent_process = "sar"

    def impacts(
        self,
        return_posterior_samples: bool = False,
        scale: str = "index",
        method: str = "auto",
    ):
        r"""Direct, indirect and total impacts on the latent index or probability scale.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"index", "probability"}, default "index"
            ``"index"``: impacts on the latent index, :math:`(I - \rho
            W)^{-1}\beta_r`.  ``"probability"``: impacts on :math:`P(y_i =
            1)`; for the latent SAR the derivative carries the density of the
            propagated normal disturbance at the threshold (Monte Carlo per
            posterior draw, conditioning on all but the unit's own
            disturbance), for the reduced form
            :math:`\operatorname{diag}(\phi(\eta))(I - \rho W)^{-1}\beta_r`.
        method : {"auto", "eigen", "sparse"}, default "auto"
            The reduced form's probability-scale multiplier diagonal, as for
            :meth:`SARLogit.impacts`.
        """
        return self._sar_binary_impacts(scale, return_posterior_samples, method)
