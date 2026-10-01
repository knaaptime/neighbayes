"""Typed prior containers for :mod:`neighbayes.models`.

Each model family declares its supported priors via a frozen dataclass.  The
dataclass serves two purposes:

* **Validation.** Constructing :class:`SARPriors` with an unknown keyword
  raises :class:`TypeError`, so typos like ``rho_lo=-1`` fail fast at model
  construction instead of being silently ignored deep inside
  ``_build_pymc_model``.
* **Discoverability.** Users get IDE autocomplete + type checking on the
  supported keys and their defaults.

The user-facing ``priors=...`` argument on every model still accepts a plain
``dict``; the model's ``__init__`` calls :func:`resolve_priors` to coerce
``dict | Priors | None`` into the canonical dataclass and rejects unknown
keys with a helpful error.

Example
-------

>>> from neighbayes.models import SAR
>>> from neighbayes.models.priors import SARPriors
>>> SAR(y=y, X=X, W=W, priors=SARPriors(rho_lower=0.0))           # typed form
>>> SAR(y=y, X=X, W=W, priors={"rho_lower": 0.0})                  # dict form

Class hierarchy
---------------

Two independent families live here:

* **User-facing model priors** — validated hyperparameter containers passed
  as ``priors=...`` to model constructors.  ``BasePriors`` (cross-section)
  and ``PanelBasePriors`` (panel) hold the common Gaussian-family keys;
  each model subclass narrows or extends them (``SARPriors``,
  ``SEMPriors``, ``TobitPriors``, ``SARProbitPriors``, flow/panel/dynamic
  variants, ...).  These are pinned by
  ``tests/test_models/test_priors_defaults_pin.py``.

* **Internal Gibbs-sampler structs** — resolved-hyperparameter containers
  the numpy/JAX kernels consume (``GibbsBasePriors`` →
  ``GaussianGibbsPriors`` / ``LogitGibbsPriors`` / ``REGibbsPriors``;
  standalone ``GibbsPriors``, ``ReducedGibbsPriors``,
  ``FlowReducedGibbsPriors``, ``ZINBGibbsPriors``, ``SEMLogitGibbsPriors``,
  ``PanelGaussianPriors``).  Models build these internally in
  ``_fit_gibbs``; they are not part of the user-facing ``priors=`` API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Type, Union

import numpy as np

# ---------------------------------------------------------------------------
# Cross-sectional dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasePriors:
    """Priors shared by every cross-sectional model.

    Attributes
    ----------
    beta_mu, beta_sigma
        Normal prior on regression coefficients.  Both default to ``None``,
        which means the model resolves a *weakly-informative, data-scaled*
        Gelman et al. (2008) prior at construction:

        * Intercept columns get ``mu = mean(y)`` and ``sigma = 2.5 * sd(y)``.
        * Slope columns get ``mu = 0`` and ``sigma = 2.5 * sd(y) / sd(x_j)``.

        Either field may be set explicitly as a scalar (broadcast to all
        coefficients) or as a vector of length ``p`` matching the design
        matrix used by ``beta``.  Setting these to large finite values
        (e.g. ``beta_sigma = 1e6``) reproduces the legacy near-improper
        prior but is not recommended — it produces Bartlett-Lindley
        penalties of tens of nats in Bayes-factor model comparison and
        causes posterior-sampling problems on collinear designs.
    sigma2_alpha, sigma2_beta
        Inverse-gamma prior on the observation-noise variance σ²:
        ``σ² ~ InverseGamma(sigma2_alpha, sigma2_beta)``.  Used by the
        Gaussian models (OLS, SAR, SEM, SDM, SDEM) for **both** NUTS and
        Gibbs paths so that posteriors agree exactly.  Following LeSage
        (2009) the conjugate Inv-Γ keeps the σ² Gibbs block in closed
        form.  Default ``alpha=2.0`` gives a finite-mean weakly
        informative prior; if ``sigma2_beta`` is ``None`` the model
        resolves it to ``Var(y)`` at construction so the prior mean is
        scale-aware (~ Var(y)).
    nu
        Student-t degrees of freedom used when ``robust=True``.  Following
        LeSage (2009) this is a **fixed** hyperparameter rather than a
        sampled quantity, which keeps the NUTS and Gibbs paths targeting the
        same posterior.  The default of 4 matches LeSage's ``rval`` and gives
        genuinely fat tails; large values (≳ 30) approach the Normal.

    References
    ----------
    Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008).
    *A weakly informative default prior distribution for logistic and
    other regression models.* Annals of Applied Statistics, 2(4), 1360-1383.
    """

    beta_mu: float | Any = None
    beta_sigma: float | Any = None
    sigma2_alpha: float = 2.0
    sigma2_beta: float | None = None
    nu: float = 4.0  # Student-t df when robust=True; LeSage's rval.


@dataclass(frozen=True)
class OLSPriors(BasePriors):
    """Priors for :class:`neighbayes.models.OLS`."""


@dataclass(frozen=True)
class SLXPriors(BasePriors):
    """Priors for :class:`neighbayes.models.SLX`."""


@dataclass(frozen=True)
class SARPriors(BasePriors):
    """Priors for :class:`neighbayes.models.SAR`.

    Adds bounds on the spatial autoregressive parameter ``rho``.
    """

    rho_lower: float = -1.0
    rho_upper: float = 1.0


@dataclass(frozen=True)
class SEMPriors(BasePriors):
    """Priors for :class:`neighbayes.models.SEM`.

    Adds bounds on the spatial error parameter ``lam``.
    """

    lam_lower: float = -1.0
    lam_upper: float = 1.0


@dataclass(frozen=True)
class SDMPriors(SARPriors):
    """Priors for :class:`neighbayes.models.SDM` (lag + WX)."""


@dataclass(frozen=True)
class SDEMPriors(SEMPriors):
    """Priors for :class:`neighbayes.models.SDEM` (SLX + spatial error)."""


@dataclass(frozen=True)
class NegBinPriors(BasePriors):
    """Priors for :class:`neighbayes.models.NegBin`.

    Adds overdispersion parameters for the NB2 likelihood.
    """

    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0


@dataclass(frozen=True)
class SARNegBinPriors(SARPriors, NegBinPriors):
    """Priors for :class:`neighbayes.models.SARNegBin`.

    Combines SAR spatial bounds with NB overdispersion parameters.
    """


# ---------------------------------------------------------------------------
# Tobit / probit dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SARTobitPriors(SARPriors):
    """Priors for :class:`neighbayes.models.SARTobit`.

    The same priors as :class:`SARPriors`: Gelman et al. (2008) on
    ``beta`` and ``IG(2, Var y)`` on ``sigma**2``.  The censored latent
    values carry a flat prior on their gap below the threshold, so the
    regression density alone defines their distribution and integrating
    them out gives the censored likelihood exactly.
    """


@dataclass(frozen=True)
class SEMTobitPriors(SEMPriors):
    """Priors for :class:`neighbayes.models.SEMTobit`."""


@dataclass(frozen=True)
class SDMTobitPriors(SDMPriors):
    """Priors for :class:`neighbayes.models.SDMTobit`."""


@dataclass(frozen=True)
class SARProbitPriors:
    """Priors for :class:`neighbayes.models.SARProbit`.

    Differs from :class:`SARPriors`: there is no ``sigma`` parameter (the
    probit link absorbs the noise scale) and the regional random-effect
    variance ``sigma_a`` has its own half-normal scale.
    """

    rho_lower: float = -0.95
    rho_upper: float = 0.95
    beta_mu: float | Any = None  # Gelman et al. (2008), link scale
    beta_sigma: float | Any = None
    sigma_a_sigma: float = 2.0


@dataclass(frozen=True)
class SARLogitPriors:
    """Priors for :class:`neighbayes.models.SARLogit`.

    Like :class:`SARProbitPriors` but without ``sigma_a_sigma``
    (the logit model has no regional random effect).  There is no
    ``sigma`` parameter because the logit link absorbs the noise scale.
    """

    rho_lower: float = -0.999
    rho_upper: float = 0.999
    beta_mu: float | Any = None  # Gelman et al. (2008), link scale
    beta_sigma: float | Any = None


# Alias — the non-spatial Logit model uses the same prior structure.
LogitPriors = SARLogitPriors


@dataclass
class SEMLogitPriors:
    """Priors for :class:`neighbayes.models.SEMLogit`.

    Like :class:`SARLogitPriors` but with ``lam_lower``/``lam_upper``
    instead of ``rho_lower``/``rho_upper``.
    """

    lam_lower: float = -0.999
    lam_upper: float = 0.999
    beta_mu: float | Any = None  # Gelman et al. (2008), link scale
    beta_sigma: float | Any = None


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------


PriorsLike = Union[
    Mapping[str, Any],
    BasePriors,
    "SARProbitPriors",
    "SARLogitPriors",
    "SEMLogitPriors",
    None,
]


def resolve_priors(
    priors: PriorsLike,
    priors_cls: Type[Any],
) -> Any:
    """Coerce a user-supplied ``priors`` argument to a dataclass instance.

    Parameters
    ----------
    priors
        Either ``None`` (use defaults), a :class:`dict` of overrides, or an
        already-constructed priors dataclass.
    priors_cls
        The dataclass type expected by the model.  Used to validate keys
        and to build the result when ``priors`` is ``None`` or a ``dict``.

    Returns
    -------
    An instance of ``priors_cls``.

    Raises
    ------
    TypeError
        If ``priors`` is a dict containing keys that are not defined on
        ``priors_cls``, or if ``priors`` is a dataclass of an incompatible
        type.
    """
    if priors is None:
        return priors_cls()
    if isinstance(priors, priors_cls):
        return priors
    if isinstance(priors, Mapping):
        allowed = {f.name for f in fields(priors_cls)}
        unknown = set(priors) - allowed
        if unknown:
            raise TypeError(
                f"Unknown prior key(s) for {priors_cls.__name__}: "
                f"{sorted(unknown)}. Allowed keys: {sorted(allowed)}."
            )
        return priors_cls(**dict(priors))
    raise TypeError(
        f"priors must be None, a dict, or a {priors_cls.__name__} instance; "
        f"got {type(priors).__name__}."
    )


def priors_as_dict(priors: Any) -> dict[str, Any]:
    """Return a plain ``dict`` view of a priors dataclass.

    Provided so that existing model code that relies on ``self.priors.get(...)``
    continues to function unchanged while migration to typed access proceeds.

    Fields whose value is ``None`` are omitted so that ``dict.get(key, default)``
    falls through to the caller-supplied default (e.g. a data-driven prior scale
    computed from ``y``).
    """
    return {k: v for k, v in asdict(priors).items() if v is not None}


# ---------------------------------------------------------------------------
# Panel dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelBasePriors:
    """Priors shared by every panel model.

    Attributes
    ----------
    beta_mu, beta_sigma
        Normal prior on regression coefficients.  Both default to
        ``None``, which means the model resolves a weakly-informative,
        data-scaled Gelman et al. (2008) prior at construction (see
        :class:`BasePriors` for the formula).
    sigma2_alpha, sigma2_beta
        Inverse-gamma prior on the observation-noise variance
        :math:`\\sigma^2`.  Default ``alpha=2.0``; ``sigma2_beta`` defaults
        to ``Var(y)`` when ``None``.
    nu
        Student-t degrees of freedom used when ``robust=True``.  Fixed
        rather than sampled; see :class:`BasePriors`.
    """

    beta_mu: float | Any = None
    beta_sigma: float | Any = None
    sigma2_alpha: float = 2.0
    sigma2_beta: float | None = None
    nu: float = 4.0  # Student-t df when robust=True; LeSage's rval.


@dataclass(frozen=True)
class PanelOLSPriors(PanelBasePriors):
    """Priors for :class:`neighbayes.models.OLSPanelFE`."""


@dataclass(frozen=True)
class PanelSLXPriors(PanelBasePriors):
    """Priors for :class:`neighbayes.models.SLXPanelFE`."""


@dataclass(frozen=True)
class PanelSARPriors(PanelBasePriors):
    """Priors for :class:`neighbayes.models.SARPanelFE` (adds rho bounds)."""

    rho_lower: float = -1.0
    rho_upper: float = 1.0


@dataclass(frozen=True)
class PanelSEMPriors(PanelBasePriors):
    """Priors for :class:`neighbayes.models.SEMPanelFE` (adds lam bounds)."""

    lam_lower: float = -1.0
    lam_upper: float = 1.0


@dataclass(frozen=True)
class PanelSDMPriors(PanelSARPriors):
    """Priors for :class:`neighbayes.models.SDMPanelFE`."""


@dataclass(frozen=True)
class PanelSDEMPriors(PanelSEMPriors):
    """Priors for :class:`neighbayes.models.SDEMPanelFE`."""


# -- Panel random-effects ----------------------------------------------------


@dataclass(frozen=True)
class PanelREMixinPriors:
    r"""Mixin providing the random-effects scale prior.

    Attributes
    ----------
    sigma_alpha_nu, sigma_alpha_scale
        Half-t prior on the random-effect scale,
        :math:`\sigma_\alpha \sim \text{half-}t_\nu(0, A)`.  The default
        ``nu = 1`` is the half-Cauchy of Gelman (2006) and Polson & Scott
        (2012), the weakly informative choice for a hierarchical variance;
        the inverse gamma, conjugate or near-improper, pulls
        :math:`\sigma_\alpha` toward zero and over-shrinks the unit effects
        when units are few or weakly identified.  ``sigma_alpha_scale``
        (``A``) defaults to ``sd(y)`` at construction, so the prior is
        scale-aware like the Gelman et al. (2008) default for
        :math:`\beta`.  NUTS and Gibbs use the same prior; the Gibbs
        sampler keeps it conditionally conjugate through the inverse-gamma
        scale mixture of Huang & Wand (2013).
    """

    sigma_alpha_nu: float = 1.0
    sigma_alpha_scale: float | None = None


@dataclass(frozen=True)
class PanelOLSREPriors(PanelOLSPriors, PanelREMixinPriors):
    """Priors for :class:`neighbayes.models.OLSPanelRE`."""


@dataclass(frozen=True)
class PanelSARREPriors(PanelSARPriors, PanelREMixinPriors):
    """Priors for :class:`neighbayes.models.SARPanelRE`."""


@dataclass(frozen=True)
class PanelSEMREPriors(PanelSEMPriors, PanelREMixinPriors):
    """Priors for :class:`neighbayes.models.SEMPanelRE`."""


@dataclass(frozen=True)
class PanelSDEMREPriors(PanelSDEMPriors, PanelREMixinPriors):
    """Priors for :class:`neighbayes.models.SDEMPanelRE`."""


# -- Panel Tobit -------------------------------------------------------------


@dataclass(frozen=True)
class PanelSARTobitPriors(PanelSARPriors):
    """Priors for :class:`neighbayes.models.SARPanelTobit`."""


@dataclass(frozen=True)
class PanelSEMTobitPriors(PanelSEMPriors):
    """Priors for :class:`neighbayes.models.SEMPanelTobit`."""


# -- Panel dynamic (tighter [-0.95, 0.95] bounds; adds phi) -----------------


@dataclass(frozen=True)
class PanelDynamicBasePriors(PanelBasePriors):
    """Priors shared by every dynamic panel model.

    Adds bounds on the AR(1) coefficient ``phi`` on the lagged dependent
    variable and tightens spatial-parameter bounds to ``[-0.95, 0.95]``.
    """

    phi_lower: float = -0.95
    phi_upper: float = 0.95


@dataclass(frozen=True)
class PanelOLSDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`neighbayes.models.OLSPanelDynamic`."""


@dataclass(frozen=True)
class PanelSLXDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`neighbayes.models.SLXPanelDynamic`."""


@dataclass(frozen=True)
class PanelSARDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`neighbayes.models.SARPanelDynamic`."""

    rho_lower: float = -0.95
    rho_upper: float = 0.95


@dataclass(frozen=True)
class PanelSEMDynamicPriors(PanelDynamicBasePriors):
    """Priors for :class:`neighbayes.models.SEMPanelDynamic`."""

    lam_lower: float = -0.95
    lam_upper: float = 0.95


@dataclass(frozen=True)
class PanelSDMRDynamicPriors(PanelSARDynamicPriors):
    """Priors for :class:`neighbayes.models.SDMRPanelDynamic`."""


@dataclass(frozen=True)
class PanelSDMUDynamicPriors(PanelSARDynamicPriors):
    """Priors for :class:`neighbayes.models.SDMUPanelDynamic`.

    Adds bounds on the time-space cross-term coefficient ``theta``.
    """

    theta_lower: float = -0.95
    theta_upper: float = 0.95


@dataclass(frozen=True)
class PanelSDEMDynamicPriors(PanelSEMDynamicPriors):
    """Priors for :class:`neighbayes.models.SDEMPanelDynamic`."""


# ---------------------------------------------------------------------------
# Gibbs sampler prior structs
# ---------------------------------------------------------------------------
#
# These carry the **resolved** numeric hyperparameters (β mean/scale as
# scalars or arrays, spatial bounds, α/σ² params) that the numpy / JAX Gibbs
# kernels consume — distinct from the user-facing validation dataclasses
# above.  They live here so every prior container is defined in one place;
# the sampler modules import them from this module.


@dataclass
class GibbsBasePriors:
    """Base priors for all Gibbs samplers.

    Subclasses add model-specific prior fields (e.g. ``sigma2_alpha``,
    ``alpha_sigma``).
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    rho_lower: float = -0.999
    rho_upper: float = 0.999


@dataclass
class GaussianGibbsPriors:
    """Prior hyperparameters for Gaussian spatial Gibbs.

    Parameters
    ----------
    beta_mu : float or ndarray
        Prior mean for β.  Scalar is broadcast to all coefficients.
    beta_sigma : float or ndarray
        Prior standard deviation for β.  Scalar is broadcast.
    sigma2_alpha : float
        Shape hyperparameter of the ``InverseGamma(sigma2_alpha,
        sigma2_beta)`` prior on σ².  Matches the NUTS path exactly so
        that posteriors — and therefore LOO/WAIC — agree between the two
        samplers.  Conjugate with the Gaussian likelihood, so the σ²
        block is an exact closed-form draw (LeSage 2009 convention).
    sigma2_beta : float
        Scale (rate) hyperparameter of the InverseGamma prior on σ².
        Models typically resolve this to ``Var(y)`` at construction so
        the prior mean is scale-aware.
    rho_lower : float
        Lower bound for ρ/λ (from spectral stability).
    rho_upper : float
        Upper bound for ρ/λ (from spectral stability).
    """

    beta_mu: float | np.ndarray = 0.0
    beta_sigma: float | np.ndarray = 1e6
    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    rho_lower: float = -0.999
    rho_upper: float = 0.999


@dataclass
class GibbsPriors:
    """Prior hyperparameters for the SAR-NB Gibbs sampler.

    All priors are weakly informative by default, matching the
    ``GaussianGibbsPriors`` convention.
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    sigma2_alpha: float = 2.0  # InverseGamma shape for σ²
    sigma2_beta: float = 1.0  # InverseGamma scale for σ²
    alpha_sigma: float = 10.0  # HalfNormal scale for α
    alpha_nu: float = 3.0  # Half-Student-t degrees of freedom for α
    rho_lower: float = -0.999
    rho_upper: float = 0.999


@dataclass
class ReducedGibbsPriors:
    """Prior hyperparameters for the reduced-form SAR-NB sampler.

    Notes
    -----
    The ``sigma2_*`` fields present on the structural-form
    :class:`GibbsPriors` are intentionally absent — this sampler has no
    :math:`\\sigma` parameter.
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    alpha_sigma: float = 2.5  # Half-Student-t scale for α
    alpha_nu: float = 3.0  # Half-Student-t degrees of freedom for α
    rho_lower: float = -0.999
    rho_upper: float = 0.999


@dataclass
class FlowReducedGibbsPriors:
    """Prior hyperparameters for the reduced-form flow NB sampler."""

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0
    rho_lower: float = -0.999
    rho_upper: float = 0.999


@dataclass
class ZINBGibbsPriors:
    """Prior hyperparameters for the ZINB Gibbs sampler.

    No σ² parameter in either equation: the logit link absorbs the
    error scale, and the reduced-form NB has no latent noise term.
    """

    # Selection equation
    gamma_mu: np.ndarray | float = 0.0
    gamma_sigma: np.ndarray | float = 1e6
    lam_lower: float = -0.999
    lam_upper: float = 0.999
    # Count equation
    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 10.0
    rho_lower: float = -0.999
    rho_upper: float = 0.999
    alpha_sigma: float = 2.5
    alpha_nu: float = 3.0


@dataclass
class LogitGibbsPriors(GibbsBasePriors):
    """Prior hyperparameters for the SAR-logit Gibbs sampler.

    All priors are weakly informative by default.  There is no σ²
    parameter (the logit link absorbs the error scale) and no α
    parameter (binary response is always Bernoulli).
    """


@dataclass
class SEMLogitGibbsPriors:
    """Prior hyperparameters for the SEM-logit Gibbs sampler.

    No σ² parameter (logit link absorbs error scale).
    """

    beta_mu: np.ndarray | float = 0.0
    beta_sigma: np.ndarray | float = 1e6
    lam_lower: float = -0.999
    lam_upper: float = 0.999


@dataclass
class REGibbsPriors(GibbsBasePriors):
    """Prior hyperparameters for RE panel Gibbs sampler.

    Inherits ``beta_mu`` / ``beta_sigma`` / ``rho_lower`` / ``rho_upper``
    from :class:`GibbsBasePriors`.  The models resolve every field to the
    same values their NUTS build uses, so both backends target one
    posterior:

    * ``sigma2_alpha``, ``sigma2_beta``: σ² ~ Inv-Γ(sigma2_alpha,
      sigma2_beta), the models' default ``Inv-Γ(2, Var(y))``.
    * ``sigma_alpha_nu``, ``sigma_alpha_scale``: σ_α ~ half-t_ν(0, A),
      half-Cauchy by default (see :class:`PanelREMixinPriors`), sampled
      through the Huang & Wand (2013) inverse-gamma mixture.
    """

    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    sigma_alpha_nu: float = 1.0
    sigma_alpha_scale: float = 1.0


@dataclass
class PanelGaussianPriors:
    r"""Prior hyperparameters for the Gaussian panel flow Gibbs sampler.

    All priors are weakly informative by default, matching the
    ``GibbsPriors`` / ``FlowGibbsPriors`` convention.

    Parameters
    ----------
    beta_mu : float, default 0.0
        Normal prior mean for :math:`\beta`.
    beta_sigma : float, default 1e6
        Normal prior standard deviation for :math:`\beta`.
    sigma2_alpha : float, default 2.0
        Inverse-Gamma shape for :math:`\sigma^2_u`.
    sigma2_beta : float, default 1.0
        Inverse-Gamma scale for :math:`\sigma^2_u`.
    sigma2_y_alpha : float, default 2.0
        Inverse-Gamma shape for :math:`\sigma^2_y`.
    sigma2_y_beta : float, default 1.0
        Inverse-Gamma scale for :math:`\sigma^2_y`.
    gamma_prior_var : float, default 1.0
        Prior variance for :math:`\gamma \sim N(0, \sigma^2_\gamma)`
        truncated to :math:`(-1, 1)`.
    rho_lower : float, default -0.999
        Lower bound for :math:`\rho_d, \rho_o`.
    rho_upper : float, default 0.999
        Upper bound for :math:`\rho_d, \rho_o`.
    """

    beta_mu: float | np.ndarray = 0.0
    beta_sigma: float | np.ndarray = 1e6
    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    sigma2_y_alpha: float = 2.0
    sigma2_y_beta: float = 1.0
    gamma_prior_var: float = 1.0
    rho_lower: float = -0.999
    rho_upper: float = 0.999


__all__ = [
    "BasePriors",
    "OLSPriors",
    "SLXPriors",
    "SARPriors",
    "SEMPriors",
    "SDMPriors",
    "SDEMPriors",
    "NegBinPriors",
    "SARNegBinPriors",
    "SARTobitPriors",
    "SEMTobitPriors",
    "SDMTobitPriors",
    "SARProbitPriors",
    "SARLogitPriors",
    "SEMLogitPriors",
    "PanelBasePriors",
    "PanelOLSPriors",
    "PanelSLXPriors",
    "PanelSARPriors",
    "PanelSEMPriors",
    "PanelSDMPriors",
    "PanelSDEMPriors",
    "PanelOLSREPriors",
    "PanelSARREPriors",
    "PanelSEMREPriors",
    "PanelSDEMREPriors",
    "PanelSARTobitPriors",
    "PanelSEMTobitPriors",
    "PanelDynamicBasePriors",
    "PanelOLSDynamicPriors",
    "PanelSLXDynamicPriors",
    "PanelSARDynamicPriors",
    "PanelSEMDynamicPriors",
    "PanelSDMRDynamicPriors",
    "PanelSDMUDynamicPriors",
    "PanelSDEMDynamicPriors",
    # Gibbs sampler prior structs
    "GibbsBasePriors",
    "GaussianGibbsPriors",
    "GibbsPriors",
    "ReducedGibbsPriors",
    "FlowReducedGibbsPriors",
    "ZINBGibbsPriors",
    "LogitGibbsPriors",
    "SEMLogitGibbsPriors",
    "REGibbsPriors",
    "PanelGaussianPriors",
    "PriorsLike",
    "resolve_priors",
    "priors_as_dict",
]
