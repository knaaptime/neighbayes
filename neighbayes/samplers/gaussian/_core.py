"""Gaussian spatial Gibbs sampler for SAR, SEM, SDM, SDEM models.

Implements a partially collapsed Gibbs sampler (van Dyk & Park, 2008) that
exploits conditional conjugacy in Gaussian spatial regression models.  Each
sweep is

1. σ² | β, ρ, y  — conjugate inverse-gamma (direct draw)
2. ρ/λ | σ², y   — 1-D slice sampling on the density with β integrated out
   under its Normal prior
3. β | ρ, σ², y  — conjugate normal (direct draw)

Step 2 marginalizes β, so step 3 must follow it immediately: the stored
``(β, σ², ρ)`` triple is then a draw from the joint posterior, not a
pairing of ρ with a β drawn under the previous ρ.  All three steps use the
model's own priors (β ~ N(μ₀, diag(s²)), σ² ~ InvGamma(a, b), ρ uniform),
so the Gibbs and NUTS paths target the same posterior.

**SAR/SDM**: with ``P = XᵀX/σ² + Λ₀⁻¹`` independent of ρ, the ρ density is
log|I - ρW| plus a quadratic in ρ whose coefficients are computed once per
sweep, so each slice evaluation is O(1) beyond the log-determinant.

**SEM/SDEM**: ``P(λ)`` depends on λ through quadratic-in-λ Gram matrices,
so each evaluation is a k×k Cholesky.

**Robust (Student-t) errors**: ``εᵢ ~ t_ν(0, σ)`` is written as the scale
mixture ``εᵢ | vᵢ ~ N(0, σ² vᵢ)``, ``vᵢ ~ InvGamma(ν/2, ν/2)`` (Geweke,
1993; LeSage's ``sar_g``).  Each sweep first draws
``vᵢ | ε ~ InvGamma((ν + 1)/2, (ν + εᵢ²/σ²)/2)`` and then runs the blocks
above on moments weighted by ``1/vᵢ``.  ν is fixed, as in the NUTS path.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.

Geweke, J. (1993). Bayesian treatment of the independent Student-t linear
model. *Journal of Applied Econometrics*, 8(S1), S19–S40.

van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers.
*Journal of the American Statistical Association*, 103(482), 790–796.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from ...models.priors import GaussianGibbsPriors
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
)
from ._loglik import _eps_log_density

# ---------------------------------------------------------------------------
# State and configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GaussianGibbsState:
    """Mutable state for the Gaussian spatial Gibbs sampler.

    Parameters
    ----------
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma2 : float
        Residual variance σ².
    rho : float
        Spatial autoregressive parameter (ρ for SAR/SDM, λ for SEM/SDEM).
    """

    beta: np.ndarray
    sigma2: float
    rho: float
    # Student-t mixing variances vᵢ (robust models only).
    v: np.ndarray | None = None


@dataclass
class GaussianGibbsCache:
    """Precomputed data that doesn't change across Gibbs sweeps.

    Parameters
    ----------
    XtX : ndarray of shape (k, k)
        X^T X matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
        Used for solving linear systems and quadratic forms involving
        (X^T X)^{-1} without forming the explicit inverse.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable for arrays of rho values.
    rho_lower : float
        Lower bound for ρ/λ.
    rho_upper : float
        Upper bound for ρ/λ.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed for SAR/SDM).
    W_sparse : csr_matrix or None
        Sparse W matrix (for SEM/SDEM residual filtering).
    """

    XtX: np.ndarray
    XtX_cho: tuple  # Cholesky factor from cho_factor(XtX)
    logdet_fn: Callable[[float], float]
    logdet_vec_fn: Callable
    rho_lower: float
    rho_upper: float
    model_type: str = "sar"
    Wy: np.ndarray | None = None
    W_sparse: sp.csr_matrix | None = None
    # SDM/SDEM additional precomputed quantities
    WX: np.ndarray | None = None
    XtWX: np.ndarray | None = None
    WXtWX: np.ndarray | None = None
    yty: float | None = None
    yTWy: float | None = None
    WyTWy: float | None = None
    XTy: np.ndarray | None = None
    XTWy: np.ndarray | None = None
    WXTy: np.ndarray | None = None
    WXTWy: np.ndarray | None = None
    # Student-t degrees of freedom; ``None`` for Gaussian errors.
    nu: float | None = None
    # Fixed-effects panels (Lee & Yu 2010): the independent observations the
    # variance counts (``None``: every row) and the coefficient m of the
    # -m·log(1 - ρ) time-effects Jacobian term.
    n_eff: int | None = None
    jacobian_shift: float = 0.0


def _jacobian(cache: GaussianGibbsCache, rho):
    """``log|I - ρW|`` as the likelihood uses it, with any time-effects term."""
    ld = cache.logdet_fn(rho)
    if cache.jacobian_shift:
        ld = ld - cache.jacobian_shift * np.log1p(-rho)
    return ld


# ---------------------------------------------------------------------------
# Block samplers
# ---------------------------------------------------------------------------


def _sample_beta_conjugate(
    r: np.ndarray,
    X: np.ndarray,
    XtX: np.ndarray,
    sigma2: float,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior.

    Model: r = X β + ε,  ε ~ N(0, σ² I)
    Prior: β ~ N(μ₀, Λ₀)

    Posterior: β | · ~ N(β̂, Σ_β)
    where Σ_β = (X^T X / σ² + Λ₀⁻¹)⁻¹
          β̂ = Σ_β (X^T r / σ² + Λ₀⁻¹ μ₀)

    Parameters
    ----------
    r : ndarray of shape (n,)
        Response (or residual) vector.
    X : ndarray of shape (n, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        X^T X (precomputed).
    sigma2 : float
        Current residual variance.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta : ndarray of shape (k,)
        New draw from the conditional posterior.
    """
    k = X.shape[1]
    beta_sigma_arr = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=float), (k,))
    beta_mu_arr = np.broadcast_to(np.asarray(priors.beta_mu, dtype=float), (k,))
    prior_prec_diag = 1.0 / beta_sigma_arr**2

    post_prec = XtX / sigma2
    post_prec[np.diag_indices_from(post_prec)] += prior_prec_diag
    Xtr = X.T @ r
    rhs = Xtr / sigma2 + prior_prec_diag * beta_mu_arr

    # Cholesky factorization: post_prec = L Lᵀ (SPD, lower-triangular L)
    # post_mean = post_prec⁻¹ @ rhs via two triangular solves
    # β = post_mean + L⁻ᵀ z,  z ~ N(0, I)  avoids forming inv(post_prec)
    # Cov(L⁻ᵀ z) = L⁻ᵀ L⁻¹ = (L Lᵀ)⁻¹ = post_prec⁻¹  ✓
    # NB: must request lower=True; the scipy default returns the *upper*
    # Cholesky U (A = UᵀU), in which case solve_triangular(U, z, trans='T')
    # yields U⁻ᵀ z whose covariance is U⁻ᵀ U⁻¹ ≠ A⁻¹ — a silent bug.
    L, lower = cho_factor(post_prec, lower=True)
    post_mean = cho_solve((L, lower), rhs)
    z = rng.standard_normal(k)
    beta = post_mean + solve_triangular(L, z, lower=lower, trans="T")
    return beta


def _sample_sigma2(
    rho: float,
    beta: np.ndarray,
    y: np.ndarray,
    Wy: np.ndarray | None,
    W_sparse: sp.csr_matrix | None,
    X: np.ndarray,
    priors: GaussianGibbsPriors,
    model_type: str,
    rng: np.random.Generator,
    n_eff: int | None = None,
) -> float:
    """Sample σ² from its conjugate Inverse-Gamma full conditional.

    ``n_eff`` is the number of independent observations (default ``len(y)``);
    a fixed-effects panel has fewer than it has rows.

    With prior ``σ² ~ InverseGamma(α, β)`` and Gaussian likelihood the
    full conditional is

    .. math::

        \\sigma^2 \\mid \\beta, \\rho, y
            \\sim
            \\mathrm{InverseGamma}\\!\\left(\\alpha + \\tfrac{n}{2},\\;
                \\beta + \\tfrac{1}{2} \\lVert \\varepsilon \\rVert^2\\right),

    where the residual depends on the model:

    - SAR/SDM:   ε = y - ρ W y - X β
    - SEM/SDEM:  ε = (I - λ W)(y - X β)

    This is the standard LeSage (2009) / Anselin Bayesian-spatial Gibbs
    update.  The same prior is placed on σ² in the NUTS path so the two
    samplers target identical posteriors.

    Parameters
    ----------
    rho : float
        Current spatial parameter (ρ for SAR/SDM, λ for SEM/SDEM).
    beta : ndarray of shape (k,)
        Current regression coefficients.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,) or None
        W @ y (for SAR/SDM).
    W_sparse : csr_matrix or None
        Sparse W (for SEM/SDEM).
    X : ndarray of shape (n, k)
        Design matrix.
    priors : GaussianGibbsPriors
        Prior hyperparameters.  Uses ``sigma2_alpha`` (shape) and
        ``sigma2_beta`` (scale/rate) for the InverseGamma prior.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    sigma2 : float
        Draw from the full conditional.
    """
    n = len(y) if n_eff is None else int(n_eff)

    if model_type in ("sar", "sdm"):
        resid = y - rho * Wy - X @ beta
        ss = np.dot(resid, resid)
    else:  # sem, sdem
        resid_raw = y - X @ beta
        eps = resid_raw - rho * (W_sparse @ resid_raw)
        ss = np.dot(eps, eps)

    a_post = priors.sigma2_alpha + n / 2.0
    b_post = priors.sigma2_beta + ss / 2.0
    return 1.0 / rng.gamma(a_post, 1.0 / b_post)


# ---------------------------------------------------------------------------
# ρ/λ | σ², y with β integrated out
# ---------------------------------------------------------------------------


def _beta_prior(priors: GaussianGibbsPriors, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the prior mean and diagonal prior precision of β as length-k arrays."""
    mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=float), (k,))
    s = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=float), (k,))
    return mu, 1.0 / s**2


@dataclass
class SarSweepTerms:
    """ρ-independent quantities for one SAR/SDM sweep at fixed σ².

    With ``P = XᵀX/σ² + Λ₀⁻¹`` and ``b(ρ) = b0 − ρ b1`` where
    ``b0 = Xᵀy/σ² + Λ₀⁻¹μ₀`` and ``b1 = XᵀWy/σ²``, the posterior of β given
    (ρ, σ²) is ``N(P⁻¹ b(ρ), P⁻¹)``, and ``b(ρ)ᵀP⁻¹b(ρ) = c00 − 2ρ c01 + ρ² c11``.
    """

    P_cho: tuple
    m0: np.ndarray
    m1: np.ndarray
    c00: float
    c01: float
    c11: float


def _sar_sweep_terms(
    cache: GaussianGibbsCache, sigma2: float, priors: GaussianGibbsPriors
) -> SarSweepTerms:
    """Factor ``P`` once per sweep and reduce ``bᵀP⁻¹b`` to a quadratic in ρ."""
    k = cache.XtX.shape[0]
    mu, pprec = _beta_prior(priors, k)
    P = cache.XtX / sigma2
    P[np.diag_indices_from(P)] += pprec
    P_cho = cho_factor(P, lower=True)
    b0 = cache.XTy / sigma2 + pprec * mu
    b1 = cache.XTWy / sigma2
    m0 = cho_solve(P_cho, b0)
    m1 = cho_solve(P_cho, b1)
    return SarSweepTerms(
        P_cho=P_cho,
        m0=m0,
        m1=m1,
        c00=float(b0 @ m0),
        c01=float(b0 @ m1),
        c11=float(b1 @ m1),
    )


def _sar_log_density_given_sigma2(
    rho: float,
    cache: GaussianGibbsCache,
    sigma2: float,
    terms: SarSweepTerms,
) -> float:
    """log p(ρ | σ², y) for SAR/SDM with β integrated out under its Normal prior.

    .. math::

        \\log p(\\rho \\mid \\sigma^2, y) = \\log|I - \\rho W|
            - \\frac{r^\\top r}{2\\sigma^2}
            + \\tfrac12 b(\\rho)^\\top P^{-1} b(\\rho) + \\text{const},

    with ``r = y − ρWy``.  ``rᵀr`` is quadratic in ρ from cached inner products
    and ``bᵀP⁻¹b`` is quadratic in ρ from ``terms``, so an evaluation costs the
    log-determinant plus O(1).
    """
    r_dot_r = cache.yty - 2.0 * rho * cache.yTWy + rho * rho * cache.WyTWy
    quad = terms.c00 - 2.0 * rho * terms.c01 + rho * rho * terms.c11
    return _jacobian(cache, rho) - 0.5 * r_dot_r / sigma2 + 0.5 * quad


def _sem_precision_terms(
    lam: float,
    cache: GaussianGibbsCache,
    sigma2: float,
    priors: GaussianGibbsPriors,
) -> tuple[tuple, np.ndarray, float]:
    """Return ``(cho(P(λ)), b(λ), y*ᵀy*)`` for SEM/SDEM at fixed σ².

    ``P(λ) = X*ᵀX*/σ² + Λ₀⁻¹`` and ``b(λ) = X*ᵀy*/σ² + Λ₀⁻¹μ₀`` with
    ``y* = (I − λW)y`` and ``X* = (I − λW)X``; the cross-products are
    quadratics in λ from the λ-independent terms on the cache.
    """
    k = cache.XtX.shape[0]
    mu, pprec = _beta_prior(priors, k)
    lam2 = lam * lam
    XtX_star = cache.XtX - lam * (cache.XtWX + cache.XtWX.T) + lam2 * cache.WXtWX
    Xty_star = cache.XTy - lam * (cache.XTWy + cache.WXTy) + lam2 * cache.WXTWy
    yty_star = cache.yty - 2.0 * lam * cache.yTWy + lam2 * cache.WyTWy
    P = XtX_star / sigma2
    P[np.diag_indices_from(P)] += pprec
    return cho_factor(P, lower=True), Xty_star / sigma2 + pprec * mu, yty_star


def _sem_log_density_given_sigma2(
    lam: float,
    cache: GaussianGibbsCache,
    sigma2: float,
    priors: GaussianGibbsPriors,
) -> float:
    """log p(λ | σ², y) for SEM/SDEM with β integrated out under its Normal prior.

    .. math::

        \\log p(\\lambda \\mid \\sigma^2, y) = \\log|I - \\lambda W|
            - \\frac{y^{*\\top} y^*}{2\\sigma^2}
            - \\tfrac12 \\log|P(\\lambda)|
            + \\tfrac12 b(\\lambda)^\\top P(\\lambda)^{-1} b(\\lambda)
            + \\text{const}.

    Each evaluation is a k×k Cholesky; no O(n) work.
    """
    P_cho, b, yty_star = _sem_precision_terms(lam, cache, sigma2, priors)
    logdet_P = 2.0 * np.sum(np.log(np.diag(P_cho[0])))
    quad = b @ cho_solve(P_cho, b)
    return _jacobian(cache, lam) - 0.5 * yty_star / sigma2 - 0.5 * logdet_P + 0.5 * quad


def _draw_beta_from_cho(
    P_cho: tuple, mean: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``β ~ N(mean, P⁻¹)`` given the lower Cholesky factor of ``P``."""
    z = rng.standard_normal(mean.shape[0])
    return mean + solve_triangular(P_cho[0], z, lower=True, trans="T")


# ---------------------------------------------------------------------------
# ρ/λ slice sampler
# ---------------------------------------------------------------------------


def _sample_rho_sar(
    rho: float,
    cache: GaussianGibbsCache,
    sigma2: float,
    terms: SarSweepTerms,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
) -> float:
    """Slice sample ρ from p(ρ | σ², y) for SAR/SDM (β integrated out)."""

    def log_density(rho_val):
        return _sar_log_density_given_sigma2(rho_val, cache, sigma2, terms)

    rho_new, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
    )
    return rho_new


def _sample_lam_sem(
    lam: float,
    cache: GaussianGibbsCache,
    sigma2: float,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
) -> float:
    """Slice sample λ from p(λ | σ², y) for SEM/SDEM (β integrated out)."""

    def log_density(lam_val):
        return _sem_log_density_given_sigma2(lam_val, cache, sigma2, priors)

    lam_new, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        lam,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
    )
    return lam_new


def _residual(
    state: GaussianGibbsState, y: np.ndarray, X: np.ndarray, cache: GaussianGibbsCache
) -> np.ndarray:
    """ε = y − ρWy − Xβ (SAR/SDM) or (I − λW)(y − Xβ) (SEM/SDEM)."""
    if cache.model_type in ("sar", "sdm"):
        return y - state.rho * cache.Wy - X @ state.beta
    raw = y - X @ state.beta
    return raw - state.rho * (cache.W_sparse @ raw)


def _weighted_cache(
    cache: GaussianGibbsCache, y: np.ndarray, X: np.ndarray, w: np.ndarray
) -> GaussianGibbsCache:
    """The cache with every cross-product weighted by ``w = 1/v`` (Ω = diag(w)).

    O(n·k²) per call; the ρ/λ density and β draw then run unchanged on
    ``yᵀΩy``, ``XᵀΩX``, … in place of their unweighted counterparts.
    """
    Wy = cache.Wy
    wy, wWy, Xw = w * y, w * Wy, X * w[:, None]
    fields = dict(
        XtX=Xw.T @ X,
        yty=float(y @ wy),
        yTWy=float(Wy @ wy),
        WyTWy=float(Wy @ wWy),
        XTy=Xw.T @ y,
        XTWy=Xw.T @ Wy,
    )
    if cache.model_type in ("sem", "sdem"):
        WX = cache.WX
        WXw = WX * w[:, None]
        fields.update(
            XtWX=Xw.T @ WX,
            WXtWX=WXw.T @ WX,
            WXTy=WXw.T @ y,
            WXTWy=WXw.T @ Wy,
        )
    return replace(cache, **fields)


def gaussian_sweep(
    state: GaussianGibbsState,
    y: np.ndarray,
    X: np.ndarray,
    cache: GaussianGibbsCache,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
) -> None:
    """One partially collapsed Gibbs sweep, updating ``state`` in place.

    Order: [v | β, σ², ρ →] σ² | β, ρ[, v] → ρ | σ²[, v] (β integrated out)
    → β | ρ, σ²[, v].  The β draw must come straight after the ρ draw that
    marginalized it.  The bracketed v block runs only for robust models.
    """
    if cache.nu is not None:
        eps = _residual(state, y, X, cache)
        nu = cache.nu
        # vᵢ | · ~ InvGamma((ν+1)/2, (ν + εᵢ²/σ²)/2)
        state.v = (0.5 * (nu + eps * eps / state.sigma2)) / rng.gamma(
            0.5 * (nu + 1.0), size=eps.shape[0]
        )
        w = 1.0 / state.v
        n_obs = eps.shape[0] if cache.n_eff is None else cache.n_eff
        a_post = priors.sigma2_alpha + 0.5 * n_obs
        b_post = priors.sigma2_beta + 0.5 * float(w @ (eps * eps))
        state.sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
        cache = _weighted_cache(cache, y, X, w)
    else:
        state.sigma2 = _sample_sigma2(
            state.rho,
            state.beta,
            y,
            cache.Wy,
            cache.W_sparse,
            X,
            priors,
            cache.model_type,
            rng,
            n_eff=cache.n_eff,
        )
    if cache.model_type in ("sar", "sdm"):
        terms = _sar_sweep_terms(cache, state.sigma2, priors)
        state.rho = _sample_rho_sar(
            state.rho, cache, state.sigma2, terms, rng, slice_state
        )
        mean = terms.m0 - state.rho * terms.m1
        state.beta = _draw_beta_from_cho(terms.P_cho, mean, rng)
    else:  # sem, sdem
        state.rho = _sample_lam_sem(
            state.rho, cache, state.sigma2, priors, rng, slice_state
        )
        P_cho, b, _ = _sem_precision_terms(state.rho, cache, state.sigma2, priors)
        state.beta = _draw_beta_from_cho(P_cho, cho_solve(P_cho, b), rng)


def _pointwise_loglik(
    state: GaussianGibbsState, y: np.ndarray, X: np.ndarray, cache: GaussianGibbsCache
) -> np.ndarray:
    """Pointwise log-likelihood with the Jacobian spread evenly over observations.

    Normal for Gaussian errors; Student-t (the mixture's marginal, as the NUTS
    path stores) for robust models.
    """
    eps = _residual(state, y, X, cache)
    ll = _eps_log_density(eps, np.sqrt(state.sigma2), cache.nu)
    ll = ll + _jacobian(cache, state.rho) / eps.shape[0]
    return np.where(np.isfinite(ll), ll, -1e10)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_gaussian_gibbs(
    y: np.ndarray,
    X: np.ndarray,
    XtX_cho: tuple,
    priors: GaussianGibbsPriors,
    rng: np.random.Generator,
) -> GaussianGibbsState:
    """Warm-start the Gibbs sampler from an OLS fit.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    GaussianGibbsState
        Initial state with OLS-based starting values.
    """
    beta_ols = cho_solve(XtX_cho, X.T @ y)
    resid = y - X @ beta_ols
    sigma2_ols = np.dot(resid, resid) / len(y)
    # Start ρ/λ at 0 (no spatial effect)
    rho_init = 0.0

    return GaussianGibbsState(
        beta=beta_ols.copy(),
        sigma2=max(sigma2_ols, 1e-6),
        rho=rho_init,
    )


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_gaussian_chain(
    y: np.ndarray,
    X: np.ndarray,
    cache: GaussianGibbsCache,
    priors: GaussianGibbsPriors,
    init: GaussianGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    progressbar: bool = True,
    chain_id: int = 0,
    progress_manager: object | None = None,
    return_state: bool = False,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain of the Gaussian spatial Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    cache : GaussianGibbsCache
        Precomputed data.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    init : GaussianGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every thin-th draw after warmup.
    rng : numpy.random.Generator, optional
        Random state.
    progressbar : bool, default True
        Show progress bar for this chain.
    chain_id : int, default 0
        Chain index (0-based) for progress bar display.
    progress_manager : object or None
        ``GibbsProgressBarManager`` instance. If provided,
        ``update()`` is called after each iteration.
    return_state : bool, default False
        Also return the final :class:`GaussianGibbsState` under the key
        ``"_final_state"``, so the chain can be resumed by a later call.
    store_log_lik : bool, default True
        Compute the pointwise log-likelihood for each retained draw.  Set
        ``False`` for a phase whose draws are discarded — it costs an O(n·k)
        pass per iteration and an ``(n_keep, n)`` array that would otherwise be
        pickled back from every parallel worker.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho`` (or ``lam``), ``beta``,
        ``sigma``, and ``log_lik``.  Each array has shape
        ``(n_keep, ...)`` where n_keep = draws // thin.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws
    model_type = cache.model_type

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = (
        np.empty((n_keep, n), dtype=np.float64)
        if store_log_lik
        else np.empty((0, 0), dtype=np.float64)
    )

    # Copy initial state
    state = GaussianGibbsState(
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        rho=init.rho,
        v=None if init.v is None else init.v.copy(),
    )

    # Adaptive slice width for ρ/λ
    rho_range = cache.rho_upper - cache.rho_lower
    slice_state = SliceWidthState(w=rho_range * 0.1)

    for i in range(total_iters):
        gaussian_sweep(state, y, X, cache, priors, rng, slice_state)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            j = (i - tune) // thin
            rho_samples[j] = state.rho
            beta_samples[j] = state.beta
            sigma_samples[j] = np.sqrt(state.sigma2)

        # Pointwise log-likelihood (including Jacobian/n)
        if store_log_lik and i >= tune and (i - tune) % thin == 0:
            log_lik_samples[(i - tune) // thin] = _pointwise_loglik(state, y, X, cache)

        # Update progress bar
        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    # Name the spatial parameter appropriately
    param_name = "rho" if model_type in ("sar", "sdm") else "lam"
    result = {
        param_name: rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "log_lik": log_lik_samples,
    }
    if return_state:
        # Lets a caller stop a chain, adapt something the chain depends on, and
        # resume from where it left off — used by the warmup Jacobian refit,
        # which pools ρ across chains partway through warmup and rebuilds the
        # interpolant on the range they found.
        result["_final_state"] = GaussianGibbsState(
            beta=state.beta.copy(),
            sigma2=state.sigma2,
            rho=state.rho,
            v=None if state.v is None else state.v.copy(),
        )

    return result
