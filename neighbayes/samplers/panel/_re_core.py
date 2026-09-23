"""Gaussian spatial Gibbs sampler for panel models with random effects.

Implements a 5-block Gibbs sampler for RE panel models:

1. β | α, σ², ρ/λ, y  — conjugate normal (direct draw)
2. σ² | β, α, ρ/λ, y  — conjugate inverse-gamma (direct draw)
3. α | β, σ², ρ/λ, y  — conjugate normal (direct draw, vectorized)
4. σ_α² | α             — conjugate inverse-gamma (direct draw)
5. ρ/λ | β, σ², σ_α², y — 1-D slice sampling on marginalized density

Blocks 1-4 are all conjugate and can be sampled directly. Only block 5
(the spatial parameter) requires slice sampling, exactly as in the FE
Gibbs sampler.

The key difference from the FE (within-transformed) Gibbs sampler is:
- FE models demean the data, eliminating α and σ_α²
- RE models keep the raw data and estimate α_i ~ N(0, σ_α²) explicitly

Conditional posteriors
----------------------
Given the model:
    y_it = ρ(Wy)_it + X_it β + α_i + ε_it    (SAR-RE)
    ε_it ~ N(0, σ²),  α_i ~ N(0, σ_α²)

Or for SEM-RE:
    y_it = X_it β + α_i + u_it
    u_it = λ(Wu)_it + ε_it
    ε_it ~ N(0, σ²),  α_i ~ N(0, σ_α²)

Block 1 (β):
    r = y - ρWy - α_expanded   (SAR)  or  r = (I-λW)(y - α_expanded)  (SEM)
    β | rest ~ N(β̂, Σ_β)  — standard conjugate normal

Block 2 (σ²):
    resid = y - ρWy - Xβ - α_expanded   (SAR)  or  (I-λW)(y - Xβ - α_expanded)  (SEM)
    σ² | rest ~ Inv-Γ(a_post, b_post)

Block 3 (α):
    SAR-RE: α_i | rest ~ N(μ_i, τ_i²)  (diagonal conditional, each unit independent)
        where τ_i² = 1 / (T/σ² + 1/σ_α²),  μ_i = τ_i² × r_i / σ²

    SEM-RE: α | rest ~ N(μ, Σ_α)  (full N×N multivariate normal)
        because the spatial filter (I-λW) couples units across space.
        Σ_α^{-1} = (1/σ²) B^T B + (1/σ_α²) I_N,  where B = (I-λW)D
        μ = Σ_α × (1/σ²) B^T (I-λW)(y - Xβ)

Block 4 (σ_α²):
    σ_α² | α ~ Inv-Γ(a_post, b_post)
    where a_post = N/2 + ε,  b_post = Σα_i²/2 + ε  (Jeffreys-like prior)

Block 5 (ρ/λ):
    SAR-RE: Collapsed density that integrates out β and σ², conditioning
    on α.  Uses within-group demeaning (Frisch-Waugh-Lovell) to eliminate
    the intercept and compute the density efficiently.

    SEM-RE: Marginalized density that integrates out α analytically,
    breaking the α-λ correlation that causes slow mixing.  The marginalized
    density is:

        log p(λ | β, σ², σ_α², y) = T·log|I-λW| + ½·log|Σ_α|
            + (1/(2σ⁴))·(D^T A^T A r)^T Σ_α (D^T A^T A r)
            - (1/(2σ²))·r^T A^T A r

    where A = I-λW, B = AD, r = y - Xβ, and Σ_α^{-1} = (1/σ²)B^T B + (1/σ_α²)I_N.

    .. note::
        The marginalized density depends on β, σ², and σ_α², which change
        every Gibbs iteration.  Therefore the log-density cannot be cached
        across iterations (unlike the SAR-RE collapsed density, which only
        depends on ρ and α).

Identification warning
----------------------
For SEM-RE models, the spatial error parameter λ is weakly identified
when random effects α are present.  The random effects absorb spatial
correlation across units, making it difficult for the data to distinguish
between λ (spatial error dependence) and σ_α² (between-unit variance).
Both Gibbs and NUTS samplers will tend to estimate λ near zero even when
the true λ is moderate, because the posterior genuinely concentrates
there.  This is a model identification issue, not a sampler bug.

For SAR-RE models, ρ is better identified because the spatial lag
enters the model directly (Wy), providing more information to separate
ρ from α.

Possible remedies for SEM-RE identification:
- Use fixed effects (within transform) instead of random effects
- Use a Spatial Durbin (SDM) specification that includes WX terms
- Use longer panels (T → ∞) which provide more information
- Use the Mundlak specification (α_i = X̄_i γ + η_i) to test for
  RE-regressor correlation, though this does not resolve the λ
  identification issue itself

References
----------
Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.).
Springer.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.

Mundlak, Y. (1978). On the pooling of time series and cross section data.
*Econometrica*, 46(1), 69–85.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from ...models.priors import REGibbsPriors
from .._utils._base import GibbsBaseState
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
)

# ---------------------------------------------------------------------------
# State and configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class REGibbsState(GibbsBaseState):
    """Mutable state for the RE panel Gibbs sampler.

    Parameters
    ----------
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma2 : float
        Residual variance σ².
    alpha : ndarray of shape (N,)
        Unit random effects.
    sigma_alpha2 : float
        Variance of random effects σ_α².
    rho : float
        Spatial autoregressive parameter (ρ for SAR, λ for SEM).
    """

    sigma2: float = 1.0
    alpha: np.ndarray = None
    sigma_alpha2: float = 1.0


@dataclass
class REGibbsCache:
    """Precomputed data that doesn't change across Gibbs sweeps for RE models.

    Extends GaussianGibbsCache with RE-specific precomputed data.

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
        One of "sar", "sem".
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed for SAR).
    W_sparse : csr_matrix or None
        Sparse W matrix (for SEM residual filtering).
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index (obs i → unit i % N).
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
    N: int = 1
    T: int = 1
    unit_idx: np.ndarray | None = None
    # SEM-RE only: λ-independent terms of BᵀB = DᵀAᵀAD, precomputed once
    # (M1 = DᵀWD, M2 = DᵀWᵀWD).  See ``_sem_re_BtB``.
    sem_BtB_M1: np.ndarray | None = None
    sem_BtB_M2: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Block samplers
# ---------------------------------------------------------------------------


def _sample_beta_re_sar(
    rho: float,
    sigma2: float,
    alpha_expanded: np.ndarray,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    XtX: np.ndarray,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior (SAR-RE).

    Model: r = y - ρWy - α_expanded = Xβ + ε,  ε ~ N(0, σ²I)

    Parameters
    ----------
    rho : float
        Current spatial autoregressive parameter.
    sigma2 : float
        Current residual variance.
    alpha_expanded : ndarray of shape (n,)
        Random effects expanded to observation level (α[unit_idx]).
    y, Wy, X, XtX, priors, rng
        As in the FE Gibbs sampler.

    Returns
    -------
    beta : ndarray of shape (k,)
    """
    r = y - rho * Wy - alpha_expanded
    return _sample_beta_conjugate(r, X, XtX, sigma2, priors, rng)


def _sample_beta_re_sem(
    lam: float,
    sigma2: float,
    alpha_expanded: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior (SEM-RE).

    Model: (I-λW)(y - α_expanded) = (I-λW)X β + ε,  ε ~ N(0, σ²I)

    Parameters
    ----------
    lam : float
        Current spatial error parameter.
    sigma2, alpha_expanded, y, X, W_sparse, priors, rng
        As described above.

    Returns
    -------
    beta : ndarray of shape (k,)
    """
    r = y - alpha_expanded
    r_star = r - lam * (W_sparse @ r)
    X_star = X - lam * (W_sparse @ X)
    XtX_star = X_star.T @ X_star
    return _sample_beta_conjugate(r_star, X_star, XtX_star, sigma2, priors, rng)


def _sample_beta_conjugate(
    r: np.ndarray,
    X: np.ndarray,
    XtX: np.ndarray,
    sigma2: float,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample β from conjugate normal posterior.

    Model: r = X β + ε,  ε ~ N(0, σ² I)
    Prior: β ~ N(μ₀, Λ₀)

    Posterior: β | · ~ N(β̂, Σ_β)
    where Σ_β = (X^T X / σ² + Λ₀⁻¹)⁻¹
          β̂ = Σ_β (X^T r / σ² + Λ₀⁻¹ μ₀)
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
    # Must request lower=True so that solve_triangular(L, z, trans='T')
    # produces L⁻ᵀ z with Cov = (L Lᵀ)⁻¹ = post_prec⁻¹.  (See
    # _gaussian_gibbs._sample_beta_conjugate for the full explanation.)
    L, lower = cho_factor(post_prec, lower=True)
    post_mean = cho_solve((L, lower), rhs)
    z = rng.standard_normal(k)
    beta = post_mean + solve_triangular(L, z, lower=lower, trans="T")
    return beta


def _sample_sigma2_re(
    rho: float,
    beta: np.ndarray,
    alpha_expanded: np.ndarray,
    y: np.ndarray,
    Wy: np.ndarray | None,
    W_sparse: sp.csr_matrix | None,
    X: np.ndarray,
    priors: REGibbsPriors,
    model_type: str,
    rng: np.random.Generator,
) -> float:
    """Sample σ² from conjugate inverse-gamma posterior (RE model).

    For SAR-RE:
        resid = y - ρWy - Xβ - α_expanded
        σ² | · ~ Inv-Γ(a_post, b_post)

    For SEM-RE:
        resid_raw = y - Xβ - α_expanded
        ε = (I - λW) resid_raw
        σ² | · ~ Inv-Γ(a_post, b_post)

    Uses Jeffreys prior p(σ²) ∝ 1/σ².
    """
    n = len(y)

    if model_type in ("sar", "sdm"):
        resid = y - rho * Wy - X @ beta - alpha_expanded
        ss = np.dot(resid, resid)
    else:  # sem, sdem
        resid_raw = y - X @ beta - alpha_expanded
        eps = resid_raw - rho * (W_sparse @ resid_raw)
        ss = np.dot(eps, eps)

    EPS = 1e-3
    a_post = n / 2 + EPS
    b_post = ss / 2 + EPS

    sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
    return sigma2


def _sem_re_unit_aggregated_terms(
    W_sparse: sp.csr_matrix, unit_idx: np.ndarray, N: int
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the λ-independent terms of ``BᵀB = DᵀAᵀAD`` for SEM-RE.

    With ``A = I - λW`` and ``D`` the ``(NT × N)`` unit-indicator matrix,

        BᵀB(λ) = DᵀD − λ(DᵀWD + DᵀWᵀD) + λ² DᵀWᵀWD

    For a balanced panel ``DᵀD = T·I_N``.  This returns ``(M1, M2)`` with
    ``M1 = DᵀWD`` and ``M2 = DᵀWᵀWD`` (dense ``N × N``), so that

        BᵀB(λ) = T·I_N − λ(M1 + M1ᵀ) + λ² M2.

    Both terms depend only on ``W`` and the panel structure, so they are
    computed once (at cache build time) instead of rebuilding the dense
    ``NT × N`` matrix ``B`` on every Gibbs sweep (``O(N²·NT)`` → ``O(N²)``).
    """
    n = len(unit_idx)
    D = sp.csr_matrix((np.ones(n), (np.arange(n), unit_idx)), shape=(n, N))
    M1 = np.asarray((D.T @ W_sparse @ D).todense())
    WD = W_sparse @ D
    M2 = np.asarray((WD.T @ WD).todense())
    return M1, M2


def _sem_re_BtB(
    lam: float, M1: np.ndarray, M2: np.ndarray, T: int, N: int
) -> np.ndarray:
    """Closed-form ``BᵀB(λ) = T·I_N − λ(M1 + M1ᵀ) + λ² M2`` (see above)."""
    return T * np.eye(N) - lam * (M1 + M1.T) + (lam * lam) * M2


def _sample_alpha_re(
    rho: float,
    beta: np.ndarray,
    sigma2: float,
    y: np.ndarray,
    Wy: np.ndarray | None,
    W_sparse: sp.csr_matrix | None,
    X: np.ndarray,
    N: int,
    T: int,
    unit_idx: np.ndarray,
    sigma_alpha2: float,
    priors: REGibbsPriors,
    model_type: str,
    rng: np.random.Generator,
    M1: np.ndarray | None = None,
    M2: np.ndarray | None = None,
) -> np.ndarray:
    """Sample α (unit random effects) from conditional posterior.

    **SAR-RE**: The conditional is conjugate and diagonal:
        α_i | rest ~ N(μ_i, τ_i²)
    where τ_i² = 1/(T/σ² + 1/σ_α²) and μ_i = τ_i² × (Σ_t r_it)/σ²,
    with r = y - ρWy - Xβ.  Each α_i is independent given the rest,
    so sampling is fully vectorized.

    **SEM-RE**: The spatial filter (I - λW) couples α values across
    units, so the conditional is multivariate normal:
        α | rest ~ N(μ_α, Σ_α)
    where Σ_α^{-1} = (1/σ²) D^T A^T A D + (1/σ_α²) I_N
    and   μ_α = Σ_α × (1/σ²) D^T A^T A r
    with A = I - λW_NT, r = y - Xβ, D = unit indicator matrix.
    This requires solving an N × N linear system.

    Parameters
    ----------
    rho : float
        Current spatial parameter (ρ for SAR, λ for SEM).
    beta : ndarray of shape (k,)
        Current regression coefficients.
    sigma2 : float
        Current residual variance.
    y : ndarray of shape (n,)
        Response vector.
    Wy : ndarray of shape (n,) or None
        W @ y (for SAR).
    W_sparse : csr_matrix or None
        Sparse W (for SEM).
    X : ndarray of shape (n, k)
        Design matrix.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.
    sigma_alpha2 : float
        Current random effects variance.
    priors : REGibbsPriors
        Prior hyperparameters.
    model_type : str
        One of "sar", "sem".
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    alpha : ndarray of shape (N,)
        New draw from the conditional posterior.
    """
    if model_type in ("sar", "sdm"):
        # SAR: α conditional is diagonal — each α_i is independent
        r = y - rho * Wy - X @ beta
        r_sum = np.bincount(unit_idx, weights=r, minlength=N)
        tau2 = 1.0 / (T / sigma2 + 1.0 / sigma_alpha2)
        mu = tau2 * r_sum / sigma2
        alpha = rng.normal(loc=mu, scale=np.sqrt(tau2))
        return alpha

    # SEM: α conditional is multivariate normal due to spatial coupling
    # Model: A(y - Xβ - Dα) = ε,  A = I - λW_NT
    # Conditional: α | rest ~ N(μ_α, Σ_α)
    #   Σ_α^{-1} = (1/σ²) D^T A^T A D + (1/σ_α²) I_N
    #   μ_α = Σ_α × (1/σ²) D^T A^T A r,  r = y - Xβ
    lam = rho  # λ is stored in state.rho for SEM
    r = y - X @ beta  # NT × 1

    # Compute A @ r = (I - λW) @ r
    Ar = r - lam * (W_sparse @ r)

    # Compute A^T @ Ar = (I - λW^T) @ Ar
    # For row-standardized W, W^T ≠ W in general.
    # W_sparse is the NT × NT block-diagonal matrix.
    WtAr = W_sparse.T @ Ar
    AtAr = Ar - lam * WtAr

    # D^T @ AtAr: sum AtAr values by unit
    DtAtAr = np.bincount(unit_idx, weights=AtAr, minlength=N)

    # B^T B = D^T A^T A D, evaluated via the closed form in λ from the
    # precomputed λ-independent terms M1 = D^T W D, M2 = D^T W^T W D:
    #   B^T B(λ) = T·I_N − λ(M1 + M1ᵀ) + λ² M2
    # (avoids rebuilding the dense NT × N matrix B every sweep).
    if M1 is None or M2 is None:
        M1, M2 = _sem_re_unit_aggregated_terms(W_sparse, unit_idx, N)
    BtB = _sem_re_BtB(lam, M1, M2, T, N)  # N × N
    # Precision matrix
    prec_alpha = (1.0 / sigma2) * BtB + (1.0 / sigma_alpha2) * np.eye(N)
    # Cholesky factorization: prec_alpha = L Lᵀ (SPD, lower-triangular L)
    # Must request lower=True so that solve_triangular(L, z, trans='T')
    # produces L⁻ᵀ z with Cov = (L Lᵀ)⁻¹ = prec_alpha⁻¹.  (See
    # _gaussian_gibbs._sample_beta_conjugate for the full explanation.)
    rhs_alpha = (1.0 / sigma2) * DtAtAr
    L, lower = cho_factor(prec_alpha, lower=True)
    mean_alpha = cho_solve((L, lower), rhs_alpha)
    z = rng.standard_normal(N)
    alpha = mean_alpha + solve_triangular(L, z, lower=lower, trans="T")
    return alpha


def _sample_sigma_alpha2(
    alpha: np.ndarray,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> float:
    """Sample σ_α² from conjugate inverse-gamma posterior.

    Prior: p(σ_α²) ∝ 1/σ_α²  (Jeffreys, approximated as Inv-Γ(ε, ε))

    Posterior: σ_α² | α ~ Inv-Γ(a_post, b_post)
    where a_post = N/2 + ε,  b_post = Σα_i²/2 + ε
    """
    N = len(alpha)
    EPS = 1e-3
    a_post = N / 2 + EPS
    b_post = np.dot(alpha, alpha) / 2 + EPS

    sigma_alpha2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
    return sigma_alpha2


# ---------------------------------------------------------------------------
# Collapsed ρ/λ log-density for RE models
# ---------------------------------------------------------------------------


def _sar_re_collapsed_log_density(
    rho: float,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    XtX_cho: tuple,
    logdet_fn: Callable[[float], float],
    n: int,
    k: int,
    N: int,
    T: int,
    unit_idx: np.ndarray,
) -> float:
    """Collapsed log p(ρ | y) for SAR-RE model.

    Integrates out β, σ², α, and σ_α² analytically.  The collapsed
    density uses the within-group residual structure:

        r = y - ρWy
        RSS(ρ) = r^T M_Z r

    where Z = [X, D] with D being the N unit dummies (via unit_idx),
    and M_Z = I - Z(Z^T Z)^{-1} Z^T.

    Using the Woodbury form:
        r^T M_Z r = r^T r - (Z^T r)^T (Z^T Z)^{-1} (Z^T r)

    However, Z^T Z has a block structure that makes direct inversion
    expensive. Instead, we use the Frisch-Waugh-Lovell (FWL) approach:

    1. Partial out unit effects from r and X:
       M_D = I - D(D^T D)^{-1} D^T  (within-group demeaning)
       r̃ = M_D r, X̃ = M_D X

    2. Then RSS(ρ) = r̃^T M_{X̃} r̃ = r̃^T r̃ - (X̃^T r̃)^T (X̃^T X̃)^{-1} (X̃^T r̃)

    This avoids forming the (k+N) × (k+N) matrix Z^T Z.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.
    y, Wy, X, XtX_cho, logdet_fn, n, k
        As in the FE collapsed density.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.

    Returns
    -------
    log_density : float
    """
    r = y - rho * Wy

    # Within-group demeaning (FWL: partial out unit effects)
    # For each unit i, compute group mean and subtract
    group_counts = np.bincount(unit_idx, minlength=N)
    r_group_sum = np.bincount(unit_idx, weights=r, minlength=N)
    r_demeaned = r - r_group_sum[unit_idx] / group_counts[unit_idx]

    X_group_sum = np.zeros((N, k))
    for j in range(k):
        X_group_sum[:, j] = np.bincount(unit_idx, weights=X[:, j], minlength=N)
    X_demeaned = X - X_group_sum[unit_idx] / group_counts[unit_idx, None]

    # Drop columns that are zero after demeaning (e.g. intercept).
    # Within-group demeaning turns constant columns into zeros, making
    # XtX_tilde singular.  The intercept is absorbed by the unit effects α.
    col_norms = np.linalg.norm(X_demeaned, axis=0)
    nonzero_mask = col_norms > 1e-10
    X_tilde = X_demeaned[:, nonzero_mask]
    k_eff = int(nonzero_mask.sum())

    # RSS via Cholesky: r̃^T r̃ - (X̃^T r̃)^T (X̃^T X̃)^{-1} (X̃^T r̃)
    XtX_tilde = X_tilde.T @ X_tilde
    Xtr_tilde = X_tilde.T @ r_demeaned
    rtr_tilde = np.dot(r_demeaned, r_demeaned)

    c, lower = cho_factor(XtX_tilde)
    rss = rtr_tilde - Xtr_tilde @ cho_solve((c, lower), Xtr_tilde)

    logdet = logdet_fn(rho)
    # Degrees of freedom: n - k_eff - N (k_eff non-zero regressors + N unit effects)
    return logdet - 0.5 * (n - k_eff - N) * np.log(max(rss, 1e-300))


def _sem_re_marginalized_log_density(
    lam: float,
    beta: np.ndarray,
    sigma2: float,
    sigma_alpha2: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_fn: Callable[[float], float],
    n: int,
    k: int,
    N: int,
    T: int,
    unit_idx: np.ndarray,
    M1: np.ndarray | None = None,
    M2: np.ndarray | None = None,
) -> float:
    """Marginalized log p(λ | β, σ², σ_α², y) for SEM-RE model.

    Integrates out α analytically from the conditional density,
    breaking the α-λ correlation that causes slow mixing in the
    standard Gibbs sampler.

    The marginalized density is derived from the joint:

        p(y | β, σ², λ, σ_α²) = ∫ p(y | β, α, σ², λ) p(α | σ_α²) dα

    which is a Gaussian integral over α.  Using the Woodbury identity
    on the resulting marginal covariance:

        Ω = σ²(A^T A)^{-1} + σ_α² D D^T

    where A = I - λW and D is the unit indicator matrix, we obtain:

        log p(λ | β, σ², σ_α², y) =
            T·log|I - λW|
            + (1/2)·log|Σ_α|
            - (1/(2σ²))·Q

    where:
        Σ_α^{-1} = (1/σ²) B^T B + (1/σ_α²) I_N
        B = A D  (NT × N matrix)
        Q = r^T A^T A r - (1/σ²)·(D^T A^T A r)^T Σ_α (D^T A^T A r)
        r = y - Xβ

    The N×N matrix Σ_α^{-1} is inverted via Cholesky, making this
    O(N³ + nnz(W)·N) per evaluation, which is efficient for moderate N.

    Parameters
    ----------
    lam : float
        Spatial error parameter.
    beta : ndarray of shape (k,)
        Current regression coefficients.
    sigma2 : float
        Current residual variance.
    sigma_alpha2 : float
        Current random effects variance.
    y, X, W_sparse, logdet_fn, n, k
        As in the cross-sectional collapsed density.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.

    Returns
    -------
    log_density : float
    """
    # Residual
    r = y - X @ beta

    # Spatial filter: A = I - λW
    Ar = r - lam * (W_sparse @ r)
    AtAr = Ar - lam * (W_sparse.T @ Ar)

    # D^T A^T A r: sum A^T A r values by unit
    DtAtAr = np.bincount(unit_idx, weights=AtAr, minlength=N)

    # B^T B = D^T A^T A D via the closed form in λ (see _sem_re_BtB):
    #   B^T B(λ) = T·I_N − λ(M1 + M1ᵀ) + λ² M2
    # from the precomputed λ-independent M1 = D^T W D, M2 = D^T W^T W D.
    if M1 is None or M2 is None:
        M1, M2 = _sem_re_unit_aggregated_terms(W_sparse, unit_idx, N)
    BtB = _sem_re_BtB(lam, M1, M2, T, N)  # N × N

    # Precision and covariance of α
    prec_alpha = (1.0 / sigma2) * BtB + (1.0 / sigma_alpha2) * np.eye(N)

    # Cholesky factorization of precision
    try:
        c_alpha, lower_alpha = cho_factor(prec_alpha)
    except np.linalg.LinAlgError:
        # Near-singular precision — λ likely near boundary
        return -np.inf

    # log|Σ_α| = -log|Σ_α^{-1}| = -log|prec_alpha|
    # Using Cholesky: log|prec_alpha| = 2·Σ log(diag(L))
    log_det_prec = 2.0 * np.sum(np.log(np.diag(c_alpha)))
    log_det_sigma_alpha = -log_det_prec  # log|Σ_α| = -log|prec_alpha|

    # Solve Σ_α (D^T A^T A r) via Cholesky
    z = cho_solve((c_alpha, lower_alpha), DtAtAr)

    # Marginalized log-likelihood (up to constants):
    #   T·log|I-λW| + ½·log|Σ_α| + (1/(2σ⁴))·(D^T A^T A r)^T Σ_α (D^T A^T A r)
    #                  - (1/(2σ²))·r^T A^T A r
    rAtAr = np.dot(r, AtAr)
    quad_alpha = np.dot(DtAtAr, z)  # (D^T A^T A r)^T Σ_α (D^T A^T A r)

    logdet = logdet_fn(lam)

    return (
        logdet
        + 0.5 * log_det_sigma_alpha
        + 0.5 * quad_alpha / (sigma2 * sigma2)
        - 0.5 * rAtAr / sigma2
    )


# ---------------------------------------------------------------------------
# Slice sampling for ρ/λ
# ---------------------------------------------------------------------------


def _sample_rho_re_sar(
    state: REGibbsState,
    cache: REGibbsCache,
    priors: REGibbsPriors,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    n: int,
    k: int,
    N: int,
    T: int,
    unit_idx: np.ndarray,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
    log_density_rho: float | None,
) -> tuple[float, float]:
    """Sample ρ via adaptive slice sampling (SAR-RE collapsed density).

    Returns
    -------
    rho : float
        New draw of ρ.
    log_density_rho : float
        Log-density at the new ρ (cached for next iteration).
    """
    # Use cached log-density if available, otherwise compute it
    if log_density_rho is None:
        log_density_rho = _sar_re_collapsed_log_density(
            state.rho, y, Wy, X, cache.XtX_cho, cache.logdet_fn, n, k, N, T, unit_idx
        )

    new_rho, new_log_density, _, _ = slice_sample_1d_adaptive(
        lambda rho: _sar_re_collapsed_log_density(
            rho, y, Wy, X, cache.XtX_cho, cache.logdet_fn, n, k, N, T, unit_idx
        ),
        state.rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
        log_density_x0=log_density_rho,
    )
    return new_rho, new_log_density


def _sample_lam_re_sem(
    state: REGibbsState,
    cache: REGibbsCache,
    priors: REGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    n: int,
    k: int,
    N: int,
    T: int,
    unit_idx: np.ndarray,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
    log_density_lam: float | None,
) -> tuple[float, float]:
    """Sample λ via adaptive slice sampling (SEM-RE marginalized density).

    Uses a marginalized density that integrates out α analytically,
    breaking the α-λ correlation that causes slow mixing in the
    standard Gibbs sampler.

    Returns
    -------
    lam : float
        New draw of λ.
    log_density_lam : float
        Log-density at the new λ (cached for next iteration).
    """
    # NOTE: Unlike the SAR-RE collapsed density (which only depends on ρ),
    # the SEM-RE marginalized density depends on β, σ², and σ_α², which
    # change every Gibbs iteration.  Therefore we cannot cache the
    # log-density across iterations — it must be recomputed each time.
    log_density_lam = _sem_re_marginalized_log_density(
        state.rho,
        state.beta,
        state.sigma2,
        state.sigma_alpha2,
        y,
        X,
        W_sparse,
        cache.logdet_fn,
        n,
        k,
        N,
        T,
        unit_idx,
        cache.sem_BtB_M1,
        cache.sem_BtB_M2,
    )

    new_lam, new_log_density, _, _ = slice_sample_1d_adaptive(
        lambda lam: _sem_re_marginalized_log_density(
            lam,
            state.beta,
            state.sigma2,
            state.sigma_alpha2,
            y,
            X,
            W_sparse,
            cache.logdet_fn,
            n,
            k,
            N,
            T,
            unit_idx,
            cache.sem_BtB_M1,
            cache.sem_BtB_M2,
        ),
        state.rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
        log_density_x0=log_density_lam,
    )
    return new_lam, new_log_density


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_re_gibbs(
    y: np.ndarray,
    X: np.ndarray,
    XtX_cho: tuple,
    N: int,
    T: int,
    unit_idx: np.ndarray,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> REGibbsState:
    """Warm-start the RE Gibbs sampler from pooled OLS + unit means.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    XtX_cho : tuple of (ndarray, bool)
        Cholesky factor of X^T X from ``scipy.linalg.cho_factor``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.
    priors : REGibbsPriors
        Prior hyperparameters.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    REGibbsState
        Initial state with OLS-based starting values.
    """
    # Pooled OLS for β
    beta_ols = cho_solve(XtX_cho, X.T @ y)
    resid = y - X @ beta_ols

    # Unit means as initial α
    group_counts = np.bincount(unit_idx, minlength=N)
    alpha_init = np.bincount(unit_idx, weights=resid, minlength=N) / group_counts

    # Residual variance after removing β and α
    resid_full = resid - alpha_init[unit_idx]
    sigma2_ols = np.dot(resid_full, resid_full) / len(y)

    # Random effects variance
    sigma_alpha2_init = max(np.var(alpha_init), 1e-6)

    # Start ρ/λ at 0 (no spatial effect)
    rho_init = 0.0

    return REGibbsState(
        beta=beta_ols.copy(),
        sigma2=max(sigma2_ols, 1e-6),
        alpha=alpha_init.copy(),
        sigma_alpha2=sigma_alpha2_init,
        rho=rho_init,
    )


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_re_chain(
    y: np.ndarray,
    X: np.ndarray,
    cache: REGibbsCache,
    priors: REGibbsPriors,
    init: REGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    progressbar: bool = True,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain of the RE panel Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    cache : REGibbsCache
        Precomputed data.
    priors : REGibbsPriors
        Prior hyperparameters.
    init : REGibbsState
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

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho`` (or ``lam``), ``beta``,
        ``sigma``, ``alpha``, ``sigma_alpha``, and ``log_lik``.
        Each array has shape ``(n_keep, ...)`` where n_keep = draws // thin.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    N = cache.N
    T = cache.T
    unit_idx = cache.unit_idx
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws
    model_type = cache.model_type

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    alpha_samples = np.empty((n_keep, N), dtype=np.float64)
    sigma_alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None

    # Copy initial state
    state = REGibbsState(
        beta=init.beta.copy(),
        sigma2=init.sigma2,
        alpha=init.alpha.copy(),
        sigma_alpha2=init.sigma_alpha2,
        rho=init.rho,
    )

    # Adaptive slice width for ρ/λ
    rho_range = cache.rho_upper - cache.rho_lower
    slice_state = SliceWidthState(w=rho_range * 0.1)

    # Cached log-density for ρ/λ
    log_density_rho = None

    # Precompute Wy for SAR
    Wy = cache.Wy

    for i in range(total_iters):
        # Expand α to observation level
        alpha_expanded = state.alpha[unit_idx]

        # --- Block 1: β | α, σ², ρ/λ, y ---
        if model_type in ("sar", "sdm"):
            state.beta = _sample_beta_re_sar(
                state.rho,
                state.sigma2,
                alpha_expanded,
                y,
                Wy,
                X,
                cache.XtX,
                priors,
                rng,
            )
        else:  # sem, sdem
            state.beta = _sample_beta_re_sem(
                state.rho,
                state.sigma2,
                alpha_expanded,
                y,
                X,
                cache.W_sparse,
                priors,
                rng,
            )

        # --- Block 2: σ² | β, α, ρ/λ, y ---
        alpha_expanded = state.alpha[unit_idx]
        state.sigma2 = _sample_sigma2_re(
            state.rho,
            state.beta,
            alpha_expanded,
            y,
            Wy,
            cache.W_sparse,
            X,
            priors,
            model_type,
            rng,
        )

        # --- Block 3: α | β, σ², ρ/λ, y ---
        state.alpha = _sample_alpha_re(
            state.rho,
            state.beta,
            state.sigma2,
            y,
            Wy,
            cache.W_sparse,
            X,
            N,
            T,
            unit_idx,
            state.sigma_alpha2,
            priors,
            model_type,
            rng,
            cache.sem_BtB_M1,
            cache.sem_BtB_M2,
        )

        # --- Block 4: σ_α² | α ---
        state.sigma_alpha2 = _sample_sigma_alpha2(
            state.alpha,
            priors,
            rng,
        )

        # --- Block 5: ρ/λ | y (collapsed, slice sampling) ---
        if model_type in ("sar", "sdm"):
            state.rho, log_density_rho = _sample_rho_re_sar(
                state,
                cache,
                priors,
                y,
                Wy,
                X,
                n,
                k,
                N,
                T,
                unit_idx,
                rng,
                slice_state,
                log_density_rho,
            )
        else:  # sem, sdem
            state.rho, log_density_rho = _sample_lam_re_sem(
                state,
                cache,
                priors,
                y,
                X,
                cache.W_sparse,
                n,
                k,
                N,
                T,
                unit_idx,
                rng,
                slice_state,
                log_density_rho,
            )

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            j = (i - tune) // thin
            rho_samples[j] = state.rho
            beta_samples[j] = state.beta
            sigma_samples[j] = np.sqrt(state.sigma2)
            alpha_samples[j] = state.alpha
            sigma_alpha_samples[j] = np.sqrt(state.sigma_alpha2)

            # Log-likelihood (Gaussian + Jacobian / N·T correction so
            # that the per-obs contributions sum to the joint log-density
            # used by NUTS).  Matches the convention in
            # ``run_gaussian_chain`` and the SAR NUTS post-sample path.
            if store_log_lik:
                alpha_exp = state.alpha[unit_idx]
                if model_type in ("sar", "sdm"):
                    resid = y - state.rho * Wy - X @ state.beta - alpha_exp
                else:
                    resid_raw = y - X @ state.beta - alpha_exp
                    resid = resid_raw - state.rho * (cache.W_sparse @ resid_raw)
                sigma = np.sqrt(state.sigma2)
                ll = (
                    -0.5 * (resid / sigma) ** 2
                    - np.log(sigma)
                    - 0.5 * np.log(2.0 * np.pi)
                )
                # T·log|I - ρ W_N| spread across all n = N·T observations.
                # ``cache.logdet_fn`` was built with the panel's T, so this
                # returns the total log-Jacobian.
                jacobian = cache.logdet_fn(state.rho)
                ll = ll + jacobian / n
                ll = np.where(np.isfinite(ll), ll, -1e10)
                log_lik_samples[j] = ll

        # Progress bar
        if progressbar and progress_manager is None:
            if i % 100 == 0:
                pass  # Progress handled by progress_manager if available

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    spatial_param = "rho" if model_type in ("sar", "sdm") else "lam"
    return {
        spatial_param: rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "alpha": alpha_samples,
        "sigma_alpha": sigma_alpha_samples,
        "log_lik": log_lik_samples,
    }
