"""Gaussian spatial Gibbs sampler for panel models with random effects.

Implements a partially collapsed Gibbs sampler (van Dyk & Park, 2008) for
RE panel models.  Each sweep draws

1. β | α, σ², ρ/λ, y        — conjugate normal
2. σ² | β, α, ρ/λ, y        — conjugate inverse-gamma
3. σ_α² | α                 — conjugate inverse-gamma
4. ρ/λ | β, σ², σ_α², y     — 1-D slice sampling with α integrated out
5. α | β, σ², σ_α², ρ/λ, y  — conjugate normal
6. σ_α | α̃, β, σ², ρ/λ, y  — the non-centred half (α̃ = α/σ_α held, α rescaled)

Step 4 marginalizes α, so step 5 must redraw α straight after it; the
stored state is then a draw from the joint posterior.  Integrating α out of
the ρ/λ update removes the α–ρ/λ correlation that would otherwise slow it.

Steps 3 and 6 interweave the centred and non-centred updates of σ_α (Yu &
Meng 2011).  The centred step alone mixes badly when each unit's periods say
little about its effect — small σ_α against σ/√T, the usual large-N, small-T
spatial panel (ESS ~1% of draws at N = 900, T = 3, σ_α = 0.3σ).  Given α̃ the
likelihood is Gaussian in σ_α, so step 6 is exact.

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

With D the ``n × N`` unit-indicator matrix, c_i the number of periods for
unit i, and A = I - ρW (or I - λW):

SAR-RE ρ step: r = y - Xβ - ρWy ~ N(Dα, σ²I), and DᵀD = diag(c) is
diagonal, so integrating α out by Woodbury gives

    log p(ρ | β, σ², σ_α², y) = log|A| - rᵀr/(2σ²)
                                + Σ_i τ_i (Dᵀr)_i² / (2σ⁴),
    τ_i = 1 / (c_i/σ² + 1/σ_α²),

a quadratic in ρ whose coefficients cost O(n) once per sweep.

SEM-RE λ step: with r = y - Xβ, B = AD and the sparse N × N precision
P(λ) = BᵀB/σ² + I/σ_α²,

    log p(λ | β, σ², σ_α², y) = log|A| - ½ log|P(λ)|
                                - rᵀAᵀAr/(2σ²) + qᵀP(λ)⁻¹q/(2σ⁴),
    q = DᵀAᵀAr.

BᵀB = diag(c) - λ(M1 + M1ᵀ) + λ²M2 with M1 = DᵀWD and M2 = DᵀWᵀWD
precomputed and sparse; each evaluation is one CHOLMOD numeric
refactorization on a fixed pattern.  The α draw uses the same P(λ).

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
- Use the spatial Durbin error specification (SDEM), which adds WX terms
- Use longer panels (T → ∞) which provide more information

References
----------
van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers.
*Journal of the American Statistical Association*, 103(482), 790–796.

Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.).
Springer.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.

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
from .._utils._spatial_normal import CholmodFactor, NotPositiveDefiniteError

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
    sigma_alpha_aux : float
        Auxiliary variable ``a`` of the Huang & Wand (2013) mixture that
        makes the half-t prior on σ_α conditionally conjugate.
    """

    sigma2: float = 1.0
    alpha: np.ndarray = None
    sigma_alpha2: float = 1.0
    sigma_alpha_aux: float = 1.0


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
    # Periods per unit, c_i (DᵀD = diag(c)).
    unit_counts: np.ndarray | None = None
    # SEM-RE only: sparse λ-independent pieces of the α precision.
    sem_alpha: "SemReAlphaStructure | None" = None


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

    Prior σ² ~ Inv-Γ(priors.sigma2_alpha, priors.sigma2_beta), the same
    prior the NUTS build uses.
    """
    n = len(y)

    if model_type in ("sar", "sdm"):
        resid = y - rho * Wy - X @ beta - alpha_expanded
        ss = np.dot(resid, resid)
    else:  # sem, sdem
        resid_raw = y - X @ beta - alpha_expanded
        eps = resid_raw - rho * (W_sparse @ resid_raw)
        ss = np.dot(eps, eps)

    a_post = priors.sigma2_alpha + n / 2
    b_post = priors.sigma2_beta + ss / 2

    sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)
    return sigma2


@dataclass
class SemReAlphaStructure:
    """λ-independent pieces of the SEM-RE α precision on one sparsity pattern.

    ``BᵀB(λ) = DᵀAᵀAD = diag(c) − λ(M1 + M1ᵀ) + λ² M2`` with ``M1 = DᵀWD``
    and ``M2 = DᵀWᵀWD``.  Each term's values are stored aligned to the CSC
    structure of ``pattern`` (their union plus the diagonal), so
    ``precision`` assembles ``P(λ) = BᵀB/σ² + I/σ_α²`` with that exact
    structure and CHOLMOD's symbolic analysis is reused on every refactor.
    """

    pattern: sp.csc_matrix
    diag_vals: np.ndarray  # c_i on the diagonal
    m1s_vals: np.ndarray  # M1 + M1ᵀ
    m2_vals: np.ndarray  # M2
    eye_vals: np.ndarray  # I_N

    def precision(
        self, lam: float, sigma2: float, sigma_alpha2: float
    ) -> sp.csc_matrix:
        vals = (
            self.diag_vals - lam * self.m1s_vals + (lam * lam) * self.m2_vals
        ) / sigma2 + self.eye_vals / sigma_alpha2
        return sp.csc_matrix(
            (vals, self.pattern.indices, self.pattern.indptr),
            shape=self.pattern.shape,
        )

    def new_factor(self) -> CholmodFactor:
        """A CHOLMOD factor analyzed on ``pattern`` (one per chain)."""
        return CholmodFactor(self.pattern)


def _sem_re_alpha_structure(
    W_sparse: sp.csr_matrix, unit_idx: np.ndarray, N: int
) -> SemReAlphaStructure:
    """Precompute the sparse λ-independent terms of the SEM-RE α precision."""
    n = len(unit_idx)
    D = sp.csr_matrix((np.ones(n), (np.arange(n), unit_idx)), shape=(n, N))
    WD = sp.csr_matrix(W_sparse) @ D
    M1 = sp.csr_matrix(D.T @ WD)
    M1s = (M1 + M1.T).tocsr()
    M2 = sp.csr_matrix(WD.T @ WD)
    counts = np.bincount(unit_idx, minlength=N).astype(np.float64)

    # Union pattern with a diagonally dominant, hence SPD, value set for the
    # symbolic analysis.  abs() keeps entries from cancelling to zero.
    S = (abs(M1s) + abs(M2)).tocsr()
    pattern = (S + sp.diags(np.asarray(S.sum(axis=1)).ravel() + 1.0)).tocsc()
    pattern.sort_indices()
    rows = pattern.indices
    cols = np.repeat(np.arange(N), np.diff(pattern.indptr))

    def _on_pattern(M: sp.spmatrix) -> np.ndarray:
        return np.asarray(sp.csr_matrix(M)[rows, cols], dtype=np.float64).ravel()

    return SemReAlphaStructure(
        pattern=pattern,
        diag_vals=_on_pattern(sp.diags(counts)),
        m1s_vals=_on_pattern(M1s),
        m2_vals=_on_pattern(M2),
        eye_vals=(rows == cols).astype(np.float64),
    )


def _sample_alpha_re(
    rho: float,
    beta: np.ndarray,
    sigma2: float,
    y: np.ndarray,
    Wy: np.ndarray | None,
    W_sparse: sp.csr_matrix | None,
    X: np.ndarray,
    N: int,
    unit_idx: np.ndarray,
    unit_counts: np.ndarray,
    sigma_alpha2: float,
    model_type: str,
    rng: np.random.Generator,
    sem_alpha: SemReAlphaStructure | None = None,
    factor: CholmodFactor | None = None,
) -> np.ndarray:
    """Sample α (unit random effects) from its conditional posterior.

    **SAR-RE**: diagonal, α_i | rest ~ N(τ_i (Dᵀr)_i / σ², τ_i) with
    r = y − ρWy − Xβ and τ_i = 1 / (c_i/σ² + 1/σ_α²).

    **SEM-RE**: the filter A = I − λW couples units, so
    α | rest ~ N(P⁻¹q/σ², P⁻¹) with the sparse precision
    P = DᵀAᵀAD/σ² + I/σ_α² and q = DᵀAᵀA r, r = y − Xβ; drawn with a
    CHOLMOD refactorization of ``factor`` on ``sem_alpha.pattern``.
    """
    if model_type in ("sar", "sdm"):
        r = y - rho * Wy - X @ beta
        r_sum = np.bincount(unit_idx, weights=r, minlength=N)
        tau2 = 1.0 / (unit_counts / sigma2 + 1.0 / sigma_alpha2)
        return rng.normal(loc=tau2 * r_sum / sigma2, scale=np.sqrt(tau2))

    r = y - X @ beta
    Ar = r - rho * (W_sparse @ r)
    AtAr = Ar - rho * (W_sparse.T @ Ar)
    q = np.bincount(unit_idx, weights=AtAr, minlength=N)
    factor.factorize(sem_alpha.precision(rho, sigma2, sigma_alpha2))
    return factor.sample(q / sigma2, rng=rng)


def _sample_sigma_alpha2(
    alpha: np.ndarray,
    sigma_alpha2: float,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Draw σ_α² under the half-t prior, returning ``(σ_α², a)``.

    Prior σ_α ~ half-t_ν(0, A), written as the inverse-gamma scale mixture of
    Huang & Wand (2013)::

        σ_α² | a ~ Inv-Γ(ν/2, ν/a),    a ~ Inv-Γ(1/2, 1/A²)

    which keeps both conditionals conjugate::

        a | σ_α²     ~ Inv-Γ((ν+1)/2, ν/σ_α² + 1/A²)
        σ_α² | α, a  ~ Inv-Γ((N+ν)/2, Σα_i²/2 + ν/a)

    ``Inv-Γ(s, r)`` has density ∝ x^(-s-1) e^(-r/x), drawn as 1/Gamma(s, 1/r).
    """
    nu = priors.sigma_alpha_nu
    A = priors.sigma_alpha_scale
    a = 1.0 / rng.gamma((nu + 1.0) / 2.0, 1.0 / (nu / sigma_alpha2 + 1.0 / A**2))
    a_post = (len(alpha) + nu) / 2.0
    b_post = np.dot(alpha, alpha) / 2.0 + nu / a
    return 1.0 / rng.gamma(a_post, 1.0 / b_post), a


def _noncentered_sigma_alpha(
    state: REGibbsState,
    cache: REGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors: REGibbsPriors,
    rng: np.random.Generator,
) -> None:
    """σ_α | α̃ with α̃ = α/σ_α held, then α = σ_α α̃ (in place).

    The residual ``r`` (``y − ρWy − Xβ`` for SAR-RE, ``A(y − Xβ)`` for
    SEM-RE with ``a`` filtered alike) is ``N(σ_α a, σ²I)`` with
    ``a = α̃[unit]``, a Gaussian likelihood in σ_α; the prior is the marginal
    half-t, and the Huang–Wand mixture variable is redrawn given σ_α² before
    its next use.
    """
    from .._utils._group_effects import noncentered_effect_sd

    sd = np.sqrt(state.sigma_alpha2)
    ct = state.alpha / sd
    a = ct[cache.unit_idx]
    if cache.model_type in ("sar", "sdm"):
        r = y - state.rho * cache.Wy - X @ state.beta
    else:
        W = cache.W_sparse
        e = y - X @ state.beta
        r = e - state.rho * (W @ e)
        a = a - state.rho * (W @ a)
    s2 = state.sigma2
    new = noncentered_effect_sd(
        sd,
        float(a @ a) / s2,
        float(a @ r) / s2,
        priors.sigma_alpha_scale,
        rng,
        nu=priors.sigma_alpha_nu,
    )
    state.sigma_alpha2 = new * new
    state.alpha = new * ct


# ---------------------------------------------------------------------------
# ρ/λ | β, σ², σ_α², y with α integrated out
# ---------------------------------------------------------------------------


def _sar_re_rho_quadratic(
    beta: np.ndarray,
    sigma2: float,
    sigma_alpha2: float,
    y: np.ndarray,
    Wy: np.ndarray,
    X: np.ndarray,
    N: int,
    unit_idx: np.ndarray,
    unit_counts: np.ndarray,
) -> tuple[float, float, float]:
    """Coefficients ``(q0, q1, q2)`` of the SAR-RE ρ density's quadratic part.

    With ``e = y − Xβ`` and ``r = e − ρWy``, the α-integrated density is
    ``log|I − ρW| + q0 + q1·ρ + q2·ρ²`` (see the module docstring), where
    the quadratic combines ``−rᵀr/(2σ²)`` and ``Σ τ_i (Dᵀr)_i²/(2σ⁴)``.
    """
    e = y - X @ beta
    tau2 = 1.0 / (unit_counts / sigma2 + 1.0 / sigma_alpha2)
    De = np.bincount(unit_idx, weights=e, minlength=N)
    Dw = np.bincount(unit_idx, weights=Wy, minlength=N)
    s2, s4 = sigma2, sigma2 * sigma2
    q0 = -0.5 * (e @ e) / s2 + 0.5 * (tau2 @ (De * De)) / s4
    q1 = (e @ Wy) / s2 - (tau2 @ (De * Dw)) / s4
    q2 = -0.5 * (Wy @ Wy) / s2 + 0.5 * (tau2 @ (Dw * Dw)) / s4
    return q0, q1, q2


def _sem_re_lam_log_density(
    lam: float,
    beta: np.ndarray,
    sigma2: float,
    sigma_alpha2: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_fn: Callable[[float], float],
    N: int,
    unit_idx: np.ndarray,
    sem_alpha: SemReAlphaStructure,
    factor: CholmodFactor,
) -> float:
    """log p(λ | β, σ², σ_α², y) for SEM-RE with α integrated out.

    ``log|A| − ½ log|P(λ)| − rᵀAᵀAr/(2σ²) + qᵀP(λ)⁻¹q/(2σ⁴)`` with
    ``r = y − Xβ``, ``q = DᵀAᵀAr`` and the sparse precision
    ``P(λ) = DᵀAᵀAD/σ² + I/σ_α²`` (one CHOLMOD refactorization).
    """
    r = y - X @ beta
    Ar = r - lam * (W_sparse @ r)
    AtAr = Ar - lam * (W_sparse.T @ Ar)
    q = np.bincount(unit_idx, weights=AtAr, minlength=N)
    s2 = sigma2
    try:
        factor.factorize(sem_alpha.precision(lam, sigma2, sigma_alpha2))
        quad = 0.5 * (q @ factor.solve(q)) / (s2 * s2) - 0.5 * factor.logdet()
    except NotPositiveDefiniteError:
        return -np.inf  # λ at the edge of its support
    return logdet_fn(lam) - 0.5 * (Ar @ Ar) / s2 + quad


def _sample_rho_re_sar(
    state: REGibbsState,
    cache: REGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
) -> float:
    """Slice sample ρ from p(ρ | β, σ², σ_α², y) for SAR-RE (α integrated out)."""
    q0, q1, q2 = _sar_re_rho_quadratic(
        state.beta,
        state.sigma2,
        state.sigma_alpha2,
        y,
        cache.Wy,
        X,
        cache.N,
        cache.unit_idx,
        cache.unit_counts,
    )

    def log_density(rho):
        return cache.logdet_fn(rho) + q0 + rho * (q1 + rho * q2)

    new_rho, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
    )
    return new_rho


def _sample_lam_re_sem(
    state: REGibbsState,
    cache: REGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    factor: CholmodFactor,
    rng: np.random.Generator,
    slice_state: SliceWidthState,
) -> float:
    """Slice sample λ from p(λ | β, σ², σ_α², y) for SEM-RE (α integrated out)."""

    def log_density(lam):
        return _sem_re_lam_log_density(
            lam,
            state.beta,
            state.sigma2,
            state.sigma_alpha2,
            y,
            X,
            cache.W_sparse,
            cache.logdet_fn,
            cache.N,
            cache.unit_idx,
            cache.sem_alpha,
            factor,
        )

    new_lam, _, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho,
        lower=cache.rho_lower,
        upper=cache.rho_upper,
        rng=rng,
        width_state=slice_state,
    )
    return new_lam


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
        # a | σ_α² is drawn before it is used, so any positive start works.
        sigma_alpha_aux=priors.sigma_alpha_scale**2,
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
        sigma_alpha_aux=init.sigma_alpha_aux,
    )

    # Adaptive slice width for ρ/λ
    rho_range = cache.rho_upper - cache.rho_lower
    slice_state = SliceWidthState(w=rho_range * 0.1)

    Wy = cache.Wy
    counts = cache.unit_counts
    # One CHOLMOD factor per chain (chains may run in threads).
    factor = cache.sem_alpha.new_factor() if cache.sem_alpha is not None else None

    for i in range(total_iters):
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

        # --- Block 3: σ_α² | α (and the half-t mixture variable) ---
        state.sigma_alpha2, state.sigma_alpha_aux = _sample_sigma_alpha2(
            state.alpha, state.sigma_alpha2, priors, rng
        )

        # --- Block 4: ρ/λ | β, σ², σ_α², y (α integrated out) ---
        if model_type in ("sar", "sdm"):
            state.rho = _sample_rho_re_sar(state, cache, y, X, rng, slice_state)
        else:  # sem, sdem
            state.rho = _sample_lam_re_sem(state, cache, y, X, factor, rng, slice_state)

        # --- Block 5: α | β, σ², σ_α², ρ/λ, y — straight after block 4 ---
        state.alpha = _sample_alpha_re(
            state.rho,
            state.beta,
            state.sigma2,
            y,
            Wy,
            cache.W_sparse,
            X,
            N,
            unit_idx,
            counts,
            state.sigma_alpha2,
            model_type,
            rng,
            cache.sem_alpha,
            factor,
        )

        # --- Block 6: σ_α | α̃, rest — non-centred, interwoven with block 3 ---
        _noncentered_sigma_alpha(state, cache, y, X, priors, rng)

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
