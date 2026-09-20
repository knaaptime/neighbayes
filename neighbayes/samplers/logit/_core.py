"""Pólya–Gamma Gibbs sampler for structural-form SAR-logit.

Orchestrates the 4-block Gibbs sweep:
  1. ω | η          (PG augmentation, h = 1)
  2. η | ω, ρ, β   (spatial-normal draw, σ² = 1)
  3. β | η, ρ      (conjugate normal, σ² = 1)
  4. ρ | β, ω, y   (collapsed 1-D slice, η integrated out)

The structural form parameterizes the latent log-odds as
``η = ρ W η + X β + ν`` with ``ν ~ N(0, I)``, and augments the
logistic likelihood with Pólya–Gamma auxiliary variables to obtain
fully conjugate Gibbs updates for η and β.

Because the logit link absorbs the error scale, σ² is fixed at 1
and does not appear as a free parameter.  The PG shape parameter is
always h = 1 (one trial per observation), so the exact sum-of-
exponentials method is used (no truncation bias).

The ρ update uses a **collapsed** (marginal) posterior that integrates
out η, avoiding the slow mixing that arises from conditioning on the
current η draw.  The collapsed log-density is:

    log p(ρ | ·) = log|I - ρW| - ½ log|P_η| + ½ rhs^T P_η⁻¹ rhs

where P_η = A_ρ^T A_ρ + diag(ω) and rhs = A_ρ^T Xβ + κ
with κ = y - ½.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from ...models.priors import LogitGibbsPriors, SEMLogitGibbsPriors
from .._utils._base import GibbsBaseState
from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import (
    CholmodFactor,
    KrylovPrecisionBasis,
    build_precision_krylov_basis,
    cg_solve,
    eval_precision_logdet_from_basis,
    eval_precision_solve_from_basis,
    lanczos_logdet,
    sample_spatial_normal,
)

# ---------------------------------------------------------------------------
# Data classes for state, priors, and precomputed cache
# ---------------------------------------------------------------------------


@dataclass
class LogitGibbsState(GibbsBaseState):
    """Mutable state carried through one Gibbs sweep (Python-loop path).

    All arrays are numpy arrays; scalars are Python floats.
    For the JAX-dense path, use :class:`JAXLogitGibbsState` instead.
    """

    eta: np.ndarray  # (n,) latent log-odds
    omega: np.ndarray  # (n,) PG auxiliary variables

    def to_jax(self) -> "JAXLogitGibbsState":
        """Convert to a JAX-compatible :class:`JAXLogitGibbsState`."""
        import jax.numpy as jnp

        return JAXLogitGibbsState(
            eta=jnp.asarray(self.eta, dtype=jnp.float64),
            beta=jnp.asarray(self.beta, dtype=jnp.float64),
            rho=jnp.float64(self.rho),
            omega=jnp.asarray(self.omega, dtype=jnp.float64),
        )


# JAX-compatible state class (equinox.Module when available, stub otherwise).
from ..._jax_dispatch import ensure_x64
from .._utils._jax_base import make_jax_state_class

JAXLogitGibbsState = make_jax_state_class(
    "JAXLogitGibbsState",
    ("eta", "beta", "rho", "omega"),
    numpy_state_cls=LogitGibbsState,
)


class LogitGibbsCache(NamedTuple):
    """Precomputed data that doesn't change across sweeps.

    Identical in structure to the NB Gibbs cache, except that
    matrix pieces are not divided by σ² (since σ² = 1).
    """

    W_sparse: sp.csr_matrix
    XtX: np.ndarray  # (k, k) = X^T X
    logdet_fn: Callable[[float], float]  # log|I - rho*W| callable
    rho_lower: float
    rho_upper: float
    cholmod_factor: CholmodFactor | None = None
    W_sym: sp.csr_matrix | None = None  # W + W^T (not divided by σ²)
    WtW: sp.csr_matrix | None = None  # W^T W (not divided by σ²)
    WtX: np.ndarray | None = None  # (n, k) = W^T X, ρ-constant
    solve_method: str = "cholmod"  # "cholmod" | "cg" | "jax_dense" | "cholmod_jax"
    logdet_P_method: str = (
        "cholmod"  # "cholmod" | "lanczos" | "jax_dense" | "cholmod_jax"
    )
    sample_method: str = "cholmod"  # "cholmod" | "jax_dense" | "cholmod_jax"
    lanczos_n_probes: int = 10
    lanczos_deg: int = 30
    # --- Krylov basis reuse for the ρ slice -------------------------------
    # When krylov_degree > 0, the slice step factors P(ρ_c) once and
    # reuses the basis for all candidates within krylov_dmax.
    # Only beneficial on high-fill-in graphs (queen contiguity, knn);
    # on ring/lattices with minimal fill-in CHOLMOD is already cheap.
    # The threshold guards against a slowdown on those graphs.
    krylov_degree: int = 0
    krylov_dmax: float = 0.4
    krylov_min_n: int = 400  # below this, CHOLMOD factorization is too cheap to beat
    # JAX dense backend fields
    W_sym_dense: object | None = None  # jax.numpy.ndarray (n, n): W + W^T
    WtW_dense: object | None = None  # jax.numpy.ndarray (n, n): W^T W
    logdet_jax: object | None = None  # callable (rho) -> jax.numpy.ndarray
    # Adaptive width for ρ slice sampler
    rho_adaptive_width: bool = True
    rho_slice_width_state: SliceWidthState | None = None


# ---------------------------------------------------------------------------
# Gibbs block samplers
# ---------------------------------------------------------------------------


def _sample_omega(
    eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 1: Draw ω | η — Pólya–Gamma augmentation.

    For binary logistic regression, h = 1 (one trial per observation).
    Since h is integer, the Devroye method is valid and typically fastest.

    Parameters
    ----------
    eta : ndarray of shape (n,)
        Current latent log-odds.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    omega : ndarray of shape (n,)
        PG(1, eta) draws.
    """
    h = np.ones(len(eta))  # shape = 1 for binary logit
    z = eta  # tilting parameters
    return sample_polyagamma(h, z, rng=rng)


def _sample_eta(
    state: LogitGibbsState,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    *,
    rng: np.random.Generator,
    cache: LogitGibbsCache | None = None,
) -> tuple[np.ndarray, CholmodFactor]:
    """Block 2: Draw η | ω, ρ, β — spatial-normal draw (σ² = 1).

    The conditional posterior is

        η | · ~ N(m_η, Σ_η)

    where Σ_η⁻¹ = A_ρ^T A_ρ + diag(ω) and
    m_η = Σ_η (A_ρ^T Xβ + κ) with κ = y - ½.

    Parameters
    ----------
    state : LogitGibbsState
        Current state (uses eta, beta, rho, omega).
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Spatial weights matrix.
    rng : numpy.random.Generator
        Random state.
    cache : LogitGibbsCache, optional
        If provided, uses precomputed matrix pieces.

    Returns
    -------
    eta_new : ndarray of shape (n,)
        New draw of the latent log-odds.
    factor : CholmodFactor
        Factorisation (for potential reuse within the sweep).
    """
    n = X.shape[0]
    rho = state.rho
    omega = state.omega
    beta = state.beta

    # Precision: P = A_ρ^T A_ρ + diag(ω) = I - ρ(W+W^T) + ρ²W^TW + diag(ω)
    if cache is not None and cache.W_sym is not None and cache.WtW is not None:
        P = (
            sp.eye(n, format="csr")
            + sp.diags(omega, format="csr")
            - rho * cache.W_sym
            + rho**2 * cache.WtW
        )
    else:
        A_rho = sp.eye(n, format="csr") - rho * W_sparse
        AtA = A_rho.T @ A_rho  # σ² = 1, no division
        P = AtA + sp.diags(omega, format="csr")

    # Mean term: P @ m = A_ρ^T Xβ + κ  where κ = y - ½
    Xbeta = X @ beta
    kappa = y - 0.5
    rhs = Xbeta - rho * W_sparse.T @ Xbeta + kappa

    # Dispatch based on sample_method
    sample_method = cache.sample_method if cache is not None else "cholmod"

    if sample_method == "jax_dense":
        # JAX dense path: build dense P, use Chebyshev or Cholesky
        import jax

        ensure_x64()
        import jax.numpy as jnp

        from .._utils._spatial_normal import jax_build_P_dense, jax_chebyshev_sample

        omega_jax = jnp.asarray(omega)
        P_dense = jax_build_P_dense(
            rho, 1.0, omega_jax, cache.W_sym_dense, cache.WtW_dense
        )
        rhs_jax = jnp.asarray(rhs)
        _jax_key_eta = jax.random.PRNGKey(rng.integers(2**31))
        draw = jax_chebyshev_sample(P_dense, rhs_jax, key=_jax_key_eta, degree=30)
        return draw.x, draw.factor

    # NumPy path: use sample_spatial_normal
    cholmod_factor = cache.cholmod_factor if cache is not None else None
    draw = sample_spatial_normal(
        P,
        rhs,
        rng=rng,
        cached_factor=cholmod_factor,
    )
    return draw.x, draw.factor


def _sample_beta(
    state: LogitGibbsState,
    X: np.ndarray,
    XtX: np.ndarray,
    priors: LogitGibbsPriors,
    A_rho_eta: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 3: Draw β | η, ρ — conjugate normal (σ² = 1).

    Posterior: β | · ~ N(m_β, Σ_β) where
        Σ_β⁻¹ = Λ₀⁻¹ + X^T X
        m_β = Σ_β (Λ₀⁻¹ μ₀ + X^T A_ρη)

    Parameters
    ----------
    state : LogitGibbsState
        Current state (uses beta).
    X : ndarray of shape (n, k)
        Design matrix.
    XtX : ndarray of shape (k, k)
        Precomputed X^T X.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    A_rho_eta : ndarray of shape (n,)
        (I - ρW) @ η.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta_new : ndarray of shape (k,)
        New draw of regression coefficients.
    """
    beta_mu = priors.beta_mu
    beta_sigma = priors.beta_sigma

    # Prior precision and mean (diagonal — use vector arithmetic, not np.diag)
    if np.isscalar(beta_sigma):
        k_beta = len(XtX)
        prior_prec_diag = np.full(k_beta, 1.0 / beta_sigma**2)
        Lambda0_inv_mu0 = np.full(k_beta, beta_mu) / beta_sigma**2
    else:
        beta_sigma = np.asarray(beta_sigma)
        prior_prec_diag = 1.0 / beta_sigma**2
        Lambda0_inv_mu0 = np.asarray(beta_mu) / beta_sigma**2

    # Posterior precision: Σ_β⁻¹ = Λ₀⁻¹ + X^TX  (σ² = 1, no division)
    Sigma_beta_inv = XtX.copy()
    Sigma_beta_inv[np.diag_indices_from(Sigma_beta_inv)] += prior_prec_diag
    rhs_beta = Lambda0_inv_mu0 + X.T @ A_rho_eta  # σ² = 1

    # One Cholesky of the SPD precision, reused for the mean and the sample.
    L, lower = cho_factor(Sigma_beta_inv, lower=True)
    m_beta = cho_solve((L, lower), rhs_beta)
    z = rng.standard_normal(len(m_beta))
    beta_new = m_beta + solve_triangular(L, z, lower=lower, trans="T")

    return beta_new


def _sample_rho(
    state: LogitGibbsState,
    cache: LogitGibbsCache,
    priors: LogitGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    log_density_current: float | None = None,
    sweep_idx: int = 0,
    tune: int = 1000,
) -> tuple[float, float]:
    """Block 4: Draw ρ — doubly-collapsed 1-D slice sampler (η AND β integrated out).

    The doubly-collapsed log-density is:

        log p(ρ | σ², ω, y) = log|I - ρW| - ½ log|P_η|
                              - ½ log|Σ_β*⁻¹|
                              + ½ κᵀ P_η⁻¹ κ
                              + ½ m_β*ᵀ Σ_β*⁻¹ m_β*

    where P_η = A_ρᵀA_ρ + diag(ω) (σ² = 1), u = A_ρᵀX = X - ρWᵀX,
    Σ_β*⁻¹ = XᵀX + V₀⁻¹ - uᵀP_η⁻¹u, and
    m_β* = Σ_β*(uᵀP_η⁻¹κ + V₀⁻¹b₀)  with κ = y - ½.

    Integrating β out (in addition to η) breaks the β–ρ posterior
    coupling that bottlenecks ρ mixing in the β-conditional variant.

    Parameters
    ----------
    state : LogitGibbsState
        Current state (uses rho, omega, beta).
    cache : LogitGibbsCache
        Precomputed data.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    rng : numpy.random.Generator
        Random state.
    log_density_current : float or None
        Cached log-density at current ρ (from previous sweep).
        **Must not** be reused across sweeps — the density changes
        because ω and β change.
    sweep_idx : int
        Current sweep index (for adaptive width decisions).
    tune : int
        Total warmup sweeps (adaptive width is frozen after warmup).

    Returns
    -------
    rho_new : float
        New draw of ρ.
    log_density_new : float
        Log-density at the new ρ (can be cached for the next sweep
        within the same iteration, but NOT across iterations).
    """
    n = len(y)
    k = X.shape[1]
    W = cache.W_sparse
    logdet_fn = cache.logdet_fn
    rho_lower = priors.rho_lower
    rho_upper = priors.rho_upper
    omega = state.omega
    Xbeta = X @ state.beta
    kappa = y - 0.5  # κ = y - ½ for binary logit

    cholmod_factor = cache.cholmod_factor
    W_sym = cache.W_sym
    WtW = cache.WtW
    solve_method = cache.solve_method
    logdet_P_method = cache.logdet_P_method

    # Precompute the base precision matrix (changes each Gibbs iteration
    # because omega changes, but is constant across ρ candidates within
    # one slice-sampler step).
    # P = I + diag(ω) - ρ*(W+W^T) + ρ²*W^T W  (σ² = 1)
    base = sp.eye(n, format="csr") + sp.diags(omega, format="csr")

    # Precompute W^T @ Xbeta (constant across ρ candidates)
    # — used only by the JAX-dense β-conditional path below.
    WtXbeta = W.T @ Xbeta

    # --- β-marginalization precomputes (scipy-sparse path, σ² = 1) ---
    beta_mu_arr = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    beta_sigma_arr = np.broadcast_to(
        np.asarray(priors.beta_sigma, dtype=np.float64), (k,)
    )
    V0_inv_diag = 1.0 / beta_sigma_arr**2
    V0_inv_b0 = beta_mu_arr * V0_inv_diag
    XtX_mat = cache.XtX  # (k, k), σ² = 1 so no division
    WtX = cache.WtX if cache.WtX is not None else W.T @ X  # (n, k), ρ-constant

    # Lanczos RNG (only needed on the Lanczos logdet path)
    _lanczos_rng = (
        np.random.default_rng(rng.integers(2**31))
        if logdet_P_method == "lanczos"
        else None
    )

    # JAX dense backend
    use_jax = solve_method == "jax_dense"
    if use_jax:
        import jax

        ensure_x64()
        import jax.numpy as jnp

        from .._utils._spatial_normal import _jax_log_density_core

        omega_jax = jnp.asarray(omega)
        _jax_key = jax.random.PRNGKey(rng.integers(2**31))
        # For σ² = 1: Xbeta_over_s2 = Xbeta, WtXbeta_over_s2 = WtXbeta
        _Xbeta_jax = jnp.asarray(Xbeta)
        _WtXbeta_jax = jnp.asarray(WtXbeta)
        _kappa_jax = jnp.asarray(kappa)
        _jax_logdens_fn = jax.jit(
            lambda rho, key: _jax_log_density_core(
                rho=rho,
                sigma2=1.0,  # σ² = 1 for logit
                omega=omega_jax,
                W_sym_dense=cache.W_sym_dense,
                WtW_dense=cache.WtW_dense,
                logdet_jax=cache.logdet_jax,
                Xbeta_over_s2=_Xbeta_jax,
                WtXbeta_over_s2=_WtXbeta_jax,
                kappa=_kappa_jax,
                key=key,
                n_probes=cache.lanczos_n_probes,
                lanczos_deg=cache.lanczos_deg,
                cg_tol=1e-8,
                cg_maxiter=n,
            ),
        )
        _warmup_key = jax.random.fold_in(_jax_key, 0)
        _ = float(_jax_logdens_fn(jnp.float64(state.rho), _warmup_key))

    # --- Krylov basis for the precision P(ρ) (scipy sparse path only) -------
    # Build once at the slice centre ρ_c = state.rho, then evaluate every
    # candidate within krylov_dmax via a Horner sum.
    #
    # The RHS for the slice is [κ, u(ρ)] where u(ρ) = X − ρ·WtX is itself
    # ρ-dependent.  We seed the basis with the ρ-independent columns
    # [κ, X, WtX] so that P(ρ)⁻¹u(ρ) = P(ρ)⁻¹X − ρ·P(ρ)⁻¹WtX is recovered
    # as a linear combination of two Horner evaluations against the same
    # single factorization.
    _krylov_basis: KrylovPrecisionBasis | None = None
    if (
        not use_jax
        and cache.krylov_degree > 0
        and n >= cache.krylov_min_n
        and W_sym is not None
        and WtW is not None
    ):
        _rho_c = float(state.rho)
        # Seed with ρ-independent columns: [κ, X, WtX]
        rhs_c = np.column_stack([kappa, X, WtX])  # (n, 1 + k + k)
        _krylov_basis = build_precision_krylov_basis(
            rho_c=_rho_c,
            base=base,
            G1=W_sym,
            G2=WtW,
            rhs=rhs_c,
            degree=cache.krylov_degree,
            cholmod_factor=cholmod_factor,
            n_probes=cache.lanczos_n_probes,
            lanczos_deg=cache.lanczos_deg,
            rng=_lanczos_rng,
            dmax=cache.krylov_dmax,
        )

    def log_density(rho: float) -> float:
        """Doubly-collapsed log-density of ρ (η and β integrated out)."""
        if use_jax:
            # NOTE: JAX path still uses β-conditional density;
            # marginalising β here is deferred to a follow-up.
            # Fixed key per slice step: all ρ candidates share the same
            # Lanczos probes, giving a deterministic density surface within
            # the step (fresh _jax_key each Gibbs iteration).
            result = _jax_logdens_fn(jnp.float64(rho), _jax_key)
            return float(result)

        # --- scipy sparse path (β-marginalized, σ² = 1) ---
        log_det_A = logdet_fn(rho)

        # u = A_ρᵀ X = X − ρ Wᵀ X   — (n, k) dense
        u = X - rho * WtX

        # --- Krylov-accelerated solve + logdet ------------------------------
        # When the basis is available and ρ is within the Krylov radius,
        # P⁻¹[κ | u(ρ)] and log|P(ρ)| come from Horner evaluations against
        # the single factored P_c.
        if (
            _krylov_basis is not None
            and abs(rho - _krylov_basis.rho_basis) <= _krylov_basis.safe_dmax
        ):
            drho = rho - _krylov_basis.rho_basis
            sol_all = eval_precision_solve_from_basis(_krylov_basis, drho)
            # sol_all columns: [P(ρ)⁻¹κ, P(ρ)⁻¹X, P(ρ)⁻¹WtX]
            z = sol_all[:, 0]
            PX = sol_all[:, 1 : 1 + k]
            PWtX = sol_all[:, 1 + k :]
            # u(ρ) = X − ρ·WtX  →  P(ρ)⁻¹u(ρ) = P(ρ)⁻¹X − ρ·P(ρ)⁻¹WtX
            M = PX - rho * PWtX
            # log|P(ρ)| via the first-order trace correction (matvec-only)
            log_det_P = eval_precision_logdet_from_basis(
                _krylov_basis,
                drho,
                rng=_lanczos_rng,
            )
        else:
            # --- Direct path: factor / Lanczos at this candidate ------------
            # Precision: P = base - ρ * W_sym + ρ² * WtW  (σ² = 1)
            if W_sym is not None and WtW is not None:
                P = base - rho * W_sym + rho**2 * WtW
            else:
                A_rho = sp.eye(n, format="csr") - rho * W
                AtA = A_rho.T @ A_rho  # σ² = 1
                P = AtA + sp.diags(omega, format="csr")

            rhs_stack = np.column_stack([kappa, u])  # (n, k+1)

            # --- log|P_η| ---
            if logdet_P_method == "lanczos":
                log_det_P = lanczos_logdet(
                    P,
                    n_probes=cache.lanczos_n_probes,
                    lanczos_deg=cache.lanczos_deg,
                    rng=_lanczos_rng,
                )
            else:
                cholmod_factor.factorize(P)
                log_det_P = cholmod_factor.logdet()

            # --- Multi-RHS solve: P [z | M] = [κ | u] ---
            if solve_method == "cg":
                sol = np.column_stack(
                    [cg_solve(P, rhs_stack[:, j]) for j in range(k + 1)]
                )
            else:
                sol = cholmod_factor.solve(rhs_stack)
            z = sol[:, 0]
            M = sol[:, 1:]

        # Σ_β*⁻¹ = XᵀX + V₀⁻¹ − uᵀM   (k × k, symmetric PD)
        Sig_inv = XtX_mat - u.T @ M
        Sig_inv[np.arange(k), np.arange(k)] += V0_inv_diag
        Lb = np.linalg.cholesky(Sig_inv)
        log_det_Sig_inv = 2.0 * float(np.sum(np.log(np.diag(Lb))))

        rhs_b = u.T @ z + V0_inv_b0
        m_b = np.linalg.solve(Lb.T, np.linalg.solve(Lb, rhs_b))

        quad_kappa = float(kappa @ z)
        quad_b = float(rhs_b @ m_b)

        result = (
            log_det_A
            - 0.5 * log_det_P
            - 0.5 * log_det_Sig_inv
            + 0.5 * quad_kappa
            + 0.5 * quad_b
        )
        if not np.isfinite(result):
            return -np.inf
        return result

    # Cache log-density at current x0
    x0 = state.rho
    if log_density_current is not None:
        log_dens_x0 = log_density_current
    else:
        log_dens_x0 = log_density(x0)

    # --- Adaptive width slice sampling ---
    width_state = cache.rho_slice_width_state
    if width_state is None:
        width_state = SliceWidthState(w=0.2)

    rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=x0,
        lower=rho_lower,
        upper=rho_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_dens_x0,
    )

    # Update width (only during burn-in)
    if cache.rho_adaptive_width and sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)

    return rho_new, log_density_new


# ---------------------------------------------------------------------------
# Pointwise log-likelihood (for ArviZ InferenceData)
# ---------------------------------------------------------------------------


def _logit_loglik_pointwise(
    y: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Pointwise log-likelihood for binary logistic model.

    log p(y_i | η_i) = y_i η_i - log(1 + exp(η_i))

    Uses the numerically stable log-sum-exp trick.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    eta : ndarray of shape (n,)
        Latent log-odds.

    Returns
    -------
    log_lik : ndarray of shape (n,)
        Pointwise log-likelihood values.

    Notes
    -----
    This is the single NumPy source for the Bernoulli-logit pointwise
    log-likelihood; the reduced-form sampler imports it rather than
    re-deriving it.  ``np.logaddexp(0, η)`` is the stable softplus — it is
    exactly the ``max(η, 0) + log1p(exp(-|η|))`` form this used to spell
    out by hand (agreeing to ~1e-15 and finite at |η| = 800), with less to
    get wrong.
    """
    return y * eta - np.logaddexp(0.0, eta)


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: LogitGibbsPriors,
    cache: LogitGibbsCache,
    init: LogitGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
) -> dict:
    """Run one chain of the SAR-logit Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Spatial weights matrix.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    cache : LogitGibbsCache
        Precomputed data.
    init : LogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.

    Returns
    -------
    dict
        Posterior samples with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws

    # Pre-allocate storage for post-warmup draws
    n_keep = draws // thin if thin > 0 else draws
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Copy initial state (we mutate state in-place)
    state = LogitGibbsState(
        eta=init.eta.copy(),
        beta=init.beta.copy(),
        rho=init.rho,
        omega=init.omega.copy(),
    )

    # Precompute X^T X (already in cache)
    XtX = cache.XtX

    for i in range(total_iters):
        # --- Block 1: ω | η ---
        state.omega = _sample_omega(state.eta, rng=rng)

        # --- Block 2: η | ω, ρ, β ---
        state.eta, _ = _sample_eta(state, y, X, W_sparse, rng=rng, cache=cache)

        # Recompute A_rho_eta with new eta (no matrix build)
        A_rho_eta = state.eta - state.rho * (W_sparse @ state.eta)

        # --- Block 3: β | η, ρ ---
        state.beta = _sample_beta(state, X, XtX, priors, A_rho_eta, rng=rng)

        # --- Block 4: ρ | β, ω, y (collapsed, η integrated out) ---
        state.rho, log_density_rho = _sample_rho(
            state,
            cache,
            priors,
            y,
            X,
            rng=rng,
            log_density_current=None,
            sweep_idx=i,
            tune=tune,
        )

        # --- Store post-warmup draws ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                if store_log_lik:
                    log_lik_samples[idx] = _logit_loglik_pointwise(y, state.eta)
                eta_norm_samples[idx] = float(state.eta @ state.eta)
                if return_eta:
                    eta_samples[idx] = state.eta

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    result = {
        "rho": rho_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ===========================================================================
# SEM-logit Gibbs sampler
# ===========================================================================
#
# Structural form:  y_i ~ Bernoulli(logit⁻¹(η_i))
#   η = Xβ + u,  u = λWu + ν,  ν ~ N(0, I)
#
# Reduced form:  η = Xβ + (I - λW)⁻¹ν
#
# Prior for η | β, λ:  η ~ N(Xβ, (A_λ'A_λ)⁻¹)  where A_λ = I - λW
#
# 4-block Gibbs sweep:
#   1. ω | η          (PG augmentation, h = 1 — identical to SAR-logit)
#   2. η | ω, β, λ   (spatial-normal draw, σ² = 1)
#      P = A_λ'A_λ + diag(ω),  rhs = A_λ'A_λ Xβ + κ
#   3. β | η, λ      (conjugate normal with SEM-style transformed data)
#      X* = A_λX,  η* = A_λη,  Σ_β⁻¹ = X*'X* + Λ₀⁻¹
#   4. λ | β, ω, y   (collapsed 1-D slice, η integrated out)
#
# Key difference from SAR-logit:
#   - η rhs:  A_λ'A_λXβ + κ  (not A_λ'Xβ + κ)
#   - β block: uses X* = A_λX, η* = A_λη  (not X, A_ρη)
# ===========================================================================


@dataclass
class SEMLogitGibbsState:
    """Mutable state for the SEM-logit Gibbs sampler (Python-loop path).

    All arrays are numpy arrays; scalars are Python floats.
    """

    eta: np.ndarray  # (n,) latent log-odds
    beta: np.ndarray  # (k,) regression coefficients
    lam: float  # spatial error parameter
    omega: np.ndarray  # (n,) PG auxiliary variables

    def to_jax(self) -> "JAXSEMLogitGibbsState":
        """Convert to a JAX-compatible :class:`JAXSEMLogitGibbsState`."""
        import jax.numpy as jnp

        return JAXSEMLogitGibbsState(
            eta=jnp.asarray(self.eta, dtype=jnp.float64),
            beta=jnp.asarray(self.beta, dtype=jnp.float64),
            lam=jnp.float64(self.lam),
            omega=jnp.asarray(self.omega, dtype=jnp.float64),
        )


JAXSEMLogitGibbsState = make_jax_state_class(
    "JAXSEMLogitGibbsState",
    ("eta", "beta", "lam", "omega"),
    numpy_state_cls=SEMLogitGibbsState,
)


class SEMLogitGibbsCache(NamedTuple):
    """Precomputed data for the SEM-logit Gibbs sampler.

    Identical in structure to LogitGibbsCache but with λ-specific fields.
    """

    W_sparse: sp.csr_matrix
    XtX: np.ndarray  # (k, k) = X^T X  (NOT X*'X* — that depends on λ)
    logdet_fn: Callable[[float], float]  # log|I - lam*W| callable
    lam_lower: float
    lam_upper: float
    cholmod_factor: CholmodFactor | None = None
    W_sym: sp.csr_matrix | None = None  # W + W^T
    WtW: sp.csr_matrix | None = None  # W^T W
    solve_method: str = "cholmod"  # "cholmod" | "cg" | "jax_dense" | "cholmod_jax"
    logdet_P_method: str = (
        "cholmod"  # "cholmod" | "lanczos" | "jax_dense" | "cholmod_jax"
    )
    sample_method: str = "cholmod"  # "cholmod" | "jax_dense" | "cholmod_jax"
    lanczos_n_probes: int = 10
    lanczos_deg: int = 30
    # --- Krylov basis reuse for the λ slice -------------------------------
    # When krylov_degree > 0, the slice step factors P(λ_c) once and
    # reuses the basis for all candidates within krylov_dmax.
    # Only beneficial on high-fill-in graphs (queen contiguity, knn);
    # on ring/lattices with minimal fill-in CHOLMOD is already cheap.
    krylov_degree: int = 0
    krylov_dmax: float = 0.4
    krylov_min_n: int = 400  # below this, CHOLMOD factorization is too cheap to beat
    # JAX dense backend fields
    W_sym_dense: object | None = None
    WtW_dense: object | None = None
    logdet_jax: object | None = None
    # Adaptive width for λ slice sampler
    lam_adaptive_width: bool = True
    lam_slice_width_state: SliceWidthState | None = None


# ---------------------------------------------------------------------------
# SEM-logit block samplers
# ---------------------------------------------------------------------------


def _sample_eta_sem(
    state: SEMLogitGibbsState,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    *,
    rng: np.random.Generator,
    cache: SEMLogitGibbsCache | None = None,
) -> tuple[np.ndarray, CholmodFactor]:
    """Block 2 (SEM): Draw η | ω, β, λ — spatial-normal draw (σ² = 1).

    The conditional posterior is

        η | · ~ N(m_η, Σ_η)

    where Σ_η⁻¹ = A_λ^T A_λ + diag(ω) and
    m_η = Σ_η (A_λ^T A_λ Xβ + κ) with κ = y - ½.

    Note the rhs differs from SAR-logit: SAR uses A_ρ'Xβ + κ,
    SEM uses A_λ'A_λXβ + κ.

    Parameters
    ----------
    state : SEMLogitGibbsState
        Current state (uses eta, beta, lam, omega).
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Spatial weights matrix.
    rng : numpy.random.Generator
        Random state.
    cache : SEMLogitGibbsCache, optional
        If provided, uses precomputed matrix pieces.

    Returns
    -------
    eta_new : ndarray of shape (n,)
        New draw of the latent log-odds.
    factor : CholmodFactor
        Factorisation (for potential reuse within the sweep).
    """
    n = X.shape[0]
    lam = state.lam
    omega = state.omega
    beta = state.beta

    # Precision: P = A_λ^T A_λ + diag(ω) = I - λ(W+W^T) + λ²W^TW + diag(ω)
    if cache is not None and cache.W_sym is not None and cache.WtW is not None:
        P = (
            sp.eye(n, format="csr")
            + sp.diags(omega, format="csr")
            - lam * cache.W_sym
            + lam**2 * cache.WtW
        )
    else:
        A_lam = sp.eye(n, format="csr") - lam * W_sparse
        AtA = A_lam.T @ A_lam
        P = AtA + sp.diags(omega, format="csr")

    # Mean term: P @ m = A_λ'A_λ Xβ + κ  where κ = y - ½
    # A_λ'A_λ Xβ = Xβ - λ(W+W')Xβ + λ²W'WXβ
    Xbeta = X @ beta
    kappa = y - 0.5
    if cache is not None and cache.W_sym is not None and cache.WtW is not None:
        rhs = Xbeta - lam * (cache.W_sym @ Xbeta) + lam**2 * (cache.WtW @ Xbeta) + kappa
    else:
        A_lam = sp.eye(n, format="csr") - lam * W_sparse
        AtA_Xbeta = A_lam.T @ (A_lam @ Xbeta)
        rhs = AtA_Xbeta + kappa

    # Dispatch based on sample_method
    sample_method = cache.sample_method if cache is not None else "cholmod"

    if sample_method == "jax_dense":
        import jax

        ensure_x64()
        import jax.numpy as jnp

        from .._utils._spatial_normal import jax_build_P_dense, jax_chebyshev_sample

        omega_jax = jnp.asarray(omega)
        P_dense = jax_build_P_dense(
            lam, 1.0, omega_jax, cache.W_sym_dense, cache.WtW_dense
        )
        rhs_jax = jnp.asarray(rhs)
        _jax_key_eta = jax.random.PRNGKey(rng.integers(2**31))
        draw = jax_chebyshev_sample(P_dense, rhs_jax, key=_jax_key_eta, degree=30)
        return draw.x, draw.factor

    # NumPy path: use sample_spatial_normal
    cholmod_factor = cache.cholmod_factor if cache is not None else None
    draw = sample_spatial_normal(
        P,
        rhs,
        rng=rng,
        cached_factor=cholmod_factor,
    )
    return draw.x, draw.factor


def _sample_beta_sem(
    state: SEMLogitGibbsState,
    X: np.ndarray,
    priors: SEMLogitGibbsPriors,
    A_lambda_eta: np.ndarray,
    W_sparse: sp.csr_matrix,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block 3 (SEM): Draw β | η, λ — conjugate normal (σ² = 1).

    Uses SEM-style transformed data:
        η* = A_λ η = (I - λW)η
        X* = A_λ X = (I - λW)X

    Posterior: β | · ~ N(m_β, Σ_β) where
        Σ_β⁻¹ = X*'X* + Λ₀⁻¹
        m_β = Σ_β (X*'η* + Λ₀⁻¹μ₀)

    Parameters
    ----------
    state : SEMLogitGibbsState
        Current state (uses beta, lam).
    X : ndarray of shape (n, k)
        Design matrix.
    priors : SEMLogitGibbsPriors
        Prior hyperparameters.
    A_lambda_eta : ndarray of shape (n,)
        (I - λW) @ η.
    W_sparse : csr_matrix
        Spatial weights matrix.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    beta_new : ndarray of shape (k,)
        New draw of regression coefficients.
    """
    lam = state.lam
    beta_mu = priors.beta_mu
    beta_sigma = priors.beta_sigma

    # Transform data: X* = (I - λW)X,  η* = A_λη (passed in)
    X_star = X - lam * (W_sparse @ X)
    eta_star = A_lambda_eta

    # Prior precision and mean (diagonal — use vector arithmetic, not np.diag)
    k = X.shape[1]
    if np.isscalar(beta_sigma):
        prior_prec_diag = np.full(k, 1.0 / beta_sigma**2)
        Lambda0_inv_mu0 = np.full(k, beta_mu) / beta_sigma**2
    else:
        beta_sigma = np.asarray(beta_sigma)
        prior_prec_diag = 1.0 / beta_sigma**2
        Lambda0_inv_mu0 = np.asarray(beta_mu) / beta_sigma**2

    # Posterior precision: Σ_β⁻¹ = Λ₀⁻¹ + X*'X*  (σ² = 1)
    XstXs = X_star.T @ X_star
    Sigma_beta_inv = XstXs.copy()
    Sigma_beta_inv[np.diag_indices_from(Sigma_beta_inv)] += prior_prec_diag
    rhs_beta = Lambda0_inv_mu0 + X_star.T @ eta_star  # σ² = 1

    # One Cholesky of the SPD precision, reused for the mean and the sample.
    L, lower = cho_factor(Sigma_beta_inv, lower=True)
    m_beta = cho_solve((L, lower), rhs_beta)
    z = rng.standard_normal(len(m_beta))
    beta_new = m_beta + solve_triangular(L, z, lower=lower, trans="T")

    return beta_new


def _sample_lam(
    state: SEMLogitGibbsState,
    cache: SEMLogitGibbsCache,
    priors: SEMLogitGibbsPriors,
    y: np.ndarray,
    X: np.ndarray,
    *,
    rng: np.random.Generator,
    log_density_current: float | None = None,
    sweep_idx: int = 0,
    tune: int = 1000,
) -> tuple[float, float]:
    """Block 4 (SEM): Draw λ — collapsed 1-D slice sampler (η integrated out).

    The collapsed log-density is:

        log p(λ | ·) = log|I - λW| - ½ log|P_η| + ½ rhs^T P_η⁻¹ rhs
                      - ½ Xβ^T A_λ^T A_λ Xβ

    where P_η = A_λ^T A_λ + diag(ω) and rhs = A_λ^T A_λ Xβ + κ
    with κ = y - ½.  σ² = 1, so no 1/σ² scaling appears.

    The term -½Xβ'A_λ'A_λXβ arises from the SEM prior normalization.
    In SAR, the analogous term -½Xβ'Xβ is constant in ρ and drops out,
    but in SEM, A_λ'A_λXβ depends on λ through W and W'W.

    Expanding the rhs:
        A_λ'A_λXβ = Xβ - λ(W+W')Xβ + λ²W'WXβ

    Expanding the correction:
        -½Xβ'A_λ'A_λXβ = ½λXβ'(W+W')Xβ - ½λ²Xβ'W'WXβ + const

    Parameters
    ----------
    state : SEMLogitGibbsState
        Current state (uses lam, omega, beta).
    cache : SEMLogitGibbsCache
        Precomputed data.
    priors : SEMLogitGibbsPriors
        Prior hyperparameters.
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    rng : numpy.random.Generator
        Random state.
    log_density_current : float or None
        Cached log-density at current λ.
    sweep_idx : int
        Current sweep index (for adaptive width decisions).
    tune : int
        Total warmup sweeps (adaptive width is frozen after warmup).

    Returns
    -------
    lam_new : float
        New draw of λ.
    log_density_new : float
        Log-density at the new λ.
    """
    n = len(y)
    W = cache.W_sparse
    logdet_fn = cache.logdet_fn
    lam_lower = priors.lam_lower
    lam_upper = priors.lam_upper
    omega = state.omega
    Xbeta = X @ state.beta
    kappa = y - 0.5

    cholmod_factor = cache.cholmod_factor
    W_sym = cache.W_sym
    WtW = cache.WtW
    solve_method = cache.solve_method
    logdet_P_method = cache.logdet_P_method

    # Precompute the base precision matrix
    # P = I + diag(ω) - λ*(W+W^T) + λ²*W^T W  (σ² = 1)
    base = sp.eye(n, format="csr") + sp.diags(omega, format="csr")

    # Precompute W_sym @ Xbeta and WtW @ Xbeta (constant across λ candidates)
    WsymXbeta = W_sym @ Xbeta if W_sym is not None else (W + W.T) @ Xbeta
    WtWXbeta = WtW @ Xbeta if WtW is not None else (W.T @ W) @ Xbeta

    # Lanczos RNG
    _lanczos_rng = (
        np.random.default_rng(rng.integers(2**31))
        if logdet_P_method == "lanczos"
        else None
    )

    # JAX dense backend
    use_jax = solve_method == "jax_dense"
    if use_jax:
        import jax

        ensure_x64()
        import jax.numpy as jnp

        omega_jax = jnp.asarray(omega)
        _jax_key = jax.random.PRNGKey(rng.integers(2**31))
        _Xbeta_jax = jnp.asarray(Xbeta)
        _WsymXbeta_jax = jnp.asarray(WsymXbeta)
        _WtWXbeta_jax = jnp.asarray(WtWXbeta)
        _kappa_jax = jnp.asarray(kappa)
        _jax_logdens_fn = jax.jit(
            lambda lam, key: _sem_jax_log_density_core(
                lam=lam,
                omega=omega_jax,
                W_sym_dense=cache.W_sym_dense,
                WtW_dense=cache.WtW_dense,
                logdet_jax=cache.logdet_jax,
                Xbeta=_Xbeta_jax,
                WsymXbeta=_WsymXbeta_jax,
                WtWXbeta=_WtWXbeta_jax,
                kappa=_kappa_jax,
                key=key,
                n_probes=cache.lanczos_n_probes,
                lanczos_deg=cache.lanczos_deg,
                cg_tol=1e-8,
                cg_maxiter=n,
            ),
        )
        _warmup_key = jax.random.fold_in(_jax_key, 0)
        _ = float(_jax_logdens_fn(jnp.float64(state.lam), _warmup_key))

    # --- Krylov basis for the precision P(λ) (scipy sparse path only) -------
    # The SEM RHS is quadratic in λ: rhs = Xβ − λ·W_sym·Xβ + λ²·WtW·Xβ + κ.
    # We seed the basis with the ρ-independent columns [κ, Xβ, W_sym·Xβ,
    # WtW·Xβ] and reconstruct the solve as a quadratic combination of Horner
    # evaluations against the single factored P_c.
    _krylov_basis: KrylovPrecisionBasis | None = None
    if (
        not use_jax
        and cache.krylov_degree > 0
        and n >= cache.krylov_min_n
        and W_sym is not None
        and WtW is not None
    ):
        _lam_c = float(state.lam)
        rhs_c = np.column_stack([kappa, Xbeta, WsymXbeta, WtWXbeta])
        _krylov_basis = build_precision_krylov_basis(
            rho_c=_lam_c,
            base=base,
            G1=W_sym,
            G2=WtW,
            rhs=rhs_c,
            degree=cache.krylov_degree,
            cholmod_factor=cholmod_factor,
            n_probes=cache.lanczos_n_probes,
            lanczos_deg=cache.lanczos_deg,
            rng=_lanczos_rng,
            dmax=cache.krylov_dmax,
        )

    def log_density(lam: float) -> float:
        """Collapsed log-density of λ (η integrated out)."""
        if use_jax:
            # Fixed key per slice step: all λ candidates share the same
            # Lanczos probes, giving a deterministic density surface within
            # the step (fresh _jax_key each Gibbs iteration).
            result = _jax_logdens_fn(jnp.float64(lam), _jax_key)
            return float(result)

        # --- scipy sparse path ---
        logdet = logdet_fn(lam)

        # RHS: Xbeta - λ*(W+W')Xbeta + λ²*W'WXbeta + κ  (σ² = 1)
        rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

        # --- Krylov-accelerated solve + logdet ------------------------------
        if (
            _krylov_basis is not None
            and abs(lam - _krylov_basis.rho_basis) <= _krylov_basis.safe_dmax
        ):
            dlam = lam - _krylov_basis.rho_basis
            sol_all = eval_precision_solve_from_basis(_krylov_basis, dlam)
            # sol_all columns: [P(λ)⁻¹κ, P(λ)⁻¹Xβ, P(λ)⁻¹W_symXβ, P(λ)⁻¹WtWXβ]
            Pkappa = sol_all[:, 0]
            PXbeta = sol_all[:, 1:2]
            PWsymXbeta = sol_all[:, 2:3]
            PWtWXbeta = sol_all[:, 3:]
            # rhs = Xβ − λ·W_symXβ + λ²·WtWXβ + κ
            m = (PXbeta - lam * PWsymXbeta + lam**2 * PWtWXbeta).ravel() + Pkappa
            log_det_P = eval_precision_logdet_from_basis(
                _krylov_basis,
                dlam,
                rng=_lanczos_rng,
            )
        else:
            # --- Direct path: factor / Lanczos at this candidate ------------
            # Precision: P = base - λ * W_sym + λ² * WtW  (σ² = 1)
            if W_sym is not None and WtW is not None:
                P = base - lam * W_sym + lam**2 * WtW
            else:
                A_lam = sp.eye(n, format="csr") - lam * W
                AtA = A_lam.T @ A_lam
                P = AtA + sp.diags(omega, format="csr")

            # --- log|P_η| ---
            if logdet_P_method == "lanczos":
                log_det_P = lanczos_logdet(
                    P,
                    n_probes=cache.lanczos_n_probes,
                    lanczos_deg=cache.lanczos_deg,
                    rng=_lanczos_rng,
                )
            else:
                cholmod_factor.factorize(P)
                log_det_P = cholmod_factor.logdet()

            # --- Solve P m = rhs ---
            if solve_method == "cg":
                m = cg_solve(P, rhs)
            else:
                m = cholmod_factor.solve(rhs)

        quad = float(rhs @ m)

        # Missing term from SEM prior: -½Xβ'A_λ'A_λXβ
        # = -½Xβ'Xβ + ½λXβ'W_symXβ - ½λ²Xβ'W'WXβ
        # The -½Xβ'Xβ is constant in λ and drops out.
        # In SAR, the analogous term -½Xβ'Xβ is also constant in ρ.
        # In SEM, A_λ'A_λXβ appears in the prior normalization,
        # so -½Xβ'A_λ'A_λXβ depends on λ and must be included.
        xbeta_correction = 0.5 * lam * float(Xbeta @ WsymXbeta) - 0.5 * lam**2 * float(
            Xbeta @ WtWXbeta
        )

        return logdet - 0.5 * log_det_P + 0.5 * quad + xbeta_correction

    # Cache log-density at current x0
    x0 = state.lam
    if log_density_current is not None:
        log_dens_x0 = log_density_current
    else:
        log_dens_x0 = log_density(x0)

    # --- Adaptive width slice sampling ---
    width_state = cache.lam_slice_width_state
    if width_state is None:
        width_state = SliceWidthState(w=0.2)

    lam_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=x0,
        lower=lam_lower,
        upper=lam_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_dens_x0,
    )

    # Update width (only during burn-in)
    if cache.lam_adaptive_width and sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)

    return lam_new, log_density_new


def _sem_jax_log_density_core(
    lam,
    omega,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    Xbeta,
    WsymXbeta,
    WtWXbeta,
    kappa,
    key,
    n_probes,
    lanczos_deg,
    cg_tol,
    cg_maxiter,
):
    """Core SEM-logit collapsed log-density, fully in JAX.

    Computes:
        log p(λ | ·) = log|I - λW| - ½ log|P| + ½ rhs'P⁻¹rhs
                      - ½ Xβ'A_λ'A_λXβ

    where P = I + diag(ω) - λ(W+W') + λ²W'W
    and rhs = Xβ - λ(W+W')Xβ + λ²W'WXβ + κ

    The term -½Xβ'A_λ'A_λXβ arises from the SEM prior normalization
    and depends on λ (unlike SAR where -½Xβ'Xβ is constant in ρ).
    Expanding: -½Xβ'A_λ'A_λXβ = ½λXβ'W_symXβ - ½λ²Xβ'W'WXβ + const
    """
    import jax.numpy as jnp

    from .._utils._spatial_normal import jax_cg_solve, jax_lanczos_logdet

    n = omega.shape[0]

    # Build P = I + diag(ω) - λ*W_sym + λ²*WtW  (σ² = 1)
    P_diag = jnp.ones(n) + omega
    P = jnp.diag(P_diag) - lam * W_sym_dense + lam**2 * WtW_dense
    P = P + 1e-6 * jnp.eye(n)  # regularization

    # RHS: Xβ - λ*(W+W')Xβ + λ²*W'WXβ + κ
    rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

    # Jacobi preconditioner
    M_inv_diag = 1.0 / jnp.where(jnp.abs(P_diag) > 1e-15, P_diag, 1.0)

    # Lanczos logdet of P
    log_det_P = jax_lanczos_logdet(
        P, key=key, n_probes=n_probes, lanczos_deg=lanczos_deg
    )

    # CG solve P m = rhs
    m = jax_cg_solve(P, rhs, M_inv_diag, tol=cg_tol, maxiter=cg_maxiter)
    quad = rhs @ m

    # log|I - lam*W|
    logdet_W = logdet_jax(lam)

    # Missing term from SEM prior: -½Xβ'A_λ'A_λXβ
    # = -½Xβ'Xβ + ½λXβ'W_symXβ - ½λ²Xβ'W'WXβ
    # The -½Xβ'Xβ is constant in λ and drops out.
    xbeta_correction = 0.5 * lam * (Xbeta @ WsymXbeta) - 0.5 * lam**2 * (
        Xbeta @ WtWXbeta
    )

    return logdet_W - 0.5 * log_det_P + 0.5 * quad + xbeta_correction


def _sem_logit_collapsed_log_density(
    lam: float,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_fn: Callable[[float], float],
    omega: np.ndarray,
    Xbeta: np.ndarray,
    n: int,
    k: int,
) -> float:
    """Collapsed log-density of λ for SEM-logit (for testing).

    log p(λ | β, ω, y) = log|I - λW| - ½ log|P_η| + ½ rhs'P_η⁻¹ rhs
                        - ½ Xβ'A_λ'A_λXβ

    where P_η = A_λ'A_λ + diag(ω) and rhs = A_λ'A_λXβ + κ.

    The term -½Xβ'A_λ'A_λXβ arises from the SEM prior normalization
    and depends on λ (unlike SAR where -½Xβ'Xβ is constant in ρ).
    """
    kappa = y - 0.5
    W_sym = W_sparse + W_sparse.T
    WtW = W_sparse.T @ W_sparse

    # RHS: Xβ - λ*(W+W')Xβ + λ²*W'WXβ + κ
    WsymXbeta = W_sym @ Xbeta
    WtWXbeta = WtW @ Xbeta
    rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

    # Precision: P = I + diag(ω) - λ*W_sym + λ²*WtW
    P = (
        sp.eye(n, format="csr")
        + sp.diags(omega, format="csr")
        - lam * W_sym
        + lam**2 * WtW
    )

    # log|I - λW|
    logdet = logdet_fn(lam)

    # log|P| via LU — prefer KLU (scikit-sparse) over scipy SuperLU
    from ..._ops._backend import _factor_solve_logdet

    P_csc = sp.csc_matrix(P)
    m, log_det_P = _factor_solve_logdet(P_csc, rhs)
    quad = float(rhs @ m)

    # Missing term from SEM prior: -½Xβ'A_λ'A_λXβ
    # = -½Xβ'Xβ + ½λXβ'W_symXβ - ½λ²Xβ'W'WXβ
    # The -½Xβ'Xβ is constant in λ and drops out.
    xbeta_correction = 0.5 * lam * float(Xbeta @ WsymXbeta) - 0.5 * lam**2 * float(
        Xbeta @ WtWXbeta
    )

    return logdet - 0.5 * log_det_P + 0.5 * quad + xbeta_correction


# ---------------------------------------------------------------------------
# SEM-logit chain runner
# ---------------------------------------------------------------------------


def run_chain_sem(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: SEMLogitGibbsPriors,
    cache: SEMLogitGibbsCache,
    init: SEMLogitGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
) -> dict:
    """Run one chain of the SEM-logit Gibbs sampler.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Spatial weights matrix.
    priors : SEMLogitGibbsPriors
        Prior hyperparameters.
    cache : SEMLogitGibbsCache
        Precomputed data.
    init : SEMLogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.

    Returns
    -------
    dict
        Posterior samples with keys ``lam``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws

    # Pre-allocate storage for post-warmup draws
    n_keep = draws // thin if thin > 0 else draws
    lam_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Copy initial state (we mutate state in-place)
    state = SEMLogitGibbsState(
        eta=init.eta.copy(),
        beta=init.beta.copy(),
        lam=init.lam,
        omega=init.omega.copy(),
    )

    for i in range(total_iters):
        # --- Block 1: ω | η (identical to SAR-logit) ---
        state.omega = _sample_omega(state.eta, rng=rng)

        # --- Block 2: η | ω, β, λ (SEM-specific rhs) ---
        state.eta, _ = _sample_eta_sem(state, y, X, W_sparse, rng=rng, cache=cache)

        # Recompute A_λ η with new eta (no matrix build)
        A_lam_eta = state.eta - state.lam * (W_sparse @ state.eta)

        # --- Block 3: β | η, λ (SEM-style transformed data) ---
        state.beta = _sample_beta_sem(state, X, priors, A_lam_eta, W_sparse, rng=rng)

        # --- Block 4: λ | β, ω, y (collapsed, η integrated out) ---
        state.lam, log_density_lam = _sample_lam(
            state,
            cache,
            priors,
            y,
            X,
            rng=rng,
            log_density_current=None,
            sweep_idx=i,
            tune=tune,
        )

        # --- Store post-warmup draws ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                lam_samples[idx] = state.lam
                beta_samples[idx] = state.beta
                if store_log_lik:
                    log_lik_samples[idx] = _logit_loglik_pointwise(y, state.eta)
                eta_norm_samples[idx] = float(state.eta @ state.eta)
                if return_eta:
                    eta_samples[idx] = state.eta

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    result = {
        "lam": lam_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
    }
    if return_eta:
        result["eta"] = eta_samples

    return result
