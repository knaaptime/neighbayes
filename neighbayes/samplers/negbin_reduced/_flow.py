"""Reduced-form PG-Gibbs sampler for spatial-lag NB flow models.

Targets the reduced-form NB2 spatial-lag flow model

.. math::

    y_{ij} \\sim \\operatorname{NegBin}(\\mu_{ij}, \\alpha), \\qquad
    \\log \\boldsymbol{\\mu} = A(\\boldsymbol{\\rho})^{-1} X \\beta

where :math:`A = I_N - \\rho_d W_d - \\rho_o W_o - \\rho_w W_w`
(unrestricted) or :math:`A = L_d \\otimes L_o` with
:math:`L_k = I_n - \\rho_k W` and :math:`\\rho_w = -\\rho_d \\rho_o`
(separable).

The sampler has four blocks per sweep:

1. **ω** — Pólya–Gamma augmentation (shared with SAR-NB core).
2. **β** — Conjugate Gaussian given :math:`\\tilde X = A^{-1} X`.
3. **ρ** — 1-D adaptive slice on each spatial parameter (β marginalized).
4. **α** — 1-D slice on log(α) (shared with SAR-NB core).

There is no :math:`\\sigma` parameter — spatial dependence enters only
through the mean propagator :math:`A^{-1}`, matching the LeSage & Pace
spatial econometrics convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..._ops._backend import _make_cached_sparse_solver
from ...models.priors import FlowReducedGibbsPriors
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from ..negbin._core import GibbsState as _StructuralState
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
from ._core import (
    _KRYLOV_DEGREE_DEFAULT,
    _KRYLOV_DMAX_DEFAULT,
    ReducedGibbsPriors,
    ReducedKrylovBasis,
    _eval_U_from_basis,
    _sample_beta,
    _sample_omega,
)

# ---------------------------------------------------------------------------
# State, priors, cache
# ---------------------------------------------------------------------------


@dataclass
class FlowReducedGibbsState:
    """Mutable state for one chain of the reduced-form flow NB Gibbs sampler.

    Parameters
    ----------
    beta : ndarray, shape (k,)
        Regression coefficients.
    rho_d, rho_o : float
        Destination and origin spatial autoregressive parameters.
    rho_w : float or None
        Cross spatial parameter (None for separable models where
        rho_w = -rho_d * rho_o is deterministic).
    alpha : float
        NB dispersion (NB2 parameterization).
    omega : ndarray, shape (N,)
        Pólya–Gamma auxiliary variables (N = n²).
    """

    beta: np.ndarray
    rho_d: float
    rho_o: float
    rho_w: Optional[float]
    alpha: float
    omega: np.ndarray


class FlowReducedGibbsCache:
    """Precomputed constants for the flow Gibbs sweep.

    Parameters
    ----------
    Wd, Wo, Ww : scipy.sparse.csr_matrix, shape (N, N)
        Flow weight matrices (destination, origin, cross).
    W_csc : scipy.sparse.csc_matrix, shape (n, n)
        Row-standardized regional weights in CSC format (for the
        separable Kronecker solve and Krylov basis).
    n : int
        Number of regions (A is N×N where N = n²).
    separable : bool
        If True, use Kronecker factorization (rho_w = -rho_d * rho_o).
    rho_lower, rho_upper : float
        Support bounds for each ρ.
    krylov_degree : int
        Krylov basis degree for the separable variant.  Default 8.
        Ignored for the unrestricted variant.
    krylov_dmax : float
        Maximum |Δρ| for Krylov basis reuse.  Default 0.15.
    n_rho_omega_cycles : int
        Number of (ω, ρ, β) Gibbs cycles per sweep.  Default 1.
    positive : bool
        If True (the model-level ``restrict_positive``), reject any
        ``ρ_k < 0`` in the unrestricted ρ conditionals — the Gibbs
        counterpart of the Dirichlet-simplex prior used by the NUTS
        path.  Ignored for the separable variant (which, like its NUTS
        counterpart, always uses the box prior).
    T : int
        Number of panel periods sharing the same per-period system
        matrix ``A`` (``1`` for the cross-section).  ``y``/``X`` are
        stacked time-first with ``N_f·T`` rows, ``N_f = n²``.
    """

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
        W_csc: sp.csc_matrix,
        n: int,
        separable: bool = False,
        rho_lower: float = -0.999,
        rho_upper: float = 0.999,
        krylov_degree: int = _KRYLOV_DEGREE_DEFAULT,
        krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
        n_rho_omega_cycles: int = 1,
        positive: bool = False,
        T: int = 1,
        krylov_reuse: bool = True,
        krylov_reuse_threshold: float = 0.15,
    ):
        self.Wd = Wd
        self.Wo = Wo
        self.Ww = Ww
        self.W_csc = W_csc
        self.n = n
        self.separable = separable
        self.rho_lower = rho_lower
        self.rho_upper = rho_upper
        self.krylov_degree = krylov_degree
        self.krylov_dmax = krylov_dmax
        self.n_rho_omega_cycles = n_rho_omega_cycles
        self.positive = positive
        self.T = int(T)
        self.Nf = int(Wd.shape[0])  # per-period flow count (n²)
        self.krylov_reuse = krylov_reuse
        self.krylov_reuse_threshold = krylov_reuse_threshold

        # Eigenvalue bounds for the regional W (n×n).
        # For row-standardized W: eigenvalues in [-1, 1].
        try:
            eigvals = sp.linalg.eigsh(W_csc.astype(np.float64), k=1, which="LM")
            self.W_eig_max = float(np.max(np.abs(eigvals[0])))
        except Exception:
            self.W_eig_max = 1.0
        try:
            eigvals = sp.linalg.eigsh(W_csc.astype(np.float64), k=1, which="SA")
            self.W_eig_min = float(eigvals[0][0])
        except Exception:
            self.W_eig_min = -1.0

        # Adaptive slice width states for each ρ parameter
        self.rho_d_slice_width_state = SliceWidthState()
        self.rho_o_slice_width_state = SliceWidthState()
        self.rho_w_slice_width_state = SliceWidthState()


# ---------------------------------------------------------------------------
# Helpers: assemble A and solve A⁻¹X
# ---------------------------------------------------------------------------


def _assemble_A_unrestricted(
    rho_d: float,
    rho_o: float,
    rho_w: float,
    Wd: sp.csr_matrix,
    Wo: sp.csr_matrix,
    Ww: sp.csr_matrix,
    N: int,
) -> sp.csr_matrix:
    """Assemble A = I_N - rho_d*Wd - rho_o*Wo - rho_w*Ww."""
    eye = sp.eye(N, format="csr", dtype=np.float64)
    return eye - rho_d * Wd - rho_o * Wo - rho_w * Ww


def flow_system_is_invertible(
    rho_d: float,
    rho_o: float,
    rho_w: float,
    lam_min: float,
    lam_max: float,
    margin: float = 1e-3,
) -> bool:
    r"""Spectral admissibility test for :math:`A = I - \rho_d W_d - \rho_o W_o - \rho_w W_w`.

    With ``W`` diagonalisable with real spectrum, the eigenvalues of ``A`` are

    .. math::
        1 - \rho_d \lambda_b - \rho_o \lambda_a - \rho_w \lambda_a \lambda_b

    over pairs of eigenvalues of ``W``.  That expression is *bilinear* in
    :math:`(\lambda_a, \lambda_b)`, so over the box
    :math:`[\lambda_{\min}, \lambda_{\max}]^2` its extrema occur at the four
    corners — checking those is exact for the box and costs four flops.

    This replaces the earlier :math:`|\rho_d| + |\rho_o| + |\rho_w| < 1` rule,
    which is *sufficient* for invertibility but far from necessary: it rejected
    ``(0.487, 0.391, -0.122)`` — the profile MLE on a well-behaved synthetic
    flow dataset, with ``min|eig(A)| = 0.244`` — and truncated the posterior
    where the likelihood was still rising.  The corner test admits that region
    while still rejecting genuinely singular draws.

    Parameters
    ----------
    rho_d, rho_o, rho_w : float
        Candidate spatial parameters.
    lam_min, lam_max : float
        Bounds on the spectrum of the regional weights matrix.
    margin : float, default 1e-3
        Required distance from singularity; guards the near-singular ridge the
        plain box prior would otherwise admit.

    Returns
    -------
    bool
        True when every corner eigenvalue is positive and at least ``margin``
        from zero.
    """
    lo = min(lam_min, lam_max)
    hi = max(lam_min, lam_max)
    for lam_a in (lo, hi):
        for lam_b in (lo, hi):
            val = 1.0 - rho_d * lam_b - rho_o * lam_a - rho_w * lam_a * lam_b
            if val <= margin:
                return False
    return True


def _stack_periods(X: np.ndarray, Nf: int, T: int) -> np.ndarray:
    """Reshape a time-first stacked ``(Nf·T, k)`` RHS to ``(Nf, T·k)``.

    Lets one factorization of the per-period ``A`` (``Nf × Nf``) cover all
    ``T`` period blocks in a single batched backsolve.
    """
    k = X.shape[1]
    return X.reshape(T, Nf, k).transpose(1, 0, 2).reshape(Nf, T * k)


def _unstack_periods(U: np.ndarray, Nf: int, T: int) -> np.ndarray:
    """Inverse of :func:`_stack_periods`: ``(Nf, T·k)`` → ``(Nf·T, k)``."""
    k = U.shape[1] // T
    return U.reshape(Nf, T, k).transpose(1, 0, 2).reshape(T * Nf, k)


def _solve_A_unrestricted(
    A: sp.csr_matrix,
    X: np.ndarray,
    T: int = 1,
) -> np.ndarray:
    """Solve (I_T ⊗ A) η = X via sparse LU, returning η = A⁻¹X per period.

    ``A`` is the per-period ``Nf × Nf`` system matrix; ``X`` is the
    time-first stacked ``(Nf·T, k)`` RHS.  One factorization covers all
    periods.  Uses KLU (scikit-sparse) when available, falling
    back to scipy SuperLU.
    """
    from ..._ops._backend import _solve_sparse_matrix

    if T == 1:
        return _solve_sparse_matrix(A, X)
    Nf = A.shape[0]
    return _unstack_periods(_solve_sparse_matrix(A, _stack_periods(X, Nf, T)), Nf, T)


def _solve_A_separable(
    rho_d: float,
    rho_o: float,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    T: int = 1,
) -> np.ndarray:
    """Solve (I_T ⊗ L_o ⊗ L_d) η = X via Kronecker two-step solve.

    L_k = I_n - rho_k * W (n×n).
    X has shape (Nf·T, k) where Nf = n², stacked time-first.
    """
    from ..._ops import kron_solve_matrix

    I_n = sp.eye(n, format="csr", dtype=np.float64)
    Ld = (I_n - rho_d * W_csc).tocsr()
    Lo = (I_n - rho_o * W_csc).tocsr()
    if T == 1:
        return kron_solve_matrix(Lo, Ld, X, n)
    Nf = n * n
    Xb = _stack_periods(X, Nf, T)
    return _unstack_periods(kron_solve_matrix(Lo, Ld, Xb, n), Nf, T)


# ---------------------------------------------------------------------------
# Kronecker-aware Krylov basis for the separable flow model
# ---------------------------------------------------------------------------


def _build_kron_krylov_basis(
    rho_d_c: float,
    rho_o_c: float,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    direction: str,
    degree: int,
) -> ReducedKrylovBasis:
    """Build a shift-invert Krylov basis for the separable Kronecker system.

    When varying ρ_d (holding ρ_o fixed), the system matrix is

    .. math::
        A(\\rho_d) = L_o \\otimes L_d(\\rho_d)
        = A_c - \\Delta\\rho_d \\, (L_o \\otimes W)

    so the Krylov recurrence uses the matvec ``(L_o ⊗ W) v`` and the
    solve ``A_c⁻¹ rhs`` (a Kronecker two-step).  Symmetrically for ρ_o,
    the matvec is ``(W ⊗ L_d) v``.

    Parameters
    ----------
    rho_d_c, rho_o_c : float
        Centre values at which ``A_c`` is factored.
    X : ndarray, shape (N, k)
        Design matrix on the flow lattice (N = n²).
    W_csc : csc_matrix, shape (n, n)
        Regional weights matrix.
    n : int
        Number of regions.
    direction : {"rho_d", "rho_o"}
        Which ρ the basis is for — determines the matvec operator.
    degree : int
        Krylov degree m.

    Returns
    -------
    ReducedKrylovBasis
        With ``rho_basis`` set to the ρ_k being varied, ``V_stack`` of
        shape ``(m+1, N, k)``, and ``solver`` set to ``None`` (the solve
        is done via Kronecker two-step, not a single factor).
    """
    from ..._ops import kron_solve_matrix

    I_n = sp.eye(n, format="csr", dtype=np.float64)
    Ld_c = (I_n - rho_d_c * W_csc).tocsr()
    Lo_c = (I_n - rho_o_c * W_csc).tocsr()
    N = n * n
    m = degree
    V_stack = np.empty((m + 1, N, X.shape[1]), dtype=np.float64)

    def _kron_solve(rhs):
        return kron_solve_matrix(Lo_c, Ld_c, rhs, n)

    # V_0 = A_c⁻¹ X
    V_stack[0] = _kron_solve(X)

    def _kron_matvec(Aop, Bop, v):
        """``(B ⊗ A) v`` for the column-major vec convention.

        ``kron_solve_matrix(Lo, Ld, ...)`` solves ``A = kron(Lo, Ld)``, so with
        ``v = vec(H)`` stacked column-major, ``(B ⊗ A) vec(H) = vec(A H Bᵀ)``:
        apply ``A`` along axis 0 and ``B`` along axis 1.
        """
        kc = v.shape[1]
        H = v.reshape(n, n, kc, order="F")
        AH = (Aop @ H.reshape(n, n * kc)).reshape(n, n, kc)
        T = AH.transpose(1, 0, 2)
        BT = (Bop @ T.reshape(n, n * kc)).reshape(n, n, kc)
        return BT.transpose(1, 0, 2).reshape(N, kc, order="F")

    # A(ρ_d) = A_c - Δρ_d (L_o ⊗ W)  and  A(ρ_o) = A_c - Δρ_o (W ⊗ L_d);
    # both derivatives are verified against explicit Kronecker products in
    # tests/test_samplers/test_kron_krylov_matvec.py.
    if direction == "rho_d":

        def _matvec(v):
            return _kron_matvec(W_csc, Lo_c, v)
    else:  # rho_o

        def _matvec(v):
            return _kron_matvec(Ld_c, W_csc, v)

    for j in range(m):
        Wv = _matvec(V_stack[j])
        V_stack[j + 1] = _kron_solve(Wv)

    rho_basis_val = rho_d_c if direction == "rho_d" else rho_o_c
    return ReducedKrylovBasis(
        rho_basis=rho_basis_val,
        solver=None,  # Kronecker solve, not a single factor
        V_stack=V_stack,
        degree=m,
    )


def build_unrestricted_krylov_basis(
    rho_d_c: float,
    rho_o_c: float,
    rho_w_c: float,
    X: np.ndarray,
    Wd: sp.csr_matrix,
    Wo: sp.csr_matrix,
    Ww: sp.csr_matrix,
    direction: str,
    degree: int = _KRYLOV_DEGREE_DEFAULT,
) -> "ReducedKrylovBasis":
    r"""Shift-invert Krylov basis for one :math:`\rho` of the unrestricted flow system.

    The unrestricted system is affine in each :math:`\rho` separately,

    .. math::
        A = I - \rho_d W_d - \rho_o W_o - \rho_w W_w
          = A_c - \Delta\rho_k W_k ,

    so the same shift-invert series the separable and cross-section samplers
    use applies here, with ``W_k`` the operator for the direction being sliced:

    .. math::
        V_0 = A_c^{-1} X, \qquad V_{j+1} = A_c^{-1} (W_k V_j),
        \qquad U(\rho_c + \Delta\rho) \approx \sum_j (\Delta\rho)^j V_j .

    Without this the unrestricted :math:`\rho` blocks refactorize ``A`` at
    every slice candidate — measured at 22.7 factorizations per sweep against
    the 3 basis builds this needs.

    ``safe_dmax`` comes from a root test on the coefficients themselves, so no
    eigenvalue bounds are required; callers must clamp their configured
    ``krylov_dmax`` to it, since the series diverges beyond that radius.

    Parameters
    ----------
    rho_d_c, rho_o_c, rho_w_c : float
        Centre point at which ``A_c`` is factored.
    X : ndarray, shape (N, k)
        Design matrix on the flow lattice.
    Wd, Wo, Ww : sparse
        Destination, origin and cross flow-weight operators.
    direction : {"rho_d", "rho_o", "rho_w"}
        Which :math:`\rho` the basis varies.
    degree : int
        Krylov degree.

    Returns
    -------
    ReducedKrylovBasis
    """
    from .._utils._spatial_normal import _series_radius
    from ._core import ReducedKrylovBasis

    ops = {"rho_d": Wd, "rho_o": Wo, "rho_w": Ww}
    if direction not in ops:
        raise ValueError(f"Unknown direction: {direction!r}")
    Wk = ops[direction]
    centres = {"rho_d": rho_d_c, "rho_o": rho_o_c, "rho_w": rho_w_c}

    N = X.shape[0]
    A_c = _assemble_A_unrestricted(rho_d_c, rho_o_c, rho_w_c, Wd, Wo, Ww, N)
    solver = _make_cached_sparse_solver(A_c)
    if solver is None:
        solver = spla.splu(A_c.tocsc())

    m = degree
    V_stack = np.empty((m + 1, N, X.shape[1]), dtype=np.float64)
    V_stack[0] = solver.solve(X)
    for j in range(m):
        V_stack[j + 1] = solver.solve(Wk @ V_stack[j])

    return ReducedKrylovBasis(
        rho_basis=centres[direction],
        solver=solver,
        V_stack=V_stack,
        degree=m,
        safe_dmax=_series_radius(V_stack),
    )


def _compute_eta_unrestricted(
    rho_d: float,
    rho_o: float,
    rho_w: float,
    Xbeta: np.ndarray,
    cache: FlowReducedGibbsCache,
) -> np.ndarray:
    """Compute η = A⁻¹ Xβ (per period) for the unrestricted model.

    Uses KLU (scikit-sparse) when available, falling back to
    scipy SuperLU.
    """
    from ..._ops._backend import _solve_sparse_vector

    Nf = cache.Nf
    A = _assemble_A_unrestricted(rho_d, rho_o, rho_w, cache.Wd, cache.Wo, cache.Ww, Nf)
    if cache.T == 1:
        return _solve_sparse_vector(A, Xbeta)
    return _solve_A_unrestricted(A, Xbeta.reshape(-1, 1), T=cache.T).ravel()


def _compute_eta_separable(
    rho_d: float,
    rho_o: float,
    Xbeta: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    T: int = 1,
) -> np.ndarray:
    """Compute η = A⁻¹ Xβ (per period) for the separable model."""
    return _solve_A_separable(rho_d, rho_o, Xbeta.reshape(-1, 1), W_csc, n, T=T).ravel()


# ---------------------------------------------------------------------------
# ρ slice density (β-marginalized)
# ---------------------------------------------------------------------------


def _rho_log_density_marginal_flow(
    rho_val: float,
    rho_name: str,
    state: FlowReducedGibbsState,
    cache: FlowReducedGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors: FlowReducedGibbsPriors,
    *,
    basis: Optional[object] = None,
    intercept_col: int = -1,
) -> float:
    r"""β-marginalized conditional log-density for one ρ parameter.

    For the unrestricted model, each ρ_k is sliced while holding the
    other two fixed.  The system matrix A depends on all three ρ's,
    so we recompute A⁻¹X at each candidate.

    For the separable model, ρ_d and ρ_o are sliced independently.
    The Kronecker structure means A⁻¹X can be computed via two n×n
    solves, and a Krylov basis can be used for each ρ_k.
    """
    if rho_val <= cache.rho_lower or rho_val >= cache.rho_upper:
        return -np.inf

    # Build the current ρ vector with the candidate value
    if rho_name == "rho_d":
        rho_d = rho_val
        rho_o = state.rho_o
        rho_w = state.rho_w if state.rho_w is not None else -rho_d * rho_o
    elif rho_name == "rho_o":
        rho_d = state.rho_d
        rho_o = rho_val
        rho_w = state.rho_w if state.rho_w is not None else -rho_d * rho_o
    elif rho_name == "rho_w":
        rho_d = state.rho_d
        rho_o = state.rho_o
        rho_w = rho_val
    else:
        raise ValueError(f"Unknown rho_name: {rho_name}")

    if not cache.separable:
        # Joint stability wall: |ρ_d|+|ρ_o|+|ρ_w| < 1 is the sufficient
        # invertibility bound for row-standardized weights.  Without it the
        # box prior admits a near-singular likelihood ridge (e.g. large ρ_w
        # with negative ρ_d/ρ_o) that the NUTS path excludes by prior.
        if not flow_system_is_invertible(
            rho_d, rho_o, rho_w, cache.W_eig_min, cache.W_eig_max
        ):
            return -np.inf
        if cache.positive and (rho_d < 0.0 or rho_o < 0.0 or rho_w < 0.0):
            return -np.inf

    k = X.shape[1]
    n = cache.n
    omega = state.omega
    alpha = state.alpha
    log_alpha = np.log(alpha)
    kappa = 0.5 * (y - alpha)

    # Prior precision and mean
    beta_sigma = priors.beta_sigma
    if np.isscalar(beta_sigma):
        V0_inv_diag = np.full(k, 1.0 / (float(beta_sigma) ** 2))
    else:
        V0_inv_diag = 1.0 / (np.asarray(beta_sigma, dtype=np.float64) ** 2)
    beta_mu = priors.beta_mu
    if np.isscalar(beta_mu):
        mu0 = np.full(k, float(beta_mu))
    else:
        mu0 = np.asarray(beta_mu, dtype=np.float64)

    # Compute U = A⁻¹X at the candidate ρ (per period for panels)
    try:
        if cache.separable:
            # Use Krylov basis if available (cross-section only)
            if basis is not None and basis.degree > 0 and cache.T == 1:
                drho = rho_val - basis.rho_basis
                if abs(drho) <= cache.krylov_dmax:
                    U = _eval_U_from_basis(basis, drho)
                else:
                    U = _solve_A_separable(rho_d, rho_o, X, cache.W_csc, n, T=cache.T)
            else:
                U = _solve_A_separable(rho_d, rho_o, X, cache.W_csc, n, T=cache.T)
        else:
            # Same shift-invert series as the separable path: A is affine in
            # each rho separately, so one basis per direction replaces a
            # refactorization at every slice candidate (~23 per sweep).
            # Clamped to the basis's own convergence radius; beyond it the
            # series diverges and would silently return nonsense.
            U = None
            if basis is not None and getattr(basis, "degree", 0) > 0 and cache.T == 1:
                drho = rho_val - basis.rho_basis
                if abs(drho) <= min(
                    cache.krylov_dmax, getattr(basis, "safe_dmax", np.inf)
                ):
                    U = _eval_U_from_basis(basis, drho)
            if U is None:
                A = _assemble_A_unrestricted(
                    rho_d, rho_o, rho_w, cache.Wd, cache.Wo, cache.Ww, cache.Nf
                )
                U = _solve_A_unrestricted(A, X, T=cache.T)
    except (RuntimeError, ValueError, spla.ArpackNoConvergence):
        return -np.inf

    # Working response and residual
    s = kappa / omega + log_alpha
    r = s - U @ mu0

    # M = V0^{-1} + U^T Ω U  (k x k)
    Uw = U * omega[:, None]
    M = U.T @ Uw
    M.flat[:: k + 1] += V0_inv_diag

    v = Uw.T @ r

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf

    from scipy.linalg import solve_triangular

    w = solve_triangular(L, v, lower=True)
    quad_pen = float(w @ w)
    rOr = float(np.dot(r, omega * r))
    log_det_M = 2.0 * float(np.sum(np.log(np.diag(L))))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)
    if not np.isfinite(result):
        return -np.inf
    return result


def _sample_rho_k(
    rho_name: str,
    state: FlowReducedGibbsState,
    cache: FlowReducedGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors: FlowReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    sweep_idx: int,
    tune: int,
    basis: Optional[object] = None,
    intercept_col: int = -1,
) -> float:
    """1-D adaptive slice on one ρ parameter with β marginalized."""
    rho_lower = cache.rho_lower
    rho_upper = cache.rho_upper
    if cache.positive and not cache.separable:
        rho_lower = max(rho_lower, 0.0)

    # Get current value and width state
    if rho_name == "rho_d":
        rho_current = state.rho_d
        width_state = cache.rho_d_slice_width_state
    elif rho_name == "rho_o":
        rho_current = state.rho_o
        width_state = cache.rho_o_slice_width_state
    elif rho_name == "rho_w":
        rho_current = state.rho_w
        width_state = cache.rho_w_slice_width_state
    else:
        raise ValueError(f"Unknown rho_name: {rho_name}")

    def log_density(rho_val: float) -> float:
        return _rho_log_density_marginal_flow(
            rho_val=rho_val,
            rho_name=rho_name,
            state=state,
            cache=cache,
            y=y,
            X=X,
            priors=priors,
            basis=basis,
            intercept_col=intercept_col,
        )

    log_dens_x0 = log_density(rho_current)
    rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
        log_density=log_density,
        x0=rho_current,
        lower=rho_lower,
        upper=rho_upper,
        width_state=width_state,
        rng=rng,
        log_density_x0=log_dens_x0,
    )
    if sweep_idx < tune:
        update_slice_width(width_state, steps_left, steps_right)

    return rho_new


# ---------------------------------------------------------------------------
# Chain runner — unrestricted (3 ρ's)
# ---------------------------------------------------------------------------


def run_chain_unrestricted(
    y: np.ndarray,
    X: np.ndarray,
    Wd: sp.csr_matrix,
    Wo: sp.csr_matrix,
    Ww: sp.csr_matrix,
    priors: FlowReducedGibbsPriors,
    cache: FlowReducedGibbsCache,
    init: FlowReducedGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain of the reduced-form flow NB Gibbs sampler (unrestricted).

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer responses (N = n²).
    X : ndarray, shape (N, k)
        Design matrix.
    Wd, Wo, Ww : scipy.sparse.csr_matrix, shape (N, N)
        Flow weight matrices.
    priors : FlowReducedGibbsPriors
    cache : FlowReducedGibbsCache
    init : FlowReducedGibbsState
    draws, tune : int
        Post-warmup draws and warmup sweeps.
    thin : int, default 1
    rng : numpy.random.Generator, optional
    chain_id : int, default 0
    progress_manager : object, optional

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``rho_d``, ``rho_o``, ``rho_w``, ``beta``, ``alpha``,
        ``log_lik``.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    rho_d_samples = np.empty(n_keep, dtype=np.float64)
    rho_o_samples = np.empty(n_keep, dtype=np.float64)
    rho_w_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    # One value per draw and flow; skipped unless requested (``None``).
    log_lik_samples = np.empty((n_keep, N), dtype=np.float64) if store_log_lik else None

    state = FlowReducedGibbsState(
        beta=np.asarray(init.beta, dtype=np.float64).copy(),
        rho_d=float(init.rho_d),
        rho_o=float(init.rho_o),
        rho_w=float(init.rho_w) if init.rho_w is not None else 0.0,
        alpha=float(init.alpha),
        omega=np.asarray(init.omega, dtype=np.float64).copy(),
    )

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Detect intercept column
    intercept_col = -1
    for _j in range(k):
        if np.all(X[:, _j] == 1.0):
            intercept_col = _j
            break

    # Shared priors object for _sample_beta (same fields)
    core_priors = ReducedGibbsPriors(
        beta_mu=priors.beta_mu,
        beta_sigma=priors.beta_sigma,
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        rho_lower=priors.rho_lower,
        rho_upper=priors.rho_upper,
    )

    for i in range(total_iters):
        # Compute η = A⁻¹ Xβ at current ρ's (per period for panels)
        Xbeta = X @ state.beta
        eta = _compute_eta_unrestricted(
            state.rho_d, state.rho_o, state.rho_w, Xbeta, cache
        )
        psi = eta - np.log(state.alpha)

        # --- Block 1: ω ---
        state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- (ρ, β) cycles ---
        _n_cycles = cache.n_rho_omega_cycles
        Xtilde = None

        # One shift-invert basis per ρ direction, replacing a refactorization
        # at every slice candidate (~23 per sweep, measured).
        _bases: dict[str, object] = {}
        if cache.krylov_degree > 0 and cache.T == 1:
            for _dir in ("rho_d", "rho_o", "rho_w"):
                try:
                    _bases[_dir] = build_unrestricted_krylov_basis(
                        state.rho_d,
                        state.rho_o,
                        state.rho_w,
                        X,
                        Wd,
                        Wo,
                        Ww,
                        _dir,
                        cache.krylov_degree,
                    )
                except (RuntimeError, ValueError):
                    _bases[_dir] = None

        for _cycle in range(_n_cycles):
            # --- ρ_d | ω, α, y (β marginalized) ---
            state.rho_d = _sample_rho_k(
                "rho_d",
                state,
                cache,
                y,
                X,
                priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                intercept_col=intercept_col,
                basis=_bases.get("rho_d"),
            )

            # --- ρ_o | ω, α, y (β marginalized) ---
            state.rho_o = _sample_rho_k(
                "rho_o",
                state,
                cache,
                y,
                X,
                priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                intercept_col=intercept_col,
                basis=_bases.get("rho_o"),
            )

            # --- ρ_w | ω, α, y (β marginalized) ---
            state.rho_w = _sample_rho_k(
                "rho_w",
                state,
                cache,
                y,
                X,
                priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                intercept_col=intercept_col,
                basis=_bases.get("rho_w"),
            )

            # --- β | ρ, ω, α, y ---
            A = _assemble_A_unrestricted(
                state.rho_d, state.rho_o, state.rho_w, Wd, Wo, Ww, cache.Nf
            )
            Xtilde = _solve_A_unrestricted(A, X, T=cache.T)

            state.beta = _sample_beta(
                beta_current=state.beta,
                Xtilde=Xtilde,
                omega=state.omega,
                y=y,
                alpha=state.alpha,
                priors=core_priors,
                rng=rng,
                rho=0.0,  # no single-ρ intercept reparam for flow
                intercept_col=-1,  # disable intercept reparam
            )

            # η = X̃ @ β
            eta = Xtilde @ state.beta

            # Draw ω for next cycle
            if _cycle < _n_cycles - 1:
                psi = eta - np.log(state.alpha)
                state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- Block 4: α | y, η ---
        alpha_state = _StructuralState(
            eta=eta,
            beta=state.beta,
            sigma2=1.0,
            rho=state.rho_d,
            alpha=state.alpha,
            omega=state.omega,
        )
        state.alpha = _sample_alpha(alpha_state, y, core_priors, rng=rng)

        # --- Store post-warmup draw ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_d_samples[idx] = state.rho_d
                rho_o_samples[idx] = state.rho_o
                rho_w_samples[idx] = state.rho_w
                beta_samples[idx] = state.beta
                alpha_samples[idx] = state.alpha
                if store_log_lik:
                    log_lik_samples[idx] = _nb_loglik_pointwise(y, eta, state.alpha)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    return {
        "rho_d": rho_d_samples,
        "rho_o": rho_o_samples,
        "rho_w": rho_w_samples,
        "beta": beta_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
    }


# ---------------------------------------------------------------------------
# Chain runner — separable (2 ρ's, Kronecker)
# ---------------------------------------------------------------------------


def run_chain_separable(
    y: np.ndarray,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    priors: FlowReducedGibbsPriors,
    cache: FlowReducedGibbsCache,
    init: FlowReducedGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain of the reduced-form flow NB Gibbs sampler (separable).

    Parameters
    ----------
    y : ndarray, shape (N,)
        Integer responses (N = n²).
    X : ndarray, shape (N, k)
        Design matrix.
    W_csc : scipy.sparse.csc_matrix, shape (n, n)
        Row-standardized regional weights.
    n : int
        Number of regions.
    priors : FlowReducedGibbsPriors
    cache : FlowReducedGibbsCache
    init : FlowReducedGibbsState
    draws, tune : int
    thin : int, default 1
    rng : numpy.random.Generator, optional
    chain_id : int, default 0
    progress_manager : object, optional

    Returns
    -------
    dict[str, np.ndarray]
        Keys: ``rho_d``, ``rho_o``, ``rho_w``, ``beta``, ``alpha``,
        ``log_lik``.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    rho_d_samples = np.empty(n_keep, dtype=np.float64)
    rho_o_samples = np.empty(n_keep, dtype=np.float64)
    rho_w_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    # One value per draw and flow; skipped unless requested (``None``).
    log_lik_samples = np.empty((n_keep, N), dtype=np.float64) if store_log_lik else None

    state = FlowReducedGibbsState(
        beta=np.asarray(init.beta, dtype=np.float64).copy(),
        rho_d=float(init.rho_d),
        rho_o=float(init.rho_o),
        rho_w=None,  # deterministic: -rho_d * rho_o
        alpha=float(init.alpha),
        omega=np.asarray(init.omega, dtype=np.float64).copy(),
    )

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Detect intercept column
    intercept_col = -1
    for _j in range(k):
        if np.all(X[:, _j] == 1.0):
            intercept_col = _j
            break

    core_priors = ReducedGibbsPriors(
        beta_mu=priors.beta_mu,
        beta_sigma=priors.beta_sigma,
        alpha_sigma=priors.alpha_sigma,
        alpha_nu=priors.alpha_nu,
        rho_lower=priors.rho_lower,
        rho_upper=priors.rho_upper,
    )

    use_krylov = cache.krylov_degree > 0 and cache.T == 1
    krylov_degree = cache.krylov_degree

    # Per-chain Krylov basis caches for reuse across sweeps.
    _prev_basis_d = None
    _prev_rho_d = None
    _prev_basis_o = None
    _prev_rho_o = None

    for i in range(total_iters):
        # Compute η = A⁻¹ Xβ at current ρ's (per period for panels)
        Xbeta = X @ state.beta
        eta = _compute_eta_separable(
            state.rho_d, state.rho_o, Xbeta, W_csc, n, T=cache.T
        )
        psi = eta - np.log(state.alpha)

        # --- Block 1: ω ---
        state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- (ρ, β) cycles ---
        _n_cycles = cache.n_rho_omega_cycles
        Xtilde = None

        # Build Krylov bases for ρ_d and ρ_o at current values (with reuse)
        basis_d = None
        basis_o = None
        if use_krylov:
            if (
                cache.krylov_reuse
                and _prev_basis_d is not None
                and abs(state.rho_d - _prev_rho_d) < cache.krylov_reuse_threshold
            ):
                basis_d = _prev_basis_d
            else:
                try:
                    basis_d = _build_kron_krylov_basis(
                        state.rho_d,
                        state.rho_o,
                        X,
                        W_csc,
                        n,
                        direction="rho_d",
                        degree=krylov_degree,
                    )
                except (RuntimeError, ValueError):
                    basis_d = None
                _prev_basis_d = basis_d
                _prev_rho_d = state.rho_d

            if (
                cache.krylov_reuse
                and _prev_basis_o is not None
                and abs(state.rho_o - _prev_rho_o) < cache.krylov_reuse_threshold
            ):
                basis_o = _prev_basis_o
            else:
                try:
                    basis_o = _build_kron_krylov_basis(
                        state.rho_d,
                        state.rho_o,
                        X,
                        W_csc,
                        n,
                        direction="rho_o",
                        degree=krylov_degree,
                    )
                except (RuntimeError, ValueError):
                    basis_o = None
                _prev_basis_o = basis_o
                _prev_rho_o = state.rho_o

        for _cycle in range(_n_cycles):
            # --- ρ_d | ω, α, y (β marginalized) ---
            state.rho_d = _sample_rho_k(
                "rho_d",
                state,
                cache,
                y,
                X,
                priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                basis=basis_d,
                intercept_col=intercept_col,
            )

            # --- ρ_o | ω, α, y (β marginalized) ---
            state.rho_o = _sample_rho_k(
                "rho_o",
                state,
                cache,
                y,
                X,
                priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                basis=basis_o,
                intercept_col=intercept_col,
            )

            # --- β | ρ, ω, α, y ---
            Xtilde = _solve_A_separable(
                state.rho_d, state.rho_o, X, W_csc, n, T=cache.T
            )

            state.beta = _sample_beta(
                beta_current=state.beta,
                Xtilde=Xtilde,
                omega=state.omega,
                y=y,
                alpha=state.alpha,
                priors=core_priors,
                rng=rng,
                rho=0.0,
                intercept_col=-1,
            )

            # η = X̃ @ β
            eta = Xtilde @ state.beta

            # Draw ω for next cycle
            if _cycle < _n_cycles - 1:
                psi = eta - np.log(state.alpha)
                state.omega = _sample_omega(y, state.alpha, psi, rng=rng)

        # --- Block 4: α | y, η ---
        alpha_state = _StructuralState(
            eta=eta,
            beta=state.beta,
            sigma2=1.0,
            rho=state.rho_d,
            alpha=state.alpha,
            omega=state.omega,
        )
        state.alpha = _sample_alpha(alpha_state, y, core_priors, rng=rng)

        # --- Store post-warmup draw ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_d_samples[idx] = state.rho_d
                rho_o_samples[idx] = state.rho_o
                rho_w_samples[idx] = -state.rho_d * state.rho_o
                beta_samples[idx] = state.beta
                alpha_samples[idx] = state.alpha
                if store_log_lik:
                    log_lik_samples[idx] = _nb_loglik_pointwise(y, eta, state.alpha)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    return {
        "rho_d": rho_d_samples,
        "rho_o": rho_o_samples,
        "rho_w": rho_w_samples,
        "beta": beta_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
    }
