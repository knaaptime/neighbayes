"""Sample from a multivariate normal with sparse spatial precision.

Draws x ~ N(m, Σ) where Σ⁻¹ = P (sparse SPD) via sparse Cholesky
factorization, conjugate gradient (CG) iterative solve, or Chebyshev
polynomial approximation.

**Factorization path** (default for moderate n):
    Uses CHOLMOD (through ``sparsax`` when installed, else ``scikit-sparse``),
    which is 5–9× faster than
    ``scipy.sparse.linalg.splu`` for SPD matrices.  CHOLMOD applies a
    fill-reducing permutation P_perm such that

        P_perm P P_permᵀ = L Lᵀ.

    Sampling from N(0, P⁻¹) therefore requires undoing that permutation:

        x = m + P_permᵀ L⁻ᵀ z,  z ~ N(0, I)

    which gives Cov(x) = P_permᵀ (L Lᵀ)⁻¹ P_perm = P⁻¹.

**Iterative path** (for large n with high fill-in):
    Uses preconditioned CG for the mean solve and Lanczos-based
    stochastic log-determinant estimation.  Avoids the O(nnz^{1.5})
    factorization cost entirely.

**JAX dense path** (for n ≤ ~5000 with JAX installed):
    Uses JAX dense matvec + vmap over Lanczos probes and Chebyshev
    draws.  3–4× faster for single draws, 20–27× per-draw when
    batching Chebyshev draws.  Requires ``jax_enable_x64=True``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import CholeskyFactor as _SkCholeskyFactor

# ---------------------------------------------------------------------------
# CHOLMOD factorization wrapper
# ---------------------------------------------------------------------------


# CHOLMOD solve modes (cholmod.h): A x = b, L' x = b, and x = P' b.
_MODE_A, _MODE_LT, _MODE_PT = 0, 5, 8


class CholmodFactor:
    """Wrapper around a CHOLMOD factorization for a fixed sparsity pattern.

    Stores the symbolic analysis so that ``factorize`` only does the
    numeric factorization when the matrix values change but the
    sparsity pattern stays the same.  This is the key optimization
    for the ρ block in the Gibbs sampler, where P_η changes with each
    candidate ρ but always has the same non-zero structure.

    Two backends, chosen once at construction:

    * **sparsax** (when importable) — its NumPy CHOLMOD bindings, which keep
      one symbolic analysis per pattern and refactor supernodally.
    * **scikit-sparse** otherwise.  scikit-sparse 0.5's ``factorize`` ends
      every factorization with a conversion to a simplicial factor, and a
      refactor of a simplicial factor is itself simplicial — 5× slower than
      supernodal at n ≈ 160,000.  So the symbolic analysis is kept untouched
      and each ``factorize`` refactors a copy of it, which runs supernodally
      whenever the analysis chose that.

    Parameters
    ----------
    pattern_matrix : sparse matrix
        Any matrix with the target sparsity pattern.  Used only for
        the symbolic analysis; the numeric values are irrelevant.
    """

    def __init__(self, pattern_matrix: sp.spmatrix) -> None:
        self._pattern_matrix = sp.csc_matrix(pattern_matrix)
        self._pattern_matrix.sort_indices()
        self._setup()

    def _setup(self) -> None:
        from ..._jax_dispatch import _sparsax_available

        P = self._pattern_matrix
        self._n = P.shape[0]
        self._sparsax = None
        if _sparsax_available():
            import sparsax_cpp

            self._sparsax = sparsax_cpp
            # The upper triangle in CSC order.  A matrix with the pattern's
            # exact CSC structure maps to it by one gather (``_upper_pos``);
            # anything else is matched on sorted (col, row) keys.
            col = np.repeat(np.arange(self._n), np.diff(P.indptr))
            upper = P.indices <= col
            self._indptr, self._indices = P.indptr.copy(), P.indices.copy()
            self._upper_pos = np.flatnonzero(upper)
            self._Ai = P.indices[upper].astype(np.int32)
            self._Aj = col[upper].astype(np.int32)
            self._keys = self._Aj.astype(np.int64) * self._n + self._Ai
            self._Ax = np.ascontiguousarray(P.data[upper], dtype=np.float64)
        else:
            self._symbolic = _SkCholeskyFactor(P)
            self._factor = self._symbolic.copy().factorize(P)

    def __getstate__(self) -> dict:
        """Support pickling: store pattern matrix, drop C factor."""
        return {"_pattern_matrix": self._pattern_matrix}

    def __setstate__(self, state: dict) -> None:
        """Reconstruct the factorization from the pattern matrix on unpickle."""
        self._pattern_matrix = state["_pattern_matrix"]
        self._setup()

    def _upper_values(self, matrix: sp.spmatrix) -> np.ndarray:
        """``matrix``'s upper-triangle values at the pattern's positions."""
        M = matrix if sp.isspmatrix_csc(matrix) else sp.csc_matrix(matrix)
        if not M.has_sorted_indices:
            M = M.sorted_indices()
        if np.array_equal(M.indptr, self._indptr) and np.array_equal(
            M.indices, self._indices
        ):
            return np.ascontiguousarray(M.data[self._upper_pos], dtype=np.float64)
        # A different structure — typically a sub-pattern, when sparse
        # arithmetic dropped explicit zeros: match entries on their keys.
        col = np.repeat(np.arange(self._n), np.diff(M.indptr))
        upper = M.indices <= col
        keys = col[upper].astype(np.int64) * self._n + M.indices[upper]
        pos = np.searchsorted(self._keys, keys)
        pos_c = np.minimum(pos, len(self._keys) - 1)
        if np.any(self._keys[pos_c] != keys):
            raise ValueError("matrix has entries outside the factor's pattern")
        out = np.zeros(len(self._keys))
        out[pos_c] = M.data[upper]
        return out

    def factorize(self, matrix: sp.spmatrix) -> None:
        """Re-factorize with new values (same sparsity pattern).

        Parameters
        ----------
        matrix : sparse matrix
            New SPD matrix with the same sparsity pattern as the
            pattern matrix passed at construction.
        """
        if self._sparsax is not None:
            # sparsax factors on first use and caches the factor by value.
            self._Ax = self._upper_values(matrix)
        else:
            self._factor = self._symbolic.copy().factorize(sp.csc_matrix(matrix))

    def _solve(self, rhs: np.ndarray, mode: int) -> np.ndarray:
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        return self._sparsax.solve_np(self._Ai, self._Aj, self._Ax, rhs, mode)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve P x = rhs."""
        if self._sparsax is not None:
            return self._solve(rhs, _MODE_A)
        return self._factor.solve(rhs)

    def logdet(self) -> float:
        """Return log|P|."""
        if self._sparsax is not None:
            return float(self._sparsax.logdet_np(self._Ai, self._Aj, self._Ax, self._n))
        return self._factor.logdet()

    def sample(
        self,
        mean_term: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw from N(m, P⁻¹) where m = P⁻¹ @ mean_term.

        CHOLMOD factors a *permuted* matrix, P_perm P P_permᵀ = L Lᵀ,
        where P_perm is a fill-reducing permutation.  A correct draw
        must undo that permutation:

            m = P⁻¹ @ mean_term            (CHOLMOD solve)
            w = L⁻ᵀ z,  z ~ N(0, I)        (solve in permuted order)
            x = m + P_permᵀ w              (undo permutation)

        which gives Cov(x - m) = P_permᵀ (L Lᵀ)⁻¹ P_perm = P⁻¹.

        Parameters
        ----------
        mean_term : ndarray of shape (n,)
            The precision-weighted mean: P @ m.
        rng : numpy.random.Generator
            Random state.

        Returns
        -------
        x : ndarray of shape (n,)
            Draw from N(m, P⁻¹).
        """
        m = self.solve(mean_term)
        z = rng.standard_normal(self._n)
        if self._sparsax is not None:
            # w = L^{-T} z in the permuted ordering, then P_permᵀ w.
            return m + self._solve(self._solve(z, _MODE_LT), _MODE_PT)
        # w = L^{-T} z in the permuted ordering.
        w = self._factor.solve(z, system="Lt")
        # Undo the fill-reducing permutation: draw[perm] = w  ==  P_permᵀ w.
        perm = self._factor.get_perm()
        draw = np.empty_like(w)
        draw[perm] = w
        return m + draw


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class SpatialNormalDraw(NamedTuple):
    """Result of a spatial-normal draw.

    Attributes
    ----------
    x : ndarray of shape (n,)
        The drawn sample.
    factor : CholmodFactor
        The factorization of the precision matrix.  Can be reused
        for subsequent solves when the precision matrix has not changed.
    """

    x: np.ndarray
    factor: CholmodFactor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_spatial_normal(
    precision: sp.csr_matrix | sp.csc_matrix,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    cached_factor: CholmodFactor | None = None,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) where P is sparse SPD.

    Uses CHOLMOD (``scikit-sparse``), which is 5–9× faster than
    ``scipy.sparse.linalg.splu`` for SPD matrices.

    Parameters
    ----------
    precision : sparse matrix of shape (n, n)
        Sparse precision matrix P. Must be symmetric positive definite.
        Typically A_ρᵀ A_ρ / σ² + diag(ω) for spatial models.
    mean_term : ndarray of shape (n,)
        The precision-weighted mean: P @ m. The actual mean is
        m = P⁻¹ @ mean_term, computed via the sparse solve.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.
    cached_factor : CholmodFactor, optional
        Pre-computed factorization of precision. If None, computed
        fresh. Passing a cached factorization saves the factorization
        cost when P has not changed between calls.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with fields ``x`` (the draw) and ``factor``
        (the factorization, for potential reuse).

    Notes
    -----
    CHOLMOD factors a permuted matrix, P_perm P P_permᵀ = L Lᵀ, where
    P_perm is a fill-reducing permutation.  The draw undoes it:

    .. math::

        x = m + P_\\text{perm}^{T} L^{-T} z, \\quad z \\sim N(0, I)

    which gives Cov(x) = P_perm^{T} (L L^T)^{-1} P_perm = P^{-1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    precision_csc = sp.csc_matrix(precision)

    if cached_factor is not None:
        factor = cached_factor
        # Re-factorize with current values (reuses symbolic analysis).
        factor.factorize(precision_csc)
    else:
        factor = CholmodFactor(precision_csc)

    x = factor.sample(mean_term, rng=rng)
    return SpatialNormalDraw(x=x, factor=factor)


# ---------------------------------------------------------------------------
# Lanczos log-determinant estimator
# ---------------------------------------------------------------------------


def lanczos_logdet(
    precision: sp.spmatrix | spla.LinearOperator,
    *,
    n_probes: int = 10,
    lanczos_deg: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate log|P| for sparse SPD P via Lanczos iteration.

    Uses the Hutchinson trace estimator applied to the matrix
    logarithm:  log|P| = tr(log(P)).  For each probe vector z,
    runs a Lanczos iteration on P starting from z to build a
    tridiagonal matrix T_m, then estimates z^T log(P) z ≈ ||z||² e₁^T log(T_m) e₁.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
    n_probes : int, default 10
        Number of probe vectors.  More probes reduce variance.
    lanczos_deg : int, default 30
        Number of Lanczos iterations per probe.  Higher values
        improve accuracy at the cost of more matrix-vector products.
    rng : numpy.random.Generator, optional
        Random state.  If None, a fresh generator is created.

    Returns
    -------
    logdet : float
        Estimated log-determinant of P.

    Notes
    -----
    The cost is O(n_probes * lanczos_deg * nnz) where nnz is the
    number of non-zeros in P.  For typical spatial precision matrices
    with n > 5000, this can be significantly faster than CHOLMOD
    factorization when fill-in is high.

    The estimator is unbiased in the limit of infinite probes and
    Lanczos depth.  With n_probes=10 and lanczos_deg=30, the
    relative error is typically < 1e-3 for well-conditioned matrices.

    **Why not a generic trace estimator?**  Variance-reduced trace
    estimators (Hutch++, XTrace) are designed for :math:`\\text{tr}(A)`
    where :math:`A` is an *explicit* linear operator (you provide
    matvecs).  However, :math:`\\log|P| = \\text{tr}(\\log(P))` requires
    computing :math:`\\log(P)` as an operator, which itself needs
    Lanczos tridiagonalization per matvec.  Using a generic estimator
    here would nest Lanczos *inside* the estimator's probes, making it
    strictly more expensive.  Our implementation combines Lanczos and
    trace estimation in a single pass — this *is* the standard
    algorithm for :math:`\\text{tr}(f(A))` (Ubaru & Saad 2016).

    References
    ----------
    .. [1] Ubaru, S., & Saad, Y. (2018). Applications of Trace Estimation
       Techniques. In T. Kozubek, M. Čermák, P. Tichý, R. Blaheta, J. Šístek,
       D. Lukáš, & J. Jaroš (Eds.), High Performance Computing in Science
       and Engineering (pp. 19–33). Springer International Publishing.
       https://doi.org/10.1007/978-3-319-97136-0_2

    .. [2] Ubaru, S., & Saad, Y. (2016). Fast methods for estimating the
       Numerical rank of large matrices.

    .. [3] Bai, Z., Fahey, G., & Golub, G. (1996). Some large-scale
       matrix computation problems.
    """

    if rng is None:
        rng = np.random.default_rng()

    n = precision.shape[0]
    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
    else:
        P_op = sp.csr_matrix(precision)

    estimates = np.empty(n_probes)
    for j in range(n_probes):
        z = rng.standard_normal(n)
        z_norm = np.linalg.norm(z)
        if z_norm == 0:
            estimates[j] = 0.0
            continue
        q = z / z_norm

        # Lanczos iteration: build tridiagonal T_m
        alpha_vals = np.empty(lanczos_deg)
        beta_vals = np.empty(lanczos_deg - 1)
        Q = np.empty((n, lanczos_deg))

        Q[:, 0] = q
        r = P_op @ q
        alpha_vals[0] = float(q @ r)
        r = r - alpha_vals[0] * q

        for i in range(1, lanczos_deg):
            beta_vals[i - 1] = np.linalg.norm(r)
            if beta_vals[i - 1] < 1e-15:
                # Invariant subspace found — truncate
                alpha_vals = alpha_vals[:i]
                beta_vals = beta_vals[: i - 1] if i > 1 else beta_vals
                Q = Q[:, :i]
                break
            q_new = r / beta_vals[i - 1]
            Q[:, i] = q_new
            r = P_op @ q_new
            alpha_vals[i] = float(q_new @ r)
            # Full reorthogonalization (one pass)
            r = r - alpha_vals[i] * q_new - beta_vals[i - 1] * Q[:, i - 1]
            # Modified Gram-Schmidt against all previous vectors
            for k in range(i):
                r = r - float(Q[:, k] @ r) * Q[:, k]

        # Build tridiagonal matrix T_m
        m = len(alpha_vals)
        T = np.diag(alpha_vals[:m])
        if m > 1:
            T += np.diag(beta_vals[: m - 1], 1) + np.diag(beta_vals[: m - 1], -1)

        # z^T log(P) z ≈ ||z||² e₁^T log(T) e₁
        eigvals, eigvecs = np.linalg.eigh(T)
        log_T_diag = np.log(np.maximum(eigvals, 1e-300))
        logdet_T = float(eigvecs[0, :] @ (log_T_diag * eigvecs[0, :]))
        estimates[j] = z_norm**2 * logdet_T

    return float(np.mean(estimates))


# ---------------------------------------------------------------------------
# Preconditioned CG solve
# ---------------------------------------------------------------------------


def cg_solve(
    precision: sp.spmatrix | spla.LinearOperator,
    rhs: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: str = "jacobi",
) -> np.ndarray:
    """Solve P x = rhs for sparse SPD P via preconditioned CG.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
    rhs : ndarray of shape (n,)
        Right-hand side vector.
    tol : float, default 1e-8
        Convergence tolerance (relative residual norm).
    maxiter : int, optional
        Maximum iterations.  Defaults to 2 * n.
    preconditioner : {"jacobi", "none"}, default "jacobi"
        Preconditioner type.  "jacobi" uses M = diag(P), which
        is cheap and effective for diagonally-dominant spatial
        precision matrices.  Not available when ``precision`` is a
        ``LinearOperator`` (use "none" instead).

    Returns
    -------
    x : ndarray of shape (n,)
        Approximate solution to P x = rhs.

    Notes
    -----
    For spatial precision matrices of the form
    P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²,
    the Jacobi preconditioner M = diag(P) = I/σ² + diag(ω) is
    very effective because the diagonal dominates for typical
    spatial weights (small ρ and bounded ω).
    """
    n = precision.shape[0]
    if maxiter is None:
        maxiter = 2 * n

    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
        # No Jacobi preconditioner for LinearOperator (no diagonal access)
        M_op = None
    else:
        P_csr = sp.csr_matrix(precision)
        P_op = P_csr

        # Build preconditioner LinearOperator (only for sparse matrices)
        if preconditioner == "jacobi":
            M_diag = P_csr.diagonal()
            M_diag = np.where(np.abs(M_diag) > 1e-15, M_diag, 1.0)
            M_inv_diag = 1.0 / M_diag
            M_op = spla.LinearOperator((n, n), matvec=lambda v: M_inv_diag * v)
        elif preconditioner == "none":
            M_op = None
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

    # scipy >= 1.12 uses rtol/atol; older versions use tol.
    # Use rtol for forward compatibility.
    x, info = spla.cg(
        P_op,
        rhs,
        rtol=tol,
        maxiter=maxiter,
        M=M_op,
    )

    if info != 0:
        # CG did not converge — return best iterate anyway
        import warnings

        warnings.warn(
            f"CG did not converge after {maxiter} iterations "
            f"(info={info}). Returning best iterate.",
            RuntimeWarning,
            stacklevel=2,
        )

    return x


# ---------------------------------------------------------------------------
# Shift-invert Krylov basis for the precision matrix P(ρ)
# ---------------------------------------------------------------------------


# Safety factor applied to the estimated radius of convergence.
_SERIES_RADIUS_SAFETY = 0.6


def _series_radius(V_stack, safety: float = _SERIES_RADIUS_SAFETY) -> float:
    r"""Estimate the usable ``|Δρ|`` from the Taylor coefficients themselves.

    The series :math:`\sum_j \Delta\rho^j U_j` converges inside
    :math:`|\Delta\rho| < R` with :math:`R^{-1} = \limsup_j \|U_j\|^{1/j}`
    (Cauchy–Hadamard), so the coefficients already on the basis reveal the
    radius at no extra cost.

    The **root** test is used rather than the ratio ``‖U_j‖/‖U_{j+1}‖``: on
    real problems consecutive norms oscillate hard (ratios seen swinging
    between 0.1 and 1.9 within one basis), so any single ratio — the last
    pair especially — is meaningless.  Taking the minimum over ``j`` of
    :math:`(\|U_0\|/\|U_j\|)^{1/j}` is both stable and conservative.

    This matters because the precision series' radius depends on ``ω`` as
    well as ``ρ_c`` — ``P = diag(ω) + AᵀA/σ²`` approaches the singular
    ``AᵀA`` as ``ω → 0``, pulling the nearest singularity toward the real
    axis.  A fixed ``dmax`` cannot be safe across that range: at ``ω=0.02,
    ρ_c=0.95`` a radius of 0.4 gives a relative solve error of 2e+04, while
    the estimate below returns 0.097 and holds the error at 1e-03.
    """
    norms = np.array(
        [float(np.linalg.norm(V_stack[j])) for j in range(V_stack.shape[0])]
    )
    if norms.size < 2:
        return float("inf")
    n0 = max(norms[0], 1e-300)
    j = np.arange(1, norms.size)
    radii = (n0 / np.maximum(norms[1:], 1e-300)) ** (1.0 / j)
    return float(safety * np.min(radii))


def _chebyshev_nodes(dmax: float, n_nodes: int) -> np.ndarray:
    """``n_nodes`` Chebyshev points of the first kind on ``[-dmax, dmax]``."""
    k = np.arange(n_nodes)
    return dmax * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n_nodes))


def _fit_logdet_poly(logdet_at, dmax: float, n_nodes: int) -> np.ndarray:
    """Interpolate ``Δρ -> log|P(ρ_c+Δρ)|`` through exact values at Chebyshev nodes.

    Returns coefficients in ascending powers of ``Δρ``.  Chebyshev nodes keep
    the Vandermonde system well conditioned and spread the interpolation error
    evenly across the radius instead of piling it up at the ends.
    """
    nodes = _chebyshev_nodes(dmax, n_nodes)
    vals = np.array([float(logdet_at(float(d))) for d in nodes], dtype=np.float64)
    V = np.vander(nodes, n_nodes, increasing=True)
    return np.linalg.solve(V, vals)


class KrylovPrecisionBasis(NamedTuple):
    """Precomputed shift-invert Krylov basis for the ρ-dependent precision.

    The structural-form SAR/SEM Gibbs samplers slice over ρ against the
    **precision**

    .. math::

        P(\\rho) = \\mathrm{base} - \\rho\\,G_1 + \\rho^2 G_2,

    where ``base = I/σ² + diag(ω)`` (fixed within a slice step), ``G_1 =
    (W+W^T)/σ²``, and ``G_2 = W^T W/σ²``.  Re-centering about ``ρ_c`` is
    **exact** — no linearization:

    .. math::

        P(\\rho_c + \\Delta\\rho) = P_c - \\Delta\\rho\\, G
        + \\Delta\\rho^2 G_2, \\qquad
        G = G_1 - 2\\rho_c G_2 = \\partial P/\\partial \\rho\\big|_{\\rho_c}.

    Matching powers of ``Δρ`` in ``P(ρ) Σ_j Δρ^j U_j = rhs`` gives the
    three-term recurrence

    .. math::

        U_0 = P_c^{-1}\\mathrm{rhs}, \\quad U_1 = P_c^{-1} G U_0, \\quad
        U_j = P_c^{-1}\\left(G U_{j-1} - G_2 U_{j-2}\\right),\\; j \\ge 2,

    so factorizing ``P_c`` **once** lets every slice candidate evaluate
    ``P(\\rho)^{-1}\\mathrm{rhs}`` via a Horner sum whose only error is
    Taylor truncation at degree ``m``.

    ``log|P(ρ)|`` rides along on the same factorization.  Because a
    factorization makes ``log|P|`` available *exactly* at any ``ρ``, the
    basis stores a polynomial interpolated through exact values at
    Chebyshev nodes over ``[-dmax, dmax]`` rather than a truncated trace
    expansion — deterministic in ``ρ`` (which slice sampling requires),
    free per candidate, and accurate across the whole radius.  The
    factorization-free CG path falls back to a second-order trace
    expansion with probes frozen at build time.

    Attributes
    ----------
    rho_basis : float
        Center ``ρ_c`` at which ``P_c`` was factored.
    V_stack : ndarray, shape (m+1, n, k_rhs)
        Taylor coefficients ``U_j`` of ``P(ρ)⁻¹rhs`` about ``ρ_c``.
    degree : int
        Krylov degree ``m`` (correction terms beyond ``V_0``).
    logdet_Pc : float
        ``log|P_c|`` — the logdet at the center.
    G_matvec : callable (n,) -> (n,)
        Cached ``G = G1 − 2ρ_c G2`` matvec driver.
    solve_at_c : callable (n, k) -> (n, k)
        Cached solver ``P_c⁻¹ rhs`` (the factored CHOLMOD factor or a
        closure over CG).
    logdet_coefs : ndarray, shape (n_nodes,)
        Coefficients of ``Δρ -> log|P(ρ_c+Δρ)|`` in ascending powers of
        ``Δρ``.
    safe_dmax : float
        Largest ``|Δρ|`` the series is trustworthy over at this center,
        from :func:`_series_radius`.  Consumers clamp their configured
        ``krylov_dmax`` to this and fall back to a direct solve beyond it.
    """

    rho_basis: float
    V_stack: np.ndarray
    degree: int
    logdet_Pc: float
    G_matvec: object  # callable (n,) -> (n,)
    solve_at_c: object  # callable (n, k) -> (n, k)
    logdet_coefs: np.ndarray = np.zeros(1)
    safe_dmax: float = 0.0


def build_precision_krylov_basis(
    rho_c: float,
    base: sp.spmatrix,
    G1: sp.spmatrix,
    G2: sp.spmatrix,
    rhs: np.ndarray,
    *,
    degree: int = 12,
    cholmod_factor: CholmodFactor | None = None,
    n_probes: int = 10,
    lanczos_deg: int = 30,
    rng: np.random.Generator | None = None,
    sigma2: float = 1.0,
    dmax: float = 0.4,
    logdet_nodes: int = 4,
) -> KrylovPrecisionBasis:
    """Build a shift-invert Krylov basis for ``P(ρ) = base − ρG1 + ρ²G2``.

    Parameters
    ----------
    rho_c : float
        Center ``ρ_c`` at which ``P_c = P(ρ_c)`` is factored.
    base, G1, G2 : sparse matrices
        Components of the precision: ``P(ρ) = base − ρ·G1 + ρ²·G2``.
        For SAR (σ²=1): ``base = I + diag(ω)``, ``G1 = W+W^T``,
        ``G2 = W^T W``.  For σ²≠1, divide ``G1``, ``G2`` and the ``I``
        term of ``base`` by ``σ²``.
    rhs : ndarray, shape (n, k_rhs)
        Right-hand side(s) the slice sampler will solve for at each
        candidate ρ.  Stored as the seed ``V_0 = P_c⁻¹ rhs``.
    degree : int, default 12
        Krylov degree ``m``.
    cholmod_factor : CholmodFactor or None
        Pre-built CHOLMOD factor with the sparsity pattern of ``P``.
        When provided, ``P_c`` is factored via CHOLMOD (fast for
        moderate n) and ``logdet_Pc`` comes from CHOLMOD.  When
        ``None``, a Lanczos run estimates ``log|P_c|`` and CG is used
        for the solves (no factorization).
    n_probes, lanczos_deg : int
        Lanczos settings for the ``log|P_c|`` estimate (CG path only).
    rng : numpy.random.Generator, optional
        RNG for the Lanczos probes (CG path only).
    sigma2 : float, default 1.0
        Error variance — only used to scale ``G1``, ``G2`` and the
        identity term of ``base`` when the caller has not already done
        so.  When the caller passes already-scaled matrices, leave at
        1.0.
    dmax : float, default 0.4
        Radius ``|Δρ|`` the caller will evaluate over.  The logdet
        interpolant is fitted on ``[-dmax, dmax]``, so this must match the
        caller's ``krylov_dmax``.
    logdet_nodes : int, default 4
        Number of Chebyshev nodes (hence exact refactorizations) used to fit
        the logdet interpolant.  Four keeps the interpolation error well
        under a nat across the radius while holding the build to roughly the
        cost of five direct candidate evaluations.  Ignored on the CG path.

    Returns
    -------
    KrylovPrecisionBasis
    """
    n = base.shape[0]
    m = degree
    k_rhs = rhs.shape[1] if rhs.ndim > 1 else 1
    if rhs.ndim == 1:
        rhs = rhs.reshape(n, 1)

    # --- Assemble and factor P_c = base − ρ_c G1 + ρ_c² G2 ---
    P_c = (base - rho_c * G1 + rho_c**2 * G2).tocsc()

    # Derivative operator G = ∂P/∂ρ = G1 − 2ρ_c G2 (the matvec driver)
    G_csr = (G1 - 2.0 * rho_c * G2).tocsr()
    G2_csr = G2.tocsr()

    def _G_matvec(v):
        return G_csr @ v

    V_stack = np.empty((m + 1, n, k_rhs), dtype=np.float64)

    if cholmod_factor is not None:
        cholmod_factor.factorize(P_c)
        logdet_Pc = cholmod_factor.logdet()
        _solve_at_c = cholmod_factor.solve
    else:
        # CG path: no factorization.  Use Lanczos once for log|P_c|.
        def _solve_at_c(r):
            return iterative_solve(P_c, r, lambda_min=1e-3, lambda_max=1e6)

        logdet_Pc = lanczos_logdet(
            P_c, n_probes=n_probes, lanczos_deg=lanczos_deg, rng=rng
        )

    # --- Exact Taylor coefficients of P(ρ_c + Δρ)⁻¹ rhs ---------------------
    # P(ρ_c+Δρ) = P_c − Δρ·G + Δρ²·G2 *exactly* (the quadratic term is not a
    # remainder — it is the whole ρ² part re-centered).  Matching powers of Δρ
    # in P(ρ)·Σ_j Δρ^j U_j = rhs gives the three-term recurrence
    #     U_0 = P_c⁻¹ rhs
    #     U_1 = P_c⁻¹ (G U_0)
    #     U_j = P_c⁻¹ (G U_{j-1} − G2 U_{j-2}),   j ≥ 2
    # which costs exactly what the old linearized two-term recurrence did but
    # carries no model error — only Taylor truncation at degree m.
    V_stack[0] = _solve_at_c(rhs)
    if m >= 1:
        V_stack[1] = _solve_at_c(G_csr @ V_stack[0])
    for j in range(2, m + 1):
        V_stack[j] = _solve_at_c(G_csr @ V_stack[j - 1] - G2_csr @ V_stack[j - 2])

    # --- log|P(ρ)| across the radius ---------------------------------------
    # With a factorization in hand, log|P| is available *exactly* at any ρ for
    # the price of one refactor.  Interpolating a handful of exact values on
    # Chebyshev nodes over [−dmax, dmax] beats any truncated trace expansion:
    # it is deterministic (no Hutchinson probes, so the slice sampler's
    # shrinkage invariant holds), costs nothing per candidate, and stays
    # accurate over the whole radius rather than only near the center.
    # Clamp the requested radius to what the coefficients say is usable, then
    # fit the logdet over the radius we will actually evaluate on.
    safe_dmax = min(float(dmax), _series_radius(V_stack))

    if cholmod_factor is not None and safe_dmax > 0.0:

        def _logdet_at(d):
            Pd = (base - (rho_c + d) * G1 + (rho_c + d) ** 2 * G2).tocsc()
            cholmod_factor.factorize(Pd)
            return cholmod_factor.logdet()

        logdet_coefs = _fit_logdet_poly(_logdet_at, safe_dmax, logdet_nodes)
        # Leave the shared factor holding P_c again so ``solve_at_c`` stays valid.
        cholmod_factor.factorize(P_c)
    else:
        # CG path: no factorization to exploit, so fall back to the
        # second-order trace expansion,
        #   log|P(ρ)| ≈ log|P_c| − Δρ·tr(A) + Δρ²·[tr(B) − ½tr(A²)],
        # with A = P_c⁻¹G, B = P_c⁻¹G2.  The traces do not depend on ρ, so they
        # are estimated once here with shared probes and frozen onto the basis.
        _rng = rng if rng is not None else np.random.default_rng()
        n_tr = max(1, int(n_probes))
        Z = _rng.standard_normal((n, n_tr))
        U1 = _solve_at_c(G_csr @ Z)  # A Z
        U2 = _solve_at_c(G_csr @ U1)  # A² Z
        Vb = _solve_at_c(G2_csr @ Z)  # B Z
        tr_A = float(np.einsum("ij,ij->", Z, U1) / n_tr)
        tr_A2 = float(np.einsum("ij,ij->", Z, U2) / n_tr)
        tr_B = float(np.einsum("ij,ij->", Z, Vb) / n_tr)
        logdet_coefs = np.array([logdet_Pc, -tr_A, tr_B - 0.5 * tr_A2])

    return KrylovPrecisionBasis(
        rho_basis=rho_c,
        V_stack=V_stack,
        degree=m,
        logdet_Pc=logdet_Pc,
        G_matvec=_G_matvec,
        solve_at_c=_solve_at_c,
        logdet_coefs=logdet_coefs,
        safe_dmax=safe_dmax,
    )


def eval_precision_solve_from_basis(
    basis: KrylovPrecisionBasis,
    drho: float,
) -> np.ndarray:
    """Evaluate ``P(ρ_c + Δρ)⁻¹ rhs`` via the Horner recurrence.

    Returns an array shaped like ``basis.V_stack[0]``: ``(n, k_rhs)``.
    """
    result = basis.V_stack[basis.degree].copy()
    for j in range(basis.degree - 1, -1, -1):
        result = basis.V_stack[j] + drho * result
    return result


def eval_precision_logdet_from_basis(
    basis: KrylovPrecisionBasis,
    drho: float,
    *,
    P_at_rho: sp.spmatrix | None = None,
    cholmod_factor: CholmodFactor | None = None,
    n_probes: int = 10,
    lanczos_deg: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """Evaluate ``log|P(ρ_c + Δρ)|`` from the basis's cached interpolant.

    Horner evaluation of the polynomial fitted by
    :func:`build_precision_krylov_basis` — no solves and no fresh probes,
    so the value is a deterministic function of ``ρ``, which the slice
    sampler's shrinkage loop requires.

    Parameters
    ----------
    basis : KrylovPrecisionBasis
    drho : float
        Offset ``Δρ = ρ − ρ_c``.
    P_at_rho : sparse matrix, optional
        Unused — kept for API symmetry with the solve path.
    cholmod_factor : CholmodFactor, optional
        Unused — the basis already holds the factored solver.
    n_probes, lanczos_deg, rng :
        Unused — retained for backward compatibility.  Probe count is set
        at build time via :func:`build_precision_krylov_basis`.
    """
    coefs = basis.logdet_coefs
    acc = coefs[-1]
    for j in range(len(coefs) - 2, -1, -1):
        acc = coefs[j] + drho * acc
    return float(acc)


# ---------------------------------------------------------------------------
# Chebyshev-accelerated iterative solve
# ---------------------------------------------------------------------------


def iterative_solve(
    A: sp.spmatrix | spla.LinearOperator,
    rhs: np.ndarray,
    *,
    lambda_min: float,
    lambda_max: float,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> np.ndarray:
    r"""Solve A x = rhs for SPD A via CG (column-by-column for multi-RHS).

    For single-RHS ``(n,)``, calls :func:`scipy.sparse.linalg.cg`
    directly.  For multi-RHS ``(n, k)``, loops over columns — each
    column is an independent CG solve.  This avoids the O(nnz^{1.5})
    factorization cost entirely.

    Convergence rate (per column):

    .. math::

        \|e^{(k)}\| \leq 2 \left(\frac{\sqrt{\kappa} - 1}
        {\sqrt{\kappa} + 1}\right)^k \|e^{(0)}\|

    where :math:`\kappa = \lambda_{\max} / \lambda_{\min}`.

    Parameters
    ----------
    A : sparse matrix or LinearOperator of shape (n, n)
        Symmetric positive definite matrix, or a ``LinearOperator``
        with a ``matvec`` method.
    rhs : ndarray of shape (n,) or (n, k)
        Right-hand side vector or matrix.
    lambda_min : float
        Lower bound on the smallest eigenvalue of A (must be > 0).
        Used to compute the adaptive ``maxiter``.  Not used as a
        preconditioner (CG discovers eigenvalues adaptively).
    lambda_max : float
        Upper bound on the largest eigenvalue of A.
    tol : float, default 1e-6
        Convergence tolerance (relative residual norm).  1e-6 is
        sufficient for MCMC applications where the Krylov polynomial
        already introduces :math:`O(\Delta\rho^{m+1})` error.
    maxiter : int, optional
        Maximum iterations per column.  Defaults to ``2 * n``
        (generous enough for CG to converge on typical spatial
        systems).  The eigenvalue-based estimate is used as a
        lower bound.

    Returns
    -------
    x : ndarray of shape (n,) or (n, k)
        Approximate solution to A x = rhs.

    Notes
    -----
    Uses :func:`scipy.sparse.linalg.cg` (preconditioned conjugate
    gradient) for each column.  CG is optimal among Krylov subspace
    methods for SPD systems and converges faster than Chebyshev
    semi-iteration because it discovers eigenvalue information
    adaptively.

    For :math:`A_\rho = I - \rho W` with row-standardized :math:`W`:

    .. math::

        \lambda_{\min}(A_\rho) = \min(1 - \rho \lambda_{\max}(W),\;
        1 - \rho \lambda_{\min}(W)), \\
        \lambda_{\max}(A_\rho) = \max(1 - \rho \lambda_{\max}(W),\;
        1 - \rho \lambda_{\min}(W)).

    At :math:`\rho = 0.9`, :math:`\kappa \approx 19` and CG converges
    in ~25 iterations.  At :math:`\rho = 0.99`, :math:`\kappa \approx 199`
    and CG converges in ~100 iterations.  Each iteration is one sparse
    matvec O(nnz).

    The ``lambda_min`` and ``lambda_max`` parameters are used only to
    compute the adaptive ``maxiter``.  CG does not need eigenvalue
    bounds for convergence (unlike Chebyshev semi-iteration), but
    they are useful for setting a tight iteration cap.

    References
    ----------
    .. [1] Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate
       gradients for solving linear systems. *Journal of Research of
       the National Bureau of Standards*, 49(6), 409–436.
    .. [2] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*.
       2nd ed. SIAM. §6.7.
    """
    if lambda_min <= 0:
        raise ValueError(
            f"lambda_min must be positive (A must be SPD), got {lambda_min}"
        )
    if lambda_max < lambda_min:
        raise ValueError(
            f"lambda_max ({lambda_max}) must be >= lambda_min ({lambda_min})"
        )
    # Degenerate case: A is a scalar multiple of identity (e.g. rho=0).
    # Solution is trivially x = rhs / lambda_min.
    if lambda_max == lambda_min:
        return np.asarray(rhs, dtype=np.float64) / lambda_min

    # Accept both sparse matrices and LinearOperator
    if isinstance(A, spla.LinearOperator):
        A_op = A
    else:
        A_op = sp.csr_matrix(A)

    rhs = np.asarray(rhs, dtype=np.float64)
    single_rhs = rhs.ndim == 1
    if single_rhs:
        rhs = rhs[:, None]  # (n, 1)

    n, k = rhs.shape

    # Adaptive maxiter from condition number.
    # CG converges in at most n iterations in exact arithmetic, but
    # rounding errors can require more.  Use max(2*n, eigenvalue_cap)
    # to handle both small-n/high-κ and large-n/low-κ cases.
    kappa = lambda_max / lambda_min
    if maxiter is None:
        if kappa <= 1.001:
            eigenvalue_cap = 10
        else:
            eigenvalue_cap = int(
                np.ceil(
                    np.log(tol) / np.log((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))
                )
            )
        maxiter = max(2 * n, eigenvalue_cap)

    # Solve each column via CG
    x = np.empty_like(rhs)
    for col in range(k):
        x_col, info = spla.cg(A_op, rhs[:, col], rtol=tol, maxiter=maxiter)
        if info != 0:
            import warnings

            warnings.warn(
                f"CG did not converge for column {col} after {maxiter} "
                f"iterations (info={info}). Returning best iterate.",
                RuntimeWarning,
                stacklevel=2,
            )
        x[:, col] = x_col

    return x.reshape(n) if single_rhs else x


# ---------------------------------------------------------------------------
# Chebyshev polynomial sampler for η draw
# ---------------------------------------------------------------------------


def _gershgorin_bounds(P: sp.spmatrix) -> tuple[float, float]:
    """Compute Gershgorin eigenvalue bounds for a symmetric sparse matrix.

    For a symmetric matrix, all eigenvalues lie in the union of
    intervals [P_ii - R_i, P_ii + R_i] where R_i = Σ_{j≠i} |P_ij|.

    Parameters
    ----------
    P : sparse matrix of shape (n, n)
        Symmetric matrix.

    Returns
    -------
    lambda_min : float
        Lower bound on the smallest eigenvalue.
    lambda_max : float
        Upper bound on the largest eigenvalue.
    """
    P_csr = sp.csr_matrix(P)
    diag = P_csr.diagonal()
    # R_i = sum of |off-diagonal| in row i
    abs_P = abs(P_csr)
    row_sums = np.asarray(abs_P.sum(axis=1)).ravel()
    R = row_sums - np.abs(diag)
    lambda_min = float(np.min(diag - R))
    lambda_max = float(np.max(diag + R))
    return lambda_min, lambda_max


def _chebyshev_coeffs_inv_sqrt(
    lambda_min: float,
    lambda_max: float,
    degree: int,
) -> np.ndarray:
    """Compute Chebyshev coefficients for f(x) = x^{-1/2} on [a, b].

    Maps [a, b] → [-1, 1] and computes the Chebyshev series
    coefficients via the DCT of f evaluated at Chebyshev nodes.

    Parameters
    ----------
    lambda_min : float
        Lower bound of the eigenvalue interval (must be > 0).
    lambda_max : float
        Upper bound of the eigenvalue interval.
    degree : int
        Polynomial degree.

    Returns
    -------
    coeffs : ndarray of shape (degree + 1,)
        Chebyshev coefficients c_0, c_1, ..., c_degree.
    """
    if lambda_min <= 0:
        raise ValueError(
            f"lambda_min must be positive for x^{{-1/2}}, got {lambda_min}"
        )

    m = degree + 1
    # Chebyshev nodes on [-1, 1]
    k = np.arange(1, m + 1)
    nodes = np.cos((2 * k - 1) * np.pi / (2 * m))

    # Map nodes from [-1, 1] to [lambda_min, lambda_max]
    # x ∈ [-1, 1] → λ = (a+b)/2 + (b-a)/2 * x
    mid = 0.5 * (lambda_min + lambda_max)
    half_range = 0.5 * (lambda_max - lambda_min)
    lam_nodes = mid + half_range * nodes

    # Evaluate f(λ) = λ^{-1/2} at the mapped nodes
    f_vals = lam_nodes ** (-0.5)

    # Chebyshev coefficients via DCT-I
    coeffs = np.zeros(m, dtype=np.float64)
    for j in range(m):
        scale = 2.0 / m if j > 0 else 1.0 / m
        coeffs[j] = scale * np.sum(f_vals * np.cos(j * (2 * k - 1) * np.pi / (2 * m)))

    return coeffs


def chebyshev_sample(
    precision: sp.spmatrix | spla.LinearOperator,
    mean_term: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    degree: int = 30,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    cg_tol: float = 1e-8,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) via Chebyshev polynomial approximation of P⁻¹ᐟ².

    Avoids the O(nnz^{1.5}) sparse factorization cost by:
    1. Computing the conditional mean m = P⁻¹ rhs via CG.
    2. Approximating P⁻¹ᐟ² z (z ~ N(0, I)) via a Chebyshev polynomial
       of P, evaluated via Clenshaw's recurrence.

    Parameters
    ----------
    precision : sparse matrix or LinearOperator of shape (n, n)
        Sparse SPD precision matrix P, or a ``LinearOperator`` with
        a ``matvec`` method.  Passing a ``LinearOperator`` avoids
        constructing the full N×N sparse matrix when the matvec
        can be computed more efficiently (e.g., via Kronecker structure).
        When passing a ``LinearOperator``, ``lambda_min`` and
        ``lambda_max`` must be provided explicitly since Gershgorin
        bounds require diagonal access.
    mean_term : ndarray of shape (n,)
        The precision-weighted mean: P @ m. The actual mean is
        m = P⁻¹ @ mean_term, computed via CG.
    rng : numpy.random.Generator, optional
        Random state. If None, a fresh generator is created.
    degree : int, default 30
        Degree of the Chebyshev polynomial approximation for P⁻¹ᐟ².
        Higher values improve accuracy at the cost of more matrix-vector
        products.  20–40 is usually sufficient for well-conditioned
        spatial precision matrices.
    lambda_min : float, optional
        Lower eigenvalue bound. If None, computed via Gershgorin circles.
    lambda_max : float, optional
        Upper eigenvalue bound. If None, computed via Gershgorin circles.
    cg_tol : float, default 1e-8
        CG convergence tolerance for the mean solve.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with fields ``x`` (the draw) and ``factor``
        (None — no factorization is available for reuse).

    Notes
    -----
    The sampling formula is:

    .. math::

        \\eta = m + \\hat{P}^{-1/2} z, \\quad z \\sim N(0, I)

    where :math:`\\hat{P}^{-1/2}` is a Chebyshev polynomial approximation
    of :math:`P^{-1/2}`.  The approximation error in the covariance is:

    .. math::

        \\text{Cov}(\\eta) = P^{-1} + O(\\|P^{-1/2} - \\hat{P}^{-1/2}\\|^2)

    For well-conditioned matrices (condition number < 100), degree 30
    typically gives relative covariance error < 1e-4.

    The cost is O((degree + t_cg) * nnz) where t_cg is the CG iteration
    count for the mean solve.  This is typically much cheaper than
    O(nnz^{1.5}) for large n with high fill-in.

    References
    ----------
    .. [1] Fox, C., & Parker, A. (2014). Convergence in Variance of Chebyshev
       Accelerated Gibbs Samplers. SIAM Journal on Scientific Computing, 36(1),
       A124–A147. https://doi.org/10.1137/120900940

    .. [2] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*.
       2nd ed. SIAM. (Background on Chebyshev polynomial iteration.)
    """

    if rng is None:
        rng = np.random.default_rng()

    n = precision.shape[0]

    # Accept both sparse matrices and LinearOperator
    if isinstance(precision, spla.LinearOperator):
        P_op = precision
        P_csr = None  # Not available for LinearOperator
    else:
        P_csr = sp.csr_matrix(precision)
        P_op = P_csr

    # --- Step 1: Compute mean via CG ---
    m = cg_solve(P_op, mean_term, tol=cg_tol)

    # --- Step 2: Compute eigenvalue bounds ---
    if lambda_min is None or lambda_max is None:
        if P_csr is not None:
            g_min, g_max = _gershgorin_bounds(P_csr)
            lambda_min = g_min if lambda_min is None else lambda_min
            lambda_max = g_max if lambda_max is None else lambda_max
        else:
            raise ValueError(
                "When precision is a LinearOperator, lambda_min and lambda_max "
                "must be provided explicitly (Gershgorin bounds require "
                "diagonal access)."
            )

    # Safety: ensure lambda_min > 0 (P is SPD)
    if lambda_min <= 0:
        lambda_min = 1e-6

    # Safety: ensure lambda_max > lambda_min (avoid division by zero)
    if lambda_max <= lambda_min:
        lambda_max = lambda_min + 1.0

    # --- Step 3: Chebyshev approximation of P^{-1/2} z ---
    coeffs = _chebyshev_coeffs_inv_sqrt(lambda_min, lambda_max, degree)

    z = rng.standard_normal(n)

    # Evaluate f(P) z = Σ_{j=0}^{degree} c_j T_j(P_mapped) z
    # where P_mapped = (2P - (λ_max + λ_min)I) / (λ_max - λ_min)
    # maps eigenvalues from [λ_min, λ_max] to [-1, 1].
    #
    # Use the three-term Chebyshev recurrence on the *unmapped* matrix P
    # (Saad 2003, §12.3), which is numerically stable:
    #   d = (λ_max - λ_min) / 2  (half-range)
    #   c = (λ_max + λ_min) / 2  (midpoint)
    #   T_0(P_m) z = z
    #   T_1(P_m) z = (P z - c z) / d
    #   T_j(P_m) z = 2 (P z_j - c z_j) / d - z_{j-1}
    #
    # This avoids forming P_mapped explicitly and keeps the
    # intermediate vectors bounded.
    d = 0.5 * (lambda_max - lambda_min)
    c = 0.5 * (lambda_max + lambda_min)
    inv_d = 1.0 / d

    # Forward recurrence for T_j(P_mapped) z
    y_prev = z.copy()  # T_0(P_m) z = z
    y_curr = (P_op @ z - c * z) * inv_d  # T_1(P_m) z = (Pz - cz) / d

    v = coeffs[0] * y_prev + coeffs[1] * y_curr

    for j in range(2, degree + 1):
        y_new = 2.0 * (P_op @ y_curr - c * y_curr) * inv_d - y_prev
        v += coeffs[j] * y_new
        y_prev = y_curr
        y_curr = y_new

    x = m + v

    return SpatialNormalDraw(x=x, factor=None)


# ---------------------------------------------------------------------------
# JAX-accelerated variants (dense matvec + vmap)
# ---------------------------------------------------------------------------
# These functions use JAX dense matrix-vector products and jax.vmap
# to batch sequential operations into single XLA kernels.  They are
# significantly faster than the scipy sparse versions for n ≤ ~5000
# where dense matvec is competitive with scipy CSR matvec.
#
# All JAX imports are lazy — these functions are only called when
# gibbs_method="jax_dense" is selected, and JAX availability is
# checked at model construction time.
# ---------------------------------------------------------------------------


from ._jax_utils import (
    check_jax_available as _check_jax_available,
)


def _jax_lanczos_probe(P_dense, z_raw, lanczos_deg):
    """Single Lanczos probe: estimate z^T log(P) z.

    Runs a Lanczos iteration on P starting from z/||z||, builds
    tridiagonal matrix T_m, and returns ||z||^2 * e_1^T log(T_m) e_1.

    This is the inner function designed to be vmapped over probes.

    Notes
    -----
    Reorthogonalization uses the full Q matrix (n × lanczos_deg)
    rather than a dynamic slice Q[:, :i], because JAX's lax.scan
    requires static slice sizes.  Columns beyond the current iteration
    are zero, so projecting them out has no effect.
    """
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]
    z_norm = jnp.linalg.norm(z_raw)
    q = z_raw / jnp.where(z_norm < 1e-15, 1.0, z_norm)

    # Pre-allocate Q matrix (n × lanczos_deg) and coefficient arrays
    Q = jnp.zeros((n, lanczos_deg))
    Q = Q.at[:, 0].set(q)
    alphas = jnp.zeros(lanczos_deg)
    betas = jnp.zeros(lanczos_deg - 1)

    # First step
    r = P_dense @ q
    alpha0 = jnp.dot(q, r)
    r = r - alpha0 * q
    alphas = alphas.at[0].set(alpha0)

    # Lanczos iteration via lax.scan
    def body(carry, i):
        Q, alphas, betas, r = carry
        beta = jnp.linalg.norm(r)
        q_new = r / jnp.where(beta < 1e-15, 1.0, beta)
        Q = Q.at[:, i].set(q_new)
        r = P_dense @ q_new
        alpha = jnp.dot(q_new, r)
        # Three-term recurrence
        r = r - alpha * q_new - beta * Q[:, i - 1]
        # Full reorthogonalization against all Q columns.
        # Columns beyond i are zero, so this is equivalent to
        # projecting out Q[:, :i+1] but with a static slice size.
        r = r - Q @ (Q.T @ r)
        alphas = alphas.at[i].set(alpha)
        betas = betas.at[i - 1].set(beta)
        return (Q, alphas, betas, r), None

    (Q, alphas, betas, _), _ = jax.lax.scan(
        body,
        (Q, alphas, betas, r),
        jnp.arange(1, lanczos_deg),
    )

    # Build tridiagonal T_m and compute e_1^T log(T_m) e_1
    T = jnp.diag(alphas) + jnp.diag(betas, 1) + jnp.diag(betas, -1)
    eigvals, eigvecs = jnp.linalg.eigh(T)
    log_T_diag = jnp.log(jnp.maximum(eigvals, 1e-300))
    logdet_T = jnp.dot(eigvecs[0, :], log_T_diag * eigvecs[0, :])
    return z_norm**2 * logdet_T


def jax_lanczos_logdet(
    P_dense: "jnp.ndarray",  # noqa: F821
    *,
    key: "jax.random.PRNGKey",  # noqa: F821
    n_probes: int = 10,
    lanczos_deg: int = 30,
) -> float:
    """Estimate log|P| for dense SPD P via JAX-accelerated Lanczos.

    Same algorithm as :func:`lanczos_logdet` but uses JAX dense matvec
    and :func:`jax.vmap` over probe vectors to batch all probes into
    a single XLA kernel.  This gives a 3–4× speedup over the numpy
    implementation for n ≤ 2000.

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.  Must be float64 for numerical
        stability — float32 causes NaN/inf in the Lanczos iteration.
    key : jax.random.PRNGKey
        JAX random key for probe vector generation.
    n_probes : int, default 10
        Number of probe vectors.
    lanczos_deg : int, default 30
        Number of Lanczos iterations per probe.

    Returns
    -------
    logdet : float or jax.numpy.ndarray
        Estimated log-determinant of P.  Returns a Python float
        when called outside ``jax.jit``, or a JAX array when called
        inside JIT (the ``float()`` conversion is deferred to the caller).

    Notes
    -----
    Requires ``jax_enable_x64=True``.  Without float64, the Lanczos
    iteration accumulates roundoff errors that cause NaN/inf.

    The vmap over probes batches all ``n_probes`` Lanczos iterations
    into a single XLA kernel, eliminating Python-loop overhead and
    enabling XLA fusion across the matvec + orthogonalization steps.

    """
    _check_jax_available()
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]

    # Generate probe vectors via vmap (jax.random.normal doesn't accept
    # key arrays directly — must vmap over individual keys)
    keys = jax.random.split(key, n_probes)
    z_all = jax.vmap(lambda k: jax.random.normal(k, shape=(n,)))(keys)

    # vmap over probes — each probe runs an independent Lanczos iteration
    estimates = jax.vmap(lambda z: _jax_lanczos_probe(P_dense, z, lanczos_deg))(z_all)
    return jnp.mean(estimates)


def jax_cg_solve(
    P_dense: "jnp.ndarray",  # noqa: F821
    rhs: "jnp.ndarray",  # noqa: F821
    M_inv_diag: "jnp.ndarray | None" = None,  # noqa: F821
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> "jnp.ndarray":  # noqa: F821
    """Solve P x = rhs for dense SPD P via JAX preconditioned CG.

    Uses ``jax.scipy.sparse.linalg.cg`` with a dense matvec
    ``lambda v: P_dense @ v``.  Machine-precision accuracy (1e-16
    relative error).

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.
    rhs : jax.numpy.ndarray of shape (n,)
        Right-hand side vector.
    M_inv_diag : jax.numpy.ndarray of shape (n,), optional
        Diagonal of the inverse Jacobi preconditioner M⁻¹.
        If None, no preconditioner is used.
    tol : float, default 1e-8
        Convergence tolerance (relative residual norm).
    maxiter : int, optional
        Maximum iterations.  Defaults to 2 * n.

    Returns
    -------
    x : jax.numpy.ndarray of shape (n,)
        Solution to P x = rhs.

    Notes
    -----
    Uses ``jax.scipy.sparse.linalg.cg`` (built-in) because:
    - Machine-precision accuracy (1e-16)
    - Simple API with no extra optional dependency

    Benchmarks (Apple M1, CPU):
        n=500:  scipy=0.56ms, JAX=0.29ms (1.9×)
        n=1000: scipy=1.28ms, JAX=0.69ms (1.9×)
        n=2000: scipy=2.31ms, JAX=2.33ms (1.0×)
        n=5000: scipy=5.81ms, JAX=18.2ms (0.3× — dense loses)
    """
    _check_jax_available()
    import jax
    import jax.scipy.sparse.linalg

    n = P_dense.shape[0]
    if maxiter is None:
        maxiter = 2 * n

    # JAX cg accepts A as a callable matvec or a dense array.
    # Passing the dense array directly is simplest and fastest.
    if M_inv_diag is not None:
        # Jacobi preconditioner: M^{-1} applied as element-wise multiply
        def M_func(v):
            return M_inv_diag * v
    else:
        M_func = None

    x, info = jax.scipy.sparse.linalg.cg(
        P_dense,
        rhs,
        tol=tol,
        maxiter=maxiter,
        M=M_func,
    )
    return x


def jax_chebyshev_sample(
    P_dense: "jnp.ndarray",  # noqa: F821
    mean_term: "jnp.ndarray",  # noqa: F821
    *,
    key: "jax.random.PRNGKey",  # noqa: F821
    degree: int = 30,
    n_draws: int = 1,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> SpatialNormalDraw:
    """Draw from N(m, P⁻¹) via JAX Chebyshev polynomial with vmap over draws.

    Same algorithm as :func:`chebyshev_sample` but uses JAX dense matvec
    and :func:`jax.vmap` over draws to batch the Clenshaw recurrence
    across multiple draws.  For a single draw, JAX dispatch overhead
    makes this slightly slower than scipy; for 10 draws, vmap gives
    a 3.6× per-draw speedup.

    Parameters
    ----------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense SPD precision matrix.
    mean_term : jax.numpy.ndarray of shape (n,)
        Precision-weighted mean: P @ m.
    key : jax.random.PRNGKey
        JAX random key for z ~ N(0, I) generation.
    degree : int, default 30
        Chebyshev polynomial degree for P⁻¹ᐟ² approximation.
    n_draws : int, default 1
        Number of independent draws.  vmap batches all draws into
        a single XLA kernel, so 10 draws costs barely more than 4.
    lambda_min : float, optional
        Lower eigenvalue bound.  If None, computed via Gershgorin.
    lambda_max : float, optional
        Upper eigenvalue bound.  If None, computed via Gershgorin.

    Returns
    -------
    SpatialNormalDraw
        Named tuple with ``x`` (the first draw, shape (n,)) and
        ``factor=None`` (no factorization available for reuse).

    Notes
    -----
    Requires ``jax_enable_x64=True``.

    The ``n_draws`` parameter must be a compile-time constant (not
    traced by JAX).  This is because ``jax.vmap`` needs to know the
    batch size at trace time.
    """
    _check_jax_available()
    import jax
    import jax.numpy as jnp

    n = P_dense.shape[0]

    # --- Step 1: Compute mean via JAX CG ---
    M_inv_diag = 1.0 / jnp.where(
        jnp.abs(jnp.diag(P_dense)) > 1e-15,
        jnp.diag(P_dense),
        1.0,
    )
    m = jax_cg_solve(P_dense, mean_term, M_inv_diag)

    # --- Step 2: Compute eigenvalue bounds ---
    if lambda_min is None or lambda_max is None:
        diag = jnp.diag(P_dense)
        abs_P = jnp.abs(P_dense)
        row_sums = jnp.sum(abs_P, axis=1)
        R = row_sums - jnp.abs(diag)
        g_min = jnp.min(diag - R)
        g_max = jnp.max(diag + R)
        lambda_min = g_min if lambda_min is None else lambda_min
        lambda_max = g_max if lambda_max is None else lambda_max

    # Safety: ensure bounds are valid (use jnp.maximum for JIT compatibility)
    lambda_min = jnp.maximum(lambda_min, 1e-6)
    lambda_max = jnp.maximum(lambda_max, lambda_min + 1.0)

    # --- Step 3: Chebyshev coefficients for x^{-1/2} on [a, b] ---
    # Inline computation (JIT-compatible, no Python-level if/float)
    md = degree + 1
    k = jnp.arange(1, md + 1, dtype=jnp.float64)
    nodes = jnp.cos((2 * k - 1) * jnp.pi / (2 * md))
    mid = 0.5 * (lambda_min + lambda_max)
    half_range = 0.5 * (lambda_max - lambda_min)
    lam_nodes = mid + half_range * nodes
    f_vals = lam_nodes ** (-0.5)
    coeffs = jnp.zeros(md, dtype=jnp.float64)
    for j in range(md):
        scale = 2.0 / md if j > 0 else 1.0 / md
        coeffs = coeffs.at[j].set(
            scale * jnp.sum(f_vals * jnp.cos(j * (2 * k - 1) * jnp.pi / (2 * md)))
        )

    # --- Step 4: Generate z vectors and compute P^{-1/2} z via Clenshaw ---
    d = 0.5 * (lambda_max - lambda_min)
    c = 0.5 * (lambda_max + lambda_min)
    inv_d = 1.0 / d

    keys = jax.random.split(key, n_draws)
    z_all = jax.vmap(lambda k: jax.random.normal(k, shape=(n,)))(keys)  # (n_draws, n)

    def _single_chebyshev_draw(z):
        """Clenshaw recurrence for one draw."""
        y_prev = z
        y_curr = (P_dense @ z - c * z) * inv_d
        v = coeffs[0] * y_prev + coeffs[1] * y_curr
        for j in range(2, degree + 1):
            y_new = 2.0 * (P_dense @ y_curr - c * y_curr) * inv_d - y_prev
            v = v + coeffs[j] * y_new
            y_prev = y_curr
            y_curr = y_new
        return m + v

    # vmap over draws
    draws = jax.vmap(_single_chebyshev_draw)(z_all)  # (n_draws, n)

    # Return first draw (Gibbs sampler only needs one)
    x = draws[0]
    return SpatialNormalDraw(x=np.asarray(x), factor=None)


def jax_build_P_dense(
    rho: float,
    sigma2: float,
    omega: "jnp.ndarray",  # noqa: F821
    W_sym_dense: "jnp.ndarray",  # noqa: F821
    WtW_dense: "jnp.ndarray",  # noqa: F821
) -> "jnp.ndarray":  # noqa: F821
    """Build the dense precision matrix P from precomputed components.

    Constructs P = I/σ² + diag(ω) - ρ(W+W^T)/σ² + ρ²W^TW/σ²
    using precomputed dense W components.  This is ~3× faster than
    building the scipy sparse version because dense arithmetic avoids
    sparse format overhead.

    Parameters
    ----------
    rho : float
        Spatial autoregressive parameter.
    sigma2 : float
        Residual variance.
    omega : jax.numpy.ndarray of shape (n,)
        PG auxiliary variables (diagonal of precision).
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T), precomputed once at model setup.
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W, precomputed once at model setup.

    Returns
    -------
    P_dense : jax.numpy.ndarray of shape (n, n)
        Dense precision matrix.
    """
    _check_jax_available()
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2
    P = (
        jnp.diag(jnp.ones(n) * inv_s2 + omega)
        - rho * W_sym_dense * inv_s2
        + rho**2 * WtW_dense * inv_s2
    )
    return P


def _jax_log_density_core(
    rho,
    sigma2,
    omega,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    Xbeta_over_s2,
    WtXbeta_over_s2,
    kappa,
    key,
    n_probes,
    lanczos_deg,
    cg_tol,
    cg_maxiter,
):
    """Core log-density computation, fully in JAX (no Python dispatch).

    This function is designed to be wrapped with ``jax.jit`` so that
    build P → Lanczos logdet → CG solve → quadratic form are fused
    into a single XLA kernel, eliminating Python dispatch overhead.

    Returns a JAX scalar (not a Python float) so it can be JIT-compiled.
    """
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2

    # Build P
    P_diag = jnp.ones(n) * inv_s2 + omega
    P = jnp.diag(P_diag) - rho * W_sym_dense * inv_s2 + rho**2 * WtW_dense * inv_s2

    # RHS
    rhs = Xbeta_over_s2 - rho * WtXbeta_over_s2 + kappa

    # Jacobi preconditioner
    M_inv_diag = 1.0 / jnp.where(jnp.abs(P_diag) > 1e-15, P_diag, 1.0)

    # Lanczos logdet of P
    log_det_P = jax_lanczos_logdet(
        P, key=key, n_probes=n_probes, lanczos_deg=lanczos_deg
    )

    # CG solve P m = rhs
    m = jax_cg_solve(P, rhs, M_inv_diag, tol=cg_tol, maxiter=cg_maxiter)

    # Quadratic form
    quad = rhs @ m

    # log|I - rho*W| via generic JAX-native logdet callable
    logdet_W = logdet_jax(rho)

    # Final log-density
    return logdet_W - 0.5 * log_det_P + 0.5 * quad


def _jax_log_density_core_exact(
    rho,
    sigma2,
    omega,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    Xbeta_over_s2,
    WtXbeta_over_s2,
    kappa,
):
    """Exact log-density computation using dense Cholesky (no stochastic approx).

    This is a deterministic variant of :func:`_jax_log_density_core` that
    replaces Lanczos logdet with dense Cholesky and CG solve with
    ``jax.scipy.linalg.cho_solve``.  It is **much faster** for small
    matrices (n \u2264 ~500) because it avoids the overhead of stochastic
    trace estimation and iterative solvers.

    Use this for mode-finding during burn-in, where exactness and speed
    matter more than O(n\u00b3) scaling.

    Returns a JAX scalar.
    """
    import jax
    import jax.numpy as jnp

    n = omega.shape[0]
    inv_s2 = 1.0 / sigma2

    # Build P
    P_diag = jnp.ones(n) * inv_s2 + omega
    P = jnp.diag(P_diag) - rho * W_sym_dense * inv_s2 + rho**2 * WtW_dense * inv_s2

    # RHS
    rhs = Xbeta_over_s2 - rho * WtXbeta_over_s2 + kappa

    # Exact log|P| via dense Cholesky
    L = jnp.linalg.cholesky(P)
    log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    # Exact solve via Cholesky
    m = jax.scipy.linalg.cho_solve((L, True), rhs)
    quad = rhs @ m

    # log|I - rho*W| via generic JAX-native logdet callable
    logdet_W = logdet_jax(rho)

    return logdet_W - 0.5 * log_det_P + 0.5 * quad


# ---------------------------------------------------------------------------
# JAX-native shift-invert Krylov basis for the precision P(ρ)  (sparsax)
# ---------------------------------------------------------------------------


def _sparsax_factor_ops_available() -> bool:
    """Return ``True`` when sparsax exposes the factor-reuse primitives."""
    try:
        import sparsax

        return (
            hasattr(sparsax, "factor")
            and hasattr(sparsax, "solve_factor")
            and hasattr(sparsax, "logdet_factor")
        )
    except ImportError:
        return False


def build_precision_krylov_basis_jax(
    Ai,
    Aj,
    Ax_c,
    G_vals,
    G2_vals,
    rhs_seed,
    *,
    n: int,
    degree: int = 12,
    dmax: float = 0.4,
    logdet_nodes: int = 4,
):
    """Build a JAX-native shift-invert Krylov basis for ``P(ρ_c)`` via sparsax.

    The JAX analog of :func:`build_precision_krylov_basis`.  Factors ``P_c``
    **once** via :func:`sparsax.factor`, then runs the exact three-term
    coefficient recurrence ``U_j = P_c⁻¹(G U_{j-1} − G2 U_{j-2})`` as a
    :func:`jax.lax.scan` of :func:`sparsax.solve_factor` calls against the
    held factor — zero refactors across the recurrence.

    Parameters
    ----------
    Ai, Aj : int32 arrays, shape (nnz,)
        COO pattern (upper triangle).
    Ax_c : float64 array, shape (nnz,)
        COO values of ``P_c = P(ρ_c)``.
    G_vals : float64 array, shape (nnz,)
        COO values of ``G = G1 − 2ρ_c G2`` (the ρ-derivative of P), at the
        same pattern positions as ``Ax_c``.
    G2_vals : float64 array, shape (nnz,)
        COO values of ``G2 = WᵀW/σ²`` at the same pattern positions.  This is
        the exact ``Δρ²`` term of the re-centered precision; omitting it would
        leave a model error that no Krylov degree can remove.
    rhs_seed : float64 array, shape (n, k_rhs)
        Right-hand side(s) the slice sampler will solve for.  Seed with the
        ρ-*independent* columns (e.g. ``[κ, X, WtX]``) so the ρ-dependent RHS
        is reconstructed as a linear combination of Horner evaluations.
    n : int
        Matrix dimension (static Python int).
    degree : int, default 12
        Krylov degree ``m``.
    dmax : float, default 0.4
        Requested evaluation radius.  The returned ``safe_dmax`` is this
        clamped to what the series can actually support.
    logdet_nodes : int, default 4
        Chebyshev nodes (hence exact factorizations) behind the logdet
        interpolant.

    Returns
    -------
    token : sparsax Factor
        The held numeric factor.
    V_stack : float64 array, shape (m+1, n, k_rhs)
        Taylor coefficients ``U_j`` of ``P(ρ)⁻¹rhs_seed`` about ``ρ_c``.
    logdet_coefs : float64 array, shape (logdet_nodes,)
        ``Δρ -> log|P(ρ_c+Δρ)|`` in ascending powers of ``Δρ``.
    safe_dmax : float64 scalar
        Largest usable ``|Δρ|``; candidates beyond it must take a direct
        solve.

    Notes
    -----
    Requires ``sparsax`` with the factor-reuse primitives (``factor``,
    ``solve_factor``, ``logdet_factor``).  Use
    :func:`_sparsax_factor_ops_available` to gate.

    The whole function is ``jax.jit``-compatible and ``vmap``-able over
    ``Ax_c`` (one factor per batch element).  Every solve is issued against a
    single factor — ``sparsax.factorization_count()`` increments by exactly 1.
    """
    import jax
    import jax.numpy as jnp
    import sparsax

    token = sparsax.factor(Ai, Aj, Ax_c, n)
    V0 = sparsax.solve_factor(token, rhs_seed)

    def _spmv(vals, V):
        """Symmetric COO matvec against a single stored triangle.

        ``G`` and ``G2`` are symmetric (``W+Wᵀ`` and ``WᵀW``) and the pattern
        stores each off-diagonal entry **once**, in the upper triangle, since
        that is what sparsax reads.  A plain scatter over the stored entries
        would therefore compute only half the product; every off-diagonal
        value has to be applied in both directions.
        """
        out = jnp.zeros((n, V.shape[1]), dtype=V.dtype)
        out = out.at[Ai].add(vals[:, None] * V[Aj])
        off_diag = (Ai != Aj)[:, None]
        return out.at[Aj].add(jnp.where(off_diag, vals[:, None] * V[Ai], 0.0))

    # Exact Taylor coefficients of P(ρ_c+Δρ)⁻¹rhs.  Because
    # P(ρ_c+Δρ) = P_c − Δρ·G + Δρ²·G2 exactly, the coefficients satisfy the
    # three-term recurrence U_j = P_c⁻¹(G U_{j-1} − G2 U_{j-2}); dropping the
    # G2 term would leave a model error no degree could remove.  scan carries
    # (U_{j-1}, U_{j-2}) and avoids dynamic-index reads that break under
    # eqx.filter_jit + vmap.
    V1 = sparsax.solve_factor(token, _spmv(G_vals, V0))

    def _scan_step(carry, _):
        V_prev, V_prev2 = carry
        V_new = sparsax.solve_factor(
            token, _spmv(G_vals, V_prev) - _spmv(G2_vals, V_prev2)
        )
        return (V_new, V_prev), V_new

    # V_0, V_1 are seeded; scan produces V_2..V_m as the collected outputs.
    _, V_tail = jax.lax.scan(_scan_step, (V1, V0), xs=None, length=max(degree - 1, 0))
    if degree == 0:
        V_stack = V0[None]
    elif degree == 1:
        V_stack = jnp.stack([V0, V1], axis=0)
    else:
        V_stack = jnp.concatenate([V0[None], V1[None], V_tail], axis=0)

    # Usable radius, read off the coefficients themselves (root test — see
    # the numpy :func:`_series_radius` for why the ratio test is unusable).
    norms = jnp.sqrt(jnp.sum(V_stack.reshape(V_stack.shape[0], -1) ** 2, axis=1))
    n0 = jnp.maximum(norms[0], 1e-300)
    jj = jnp.arange(1, norms.shape[0], dtype=jnp.float64)
    safe_dmax = jnp.minimum(
        jnp.float64(dmax),
        _SERIES_RADIUS_SAFETY
        * jnp.min((n0 / jnp.maximum(norms[1:], 1e-300)) ** (1.0 / jj)),
    )

    # log|P(ρ)| by interpolation through *exact* node values.  P(ρ_c+Δρ)
    # shares P_c's pattern — its COO values are just Ax_c − Δρ·G + Δρ²·G2 — so
    # each node is one exact factorization: nothing stochastic, and accurate
    # across the whole radius.
    #
    # A `selinv`-based expansion was tried instead, to get the logdet off the
    # single held factor at zero extra factorizations.  It is *correct* (tr(A)
    # and tr(B) come out exact) but far slower: at n=900 the selected inverse
    # costs 3.99 ms against 0.37 ms for a factorization and 0.34 ms for an
    # entire direct candidate — ~12 candidates' worth of work to avoid 4
    # factorizations.  Nodes win on both cost and accuracy (0.03 vs 0.23 nats).
    cos_factors = jnp.asarray(_chebyshev_nodes(1.0, logdet_nodes), dtype=jnp.float64)
    nodes = safe_dmax * cos_factors

    def _node_logdet(d):
        return sparsax.logdet(Ai, Aj, Ax_c - d * G_vals + d * d * G2_vals, n)

    node_vals = jnp.stack([_node_logdet(nodes[i]) for i in range(logdet_nodes)])
    vander = jnp.vander(nodes, logdet_nodes, increasing=True)
    logdet_coefs = jnp.linalg.solve(vander, node_vals)

    return token, V_stack, logdet_coefs, safe_dmax


def eval_precision_solve_from_basis_jax(V_stack, drho):
    """Evaluate ``P(ρ_c + Δρ)⁻¹ rhs`` via the Horner recurrence (pure JAX)."""

    degree = V_stack.shape[0] - 1
    result = V_stack[degree]
    for j in range(degree - 1, -1, -1):
        result = V_stack[j] + drho * result
    return result


def eval_precision_logdet_from_basis_jax(logdet_coefs, drho):
    """Evaluate ``log|P(ρ_c + Δρ)|`` from the basis's cached interpolant.

    Horner over the coefficients fitted by
    :func:`build_precision_krylov_basis_jax` through exact node logdets — no
    solves, no probes, and deterministic in ρ as slice sampling requires.

    Parameters
    ----------
    logdet_coefs : float64 array
        Ascending-power coefficients from the build.
    drho : float64 scalar
        Offset ``Δρ = ρ − ρ_c``.
    """
    acc = logdet_coefs[-1]
    for j in range(logdet_coefs.shape[0] - 2, -1, -1):
        acc = logdet_coefs[j] + drho * acc
    return acc
