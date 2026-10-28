"""Sparse factorization of spatial precision matrices, and the solves around it.

:class:`CholmodFactor` factors an SPD precision with CHOLMOD (through
``sparsax`` when installed, else ``scikit-sparse``), which is 5–9× faster than
``scipy.sparse.linalg.splu`` for SPD matrices, and draws from :math:`N(m,
P^{-1})` by undoing its fill-reducing permutation:

.. math::
    P_{perm} P P_{perm}^\top = LL^\top, \\qquad
    x = m + P_{perm}^\top L^{-\top} z, \\quad z \\sim N(0, I).

:func:`cg_solve` is the preconditioned conjugate-gradient solve the flow
samplers use, and :func:`_series_radius` the safe radius of a Krylov series in
:math:`\\Delta\rho`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import CholeskyFactor as _SkCholeskyFactor
from sksparse.cholmod import CholmodNotPositiveDefiniteError

# ---------------------------------------------------------------------------
# CHOLMOD factorization wrapper
# ---------------------------------------------------------------------------


# CHOLMOD solve modes (cholmod.h): A x = b, L' x = b, and x = P' b.
_MODE_A, _MODE_LT, _MODE_PT = 0, 5, 8


class NotPositiveDefiniteError(np.linalg.LinAlgError):
    """A matrix handed to :class:`CholmodFactor` is not positive definite.

    Raised by whichever call first factorizes: ``factorize`` on the
    scikit-sparse path, the first ``solve``/``logdet``/``sample`` on the
    sparsax path, which factors on first use.
    """


#: What a sparse factorization of ``I − ρW`` (or of a precision built from
#: it) raises when the matrix is singular or indefinite, as at the edge of a
#: parameter's support: :class:`NotPositiveDefiniteError` and numpy's errors
#: are ``LinAlgError``; sparsax's LU and SuperLU raise ``RuntimeError``.
FACTORIZATION_ERRORS = (np.linalg.LinAlgError, RuntimeError)


def _sparsax_call(fn, *args):
    """``fn(*args)``, re-raising sparsax's not-positive-definite failure as ours."""
    try:
        return fn(*args)
    except RuntimeError as exc:
        if "not positive definite" in str(exc):
            raise NotPositiveDefiniteError(str(exc)) from exc
        raise


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
            return
        try:
            self._factor = self._symbolic.copy().factorize(sp.csc_matrix(matrix))
        except CholmodNotPositiveDefiniteError as exc:
            raise NotPositiveDefiniteError(str(exc)) from exc

    def _solve(self, rhs: np.ndarray, mode: int) -> np.ndarray:
        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        return _sparsax_call(
            self._sparsax.solve_np, self._Ai, self._Aj, self._Ax, rhs, mode
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Solve P x = rhs."""
        if self._sparsax is not None:
            return self._solve(rhs, _MODE_A)
        return self._factor.solve(rhs)

    def logdet(self) -> float:
        """Return log|P|."""
        if self._sparsax is not None:
            return float(
                _sparsax_call(
                    self._sparsax.logdet_np, self._Ai, self._Aj, self._Ax, self._n
                )
            )
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
# Preconditioned CG solve
# ---------------------------------------------------------------------------


def cg_solve(
    precision: sp.spmatrix | spla.LinearOperator,
    rhs: np.ndarray,
    *,
    tol: float = 1e-8,
    maxiter: int | None = None,
    preconditioner: str = "jacobi",
    diagonal: np.ndarray | None = None,
    x0: np.ndarray | None = None,
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
        precision matrices.  For a ``LinearOperator``, which has no diagonal to
        read, pass it as ``diagonal``; without one the solve is unpreconditioned.
    diagonal : ndarray of shape (n,), optional
        ``diag(P)`` for the Jacobi preconditioner of a ``LinearOperator``.
    x0 : ndarray of shape (n,), optional
        Starting iterate (a warm start); zero by default.

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
        M_op = None
        if diagonal is not None and preconditioner == "jacobi":
            inv = 1.0 / np.asarray(diagonal, dtype=np.float64)
            M_op = spla.LinearOperator((n, n), matvec=lambda v: inv * v)
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
        x0=x0,
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
