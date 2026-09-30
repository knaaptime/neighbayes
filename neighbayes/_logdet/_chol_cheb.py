"""Cholesky-Chebyshev log-determinant for SPD ``I - ρW``.

For row-standardized ``W`` with spectrum in ``[-1, 1]``, the D-symmetrized
matrix ``W_sym = D^{-1/2} A D^{-1/2}`` is symmetric with the same eigenvalues
as ``W``.  This makes ``I - ρW_sym`` **symmetric positive definite** (SPD)
for all ``|ρ| < 1``, enabling **sparse Cholesky** factorization:

* **Exact**: ``log|det(I - ρW)| = 2 Σ log(diag(L))`` — no stochastic noise.
* **Fast**: CHOLMOD sparse Cholesky is ~2× faster than sparse LU for SPD.
* **Scalable**: ``O(nnz^{1.5})`` per factorization, no ``O(n³)`` eigendecomposition.
* **Full range**: works for ``ρ ∈ (-1, 1)`` — the entire stable region.
  The order is selected from the interval's Bernstein-ellipse convergence
  rate (:func:`~._chebyshev.cheb_order_for_tolerance`) — its distance to the
  ``ρ = ±1`` singularities, not its width — so it rises for wide intervals
  (17 nodes for ``[0.1, 0.8]``, 52 for ``[-0.95, 0.95]``) and falls sharply
  for narrow ones (6 for ``[0.55, 0.65]``).

The method evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
in ``[ρ_min, ρ_max]``, then fits a Chebyshev polynomial in ``ρ`` for ``O(order)``
per-``ρ`` evaluation via Clenshaw recurrence.

**Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity pattern,
so CHOLMOD's symbolic analysis (AMD ordering + elimination tree) is performed
only once and reused for all subsequent numeric factorizations via
``factor.factorize()``.  This saves ~64% of per-node cost.

**When to use**: ``n ∈ (500, 60000]``, any ``ρ ∈ (-1, 1)``.  For ``n ≤ 500``
use ``eigenvalue`` (exact eigendecomposition).  For ``n > 60000`` use
``cheb_stochastic`` (avoids ``O(nnz^{1.5})`` Cholesky fill-in).  For
non-symmetric ``W`` (directed graphs: KNN, travel time) use ``aaa`` (rational
approximation via sparse LU).

**Benchmark** (2D rook grid, adaptive order, ρ ∈ [0.1, 0.8], 200 probes,
stochastic error averaged over 8 probe draws, 2026-08):

========== ============= ============= =========== ============= ==========
n          chol setup    chol eval     chol error  stoch setup   stoch err
========== ============= ============= =========== ============= ==========
484        3.3ms         1.7μs         3.7e-8      5.3ms         0.01
4,900      33ms          1.7μs         2.9e-7      34ms          0.05
10,000     98ms          1.7μs         5.7e-7      51ms          0.08
19,881     237ms         1.7μs         1.1e-6      101ms         0.08
40,000     624ms         1.7μs         2.2e-6      252ms         0.17
59,536     1.16s         1.7μs         3.2e-6      362ms         0.14
========== ============= ============= =========== ============= ==========

Cholesky-Chebyshev remains the accuracy leader across this range — exact
(3.7e-8 to 3.2e-6) against 0.01-0.17 for stochastic — and ~34× faster to
evaluate (1.7μs vs ~58μs).  Its setup grows with Cholesky fill-in, reaching
~3.2× the stochastic cost by n≈60k, which is why the auto-selection cutoff sits
at 60,000 rather than where the setup curves first cross.

The stochastic column improved 11-27× over the previous release from the exact
low-order moments (``n_exact``, see :mod:`._cheb_stochastic`), at 5-10% more
setup: on this *narrow* interval the order rule still returns 15, so the depth
is bought without paying for a higher order.  Over the full stability region
``[-0.99, 0.99]`` the order must rise too and the setup roughly triples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._chebyshev import (
    cheb_order_for_tolerance,
    cheb_tail_error,
    chebyshev_coeffs_dct1,
    chebyshev_gauss_nodes,
)
from ._clenshaw import clenshaw_scalar, clenshaw_vec


@dataclass(frozen=True)
class CholChebPrecompute:
    """Precomputed Chebyshev coefficients from Cholesky log-determinant.

    Attributes
    ----------
    coeffs : np.ndarray, shape (order,)
        Chebyshev coefficients of ``log|I - ρW|`` in ``ρ``.
    rho_min : float
        Lower bound of the ρ approximation interval.
    rho_max : float
        Upper bound of the ρ approximation interval.
    order : int
        Chebyshev polynomial degree.
    n : int
        Matrix dimension.
    err_est : float
        A-posteriori truncation-error estimate from the coefficient tail (see
        :func:`~._chebyshev.cheb_tail_error`).  ``nan`` when not computed.
    """

    coeffs: np.ndarray
    rho_min: float
    rho_max: float
    order: int
    n: int
    err_est: float = float("nan")


def _d_symmetrize(W: sp.csr_matrix) -> sp.csc_matrix:
    """D-symmetrize row-standardized ``W``.

    For ``W = D⁻¹A`` (row-standardized, ``A`` symmetric adjacency),
    ``W_sym = D^{1/2} W D^{-1/2} = D^{-1/2} A D^{-1/2}`` is symmetric
    with the **same eigenvalues** as ``W``.

    The symmetrizing degrees are recovered from the *values* of ``W``
    via the edge ratios ``D[i]/D[j] = W[j,i]/W[i,j]`` (BFS propagation).
    The neighbor count (``getnnz``) equals the standardizing degree only
    for binary adjacency — using it for weighted graphs breaks the
    symmetry that CHOLMOD relies on (it reads a single triangle) and
    silently corrupts the log-determinant.

    This makes ``I - ρW_sym`` SPD for ``|ρ| < 1``, enabling sparse Cholesky.

    Raises
    ------
    ValueError
        If no symmetrizing diagonal exists (directed graph, or weights
        inconsistent with ``W = D⁻¹A`` for symmetric ``A``).  Pass
        ``logdet_method="aaa"`` for such matrices.
    """
    n = W.shape[0]
    W = sp.csr_matrix(W)

    # Fast path: W already symmetric — no scaling needed.
    diff = (W - W.T).tocoo()
    if diff.nnz == 0 or np.all(np.abs(diff.data) <= 1e-12):
        return sp.csc_matrix(W)

    # Numba-accelerated path (opt-in via NEIGHBAYES_NUMBA_SYMMETRIZE=1).
    #
    # The pure-NumPy scipy path (below) takes ~40 ms on an 85k-node matrix
    # and is called once per precompute.  The numba kernel is ~25 ms warm
    # but pays a ~1.2 s JIT compilation on the first-ever call (cached to
    # disk via ``cache=True``).  Since symmetrization is not a hot loop,
    # the compile penalty outweighs the warm savings unless the kernel is
    # reused across many matrices in the same session.  Default: off.
    import os

    if os.environ.get("NEIGHBAYES_NUMBA_SYMMETRIZE", "") == "1":
        from ._slq import _numba_available

        if _numba_available:
            from ._slq import _d_symmetrize_numba

            W_sym = _d_symmetrize_numba(W)
            if W_sym is None:
                raise ValueError(
                    "cheb_cholesky requires a D-symmetrizable W (row-"
                    "standardized undirected graph); no valid symmetrizing "
                    'diagonal was found. Use logdet_method="aaa" for '
                    "directed or non-symmetrizable W."
                )
            sym_diff = (W_sym - W_sym.T).tocoo()
            sym_err = float(np.abs(sym_diff.data).max()) if sym_diff.nnz else 0.0
            if sym_err > 1e-10:
                raise ValueError(
                    f"D-symmetrization failed (max asymmetry {sym_err:.2e}); "
                    "W is not of the form D^-1 A with symmetric A. Use "
                    'logdet_method="aaa" for this weights matrix.'
                )
            return W_sym

    # Pure-NumPy path (default).
    from ._slq import _recover_symmetrizing_diagonal

    D = _recover_symmetrizing_diagonal(W)
    if D is None or not np.all(np.isfinite(D)) or np.any(D <= 0):
        raise ValueError(
            "cheb_cholesky requires a D-symmetrizable W (row-standardized "
            "undirected graph); no valid symmetrizing diagonal was found. "
            'Use logdet_method="aaa" for directed or non-symmetrizable W.'
        )

    D_sqrt = np.sqrt(D)
    D_inv_sqrt = 1.0 / D_sqrt
    # W_sym = D^{1/2} W D^{-1/2}  — sparse scaling, no densification
    # W_sym[i,j] = sqrt(d_i) * W[i,j] / sqrt(d_j)
    W_coo = W.tocoo()
    scaled_data = D_sqrt[W_coo.row] * W_coo.data * D_inv_sqrt[W_coo.col]
    W_sym = sp.csc_matrix((scaled_data, (W_coo.row, W_coo.col)), shape=(n, n))

    # Hard guard: CHOLMOD reads one triangle of its input, so a
    # non-symmetric W_sym would produce a silently wrong logdet.
    sym_diff = (W_sym - W_sym.T).tocoo()
    sym_err = float(np.abs(sym_diff.data).max()) if sym_diff.nnz else 0.0
    if sym_err > 1e-10:
        raise ValueError(
            f"D-symmetrization failed (max asymmetry {sym_err:.2e}); W is "
            "not of the form D^-1 A with symmetric A. Use "
            'logdet_method="aaa" for this weights matrix.'
        )
    return W_sym


def _clamp_interval(rho_min: float, rho_max: float) -> tuple[float, float]:
    """Clamp a ρ interval away from the logarithmic singularities at ``±1``."""
    return max(float(rho_min), -0.99), min(float(rho_max), 0.99)


class _ChebContextBase:
    """Interval fitting for a Chebyshev interpolant, given exact node values.

    The factorizer and the interpolant are independent choices: everything below
    depends on the factorizer only through :meth:`logdet_at`, so a subclass
    supplies exact node values by whatever means suits ``W`` — sparse Cholesky
    when it is D-symmetrizable (:class:`CholChebContext`), sparse LU otherwise
    (:class:`LUChebContext`) — and inherits the node placement, the order rule
    and the coefficient fit unchanged.

    Subclasses must define ``n`` and :meth:`logdet_at`.
    """

    __slots__ = ()

    def logdet_at(self, rho_nodes: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def coeffs_on(
        self,
        rho_min: float = 0.1,
        rho_max: float = 0.8,
        order: int | None = None,
        tol: float | None = None,
    ) -> CholChebPrecompute:
        """Fit the Chebyshev interpolant on ``[rho_min, rho_max]``.

        Parameters
        ----------
        rho_min, rho_max : float
            The ρ interval, clamped to ``[-0.99, 0.99]``.
        order : int or None, default None
            Number of nodes.  ``None`` selects it from the interval, ``n`` and
            ``tol`` via :func:`~._chebyshev.cheb_order_for_tolerance`.
        tol : float, optional
            Target absolute error when ``order`` is ``None``.  Defaults to the
            relative target ``DEFAULT_CHEB_RTOL · n``.
        """
        rho_min, rho_max = _clamp_interval(rho_min, rho_max)
        if rho_max <= rho_min:
            raise ValueError(
                f"Invalid rho interval: rho_min={rho_min}, rho_max={rho_max}."
            )
        if order is None:
            order = cheb_order_for_tolerance(rho_min, rho_max, self.n, tol=tol)

        rho_nodes, _ = chebyshev_gauss_nodes(order, rho_min, rho_max)
        coeffs = chebyshev_coeffs_dct1(self.logdet_at(rho_nodes))

        return CholChebPrecompute(
            coeffs=coeffs,
            rho_min=rho_min,
            rho_max=rho_max,
            order=order,
            n=self.n,
            err_est=cheb_tail_error(coeffs),
        )


class LUChebContext(_ChebContextBase):
    """Reusable sparse-LU symbolic analysis for one ``W``, Chebyshev nodes.

    The LU counterpart of :class:`CholChebContext`, and the fourth cell of the
    factorizer × interpolant grid: it pairs the factorizer that works on *any*
    sparse ``W`` with the interpolant that carries a convergence rate computable
    from the interval alone.  Directed weights — KNN, travel time — admit no
    symmetrizing diagonal, so Cholesky is unavailable and this is the only way to
    put a Chebyshev interpolant on them.

    KLU's symbolic analysis is computed once and reused for every later numeric
    factorization, exactly as CHOLMOD's is in :class:`CholChebContext`, since all
    ``I - ρW`` share one sparsity pattern.  It falls back to scipy SuperLU when
    ``sksparse.klu`` is unavailable.

    Expect it to be slower than :class:`CholChebContext` on weights that *are*
    D-symmetrizable — LU does roughly twice the work of Cholesky on an SPD system
    — so prefer the Cholesky context where it applies.  On fragmented graphs that
    ordering can reverse; see the empirical results in the paper.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix.  No symmetrizability requirement.
    """

    __slots__ = ("W", "_eye", "_lu", "n")

    def __init__(self, W):
        if sp.issparse(W) or hasattr(W, "format"):
            W_sp = sp.csr_matrix(W, dtype=np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        self.n = int(W_sp.shape[0])
        self.W = W_sp
        self._eye = sp.eye(self.n, format="csc")
        self._lu = None

    def logdet_at(self, rho_nodes: np.ndarray) -> np.ndarray:
        """Exact ``log|I - ρW|`` at each node, by sparse LU."""
        from ._aaa import _make_reusable_lu_logdet

        if self._lu is None:
            self._lu = _make_reusable_lu_logdet()
        out = np.empty(len(rho_nodes), dtype=np.float64)
        for i, rho in enumerate(rho_nodes):
            out[i] = self._lu(sp.csc_matrix(self._eye - float(rho) * self.W))
        return out


def lu_cheb_logdet_precompute(
    W,
    order: int | None = None,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    tol: float | None = None,
) -> CholChebPrecompute:
    """Chebyshev interpolant of ``log|I - ρW|`` from exact sparse-LU nodes.

    The LU counterpart of :func:`chol_cheb_logdet_precompute`, applicable to any
    sparse ``W`` including directed graphs.  A one-shot wrapper over
    :class:`LUChebContext`; hold the context directly to refit on a second
    interval without repeating the symbolic analysis.
    """
    return LUChebContext(W).coeffs_on(
        rho_min=rho_min, rho_max=rho_max, order=order, tol=tol
    )


def _pattern_factor(ctx):
    """``ctx``'s CHOLMOD factor for the pattern of ``I - ρ W_sym``, built once.

    The symbolic analysis is done on ``I - ½ W_sym`` (SPD, since the spectrum of
    ``W_sym`` lies in [-1, 1]) rather than on the first matrix factored: at
    ρ = 0 sparse arithmetic drops W's entries, and an analysis of the identity
    would not cover any other ρ.
    """
    from ..samplers._utils._spatial_normal import CholmodFactor

    if ctx._factor is None:
        ctx._factor = CholmodFactor(sp.csc_matrix(ctx._eye - 0.5 * ctx.W_sym))
    return ctx._factor


class CholChebContext(_ChebContextBase):
    """Reusable D-symmetrization and CHOLMOD symbolic analysis for one ``W``.

    Everything a Chebyshev interpolant of ``log|I - ρW|`` needs that does *not*
    depend on the interval — the symmetrized matrix ``W_sym``, the identity, and
    CHOLMOD's symbolic analysis (AMD ordering + elimination tree) — is built
    once here and reused by every call to :meth:`coeffs_on`.  Fitting a *second*
    interpolant on a different interval therefore costs only its numeric
    factorizations.

    That is what makes a warmup-adaptive refit affordable.  A sampler past
    warmup knows ρ to within a fraction of the prior's support, and a narrow
    interval both needs fewer nodes (see
    :func:`~._chebyshev.cheb_order_for_tolerance`) and resolves the Jacobian far
    more accurately over the region the posterior actually occupies — but only
    if the refit does not re-pay the setup.  Measured on the rook lattice
    (n = 10,000): a fresh build at 15 nodes costs ~77 ms, a refit at 12 through
    this context ~60 ms, at 6 nodes ~30 ms.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, row-standardized).

    Raises
    ------
    ValueError
        If ``W`` admits no symmetrizing diagonal (directed graph); use
        ``logdet_method="aaa"`` for those.
    """

    __slots__ = ("W_sym", "_eye", "_factor", "n")

    def __init__(self, W):
        if sp.issparse(W) or hasattr(W, "format"):
            W_sp = sp.csr_matrix(W, dtype=np.float64)
        else:
            W_sp = sp.csr_matrix(np.asarray(W, dtype=np.float64))
        self.n = int(W_sp.shape[0])
        self.W_sym = _d_symmetrize(W_sp)
        self._eye = sp.eye(self.n, format="csc")
        self._factor = None

    def logdet_at(self, rho_nodes: np.ndarray) -> np.ndarray:
        """Exact ``log|I - ρW|`` at each node, by sparse Cholesky.

        The first call performs CHOLMOD's symbolic analysis and caches it on the
        instance; every later node — and every later interval — reuses it, since
        the sparsity pattern of ``I - ρW_sym`` does not depend on ρ.
        """
        out = np.empty(len(rho_nodes), dtype=np.float64)
        factor = _pattern_factor(self)
        for i, rho in enumerate(rho_nodes):
            # Numeric factorization only — symbolic analysis reused.
            factor.factorize(sp.csc_matrix(self._eye - float(rho) * self.W_sym))
            out[i] = factor.logdet()
        return out


def chol_cheb_logdet_precompute(
    W,
    order: int | None = None,
    rho_min: float = 0.1,
    rho_max: float = 0.8,
    tol: float | None = None,
) -> CholChebPrecompute:
    """Precompute Chebyshev coefficients via sparse Cholesky log-determinant.

    Evaluates ``log|det(I - ρW)|`` exactly at ``order`` Chebyshev nodes
    in ``[ρ_min, ρ_max]`` via sparse Cholesky factorization, then fits
    a Chebyshev polynomial in ``ρ``.

    **Symbolic reuse**: all ``I - ρW`` matrices share the same sparsity
    pattern, so CHOLMOD's symbolic analysis (AMD ordering + elimination
    tree) is performed only once and reused for all subsequent numeric
    factorizations.

    This is a one-shot convenience wrapper over :class:`CholChebContext`.  To
    fit more than one interval for the same ``W`` — a warmup-adaptive refit —
    hold the context and call :meth:`CholChebContext.coeffs_on` repeatedly, so
    the symmetrization and symbolic analysis are paid once rather than per fit.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix
        Spatial weights matrix (dense or sparse, row-standardized).
    order : int or None, default None
        Chebyshev polynomial degree.  If ``None``, selected from the interval's
        Bernstein-ellipse convergence rate, ``n``, and ``tol``.
    rho_min : float, default 0.1
        Lower bound of the ρ approximation interval.  Clamped to -0.99
        to avoid the singularity at ρ = -1.
    rho_max : float, default 0.8
        Upper bound of the ρ approximation interval.  Clamped to 0.99
        to avoid the singularity at ρ = 1.
    tol : float, optional
        Target absolute error used to pick ``order`` when it is ``None``.
        Defaults to the relative target ``DEFAULT_CHEB_RTOL · n``.

    Returns
    -------
    CholChebPrecompute
        Precomputed Chebyshev coefficients.
    """
    return CholChebContext(W).coeffs_on(
        rho_min=rho_min, rho_max=rho_max, order=order, tol=tol
    )


def chol_cheb_logdet_eval(pre: CholChebPrecompute, rho: float) -> float:
    """Evaluate ``log|I - ρW|`` from precomputed Chebyshev coefficients.

    Uses Clenshaw recurrence: ``O(order)`` per evaluation.

    Parameters
    ----------
    pre : CholChebPrecompute
        Precomputed coefficients from :func:`chol_cheb_logdet_precompute`.
    rho : float
        Spatial autoregressive parameter.
    """
    return clenshaw_scalar(pre.coeffs, rho, pre.rho_min, pre.rho_max)


def chol_cheb_logdet_eval_vec(
    pre: CholChebPrecompute, rho_arr: np.ndarray
) -> np.ndarray:
    """Vectorized evaluation over an array of ρ values."""
    return clenshaw_vec(pre.coeffs, rho_arr, pre.rho_min, pre.rho_max)
