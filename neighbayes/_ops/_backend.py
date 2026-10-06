"""Sparse backend selection, dense LU fast path, and Kronecker factor helpers."""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

# For n in the regime that fits in memory (n^2 weights matrix), calling
# ``scipy.linalg.lu_factor`` (LAPACK ``dgetrf``) on the dense ``L_k = I - rho_k W``
# is several times faster than ``scipy.sparse.linalg.splu``: SuperLU spends
# most of its time in symbolic factorization overhead at these sizes, whereas
# ``dgetrf`` is a single BLAS-3 kernel.  The forward and adjoint passes share
# the same factorization (``lu_solve(..., trans=0)`` vs ``trans=1``), so one
# factorization per Kronecker leg covers both directions.
#
# The threshold below caps the dense path so very large problems still use
# SuperLU.  Tunable via the ``NEIGHBAYES_KRON_DENSE_MAX`` env var
# (see :mod:`neighbayes._config`).


def _kron_dense_max() -> int:
    """Largest ``n`` for which the Kronecker Ops use dense LAPACK over SuperLU."""
    try:
        return int(os.environ.get("NEIGHBAYES_KRON_DENSE_MAX", "512"))
    except (TypeError, ValueError):
        return 512


@lru_cache(maxsize=1)
def _klu_available() -> bool:
    """Return ``True`` when ``sksparse.klu`` (scikit-sparse) is importable."""
    try:
        return importlib.util.find_spec("sksparse.klu") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _umfpack_available() -> bool:
    """Return ``True`` when ``sksparse.umfpack`` (scikit-sparse) is importable."""
    try:
        return importlib.util.find_spec("sksparse.umfpack") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def _warn_sparse_auto_scipy_fallback_once() -> None:
    """Emit a one-time advisory warning for auto fallback to scipy sparse solve."""
    warnings.warn(
        "NEIGHBAYES_SPARSE_BACKEND=auto selected scipy sparse solves because "
        "KLU (from 'scikit-sparse') is not available. Estimation is typically "
        "faster with 'scikit-sparse' installed.",
        RuntimeWarning,
        stacklevel=3,
    )


_SPARSE_BACKEND_AVAILABLE = {
    "klu": _klu_available,
    "umfpack": _umfpack_available,
}


@lru_cache(maxsize=1)
def _select_sparse_backend() -> str:
    """Resolve sparse solve backend from env vars with robust fallback.

    Environment
    -----------
    NEIGHBAYES_SPARSE_BACKEND : {"auto", "scipy", "klu", "umfpack"}
        Default ``auto``. ``auto`` prefers ``klu`` (fastest for the
        structured ``I - rho W`` systems at the mean degrees typical of
        contiguity and KNN weights), then falls back to scipy's SuperLU.
        ``klu`` and ``umfpack`` are both provided by ``scikit-sparse``.
    NEIGHBAYES_SPARSE_STRICT : {"0", "1", "false", "true"}
        If truthy, missing requested optional backends raise ImportError.

    Notes
    -----
    Under ``auto`` the returned name is ``"klu"``, but
    :func:`_sparse_factor` then routes each sparsity pattern to KLU or
    UMFPACK by the work per factor entry of its factorization
    (:func:`neighbayes._lu_route.route`, see :func:`_new_working_factor`).
    The crossover is a property of the graph, not of its size: with the
    symbolic analysis reused, KLU is 6× faster than UMFPACK on a KNN graph
    with 4 neighbours at ``n = 6,400`` and UMFPACK 2.6× faster with 32.
    Naming a backend explicitly disables the routing.
    """
    requested = os.environ.get("NEIGHBAYES_SPARSE_BACKEND", "auto").strip().lower()
    strict = os.environ.get("NEIGHBAYES_SPARSE_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if requested in {"", "auto"}:
        if _klu_available():
            return "klu"
        _warn_sparse_auto_scipy_fallback_once()
        return "scipy"

    if requested in {"scipy", "superlu"}:
        return "scipy"

    if requested in _SPARSE_BACKEND_AVAILABLE:
        if _SPARSE_BACKEND_AVAILABLE[requested]():
            return requested
        msg = (
            f"NEIGHBAYES_SPARSE_BACKEND={requested} requested, but "
            f"'sksparse.{requested}' is not available. Install 'scikit-sparse' "
            "for this backend. Falling back to scipy backend."
        )
        if strict:
            raise ImportError(msg)
        warnings.warn(msg, RuntimeWarning)
        return "scipy"

    msg = (
        f"Unknown NEIGHBAYES_SPARSE_BACKEND='{requested}'. "
        "Valid values are: auto, scipy, klu, umfpack. Falling back to auto."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)
    return "klu" if _klu_available() else "scipy"


@lru_cache(maxsize=1)
def _get_klu_factor():
    """Import and return ``sksparse.klu.KLUFactor``."""
    return importlib.import_module("sksparse.klu").KLUFactor


@lru_cache(maxsize=1)
def _get_umf_factor():
    """Import and return ``sksparse.umfpack.UMFFactor``."""
    return importlib.import_module("sksparse.umfpack").UMFFactor


_FACTOR_CLASSES = {"klu": _get_klu_factor, "umfpack": _get_umf_factor}

# Working factors per thread, keyed on sparsity pattern (LRU).  Each holds one
# symbolic analysis and is refactored in place for every new set of values, so
# a repeated pattern pays the analysis once — 1.3–2.3× cheaper per call than a
# fresh factorization.  Thread-local, so concurrent callers never share one.
_FACTOR_CACHE_SIZE = 16
_factor_cache = threading.local()


def _auto_backend_requested() -> bool:
    requested = os.environ.get("NEIGHBAYES_SPARSE_BACKEND", "auto").strip().lower()
    return requested in {"", "auto"}


def _pattern_key(A_csc, backend: str):
    return (
        backend,
        A_csc.shape,
        A_csc.nnz,
        hash(A_csc.indptr.tobytes()),
        hash(A_csc.indices.tobytes()),
    )


def _new_working_factor(A_csc, backend: str):
    """``(backend, factor)`` for a new pattern, numerically factored at ``A_csc``.

    Under ``auto`` with both bindings importable, the pattern is routed to KLU
    or UMFPACK at the factor-and-solve threshold of
    :func:`neighbayes._lu_route.route`.  The route depends on the pattern
    alone, so every thread and every run makes the same choice, and with it
    the same rounding.
    """
    if backend == "klu" and _auto_backend_requested() and _umfpack_available():
        from .._lu_route import SOLVE_THRESHOLD, route_matrix

        backend = route_matrix(A_csc, SOLVE_THRESHOLD)
    factor = _FACTOR_CLASSES[backend]()(A_csc)  # symbolic analysis
    factor.factorize(A_csc)
    return backend, factor


def _refactor(A_csc, backend: str):
    """This thread's working factor for ``A_csc``'s pattern, refactored at ``A_csc``.

    The factor is shared by every later call on the same pattern and thread;
    callers that keep it past their next factorization must copy it (as
    :func:`_sparse_factor` does).
    """
    if backend not in _FACTOR_CLASSES:
        raise ValueError(f"Unknown sparse backend: {backend!r}")
    if not A_csc.has_sorted_indices:
        A_csc = A_csc.sorted_indices()
    cache = getattr(_factor_cache, "entries", None)
    if cache is None:
        cache = _factor_cache.entries = OrderedDict()
    key = _pattern_key(A_csc, backend)
    entry = cache.get(key)
    if entry is not None and (
        np.array_equal(entry[0], A_csc.indptr)
        and np.array_equal(entry[1], A_csc.indices)
    ):
        cache.move_to_end(key)
        factor = entry[2]
        factor.factorize(A_csc)
        return factor
    _, factor = _new_working_factor(A_csc, backend)
    cache[key] = (A_csc.indptr.copy(), A_csc.indices.copy(), factor)
    if len(cache) > _FACTOR_CACHE_SIZE:
        cache.popitem(last=False)
    return factor


def _sparse_factor(A_csc, backend: str):
    """Factorize ``A_csc`` with the requested SuiteSparse backend.

    Reuses the symbolic analysis of an earlier call with the same sparsity
    pattern (:func:`_refactor`) and returns a private copy of the numeric
    factor, so the caller may hold it across later factorizations.  Under
    ``auto`` the factor may be UMFPACK's even though ``backend`` is ``"klu"``;
    both expose ``solve``.
    """
    return _refactor(A_csc, backend).copy()


def _is_suitesparse(backend: str) -> bool:
    """Return ``True`` for backends that yield a reusable ``scikit-sparse`` factor."""
    return backend in _SPARSE_BACKEND_AVAILABLE


def _writable_f64(rhs: np.ndarray) -> np.ndarray:
    """``rhs`` as a writable float64 array, copying only when it must.

    ``scikit-sparse`` solves take typed memoryviews, which reject read-only
    buffers, and JAX hands ``pure_callback`` functions read-only views of its
    device arrays.  ``np.asarray`` keeps those views, so the UMFPACK solve
    raised ``buffer source array is read-only`` from every JAX host callback.
    """
    return np.require(rhs, dtype=np.float64, requirements=["W"])


def _solve_sparse_vector(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A x = rhs`` for vector RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = _writable_f64(rhs)
    if _is_suitesparse(backend):
        factor = _refactor(A.tocsc(), backend)
        return np.asarray(factor.solve(rhs64), dtype=np.float64)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


def _solve_sparse_matrix(A: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``A X = rhs`` for matrix RHS using configured sparse backend."""
    backend = _select_sparse_backend()
    rhs64 = _writable_f64(rhs)
    if _is_suitesparse(backend):
        # KLU and UMFPACK factors both accept a 2-D RHS directly (single
        # factorization, batched solve).
        factor = _refactor(A.tocsc(), backend)
        return np.asarray(factor.solve(rhs64), dtype=np.float64)
    lu = sp.linalg.splu(A.tocsc())
    return np.asarray(lu.solve(rhs64), dtype=np.float64)


def _factor_solve_logdet(A: sp.spmatrix, rhs: np.ndarray) -> tuple[np.ndarray, float]:
    """Factorize ``A``, solve ``A x = rhs``, and return ``(x, log|det A|)``.

    Uses a ``scikit-sparse`` backend (KLU or UMFPACK) when available, falling
    back to scipy SuperLU.  The logdet comes from UMFPACK's own determinant
    routine where that backend factored ``A``, and from the factor diagonals
    otherwise; see :mod:`neighbayes._logdet._aaa` for why the distinction is
    worth making.
    """
    backend = _select_sparse_backend()
    rhs64 = _writable_f64(rhs)
    A_csc = A.tocsc() if not sp.isspmatrix_csc(A) else A
    if _is_suitesparse(backend):
        from .._logdet._aaa import _lu_logdet_from_factor, _umf_logdet_from_factor

        factor = _refactor(A_csc, backend)
        x = np.asarray(factor.solve(rhs64), dtype=np.float64)
        # Dispatch on the factor, not the name: auto may have chosen UMFPACK.
        if hasattr(factor, "slogdet"):
            return x, _umf_logdet_from_factor(factor)
        return x, _lu_logdet_from_factor(factor)
    lu = sp.linalg.splu(A_csc)
    x = np.asarray(lu.solve(rhs64), dtype=np.float64)
    logdet = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
    return x, logdet


class _CachedSolverOpMixin:
    """Pickling for Ops that keep a numeric factor between ``perform`` calls.

    The cached factor (scikit-sparse, SuperLU or LAPACK) cannot be pickled, and
    PyTensor pickles Ops -- the Numba backend does so to cache compiled graphs
    that call back into Python.  The cache is dropped from the pickled state and
    rebuilt on the next ``perform``.
    """

    _CACHE_ATTRS = ("_cached_solver", "_cached_backend", "_cached_rho", "_cached_rhos")

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in self._CACHE_ATTRS:
            if name in state:
                state[name] = None
        return state


class _SparseFactorSolver:
    """Adapter exposing a ``SuperLU``-like ``solve`` over a ``scikit-sparse`` factor.

    ``sksparse`` KLU factors solve ``A x = rhs`` for both 1-D and
    2-D right-hand sides but do not accept a ``trans`` argument.  Callers
    that need the adjoint build ``A^T`` explicitly and solve with
    ``trans="N"``.  UMFPACK factors *do* take ``trans``, but this adapter
    holds the KLU restriction for both so the two backends stay
    interchangeable at every call site.
    """

    __slots__ = ("_factor",)

    def __init__(self, factor) -> None:
        self._factor = factor

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        if trans != "N":
            raise ValueError("sparse factor solver supports trans='N' only")
        return np.asarray(self._factor.solve(_writable_f64(rhs)), dtype=np.float64)


def _make_cached_sparse_solver(
    A: sp.spmatrix, backend: str | None = None
) -> _SparseFactorSolver | None:
    """Build a reusable SuiteSparse factor solver for repeated solves.

    Parameters
    ----------
    A : scipy.sparse matrix
        Matrix to factorize.
    backend : {"klu", "umfpack", "scipy"} or None, optional
        Backend to use.  When ``None`` the configured backend is resolved.

    Returns
    -------
    _SparseFactorSolver | None
        Reusable solver, or ``None`` when the resolved backend is scipy
        (no reusable SuiteSparse factor) or factorization fails.
    """
    if backend is None:
        backend = _select_sparse_backend()
    if not _is_suitesparse(backend):
        return None
    from .._lu_route import lu_backend_errors

    try:
        return _SparseFactorSolver(_sparse_factor(A.tocsc(), backend))
    except lu_backend_errors() as exc:
        warnings.warn(
            f"Sparse LU backend {backend!r} failed ({type(exc).__name__}: {exc}); "
            "falling back to SciPy's SuperLU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


class _DenseLU:
    """Lightweight wrapper exposing the same ``solve`` API as ``SuperLU``.

    Holds a LAPACK ``(lu, piv)`` pair from :func:`scipy.linalg.lu_factor`
    and dispatches via :func:`scipy.linalg.lu_solve`.  ``trans="T"`` maps to
    LAPACK ``trans=1`` (transpose, no conjugate, real matrices).
    """

    __slots__ = ("_lu", "_piv")

    def __init__(self, A_dense: np.ndarray) -> None:
        self._lu, self._piv = sla.lu_factor(
            A_dense, overwrite_a=False, check_finite=False
        )

    def solve(self, rhs: np.ndarray, trans: str = "N") -> np.ndarray:
        t = 1 if trans == "T" else 0
        return sla.lu_solve((self._lu, self._piv), rhs, trans=t, check_finite=False)


def _factor_kron_factor(
    W_dense: np.ndarray,
    W_sparse: sp.csr_matrix,
    rho: float,
    n: int,
    I_dense: np.ndarray | None = None,
):
    """Return an LU factorization of ``I - rho * W`` using dense LAPACK when small.

    Falls back to ``scipy.sparse.linalg.splu`` for ``n > NEIGHBAYES_KRON_DENSE_MAX``.
    The returned object exposes ``.solve(rhs, trans=...)`` regardless of path.
    """
    if n <= _kron_dense_max() and W_dense is not None:
        I_ref = I_dense if I_dense is not None else np.eye(n, dtype=np.float64)
        L = I_ref - float(rho) * W_dense
        return _DenseLU(L)
    L_sparse = (
        sp.eye(n, format="csr", dtype=np.float64) - float(rho) * W_sparse
    ).tocsc()
    return sp.linalg.splu(L_sparse)
