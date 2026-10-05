"""KLU or UMFPACK: routing sparsax's non-symmetric LU by the sparsity pattern.

sparsax offers two interchangeable sparse LU backends for ``A = I − ρW`` and its
relatives: KLU (``lu_*``) and UMFPACK (``umf_*``), with the same arguments, the
same results up to rounding and the same JIT, vmap and autodiff behaviour.
Neither dominates.  :func:`sparsax_lu` routes each sparsity pattern by the work
per factor entry of its factorization (:mod:`neighbayes._lu_route`), which
depends on the pattern alone, so a fixed seed reproduces the same draws.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

_BACKENDS = ("klu", "umfpack")


class SparsaxLU(NamedTuple):
    """One sparsax LU backend, as functions with the ``sparsax.lu_*`` signatures."""

    backend: str
    solve: Callable  # (Ai, Aj, Ax, b) -> x
    factor: Callable  # (Ai, Aj, Ax, n) -> token
    solve_factor: Callable  # (token, b, trans=False) -> x
    logdet: Callable  # (Ai, Aj, Ax, n) -> log|det A|
    logdet_factor: Callable  # (token) -> log|det A|


def _functions(name: str) -> SparsaxLU:
    import sparsax

    if name == "umfpack":
        return SparsaxLU(
            "umfpack",
            sparsax.umf_solve,
            sparsax.umf_factor,
            sparsax.umf_solve_factor,
            sparsax.umf_logdet,
            sparsax.umf_logdet_factor,
        )
    return SparsaxLU(
        "klu",
        sparsax.lu_solve,
        sparsax.lu_factor,
        sparsax.lu_solve_factor,
        sparsax.lu_logdet,
        sparsax.lu_logdet_factor,
    )


def _has_umfpack() -> bool:
    import sparsax

    return hasattr(sparsax, "umf_factor")


def set_sparsax_lu_cache_size(size: int) -> None:
    """Set the numeric-factor cache size of both sparsax LU backends.

    Samplers size the cache before any solver is routed, and solvers on
    different patterns within one fit may route differently, so both caps are
    set.  A cap costs no memory until factors fill it.

    Also bounds sparsax's factor tokens where supported.  A token pins its
    factor (and keeps it from being recycled), and a solver holds at most its
    latest one, so a few per cached factor suffice; a solver whose token is
    released refactors (``CachedSparseSolver._with_token``).
    """
    import sparsax

    sparsax.set_lu_cache_size(size)
    if _has_umfpack():
        sparsax.set_umf_cache_size(size)
    if hasattr(sparsax, "set_token_cache_size"):
        sparsax.set_token_cache_size(max(16, size // 2))


def _pinned_backend() -> str | None:
    """``"klu"`` or ``"umfpack"`` when ``NEIGHBAYES_SPARSE_BACKEND`` names one."""
    requested = os.environ.get("NEIGHBAYES_SPARSE_BACKEND", "").strip().lower()
    return requested if requested in _BACKENDS else None


def sparsax_lu(Ai, Aj, n: int, *, backend: str | None = None) -> SparsaxLU:
    """Return the faster sparsax LU backend for this sparsity pattern.

    Parameters
    ----------
    Ai, Aj : array_like of int
        COO indices of the pattern that every later call will share.
    n : int
        Matrix dimension.
    backend : {"klu", "umfpack"} or None
        Pin a backend.  ``None`` defers to ``NEIGHBAYES_SPARSE_BACKEND``, which
        pins either backend by name, and otherwise routes by the pattern.

    Returns
    -------
    SparsaxLU
        The chosen backend's ``solve``, ``factor``, ``solve_factor``,
        ``logdet`` and ``logdet_factor``.

    Notes
    -----
    The route is :func:`neighbayes._lu_route.route` at its factor-and-solve
    threshold: UMFPACK when KLU's work per factor entry exceeds 66, else KLU.
    It costs one KLU factorization per pattern per process.  A misroute is
    expensive in both directions far from the threshold (UMFPACK 7× faster on
    a dense flow pattern, KLU 2× faster on a sparse KNN graph) and costs at
    most a few percent near it.
    """
    from ..._lu_route import SOLVE_THRESHOLD, route

    choice = backend if backend is not None else _pinned_backend()
    if choice is not None and choice not in _BACKENDS:
        raise ValueError(f"backend must be 'klu' or 'umfpack', got {choice!r}")
    if not _has_umfpack():
        return _functions("klu")
    if choice is not None:
        return _functions(choice)
    rows = np.asarray(Ai, dtype=np.int64)
    cols = np.asarray(Aj, dtype=np.int64)
    return _functions(route(rows, cols, int(n), SOLVE_THRESHOLD))
