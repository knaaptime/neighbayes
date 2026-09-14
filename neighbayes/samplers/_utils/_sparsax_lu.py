"""KLU or UMFPACK: routing sparsax's non-symmetric LU by measurement.

sparsax offers two interchangeable sparse LU backends for ``A = I − ρW`` and its
relatives: KLU (``lu_*``) and UMFPACK (``umf_*``), with the same arguments, the
same results and the same JIT, vmap and autodiff behaviour.  Neither dominates.
KLU's left-looking factorization is several times faster on the sparse graphs of
contiguity and KNN weights; UMFPACK's multifrontal factorization, which hands
dense frontal matrices to BLAS, wins several-fold once fill-in grows.  The
crossover mean degree falls as ``n`` grows, so no fixed threshold routes
correctly at every size (see :class:`neighbayes._logdet._aaa._ReusableLULogdet`,
which measured the same crossover for the scikit-sparse backends).

:func:`sparsax_lu` therefore times a numeric factorization and a solve on each
backend, keeps the faster, and remembers the choice for the sparsity pattern, so
every later solver built on that pattern skips the race.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

_BACKENDS = ("klu", "umfpack")

#: Routing decisions by sparsity pattern: ``(n, hash of Ai and Aj) -> name``.
#: With the symbolic analysis cached, per-call cost depends on the pattern and
#: not the values, so a decision holds for every solver on that pattern.
_ROUTES: dict[tuple, str] = {}
_ROUTES_LOCK = threading.Lock()


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
    """
    import sparsax

    sparsax.set_lu_cache_size(size)
    if _has_umfpack():
        sparsax.set_umf_cache_size(size)


def _pinned_backend() -> str | None:
    """``"klu"`` or ``"umfpack"`` when ``NEIGHBAYES_SPARSE_BACKEND`` names one."""
    requested = os.environ.get("NEIGHBAYES_SPARSE_BACKEND", "").strip().lower()
    return requested if requested in _BACKENDS else None


#: Distinct scale factors for the probe's timed calls (see :func:`_probe_seconds`).
_NUDGES = itertools.count(1)


def _probe_seconds(lu: SparsaxLU, Ai, Aj, Ax, n: int, repeats: int = 2) -> float:
    """Seconds for one numeric factorization plus one solve with ``lu``.

    An untimed first call absorbs what a sampler pays once per pattern
    (dispatch setup and the symbolic analysis), so the timed calls measure the
    cost paid at every new ρ.  The fastest of ``repeats`` is returned, since
    interference only ever adds time.

    Each timed call scales the values by a factor no earlier call has used.
    sparsax caches factors by value, so a reused nudge would time a cache hit,
    which costs only dispatch and value hashing -- the same for both backends,
    so the race would be decided by noise.
    """
    import jax
    import jax.numpy as jnp

    b = jnp.ones(n, dtype=jnp.float64)
    jax.block_until_ready(lu.solve_factor(lu.factor(Ai, Aj, Ax, n), b))
    best = float("inf")
    for _ in range(repeats):
        Ax_new = jax.block_until_ready(Ax * (1.0 + 1e-9 * next(_NUDGES)))
        start = time.perf_counter()
        jax.block_until_ready(lu.solve_factor(lu.factor(Ai, Aj, Ax_new, n), b))
        best = min(best, time.perf_counter() - start)
    return best


def _race(Ai, Aj, Ax, n: int) -> str:
    """Probe every backend and return the fastest one's name."""
    import jax.numpy as jnp

    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    Ax = jnp.asarray(Ax, dtype=jnp.float64)
    timings: dict[str, float] = {}
    for name in _BACKENDS:
        try:
            timings[name] = _probe_seconds(_functions(name), Ai, Aj, Ax, n)
        except Exception as exc:  # noqa: BLE001 - a failing backend is skipped
            warnings.warn(
                f"sparsax {name} failed while probing the LU route "
                f"({type(exc).__name__}: {exc}); it will not be used for this "
                "sparsity pattern.",
                RuntimeWarning,
                stacklevel=4,
            )
    if not timings:
        return "klu"
    return min(timings, key=timings.__getitem__)


def sparsax_lu(Ai, Aj, Ax, n: int, *, backend: str | None = None) -> SparsaxLU:
    """Return the faster sparsax LU backend for this sparsity pattern.

    Parameters
    ----------
    Ai, Aj : array_like of int32
        COO indices of the pattern that every later call will share.
    Ax : array_like of float64
        Representative values on that pattern, for example ``I − ρW`` at a
        mid-range ρ.  The probe factorizes them, so they must be nonsingular.
    n : int
        Matrix dimension.
    backend : {"klu", "umfpack"} or None
        Pin a backend and skip the probe.  ``None`` defers to
        ``NEIGHBAYES_SPARSE_BACKEND``, which pins either backend by name, and
        otherwise measures.

    Returns
    -------
    SparsaxLU
        The chosen backend's ``solve``, ``factor``, ``solve_factor``,
        ``logdet`` and ``logdet_factor``.

    Notes
    -----
    The probe costs two factorizations and two solves per backend, once per
    pattern per process.  A sampler factorizes thousands of times, and a
    misroute is expensive in both directions: on the scikit-sparse
    measurement in ``_ReusableLULogdet`` (``n = 3,000``), UMFPACK was 18.7×
    faster at mean degree 65 and KLU 2.8× faster at degree 2.  A backend that
    raises during the probe is skipped with a warning; if both raise, KLU is
    returned so that the error surfaces at the caller's first real call.
    """
    choice = backend if backend is not None else _pinned_backend()
    if choice is not None and choice not in _BACKENDS:
        raise ValueError(f"backend must be 'klu' or 'umfpack', got {choice!r}")
    if not _has_umfpack():
        return _functions("klu")
    if choice is not None:
        return _functions(choice)

    Ai_np = np.asarray(Ai, dtype=np.int32)
    Aj_np = np.asarray(Aj, dtype=np.int32)
    key = (int(n), hash((Ai_np.tobytes(), Aj_np.tobytes())))
    with _ROUTES_LOCK:
        winner = _ROUTES.get(key)
        if winner is None:
            winner = _race(Ai, Aj, Ax, int(n))
            _ROUTES[key] = winner
    return _functions(winner)
