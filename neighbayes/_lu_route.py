r"""KLU or UMFPACK, decided by the sparsity pattern alone.

SuiteSparse offers two sparse LU factorizations for ``A = I − ρW`` and its
relatives, with the same results up to rounding.  KLU's left-looking
factorization is faster on the sparse graphs of contiguity and few-neighbour
KNN weights; UMFPACK's multifrontal factorization, which hands dense frontal
matrices to BLAS, is faster once those fronts grow.  One number decides
between them: KLU's work per factor entry,

.. math::

    d = \frac{\text{flops}}{\operatorname{nnz}(L) + \operatorname{nnz}(U)},

roughly the size of a typical front.  Mean degree does not decide it, because
the crossover degree falls as ``n`` grows, and ``n`` does not decide it
either.  KLU reports ``d`` exactly, block-triangular form and all, which
matters for directed patterns: a symmetrized estimate overstated ``d``
fourfold on a directed KNN flow pattern, where KLU was 1.7× faster.

**Why not time both.**  The route used to be decided by timing the two
backends.  Timing is noisy, so the choice, and with it the rounding of every
solve, could differ between runs; a sampler amplifies a 1e-15 difference into
a different chain within a few dozen sweeps, so a fixed seed did not fix the
draws.  ``d`` depends on the pattern alone: it is computed at canonical,
diagonally dominant values (diagonal 1, each off-diagonal ``−0.5/degree``), at
which KLU's threshold pivoting keeps the diagonal and the factor's structure
is the symbolic one.

**Thresholds.**  Measured on an Apple M3 Max (Accelerate BLAS) over 40
patterns — rook and queen grids to ``n = 25,600``, KNN with 4-96 neighbours at
``n`` = 400, 1,600 and 6,400, and flow Kronecker patterns to ``N = 10,000``:

============================  ==============  ==============  ==========
workload                      KLU wins at     UMFPACK wins    threshold
============================  ==============  ==============  ==========
factor + solve                ``d`` ≤ 65.8    ``d`` ≥ 66.6    66
factor + log-determinant      ``d`` ≤ 46.0    ``d`` ≥ 48.5    47
============================  ==============  ==============  ==========

A log-determinant moves the crossover down because KLU must materialize ``L``
and ``U`` to read their diagonals while UMFPACK reads the determinant out of
the library.  Away from a threshold the gap is wide (UMFPACK 7× faster at
``d = 724``, KLU 4–8× faster at ``d < 5``), while next to it the backends are
within about 15% of each other; the one pattern on the wrong side of the solve
threshold costs 4%.  On other hardware the crossover may shift, which moves
only those near-ties.  ``NEIGHBAYES_SPARSE_BACKEND`` (sparse solves) and
``NEIGHBAYES_LOGDET_LU_BACKEND`` (log-determinant grids) pin a backend.

Computing ``d`` costs one KLU factorization per pattern per process, after
which it is remembered.
"""

from __future__ import annotations

import threading

import numpy as np
import scipy.sparse as sp

#: Work per factor entry above which UMFPACK factors and solves faster.
SOLVE_THRESHOLD = 66.0

#: Work per factor entry above which UMFPACK factors and extracts a
#: log-determinant faster.
LOGDET_THRESHOLD = 47.0

_DENSITY: dict[tuple, float | None] = {}
_DENSITY_LOCK = threading.Lock()


def _structure(rows, cols, n: int) -> sp.csr_matrix:
    """The pattern as a canonical CSR matrix of ones (sorted, no duplicates)."""
    S = sp.csr_matrix(
        (np.ones(np.size(rows)), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )
    S.sum_duplicates()
    S.data[:] = 1.0
    return S


def front_density(rows, cols, n: int) -> float | None:
    """KLU's flops per factor entry for the pattern ``(rows, cols)``.

    ``None`` when scikit-sparse's KLU is unavailable.  Duplicate entries and
    their order do not matter, and the result is remembered per pattern.
    """
    S = _structure(rows, cols, n)
    key = (int(n), hash((S.indptr.tobytes(), S.indices.tobytes())))
    with _DENSITY_LOCK:
        if key in _DENSITY:
            return _DENSITY[key]
    try:
        from sksparse.klu import klu_factor
    except ImportError:
        density = None
    else:
        C = S.tocoo()
        off = C.row != C.col
        degree = np.bincount(C.row[off], minlength=n)
        A = sp.eye(n, format="csc") - sp.csc_matrix(
            (0.5 / degree[C.row[off]], (C.row[off], C.col[off])), shape=(n, n)
        )
        info = klu_factor(A).info
        density = float(info.flops) / max(float(info.lnz + info.unz), 1.0)
    with _DENSITY_LOCK:
        _DENSITY[key] = density
    return density


def route(rows, cols, n: int, threshold: float) -> str:
    """``"umfpack"`` when the pattern's :func:`front_density` exceeds ``threshold``.

    Otherwise ``"klu"``, including when the density cannot be computed.
    """
    density = front_density(rows, cols, n)
    return "umfpack" if density is not None and density > threshold else "klu"


def route_matrix(A, threshold: float) -> str:
    """:func:`route` for a scipy sparse matrix."""
    C = sp.coo_matrix(A)
    return route(C.row, C.col, C.shape[0], threshold)
