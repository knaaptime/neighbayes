"""Fill-in estimation for sparse factorization routing.

Estimates the ratio ``nnz(W²) / nnz(W)`` — the growth in nonzeros from the
first sparse-sparse product — in ``O(nnz)`` without materializing ``W²``.
This ratio directly predicts Cholesky/LU factorization cost: the factor's
nonzero count is bounded below by the fill-in from the first elimination
steps, which ``W²`` captures.

The estimate is an **upper bound**: for each row ``i`` it sums the degrees of
``i``'s neighbors, which overcounts shared neighbors but never undercounts.
This errs toward caution — routing to ``stoch_cheb`` when the factor
*might* be expensive — which is the safe failure mode.

For dense ``W`` (e.g. a KNN graph stored as a dense array), the ratio is
approximately ``n`` (every row has degree ``n``, every neighbor has degree
``n``), so a dense ``n = 2,000`` matrix gives a ratio near 2,000.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import scipy.sparse as sp

_NUMBA_OK = importlib.util.find_spec("numba") is not None


if _NUMBA_OK:
    import numba

    @numba.njit(cache=True)
    def _estimate_w2_nnz_numba(indptr: np.ndarray, indices: np.ndarray, n: int) -> int:
        """Upper-bound ``nnz(W²)`` in ``O(nnz)`` via Numba.

        For each row ``i``, ``W²[i, :]`` is the union of neighbor rows of ``i``.
        Upper bound: sum of degrees of ``i``'s neighbors (overcounts shared
        neighbors, but never undercounts).
        """
        row_degrees = np.empty(n, dtype=np.int64)
        for i in range(n):
            row_degrees[i] = indptr[i + 1] - indptr[i]
        est = np.int64(0)
        for i in range(n):
            for p in range(indptr[i], indptr[i + 1]):
                est += row_degrees[indices[p]]
        return est

    _estimate_w2_nnz = _estimate_w2_nnz_numba
else:
    _estimate_w2_nnz = None


def estimate_fillin_ratio(W: sp.csr_matrix) -> float:
    """Estimate ``nnz(W²) / nnz(W)`` — the fill-in ratio for the first product.

    Parameters
    ----------
    W
        Sparse CSR matrix (spatial weights).  Dense arrays should be
        converted to CSR before calling.

    Returns
    -------
    float
        Estimated ``nnz(W²) / nnz(W)``.  Returns ``inf`` if ``W`` has zero
        nonzeros.

    Notes
    -----
    The estimate is an upper bound on the true ``nnz(W²)`` because it
    double-counts neighbors shared between multiple neighbors of the same
    row.  For a regular lattice (e.g. 2D rook with degree ``d``) the ratio
    is approximately ``d``; for a dense matrix it is approximately ``n``;
    for a hub-dominated graph it can be much larger than the mean degree.
    """
    n = W.shape[0]
    nnz = W.nnz
    if nnz == 0:
        return float("inf")
    indptr = W.indptr.astype(np.int64, copy=False)
    indices = W.indices.astype(np.int64, copy=False)
    if _estimate_w2_nnz is not None:
        est = int(_estimate_w2_nnz(indptr, indices, n))
    else:
        row_degrees = np.diff(indptr)
        est = int(row_degrees[indices].sum())
    return est / nnz
