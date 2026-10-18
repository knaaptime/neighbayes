"""Tests for ``_recover_symmetrizing_diagonal``.

The routine recovers ``D`` with ``D^{1/2} W D^{-1/2}`` symmetric, which is what
lets ``cheb_cholesky`` factorize a row-standardized undirected ``W``.  It is
also the routing predicate's applicability test (``_is_symmetric_W``), so an
error here silently sends a Cholesky-capable problem down the LU path — or, far
worse, lets a wrong ``W_sym`` through to CHOLMOD, which reads one triangle and
would return a wrong log-determinant without complaint.

The reference implementation below is the Python BFS the vectorized version
replaced.  It is kept here, rather than deleted with the original, precisely so
the replacement has something independent to be checked against: it walks the
graph edge by edge with scalar CSR lookups and shares no code with the
production path.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from neighbayes._logdet._symmetrize import _recover_symmetrizing_diagonal


def _reference_bfs(W: sp.csr_matrix) -> np.ndarray | None:
    """Original scalar-indexing BFS; independent check on the vectorized path."""
    n = W.shape[0]
    pattern = (W != 0).tocsr()
    if (pattern != pattern.T.tocsr()).nnz > 0:
        return None
    D = np.full(n, np.nan, dtype=np.float64)
    W_coo = W.tocoo()
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(W_coo.row, W_coo.col):
        if i != j:
            adj[i].append(j)
    W_csr = W.tocsr()
    for seed in range(n):
        if not np.isnan(D[seed]):
            continue
        D[seed] = 1.0
        queue = deque([seed])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if np.isnan(D[j]):
                    wij = W_csr[i, j]
                    wji = W_csr[j, i]
                    if abs(wij) < 1e-300 or abs(wji) < 1e-300:
                        continue
                    D[j] = D[i] * wij / wji
                    queue.append(j)
    if np.any(np.isnan(D)):
        D[np.isnan(D)] = 1.0
    return D


def _normalize_per_component(W: sp.csr_matrix, D: np.ndarray) -> np.ndarray:
    """``D`` is defined only up to a scalar per component; pin the first entry."""
    _, labels = connected_components((W != 0).astype(float), directed=False)
    out = D.astype(np.float64).copy()
    for c in np.unique(labels):
        mask = labels == c
        out[mask] = out[mask] / out[mask][0]
    return out


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _row_standardize(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / deg) @ A)


def _lattice(side: int, queen: bool = False, weighted: bool = False, seed: int = 0):
    n = side * side
    offsets = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if not queen
        else [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    )
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    rows.append(i * side + j)
                    cols.append(ni * side + nj)
    if weighted:
        rng = np.random.default_rng(seed)
        A = sp.coo_matrix(
            (rng.uniform(0.5, 3.0, len(rows)), (rows, cols)), shape=(n, n)
        ).tocsr()
        A = sp.csr_matrix((A + A.T) / 2.0)  # symmetric kernel weights
    else:
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return _row_standardize(A)


def _directed(n: int = 25, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = rng.integers(0, n, size=n * k)
    keep = rows != cols
    A = sp.coo_matrix(
        (np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n)
    ).tocsr()
    A.data[:] = 1.0
    return _row_standardize(A)


def _with_isolate(side: int = 6):
    W = _lattice(side).tolil()
    W[0, :] = 0
    W[:, 0] = 0
    return sp.csr_matrix(W)


CASES = {
    "rook binary": _lattice(8),
    "queen binary": _lattice(8, queen=True),
    "rook kernel-weighted": _lattice(8, weighted=True),
    "queen kernel-weighted": _lattice(6, queen=True, weighted=True),
    "disconnected": sp.csr_matrix(sp.block_diag([_lattice(5), _lattice(4)])),
    "isolated node": _with_isolate(),
    "single node": sp.csr_matrix(np.zeros((1, 1))),
}


class TestAgreesWithReferenceBFS:
    @pytest.mark.parametrize("label", sorted(CASES))
    def test_matches(self, label):
        W = sp.csr_matrix(CASES[label])
        got = _recover_symmetrizing_diagonal(W)
        want = _reference_bfs(W)
        assert got is not None and want is not None
        np.testing.assert_allclose(
            _normalize_per_component(W, got),
            _normalize_per_component(W, want),
            rtol=1e-10,
            atol=0.0,
        )

    def test_directed_returns_none(self):
        """Asymmetric sparsity has no symmetrizing diagonal."""
        W = sp.csr_matrix(_directed())
        assert _recover_symmetrizing_diagonal(W) is None
        assert _reference_bfs(W) is None

    def test_empty(self):
        out = _recover_symmetrizing_diagonal(sp.csr_matrix((0, 0)))
        assert out is not None and out.size == 0


class TestSymmetrizesInPractice:
    """The contract that actually matters downstream."""

    @pytest.mark.parametrize("label", sorted(CASES))
    def test_D_symmetrizes_W(self, label):
        W = sp.csr_matrix(CASES[label])
        D = _recover_symmetrizing_diagonal(W)
        assert D is not None
        assert np.all(D > 0) and np.all(np.isfinite(D))
        s = np.sqrt(D)
        coo = W.tocoo()
        W_sym = sp.csr_matrix(
            (s[coo.row] * coo.data / s[coo.col], (coo.row, coo.col)), shape=W.shape
        )
        asym = (W_sym - W_sym.T).tocoo()
        assert (0.0 if asym.nnz == 0 else np.abs(asym.data).max()) < 1e-12

    @pytest.mark.parametrize("label", sorted(CASES))
    def test_preserves_spectrum(self, label):
        """Similarity transform — the eigenvalues, and so the logdet, are unchanged."""
        W = sp.csr_matrix(CASES[label])
        if W.shape[0] > 40:
            pytest.skip("dense eigendecomposition kept small")
        D = _recover_symmetrizing_diagonal(W)
        s = np.sqrt(D)
        W_sym = np.diag(s) @ W.toarray() @ np.diag(1.0 / s)
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvals(W.toarray()).real),
            np.sort(np.linalg.eigvals(W_sym).real),
            atol=1e-10,
        )


class TestSignAndScale:
    def test_negative_ratio_propagates_to_a_rejected_D(self):
        """A sign-inconsistent W must not be silently accepted.

        Log-space accumulation would lose the sign; the parity bit keeps it, so
        ``D`` comes back negative and ``_d_symmetrize`` rejects it as before.
        """
        from neighbayes._logdet._chol_cheb import _d_symmetrize

        W = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
        D = _recover_symmetrizing_diagonal(W)
        assert D is not None and np.any(D < 0)
        with pytest.raises(ValueError, match="D-symmetrizable"):
            _d_symmetrize(W)

    @pytest.mark.parametrize("n", [400, 1200])
    def test_survives_long_paths_without_overflow(self, n):
        """A chain whose edge ratios compound past float range must still work.

        ``D[i] ∝ r^i``, so at n = 1200 with r = 3 the raw diagonal spans
        ``3^1200 ≈ 1e572`` — far outside float64.  The old multiplicative
        propagation seeded ``D[0] = 1`` and marched upward, overflowing to
        ``inf``.  Log-space accumulation plus per-component centring keeps the
        whole chain representable, because ``D`` is free up to a scalar.
        """
        r = 3.0
        rows, cols, vals = [], [], []
        for i in range(n - 1):
            # W[i, i+1] = r * W[i+1, i] gives a constant ratio along the chain
            rows += [i, i + 1]
            cols += [i + 1, i]
            vals += [r, 1.0]
        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

        if n > 700:  # the reference implementation cannot represent this one
            with np.errstate(over="ignore"):
                assert not np.all(np.isfinite(_reference_bfs(W)))

        D = _recover_symmetrizing_diagonal(W)
        assert D is not None
        assert np.all(np.isfinite(D)) and np.all(D > 0)
        # D[i] ∝ r^i, so the ratio between consecutive entries is exactly r.
        np.testing.assert_allclose(D[1:] / D[:-1], r, rtol=1e-9)
