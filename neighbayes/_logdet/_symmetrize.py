"""D-symmetrization of a row-standardized undirected ``W``.

A row-standardized undirected graph ``W = D⁻¹A`` (``A`` symmetric) is not
literally symmetric, but it is diagonally similar to the symmetric
``W_sym = D^{1/2} W D^{-1/2}``, which has the same (real) eigenvalues.  The
Cholesky log-determinants and the stochastic Chebyshev expansion work on
``W_sym``; these routines recover ``D`` from ``W`` in ``O(nnz)``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_numba_available = False
try:
    from numba import njit as _njit

    _numba_available = True
except ImportError:
    pass


if _numba_available:

    @_njit(cache=True)
    def _symmetrize_kernel(indptr, indices, data, n):
        """JIT-compiled symmetrization: D-recovery + sparse scaling in one pass.

        Returns ``(out_indices, out_data, out_indptr, ok)``.  When ``ok`` is
        ``False`` the input has an asymmetric sparsity pattern or no valid
        symmetrizing diagonal, and the caller must fall back or raise.
        """
        nnz = len(data)

        # -- Build matched edge list + check pattern symmetry ----------------
        n_edges = 0
        for i in range(n):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if i != j and abs(data[p]) >= 1e-300:
                    n_edges += 1

        edge_i = np.empty(n_edges, dtype=np.int64)
        edge_j = np.empty(n_edges, dtype=np.int64)
        edge_fwd = np.empty(n_edges, dtype=np.float64)
        edge_rev = np.empty(n_edges, dtype=np.float64)

        symmetric = True
        k = 0
        for i in range(n):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if i != j and abs(data[p]) >= 1e-300:
                    edge_i[k] = i
                    edge_j[k] = j
                    edge_fwd[k] = data[p]
                    # Binary-search for (j, i) in row j (CSR is sorted).
                    lo = indptr[j]
                    hi = indptr[j + 1]
                    found = False
                    while lo < hi:
                        mid = lo + (hi - lo) // 2
                        if indices[mid] == i:
                            edge_rev[k] = data[mid]
                            found = True
                            break
                        elif indices[mid] < i:
                            lo = mid + 1
                        else:
                            hi = mid
                    if not found:
                        symmetric = False
                        edge_rev[k] = 0.0
                    k += 1

        if not symmetric:
            empty_idx = np.zeros(0, dtype=np.int64)
            empty_dat = np.zeros(0, dtype=np.float64)
            empty_ptr = np.zeros(n + 1, dtype=np.int64)
            return empty_idx, empty_dat, empty_ptr, False

        # -- Build CSR adjacency for BFS (upper-triangle edges only) ----------
        deg = np.zeros(n, dtype=np.int64)
        for k in range(n_edges):
            if edge_i[k] < edge_j[k]:
                deg[edge_i[k]] += 1
                deg[edge_j[k]] += 1

        adj_indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            adj_indptr[i + 1] = adj_indptr[i] + deg[i]

        adj_indices = np.empty(adj_indptr[n], dtype=np.int64)
        adj_edge = np.empty(adj_indptr[n], dtype=np.int64)
        fill = np.zeros(n, dtype=np.int64)

        for k in range(n_edges):
            if edge_i[k] < edge_j[k]:
                i = edge_i[k]
                j = edge_j[k]
                pos = adj_indptr[i] + fill[i]
                adj_indices[pos] = j
                adj_edge[pos] = k
                fill[i] += 1
                pos = adj_indptr[j] + fill[j]
                adj_indices[pos] = i
                adj_edge[pos] = k
                fill[j] += 1

        # -- Union-find for connected components ------------------------------
        parent_uf = np.arange(n, dtype=np.int64)
        rank_uf = np.zeros(n, dtype=np.int64)

        for k in range(n_edges):
            if edge_i[k] < edge_j[k]:
                i = edge_i[k]
                j = edge_j[k]
                ri = i
                while parent_uf[ri] != ri:
                    ri = parent_uf[ri]
                rj = j
                while parent_uf[rj] != rj:
                    rj = parent_uf[rj]
                if ri != rj:
                    if rank_uf[ri] < rank_uf[rj]:
                        parent_uf[ri] = rj
                    elif rank_uf[ri] > rank_uf[rj]:
                        parent_uf[rj] = ri
                    else:
                        parent_uf[rj] = ri
                        rank_uf[ri] += 1

        # Path compression
        for i in range(n):
            r = i
            while parent_uf[r] != r:
                r = parent_uf[r]
            x = i
            while parent_uf[x] != r:
                nxt = parent_uf[x]
                parent_uf[x] = r
                x = nxt

        comp_min = np.full(n, n, dtype=np.int64)
        for i in range(n):
            r = parent_uf[i]
            if i < comp_min[r]:
                comp_min[r] = i

        # -- BFS spanning forest from min-index seeds ------------------------
        tree_parent = np.arange(n, dtype=np.int64)
        logabs = np.zeros(n, dtype=np.float64)
        neg = np.zeros(n, dtype=np.int64)
        visited = np.zeros(n, dtype=np.uint8)

        queue = np.empty(n, dtype=np.int64)
        q_head = 0
        q_tail = 0

        for i in range(n):
            if comp_min[parent_uf[i]] == i and visited[i] == 0:
                visited[i] = 1
                queue[q_tail] = i
                q_tail += 1

        while q_head < q_tail:
            node = queue[q_head]
            q_head += 1
            for p in range(adj_indptr[node], adj_indptr[node + 1]):
                nbr = adj_indices[p]
                if visited[nbr] == 0:
                    visited[nbr] = 1
                    tree_parent[nbr] = node
                    queue[q_tail] = nbr
                    q_tail += 1

                    ek = adj_edge[p]
                    if edge_i[ek] == node and edge_j[ek] == nbr:
                        ratio_val = (
                            edge_fwd[ek] / edge_rev[ek]
                            if abs(edge_rev[ek]) >= 1e-300
                            else 1.0
                        )
                    else:
                        ratio_val = (
                            edge_rev[ek] / edge_fwd[ek]
                            if abs(edge_fwd[ek]) >= 1e-300
                            else 1.0
                        )
                    logabs[nbr] = np.log(abs(ratio_val)) if abs(ratio_val) > 0 else 0.0
                    neg[nbr] = 1 if ratio_val < 0 else 0

        # -- Pointer doubling: accumulate path sums --------------------------
        anc = tree_parent.copy()
        while True:
            converged = True
            for i in range(n):
                if anc[i] != anc[anc[i]]:
                    converged = False
                    break
            if converged:
                break
            for i in range(n):
                if anc[i] != i:
                    logabs[i] = logabs[i] + logabs[anc[i]]
                    neg[i] = neg[i] + neg[anc[i]]
                    anc[i] = anc[anc[i]]

        # -- Per-component centring + exponentiate ---------------------------
        comp_sum = np.zeros(n, dtype=np.float64)
        comp_count = np.zeros(n, dtype=np.int64)
        for i in range(n):
            r = parent_uf[i]
            comp_sum[r] += logabs[i]
            comp_count[r] += 1

        D = np.ones(n, dtype=np.float64)
        for i in range(n):
            r = parent_uf[i]
            if comp_count[r] > 0:
                centered = logabs[i] - comp_sum[r] / comp_count[r]
                if neg[i] % 2 == 0:
                    D[i] = np.exp(centered)
                else:
                    D[i] = -np.exp(centered)

        # -- Validate D -------------------------------------------------------
        for i in range(n):
            if not np.isfinite(D[i]) or D[i] <= 0:
                empty_idx = np.zeros(0, dtype=np.int64)
                empty_dat = np.zeros(0, dtype=np.float64)
                empty_ptr = np.zeros(n + 1, dtype=np.int64)
                return empty_idx, empty_dat, empty_ptr, False

        # -- Scale W → W_sym in-place (same CSR structure) --------------------
        D_sqrt = np.sqrt(D)
        D_inv_sqrt = 1.0 / D_sqrt

        out_data = np.empty(nnz, dtype=np.float64)
        for i in range(n):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                out_data[p] = D_sqrt[i] * data[p] * D_inv_sqrt[j]

        return indices.copy(), out_data, indptr.copy(), True


def _d_symmetrize_numba(W: sp.csr_matrix) -> sp.csc_matrix | None:
    """Numba-accelerated D-symmetrization.

    Returns the symmetrized CSC matrix, or ``None`` if W has an asymmetric
    sparsity pattern or no valid symmetrizing diagonal.  The caller is
    responsible for raising the appropriate error.
    """
    n = W.shape[0]
    if n == 0:
        return sp.csc_matrix((0, 0))
    Wc = sp.csr_matrix(W, dtype=np.float64)
    Wc.sum_duplicates()
    Wc.eliminate_zeros()
    Wc.sort_indices()

    out_indices, out_data, out_indptr, ok = _symmetrize_kernel(
        Wc.indptr.astype(np.int64),
        Wc.indices.astype(np.int64),
        Wc.data.astype(np.float64),
        n,
    )
    if not ok:
        return None
    return sp.csc_matrix((out_data, out_indices, out_indptr), shape=(n, n))


def _recover_symmetrizing_diagonal(W: sp.csr_matrix) -> np.ndarray | None:
    """Recover D such that D^{1/2} W D^{-1/2} is symmetric.

    For W = D⁻¹A (row-standardized, A symmetric), D[i]/D[j] = W[j,i]/W[i,j]
    for each edge (i,j), so ``log D`` is a potential on the graph and is
    recovered by accumulating edge log-ratios along a spanning forest.

    The traversal is a BFS spanning forest from ``scipy.sparse.csgraph``,
    seeded at the lowest-index node of each connected component, and the
    accumulation is done by pointer doubling — ``O(log depth)`` vectorized
    passes rather than a Python loop over edges.

    Accumulating in log space, then centring each component before
    exponentiating, also extends the range of graphs that can be symmetrized at
    all: the multiplicative propagation it replaces overflowed once the edge
    ratios compounded past ``~1e308`` along a path, whereas the same graph is
    representable here because ``D`` is free up to a per-component scalar.  The
    sign of each ratio is carried separately as a parity bit, so a
    sign-inconsistent ``W`` still yields a negative ``D`` and is rejected by the
    caller rather than silently losing its sign to ``log|·|``.

    Edges whose value is below ``1e-300`` in either direction are excluded from
    the traversal; a node they leave isolated becomes its own component and so
    comes back as ``D = 1``.

    Returns
    -------
    np.ndarray or None
        D (up to scalar multiple), or None if W has asymmetric sparsity
        (directed graph — D-symmetrization not applicable).
    """
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    n = W.shape[0]

    # Check symmetric sparsity pattern without densifying: the boolean
    # patterns differ iff their sparse XOR has any stored entries.
    pattern = (W != 0).tocsr()
    if (pattern != pattern.T.tocsr()).nnz > 0:
        return None

    if n == 0:
        return np.ones(0, dtype=np.float64)

    # Drop stored zeros so the surviving pattern is exactly the one the check
    # above proved symmetric.  W and Wᵀ then share an index structure, and
    # entry k of ``fwd`` is W[i,j] while entry k of ``rev`` is W[j,i].
    Wc = sp.csr_matrix(W, dtype=np.float64)
    Wc.sum_duplicates()
    Wc.eliminate_zeros()
    Wc.sort_indices()
    A = Wc
    B = Wc.T.tocsr()
    B.sort_indices()

    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
    cols = A.indices.astype(np.int64)
    fwd = A.data  # W[i, j]
    rev = B.data  # W[j, i]

    # Traversal graph: off-diagonal edges carrying usable values both ways.
    # The predicate is symmetric in (i, j), so the graph is undirected.
    usable = (rows != cols) & (np.abs(fwd) >= 1e-300) & (np.abs(rev) >= 1e-300)
    graph = sp.csr_matrix(
        (np.ones(int(usable.sum())), (rows[usable], cols[usable])),
        shape=(n, n),
    )

    # Per-edge log-ratio: D[j] = D[i] · W[i,j] / W[j,i].
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, fwd / np.where(usable, rev, 1.0), 1.0)
    edge_logabs = np.log(np.abs(ratio))
    edge_neg = (ratio < 0).astype(np.int8)

    # Flat row-major keys are already sorted (CSR with sorted indices), so a
    # single searchsorted resolves every (parent, child) lookup at once.
    keys = rows * n + cols

    n_comp, labels = connected_components(graph, directed=False)

    # BFS spanning forest, seeded at the lowest-index node of each component
    # (matching the seeding order of the loop this replaces).
    order = np.argsort(labels, kind="stable")
    seeds = order[np.searchsorted(labels[order], np.arange(n_comp))]

    # Single super-root BFS instead of per-component loop.
    #
    # A dummy node (index ``n``) is connected to all seeds, then one
    # ``breadth_first_order`` call traverses the entire forest.  Each seed's
    # predecessor is the dummy (index ``n``), so it is treated as a root
    # (self-parented); every other node inherits its BFS parent from the
    # original graph.  This replaces ``n_comp`` Python-level scipy calls with
    # one, turning a 73-second loop on a 39,906-component matrix into <0.1s.
    parent = np.arange(n, dtype=np.int64)  # roots/islands self-parented
    if n_comp > 0 and seeds.size > 0:
        super_n = n + 1
        ext_rows = np.concatenate(
            [rows[usable], np.full(seeds.size, n, dtype=np.int64)]
        )
        ext_cols = np.concatenate([cols[usable], seeds.astype(np.int64)])
        ext_graph = sp.csr_matrix(
            (np.ones(ext_rows.size), (ext_rows, ext_cols)),
            shape=(super_n, super_n),
        )
        ext_graph = (ext_graph + ext_graph.T).tocsr()
        _nodes, pred = breadth_first_order(
            ext_graph, n, directed=False, return_predecessors=True
        )
        # Seeds have pred == n (the dummy) or -1; all others get their real parent.
        valid = (pred[:n] >= 0) & (pred[:n] < n)
        parent[valid] = pred[:n][valid]

    # Edge into each node from its BFS parent; roots contribute nothing.
    is_root = parent == np.arange(n, dtype=np.int64)
    logabs = np.zeros(n, dtype=np.float64)
    neg = np.zeros(n, dtype=np.int64)
    if keys.size:
        pos = np.searchsorted(keys, parent * n + np.arange(n, dtype=np.int64))
        pos = np.clip(pos, 0, keys.size - 1)
        logabs = np.where(is_root, 0.0, edge_logabs[pos])
        neg = np.where(is_root, 0, edge_neg[pos]).astype(np.int64)

    # Pointer doubling: accumulate the path sum from each node to its root in
    # O(log depth) passes.  Roots are self-parents, so they are fixed points.
    anc = parent
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            break
        logabs = logabs + logabs[anc]
        neg = neg + neg[anc]
        anc = nxt

    # ``D`` is defined only up to a scalar multiple *per connected component*
    # (every edge of W_sym rescales by the same factor within a component), so
    # centring each component's log before exponentiating is free and keeps the
    # representable range centred on 1.  Without it, a graph whose edge ratios
    # compound in one direction — a long chain, a steep density gradient —
    # overflows to ``inf`` at one end and underflows to ``0`` at the other, and
    # both are rejected downstream as a failed symmetrization.
    #
    # Via bincount rather than a mask per component: contiguity weights routinely
    # have many islands, and ``labels == c`` in a loop is O(n · n_comp).
    logabs -= (np.bincount(labels, weights=logabs) / np.bincount(labels))[labels]

    # Every node is in some component and every component is traversed from its
    # seed, so an isolated node is its own component: centring leaves its log at
    # zero and it comes back as D = 1 without a special case.
    return np.where(neg % 2 == 0, 1.0, -1.0) * np.exp(logabs)
