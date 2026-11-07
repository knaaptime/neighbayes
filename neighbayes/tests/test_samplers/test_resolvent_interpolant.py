"""Functions of the multiplier across posterior draws, from a few factorizations.

:class:`ResolventInterpolant` must reproduce dense inverses at every draw, by
either route to the diagonal, including draws next to ``ρ = ±1``, where a power
series would need tens of thousands of terms.  :meth:`CachedSparseSolver.solve_batch`
must agree with separate solves, on either backend, and compile once.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers._utils import _sparsax_utils
from neighbayes.samplers._utils._sparsax_utils import (
    CachedSparseSolver,
    ResolventInterpolant,
    _multiplier_diagonal,
    fold_resolvent_columns,
)


def _rook(side: int) -> sp.csr_matrix:
    """A row-standardized rook lattice: D-symmetrizable, with eigenvalues ±1."""
    n = side * side
    rows, cols = [], []
    for i in range(side):
        for j in range(side):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < side and 0 <= jj < side:
                    rows.append(i * side + j)
                    cols.append(ii * side + jj)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return sp.csr_matrix(sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A)


def _directed(n: int, seed: int = 0) -> sp.csr_matrix:
    """A row-standardized directed graph, with no symmetrizing diagonal."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n):
        for j in rng.choice(np.delete(np.arange(n), i), 3, replace=False):
            rows.append(i)
            cols.append(j)
    A = sp.csr_matrix((np.ones(3 * n), (rows, cols)), shape=(n, n))
    return sp.csr_matrix(sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A)


GRAPHS = {"rook": _rook(5), "directed": _directed(25)}
RANGES = {
    "interior": (0.2, 0.6),
    "near_bound": (0.9, 0.9988),
    "negative": (-0.9, -0.2),
}


def _dense(W, rho):
    return np.linalg.inv(np.eye(W.shape[0]) - rho * W.toarray())


@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize("span", list(RANGES))
def test_functions_match_the_dense_multiplier(graph, span):
    W = GRAPHS[graph]
    n = W.shape[0]
    rng = np.random.default_rng(1)
    rho = rng.uniform(*RANGES[span], 40)
    B = rng.standard_normal((n, 2))
    F = ResolventInterpolant(W, rho, rhs=B, row_sums=True, col_sums=True, diag=True)
    L = F.weights(rho)
    for g in range(rho.size):
        S = _dense(W, rho[g])
        np.testing.assert_allclose(
            np.tensordot(L[g], F.values["rhs"], axes=1), S @ B, rtol=1e-8, atol=1e-9
        )
        np.testing.assert_allclose(L[g] @ F.values["row_sums"], S.sum(1), rtol=1e-8)
        np.testing.assert_allclose(L[g] @ F.values["col_sums"], S.sum(0), rtol=1e-8)
        np.testing.assert_allclose(L[g] @ F.values["diag"], np.diag(S), rtol=1e-8)


def test_one_draw_needs_no_range():
    W = GRAPHS["rook"]
    F = ResolventInterpolant(W, [0.4], diag=True)
    d = F.weights([0.4])[0] @ F.values["diag"]
    np.testing.assert_allclose(d, np.diag(_dense(W, 0.4)), rtol=1e-10)


def test_the_node_count_does_not_grow_near_the_bound():
    """The series' order is 1/(1 − ρ); the nodes stay in the tens at any ρ."""
    W = GRAPHS["rook"]
    rng = np.random.default_rng(2)
    interior = ResolventInterpolant(W, rng.uniform(0.2, 0.6, 50), row_sums=True)
    near = ResolventInterpolant(W, rng.uniform(0.9, 0.9995, 50), row_sums=True)
    everywhere = ResolventInterpolant(
        W, np.r_[-0.999, 0.999, rng.uniform(-0.9, 0.9, 50)], row_sums=True
    )
    assert interior.values["row_sums"].shape[0] <= 33
    assert near.values["row_sums"].shape[0] <= 33
    assert everywhere.values["row_sums"].shape[0] <= 129


@pytest.mark.requires_jax
def test_selected_inversion_matches_the_columns(monkeypatch):
    W = sp.csc_matrix(GRAPHS["rook"])
    n = W.shape[0]
    solver = CachedSparseSolver([W], n)
    rho = np.array([-0.7, 0.2, 0.95])
    selected = _multiplier_diagonal(W, solver)(rho)
    monkeypatch.setattr(
        "neighbayes.samplers.negbin_reduced._core._symmetrizing_diagonal",
        lambda W_csc: None,
    )
    by_columns = _multiplier_diagonal(W, solver)(rho)
    np.testing.assert_allclose(selected, by_columns, rtol=1e-11)
    for r, d in zip(rho, selected):
        np.testing.assert_allclose(d, np.diag(_dense(GRAPHS["rook"], r)), rtol=1e-11)


def test_fold_sums_every_column_across_blocks(monkeypatch):
    W = GRAPHS["directed"]
    n = W.shape[0]
    solver = CachedSparseSolver([W], n)
    # A budget of three columns a node forces several blocks.
    monkeypatch.setattr(_sparsax_utils, "_BATCH_BYTES", 3 * 8 * n * 8)
    rho = np.array([0.1, 0.5, 0.8])
    got = fold_resolvent_columns(solver, rho, lambda C, i: (C**2).sum(axis=1) + i)
    for i, r in enumerate(rho):
        S = _dense(W, r)
        np.testing.assert_allclose(
            got[i], (S**2).sum(1) + i * np.ceil(n / 3), rtol=1e-10
        )


@pytest.mark.parametrize("pad_to", [None, 8])
def test_solve_batch_matches_separate_solves(pad_to):
    W1, W2 = GRAPHS["directed"], sp.csr_matrix(_directed(25, seed=1))
    n = W1.shape[0]
    rng = np.random.default_rng(3)
    coeffs = np.array([[-0.3, 0.1], [0.2, -0.4], [-0.5, -0.1]])
    rhs = rng.standard_normal((3, n, 4))
    solver = CachedSparseSolver([W1, W2], n)
    got = solver.solve_batch(coeffs, rhs, pad_to=pad_to)
    assert got.shape == rhs.shape
    for b in range(3):
        np.testing.assert_allclose(got[b], solver.solve(coeffs[b], rhs[b]), rtol=1e-10)


def test_solve_batch_checks_its_shapes():
    W = GRAPHS["rook"]
    solver = CachedSparseSolver([W], W.shape[0])
    with pytest.raises(ValueError, match="rhs must be"):
        solver.solve_batch(np.zeros((2, 1)), np.zeros((3, W.shape[0], 1)))


def test_without_sparsax_the_answers_are_the_same(monkeypatch):
    W = GRAPHS["rook"]
    rho = np.array([0.1, 0.5, 0.97])
    F = ResolventInterpolant(W, rho, row_sums=True, col_sums=True, diag=True)
    monkeypatch.setattr("neighbayes._jax_dispatch._sparsax_available", lambda: False)
    plain = ResolventInterpolant(W, rho, row_sums=True, col_sums=True, diag=True)
    for name in ("row_sums", "col_sums", "diag"):
        np.testing.assert_allclose(
            F.weights(rho) @ F.values[name],
            plain.weights(rho) @ plain.values[name],
            rtol=1e-9,
        )


@pytest.mark.requires_jax
def test_solve_batch_compiles_once_per_structure():
    """A new solver on the same pattern, and a short padded batch, reuse the program."""
    import jax.monitoring

    events = []

    def listener(name, _secs, **_kw):
        if name == "/jax/core/compile/backend_compile_duration":
            events.append(name)

    W = _rook(6)
    n = W.shape[0]
    rhs = np.ones((8, n, 3))
    first = CachedSparseSolver([W], n)
    first.solve_batch(np.full((8, 1), -0.2), rhs)
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        CachedSparseSolver([W], n).solve_batch(np.full((8, 1), -0.3), rhs)
        first.solve_batch(np.full((3, 1), -0.4), rhs[:3], pad_to=8)
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert not events
