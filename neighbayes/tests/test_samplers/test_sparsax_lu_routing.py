"""Routing sparse LU between KLU and UMFPACK by the sparsity pattern."""

import numpy as np
import pytest
import scipy.sparse as sp

sparsax = pytest.importorskip("sparsax")
if not hasattr(sparsax, "umf_factor"):
    pytest.skip("sparsax built without UMFPACK", allow_module_level=True)
pytest.importorskip("sksparse.klu")

from neighbayes import _lu_route as route_mod  # noqa: E402
from neighbayes._jax_dispatch import ensure_x64  # noqa: E402
from neighbayes.samplers._utils import _sparsax_lu as lu_mod  # noqa: E402

ensure_x64()


def _rook(side):
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = (sp.kron(eye, path) + sp.kron(path, eye)).tocsr()
    return sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W


def _system(side=12, rho=0.4):
    """``I − ρW`` for row-standardised rook contiguity, as COO plus dense."""
    A = (sp.identity(side * side) - rho * _rook(side)).tocoo()
    return (
        A.row.astype(np.int32),
        A.col.astype(np.int32),
        A.data.astype(np.float64),
        A.toarray(),
    )


def _flow(side=6):
    """The unrestricted flow pattern ``I − W⊗I − I⊗W − W⊗W`` of a rook grid."""
    W = _rook(side)
    eye = sp.identity(side * side)
    return (
        sp.identity(W.shape[0] ** 2)
        - 0.2 * (sp.kron(W, eye) + sp.kron(eye, W) + sp.kron(W, W))
    ).tocoo()


@pytest.fixture(autouse=True)
def _fresh_routes(monkeypatch):
    monkeypatch.delenv("NEIGHBAYES_SPARSE_BACKEND", raising=False)
    monkeypatch.setattr(route_mod, "_DENSITY", {})


@pytest.mark.parametrize("backend", ["klu", "umfpack"])
def test_backends_are_interchangeable(backend):
    Ai, Aj, Ax, A = _system()
    n = A.shape[0]
    lu = lu_mod.sparsax_lu(Ai, Aj, n, backend=backend)
    assert lu.backend == backend

    b = np.linspace(-1.0, 1.0, n)
    np.testing.assert_allclose(
        np.asarray(lu.solve(Ai, Aj, Ax, b)), np.linalg.solve(A, b), atol=1e-10
    )
    tok = lu.factor(Ai, Aj, Ax, n)
    np.testing.assert_allclose(
        np.asarray(lu.solve_factor(tok, b, trans=True)),
        np.linalg.solve(A.T, b),
        atol=1e-10,
    )
    logdet = np.linalg.slogdet(A)[1]
    assert abs(float(lu.logdet(Ai, Aj, Ax, n)) - logdet) < 1e-9
    assert abs(float(lu.logdet_factor(tok)) - logdet) < 1e-9


def test_density_depends_on_the_pattern_alone(monkeypatch):
    """Entry order, duplicates and values must not move the statistic.

    The route is what fixes the rounding of every solve, so anything but the
    pattern reaching it would let one fit's draws depend on another's.
    """
    A = _flow(side=4)
    first = route_mod.front_density(A.row, A.col, A.shape[0])
    monkeypatch.setattr(route_mod, "_DENSITY", {})
    order = np.random.default_rng(0).permutation(A.nnz)
    rows = np.concatenate([A.row[order], A.row[:10]])  # shuffled, with duplicates
    cols = np.concatenate([A.col[order], A.col[:10]])
    assert route_mod.front_density(rows, cols, A.shape[0]) == first


def test_sparse_graph_routes_to_klu_and_dense_flow_to_umfpack():
    Ai, Aj, _, A = _system()
    flow = _flow()
    for threshold in (route_mod.SOLVE_THRESHOLD, route_mod.LOGDET_THRESHOLD):
        assert route_mod.route(Ai, Aj, A.shape[0], threshold) == "klu"
        assert route_mod.route_matrix(flow, threshold) == "umfpack"
    assert lu_mod.sparsax_lu(Ai, Aj, A.shape[0]).backend == "klu"
    assert lu_mod.sparsax_lu(flow.row, flow.col, flow.shape[0]).backend == "umfpack"


def test_environment_pins_backend_without_routing(monkeypatch):
    monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "umfpack")

    def _no_route(*args, **kwargs):
        raise AssertionError("a pinned backend must not be routed")

    monkeypatch.setattr(route_mod, "route", _no_route)
    Ai, Aj, _, A = _system()
    assert lu_mod.sparsax_lu(Ai, Aj, A.shape[0]).backend == "umfpack"


def test_without_klu_the_route_is_klu(monkeypatch):
    """With no density to read, route to KLU, sparsax's default."""
    import builtins

    real_import = builtins.__import__

    def _no_klu(name, *args, **kwargs):
        if name == "sksparse.klu":
            raise ImportError("klu disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_klu)
    flow = _flow()
    assert route_mod.front_density(flow.row, flow.col, flow.shape[0]) is None
    assert route_mod.route_matrix(flow, route_mod.SOLVE_THRESHOLD) == "klu"


def test_cache_size_sets_both_backends(monkeypatch):
    seen = {}
    monkeypatch.setattr(sparsax, "set_lu_cache_size", lambda k: seen.update(klu=k))
    monkeypatch.setattr(sparsax, "set_umf_cache_size", lambda k: seen.update(umfpack=k))
    lu_mod.set_sparsax_lu_cache_size(48)
    assert seen == {"klu": 48, "umfpack": 48}


def test_cached_solver_refactors_a_stale_token():
    """A token the token cache released is refactored, not raised.

    sparsax reports a stale token as INVALID_ARGUMENT, which JAX raises as
    ``ValueError``; catching ``RuntimeError`` instead let it escape.
    """
    from neighbayes.samplers._utils._sparsax_utils import CachedSparseSolver

    if not hasattr(sparsax, "set_token_cache_size"):
        pytest.skip("sparsax without a token cache")
    W = _rook(6)
    n = W.shape[0]
    b = np.linspace(-1.0, 1.0, n)
    held = CachedSparseSolver([W], n)
    other = CachedSparseSolver([W], n)
    if held._lu is None or not held._has_lu_factor:
        pytest.skip("solver does not hold factor tokens")
    expected = np.linalg.solve(np.eye(n) - 0.3 * W.toarray(), b)
    sparsax.set_token_cache_size(16)
    try:
        np.testing.assert_allclose(held.solve([-0.3], b), expected, atol=1e-10)
        stale = held._last_token
        for rho in np.linspace(0.01, 0.2, 40):  # release the held token
            other.solve([-float(rho)], b)
        np.testing.assert_allclose(held.solve([-0.3], b), expected, atol=1e-10)
        assert held._last_token is not stale  # the refactor path ran
    finally:
        lu_mod.set_sparsax_lu_cache_size(32)
