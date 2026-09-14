"""Routing sparsax's non-symmetric LU between KLU and UMFPACK."""

import numpy as np
import pytest
import scipy.sparse as sp

sparsax = pytest.importorskip("sparsax")
if not hasattr(sparsax, "umf_factor"):
    pytest.skip("sparsax built without UMFPACK", allow_module_level=True)

from neighbayes._jax_dispatch import ensure_x64  # noqa: E402
from neighbayes.samplers._utils import _sparsax_lu as lu_mod  # noqa: E402

ensure_x64()


def _system(side=12, rho=0.4):
    """``I − ρW`` for row-standardised rook contiguity, as COO plus dense."""
    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.identity(side)
    W = (sp.kron(eye, path) + sp.kron(path, eye)).tocsr()
    W = sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W
    A = (sp.identity(side * side) - rho * W).tocoo()
    return (
        A.row.astype(np.int32),
        A.col.astype(np.int32),
        A.data.astype(np.float64),
        A.toarray(),
    )


@pytest.fixture(autouse=True)
def _fresh_routes(monkeypatch):
    monkeypatch.delenv("NEIGHBAYES_SPARSE_BACKEND", raising=False)
    monkeypatch.setattr(lu_mod, "_ROUTES", {})


@pytest.mark.parametrize("backend", ["klu", "umfpack"])
def test_backends_are_interchangeable(backend):
    Ai, Aj, Ax, A = _system()
    n = A.shape[0]
    lu = lu_mod.sparsax_lu(Ai, Aj, Ax, n, backend=backend)
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


def test_environment_pins_backend_without_probing(monkeypatch):
    monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "umfpack")

    def _no_probe(*args, **kwargs):
        raise AssertionError("a pinned backend must not be probed")

    monkeypatch.setattr(lu_mod, "_probe_seconds", _no_probe)
    Ai, Aj, Ax, A = _system()
    assert lu_mod.sparsax_lu(Ai, Aj, Ax, A.shape[0]).backend == "umfpack"


def test_auto_keeps_the_faster_backend_and_remembers_it(monkeypatch):
    calls = []

    def _fake_probe(lu, Ai, Aj, Ax, n):
        calls.append(lu.backend)
        return {"klu": 2.0, "umfpack": 1.0}[lu.backend]

    monkeypatch.setattr(lu_mod, "_probe_seconds", _fake_probe)
    Ai, Aj, Ax, A = _system()
    n = A.shape[0]
    assert lu_mod.sparsax_lu(Ai, Aj, Ax, n).backend == "umfpack"
    # Same pattern, new values: the recorded route is reused without a race.
    assert lu_mod.sparsax_lu(Ai, Aj, 1.1 * Ax, n).backend == "umfpack"
    assert calls == ["klu", "umfpack"]


def test_backend_failing_the_probe_is_skipped(monkeypatch):
    def _fake_probe(lu, Ai, Aj, Ax, n):
        if lu.backend == "umfpack":
            raise RuntimeError("probe failure")
        return 1.0

    monkeypatch.setattr(lu_mod, "_probe_seconds", _fake_probe)
    Ai, Aj, Ax, A = _system()
    with pytest.warns(RuntimeWarning, match="umfpack"):
        assert lu_mod.sparsax_lu(Ai, Aj, Ax, A.shape[0]).backend == "klu"


@pytest.mark.parametrize("backend", ["klu", "umfpack"])
def test_repeated_probe_times_real_factorizations(backend):
    """A second probe on the same values must factorize, not hit the value cache."""
    Ai, Aj, Ax, A = _system()
    n = A.shape[0]
    lu = lu_mod._functions(backend)
    lu_mod._probe_seconds(lu, Ai, Aj, Ax, n)
    before = sparsax.factorization_count()
    lu_mod._probe_seconds(lu, Ai, Aj, Ax, n, repeats=2)
    assert sparsax.factorization_count() - before == 2


def test_real_probe_chooses_a_backend():
    Ai, Aj, Ax, A = _system()
    assert lu_mod.sparsax_lu(Ai, Aj, Ax, A.shape[0]).backend in ("klu", "umfpack")


def test_cache_size_sets_both_backends(monkeypatch):
    seen = {}
    monkeypatch.setattr(sparsax, "set_lu_cache_size", lambda k: seen.update(klu=k))
    monkeypatch.setattr(sparsax, "set_umf_cache_size", lambda k: seen.update(umfpack=k))
    lu_mod.set_sparsax_lu_cache_size(48)
    assert seen == {"klu": 48, "umfpack": 48}
