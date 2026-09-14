"""Tests for AAA rational approximation log-determinant."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes._logdet import make_logdet_numpy_fn, make_logdet_numpy_vec_fn
from neighbayes._logdet._aaa import (
    AAAPrecompute,
    _aaa_algorithm,
    _aaa_algorithm_lazy,
    _adaptive_n_coarse,
    _lu_logdet,
    aaa_logdet_eval,
    aaa_logdet_eval_vec,
    aaa_logdet_precompute,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_symmetric_W():
    """Small symmetric rook contiguity W (for ground-truth comparison)."""
    from libpysal import graph

    from neighbayes import dgp

    gdf = dgp.simulate_sar(n=20, create_gdf=True)
    W = graph.Graph.build_contiguity(gdf, rook=True).transform("r").sparse.toarray()
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_nonsymmetric_W():
    """Small non-symmetric W (KNN-like directed graph)."""
    np.random.seed(42)
    n = 20
    # Build a directed KNN graph: each node connects to 3 random neighbors
    rows, cols, vals = [], [], []
    for i in range(n):
        # Pick 3 random neighbors (not self)
        neighbors = np.random.choice(
            [j for j in range(n) if j != i], size=3, replace=False
        )
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Row-standardize
    degrees = np.array(A.getnnz(axis=1), dtype=np.float64)
    D_inv = sp.diags(1.0 / degrees)
    W = D_inv @ A
    return sp.csr_matrix(W.astype(np.float64))


@pytest.fixture
def small_eigs(small_symmetric_W):
    return np.linalg.eigvals(small_symmetric_W.toarray())


# ---------------------------------------------------------------------------
# LU logdet
# ---------------------------------------------------------------------------


class TestLULogdet:
    def test_matches_eigenvalue(self, small_symmetric_W, small_eigs):
        """LU logdet should match eigenvalue computation."""
        rho = 0.5
        A = sp.eye(small_symmetric_W.shape[0], format="csc") - rho * small_symmetric_W
        lu_ld = _lu_logdet(A)
        eig_ld = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(lu_ld - eig_ld) < 1e-8

    def test_nonsymmetric(self, small_nonsymmetric_W):
        """LU logdet should work for non-symmetric W."""
        rho = 0.3
        n = small_nonsymmetric_W.shape[0]
        A = sp.eye(n, format="csc") - rho * small_nonsymmetric_W
        lu_ld = _lu_logdet(A)
        # Compare with numpy dense
        dense_ld = np.linalg.slogdet(A.toarray())[1]
        assert abs(lu_ld - dense_ld) < 1e-6


class TestLUBackendRouting:
    """KLU/UMFPACK selection, extraction, and the failure ladder."""

    def _grid(self, W, rhos):
        eye = sp.eye(W.shape[0], format="csc")
        Wc = sp.csc_matrix(W)
        return [(eye - r * Wc).tocsc() for r in rhos]

    def test_umf_extractor_matches_diagonal_formula(self, small_nonsymmetric_W):
        """UMFPACK's determinant routine must agree with the L/U diagonals.

        The two are different code paths to the same number — the library's
        mantissa/exponent split versus a log-sum over the factor diagonals —
        so agreement here is what licenses using the cheaper one.
        """
        umfpack = pytest.importorskip("sksparse.umfpack")
        from neighbayes._logdet._aaa import (
            _lu_logdet_from_factor,
            _umf_logdet_from_factor,
        )

        for A in self._grid(small_nonsymmetric_W, (0.15, 0.5, 0.85)):
            factor = umfpack.umf_factor(A)
            dense = np.linalg.slogdet(A.toarray())[1]
            assert abs(_umf_logdet_from_factor(factor) - dense) < 1e-8
            assert (
                abs(_umf_logdet_from_factor(factor) - _lu_logdet_from_factor(factor))
                < 1e-8
            )

    @pytest.mark.parametrize("backend", ["klu", "umfpack", "scipy"])
    def test_pinned_backend_is_used_and_exact(self, small_nonsymmetric_W, backend):
        """Each pinned backend reports itself and returns the exact logdet."""
        if backend != "scipy":
            pytest.importorskip(f"sksparse.{backend}")
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        evaluate = _make_reusable_lu_logdet(backend=backend)
        for A in self._grid(small_nonsymmetric_W, (0.1, 0.4, 0.7, 0.2)):
            assert abs(evaluate(A) - np.linalg.slogdet(A.toarray())[1]) < 1e-8
        assert evaluate.backend == backend

    def test_pinned_backend_skips_the_probe(self, small_nonsymmetric_W):
        """Pinning a backend must not factorize with the other one."""
        pytest.importorskip("sksparse.umfpack")
        import neighbayes._logdet._aaa as aaa_mod
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        seen = []
        real = aaa_mod._load_lu_backend

        def spy(name):
            seen.append(name)
            return real(name)

        aaa_mod._load_lu_backend = spy
        try:
            evaluate = _make_reusable_lu_logdet(backend="umfpack")
            for A in self._grid(small_nonsymmetric_W, (0.2, 0.5)):
                evaluate(A)
        finally:
            aaa_mod._load_lu_backend = real
        assert set(seen) == {"umfpack"}

    def test_auto_probes_once_then_uses_the_winner(
        self, small_nonsymmetric_W, monkeypatch
    ):
        """Auto factorizes with every candidate on call 1 and one thereafter.

        This is the probe's whole cost model: one extra factorization per
        losing backend, never one per node.
        """
        pytest.importorskip("sksparse.klu")
        pytest.importorskip("sksparse.umfpack")
        import neighbayes._logdet._aaa as aaa_mod
        from neighbayes._logdet._aaa import _LU_BACKENDS, _ReusableLULogdet

        # This test pins the full-race behaviour: both candidates factorize
        # once, fastest keeps the work.  The stakes gate would short-circuit
        # the race on a matrix this small, and a memoized route from an
        # earlier test would skip it entirely, so neutralize both.
        monkeypatch.setenv("NEIGHBAYES_LOGDET_LU_PROBE_GATE", "0")
        monkeypatch.setattr(aaa_mod, "_LU_ROUTE_MEMO", {})

        # _evaluate_one resolves the backend on every factorization, so
        # counting here counts factorizations per backend.
        counts: dict[str, int] = {}
        real = aaa_mod._load_lu_backend

        def counting(name):
            counts[name] = counts.get(name, 0) + 1
            return real(name)

        aaa_mod._load_lu_backend = counting
        try:
            evaluate = _ReusableLULogdet(backend=None)
            grid = self._grid(small_nonsymmetric_W, (0.1, 0.3, 0.5, 0.7))
            values = [evaluate(A) for A in grid]
        finally:
            aaa_mod._load_lu_backend = real

        for A, value in zip(grid, values, strict=True):
            assert abs(value - np.linalg.slogdet(A.toarray())[1]) < 1e-8
        winner = evaluate.backend
        assert winner in _LU_BACKENDS
        assert counts[winner] == len(grid)
        for loser in set(_LU_BACKENDS) - {winner}:
            assert counts.get(loser, 0) == 1

    def test_broken_klu_falls_to_umfpack_not_superlu(self, small_nonsymmetric_W):
        """A KLU failure must demote to the other SuiteSparse backend, loudly.

        The trap this guards is a catch-all that treats any KLU hiccup as
        "no SuiteSparse available" and drops straight to SuperLU, the slowest
        of the three.
        """
        pytest.importorskip("sksparse.umfpack")
        import neighbayes._logdet._aaa as aaa_mod
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        real = aaa_mod._load_lu_backend

        def broken_klu(name):
            if name == "klu":
                raise RuntimeError("klu wedged for test")
            return real(name)

        aaa_mod._load_lu_backend = broken_klu
        try:
            evaluate = _make_reusable_lu_logdet(backend="klu")
            with pytest.warns(RuntimeWarning, match="klu"):
                first = evaluate(self._grid(small_nonsymmetric_W, (0.3,))[0])
        finally:
            aaa_mod._load_lu_backend = real

        A = self._grid(small_nonsymmetric_W, (0.3,))[0]
        assert abs(first - np.linalg.slogdet(A.toarray())[1]) < 1e-8

    def test_env_var_pins_backend(self, small_nonsymmetric_W, monkeypatch):
        """NEIGHBAYES_LOGDET_LU_BACKEND selects without an explicit argument."""
        pytest.importorskip("sksparse.umfpack")
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        monkeypatch.setenv("NEIGHBAYES_LOGDET_LU_BACKEND", "umfpack")
        evaluate = _make_reusable_lu_logdet()
        A = self._grid(small_nonsymmetric_W, (0.4,))[0]
        assert abs(evaluate(A) - np.linalg.slogdet(A.toarray())[1]) < 1e-8
        assert evaluate.backend == "umfpack"

    def test_unknown_env_var_warns_and_falls_back_to_auto(self, monkeypatch):
        """A typo in the env var must not silently pin the wrong thing."""
        from neighbayes._logdet._aaa import _lu_backend_preference

        monkeypatch.setenv("NEIGHBAYES_LOGDET_LU_BACKEND", "umfpak")
        with pytest.warns(RuntimeWarning, match="NEIGHBAYES_LOGDET_LU_BACKEND"):
            assert _lu_backend_preference() == "auto"

    def test_routing_agrees_across_backends_on_a_directed_graph(self):
        """All three backends must return the same curve, not merely a number."""
        pytest.importorskip("sksparse.klu")
        pytest.importorskip("sksparse.umfpack")
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        W = _knn_W(300, k=6)
        rhos = np.linspace(0.05, 0.85, 9)
        grids = {
            b: [_make_reusable_lu_logdet(backend=b)] for b in ("klu", "umfpack", None)
        }
        curves = {}
        for name, (evaluate,) in grids.items():
            curves[name] = np.array([evaluate(A) for A in self._grid(W, rhos)])
        assert np.max(np.abs(curves["klu"] - curves["umfpack"])) < 1e-8
        assert np.max(np.abs(curves["klu"] - curves[None])) < 1e-8


class TestReusableLULogdet:
    def test_reuse_matches_single_shot(self, small_nonsymmetric_W):
        """Symbolic-reuse evaluator must match single-shot LU across ρ values.

        Exercises the second-and-later calls (numeric refactor reusing the
        symbolic analysis), which is where the KLU path diverges from a
        fresh factorization.
        """
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        n = small_nonsymmetric_W.shape[0]
        eye = sp.eye(n, format="csc")
        W = sp.csc_matrix(small_nonsymmetric_W)
        evaluate = _make_reusable_lu_logdet()
        for rho in (0.1, 0.4, 0.7, 0.2):
            reused = evaluate(eye - rho * W)
            single = _lu_logdet(eye - rho * W)
            assert abs(reused - single) < 1e-9

    def test_falls_back_without_klu(self, small_nonsymmetric_W, monkeypatch):
        """Evaluator must still be correct when KLU import fails."""
        import builtins

        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        real_import = builtins.__import__

        def _no_klu(name, *args, **kwargs):
            if name == "sksparse.klu" or name.endswith(".klu"):
                raise ImportError("klu disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_klu)

        n = small_nonsymmetric_W.shape[0]
        eye = sp.eye(n, format="csc")
        W = sp.csc_matrix(small_nonsymmetric_W)
        evaluate = _make_reusable_lu_logdet()
        for rho in (0.2, 0.5):
            reused = evaluate(eye - rho * W)
            dense = np.linalg.slogdet((eye - rho * W).toarray())[1]
            assert abs(reused - dense) < 1e-6

    def test_route_memo_skips_second_race(self, small_nonsymmetric_W, monkeypatch):
        """A second context on the same pattern must not re-pay the race.

        The memo's whole value is that the routing decision — a function of
        the sparsity pattern alone, since per-node cost under symbolic reuse
        carries no value dependence — transfers across instances: the refit,
        the matched-budget comparison, one fit per model on a shared W.
        """
        pytest.importorskip("sksparse.klu")
        pytest.importorskip("sksparse.umfpack")
        import neighbayes._logdet._aaa as aaa_mod
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        # Pin the race on (no gate, no inherited route), because the point is
        # to observe the memo being *written*, then observe it being *read*.
        monkeypatch.setenv("NEIGHBAYES_LOGDET_LU_PROBE_GATE", "0")
        monkeypatch.setattr(aaa_mod, "_LU_ROUTE_MEMO", {})

        n = small_nonsymmetric_W.shape[0]
        eye = sp.eye(n, format="csc")
        W = sp.csc_matrix(small_nonsymmetric_W)
        grid = [eye - rho * W for rho in (0.2, 0.5)]

        counts: dict[str, int] = {}
        real = aaa_mod._load_lu_backend

        def counting(name):
            counts[name] = counts.get(name, 0) + 1
            return real(name)

        aaa_mod._load_lu_backend = counting
        try:
            first = _make_reusable_lu_logdet()
            for A in grid:
                first(A)
            winner = first.backend
            settled = dict(counts)
            second = _make_reusable_lu_logdet()
            for A in grid:
                second(A)
        finally:
            aaa_mod._load_lu_backend = real

        # The second context factorized with the memoized winner only.
        for name, count in counts.items():
            if name == winner:
                assert count == 2 * len(grid)
            else:
                assert count == settled.get(name, 0)  # no new factorizations
        assert second.backend == winner

    def test_stakes_gate_skips_race_on_cheap_grid(
        self, small_nonsymmetric_W, monkeypatch
    ):
        """A first candidate under the gate settles without racing the other.

        Below the gate the grid is milliseconds outright, so the second
        backend's probe factorization cannot pay for itself; the gate
        bounds the possible misroute by the gate value itself.
        """
        pytest.importorskip("sksparse.klu")
        pytest.importorskip("sksparse.umfpack")
        import neighbayes._logdet._aaa as aaa_mod
        from neighbayes._logdet._aaa import _make_reusable_lu_logdet

        monkeypatch.setenv("NEIGHBAYES_LOGDET_LU_PROBE_GATE", "10")
        monkeypatch.setattr(aaa_mod, "_LU_ROUTE_MEMO", {})

        seen = []
        real = aaa_mod._load_lu_backend

        def spy(name):
            seen.append(name)
            return real(name)

        aaa_mod._load_lu_backend = spy
        try:
            evaluate = _make_reusable_lu_logdet()
            n = small_nonsymmetric_W.shape[0]
            eye = sp.eye(n, format="csc")
            W = sp.csc_matrix(small_nonsymmetric_W)
            value = evaluate(eye - 0.3 * W)
        finally:
            aaa_mod._load_lu_backend = real

        # Small-W factorizations land far under a 10s gate, so only the
        # KLU-first candidate ran — no UMFPACK probe.
        assert seen == ["klu"]
        dense = np.linalg.slogdet((eye - 0.3 * W).toarray())[1]
        assert abs(value - dense) < 1e-8
        assert evaluate.backend == "klu"


# ---------------------------------------------------------------------------
# AAA algorithm
# ---------------------------------------------------------------------------


class TestAAAAlgorithm:
    def test_polynomial_function(self):
        """AAA should exactly reproduce a polynomial with enough points."""
        z = np.linspace(-1, 1, 100)
        f = z**2  # simple polynomial
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-12)
        # Should need very few support points for a polynomial
        assert len(sp_z) <= 10
        # Evaluate at a test point
        rho = 0.5
        diff = rho - sp_z
        n_val = np.sum(w * sp_f / diff)
        d_val = np.sum(w / diff)
        result = n_val / d_val
        assert abs(result - 0.25) < 1e-10

    def test_rational_function(self):
        """AAA should exactly reproduce a rational function."""
        z = np.linspace(0.1, 0.9, 100)
        f = 1.0 / (1.0 - z)  # has a pole at z=1
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-12)
        # Should converge fast for a simple rational
        assert len(sp_z) <= 10
        # Evaluate
        rho = 0.5
        diff = rho - sp_z
        n_val = np.sum(w * sp_f / diff)
        d_val = np.sum(w / diff)
        result = n_val / d_val
        assert abs(result - 2.0) < 1e-10

    def test_log_function(self):
        """AAA should approximate log(1 - rho*x) well."""
        z = np.linspace(0.1, 0.8, 200)
        x = 0.7  # fixed eigenvalue
        f = np.log(np.abs(1.0 - z * x))
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-10)
        # Should converge in a few points
        assert len(sp_z) <= 20
        # Check accuracy at several points
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            diff = rho - sp_z
            n_val = np.sum(w * sp_f / diff)
            d_val = np.sum(w / diff)
            result = n_val / d_val
            exact = np.log(np.abs(1.0 - rho * x))
            assert abs(result - exact) < 1e-8, f"rho={rho}: {result} vs {exact}"


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------


class TestPrecompute:
    def test_returns_precompute(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        assert isinstance(pre, AAAPrecompute)
        assert pre.n == small_symmetric_W.shape[0]
        assert len(pre.support_points) >= 2
        assert len(pre.support_points) == len(pre.support_values)
        assert len(pre.support_points) == len(pre.weights)

    def test_nonsymmetric_W(self, small_nonsymmetric_W):
        """Should work with non-symmetric W."""
        pre = aaa_logdet_precompute(small_nonsymmetric_W, rho_min=0.1, rho_max=0.8)
        assert isinstance(pre, AAAPrecompute)
        assert pre.n == small_nonsymmetric_W.shape[0]

    def test_custom_interval(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.2, rho_max=0.7)
        assert pre.rho_min == 0.2
        assert pre.rho_max == 0.7


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_accuracy_symmetric(self, small_symmetric_W, small_eigs):
        """AAA should match exact logdet for symmetric W."""
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = aaa_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_accuracy_nonsymmetric(self, small_nonsymmetric_W):
        """AAA should match exact logdet for non-symmetric W."""
        pre = aaa_logdet_precompute(small_nonsymmetric_W, rho_min=0.1, rho_max=0.8)
        n = small_nonsymmetric_W.shape[0]
        for rho in [0.15, 0.3, 0.45, 0.6, 0.75]:
            A = sp.eye(n, format="csc") - rho * small_nonsymmetric_W
            exact = np.linalg.slogdet(A.toarray())[1]
            approx = aaa_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_exact_at_support_points(self, small_symmetric_W, small_eigs):
        """At support points, the approximant should be exact."""
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        for sp_rho, sp_val in zip(pre.support_points, pre.support_values):
            val = aaa_logdet_eval(pre, float(sp_rho))
            assert abs(val - sp_val) < 1e-10


# ---------------------------------------------------------------------------
# Vectorized evaluation
# ---------------------------------------------------------------------------


class TestVectorized:
    def test_vec_matches_scalar(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.15, 0.75, 50)
        vec_vals = aaa_logdet_eval_vec(pre, rhos)
        scalar_vals = np.array([aaa_logdet_eval(pre, r) for r in rhos])
        assert np.allclose(vec_vals, scalar_vals, atol=1e-10)

    def test_vec_shape(self, small_symmetric_W):
        pre = aaa_logdet_precompute(small_symmetric_W, rho_min=0.1, rho_max=0.8)
        rhos = np.linspace(0.1, 0.8, 30)
        vals = aaa_logdet_eval_vec(pre, rhos)
        assert vals.shape == (30,)


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    def test_scalar_factory(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.3, 0.5, 0.7]:
            exact = np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_vec_factory(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_vec_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        rhos = np.linspace(0.15, 0.75, 40)
        vals = fn(rhos)
        exact = np.array([np.sum(np.log(np.abs(1.0 - r * small_eigs))) for r in rhos])
        assert np.allclose(vals, exact, atol=1e-6)

    def test_factory_nonsymmetric(self, small_nonsymmetric_W):
        """Factory should work with non-symmetric W via AAA."""
        n = small_nonsymmetric_W.shape[0]
        eigs = np.linalg.eigvals(small_nonsymmetric_W.toarray())
        fn = make_logdet_numpy_fn(
            small_nonsymmetric_W, eigs, method="aaa", rho_min=0.1, rho_max=0.8
        )
        for rho in [0.2, 0.5]:
            exact = np.sum(np.log(np.abs(1.0 - rho * eigs)))
            approx = fn(rho)
            assert abs(approx - exact) < 1e-6, f"rho={rho}: {approx} vs {exact}"

    def test_factory_T(self, small_symmetric_W, small_eigs):
        fn = make_logdet_numpy_fn(
            small_symmetric_W, small_eigs, method="aaa", rho_min=0.1, rho_max=0.8, T=3
        )
        rho = 0.5
        exact = 3.0 * np.sum(np.log(np.abs(1.0 - rho * small_eigs)))
        assert abs(fn(rho) - exact) < 1e-6


# ---------------------------------------------------------------------------
# Coarse-grid size = LU factorization count
# ---------------------------------------------------------------------------


def _knn_W(n: int, k: int = 6, seed: int = 0) -> sp.csr_matrix:
    """Row-standardized directed KNN graph (non-symmetric → AAA path)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(n, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    A = sp.csr_matrix((np.ones(n * k), (rows, idx.ravel())), shape=(n, n))
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / deg) @ A).tocsr()


class TestAdaptiveNCoarse:
    """The coarse-grid size (= number of exact LU factorizations) is adaptive.

    The old docstring advertised "~6-15 LU factorizations" — that was the AAA
    *support* count; the code actually factorized at every one of the 30
    coarse-grid nodes.  These tests pin the corrected, adaptive behavior.
    """

    def test_interval_within_0_9_draws_14(self):
        # Within |ρ| ≤ 0.9 an estimate lies at least 0.1 from the nearest
        # singularity; 14 nodes (7 support points) kept the ML bias in ρ under
        # 1e-4 there on every weights matrix in the calibration sweeps.
        assert _adaptive_n_coarse(0.1, 0.8) == 14
        assert _adaptive_n_coarse(-0.9, 0.9) == 14

    def test_interval_past_0_9_draws_18(self):
        # 18 nodes (9 support points) covered true ρ out to |ρ| = 0.98.
        assert _adaptive_n_coarse(-0.5, 0.95) == 18
        assert _adaptive_n_coarse(-0.95, 0.5) == 18
        assert _adaptive_n_coarse(0.85, 0.99) == 18
        assert _adaptive_n_coarse(-0.99, 0.99) == 18

    def test_node_cap_env_var(self, monkeypatch):
        monkeypatch.setenv("NEIGHBAYES_LOGDET_NODE_CAP", "12")
        assert _adaptive_n_coarse(-0.99, 0.99) == 12

    def test_n_coarse_equals_lu_factorization_count(self):
        """`_aaa_algorithm_lazy` evaluates exactly n_coarse times (one LU each)."""
        calls = {"n": 0}

        def counting_eval(rho):
            calls["n"] += 1
            return np.log(abs(1.0 - 0.5 * rho))  # cheap stand-in for one LU

        z = np.linspace(0.1, 0.8, 200)
        for n_coarse in (12, 16, 30):
            calls["n"] = 0
            _aaa_algorithm_lazy(z, counting_eval, tol=1e-12, n_coarse=n_coarse)
            assert calls["n"] == n_coarse

    def test_precompute_respects_explicit_n_coarse(self, monkeypatch):
        """An explicit n_coarse overrides the adaptive default (count LUs)."""
        import neighbayes._logdet._aaa as aaa_mod

        W = _knn_W(200, k=6)
        real = aaa_mod._make_reusable_lu_logdet

        counter = {"n": 0}

        def counting_factory():
            evaluate = real()

            def wrapped(A):
                counter["n"] += 1
                return evaluate(A)

            return wrapped

        monkeypatch.setattr(aaa_mod, "_make_reusable_lu_logdet", counting_factory)
        aaa_logdet_precompute(W, rho_min=0.1, rho_max=0.8, n_coarse=12)
        assert counter["n"] == 12

    def test_default_precompute_uses_adaptive(self, monkeypatch):
        """With n_coarse unset, the default interval factorizes 14 times."""
        import neighbayes._logdet._aaa as aaa_mod

        W = _knn_W(200, k=6)
        real = aaa_mod._make_reusable_lu_logdet
        counter = {"n": 0}

        def counting_factory():
            evaluate = real()

            def wrapped(A):
                counter["n"] += 1
                return evaluate(A)

            return wrapped

        monkeypatch.setattr(aaa_mod, "_make_reusable_lu_logdet", counting_factory)
        aaa_logdet_precompute(W, rho_min=0.1, rho_max=0.8)  # adaptive → 14
        assert counter["n"] == 14


class TestWideIntervalAccuracy:
    """AAA holds accuracy on wide/near-singular intervals with the full grid."""

    def test_wide_interval_symmetric(self):
        W = _knn_W(400, k=6)  # directed → genuine AAA path
        n = W.shape[0]
        eye = sp.eye(n, format="csc")
        Wc = sp.csc_matrix(W)
        # A fixed budget of 20 factorizations, so the test measures the algorithm's
        # pointwise accuracy on a wide interval.  The default count targets the
        # bias an estimate of ρ carries, not pointwise error (see
        # ``_adaptive_n_coarse``), and returns 18 here.
        pre = aaa_logdet_precompute(W, rho_min=-0.95, rho_max=0.95, n_coarse=20)
        for rho in (-0.9, -0.5, 0.0, 0.5, 0.9):
            exact = np.linalg.slogdet((eye - rho * Wc).toarray())[1]
            approx = aaa_logdet_eval(pre, rho)
            assert abs(approx - exact) < 1e-5, f"rho={rho}: {approx} vs {exact}"


def _rook_W(side: int):
    """Row-standardized rook contiguity on a side × side lattice."""
    import scipy.sparse as sp

    path = sp.diags([np.ones(side - 1), np.ones(side - 1)], [-1, 1])
    eye = sp.eye(side)
    A = (sp.kron(path, eye) + sp.kron(eye, path)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sp.diags(1.0 / deg) @ A).tocsr()


class TestSpuriousPoles:
    """Poles inside the unit disk approximate no singularity and are removed."""

    def test_fit_has_no_pole_inside_unit_disk(self):
        # On a 50 × 50 rook lattice the raw AAA fit at 16 nodes on the full
        # stability region places a pole at ρ ≈ -0.29, where the Jacobian is
        # analytic; the fitted approximant must not keep it.
        from neighbayes._logdet._aaa import aaa_poles, chol_aaa_logdet_precompute

        pre = chol_aaa_logdet_precompute(
            _rook_W(50), rho_min=-0.99, rho_max=0.99, n_coarse=16
        )
        assert np.all(np.abs(aaa_poles(pre.support_points, pre.weights)) >= 1.0)

    def test_cleanup_removes_the_raw_fits_pole(self):
        from neighbayes._logdet._aaa import (
            CholAAAContext,
            _aaa_algorithm,
            _remove_spurious_poles,
            aaa_poles,
        )

        ctx = CholAAAContext(_rook_W(50))
        lo, hi, m = -0.99, 0.99, 16
        k = np.arange(1, m + 1)
        z = 0.5 * (hi - lo) * np.cos((2 * k - 1) * np.pi / (2 * m)) + 0.5 * (hi + lo)
        f = np.array([ctx.logdet_at(r) for r in z])
        sp_z, sp_f, w = _aaa_algorithm(z, f, tol=1e-13, max_iter=30)
        assert np.any(np.abs(aaa_poles(sp_z, w)) < 1.0)
        sp_z, sp_f, w = _remove_spurious_poles(z, f, sp_z, sp_f, w)
        assert np.all(np.abs(aaa_poles(sp_z, w)) >= 1.0)
        # The cleaned fit still interpolates the function to the accuracy of a
        # clean fit with one support point fewer.
        grid = np.linspace(-0.9, 0.9, 41)
        exact = np.array([ctx.logdet_at(r) for r in grid])
        diff = grid[:, None] - sp_z[None, :]
        approx = (w / diff) @ sp_f / (w / diff).sum(axis=1)
        assert np.max(np.abs(approx - exact)) < 0.05

    def test_reach_shrinks_as_nodes_are_added(self):
        from neighbayes._logdet._aaa import aaa_reach, chol_aaa_logdet_precompute

        W = _rook_W(50)
        reach = [
            aaa_reach(
                chol_aaa_logdet_precompute(W, rho_min=-0.99, rho_max=0.99, n_coarse=m)
            )[0]
            for m in (14, 18, 22)
        ]
        assert 0.0 < reach[2] < reach[1] < reach[0] < 0.1
