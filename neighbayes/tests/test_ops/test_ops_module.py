"""Tests for neighbayes._ops — Kronecker-factored flow solve ops.

Covers:
- Numerical equivalence of KroneckerFlowSolveOp vs reference Kronecker solve.
- Numerical equivalence of KroneckerFlowSolveMatrixOp vs reference.
- VJP correctness via pytensor.gradient.verify_grad.
- logp compile smoke test at a moderate n to catch regressions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.sparse as sp
from pytensor.gradient import verify_grad

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring_W(n: int) -> sp.csr_matrix:
    """Row-standardized ring-contiguity weight matrix (n × n)."""
    row = np.concatenate([np.arange(n), np.arange(n)])
    col = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
    data = np.ones(2 * n, dtype=np.float64)
    W = sp.csr_matrix((data, (row, col)), shape=(n, n))
    # row-standardize
    d = np.array(W.sum(axis=1)).ravel()
    W = sp.diags(1.0 / d) @ W
    return W.tocsr()


def _kron_ref(rd, ro, W, b):
    """Reference Kronecker solve: eta = (Lo ⊗ Ld)^{-1} b via dense Kronecker.

    A = I - rho_d (I⊗W) - rho_o (W⊗I) + rho_d*rho_o (W⊗W)
      = (I - rho_o W) ⊗ (I - rho_d W) = Lo ⊗ Ld
    """
    n = W.shape[0]
    I = np.eye(n)
    Ld = I - rd * W.toarray()
    Lo = I - ro * W.toarray()
    A = np.kron(Lo, Ld)
    return np.linalg.solve(A, b)


def _kron_ref_matrix(rd, ro, W, B):
    """Reference Kronecker matrix solve column by column."""
    return np.column_stack([_kron_ref(rd, ro, W, B[:, t]) for t in range(B.shape[1])])


def _flow_weight_mats(
    W: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Build unrestricted flow weight matrices from an n x n spatial weight matrix."""
    n = W.shape[0]
    I = sp.eye(n, format="csr")
    Wd = sp.kron(I, W, format="csr")
    Wo = sp.kron(W, I, format="csr")
    Ww = sp.kron(W, W, format="csr")
    return Wd, Wo, Ww


def _unrestricted_ref(rd, ro, rw, W, b):
    """Reference unrestricted flow solve using a dense n^2 x n^2 system."""
    Wd, Wo, Ww = _flow_weight_mats(W)
    N = W.shape[0] ** 2
    A = (sp.eye(N, format="csr") - rd * Wd - ro * Wo - rw * Ww).toarray()
    return np.linalg.solve(A, b)


def _unrestricted_ref_matrix(rd, ro, rw, W, B):
    """Reference unrestricted matrix solve column by column."""
    return np.column_stack(
        [_unrestricted_ref(rd, ro, rw, W, B[:, t]) for t in range(B.shape[1])]
    )


# ---------------------------------------------------------------------------
# SparseFlowSolveOp — numerical correctness
# ---------------------------------------------------------------------------


class TestSparseFlowSolveOp:
    @pytest.mark.parametrize("n", [3, 4])
    def test_matches_reference_unrestricted_solve(self, n):
        from neighbayes._ops import SparseFlowSolveOp

        rng = np.random.default_rng(10)
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rd, ro, rw = 0.2, -0.1, 0.05
        b = rng.normal(size=n * n)

        solve_op = SparseFlowSolveOp(Wd, Wo, Ww)
        rd_t = pt.dscalar("rd")
        ro_t = pt.dscalar("ro")
        rw_t = pt.dscalar("rw")
        b_t = pt.dvector("b")
        eta_t = solve_op(rd_t, ro_t, rw_t, b_t)
        f = pytensor.function([rd_t, ro_t, rw_t, b_t], eta_t)

        got = f(rd, ro, rw, b)
        ref = _unrestricted_ref(rd, ro, rw, W, b)
        np.testing.assert_allclose(got, ref, atol=1e-10)


class TestSparseFlowSolveOpVJP:
    def test_vjp_rho_and_rhs(self):
        from neighbayes._ops import SparseFlowSolveOp

        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rng = np.random.default_rng(11)
        b_val = rng.normal(size=n * n)
        rd_val = np.asarray(0.2)
        ro_val = np.asarray(-0.1)
        rw_val = np.asarray(0.05)

        solve_op = SparseFlowSolveOp(Wd, Wo, Ww)

        def f_scalar(rd, ro, rw, b):
            return pt.sum(solve_op(rd, ro, rw, b))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, rw_val, b_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# SparseFlowSolveMatrixOp — numerical correctness
# ---------------------------------------------------------------------------


class TestSparseFlowSolveMatrixOp:
    @pytest.mark.parametrize("T", [1, 3])
    def test_matches_reference_unrestricted_matrix_solve(self, T):
        from neighbayes._ops import SparseFlowSolveMatrixOp

        n = 3
        rng = np.random.default_rng(12)
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rd, ro, rw = 0.15, 0.1, -0.05
        B = rng.normal(size=(n * n, T))

        solve_op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        rw_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, rw_t, B_t)
        f = pytensor.function([rd_t, ro_t, rw_t, B_t], H_t)

        got = f(rd, ro, rw, B)
        ref = _unrestricted_ref_matrix(rd, ro, rw, W, B)
        np.testing.assert_allclose(got, ref, atol=1e-10)


class TestSparseFlowSolveMatrixOpVJP:
    def test_vjp_rho_and_rhs(self):
        from neighbayes._ops import SparseFlowSolveMatrixOp

        n, T = 3, 2
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        rng = np.random.default_rng(13)
        B_val = rng.normal(size=(n * n, T))
        rd_val = np.asarray(0.15)
        ro_val = np.asarray(0.1)
        rw_val = np.asarray(-0.05)

        solve_op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)

        def f_scalar(rd, ro, rw, B):
            return pt.sum(solve_op(rd, ro, rw, B))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, rw_val, B_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# KroneckerFlowSolveOp — numerical correctness
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveOp:
    @pytest.mark.parametrize("n", [4, 6])
    def test_matches_reference_kron_solve(self, n):
        from neighbayes._ops import KroneckerFlowSolveOp

        rng = np.random.default_rng(0)
        W = _ring_W(n)
        rd, ro = 0.3, -0.2
        b = rng.normal(size=n * n)

        solve_op = KroneckerFlowSolveOp(W, n)
        rd_t = pt.dscalar("rd")
        ro_t = pt.dscalar("ro")
        b_t = pt.dvector("b")
        eta_t = solve_op(rd_t, ro_t, b_t)
        f = pytensor.function([rd_t, ro_t, b_t], eta_t)

        got = f(rd, ro, b)
        ref = _kron_ref(rd, ro, W, b)
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_output_shape(self):
        from neighbayes._ops import KroneckerFlowSolveOp

        n = 5
        W = _ring_W(n)
        rng = np.random.default_rng(1)
        b = rng.normal(size=n * n)

        solve_op = KroneckerFlowSolveOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        b_t = pt.dvector()
        eta_t = solve_op(rd_t, ro_t, b_t)
        f = pytensor.function([rd_t, ro_t, b_t], eta_t)
        assert f(0.1, -0.1, b).shape == (n * n,)


# ---------------------------------------------------------------------------
# KroneckerFlowSolveOp — VJP (verify_grad)
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveOpVJP:
    def test_vjp_rho_d_rho_o(self):
        """verify_grad checks all inputs via finite differences."""
        from neighbayes._ops import KroneckerFlowSolveOp

        n = 3
        W = _ring_W(n)
        rng = np.random.default_rng(2)
        b_val = rng.normal(size=n * n)
        rd_val = np.asarray(0.25)
        ro_val = np.asarray(-0.15)

        solve_op = KroneckerFlowSolveOp(W, n)

        def f_scalar(rd, ro, b):
            # Sum to get a scalar for verify_grad
            return pt.sum(solve_op(rd, ro, b))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, b_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# KroneckerFlowSolveMatrixOp — numerical correctness
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveMatrixOp:
    @pytest.mark.parametrize("T", [1, 3])
    def test_matches_reference(self, T):
        from neighbayes._ops import KroneckerFlowSolveMatrixOp

        n = 4
        rng = np.random.default_rng(3)
        W = _ring_W(n)
        rd, ro = 0.2, 0.1
        B = rng.normal(size=(n * n, T))

        solve_op = KroneckerFlowSolveMatrixOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, B_t)
        f = pytensor.function([rd_t, ro_t, B_t], H_t)

        got = f(rd, ro, B)
        ref = _kron_ref_matrix(rd, ro, W, B)
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_output_shape(self):
        from neighbayes._ops import KroneckerFlowSolveMatrixOp

        n, T = 4, 5
        W = _ring_W(n)
        rng = np.random.default_rng(4)
        B = rng.normal(size=(n * n, T))

        solve_op = KroneckerFlowSolveMatrixOp(W, n)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        B_t = pt.dmatrix()
        H_t = solve_op(rd_t, ro_t, B_t)
        f = pytensor.function([rd_t, ro_t, B_t], H_t)
        assert f(0.1, -0.1, B).shape == (n * n, T)


# ---------------------------------------------------------------------------
# KroneckerFlowSolveMatrixOp — VJP (verify_grad)
# ---------------------------------------------------------------------------


class TestKroneckerFlowSolveMatrixOpVJP:
    def test_vjp_rho_d_rho_o(self):
        from neighbayes._ops import KroneckerFlowSolveMatrixOp

        n, T = 3, 2
        W = _ring_W(n)
        rng = np.random.default_rng(5)
        B_val = rng.normal(size=(n * n, T))
        rd_val = np.asarray(0.2)
        ro_val = np.asarray(-0.1)

        solve_op = KroneckerFlowSolveMatrixOp(W, n)

        def f_scalar(rd, ro, B):
            return pt.sum(solve_op(rd, ro, B))

        verify_grad(
            f_scalar,
            [rd_val, ro_val, B_val],
            rng=rng,
            eps=1e-5,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )


# ---------------------------------------------------------------------------
# Smoke test: logp compiles for SARNegBinFlowSeparable at moderate n
# ---------------------------------------------------------------------------


class TestSeparableNegBinLogpCompiles:
    @pytest.mark.slow
    def test_logp_compiles_n20(self):
        """Check that the Kronecker op compiles inside PyMC at n=20."""
        from neighbayes.graph import flow_weight_matrices
        from neighbayes.models.flow._flow import SARNegBinFlowSeparable

        n = 20
        rng = np.random.default_rng(6)
        pytest.importorskip("libpysal")
        from libpysal.graph import Graph

        focal = np.concatenate([np.arange(n), np.arange(n)])
        neighbor = np.concatenate([(np.arange(n) - 1) % n, (np.arange(n) + 1) % n])
        weights = np.ones(2 * n, dtype=float)
        graph = Graph.from_arrays(focal, neighbor, weights).transform("r")

        y = rng.poisson(5.0, size=(n, n)).astype(np.int64)
        X = rng.normal(size=(n * n, 2))

        model_obj = SARNegBinFlowSeparable(y, X, graph)
        pm_model = model_obj._build_pymc_model()

        # Just test that logp can be evaluated at the initial point
        import pymc as pm

        with pm_model:
            ip = pm_model.initial_point()
            lp = pm_model.compile_logp()(ip)
        assert np.isfinite(lp)


# ---------------------------------------------------------------------------
# SparseSARSolveOp tests
# ---------------------------------------------------------------------------


class TestSparseSARSolveOp:
    """Tests for the cross-sectional SAR sparse solve Op."""

    def test_matches_reference_solve(self):
        """SparseSARSolveOp output matches numpy.linalg.solve."""
        from neighbayes._ops import SparseSARSolveOp

        n = 20
        W = _ring_W(n)
        W_dense = W.toarray()
        rng = np.random.default_rng(42)
        b = rng.standard_normal(n)
        rho = 0.4

        # Reference: dense solve
        A_ref = np.eye(n) - rho * W_dense
        eta_ref = np.linalg.solve(A_ref, b)

        # Op: sparse solve
        op = SparseSARSolveOp(W)
        rho_pt = pt.scalar("rho")
        b_pt = pt.vector("b")
        eta_pt = op(rho_pt, b_pt)
        fn = pytensor.function([rho_pt, b_pt], eta_pt)
        eta_op = fn(np.float64(rho), b)

        np.testing.assert_allclose(eta_op, eta_ref, rtol=1e-10, atol=1e-12)

    def test_vjp_rho_and_b(self):
        """VJP (gradient) w.r.t. rho and b is numerically correct."""
        from neighbayes._ops import SparseSARSolveOp

        n = 10
        W = _ring_W(n)
        rng = np.random.default_rng(123)
        b = rng.standard_normal(n).astype(np.float64)
        rho_val = np.float64(0.3)

        op = SparseSARSolveOp(W)

        # Check gradient w.r.t. rho via finite differences
        rho_pt = pt.scalar("rho")
        b_pt = pt.vector("b")
        eta = op(rho_pt, b_pt)
        loss = eta.sum()
        grad_rho = pytensor.grad(loss, rho_pt)
        grad_b = pytensor.grad(loss, b_pt)
        fn = pytensor.function([rho_pt, b_pt], [loss, grad_rho, grad_b])
        _, grad_rho_num, grad_b_num = fn(rho_val, b)

        # Finite difference check for rho
        eps = 1e-5
        f_plus, _, _ = fn(rho_val + eps, b)
        f_minus, _, _ = fn(rho_val - eps, b)
        grad_rho_fd = (f_plus - f_minus) / (2 * eps)
        np.testing.assert_allclose(
            float(grad_rho_num), grad_rho_fd, rtol=1e-4, atol=1e-4
        )

        # Finite difference check for b
        grad_b_fd = np.zeros_like(b)
        for i in range(n):
            b_plus = b.copy()
            b_plus[i] += eps
            b_minus = b.copy()
            b_minus[i] -= eps
            f_plus_b, _, _ = fn(rho_val, b_plus)
            f_minus_b, _, _ = fn(rho_val, b_minus)
            grad_b_fd[i] = (f_plus_b - f_minus_b) / (2 * eps)
        np.testing.assert_allclose(grad_b_num, grad_b_fd, rtol=1e-4, atol=1e-4)


class TestOptionalSparseBackends:
    def test_selects_klu_when_requested_and_installed(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes._ops import _select_sparse_backend

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        _select_sparse_backend.cache_clear()
        assert _select_sparse_backend() == "klu"

    def test_sparse_vector_solver_routes_to_klu_backend(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "0")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            assert backend == "klu"
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        # Immediate solves use the cached working factor directly (no copy).
        monkeypatch.setattr(ops_mod._backend, "_refactor", _fake_sparse_factor)

        A = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64))
        rhs = np.array([1.0, 2.0], dtype=np.float64)
        got = ops_mod._solve_sparse_vector(A, rhs)
        ref = np.linalg.solve(A.toarray(), rhs)

        assert called["klu"] >= 1
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_sparse_flow_solver_routes_to_klu_backend(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod
        from neighbayes._ops import SparseFlowSolveOp

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "0")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            assert backend == "klu"
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        monkeypatch.setattr(ops_mod._backend, "_sparse_factor", _fake_sparse_factor)

        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveOp(Wd, Wo, Ww)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        rw_t = pt.dscalar()
        b_t = pt.dvector()
        f = pytensor.function([rd_t, ro_t, rw_t, b_t], op(rd_t, ro_t, rw_t, b_t))

        b = np.arange(1, n * n + 1, dtype=np.float64)
        got = f(0.2, -0.1, 0.05, b)
        ref = _unrestricted_ref(0.2, -0.1, 0.05, W, b)

        assert called["klu"] >= 1
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_sparse_flow_matrix_solver_routes_to_klu_backend(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod
        from neighbayes._ops import SparseFlowSolveMatrixOp

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            assert backend == "klu"
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        monkeypatch.setattr(ops_mod._backend, "_sparse_factor", _fake_sparse_factor)

        n, T = 3, 2
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        rw_t = pt.dscalar()
        B_t = pt.dmatrix()
        f = pytensor.function([rd_t, ro_t, rw_t, B_t], op(rd_t, ro_t, rw_t, B_t))

        B = np.arange(1, n * n * T + 1, dtype=np.float64).reshape(n * n, T)
        got = f(0.15, 0.1, -0.05, B)
        ref = _unrestricted_ref_matrix(0.15, 0.1, -0.05, W, B)

        # sksparse KLU factorizes once and batch-solves the T columns.
        assert called["klu"] >= 1
        np.testing.assert_allclose(got, ref, atol=1e-10)

    def test_sparse_sar_forward_reuses_klu_factorization(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod
        from neighbayes._ops import SparseSARSolveOp

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "0")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        monkeypatch.setattr(ops_mod._backend, "_sparse_factor", _fake_sparse_factor)

        n = 8
        W = _ring_W(n)
        op = SparseSARSolveOp(W)
        b = np.linspace(1.0, 2.0, n, dtype=np.float64)

        got1 = op._solve_forward(0.25, b)
        got2 = op._solve_forward(0.25, b)

        assert called["klu"] == 1
        np.testing.assert_allclose(got1, got2, atol=1e-12)

    def test_sparse_sar_adjoint_reuses_klu_factorization(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod
        from neighbayes._ops import _SparseSARVJPOp

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "0")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        monkeypatch.setattr(ops_mod._backend, "_sparse_factor", _fake_sparse_factor)

        n = 8
        W = _ring_W(n)
        op = _SparseSARVJPOp(W)
        g = np.linspace(1.0, 2.0, n, dtype=np.float64)

        got1 = op._solve_adjoint(0.25, g)
        got2 = op._solve_adjoint(0.25, g)

        assert called["klu"] == 1
        np.testing.assert_allclose(got1, got2, atol=1e-12)


class TestFlowSparseLUCache:
    def test_sparse_flow_scipy_lu_reused_for_same_rhos(self, monkeypatch):
        from neighbayes import _ops as ops_mod
        from neighbayes._ops import SparseFlowSolveOp

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "scipy")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        ops_mod._select_sparse_backend.cache_clear()

        calls = {"splu": 0}
        orig_splu = ops_mod.sp.linalg.splu

        def _counting_splu(*args, **kwargs):
            calls["splu"] += 1
            return orig_splu(*args, **kwargs)

        monkeypatch.setattr(ops_mod.sp.linalg, "splu", _counting_splu)

        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveOp(Wd, Wo, Ww)
        rd_t = pt.dscalar()
        ro_t = pt.dscalar()
        rw_t = pt.dscalar()
        b_t = pt.dvector()
        f = pytensor.function([rd_t, ro_t, rw_t, b_t], op(rd_t, ro_t, rw_t, b_t))

        b = np.linspace(-1.0, 1.0, n * n)
        _ = f(0.2, -0.1, 0.05, b)
        _ = f(0.2, -0.1, 0.05, b)

        assert calls["splu"] == 1

    @staticmethod
    def _counting_klu_env(monkeypatch):
        """Force the KLU sparse path and return a factor-call counter."""
        pytest.importorskip("sksparse.klu")
        from neighbayes import _ops as ops_mod

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "klu")
        monkeypatch.setenv("NEIGHBAYES_SPARSE_STRICT", "1")
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "0")
        ops_mod._select_sparse_backend.cache_clear()

        called = {"klu": 0}

        def _fake_sparse_factor(A_csc, backend):
            called["klu"] += 1
            assert backend == "klu"
            A_dense = A_csc.toarray()

            class _F:
                def solve(self, rhs):
                    return np.linalg.solve(A_dense, np.asarray(rhs))

            return _F()

        monkeypatch.setattr(ops_mod._backend, "_sparse_factor", _fake_sparse_factor)
        return called

    def test_sparse_flow_forward_reuses_klu_factorization(self, monkeypatch):
        from neighbayes._ops import SparseFlowSolveOp

        called = self._counting_klu_env(monkeypatch)
        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveOp(Wd, Wo, Ww)
        b = np.linspace(1.0, 2.0, n * n, dtype=np.float64)

        got1 = op._solve_forward(0.2, -0.1, 0.05, b)
        got2 = op._solve_forward(0.2, -0.1, 0.05, b)

        assert called["klu"] == 1
        np.testing.assert_allclose(got1, got2, atol=1e-12)

    def test_sparse_flow_adjoint_reuses_klu_factorization(self, monkeypatch):
        from neighbayes._ops import SparseFlowSolveOp

        called = self._counting_klu_env(monkeypatch)
        n = 3
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveOp(Wd, Wo, Ww)
        g = np.linspace(1.0, 2.0, n * n, dtype=np.float64)

        got1 = op._vjp_op._solve_adjoint(0.2, -0.1, 0.05, g)
        got2 = op._vjp_op._solve_adjoint(0.2, -0.1, 0.05, g)

        assert called["klu"] == 1
        np.testing.assert_allclose(got1, got2, atol=1e-12)

    def test_sparse_flow_matrix_reuses_klu_factorization(self, monkeypatch):
        from neighbayes._ops import SparseFlowSolveMatrixOp

        called = self._counting_klu_env(monkeypatch)
        n, T = 3, 2
        W = _ring_W(n)
        Wd, Wo, Ww = _flow_weight_mats(W)
        op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
        B = np.arange(1, n * n * T + 1, dtype=np.float64).reshape(n * n, T)

        got1 = op._solve_forward_matrix(0.15, 0.1, -0.05, B)
        got2 = op._solve_forward_matrix(0.15, 0.1, -0.05, B)
        v1 = op._vjp_op._solve_adjoint_matrix(0.15, 0.1, -0.05, B)
        v2 = op._vjp_op._solve_adjoint_matrix(0.15, 0.1, -0.05, B)

        # One factorization each for the forward and the adjoint system,
        # reused across the repeat call at the same rhos.
        assert called["klu"] == 2
        np.testing.assert_allclose(got1, got2, atol=1e-12)
        np.testing.assert_allclose(v1, v2, atol=1e-12)


class TestSparseSARSolveOpNumbaDispatch:
    def test_numba_dense_path_matches_default(self, monkeypatch):
        pytest.importorskip("numba")
        from neighbayes._ops import SparseSARSolveOp

        # Keep dense path active for this small n.
        monkeypatch.delenv("NEIGHBAYES_KRON_DENSE_MAX", raising=False)

        n = 8
        W = _ring_W(n)
        rng = np.random.default_rng(2026)
        b = rng.standard_normal(n)

        op = SparseSARSolveOp(W)
        rho = pt.scalar("rho")
        b_t = pt.vector("b")
        eta = op(rho, b_t)

        f_default = pytensor.function([rho, b_t], eta)
        f_numba = pytensor.function([rho, b_t], eta, mode="NUMBA")

        np.testing.assert_allclose(
            f_numba(0.2, b),
            f_default(0.2, b),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_numba_sparse_path_has_no_pytensor_fallback_warning(self, monkeypatch):
        pytest.importorskip("numba")
        from neighbayes._ops import SparseSARSolveOp

        # Force sparse path by lowering dense threshold below n.
        monkeypatch.setenv("NEIGHBAYES_KRON_DENSE_MAX", "2")

        n = 10
        W = _ring_W(n)
        rng = np.random.default_rng(2027)
        b = rng.standard_normal(n)

        op = SparseSARSolveOp(W)
        rho = pt.scalar("rho")
        b_t = pt.vector("b")
        eta = op(rho, b_t)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f_numba = pytensor.function([rho, b_t], eta, mode="NUMBA")
            _ = f_numba(0.2, b)

        msgs = [str(w.message) for w in caught]
        assert not any("Numba will use object mode to run" in m for m in msgs)


class TestSparseFactorCache:
    """The scikit-sparse fallback reuses one symbolic analysis per pattern."""

    @staticmethod
    def _system(n=300, seed=0):
        W = sp.random(n, n, density=0.02, random_state=seed, format="csc")
        W = sp.diags(1.0 / np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)) @ W
        eye = sp.eye(n, format="csc")
        return lambda rho: (eye - rho * W).tocsc()

    @pytest.mark.parametrize("backend", ["klu", "umfpack"])
    def test_refactor_reuses_analysis_and_copies_are_private(
        self, monkeypatch, backend
    ):
        pytest.importorskip(f"sksparse.{backend}")
        from neighbayes._ops import _backend

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", backend)
        A = self._system()
        b = np.random.default_rng(1).standard_normal(A(0.3).shape[0])

        held = _backend._sparse_factor(A(0.3), backend)
        work = _backend._refactor(A(0.3), backend)
        # Same pattern: the working factor is reused, refactored in place.
        assert _backend._refactor(A(0.6), backend) is work
        np.testing.assert_allclose(A(0.6) @ work.solve(b), b, atol=1e-12)
        # The copy handed out earlier still solves at its own values.
        np.testing.assert_allclose(A(0.3) @ held.solve(b), b, atol=1e-12)

    def test_auto_route_returns_a_working_factor(self, monkeypatch):
        pytest.importorskip("sksparse.klu")
        pytest.importorskip("sksparse.umfpack")
        from neighbayes._ops import _backend

        monkeypatch.setenv("NEIGHBAYES_SPARSE_BACKEND", "auto")
        A = self._system(seed=2)
        b = np.ones(A(0.4).shape[0])
        x, logdet = _backend._factor_solve_logdet(A(0.4), b)
        np.testing.assert_allclose(A(0.4) @ x, b, atol=1e-12)
        np.testing.assert_allclose(
            logdet, np.linalg.slogdet(A(0.4).toarray())[1], rtol=1e-10
        )


@pytest.mark.parametrize("backend", ["klu", "umfpack"])
def test_sparse_solves_accept_read_only_rhs(backend):
    """JAX ``pure_callback`` hands host code read-only views of its arrays.

    scikit-sparse solves take typed memoryviews, which reject read-only
    buffers; the UMFPACK path raised ``buffer source array is read-only``
    from every JAX host callback until the right-hand side was made writable.
    """
    pytest.importorskip(f"sksparse.{backend}")
    from neighbayes._ops._backend import _make_cached_sparse_solver

    rng = np.random.default_rng(0)
    A = sp.random(30, 30, density=0.15, random_state=1, format="csc") + 4 * sp.eye(30)
    for rhs in (rng.normal(size=30), rng.normal(size=(30, 3))):
        ro = rhs.copy()
        ro.flags.writeable = False
        solver = _make_cached_sparse_solver(A.tocsc(), backend)
        np.testing.assert_allclose(A @ solver.solve(ro), rhs, atol=1e-10)
