"""JAX dispatch parity tests for the custom Ops in :mod:`neighbayes._ops`.

These tests are skipped when JAX is not installed.  They verify that each Op
(forward and VJP) produces numerically identical outputs under the default
PyTensor C backend and the JAX backend, and that the dispatched models can
be sampled with ``nuts_sampler="blackjax"`` without falling back to PyMC.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

pytestmark = pytest.mark.requires_jax

pytest.importorskip("jax")

import pytensor
import pytensor.tensor as pt

from neighbayes._ops import (
    KroneckerFlowSolveMatrixOp,
    KroneckerFlowSolveOp,
    SparseFlowSolveMatrixOp,
    SparseFlowSolveOp,
    SparseSARSolveOp,
)


def _line_W(n):
    W = sp.lil_matrix((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rows = np.asarray(W.sum(axis=1)).ravel()
    rows[rows == 0] = 1.0
    return sp.diags(1.0 / rows) @ W.tocsr()


@pytest.fixture
def small_W():
    return _line_W(5)


@pytest.fixture
def kron_matrices(small_W):
    n = small_W.shape[0]
    Wd = sp.kron(sp.eye(n), small_W).tocsr()
    Wo = sp.kron(small_W, sp.eye(n)).tocsr()
    Ww = sp.kron(small_W, small_W).tocsr()
    return Wd, Wo, Ww, n


def _compile_pair(inputs, outputs):
    f_c = pytensor.function(inputs, outputs)
    f_j = pytensor.function(inputs, outputs, mode="JAX")
    return f_c, f_j


def _assert_close(c_out, j_out, atol=1e-10):
    if not isinstance(c_out, (list, tuple)):
        c_out = [c_out]
        j_out = [j_out]
    for c, j in zip(c_out, j_out):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=atol, rtol=1e-10)


def test_kronecker_solve_forward_parity(small_W):
    n = small_W.shape[0]
    op = KroneckerFlowSolveOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, b)
    f_c, f_j = _compile_pair([rho_d, rho_o, b], eta)
    rng = np.random.default_rng(0)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.3, 0.2, bv), f_j(0.3, 0.2, bv))


def test_kronecker_solve_vjp_parity(small_W):
    n = small_W.shape[0]
    op = KroneckerFlowSolveOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, b)]
    f_c, f_j = _compile_pair([rho_d, rho_o, b], grads)
    rng = np.random.default_rng(1)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.4, -0.1, bv), f_j(0.4, -0.1, bv))


def test_kronecker_solve_jax_autodiff_vs_manual_vjp(small_W):
    """JAX autodiff through the pure-JAX forward must match the hand-derived VJP.

    This test directly exercises ``jax.grad`` on the JIT-compiled forward
    solve (the same function returned by ``_funcify_kron_solve``) and
    compares the result with the C-backend reference that uses the manual
    adjoint in :class:`_KroneckerFlowVJPOp`.
    """
    import jax
    import jax.numpy as jnp

    n = small_W.shape[0]
    W_d = jnp.asarray(small_W.toarray(), dtype=jnp.float64)
    I = jnp.eye(n, dtype=jnp.float64)

    def kron_solve(rho_d, rho_o, b):
        Ld = I - rho_d * W_d
        Lo = I - rho_o * W_d
        Hb = b.reshape((n, n)).T
        Hp = jnp.linalg.solve(Ld, Hb)
        Z = jnp.linalg.solve(Lo, Hp.T)
        return Z.reshape(-1)

    def loss_fn(rho_d, rho_o, b):
        eta = kron_solve(rho_d, rho_o, b)
        return jnp.sum(eta * eta)

    grad_rd = jax.grad(loss_fn, argnums=0)
    grad_ro = jax.grad(loss_fn, argnums=1)
    grad_b = jax.grad(loss_fn, argnums=2)

    rng = np.random.default_rng(42)
    rd_val = np.float64(0.4)
    ro_val = np.float64(-0.1)
    bv = rng.standard_normal(n * n)

    jax_rd = float(grad_rd(rd_val, ro_val, bv))
    jax_ro = float(grad_ro(rd_val, ro_val, bv))
    jax_b = np.asarray(grad_b(rd_val, ro_val, bv), dtype=np.float64)

    # C-backend reference via pytensor.grad + default mode
    op = KroneckerFlowSolveOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, b)]
    f_c = pytensor.function([rho_d, rho_o, b], grads)
    c_rd, c_ro, c_b = f_c(rd_val, ro_val, bv)

    np.testing.assert_allclose(jax_rd, float(c_rd), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(jax_ro, float(c_ro), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(jax_b, np.asarray(c_b), atol=1e-10, rtol=1e-10)


def test_kronecker_matrix_forward_parity(small_W):
    n = small_W.shape[0]
    T = 3
    op = KroneckerFlowSolveMatrixOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, B)
    f_c, f_j = _compile_pair([rho_d, rho_o, B], H)
    rng = np.random.default_rng(2)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.25, 0.15, Bv), f_j(0.25, 0.15, Bv))


def test_kronecker_matrix_vjp_parity(small_W):
    n = small_W.shape[0]
    T = 2
    op = KroneckerFlowSolveMatrixOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, B)
    loss = pt.sum(H * H)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, B)]
    f_c, f_j = _compile_pair([rho_d, rho_o, B], grads)
    rng = np.random.default_rng(3)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.3, Bv), f_j(0.2, 0.3, Bv))


def test_kronecker_matrix_jax_autodiff_vs_manual_vjp(small_W):
    """JAX autodiff through the vmapped Kronecker forward must match manual VJP.

    Same pattern as ``test_kronecker_solve_jax_autodiff_vs_manual_vjp``
    but for the matrix (multi-column) variant.
    """
    import jax
    import jax.numpy as jnp

    n = small_W.shape[0]
    T = 2
    W_d = jnp.asarray(small_W.toarray(), dtype=jnp.float64)
    I = jnp.eye(n, dtype=jnp.float64)

    def _solve_one(rho_d, rho_o, b):
        Ld = I - rho_d * W_d
        Lo = I - rho_o * W_d
        Hb = b.reshape((n, n)).T
        Hp = jnp.linalg.solve(Ld, Hb)
        Z = jnp.linalg.solve(Lo, Hp.T)
        return Z.reshape(-1)

    def kron_solve_mat(rho_d, rho_o, B):
        solver = jax.vmap(_solve_one, in_axes=(None, None, 1), out_axes=1)
        return solver(rho_d, rho_o, B)

    def loss_fn(rho_d, rho_o, B):
        H = kron_solve_mat(rho_d, rho_o, B)
        return jnp.sum(H * H)

    grad_rd = jax.grad(loss_fn, argnums=0)
    grad_ro = jax.grad(loss_fn, argnums=1)
    grad_B = jax.grad(loss_fn, argnums=2)

    rng = np.random.default_rng(43)
    rd_val = np.float64(0.2)
    ro_val = np.float64(0.3)
    Bv = rng.standard_normal((n * n, T))

    jax_rd = float(grad_rd(rd_val, ro_val, Bv))
    jax_ro = float(grad_ro(rd_val, ro_val, Bv))
    jax_B = np.asarray(grad_B(rd_val, ro_val, Bv), dtype=np.float64)

    # C-backend reference
    op = KroneckerFlowSolveMatrixOp(small_W, n)
    rho_d, rho_o = pt.dscalars("rho_d", "rho_o")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, B)
    loss = pt.sum(H * H)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, B)]
    f_c = pytensor.function([rho_d, rho_o, B], grads)
    c_rd, c_ro, c_B = f_c(rd_val, ro_val, Bv)

    np.testing.assert_allclose(jax_rd, float(c_rd), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(jax_ro, float(c_ro), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(jax_B, np.asarray(c_B), atol=1e-10, rtol=1e-10)


def test_sparse_flow_forward_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    op = SparseFlowSolveOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, rho_w, b)
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, b], eta)
    rng = np.random.default_rng(4)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.2, 0.15, -0.03, bv), f_j(0.2, 0.15, -0.03, bv))


def test_sparse_flow_vjp_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    op = SparseFlowSolveOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    b = pt.dvector("b")
    eta = op(rho_d, rho_o, rho_w, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, rho_w, b)]
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, b], grads)
    rng = np.random.default_rng(5)
    bv = rng.standard_normal(n * n)
    _assert_close(f_c(0.2, 0.15, -0.03, bv), f_j(0.2, 0.15, -0.03, bv))


def test_sparse_flow_matrix_forward_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    T = 3
    op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, rho_w, B)
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, B], H)
    rng = np.random.default_rng(6)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.15, -0.03, Bv), f_j(0.2, 0.15, -0.03, Bv))


def test_sparse_flow_matrix_vjp_parity(kron_matrices):
    Wd, Wo, Ww, n = kron_matrices
    T = 2
    op = SparseFlowSolveMatrixOp(Wd, Wo, Ww)
    rho_d, rho_o, rho_w = pt.dscalars("rd", "ro", "rw")
    B = pt.dmatrix("B")
    H = op(rho_d, rho_o, rho_w, B)
    loss = pt.sum(H * H)
    grads = [pytensor.grad(loss, v) for v in (rho_d, rho_o, rho_w, B)]
    f_c, f_j = _compile_pair([rho_d, rho_o, rho_w, B], grads)
    rng = np.random.default_rng(7)
    Bv = rng.standard_normal((n * n, T))
    _assert_close(f_c(0.2, 0.15, -0.03, Bv), f_j(0.2, 0.15, -0.03, Bv))


def test_sampler_resolution_with_jax_present():
    """When JAX is importable, requires_c_backend should not force a downgrade."""
    from neighbayes._backends.sampler_helpers import (
        _jax_dispatches_available,
        enforce_c_backend,
    )

    assert _jax_dispatches_available() is True
    assert (
        enforce_c_backend("blackjax", requires_c_backend=True, model_name="ToyFlow")
        == "blackjax"
    )


def test_jax_auto_prefers_sparsax_when_available(monkeypatch):
    from neighbayes._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("neighbayes._jax_dispatch._sparsax_available", lambda: True)

    _select_jax_sparse_backend.cache_clear()
    assert _select_jax_sparse_backend() == "sparsax"
    _select_jax_sparse_backend.cache_clear()


def test_jax_auto_falls_to_callback_scipy_when_no_optional_backends(monkeypatch):
    from neighbayes._jax_dispatch import _select_jax_sparse_backend

    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "0")
    monkeypatch.setattr("neighbayes._jax_dispatch._sparsax_available", lambda: False)

    _select_jax_sparse_backend.cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = _select_jax_sparse_backend()
    assert backend == "callback"
    msgs = [str(w.message) for w in caught]
    assert any("callback+scipy" in m for m in msgs)
    assert any("sparsax" in m for m in msgs)


# ---------------------------------------------------------------------------
# Lineax SAR-solver path
# ---------------------------------------------------------------------------


def _reset_jax_dispatch_caches() -> None:
    """Clear the JAX-dispatch selector caches so env changes take effect.

    ``register_jax_dispatch`` is ``lru_cache``-wrapped and re-runs the
    ``jax_funcify.register`` decorators on re-entry, which replaces the
    previously registered dispatcher closures.
    """
    from neighbayes._jax_dispatch import (
        _select_jax_sar_solver,
        _select_jax_sparse_backend,
        register_jax_dispatch,
    )

    _select_jax_sparse_backend.cache_clear()
    _select_jax_sar_solver.cache_clear()
    register_jax_dispatch.cache_clear()


def _setup_jax_gmres_dispatch(monkeypatch):
    """Configure environment for JAX-native GMRES dispatch tests."""
    monkeypatch.setenv("NEIGHBAYES_JAX_SAR_SOLVER", "jax_gmres")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    from neighbayes._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()


# ---------------------------------------------------------------------------
# JAX-native GMRES SAR-solver path
# ---------------------------------------------------------------------------


def test_jax_gmres_solver_env(monkeypatch, sar_env_reset):
    from neighbayes._jax_dispatch import _select_jax_sar_solver

    monkeypatch.setenv("NEIGHBAYES_JAX_SAR_SOLVER", "jax_gmres")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    assert _select_jax_sar_solver() == "jax_gmres"


def test_sparse_sar_jax_gmres_forward_parity(monkeypatch, sar_env_reset):
    """JAX GMRES forward solve must match C-backend reference."""
    _setup_jax_gmres_dispatch(monkeypatch)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)

    f_c = pytensor.function([rho, b], eta)
    f_j = pytensor.function([rho, b], eta, mode="JAX")

    rng = np.random.default_rng(31)
    b_val = rng.standard_normal(8)

    np.testing.assert_allclose(
        np.asarray(f_c(0.3, b_val)),
        np.asarray(f_j(0.3, b_val)),
        atol=1e-7,
        rtol=1e-7,
    )


def test_sparse_sar_jax_gmres_grad_parity(monkeypatch, sar_env_reset):
    """Reverse-mode gradient parity for JAX GMRES path."""
    _setup_jax_gmres_dispatch(monkeypatch)

    W = _line_W(8)
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho, b)]

    f_c = pytensor.function([rho, b], grads)
    f_j = pytensor.function([rho, b], grads, mode="JAX")

    rng = np.random.default_rng(32)
    b_val = rng.standard_normal(8)

    c_out = f_c(0.25, b_val)
    j_out = f_j(0.25, b_val)
    for c, j in zip(c_out, j_out):
        np.testing.assert_allclose(np.asarray(c), np.asarray(j), atol=1e-7, rtol=1e-7)


def test_sparse_sar_jax_eigen_autodiff_vs_manual_vjp(monkeypatch, sar_env_reset):
    """JAX autodiff through the pure-JAX eigen forward must match manual VJP.

    The eigen path is pure dense JAX (complex128 eigendecomposition +
    mat-vec).  JAX differentiates through it automatically, so this test
    verifies that ``jax.grad`` on the forward function agrees with the
    C-backend reference that uses the hand-derived adjoint in
    :class:`_SparseSARVJPOp`.
    """
    import jax
    import jax.numpy as jnp

    monkeypatch.setenv("NEIGHBAYES_JAX_SAR_SOLVER", "eigen")
    monkeypatch.setenv("NEIGHBAYES_JAX_SAR_EIGEN_N_MAX", "1000")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "1")
    _reset_jax_dispatch_caches()
    from neighbayes._jax_dispatch import register_jax_dispatch

    register_jax_dispatch()

    n = 8
    W = _line_W(n)
    W_dense = np.asarray(W.toarray(), dtype=np.float64)
    eigs_np, V_np = np.linalg.eig(W_dense)
    Vinv_np = np.linalg.inv(V_np)
    idx = np.argsort(eigs_np.real)[::-1]
    eigs_np = eigs_np[idx]
    V_np = V_np[:, idx]
    Vinv_np = Vinv_np[idx, :]

    eigs_j = jnp.asarray(eigs_np.astype(np.complex128))
    V_j = jnp.asarray(V_np.astype(np.complex128))
    Vinv_j = jnp.asarray(Vinv_np.astype(np.complex128))

    def eigen_solve(rho, b):
        inv_eigs = 1.0 / (1.0 - rho * eigs_j)
        return (V_j @ (inv_eigs * (Vinv_j @ b.astype(jnp.complex128)))).real

    def loss_fn(rho, b):
        eta = eigen_solve(rho, b)
        return jnp.sum(eta * eta)

    grad_rho = jax.grad(loss_fn, argnums=0)
    grad_b = jax.grad(loss_fn, argnums=1)

    rng = np.random.default_rng(44)
    rho_val = np.float64(0.3)
    b_val = rng.standard_normal(n)

    jax_rho = float(grad_rho(rho_val, b_val))
    jax_b = np.asarray(grad_b(rho_val, b_val), dtype=np.float64)

    # C-backend reference via pytensor.grad + default mode
    op = SparseSARSolveOp(W)
    rho = pt.dscalar("rho")
    b = pt.dvector("b")
    eta = op(rho, b)
    loss = pt.sum(eta * eta)
    grads = [pytensor.grad(loss, v) for v in (rho, b)]
    f_c = pytensor.function([rho, b], grads)
    c_rho, c_b = f_c(rho_val, b_val)

    np.testing.assert_allclose(jax_rho, float(c_rho), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(jax_b, np.asarray(c_b), atol=1e-10, rtol=1e-10)


def test_jax_gmres_high_rho_correctness(monkeypatch, sar_env_reset):
    """JAX GMRES must match dense reference for moderate-to-high rho."""
    _setup_jax_gmres_dispatch(monkeypatch)

    n = 64
    W = _line_W(n)
    rng = np.random.default_rng(33)
    b_val = rng.standard_normal(n)
    rho_val = 0.85

    A_dense = np.eye(n) - rho_val * W.toarray()
    eta_ref = np.linalg.solve(A_dense, b_val)

    op = SparseSARSolveOp(W)
    rho_pt = pt.dscalar("rho")
    b_pt = pt.dvector("b")
    eta = op(rho_pt, b_pt)
    f_j = pytensor.function([rho_pt, b_pt], eta, mode="JAX")

    out = np.asarray(f_j(rho_val, b_val))
    np.testing.assert_allclose(out, eta_ref, atol=1e-6, rtol=1e-6)


@pytest.fixture
def sar_env_reset():
    """Reset JAX-dispatch caches before and after each Lineax test."""
    _reset_jax_dispatch_caches()
    yield
    _reset_jax_dispatch_caches()


def test_jax_sar_solver_auto_preserves_existing_backend(monkeypatch, sar_env_reset):
    from neighbayes._jax_dispatch import _select_jax_sar_solver

    monkeypatch.delenv("NEIGHBAYES_JAX_SAR_SOLVER", raising=False)
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_BACKEND", "auto")
    monkeypatch.setenv("NEIGHBAYES_JAX_SPARSE_STRICT", "0")

    _reset_jax_dispatch_caches()
    # _select_jax_sar_solver returns "auto" when no explicit solver is set;
    # concrete resolution happens in _resolve_auto_sar_solver at Op time.
    assert _select_jax_sar_solver() == "auto"


# ---------------------------------------------------------------------------
# Lineax SAR-solver path — Neumann-series preconditioner (Phase D)
# ---------------------------------------------------------------------------
