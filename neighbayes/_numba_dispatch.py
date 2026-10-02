"""Numba dispatch registrations for selected custom Ops in :mod:`neighbayes._ops`.

Targets the cross-sectional SAR sparse solve family
(:class:`~neighbayes._ops.SparseSARSolveOp`, :class:`~neighbayes._ops._SparseSARVJPOp`)
and the Kronecker-factored NB flow Ops
(:class:`~neighbayes._ops.KroneckerFlowSolveOp`,
:class:`~neighbayes._ops._KroneckerFlowVJPOp`,
:class:`~neighbayes._ops.KroneckerFlowSolveMatrixOp`,
:class:`~neighbayes._ops._KroneckerFlowVJPMatrixOp`).

When a dense ``W`` view is available (small ``n``), the cross-sectional SAR
and the *vector* Kronecker Ops use pure Numba nopython kernels based on
``np.linalg.solve``.  Other paths (Kronecker matrix variants with batched
Fortran-order reshapes; sparse SAR with no dense ``W``) register explicit
:func:`numba.objmode` wrappers around each Op's existing ``perform``
implementation.  Either branch suppresses PyTensor's generic
``"Numba will use object mode to run …'s perform method"`` warning while
preserving correctness.

The general 3-rho sparse NB flow Ops (``SparseFlowSolveOp`` family) are
intentionally **not** registered: PyTensor's default per-Op object-mode
fallback works correctly downstream, whereas an explicit objmode wrapper over
the multi-output ``_SparseFlowVJP*`` Ops drops the static shape information
that PyTensor's elemwise codegen requires (``input_bc_patterns must be
literal``).  The cost is one informational PyTensor warning per Op for a path
that already bottlenecks on a sparse SuperLU factorization per gradient
evaluation.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def _numba_available() -> bool:
    """Return ``True`` if Numba and PyTensor's Numba dispatch are importable."""
    return (
        importlib.util.find_spec("numba") is not None
        and importlib.util.find_spec("pytensor.link.numba.dispatch.basic") is not None
    )


@lru_cache(maxsize=1)
def register_numba_dispatch() -> bool:
    """Register Numba dispatches for SAR sparse solve Ops.

    Idempotent (cached). Returns ``True`` if registration ran, ``False`` if
    Numba dispatch is not available.
    """
    if not _numba_available():
        return False

    import numba
    import numpy as np
    from pytensor.link.numba.dispatch.basic import numba_funcify

    from ._ops import (
        KroneckerFlowSolveMatrixOp,
        KroneckerFlowSolveOp,
        SparseSARSolveOp,
        _KroneckerFlowVJPMatrixOp,
        _KroneckerFlowVJPOp,
        _SparseSARVJPOp,
    )

    # A scalar Op output (a gradient w.r.t. rho) must reach PyTensor as a 0-d
    # array; a bare float passes through the Numba backend unconverted.
    _SCALAR = numba.types.Array(numba.types.float64, 0, "C")

    @numba_funcify.register(SparseSARSolveOp)
    def _funcify_sparse_sar_solve(op, **kwargs):
        n = op._n
        use_dense = op._W_dense is not None

        if use_dense:
            W_dense = np.asarray(op._W_dense, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def sparse_sar_solve(rho, b):
                A = I - rho * W_dense
                return np.linalg.solve(A, b)

            return sparse_sar_solve

        ret_sig = numba.types.float64[:]

        def _py_solve(rho, b):
            outputs = [[None]]
            op.perform(None, [np.asarray(rho), np.asarray(b)], outputs)
            return outputs[0][0]

        @numba.njit
        def sparse_sar_solve(rho, b):
            with numba.objmode(ret=ret_sig):
                ret = _py_solve(rho, b)
            return ret

        return sparse_sar_solve

    @numba_funcify.register(_SparseSARVJPOp)
    def _funcify_sparse_sar_vjp(op, **kwargs):
        n = op._n
        use_dense = op._W_dense is not None

        if use_dense:
            W_dense = np.asarray(op._W_dense, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def sparse_sar_vjp(rho, eta, g):
                A_t = (I - rho * W_dense).T
                v = np.linalg.solve(A_t, g)
                grad_rho = np.dot(v, W_dense @ eta)
                # 0-d arrays, as PyTensor expects for scalar outputs.
                return np.asarray(grad_rho), v

            return sparse_sar_vjp

        ret_sig = numba.types.Tuple((_SCALAR, numba.types.float64[:]))

        def _py_vjp(rho, eta, g):
            outputs = [[None], [None]]
            op.perform(None, [np.asarray(rho), np.asarray(eta), np.asarray(g)], outputs)
            return outputs[0][0], outputs[1][0]

        @numba.njit
        def sparse_sar_vjp(rho, eta, g):
            with numba.objmode(ret=ret_sig):
                ret = _py_vjp(rho, eta, g)
            return ret

        return sparse_sar_vjp

    # ------------------------------------------------------------------
    # Kronecker NB-flow Ops
    # ------------------------------------------------------------------
    #
    # The vector forward / VJP perform two ``n x n`` dense LU solves
    # (``np.linalg.solve``) plus a handful of dense matmuls — fully
    # expressible in nopython mode when ``W_dense`` is cached
    # (``n <= _kron_dense_max()``).  The matrix variants do batched
    # Fortran-order reshapes and ``scipy.sparse.linalg`` factorizations that
    # are not njittable; we wrap their ``perform`` in ``numba.objmode``
    # purely to silence PyTensor's "object mode" warning.
    #
    # Key reshape identity used below: for a 1D ``arr`` of length ``n*n``
    # ``arr.reshape((n, n), order='F') == arr.reshape((n, n)).T`` and
    # ``M.ravel(order='F') == M.T.ravel()`` for a 2D ``M``.

    @numba_funcify.register(KroneckerFlowSolveOp)
    def _funcify_kron_solve(op, **kwargs):
        n = op._n
        W_dense_view = op._W_dense

        if W_dense_view is not None:
            W_dense = np.asarray(W_dense_view, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def kron_solve(rho_d, rho_o, b):
                Ld = I - rho_d * W_dense
                Lo = I - rho_o * W_dense
                # PyTensor's numba lowering may pass a non-contiguous view;
                # ``np.ascontiguousarray`` is a no-op when ``b`` is already
                # C-contiguous and is required by numba's ``reshape``.
                b_c = np.ascontiguousarray(b)
                Hb = b_c.reshape(n, n).T  # F-order reshape (n*n,) -> (n,n)
                # (Lo ⊗ Ld) vec(H) = vec(Ld H Loᵀ): solve Ld H' = Hb, then
                # H Loᵀ = H', i.e. Lo Hᵀ = H'ᵀ.  Lo, not Loᵀ -- a transposed
                # solve here only agrees with ``perform`` when W is symmetric.
                Hp = np.linalg.solve(Ld, Hb)  # Ld H' = Hb
                Z = np.linalg.solve(Lo, Hp.T)  # Lo Z = H'^T, Z = H^T
                return np.ascontiguousarray(Z).ravel()  # = Z.T.ravel(order='F')

            return kron_solve

        ret_sig = numba.types.float64[:]

        def _py_solve(rd, ro, b):
            outputs = [[None]]
            op.perform(None, [np.asarray(rd), np.asarray(ro), np.asarray(b)], outputs)
            return outputs[0][0]

        @numba.njit
        def kron_solve(rho_d, rho_o, b):
            with numba.objmode(ret=ret_sig):
                ret = _py_solve(rho_d, rho_o, b)
            return ret

        return kron_solve

    @numba_funcify.register(_KroneckerFlowVJPOp)
    def _funcify_kron_vjp(op, **kwargs):
        n = op._n
        W_dense_view = op._W_dense

        if W_dense_view is not None:
            W_dense = np.asarray(W_dense_view, dtype=np.float64)
            I = np.eye(n, dtype=np.float64)

            @numba.njit
            def kron_vjp(rho_d, rho_o, eta, g):
                Ld = I - rho_d * W_dense
                Lo = I - rho_o * W_dense

                # ``ascontiguousarray`` guards against non-contiguous views
                # PyTensor may pass; both reshapes require a contiguous src.
                eta_c = np.ascontiguousarray(eta)
                g_c = np.ascontiguousarray(g)
                H_eta = eta_c.reshape(n, n).T  # F-order reshape
                Hg = g_c.reshape(n, n).T  # F-order reshape

                # Adjoint solve: (Lo^T ⊗ Ld^T) v = g  ⇒  Ld^T H_v Lo = Hg
                P = np.linalg.solve(Ld.T, Hg)  # Ld^T P = Hg
                Q = np.linalg.solve(Lo.T, P.T)  # Lo^T Q = P^T  (Q = H_v^T)
                H_v = Q.T

                W_H = W_dense @ H_eta
                Ld_H = Ld @ H_eta
                grad_rd = np.sum(H_v * (W_H @ Lo.T))
                grad_ro = np.sum(H_v * (Ld_H @ W_dense.T))
                grad_b = np.ascontiguousarray(H_v).T.ravel()  # H_v.ravel(order='F')
                # 0-d arrays, as PyTensor expects for scalar outputs.
                return np.asarray(grad_rd), np.asarray(grad_ro), grad_b

            return kron_vjp

        ret_sig = numba.types.Tuple((_SCALAR, _SCALAR, numba.types.float64[:]))

        def _py_vjp(rd, ro, eta, g):
            outputs = [[None], [None], [None]]
            op.perform(
                None,
                [np.asarray(rd), np.asarray(ro), np.asarray(eta), np.asarray(g)],
                outputs,
            )
            return outputs[0][0], outputs[1][0], outputs[2][0]

        @numba.njit
        def kron_vjp(rho_d, rho_o, eta, g):
            with numba.objmode(ret=ret_sig):
                ret = _py_vjp(rho_d, rho_o, eta, g)
            return ret

        return kron_vjp

    @numba_funcify.register(KroneckerFlowSolveMatrixOp)
    def _funcify_kron_solve_matrix(op, **kwargs):
        # Matrix variant uses batched Fortran-order reshapes and scipy LU
        # factorizations: not expressible in nopython mode.  Wrap perform
        # in objmode to suppress PyTensor's generic fallback warning.
        ret_sig = numba.types.float64[:, :]

        def _py_solve(rd, ro, B):
            outputs = [[None]]
            op.perform(None, [np.asarray(rd), np.asarray(ro), np.asarray(B)], outputs)
            return outputs[0][0]

        @numba.njit
        def kron_solve_mat(rho_d, rho_o, B):
            with numba.objmode(ret=ret_sig):
                ret = _py_solve(rho_d, rho_o, B)
            return ret

        return kron_solve_mat

    @numba_funcify.register(_KroneckerFlowVJPMatrixOp)
    def _funcify_kron_vjp_matrix(op, **kwargs):
        ret_sig = numba.types.Tuple((_SCALAR, _SCALAR, numba.types.float64[:, :]))

        def _py_vjp(rd, ro, H_eta, G):
            outputs = [[None], [None], [None]]
            op.perform(
                None,
                [np.asarray(rd), np.asarray(ro), np.asarray(H_eta), np.asarray(G)],
                outputs,
            )
            return outputs[0][0], outputs[1][0], outputs[2][0]

        @numba.njit
        def kron_vjp_mat(rho_d, rho_o, H_eta, G):
            with numba.objmode(ret=ret_sig):
                ret = _py_vjp(rho_d, rho_o, H_eta, G)
            return ret

        return kron_vjp_mat

    # ------------------------------------------------------------------
    # General sparse NB-flow Ops (3 rho parameters)
    # ------------------------------------------------------------------
    #
    # The forward solve uses scipy.sparse.linalg.splu on the full
    # N x N (N = n^2) system matrix, which Numba cannot lower.  We deliberately
    # do **not** register these Ops: PyTensor's default per-Op object-mode
    # fallback works correctly downstream, whereas explicit objmode wrappers
    # over the multi-output ``_SparseFlowVJP*`` Ops drop the static shape
    # information that PyTensor's elemwise codegen requires
    # (``input_bc_patterns must be literal``).  The cost is one informational
    # PyTensor warning per Op, which is acceptable for a path that already
    # bottlenecks on a sparse SuperLU factorization per gradient evaluation.

    return True
