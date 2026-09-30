"""PyTensor Ops for spatial-flow ``(I - rho W)^{-1}`` solves (vector and matrix RHS)."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
from pytensor.graph.basic import Apply

from ._backend import (
    _CachedSolverOpMixin,
    _make_cached_sparse_solver,
    _select_sparse_backend,
    _solve_sparse_matrix,
    _solve_sparse_vector,
)
from ._instrument import _op_id_counter


class _SparseFlowVJPOp(_CachedSolverOpMixin, pt.Op):
    r"""Vector-Jacobian product (VJP) for :class:`SparseFlowSolveOp`.

    Computes all four partial derivatives of the scalar loss :math:`L` with
    respect to the inputs :math:`(\\rho_d, \\rho_o, \\rho_w, b)` of the
    forward Op in a single ``perform`` call.

    Algorithm
    ---------
     1. Consume the forward solution :math:`\eta = A^{-1} b` from the
         parent Op output.
     2. **Adjoint solve** :math:`v = (A^\top)^{-1} g`.
    3. **Sensitivity scalars** for each :math:`k \\in \\{d, o, w\\}`:

       .. math::

           \\frac{\\partial L}{\\partial \\rho_k}
           = -v^\\top W_k \\eta

    4. **Gradient w.r.t. the RHS vector** :math:`b`:

       .. math::

           \\frac{\\partial L}{\\partial b} = v

    Parameters
    ----------
    Wd, Wo, Ww : scipy.sparse.csr_matrix
        Kronecker flow weight matrices, shared with the parent
        :class:`SparseFlowSolveOp` instance (not copied).
    """

    __props__ = ("_op_id",)

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
    ) -> None:
        self._Wd = Wd
        self._Wo = Wo
        self._Ww = Ww
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint(
        self, rd: float, ro: float, rw: float, g: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho)^T v = g`` with lightweight LU cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        g64 = np.asarray(g, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(
                self._cached_solver.solve(g64, trans="T"), dtype=np.float64
            )

        # KLU: reuse the cached factor of A^T when backend and rho
        # are unchanged, otherwise build and cache a fresh one.  (The factor
        # has no ``trans`` support, so A^T is factorized directly.)
        if (
            self._cached_backend == backend
            and self._cached_rhos == rhos
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        A_t = (
            self._I - rhos[0] * self._Wd.T - rhos[1] * self._Wo.T - rhos[2] * self._Ww.T
        )
        cached_solver = _make_cached_sparse_solver(A_t, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        return _solve_sparse_vector(A_t, g64)

    def make_node(self, rho_d, rho_o, rho_w, eta, g):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(
            self,
            [rho_d, rho_o, rho_w, eta, g],
            [pt.dscalar(), pt.dscalar(), pt.dscalar(), pt.dvector()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, rw, eta, g = inputs
        eta = np.asarray(eta, dtype=np.float64)
        v = self._solve_adjoint(
            float(rd), float(ro), float(rw), np.asarray(g, dtype=np.float64)
        )
        outputs[0][0] = np.asarray(float(v @ (self._Wd @ eta)), dtype=np.float64)
        outputs[1][0] = np.asarray(float(v @ (self._Wo @ eta)), dtype=np.float64)
        outputs[2][0] = np.asarray(float(v @ (self._Ww @ eta)), dtype=np.float64)
        outputs[3][0] = np.asarray(v, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        # Three scalar outputs + one vector matching b/g shape
        eta_shape = input_shapes[3]
        return [(), (), (), eta_shape]

    def pullback(self, inputs, outputs, output_grads):
        # Second-order gradients are not required for NUTS (first-order only).
        return [pt.zeros_like(inp) for inp in inputs]


class SparseFlowSolveOp(_CachedSolverOpMixin, pt.Op):
    r"""Differentiable sparse solve :math:`\eta = A(\rho)^{-1} b`.

    Wraps :func:`scipy.sparse.linalg.spsolve` as a pytensor
    :class:`~pytensor.graph.op.Op` with analytically exact first-order
    gradients derived via the adjoint method.

    The system matrix is:

    .. math::

        A(\rho_d, \rho_o, \rho_w)
        = I_N - \rho_d W_d - \rho_o W_o - \rho_w W_w

    where :math:`W_d = I_n \otimes W`, :math:`W_o = W \otimes I_n`,
    :math:`W_w = W \otimes W` are the Kronecker-product flow weight matrices
    and :math:`N = n^2`.

    This Op is used by :class:`~neighbayes.models.flow.SARNegBinFlow` to embed
    the implicit spatial filter on the **log-mean** of a count observation
    model:

    .. math::

        \eta &= A^{-1} X\beta \\
        \mu_{ij} &= \exp(\eta_{ij}) \\
        y_{ij} &\sim \operatorname{NegBin}(\mu_{ij}, \alpha)

    The Jacobian log-determinant :math:`\log|A(\rho)|` is added separately
    via :func:`~neighbayes.logdet.flow_logdet_pytensor` (identical to the
    Gaussian SAR flow model).

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of :math:`A\eta = b`
    gives :math:`dA\, \eta + A\, d\eta = db`, so:

    .. math::

        d\eta = A^{-1}(db - dA\, \eta)

    The VJPs are:

    .. math::

        \frac{\partial L}{\partial \rho_k}
        = g^\top \frac{\partial \eta}{\partial \rho_k}
        = -v^\top W_k \eta,
        \qquad
        \frac{\partial L}{\partial b} = v

    where :math:`v = (A^\top)^{-1} g` and
    :math:`g = \partial L / \partial \eta` is the upstream gradient.
    See :class:`_SparseFlowVJPOp` for the implementation.

    Per-gradient-evaluation cost: **2 sparse direct solves** (SuperLU) +
    3 sparse matrix-vector products.  For :math:`n \leq 100`
    (:math:`N \leq 10^4`) this is fast enough for NUTS sampling.

    Parameters
    ----------
    Wd : scipy.sparse.csr_matrix, shape (N, N)
        Destination weight matrix :math:`W_d = I_n \otimes W`.
    Wo : scipy.sparse.csr_matrix, shape (N, N)
        Origin weight matrix :math:`W_o = W \otimes I_n`.
    Ww : scipy.sparse.csr_matrix, shape (N, N)
        Network weight matrix :math:`W_w = W \otimes W`.

    Examples
    --------
    >>> from neighbayes._ops import SparseFlowSolveOp
    >>> from neighbayes.graph import flow_weight_matrices
    >>> import pytensor.tensor as pt, pytensor
    >>> wms = flow_weight_matrices(G)
    >>> op = SparseFlowSolveOp(wms["destination"], wms["origin"], wms["network"])
    >>> rho_d, rho_o, rho_w = pt.scalars("rho_d", "rho_o", "rho_w")
    >>> b = pt.vector("b")
    >>> eta = op(rho_d, rho_o, rho_w, b)
    >>> fn = pytensor.function([rho_d, rho_o, rho_w, b], eta)
    """

    __props__ = ("_op_id",)

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
    ) -> None:
        self._Wd = Wd.tocsr().astype(np.float64)
        self._Wo = Wo.tocsr().astype(np.float64)
        self._Ww = Ww.tocsr().astype(np.float64)
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseFlowVJPOp(self._Wd, self._Wo, self._Ww)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward(
        self, rd: float, ro: float, rw: float, b: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho) eta = b`` with lightweight LU cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        b64 = np.asarray(b, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        # KLU: reuse the cached factor when backend and rho are
        # unchanged, otherwise build and cache a fresh one.
        if (
            self._cached_backend == backend
            and self._cached_rhos == rhos
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        A = self._I - rhos[0] * self._Wd - rhos[1] * self._Wo - rhos[2] * self._Ww
        cached_solver = _make_cached_sparse_solver(A, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        return _solve_sparse_vector(A, b64)

    def make_node(self, rho_d, rho_o, rho_w, b):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho_d, rho_o, rho_w, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        """Compute :math:`\\eta = A(\\rho)^{-1} b` via a sparse direct solver.

        Uses a single SuperLU factorization via :func:`scipy.sparse.linalg.splu`.
        """
        rd, ro, rw, b = inputs
        outputs[0][0] = self._solve_forward(
            float(rd), float(ro), float(rw), np.asarray(b, dtype=np.float64)
        )

    def infer_shape(self, node, input_shapes):
        # Output has same shape as b
        return [input_shapes[3]]

    def pullback(self, inputs, outputs, output_grads):
        """Compute the VJP via the adjoint method.

        Delegates to :class:`_SparseFlowVJPOp`, which performs:

        1. Forward re-solve: :math:`\\eta = A^{-1} b`.
        2. Adjoint solve: :math:`v = (A^\\top)^{-1} g`.
        3. Sensitivity scalars:
           :math:`\\partial L / \\partial \\rho_k = -v^\\top W_k \\eta`.
        4. Gradient w.r.t. :math:`b`: :math:`v`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, rho_w, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic forward output; not used directly here).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \\partial L / \\partial \\eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_rho_w, grad_b]``.
        """
        rd, ro, rw, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rd, grad_ro, grad_rw, grad_b = self._vjp_op(rd, ro, rw, eta, g)
        return [grad_rd, grad_ro, grad_rw, grad_b]


class _SparseFlowVJPMatrixOp(_CachedSolverOpMixin, pt.Op):
    """Vector-Jacobian product (VJP) for :class:`SparseFlowSolveMatrixOp`.

    Same adjoint-method derivation as :class:`_SparseFlowVJPOp`, extended to
    a matrix right-hand side :math:`B` of shape :math:`(N, T)`.  One
    LU factorization covers all :math:`T` columns.

    Algorithm
    ---------
    1. **Forward re-solve** :math:`H = A^{-1} B`, result shape :math:`(N, T)`.
    2. **Adjoint solve** :math:`V = (A^\\top)^{-1} G`, same shape.
    3. **Sensitivity scalars** for each :math:`k \\in \\{d, o, w\\}`:

       .. math::

           \\frac{\\partial L}{\\partial \\rho_k}
           = -\\sum_t v_t^\\top W_k h_t
           = -\\operatorname{sum}(V \\odot (W_k H))

    4. **Gradient w.r.t. B**: :math:`V`.
    """

    __props__ = ("_op_id",)

    def __init__(self, Wd, Wo, Ww):
        self._Wd = Wd
        self._Wo = Wo
        self._Ww = Ww
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint_matrix(
        self, rd: float, ro: float, rw: float, G: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho)^T V = G`` for matrix RHS with cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        G64 = np.asarray(G, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(
                self._cached_solver.solve(G64, trans="T"), dtype=np.float64
            )

        # KLU: reuse the cached factor of A^T (one factorization
        # covers all T columns) when backend and rho are unchanged.
        if (
            self._cached_backend == backend
            and self._cached_rhos == rhos
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(G64), dtype=np.float64)

        A_t = (
            self._I - rhos[0] * self._Wd.T - rhos[1] * self._Wo.T - rhos[2] * self._Ww.T
        )
        cached_solver = _make_cached_sparse_solver(A_t, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(G64), dtype=np.float64)

        return _solve_sparse_matrix(A_t, G64)

    def make_node(self, rho_d, rho_o, rho_w, H, G):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        H = pt.as_tensor_variable(H)
        G = pt.as_tensor_variable(G)
        return Apply(
            self,
            [rho_d, rho_o, rho_w, H, G],
            [pt.dscalar(), pt.dscalar(), pt.dscalar(), pt.dmatrix()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, rw, H, G = inputs
        H = np.asarray(H, dtype=np.float64)
        V = self._solve_adjoint_matrix(
            float(rd), float(ro), float(rw), np.asarray(G, dtype=np.float64)
        )
        outputs[0][0] = np.asarray(np.sum(V * (self._Wd @ H)), dtype=np.float64)
        outputs[1][0] = np.asarray(np.sum(V * (self._Wo @ H)), dtype=np.float64)
        outputs[2][0] = np.asarray(np.sum(V * (self._Ww @ H)), dtype=np.float64)
        outputs[3][0] = np.asarray(V, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        H_shape = input_shapes[3]
        return [(), (), (), H_shape]

    def pullback(self, inputs, outputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class SparseFlowSolveMatrixOp(_CachedSolverOpMixin, pt.Op):
    r"""Differentiable sparse solve :math:`H = A(\rho)^{-1} B` for matrix RHS.

    Extends :class:`SparseFlowSolveOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}`, which arises in panel NB flow
    models where :math:`T` time periods share the same system matrix
    :math:`A(\rho_d, \rho_o, \rho_w)`.

    One LU factorization of :math:`A` covers all :math:`T` columns, so the
    cost per gradient evaluation is **2 sparse direct solves** (vs. :math:`2T`
    for a per-period loop).

    Parameters
    ----------
    Wd, Wo, Ww : scipy.sparse.csr_matrix, shape (N, N)
        Kronecker flow weight matrices (shared with parent model; not copied).
    """

    __props__ = ("_op_id",)

    def __init__(self, Wd, Wo, Ww):
        self._Wd = Wd.tocsr().astype(np.float64)
        self._Wo = Wo.tocsr().astype(np.float64)
        self._Ww = Ww.tocsr().astype(np.float64)
        self._I = sp.eye(Wd.shape[0], format="csr", dtype=np.float64)
        self._cached_rhos: tuple[float, float, float] | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseFlowVJPMatrixOp(self._Wd, self._Wo, self._Ww)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward_matrix(
        self, rd: float, ro: float, rw: float, B: np.ndarray
    ) -> np.ndarray:
        """Solve ``A(rho) H = B`` for matrix RHS with cache reuse."""
        rhos = (float(rd), float(ro), float(rw))
        B64 = np.asarray(B, dtype=np.float64)
        backend = _select_sparse_backend()

        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rhos != rhos:
                A = (
                    self._I
                    - rhos[0] * self._Wd
                    - rhos[1] * self._Wo
                    - rhos[2] * self._Ww
                )
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(B64), dtype=np.float64)

        # KLU: reuse the cached factor (one factorization covers all
        # T columns) when backend and rho are unchanged.
        if (
            self._cached_backend == backend
            and self._cached_rhos == rhos
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(B64), dtype=np.float64)

        A = self._I - rhos[0] * self._Wd - rhos[1] * self._Wo - rhos[2] * self._Ww
        cached_solver = _make_cached_sparse_solver(A, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rhos = rhos
            return np.asarray(self._cached_solver.solve(B64), dtype=np.float64)

        return _solve_sparse_matrix(A, B64)

    def make_node(self, rho_d, rho_o, rho_w, B):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        rho_w = pt.as_tensor_variable(rho_w)
        B = pt.as_tensor_variable(B)
        return Apply(self, [rho_d, rho_o, rho_w, B], [pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        rd, ro, rw, B = inputs
        outputs[0][0] = self._solve_forward_matrix(
            float(rd), float(ro), float(rw), np.asarray(B, dtype=np.float64)
        )

    def infer_shape(self, node, input_shapes):
        return [input_shapes[3]]

    def pullback(self, inputs, outputs, output_grads):
        rd, ro, rw, B = inputs
        H = outputs[0]
        G = output_grads[0]
        grad_rd, grad_ro, grad_rw, grad_B = self._vjp_op(rd, ro, rw, H, G)
        return [grad_rd, grad_ro, grad_rw, grad_B]
