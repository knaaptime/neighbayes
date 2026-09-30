"""PyTensor Ops for the SAR reduced form ``(I - rho W)^{-1}`` and its VJP."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
from pytensor.graph.basic import Apply

from ._backend import (
    _CachedSolverOpMixin,
    _DenseLU,
    _kron_dense_max,
    _make_cached_sparse_solver,
    _select_sparse_backend,
    _solve_sparse_vector,
)
from ._instrument import _op_id_counter


class _SparseSARVJPOp(_CachedSolverOpMixin, pt.Op):
    r"""Vector-Jacobian product for :class:`SparseSARSolveOp`.

    Computes partial derivatives of a scalar loss :math:`L` with respect to
    the inputs :math:`(\rho, b)` of the forward Op using the adjoint method.

    Algorithm
    ---------
    1. **Adjoint solve** :math:`v = (I - \rho W^\top)^{-1} g`.
    2. **Sensitivity scalar** for :math:`\rho`:

       .. math::

           \frac{\partial L}{\partial \rho}
           = v^\top W \eta

    3. **Gradient w.r.t.** :math:`b`: :math:`v`.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, eigendecomposition=None) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = W.shape[0]
        self._I = sp.eye(self._n, format="csr", dtype=np.float64)
        self._W_dense = W.toarray() if self._n <= _kron_dense_max() else None
        self._eigendecomposition = eigendecomposition
        self._cached_rho: float | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_adjoint(self, rho_val: float, g: np.ndarray) -> np.ndarray:
        """Solve ``(I - rho W^T) v = g`` with lightweight factor cache reuse."""
        n = self._n
        rho_f = float(rho_val)
        g64 = np.asarray(g, dtype=np.float64)

        if n <= _kron_dense_max() and self._W_dense is not None:
            if self._cached_backend != "dense" or self._cached_rho != rho_f:
                A_T_dense = np.eye(n, dtype=np.float64) - rho_f * self._W_dense.T
                self._cached_solver = _DenseLU(A_T_dense)
                self._cached_backend = "dense"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        backend = _select_sparse_backend()
        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rho != rho_f:
                A_T = self._I - rho_f * self._W.transpose()
                self._cached_solver = sp.linalg.splu(A_T.tocsc())
                self._cached_backend = "scipy"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        if (
            self._cached_backend == backend
            and self._cached_rho == rho_f
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        A_T = self._I - rho_f * self._W.transpose()
        cached_solver = _make_cached_sparse_solver(A_T, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(g64), dtype=np.float64)

        return _solve_sparse_vector(A_T, g64)

    def make_node(self, rho, eta, g):
        rho = pt.as_tensor_variable(rho)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(self, [rho, eta, g], [pt.dscalar(), pt.dvector()])

    def perform(self, node, inputs, outputs):
        rho_val, eta, g = inputs
        # Adjoint solve: v = (I - rho * W^T)^{-1} g
        # For symmetric-like W (queen contiguity), W^T ≈ W, but we use W^T for correctness.
        v = self._solve_adjoint(float(rho_val), np.asarray(g, dtype=np.float64))

        eta = np.asarray(eta, dtype=np.float64)
        # dL/drho = v^T W eta
        W_eta = self._W @ eta
        outputs[0][0] = np.asarray(float(v @ W_eta), dtype=np.float64)
        # dL/db = v
        outputs[1][0] = np.asarray(v, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        eta_shape = input_shapes[1]
        return [(), eta_shape]

    def pullback(self, inputs, outputs, output_grads):
        # Second-order gradients not required for NUTS.
        return [pt.zeros_like(inp) for inp in inputs]


class SparseSARSolveOp(_CachedSolverOpMixin, pt.Op):
    r"""Differentiable sparse solve :math:`\eta = (I - \rho W)^{-1} b`.

    Wraps :func:`scipy.sparse.linalg.splu` as a pytensor
    :class:`~pytensor.graph.op.Op` with analytically exact first-order
    gradients derived via the adjoint method.

    The system matrix is:

    .. math::

        A(\rho) = I_n - \rho W

    where :math:`W` is a row-standardized spatial weight matrix.

    This Op is used by :class:`~neighbayes.models.sar_negbin.SARNegBin`
    to embed the SAR-in-mean reduced form on the **log-mean** of a
    Negative Binomial observation model:

    .. math::

        \eta &= (I - \rho W)^{-1} X\beta \\
        \mu_i &= \exp(\eta_i) \\
        y_i &\sim \operatorname{NegBin}(\mu_i, \alpha)

    The Jacobian log-determinant :math:`\log|I - \rho W|` is added separately
    via the model's ``_logdet_pytensor_fn``.

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of
    :math:`(I - \rho W)\eta = b` gives:

    .. math::

        \frac{\partial L}{\partial \rho}
        = g^\top \frac{\partial \eta}{\partial \rho}
        = g^\top (I - \rho W)^{-1} W \eta
        = v^\top W \eta

    .. math::

        \frac{\partial L}{\partial b}
        = g^\top (I - \rho W)^{-1}
        = v

    where :math:`v = (I - \rho W^\top)^{-1} g` is the **adjoint solution**
    and :math:`g = \partial L / \partial \eta` is the upstream gradient.

    Per-gradient-evaluation cost: **2 sparse direct solves** (one forward,
    one adjoint) + 1 sparse matrix-vector product.  For queen-contiguity
    :math:`W` with :math:`n \leq 10{,}000` this is fast enough for NUTS
    sampling.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix.

    Examples
    --------
    >>> from neighbayes._ops import SparseSARSolveOp
    >>> import pytensor.tensor as pt, pytensor
    >>> op = SparseSARSolveOp(W_csr)
    >>> rho = pt.scalar("rho")
    >>> b = pt.vector("b")
    >>> eta = op(rho, b)
    >>> fn = pytensor.function([rho, b], eta)
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, eigendecomposition=None) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = W.shape[0]
        self._I = sp.eye(self._n, format="csr", dtype=np.float64)
        self._W_dense = W.toarray() if self._n <= _kron_dense_max() else None
        # Pre-allocate dense identity once to avoid repeated np.eye() calls
        # during NUTS sampling when rho changes frequently.
        self._I_dense = (
            np.eye(self._n, dtype=np.float64) if self._W_dense is not None else None
        )
        # Shared eigendecomposition cache (eigs, V, Vinv) in complex128,
        # consumed from the model's _W_eigendecomposition property.
        # When provided, _build_eigen_sar_paths reuses it instead of
        # computing its own decomposition.
        self._eigendecomposition = eigendecomposition
        self._cached_rho: float | None = None
        self._cached_backend: str | None = None
        self._cached_solver = None
        self._vjp_op = _SparseSARVJPOp(self._W, eigendecomposition=eigendecomposition)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def _solve_forward(self, rho_val: float, b: np.ndarray) -> np.ndarray:
        """Solve ``(I - rho W) eta = b`` with lightweight factor cache reuse."""
        n = self._n
        rho_f = float(rho_val)
        b64 = np.asarray(b, dtype=np.float64)

        if n <= _kron_dense_max() and self._W_dense is not None:
            if self._cached_backend != "dense" or self._cached_rho != rho_f:
                A_dense = self._I_dense - rho_f * self._W_dense
                self._cached_solver = _DenseLU(A_dense)
                self._cached_backend = "dense"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        backend = _select_sparse_backend()
        if backend == "scipy":
            if self._cached_backend != "scipy" or self._cached_rho != rho_f:
                A = self._I - rho_f * self._W
                self._cached_solver = sp.linalg.splu(A.tocsc())
                self._cached_backend = "scipy"
                self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        if (
            self._cached_backend == backend
            and self._cached_rho == rho_f
            and self._cached_solver is not None
        ):
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        A = self._I - rho_f * self._W
        cached_solver = _make_cached_sparse_solver(A, backend)
        if cached_solver is not None:
            self._cached_solver = cached_solver
            self._cached_backend = backend
            self._cached_rho = rho_f
            return np.asarray(self._cached_solver.solve(b64), dtype=np.float64)

        return _solve_sparse_vector(A, b64)

    def make_node(self, rho, b):
        rho = pt.as_tensor_variable(rho)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`\eta = (I - \rho W)^{-1} b` via a sparse direct solver."""
        rho_val, b = inputs
        outputs[0][0] = self._solve_forward(
            float(rho_val), np.asarray(b, dtype=np.float64)
        )

    def infer_shape(self, node, input_shapes):
        return [input_shapes[1]]

    def pullback(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the adjoint method.

        Delegates to :class:`_SparseSARVJPOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic forward output; not used directly here).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \partial L / \partial \eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho, grad_b]``.
        """
        rho, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rho, grad_b = self._vjp_op(rho, eta, g)
        return [grad_rho, grad_b]
