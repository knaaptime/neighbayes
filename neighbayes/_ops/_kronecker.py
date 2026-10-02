"""PyTensor Ops for Kronecker-structured spatial-flow solves and their VJPs."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp
from pytensor.graph.basic import Apply

from ._backend import (
    _factor_kron_factor,
    _kron_dense_max,
)
from ._instrument import _op_id_counter

# ---------------------------------------------------------------------------
# Kronecker-factored ops for separable NB flow models
# (rho_w = -rho_d * rho_o  =>  A = L_d ⊗ L_o,  L_k = I_n - rho_k * W)
# ---------------------------------------------------------------------------


class _KroneckerFlowVJPOp(pt.Op):
    r"""Vector-Jacobian product for :class:`KroneckerFlowSolveOp`.

    Computes the partial derivatives of a scalar loss :math:`L` with respect
    to the inputs :math:`(\rho_d, \rho_o, b)` of the forward Op using the
    **Kronecker adjoint method** — all arithmetic stays in :math:`n \times n`
    space; no :math:`N \times N` (:math:`N = n^2`) matrix is formed.

    Algorithm
    ---------
    Let :math:`L_k = I_n - \rho_k W`,
    :math:`H_x = \operatorname{mat}(x) \in \mathbb{R}^{n \times n}` denote
    column-major reshaping (``order='F'``), and
    :math:`A = L_o \otimes L_d` the system matrix.

    **1. Forward re-solve** :math:`\eta = (L_o \otimes L_d)^{-1} b`.

    The vec-permutation identity
    :math:`(L_o \otimes L_d)\operatorname{vec}(H) = \operatorname{vec}(L_d H L_o^\top)`
    gives the equivalent dense system :math:`L_d H_\eta L_o^\top = H_b`, solved as:

    .. math::

        H' = L_d^{-1} H_b, \qquad
        Z = L_o^{-\top} H'^{\,\top}, \qquad
        \eta = \operatorname{vec}(Z^{\top})

    (Here :math:`Z = H_\eta^\top`, so :math:`Z^\top = H_\eta`.)

    **2. Adjoint solve** :math:`v = (L_o \otimes L_d)^{-\top} g`.

    Since :math:`(L_o \otimes L_d)^\top = L_o^\top \otimes L_d^\top`,
    the vec-identity gives the dense system :math:`L_d^\top H_v L_o = H_g`:

    .. math::

        P = L_d^{-\top} H_g, \qquad
        Q = L_o^{-\top} P^{\top}, \qquad
        v = \operatorname{vec}(Q^{\top})

    (Here :math:`Q = H_v^\top`, so :math:`Q^\top = H_v`.)

    **3. Sensitivity scalars** via the vec-permutation trace identity.

    For :math:`A = L_o \otimes L_d`:

    .. math::

        \frac{\partial A}{\partial \rho_d} = L_o \otimes (-W), \qquad
        \frac{\partial A}{\partial \rho_o} = (-W) \otimes L_d

    Using :math:`\partial L/\partial \rho_k = -v^\top (\partial A/\partial \rho_k)\eta`
    and :math:`(B \otimes C)\operatorname{vec}(H) = \operatorname{vec}(C H B^\top)`:

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = v^\top (L_o \otimes W)\eta
        = \operatorname{tr}\!\left(H_v^\top W H_\eta L_o^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(W H_\eta L_o^\top)_{ij}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = v^\top (W \otimes L_d)\eta
        = \operatorname{tr}\!\left(H_v^\top L_d H_\eta W^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(L_d H_\eta W^\top)_{ij}

    **4.** :math:`\partial L / \partial b = v`.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix on the *n* spatial units.
        The Kronecker flow matrices :math:`W_d = I_n \otimes W` and
        :math:`W_o = W \otimes I_n` are implicit.
    n : int
        Number of spatial units.  :math:`N = n^2` is the number of O-D pairs.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        # Cached dense view of W; ~n^2 * 8 bytes (trivial for n <= 500).
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, eta, g):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        eta = pt.as_tensor_variable(eta)
        g = pt.as_tensor_variable(g)
        return Apply(
            self,
            [rho_d, rho_o, eta, g],
            [pt.dscalar(), pt.dscalar(), pt.dvector()],
        )

    def perform(self, node, inputs, outputs):
        rd, ro, eta, g = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)
        Ld = self._I - float(rd) * self._W  # only used for L_d @ H below
        Lo = self._I - float(ro) * self._W  # only used for L_o @ ... below

        H_eta = np.asarray(eta, dtype=np.float64).reshape(n, n, order="F")

        # (Lo⊗Ld)^T = Lo^T⊗Ld^T,  (Lo^T⊗Ld^T) vec(Y) = vec(Ld^T Y Lo)
        # Adjoint solve: v = (Lo ⊗ Ld)^{-T} g
        Hg = g.reshape(n, n, order="F")
        P = np.asarray(
            lu_d.solve(np.asarray(Hg, dtype=np.float64), trans="T"), dtype=np.float64
        )
        Q = np.asarray(lu_o.solve(P.T, trans="T"), dtype=np.float64)
        H_v = Q.T  # (n, n)

        # Sensitivities for A = Lo ⊗ Ld:
        #   dA/drho_d = Lo ⊗ (-W),  dL/drho_d = v^T (Lo⊗W) eta
        #             = tr(H_v^T W H_eta Lo^T) = sum(H_v * (W H_eta) @ Lo^T)
        #   dA/drho_o = (-W) ⊗ Ld,  dL/drho_o = v^T (W⊗Ld) eta
        #             = tr(H_v^T Ld H_eta W^T) = sum(H_v * (Ld H_eta) @ W^T)
        W_H = self._W @ H_eta  # (n, n)
        Ld_H = Ld @ H_eta  # (n, n)
        # sum(H_v * (W_H @ Lo.T)) = sum(H_v * (Lo @ W_H.T).T)  [avoids Lo.toarray()]
        # sum(H_v * (Ld_H @ W.T)) = sum(H_v * (W @ Ld_H.T).T)  [avoids W.toarray()]
        outputs[0][0] = np.asarray(np.sum(H_v * (Lo @ W_H.T).T), dtype=np.float64)
        outputs[1][0] = np.asarray(np.sum(H_v * (self._W @ Ld_H.T).T), dtype=np.float64)
        outputs[2][0] = H_v.ravel(order="F").astype(np.float64)  # v = vec(H_v)

    def infer_shape(self, node, input_shapes):
        eta_shape = input_shapes[2]
        return [(), (), eta_shape]

    def pullback(self, inputs, outputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class KroneckerFlowSolveOp(pt.Op):
    r"""Differentiable Kronecker-factored solve for separable NB flow models.

    Computes :math:`\eta = A(\rho_d, \rho_o)^{-1} b` where the system matrix
    exploits the separability constraint :math:`\rho_w = -\rho_d \rho_o`:

    .. math::

        A(\rho_d, \rho_o)
        = I_N - \rho_d (I_n \otimes W) - \rho_o (W \otimes I_n)
          + \rho_d \rho_o (W \otimes W)
        = (I_n - \rho_o W) \otimes (I_n - \rho_d W)
        = L_o \otimes L_d

    where :math:`N = n^2`, :math:`L_k = I_n - \rho_k W`, :math:`W_d = I_n \otimes W`,
    and :math:`W_o = W \otimes I_n`.  Note the order: the **left** Kronecker
    factor :math:`L_o` is associated with :math:`\rho_o` (origin effect) and the
    **right** factor :math:`L_d` with :math:`\rho_d` (destination effect).

    Algorithm
    ---------
    Via the vec-permutation identity
    :math:`(A \otimes B)\operatorname{vec}(X) = \operatorname{vec}(B X A^\top)`,
    the solve :math:`(L_o \otimes L_d)\eta = b` is equivalent to
    :math:`L_d H L_o^\top = B` where
    :math:`B = \operatorname{mat}(b) \in \mathbb{R}^{n \times n}` uses
    column-major (Fortran) ordering.  This is solved in two steps:

    1. :math:`H' = L_d^{-1} B` — sparse solve with :math:`n` RHS columns.
    2. :math:`Z = L_o^{-\top} H'^{\,\top}` — second sparse solve
       (:math:`Z = H_\eta^\top`).
    3. :math:`\eta = \operatorname{vec}(Z^\top)`.

    Complexity
    ----------
    Each gradient evaluation requires **4** :math:`n \times n` sparse
    factorizations (2 forward + 2 adjoint) plus **2** dense
    :math:`n \times n` matrix products — all :math:`O(n^3)`.

    Compare to :class:`SparseFlowSolveOp` which requires **2**
    :math:`N \times N` (:math:`N = n^2`) factorizations — :math:`O(n^6)`.

    Speedup at representative sizes:

    ======  ===============  ========================
    n       N = n²           Approx. speedup
    ======  ===============  ========================
    10      100              ×100
    50      2 500            ×15 000
    100     10 000           ×250 000
    ======  ===============  ========================

    Gradient derivation
    -------------------
    For a scalar loss :math:`L`, implicit differentiation of
    :math:`(L_o \otimes L_d)\eta = b` and the formula
    :math:`dL/d\rho_k = -v^\top (\partial A/\partial \rho_k) \eta` give:

    .. math::

        \frac{\partial A}{\partial \rho_d} = L_o \otimes (-W), \qquad
        \frac{\partial A}{\partial \rho_o} = (-W) \otimes L_d

    where :math:`v = A^{-\top} g` is the adjoint solution and
    :math:`g = \partial L / \partial \eta`.  Using the vec-permutation identity
    with :math:`H_x = \operatorname{mat}(x)` (column-major reshape):

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = \operatorname{tr}\!\left(H_v^\top W H_\eta L_o^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(W H_\eta L_o^\top)_{ij}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = \operatorname{tr}\!\left(H_v^\top L_d H_\eta W^\top\right)
        = \sum_{ij}(H_v)_{ij}\,(L_d H_\eta W^\top)_{ij}

    See :class:`_KroneckerFlowVJPOp` for the implementation.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix on the *n* spatial units.
        Only this :math:`n \times n` matrix is stored; the
        :math:`N \times N` Kronecker matrices are never allocated.
    n : int
        Number of spatial units.

    Notes
    -----
    ``rho_w`` is **not** an input to this Op.  The caller declares
    ``rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)`` for trace
    reporting; the Op implicitly uses the factorized form.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._vjp_op = _KroneckerFlowVJPOp(self._W, n)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, b):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        b = pt.as_tensor_variable(b)
        return Apply(self, [rho_d, rho_o, b], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`\eta = (L_o \otimes L_d)^{-1} b`.

        Applies the two-step Kronecker solve:

        1. :math:`L_d H' = H_b` — sparse solve (``spsolve(Ld, Hb)``).
        2. :math:`L_o Z = H'^\top` — second sparse solve, yielding
           :math:`Z = H_\eta^\top` (from :math:`H_\eta L_o^\top = H'`).
        3. :math:`\eta = \operatorname{vec}(Z^\top)` — column-major flatten.
        """
        rd, ro, b = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)

        Hb = b.reshape(n, n, order="F")
        Hp = np.asarray(lu_d.solve(np.asarray(Hb, dtype=np.float64)), dtype=np.float64)
        Z = np.asarray(lu_o.solve(Hp.T), dtype=np.float64)
        outputs[0][0] = Z.T.ravel(order="F").astype(np.float64)

    def infer_shape(self, node, input_shapes):
        return [input_shapes[2]]

    def pullback(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the Kronecker adjoint method.

        Delegates to :class:`_KroneckerFlowVJPOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, b]``.
        outputs : list of TensorVariable
            ``[eta]`` (symbolic; not used directly).
        output_grads : list of TensorVariable
            ``[g]`` where :math:`g = \partial L / \partial \eta`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_b]``.
        """
        rd, ro, b = inputs
        eta = outputs[0]
        g = output_grads[0]
        grad_rd, grad_ro, grad_b = self._vjp_op(rd, ro, eta, g)
        return [grad_rd, grad_ro, grad_b]


class _KroneckerFlowVJPMatrixOp(pt.Op):
    r"""Vector-Jacobian product for :class:`KroneckerFlowSolveMatrixOp`.

    Extends :class:`_KroneckerFlowVJPOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}` (:math:`T` time periods).

    Two sparse factorizations — one for :math:`L_d` and one for :math:`L_o^\top`
    — cover all :math:`T` columns simultaneously via batched
    :math:`(n, nT)` right-hand sides, so cost is still **4** :math:`n \times n`
    factorizations regardless of :math:`T`.

    Algorithm
    ---------
    For :math:`A = L_o \otimes L_d` and each period :math:`t`, the system
    :math:`(L_o \otimes L_d) \eta_t = b_t` is equivalent to
    :math:`L_d H_{\eta,t} L_o^\top = H_{b,t}` (vec-permutation identity).
    The :math:`T` slices are solved in batch:

    1. Pack :math:`B` as :math:`(n, nT)` in column-major order so that
       columns :math:`[tn : (t+1)n]` hold :math:`H_{b,t}`.
       Solve :math:`L_d H' = R_{(n,nT)}` — one factorization covering all :math:`T`.
    2. Permute slices to build the transposed batch
       :math:`[(H'_t)^\top]_t` as an :math:`(n, nT)` matrix.
       Solve :math:`L_o^\top Z = \mathrm{RHS2}_{(n,nT)}` — second factorization.
    3. Permute and reshape to :math:`(N, T)`:  column :math:`t` is
       :math:`\operatorname{vec}(Z_t^\top) = \eta_t`.

    The adjoint pass uses the same layout with :math:`L_d^\top` and
    :math:`L_o^\top` instead.

    Sensitivity scalars (summed over all :math:`T` periods):

    .. math::

        \frac{\partial L}{\partial \rho_d}
        = \sum_{t} \operatorname{tr}\!\left(H_{v,t}^\top W H_{\eta,t} L_o^\top\right)
        = \sum_{ijt} (H_v)_{ijt}\,(W H_\eta L_o^\top)_{ijt}

    .. math::

        \frac{\partial L}{\partial \rho_o}
        = \sum_{t} \operatorname{tr}\!\left(H_{v,t}^\top L_d H_{\eta,t} W^\top\right)
        = \sum_{ijt} (H_v)_{ijt}\,(L_d H_\eta W^\top)_{ijt}

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix.
    n : int
        Number of spatial units.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, H_eta, G):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        H_eta = pt.as_tensor_variable(H_eta)
        G = pt.as_tensor_variable(G)
        return Apply(
            self,
            [rho_d, rho_o, H_eta, G],
            [pt.dscalar(), pt.dscalar(), pt.dmatrix()],
        )

    def _kron_solve(self, lu_o, lu_d, rhs, *, transpose_d=False, transpose_o=True):
        """Two-step Kronecker solve used for both forward and adjoint passes.

        Both ``(Lo⊗Ld) η = B`` (forward) and ``(Lo^T⊗Ld^T) v = G`` (adjoint)
        use one LU factorization per Kronecker factor. The second step solves
        against ``Lo`` for the forward path (``transpose_o=False``) and against
        ``Lo^T`` for the adjoint path (``transpose_o=True``, the default for
        backward compatibility with the VJP).

        For each period t the system ``L_first H'_t = H_b_t`` is solved
        simultaneously for all T periods via a single (n, n*T) RHS.
        The second step ``L_second Z_t = H'_t^T`` is likewise batched.

        Parameters
        ----------
        lu_o : scipy.sparse.linalg.SuperLU
            LU factorization of ``Lo``.
        lu_d : scipy.sparse.linalg.SuperLU
            LU factorization of ``Ld``.
        rhs : ndarray, shape (N, T)
            Right-hand-side matrix.
        transpose_d : bool, default False
            If True, solve the first step against ``Ld^T``.
        transpose_o : bool, default True
            If True (default, adjoint convention), solve the second step
            against ``Lo^T``. Set to False for the forward solve, which
            needs ``Lo^{-1}`` (consistent with the single-vector forward).

        Returns
        -------
        ndarray, shape (N, T)
            Solution columns ``η_t = H_eta_t.ravel('F')``.
        """
        n, T = self._n, rhs.shape[1]
        # Step 1: L_first H'_t = H_b_t  (batch: (n, n*T) solve)
        R = rhs.reshape(n, n * T, order="F")  # (n, n*T): col t*n+j = col j of H_b_t
        Hp = np.asarray(
            lu_d.solve(
                np.asarray(R, dtype=np.float64), trans="T" if transpose_d else "N"
            ),
            dtype=np.float64,
        )
        Hp3 = Hp.reshape(n, n, T, order="F")  # (n, n, T): Hp3[:,:,t] = H'_t
        # Step 2: L_second Z_t = H'_t^T  (batch: (n, n*T) solve)
        # Pack RHS so that col t*n+j = H'_t[j,:] (j-th row of H'_t = j-th col of H'_t^T)
        # Hp3.transpose(2,0,1) shape (T,n,n): result[t,j,:] = H'_t[j,:]
        # C-order reshape to (T*n, n): result_2d[t*n+j, i] = H'_t[j, i]
        # Transpose to (n, T*n): RHS2[:, t*n+j] = H'_t[j, :] ✓
        RHS2 = Hp3.transpose(2, 0, 1).reshape(T * n, n).T  # (n, n*T)
        Z_h = np.asarray(
            lu_o.solve(
                np.asarray(RHS2, dtype=np.float64),
                trans="T" if transpose_o else "N",
            ),
            dtype=np.float64,
        )
        Z3 = Z_h.reshape(n, n, T, order="F")  # (n, n, T): Z3[:,:,t] = Z_t
        # result[:, t] = Z_t^T.ravel('F') = H_eta_t.ravel('F')
        # Z3.transpose(1,0,2): result[j,i,t] = Z3[i,j,t] = Z_t[i,j]
        # F-order reshape (n,n,T) → (N,T): result[i+n*j, t] = Z3[j,i,t] = Z_t[j,i] = Z_t^T[i,j]
        return Z3.transpose(1, 0, 2).reshape(n * n, T, order="F")  # (N, T)

    def perform(self, node, inputs, outputs):
        rd, ro, H_eta, G = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)
        Ld = self._I - float(rd) * self._W  # used only for sparse matmul below
        Lo = self._I - float(ro) * self._W  # used only for sparse multiply below

        H_eta = np.asarray(H_eta, dtype=np.float64)
        H_v = self._kron_solve(
            lu_o, lu_d, np.asarray(G, dtype=np.float64), transpose_d=True
        )

        # Reshape to (n, n, T) for Kronecker trace sums over all T periods
        T = H_eta.shape[1]
        He = H_eta.reshape(n, n, T, order="F")
        Hv = H_v.reshape(n, n, T, order="F")

        # Sparse matmul: W @ He_t and Ld @ He_t for all T simultaneously
        He_2d = He.reshape(n, n * T)  # (n, n*T)
        W_He = (self._W @ He_2d).reshape(n, n, T)  # (n, n, T)
        Ld_He = (Ld @ He_2d).reshape(n, n, T)  # (n, n, T)

        # sum(Hv * WHe_LoT)  = sum_{jk} Lo[j,k]  * S_d[j,k]
        # sum(Hv * LdHe_WT)  = sum_{jk} W[j,k]   * S_o[j,k]
        # where S[j,k] = sum_{it} XHe[i,k,t] * Hv[i,j,t]
        S_d = np.einsum("ikt,ijt->jk", W_He, Hv)  # (n, n)
        S_o = np.einsum("ikt,ijt->jk", Ld_He, Hv)  # (n, n)

        outputs[0][0] = np.asarray(Lo.multiply(S_d).sum(), dtype=np.float64)
        outputs[1][0] = np.asarray(self._W.multiply(S_o).sum(), dtype=np.float64)
        outputs[2][0] = np.asarray(H_v, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        H_shape = input_shapes[2]
        return [(), (), H_shape]

    def pullback(self, inputs, outputs, output_grads):
        return [pt.zeros_like(inp) for inp in inputs]


class KroneckerFlowSolveMatrixOp(pt.Op):
    r"""Kronecker-factored solve for separable panel NB flow models.

    Extends :class:`KroneckerFlowSolveOp` to a matrix right-hand side
    :math:`B \in \mathbb{R}^{N \times T}` that arises when :math:`T` time
    periods share the same system matrix :math:`A = L_o \otimes L_d`
    (see :class:`KroneckerFlowSolveOp` for the factorization derivation).

    Two :math:`n \times n` factorizations cover all :math:`T` columns
    simultaneously:

    .. math::

        \eta_t = (L_o \otimes L_d)^{-1} b_t, \quad t = 1, \ldots, T

    where columns :math:`b_t` of :math:`B` are the per-period :math:`X_t\beta`
    vectors stacked as :math:`(N, T)`.

    The batched solve reshapes :math:`B` to :math:`(n, nT)` so that a single
    call to :func:`scipy.sparse.linalg.spsolve` covers all :math:`T` columns
    with one :math:`L_d` factorization, followed by one :math:`L_o^\top`
    factorization for the second step.

    Complexity
    ----------
    **4** :math:`n \times n` sparse factorizations per gradient step, regardless
    of :math:`T`.  Compare to :class:`SparseFlowSolveMatrixOp` which requires
    **2** :math:`N \times N` factorizations (:math:`N = n^2`).

    Parameters
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Row-standardized spatial weight matrix.  Never stored as
        :math:`N \times N` Kronecker matrices.
    n : int
        Number of spatial units.
    """

    __props__ = ("_op_id",)

    def __init__(self, W: sp.csr_matrix, n: int) -> None:
        self._W = W.tocsr().astype(np.float64)
        self._n = n
        self._I = sp.eye(n, format="csr", dtype=np.float64)
        self._W_dense = self._W.toarray() if n <= _kron_dense_max() else None
        self._I_dense = (
            np.eye(n, dtype=np.float64) if self._W_dense is not None else None
        )
        self._vjp_op = _KroneckerFlowVJPMatrixOp(self._W, n)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho_d, rho_o, B):
        rho_d = pt.as_tensor_variable(rho_d)
        rho_o = pt.as_tensor_variable(rho_o)
        B = pt.as_tensor_variable(B)
        return Apply(self, [rho_d, rho_o, B], [pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        r"""Compute :math:`H = (L_o \otimes L_d)^{-1} B` for all :math:`T` columns.

        Applies the batched two-step Kronecker solve to
        :math:`B \in \mathbb{R}^{N \times T}` using one :math:`L_d`
        factorization and one :math:`L_o^\top` factorization:

        1. :math:`L_d H' = R_{(n,nT)}` — batch all :math:`T` slices
           side-by-side.
        2. :math:`L_o^\top Z = \mathrm{RHS2}_{(n,nT)}` — second batch solve,
           with slices permuted to present transposed :math:`H'_t` columns.
        3. Permute and reshape :math:`Z` back to :math:`(N, T)`.
        """
        rd, ro, B = inputs
        n = self._n
        lu_d = _factor_kron_factor(self._W_dense, self._W, rd, n, self._I_dense)
        lu_o = _factor_kron_factor(self._W_dense, self._W, ro, n, self._I_dense)

        # Forward: (Lo⊗Ld) η = b  →  Ld H' Lo^T = H_b
        # Step 1: Ld H' = R;  Step 2: Lo H_eta^T = H'^T (batched over T)
        # transpose_o=False makes step 2 use Lo (not Lo^T), matching the
        # single-vector forward in KroneckerFlowSolveOp.perform.
        result = self._vjp_op._kron_solve(
            lu_o,
            lu_d,
            np.asarray(B, dtype=np.float64),
            transpose_o=False,
        )
        outputs[0][0] = np.asarray(result, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        return [input_shapes[2]]

    def pullback(self, inputs, outputs, output_grads):
        r"""Compute VJPs via the Kronecker adjoint method.

        Delegates to :class:`_KroneckerFlowVJPMatrixOp`.

        Parameters
        ----------
        inputs : list of TensorVariable
            ``[rho_d, rho_o, B]``.
        outputs : list of TensorVariable
            ``[H]`` (symbolic; not used directly).
        output_grads : list of TensorVariable
            ``[G]`` where :math:`G = \partial L / \partial H \in \mathbb{R}^{N \times T}`.

        Returns
        -------
        list of TensorVariable
            ``[grad_rho_d, grad_rho_o, grad_B]``.
        """
        rd, ro, B = inputs
        H = outputs[0]
        G = output_grads[0]
        grad_rd, grad_ro, grad_B = self._vjp_op(rd, ro, H, G)
        return [grad_rd, grad_ro, grad_B]
