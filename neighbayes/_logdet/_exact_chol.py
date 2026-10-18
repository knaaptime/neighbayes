r"""Exact ``log|I − ρW|`` at every ρ by sparse Cholesky (``exact`` on ``chol``).

For a D-symmetrizable ``W`` (a row-standardized undirected graph), ``I − ρW``
has the determinant of the SPD ``I − ρW_sym``, factored with scikit-sparse's
CHOLMOD.  The symbolic analysis is done once (:class:`~._chol_cheb.CholChebContext`)
and each ρ costs one numeric factorization.

The gradient ``g(ρ) = −tr(W(I − ρW)⁻¹)`` would take ``n`` solves exactly; it is
the central difference of exact values instead, two factorizations at
``ρ ± h``, accurate to ``O(h²)`` (about 1e-9 relative at the default ``h``).
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from pytensor.graph.basic import Apply

from .._numba_dispatch import register_logdet_numba_dispatch
from .._ops._instrument import _op_id_counter
from ._chol_cheb import CholChebContext

#: Step of the central-difference gradient.
_GRAD_STEP = 1e-5
#: ``I − ρW_sym`` is SPD for ``|ρ| < 1`` (a row-standardized ``W`` has its
#: spectrum in ``[−1, 1]``); within this margin of the edge the value is ``−inf``.
_PD_MARGIN = 1e-10


class ExactCholLogdet:
    """``log|I − ρW|`` and its gradient at any ρ, by CHOLMOD.

    Raises ``ValueError`` at construction if ``W`` admits no symmetrizing
    diagonal (a directed graph); use ``logdet_decomp="lu"`` with an
    interpolating method there.
    """

    def __init__(self, W):
        self._ctx = CholChebContext(W)
        self.n = self._ctx.n

    def value(self, rho) -> np.ndarray:
        """Exact ``log|I − ρW|`` for each ρ; ``−inf`` at and beyond ``|ρ| = 1``.

        ``−inf`` is the limit at the stability edge, where the eigenvalue form
        gives the same, and keeps a sampler's proposals there out of support.
        """
        r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        out = np.full(r.shape, -np.inf)
        inside = np.abs(r) < 1.0 - _PD_MARGIN
        if inside.any():
            out[inside] = self._ctx.logdet_at(r[inside])
        return out

    def grad(self, rho) -> np.ndarray:
        """``d/dρ log|I − ρW|`` for each ρ (central difference of exact values)."""
        r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        both = self.value(np.concatenate([r + _GRAD_STEP, r - _GRAD_STEP]))
        return (both[: r.size] - both[r.size :]) / (2.0 * _GRAD_STEP)


class ExactCholLogdetGradOp(pt.Op):
    """``T · d/dρ log|I − ρW|`` (no second derivative: NUTS needs none)."""

    __props__ = ("_op_id",)

    def __init__(self, evaluator: ExactCholLogdet, T: int = 1):
        self._evaluator = evaluator
        self._T = int(T)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho):
        rho = pt.as_tensor_variable(rho, dtype="float64")
        return Apply(self, [rho], [pt.dscalar()])

    def perform(self, node, inputs, outputs):
        g = self._evaluator.grad(float(inputs[0]))[0]
        outputs[0][0] = np.asarray(self._T * g, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        return [()]

    def pullback(self, inputs, outputs, output_grads):
        return [pt.zeros_like(inputs[0])]


class ExactCholLogdetOp(pt.Op):
    """``T · log|I − ρW|`` by CHOLMOD, differentiable in ρ."""

    __props__ = ("_op_id",)

    def __init__(self, evaluator: ExactCholLogdet, T: int = 1):
        self._evaluator = evaluator
        self._T = int(T)
        self._grad = ExactCholLogdetGradOp(evaluator, T)
        self._op_id = next(_op_id_counter)
        super().__init__()

    def make_node(self, rho):
        rho = pt.as_tensor_variable(rho, dtype="float64")
        return Apply(self, [rho], [pt.dscalar()])

    def perform(self, node, inputs, outputs):
        v = self._evaluator.value(float(inputs[0]))[0]
        outputs[0][0] = np.asarray(self._T * v, dtype=np.float64)

    def infer_shape(self, node, input_shapes):
        return [()]

    def pullback(self, inputs, outputs, output_grads):
        return [output_grads[0] * self._grad(inputs[0])]


def exact_chol_logdet_fn(evaluator: ExactCholLogdet, T: int = 1):
    """A PyTensor ``(rho) -> T·log|I − ρW|`` expression, differentiable in ρ."""
    op = ExactCholLogdetOp(evaluator, T)
    return lambda rho: op(rho)


register_logdet_numba_dispatch()
