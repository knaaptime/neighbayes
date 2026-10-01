"""Custom pytensor Ops for differentiable sparse linear solves.

This subpackage provides two families of differentiable pytensor
:class:`~pytensor.graph.op.Op` classes that wrap
:func:`scipy.sparse.linalg.spsolve` with analytically derived gradients via
the **adjoint method**.

**General (unrestricted) flow models** — :class:`SparseFlowSolveOp` and
:class:`SparseFlowSolveMatrixOp` — handle the full three-parameter system
matrix :math:`A(\\rho_d, \\rho_o, \\rho_w)`.

**Separable flow models** — :class:`KroneckerFlowSolveOp` and
:class:`KroneckerFlowSolveMatrixOp` — exploit the constraint
:math:`\\rho_w = -\\rho_d \\rho_o` so that the system matrix factors as
:math:`A = L_o \\otimes L_d`.

**SAR models** — :class:`SparseSARSolveOp` — handles the single-parameter
system matrix :math:`A(\\rho) = I_N - \\rho W`.

All solves also have corresponding VJP (vector-Jacobian product) Ops for
gradient-based inference.

The public API is re-exported from the submodules so that
``from neighbayes._ops import SparseFlowSolveOp`` continues to work.
"""

from __future__ import annotations

import scipy.sparse as sp  # re-exported for test compatibility

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
from ._backend import (
    _DenseLU,
    _factor_kron_factor,
    _klu_available,
    _kron_dense_max,
    _make_cached_sparse_solver,
    _select_sparse_backend,
    _solve_sparse_matrix,
    _solve_sparse_vector,
    _SparseFactorSolver,
    _warn_sparse_auto_scipy_fallback_once,
)
from ._flow import (
    FlowLogdetOp,
    SparseFlowSolveMatrixOp,
    SparseFlowSolveOp,
    _SparseFlowVJPMatrixOp,
    _SparseFlowVJPOp,
)
from ._instrument import (
    _INSTRUMENT_ENABLED,
    _measure_callback,
    _record_callback,
    get_callback_stats,
    reset_callback_stats,
)
from ._kron_solve import kron_solve_matrix, kron_solve_vec
from ._kronecker import (
    KroneckerFlowSolveMatrixOp,
    KroneckerFlowSolveOp,
    _KroneckerFlowVJPMatrixOp,
    _KroneckerFlowVJPOp,
)
from ._sar import (
    SparseSARSolveOp,
    _SparseSARVJPOp,
)

__all__ = [
    # Flow Ops
    "FlowLogdetOp",
    "SparseFlowSolveOp",
    "SparseFlowSolveMatrixOp",
    # Kronecker Flow Ops
    "KroneckerFlowSolveOp",
    "KroneckerFlowSolveMatrixOp",
    # SAR Ops
    "SparseSARSolveOp",
    # Standalone utilities
    "kron_solve_vec",
    "kron_solve_matrix",
    # Instrumentation
    "reset_callback_stats",
    "get_callback_stats",
    # Backend helpers
    "_select_sparse_backend",
    "_make_cached_sparse_solver",
]


# ---------------------------------------------------------------------------
# Callback instrumentation (gated by NEIGHBAYES_OP_INSTRUMENT env var)
# ---------------------------------------------------------------------------


def _instrument_op_perform_class(op_cls, name: str) -> None:
    """Wrap ``perform`` to record callback timing/counters for benchmarks."""
    original = getattr(op_cls, "perform", None)
    if original is None or getattr(original, "_callback_instrumented", False):
        return

    def _wrapped(self, node, inputs, outputs):
        with _measure_callback(name):
            return original(self, node, inputs, outputs)

    _wrapped._callback_instrumented = True  # type: ignore[attr-defined]
    setattr(op_cls, "perform", _wrapped)


if _INSTRUMENT_ENABLED:
    for _cls_name, _cls in (
        ("_SparseFlowVJPOp", _SparseFlowVJPOp),
        ("SparseFlowSolveOp", SparseFlowSolveOp),
        ("_SparseFlowVJPMatrixOp", _SparseFlowVJPMatrixOp),
        ("SparseFlowSolveMatrixOp", SparseFlowSolveMatrixOp),
        ("_KroneckerFlowVJPOp", _KroneckerFlowVJPOp),
        ("KroneckerFlowSolveOp", KroneckerFlowSolveOp),
        ("_KroneckerFlowVJPMatrixOp", _KroneckerFlowVJPMatrixOp),
        ("KroneckerFlowSolveMatrixOp", KroneckerFlowSolveMatrixOp),
        ("_SparseSARVJPOp", _SparseSARVJPOp),
        ("SparseSARSolveOp", SparseSARSolveOp),
    ):
        _instrument_op_perform_class(_cls, _cls_name)


# ---------------------------------------------------------------------------
# Numba dispatch registration (no-op when Numba is not installed)
# ---------------------------------------------------------------------------
from .._numba_dispatch import register_numba_dispatch as _register_numba_dispatch

_register_numba_dispatch()


# ---------------------------------------------------------------------------
# JAX dispatch registration (no-op when JAX is not installed)
# ---------------------------------------------------------------------------
from .._jax_dispatch import register_jax_dispatch as _register_jax_dispatch

_register_jax_dispatch()
