r"""The structure of a separable flow design, and the filter solves it takes.

Model: ``y_od`` given ``η = A⁻¹Xβ`` with the separable filter ``A = L_o ⊗ L_d``,
``L_k = I_n − ρ_k W``.  On the ``n × n`` flow array (rows = origins) ``A⁻¹``
acts as ``Y ↦ L_o⁻¹ Y L_d⁻ᵀ``, so every solve is a pair of ``n × n``
factorizations with ``n`` right-hand sides; nothing ``N × N`` exists.

**Design structure.**  Flow design columns are of two kinds on the flow array:

* *rank one*: the intercept (``1·1ᵀ``), destination attributes (``1·xᵀ``) and
  origin attributes (``x·1ᵀ``).  ``A⁻¹`` keeps them rank one,
  ``(L_o⁻¹u)(L_d⁻¹v)ᵀ``, so each costs two ``n``-vector solves.
* *full*: log-distance, the intra dummy, ``intra × x`` and anything else.  Each
  costs a two-sided ``n × n`` solve.

Columns are classified from the data (constant rows or constant columns), not
from their names.  ``T`` periods stacked time-first share ``W`` and ``ρ``; a
column is rank one only if it is rank one in every period, and vectors and
full-rank arrays are de-duplicated across periods, so a time-invariant column
(log-distance, the intra dummy) is stored and solved once.  The separable
flow samplers (:mod:`._flow_exact`, :mod:`._flow_structured_fe` and the zero
models') build on these pieces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Design classification
# ---------------------------------------------------------------------------


@dataclass
class FlowDesignStructure:
    """Rank-one and full-rank columns of a (panel) flow design.

    Period ``t``'s rank-one column ``j`` is ``U[:, u_idx[t, j]] V[:, v_idx[t, j]]ᵀ``
    and its full-rank column ``j`` is ``F[f_idx[t, j]]`` (an ``n × n`` array).
    ``U``, ``V`` and ``F`` are de-duplicated across columns and periods.
    """

    n: int
    T: int
    cheap_cols: np.ndarray
    full_cols: np.ndarray
    U: np.ndarray  # (n, n_u) origin-side vectors
    V: np.ndarray  # (n, n_v) destination-side vectors
    u_idx: np.ndarray  # (T, k_c)
    v_idx: np.ndarray  # (T, k_c)
    F: list  # distinct full-rank (n, n) arrays
    f_idx: np.ndarray  # (T, k_e)

    @property
    def k(self) -> int:
        return len(self.cheap_cols) + len(self.full_cols)


class _Dedup:
    """Index distinct arrays by content."""

    def __init__(self):
        self.items: list[np.ndarray] = []
        self._index: dict[bytes, int] = {}

    def add(self, a: np.ndarray) -> int:
        key = np.ascontiguousarray(a).tobytes()
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(np.ascontiguousarray(a))
        return self._index[key]


def _rank_one(Y: np.ndarray):
    """``(u, v)`` with ``Y = u vᵀ`` for constant-row or constant-column ``Y``."""
    if np.all(Y == Y[0:1, :]):
        return np.ones(Y.shape[0]), Y[0].copy()
    if np.all(Y == Y[:, 0:1]):
        return Y[:, 0].copy(), np.ones(Y.shape[1])
    return None


def classify_flow_design(X: np.ndarray, n: int, T: int = 1) -> FlowDesignStructure:
    """Split a time-first stacked flow design into rank-one and full-rank parts."""
    X = np.asarray(X, dtype=np.float64)
    N = n * n
    if X.shape[0] != N * T:
        raise ValueError(f"X has {X.shape[0]} rows, expected n²·T = {N * T}.")
    blocks = X.reshape(T, n, n, X.shape[1])  # [t, o, d, j]
    cheap, full = [], []
    for j in range(X.shape[1]):
        factors = [_rank_one(blocks[t, :, :, j]) for t in range(T)]
        (cheap if all(f is not None for f in factors) else full).append((j, factors))
    Ud, Vd, Fd = _Dedup(), _Dedup(), _Dedup()
    u_idx = np.zeros((T, len(cheap)), dtype=int)
    v_idx = np.zeros((T, len(cheap)), dtype=int)
    for c, (_, factors) in enumerate(cheap):
        for t, (u, v) in enumerate(factors):
            u_idx[t, c] = Ud.add(u)
            v_idx[t, c] = Vd.add(v)
    f_idx = np.zeros((T, len(full)), dtype=int)
    for c, (j, _) in enumerate(full):
        for t in range(T):
            f_idx[t, c] = Fd.add(blocks[t, :, :, j])
    stack = lambda items: np.column_stack(items) if items else np.zeros((n, 0))  # noqa: E731
    return FlowDesignStructure(
        n=n,
        T=T,
        cheap_cols=np.asarray([j for j, _ in cheap], dtype=int),
        full_cols=np.asarray([j for j, _ in full], dtype=int),
        U=stack(Ud.items),
        V=stack(Vd.items),
        u_idx=u_idx,
        v_idx=v_idx,
        F=Fd.items,
        f_idx=f_idx,
    )


# ---------------------------------------------------------------------------
# I − ρW solvers
# ---------------------------------------------------------------------------


class _FilterSolver:
    """``(I − ρW)⁻¹`` with the package's SuiteSparse routing, refactored per ρ."""

    def __init__(self, W_csc: sp.csc_matrix):
        from .._utils._spatial_normal import CholmodFactor
        from ._core import _make_cholmod_pattern, make_sar_solver

        n = W_csc.shape[0]
        W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
        self._solver = make_sar_solver(CholmodFactor(pattern), W_csc, W_sym, WtW, n)
        self.rho: float | None = None

    def at(self, rho: float) -> "_FilterSolver":
        if rho != self.rho:
            self._solver.factorize(float(rho))
            self.rho = float(rho)
        return self

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(self._solver.solve(np.asarray(rhs, dtype=np.float64)))

    def logdet(self) -> float:
        """``log|det(I − ρW)|`` from the current factorization."""
        return float(self._solver.logdet())


# ---------------------------------------------------------------------------
# Per-period algebra
# ---------------------------------------------------------------------------


def _low_rank(A, B, u_idx, v_idx, coef):
    """``Σ_j coef_j a_j b_jᵀ`` as an ``n × n`` array."""
    return (A[:, u_idx] * coef) @ B[:, v_idx].T


def _prior_arrays(priors, k):
    mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,)).copy()
    sd = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    return mu, 1.0 / sd**2
