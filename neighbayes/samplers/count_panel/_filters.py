"""Spatial filters ``U = (I_T ⊗ A(ρ))⁻¹X`` for the count panel kernel.

Each filter knows its ρ names, their box bounds, the admissible region, and how
to solve the per-period system for a time-first stacked ``(N·T, k)`` right-hand
side (``N`` groups per period).  One factorization of ``A`` covers all periods.

A filter whose group effects enter inside it also gives the per-period ``A`` at
ρ (:meth:`effect_operator`), ``AᵀA`` with its log-determinant when the solve's
factorization already holds it (:meth:`effect_gram`), and a sparsity pattern
covering ``AᵀA`` at every ρ (:meth:`effect_pattern`); see
:class:`~neighbayes.samplers._utils._group_effects.SARGroupProjector`.
:class:`NoFilter` returns ``None``: with ``A = I`` inside and outside
coincide.  The separable flow filter builds its ``n² × n²`` operator
explicitly, which suits moderate ``n``; the structured flow kernels are the
scale path.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _stack(X: np.ndarray, N: int, T: int) -> np.ndarray:
    """``(N·T, k)`` time-first → ``(N, T·k)``: one RHS block per period."""
    k = X.shape[1]
    return X.reshape(T, N, k).transpose(1, 0, 2).reshape(N, T * k)


def _unstack(U: np.ndarray, N: int, T: int) -> np.ndarray:
    k = U.shape[1] // T
    return U.reshape(N, T, k).transpose(1, 0, 2).reshape(T * N, k)


class NoFilter:
    """``A = I``: the aspatial model (ρ fixed at zero)."""

    names: tuple[str, ...] = ()

    def admissible(self, rho: dict) -> bool:
        return True

    def bounds(self, name: str) -> tuple[float, float]:
        raise KeyError(name)

    def solve(self, rho: dict, X: np.ndarray) -> np.ndarray:
        return X

    def effect_operator(self, rho: dict) -> None:
        return None

    def init(self, rng: np.random.Generator) -> dict:
        return {}


def _gram_pattern(*Ws: sp.spmatrix) -> sp.csc_matrix:
    """A matrix with the pattern of ``AᵀA`` for every ``A = I − Σ ρ_k W_k``.

    Built from ``I + Σ |W_k|`` so that no entry cancels for want of a sign.
    """
    n = Ws[0].shape[0]
    B = sp.eye(n, format="csc") + sum(abs(sp.csc_matrix(W)) for W in Ws)
    return (B.T @ B).tocsc()


class SARFilter:
    """Place-based ``A = I_N − ρW`` with the package's SuiteSparse routing."""

    names = ("rho",)

    def __init__(self, W: sp.spmatrix, T: int, lower: float, upper: float):
        from ..negbin_reduced._flow_structured import _FilterSolver

        self.W_csc = sp.csc_matrix(W, dtype=np.float64)
        self.N = self.W_csc.shape[0]
        self.T = int(T)
        self._bounds = (float(lower), float(upper))
        self._solver = _FilterSolver(self.W_csc)
        self._gram_parts = None

    def admissible(self, rho: dict) -> bool:
        lo, hi = self._bounds
        return lo < rho["rho"] < hi

    def bounds(self, name: str) -> tuple[float, float]:
        return self._bounds

    def solve(self, rho: dict, X: np.ndarray) -> np.ndarray:
        U = self._solver.at(float(rho["rho"])).solve(_stack(X, self.N, self.T))
        return _unstack(np.asarray(U).reshape(self.N, -1), self.N, self.T)

    def effect_operator(self, rho: dict) -> sp.csc_matrix:
        """``A = I − ρW``, which the group effects pass through."""
        return (sp.eye(self.N, format="csc") - float(rho["rho"]) * self.W_csc).tocsc()

    def effect_gram(self, rho: dict):
        """``(AᵀA, log|AᵀA|)`` at ρ, the log-determinant ``2 log|A|`` from the
        factorization :meth:`solve` holds (refactored only for a new ρ)."""
        r = float(rho["rho"])
        if self._gram_parts is None:
            self._gram_parts = (
                sp.eye(self.N, format="csc"),
                (self.W_csc + self.W_csc.T).tocsc(),
                (self.W_csc.T @ self.W_csc).tocsc(),
            )
        eye, W_sym, WtW = self._gram_parts
        gram = (eye - r * W_sym + r * r * WtW).tocsc()
        return gram, 2.0 * self._solver.at(r).logdet()

    def effect_pattern(self) -> sp.csc_matrix:
        return _gram_pattern(self.W_csc)

    def init(self, rng: np.random.Generator) -> dict:
        return {"rho": float(rng.uniform(-0.1, 0.1))}


class FlowSeparableFilter:
    """Separable flow filter ``A = L_o ⊗ L_d`` (``ρ_w = −ρ_d ρ_o``)."""

    names = ("rho_d", "rho_o")

    def __init__(self, W: sp.spmatrix, T: int, lower: float, upper: float):
        self.W_csc = sp.csc_matrix(W, dtype=np.float64)
        self.n = self.W_csc.shape[0]
        self.T = int(T)
        self._bounds = (float(lower), float(upper))
        self._unit_solvers = None

    def admissible(self, rho: dict) -> bool:
        lo, hi = self._bounds
        return all(lo < rho[k] < hi for k in self.names)

    def bounds(self, name: str) -> tuple[float, float]:
        return self._bounds

    def solve(self, rho: dict, X: np.ndarray) -> np.ndarray:
        from ..negbin_reduced._flow import _solve_A_separable

        return _solve_A_separable(
            rho["rho_d"], rho["rho_o"], X, self.W_csc, self.n, T=self.T
        )

    def effect_operator(self, rho: dict) -> sp.csc_matrix:
        """``A = L_o ⊗ L_d`` as an explicit ``n² × n²`` matrix (moderate ``n``)."""
        eye = sp.eye(self.n, format="csc")
        Lo = (eye - float(rho["rho_o"]) * self.W_csc).tocsc()
        Ld = (eye - float(rho["rho_d"]) * self.W_csc).tocsc()
        return sp.kron(Lo, Ld, format="csc")

    def effect_gram(self, rho: dict):
        """``(AᵀA, log|AᵀA|)``, ``log|A| = n(log|L_o| + log|L_d|)`` from two
        ``n × n`` factorizations."""
        from ..negbin_reduced._flow_structured import _FilterSolver

        if self._unit_solvers is None:
            self._unit_solvers = (_FilterSolver(self.W_csc), _FilterSolver(self.W_csc))
        so, sd = self._unit_solvers
        A = self.effect_operator(rho)
        logdet = (
            so.at(float(rho["rho_o"])).logdet() + sd.at(float(rho["rho_d"])).logdet()
        )
        return (A.T @ A).tocsc(), 2.0 * self.n * logdet

    def effect_pattern(self) -> sp.csc_matrix:
        eye = sp.eye(self.n, format="csc")
        W = self.W_csc
        return _gram_pattern(sp.kron(eye, W), sp.kron(W, eye), sp.kron(W, W))

    def init(self, rng: np.random.Generator) -> dict:
        return {k: float(rng.uniform(-0.1, 0.1)) for k in self.names}


class FlowUnrestrictedFilter:
    """Unrestricted flow filter ``A = I − ρ_d W_d − ρ_o W_o − ρ_w W_w``."""

    names = ("rho_d", "rho_o", "rho_w")

    def __init__(
        self,
        Wd: sp.csr_matrix,
        Wo: sp.csr_matrix,
        Ww: sp.csr_matrix,
        W: sp.spmatrix,
        T: int,
        lower: float,
        upper: float,
        positive: bool,
    ):
        self.Wd, self.Wo, self.Ww = Wd, Wo, Ww
        self.Nf = Wd.shape[0]
        self.T = int(T)
        self.positive = bool(positive)
        self._bounds = (0.0 if positive else float(lower), float(upper))
        from ..negbin_reduced._flow import real_spectrum_bounds

        self.eig_min, self.eig_max = real_spectrum_bounds(W)

    def admissible(self, rho: dict) -> bool:
        from ..negbin_reduced._flow import flow_system_is_invertible

        lo, hi = self._bounds
        r = [rho[k] for k in self.names]
        if not all(lo < v < hi for v in r):
            return False
        return flow_system_is_invertible(*r, self.eig_min, self.eig_max)

    def bounds(self, name: str) -> tuple[float, float]:
        return self._bounds

    def solve(self, rho: dict, X: np.ndarray) -> np.ndarray:
        from ..negbin_reduced._flow import (
            _assemble_A_unrestricted,
            _solve_A_unrestricted,
        )

        A = _assemble_A_unrestricted(
            rho["rho_d"], rho["rho_o"], rho["rho_w"], self.Wd, self.Wo, self.Ww, self.Nf
        )
        return _solve_A_unrestricted(A, X, T=self.T)

    def effect_operator(self, rho: dict) -> sp.csc_matrix:
        """``A = I − ρ_d W_d − ρ_o W_o − ρ_w W_w``, which the pair effects pass through."""
        from ..negbin_reduced._flow import _assemble_A_unrestricted

        return _assemble_A_unrestricted(
            rho["rho_d"], rho["rho_o"], rho["rho_w"], self.Wd, self.Wo, self.Ww, self.Nf
        ).tocsc()

    def effect_gram(self, rho: dict):
        """``(AᵀA, None)``: no factorization of ``A`` is kept to read
        ``log|AᵀA|`` from, so the projector factors ``AᵀA`` itself."""
        A = self.effect_operator(rho)
        return (A.T @ A).tocsc(), None

    def effect_pattern(self) -> sp.csc_matrix:
        return _gram_pattern(self.Wd, self.Wo, self.Ww)

    def init(self, rng: np.random.Generator) -> dict:
        lo = 0.0 if self.positive else -0.1
        return {
            "rho_d": float(rng.uniform(lo, 0.1)),
            "rho_o": float(rng.uniform(lo, 0.1)),
            "rho_w": float(rng.uniform(0.0 if self.positive else -0.05, 0.05)),
        }
