"""The zero part of the place two-part count models (ZINB, hurdle).

Its design and weights default to the count part's.  The flows build their own
zero part (:mod:`..flow_panel`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._base._shared import _parse_W


def _selection_design(Z, formula_zero, data, n: int):
    """The selection design and its column names, or ``(None, None)`` for ``Z = X``."""
    if Z is not None and formula_zero is not None:
        raise ValueError("Pass Z or formula_zero, not both.")
    if formula_zero is not None:
        if data is None:
            raise ValueError("formula_zero needs data (formula mode).")
        from formulaic import model_matrix

        rhs = formula_zero.split("~", 1)[-1].strip()
        mm = model_matrix(rhs, data)
        names, Z_arr = list(mm.columns), np.asarray(mm, dtype=np.float64)
    elif Z is not None:
        if isinstance(Z, pd.DataFrame):
            names = [str(c) for c in Z.columns]
        Z_arr = np.asarray(Z, dtype=np.float64)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if not isinstance(Z, pd.DataFrame):
            names = [f"z{j}" for j in range(Z_arr.shape[1])]
    else:
        return None, None
    if Z_arr.shape[0] != n:
        raise ValueError(f"Z has {Z_arr.shape[0]} rows but y has {n} observations.")
    return Z_arr, names


class ZeroPartMixin:
    """The zero part's design, weights, bounds and per-period solves."""

    def _init_zero_part(
        self,
        data: Optional[pd.DataFrame],
        X_zero=None,
        formula_zero: Optional[str] = None,
        W_zero=None,
    ) -> None:
        """``_Z`` and ``_W_zero_sparse``: ``X`` and ``W`` unless given.

        Runs before a panel's effects drop the columns they absorb, so a
        default ``Z`` is the raw design.
        """
        if formula_zero is not None and data is not None and self._is_panel:
            # The panel's rows, sorted as the base class sorted them.
            order = list(self._panel_index.columns)
            data = data.sort_values(order).reset_index(drop=True)
        self._Z, self._sel_feature_names = _selection_design(
            X_zero, formula_zero, data, self._N * self._T
        )
        if self._Z is None:
            self._Z = np.asarray(self._X, dtype=np.float64).copy()
            self._sel_feature_names = list(self._feature_names)
        if W_zero is not None:
            self._W_zero_sparse, self._is_sel_row_std = _parse_W(W_zero, self._N)
            self._same_W = False
        else:
            self._W_zero_sparse = self._W_sparse
            self._is_sel_row_std = self._is_row_std
            self._same_W = True

    def _bounds(self, zero: bool = False) -> tuple[float, float]:
        """ρ's prior bounds in the count part, or in the zero part (``*_zero`` keys)."""
        b = self._logdet_bounds
        z = "_zero" if zero else ""
        lo = float(self.priors.get(f"rho_lower{z}", b.rho_min))
        hi = float(self.priors.get(f"rho_upper{z}", b.rho_max))
        return max(lo, b.rho_min), min(hi, b.rho_max)

    def _solve_periods(self, W, coef: float, b: np.ndarray) -> np.ndarray:
        """``(I_T ⊗ (I − coef·W))⁻¹ b`` for a time-first stacked ``b``."""
        N, T = self._N, self._T
        A = (sp.eye(N, format="csc") - coef * W).tocsc()
        out = sp.linalg.splu(A).solve(np.ascontiguousarray(b.reshape(T, N).T))
        return out.T.reshape(-1)

    def _filters(self):
        """The count and zero parts' filters for the panel Gibbs kernels."""
        from ...samplers.count_panel._filters import NoFilter, SARFilter

        if not self._count_spatial:
            return NoFilter(), NoFilter()
        return (
            SARFilter(self._W_sparse, self._T, *self._bounds()),
            SARFilter(self._W_zero_sparse, self._T, *self._bounds(zero=True)),
        )
