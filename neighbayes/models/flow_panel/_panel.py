"""Panel extensions for Bayesian spatial flow (origin-destination) models.

This module introduces a panel-flow base class and four panel model
variants that extend the cross-sectional flow models to balanced panel data.

The panel stack uses time-first ordering. For each period t, the response is
an n^2-length vectorized origin-destination flow array, and all periods are
stacked to length n^2 * T.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
from ..._logdet import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ..._ops import kron_solve_matrix
from ...graph import _weights_to_csr, flow_lags, flow_trace_blocks
from .._base._nb import nb_alpha_rv
from .._mixins._count_panel import CountPanelFEMixin, pop_mundlak, require_counts
from .._mixins._flow_shared import FlowSharedMethods
from .._mixins._zinb import ZINBMixin
from ..flow import (
    _compute_ols_flow_effects,
)
from ..panel_base import SpatialPanelModel, _absorbed_columns, _demean_panel


class FlowPanelModel(FlowSharedMethods, SpatialPanelModel):
    """Abstract base class for balanced panel spatial flow models.

    Parameters
    ----------
    y : array-like
        Stacked panel response in one of these forms:
        - shape (T, n, n)
        - shape (T, n^2)
        - shape (n^2 * T,)
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on n units.
    X : np.ndarray or pandas.DataFrame, shape (n^2 * T, p)
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods.
    col_names : list[str], optional
        Feature names for X.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects.
        If omitted, inferred from column names with ``dest_`` prefix.
    model : int, default 0
        Fixed-effects transform mode:
        0 pooled, 1 pair FE, 2 time FE, 3 two-way FE.
    priors : dict, optional
        Prior overrides.
    logdet_method : str, optional
        Flow log-determinant method; ``None`` (default) auto-selects.
        Concrete subclasses override the default with their recommended
        method (see the cross-sectional :class:`~neighbayes.models.flow.FlowModel`).
    restrict_positive : bool, default True
        If True, use simplex-constrained rho parameters.
    robust : bool, default False
        If True, use Student-t observation errors.
    """

    def __init__(
        self,
        y: Union[np.ndarray, pd.Series],
        X: Union[np.ndarray, pd.DataFrame],
        W,
        T: int,
        col_names: Optional[list[str]] = None,
        k: Optional[int] = None,
        priors: Optional[dict] = None,
        logdet_method: Optional[str] = None,
        restrict_positive: bool = True,
        robust: bool = False,
        symmetric_xo_xd: Optional[bool] = None,
        effects: int = 0,
    ):
        self.priors = priors or {}
        self.logdet_method = logdet_method
        self.restrict_positive = restrict_positive
        self.robust = robust
        self.effects = int(effects)
        if self.effects not in (0, 1, 2, 3):
            raise ValueError("effects must be one of {0,1,2,3}.")

        self._is_row_std = True  # Graph is assumed row-standardized
        self._idata: Optional[xr.DataTree] = None
        self._pymc_model: Optional[pm.Model] = None

        # Validate and extract n x n W
        self._W_sparse: sp.csr_matrix = _weights_to_csr(W)
        self._n: int = self._W_sparse.shape[0]
        self._N_flow: int = self._n * self._n

        # Validate T
        self._T: int = int(T)
        if self._T <= 0:
            raise ValueError(f"T must be positive, got {T}.")

        # Validate y
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 3:
            expected = (self._T, self._n, self._n)
            if y_arr.shape != expected:
                raise ValueError(
                    f"y with 3 dims must have shape {expected}, got {y_arr.shape}."
                )
            y_vec = y_arr.reshape(self._T, self._N_flow).reshape(-1)
        elif y_arr.ndim == 2:
            if y_arr.shape == (self._T, self._N_flow):
                y_vec = y_arr.reshape(-1)
            elif y_arr.shape == (self._n, self._n) and self._T == 1:
                y_vec = y_arr.ravel()
            else:
                raise ValueError(
                    "y with 2 dims must have shape (T, n^2) or (n, n) when T=1. "
                    f"Got {y_arr.shape}."
                )
        elif y_arr.ndim == 1:
            expected_len = self._N_flow * self._T
            if y_arr.shape[0] != expected_len:
                raise ValueError(
                    f"y vector must have length n^2*T={expected_len}, got {y_arr.shape[0]}."
                )
            y_vec = y_arr
        else:
            raise ValueError("y must be a 1-D, 2-D, or 3-D array.")
        self._y_raw = y_vec

        # Validate X
        if isinstance(X, pd.DataFrame):
            if col_names is None:
                col_names = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]

        expected_rows = self._N_flow * self._T
        if X_arr.shape[0] != expected_rows:
            raise ValueError(
                f"X must have n^2*T={expected_rows} rows, got {X_arr.shape[0]}."
            )
        self._X_raw = X_arr

        if col_names is not None:
            self._feature_names: list[str] = list(col_names)
        elif X_arr.shape[1] == 0:
            self._feature_names = []
        else:
            self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

        if k is not None:
            self._k: int = int(k)
            self._k_d: int = int(k)
            self._k_o: int = int(k)
        else:
            dest_cols = [
                name for name in self._feature_names if name.startswith("dest_")
            ]
            orig_cols = [
                name for name in self._feature_names if name.startswith("orig_")
            ]
            self._k_d = len(dest_cols)
            self._k_o = len(orig_cols)
            self._k = self._k_d  # backward compat alias

        # Locate β_intra slice for the Thomas-Agnan & LeSage (2014) intra
        # contribution.
        if self._k_d > 0:
            intra_cols = [
                i
                for i, name in enumerate(self._feature_names)
                if name.startswith("intra_")
            ]
            self._intra_idx: Optional[np.ndarray] = (
                np.asarray(intra_cols, dtype=np.int64) if intra_cols else None
            )
        else:
            self._intra_idx = None

        # Detect Xo == Xd symmetry on the (undemeaned) raw design.
        if (
            symmetric_xo_xd is None
            and self._k_d > 0
            and self._k_d == self._k_o
            and X_arr.shape[1] >= 2 + self._k_d + self._k_o
        ):
            dest_block = X_arr[:, 2 : 2 + self._k_d]
            orig_block = X_arr[:, 2 + self._k_d : 2 + self._k_d + self._k_o]
            self._symmetric_xo_xd: bool = bool(np.array_equal(dest_block, orig_block))
        else:
            self._symmetric_xo_xd = (
                bool(symmetric_xo_xd)
                if symmetric_xo_xd is not None
                else (self._k_d == self._k_o)
            )

        # Demean panel data using N_flow panel units (OD pairs)
        self._y, self._X = _demean_panel(
            self._y_raw,
            self._X_raw,
            self._N_flow,
            self._T,
            self.effects,
        )

        # Columns fixed within the demeaning groups (intercept, intra indicator
        # and log distance under pair effects; any time-invariant attribute)
        # are zero after demeaning: absorbed by the fixed effects, not
        # identified.  Drop them from the sampled design, as SpatialPanelModel
        # does; effects read beta in the full layout via _beta_layout, with NaN
        # for absorbed columns.
        self._design_feature_names = list(self._feature_names)
        self._beta_keep: Optional[np.ndarray] = None
        if self.effects != 0:
            absorbed = _absorbed_columns(self._X, X_arr)
            if absorbed.any():
                keep = np.flatnonzero(~absorbed)
                self._beta_keep = keep
                self._X = self._X[:, keep]
                self._feature_names = [self._feature_names[j] for j in keep]
                slopes = [
                    name
                    for name, a in zip(self._design_feature_names, absorbed)
                    if a and name.startswith(("dest_", "orig_", "intra_x"))
                ]
                if slopes:
                    warnings.warn(
                        f"{slopes} do not vary within the fixed-effect groups and "
                        "are absorbed; their effects are not identified (NaN).",
                        UserWarning,
                        stacklevel=2,
                    )

        # Build flow weight matrices on N_flow = n^2 system

        # Cache the symmetric 3x3 Kronecker trace matrix used by Bayesian
        # LM diagnostics on flow models: T[i,j] = tr(W_i' W_j) + tr(W_i W_j)
        # for (W_d, W_o, W_w).  Computed in O(nnz) from the n x n base graph.
        self._T_flow_traces: np.ndarray = flow_trace_blocks(self._W_sparse)

        # Spatial lags on demeaned/stationary panel stack
        # Matrix-free: W⊗W alone would hold nnz(W)² entries.
        self._Wd_y, self._Wo_y, self._Ww_y = flow_lags(
            self._W_sparse, self._y, T=self._T
        )

        # Pre-compute logdet data for separable constraint: log|Lo⊗Ld| = n*f(ρ_d) + n*f(ρ_o).
        # Also keep _W_eigs for backward compatibility.
        self._W_eigs: Optional[np.ndarray] = None
        self._separable_logdet_fn = None
        self._separable_logdet_numpy_fn = None
        _SEPARABLE_METHODS = {
            "eigenvalue",
            "chebyshev",
            "cheb_cholesky",
            "aaa",
            "cheb_stochastic",
        }
        if self.logdet_method is None or self.logdet_method in _SEPARABLE_METHODS:
            from ..._logdet._config import resolve_logdet_method

            self._separable_logdet_fn = make_flow_separable_logdet(
                self._W_sparse,
                self._n,
                method=self.logdet_method,
            )
            self._separable_logdet_numpy_fn = make_flow_separable_logdet_numpy(
                self._W_sparse,
                self._n,
                method=self.logdet_method,
            )
            # Populate ``_W_eigs`` only when the resolved method is eigenvalue
            # (auto-selection may resolve None to eigenvalue for small n).
            resolved = resolve_logdet_method(
                self.logdet_method, n=self._n, W=self._W_sparse
            )
            if resolved == "eigenvalue":
                self._W_eigs = np.linalg.eigvals(
                    self._W_sparse.toarray().astype(np.float64)
                ).real

        # The unrestricted panel-flow log-determinant will use the resolvent-
        # Kronecker gradient (per-period, block over T); the old "traces" value
        # method was removed (noise-amplified for large directed W).
        self._traces = None

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return the PyMC model."""

    # ------------------------------------------------------------------
    # Fixed-effects likelihood (Lee & Yu 2010; see SpatialPanelModel)
    # ------------------------------------------------------------------
    # The panel units are the N_flow origin-destination pairs.  With time
    # effects and a row-standardized W, each of W_d, W_o, W_w has unit row
    # sums, so the flow filter removes the eigenvalue 1 - ρ_d - ρ_o - ρ_w
    # ((1 - ρ_d)(1 - ρ_o) when separable).

    @property
    def _fe_unit(self) -> bool:
        return self._lee_yu and self.effects in (1, 3)

    @property
    def _fe_time(self) -> bool:
        return self._lee_yu and self.effects in (2, 3)

    @property
    def _flow_jacobian_shift(self) -> float:
        if not self._fe_time or not self._W_row_standardized():
            return 0.0
        return float(self._jacobian_T)

    @property
    def _flow_fe_dims(self) -> dict:
        """Likelihood dimensions for the flow resolvent samplers."""
        return {
            "jacobian_T": self._jacobian_T,
            "n_eff": self._n_effective,
            "jacobian_shift": self._flow_jacobian_shift,
        }

    def _flow_jacobian(self, logdet, one_minus_rowsum, lib):
        """``jacobian_T·log|A|``, less the time-effects term (``lib``: np or pt)."""
        val = self._jacobian_T * logdet
        m = self._flow_jacobian_shift
        if m:
            val = val - m * lib.log(one_minus_rowsum)
        return val

    def _fe_dof_sigma(self, sigma) -> None:
        """Count ``_n_effective`` observations in a Normal over all rows (σ prior)."""
        surplus = int(np.asarray(self._y).shape[0]) - int(self._n_effective)
        if surplus:
            pm.Potential("fe_dof", surplus * pt.log(sigma))

    def _beta_layout(self, posterior) -> np.ndarray:
        """Posterior ``beta`` draws in the full design layout.

        The effects code indexes the design by position (intercept, intra
        indicator, dest, orig, intra blocks).  Columns absorbed by the fixed
        effects were not sampled and come back as NaN.
        """
        beta = posterior["beta"].values.reshape(-1, len(self._feature_names))
        n_mundlak = len(getattr(self, "_mundlak_names", ()))
        if n_mundlak:
            beta = beta[:, : beta.shape[1] - n_mundlak]
        if getattr(self, "_beta_keep", None) is None:
            return beta
        full = np.full((beta.shape[0], len(self._design_feature_names)), np.nan)
        full[:, self._beta_keep] = beta
        return full

    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute posterior effects per draw."""

    # ------------------------------------------------------------------
    # Public API (fit, spatial_diagnostics_decision, etc.) and sparse filter
    # helpers (_assemble_A, _A_solver, _solve_A, _attach_complete_log_likelihood,
    # etc.) inherited from FlowSharedMethods — see .._mixins._flow_shared
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def spatial_effects(
        self,
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Summarize posterior origin/destination/intra/network/total effects.

        See :meth:`neighbayes.models.flow.FlowModel.spatial_effects` for the
        ``mode`` semantics (auto / combined / separate destination-origin
        sides per Thomas-Agnan & LeSage 2014, §83.5.2).
        """
        from ...diagnostics.spatial_effects import _compute_bayesian_pvalue
        from ..flow import _EFFECT_KEYS

        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        if self._k == 0:
            raise RuntimeError(
                "Cannot compute spatial effects: no `dest_*` columns detected "
                "in the design matrix.  Pass `k=` explicitly when constructing "
                "the model."
            )
        if mode not in {"auto", "combined", "separate"}:
            raise ValueError(
                f"mode must be 'auto', 'combined', or 'separate'; got {mode!r}."
            )

        posterior = self._compute_spatial_effects_posterior(draws=draws)

        if mode == "auto":
            effective_mode = "combined" if self._symmetric_xo_xd else "separate"
        else:
            effective_mode = mode

        if effective_mode == "combined":
            display = [("combined", eff) for eff in _EFFECT_KEYS]
        else:
            display = [(side, eff) for side in ("dest", "orig") for eff in _EFFECT_KEYS]

        feature_names = [
            name[len("dest_") :] if name.startswith("dest_") else name
            for name in self._design_feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(feature_names) != self._k_d:
            feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :] if name.startswith("orig_") else name
            for name in self._design_feature_names
            if name.startswith("orig_")
        ][: self._k_o]
        if len(orig_feature_names) != self._k_o:
            orig_feature_names = [f"y{i}" for i in range(self._k_o)]

        # For combined mode: when k_d == k_o, combined effects are the sum
        # of dest and orig (same variables), so use dest names.
        # When k_d != k_o, combined effects are concatenated (different variables).
        if self._k_d == self._k_o:
            combined_feature_names = feature_names
        else:
            combined_feature_names = feature_names + orig_feature_names

        alpha = (1.0 - ci) / 2.0
        rows = []
        for side, effect_name in display:
            key = effect_name if side == "combined" else f"{side}_{effect_name}"
            samples = posterior[key]
            means = samples.mean(axis=0)
            lower = np.quantile(samples, alpha, axis=0)
            upper = np.quantile(samples, 1.0 - alpha, axis=0)
            pvals = _compute_bayesian_pvalue(samples)
            if side == "combined":
                fnames = combined_feature_names
            elif side == "dest":
                fnames = feature_names
            else:
                fnames = orig_feature_names
            for j, fname in enumerate(fnames):
                rows.append(
                    {
                        "predictor": fname,
                        "side": side,
                        "effect": effect_name,
                        "mean": float(means[j]),
                        "ci_lower": float(lower[j]),
                        "ci_upper": float(upper[j]),
                        "bayes_pvalue": float(pvals[j]),
                    }
                )

        df = pd.DataFrame(rows).set_index(["predictor", "side", "effect"])
        if return_posterior_samples:
            return df, posterior
        return df

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw a single posterior-predictive replicate for the full panel.

        Default Gaussian implementation: :math:`y_{rep,t} = A^{-1}(X_t \\beta + \\sigma\\varepsilon_t)`
        for each period ``t``, with a single sparse :math:`LU` factorization
        of :math:`A` reused across periods.  Subclasses (NB variants)
        override this method.
        """
        N = self._N_flow
        T = self._T
        Xb = self._X @ beta  # (N*T,)
        Xb_mat = Xb.reshape(T, N).T  # (N, T)
        if sigma is not None:
            noise = rng.normal(scale=float(sigma), size=(N, T))
            rhs = Xb_mat + noise
        else:
            rhs = Xb_mat
        # Cached symbolic analysis: A = I - ρ_d W_d - ρ_o W_o - ρ_w W_w
        # shares its sparsity pattern across draws, so sparsax reuses one
        # fill-reducing analysis; scipy fallback still avoids re-symbolic work.
        out = self._solve_A(rho_d, rho_o, rho_w, rhs)  # (N, T)
        return out.T.reshape(-1)  # back to time-first stacked vector

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive samples ``y_rep`` for the full panel stack.

        Parameters
        ----------
        n_draws : int, optional
            Number of posterior draws to use.  Defaults to all.
        random_seed : int, optional
            Seed for the posterior-predictive sampler.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_draws, N_flow * T)`` with posterior-predictive
            flows in time-first stacked order.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d = post["rho_d"].values.reshape(-1)
        rho_o = post["rho_o"].values.reshape(-1)
        rho_w = post["rho_w"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = len(rho_d)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d = rho_d[:total]
            rho_o = rho_o[:total]
            rho_w = rho_w[:total]
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep_period(
                float(rho_d[g]),
                float(rho_o[g]),
                float(rho_w[g]),
                beta_draws[g],
                sigma_g,
                rng,
            )
        return out

    # ------------------------------------------------------------------
    # Internal effects helpers
    # ------------------------------------------------------------------

    def _compute_flow_effects_from_draws(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        rho_w_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute LeSage flow effects from posterior draws.

        Effects are computed using one-period :math:`n^2 \\times n^2` system
        matrices, which are time-invariant under static panel parameters.  See
        :func:`~neighbayes.models.flow._compute_flow_effects` for the
        decomposition.  One sparse :math:`LU` factorization per draw covers all
        :math:`n` shock columns and all :math:`k` predictors.
        """
        k_d = self._k_d
        k_o = self._k_o

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            rho_w_draws = rho_w_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        # Exact LeSage decomposition from W-only moments (no n²-sized arrays).
        return self._flow_effects_for_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws[:, dest_start : dest_start + k_d],
            beta_draws[:, orig_start : orig_start + k_o],
            beta_draws[:, intra_start : intra_start + k_d] if has_intra else None,
        )

    def _compute_flow_effects_kron(
        self,
        rho_d_draws: np.ndarray,
        rho_o_draws: np.ndarray,
        beta_draws: np.ndarray,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Compute LeSage flow effects via Kronecker-factored solve.

        Replaces the :math:`N\\times N` sparse factorization in
        :meth:`_compute_flow_effects_from_draws` with two :math:`n\\times n`
        solves via :func:`~neighbayes._ops.kron_solve_matrix`, exploiting
        :math:`A = L_o \\otimes L_d`.
        """
        k_d = self._k_d
        k_o = self._k_o

        dest_start = 2
        orig_start = 2 + k_d
        intra_start = 2 + k_d + k_o
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k_d
        )

        n_draws_total = len(rho_d_draws)
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            rho_d_draws = rho_d_draws[:n_draws_total]
            rho_o_draws = rho_o_draws[:n_draws_total]
            beta_draws = beta_draws[:n_draws_total]

        # Exact LeSage decomposition from W-only moments (no n²-sized arrays).
        return self._flow_effects_for_draws(
            rho_d_draws,
            rho_o_draws,
            -rho_d_draws * rho_o_draws,  # separable: ρ_w = −ρ_d·ρ_o
            beta_draws[:, dest_start : dest_start + k_d],
            beta_draws[:, orig_start : orig_start + k_o],
            beta_draws[:, intra_start : intra_start + k_d] if has_intra else None,
        )


class _ResolventFlowPanelMixin:
    """Shared ``fit`` dispatch for unrestricted flow panel models.

    Provides ``sampler="gibbs"`` (resolvent-gradient MALA) and
    ``sampler="nuts"`` (PyMC NUTS via ``FlowPanelModel.fit``) paths,
    matching the regular panel model pattern.  The mixin is listed
    first in the bases so its ``fit`` wins and ``FlowPanelModel.fit``
    is reached via explicit delegation.
    """

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str | None = None,
        step_size: float = 5e-4,
        n_probes: int = 48,
        logdet_method: str = "auto",
        n_quad: int = 8,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Parameters
        ----------
        sampler : {"gibbs", "nuts", None}, default None
            ``"gibbs"`` (default) uses the resolvent-gradient MALA sampler;
            ``"nuts"`` uses PyMC NUTS via the base class.
        step_size, n_probes, logdet_method, n_quad : float/int/str
            Resolvent sampler parameters (Gibbs path only).
        n_jobs : int, default -1
            Parallel workers for the Gibbs path (``-1`` = all CPUs).
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the pointwise log-likelihood
            (one value per draw, chain, and flow-period) for ``az.loo``,
            on either sampler.  Off by default, as in PyMC.
        """
        if sampler is None:
            sampler = "gibbs"
        if sampler == "nuts":
            return FlowPanelModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        elif sampler != "gibbs":
            raise ValueError(
                f"sampler must be 'gibbs', 'nuts', or None, got {sampler!r}."
            )
        # --- Gibbs (resolvent) path ---
        self._pymc_model = None
        self._idata = self._sample_resolvent(
            draws=draws,
            tune=tune,
            chains=chains,
            step_size=step_size,
            n_probes=n_probes,
            logdet_method=logdet_method,
            n_quad=n_quad,
            coord_names=list(getattr(self, "_feature_names", []) or []) or None,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            compute_log_likelihood=bool(
                (idata_kwargs or {}).get("log_likelihood", False)
            ),
        )
        return self._idata

    @abstractmethod
    def _sample_resolvent(self, **kwargs) -> xr.DataTree:
        """Subclass hook: call the appropriate resolvent sampling function."""
        ...


class SARFlowPanel(_ResolventFlowPanelMixin, FlowPanelModel):
    """Panel spatial-lag origin-destination flow model with unrestricted dependence.

    For each period :math:`t`, the vectorized flow matrix
    :math:`y_t \\in \\mathbb{R}^{N}` with :math:`N = n^2` satisfies

    .. math::

        y_t = \\rho_d W_d y_t + \\rho_o W_o y_t + \\rho_w W_w y_t + X_t \\beta + \\varepsilon_t,
        \\qquad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2 I_N).

    The panel stack is time-first across :math:`T` periods. The ``model``
    argument controls pooled, pair fixed-effects, time fixed-effects, or
    two-way demeaning before the likelihood is evaluated. The Jacobian
    contribution scales as :math:`T \\log |A(\\rho_d, \\rho_o, \\rho_w)|`
    (:math:`T - 1` under pair effects; see :class:`SpatialPanelModel`).
    ``sampler="nuts"`` evaluates it exactly from ``n × n`` trace moments.

    Parameters
    ----------
    y : array-like
        Stacked panel response in shape ``(T, n, n)``, ``(T, n^2)``, or
        ``(n^2 * T,)``.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on ``n`` units.
    X : np.ndarray or pandas.DataFrame, shape ``(n^2 * T, p)``
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods (must be a positive integer).
    col_names : list of str, optional
        Feature names for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects;
        inferred from columns prefixed ``dest_`` if omitted.
    model : int, default 0
        Fixed-effects transform: ``0`` pooled, ``1`` pair FE, ``2`` time
        FE, ``3`` two-way FE.
    logdet_method : str, default "resolvent"
        Log-determinant method.  The default ``"resolvent"`` samples via the
        per-period resolvent-Kronecker gradient sampler (recommended).
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("rho_simplex", a=ones(4))`` to enforce
        :math:`\\rho_d, \\rho_o, \\rho_w \\geq 0` and
        :math:`\\rho_d + \\rho_o + \\rho_w \\leq 1`. If False, three
        independent ``pm.Uniform(rho_lower, rho_upper)`` priors are used
        with a differentiable quadratic-wall stability potential.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers.  The degrees of freedom :math:`\\nu` are
        **fixed** at ``priors["nu"]`` (default 4, LeSage's ``rval``).
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float or array, default Gelman et al. (2008) — Normal prior mean for ``beta`` (``mean(y)`` on the intercept, 0 otherwise).
        - ``beta_sigma`` : float or array, default Gelman et al. (2008) — Normal prior std for ``beta``, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` : float, default 2 and ``Var(y)`` — InverseGamma prior on ``sigma**2``.
        - ``rho_lower`` : float, default -1.0 — Lower bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``rho_upper`` : float, default 1.0 — Upper bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``nu`` : float, default 4.0 — Fixed Student-t degrees of freedom (only when ``robust=True``).
    """

    def __init__(self, *args, **kwargs):
        # Default to the resolvent-gradient panel sampler (subclasses that need the
        # PyMC path — e.g. the NB count panel — set a different logdet_method).
        kwargs.setdefault("logdet_method", "resolvent")
        super().__init__(*args, **kwargs)

    def _sample_resolvent(self, **kwargs) -> xr.DataTree:
        from ...samplers.gaussian._flow_resolvent import sample_flow_resolvent

        return sample_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            T=self._T,
            restrict_positive=self.restrict_positive,
            fe_dims=self._flow_fe_dims,
            priors=self._flow_gaussian_priors(),
            **kwargs,
        )

    def _trace_logdet(self):
        """Exact ``log|A(ρ)|`` value-and-gradient on the ``n²`` flow system."""
        if getattr(self, "_trace_logdet_obj", None) is None:
            from ..._logdet._flow_kron_traces import FlowKronTraceLogdet

            self._trace_logdet_obj = FlowKronTraceLogdet(self._W_sparse)
        return self._trace_logdet_obj

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import FlowLogdetOp

        pv = self._flow_gaussian_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential("stability", pt.switch(slack > 0.0, 0.0, -1e6 * slack**2))

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)

            # A y = X β + ε, so y | ρ is Normal about the lagged terms plus X β,
            # with the change of variables carried by log|A| per period.
            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            if self.robust:
                pm.StudentT("obs", nu=self._nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            logdet, _ = FlowLogdetOp(self._trace_logdet())(rho_d, rho_o, rho_w)
            pm.Potential(
                "jacobian",
                self._flow_jacobian(logdet, 1 - rho_d - rho_o - rho_w, pt),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        ld = self._trace_logdet()
        rho = [
            np.asarray(posterior[k].values.reshape(-1), dtype=np.float64)
            for k in ("rho_d", "rho_o", "rho_w")
        ]
        logdet = np.array([ld(*r)[0] for r in zip(*rho)])
        return self._flow_jacobian(logdet, 1 - rho[0] - rho[1] - rho[2], np)

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = self._beta_layout(idata.posterior)
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws,
            draws=draws,
        )


class SARFlowSeparablePanel(FlowPanelModel):
    """Panel separable spatial-lag flow model with :math:`\\rho_w = -\\rho_d \\rho_o`.

    For each period :math:`t`,

    .. math::

        y_t = \\rho_d W_d y_t + \\rho_o W_o y_t - \\rho_d \\rho_o W_w y_t + X_t \\beta + \\varepsilon_t,
        \\qquad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2 I_N).

    Under the separability restriction,
    :math:`A = I_N - \\rho_d W_d - \\rho_o W_o + \\rho_d \\rho_o W_w`
    factorizes into Kronecker blocks, which enables the exact or
    approximated eigenvalue-based log-determinant used by this class.

    Parameters
    ----------
    y : array-like
        Stacked panel response in shape ``(T, n, n)``, ``(T, n^2)``, or
        ``(n^2 * T,)``.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on ``n`` units.
    X : np.ndarray or pandas.DataFrame, shape ``(n^2 * T, p)``
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods (must be a positive integer).
    col_names : list of str, optional
        Feature names for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects;
        inferred from columns prefixed ``dest_`` if omitted.
    model : int, default 0
        Fixed-effects transform: ``0`` pooled, ``1`` pair FE, ``2`` time
        FE, ``3`` two-way FE.
    logdet_method : {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"} or None, default None
        ``None`` auto-selects (``aaa`` for directed W, ``cheb_cholesky`` for
        symmetric, ``eigenvalue`` for small n).
        Method for the Kronecker-factored log-determinant.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers.  The degrees of freedom :math:`\\nu` are
        **fixed** at ``priors["nu"]`` (default 4, LeSage's ``rval``).
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float or array, default Gelman et al. (2008) — Normal prior mean for ``beta`` (``mean(y)`` on the intercept, 0 otherwise).
        - ``beta_sigma`` : float or array, default Gelman et al. (2008) — Normal prior std for ``beta``, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` : float, default 2 and ``Var(y)`` — InverseGamma prior on ``sigma**2``.
        - ``rho_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``rho_d`` and ``rho_o``.
        - ``rho_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``rho_d`` and ``rho_o``.
        - ``nu`` : float, default 4.0 — Fixed Student-t degrees of freedom (only when ``robust=True``).

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowPanelModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\\rho` components.
    """

    def __init__(self, y, X, W, **kwargs):
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SARFlowSeparablePanel logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, X, W, **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        pv = self._flow_gaussian_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARFlowSeparablePanel requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            rho_w = pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._flow_jacobian(
                    self._separable_logdet_fn(rho_d, rho_o),
                    (1 - rho_d) * (1 - rho_o),
                    pt,
                ),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        rho_d = np.asarray(posterior["rho_d"].values.reshape(-1), dtype=np.float64)
        rho_o = np.asarray(posterior["rho_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )
        return self._flow_jacobian(
            self._separable_logdet_numpy_fn(rho_d, rho_o), (1 - rho_d) * (1 - rho_o), np
        )

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = self._beta_layout(idata.posterior)
        return self._compute_flow_effects_kron(
            rho_d_draws,
            rho_o_draws,
            beta_draws,
            draws=draws,
        )


class OLSFlowPanel(FlowPanelModel):
    r"""Non-spatial Bayesian OD-flow gravity model for balanced panel data.

    Panel analogue of :class:`~neighbayes.models.flow.OLSFlow`: implements
    the conventional log-linear gravity specification of
    :cite:t:`thomas-agnan2014SpatialEconometric` (eq. 83.2) with no spatial
    lag terms,

    .. math::

        y_{t} = X_{t}\,\beta + \varepsilon_{t}, \quad
        \varepsilon_{t} \sim \mathcal{N}(0, \sigma^{2} I_{N}),

    on a balanced panel of :math:`T` periods, applying the same
    fixed-effects within transform (`model` argument) as the spatial panel
    flow models.  Provided as the canonical null model for Bayesian LM
    diagnostics on panel flow data.

    Parameters
    ----------
    y : array-like
        Stacked panel response in shape ``(T, n, n)``, ``(T, n^2)``, or
        ``(n^2 * T,)``.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on ``n`` units. Required for API
        symmetry but not used in estimation.
    X : np.ndarray or pandas.DataFrame, shape ``(n^2 * T, p)``
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods.
    col_names : list of str, optional
        Feature names for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of destination/origin covariate pairs used by flow
        effects; inferred from columns prefixed ``dest_`` if omitted.
    model : int, default 0
        Fixed-effects transform: ``0`` pooled, ``1`` pair FE, ``2``
        time FE, ``3`` two-way FE.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu``, ``beta_sigma`` (float or array, default Gelman et al.
          2008): Normal prior on :math:`\beta`, scaled to ``sd(y)`` and each
          column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` (float, default 2 and ``Var(y)``):
          InverseGamma prior on :math:`\sigma^2`.
        - ``nu`` (float, default 4.0): Fixed Student-t degrees of
          freedom (only used when ``robust=True``).

        Spatial keys (``rho_*``) are ignored in this aspatial baseline.
    robust : bool, default False
        If True, replace the Normal error with Student-t.
    symmetric_xo_xd : bool, optional
        Whether to constrain origin and destination covariate effects
        to be equal. Forwarded to :class:`FlowPanelModel`.

    Notes
    -----
    All log-determinant precomputation is skipped (``A = I_N`` with
    :math:`|A| = 1`).
    """

    def __init__(self, y, X, W, T, **kwargs):
        # Skip log-determinant precomputation: A = I_N has |A| = 1.
        kwargs.pop("logdet_method", None)
        kwargs.pop("restrict_positive", None)
        super().__init__(y, X, W, T, logdet_method="none", **kwargs)

    def _build_pymc_model(self) -> pm.Model:
        pv = self._flow_gaussian_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)
            mu = pt.dot(X_t, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

        return model

    def _simulate_y_rep_period(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Posterior-predictive replicate ``y_rep = X β + σ ε`` (full panel stack)."""
        Xb = self._X @ beta  # (N_flow * T,)
        if sigma is None:
            return Xb
        return Xb + rng.normal(scale=float(sigma), size=Xb.shape[0])

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the OLS panel gravity model.

        Overrides the base implementation, which expects ``rho_d``,
        ``rho_o``, ``rho_w`` posterior arrays that this model does not
        sample.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        sigma_draws = (
            post["sigma"].values.reshape(-1) if "sigma" in post.data_vars else None
        )

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            if sigma_draws is not None:
                sigma_draws = sigma_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            sigma_g = float(sigma_draws[g]) if sigma_draws is not None else None
            out[g] = self._simulate_y_rep_period(
                0.0, 0.0, 0.0, beta_draws[g], sigma_g, rng
            )
        return out

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        r"""Closed-form Thomas-Agnan & LeSage (2014, Table 83.1) effects.

        Identical to
        :meth:`neighbayes.models.flow.OLSFlow._compute_spatial_effects_posterior`:
        with :math:`A = I_N` the response to any shock equals the shock
        itself, so the per-region averages collapse to closed-form
        expressions in :math:`\beta_d`, :math:`\beta_o`, and
        :math:`\beta_{\text{intra}}`.  Effects are time-invariant under
        the static panel parameters of this model.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        # Local import avoids a circular import at module load time.
        from ..flow import _EFFECT_KEYS

        idata = self._idata
        n = self._n
        k = self._k
        beta_draws = self._beta_layout(idata.posterior)

        dest_start = 2
        orig_start = 2 + k
        intra_start = 2 + 2 * k
        has_intra = (
            self._intra_idx is not None and beta_draws.shape[1] >= intra_start + k
        )

        n_draws_total = beta_draws.shape[0]
        if draws is not None:
            n_draws_total = min(draws, n_draws_total)
            beta_draws = beta_draws[:n_draws_total]

        bd = beta_draws[:, dest_start : dest_start + k]
        bo = beta_draws[:, orig_start : orig_start + k]
        bi = (
            beta_draws[:, intra_start : intra_start + k]
            if has_intra
            else np.zeros((n_draws_total, k), dtype=np.float64)
        )

        zeros = np.zeros_like(bd)
        out: dict[str, np.ndarray] = {}
        out["dest_total"] = bd + bi / n
        out["dest_destination"] = bd * (n - 1) / n
        out["dest_intra"] = (bd + bi) / n
        out["dest_origin"] = zeros.copy()
        out["dest_network"] = zeros.copy()

        out["orig_total"] = bo
        out["orig_origin"] = bo * (n - 1) / n
        out["orig_intra"] = bo / n
        out["orig_destination"] = zeros.copy()
        out["orig_network"] = zeros.copy()

        for eff in _EFFECT_KEYS:
            out[eff] = out[f"dest_{eff}"] + out[f"orig_{eff}"]

        return out


# ---------------------------------------------------------------------------
# NB2 count panel flows, with pooled pair effects and period effects
# ---------------------------------------------------------------------------

_COUNT_FE_DOC = """
    Effects
    -------
    ``effects`` selects ``0`` pooled, ``1`` pair effects, ``2`` period effects
    or ``3`` both.  The within transform of the Gaussian panels does not carry to
    a log link, so the effects are parameters: pair effects (one per
    origin–destination pair) sit outside the spatial filter — ``A⁻¹`` commutes
    with the pair dummies, so this is the inside-filter model reparameterized —
    and are integrated out of every ρ update and drawn jointly with β, held as a
    pair index rather than ``n²`` dummy columns.

    Pair effects are **partially pooled**: ``C_ij ~ N(μ, σ²)`` with σ learned
    (half-t(3, ``group_effect_sd_scale``) prior, default scale 1).  A pair seen
    in few periods is shrunk toward the common mean in proportion to how little
    its data say; with weekly sparse flows a fixed, wide prior instead leaves
    the dispersion α and ρ with an incidental-parameter bias and lets ρ drift
    to a degenerate mode near 1.  Pair effects absorb nothing: the intercept
    and time-invariant columns (intra indicator, log distance) stay in the
    design and anchor the level.  ``mundlak=True`` adds the pair means of the
    time-varying columns inside the filter, absorbing a correlation between the
    pair effects and ``X``.

    Period effects join the coefficient block under a fixed prior
    (``time_effect_mu``, default 0 beside pair effects and ``log mean y``
    without; ``time_effect_sigma``, default 2.5): ``T − 1`` of them (first
    period the baseline) beside pair effects, all ``T`` without (the intercept
    is then absorbed).  ``alpha_fixed`` holds α at a value; a large one (10-20
    times the typical mean count) makes the model an essentially Poisson one
    that keeps the exact Pólya–Gamma sampler (see
    :mod:`neighbayes.models._base._nb`).

    ``fit(store_group_effects=None)`` keeps every draw of the pair effects when
    they fit in about 500 MB and otherwise only their posterior mean and sd
    (``idata["group_effect_summary"]``); ``posterior_predictive`` needs the
    draws.  Effects run on the NumPy Gibbs backend, and on JAX for the
    separable panels (the structured sweep that scales).
"""


def _pop_effects(kwargs: dict) -> int:
    effects = kwargs.pop("effects", 0)
    model = kwargs.pop("model", None)
    return effects if model is None else model


class _FlowCountPanelMixin(CountPanelFEMixin):
    """Shared construction, effects and prediction for count flow panels."""

    def _finish_count_init(self, y_arr: np.ndarray, effects, mundlak=False) -> None:
        self._y_int_vec = np.asarray(y_arr).reshape(-1).astype(np.int64)
        self._init_count_fe(effects, self._N_flow, self._T, mundlak)
        self.effects = self._count_effects

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords.update(self._fe_coords())
        return coords

    def _compute_jacobian_log_det(self, posterior) -> None:
        # The filter acts on the latent log-mean; the count density on the
        # observed counts is already the complete pointwise likelihood.
        return None

    def _eta_reduced(self, rho: dict, beta: np.ndarray) -> np.ndarray:
        """``(I_T ⊗ A(ρ))⁻¹ Xβ`` for one draw, time-first stacked."""
        raise NotImplementedError

    def _rho_draws(self) -> dict:
        post = self._idata.posterior
        return {
            k: post[k].values.reshape(-1)
            for k in ("rho_d", "rho_o", "rho_w")
            if k in post.data_vars
        }

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flow counts, time-first stacked.

        Returns an array of shape ``(n_draws, n² · T)``.
        """
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")
        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
        rho = self._rho_draws()
        alpha = post["alpha"].values.reshape(-1) if "alpha" in post.data_vars else None
        offsets = self._fe_offset_draws(total)
        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            eta = self._eta_reduced(
                {k: float(v[g]) for k, v in rho.items()}, beta_draws[g]
            )
            if offsets is not None:
                eta = eta + offsets[g]
            out[g] = self._count_draw(
                rng, eta, None if alpha is None else float(alpha[g])
            )
        return out


class _UnrestrictedCountFlowPanel(_FlowCountPanelMixin):
    """3-ρ filter pieces of the unrestricted NB flow panel."""

    def _eta_reduced(self, rho: dict, beta: np.ndarray) -> np.ndarray:
        N, T = self._N_flow, self._T
        Xb = (self._X @ beta).reshape(T, N).T
        return self._solve_A(rho["rho_d"], rho["rho_o"], rho["rho_w"], Xb).T.reshape(-1)

    def _unrestricted_filter(self):
        from ...samplers.count_panel._filters import FlowUnrestrictedFilter

        return FlowUnrestrictedFilter(
            self._Wd,
            self._Wo,
            self._Ww,
            self._W_sparse,
            self._T,
            self.priors.get("rho_lower", -0.999),
            self.priors.get("rho_upper", 0.999),
            self.restrict_positive,
        )

    def _build_pymc_count_model(self) -> pm.Model:
        from ..._ops import SparseFlowSolveMatrixOp

        pv = self._flow_count_priors()
        N, T = self._N_flow, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            if self.restrict_positive:
                rho_simplex = pm.Dirichlet("rho_simplex", a=np.ones(4))
                rho_d = pm.Deterministic("rho_d", rho_simplex[0])
                rho_o = pm.Deterministic("rho_o", rho_simplex[1])
                rho_w = pm.Deterministic("rho_w", rho_simplex[2])
            else:
                rho_lower = self.priors.get("rho_lower", -1.0)
                rho_upper = self.priors.get("rho_upper", 1.0)
                rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
                rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
                rho_w = pm.Uniform("rho_w", lower=rho_lower, upper=rho_upper)
                slack = 1.0 - rho_d - rho_o - rho_w
                pm.Potential("stability", pt.switch(slack > 0.0, 0.0, -1e6 * slack**2))
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            Xb_mat = pt.reshape(pt.dot(X_t, beta), (T, N)).T
            solve_op = SparseFlowSolveMatrixOp(self._Wd, self._Wo, self._Ww)
            eta = pt.reshape(solve_op(rho_d, rho_o, rho_w, Xb_mat).T, (N * T,))
            eta = eta + self._pymc_fe_offset()
            lam = pm.Deterministic("lambda", pt.exp(eta))
            alpha = nb_alpha_rv(pv["alpha_fixed"], pv["alpha_nu"], pv["alpha_sigma"])
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)
            # No |A| Jacobian: y is not transformed, the filter acts on the mean.
        return model


class _SeparableCountFlowPanel(_FlowCountPanelMixin):
    """Kronecker-filter pieces of the separable NB flow panel."""

    def _eta_reduced(self, rho: dict, beta: np.ndarray) -> np.ndarray:
        n, N, T = self._n, self._N_flow, self._T
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rho["rho_d"] * self._W_sparse).tocsr()
        Lo = (I_n - rho["rho_o"] * self._W_sparse).tocsr()
        Xb = (self._X @ beta).reshape(T, N).T
        return kron_solve_matrix(Lo, Ld, Xb, n).T.reshape(-1)

    def _separable_filter(self):
        from ...samplers.count_panel._filters import FlowSeparableFilter

        return FlowSeparableFilter(
            self._W_sparse,
            self._T,
            self.priors.get("rho_lower", -0.999),
            self.priors.get("rho_upper", 0.999),
        )

    def _build_pymc_count_model(self) -> pm.Model:
        from ..._ops import KroneckerFlowSolveMatrixOp

        pv = self._flow_count_priors()
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)
        n, N, T = self._n, self._N_flow, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            Xb_mat = pt.reshape(pt.dot(X_t, beta), (T, N)).T
            solve_op = KroneckerFlowSolveMatrixOp(self._W_sparse, n)
            eta = pt.reshape(solve_op(rho_d, rho_o, Xb_mat).T, (N * T,))
            eta = eta + self._pymc_fe_offset()
            lam = pm.Deterministic("lambda", pt.exp(eta))
            alpha = nb_alpha_rv(pv["alpha_fixed"], pv["alpha_nu"], pv["alpha_sigma"])
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)
            # No |A| Jacobian for the count likelihood: the filter enters only
            # through the mean.  (Copying the Gaussian Jacobian biases ρ toward
            # the negative-logdet region.)
        return model


class _AspatialCountFlowPanel(_FlowCountPanelMixin):
    """``A = I`` pieces of the NB gravity panel."""

    _count_spatial = False

    def _eta_reduced(self, rho: dict, beta: np.ndarray) -> np.ndarray:
        return self._X @ beta

    def _build_pymc_count_model(self) -> pm.Model:
        pv = self._flow_count_priors()
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal(
                "beta", mu=pv["beta_mu"], sigma=pv["beta_sigma"], dims="coefficient"
            )
            eta = pt.dot(X_t, beta) + self._pymc_fe_offset()
            lam = pm.Deterministic("lambda", pt.exp(eta))
            alpha = nb_alpha_rv(pv["alpha_fixed"], pv["alpha_nu"], pv["alpha_sigma"])
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)
        return model

    def _fit_count_aspatial(
        self,
        *,
        sampler,
        draws,
        tune,
        chains,
        random_seed,
        progressbar,
        n_jobs,
        idata_kwargs,
        store_group_effects,
        model_type,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Gibbs with ρ fixed at 0 (pair effects integrated out), or NUTS."""
        if sampler == "nuts":
            return FlowPanelModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        if sampler != "gibbs":
            raise ValueError(f"sampler must be 'gibbs' or 'nuts', got {sampler!r}")
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        from ...samplers.count_panel._filters import NoFilter

        pv = self._flow_count_priors()
        return self._run_count_panel_gibbs(
            NoFilter(),
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            store_group_effects=store_group_effects,
            model_type=model_type,
        )


class SARNegBinFlowPanel(_UnrestrictedCountFlowPanel, SARFlowPanel):
    """Panel NB2 SAR flow model with unrestricted dependence parameters."""

    __doc__ = __doc__ + _COUNT_FE_DOC

    def __init__(self, y, X, W, **kwargs):
        # Count model: no |A| change-of-variables Jacobian, so it keeps the PyMC
        # path (not the Gaussian resolvent sampler); "none" routes fit accordingly.
        kwargs.setdefault("logdet_method", "none")
        effects = _pop_effects(kwargs)
        mundlak = pop_mundlak(kwargs)
        y_arr = require_counts(y, "SARNegBinFlowPanel")
        super().__init__(y_arr.astype(np.float64), X, W, effects=0, **kwargs)
        self._finish_count_init(y_arr, effects, mundlak)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        gibbs_backend: str = "auto",
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_group_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the NB2 SAR flow panel posterior.

        ``sampler="gibbs"`` (default) runs the reduced-form Pólya–Gamma Gibbs
        sampler with per-period Kronecker solves — the recommended path for NB
        models.  ``sampler="nuts"`` uses the PyMC count path (exact likelihood,
        much slower).  The count likelihood carries no ``|A|`` change-of-variables
        term on either path.  With ``attach_log_abs_det`` (default) the per-draw
        spatial-filter Jacobian ``T·log|A(ρ)|`` is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics (never folded into
        ``log_likelihood``); set it ``False`` to skip the per-draw resolvent cost
        at very large ``N``.

        ``gibbs_backend="auto"`` (default) takes JAX when it is installed and the
        configuration has a JAX kernel, else NumPy.  The unrestricted JAX kernel
        covers the cross-section (``T = 1``) without effects; panels and pair or
        period effects run on NumPy.  With pair or period effects
        (``effects != 0``) the sweep also integrates out the pair effects; see
        the class docstring.

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` on either sampler; off by default, as in PyMC.
        """
        log_lik = bool((idata_kwargs or {}).get("log_likelihood", False))
        if sampler == "gibbs":
            gibbs_backend = self._resolve_gibbs_backend(
                gibbs_backend, jax=not self._count_effects and self._T == 1
            )
            if self._count_effects:
                pv = self._flow_count_priors()
                idata = self._run_count_panel_gibbs(
                    self._unrestricted_filter(),
                    beta_mu=pv["beta_mu"],
                    beta_sigma=pv["beta_sigma"],
                    alpha_sigma=pv["alpha_sigma"],
                    alpha_nu=pv["alpha_nu"],
                    alpha_fixed=pv["alpha_fixed"],
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    log_likelihood=log_lik,
                    store_group_effects=store_group_effects,
                    model_type="nb_sar_flow_panel_fe",
                )
            else:
                idata = self._fit_gibbs(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    gibbs_backend=gibbs_backend,
                    log_likelihood=log_lik,
                )
        elif sampler == "nuts":
            idata = FlowPanelModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        else:
            raise ValueError(f"sampler must be 'gibbs' or 'nuts', got {sampler!r}")
        if attach_log_abs_det:
            self._attach_flow_log_abs_det(idata)
        return idata

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample the pooled posterior via reduced-form PG-Gibbs (unrestricted 3-ρ)."""
        from ..flow._nb_gibbs import run_negbin_flow_gibbs

        return run_negbin_flow_gibbs(
            self,
            separable=False,
            model_type="nb_sar_flow_panel",
            omega_size=self._N_flow * self._T,
            T=self._T,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            gibbs_backend=gibbs_backend,
            log_likelihood=log_likelihood,
        )

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        rho_w_draws = idata.posterior["rho_w"].values.reshape(-1)
        beta_draws = self._beta_layout(idata.posterior)
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws,
            draws=draws,
        )

    def _build_pymc_model(self) -> pm.Model:
        return self._build_pymc_count_model()


class SARNegBinFlowSeparablePanel(_SeparableCountFlowPanel, SARFlowSeparablePanel):
    """Panel separable NB2 SAR flow model."""

    __doc__ = __doc__ + _COUNT_FE_DOC

    def __init__(self, y, X, W, **kwargs):
        effects = _pop_effects(kwargs)
        mundlak = pop_mundlak(kwargs)
        y_arr = require_counts(y, "SARNegBinFlowSeparablePanel")
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SARNegBinFlowSeparablePanel logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y_arr.astype(np.float64), X, W, effects=0, **kwargs)
        self._finish_count_init(y_arr, effects, mundlak)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        gibbs_backend: str = "auto",
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_group_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the separable NB2 SAR flow panel posterior.

        ``sampler="gibbs"`` (default) runs the reduced-form Pólya–Gamma Gibbs
        sampler with per-period Kronecker solves; ``sampler="nuts"`` uses the
        PyMC count path (exact likelihood, much slower).  With
        ``attach_log_abs_det`` (default) the per-draw Jacobian ``T·log|A(ρ)|``
        (using the separability relation ``ρ_w = −ρ_d ρ_o``) is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics — not folded into the
        count model's ``log_likelihood``.

        ``gibbs_backend="jax"`` runs the same structured sweep compiled with JAX,
        chains on threads; ``"numpy"`` runs it on the host; ``"auto"`` (default)
        takes JAX when it is installed, else NumPy.  With
        effects (``effects != 0``) the structured sweep also integrates
        out the pair and period effects, on either backend (see the class
        docstring).

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` on either sampler; off by default, as in PyMC.
        """
        log_lik = bool((idata_kwargs or {}).get("log_likelihood", False))
        if sampler == "gibbs":
            gibbs_backend = self._resolve_gibbs_backend(gibbs_backend, jax=True)
            if self._count_effects:
                idata = self._fit_gibbs_fe(
                    gibbs_backend=gibbs_backend,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    log_likelihood=log_lik,
                    store_group_effects=store_group_effects,
                )
            else:
                idata = self._fit_gibbs(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    random_seed=random_seed,
                    progressbar=progressbar,
                    n_jobs=n_jobs,
                    gibbs_backend=gibbs_backend,
                    log_likelihood=log_lik,
                )
        elif sampler == "nuts":
            idata = FlowPanelModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        else:
            raise ValueError(f"sampler must be 'gibbs' or 'nuts', got {sampler!r}")
        if attach_log_abs_det:
            self._attach_flow_log_abs_det(idata)
        return idata

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        n_jobs: int = -1,
        gibbs_backend: str = "numpy",
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample the pooled posterior via reduced-form PG-Gibbs (separable 2-ρ)."""
        from ..flow._nb_gibbs import run_negbin_flow_gibbs

        return run_negbin_flow_gibbs(
            self,
            separable=True,
            model_type="nb_sar_flow_sep_panel",
            omega_size=self._N_flow * self._T,
            T=self._T,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            gibbs_backend=gibbs_backend,
            log_likelihood=log_likelihood,
        )

    def _fit_gibbs_fe(
        self,
        *,
        gibbs_backend: str = "numpy",
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_group_effects: Optional[bool],
    ) -> xr.DataTree:
        """Structured sweep with pair and period effects integrated out."""
        from types import SimpleNamespace

        from ...samplers._utils._seeds import spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.negbin_reduced._flow_structured import classify_flow_design
        from ...samplers.negbin_reduced._flow_structured_fe import (
            run_chain_separable_structured_fe,
        )

        pv = self._flow_count_priors()
        fp = self._count_fe_priors()
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)
        priors = SimpleNamespace(
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )
        keep_groups = self._store_group_draws(store_group_effects, chains, draws)
        y = self._y_int_vec.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        struct = classify_flow_design(X, self._n, self._T)
        W_csc = self._W_sparse.tocsc()
        k = X.shape[1]
        pair = self._pair_args(fp) if self._fe_groups else None
        periods = (
            (fp["time_effect_mu"], fp["time_effect_sigma"], self._n_tau)
            if self._fe_periods
            else None
        )

        def _init(rng):
            return SimpleNamespace(
                beta=rng.normal(0.0, 0.1, size=k),
                rho_d=rng.uniform(-0.1, 0.1),
                rho_o=rng.uniform(-0.1, 0.1),
                alpha=1.0 if pv["alpha_fixed"] is None else pv["alpha_fixed"],
            )

        if gibbs_backend == "jax":
            from ...samplers._utils._seeds import seed_sequence_to_int
            from ...samplers.negbin_reduced._flow_structured_fe_jax import (
                run_chains_jax_flow_structured_fe,
            )

            int_seeds = [
                seed_sequence_to_int(s) for s in spawn_chain_seeds(random_seed, chains)
            ]
            results = run_chains_jax_flow_structured_fe(
                y,
                W_csc,
                self._n,
                priors,
                [_init(np.random.default_rng(s)) for s in int_seeds],
                draws,
                tune,
                struct=struct,
                pair_effects=pair,
                period_effects=periods,
                keep_group_draws=keep_groups,
                jax_seeds=int_seeds,
                store_log_lik=log_likelihood,
            )
            return self._assemble_count_idata(
                results, ["rho_d", "rho_o", "rho_w"], keep_groups, log_likelihood
            )

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            rng = np.random.default_rng(seed)
            init = _init(rng)
            return run_chain_separable_structured_fe(
                y,
                X,
                W_csc,
                self._n,
                priors,
                init,
                draws,
                tune,
                T=self._T,
                pair_effects=pair,
                period_effects=periods,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
                rng=rng,
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                store_group_draws=keep_groups,
                struct=struct,
            )

        seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type="nb_sar_flow_sep_panel_fe",
            timeout=None,
        )
        return self._assemble_count_idata(
            results, ["rho_d", "rho_o", "rho_w"], keep_groups, log_likelihood
        )

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = self._beta_layout(idata.posterior)
        return self._compute_flow_effects_kron(
            rho_d_draws,
            rho_o_draws,
            beta_draws,
            draws=draws,
        )

    def _build_pymc_model(self) -> pm.Model:
        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARNegBinFlowSeparablePanel requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )
        return self._build_pymc_count_model()


class NegBinFlowPanel(_AspatialCountFlowPanel, OLSFlowPanel):
    """Aspatial panel OD-flow NB2 gravity baseline.

    ``fit()`` runs Pólya–Gamma Gibbs with ρ fixed at zero (pair effects
    integrated out); ``sampler="nuts"`` runs PyMC.
    """

    __doc__ = __doc__ + _COUNT_FE_DOC

    def __init__(self, y, X, W, T, **kwargs):
        effects = _pop_effects(kwargs)
        mundlak = pop_mundlak(kwargs)
        y_arr = require_counts(y, "NegBinFlowPanel")
        super().__init__(y_arr.astype(np.float64), X, W, T, effects=0, **kwargs)
        self._finish_count_init(y_arr, effects, mundlak)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_group_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior: Gibbs (default, ρ fixed at 0) or ``"nuts"``."""
        return self._fit_count_aspatial(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            idata_kwargs=idata_kwargs,
            store_group_effects=store_group_effects,
            model_type="nb_flow_panel",
            **sample_kwargs,
        )

    def _build_pymc_model(self) -> pm.Model:
        return self._build_pymc_count_model()


# ---------------------------------------------------------------------------
# Zero-inflated NB flows (separable), cross-section and panel
# ---------------------------------------------------------------------------


class SARZINBFlowSeparablePanel(
    ZINBMixin, _SeparableCountFlowPanel, SARFlowSeparablePanel
):
    r"""Zero-inflated separable SAR flow panel.

    .. math::

        \eta^{\mathrm{sel}}_t &= (L_o^{\lambda} \otimes L_d^{\lambda})^{-1} Z_t\gamma,
        \qquad z \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta^{\mathrm{sel}})), \\
        \eta^{\mathrm{cnt}}_t &= (L_o^{\rho} \otimes L_d^{\rho})^{-1} X_t\beta
        + \tau_t + C, \qquad y \mid z = 1 \sim \mathrm{NB2}(e^{\eta^{\mathrm{cnt}}}, \alpha),

    with :math:`L_k^{\lambda} = I - \lambda_k W_{\mathrm{sel}}`,
    :math:`L_k^{\rho} = I - \rho_k W` and per-cell structural zeros
    (``y = 0`` when ``z = 0``).  ``effects`` sets pair and period effects on the
    **count** equation only.

    Sampling: ``fit()`` runs the structured ``n × n`` Pólya–Gamma sweep
    (:mod:`neighbayes.samplers.zinb._flow_structured`), on NumPy or
    ``gibbs_backend="jax"``; ``sampler="nuts"`` fits the same model in PyMC.

    Parameters
    ----------
    y, X, W, T, col_names, k, priors, symmetric_xo_xd
        As :class:`SARNegBinFlowSeparablePanel`.
    Z : array-like or pandas.DataFrame, optional
        Selection design, time-first stacked ``(n²·T, p)``.  Default: ``X``
        (then selection impacts are available).
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        ``n × n`` selection weights.  Default: ``W``.
    effects : int, default 0
        Count-equation effects: ``0`` pooled, ``1`` pair, ``2`` period,
        ``3`` both.
    priors : dict, optional
        Count keys as :class:`SARNegBinFlowSeparablePanel` (incl.
        ``alpha_fixed``) plus ``gamma_mu``, ``gamma_sigma``, ``lam_lower`` and
        ``lam_upper`` for the selection equation.
    """

    def __init__(self, y, X, W, Z=None, W_sel=None, **kwargs):
        effects = _pop_effects(kwargs)
        mundlak = pop_mundlak(kwargs)
        y_arr = require_counts(y, type(self).__name__)
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"logdet_method must be None (auto) or one of {sorted(_VALID)}; "
                f"got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y_arr.astype(np.float64), X, W, effects=0, **kwargs)
        # Selection design, before the count effects drop absorbed X columns.
        self._sel_is_X = Z is None
        if Z is None:
            self._Z = np.asarray(self._X, dtype=np.float64).copy()
            self._sel_feature_names = list(self._feature_names)
        else:
            if isinstance(Z, pd.DataFrame):
                self._sel_feature_names = [str(c) for c in Z.columns]
            Z_arr = np.asarray(Z, dtype=np.float64)
            Z_arr = Z_arr[:, None] if Z_arr.ndim == 1 else Z_arr
            if Z_arr.shape[0] != self._N_flow * self._T:
                raise ValueError(
                    f"Z must have n²·T = {self._N_flow * self._T} rows, "
                    f"got {Z_arr.shape[0]}."
                )
            if not isinstance(Z, pd.DataFrame):
                self._sel_feature_names = [f"z{j}" for j in range(Z_arr.shape[1])]
            self._Z = Z_arr
        if W_sel is None:
            self._W_sel_sparse, self._same_W = self._W_sparse, True
        else:
            self._W_sel_sparse, self._same_W = _weights_to_csr(W_sel), False
            if self._W_sel_sparse.shape != self._W_sparse.shape:
                raise ValueError("W_sel must be n × n, like W.")
        self._is_sel_row_std = True
        self._effects_equation = "count"
        self._finish_count_init(y_arr, effects, mundlak)

    def _model_coords(self, extra: Optional[dict] = None) -> dict:
        coords = super()._model_coords(extra)
        coords["sel_coefficient"] = list(self._sel_feature_names)
        return coords

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------

    def _zinb_flow_priors(self):
        from types import SimpleNamespace

        from .._base._shared import gelman_default_beta_prior

        pv = self._flow_count_priors()
        p = self._Z.shape[1]
        g_mu, g_sd = gelman_default_beta_prior(
            np.full(self._Z.shape[0], 0.5),
            self._Z,
            list(self._sel_feature_names),
            link="logit",
        )
        gamma_mu = np.broadcast_to(
            np.asarray(self.priors.get("gamma_mu", g_mu), dtype=float), (p,)
        ).copy()
        gamma_sigma = np.broadcast_to(
            np.asarray(self.priors.get("gamma_sigma", g_sd), dtype=float), (p,)
        ).copy()
        return SimpleNamespace(
            beta_mu=pv["beta_mu"],
            beta_sigma=pv["beta_sigma"],
            gamma_mu=gamma_mu,
            gamma_sigma=gamma_sigma,
            alpha_sigma=pv["alpha_sigma"],
            alpha_nu=pv["alpha_nu"],
            alpha_fixed=pv["alpha_fixed"],
            rho_bounds=(
                float(self.priors.get("rho_lower", -0.999)),
                float(self.priors.get("rho_upper", 0.999)),
            ),
            lam_bounds=(
                float(self.priors.get("lam_lower", -0.999)),
                float(self.priors.get("lam_upper", 0.999)),
            ),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        gibbs_backend: str = "auto",
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        store_group_effects: Optional[bool] = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Sample the posterior: structured Gibbs (default) or ``sampler="nuts"``.

        ``gibbs_backend="auto"`` (default) takes JAX when it is installed, else
        NumPy; ``"numpy"`` or ``"jax"`` pins one.
        ``store_group_effects`` keeps every draw of the pair effects (default:
        when they fit in about 500 MB).  ``idata_kwargs={"log_likelihood":
        True}`` stores the marginal ZINB pointwise log-likelihood.
        """
        if sampler == "nuts":
            return FlowPanelModel.fit(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                idata_kwargs=idata_kwargs,
                **sample_kwargs,
            )
        if sampler != "gibbs":
            raise ValueError(f"sampler must be 'gibbs' or 'nuts', got {sampler!r}")
        if sample_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(sample_kwargs)}")
        return self._fit_gibbs_zinb(
            backend=self._resolve_gibbs_backend(gibbs_backend, jax=True),
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            n_jobs=n_jobs,
            log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            store_group_effects=store_group_effects,
        )

    def _zinb_fe_args(self):
        fp = self._count_fe_priors()
        pair = self._pair_args(fp) if self._fe_groups else None
        periods = (
            (fp["time_effect_mu"], fp["time_effect_sigma"], self._n_tau)
            if self._fe_periods
            else None
        )
        return pair, periods

    def _fit_gibbs_zinb(
        self,
        *,
        backend: str,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        n_jobs: int,
        log_likelihood: bool,
        store_group_effects: Optional[bool],
    ) -> xr.DataTree:
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds
        from ...samplers.gaussian._chain_runner import run_chains
        from ...samplers.zinb._flow_structured import run_chain_zinb_flow_structured

        pr = self._zinb_flow_priors()
        pair, periods = self._zinb_fe_args()
        keep_groups = self._store_group_draws(store_group_effects, chains, draws)
        y = self._y_int_vec.astype(np.float64)
        X = np.asarray(self._X, dtype=np.float64)
        Z = np.asarray(self._Z, dtype=np.float64)
        W_csc, W_sel_csc = self._W_sparse.tocsc(), self._W_sel_sparse.tocsc()
        scalars = ["rho_d", "rho_o", "rho_w", "lam_d", "lam_o", "lam_w"]
        extra = {"gamma": ("sel_coefficient", self._sel_feature_names)}

        if backend == "jax":
            from ...samplers.zinb._flow_structured_jax import (
                run_chains_jax_zinb_flow_structured,
            )

            int_seeds = [
                seed_sequence_to_int(s) for s in spawn_chain_seeds(random_seed, chains)
            ]
            results = run_chains_jax_zinb_flow_structured(
                y,
                X,
                Z,
                W_csc,
                W_sel_csc,
                self._n,
                self._T,
                pr,
                draws,
                tune,
                pair_effects=pair,
                period_effects=periods,
                keep_group_draws=keep_groups,
                jax_seeds=int_seeds,
                store_log_lik=log_likelihood,
            )
            return self._assemble_count_idata(
                results, scalars, keep_groups, log_likelihood, extra=extra
            )

        def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
            return run_chain_zinb_flow_structured(
                y,
                X,
                Z,
                W_csc,
                W_sel_csc,
                self._n,
                self._T,
                pr,
                draws,
                tune,
                pair_effects=pair,
                period_effects=periods,
                rho_bounds=pr.rho_bounds,
                lam_bounds=pr.lam_bounds,
                rng=np.random.default_rng(seed),
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                store_group_draws=keep_groups,
            )

        seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type=type(self).__name__,
            timeout=None,
        )
        return self._assemble_count_idata(
            results, scalars, keep_groups, log_likelihood, extra=extra
        )

    def _build_pymc_model(self) -> pm.Model:
        """The same ZINB and priors as the Gibbs path, in PyMC."""
        from ..._ops import KroneckerFlowSolveMatrixOp

        pr = self._zinb_flow_priors()
        n, N, T = self._n, self._N_flow, self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        Z_t = pt.as_tensor_variable(self._Z.astype(np.float64))
        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", *pr.rho_bounds)
            rho_o = pm.Uniform("rho_o", *pr.rho_bounds)
            pm.Deterministic("rho_w", -rho_d * rho_o)
            lam_d = pm.Uniform("lam_d", *pr.lam_bounds)
            lam_o = pm.Uniform("lam_o", *pr.lam_bounds)
            pm.Deterministic("lam_w", -lam_d * lam_o)
            beta = pm.Normal(
                "beta", mu=pr.beta_mu, sigma=pr.beta_sigma, dims="coefficient"
            )
            gamma = pm.Normal(
                "gamma", mu=pr.gamma_mu, sigma=pr.gamma_sigma, dims="sel_coefficient"
            )
            Xb = pt.reshape(pt.dot(X_t, beta), (T, N)).T
            Zg = pt.reshape(pt.dot(Z_t, gamma), (T, N)).T
            eta_cnt = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_sparse, n)(rho_d, rho_o, Xb).T,
                (N * T,),
            )
            eta_sel = pt.reshape(
                KroneckerFlowSolveMatrixOp(self._W_sel_sparse, n)(lam_d, lam_o, Zg).T,
                (N * T,),
            )
            eta_cnt = eta_cnt + self._pymc_fe_offset()
            alpha = nb_alpha_rv(pr.alpha_fixed, pr.alpha_nu, pr.alpha_sigma)
            pm.ZeroInflatedNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_sel),
                mu=pt.exp(eta_cnt),
                alpha=alpha,
                observed=self._y_int_vec,
            )
        return model

    # ------------------------------------------------------------------
    # Per-draw predictors and impacts
    # ------------------------------------------------------------------

    def _kron_eta(self, W, rd: float, ro: float, b: np.ndarray) -> np.ndarray:
        n, N, T = self._n, self._N_flow, self._T
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rd * W).tocsr()
        Lo = (I_n - ro * W).tocsr()
        return kron_solve_matrix(Lo, Ld, b.reshape(T, N).T, n).T.reshape(-1)

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at draw ``g`` (pair-effect posterior mean if summarized)."""
        f = self._flat_draw
        eta_sel = self._kron_eta(
            self._W_sel_sparse,
            float(f("lam_d")[g]),
            float(f("lam_o")[g]),
            self._Z @ f("gamma")[g],
        )
        eta_cnt = self._kron_eta(
            self._W_sparse,
            float(f("rho_d")[g]),
            float(f("rho_o")[g]),
            self._X @ f("beta")[g],
        )
        post = self._idata.posterior
        if self._fe_groups:
            if "group_effect" in post.data_vars:
                c = f("group_effect")[g]
            else:
                c = self._idata["group_effect_summary"]["mean"].values
            eta_cnt = eta_cnt + np.asarray(c)[self._row_group]
        if self._fe_periods:
            eta_cnt = eta_cnt + self._D_tau @ f("time_effect")[g]
        return eta_sel, eta_cnt

    def _compute_spatial_effects_posterior(self, draws: Optional[int] = None) -> dict:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        post = self._idata.posterior
        if self._effects_equation == "selection":
            coef = post["gamma"].values.reshape(-1, len(self._sel_feature_names))
            return self._compute_flow_effects_kron(
                post["lam_d"].values.reshape(-1),
                post["lam_o"].values.reshape(-1),
                coef,
                draws=draws,
            )
        return self._compute_flow_effects_kron(
            post["rho_d"].values.reshape(-1),
            post["rho_o"].values.reshape(-1),
            self._beta_layout(post),
            draws=draws,
        )

    def spatial_effects(
        self,
        equation: str = "count",
        draws: Optional[int] = None,
        return_posterior_samples: bool = False,
        ci: float = 0.95,
        mode: str = "auto",
    ):
        """Origin, destination, intra and network effects for one equation.

        ``equation="count"`` (default) on the count log-mean through
        ``ρ``; ``"selection"`` on the activation log-odds through ``λ`` and γ,
        available when the selection design is the flow design (``Z=None``).
        Other arguments as :meth:`FlowPanelModel.spatial_effects`.
        """
        if equation not in ("count", "selection"):
            raise ValueError(
                f"equation must be 'count' or 'selection', got {equation!r}"
            )
        if equation == "selection" and not self._sel_is_X:
            raise NotImplementedError(
                "Selection flow effects need the flow design layout; they are "
                "available when the model is built with Z=None (Z = X)."
            )
        self._effects_equation = equation
        try:
            return FlowPanelModel.spatial_effects(
                self,
                draws=draws,
                return_posterior_samples=return_posterior_samples,
                ci=ci,
                mode=mode,
            )
        finally:
            self._effects_equation = "count"


class SARZINBFlowSeparable(SARZINBFlowSeparablePanel):
    """Cross-sectional zero-inflated separable SAR flow model (one period).

    :class:`SARZINBFlowSeparablePanel` with ``T = 1`` and no effects; see
    there for the model, sampling and arguments.  ``y`` is the ``n²`` flow
    vector (or the ``n × n`` matrix).

    On sparse flows (active pairs averaging about one count) the selection
    equation is weakly identified in a cross-section: only the shape of the
    count distribution separates structural from sampling zeros, and NB with a
    smaller α explains the zeros nearly as well.  Check λ against its prior and
    the selection R-hat, or use the panel with pair effects, or
    :class:`SARHurdleNBFlowSeparable`.
    """

    def __init__(self, y, X, W, **kwargs):
        if _pop_effects(kwargs):
            raise ValueError(
                "SARZINBFlowSeparable is a cross-section; pair and period effects need "
                "SARZINBFlowSeparablePanel."
            )
        kwargs.pop("T", None)
        super().__init__(y, X, W, T=1, **kwargs)


# ---------------------------------------------------------------------------
# Panel SEM-Flow variants (spatial-error analogues of SARFlowPanel)
# ---------------------------------------------------------------------------


class _SEMFlowPanelMixin:
    """Shared init helper to precompute design-matrix lags for SEM panel models."""

    def _init_sem_lags(self) -> None:
        T = self._T
        # Lags of the (already-demeaned) design matrix.  Constants — no
        # parameter dependence, so we precompute once.
        self._Wd_X, self._Wo_X, self._Ww_X = flow_lags(
            self._W_sparse, self._X.astype(np.float64), T=T
        )


class SEMFlowPanel(_ResolventFlowPanelMixin, _SEMFlowPanelMixin, FlowPanelModel):
    """Panel spatial-error flow model with three free spatial parameters.

    Panel analogue of :class:`~neighbayes.models.flow.SEMFlow`: applies the
    Kronecker spatial filter (:math:`W_d`, :math:`W_o`, :math:`W_w`) to the
    disturbance rather than the dependent variable, period by period:

    .. math::

        y_t = X_t \\beta + u_t, \\qquad B u_t = \\varepsilon_t,
        \\quad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2 I_N).

    The Jacobian contribution scales as :math:`T \\cdot \\log|B|` — identical
    in form to :class:`SARFlowPanel`. Marginal mean is :math:`X_t \\beta`,
    so there are no :math:`X`-mediated spillovers; effects collapse to the
    closed-form expressions used by :class:`OLSFlowPanel`.

    Parameters
    ----------
    y : array-like
        Stacked panel response in shape ``(T, n, n)``, ``(T, n^2)``, or
        ``(n^2 * T,)``.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on ``n`` units.
    X : np.ndarray or pandas.DataFrame, shape ``(n^2 * T, p)``
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods (must be a positive integer).
    col_names : list of str, optional
        Feature names for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects;
        inferred from columns prefixed ``dest_`` if omitted.
    model : int, default 0
        Fixed-effects transform: ``0`` pooled, ``1`` pair FE, ``2`` time
        FE, ``3`` two-way FE.
    logdet_method : str, default "resolvent"
        Log-determinant method.  The default ``"resolvent"`` samples via the
        per-period resolvent-gradient sampler (recommended).
    restrict_positive : bool, default True
        If True, use ``pm.Dirichlet("lam_simplex", a=ones(4))`` to enforce
        :math:`\\lambda_d, \\lambda_o, \\lambda_w \\geq 0` and
        :math:`\\lambda_d + \\lambda_o + \\lambda_w \\leq 1`. If False,
        three independent ``pm.Uniform(lam_lower, lam_upper)`` priors are
        used with a differentiable quadratic-wall stability potential.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers.  The degrees of freedom :math:`\\nu` are
        **fixed** at ``priors["nu"]`` (default 4, LeSage's ``rval``).
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float or array, default Gelman et al. (2008) — Normal prior mean for ``beta`` (``mean(y)`` on the intercept, 0 otherwise).
        - ``beta_sigma`` : float or array, default Gelman et al. (2008) — Normal prior std for ``beta``, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` : float, default 2 and ``Var(y)`` — InverseGamma prior on ``sigma**2``.
        - ``lam_lower`` : float, default -1.0 — Lower bound of Uniform prior on each λ (only when ``restrict_positive=False``).
        - ``lam_upper`` : float, default 1.0 — Upper bound of Uniform prior on each λ (only when ``restrict_positive=False``).
        - ``nu`` : float, default 4.0 — Fixed Student-t degrees of freedom (only when ``robust=True``).
    """

    def __init__(self, y, X, W, T, **kwargs):
        # Default to the resolvent-gradient SEM panel sampler (parallel to the
        # cross-sectional SEMFlow); the separable subclass sets its own method.
        kwargs.setdefault("logdet_method", "resolvent")
        super().__init__(y, X, W, T, **kwargs)
        self._init_sem_lags()

    def _sample_resolvent(self, **kwargs) -> xr.DataTree:
        from ...samplers.gaussian._flow_resolvent import sample_sem_flow_resolvent

        return sample_sem_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            T=self._T,
            restrict_positive=self.restrict_positive,
            fe_dims=self._flow_fe_dims,
            priors=self._flow_gaussian_priors(),
            **kwargs,
        )

    def _build_pymc_model(self) -> pm.Model:
        # The unrestricted SEM flow panel samples via the resolvent-gradient
        # sampler (``fit`` → ``sample_sem_flow_resolvent``); the legacy "traces"
        # Jacobian was removed.  Only reached if a non-resolvent logdet_method
        # is forced.
        raise NotImplementedError(
            "SEMFlowPanel samples via the resolvent-gradient sampler "
            "(logdet_method='resolvent'); the legacy 'traces' PyMC path was removed."
        )

    def _simulate_y_rep_period(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM panel posterior-predictive: ``y_rep,t = X_t β + B^{-1} ε_t``."""
        N = self._N_flow
        T = self._T
        Xb = self._X @ beta  # (N*T,)
        if sigma is None:
            return Xb
        B = self._assemble_A(lam_d, lam_o, lam_w).tocsc()
        lu = sp.linalg.splu(B)
        eps = rng.normal(scale=float(sigma), size=(N, T))
        u = lu.solve(eps)  # (N, T)
        u_stacked = u.T.reshape(-1)  # back to time-first
        return Xb + u_stacked

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        """Closed-form effects (delegates to OLSFlowPanel logic)."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return _compute_ols_flow_effects(
            self._idata,
            beta_draws=self._beta_layout(self._idata.posterior),
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._design_feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )


class SEMFlowSeparablePanel(_SEMFlowPanelMixin, FlowPanelModel):
    """Panel separable spatial-error flow model with :math:`\\lambda_w = -\\lambda_d \\lambda_o`.

    Panel analogue of :class:`~neighbayes.models.flow.SEMFlowSeparable` and
    spatial-error counterpart of :class:`SARFlowSeparablePanel`. Uses the
    eigenvalue / Chebyshev factorization of :math:`\\log|B|` with the panel
    Jacobian scaling :math:`T \\cdot \\log|B|`.

    Parameters
    ----------
    y : array-like
        Stacked panel response in shape ``(T, n, n)``, ``(T, n^2)``, or
        ``(n^2 * T,)``.
    W : libpysal.graph.Graph or scipy.sparse / dense (n×n) matrix
        Row-standardized graph on ``n`` units.
    X : np.ndarray or pandas.DataFrame, shape ``(n^2 * T, p)``
        Stacked panel design matrix in time-first order.
    T : int
        Number of panel periods (must be a positive integer).
    col_names : list of str, optional
        Feature names for ``X``. Inferred from a DataFrame if omitted.
    k : int, optional
        Number of destination/origin covariate pairs used by flow effects;
        inferred from columns prefixed ``dest_`` if omitted.
    model : int, default 0
        Fixed-effects transform: ``0`` pooled, ``1`` pair FE, ``2`` time
        FE, ``3`` two-way FE.
    logdet_method : {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"} or None, default None
        ``None`` auto-selects (``aaa`` for directed W, ``cheb_cholesky`` for
        symmetric, ``eigenvalue`` for small n).
        Method for the Kronecker-factored log-determinant.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers.  The degrees of freedom :math:`\\nu` are
        **fixed** at ``priors["nu"]`` (default 4, LeSage's ``rval``).
    symmetric_xo_xd : bool, optional
        If ``None`` (default), origin and destination design blocks are
        compared and symmetry is auto-detected.
    priors : dict, optional
        Override default priors. Supported keys:

        - ``beta_mu`` : float or array, default Gelman et al. (2008) — Normal prior mean for ``beta`` (``mean(y)`` on the intercept, 0 otherwise).
        - ``beta_sigma`` : float or array, default Gelman et al. (2008) — Normal prior std for ``beta``, scaled to ``sd(y)`` and each column's sd.
        - ``sigma2_alpha``, ``sigma2_beta`` : float, default 2 and ``Var(y)`` — InverseGamma prior on ``sigma**2``.
        - ``lam_lower`` : float, default -0.999 — Lower bound of Uniform prior on ``lam_d`` and ``lam_o``.
        - ``lam_upper`` : float, default 0.999 — Upper bound of Uniform prior on ``lam_d`` and ``lam_o``.
        - ``nu`` : float, default 4.0 — Fixed Student-t degrees of freedom (only when ``robust=True``).

    Notes
    -----
    The ``restrict_positive`` argument inherited from :class:`FlowPanelModel`
    has no effect on this class — separable variants always use Uniform
    priors on the individual :math:`\\lambda` components.
    """

    def __init__(self, y, X, W, T, **kwargs):
        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SEMFlowSeparablePanel logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y, X, W, T, **kwargs)
        self._init_sem_lags()

    def _build_pymc_model(self) -> pm.Model:
        pv = self._flow_gaussian_priors()
        beta_mu, beta_sigma = pv["beta_mu"], pv["beta_sigma"]
        lam_lower = self.priors.get("lam_lower", -0.999)
        lam_upper = self.priors.get("lam_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SEMFlowSeparablePanel requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )

        Wd_y_t = pt.as_tensor_variable(self._Wd_y.astype(np.float64))
        Wo_y_t = pt.as_tensor_variable(self._Wo_y.astype(np.float64))
        Ww_y_t = pt.as_tensor_variable(self._Ww_y.astype(np.float64))
        Wd_X_t = pt.as_tensor_variable(self._Wd_X.astype(np.float64))
        Wo_X_t = pt.as_tensor_variable(self._Wo_X.astype(np.float64))
        Ww_X_t = pt.as_tensor_variable(self._Ww_X.astype(np.float64))
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            lam_d = pm.Uniform("lam_d", lower=lam_lower, upper=lam_upper)
            lam_o = pm.Uniform("lam_o", lower=lam_lower, upper=lam_upper)
            lam_w = pm.Deterministic("lam_w", -lam_d * lam_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = self._flow_sigma(pv)
            self._fe_dof_sigma(sigma)

            mu = (
                lam_d * Wd_y_t
                + lam_o * Wo_y_t
                + lam_w * Ww_y_t
                + pt.dot(X_t, beta)
                - lam_d * pt.dot(Wd_X_t, beta)
                - lam_o * pt.dot(Wo_X_t, beta)
                - lam_w * pt.dot(Ww_X_t, beta)
            )
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._flow_jacobian(
                    self._separable_logdet_fn(lam_d, lam_o),
                    (1 - lam_d) * (1 - lam_o),
                    pt,
                ),
            )

        return model

    def _compute_jacobian_log_det(self, posterior) -> np.ndarray:
        lam_d = np.asarray(posterior["lam_d"].values.reshape(-1), dtype=np.float64)
        lam_o = np.asarray(posterior["lam_o"].values.reshape(-1), dtype=np.float64)
        if self._separable_logdet_numpy_fn is None:
            raise RuntimeError(
                "Missing separable numeric logdet evaluator. "
                "Initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )
        return self._flow_jacobian(
            self._separable_logdet_numpy_fn(lam_d, lam_o), (1 - lam_d) * (1 - lam_o), np
        )

    def _simulate_y_rep_period(
        self,
        lam_d: float,
        lam_o: float,
        lam_w: float,
        beta: np.ndarray,
        sigma: Optional[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """SEM panel posterior-predictive using Kronecker solve for ``B^{-1}``."""
        N = self._N_flow
        T = self._T
        n = self._n
        Xb = self._X @ beta
        if sigma is None:
            return Xb
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - lam_d * self._W_sparse).tocsr()
        Lo = (I_n - lam_o * self._W_sparse).tocsr()
        eps = rng.normal(scale=float(sigma), size=(N, T))
        u = kron_solve_matrix(Lo, Ld, eps, n)
        return Xb + u.T.reshape(-1)

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")
        return _compute_ols_flow_effects(
            self._idata,
            beta_draws=self._beta_layout(self._idata.posterior),
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._design_feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )
