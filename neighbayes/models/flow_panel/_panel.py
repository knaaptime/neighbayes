"""Panel extensions for Bayesian spatial flow (origin-destination) models.

This module introduces a panel-flow base class and four panel model
variants that extend the cross-sectional flow models to balanced panel data.

The panel stack uses time-first ordering. For each period t, the response is
an n^2-length vectorized origin-destination flow array, and all periods are
stacked to length n^2 * T.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import az, pm
from ..._logdet import (
    make_flow_separable_logdet,
    make_flow_separable_logdet_numpy,
)
from ..._ops import kron_solve_matrix
from ...graph import _weights_to_csr, flow_trace_blocks, flow_weight_matrices
from .._mixins._flow_shared import FlowSharedMethods
from ..flow import (
    _build_flow_effect_masks,
    _compute_flow_effects_lesage,
    _compute_ols_flow_effects,
)
from ..panel_base import SpatialPanelModel, _demean_panel


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
        self._idata: Optional[az.InferenceData] = None
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

        # Keep aliases matching flow model naming
        self._y = self._y
        self._X = self._X

        # Build flow weight matrices on N_flow = n^2 system
        wms = flow_weight_matrices(self._W_sparse)
        self._Wd: sp.csr_matrix = wms["destination"]
        self._Wo: sp.csr_matrix = wms["origin"]
        self._Ww: sp.csr_matrix = wms["network"]

        # Cache region-shock masks for LeSage effects decomposition.
        self._dmask, self._omask, self._imask = _build_flow_effect_masks(self._n)

        # Cache the symmetric 3x3 Kronecker trace matrix used by Bayesian
        # LM diagnostics on flow models: T[i,j] = tr(W_i' W_j) + tr(W_i W_j)
        # for (W_d, W_o, W_w).  Computed in O(nnz) from the n x n base graph.
        self._T_flow_traces: np.ndarray = flow_trace_blocks(self._W_sparse)

        # Spatial lags on demeaned/stationary panel stack
        self._Wd_y = self._sparse_flow_panel_lag(self._y, self._Wd)
        self._Wo_y = self._sparse_flow_panel_lag(self._y, self._Wo)
        self._Ww_y = self._sparse_flow_panel_lag(self._y, self._Ww)

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

    def _sparse_flow_panel_lag(
        self, v: np.ndarray, W_flow: sp.csr_matrix
    ) -> np.ndarray:
        """Apply panel flow lag I_T kron W_flow to time-first stacked vector."""
        chunks = v.reshape(self._T, self._N_flow)
        return np.asarray((W_flow @ chunks.T).T, dtype=np.float64).reshape(-1)

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
            for name in self._feature_names
            if name.startswith("dest_")
        ][: self._k_d]
        if len(feature_names) != self._k_d:
            feature_names = [f"x{i}" for i in range(self._k_d)]

        orig_feature_names = [
            name[len("orig_") :] if name.startswith("orig_") else name
            for name in self._feature_names
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
        :func:`~neighbayes.models.flow._compute_flow_effects_lesage` for the
        decomposition.  One sparse :math:`LU` factorization per draw covers all
        :math:`n` shock columns and all :math:`k` predictors.
        """
        n = self._n
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

        from ..flow import _EFFECT_KEYS

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            rw = float(rho_w_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

            solver = self._A_solver

            def _solve(
                rhs: np.ndarray, _s=solver, _rd=rd, _ro=ro, _rw=rw
            ) -> np.ndarray:
                return _s.solve([-_rd, -_ro, -_rw], rhs)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

        return out

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
        n = self._n
        k_d = self._k_d
        k_o = self._k_o
        W = self._W_sparse.tocsr()
        I_n = sp.eye(n, format="csr", dtype=np.float64)

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

        from ..flow import _EFFECT_KEYS

        out: dict[str, np.ndarray] = {}
        for side in ("dest", "orig"):
            k_side = k_d if side == "dest" else k_o
            for eff in _EFFECT_KEYS:
                out[f"{side}_{eff}"] = np.zeros(
                    (n_draws_total, k_side), dtype=np.float64
                )
        k_combined = k_d + k_o if k_d != k_o else k_d
        for eff in _EFFECT_KEYS:
            out[eff] = np.zeros((n_draws_total, k_combined), dtype=np.float64)

        for idx in range(n_draws_total):
            rd = float(rho_d_draws[idx])
            ro = float(rho_o_draws[idx])
            beta_d_vec = beta_draws[idx, dest_start : dest_start + k_d]
            beta_o_vec = beta_draws[idx, orig_start : orig_start + k_o]
            beta_intra_vec = (
                beta_draws[idx, intra_start : intra_start + k_d] if has_intra else None
            )

            Ld = (I_n - rd * W).tocsr()
            Lo = (I_n - ro * W).tocsr()

            def _solve(rhs: np.ndarray, _Lo=Lo, _Ld=Ld, _n=n) -> np.ndarray:
                return kron_solve_matrix(_Lo, _Ld, rhs, _n)

            res = _compute_flow_effects_lesage(
                _solve,
                self._dmask,
                self._omask,
                self._imask,
                beta_d_vec,
                beta_o_vec,
                n,
                k_d,
                k_o=k_o,
                beta_intra=beta_intra_vec,
            )
            for key, arr in res.items():
                out[key][idx, : len(arr)] = arr

        return out


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
        logdet_method: str = "jax",
        n_quad: int = 8,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
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
            (one value per draw, chain, and flow-period) for ``az.loo`` /
            ``az.waic``, on either sampler.  Off by default, as in PyMC.
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
    def _sample_resolvent(self, **kwargs) -> az.InferenceData:
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
    contribution scales as :math:`T \\log |A(\\rho_d, \\rho_o, \\rho_w)|`.

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

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
        - ``rho_lower`` : float, default -1.0 — Lower bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``rho_upper`` : float, default 1.0 — Upper bound of Uniform prior on each ρ (only when ``restrict_positive=False``).
        - ``nu`` : float, default 4.0 — Fixed Student-t degrees of freedom (only when ``robust=True``).
    """

    def __init__(self, *args, **kwargs):
        # Default to the resolvent-gradient panel sampler (subclasses that need the
        # PyMC path — e.g. the NB count panel — set a different logdet_method).
        kwargs.setdefault("logdet_method", "resolvent")
        super().__init__(*args, **kwargs)

    def _sample_resolvent(self, **kwargs) -> az.InferenceData:
        from ...samplers.gaussian._flow_resolvent import sample_flow_resolvent

        return sample_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            T=self._T,
            restrict_positive=self.restrict_positive,
            **kwargs,
        )

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import SparseFlowSolveMatrixOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        N = self._N_flow
        T = self._T
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
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            # Spatial filter: eta = A^{-1} X beta, then y = eta + epsilon
            Xb = pt.dot(X_t, beta)
            Xb_mat = pt.reshape(Xb, (T, N)).T  # (N, T)
            solve_op = SparseFlowSolveMatrixOp(self._Wd, self._Wo, self._Ww)
            eta_mat = solve_op(rho_d, rho_o, rho_w, Xb_mat)  # (N, T)
            mu = pt.reshape(eta_mat.T, (N * T,))

            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            # Jacobian: T * log|A| — but we don't have a differentiable
            # logdet for the unrestricted 3-ρ case in PyTensor.  The
            # resolvent sampler (sampler="gibbs") handles this correctly;
            # NUTS users should be aware that the Jacobian is not included
            # in this path.  For proper NUTS inference use sampler="gibbs".

        return model

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
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
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

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
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
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
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
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

            mu = rho_d * Wd_y_t + rho_o * Wo_y_t + rho_w * Ww_y_t + pt.dot(X_t, beta)
            if self.robust:
                nu = self._nu
                pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_t)
            else:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=y_t)

            pm.Potential(
                "jacobian",
                self._T * self._separable_logdet_fn(rho_d, rho_o),
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
        return self._T * self._separable_logdet_numpy_fn(rho_d, rho_o)

    def _compute_spatial_effects_posterior(
        self,
        draws: Optional[int] = None,
    ) -> dict[str, np.ndarray]:
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet. Call fit() first.")

        idata = self._idata
        rho_d_draws = idata.posterior["rho_d"].values.reshape(-1)
        rho_o_draws = idata.posterior["rho_o"].values.reshape(-1)
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
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

        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\beta`.
        - ``beta_sigma`` (float, default 1e6): Normal prior std for
          :math:`\beta`.
        - ``sigma_sigma`` (float, default 10.0): HalfNormal prior std
          for :math:`\sigma`.
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
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))
        y_t = pt.as_tensor_variable(self._y.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)
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
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )

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


class SARNegBinFlowPanel(SARFlowPanel):
    """Panel NB2 SAR flow model with unrestricted dependence parameters."""

    def __init__(self, y, X, W, **kwargs):
        # Count model: no |A| change-of-variables Jacobian, so it keeps the PyMC
        # path (not the Gaussian resolvent sampler); "none" routes fit accordingly.
        kwargs.setdefault("logdet_method", "none")
        effects_mode = int(kwargs.get("effects", kwargs.get("model", 0)))
        if effects_mode != 0:
            raise ValueError(
                "SARNegBinFlowPanel currently supports effects=0 only. "
                "Within-transformed FE panels are not valid for count models."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "SARNegBinFlowPanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "SARNegBinFlowPanel requires non-negative integer observations."
            )

        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
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

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` / ``az.waic`` on either sampler; off by default, as in PyMC.
        """
        if sampler == "gibbs":
            idata = self._fit_gibbs(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                n_jobs=n_jobs,
                log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            )
        elif sampler == "nuts":
            idata = super().fit(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                sampler="nuts",
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
        krylov_reuse: bool = True,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
        """Sample posterior via reduced-form PG-Gibbs (unrestricted 3-ρ panel)."""
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
            krylov_reuse=krylov_reuse,
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
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_from_draws(
            rho_d_draws,
            rho_o_draws,
            rho_w_draws,
            beta_draws,
            draws=draws,
        )

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """NB2 posterior-predictive replicate for the full panel stack."""
        N = self._N_flow
        T = self._T
        Xb = self._X @ beta
        Xb_mat = Xb.reshape(T, N).T
        eta_mat = self._solve_A(rho_d, rho_o, rho_w, Xb_mat)
        eta = eta_mat.T.reshape(-1)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        if alpha is None:
            raise ValueError("alpha is required for NegBin posterior_predictive")
        p = alpha / (alpha + lam)
        return rng.negative_binomial(alpha, p).astype(np.float64)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the NB2 SAR panel model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        rho_w_draws = post["rho_w"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        alpha_draws = post["alpha"].values.reshape(-1)

        total = len(rho_d_draws)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            rho_w_draws = rho_w_draws[:total]
            beta_draws = beta_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            out[g] = self._simulate_y_rep_period(
                float(rho_d_draws[g]),
                float(rho_o_draws[g]),
                float(rho_w_draws[g]),
                beta_draws[g],
                None,
                rng,
                alpha=float(alpha_draws[g]),
            )
        return out

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import SparseFlowSolveMatrixOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        N = self._N_flow
        T = self._T
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

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            Xb = pt.dot(X_t, beta)
            Xb_mat = pt.reshape(Xb, (T, N)).T
            solve_op = SparseFlowSolveMatrixOp(self._Wd, self._Wo, self._Ww)
            eta_mat = solve_op(rho_d, rho_o, rho_w, Xb_mat)
            eta = pt.reshape(eta_mat.T, (N * T,))
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            # No |A| change-of-variables Jacobian for the count likelihood: the NB
            # mean is η = A⁻¹Xβ and y is modelled directly (adding it biases β).

        return model


class SARNegBinFlowSeparablePanel(SARFlowSeparablePanel):
    """Panel separable NB2 SAR flow model."""

    def __init__(self, y, X, W, **kwargs):
        effects_mode = int(kwargs.get("effects", kwargs.get("model", 0)))
        if effects_mode != 0:
            raise ValueError(
                "SARNegBinFlowSeparablePanel currently supports effects=0 only. "
                "Within-transformed FE panels are not valid for count models."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "SARNegBinFlowSeparablePanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "SARNegBinFlowSeparablePanel requires non-negative integer observations."
            )

        method = kwargs.pop("logdet_method", None)
        _VALID = {"eigenvalue", "chebyshev", "cheb_cholesky", "aaa", "cheb_stochastic"}
        if method is not None and method not in _VALID:
            raise ValueError(
                f"SARNegBinFlowSeparablePanel logdet_method must be None (auto) or one of "
                f"{sorted(_VALID)}; got {method!r}."
            )
        kwargs["logdet_method"] = method
        super().__init__(y_arr.astype(np.float64), X, W, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        *,
        sampler: str = "gibbs",
        attach_log_abs_det: bool = True,
        progressbar: bool = True,
        n_jobs: int = -1,
        idata_kwargs: Optional[dict] = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Sample the separable NB2 SAR flow panel posterior.

        ``sampler="gibbs"`` (default) runs the reduced-form Pólya–Gamma Gibbs
        sampler with per-period Kronecker solves; ``sampler="nuts"`` uses the
        PyMC count path (exact likelihood, much slower).  With
        ``attach_log_abs_det`` (default) the per-draw Jacobian ``T·log|A(ρ)|``
        (using the separability relation ``ρ_w = −ρ_d ρ_o``) is recorded in
        ``sample_stats["log_abs_det"]`` for diagnostics — not folded into the
        count model's ``log_likelihood``.

        ``idata_kwargs={"log_likelihood": True}`` stores the pointwise
        log-likelihood (one value per draw, chain, and flow-period) for
        ``az.loo`` / ``az.waic`` on either sampler; off by default, as in PyMC.
        """
        if sampler == "gibbs":
            idata = self._fit_gibbs(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                progressbar=progressbar,
                n_jobs=n_jobs,
                log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
            )
        elif sampler == "nuts":
            idata = super().fit(
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
        krylov_reuse: bool = True,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
        """Sample posterior via reduced-form PG-Gibbs (separable 2-ρ panel)."""
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
            krylov_reuse=krylov_reuse,
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
        beta_draws = idata.posterior["beta"].values.reshape(
            -1, len(self._feature_names)
        )
        return self._compute_flow_effects_kron(
            rho_d_draws,
            rho_o_draws,
            beta_draws,
            draws=draws,
        )

    def _simulate_y_rep_period(
        self,
        rho_d: float,
        rho_o: float,
        rho_w: float,  # ignored; rho_w = -rho_d * rho_o
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """NB2 posterior-predictive replicate using Kronecker solve."""
        N = self._N_flow
        T = self._T
        n = self._n
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        Ld = (I_n - rho_d * self._W_sparse).tocsr()
        Lo = (I_n - rho_o * self._W_sparse).tocsr()
        Xb = self._X @ beta
        Xb_mat = Xb.reshape(T, N).T  # (N, T)
        eta_mat = kron_solve_matrix(Lo, Ld, Xb_mat, n)
        eta = eta_mat.T.reshape(-1)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        if alpha is None:
            raise ValueError("alpha is required for NegBin posterior_predictive")
        p = alpha / (alpha + lam)
        return rng.negative_binomial(alpha, p).astype(np.float64)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the NB2 separable SAR panel model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        rho_d_draws = post["rho_d"].values.reshape(-1)
        rho_o_draws = post["rho_o"].values.reshape(-1)
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        alpha_draws = post["alpha"].values.reshape(-1)

        total = len(rho_d_draws)
        if n_draws is not None:
            total = min(int(n_draws), total)
            rho_d_draws = rho_d_draws[:total]
            rho_o_draws = rho_o_draws[:total]
            beta_draws = beta_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            out[g] = self._simulate_y_rep_period(
                float(rho_d_draws[g]),
                float(rho_o_draws[g]),
                0.0,  # rho_w ignored for separable
                beta_draws[g],
                None,
                rng,
                alpha=float(alpha_draws[g]),
            )
        return out

    def _build_pymc_model(self) -> pm.Model:
        from ..._ops import KroneckerFlowSolveMatrixOp

        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)
        rho_lower = self.priors.get("rho_lower", -0.999)
        rho_upper = self.priors.get("rho_upper", 0.999)

        if self._separable_logdet_fn is None:
            raise RuntimeError(
                "SARNegBinFlowSeparablePanel requires precomputed logdet data; "
                "initialize with a separable logdet_method (None/auto, "
                "eigenvalue, chebyshev, cheb_cholesky, aaa, or cheb_stochastic)."
            )
        n = self._n
        N = self._N_flow
        T = self._T
        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            rho_d = pm.Uniform("rho_d", lower=rho_lower, upper=rho_upper)
            rho_o = pm.Uniform("rho_o", lower=rho_lower, upper=rho_upper)
            pm.Deterministic("rho_w", -rho_d * rho_o)

            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)

            Xb = pt.dot(X_t, beta)
            Xb_mat = pt.reshape(Xb, (T, N)).T
            solve_op = KroneckerFlowSolveMatrixOp(self._W_sparse, n)
            eta_mat = solve_op(rho_d, rho_o, Xb_mat)
            eta = pt.reshape(eta_mat.T, (N * T,))
            lam = pm.Deterministic("lambda", pt.exp(eta))

            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

            # No |A| change-of-variables Jacobian for the count likelihood:
            # the NB mean is η = A⁻¹Xβ and y is modelled directly, so the
            # spatial filter enters only through the mean.  (The Gaussian
            # separable panel keeps the Jacobian; copying it here biases
            # ρ toward the negative-logdet region.)

        return model


class NegBinFlowPanel(OLSFlowPanel):
    """Aspatial panel OD-flow NB2 gravity baseline."""

    def __init__(self, y, X, W, T, **kwargs):
        effects_mode = int(kwargs.get("effects", kwargs.get("model", 0)))
        if effects_mode != 0:
            raise ValueError(
                "NegBinFlowPanel currently supports effects=0 only. "
                "Within-transformed FE panels are not valid for count models."
            )

        y_arr = np.asarray(y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_rounded = np.round(y_arr).astype(np.int64)
            if not np.allclose(y_arr, y_rounded):
                raise ValueError(
                    "NegBinFlowPanel requires integer-valued observations; "
                    f"got dtype {y_arr.dtype} with non-integer values."
                )
            y_arr = y_rounded
        if np.any(y_arr < 0):
            raise ValueError(
                "NegBinFlowPanel requires non-negative integer observations."
            )
        super().__init__(y_arr.astype(np.float64), X, W, T, **kwargs)
        self._y_int_vec: np.ndarray = y_arr.reshape(-1).astype(np.int64)

    def _simulate_y_rep_period(
        self,
        rho_d: float,  # unused
        rho_o: float,  # unused
        rho_w: float,  # unused
        beta: np.ndarray,
        sigma: Optional[float],  # unused
        rng: np.random.Generator,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """NB2 posterior-predictive replicate ``y_rep`` (full panel stack)."""
        eta = self._X @ beta  # (N_flow * T,)
        lam = np.exp(np.clip(eta, -50.0, 50.0))
        if alpha is None:
            raise ValueError("alpha is required for NegBin posterior_predictive")
        p = alpha / (alpha + lam)
        return rng.negative_binomial(alpha, p).astype(np.float64)

    def posterior_predictive(
        self,
        n_draws: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw posterior-predictive flows for the NB2 panel gravity model."""
        if self._idata is None:
            raise RuntimeError("Model has not been fit yet.  Call fit() first.")

        post = self._idata.posterior
        beta_draws = post["beta"].values.reshape(-1, len(self._feature_names))
        alpha_draws = post["alpha"].values.reshape(-1)

        total = beta_draws.shape[0]
        if n_draws is not None:
            total = min(int(n_draws), total)
            beta_draws = beta_draws[:total]
            alpha_draws = alpha_draws[:total]

        rng = np.random.default_rng(random_seed)
        out = np.empty((total, self._N_flow * self._T), dtype=np.float64)
        for g in range(total):
            out[g] = self._simulate_y_rep_period(
                0.0,
                0.0,
                0.0,
                beta_draws[g],
                None,
                rng,
                alpha=float(alpha_draws[g]),
            )
        return out

    def _build_pymc_model(self) -> pm.Model:
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 10.0)
        alpha_sigma = self.priors.get("alpha_sigma", 10.0)

        X_t = pt.as_tensor_variable(self._X.astype(np.float64))

        with pm.Model(coords=self._model_coords()) as model:
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfNormal("alpha", sigma=alpha_sigma)
            eta = pt.dot(X_t, beta)
            lam = pm.Deterministic("lambda", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=lam, alpha=alpha, observed=self._y_int_vec)

        return model


# ---------------------------------------------------------------------------
# Panel SEM-Flow variants (spatial-error analogues of SARFlowPanel)
# ---------------------------------------------------------------------------


def _sparse_flow_panel_lag_matrix(
    M: np.ndarray, W_flow: sp.csr_matrix, T: int, N_flow: int
) -> np.ndarray:
    """Apply :math:`I_T \\otimes W_{flow}` to a stacked panel design matrix.

    Parameters
    ----------
    M : np.ndarray, shape ``(N_flow * T, p)``
        Time-first stacked design matrix.
    W_flow : scipy.sparse matrix, shape ``(N_flow, N_flow)``
        Flow weight matrix (one of ``W_d``, ``W_o``, ``W_w``).
    T, N_flow : int
        Panel dimensions.

    Returns
    -------
    np.ndarray, shape ``(N_flow * T, p)``
        ``W_flow`` applied to each period block independently.
    """
    p = M.shape[1] if M.ndim == 2 else 1
    chunks = M.reshape(T, N_flow, p)
    out = np.empty_like(chunks)
    for t in range(T):
        out[t] = W_flow @ chunks[t]
    return out.reshape(T * N_flow, p)


class _SEMFlowPanelMixin:
    """Shared init helper to precompute design-matrix lags for SEM panel models."""

    def _init_sem_lags(self) -> None:
        T = self._T
        N = self._N_flow
        # Lags of the (already-demeaned) design matrix.  Constants — no
        # parameter dependence, so we precompute once.
        self._Wd_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Wd, T, N
        )
        self._Wo_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Wo, T, N
        )
        self._Ww_X: np.ndarray = _sparse_flow_panel_lag_matrix(
            self._X.astype(np.float64), self._Ww, T, N
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

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
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

    def _sample_resolvent(self, **kwargs) -> az.InferenceData:
        from ...samplers.gaussian._flow_resolvent import sample_sem_flow_resolvent

        return sample_sem_flow_resolvent(
            self._W_sparse,
            self._y,
            self._X,
            T=self._T,
            restrict_positive=self.restrict_positive,
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
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
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

        - ``beta_mu`` : float, default 0.0 — Normal prior mean for ``beta``.
        - ``beta_sigma`` : float, default 1e6 — Normal prior std for ``beta``.
        - ``sigma_sigma`` : float, default 10.0 — HalfNormal prior std for ``sigma``.
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
        beta_mu = self.priors.get("beta_mu", 0.0)
        beta_sigma = self.priors.get("beta_sigma", 1e6)
        sigma_sigma = self.priors.get("sigma_sigma", 10.0)
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
            sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

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
                self._T * self._separable_logdet_fn(lam_d, lam_o),
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
        return self._T * self._separable_logdet_numpy_fn(lam_d, lam_o)

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
            n=self._n,
            k_d=self._k_d,
            k_o=self._k_o,
            feature_names=self._feature_names,
            intra_idx=self._intra_idx,
            draws=draws,
        )
