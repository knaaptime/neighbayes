"""Base classes and helpers for spatial panel models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from formulaic import model_matrix
from libpysal.graph import Graph

from .._backends.sampler_helpers import jax_available
from .._lazy_deps import az, pm
from ._base._shared import (
    SharedSpatialMethods,
    _check_row_standardization,
)
from ._base._structure import PanelStructure

# ---------------------------------------------------------------------------
# Effects specification helpers
# ---------------------------------------------------------------------------

_EFFECTS_MAP: dict[str, int] = {
    "pooled": 0,
    "unit": 1,
    "time": 2,
    "two_way": 3,
}

_EFFECTS_NAMES: dict[int, str] = {v: k for k, v in _EFFECTS_MAP.items()}


def _resolve_effects(effects: Union[str, int]) -> int:
    """Map ``effects`` argument to an integer FE mode (0–3).

    Accepts integers (0–3) or strings ("pooled", "unit", "time", "two_way").
    """
    if isinstance(effects, str):
        key = effects.strip().lower()
        if key not in _EFFECTS_MAP:
            valid = ", ".join(sorted(_EFFECTS_MAP))
            raise ValueError(
                f"effects={effects!r} is not recognized. "
                f"Valid strings: {valid}; valid ints: 0–3."
            )
        return _EFFECTS_MAP[key]
    if isinstance(effects, int) and 0 <= effects <= 3:
        return effects
    raise ValueError(
        f"effects={effects!r} is not recognized. "
        "Use an int 0–3 or one of: 'pooled', 'unit', 'time', 'two_way'."
    )


def _demean_panel(y: np.ndarray, X: np.ndarray, N: int, T: int, effects: int):
    """Apply panel demeaning transformation.

    Implements the within-transformation for two-way fixed-effects panel
    models prior to the spatial filter.  In the SAR-FE setting we model

    .. math::

        y_{it} = \\rho \\sum_j W_{ij} y_{jt} + X_{it}\\beta + \\mu_i
                 + \\alpha_t + \\varepsilon_{it},

    and concentrate out the fixed effects by demeaning *both* sides of
    the equation before the spatial lag is applied.  Because :math:`W`
    operates only across units (within a period), the within-period
    demeaning commutes with :math:`W` (i.e. ``W (M_T y) = M_T (W y)``)
    so the order of "demean then apply :math:`W`" or "apply :math:`W`
    then demean" yields the same likelihood — a fact exploited in
    Lee & Yu (2010) and Elhorst (2014, ch. 3).  This is why
    :func:`neighbayes.models.panel.SARPanel` builds ``Wy`` from the
    *demeaned* ``y`` returned here without an additional Jacobian
    correction beyond the standard :math:`T\\,\\log|I_N - \\rho W|`
    panel Jacobian.

    References
    ----------
    Lee, L.-F. & Yu, J. (2010). Estimation of spatial autoregressive
    panel data models with fixed effects.  *Journal of Econometrics*,
    154(2), 165–185.

    Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional
    Data to Spatial Panels*. Springer.

    Parameters
    ----------
    y : np.ndarray
        Stacked dependent variable of shape ``(N*T,)``.
    X : np.ndarray
        Stacked regressor matrix of shape ``(N*T, k)``.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    effects : int
        Fixed-effects mode: 0 pooled, 1 unit FE, 2 time FE, 3 two-way FE.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Demeaned ``(y, X)`` arrays in stacked format.
    """
    y2 = y.reshape(T, N)
    X3 = X.reshape(T, N, X.shape[1])

    if effects in (1, 3) and T < 2:
        raise ValueError(
            f"Unit fixed effects (effects={effects}) require T >= 2 to identify "
            "within-unit variation, but T=" + str(T) + " was supplied. "
            "Use effects=0 (pooled) or effects=2 (time FE) when T=1."
        )

    if effects == 0:
        y_with = y2
        X_with = X3
    elif effects == 1:
        y_with = y2 - y2.mean(axis=0, keepdims=True)
        X_with = X3 - X3.mean(axis=0, keepdims=True)
    elif effects == 2:
        y_with = y2 - y2.mean(axis=1, keepdims=True)
        X_with = X3 - X3.mean(axis=1, keepdims=True)
    elif effects == 3:
        y_i = y2.mean(axis=0, keepdims=True)
        y_t = y2.mean(axis=1, keepdims=True)
        y_g = y2.mean()
        y_with = y2 - y_i - y_t + y_g

        X_i = X3.mean(axis=0, keepdims=True)
        X_t = X3.mean(axis=1, keepdims=True)
        X_g = X3.mean(axis=(0, 1), keepdims=True)
        X_with = X3 - X_i - X_t + X_g
    else:
        raise ValueError("effects must be one of {0,1,2,3}")

    return y_with.reshape(-1), X_with.reshape(-1, X.shape[1])


def _as_dense_W(W: Union[Graph, sp.spmatrix, np.ndarray], N: int, T: int) -> np.ndarray:
    """Convert graph/sparse/array weights into dense panel-compatible matrix.

    Parameters
    ----------
    W : Graph, scipy.sparse, or np.ndarray
        Either an ``N x N`` cross-sectional matrix or an ``(N*T) x (N*T)``
        block-diagonal panel matrix. Public APIs accept only Graph or sparse
        inputs; ndarray is supported here for internal use.
    N : int
        Number of units.
    T : int
        Number of periods.

    Returns
    -------
    np.ndarray
        Dense panel weights matrix.
    """
    if isinstance(W, Graph):
        Wn = W.sparse.toarray().astype(float)
    elif sp.issparse(W):
        Wn = W.toarray().astype(float)
    else:
        Wn = np.asarray(W, dtype=float)

    if Wn.shape == (N, N):
        return np.kron(np.eye(T), Wn)
    if Wn.shape == (N * T, N * T):
        return Wn

    raise ValueError(
        f"W has shape {Wn.shape}; expected (N,N)=({N},{N}) or (N*T,N*T)=({N * T},{N * T})."
    )


def _parse_panel_W(
    W: Union[Graph, sp.spmatrix],
    N: int,
    T: int,
) -> sp.csr_matrix:
    """Validate W and return it as a CSR sparse matrix sized ``(N, N)``.

    Accepts a :class:`libpysal.graph.Graph` or any :class:`scipy.sparse`
    matrix. Raises a :class:`ValueError` if the shape is incompatible with
    *N* (and optionally *T*). Issues a :class:`UserWarning` when *W* does not
    appear to be row-standardized.

    Returns the CSR representation of the ``N x N`` cross-sectional matrix;
    callers that need the full ``(N*T) x (N*T)`` Kronecker form should use
    :func:`_as_dense_W` or build the sparse Kronecker product themselves.
    """
    if isinstance(W, Graph):
        W_csr = W.sparse.tocsr().astype(np.float64)
    elif sp.issparse(W):
        W_csr = W.tocsr().astype(np.float64)
    elif hasattr(W, "sparse") and hasattr(W, "transform"):
        raise TypeError(
            "W appears to be a legacy libpysal.weights.W object. "
            "Convert it to a libpysal.graph.Graph first: "
            "Graph.from_W(w), or pass w.sparse (the scipy sparse matrix) directly."
        )
    else:
        raise TypeError(
            f"W must be a libpysal.graph.Graph or a scipy sparse matrix, "
            f"got {type(W).__name__}."
        )

    if W_csr.ndim != 2 or W_csr.shape[0] != W_csr.shape[1]:
        raise ValueError(f"W must be square, got shape {W_csr.shape}.")

    if W_csr.shape[0] == N:
        pass  # N x N unit matrix — expected
    elif W_csr.shape[0] == N * T:
        # Caller passed the full block matrix; extract N x N block for storage.
        # We keep it as-is but raise if neither shape matches.
        pass
    else:
        raise ValueError(
            f"W has shape {W_csr.shape} but data has N={N} units (T={T} periods). "
            f"W must be ({N},{N}) or ({N * T},{N * T})."
        )

    return W_csr, _check_row_standardization(W_csr)


class SpatialPanelModel(SharedSpatialMethods, ABC):
    """Base class for static spatial panel models with FE transforms.

    Holds the within-transformation, panel-aware sorting, and weights
    handling shared by all static fixed-effects panel model subclasses.
    Not instantiated directly.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``, ``unit_col``, and ``time_col``.
    data : pandas.DataFrame, optional
        Long-format panel data when using formula mode. Must contain
        the response, regressors, ``unit_col``, and ``time_col``.
    y : array-like, optional
        Stacked response of shape ``(N*T,)`` in unit-major order.
        Required in matrix mode.
    X : array-like or pandas.DataFrame, optional
        Stacked design matrix of shape ``(N*T, k)``. Required in matrix
        mode. DataFrame columns are preserved as feature names.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(N, N)`` (preferred — broadcast over
        time periods) or the full ``(N*T, N*T)`` block-diagonal panel
        matrix. Accepts a :class:`libpysal.graph.Graph` or any
        :class:`scipy.sparse` matrix. The legacy
        :class:`libpysal.weights.W` object is **not** accepted; pass
        ``w.sparse`` or ``libpysal.graph.Graph.from_W(w)``. Should be
        row-standardized; a :class:`UserWarning` is raised otherwise.
    unit_col : str, optional
        Column in ``data`` identifying the cross-sectional unit.
        Required in formula mode for panel sorting and N/T inference.
    time_col : str, optional
        Column in ``data`` identifying the time period. Required in
        formula mode.
    N : int, optional
        Number of cross-sectional units. Required in matrix mode if
        not inferable from ``W`` or the data shape.
    T : int, optional
        Number of time periods. Required in matrix mode if not
        inferable.
    effects : str or int, default 0
        Fixed-effects specification: ``"pooled"`` (or ``0``),
        ``"unit"`` (or ``1``), ``"time"`` (or ``2``),
        ``"two_way"`` (or ``3``).  The within transformation is
        applied to ``y`` and ``X`` before likelihood evaluation.
    priors : dict, optional
        Override default priors. Supported keys depend on the subclass;
        each subclass docstring lists its keys with defaults.
    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects from the size and symmetry of the cross-sectional
        ``N x N`` weights: ``"eigenvalue"`` for ``N <= 500``; for
        ``N <= 20000`` the exact interpolating methods ``"cheb_cholesky"``
        (symmetric ``W``) or ``"aaa"`` (non-symmetric ``W``); and
        ``"cheb_stochastic"`` above that.  Stochastic opt-ins
        (``"chebyshev"``, ``"slq"``) and the environment variables
        controlling the cutoffs are documented on the cross-sectional
        ``SpatialModel`` base class.
    logdet_refit : bool, default True
        Rebuild the log-determinant interpolant halfway through warmup, on
        the range the chains have found rather than the interval implied by
        the prior.  A warmup posterior is typically one to two orders of
        magnitude narrower than the prior, which needs far fewer interpolation
        nodes and drives the approximation error over the posterior's support
        down to the factorization's roundoff floor.  Applies to
        ``"cheb_cholesky"``, ``"lu_cheb"``, ``"aaa"`` and ``"chol_aaa"``;
        ignored otherwise.

        The interpolant is only valid on its interval, so the refit window
        becomes the sampler's support.  The window is padded by
        ``logdet_refit_pad_sd`` warmup standard deviations, recorded in
        ``idata.attrs["logdet_refit_window"]``, and a warning is raised if
        the retained draws ever reach an edge the refit introduced.
    logdet_refit_pad_sd : float, default 10.0
        Padding for the refit window, in warmup posterior standard
        deviations.  At the default the truncated tail is ~1e-23 under
        normality, and the padding costs a node or two at most.
    logdet_aaa_check : bool, default True
        For the AAA methods (``"aaa"``, ``"chol_aaa"``), set the number of
        exact factorizations from where the posterior lies.  Warmup starts on a
        14-node fit; halfway through, nodes are added only if the warmup
        posterior lies closer to a singularity of the Jacobian than four times
        the distance at which the fit's poles resolve it.  The count used is
        recorded in ``idata.attrs["logdet_aaa_nodes"]``.  Off, or on a path
        without a warmup midpoint, the count is fixed by the prior interval:
        14 nodes within ``|ρ| ≤ 0.9``, 18 otherwise.
    robust : bool, default False
        If True, replace the Normal error with Student-t for robustness
        to heavy-tailed outliers.  The degrees of freedom :math:`\\nu` are
        **fixed** at ``priors["nu"]`` (default 4, LeSage's ``rval``).
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for
        subclasses that include ``WX`` terms (``SLXPanelFE``,
        ``SDMPanelFE``, ``SDEMPanelFE`` and their RE/dynamic
        analogues). By default all non-constant columns are lagged.
        Pass a subset, e.g. ``w_vars=["income", "density"]``.
    """

    # Emit a ResourceWarning before materializing very large dense panel
    # weight matrices. Tests may monkeypatch this value.
    _DENSE_W_WARN_BYTES: int = 100 * 1024 * 1024

    # Use the Panel-prefixed LM tests and panel decision specs in the shared
    # spatial_diagnostics_decision (see SharedSpatialMethods).
    _panel_diagnostics = True

    # Subclasses that include WX coefficients in the posterior beta
    # vector (SDM, SDEM, SLX) should set this to True.
    _has_wx_in_beta: bool = False

    # Subclasses with a spatial autoregressive term should set this to
    # "rho" (spatial lag) or "lam" (spatial error).  OLS/SLX leave it None.
    _jacobian_param: str | None = None

    # Sampler-registry wiring (mirrors ``SpatialModel``).  Gaussian FE
    # subclasses set ``_likelihood="gaussian"``, ``_gibbs_key=("gaussian",
    # "panel_fe")`` and the ``_gibbs_class``/``_model_type`` the generic
    # ``_fit_gibbs`` resolves at runtime.  OLS/SLX are NUTS-only (no
    # ``_gibbs_key``).
    _likelihood: str = ""
    _gibbs_key: tuple[str, str] | None = None
    _gibbs_class: str | None = None
    _model_type: str = ""

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        unit_col: Optional[str] = None,
        time_col: Optional[str] = None,
        N: Optional[int] = None,
        T: Optional[int] = None,
        effects: Union[str, int] = 0,
        priors: Optional[dict] = None,
        logdet_method: str | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
        logdet_refit: bool = True,
        logdet_refit_pad_sd: float = 10.0,
        logdet_aaa_check: bool = True,
    ):
        if W is None:
            raise ValueError("W is required.")

        # Resolve typed priors (dataclass) and dict view.
        from .priors import PanelBasePriors, priors_as_dict, resolve_priors

        _priors_cls = getattr(self.__class__, "_priors_cls", PanelBasePriors)
        self.priors_obj = resolve_priors(priors, _priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.logdet_method = logdet_method
        self.logdet_refit = bool(logdet_refit)
        self.logdet_refit_pad_sd = float(logdet_refit_pad_sd)
        self.logdet_aaa_check = bool(logdet_aaa_check)
        self.model = _resolve_effects(effects)
        self.effects = _EFFECTS_NAMES[self.model]
        self.robust = robust
        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None
        self._W_dense_cache: Optional[np.ndarray] = None

        if formula is not None:
            if data is None:
                raise ValueError("data is required with formula mode.")
            if unit_col is None or time_col is None:
                raise ValueError("unit_col and time_col are required in formula mode.")

            d = data.sort_values([time_col, unit_col]).reset_index(drop=True)
            lhs, rhs = formula.split("~", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()

            X_mm = model_matrix(rhs, d)
            self._feature_names = list(X_mm.columns)
            y_arr = np.asarray(d[lhs], dtype=float)
            X_arr = np.asarray(X_mm, dtype=float)

            units = d[unit_col].nunique()
            times = d[time_col].nunique()
            if units * times != len(d):
                raise ValueError(
                    "Data are not a balanced panel after sorting by time/unit."
                )
            self._N = units
            self._T = times
            self._panel_index = d[[time_col, unit_col]].copy()
        elif y is not None and X is not None:
            y_arr = np.asarray(y, dtype=float).reshape(-1)
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
                X_arr = X.to_numpy(dtype=float)
            else:
                X_arr = np.asarray(X, dtype=float)
                self._feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

            if N is None or T is None:
                raise ValueError("N and T are required in matrix mode.")
            self._N = int(N)
            self._T = int(T)
            self._panel_index = None
            if self._N * self._T != len(y_arr):
                raise ValueError("N*T must equal number of stacked observations.")
        else:
            raise ValueError(
                "Provide either (formula,data,unit_col,time_col) or (y,X,N,T)."
            )

        self._y_raw = y_arr
        self._X_raw = X_arr

        self._wx_column_indices = self._spatial_lag_column_indices(
            self._X_raw, self._feature_names
        )
        if w_vars is not None:
            unknown = [v for v in w_vars if v not in self._feature_names]
            if unknown:
                raise ValueError(
                    f"w_vars contains names not found in X columns: {unknown}. "
                    f"Available: {self._feature_names}"
                )
            self._wx_column_indices = [
                i for i in self._wx_column_indices if self._feature_names[i] in w_vars
            ]
        self._wx_feature_names = [
            self._feature_names[i] for i in self._wx_column_indices
        ]

        # Validate W and store as N×N CSR. Dense expansion is deferred.
        self._W_sparse, self._is_row_std = _parse_panel_W(W, self._N, self._T)
        self._structure = PanelStructure(self._W_sparse, self._N, self._T)
        # Eigenvalues of the N×N matrix are deferred — see ``_W_eigs`` property.

        # Resolve the logdet method and rho/lambda bounds exactly once,
        # passing the N×N W so auto-selection can honour graph directedness.
        # For row-standardized W the spectral stability interval is
        # always approximately (-1, 1), so no eigenvalue computation
        # is needed here.
        from .._logdet import resolve_logdet_bounds

        self._logdet_bounds = resolve_logdet_bounds(
            self.logdet_method,
            n=self._W_sparse.shape[0],
            priors=self.priors,
            W=self._W_sparse,
        )
        self._resolved_logdet_method = self._logdet_bounds.method

        self._y, self._X = _demean_panel(
            self._y_raw, self._X_raw, self._N, self._T, self.model
        )

        # For FE models (effects != 0), demeaning zeros out intercept/constant
        # columns.  Drop them from the design matrix so that both NUTS and
        # Gibbs see the same X, and the posterior beta has a consistent
        # dimension regardless of sampler.
        if self.model != 0:
            ni = self._nonintercept_indices
            if len(ni) < self._X.shape[1]:
                # Build a mapping: for each original column index, how many
                # dropped (constant) columns precede it?
                dropped = sorted(set(range(self._X.shape[1])) - set(ni))
                shift = {
                    orig: orig - sum(d < orig for d in dropped)
                    for orig in range(self._X.shape[1])
                }
                self._X = self._X[:, ni]
                self._feature_names = [self._feature_names[i] for i in ni]
                # Shift spatial-lag column indices to account for dropped cols.
                self._wx_column_indices = [shift[j] for j in self._wx_column_indices]
                self._wx_feature_names = [
                    self._feature_names[j] for j in self._wx_column_indices
                ]

        self._Wy = self._sparse_panel_lag(self._y)
        if self._wx_column_indices:
            # Single batched sparse multiply across all WX columns, replacing
            # the per-column Python loop that previously paid an O(k_wx)
            # overhead.
            self._WX = self._sparse_panel_lag(self._X[:, self._wx_column_indices])
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=float)

    def _sparse_panel_lag(self, v: np.ndarray) -> np.ndarray:
        """Apply the panel spatial lag ``W ⊗ I_T`` to a stacked vector/matrix.

        Thin delegator to :meth:`PanelStructure.spatial_lag` (kept as a method
        because samplers and diagnostics call ``model._sparse_panel_lag``).
        Accepts a 1-D stacked vector of length ``N*T`` or a 2-D ``(N*T, k)``
        matrix whose columns are all lagged in one batched sparse multiply.
        """
        return self._structure.spatial_lag(v)

    def _batch_sparse_lag(
        self,
        resid: np.ndarray,
        T_eff: int | None = None,
    ) -> np.ndarray:
        """Apply panel spatial lag to a batch of stacked residual draws.

        Thin delegator to :meth:`PanelStructure.batch_spatial_lag` (kept as a
        method because samplers call ``model._batch_sparse_lag``).  ``resid``
        has shape ``(n_draws, N*T_eff)`` (``T_eff`` defaults to ``T``; dynamic
        panels pass ``T-1``) and the result has the same shape.
        """
        return self._structure.batch_spatial_lag(resid, T_eff)

    @property
    def _W_dense(self) -> np.ndarray:
        """Dense (N*T)×(N*T) weight matrix, materialized lazily on first access."""
        if self._W_dense_cache is None:
            # If W is N x N, dense panel matrix is (N*T) x (N*T); otherwise
            # caller supplied full panel matrix already.
            n_nt = (
                self._N * self._T
                if self._W_sparse.shape[0] == self._N
                else int(self._W_sparse.shape[0])
            )
            nbytes = n_nt * n_nt * 8
            if nbytes > int(self._DENSE_W_WARN_BYTES):
                warnings.warn(
                    f"Materialising a dense panel weight matrix of size {n_nt}x{n_nt} "
                    f"(~{nbytes / 1024**2:.0f} MB).",
                    ResourceWarning,
                    stacklevel=2,
                )
            self._W_dense_cache = _as_dense_W(self._W_sparse, self._N, self._T)
        return self._W_dense_cache

    @property
    def _W_sparse_NT(self) -> "sp.csr_matrix":
        """Sparse (N*T)×(N*T) Kronecker-block weight matrix ``I_T ⊗ W_n``.

        Delegates to :meth:`PanelStructure.W_sparse_NT` (kept as a property
        because RE/dynamic samplers read ``model._W_sparse_NT``).
        """
        return self._structure.W_sparse_NT()

    @property
    def _W_pt_sparse(self):
        """PyTensor sparse operator for the PyMC model (delegates to structure).

        Structure-cached so repeated PyMC model builds reuse the same symbolic
        sparse weight operator.
        """
        return self._structure.W_pt_sparse()

    @property
    def _intercept_dropped(self) -> bool:
        """Whether the intercept column was dropped from the posterior beta.

        Checks the actual posterior beta dimension against the design
        matrix.  When Gibbs sampling is used with FE models, the
        intercept column (all zeros after demeaning) is dropped, so
        the posterior beta has fewer columns than the full design.
        NUTS samplers keep the full design matrix, so dimensions match.

        Returns True when the posterior beta has exactly the number of
        columns expected if the intercept was dropped.
        """
        n_intercept = self._X.shape[1] - len(self._nonintercept_indices)
        if n_intercept == 0:
            return False
        if not hasattr(self, "_idata") or self._idata is None:
            return False
        try:
            beta = self._idata.posterior["beta"]
            beta_cols = beta.shape[-1]
        except (KeyError, AttributeError):
            return False
        # Compute expected beta sizes with and without intercept.
        # Models with WX in beta (SDM, SDEM, SLX) include the WX block.
        kw = self._WX.shape[1] if self._has_wx_in_beta else 0
        expected_with = self._X.shape[1] + kw
        expected_without = len(self._nonintercept_indices) + kw
        return beta_cols == expected_without and beta_cols < expected_with

    @property
    def _beta_nonintercept_indices(self) -> list[int]:
        """Indices into the *posterior beta* for non-intercept columns.

        When the intercept is present in beta (NUTS or pooled models),
        these are the same as ``_nonintercept_indices``.
        When the intercept was dropped (Gibbs + FE), the indices are
        shifted to account for the missing column.
        """
        if not self._intercept_dropped:
            return list(self._nonintercept_indices)
        # Intercept was dropped: recompute indices relative to the
        # smaller beta vector by counting how many intercept/constant
        # columns precede each non-intercept column.
        ni = self._nonintercept_indices
        n_const = ni[0] if ni else 0  # number of constant cols before first non-const
        return [i - n_const for i in ni]

    @property
    def _beta_wx_column_indices(self) -> list[int]:
        """Indices into the *posterior beta1* (first k cols of beta) for WX columns.

        When the intercept is present in beta1, these are the same as
        ``_wx_column_indices``.  When the intercept was dropped (Gibbs + FE),
        the indices are shifted to account for the missing column.
        """
        if not self._intercept_dropped:
            return list(self._wx_column_indices)
        ni = self._nonintercept_indices
        n_const = ni[0] if ni else 0
        return [i - n_const for i in self._wx_column_indices]

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model."""

    @abstractmethod
    @abstractmethod
    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)`` where each
            array has shape ``(G, k)`` or ``(G, k_wx)``.
        """

    # _fitted_mean_from_posterior: concrete default in SharedSpatialMethods.

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        sampler: str | None = None,
        gibbs_backend: str = "auto",
        thin: int = 1,
        n_jobs: int = -1,
        idata_kwargs: dict[str, Any] | None = None,
        **sample_kwargs,
    ) -> az.InferenceData:
        """Draw samples from the posterior for the panel model.

        Mirrors :meth:`SpatialModel.fit`: dispatches to this model's Gibbs
        sampler (``sampler="gibbs"``) or NUTS (``sampler="nuts"``).  When
        ``sampler`` is ``None`` (default), Gibbs is used if the model has a
        registered Gibbs sampler (Gaussian FE families), otherwise NUTS.  (The
        two ``fit`` bodies are duplicated across the cross-section and panel
        base classes until Phase 5c collapses the hierarchies.)

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup steps, and number of chains.
        random_seed : int, optional
            Seed for reproducibility.
        progressbar : bool, default True
            Show progress bar(s) during sampling.
        sampler : {"gibbs", "nuts", None}, default None
            Sampling method.  ``None`` auto-selects Gibbs when this model has
            one, else NUTS.
        gibbs_backend : {"auto", "jax", "numpy"}, default "auto"
            Execution backend for the Gibbs sampler.  ``"auto"`` uses JAX when
            installed and supported, otherwise NumPy.  Ignored for NUTS.
        thin : int, default 1
            Keep every ``thin``-th post-warmup Gibbs draw (Gibbs only).
        n_jobs : int, default -1
            Parallel workers for the NumPy Gibbs path (Gibbs only).
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` (NUTS only).  ``{"log_likelihood": True}``
            reconstructs the complete Jacobian-corrected pointwise
            log-likelihood.
        **sample_kwargs
            For NUTS, forwarded to ``pm.sample`` (``target_accept``,
            ``nuts_sampler="blackjax"``/``"numpyro"``/``"nutpie"``, ...).  For
            Gibbs, the family's declared options (``slice_width``, ...); an
            unsupported key raises.

        Returns
        -------
        arviz.InferenceData
            Posterior samples and diagnostics.
        """
        from ..samplers._registry import pop_options, resolve, resolve_backend

        entry = resolve(*self._gibbs_key) if self._gibbs_key is not None else None
        if sampler is None:
            sampler = "gibbs" if entry is not None else "nuts"

        if sampler == "gibbs":
            if entry is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no Gibbs sampler. "
                    "Use sampler='nuts' (the default)."
                )
            if self.robust and not entry.supports_robust:
                raise NotImplementedError(
                    "Gibbs sampling is not supported for robust (Student-t) "
                    "models. Use sampler='nuts'."
                )
            backend = resolve_backend(gibbs_backend, entry, jax_ok=jax_available())
            family_opts = pop_options(sample_kwargs, entry)
            self._idata = entry.run(
                self,
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                backend=backend,
                **family_opts,
            )
            return self._postprocess_idata(self._idata)

        if sampler != "nuts":
            raise ValueError(
                f"sampler must be 'gibbs', 'nuts', or None, got {sampler!r}"
            )

        idata = self._fit_nuts_and_reconstruct(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=progressbar,
            idata_kwargs=idata_kwargs,
            sample_kwargs=sample_kwargs,
        )
        return self._postprocess_idata(idata)

    def _fit_nuts_and_reconstruct(
        self,
        *,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int],
        progressbar: bool,
        idata_kwargs: dict[str, Any] | None,
        sample_kwargs: dict[str, Any],
    ) -> az.InferenceData:
        """Shared NUTS path for panel models: sample, then reconstruct log-lik.

        The Jacobian-corrected pointwise log-likelihood is reconstructed only
        for Gaussian spatial-lag/error panel families (``_likelihood ==
        "gaussian"`` with a ``rho``/``lam`` Jacobian).  OLS/SLX capture it
        natively and non-Gaussian panels (e.g. Tobit) reconstruct in their own
        ``fit``.
        """
        sample_kwargs = dict(sample_kwargs)
        target_accept = sample_kwargs.pop("target_accept", 0.9)
        idata_kwargs = dict(idata_kwargs or {})
        compute_log_likelihood = bool(idata_kwargs.get("log_likelihood", False))
        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")

        _, compute_log_likelihood = self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=compute_log_likelihood,
            sample_kwargs=sample_kwargs,
        )

        if (
            compute_log_likelihood
            and self._likelihood == "gaussian"
            and self._jacobian_param in {"rho", "lam"}
        ):
            self._reconstruct_panel_log_likelihood(
                spatial_param=self._jacobian_param,
                nuts_sampler=nuts_sampler,
                T_eff=self._T,
            )

        return self._idata

    def _fit_gibbs(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "numpy",
        slice_width: float | None = None,
        chain_method: str | None = None,
    ) -> az.InferenceData:
        """Sample a Gaussian FE panel posterior via 3-block Gaussian Gibbs.

        Generic over the FE families: resolves the Gibbs class from
        ``_gibbs_class``, builds the ``[X, WX]`` design when ``_has_wx_in_beta``,
        and passes ``T`` and the block-diagonal ``W_sparse_NT``.  SAR/SDM
        (``_jacobian_param == "rho"``) additionally pass ``Wy``.  Mirrors
        :meth:`SpatialModel._fit_gibbs` with panel-specific inputs.
        """
        if self._gibbs_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Gibbs sampling. "
                "Use sampler='nuts' (the default)."
            )
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
            )

        import importlib

        from ..samplers.gaussian import GaussianGibbsPriors

        gibbs_module = importlib.import_module(
            "..samplers.gaussian", package=__package__
        )
        GibbsClass = getattr(gibbs_module, self._gibbs_class)

        if self._has_wx_in_beta:
            Z = np.hstack([self._X, self._WX])
            feature_names = list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        else:
            Z = self._X
            feature_names = list(self._feature_names)

        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            Z, feature_names
        )
        priors = GaussianGibbsPriors(
            beta_mu=self.priors.get("beta_mu", default_beta_mu),
            beta_sigma=self.priors.get("beta_sigma", default_beta_sigma),
            sigma2_alpha=self.priors.get("sigma2_alpha", 2.0),
            sigma2_beta=self.priors.get("sigma2_beta", float(np.var(self._y))),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        from .._logdet._warmup import sampler_builds_evaluators

        method = self._logdet_bounds.method
        sampler_builds_logdet = sampler_builds_evaluators(
            method,
            self._W_sparse_NT is not None,
            self.logdet_refit,
            self.logdet_aaa_check,
        )

        gibbs_kwargs: dict[str, Any] = dict(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse_NT,
            priors=priors,
            # With the refit or the AAA node check on, the sampler builds its own
            # interpolant for warmup; building one here would be discarded.
            logdet_fn=None if sampler_builds_logdet else self._logdet_numpy_fn,
            logdet_vec_fn=None if sampler_builds_logdet else self._logdet_numpy_vec_fn,
            feature_names=feature_names,
            model_type=self._model_type,
            W_eigs=self._logdet_eigs,
            logdet_method=method,
            T=self._T,
            logdet_refit=self.logdet_refit,
            logdet_refit_pad_sd=self.logdet_refit_pad_sd,
            logdet_aaa_check=self.logdet_aaa_check,
        )
        # SAR/SDM need Wy; SEM/SDEM do not.
        if self._jacobian_param == "rho":
            gibbs_kwargs["Wy"] = self._Wy

        gibbs = GibbsClass(**gibbs_kwargs)

        self._idata = gibbs.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            thin=thin,
            n_jobs=n_jobs,
            progressbar=progressbar,
            gibbs_method=gibbs_method,
            slice_width=slice_width,
            chain_method=chain_method,
        )
        return self._idata

        # _fit_nuts inherited from SharedSpatialMethods.

    def _reconstruct_panel_log_likelihood(
        self,
        *,
        spatial_param: str,
        nuts_sampler: str,
        T_eff: int | None = None,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for static panel models.

        Delegates to :meth:`SharedSpatialMethods._reconstruct_gaussian_log_likelihood`
        with panel-specific spatial lag and random-effects handling.
        """
        T_mult = int(self._T if T_eff is None else T_eff)

        alpha_component = None
        if (
            hasattr(self, "_idata")
            and "alpha" in self._idata.posterior
            and hasattr(self, "_unit_idx")
        ):
            alpha_draws = self._idata.posterior["alpha"].values
            alpha_flat = alpha_draws.reshape(-1, alpha_draws.shape[-1])
            alpha_component = alpha_flat[:, np.asarray(self._unit_idx, dtype=np.int64)]

        self._reconstruct_gaussian_log_likelihood(
            spatial_param=spatial_param,
            nuts_sampler=nuts_sampler,
            spatial_lag_fn=lambda resid: self._batch_sparse_lag(resid, T_eff=T_mult),
            alpha_component=alpha_component,
        )

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        return self._pymc_model

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
            f"k={k}, model={self.model}, features={self._feature_names})"
        )
