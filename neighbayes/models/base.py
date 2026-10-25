"""Base class for Bayesian spatial regression models, cross-section or panel."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph

from .._backends.sampler_helpers import jax_available, select_backend
from .._lazy_deps import pm, xr
from .._logdet import (
    logdet_key,
    resolve_logdet_bounds,
)
from ._base._panel import (
    _EFFECTS_NAMES,
    _as_dense_W,
    _demean_panel,
    _resolve_effects,
)
from ._base._shared import (
    SharedSpatialMethods,
    resolve_W,
)
from ._base._structure import SpatialStructure


class SpatialModel(SharedSpatialMethods, ABC):
    """Base class for Bayesian spatial regression models. Models follow the notation
    of :cite:p:`anselin1988SpatialEconometrics` and :cite:p:`lesage2009IntroductionSpatial`.
    The API supports both formula and matrix input modes.

    A model is a panel when time is given (``time_col``, or ``T``); a
    cross-section is the ``T = 1`` case.  Panel data are stacked by period,
    then unit, and ``W`` acts within each period.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + rev_rating"``.
        If provided, ``data`` must also be supplied. An intercept is included
        by default; suppress with ``"y ~ x - 1"``.
    data : DataFrame or GeoDataFrame, optional
        Data source when using formula mode; long format for a panel.
    y : array-like, optional
        Dependent variable. Required in matrix mode; stacked ``(N*T,)`` for a
        panel.
    X : array-like, optional
        Predictor matrix. Required in matrix mode. If a DataFrame, column
        names are preserved for labelling.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights matrix of shape ``(N, N)`` (for a panel, also the full
        ``(N*T, N*T)`` block matrix). Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` to use the
        underlying sparse matrix, or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardized; a :class:`UserWarning` is raised if not.
        A cross-section may omit it, for specification tests only.
    unit_col, time_col : str, optional
        Columns of ``data`` identifying the unit and the period (formula-mode
        panels).
    N, T : int, optional
        Numbers of units and periods (matrix-mode panels).
    effects : str or int, default 0
        Panel fixed effects: ``"pooled"`` (or ``0``), ``"unit"`` (or ``1``),
        ``"time"`` (or ``2``), ``"two_way"`` (or ``3``).  The within
        transformation is applied to ``y`` and ``X`` before likelihood
        evaluation.
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : {"eigenvalue", "cheb", "aaa", "spline", "stochastic"}, optional
        How ``log|I - rho*W|`` is computed over rho: summed over the
        eigenvalues of ``W``; interpolated (Chebyshev, AAA rational, or cubic
        spline) through values from a sparse factorization, exact to the
        interpolant's tolerance; or expanded stochastically with no
        factorization, the one method with sampling error.  ``None`` (default)
        selects from the size and symmetry of ``W``: ``"eigenvalue"`` for
        ``n <= 500``; ``"aaa"`` up to ``n = 60000``
        (``NEIGHBAYES_LOGDET_FACTOR_MAX_N``) unless the factorization's fill-in
        would dominate; ``"stochastic"`` beyond.
    logdet_decomp : {"auto", "chol", "lu"}, default "auto"
        The factorization an interpolating method uses: sparse Cholesky
        (undirected ``W``; scikit-sparse CHOLMOD) or sparse LU (KLU or UMFPACK,
        chosen by the sparsity pattern).  ``"auto"`` takes ``"chol"`` where the
        method has it and ``W`` is undirected, else ``"lu"``.  Ignored by
        ``"eigenvalue"`` and ``"stochastic"``; ``"spline"`` on ``"chol"``
        raises.
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal,
        yielding a model that is robust to heavy-tailed outliers. When
        ``robust=True``, the errors follow a Student-t with **fixed**
        degrees of freedom :math:`\\nu`, set via ``priors={"nu": value}``
        (default 4, LeSage's ``rval``).  Fixing :math:`\\nu` is what keeps
        the NUTS and Gibbs paths targeting the same posterior.
    durbin_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for models that
        include ``WX`` terms (SLX, SDM, SDEM and their panel/Tobit variants).
        By default all non-constant columns are lagged. Pass a subset to
        restrict which variables receive a spatial lag, e.g.
        ``durbin_vars=["income", "density"]``.
    logdet_adapt : bool, default True
        Halfway through warmup, let the Gibbs sampler fit the log-determinant
        to where the chains have found the posterior rather than to the prior
        interval.  An interpolant on Cholesky or LU values (``"cheb"``,
        ``"aaa"``) is refit on that range, padded by ten warmup standard
        deviations; the window becomes the sampler's support, is recorded in
        ``idata.attrs["logdet_refit_window"]``, and a warning is raised if the
        retained draws reach an edge it introduced.  ``"aaa"`` also adds exact
        factorizations when the posterior lies near a singularity
        (``idata.attrs["logdet_aaa_nodes"]``).  ``"stochastic"`` grows its
        Hutchinson probes from 50 until the bias they leave in the posterior
        mean of the spatial parameter is expected to stay under half the
        run's own Monte Carlo error, ``1/sqrt(ESS)`` estimated in warmup: at
        most 1,000 probes up to 100,000 units, fewer beyond
        (``idata.attrs["logdet_probes"]``).
        ``False`` keeps the fit on the prior interval.

    Attributes
    ----------
    _spatial_params : tuple[str, ...]
        Spatial autoregressive parameters in the model (e.g. ``("rho",)``
        for SAR, ``("lam",)`` for SEM).  Empty for OLS and SLX.
    _lag_terms : tuple[str, ...]
        Lagged terms present in the model specification (e.g. ``("Wy",)``
        for SAR, ``("WX",)`` for SLX, ``("Wy", "WX")`` for SDM).
    _jacobian_param : str or None
        Name of the parameter that appears in the Jacobian determinant
        ``log|I - param * W|``.  ``"rho"`` for SAR/SDM, ``"lam"`` for
        SEM/SDEM, ``None`` for OLS/SLX (no Jacobian).
    _has_wx_in_beta : bool
        Whether the ``beta`` coefficient vector includes WX coefficients
        (i.e. whether the design matrix is ``[X, WX]`` rather than just
        ``X``).  True for SLX, SDM, SDEM.
    _gibbs_class : str or None
        Fully-qualified class name of the Gibbs sampler for this model
        (e.g. ``"GaussianSARGibbs"``), or ``None`` if no Gibbs sampler
        exists.  Used to look up the sampler at runtime to avoid circular
        imports.
    _model_type : str
        Short lowercase model name used as the ``model_type`` argument to
        the Gibbs sampler (e.g. ``"sar"``, ``"sdm"``).  Also used for
        DataTree coordinate labels.
    _is_panel : bool
        Whether the model was given time.  Flow models, which build
        themselves, declare it on the class.
    """

    # --- Declarative model metadata ----------------------------------------
    # Subclasses override these to declare their spatial structure.
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = ""
    _likelihood: str = ""
    _gibbs_key: tuple[str, str] | None = None
    _is_panel: bool = False

    # Emit a ResourceWarning before materializing very large dense panel
    # weight matrices. Tests may monkeypatch this value.
    _DENSE_W_WARN_BYTES: int = 100 * 1024 * 1024

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        *,
        unit_col: Optional[str] = None,
        time_col: Optional[str] = None,
        N: Optional[int] = None,
        T: Optional[int] = None,
        effects: Union[str, int] = 0,
        priors: Optional[Union[dict, Any]] = None,
        logdet_method: str | None = None,
        robust: bool = False,
        durbin_vars: Optional[list] = None,
        logdet_decomp: str = "auto",
        logdet_adapt: bool = True,
    ):
        from .priors import (
            BasePriors,
            PanelBasePriors,
            priors_as_dict,
            resolve_priors,
        )

        panel = time_col is not None or T is not None
        self._is_panel = panel
        # Panel decision trees use the "Panel-"-prefixed LM tests.
        self._panel_diagnostics = panel

        # Resolve typed priors (dataclass) and dict view.
        # A design may set its own priors class on the instance.
        default_priors = PanelBasePriors if panel else BasePriors
        _priors_cls = getattr(self, "_priors_cls", default_priors)
        self.priors_obj = resolve_priors(priors, _priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.logdet_method = logdet_method
        self.logdet_decomp = logdet_decomp
        self.logdet_adapt = bool(logdet_adapt)
        self.model = _resolve_effects(effects)
        self.effects = _EFFECTS_NAMES[self.model]
        self.robust = robust

        self._idata: Optional[xr.DataTree] = None
        self._pymc_model: Optional[pm.Model] = None

        if panel:
            y_arr, X_arr = self._parse_panel_data(
                formula, data, y, X, unit_col, time_col, N, T
            )
        else:
            y_arr, X_arr = self._parse_cross_section_data(formula, data, y, X)
        self._y_raw = y_arr
        self._X_raw = X_arr

        if W is None:
            # W-free mode (cross-section only): no spatial structure; the
            # specification tests require W to be supplied.
            if panel:
                raise ValueError("W is required.")
            if durbin_vars is not None:
                raise ValueError("durbin_vars requires a spatial weights matrix W.")
            self._W_sparse = None
            self._structure = None
            self._is_row_std = False
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            self._y, self._X = y_arr, X_arr
            self._Wy = np.zeros(len(y_arr), dtype=np.float64)
            self._WX = np.empty((X_arr.shape[0], 0), dtype=np.float64)
            return

        # Validate W and store it as N×N CSR; dense expansion is deferred.
        self._W_sparse, self._is_row_std = resolve_W(W, self._N, self._T)
        self._structure = SpatialStructure(self._W_sparse, self._N, self._T)
        # Resolve the logdet method and rho/lambda bounds exactly once, passing
        # W so auto-selection can honour graph directedness.  Eigenvalues stay
        # lazy (see the ``_logdet_eigs`` cached property) so init never pays
        # the O(n³) eigendecomposition for methods that do not need it.  For
        # row-standardized W the spectral stability interval is approximately
        # (-1, 1), so no eigenvalues are needed to resolve the bounds either.
        n_W = self._W_sparse.shape[0]
        self._logdet_bounds = resolve_logdet_bounds(
            logdet_key(self.logdet_method, self.logdet_decomp, W=self._W_sparse),
            n=n_W,
            priors=self.priors,
            W=self._W_sparse,
        )
        self._resolved_logdet_method = self._logdet_bounds.method

        self._wx_column_indices = self._spatial_lag_column_indices(
            X_arr, self._feature_names
        )
        if durbin_vars is not None:
            unknown = [v for v in durbin_vars if v not in self._feature_names]
            if unknown:
                raise ValueError(
                    f"durbin_vars contains names not found in X columns: {unknown}. "
                    f"Available: {self._feature_names}"
                )
            self._wx_column_indices = [
                i
                for i in self._wx_column_indices
                if self._feature_names[i] in durbin_vars
            ]
        self._wx_feature_names = [
            self._feature_names[i] for i in self._wx_column_indices
        ]

        if panel:
            self._y, self._X = _demean_panel(y_arr, X_arr, self._N, self._T, self.model)
        else:
            self._y, self._X = y_arr, X_arr

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
            # One batched sparse multiply across all WX columns.
            self._WX = self._sparse_panel_lag(self._X[:, self._wx_column_indices])
        else:
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)

    def _parse_cross_section_data(self, formula, data, y, X):
        """``(y, X)`` of a cross-section; sets the names, spec and ``N, T = n, 1``."""
        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            y_arr, X_arr, self._feature_names, self._model_spec = (
                self._parse_formula_with_spec(formula, data)
            )
        elif y is not None and X is not None:
            y_arr, X_arr, self._feature_names = self._parse_matrices(y, X)
            self._model_spec = None
        else:
            raise ValueError("Provide either (formula, data) or (y, X).")
        self._N, self._T = len(y_arr), 1
        self._panel_index = None
        return y_arr, X_arr

    def _parse_panel_data(self, formula, data, y, X, unit_col, time_col, N, T):
        """Stacked ``(y, X)`` of a balanced panel, sorted by period, then unit."""
        if formula is not None:
            if data is None:
                raise ValueError("data is required with formula mode.")
            if unit_col is None or time_col is None:
                raise ValueError("unit_col and time_col are required in formula mode.")
            d = data.sort_values([time_col, unit_col]).reset_index(drop=True)
            y_arr, X_arr, self._feature_names, self._model_spec = (
                self._parse_formula_with_spec(formula, d)
            )
            units = d[unit_col].nunique()
            times = d[time_col].nunique()
            if units * times != len(d):
                raise ValueError(
                    "Data are not a balanced panel after sorting by time/unit."
                )
            self._N, self._T = units, times
            self._panel_index = d[[time_col, unit_col]].copy()
        elif y is not None and X is not None:
            y_arr, X_arr, self._feature_names = self._parse_matrices(y, X)
            y_arr = y_arr.reshape(-1)
            self._model_spec = None
            if N is None or T is None:
                raise ValueError("N and T are required in matrix mode.")
            self._N, self._T = int(N), int(T)
            self._panel_index = None
            if self._N * self._T != len(y_arr):
                raise ValueError("N*T must equal number of stacked observations.")
        else:
            raise ValueError(
                "Provide either (formula,data,unit_col,time_col) or (y,X,N,T)."
            )
        return y_arr, X_arr

    # ------------------------------------------------------------------
    # The spatial structure
    # ------------------------------------------------------------------

    def _sparse_panel_lag(self, v: np.ndarray) -> np.ndarray:
        """Apply the spatial lag ``W ⊗ I_T`` to a stacked vector/matrix.

        Thin delegator to :meth:`SpatialStructure.spatial_lag` (kept as a
        method because samplers and diagnostics call ``model._sparse_panel_lag``).
        Accepts a 1-D stacked vector of length ``N*T`` or a 2-D ``(N*T, k)``
        matrix whose columns are all lagged in one batched sparse multiply.
        """
        return self._structure.spatial_lag(v)

    def _batch_sparse_lag(
        self,
        resid: np.ndarray,
        T_eff: int | None = None,
    ) -> np.ndarray:
        """Apply the spatial lag to a batch of stacked residual draws.

        Thin delegator to :meth:`SpatialStructure.batch_spatial_lag` (kept as a
        method because samplers call ``model._batch_sparse_lag``).  ``resid``
        has shape ``(n_draws, N*T_eff)`` (``T_eff`` defaults to ``T``; dynamic
        panels pass ``T-1``) and the result has the same shape.
        """
        return self._structure.batch_spatial_lag(resid, T_eff)

    @cached_property
    def _W_dense(self) -> np.ndarray:
        """Dense ``W`` (``(N*T)×(N*T)`` for a panel), materialized lazily."""
        if not self._is_panel:
            return np.asarray(self._W_sparse.toarray(), dtype=np.float64)
        # If W is N x N, the dense panel matrix is (N*T) x (N*T); otherwise the
        # caller supplied the full panel matrix already.
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
        return _as_dense_W(self._W_sparse, self._N, self._T)

    @property
    def _W_sparse_NT(self) -> "sp.csr_matrix":
        """Sparse ``(N*T)×(N*T)`` Kronecker-block weight matrix ``I_T ⊗ W_n``.

        ``W`` itself in a cross-section.  Delegates to
        :meth:`SpatialStructure.W_sparse_NT` (kept as a property because
        RE/dynamic samplers read ``model._W_sparse_NT``).
        """
        if self._structure is None:
            return self._W_sparse
        return self._structure.W_sparse_NT()

    @cached_property
    def _W_pt_sparse(self):
        """PyTensor sparse operator applied in the PyMC model, ``I_T ⊗ W``.

        Cached so repeated PyMC model builds reuse the same symbolic sparse
        operator and avoid the ``O(n²)`` dense materialization that
        ``pt.as_tensor_variable(self._W_dense)`` performs each time.

        Use with :func:`pytensor.sparse.structured_dot` (vector inputs must
        first be reshaped to ``(n, 1)`` because the vector overload's
        backward pass is broken in PyTensor).
        """
        from pytensor import sparse as _pts

        return _pts.as_sparse_variable(sp.csc_matrix(self._W_sparse_NT))

    # ------------------------------------------------------------------
    # Fixed-effects likelihood (Lee & Yu 2010)
    # ------------------------------------------------------------------
    #
    # Demeaning over T periods leaves N(T-1) independent observations, not
    # NT: the within-transformed errors are correlated within each unit.
    # Lee & Yu's transformation approach (an orthonormal F with F F' equal to
    # the demeaning projector) gives the exact likelihood of the transformed
    # data.  Its sum of squares equals the demeaned one, so only two
    # quantities change: the observations the variance counts and the
    # Jacobian multiplier.  With time effects and a row-standardized W the
    # transformation also removes the eigenvalue 1 - ρ of I - ρW, so
    #
    #   effects   observations     Jacobian
    #   unit      N(T-1)           (T-1)·log|I - ρW|
    #   time      (N-1)T           T·[log|I - ρW| - log(1 - ρ)]
    #   two-way   (N-1)(T-1)       (T-1)·[log|I - ρW| - log(1 - ρ)]
    #
    # The direct approach (all NT observations, T·log|I - ρW|) underestimates
    # σ² by (T-1)/T under unit effects.  Exact for Gaussian errors; with
    # ``robust=True`` the same dimensions are the best available correction.
    # Dynamic and Tobit panels set ``_lee_yu = False``: their fixed-T theory
    # (or likelihood) differs.  A cross-section has no effects, so it counts
    # every observation and applies the Jacobian once.

    _lee_yu: bool = True

    @property
    def _fe_unit(self) -> bool:
        return self._is_panel and self._lee_yu and self.model in (1, 3)

    @property
    def _fe_time(self) -> bool:
        return self._is_panel and self._lee_yu and self.model in (2, 3)

    @property
    def _jacobian_T(self) -> int:
        T = getattr(self, "_T", 1)
        return T - 1 if self._fe_unit else T

    @property
    def _jacobian_shift(self) -> float:
        if not self._fe_time or self._jacobian_param is None:
            return 0.0
        if not self._W_row_standardized():
            return 0.0
        return float(self._jacobian_T)

    @property
    def _n_effective(self) -> int:
        n = int(np.asarray(self._y).shape[0])
        if not (self._fe_unit or self._fe_time):
            return n
        units = n // self._T
        return (units - int(self._fe_time)) * (self._T - int(self._fe_unit))

    def _W_row_standardized(self) -> bool:
        """Whether W's rows sum to one; warns once when time effects need it."""
        if getattr(self, "_row_std_checked", None) is None:
            rs = np.asarray(self._W_sparse.sum(axis=1)).ravel()
            ok = bool(np.allclose(rs[rs != 0], 1.0))
            self._row_std_checked = ok
            if not ok:
                warnings.warn(
                    "Time fixed effects: the exact Lee & Yu (2010) Jacobian "
                    "term -log(1 - rho) requires a row-standardized W, which "
                    "this W is not; it is omitted (an O(1/N) approximation).",
                    UserWarning,
                    stacklevel=3,
                )
        return self._row_std_checked

    @property
    def _intercept_dropped(self) -> bool:
        """Whether the intercept column was dropped from the posterior beta.

        Checks the actual posterior beta dimension against the design
        matrix.  When Gibbs sampling is used with FE models, the
        intercept column (all zeros after demeaning) is dropped, so
        the posterior beta has fewer columns than the full design.
        NUTS samplers keep the full design matrix, so dimensions match.
        A cross-section never drops it.

        Returns True when the posterior beta has exactly the number of
        columns expected if the intercept was dropped.
        """
        if not self._is_panel:
            return False
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

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

    @abstractmethod
    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute direct, indirect, and total effects for each posterior draw.

        Returns
        -------
        tuple of np.ndarray
            ``(direct_samples, indirect_samples, total_samples)`` where each
            array has shape ``(G, k)`` or ``(G, k_wx)``, with *G* being the
            total number of posterior draws and *k* / *k_wx* being the
            number of covariates for which effects are reported.
        """

    # _fitted_mean_from_posterior: concrete default in SharedSpatialMethods
    # (dispatches on _jacobian_param and _has_wx_in_beta).  Subclasses
    # with random effects or dynamic terms override it.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float | None = None,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        sampler: str | None = None,
        backend: str | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        idata_kwargs: dict[str, Any] | None = None,
        **sample_kwargs,
    ) -> xr.DataTree:
        """Draw samples from the posterior.

        Dispatches to this model's Gibbs sampler (``sampler="gibbs"``) or NUTS
        (``sampler="nuts"``).  When ``sampler`` is ``None`` (default), Gibbs is
        used if the model has a registered Gibbs sampler, otherwise NUTS.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup steps, and number of chains.
        target_accept : float, optional
            Target acceptance rate for NUTS.  NUTS-only: passing it with the
            Gibbs sampler raises ``TypeError``.  Defaults to ``0.9`` for NUTS.
        random_seed : int, optional
            Seed for reproducibility.
        progressbar : bool, default True
            Show progress bar(s) during sampling.
        sampler : {"gibbs", "nuts", None}, default None
            Sampling method.  ``None`` auto-selects Gibbs when this model has
            one, else NUTS.
        backend : str, optional
            What executes the sampler.  For Gibbs, ``"auto"`` (the default),
            ``"jax"`` or ``"numpy"``; ``"auto"`` takes JAX when it is installed
            and supported by the family, else NumPy.  For NUTS, ``"pymc"`` (the
            default), ``"nutpie"``, ``"numpyro"`` or ``"blackjax"``.
        thin : int, default 1
            Keep every ``thin``-th post-warmup Gibbs draw (Gibbs only).
        n_jobs : int, default -1
            Parallel workers for the NumPy Gibbs path (Gibbs only).
        idata_kwargs : dict, optional
            ``{"log_likelihood": True}`` stores the complete Jacobian-corrected
            pointwise log-likelihood that ``az.loo`` /
            ``az.compare`` need, for Gibbs and NUTS alike.  Off by default, as
            in PyMC: it holds one value per draw, chain, and observation (16 GB
            at n = 250,000 with 4 × 2,000 draws).  For NUTS the dict is also
            passed to ``pm.sample``.
        **sample_kwargs
            For NUTS, forwarded to ``pm.sample``; for Gibbs, the family's
            declared options (an unsupported key raises).

        Returns
        -------
        xarray.DataTree
        """
        from ..samplers._registry import (
            pop_options,
            resolve,
            resolve_backend,
            run_entry,
        )

        gibbs_key = self._gibbs_key
        entry = resolve(*gibbs_key) if gibbs_key is not None else None
        if sampler is None:
            sampler = "gibbs" if entry is not None else "nuts"
        backend = select_backend(sampler, backend, sample_kwargs)

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
            if target_accept is not None:
                raise TypeError(
                    "target_accept is a NUTS-only argument and is not valid for "
                    "the Gibbs sampler (sampler='gibbs'). Remove it, or use "
                    "sampler='nuts'."
                )
            backend = resolve_backend(backend, entry, jax_ok=jax_available())
            family_opts = pop_options(sample_kwargs, entry)
            self._idata = run_entry(
                entry,
                self,
                log_likelihood=bool((idata_kwargs or {}).get("log_likelihood", False)),
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
            self._idata = self._postprocess_idata(self._idata)
            return self._idata

        nuts_sampler = backend
        compute_log_likelihood = bool((idata_kwargs or {}).get("log_likelihood", False))
        _, compute_log_likelihood = self._fit_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9 if target_accept is None else target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
            nuts_sampler=nuts_sampler,
            idata_kwargs=idata_kwargs,
            compute_log_likelihood=compute_log_likelihood,
            sample_kwargs=sample_kwargs,
        )
        # Gaussian spatial models capture only the Normal part natively; add
        # the Jacobian-corrected pointwise log-likelihood (Pattern A+J).  A
        # panel's OLS/SLX capture it natively, and non-Gaussian panels (Tobit)
        # rebuild it in ``_postprocess_idata``.
        if compute_log_likelihood and self._likelihood == "gaussian":
            if not self._is_panel:
                self._reconstruct_cross_sectional_log_likelihood(
                    nuts_sampler=nuts_sampler
                )
            elif self._jacobian_param in {"rho", "lam"}:
                self._reconstruct_panel_log_likelihood(
                    spatial_param=self._jacobian_param,
                    nuts_sampler=nuts_sampler,
                    T_eff=self._T,
                )
        self._idata = self._postprocess_idata(self._idata)
        return self._idata

        # _fit_nuts inherited from SharedSpatialMethods.

    def _reconstruct_cross_sectional_log_likelihood(
        self,
        *,
        nuts_sampler: str,
    ) -> None:
        """Rebuild complete pointwise log-likelihood for cross-sectional models.

        Delegates to :meth:`SharedSpatialMethods._reconstruct_gaussian_log_likelihood`.
        """
        self._reconstruct_gaussian_log_likelihood(
            spatial_param=self._jacobian_param,
            nuts_sampler=nuts_sampler,
        )

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
            and "unit_effect" in self._idata.posterior
            and hasattr(self, "_unit_idx")
        ):
            alpha_draws = self._idata.posterior["unit_effect"].values
            alpha_flat = alpha_draws.reshape(-1, alpha_draws.shape[-1])
            alpha_component = alpha_flat[:, np.asarray(self._unit_idx, dtype=np.int64)]

        self._reconstruct_gaussian_log_likelihood(
            spatial_param=spatial_param,
            nuts_sampler=nuts_sampler,
            spatial_lag_fn=lambda resid: self._batch_sparse_lag(resid, T_eff=T_mult),
            alpha_component=alpha_component,
        )

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
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Sample the posterior with the partially collapsed Gaussian Gibbs sampler.

        Uses the model's :attr:`_gibbs_class` attribute to resolve the
        appropriate Gibbs sampler class at runtime.  Only models with
        ``_gibbs_class is not None`` support Gibbs sampling; calling this
        method on OLS or SLX raises :exc:`NotImplementedError`.  A panel
        passes the block-diagonal ``W ⊗ I_T`` and its fixed-effects
        dimensions (Lee & Yu 2010; see :attr:`_jacobian_T`).

        Parameters
        ----------
        draws : int, default 2000
            Number of post-warmup draws per chain.
        tune : int, default 1000
            Number of warmup (burn-in) draws per chain.
        chains : int, default 4
            Number of independent chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int, default 1
            Keep every ``thin``-th draw after warmup.
        n_jobs : int, default -1
            Number of parallel workers for the NumPy path.  ``-1`` uses
            all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "numpy"
            Execution backend (the resolved ``backend``): ``"numpy"``
            for Python-loop Gibbs or ``"jax"`` for full-JIT Gibbs; both
            slice-sample ρ/λ.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.

        Returns
        -------
        xarray.DataTree
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.

        Raises
        ------
        NotImplementedError
            If the model has no Gibbs sampler (``_gibbs_class is None``).
        """
        if self._gibbs_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Gibbs sampling. "
                f"Use sampler='nuts' (the default)."
            )

        # --- Resolve Gibbs class (lazy import to avoid circular deps) ---
        import importlib

        from ..samplers.gaussian import GaussianGibbsPriors

        gibbs_module = importlib.import_module(
            "..samplers.gaussian", package=__package__
        )
        GibbsClass = getattr(gibbs_module, self._gibbs_class)

        # --- Build design matrix and feature names ---
        if self._has_wx_in_beta:
            Z = np.hstack([self._X, self._WX])  # (n, 2k)
            feature_names = list(self._feature_names) + [
                f"W*{name}" for name in self._wx_feature_names
            ]
        else:
            Z = self._X
            feature_names = list(self._feature_names)

        # --- Build priors ---
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

        # --- Build Gibbs sampler kwargs ---
        from .._logdet._warmup import sampler_builds_evaluators

        sampler_builds_logdet = sampler_builds_evaluators(
            self._logdet_bounds.method,
            self._W_sparse is not None,
            self.logdet_adapt,
            self.logdet_adapt,
            self.logdet_adapt,
        )
        gibbs_kwargs: dict[str, Any] = dict(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse_NT,
            priors=priors,
            # With the refit or the AAA node check on, the sampler builds its own
            # interpolant for warmup and replaces it partway through if needed.
            # Forcing these lazy properties here would build an interpolant on
            # the prior interval that nothing ever evaluates.
            logdet_fn=None if sampler_builds_logdet else self._logdet_numpy_fn,
            logdet_vec_fn=None if sampler_builds_logdet else self._logdet_numpy_vec_fn,
            feature_names=feature_names,
            model_type=self._model_type,
            W_eigs=self._logdet_eigs,
            logdet_method=self._logdet_bounds.method,
            logdet_adapt=self.logdet_adapt,
            T=self._T,
            # Fixed-effects dimensions (Lee & Yu 2010; see _jacobian_T).
            jacobian_T=self._jacobian_T,
            jacobian_shift=self._jacobian_shift,
            n_eff=self._n_effective,
        )
        # SAR/SDM need Wy; SEM/SDEM do not
        if self._jacobian_param == "rho":
            gibbs_kwargs["Wy"] = self._Wy

        if self.robust:
            # Student-t errors as a normal scale mixture with the same fixed ν
            # the NUTS path uses.
            gibbs_kwargs["nu"] = self._nu
        gibbs = GibbsClass(**gibbs_kwargs)

        self._fitted_probes = None
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
            log_likelihood=log_likelihood,
        )
        from .._logdet._probe_check import WarmupProbes

        if isinstance(gibbs.warmup_jacobian, WarmupProbes):
            self._fitted_probes = gibbs.warmup_jacobian
        return self._idata

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        For a Gibbs-fitted cross-section the PyMC model is not constructed
        during sampling; it is built lazily on first access so that downstream
        consumers (e.g. bridge sampling for marginal likelihoods) can evaluate
        ``logp`` and the prior under the same model definition used by the
        NUTS path.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        if self._is_panel:
            return self._pymc_model
        if self._pymc_model is None and self._idata is not None:
            try:
                self._pymc_model = self._build_pymc_model()
            except TypeError:
                self._pymc_model = self._build_pymc_model(nuts_sampler="pymc")
        return self._pymc_model

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n, k = self._X.shape
        if self._is_panel:
            return (
                f"{self.__class__.__name__}(N={self._N}, T={self._T}, n={n}, "
                f"k={k}, model={self.model}, features={self._feature_names})"
            )
        return (
            f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
        )
