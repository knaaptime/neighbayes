"""Base class for Bayesian spatial regression models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from libpysal.graph import Graph

from .._backends.sampler_helpers import jax_available
from .._lazy_deps import az, pm
from .._logdet import (
    resolve_logdet_bounds,
)
from ._base._shared import (
    SharedSpatialMethods,
    _parse_W,
)
from ._base._structure import CrossSectionStructure


class SpatialModel(SharedSpatialMethods, ABC):
    """Base class for Bayesian spatial regression models. Models follow the notation
    of :cite:p:`anselin1988SpatialEconometrics` and :cite:p:`lesage2009IntroductionSpatial`.
    The API supports both formula and matrix input modes.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula string, e.g. ``"price ~ poverty + rev_rating"``.
        If provided, ``data`` must also be supplied. An intercept is included
        by default; suppress with ``"y ~ x - 1"``.
    data : DataFrame or GeoDataFrame, optional
        Data source when using formula mode.
    y : array-like, optional
        Dependent variable. Required in matrix mode.
    X : array-like, optional
        Predictor matrix. Required in matrix mode. If a DataFrame, column
        names are preserved for labelling.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights matrix of shape ``(n, n)``. Accepts a
        :class:`libpysal.graph.Graph` (the modern libpysal graph API) or any
        :class:`scipy.sparse` matrix.  The legacy :class:`libpysal.weights.W`
        object is **not** accepted directly; pass ``w.sparse`` to use the
        underlying sparse matrix, or convert with
        ``libpysal.graph.Graph.from_W(w)``.
        W should be row-standardized; a :class:`UserWarning` is raised if not.
    priors : dict, optional
        Override default priors. Keys depend on the model subclass; see
        each model's docstring for supported keys.
    logdet_method : str, optional
        How to compute ``log|I - rho*W|``.  ``None`` (default) auto-selects
        from the size and symmetry of ``W``:

        - ``"eigenvalue"`` for ``n <= 500``: exact; one ``O(n^3)``
          eigendecomposition, then ``O(n)`` per evaluation.
        - ``"cheb_cholesky"`` for symmetric ``W`` (undirected graph) with
          ``n <= 20000``: exact; sparse Cholesky at Chebyshev nodes with
          symbolic reuse, then ~1.3 us per rho via Clenshaw recurrence.
        - ``"aaa"`` for non-symmetric ``W`` (directed graph: KNN, travel
          time, flows) with ``n <= 20000``: exact; sparse LU on an adaptive
          coarse grid with AAA rational interpolation, ~5 us per rho.
        - ``"cheb_stochastic"`` for larger ``n``, where factorization
          fill-in gets expensive: stochastic Chebyshev expansion (Han et
          al. 2015), with probe information computed once and reused.

        Both cutoffs are settable via the ``NEIGHBAYES_LOGDET_EIGEN_MAX_N``
        and ``NEIGHBAYES_LOGDET_CHEB_MAX_N`` environment variables.
        ``"chebyshev"`` (Barry-Pace Hutchinson stochastic traces) and
        ``"slq"`` (stochastic Lanczos quadrature) are available as explicit
        opt-ins; both inject stochastic error into the log-density.  The
        resolved choice is recorded on ``_resolved_logdet_method``.
    robust : bool, default False
        If True, use a Student-t error distribution instead of Normal,
        yielding a model that is robust to heavy-tailed outliers. When
        ``robust=True``, the errors follow a Student-t with **fixed**
        degrees of freedom :math:`\\nu`, set via ``priors={"nu": value}``
        (default 4, LeSage's ``rval``).  Fixing :math:`\\nu` is what keeps
        the NUTS and Gibbs paths targeting the same posterior.
    w_vars : list of str, optional
        Names of X columns to spatially lag. Only relevant for models that
        include ``WX`` terms (SLX, SDM, SDEM and their panel/Tobit variants).
        By default all non-constant columns are lagged. Pass a subset to
        restrict which variables receive a spatial lag, e.g.
        ``w_vars=["income", "density"]``.

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
        InferenceData coordinate labels.
    """

    # --- Declarative model metadata ----------------------------------------
    # Subclasses override these to declare their spatial structure.
    _spatial_params: tuple[str, ...] = ()
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = None
    _has_wx_in_beta: bool = False
    _gibbs_class: str | None = None
    _model_type: str = ""

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        W: Optional[Union[Graph, sp.spmatrix]] = None,
        priors: Optional[Union[dict, Any]] = None,
        logdet_method: str | None = None,
        robust: bool = False,
        w_vars: Optional[list] = None,
        logdet_refit: bool = True,
        logdet_refit_pad_sd: float = 10.0,
        logdet_aaa_check: bool = True,
    ):
        # Resolve typed priors (dataclass) and dict view.
        from .priors import BasePriors, priors_as_dict, resolve_priors

        _priors_cls = getattr(self.__class__, "_priors_cls", BasePriors)
        self.priors_obj = resolve_priors(priors, _priors_cls)
        self.priors = priors_as_dict(self.priors_obj)
        self.logdet_method = logdet_method
        self.robust = robust
        self.logdet_refit = bool(logdet_refit)
        self.logdet_refit_pad_sd = float(logdet_refit_pad_sd)
        self.logdet_aaa_check = bool(logdet_aaa_check)

        self._idata: Optional[az.InferenceData] = None
        self._pymc_model: Optional[pm.Model] = None

        if formula is not None:
            if data is None:
                raise ValueError("data must be provided when using formula mode.")
            self._y, self._X, self._feature_names = self._parse_formula(formula, data)
        elif y is not None and X is not None:
            self._y, self._X, self._feature_names = self._parse_matrices(y, X)
        else:
            raise ValueError("Provide either (formula, data) or (y, X).")

        if W is not None:
            # Validate W and store as CSR sparse matrix.
            # Dense conversion is deferred to _W_dense (lazy property).
            self._W_sparse, self._is_row_std = _parse_W(W, len(self._y))
            self._structure = CrossSectionStructure(self._W_sparse)
            # Resolve the logdet method and rho/lambda bounds exactly once.
            # Eigenvalues stay lazy (see the ``_logdet_eigs`` cached property)
            # so init never pays the O(n³) eigendecomposition for methods that
            # do not need it.  For row-standardized W the spectral stability
            # interval is approximately (-1, 1), so no eigenvalues are needed
            # to resolve the bounds either.
            self._logdet_bounds = resolve_logdet_bounds(
                self.logdet_method,
                n=len(self._y),
                priors=self.priors,
                W=self._W_sparse,
            )
            self._resolved_logdet_method = self._logdet_bounds.method
            self._wx_column_indices = self._spatial_lag_column_indices(
                self._X, self._feature_names
            )
            if w_vars is not None:
                unknown = [v for v in w_vars if v not in self._feature_names]
                if unknown:
                    raise ValueError(
                        f"w_vars contains names not found in X columns: {unknown}. "
                        f"Available: {self._feature_names}"
                    )
                self._wx_column_indices = [
                    i
                    for i in self._wx_column_indices
                    if self._feature_names[i] in w_vars
                ]
            self._wx_feature_names = [
                self._feature_names[i] for i in self._wx_column_indices
            ]
            self._Wy: np.ndarray = np.asarray(
                self._W_sparse @ self._y, dtype=np.float64
            )
            if self._wx_column_indices:
                self._WX = np.asarray(
                    self._W_sparse @ self._X[:, self._wx_column_indices],
                    dtype=np.float64,
                )
            else:
                self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
        else:
            # W-free mode: no spatial structure; spec tests require W to be supplied.
            self._W_sparse = None
            self._structure = None
            self._is_row_std = False
            self._wx_column_indices: list[int] = []
            self._wx_feature_names: list[str] = []
            self._Wy = np.zeros(len(self._y), dtype=np.float64)
            self._WX = np.empty((self._X.shape[0], 0), dtype=np.float64)
            if w_vars is not None:
                raise ValueError("w_vars requires a spatial weights matrix W.")

    @cached_property
    def _W_dense(self) -> np.ndarray:
        """Dense weight matrix, materialized lazily on first access."""
        return np.asarray(self._W_sparse.toarray(), dtype=np.float64)

    @cached_property
    def _W_pt_sparse(self):
        """PyTensor sparse variable wrapping :attr:`_W_sparse`.

        Cached so repeated PyMC model builds reuse the same symbolic sparse
        operator and avoid the ``O(n²)`` dense materialization that
        ``pt.as_tensor_variable(self._W_dense)`` performs each time.

        Use with :func:`pytensor.sparse.structured_dot` (vector inputs must
        first be reshaped to ``(n, 1)`` because the vector overload's
        backward pass is broken in PyTensor).
        """
        import scipy.sparse as _sp
        from pytensor import sparse as _pts

        return _pts.as_sparse_variable(_sp.csc_matrix(self._W_sparse))

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_pymc_model(self) -> pm.Model:
        """Construct and return a pm.Model. Subclasses implement this."""

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
        gibbs_backend: str = "auto",
        thin: int = 1,
        n_jobs: int = -1,
        idata_kwargs: dict[str, Any] | None = None,
        **sample_kwargs,
    ) -> az.InferenceData:
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
        gibbs_backend : {"auto", "jax", "numpy"}, default "auto"
            Execution backend for the Gibbs sampler.  ``"auto"`` uses JAX when
            installed and supported by the family, else NumPy.  Ignored for NUTS.
        thin : int, default 1
            Keep every ``thin``-th post-warmup Gibbs draw (Gibbs only).
        n_jobs : int, default -1
            Parallel workers for the NumPy Gibbs path (Gibbs only).
        idata_kwargs : dict, optional
            Passed to ``pm.sample`` (NUTS only).  ``{"log_likelihood": True}``
            reconstructs the complete Jacobian-corrected pointwise
            log-likelihood.
        **sample_kwargs
            For NUTS, forwarded to ``pm.sample`` (``nuts_sampler=...``); for
            Gibbs, the family's declared options (an unsupported key raises).

        Returns
        -------
        arviz.InferenceData
        """
        from ..samplers._registry import pop_options, resolve, resolve_backend

        gibbs_key = getattr(self, "_gibbs_key", None)
        entry = resolve(*gibbs_key) if gibbs_key is not None else None
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
            if target_accept is not None:
                raise TypeError(
                    "target_accept is a NUTS-only argument and is not valid for "
                    "the Gibbs sampler (sampler='gibbs'). Remove it, or use "
                    "sampler='nuts'."
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
            self._idata = self._postprocess_idata(self._idata)
            return self._idata

        if sampler != "nuts":
            raise ValueError(
                f"sampler must be 'gibbs', 'nuts', or None, got {sampler!r}"
            )

        nuts_sampler = sample_kwargs.pop("nuts_sampler", "pymc")
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
        # the Jacobian-corrected pointwise log-likelihood (Pattern A+J).
        if compute_log_likelihood and getattr(self, "_likelihood", None) == "gaussian":
            self._reconstruct_cross_sectional_log_likelihood(nuts_sampler=nuts_sampler)
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
        mala_step_size: float = 0.05,
        use_mala: bool = True,
        use_slice: bool = True,
        slice_width: float | None = None,
        chain_method: str | None = None,
    ) -> az.InferenceData:
        """Sample posterior via 3-block Gaussian Gibbs.

        Uses the model's :attr:`_gibbs_class` attribute to resolve the
        appropriate Gibbs sampler class at runtime.  Only models with
        ``_gibbs_class is not None`` support Gibbs sampling; calling this
        method on OLS or SLX raises :exc:`NotImplementedError`.

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
            Execution backend: ``"numpy"`` for Python-loop Gibbs with
            adaptive slice sampling, or ``"jax"`` for full-JIT Gibbs
            with MALA for ρ/λ.
        mala_step_size : float, default 0.05
            Initial MALA step size for the JAX path.
        use_mala : bool, default True
            If True, use MALA for the ρ/λ update in the JAX path.
        use_slice : bool, default True
            If True, use slice sampling for the ρ/λ update.
        slice_width : float or None, default None
            Initial step-out width for slice sampling.
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.

        Returns
        -------
        arviz.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.

        Raises
        ------
        NotImplementedError
            If the model has no Gibbs sampler (``_gibbs_class is None``)
            or uses a robust (Student-t) likelihood.
        """
        if self._gibbs_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support Gibbs sampling. "
                f"Use sampler='nuts' (the default)."
            )
        if self.robust:
            raise NotImplementedError(
                "Gibbs sampling is not yet supported for robust (Student-t) "
                "models. Use sampler='nuts' (the default)."
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
            self.logdet_refit,
            self.logdet_aaa_check,
        )
        gibbs_kwargs: dict[str, Any] = dict(
            y=self._y,
            X=Z,
            W_sparse=self._W_sparse,
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
            logdet_refit=self.logdet_refit,
            logdet_refit_pad_sd=self.logdet_refit_pad_sd,
            logdet_aaa_check=self.logdet_aaa_check,
        )
        # SAR/SDM need Wy; SEM/SDEM do not
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

    @property
    def pymc_model(self) -> Optional[pm.Model]:
        """Return the PyMC model object built for the most recent fit.

        For Gibbs-fitted models the PyMC model is not constructed during
        sampling; it is built lazily on first access so that downstream
        consumers (e.g. bridge sampling for marginal likelihoods) can
        evaluate ``logp`` and the prior under the same model definition
        used by the NUTS path.

        Returns
        -------
        pymc.Model or None
            The model object used by :meth:`fit`, or ``None`` if the instance
            has not been fit yet.
        """
        if self._pymc_model is None and self._idata is not None:
            try:
                self._pymc_model = self._build_pymc_model()
            except TypeError:
                self._pymc_model = self._build_pymc_model(nuts_sampler="pymc")
        return self._pymc_model

    @abstractmethod
    def _compute_spatial_effects_posterior(
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
    # Internals
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n, k = self._X.shape
        return (
            f"{self.__class__.__name__}(n={n}, k={k}, features={self._feature_names})"
        )
