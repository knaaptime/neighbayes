r"""Structural-form SAR-logit with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i)), \quad
    \eta = \rho W \eta + X\beta + \nu, \quad
    \nu \sim N(0, I)

This is the *structural* (latent-field) SAR-logit: the spatial lag acts on the
latent log-odds :math:`\eta`, and an i.i.d. Gaussian innovation :math:`\nu`
enters through :math:`(I-\rho W)^{-1}`.  The logit link absorbs the error scale
(σ² = 1).  Contrast with the canonical reduced-form :class:`SARLogit`, where the
spatial lag is a deterministic mean-propagator (no latent field) — that is the
default and the model with the richer probability-scale impacts.

Use this model when you specifically want the latent spatially-smoothed
log-odds field rather than the reduced-form mean propagator.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import az
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._slice import SliceWidthState
from ...samplers._utils._sparsax_utils import resolve_pg_jax_backend
from ...samplers._utils._spatial_normal import CholmodFactor
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.logit import (
    LogitGibbsCache,
    LogitGibbsPriors,
    LogitGibbsState,
    run_chain,
)
from ...samplers.logit._jax import run_chains_jax_vectorized
from ..base import SpatialModel
from ..priors import SARLogitPriors, resolve_priors


class SARLogitStructural(SpatialModel):
    """Bayesian structural-form SAR-logit with Pólya–Gamma Gibbs sampler.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``. Requires
        ``data``. An intercept is included by default; suppress with
        ``"y ~ x - 1"``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Binary dependent variable of shape ``(n,)``. Required in matrix
        mode. Must contain only 0 and 1.
    X : array-like or pandas.DataFrame, optional
        Design matrix. Required in matrix mode.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights of shape ``(n, n)``.
    priors : dict or SARLogitPriors, optional
        Override default priors. Supported keys:

        - ``rho_lower`` (float, default -0.999): Lower bound of the
          Uniform prior on :math:`\\rho`.
        - ``rho_upper`` (float, default 0.999): Upper bound of the
          Uniform prior on :math:`\\rho`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 10.0): Normal prior std for
          :math:`\\beta`.

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\rho W|`. ``None`` (default)
        auto-selects based on ``n``.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The structural form parameterizes the latent log-odds as
    ``eta = rho * W @ eta + X @ beta + nu`` with ``nu ~ N(0, I)``,
    and augments the logistic likelihood with Pólya–Gamma auxiliary
    variables to obtain fully conjugate Gibbs updates for η and β.

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics.  Impacts are reported on the log-odds scale; for
    probability-scale impacts use the reduced-form :class:`SARLogit`.
    """

    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sar_logit_structural"
    _likelihood: str = "binary"
    _gibbs_key: tuple[str, str] | None = ("binary_structural", "cross_section")
    _priors_cls = SARLogitPriors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError(
                "robust=True is not supported for SARLogitStructural."
            )

        # Validate y is binary
        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")

        # Precompute logdet callable for the ρ slice sampler.
        self._logdet_fn = self._logdet_numpy_fn

    # ------------------------------------------------------------------
    # Auto-selection for Gibbs path
    # ------------------------------------------------------------------

    _JAX_DENSE_THRESHOLD: int = 10000

    def _initialize_from_ols(self, rng):
        """Warm-start the Gibbs sampler from a spatial profile likelihood.

        For each ρ on a coarse grid, computes X̃ = (I − ρW)⁻¹X and
        the OLS estimate β̂ = (X̃ᵀX̃)⁻¹X̃ᵀy, then picks the (ρ, β)
        that maximizes the Gaussian log-likelihood on y (treating y as
        continuous).  This places the chain near the posterior mode even
        at high ρ, where starting at ρ = 0 can leave the chain stuck in
        a wrong mode.

        Falls back to a simple OLS on X (ρ = 0) if the grid search
        fails for all ρ values.
        """
        y = self._y
        X = self._X
        self._W_sparse.tocsr()
        W_csc = self._W_sparse.tocsc()
        n, k = X.shape

        # --- Profile-log-likelihood initialization ---
        # Cached sparse solver: A = I - ρW shares its sparsity pattern across
        # the grid, so the symbolic analysis is computed once (sparsax) or
        # the pattern is pre-assembled (scipy fallback).
        from ...samplers._utils._sparsax_utils import (
            CachedSparseSolver,
            profile_loglik_rho_grid,
        )

        _best_rho, _best_beta, _best_ll = profile_loglik_rho_grid(y, X, W_csc)

        _rho_jitter = 0.02
        beta_init = _best_beta + 0.1 * rng.standard_normal(k)
        rho_init = float(
            np.clip(
                _best_rho + _rho_jitter * rng.standard_normal(),
                self._logdet_bounds.rho_min + 0.01,
                self._logdet_bounds.rho_max - 0.01,
            )
        )

        # η₀: (I − ρ₀W)⁻¹Xβ₀ — spatially structured starting values
        try:
            _init_solver = CachedSparseSolver([W_csc], n)
            eta_init = _init_solver.solve([-rho_init], X @ beta_init)
        except Exception:
            eta_init = X @ beta_init

        # ω₀: draw from PG(1, η)
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        return LogitGibbsState(
            eta=eta_init,
            beta=beta_init,
            rho=rho_init,
            omega=omega_init,
        )

    def _fit_gibbs(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        backend: str = "numpy",
        return_eta: bool = False,
        pg_n_terms: int = 25,
        n_probes: int = 5,
        lanczos_deg: int = 15,
        krylov_degree: int = 0,
        krylov_dmax: float = 0.4,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
        """Sample posterior via Pólya–Gamma block Gibbs.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup draws, and number of chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int
            Keep every ``thin``-th draw. Default 1 (no thinning).
        n_jobs : int
            Number of parallel chains. -1 = all CPUs.
        progressbar : bool
            Show per-chain progress bars.
        backend : {"numpy", "jax"}
            Execution backend.  ``"numpy"`` uses the CHOLMOD factorization
            path; ``"jax"`` uses the JAX-accelerated dense path (requires
            float64; viable for n ≲ 10 000).
        return_eta : bool
            If True, store the full latent field η in the posterior.
            Default False — η is n × draws × chains, which can be large.
        pg_n_terms : int, default 25
            Ignored (kept for API compatibility).  Only relevant on the
            JAX path.
        n_probes : int, default 5
            Number of Lanczos probe vectors for stochastic log|P|
            estimation.  Only used on the JAX path.
        lanczos_deg : int, default 15
            Lanczos iteration depth for log|P| estimation.  Only used
            on the JAX path.
        krylov_degree : int, default 12
            Krylov basis degree for the ρ-slice factor-reuse path
            (JAX + sparsax, or NumPy + CHOLMOD).  Set 0 to disable.
        krylov_dmax : float, default 0.4
            Maximum |Δρ| for the Krylov basis reuse radius.

        Returns
        -------
        az.InferenceData
            With posterior, log_likelihood, and observed_data groups.
        """
        y = self._y
        X = self._X
        W_sparse = self._W_sparse
        n, k = X.shape

        # Build priors from the typed priors object
        priors_obj = resolve_priors(
            self.priors if isinstance(self.priors, dict) else None,
            SARLogitPriors,
        )
        if isinstance(self.priors, SARLogitPriors):
            priors_obj = self.priors

        priors = LogitGibbsPriors(
            beta_mu=priors_obj.beta_mu,
            beta_sigma=priors_obj.beta_sigma,
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # Build cache
        XtX = X.T @ X

        # Precompute matrix pieces for the precision expansion:
        # P = I + diag(ω) - ρ*(W+W^T) + ρ²*W^T W  (σ² = 1)
        W_sym = W_sparse + W_sparse.T
        WtW = W_sparse.T @ W_sparse

        # Create CHOLMOD factor for the precision matrix sparsity pattern.
        _P0 = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW
        cholmod_factor = CholmodFactor(_P0)

        # Map the resolved backend onto the sampler's solve/logdet/sample paths.
        method, _jax_parts = resolve_pg_jax_backend(
            backend,
            W_sparse=W_sparse,
            W_sym=W_sym,
            WtW=WtW,
            n=n,
            logdet_bounds=self._logdet_bounds,
        )
        solve_method = logdet_P_method = sample_method = method
        W_sym_dense = _jax_parts["W_sym_dense"]
        WtW_dense = _jax_parts["WtW_dense"]
        logdet_jax = _jax_parts["logdet_jax"]
        sparsax_pattern = _jax_parts["sparsax_pattern"]

        cache = LogitGibbsCache(
            W_sparse=W_sparse,
            XtX=XtX,
            logdet_fn=self._logdet_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            cholmod_factor=cholmod_factor,
            W_sym=W_sym,
            WtW=WtW,
            WtX=np.asarray(W_sparse.T @ X, dtype=np.float64),
            solve_method=solve_method,
            logdet_P_method=logdet_P_method,
            sample_method=sample_method,
            krylov_degree=krylov_degree,
            krylov_dmax=krylov_dmax,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            rho_adaptive_width=True,
            rho_slice_width_state=SliceWidthState(w=0.2),
        )

        # Derive per-chain seeds
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]

        # Define the per-chain function
        _use_jax_full = sample_method in ("jax_dense", "cholmod_jax")

        if _use_jax_full:
            if return_eta:
                raise NotImplementedError(
                    "return_eta=True is not supported with gibbs_backend='jax'. "
                    "Use gibbs_backend='numpy' if you need the full latent field "
                    "stored."
                )
            chain_inits = [
                self._initialize_from_ols(np.random.default_rng(seed)) for seed in seeds
            ]
            chain_results = run_chains_jax_vectorized(
                y=y,
                X=X,
                W_sparse=W_sparse,
                W_sym_dense=W_sym_dense,
                WtW_dense=WtW_dense,
                logdet_jax=logdet_jax,
                priors=priors,
                inits=chain_inits,
                draws=draws,
                tune=tune,
                thin=thin,
                jax_seeds=seeds,
                pg_n_terms=pg_n_terms,
                n_probes=n_probes,
                lanczos_deg=lanczos_deg,
                progressbar=progressbar,
                sparsax_pattern=sparsax_pattern,
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
                store_log_lik=log_likelihood,
            )
        else:

            def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
                rng = np.random.default_rng(seed)
                init = self._initialize_from_ols(rng)
                progress_chain_id = chain_id if chain_id_kw is None else chain_id_kw
                return run_chain(
                    y=y,
                    X=X,
                    W_sparse=W_sparse,
                    priors=priors,
                    cache=cache,
                    init=init,
                    draws=draws,
                    tune=tune,
                    thin=thin,
                    return_eta=return_eta,
                    rng=rng,
                    progress_manager=progress_manager,
                    chain_id=progress_chain_id,
                    store_log_lik=log_likelihood,
                )

            parallel = n_jobs != 1
            chain_results = run_chains(
                chain_fn=_run_one_chain,
                n_chains=chains,
                seeds=seeds,
                n_jobs=n_jobs,
                progressbar=progressbar,
                parallel=parallel,
                draws=draws,
                tune=tune,
                model_type="sar_logit_structural",
            )

        # Assemble InferenceData
        param_keys = ["rho"]
        if return_eta:
            param_keys.append("eta")

        posterior_samples = {}
        for key in param_keys:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)

        posterior_samples["beta"] = np.stack([c["beta"] for c in chain_results], axis=0)

        feature_names = list(self._feature_names)
        coords = {
            "coefficient": feature_names,
        }
        dims = {
            "beta": ["coefficient"],
        }
        if return_eta:
            coords["obs_id"] = list(range(n))
            dims["eta"] = ["obs_id"]

        log_lik = (
            np.stack([c["log_lik"] for c in chain_results], axis=0)
            if log_likelihood
            else None
        )

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik} if log_likelihood else None,
            observed_data={"obs": y},
            coords=coords,
            dims=dims,
        )

        self._idata = idata
        return idata

    def _build_pymc_model(self):
        """Not supported — SARLogitStructural uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARLogitStructural does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def fitted_probabilities(self) -> np.ndarray:
        """Compute fitted probabilities at posterior mean parameters.

        Returns P(y=1) = logit⁻¹(η) where η = (I − ρW)⁻¹ Xβ at the
        posterior mean of ρ and β.
        """
        self._require_fit()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        A_rho_inv = sp.linalg.spsolve(
            sp.eye(self._X.shape[0], format="csr") - rho * self._W_sparse,
            self._X @ beta,
        )
        return 1.0 / (1.0 + np.exp(-A_rho_inv))

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """For the logit model, the fitted mean is the fitted probability."""
        return self.fitted_probabilities()

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts on the log-odds scale for each draw."""
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        mean_diag = self._batch_mean_diag(rho_draws)
        mean_row_sum = self._batch_mean_row_sum(rho_draws)

        ni = self._nonintercept_indices
        direct_samples = mean_diag[:, None] * beta_draws[:, ni]
        total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
        indirect_samples = total_samples - direct_samples

        return direct_samples, indirect_samples, total_samples
