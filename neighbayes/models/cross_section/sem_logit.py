r"""Structural-form SEM-logit with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i)), \quad
    \eta = X\beta + u, \quad
    u = \lambda W u + \nu, \quad
    \nu \sim N(0, I)

The logit link absorbs the error scale, so σ² is fixed at 1 and does
not appear as a free parameter.  The Pólya–Gamma augmentation yields
fully conjugate Gibbs updates for η, β, and λ (via collapsed slice
sampling).

Use this model when:
- The response is binary (0/1).
- You need spatial autocorrelation in the error term (not the outcome).
- NUTS is slow or unreliable for the spatial parameter λ.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
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
    SEMLogitGibbsCache,
    SEMLogitGibbsPriors,
    SEMLogitGibbsState,
    run_chain_sem,
)
from ...samplers.logit._jax import (
    run_chains_jax_sem_vectorized,
)
from ..base import SpatialModel
from ..priors import SEMLogitPriors, resolve_priors


class SEMLogit(SpatialModel):
    """Bayesian structural-form SEM-logit with Pólya–Gamma Gibbs sampler.

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
    priors : dict or SEMLogitPriors, optional
        Override default priors. Supported keys:

        - ``lam_lower`` (float, default -0.999): Lower bound of the
          Uniform prior on :math:`\\lambda`.
        - ``lam_upper`` (float, default 0.999): Upper bound of the
          Uniform prior on :math:`\\lambda`.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for
          :math:`\\beta`.
        - ``beta_sigma`` (float, default 10.0): Normal prior std for
          :math:`\\beta`.

    logdet_method : str, optional
        How to compute :math:`\\log|I - \\lambda W|`. ``None`` (default)
        auto-selects based on ``n``.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The structural form parameterizes the latent log-odds as
    ``eta = X @ beta + u`` with ``u = lam * W @ u + nu``,
    ``nu ~ N(0, I)``, and augments the logistic likelihood with
    Pólya–Gamma auxiliary variables to obtain fully conjugate Gibbs
    updates for η and β.

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics.

    The ``fit()`` method does **not** accept ``nuts_sampler`` or
    ``target_accept`` kwargs — these are NUTS-specific and will raise
    ``TypeError`` if passed.

    Because the logit link absorbs the error scale, σ² is fixed at 1
    and does not appear in the posterior.  The PG shape parameter is
    always h = 1 (one trial per observation), so the Devroye method
    is valid and typically fastest.
    """

    _spatial_params: tuple[str, ...] = ("lam",)
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "lam"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sem_logit"
    _likelihood: str = "binary"
    _gibbs_key: tuple[str, str] | None = ("binary_structural", "cross_section")
    _priors_cls = SEMLogitPriors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError("robust=True is not supported for SEMLogit.")

        # Validate y is binary
        if not np.isin(self._y, [0.0, 1.0]).all():
            raise ValueError("y must be binary with values in {0, 1}.")

        # Precompute logdet callable for the λ slice sampler.
        self._logdet_fn = self._logdet_numpy_fn

    # ------------------------------------------------------------------
    # Auto-selection for Gibbs path
    # ------------------------------------------------------------------

    _JAX_DENSE_THRESHOLD: int = 10000

    def _initialize_from_ols(self, rng):
        """Warm-start the Gibbs sampler from a linear probability model.

        Fits OLS of y on X (treating y as continuous), then uses the
        fitted values as initial η.  Starts λ at 0 (no spatial
        dependence in the error).

        Returns a SEMLogitGibbsState with reasonable starting values.
        """
        y = self._y
        X = self._X
        n, k = X.shape

        # β₀: OLS on y (linear probability model)
        try:
            beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.zeros(k)

        # η₀: X @ β₀ (no spatial structure)
        eta_init = X @ beta_init

        # λ₀: start at 0 (no spatial dependence in error)
        lam_init = 0.0

        # ω₀: draw from PG(1, η)
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        return SEMLogitGibbsState(
            eta=eta_init,
            beta=beta_init,
            lam=lam_init,
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
            path (the default); ``"jax"`` uses the JAX-accelerated dense path
            (requires float64; viable for n ≲ 10 000).
        return_eta : bool
            If True, store the full latent field η in the posterior.
            Default False — η is n × draws × chains, which can be large.
        pg_n_terms : int, default 25
            Ignored (kept for API compatibility).  PG draws now use the
            exact sum-of-exponentials method which does not require
            truncation.  Only relevant on the JAX path.
        n_probes : int, default 5
            Number of Lanczos probe vectors for stochastic log|P|
            estimation.  Only used on the JAX path.
        lanczos_deg : int, default 15
            Lanczos iteration depth for log|P| estimation.  Only used
            on the JAX path.
        krylov_degree : int, default 12
            Krylov basis degree for the λ-slice factor-reuse path
            (JAX + sparsax, or NumPy + CHOLMOD).  Set 0 to disable.
        krylov_dmax : float, default 0.4
            Maximum |Δλ| for the Krylov basis reuse radius.

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
            SEMLogitPriors,
        )
        if isinstance(self.priors, SEMLogitPriors):
            priors_obj = self.priors

        priors = SEMLogitGibbsPriors(
            beta_mu=priors_obj.beta_mu,
            beta_sigma=priors_obj.beta_sigma,
            lam_lower=self._logdet_bounds.rho_min,
            lam_upper=self._logdet_bounds.rho_max,
        )

        # Build cache
        XtX = X.T @ X

        # Precompute matrix pieces for the precision expansion:
        # P = I + diag(ω) - λ*(W+W^T) + λ²*W^T W  (σ² = 1)
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

        cache = SEMLogitGibbsCache(
            W_sparse=W_sparse,
            XtX=XtX,
            logdet_fn=self._logdet_fn,
            lam_lower=priors.lam_lower,
            lam_upper=priors.lam_upper,
            cholmod_factor=cholmod_factor,
            W_sym=W_sym,
            WtW=WtW,
            solve_method=solve_method,
            logdet_P_method=logdet_P_method,
            sample_method=sample_method,
            krylov_degree=krylov_degree,
            krylov_dmax=krylov_dmax,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            logdet_jax=logdet_jax,
            lam_adaptive_width=True,
            lam_slice_width_state=SliceWidthState(w=0.2),
        )

        # Derive per-chain seeds
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]

        # Define the per-chain function
        _use_jax_full = sample_method in ("jax_dense", "cholmod_jax")

        # JAX dense path: the Gibbs step compiles once and the chains run in
        # parallel threads (see run_chains_chunked).
        if _use_jax_full:
            if return_eta:
                raise NotImplementedError(
                    "return_eta=True is not supported with gibbs_backend='jax'. "
                    "Use gibbs_backend='numpy' if you need the full latent field stored."
                )
            chain_inits = [
                self._initialize_from_ols(np.random.default_rng(seed)) for seed in seeds
            ]
            chain_results = run_chains_jax_sem_vectorized(
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
                return run_chain_sem(
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

            # Non-JAX paths parallelize across chains when the user
            # requests multiple workers.
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
                model_type="sem_logit",
            )

        # Assemble InferenceData
        param_keys = ["lam"]
        if return_eta:
            param_keys.append("eta")

        posterior_samples = {}
        for key in param_keys:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack([c["beta"] for c in chain_results], axis=0)

        # Feature names for coords
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

        # Log-likelihood: shape (chains, n_keep, n)
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
        """Not supported — SEMLogit uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SEMLogit does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def fitted_probabilities(self) -> np.ndarray:
        """Compute fitted probabilities at posterior mean parameters.

        Returns the probability P(y=1) = logit⁻¹(η) where η = Xβ
        (the SEM spatial error affects the variance of η, not the mean).

        Returns
        -------
        probs : ndarray of shape (n,)
            Fitted probabilities at posterior mean.
        """
        self._require_fit()
        beta = self._posterior_mean("beta")
        eta = self._X @ beta
        return 1.0 / (1.0 + np.exp(-eta))

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        For the logit model, the fitted mean is the fitted probability.
        """
        return self.fitted_probabilities()

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts for SEM-logit.

        For SEM, direct effects equal β and indirect effects are zero.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        beta_draws = _get_posterior_draws(idata, "beta")

        ni = self._nonintercept_indices
        direct_samples = beta_draws[:, ni]
        total_samples = beta_draws[:, ni]
        indirect_samples = total_samples - direct_samples  # zeros

        return direct_samples, indirect_samples, total_samples
