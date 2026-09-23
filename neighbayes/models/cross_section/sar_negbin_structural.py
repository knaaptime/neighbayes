r"""Structural-form SAR Negative Binomial with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{NegBin}(\exp(\eta_i), \alpha), \quad
    \eta = \rho W \eta + X\beta + \nu, \quad
    \nu \sim N(0, \sigma^2 I)

Same observable likelihood as :class:`SARNegBin` (reduced form)
but sampled via Pólya–Gamma data augmentation and block Gibbs on the
structural form, yielding substantially higher ESS/s for n > ~1000.

Use this model when:
- You want the structural form with explicit ``sigma2`` noise.
- You need reliable ρ and α posteriors without long tuning.
- You want to compare with the NUTS path for validation.

Use :class:`SARNegBin` (canonical reduced form, PG-Gibbs) for
most spatial-econometric work, and :class:`SARNegBin` when
you need the full PyMC model graph for custom inference (NUTS).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import az
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._slice import SliceWidthState
from ...samplers._utils._sparsax_utils import resolve_pg_jax_backend
from ...samplers._utils._spatial_normal import CholmodFactor
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.negbin import GibbsCache, GibbsPriors, GibbsState, run_chain
from ...samplers.negbin._jax import run_chains_jax_vectorized
from ..base import SpatialModel
from ..priors import SARNegBinPriors


class SARNegBinStructural(SpatialModel):
    """Bayesian structural-form SAR-NB with Pólya–Gamma Gibbs sampler.

    Parameters
    ----------
    formula, data, y, X, W, priors, logdet_method
        Same interface as :class:`SARNegBin`.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The structural form parameterizes the latent log-mean as
    ``eta = rho * W @ eta + X @ beta + nu`` with ``nu ~ N(0, sigma^2 I)``,
    and augments the NB likelihood with Pólya–Gamma auxiliary variables
    to obtain fully conjugate Gibbs updates for eta, beta, and sigma^2.

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics (``spatial_diagnostics()``, ``spatial_effects()``,
    ``summary()``).

    The ``fit()`` method does **not** accept ``nuts_sampler`` or
    ``target_accept`` kwargs — these are NUTS-specific and will raise
    ``TypeError`` if passed.

    α (NB dispersion) mixing can be slower than ρ or β. Monitor ESS
    for α specifically and use longer runs if needed.
    """

    _priors_cls = SARNegBinPriors
    _likelihood: str = "count"
    _gibbs_key: tuple[str, str] | None = ("count_structural", "cross_section")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError(
                "robust=True is not supported for SARNegBinStructural."
            )

        # Validate y is integer and non-negative
        y_round = np.round(self._y).astype(np.int64)
        if not np.allclose(self._y, y_round):
            raise ValueError(
                "SARNegBinStructural requires integer-valued observations."
            )
        if np.any(y_round < 0):
            raise ValueError(
                "SARNegBinStructural requires non-negative integer observations."
            )
        self._y_int = y_round
        self._y = y_round.astype(np.float64)

        # Precompute logdet callable for the ρ slice sampler.
        # Small n: eigenvalue method (one-time O(n³), then O(n) per eval).
        # Large n: Chebyshev approximation (one-time O(nnz * order), then
        # O(order) per eval via Clenshaw recurrence).
        self._logdet_fn = self._logdet_numpy_fn

    # ------------------------------------------------------------------
    # Auto-selection for decoupled Gibbs path
    # ------------------------------------------------------------------

    # Threshold for JAX dense backend (O(n²) memory, competitive matvec).
    # Raised from 5000 to 10000: on a 128 GB machine the 800 MB dense
    # matrix is trivial; the real limit is the O(n³) Cholesky time
    # (~1 s per solve at n=10 000).
    _JAX_DENSE_THRESHOLD: int = 10000

    def _initialize_from_glm(self, rng):
        """Warm-start the Gibbs sampler from a non-spatial NB GLM fit.

        Returns a GibbsState with reasonable starting values for all
        parameters.
        """
        y = self._y
        X = self._X
        n, k = X.shape

        # Fit a simple NB GLM via method of moments for warm start
        # β₀: OLS on log(y + 0.5) as a rough estimate
        y_offset = np.log(np.maximum(y, 0.5))
        try:
            beta_init = np.linalg.lstsq(X, y_offset, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.zeros(k)

        # η₀: X @ β₀ (no spatial structure)
        eta_init = X @ beta_init

        # σ²₀: residual variance from OLS
        residuals = y_offset - eta_init
        sigma2_init = max(float(residuals @ residuals) / n, 0.01)

        # ρ₀: start at 0 (no spatial dependence)
        rho_init = 0.0

        # α₀: method of moments estimate
        mu_init = np.exp(eta_init)
        np.mean(mu_init)
        y_mean = np.mean(y)
        y_var = np.var(y)
        if y_var > y_mean and y_mean > 0:
            alpha_init = y_mean**2 / (y_var - y_mean)
            alpha_init = max(alpha_init, 0.1)
        else:
            alpha_init = 1.0

        # ω₀: draw from PG(y + α, η)
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(y + alpha_init, eta_init, rng=rng)

        return GibbsState(
            eta=eta_init,
            beta=beta_init,
            sigma2=sigma2_init,
            rho=rho_init,
            alpha=alpha_init,
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
            Thinning is for memory management, not statistical efficiency.
        n_jobs : int
            Number of parallel chains. -1 = all CPUs.
        progressbar : bool
            Show per-chain progress bars.
        backend : {"numpy", "jax"}
            Execution backend.  ``"numpy"`` uses the CHOLMOD ``factorize``
            path (the default — fastest on CPU for sparse W); ``"jax"`` uses
            the JAX-accelerated dense path (dense matvec + vmap over Lanczos
            probes and Chebyshev draws; requires float64; viable for
            n ≲ 10 000 on machines with ≥ 32 GB RAM).
        return_eta : bool
            If True, store the full latent field η in the posterior.
            Default False — η is n × draws × chains, which can be large.
            A scalar summary ||η||² is always stored.
        pg_n_terms : int, default 25
            Number of alternating-series terms for the JAX Pólya–Gamma
            sampler.  The draw's mean is exact at any K (a closed-form
            tail-mean correction is applied); higher values reduce the
            residual O(1/K³) tail-variance deficit at the cost of more
            compute.  Values below 20 can destabilize the Gibbs chain.
            Only used on the JAX path.
        n_probes : int, default 5
            Number of Lanczos probe vectors for stochastic log|P|
            estimation.  Only used on the JAX path.
        lanczos_deg : int, default 15
            Lanczos iteration depth for log|P| estimation.  Only used
            on the JAX path.

        Returns
        -------
        az.InferenceData
            With posterior, log_likelihood, and observed_data groups.
        """
        y = self._y
        X = self._X
        W_sparse = self._W_sparse
        n, k = X.shape

        if n < 900:
            warnings.warn(
                f"SAR Negative Binomial models require large samples for "
                f"reliable spatial parameter recovery. With n={n}, "
                f"posterior estimates of ρ and α may be severely "
                f"attenuated. n ≥ 900 is recommended.",
                UserWarning,
                stacklevel=2,
            )

        # Build priors
        priors = GibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            sigma2_alpha=self.priors.get("sigma2_alpha", 2.0),
            sigma2_beta=self.priors.get("sigma2_beta", 1.0),
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            rho_lower=self._logdet_bounds.rho_min,
            rho_upper=self._logdet_bounds.rho_max,
        )

        # Build cache
        XtX = X.T @ X

        # Create CHOLMOD factor for the precision matrix sparsity pattern.
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # The sparsity pattern is the union of diag, W+W^T, and W^T W,
        # which is the same for any ρ≠0.  Use a representative ρ to build
        # the pattern matrix (ρ=0 gives a diagonal matrix, which has the
        # wrong pattern and forces CHOLMOD to redo symbolic analysis each
        # call — very expensive).
        # Precompute matrix pieces for the precision expansion:
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # These are constant across the entire sampling run.
        # Division by σ² happens at runtime since σ² changes each iteration.
        W_sym = W_sparse + W_sparse.T
        WtW = W_sparse.T @ W_sparse

        # CHOLMOD factor for the precision matrix sparsity pattern.
        # P = I/σ² + diag(ω) - ρ*(W+W^T)/σ² + ρ²*W^T W/σ²
        # The sparsity pattern is the union of diag, W+W^T, and W^T W,
        # which is the same for any ρ≠0.  Use a representative ρ to build
        # the pattern matrix (ρ=0 gives a diagonal matrix, which has the
        # wrong pattern and forces CHOLMOD to redo symbolic analysis each
        # call — very expensive).
        #
        # We only build the pattern matrix here; the actual ``CholmodFactor``
        # is created *inside* each chain (see ``_run_one_chain`` below).
        # CHOLMOD's C handle does not survive pickling (``__setstate__``
        # rebuilds it on unpickle), so sharing one factor across joblib
        # workers gives no savings, and the concurrent symbolic-analysis
        # calls at unpickle time can deadlock on macOS + Accelerate.
        # Building per-chain isolates each chain's CHOLMOD state.
        # Any ρ≠0 gives the correct pattern; ρ=0.5 is arbitrary.
        _cholmod_pattern = sp.eye(n, format="csr") + 0.5 * W_sym + 0.25 * WtW

        # Map the resolved backend onto the sampler's solve/logdet/sample paths.
        # ``auto`` prefers exact CHOLMOD factorization (3× faster than jax_dense
        # at n=2500 on CPU); ``jax`` is the opt-in dense path.
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

        cache = GibbsCache(
            W_sparse=W_sparse,
            XtX=XtX,
            logdet_fn=self._logdet_fn,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            cholmod_factor=None,  # per-chain; built inside _run_one_chain
            W_sym_over_s2=W_sym,
            WtW_over_s2=WtW,
            solve_method=solve_method,
            logdet_P_method=logdet_P_method,
            sample_method=sample_method,
            krylov_degree=krylov_degree,
            krylov_dmax=krylov_dmax,
            W_sym_dense=W_sym_dense,
            WtW_dense=WtW_dense,
            W_eigs=None,  # JAX path uses logdet_jax instead
            logdet_jax=logdet_jax,
            rho_mode_update_freq=5,  # recompute mode every 5 sweeps during burn-in
            rho_mode_w_factor=2.0,
            rho_adaptive_width=True,
            rho_slice_width_state=None,  # per-chain; built inside _run_one_chain
        )

        # Derive per-chain seeds
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]

        # Define the per-chain function
        # When using the full-JIT JAX backend, delegate to run_chain_jax()
        # which composes all Gibbs blocks into a single @jax.jit function,
        # eliminating Python→JAX dispatch overhead (~30ms/call × 6 calls).
        _use_jax_full = sample_method in ("jax_dense", "cholmod_jax")

        # JAX dense path: the Gibbs step compiles once and the chains run in
        # parallel threads (see run_chains_chunked), avoiding joblib re-JIT cost.
        if _use_jax_full:
            if return_eta:
                raise NotImplementedError(
                    "return_eta=True is not supported with gibbs_backend='jax'. "
                    "Use gibbs_backend='numpy' if you need the full latent field stored."
                )
            chain_inits = [
                self._initialize_from_glm(np.random.default_rng(seed)) for seed in seeds
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
                init = self._initialize_from_glm(rng)
                # Per-chain CHOLMOD factor and slice-width state.
                # The C handle inside ``CholmodFactor`` does not survive
                # pickling (``__setstate__`` re-runs symbolic analysis on
                # unpickle), so building per-chain costs the same as the
                # parent build while avoiding concurrent CHOLMOD init in
                # joblib workers (which can deadlock on macOS+Accelerate).
                # ``SliceWidthState`` must also be per-chain so adaptive
                # widths from one chain don't pollute the next.
                chain_cache = cache._replace(
                    cholmod_factor=(
                        CholmodFactor(_cholmod_pattern)
                        if _cholmod_pattern is not None
                        else None
                    ),
                    rho_slice_width_state=SliceWidthState(w=0.2),
                )
                return run_chain(
                    y=y,
                    X=X,
                    W_sparse=W_sparse,
                    priors=priors,
                    cache=chain_cache,
                    init=init,
                    draws=draws,
                    tune=tune,
                    thin=thin,
                    return_eta=return_eta,
                    rng=rng,
                    chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                    progress_manager=progress_manager,
                    store_log_lik=log_likelihood,
                )

            # Run chains.  Non-JAX paths parallelize across chains when
            # the user requests multiple workers.
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
                model_type="sar_negbin",
            )

        # Assemble InferenceData
        # Stack chain results: each key has shape (n_keep, ...) per chain
        param_keys = ["rho", "sigma", "alpha"]
        if return_eta:
            param_keys.append("eta")

        posterior_samples = {}
        for key in param_keys:
            arrays = [c[key] for c in chain_results]
            # Shape: (chains, n_keep, ...) for scalar params
            # Shape: (chains, n_keep, k) for beta
            # Shape: (chains, n_keep, n) for eta
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
        """Not supported — SARNegBinStructural uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARNegBinStructural does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling, or use "
            "SARNegBin for NUTS-based inference."
        )

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        Returns the expected count E[y] = exp(eta) where eta is computed
        from the posterior mean of rho and beta via the reduced form:
        eta = (I - rho * W)^{-1} X @ beta.
        """
        self._require_fit()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        A_rho_inv = sp.linalg.spsolve(
            sp.eye(self._X.shape[0], format="csr") - rho * self._W_sparse,
            self._X @ beta,
        )
        return np.exp(A_rho_inv)

    def _compute_spatial_effects_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts on the log-mean scale for each draw."""
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
