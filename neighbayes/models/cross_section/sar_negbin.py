r"""Reduced-form spatial autoregressive Negative Binomial (SAR-NB) model.

This is the **canonical** Bayesian SAR-NB model in
:mod:`neighbayes`: spatial dependence enters purely through the mean
propagator :math:`(I - \rho W)^{-1}` and overdispersion is captured by
the Negative Binomial dispersion parameter :math:`\alpha`.  There is no
latent spatial random effect (no :math:`\sigma`).

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \mu = \exp(\eta), \qquad
    \eta = (I - \rho W)^{-1} X \beta.

This is the reduced form that the spatial-econometrics literature
(LeSage & Pace 2009) writes down by default and is the preferred
specification when there is no substantive reason to model an
additional unobserved spatially-smoothed latent field.

Sampling
--------
Two samplers are available:

* **Pólya-Gamma Gibbs** (default): call
  :meth:`fit() <fit>` or :meth:`fit(sampler="gibbs") <fit>`.
  Each sweep performs one Pólya-Gamma augmentation step, one
  conjugate-Gaussian update for :math:`\beta`, one 1-D adaptive
  slice sample for :math:`\rho`, and one 1-D slice sample for
  :math:`\log\alpha`.

* **NUTS**: call :meth:`fit(sampler="nuts") <fit>`.
  Builds a PyMC model with the reduced-form likelihood and the
  log-determinant Jacobian :math:`\log|I - \rho W|`, then samples
  with NUTS.

See also
--------
SARNegBinStructural
    P\u00f3lya-Gamma Gibbs sampler for the structural form (with
    :math:`\sigma`).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import az, pm
from ...samplers._utils._slice import SliceWidthState
from ...samplers.negbin_reduced import (  # noqa: F401 — import side-effect registers the Gibbs entry
    ReducedGibbsCache,
    ReducedGibbsPriors,
    ReducedGibbsState,
    run_chain,
)
from ...samplers.negbin_reduced._core import (
    _KRYLOV_DEGREE_DEFAULT,
    _KRYLOV_DMAX_DEFAULT,
)
from ..base import SpatialModel
from ..priors import SARNegBinPriors


class SARNegBin(SpatialModel):
    _priors_cls = SARNegBinPriors
    _likelihood: str = "count"
    _gibbs_key: tuple[str, str] | None = ("count", "cross_section")

    #: Maximum n for which the JAX dense backend is used automatically.
    _JAX_DENSE_THRESHOLD: int = 10000

    #: Threshold above which count-scale spatial effects use the sparse
    #: Hutchinson path instead of the eigendecomposition path.
    _COUNT_EFFECTS_EIGEN_MAX_N: int = 2000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError("robust=True is not supported for SARNegBin.")

        y_round = np.round(self._y).astype(np.int64)
        if not np.allclose(self._y, y_round):
            raise ValueError("SARNegBin requires integer-valued observations.")
        if np.any(y_round < 0):
            raise ValueError("SARNegBin requires non-negative integer observations.")

        self._y_int = y_round
        self._y = y_round.astype(np.float64)
        self._Wy = np.asarray(self._W_sparse @ self._y, dtype=np.float64)

    # ------------------------------------------------------------------
    # PyMC graph (reduced-form NUTS)
    # ------------------------------------------------------------------
    def _build_pymc_model(self) -> pm.Model:
        r"""Build the reduced-form PyMC model for NUTS sampling.

        The model is

        .. math::

            y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
            \mu = \exp(\eta), \quad
            \eta = (I - \rho W)^{-1} X \beta,

        No latent :math:`\sigma` or :math:`z` appears — this is the reduced
        form, not the structural form.

        **No log-determinant Jacobian.**  :math:`\log|I - \rho W|` belongs in
        the Gaussian SAR because there :math:`y` itself is transformed,
        :math:`(I - \rho W) y = X\beta + \varepsilon`, so the change of
        variables contributes :math:`|I - \rho W|`.  Here :math:`y` is not
        transformed at all: the spatial structure enters only through the
        mean, and the likelihood is the plain negative binomial pmf.  Adding
        the term anyway biases :math:`\rho` toward zero, because
        :math:`\log|I - \rho W|` decreases monotonically in :math:`\rho`.
        """
        bounds = self._logdet_bounds
        rho_lower = bounds.rho_min
        rho_upper = bounds.rho_max
        default_beta_mu, default_beta_sigma = self._gelman_default_beta_prior(
            self._X, list(self._feature_names)
        )
        beta_mu = self.priors.get("beta_mu", default_beta_mu)
        beta_sigma = self.priors.get("beta_sigma", default_beta_sigma)
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = pm.HalfStudentT("alpha", nu=alpha_nu, sigma=alpha_sigma)

            # Reduced form: η = (I - ρW)⁻¹ Xβ
            from ..._ops import SparseSARSolveOp

            _sar_solve_op = SparseSARSolveOp(self._W_sparse)
            Xbeta = pt.dot(self._X, beta)
            eta = _sar_solve_op(rho, Xbeta)
            mu = pm.Deterministic("mu", pt.exp(eta))
            pm.NegativeBinomial("obs", mu=mu, alpha=alpha, observed=self._y_int)

            # Deliberately no log|I - ρW| Potential here: y is not linearly
            # transformed, so there is no change of variables to correct for.
            # See the docstring.

        return model

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
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
        init_jitter: float = 0.1,
        slice_width: float = 0.4,
        krylov_degree: int = _KRYLOV_DEGREE_DEFAULT,
        krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
        n_rho_omega_cycles: int = 1,
        krylov_reuse: bool = True,
        timeout: float | None = None,
    ) -> "az.InferenceData":
        r"""Sample the reduced-form posterior via Pólya-Gamma block Gibbs.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup sweeps, and number of chains.
        random_seed : int, optional
            Seed for the per-chain RNGs.  Each chain receives a distinct
            child seed.
        thin : int, default 1
            Keep every ``thin``-th post-warmup draw.
        n_jobs : int, default 1
            Number of parallel worker processes.  ``1`` runs chains
            sequentially (recommended for small problems); ``-1`` uses all
            available CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.
        backend : {"numpy", "jax"}
            Execution backend.  ``"numpy"`` uses the CHOLMOD/SPLU
            factorization path with adaptive slice sampling for ρ (the
            default); ``"jax"`` uses the JAX-accelerated sparse path with
            slice+Krylov sampling (requires float64; viable for n ≲ 10 000).
        init_jitter : float, default 0.1
            Std-dev of the Gaussian jitter applied to the profile-loglik
            initial state.
        slice_width : float, default 0.4
            Initial stepping-out width for the ρ slice sampler (JAX path),
            adapted during warmup and fixed for the draws.
        krylov_degree : int, default 12
            Krylov basis degree for the shift-invert polynomial
            approximation of :math:`(I - \rho W)^{-1} X` inside the ρ-slice
            density.  Used by both backends.
        krylov_dmax : float, default 0.4
            Maximum :math:`|\Delta\rho|` for which the Krylov basis is used.
            Used by both backends.
        n_rho_omega_cycles : int, default 1
            Number of :math:`(\omega, \rho, \beta)` Gibbs cycles per sweep
            (NumPy path only).
        timeout : float or None, default None
            Maximum wall-clock seconds to wait for all chains to finish when
            ``n_jobs != 1``; ``None`` waits indefinitely.

        Returns
        -------
        arviz.InferenceData
            Posterior draws of ``rho``, ``beta``, ``alpha`` and pointwise
            ``log_likelihood`` for the observed counts.
        """
        from ...samplers._utils._idata import gibbs_to_inference_data
        from ...samplers.gaussian._chain_runner import run_chains

        n, k = self._X.shape

        if n < 900:
            warnings.warn(
                f"SAR Negative Binomial models require large samples for "
                f"reliable spatial parameter recovery. With n={n}, "
                f"posterior estimates of ρ and α may be severely "
                f"attenuated. n ≥ 900 is recommended.",
                UserWarning,
                stacklevel=2,
            )

        bounds = self._logdet_bounds
        rho_lower = float(bounds.rho_min)
        rho_upper = float(bounds.rho_max)

        priors = ReducedGibbsPriors(
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 1e6),
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )

        W_csr = self._W_sparse.tocsr()
        W_csc = self._W_sparse.tocsc()
        X = np.ascontiguousarray(self._X, dtype=np.float64)

        # ── JAX sparse path ──
        if backend == "jax":
            from ...samplers.negbin_reduced._jax import run_chains_jax_reduced

            rng = np.random.default_rng(random_seed)
            chain_seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]

            # Smart initialization (same as NumPy path)
            _log_y = np.log(self._y + 0.5)
            from ...samplers._utils._sparsax_utils import (
                CachedSparseSolver,
                profile_loglik_rho_grid,
            )

            _best_rho, _best_beta, _best_ll = profile_loglik_rho_grid(_log_y, X, W_csc)
            _rho_init_mle = float(
                np.clip(_best_rho, rho_lower + 0.05, rho_upper - 0.05)
            )
            _beta_init_mle = _best_beta
            try:
                _init_solver = CachedSparseSolver([W_csc], n)
                _Xtilde_init = _init_solver.solve([-_rho_init_mle], X)
                _eta_init = _Xtilde_init @ _beta_init_mle
                _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
                _alpha_init_mle = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))
            except Exception:
                _alpha_init_mle = 1.0

            # Detect intercept column
            intercept_col = -1
            for _j in range(k):
                if np.all(X[:, _j] == 1.0):
                    intercept_col = _j
                    break

            chain_inits = []
            for seed in chain_seeds:
                chain_rng = np.random.default_rng(seed)
                _rho_jitter = min(init_jitter, 0.02)
                beta_init = _beta_init_mle + init_jitter * chain_rng.standard_normal(k)
                rho_init = float(
                    np.clip(
                        _rho_init_mle + _rho_jitter * chain_rng.standard_normal(),
                        rho_lower + 0.01,
                        rho_upper - 0.01,
                    )
                )
                alpha_init = float(
                    np.clip(
                        _alpha_init_mle
                        * np.exp(init_jitter * chain_rng.standard_normal()),
                        0.05,
                        50.0,
                    )
                )
                omega_init = 0.25 * np.ones(n, dtype=np.float64)
                chain_inits.append(
                    ReducedGibbsState(
                        beta=beta_init,
                        rho=rho_init,
                        alpha=alpha_init,
                        omega=omega_init,
                    )
                )

            chain_results = run_chains_jax_reduced(
                y=self._y,
                X=X,
                W_sparse=self._W_sparse,
                priors=priors,
                inits=chain_inits,
                draws=draws,
                tune=tune,
                thin=thin,
                jax_seeds=chain_seeds,
                progressbar=progressbar,
                intercept_col=intercept_col,
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
                slice_width=slice_width,
                krylov_reuse=krylov_reuse,
            )

            # Stack chains
            stacked = {
                "rho": np.stack([s["rho"] for s in chain_results], axis=0),
                "beta": np.stack([s["beta"] for s in chain_results], axis=0),
                "alpha": np.stack([s["alpha"] for s in chain_results], axis=0),
            }
            log_lik = np.stack([s["log_lik"] for s in chain_results], axis=0)

            idata = gibbs_to_inference_data(
                posterior_samples=stacked,
                log_likelihood={"obs": log_lik},
                observed_data={"obs": self._y_int},
                coords={
                    "coefficient": list(self._feature_names),
                    "obs_id": list(range(n)),
                },
                dims={
                    "beta": ["coefficient"],
                    "obs": ["obs_id"],
                },
            )
            self._idata = idata
            return idata

        # ── NumPy / SciPy factorize path ──
        # For A_ρ = I − ρW: λ_min(A_ρ) = 1 − ρ·λ_max(W), λ_max(A_ρ) = 1 − ρ·λ_min(W).
        # Spectrum bounds for the solve path.  Deliberately *not* from
        # ``_W_eigs``: that densifies W for an O(n^3) eigendecomposition, and
        # only bounds are needed here.  See ``_W_spectral_bounds``.
        W_eig_max, W_eig_min = self._W_spectral_bounds

        # Precompute CHOLMOD pattern for the normal-equations matrix
        # A^T A = I − ρ(W+W^T) + ρ² W^T W  (SPD for any valid ρ).
        # When CHOLMOD is available, the sampler uses this instead of
        # ``splu`` (scipy SuperLU) to avoid Apple Accelerate BLAS deadlocks
        # on macOS under concurrent process access.
        #
        # IMPORTANT: We pass the *pattern matrix* (a sparse CSC matrix)
        # to the worker, NOT a CholmodFactor object.  Creating a
        # CholmodFactor in the parent process calls CHOLMOD/BLAS, which
        # accumulates state across many fit() calls and eventually
        # causes deadlocks on macOS.  The worker creates the
        # CholmodFactor from the pattern matrix on its side.
        from ...samplers.negbin_reduced._core import _make_cholmod_pattern

        W_sym, WtW, pattern = _make_cholmod_pattern(W_csc, n)
        cholmod_pattern = pattern

        rng = np.random.default_rng(random_seed)
        chain_seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]

        # --- Smart initialization ---
        # At high ρ, β and ρ are strongly correlated: the intercept
        # β₀ at ρ = 0.8 is much smaller than at ρ = 0 because the
        # spatial multiplier (I − ρW)⁻¹ amplifies Xβ.  Starting the
        # chain at (ρ ≈ 0, β ≈ 0) places it in a completely wrong
        # mode that the Gibbs sampler cannot escape.
        #
        # We use a profile-log-likelihood initialization on log(y+0.5):
        #   1. For each ρ on a coarse grid, compute
        #      X̃ = (I − ρW)⁻¹ X and OLS β̂ = (X̃ᵀX̃)⁻¹ X̃ᵀ log(y)
        #   2. Pick the (ρ, β) that maximizes the Gaussian log-lik
        #   3. Estimate α from method-of-moments on Pearson residuals
        _log_y = np.log(self._y + 0.5)
        from ...samplers._utils._sparsax_utils import (
            CachedSparseSolver,
            profile_loglik_rho_grid,
        )

        _best_rho, _best_beta, _best_ll = profile_loglik_rho_grid(_log_y, X, W_csc)
        _rho_init_mle = float(np.clip(_best_rho, rho_lower + 0.05, rho_upper - 0.05))
        _beta_init_mle = _best_beta
        # Estimate α from the Pearson dispersion of the Gaussian fit
        # on log(y).  The residual variance σ² from the profile
        # log-likelihood gives the dispersion on the log scale.
        # For NB data: Var(log y) ≈ 1/α + 1/(2μ) (delta method), so
        # α ≈ 1/σ² when μ is large.  Cap at a reasonable range.
        try:
            _init_solver = CachedSparseSolver([W_csc], n)
            _Xtilde_init = _init_solver.solve([-_rho_init_mle], X)
            _eta_init = _Xtilde_init @ _beta_init_mle
            _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
            _alpha_init_mle = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))
        except Exception:
            _alpha_init_mle = 1.0

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            chain_rng = np.random.default_rng(seed)
            # Jitter around the profile-loglik estimates.
            # Use a smaller jitter for ρ than for β because the
            # posterior is extremely peaked in ρ at high spatial
            # autocorrelation, and a large ρ jitter can push the
            # chain into a wrong mode that the Gibbs sampler
            # cannot escape.
            _rho_jitter = min(init_jitter, 0.02)
            beta_init = _beta_init_mle + init_jitter * chain_rng.standard_normal(k)
            rho_init = float(
                np.clip(
                    _rho_init_mle + _rho_jitter * chain_rng.standard_normal(),
                    rho_lower + 0.01,
                    rho_upper - 0.01,
                )
            )
            alpha_init = float(
                np.clip(
                    _alpha_init_mle * np.exp(init_jitter * chain_rng.standard_normal()),
                    0.05,
                    50.0,
                )
            )
            omega_init = 0.25 * np.ones(n, dtype=np.float64)
            init = ReducedGibbsState(
                beta=beta_init,
                rho=rho_init,
                alpha=alpha_init,
                omega=omega_init,
            )
            cache = ReducedGibbsCache(
                W_sparse=W_csr,
                W_csc=W_csc,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
                rho_adaptive_width=True,
                rho_slice_width_state=SliceWidthState(w=0.2),
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
                cholmod_pattern=cholmod_pattern,
                W_sym=W_sym,
                WtW=WtW,
                W_eig_max=W_eig_max,
                W_eig_min=W_eig_min,
                n_rho_omega_cycles=n_rho_omega_cycles,
                krylov_reuse=krylov_reuse,
            )
            return run_chain(
                y=self._y,
                X=X,
                W_sparse=W_csr,
                priors=priors,
                cache=cache,
                init=init,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=chain_rng,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
            )

        chain_samples = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=chain_seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=(n_jobs != 1),
            draws=draws,
            tune=tune,
            model_type="sar_negbin",
            timeout=timeout,
        )

        # Stack chains: each entry → (chains, draws, ...)
        stacked = {
            "rho": np.stack([s["rho"] for s in chain_samples], axis=0),
            "beta": np.stack([s["beta"] for s in chain_samples], axis=0),
            "alpha": np.stack([s["alpha"] for s in chain_samples], axis=0),
        }
        log_lik = np.stack([s["log_lik"] for s in chain_samples], axis=0)

        idata = gibbs_to_inference_data(
            posterior_samples=stacked,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": self._y_int},
            coords={
                "coefficient": list(self._feature_names),
                "obs_id": list(range(n)),
            },
            dims={
                "beta": ["coefficient"],
                "obs": ["obs_id"],
            },
        )
        self._idata = idata
        return idata

    # ------------------------------------------------------------------
    # Posterior-mean fitted values
    # ------------------------------------------------------------------
    def _fitted_mean_from_posterior(self) -> np.ndarray:
        r"""Posterior-mean fitted expected counts.

        Computes :math:`\exp(\eta)` where
        :math:`\eta = (I - \rho W)^{-1} X\beta` at the posterior means of
        :math:`\rho` and :math:`\beta`.  There is no :math:`\sigma z`
        term because the reduced form has no latent random effect.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        n = self._X.shape[0]
        A = sp.eye(n, format="csr", dtype=np.float64) - rho * self._W_sparse
        eta = sp.linalg.spsolve(A, self._X @ beta)
        return np.exp(eta)

    # ------------------------------------------------------------------
    # Spatial effects (log-mean and count scales)
    # ------------------------------------------------------------------

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

    def _compute_count_scale_spatial_effects_posterior(
        self,
        method: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute posterior impacts on the expected-count scale for each draw.

        Notes
        -----
        For the SAR-NB model with

        .. math::

            \mu = \exp\{(I - \rho W)^{-1} X\beta\},

        the average partial-effect matrix for covariate :math:`x_r` on the
        response scale is

        .. math::

            \frac{\partial \mu}{\partial x_r'} =
            \operatorname{diag}(\mu) (I - \rho W)^{-1} \beta_r.

        Direct, indirect, and total effects are the average diagonal, the
        average off-diagonal sum, and their sum respectively. This is more
        expensive than the log-mean-scale formula because it requires the
        diagonal of the spatial multiplier for each posterior draw.

        For n ≤ ``_COUNT_EFFECTS_EIGEN_MAX_N`` (default 2000), this uses
        the shared eigendecomposition cache (:attr:`_W_eigendecomposition`)
        to avoid per-draw sparse LU factorization, reducing complexity from
        :math:`O(\text{nnz}^{1.5})` per draw to :math:`O(n^2)` per draw.

        For n > ``_COUNT_EFFECTS_EIGEN_MAX_N``, this uses sparse solves
        with Hutchinson diagonal estimation, reducing memory from
        :math:`O(n^2)` to :math:`O(\text{nnz})` and avoiding the
        :math:`O(n^3)` eigendecomposition entirely.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data
        rho_draws = _get_posterior_draws(idata, "rho")
        beta_draws = _get_posterior_draws(idata, "beta")

        n = self._X.shape[0]
        ni = self._nonintercept_indices
        n_draws = rho_draws.shape[0]
        n_effects = len(ni)

        if method not in {"auto", "eigen", "sparse"}:
            raise ValueError(
                f"method must be one of {{'auto', 'eigen', 'sparse'}}, got {method!r}."
            )

        use_sparse = method == "sparse" or (
            method == "auto" and n > self._COUNT_EFFECTS_EIGEN_MAX_N
        )
        if use_sparse:
            # Sparse solves + Hutchinson diagonal estimation; avoids the
            # O(n³) eigendecomposition and O(n²) per-draw matmuls.
            return self._compute_count_scale_spatial_effects_posterior_sparse(
                rho_draws=rho_draws,
                beta_draws=beta_draws,
                n=n,
                ni=ni,
                n_draws=n_draws,
                n_effects=n_effects,
            )

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        # Use shared eigendecomposition cache (complex128 throughout).
        # Row-standardized W is generally non-symmetric, so V and Vinv
        # are complex.  Taking .real prematurely drops imaginary parts and
        # produces wrong results for eta, diag, and row sums.
        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        eigs_c = decomp[0]  # complex128, (n,)
        V_c = decomp[1]  # complex128, (n, n)
        Vinv_c = decomp[2]  # complex128, (n, n)

        # Precompute Vinv @ X (complex128, (n, k)) — reused for every draw
        VinvX = Vinv_c @ self._X.astype(np.complex128)  # (n, k)

        # Precompute Vinv @ 1 (complex128, (n,)) for row sums
        ones_c = np.ones(n, dtype=np.complex128)
        Vinv_ones = Vinv_c @ ones_c  # (n,)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            inv_eigs_c = 1.0 / (1.0 - float(rho) * eigs_c)  # complex128

            # eta = V @ diag(inv_eigs) @ Vinv @ X @ beta
            coeff = inv_eigs_c * (VinvX @ beta.astype(np.complex128))  # (n,)
            eta = (V_c @ coeff).real.astype(np.float64)  # (n,)
            mu = np.exp(np.clip(eta, -50.0, 50.0))

            # diag((I - rho W)^{-1}) = diag(V @ diag(inv_eigs) @ Vinv)
            # = sum_j V_{ij} * Vinv_{ji} / (1 - rho lambda_j)
            # (element-wise product of V and Vinv^T, weighted by inv_eigs)
            multiplier_diag = ((V_c * Vinv_c.T) @ inv_eigs_c).real.astype(
                np.float64
            )  # (n,)

            if self._is_row_std:
                multiplier_row_sums = self._multiplier_row_sums(rho)
            else:
                # row_sum_i = (V @ diag(inv_eigs) @ Vinv @ 1)_i
                multiplier_row_sums = (V_c @ (inv_eigs_c * Vinv_ones)).real.astype(
                    np.float64
                )  # (n,)

            direct_base = float(np.mean(mu * multiplier_diag))
            total_base = float(np.mean(mu * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    @staticmethod
    def _hutchinson_diag(
        A_solve: callable,
        n: int,
        n_probes: int = 20,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r"""Estimate the diagonal of :math:`A^{-1}` via Hutchinson's method.

        Given a callable ``A_solve(b)`` that returns :math:`A^{-1} b``, this
        estimates :math:`\operatorname{diag}(A^{-1})` using Rademacher
        (±1) probe vectors:

        .. math::

            \widehat{d_i} = \frac{1}{K} \sum_{k=1}^{K}
                z_{ki} \, [A^{-1} z_k]_i,

        where :math:`z_k \sim \operatorname{Rademacher}(\pm 1)`.
        With 20 probes the relative error is typically < 5%.

        Parameters
        ----------
        A_solve : callable
            Function that takes a vector or matrix ``b`` and returns
            :math:`A^{-1} b`.
        n : int
            Dimension of ``A``.
        n_probes : int, default 20
            Number of Rademacher probe vectors.
        rng : numpy.random.Generator, optional
            Random number generator.  If ``None``, a default generator
            with seed 42 is used for reproducibility.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Estimated diagonal of :math:`A^{-1}`.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        Z = rng.choice(np.array([-1.0, 1.0]), size=(n, n_probes))
        AinvZ = np.asarray(A_solve(Z), dtype=np.float64)
        return np.mean(Z * AinvZ, axis=1)

    def _compute_count_scale_spatial_effects_posterior_sparse(
        self,
        rho_draws: np.ndarray,
        beta_draws: np.ndarray,
        n: int,
        ni: list[int],
        n_draws: int,
        n_effects: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Count-scale spatial effects via sparse solves + Hutchinson.

        For large W where eigendecomposition is infeasible, this method uses:

        - A single sparse LU factorization of :math:`A = I - \rho W` per draw
          (KLU when available, SuperLU otherwise), reused for the
          :math:`\eta` solve, the Hutchinson probes, and the row-sum solve.
        - **Batched** matrix solve: the right-hand sides for :math:`\eta`,
          the optional row-sum vector :math:`\mathbf{1}`, and the
          Hutchinson probe matrix :math:`Z \in \{-1, +1\}^{n \times K}` are
          stacked into a single ``(n, 22)`` RHS and resolved with one
          ``solver.solve`` call per draw.
        - Hutchinson diagonal estimator for :math:`\operatorname{diag}(A^{-1})`.
        - Closed-form :math:`1/(1-\rho)` row sums for row-standardized :math:`W`,
          one extra sparse solve otherwise.

        Complexity is one LU factor plus a single batched triangular solve
        per draw.

        Parameters
        ----------
        rho_draws : np.ndarray, shape (G,)
        beta_draws : np.ndarray, shape (G, k)
        n : int
        ni : list[int]
        n_draws : int
        n_effects : int

        Returns
        -------
        direct_samples : np.ndarray, shape (G, n_effects)
        indirect_samples : np.ndarray, shape (G, n_effects)
        total_samples : np.ndarray, shape (G, n_effects)
        """
        from ..._ops import _make_cached_sparse_solver

        W = self._W_sparse
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        n_probes = 20

        # Pre-sample all Hutchinson probes up front so the per-draw RHS
        # assembly is purely vectorized.
        Z = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(n, n_probes),
        )

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            rho_f = float(rho)
            A = (I_n - rho_f * W).tocsc()

            # Factorize A once and reuse for all per-draw RHSes.
            solver = _make_cached_sparse_solver(A)
            if solver is None:
                solver = sp.linalg.splu(A)

            # Stack RHSes: [Xβ, ones (if needed), Z_1, ..., Z_K] → (n, 22 or 21).
            Xbeta = self._X @ beta
            if self._is_row_std:
                # Closed-form 1/(1-ρ) row sums; no need to solve A x = 1.
                rhs = np.empty((n, 1 + n_probes), dtype=np.float64)
                rhs[:, 0] = Xbeta
                rhs[:, 1:] = Z
                sol = np.asarray(solver.solve(rhs), dtype=np.float64)
                eta = sol[:, 0]
                AinvZ = sol[:, 1:]
                multiplier_row_sums = self._multiplier_row_sums(rho_f)
            else:
                rhs = np.empty((n, 2 + n_probes), dtype=np.float64)
                rhs[:, 0] = Xbeta
                rhs[:, 1] = ones
                rhs[:, 2:] = Z
                sol = np.asarray(solver.solve(rhs), dtype=np.float64)
                eta = sol[:, 0]
                multiplier_row_sums = sol[:, 1]
                AinvZ = sol[:, 2:]

            mu = np.exp(np.clip(eta, -50.0, 50.0))

            # Hutchinson: diag(A⁻¹) ≈ mean over probes of z ⊙ A⁻¹z.
            multiplier_diag = np.mean(Z * AinvZ, axis=1)

            direct_base = float(np.mean(mu * multiplier_diag))
            total_base = float(np.mean(mu * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def spatial_effects(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logmean",
        method: str = "auto",
    ):
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"logmean", "count"}, default "logmean"
            Scale on which impacts are reported.

            ``"logmean"`` returns the current default impacts on the linear
            predictor scale :math:`\log \mu`.

            ``"count"`` returns impacts on the expected-count scale
            :math:`\mu = \exp(\eta)`. This is exact but more expensive because
            it requires the diagonal of the spatial multiplier for each
            posterior draw.
        method : {"auto", "eigen", "sparse"}, default "auto"
            Only used when ``scale="count"``. ``"eigen"`` materializes the
            eigendecomposition of :math:`W` (fast for small :math:`n` but
            O(n³) memory/time); ``"sparse"`` uses one sparse LU per draw
            plus a Hutchinson diagonal estimator; ``"auto"`` picks sparse
            when :math:`n` exceeds
            :attr:`_COUNT_EFFECTS_EIGEN_MAX_N` (default 2000).
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        if scale == "logmean":
            return super().spatial_effects(
                return_posterior_samples=return_posterior_samples
            )
        if scale != "count":
            raise ValueError("scale must be either 'logmean' or 'count'.")

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_count_scale_spatial_effects_posterior(method=method)
        )

        k_effects = direct_samples.shape[1]
        if (
            hasattr(self, "_wx_feature_names")
            and len(self._wx_feature_names) == k_effects
        ):
            feature_names = list(self._wx_feature_names)
        elif (
            hasattr(self, "_nonintercept_feature_names")
            and len(self._nonintercept_feature_names) == k_effects
        ):
            feature_names = list(self._nonintercept_feature_names)
        else:
            feature_names = list(self._feature_names[:k_effects])

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=self.__class__.__name__,
        )
        df.attrs["scale"] = scale

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df
