r"""Reduced-form SAR-logit with Pólya–Gamma Gibbs sampler.

.. math::

    y_i \sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i)), \quad
    \eta = (I - \rho W)^{-1} X\beta

This is the canonical spatial binary model: the spatial lag enters the
*linear predictor* as a deterministic mean-propagator (there is **no**
latent noise field, so σ does not appear).  The ``|I − ρW|`` Jacobian
cancels when β is marginalized out, making the ρ conditional linear and
Krylov-accelerable.  The Pólya–Gamma augmentation yields fully conjugate
Gibbs updates for β and ρ (via a collapsed slice sampler).

Both backends fit the same model: ``gibbs_backend="jax"`` (the default via
``"auto"``) runs the chains in parallel threads, one ``jax.jit`` program each;
``"numpy"`` uses the CHOLMOD factorization path.  For the *structural*
latent-field SAR-logit, use :class:`SARLogitStructural`.

Use this model when:
- The response is binary (0/1).
- You need spatial autocorrelation in the log-odds.
- NUTS is slow or unreliable for the spatial parameter ρ.

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
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.logit import LogitGibbsPriors
from ...samplers.logit_reduced import (
    ReducedLogitGibbsState,
    run_chain,
    run_chains_jax_reduced_logit,
)
from ...samplers.negbin_reduced._core import (
    _KRYLOV_DEGREE_DEFAULT,
    _KRYLOV_DMAX_DEFAULT,
    ReducedGibbsCache,
    _make_cholmod_pattern,
)
from ..base import SpatialModel
from ..priors import SARLogitPriors, resolve_priors


class SARLogit(SpatialModel):
    """Bayesian reduced-form SAR-logit with Pólya–Gamma Gibbs sampler.

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
    The reduced form parameterizes the log-odds as the deterministic
    mean-propagator ``eta = (I - rho * W)^{-1} X @ beta`` (no latent noise
    field), and augments the logistic likelihood with Pólya–Gamma auxiliary
    variables to obtain fully conjugate Gibbs updates for β and a
    β-marginalized collapsed slice update for ρ (the ``|I − ρW|`` Jacobian
    cancels, so ρ is Krylov-accelerable).

    The sampler bypasses PyMC's NUTS entirely. It produces an
    ``arviz.InferenceData`` object compatible with all downstream
    diagnostics (``spatial_diagnostics()``, ``spatial_effects()``,
    ``summary()``).

    The ``fit()`` method does **not** accept ``nuts_sampler`` or
    ``target_accept`` kwargs — these are NUTS-specific and will raise
    ``TypeError`` if passed.

    Because the logit link absorbs the error scale, σ² is fixed at 1
    and does not appear in the posterior.  The PG shape parameter is
    always h = 1 (one trial per observation).
    """

    _spatial_params: tuple[str, ...] = ("rho",)
    _lag_terms: tuple[str, ...] = ("Wy",)
    _jacobian_param: str | None = "rho"
    _gibbs_class: str | None = None  # Gibbs-only, no NUTS
    _model_type: str = "sar_logit"
    _likelihood: str = "binary"
    _gibbs_key: tuple[str, str] | None = ("binary", "cross_section")
    _priors_cls = SARLogitPriors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.robust:
            raise NotImplementedError("robust=True is not supported for SARLogit.")

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
        """Warm-start the reduced-form Gibbs sampler from a spatial profile.

        For each ρ on a coarse grid, computes X̃ = (I − ρW)⁻¹X and the OLS
        estimate β̂ = (X̃ᵀX̃)⁻¹X̃ᵀy (linear probability proxy), then picks the
        (ρ, β) maximising the Gaussian log-likelihood on y.  Returns a
        :class:`ReducedLogitGibbsState` (β, ρ, ω) — no latent η field.
        """
        y = self._y
        X = self._X
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

        # Jitter around the profile-loglik estimates (smaller for ρ — the
        # posterior is extremely peaked in ρ at high spatial autocorrelation).
        _rho_jitter = 0.02
        beta_init = _best_beta + 0.1 * rng.standard_normal(k)
        rho_init = float(
            np.clip(
                _best_rho + _rho_jitter * rng.standard_normal(),
                self._logdet_bounds.rho_min + 0.01,
                self._logdet_bounds.rho_max - 0.01,
            )
        )

        # ω₀: draw from PG(1, η) at the profile η.
        try:
            _init_solver = CachedSparseSolver([W_csc], n)
            eta_init = _init_solver.solve([-rho_init], X @ beta_init)
        except Exception:
            eta_init = X @ beta_init
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)

        return ReducedLogitGibbsState(
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
        init_jitter: float = 0.1,
        slice_width: float = 0.4,
        krylov_degree: int = _KRYLOV_DEGREE_DEFAULT,
        krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
        krylov_reuse: bool = True,
        timeout: float | None = None,
    ) -> az.InferenceData:
        r"""Sample the reduced-form posterior via Pólya–Gamma block Gibbs.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup draws, and number of chains.
        random_seed : int or None
            Seed for reproducibility.
        thin : int
            Keep every ``thin``-th draw. Default 1 (no thinning).
        n_jobs : int
            Number of parallel chains (NumPy path). -1 = all CPUs.
        progressbar : bool
            Show per-chain progress bars.
        backend : {"numpy", "jax"}
            Execution backend.  ``"jax"`` (the default via ``"auto"``) runs
            the chains in parallel threads, one ``jax.jit`` program each; ``"numpy"``
            uses the CHOLMOD factorization path with adaptive slice sampling.
        init_jitter : float, default 0.1
            Std-dev of the Gaussian jitter applied to the profile-loglik
            initial state.
        slice_width : float, default 0.4
            Stepping-out width for the ρ slice sampler (JAX path).
        krylov_degree : int
            Krylov basis degree for the shift-invert polynomial
            approximation of :math:`(I - \rho W)^{-1} X` inside the ρ-slice
            density.  Used by both backends.
        krylov_dmax : float
            Maximum :math:`|\Delta\rho|` for which the Krylov basis is used.
        timeout : float or None, default None
            Maximum wall-clock seconds for the NumPy parallel chains.

        Returns
        -------
        az.InferenceData
            With posterior (``rho``, ``beta``), log_likelihood, and
            observed_data groups.
        """
        y = np.asarray(self._y, dtype=np.float64)
        X = np.ascontiguousarray(self._X, dtype=np.float64)
        n, k = X.shape

        bounds = self._logdet_bounds
        rho_lower = float(bounds.rho_min)
        rho_upper = float(bounds.rho_max)

        # Build priors from the typed priors object.
        priors_obj = resolve_priors(
            self.priors if isinstance(self.priors, dict) else None,
            SARLogitPriors,
        )
        if isinstance(self.priors, SARLogitPriors):
            priors_obj = self.priors

        priors = LogitGibbsPriors(
            beta_mu=priors_obj.beta_mu,
            beta_sigma=priors_obj.beta_sigma,
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )

        W_csr = self._W_sparse.tocsr()

        # Per-chain seeds.
        rng = np.random.default_rng(random_seed)
        chain_seeds = [int(s) for s in rng.integers(0, 2**31, size=chains)]

        # ── JAX path (chains in parallel threads) ──
        if backend == "jax":
            chain_inits = [
                self._initialize_from_ols(np.random.default_rng(s)) for s in chain_seeds
            ]
            intercept_col = -1
            for _j in range(k):
                if np.all(X[:, _j] == 1.0):
                    intercept_col = _j
                    break

            chain_results = run_chains_jax_reduced_logit(
                y=y,
                X=X,
                W_sparse=self._W_sparse,
                priors=priors,
                inits=chain_inits,
                draws=draws,
                tune=tune,
                thin=thin,
                intercept_col=intercept_col,
                krylov_degree=krylov_degree,
                krylov_dmax=krylov_dmax,
                slice_width=slice_width,
                jax_seeds=chain_seeds,
                progressbar=progressbar,
                krylov_reuse=krylov_reuse,
            )
        else:
            # ── NumPy / CHOLMOD path ──
            W_csc = self._W_sparse.tocsc()
            # Spectrum bounds for the solve path.  Deliberately *not* from
            # ``_W_eigs``: that densifies W for an O(n^3) eigendecomposition, and
            # only bounds are needed here.  See ``_W_spectral_bounds``.
            W_eig_max, W_eig_min = self._W_spectral_bounds
            W_sym, WtW, cholmod_pattern = _make_cholmod_pattern(W_csc, n)

            def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
                chain_rng = np.random.default_rng(seed)
                init = self._initialize_from_ols(chain_rng)
                cache = ReducedGibbsCache(
                    W_sparse=W_csr,
                    W_csc=W_csc,
                    rho_lower=rho_lower,
                    rho_upper=rho_upper,
                    rho_adaptive_width=True,
                    rho_slice_width_state=SliceWidthState(w=slice_width),
                    krylov_degree=krylov_degree,
                    krylov_dmax=krylov_dmax,
                    cholmod_pattern=cholmod_pattern,
                    W_sym=W_sym,
                    WtW=WtW,
                    W_eig_max=W_eig_max,
                    W_eig_min=W_eig_min,
                    n_rho_omega_cycles=1,
                    krylov_reuse=krylov_reuse,
                )
                return run_chain(
                    y=y,
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

            chain_results = run_chains(
                chain_fn=_run_one_chain,
                n_chains=chains,
                seeds=chain_seeds,
                n_jobs=n_jobs,
                progressbar=progressbar,
                parallel=(n_jobs != 1),
                draws=draws,
                tune=tune,
                model_type="sar_logit",
                timeout=timeout,
            )

        # Assemble InferenceData (varnames: rho, beta).
        posterior_samples = {
            "rho": np.stack([c["rho"] for c in chain_results], axis=0),
            "beta": np.stack([c["beta"] for c in chain_results], axis=0),
        }
        log_lik = np.stack([c["log_lik"] for c in chain_results], axis=0)

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": y},
            coords={"coefficient": list(self._feature_names)},
            dims={"beta": ["coefficient"]},
        )

        self._idata = idata
        return idata

    def _build_pymc_model(self):
        """Not supported — SARLogit uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARLogit does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def fitted_probabilities(self) -> np.ndarray:
        """Compute fitted probabilities at posterior mean parameters.

        Returns the probability P(y=1) = logit⁻¹(η) where η is computed
        from the posterior mean of ρ and β via the reduced form:
        η = (I - ρW)⁻¹ Xβ.

        Returns
        -------
        probs : ndarray of shape (n,)
            Fitted probabilities at posterior mean.
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
        """Compute fitted values at posterior mean parameters.

        For the logit model, the fitted mean is the fitted probability.
        """
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

    #: Threshold above which probability-scale spatial effects use the
    #: sparse Hutchinson path instead of the eigendecomposition path.
    #: See :attr:`SARNegBin._COUNT_EFFECTS_EIGEN_MAX_N` for the
    #: cost model — the logit case is identical structurally.
    _PROBABILITY_EFFECTS_EIGEN_MAX_N: int = 2000

    def _compute_probability_scale_spatial_effects_posterior(
        self,
        method: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute posterior impacts on the probability scale for each draw.

        Notes
        -----
        For the SAR-logit model with

        .. math::

            \eta = (I - \rho W)^{-1} X\beta, \qquad p = \sigma(\eta),

        the average partial-effect matrix for covariate :math:`x_r` on the
        response (probability) scale is

        .. math::

            \frac{\partial p}{\partial x_r'} =
            \operatorname{diag}\bigl(p \odot (1 - p)\bigr)
            (I - \rho W)^{-1} \beta_r,

        which matches LeSage & Pace (2009) and §3.6 of the spatial PG paper.
        Direct, indirect, and total effects are the average diagonal, the
        average row sum minus diagonal, and the average row sum of this
        matrix respectively.

        For ``n ≤ _PROBABILITY_EFFECTS_EIGEN_MAX_N`` (default 2000) this
        uses the shared eigendecomposition cache; otherwise it falls back
        to per-draw sparse LU + Hutchinson diagonal estimation.
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
            method == "auto" and n > self._PROBABILITY_EFFECTS_EIGEN_MAX_N
        )
        if use_sparse:
            return self._compute_probability_scale_spatial_effects_posterior_sparse(
                rho_draws=rho_draws,
                beta_draws=beta_draws,
                n=n,
                ni=ni,
                n_draws=n_draws,
                n_effects=n_effects,
            )

        direct_samples = np.empty((n_draws, n_effects), dtype=np.float64)
        total_samples = np.empty((n_draws, n_effects), dtype=np.float64)

        decomp = self._W_eigendecomposition
        if decomp is None:
            raise ValueError("No spatial weights matrix available.")
        eigs_c = decomp[0]
        V_c = decomp[1]
        Vinv_c = decomp[2]

        VinvX = Vinv_c @ self._X.astype(np.complex128)
        ones_c = np.ones(n, dtype=np.complex128)
        Vinv_ones = Vinv_c @ ones_c

        for draw_idx, (rho, beta) in enumerate(
            zip(rho_draws, beta_draws, strict=False)
        ):
            inv_eigs_c = 1.0 / (1.0 - float(rho) * eigs_c)

            coeff = inv_eigs_c * (VinvX @ beta.astype(np.complex128))
            eta = (V_c @ coeff).real.astype(np.float64)
            # Stable sigmoid; clip to avoid overflow in the rare extreme draw.
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0)))
            w = p * (1.0 - p)

            multiplier_diag = ((V_c * Vinv_c.T) @ inv_eigs_c).real.astype(np.float64)
            if self._is_row_std:
                multiplier_row_sums = self._multiplier_row_sums(rho)
            else:
                multiplier_row_sums = (V_c @ (inv_eigs_c * Vinv_ones)).real.astype(
                    np.float64
                )

            direct_base = float(np.mean(w * multiplier_diag))
            total_base = float(np.mean(w * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def _compute_probability_scale_spatial_effects_posterior_sparse(
        self,
        rho_draws: np.ndarray,
        beta_draws: np.ndarray,
        n: int,
        ni: list[int],
        n_draws: int,
        n_effects: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Probability-scale spatial effects via sparse solves + Hutchinson.

        Direct port of
        :meth:`SARNegBin._compute_count_scale_spatial_effects_posterior_sparse`
        with the count-mean weight :math:`\mu` replaced by the Bernoulli
        variance :math:`p(1-p)`.
        """
        from ..._ops import _make_cached_sparse_solver

        W = self._W_sparse
        I_n = sp.eye(n, format="csr", dtype=np.float64)
        ones = np.ones(n, dtype=np.float64)
        rng = np.random.default_rng(42)
        n_probes = 20

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

            # KLU reusable factor when available, else scipy SuperLU.
            solver = _make_cached_sparse_solver(A)
            if solver is None:
                solver = sp.linalg.splu(A)

            Xbeta = self._X @ beta
            if self._is_row_std:
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

            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -50.0, 50.0)))
            w = p * (1.0 - p)

            multiplier_diag = np.mean(Z * AinvZ, axis=1)

            direct_base = float(np.mean(w * multiplier_diag))
            total_base = float(np.mean(w * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def spatial_effects(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logodds",
        method: str = "auto",
    ):
        r"""Compute Bayesian inference for direct, indirect, and total impacts.

        Parameters
        ----------
        return_posterior_samples : bool, optional
            If ``True``, also return the posterior draws for each effect type.
        scale : {"logodds", "probability"}, default "logodds"
            Scale on which impacts are reported.

            ``"logodds"`` returns impacts on the linear-predictor (log-odds)
            scale :math:`\eta = (I - \rho W)^{-1} X\beta`. This is the
            cheap default; effects are linear in :math:`\beta` and do not
            require evaluating the link.

            ``"probability"`` returns response-scale impacts

            .. math::

                \partial p / \partial x_r =
                \operatorname{diag}(p(1-p))\,(I - \rho W)^{-1}\beta_r,

            which match the LeSage-Pace (2009) and PG-paper formulas. This
            is more expensive because it requires the diagonal of the
            spatial multiplier and the fitted probabilities for each draw.
        method : {"auto", "eigen", "sparse"}, default "auto"
            Only used when ``scale="probability"``. ``"eigen"`` uses the
            shared eigendecomposition cache (O(n²) per draw); ``"sparse"``
            uses one sparse LU per draw plus a Hutchinson diagonal
            estimator (O(nnz) per draw); ``"auto"`` picks sparse when
            :math:`n` exceeds :attr:`_PROBABILITY_EFFECTS_EIGEN_MAX_N`
            (default 2000).
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        if scale == "logodds":
            return super().spatial_effects(
                return_posterior_samples=return_posterior_samples
            )
        if scale != "probability":
            raise ValueError("scale must be either 'logodds' or 'probability'.")

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_probability_scale_spatial_effects_posterior(method=method)
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
