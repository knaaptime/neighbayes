r"""Spatial autoregressive negative binomial: the latent SAR, or the reduced form.

The unmarked ``SARNegBin`` is the latent SAR (ADR 0004), which propagates the
negative binomial's own gamma disturbance through the multiplier
(:mod:`._latent_negbin`).  ``latent=False`` is the reduced form, the field's
spatial lag model of counts (Lambert, Brown and Florax 2010): spatial
dependence enters only through the mean propagator :math:`(I - \rho W)^{-1}`
and overdispersion through the dispersion :math:`\alpha`, no disturbance
propagating,

.. math::

    y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
    \mu = \exp(\eta), \qquad
    \eta = (I - \rho W)^{-1} X \beta.

Sampling the reduced form
-------------------------
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

Given time it is a panel (:mod:`..panel._count`), which takes random unit
effects inside the filter.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pytensor.tensor as pt
import scipy.sparse as sp

from ..._lazy_deps import pm, xr
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
    pg_information_ratio,
    warn_pg_information,
)
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._mixins._designs import CountDesigns
from ..base import SpatialModel
from ..panel._count import _CountPanel
from ..priors import PanelCountPriors, SARNegBinPriors
from ._latent_negbin import LatentNegBinMixin


class SARNegBin(CountDesigns, LatentNegBinMixin, _CountPanel, SpatialModel):
    r"""Bayesian SAR negative binomial: the latent SAR, or the reduced form with ``latent=False``.

    The latent SAR's multiplier propagates the negative binomial's own
    disturbance as well as the covariates:

    .. math::

        y_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
        (I - \rho W)\eta = X\beta + \varepsilon, \qquad
        \varepsilon_i = \log\gamma_i,\;
        \gamma_i \sim \mathrm{Gamma}(\alpha, \alpha),

    so that :math:`\rho = 0` is :class:`NegBin`.  It is sampled by an exact
    Gibbs sweep over the log-means (:mod:`neighbayes.samplers.negbin_latent`),
    on JAX by default; there is no NUTS path.  Its pointwise
    log-likelihood integrates each log-mean out given the rest, and its
    impacts are on the log-mean or, ``impacts(scale="count")``, the expected
    count.

    ``latent=False`` is the reduced form,

    .. math::

        y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \qquad
        \mu = \exp(\eta), \qquad
        \eta = (I - \rho W)^{-1} X \beta ,

    whose spatial lag acts on the log-mean only; ``y`` is not transformed, so
    the likelihood carries no :math:`\log|I - \rho W|` Jacobian.  Its
    ``fit()`` defaults to Pólya–Gamma Gibbs, and ``sampler="nuts"`` fits it in
    PyMC.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula, e.g. ``"y ~ x1 + x2"``.  Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Non-negative integer counts of shape ``(n,)`` (matrix mode).
    X : array-like or pandas.DataFrame, optional
        Design matrix (matrix mode).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights, ``n × n`` (``N × N`` in a panel), row-standardized.
    priors : dict, optional
        Override default priors: ``beta_mu``, ``beta_sigma`` (default Gelman
        et al. 2008 on the log scale), ``rho_lower``, ``rho_upper``,
        ``alpha_sigma``, ``alpha_nu`` (Half-Student-t on α) and
        ``alpha_fixed`` (hold α; a large value is essentially Poisson).  See
        :class:`~neighbayes.models.priors.SARNegBinPriors`; a panel's are
        :class:`~neighbayes.models.priors.PanelCountPriors`, which add the
        effects' ``unit_effect_mu``, ``unit_effect_sd_scale``,
        ``time_effect_mu`` and ``time_effect_sigma``.
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
    time_col, unit_col : str, optional
        Formula-mode panel: the period and unit columns of ``data``.  Given
        time, the model is a panel (:mod:`neighbayes.models.panel._count`),
        stacked by period, then unit, whose Gibbs sampler integrates β, the
        period effects and the unit effects out of every ρ update.
    N, T : int, optional
        Matrix-mode panel: the numbers of units and periods.
    unit_effects : {None, "random"}, default None
        A panel's random unit effects inside the filter, ``unit_effect``.  A
        cross-section takes none: one draw per observation is a disturbance,
        not a unit effect.
    time_effects : bool, default False
        A panel's fixed time effects, ``time_effect``.
    mundlak : bool, default False
        Add the unit means of the time-varying columns to a panel's design
        (inside the filter), absorbing a correlation between the unit effects
        and ``X``.  Needs unit effects.
    latent : bool, default True
        The latent SAR; ``False`` fits the reduced form.  As a panel the latent
        SAR takes random unit effects and period effects, both inside the
        filter with the covariates and each period's gamma disturbance; with
        period effects and a row-standardized ``W`` its target carries Lee and
        Yu's (2010) time-effects term, :math:`-T\log(1 - \rho)`, so that the
        period effects are flat on each period's log-mean level.
    """

    _DESIGNS = {
        "static": {},
        "latent": {"_gibbs_key": ("count_latent", "cross_section")},
        "panel": {"_priors_cls": PanelCountPriors, "_gibbs_key": ("count", "panel")},
    }
    # The latent SAR's log-mean impacts and coordinates are the reduced form's.
    _FALLBACK = {"latent": "static"}
    _latent_panel = True
    _latent_default = True

    _priors_cls = SARNegBinPriors
    _gibbs_key: tuple[str, str] | None = ("count", "cross_section")

    # ------------------------------------------------------------------
    # PyMC graph (reduced-form NUTS)
    # ------------------------------------------------------------------
    def _build_pymc_model_static(self) -> pm.Model:
        r"""Build the reduced-form PyMC model for NUTS sampling.

        The model is

        .. math::

            y_i \sim \mathrm{NegBin}(\mu_i, \alpha), \quad
            \mu = \exp(\eta), \quad
            \eta = (I - \rho W)^{-1} X \beta,

        No disturbance propagates: this is the reduced form.

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
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        alpha_sigma = self.priors.get("alpha_sigma", 2.5)
        alpha_nu = self.priors.get("alpha_nu", 3.0)

        with pm.Model(coords=self._model_coords()) as model:
            rho = pm.Uniform("rho", lower=rho_lower, upper=rho_upper)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sigma, dims="coefficient")
            alpha = nb_alpha_rv(nb_alpha_fixed(self.priors), alpha_nu, alpha_sigma)

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
    def _fit_gibbs_static(
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
        log_likelihood: bool = False,
    ) -> "xr.DataTree":
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
        xarray.DataTree
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

        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        priors = ReducedGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
            alpha_fixed=nb_alpha_fixed(self.priors),
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
            _init_solver = CachedSparseSolver([W_csc], n)
            _Xtilde_init = _init_solver.solve([-_rho_init_mle], X)
            _eta_init = _Xtilde_init @ _beta_init_mle
            _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
            _alpha_init_mle = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))

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
                store_log_lik=log_likelihood,
            )

            # Stack chains
            stacked = {
                "rho": np.stack([s["rho"] for s in chain_results], axis=0),
                "beta": np.stack([s["beta"] for s in chain_results], axis=0),
                "alpha": np.stack([s["alpha"] for s in chain_results], axis=0),
            }
            # The PG augmentation's tightness, at each chain's first kept draw.
            eye = sp.identity(n, format="csc")
            self._warn_pg_information(
                [
                    pg_information_ratio(
                        self._y,
                        sp.linalg.spsolve(
                            (eye - s["rho"][0] * self._W_sparse).tocsc(),
                            X @ s["beta"][0],
                        ),
                        float(s["alpha"][0]),
                    )
                    for s in chain_results
                ],
                stacked["alpha"],
            )
            log_lik = (
                np.stack([s["log_lik"] for s in chain_results], axis=0)
                if log_likelihood
                else None
            )

            idata = gibbs_to_inference_data(
                posterior_samples=stacked,
                log_likelihood={"obs": log_lik} if log_likelihood else None,
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
        _init_solver = CachedSparseSolver([W_csc], n)
        _Xtilde_init = _init_solver.solve([-_rho_init_mle], X)
        _eta_init = _Xtilde_init @ _beta_init_mle
        _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
        _alpha_init_mle = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))

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
                store_log_lik=log_likelihood,
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
        self._warn_pg_information(
            [s["pg_information_ratio"] for s in chain_samples], stacked["alpha"]
        )
        log_lik = (
            np.stack([s["log_lik"] for s in chain_samples], axis=0)
            if log_likelihood
            else None
        )

        idata = gibbs_to_inference_data(
            posterior_samples=stacked,
            log_likelihood={"obs": log_lik} if log_likelihood else None,
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

    def _warn_pg_information(self, ratios, alpha_draws) -> None:
        """Warn when the PG augmentation over-states the data (see the sampler)."""
        warn_pg_information(
            ratios, float(np.median(alpha_draws[:, 0])), float(np.mean(self._y))
        )

    # ------------------------------------------------------------------
    # Posterior-mean fitted values
    # ------------------------------------------------------------------
    def _fitted_mean_from_posterior_static(self) -> np.ndarray:
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
    # Impacts (log-mean and count scales)
    # ------------------------------------------------------------------

    def _compute_impacts_posterior_static(
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

    def _compute_count_scale_impacts_posterior(
        self,
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
        average off-diagonal sum, and their sum respectively.

        :math:`(I - \rho W)^{-1}X`, the multiplier's diagonal and its row sums
        are all functions of the scalar :math:`\rho`, computed exactly at a few
        values and interpolated across the draws
        (:class:`~neighbayes.samplers._utils._sparsax_utils.ResolventInterpolant`),
        so no draw factors anything.
        """
        from ...diagnostics.lmtests import _get_posterior_draws
        from ...samplers._utils._sparsax_utils import ResolventInterpolant

        idata = self.inference_data
        rho = _get_posterior_draws(idata, "rho")
        beta = _get_posterior_draws(idata, "beta")
        ni = self._nonintercept_indices
        F = ResolventInterpolant(
            self._W_sparse, rho, rhs=self._X, diag=True, row_sums=True
        )
        L = F.weights(rho)
        SX, diag, rows = F.values["rhs"], F.values["diag"], F.values["row_sums"]

        direct = np.empty((rho.size, len(ni)), dtype=np.float64)
        total = np.empty((rho.size, len(ni)), dtype=np.float64)
        for g in range(rho.size):
            eta = L[g] @ (SX @ beta[g])
            mu = np.exp(np.clip(eta, -50.0, 50.0))
            direct[g] = np.mean(mu * (L[g] @ diag)) * beta[g, ni]
            total[g] = np.mean(mu * (L[g] @ rows)) * beta[g, ni]
        return direct, total - direct, total

    def impacts(
        self,
        return_posterior_samples: bool = False,
        scale: str = "logmean",
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
            :math:`\mu = \exp(\eta)`, which also needs the diagonal of the
            spatial multiplier at each posterior draw.
        """
        if scale == "logmean":
            return super().impacts(return_posterior_samples=return_posterior_samples)
        if scale != "count":
            raise ValueError("scale must be either 'logmean' or 'count'.")
        if self._design == "latent":
            return self._latent_impacts(scale, return_posterior_samples)
        if self._design != "static":
            raise NotImplementedError(
                "Count-scale impacts are implemented for the cross-section only."
            )

        self._require_fit()
        return self._effects_frame(
            *self._compute_count_scale_impacts_posterior(),
            return_posterior_samples=return_posterior_samples,
            scale=scale,
        )
