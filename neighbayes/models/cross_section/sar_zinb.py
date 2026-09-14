r"""Zero-inflated SAR Negative Binomial (ZINB-SAR) model.

Composes a reduced-form SAR-logit selection equation with a
reduced-form SAR-NB count equation via a zero-allocation block:

.. math::

    d_i \mid \eta_i^{\mathrm{sel}} &\sim \mathrm{Bernoulli}(\mathrm{logit}^{-1}(\eta_i^{\mathrm{sel}})) \\
    \eta^{\mathrm{sel}} &= (I - \lambda W_{\mathrm{sel}})^{-1} Z\gamma \\
    y_i \mid d_i, \eta_i^{\mathrm{cnt}}, \alpha &\sim \begin{cases}
        0 & \text{if } d_i = 0 \\
        \mathrm{NegBin}(\exp(\eta_i^{\mathrm{cnt}}), \alpha) & \text{if } d_i = 1
    \end{cases} \\
    \eta^{\mathrm{cnt}} &= (I - \rho W_{\mathrm{cnt}})^{-1} X\beta

Both equations are reduced-form: the spatial lag enters each linear
predictor as a deterministic mean-propagator (no latent noise field), so
the ``|I − λW|`` / ``|I − ρW|`` Jacobians cancel under marginalization.
The logit link fixes σ² = 1 in the selection equation.  The Pólya–Gamma
augmentation yields fully conjugate Gibbs updates for all blocks except
ρ, λ, and α, which use 1-D adaptive slice sampling.  The NumPy and JAX
backends fit the same model block-for-block.

Use this model when:
- The response is a non-negative integer count with excess zeros.
- You need spatial autocorrelation in both the selection (binary)
  and count (intensity) processes.
- The two processes may operate on different spatial scales
  (different W matrices).

References
----------
Lambert, D. (1992). Zero-inflated Poisson regression, with an
application to defects in manufacturing. *Technometrics* 34(1), 1–14.

Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*JASA* 108(504), 1339–1349.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..._lazy_deps import az
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._slice import SliceWidthState
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.negbin_reduced import ReducedGibbsCache
from ...samplers.negbin_reduced._core import _make_cholmod_pattern
from ...samplers.zinb import (
    ZINBGibbsCache,
    ZINBGibbsPriors,
    ZINBGibbsState,
    run_zinb_chain,
)
from .._base._shared import _parse_W
from ..base import SpatialModel


class SARZINB(SpatialModel):
    """Bayesian zero-inflated SAR Negative Binomial with PG-Gibbs sampler.

    Parameters
    ----------
    formula : str, optional
        Wilkinson-style formula for the **count** equation, e.g.
        ``"y ~ x1 + x2"``. Requires ``data``.
    data : pandas.DataFrame or geopandas.GeoDataFrame, optional
        Data source for formula mode.
    y : array-like, optional
        Non-negative integer counts of shape ``(n,)``. Required in
        matrix mode.
    X : array-like, optional
        Count covariate matrix of shape ``(n, k)``. Required in matrix
        mode.
    Z : array-like, optional
        Selection covariate matrix of shape ``(n, p)``. If ``None``,
        defaults to ``X`` (same covariates for both equations).
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights for the **count** equation, shape ``(n, n)``.
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights for the **selection** equation. If ``None``,
        uses ``W`` (same weights for both equations).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``gamma_mu`` (float, default 0.0): Normal prior mean for γ.
        - ``gamma_sigma`` (float, default 1e6): Normal prior std for γ.
        - ``lam_lower`` (float, default -0.999): Lower bound for λ.
        - ``lam_upper`` (float, default 0.999): Upper bound for λ.
        - ``beta_mu`` (float, default 0.0): Normal prior mean for β.
        - ``beta_sigma`` (float, default 10.0): Normal prior std for β.
        - ``rho_lower`` (float, default -0.999): Lower bound for ρ.
        - ``rho_upper`` (float, default 0.999): Upper bound for ρ.
        - ``alpha_sigma`` (float, default 2.5): Half-Normal scale for α.
        - ``alpha_nu`` (float, default 3.0): Half-Normal ν for α.

    logdet_method : str, optional
        How to compute log|I − ρW|. ``None`` (default) auto-selects.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    The model is fit with a custom 9-block Gibbs sampler that composes
    the SAR-logit blocks (ω^sel, η^sel, γ, λ), the zero-allocation
    block (z), and the reduced-form SAR-NB blocks (ω^cnt, β, ρ, α).
    No PyMC model is constructed; calling ``_build_pymc_model``
    raises ``NotImplementedError``.
    """

    _spatial_params: tuple[str, ...] = ("rho", "lam")
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "rho"  # count equation Jacobian
    _gibbs_class: str | None = None
    _model_type: str = "zinb_sar"
    _likelihood: str = "count"
    _gibbs_key: tuple[str, str] | None = ("zinb", "cross_section")

    def __init__(
        self,
        formula=None,
        data=None,
        y=None,
        X=None,
        Z=None,
        W=None,
        W_sel=None,
        priors=None,
        logdet_method=None,
        robust=False,
        **kwargs,
    ):
        if robust:
            raise NotImplementedError("robust=True is not supported for SARZINB.")

        # Initialize base class with count equation data
        super().__init__(
            formula=formula,
            data=data,
            y=y,
            X=X,
            W=W,
            priors=priors,
            logdet_method=logdet_method,
            robust=False,
            **kwargs,
        )

        # Validate y is non-negative integer
        if np.any(self._y < 0):
            raise ValueError("y must be non-negative for ZINB models.")
        if not np.allclose(self._y, np.round(self._y)):
            raise ValueError("y must be integer-valued for ZINB models.")

        # Store integer y
        self._y_int = np.round(self._y).astype(np.int64)

        # Binary activity indicator: d = 1(y > 0)
        self._d = (self._y > 0).astype(np.float64)

        # Selection covariates Z
        if Z is not None:
            Z_arr = np.asarray(Z, dtype=np.float64)
            if Z_arr.ndim == 1:
                Z_arr = Z_arr.reshape(-1, 1)
            if Z_arr.shape[0] != len(self._y):
                raise ValueError(
                    f"Z has {Z_arr.shape[0]} rows but y has {len(self._y)} observations."
                )
            self._Z = Z_arr
            self._sel_feature_names = [f"z{j}" for j in range(Z_arr.shape[1])]
        else:
            self._Z = self._X.copy()
            self._sel_feature_names = list(self._feature_names)

        # Selection weights W_sel
        if W_sel is not None:
            self._W_sel_sparse, self._is_sel_row_std = _parse_W(W_sel, len(self._y))
            self._same_W = False
        else:
            self._W_sel_sparse = self._W_sparse
            self._is_sel_row_std = self._is_row_std
            self._same_W = True

        # Precompute logdet callable for the count equation ρ slice sampler
        self._logdet_fn = self._logdet_numpy_fn

    # ------------------------------------------------------------------
    # Selection-equation helpers
    # ------------------------------------------------------------------

    @cached_property
    def _sel_logdet_grad_numpy_vec_fn(self):
        """Vectorized ``(λ_arr) -> g(λ)`` logdet-gradient evaluator for W_sel.

        Mirrors :attr:`SpatialModel._logdet_grad_numpy_vec_fn` on the
        selection-equation weights; ``method=None`` auto-resolves to the fast
        surrogate for ``W_sel``'s own size/symmetry, only touching the
        eigenvalue path when that is the resolved method (tiny n).
        """
        from ..._logdet import make_logdet_grad_numpy_vec_fn

        return make_logdet_grad_numpy_vec_fn(
            self._W_sel_sparse,
            eigs=None,
            method=None,
            rho_min=float(self.priors.get("lam_lower", self._logdet_bounds.rho_min)),
            rho_max=float(self.priors.get("lam_upper", self._logdet_bounds.rho_max)),
        )

    def _sel_batch_mean_diag(self, lam_draws: np.ndarray) -> np.ndarray:
        """``(1/n) tr((I − λ W_sel)⁻¹)`` per draw for the selection equation.

        Direct analogue of :meth:`SpatialModel._batch_mean_diag` on the
        selection-equation weights.  When ``W_sel`` is the count-equation
        ``W`` the shared helper is reused; otherwise a resolvent
        (logdet-gradient) evaluator is built once for ``W_sel`` so the
        selection direct effect avoids the O(n³) eigendecomposition too.
        """
        if self._same_W:
            return self._batch_mean_diag(lam_draws)
        lam_draws = np.asarray(lam_draws, dtype=np.float64)
        n = int(self._W_sel_sparse.shape[0])
        g = np.asarray(self._sel_logdet_grad_numpy_vec_fn(lam_draws), dtype=np.float64)
        return 1.0 - (lam_draws / n) * g

    @cached_property
    def _sel_nonintercept_indices(self) -> list[int]:
        """Indices of non-constant columns in Z (selection covariates)."""
        indices: list[int] = []
        for j, name in enumerate(self._sel_feature_names):
            column = self._Z[:, j]
            is_named_intercept = name.lower() == "intercept"
            is_constant = np.allclose(column, column[0])
            if not (is_named_intercept or is_constant):
                indices.append(j)
        return indices

    @cached_property
    def _sel_nonintercept_feature_names(self) -> list[str]:
        """Feature names for non-intercept selection covariates."""
        return [self._sel_feature_names[i] for i in self._sel_nonintercept_indices]

    def _initialize_from_ols(self, rng):
        """Warm-start the ZINB Gibbs sampler.

        Uses profile-log-likelihood initialization for both equations:
        the selection equation is initialized from a spatial logit
        profile, and the count equation from a spatial NB profile on
        log(y+0.5).
        """
        n = len(self._y)
        y = self._y
        d = self._d
        X = self._X
        Z = self._Z
        W_cnt_csc = self._W_sparse.tocsc()
        W_sel_csc = self._W_sel_sparse.tocsc()
        k = X.shape[1]
        p = Z.shape[1]

        # --- Selection equation initialization ---
        # Profile log-likelihood on d (binary) using linear probability model.
        # Cached sparse solver: A = I - λW shares its pattern across the grid.
        from ...samplers._utils._sparsax_utils import (
            CachedSparseSolver,
            profile_loglik_rho_grid,
        )

        _best_lam, _best_gamma, _best_ll_sel = profile_loglik_rho_grid(d, Z, W_sel_csc)

        lam_init = float(
            np.clip(
                _best_lam + 0.02 * rng.standard_normal(),
                self._logdet_bounds.rho_min + 0.01,
                self._logdet_bounds.rho_max - 0.01,
            )
        )
        gamma_init = _best_gamma + 0.1 * rng.standard_normal(p)

        # η^sel from the selection profile
        try:
            _sel_solver = CachedSparseSolver([W_sel_csc], n)
            eta_sel_init = _sel_solver.solve([-lam_init], Z @ gamma_init)
        except Exception:
            eta_sel_init = Z @ gamma_init

        # ω^sel: PG(1, η^sel)
        from ...samplers._utils._polyagamma import sample_polyagamma

        omega_sel_init = sample_polyagamma(np.ones(n), eta_sel_init, rng=rng)

        # --- Count equation initialization ---
        # Profile log-likelihood on log(y+0.5) using ONLY positive
        # observations.  Structural zeros (d=0) should not influence
        # the count equation initialization.  The cached sparse solver
        # reuses the (I - ρW) pattern across the grid.
        pos_mask = y > 0
        n_pos = int(np.sum(pos_mask))
        if n_pos > k:
            _log_y = np.log(y[pos_mask] + 0.5)
            _X_pos = X[pos_mask]
        else:
            # Too few positive obs — use all with log(y+0.5)
            _log_y = np.log(y + 0.5)
            _X_pos = X
            pos_mask = np.ones(n, dtype=bool)
            n_pos = n
        _cnt_grid_solver = CachedSparseSolver([W_cnt_csc], n)
        _best_rho, _best_beta, _best_ll_cnt = 0.0, np.zeros(k), -np.inf
        for _rho_g in np.arange(0.05, 0.96, 0.05):
            try:
                _Xtilde_g = _cnt_grid_solver.solve([-float(_rho_g)], _X_pos)
                _beta_g = np.linalg.lstsq(_Xtilde_g, _log_y, rcond=None)[0]
                _eta_g = _Xtilde_g @ _beta_g
                _sig2_g = float(np.mean((_log_y - _eta_g) ** 2))
                if _sig2_g > 1e-10:
                    _ll_g = -0.5 * n_pos * np.log(_sig2_g) - 0.5 * n_pos
                    if _ll_g > _best_ll_cnt:
                        _best_ll_cnt = _ll_g
                        _best_rho = _rho_g
                        _best_beta = _beta_g.copy()
            except Exception:
                pass

        rho_init = float(
            np.clip(
                _best_rho + 0.02 * rng.standard_normal(),
                self._logdet_bounds.rho_min + 0.01,
                self._logdet_bounds.rho_max - 0.01,
            )
        )
        beta_init = _best_beta + 0.1 * rng.standard_normal(k)

        # Estimate α from Pearson residuals on positive observations
        try:
            _Xtilde_init = _cnt_grid_solver.solve([-rho_init], X)
            _eta_init = _Xtilde_init[pos_mask] @ beta_init
            _resid2 = float(np.mean((_log_y - _eta_init) ** 2))
            alpha_init = float(np.clip(1.0 / max(_resid2, 0.01), 0.5, 50.0))
        except Exception:
            alpha_init = 1.0

        # Jitter α
        alpha_init = float(
            np.clip(
                alpha_init * np.exp(0.1 * rng.standard_normal()),
                0.05,
                50.0,
            )
        )

        # ω^cnt: start at 0.25 (uninformative)
        omega_cnt_init = 0.25 * np.ones(n, dtype=np.float64)

        # z: initialize from data (z=1 for y>0, draw for y=0)
        z_init = np.ones(n, dtype=np.int8)
        zero_mask = y == 0
        if np.any(zero_mask):
            # Rough estimate: 50% of zeros are from the count process
            z_init[zero_mask] = rng.binomial(
                1, 0.5, size=int(np.sum(zero_mask))
            ).astype(np.int8)

        return ZINBGibbsState(
            eta_sel=eta_sel_init,
            gamma=gamma_init,
            lam=lam_init,
            omega_sel=omega_sel_init,
            beta=beta_init,
            rho=rho_init,
            alpha=alpha_init,
            omega_cnt=omega_cnt_init,
            z=z_init,
        )

    def _fit_gibbs(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: Optional[int] = None,
        thin: int = 1,
        n_jobs: int = 1,
        progressbar: bool = True,
        backend: str = "numpy",
        timeout: float | None = None,
    ) -> az.InferenceData:
        """Sample posterior via 9-block Pólya–Gamma Gibbs.

        Parameters
        ----------
        draws, tune, chains : int
            Post-warmup draws, warmup draws, and number of chains.
        random_seed : int, optional
            Seed for reproducibility.
        thin : int
            Keep every ``thin``-th draw. Default 1.
        n_jobs : int
            Number of parallel chains. 1 = sequential.
        progressbar : bool
            Show per-chain progress bars.
        backend : {"numpy", "jax"}
            Execution backend — both fit the *same* reduced-form ZINB (reduced
            SAR-logit selection + reduced SAR-NB count).  ``"jax"`` (the default
            via ``"auto"``) runs the chains in parallel threads with
            sparsax-KLU solves and on-device Pólya-Gamma; ``"numpy"`` uses the
            CHOLMOD 9-block Gibbs.
        timeout : float or None
            Maximum wall-clock seconds for parallel chains.

        Returns
        -------
        arviz.InferenceData
            Posterior draws of ``lam``, ``gamma``, ``rho``, ``beta``,
            ``alpha`` and pointwise ``log_likelihood``.
        """
        n, k = self._X.shape
        self._Z.shape[1]

        if n < 900:
            warnings.warn(
                f"Zero-inflated NB models require large samples for "
                f"reliable spatial parameter recovery. With n={n}, "
                f"posterior estimates of ρ, λ, and α may be severely "
                f"attenuated. n ≥ 900 is recommended.",
                UserWarning,
                stacklevel=2,
            )

        bounds = self._logdet_bounds
        rho_lower = float(bounds.rho_min)
        rho_upper = float(bounds.rho_max)

        # Build priors
        priors = ZINBGibbsPriors(
            gamma_mu=self.priors.get("gamma_mu", 0.0),
            gamma_sigma=self.priors.get("gamma_sigma", 1e6),
            lam_lower=self.priors.get("lam_lower", rho_lower),
            lam_upper=self.priors.get("lam_upper", rho_upper),
            beta_mu=self.priors.get("beta_mu", 0.0),
            beta_sigma=self.priors.get("beta_sigma", 10.0),
            rho_lower=self.priors.get("rho_lower", rho_lower),
            rho_upper=self.priors.get("rho_upper", rho_upper),
            alpha_sigma=self.priors.get("alpha_sigma", 2.5),
            alpha_nu=self.priors.get("alpha_nu", 3.0),
        )

        # ── JAX path ──
        # Both equations reduced-form (Krylov-only slice, sparsax-KLU solves,
        # on-device Pólya-Gamma via pgjax, chains in parallel threads).
        # The NumPy path (below) fits the identical reduced-form model.
        if backend == "jax":
            from ...samplers.zinb._jax import run_chains_jax_zinb

            seed_rng = np.random.default_rng(random_seed)
            chain_seeds = [int(s) for s in seed_rng.integers(0, 2**31, size=chains)]
            chain_inits = [
                self._initialize_from_ols(np.random.default_rng(seed))
                for seed in chain_seeds
            ]

            chain_results = run_chains_jax_zinb(
                y=self._y,
                d=self._d,
                Z=self._Z,
                X=self._X,
                W_sel_sparse=self._W_sel_sparse.tocsr(),
                W_cnt_sparse=self._W_sparse.tocsr(),
                priors=priors,
                inits=chain_inits,
                draws=draws,
                tune=tune,
                thin=thin,
                jax_seeds=chain_seeds,
                progressbar=progressbar,
            )

            stacked = {
                "lam": np.stack([c["lam"] for c in chain_results], axis=0),
                "gamma": np.stack([c["gamma"] for c in chain_results], axis=0),
                "rho": np.stack([c["rho"] for c in chain_results], axis=0),
                "beta": np.stack([c["beta"] for c in chain_results], axis=0),
                "alpha": np.stack([c["alpha"] for c in chain_results], axis=0),
            }
            log_lik = np.stack([c["log_lik"] for c in chain_results], axis=0)

            idata = gibbs_to_inference_data(
                posterior_samples=stacked,
                log_likelihood={"obs": log_lik},
                observed_data={"obs": self._y_int},
                coords={
                    "sel_coefficient": list(self._sel_feature_names),
                    "coefficient": list(self._feature_names),
                    "obs_id": list(range(n)),
                },
                dims={
                    "gamma": ["sel_coefficient"],
                    "beta": ["coefficient"],
                    "obs": ["obs_id"],
                },
            )
            self._idata = idata
            return idata

        # --- Count equation cache (reduced-form SAR-NB) ---
        W_cnt_csr = self._W_sparse.tocsr()
        W_cnt_csc = self._W_sparse.tocsc()

        # Spectrum bounds for the solve path.  Deliberately *not* from
        # ``_W_eigs``: that densifies W for an O(n^3) eigendecomposition, and
        # only bounds are needed here.  See ``_W_spectral_bounds``.
        W_eig_max, W_eig_min = self._W_spectral_bounds

        W_cnt_sym, W_cnt_tW, cnt_pattern = _make_cholmod_pattern(W_cnt_csc, n)
        cnt_cholmod_pattern = cnt_pattern

        # --- Selection equation cache (reduced-form SAR-logit on W_sel) ---
        # Same ReducedGibbsCache shape as the count; the λ slice uses the
        # direct-CG path (basis=None), so krylov_degree is unused here.  For
        # W_sel eigen bounds reuse the count's when the weights match, else use
        # the row-standardized default [−1, 1] (safe for the CG SPD check).
        W_sel_csr = self._W_sel_sparse.tocsr()
        W_sel_csc = self._W_sel_sparse.tocsc()
        W_sel_sym, W_sel_tW, sel_pattern = _make_cholmod_pattern(W_sel_csc, n)
        if self._same_W:
            W_sel_eig_max, W_sel_eig_min = W_eig_max, W_eig_min
        else:
            W_sel_eig_max, W_sel_eig_min = 1.0, -1.0

        sel_cache = ReducedGibbsCache(
            W_sparse=W_sel_csr,
            W_csc=W_sel_csc,
            rho_lower=priors.lam_lower,
            rho_upper=priors.lam_upper,
            rho_adaptive_width=True,
            rho_slice_width_state=SliceWidthState(w=0.2),
            krylov_degree=0,
            krylov_dmax=0.15,
            cholmod_pattern=sel_pattern,
            W_sym=W_sel_sym,
            WtW=W_sel_tW,
            W_eig_max=W_sel_eig_max,
            W_eig_min=W_sel_eig_min,
            n_rho_omega_cycles=1,
        )

        cnt_cache = ReducedGibbsCache(
            W_sparse=W_cnt_csr,
            W_csc=W_cnt_csc,
            rho_lower=priors.rho_lower,
            rho_upper=priors.rho_upper,
            rho_adaptive_width=True,
            rho_slice_width_state=SliceWidthState(w=0.2),
            krylov_degree=8,
            krylov_dmax=0.15,
            cholmod_pattern=cnt_cholmod_pattern,
            W_sym=W_cnt_sym,
            WtW=W_cnt_tW,
            W_eig_max=W_eig_max,
            W_eig_min=W_eig_min,
            n_rho_omega_cycles=1,
        )

        # --- ZINB cache ---
        zinb_cache = ZINBGibbsCache(
            sel_cache=sel_cache,
            cnt_cache=cnt_cache,
            y=self._y,
            d=self._d,
            Z=self._Z,
            X=self._X,
            W_sel_sparse=W_sel_csr,
            W_cnt_sparse=W_cnt_csr,
            same_W=self._same_W,
        )

        # Derive per-chain seeds
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]

        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            chain_rng = np.random.default_rng(seed)
            init = self._initialize_from_ols(chain_rng)
            progress_chain_id = chain_id if chain_id_kw is None else chain_id_kw
            return run_zinb_chain(
                y=self._y,
                d=self._d,
                Z=self._Z,
                X=self._X,
                W_sel_sparse=W_sel_csr,
                W_cnt_sparse=W_cnt_csr,
                priors=priors,
                cache=zinb_cache,
                init=init,
                draws=draws,
                tune=tune,
                thin=thin,
                rng=chain_rng,
                chain_id=progress_chain_id,
                progress_manager=progress_manager,
            )

        chain_results = run_chains(
            chain_fn=_run_one_chain,
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=(n_jobs != 1),
            draws=draws,
            tune=tune,
            model_type="zinb_sar",
            timeout=timeout,
        )

        # Stack chains
        stacked = {
            "lam": np.stack([c["lam"] for c in chain_results], axis=0),
            "gamma": np.stack([c["gamma"] for c in chain_results], axis=0),
            "rho": np.stack([c["rho"] for c in chain_results], axis=0),
            "beta": np.stack([c["beta"] for c in chain_results], axis=0),
            "alpha": np.stack([c["alpha"] for c in chain_results], axis=0),
        }
        log_lik = np.stack([c["log_lik"] for c in chain_results], axis=0)

        idata = gibbs_to_inference_data(
            posterior_samples=stacked,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": self._y_int},
            coords={
                "sel_coefficient": list(self._sel_feature_names),
                "coefficient": list(self._feature_names),
                "obs_id": list(range(n)),
            },
            dims={
                "gamma": ["sel_coefficient"],
                "beta": ["coefficient"],
                "obs": ["obs_id"],
            },
        )

        self._idata = idata
        return idata

    def _build_pymc_model(self):
        """Not supported — SARZINB uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            "SARZINB does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def _compute_spatial_effects_posterior(
        self, equation: str = "count"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts for each draw.

        Parameters
        ----------
        equation : {"count", "selection"}, default "count"
            Which equation to compute impacts for.
        """
        from ...diagnostics.lmtests import _get_posterior_draws

        idata = self.inference_data

        if equation == "count":
            rho_draws = _get_posterior_draws(idata, "rho")
            beta_draws = _get_posterior_draws(idata, "beta")

            mean_diag = self._batch_mean_diag(rho_draws)
            mean_row_sum = self._batch_mean_row_sum(rho_draws)

            ni = self._nonintercept_indices
            direct_samples = mean_diag[:, None] * beta_draws[:, ni]
            total_samples = mean_row_sum[:, None] * beta_draws[:, ni]
            indirect_samples = total_samples - direct_samples
        elif equation == "selection":
            lam_draws = _get_posterior_draws(idata, "lam")
            gamma_draws = _get_posterior_draws(idata, "gamma")

            # Direct-effect trace (1/n)tr((I−λW_sel)⁻¹) via the resolvent
            # (logdet gradient) — no eigendecomposition of W_sel.
            mean_diag = self._sel_batch_mean_diag(lam_draws)

            # Mean row sum of (I - lam * W_sel)^{-1}
            if self._is_sel_row_std:
                mean_row_sum = 1.0 / (1.0 - lam_draws)
            else:
                # The non-row-standardized total effect is a column-weighted
                # bilinear form (1'S1), not a trace, so it keeps the
                # eigenvector decomposition of W_sel.
                from ...diagnostics.spatial_effects import _chunked_eig_means

                n = self._Z.shape[0]
                W_sel_dense = self._W_sel_sparse.toarray().astype(np.float64)
                eigvals_sel, V_sel = np.linalg.eig(W_sel_dense)
                c_sel = np.linalg.solve(V_sel, np.ones(n))
                V_col_sums_sel = V_sel.sum(axis=0)
                mean_row_sum = _chunked_eig_means(
                    lam_draws, eigvals_sel, weights=V_col_sums_sel * c_sel
                )

            ni = self._sel_nonintercept_indices
            direct_samples = mean_diag[:, None] * gamma_draws[:, ni]
            total_samples = mean_row_sum[:, None] * gamma_draws[:, ni]
            indirect_samples = total_samples - direct_samples
        else:
            raise ValueError(
                f"equation must be 'count' or 'selection', got '{equation}'"
            )

        return direct_samples, indirect_samples, total_samples

    def spatial_effects(
        self,
        equation: str = "count",
        return_posterior_samples: bool = False,
    ) -> "pd.DataFrame | tuple[pd.DataFrame, dict[str, np.ndarray]]":
        """Compute Bayesian inference for direct, indirect, and total impacts.

        The ZINB-SAR model has two spatial lag equations — a count
        equation with parameter ρ and a selection (logit) equation
        with parameter λ — each with its own LeSage–Pace impact
        decomposition.  Use the ``equation`` parameter to select which
        equation's impacts to report.

        Parameters
        ----------
        equation : {"count", "selection"}, default "count"
            Which equation to compute impacts for.

            - ``"count"``: impacts of X on E[y | d=1] through
              (I − ρW)⁻¹ Xβ.
            - ``"selection"``: impacts of Z on P(d=1) on the log-odds
              scale through (I − λW_sel)⁻¹ Zγ.
        return_posterior_samples : bool, default False
            If ``True``, return a ``(DataFrame, dict)`` tuple where the
            dict contains the full posterior draws.

        Returns
        -------
        pd.DataFrame or tuple of (pd.DataFrame, dict)
            Impact summary table, optionally with posterior draws.
        """
        from ...diagnostics.spatial_effects import _build_effects_dataframe

        self._require_fit()
        direct_samples, indirect_samples, total_samples = (
            self._compute_spatial_effects_posterior(equation=equation)
        )

        # Determine feature names for the selected equation.
        if equation == "selection":
            k_effects = direct_samples.shape[1]
            sel_names = self._sel_nonintercept_feature_names
            if len(sel_names) == k_effects:
                feature_names = list(sel_names)
            else:
                feature_names = list(self._sel_feature_names[:k_effects])
        else:
            k_effects = direct_samples.shape[1]
            if (
                hasattr(self, "_wx_feature_names")
                and len(self._wx_feature_names) == k_effects
            ):
                feature_names = list(self._wx_feature_names)
            elif len(self._nonintercept_feature_names) == k_effects:
                feature_names = list(self._nonintercept_feature_names)
            else:
                feature_names = list(self._feature_names[:k_effects])

        model_type = f"{self.__class__.__name__} ({equation})"

        df = _build_effects_dataframe(
            direct_samples=direct_samples,
            indirect_samples=indirect_samples,
            total_samples=total_samples,
            feature_names=feature_names,
            model_type=model_type,
        )

        if return_posterior_samples:
            posterior_samples = {
                "direct": direct_samples,
                "indirect": indirect_samples,
                "total": total_samples,
            }
            return df, posterior_samples
        return df

    def _fitted_mean_from_posterior(self) -> np.ndarray:
        """Compute posterior-mean fitted expected counts.

        E[y_i] = π_i · exp(η_i^cnt) where π_i = logit⁻¹(η_i^sel)
        and η_i^cnt = (I - ρW)^{-1} Xβ at posterior means.
        """
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        lam = float(self._posterior_mean("lam"))
        gamma = self._posterior_mean("gamma")
        n = self._X.shape[0]

        # Count equation: η^cnt = (I - ρW)^{-1} Xβ
        A_cnt = sp.eye(n, format="csr", dtype=np.float64) - rho * self._W_sparse
        eta_cnt = sp.linalg.spsolve(A_cnt, self._X @ beta)

        # Selection equation: η^sel = (I - λW_sel)^{-1} Zγ
        A_sel = sp.eye(n, format="csr", dtype=np.float64) - lam * self._W_sel_sparse
        eta_sel = sp.linalg.spsolve(A_sel, self._Z @ gamma)

        pi = 1.0 / (1.0 + np.exp(-eta_sel))
        return pi * np.exp(eta_cnt)

    def corridor_probabilities(self) -> np.ndarray:
        """Posterior-mean corridor activation probabilities.

        Returns π_i = logit⁻¹(η_i^sel) at posterior means of λ and γ.

        Returns
        -------
        pi : ndarray of shape (n,)
            Fitted activation probabilities.
        """
        lam = float(self._posterior_mean("lam"))
        gamma = self._posterior_mean("gamma")
        n = self._X.shape[0]

        A_sel = sp.eye(n, format="csr", dtype=np.float64) - lam * self._W_sel_sparse
        eta_sel = sp.linalg.spsolve(A_sel, self._Z @ gamma)
        return 1.0 / (1.0 + np.exp(-eta_sel))

    def zero_attribution(self) -> dict[str, np.ndarray]:
        """Decompose observed zeros into structural vs sampling zeros.

        For each observation with y_i = 0, computes the posterior
        probability that the zero is structural (d_i = 0) versus
        sampling (d_i = 1 but NB draw was zero).

        Returns
        -------
        dict with keys:
            ``structural_prob`` : ndarray of shape (n_zero,)
                P(d_i = 0 | y_i = 0) for each zero observation.
            ``sampling_prob`` : ndarray of shape (n_zero,)
                P(d_i = 1, NB zero | y_i = 0) for each zero observation.
            ``zero_indices`` : ndarray of shape (n_zero,)
                Indices of zero observations.
        """
        self._require_fit()
        rho = float(self._posterior_mean("rho"))
        beta = self._posterior_mean("beta")
        lam = float(self._posterior_mean("lam"))
        gamma = self._posterior_mean("gamma")
        alpha = float(self._posterior_mean("alpha"))
        n = self._X.shape[0]

        # Count equation: η^cnt
        A_cnt = sp.eye(n, format="csr", dtype=np.float64) - rho * self._W_sparse
        eta_cnt = sp.linalg.spsolve(A_cnt, self._X @ beta)

        # Selection equation: η^sel
        A_sel = sp.eye(n, format="csr", dtype=np.float64) - lam * self._W_sel_sparse
        eta_sel = sp.linalg.spsolve(A_sel, self._Z @ gamma)

        zero_mask = self._y == 0
        zero_idx = np.where(zero_mask)[0]

        pi = 1.0 / (1.0 + np.exp(-eta_sel[zero_mask]))
        log_p_nb_zero = alpha * np.log(alpha / (np.exp(eta_cnt[zero_mask]) + alpha))
        p_nb_zero = np.exp(np.clip(log_p_nb_zero, -700, 0))

        # P(structural | y=0) = (1-π) / ((1-π) + π·p_nb_zero)
        structural = 1.0 - pi
        sampling = pi * p_nb_zero
        total = structural + sampling
        total = np.where(total > 0, total, 1.0)

        return {
            "structural_prob": structural / total,
            "sampling_prob": sampling / total,
            "zero_indices": zero_idx,
        }
