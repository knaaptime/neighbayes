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
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..._lazy_deps import xr
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
from .._base._nb import nb_alpha_fixed, nb_alpha_rv
from .._base._shared import _parse_W
from .._mixins._zinb import ZINBMixin
from ..base import SpatialModel
from ..priors import SARZINBPriors


def _selection_design(Z, sel_formula, data, n: int):
    """The selection design and its column names, or ``(None, None)`` for ``Z = X``."""
    if Z is not None and sel_formula is not None:
        raise ValueError("Pass Z or sel_formula, not both.")
    if sel_formula is not None:
        if data is None:
            raise ValueError("sel_formula needs data (formula mode).")
        from formulaic import model_matrix

        rhs = sel_formula.split("~", 1)[-1].strip()
        mm = model_matrix(rhs, data)
        names, Z_arr = list(mm.columns), np.asarray(mm, dtype=np.float64)
    elif Z is not None:
        if isinstance(Z, pd.DataFrame):
            names = [str(c) for c in Z.columns]
        Z_arr = np.asarray(Z, dtype=np.float64)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if not isinstance(Z, pd.DataFrame):
            names = [f"z{j}" for j in range(Z_arr.shape[1])]
    else:
        return None, None
    if Z_arr.shape[0] != n:
        raise ValueError(f"Z has {Z_arr.shape[0]} rows but y has {n} observations.")
    return Z_arr, names


class SARZINB(ZINBMixin, SpatialModel):
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
    Z : array-like or pandas.DataFrame, optional
        Selection covariate matrix of shape ``(n, p)``; a DataFrame's column
        names label γ. If ``None`` (and no ``sel_formula``), defaults to ``X``
        (same covariates for both equations).
    sel_formula : str, optional
        Right-hand side for the **selection** equation, e.g. ``"~ z1 + z2"``,
        evaluated on ``data`` (formula mode). Alternative to ``Z``.
    W : libpysal.graph.Graph or scipy.sparse matrix
        Spatial weights for the **count** equation, shape ``(n, n)``.
    W_sel : libpysal.graph.Graph or scipy.sparse matrix, optional
        Spatial weights for the **selection** equation. If ``None``,
        uses ``W`` (same weights for both equations).
    priors : dict, optional
        Override default priors. Supported keys:

        - ``gamma_mu``, ``gamma_sigma`` (float or array, default Gelman et al.
          2008 on the logit scale, centred at even odds): Normal prior on γ.
        - ``lam_lower`` (float, default -0.999): Lower bound for λ.
        - ``lam_upper`` (float, default 0.999): Upper bound for λ.
        - ``beta_mu``, ``beta_sigma`` (float or array, default Gelman et al.
          2008 on the log scale): Normal prior on β, intercept at
          ``log(mean(y))`` with scale 2.5 and slopes ``2.5 / sd(x_j)``.
        - ``rho_lower`` (float, default -0.999): Lower bound for ρ.
        - ``rho_upper`` (float, default 0.999): Upper bound for ρ.
        - ``alpha_sigma``, ``alpha_nu`` (float, default 2.5 and 3.0):
          Half-Student-t(ν, σ) prior on the NB dispersion α.
        - ``alpha_fixed`` (float, optional): hold α at this value instead;
          a large value (10-20 times the typical mean count) makes the count
          equation essentially Poisson on the same exact sampler.

        See :class:`~neighbayes.models.priors.SARZINBPriors`.

    logdet_method : str, optional
        How to compute log|I − ρW|. ``None`` (default) auto-selects.
    robust : bool, default False
        Not supported. Raises ``NotImplementedError`` if True.

    Notes
    -----
    ``fit()`` defaults to a 9-block Gibbs sampler that composes the SAR-logit
    blocks (ω^sel, η^sel, γ, λ), the zero-allocation block (z), and the
    reduced-form SAR-NB blocks (ω^cnt, β, ρ, α); ``sampler="nuts"`` fits the
    same model and priors in PyMC (``ZeroInflatedNegativeBinomial``).

    ``corridor_probabilities``, ``zero_attribution`` and the fitted means are
    posterior expectations averaged over draws; ``posterior_predictive``
    simulates ``d ~ Bern(π)``, ``y = d · NB``.
    """

    _spatial_params: tuple[str, ...] = ("rho", "lam")
    _lag_terms: tuple[str, ...] = ()
    _jacobian_param: str | None = "rho"  # count equation Jacobian
    _gibbs_class: str | None = None
    _model_type: str = "zinb_sar"
    _likelihood: str = "count"
    _gibbs_key: tuple[str, str] | None = ("zinb", "cross_section")
    _priors_cls = SARZINBPriors

    def __init__(
        self,
        formula=None,
        data=None,
        y=None,
        X=None,
        Z=None,
        W=None,
        sel_formula=None,
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
        self._Z, self._sel_feature_names = _selection_design(
            Z, sel_formula, data, len(self._y)
        )
        if self._Z is None:
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
        else:
            # Too few positive obs — use all with log(y+0.5)
            _log_y = np.log(y + 0.5)
            pos_mask = np.ones(n, dtype=bool)
            n_pos = n
        _cnt_grid_solver = CachedSparseSolver([W_cnt_csc], n)
        _best_rho, _best_beta, _best_ll_cnt = 0.0, np.zeros(k), -np.inf
        for _rho_g in np.arange(0.05, 0.96, 0.05):
            try:
                # Filter all of X, then keep the positive rows: the filter
                # mixes every row, zeros included.
                _Xtilde_g = _cnt_grid_solver.solve([-float(_rho_g)], X)[pos_mask]
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

        fixed = nb_alpha_fixed(self.priors)
        if fixed is not None:
            alpha_init = fixed

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

    def _zinb_priors(self) -> ZINBGibbsPriors:
        """Resolved priors, shared by the Gibbs and NUTS paths.

        Gelman et al. (2008) defaults on each equation's link scale: log for
        the count equation; logit for the latent selection, centred at even
        odds because the structural zeros are not observed.
        """
        from .._base._shared import gelman_default_beta_prior

        bounds = self._logdet_bounds
        rho_lower, rho_upper = float(bounds.rho_min), float(bounds.rho_max)
        beta_mu, beta_sigma = self._resolved_beta_prior(link="log")
        g_mu, g_sd = gelman_default_beta_prior(
            np.full(self._Z.shape[0], 0.5),
            self._Z,
            list(self._sel_feature_names),
            link="logit",
        )
        p = self._Z.shape[1]
        return ZINBGibbsPriors(
            gamma_mu=np.broadcast_to(
                np.asarray(self.priors.get("gamma_mu", g_mu), dtype=float), (p,)
            ).copy(),
            gamma_sigma=np.broadcast_to(
                np.asarray(self.priors.get("gamma_sigma", g_sd), dtype=float), (p,)
            ).copy(),
            lam_lower=float(self.priors.get("lam_lower", rho_lower)),
            lam_upper=float(self.priors.get("lam_upper", rho_upper)),
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            rho_lower=float(self.priors.get("rho_lower", rho_lower)),
            rho_upper=float(self.priors.get("rho_upper", rho_upper)),
            alpha_sigma=float(self.priors.get("alpha_sigma", 2.5)),
            alpha_nu=float(self.priors.get("alpha_nu", 3.0)),
            alpha_fixed=nb_alpha_fixed(self.priors),
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
        log_likelihood: bool = False,
    ) -> xr.DataTree:
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
        xarray.DataTree
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

        priors = self._zinb_priors()

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
                store_log_lik=log_likelihood,
            )

            stacked = {
                "lam": np.stack([c["lam"] for c in chain_results], axis=0),
                "gamma": np.stack([c["gamma"] for c in chain_results], axis=0),
                "rho": np.stack([c["rho"] for c in chain_results], axis=0),
                "beta": np.stack([c["beta"] for c in chain_results], axis=0),
                "alpha": np.stack([c["alpha"] for c in chain_results], axis=0),
            }
            log_lik = (
                np.stack([c["log_lik"] for c in chain_results], axis=0)
                if log_likelihood
                else None
            )

            idata = gibbs_to_inference_data(
                posterior_samples=stacked,
                log_likelihood={"obs": log_lik} if log_likelihood else None,
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
                store_log_lik=log_likelihood,
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
        log_lik = (
            np.stack([c["log_lik"] for c in chain_results], axis=0)
            if log_likelihood
            else None
        )

        idata = gibbs_to_inference_data(
            posterior_samples=stacked,
            log_likelihood={"obs": log_lik} if log_likelihood else None,
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
        r"""Reduced-form ZINB for NUTS: the same model and priors as the Gibbs path.

        ``y ~ ZINB(ψ = π, μ = exp(η_cnt), α)`` with
        ``η_sel = (I − λW_sel)⁻¹Zγ`` and ``η_cnt = (I − ρW)⁻¹Xβ``; PyMC's ``ψ`` is
        the probability of the NB component, i.e. the activation probability π.
        No ``log|I − ρW|`` terms: neither ``y`` nor ``d`` is transformed.
        """
        import pytensor.tensor as pt

        from ..._lazy_deps import pm
        from ..._ops import SparseSARSolveOp

        pv = self._zinb_priors()
        with pm.Model(coords=self._model_coords(self._zinb_coords())) as model:
            lam = pm.Uniform("lam", lower=pv.lam_lower, upper=pv.lam_upper)
            gamma = pm.Normal(
                "gamma", mu=pv.gamma_mu, sigma=pv.gamma_sigma, dims="sel_coefficient"
            )
            rho = pm.Uniform("rho", lower=pv.rho_lower, upper=pv.rho_upper)
            beta = pm.Normal(
                "beta", mu=pv.beta_mu, sigma=pv.beta_sigma, dims="coefficient"
            )
            alpha = nb_alpha_rv(pv.alpha_fixed, pv.alpha_nu, pv.alpha_sigma)
            eta_sel = SparseSARSolveOp(self._W_sel_sparse)(
                lam, pt.dot(pt.as_tensor_variable(self._Z), gamma)
            )
            eta_cnt = SparseSARSolveOp(self._W_sparse)(
                rho, pt.dot(pt.as_tensor_variable(self._X), beta)
            )
            pm.ZeroInflatedNegativeBinomial(
                "obs",
                psi=pm.math.sigmoid(eta_sel),
                mu=pt.exp(eta_cnt),
                alpha=alpha,
                observed=self._y_int,
            )
        return model

    def _zinb_coords(self) -> dict:
        return {"sel_coefficient": list(self._sel_feature_names)}

    def _part_etas(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        """``(η_sel, η_cnt)`` at posterior draw ``g``."""
        n = self._X.shape[0]
        lam = float(self._flat_draw("lam")[g])
        rho = float(self._flat_draw("rho")[g])
        gamma = self._flat_draw("gamma")[g]
        beta = self._flat_draw("beta")[g]
        eye = sp.eye(n, format="csc", dtype=np.float64)
        eta_sel = sp.linalg.splu((eye - lam * self._W_sel_sparse).tocsc()).solve(
            self._Z @ gamma
        )
        eta_cnt = sp.linalg.splu((eye - rho * self._W_sparse).tocsc()).solve(
            self._X @ beta
        )
        return eta_sel, eta_cnt
