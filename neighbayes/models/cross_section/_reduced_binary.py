r"""The reduced-form SAR logit and probit.

Shared by :class:`~neighbayes.models.SARLogit` and
``SARProbit(latent=False)``:

.. math::

    y_i \sim \mathrm{Bernoulli}(F(\eta_i)), \qquad
    \eta = (I - \rho W)^{-1} X\beta,

with :math:`F` the logistic or normal distribution function.  The spatial lag
enters the linear predictor as a deterministic mean-propagator, so the
``|I − ρW|`` Jacobian cancels when β is marginalized out.  The sampler is
:mod:`neighbayes.samplers.binary_reduced`: Pólya–Gamma (logit) or Albert–Chib
(probit) augmentation, conjugate β, and a β-collapsed slice on ρ.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.special import expit, ndtr

from ..._lazy_deps import xr
from ...samplers._utils._idata import gibbs_to_inference_data
from ...samplers._utils._slice import SliceWidthState
from ...samplers.binary_reduced import (
    ReducedBinaryGibbsState,
    run_chain,
    run_chains_jax_reduced_binary,
)
from ...samplers.gaussian._chain_runner import run_chains
from ...samplers.negbin_reduced._core import (
    _KRYLOV_DEGREE_DEFAULT,
    _KRYLOV_DMAX_DEFAULT,
    ReducedGibbsCache,
    _make_cholmod_pattern,
)
from ..priors import BinaryGibbsPriors


def _inverse_link(link: str, eta: np.ndarray) -> np.ndarray:
    """``P(y = 1)`` at the linear predictor ``eta``."""
    return expit(eta) if link == "logit" else ndtr(eta)


def _link_density(link: str, eta: np.ndarray) -> np.ndarray:
    """``F'(eta)``: the inverse link's derivative."""
    if link == "logit":
        p = expit(eta)
        return p * (1.0 - p)
    return np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi)


class ReducedBinaryMixin:
    """The reduced-form SAR binary model's Gibbs fit, fitted values and impacts.

    The host sets ``_link`` and ``_index_scale``; the latent design, where the
    class has one, dispatches to :class:`._latent.LatentMixin`.
    """

    def _initialize_from_ols(self, rng, init_jitter: float = 0.1):
        """Warm-start the reduced-form Gibbs sampler from a spatial profile.

        For each ρ on a coarse grid, computes X̃ = (I − ρW)⁻¹X and the OLS
        estimate β̂ = (X̃ᵀX̃)⁻¹X̃ᵀy (linear probability proxy), then picks the
        (ρ, β) maximising the Gaussian log-likelihood on y, and jitters β by
        ``init_jitter`` and ρ by at most 0.02.  Returns a
        :class:`ReducedBinaryGibbsState` (β, ρ, ω) — no latent η field.  The
        logit's ω is drawn from PG(1, η) at the profile η; the probit's is one.
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
        _rho_jitter = min(init_jitter, 0.02)
        beta_init = _best_beta + init_jitter * rng.standard_normal(k)
        rho_init = float(
            np.clip(
                _best_rho + _rho_jitter * rng.standard_normal(),
                self._logdet_bounds.rho_min + 0.01,
                self._logdet_bounds.rho_max - 0.01,
            )
        )

        if self._link == "logit":
            # ω₀: draw from PG(1, η) at the profile η.
            _init_solver = CachedSparseSolver([W_csc], n)
            eta_init = _init_solver.solve([-rho_init], X @ beta_init)
            from ...samplers._utils._polyagamma import sample_polyagamma

            omega_init = sample_polyagamma(np.ones(n), eta_init, rng=rng)
        else:
            omega_init = np.ones(n)

        return ReducedBinaryGibbsState(
            beta=beta_init,
            rho=rho_init,
            omega=omega_init,
        )

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
        krylov_reuse: bool = True,
        timeout: float | None = None,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        r"""Sample the reduced-form posterior by block Gibbs (Pólya–Gamma or Albert–Chib).

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
        xr.DataTree
            With posterior (``rho``, ``beta``), log_likelihood, and
            observed_data groups.
        """
        y = np.asarray(self._y, dtype=np.float64)
        X = np.ascontiguousarray(self._X, dtype=np.float64)
        n, k = X.shape

        bounds = self._logdet_bounds
        rho_lower = float(bounds.rho_min)
        rho_upper = float(bounds.rho_max)

        # Link-scale Gelman et al. (2008) default unless overridden.
        link = self._link
        beta_mu, beta_sigma = self._resolved_beta_prior(link=link)
        priors = BinaryGibbsPriors(
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
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
                self._initialize_from_ols(np.random.default_rng(s), init_jitter)
                for s in chain_seeds
            ]
            intercept_col = -1
            for _j in range(k):
                if np.all(X[:, _j] == 1.0):
                    intercept_col = _j
                    break

            chain_results = run_chains_jax_reduced_binary(
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
                store_log_lik=log_likelihood,
                link=link,
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
                init = self._initialize_from_ols(chain_rng, init_jitter)
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
                    store_log_lik=log_likelihood,
                    link=link,
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
                model_type=self._model_type,
                timeout=timeout,
            )

        # Assemble DataTree (varnames: rho, beta).
        posterior_samples = {
            "rho": np.stack([c["rho"] for c in chain_results], axis=0),
            "beta": np.stack([c["beta"] for c in chain_results], axis=0),
        }
        log_lik = (
            np.stack([c["log_lik"] for c in chain_results], axis=0)
            if log_likelihood
            else None
        )

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik} if log_likelihood else None,
            observed_data={"obs": y},
            coords={"coefficient": list(self._feature_names)},
            dims={"beta": ["coefficient"]},
        )

        self._idata = idata
        return idata

    def _build_pymc_model_static(self):
        """Not supported — the reduced form uses a Gibbs sampler, not NUTS."""
        raise NotImplementedError(
            f"{type(self).__name__} does not build a PyMC model. "
            "Use the fit() method for Gibbs sampling."
        )

    def _fitted_probabilities_static(self) -> np.ndarray:
        """Compute fitted probabilities at posterior mean parameters.

        Returns the probability P(y=1) = F(η), with F the link's inverse,
        where η is computed from the posterior mean of ρ and β via the reduced
        form: η = (I - ρW)⁻¹ Xβ.

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
        return _inverse_link(self._link, A_rho_inv)

    def _fitted_mean_from_posterior_static(self) -> np.ndarray:
        """Compute fitted values at posterior mean parameters.

        For a binary model, the fitted mean is the fitted probability.
        """
        return self._fitted_probabilities_static()

    def _compute_impacts_posterior(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior impacts on the latent index (link) scale for each draw."""
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

    #: Threshold above which probability-scale impacts use the
    #: sparse Hutchinson path instead of the eigendecomposition path.
    #: See :attr:`SARNegBin._COUNT_EFFECTS_EIGEN_MAX_N` for the
    #: cost model — the binary case is identical structurally.
    _PROBABILITY_EFFECTS_EIGEN_MAX_N: int = 2000

    def _compute_probability_scale_impacts_posterior(
        self,
        method: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute posterior impacts on the probability scale for each draw.

        Notes
        -----
        For the reduced-form SAR with

        .. math::

            \eta = (I - \rho W)^{-1} X\beta, \qquad p = F(\eta),

        the average partial-effect matrix for covariate :math:`x_r` on the
        response (probability) scale is

        .. math::

            \frac{\partial p}{\partial x_r'} =
            \operatorname{diag}\bigl(F'(\eta)\bigr)
            (I - \rho W)^{-1} \beta_r,

        with :math:`F'(\eta) = p(1 - p)` for the logit (LeSage & Pace 2009;
        §3.6 of the spatial PG paper) and :math:`\phi(\eta)` for the probit.
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
            return self._compute_probability_scale_impacts_posterior_sparse(
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
            w = _link_density(self._link, eta)

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

    def _compute_probability_scale_impacts_posterior_sparse(
        self,
        rho_draws: np.ndarray,
        beta_draws: np.ndarray,
        n: int,
        ni: list[int],
        n_draws: int,
        n_effects: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Probability-scale impacts via sparse solves + Hutchinson.

        Direct port of
        :meth:`SARNegBin._compute_count_scale_impacts_posterior_sparse`
        with the count-mean weight :math:`\mu` replaced by the link's density
        :math:`F'(\eta)`.
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

            w = _link_density(self._link, eta)

            multiplier_diag = np.mean(Z * AinvZ, axis=1)

            direct_base = float(np.mean(w * multiplier_diag))
            total_base = float(np.mean(w * multiplier_row_sums))

            direct_samples[draw_idx] = direct_base * beta[ni]
            total_samples[draw_idx] = total_base * beta[ni]

        indirect_samples = total_samples - direct_samples
        return direct_samples, indirect_samples, total_samples

    def _sar_binary_impacts(
        self, scale: str, return_posterior_samples: bool, method: str
    ):
        """Impacts on the latent index (``_index_scale``) or the probability scale.

        The latent design's probability-scale impacts integrate over the
        propagated disturbances (:meth:`._latent.LatentMixin._latent_impacts`).
        """
        if scale == self._index_scale or self._design == "latent":
            return self._latent_impacts(scale, return_posterior_samples)
        if scale != "probability":
            raise ValueError(
                f"scale must be either {self._index_scale!r} or 'probability'."
            )
        self._require_fit()
        return self._effects_frame(
            *self._compute_probability_scale_impacts_posterior(method=method),
            return_posterior_samples=return_posterior_samples,
            scale=scale,
        )
