"""GibbsEstimation-style orchestrator for RE panel Gibbs samplers.

Provides ``GaussianSARREGibbs`` and ``GaussianSEMREGibbs`` classes that
handle chain running, DataTree assembly, and method dispatch for
the 5-block RE panel Gibbs sampler (β, σ², α, σ_α², ρ/λ).

The architecture mirrors ``_gibbs_estimation.py`` for the FE (3-block)
Gibbs sampler, but adds the α and σ_α² blocks for random effects.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import xr

_log = logging.getLogger(__name__)

from .._utils._idata import gibbs_to_inference_data
from ..gaussian._chain_runner import run_chains
from ._re_core import (
    REGibbsCache,
    REGibbsPriors,
    _initialize_re_gibbs,
    _sem_re_alpha_structure,
    run_re_chain,
)


class REGibbsEstimation:
    """Base class for RE panel Gibbs sampler configuration and execution.

    Encapsulates the data, priors, cache, and chain-running logic for
    the 5-block RE Gibbs sampler (β, σ², α, σ_α², ρ/λ).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR).
    priors : REGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable for arrays of rho values.
    feature_names : list of str
        Names for the columns of X (for DataTree coords).
    model_type : str
        One of "sar", "sem".
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method for JAX path (auto-selected when None).
    logdet_adapt : bool, default True
        Size a stochastic log-determinant's probe pool halfway through warmup
        (:class:`~neighbayes._logdet._probe_check.WarmupProbes`); ``logdet_fn``
        and ``logdet_vec_fn`` may then be ``None``.
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        Wy: np.ndarray | None,
        priors: REGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str,
        N: int,
        T: int,
        unit_idx: np.ndarray,
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        logdet_adapt: bool = True,
    ):
        self.y = y
        self.X = X
        self.W_sparse = W_sparse
        self.Wy = Wy
        self.priors = priors
        self.logdet_fn = logdet_fn
        self.logdet_vec_fn = logdet_vec_fn
        self.feature_names = feature_names
        self.model_type = model_type
        self.N = N
        self.T = T
        self.unit_idx = unit_idx
        self.W_eigs = W_eigs
        self.logdet_method = logdet_method
        self.logdet_adapt = bool(logdet_adapt)
        self.warmup_probes = None
        self.n, self.k = X.shape

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Run Gibbs chains and assemble DataTree.

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
            Number of parallel workers. ``-1`` uses all CPUs.
        progressbar : bool, default True
            Show per-chain progress bars.

        Returns
        -------
        xr.DataTree
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.
        """
        # Build cache
        cache = self._build_cache()

        spatial_param = self._spatial_param_name()
        _log.info(
            f"RE Gibbs sampling ({chains} chains, 5-block: β, σ², α, σ_α², {spatial_param})"
        )
        t_start = time.time()

        # Derive per-chain seeds; a scouting phase draws from the extra one, so
        # runs without it keep their seeding.
        from .._utils._seeds import spawn_chain_seeds

        all_seeds = spawn_chain_seeds(random_seed, chains, extra=1)
        seeds = all_seeds[:chains]
        scouting_seeds = all_seeds[chains].spawn(chains)

        def _chain_fn(inits, phase_draws, phase_tune, *, scouting=False):
            def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
                rng = np.random.default_rng(
                    scouting_seeds[chain_id] if scouting else seed
                )
                init = inits[chain_id]
                if init is None:
                    init = _initialize_re_gibbs(
                        self.y,
                        self.X,
                        cache.XtX_cho,
                        self.N,
                        self.T,
                        self.unit_idx,
                        self.priors,
                        rng,
                    )
                return run_re_chain(
                    y=self.y,
                    X=self.X,
                    cache=cache,
                    priors=self.priors,
                    init=init,
                    draws=phase_draws,
                    tune=phase_tune,
                    thin=thin,
                    rng=rng,
                    progressbar=progressbar,
                    chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                    progress_manager=progress_manager,
                    store_log_lik=log_likelihood and not scouting,
                    return_state=scouting,
                )

            return _run_one_chain

        parallel = n_jobs != 1
        inits: list = [None] * chains
        tune_remaining = tune
        # A stochastic log-determinant starts warmup on a small probe pool and
        # sizes it halfway, on where the chains settled, before any retained
        # draw (see GibbsEstimation.fit).
        wp = self.warmup_probes = self._warmup_probes()
        if wp is not None:
            start = wp.initial(tune)
            cache.logdet_fn, cache.logdet_vec_fn = start.scalar_fn, start.vec_fn
        if wp is not None and wp.adapts(tune):
            tune_a = tune // 2
            warm = run_chains(
                chain_fn=_chain_fn(inits, tune_a, 0, scouting=True),
                n_chains=chains,
                seeds=seeds,
                n_jobs=n_jobs,
                progressbar=False,
                parallel=parallel,
                draws=tune_a,
                tune=0,
                model_type=f"re_{self.model_type}",
            )
            inits = [c["_final_state"] for c in warm]
            tune_remaining = tune - tune_a
            adapted = wp.adapt(
                np.stack([c[spatial_param][tune_a // 2 :] for c in warm]),
                draws=draws,
                chains=chains,
            )
            if adapted is not None:
                cache.logdet_fn, cache.logdet_vec_fn = adapted.scalar_fn, adapted.vec_fn

        chain_results = run_chains(
            chain_fn=_chain_fn(inits, draws, tune_remaining),
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=parallel,
            draws=draws,
            tune=tune_remaining,
            model_type=f"re_{self.model_type}",
            # The scouting phase just spawned this pool with the same workers.
            reuse_workers=inits[0] is not None,
        )

        # Assemble DataTree
        idata = self._assemble_idata(chain_results, log_likelihood=log_likelihood)
        if wp is not None:
            wp.record(idata, np.concatenate([c[spatial_param] for c in chain_results]))
        elapsed = time.time() - t_start
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        return idata

    def _warmup_probes(self):
        """The probe check of a stochastic log-determinant, or ``None`` for any other."""
        if not self.logdet_adapt or self.logdet_method != "stoch_cheb":
            return None
        from ..._logdet._probe_check import WarmupProbes

        return WarmupProbes.for_sampler(
            self.W_sparse,
            T=self.T,
            rho_min=self.priors.rho_lower,
            rho_max=self.priors.rho_upper,
        )

    def _build_cache(self) -> REGibbsCache:
        """Build the REGibbsCache from model data."""
        from scipy.linalg import cho_factor

        XtX = self.X.T @ self.X
        XtX_cho = cho_factor(XtX)

        # SEM-RE: the λ-independent pieces of the sparse α precision, on one
        # fixed pattern so each chain's CHOLMOD factor is analyzed once.
        sem_alpha = None
        if self.model_type == "sem" and self.W_sparse is not None:
            sem_alpha = _sem_re_alpha_structure(self.W_sparse, self.unit_idx, self.N)

        return REGibbsCache(
            XtX=XtX,
            XtX_cho=XtX_cho,
            logdet_fn=self.logdet_fn,
            logdet_vec_fn=self.logdet_vec_fn,
            rho_lower=self.priors.rho_lower,
            rho_upper=self.priors.rho_upper,
            model_type=self.model_type,
            Wy=self.Wy,
            W_sparse=self.W_sparse,
            N=self.N,
            T=self.T,
            unit_idx=self.unit_idx,
            unit_counts=np.bincount(self.unit_idx, minlength=self.N).astype(np.float64),
            sem_alpha=sem_alpha,
        )

    def _assemble_idata(
        self,
        chain_results: list[dict],
        *,
        log_likelihood: bool = False,
    ) -> xr.DataTree:
        """Convert chain output dicts to DataTree.

        Parameters
        ----------
        chain_results : list of dict
            One dict per chain, each containing parameter trace arrays.

        Returns
        -------
        xr.DataTree
        """
        spatial_param = self._spatial_param_name()

        # Stack chain results
        posterior_samples = {}
        for key in [spatial_param, "sigma", "unit_effect_sd"]:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)  # (chains, n_keep)

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack(
            [c["beta"] for c in chain_results], axis=0
        )  # (chains, n_keep, k)

        # alpha has shape (n_keep, N) per chain
        posterior_samples["unit_effect"] = np.stack(
            [c["unit_effect"] for c in chain_results], axis=0
        )  # (chains, n_keep, N)

        # Feature names for coords
        coords = {
            "coefficient": self.feature_names,
            "unit": list(range(self.N)),
        }
        dims = {
            "beta": ["coefficient"],
            "unit_effect": ["unit"],
        }

        # Log-likelihood, shape (chains, n_keep, n), only when requested.
        ll = None
        if log_likelihood:
            ll = {"obs": np.stack([c["log_lik"] for c in chain_results], axis=0)}

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood=ll,
            observed_data={"obs": self.y},
            coords=coords,
            dims=dims,
        )

        return idata

    @abstractmethod
    def _spatial_param_name(self) -> str:
        """Return the name of the spatial parameter ('rho' or 'lam')."""
        ...


class GaussianSARREGibbs(REGibbsEstimation):
    """Gibbs sampler for SAR panel model with random effects.

    5-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    α (conjugate normal, vectorized), σ_α² (conjugate Inv-Γ),
    ρ (collapsed slice sampling).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    priors : REGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method (auto-selected when None).
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        Wy: np.ndarray,
        priors: REGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        N: int,
        T: int,
        unit_idx: np.ndarray,
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        logdet_adapt: bool = True,
    ):
        super().__init__(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=Wy,
            priors=priors,
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            feature_names=feature_names,
            model_type="sar",
            N=N,
            T=T,
            unit_idx=unit_idx,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            logdet_adapt=logdet_adapt,
        )

    def _spatial_param_name(self) -> str:
        return "rho"


class GaussianSEMREGibbs(REGibbsEstimation):
    """Gibbs sampler for SEM panel model with random effects.

    5-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    α (conjugate normal, vectorized), σ_α² (conjugate Inv-Γ),
    λ (collapsed slice sampling).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    priors : REGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - lam*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    unit_idx : ndarray of shape (n,)
        Maps observation index to unit index.
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method (auto-selected when None).
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        priors: REGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        N: int,
        T: int,
        unit_idx: np.ndarray,
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        logdet_adapt: bool = True,
    ):
        super().__init__(
            y=y,
            X=X,
            W_sparse=W_sparse,
            Wy=None,  # SEM doesn't use Wy
            priors=priors,
            logdet_fn=logdet_fn,
            logdet_vec_fn=logdet_vec_fn,
            feature_names=feature_names,
            model_type="sem",
            N=N,
            T=T,
            unit_idx=unit_idx,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            logdet_adapt=logdet_adapt,
        )

    def _spatial_param_name(self) -> str:
        return "lam"
