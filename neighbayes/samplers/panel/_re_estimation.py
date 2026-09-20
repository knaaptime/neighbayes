"""GibbsEstimation-style orchestrator for RE panel Gibbs samplers.

Provides ``GaussianSARREGibbs`` and ``GaussianSEMREGibbs`` classes that
handle chain running, InferenceData assembly, and method dispatch for
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

from ..._lazy_deps import az

_log = logging.getLogger(__name__)

from .._utils._idata import gibbs_to_inference_data
from ..gaussian._chain_runner import run_chains
from ._re_core import (
    REGibbsCache,
    REGibbsPriors,
    _initialize_re_gibbs,
    _sem_re_unit_aggregated_terms,
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
        Names for the columns of X (for InferenceData coords).
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
    ) -> az.InferenceData:
        """Run Gibbs chains and assemble InferenceData.

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
        az.InferenceData
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

        # Derive per-chain seeds
        from .._utils._seeds import spawn_chain_seeds

        seeds = spawn_chain_seeds(random_seed, chains)

        # Define per-chain function
        def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
            rng = np.random.default_rng(seed)
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
                draws=draws,
                tune=tune,
                thin=thin,
                rng=rng,
                progressbar=progressbar,
                chain_id=chain_id_kw if chain_id_kw is not None else chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
            )

        # Run chains: n_jobs=1 → sequential, n_jobs≠1 → parallel via joblib
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
            model_type=f"re_{self.model_type}",
        )

        # Assemble InferenceData
        idata = self._assemble_idata(chain_results, log_likelihood=log_likelihood)
        elapsed = time.time() - t_start
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        return idata

    def _build_cache(self) -> REGibbsCache:
        """Build the REGibbsCache from model data."""
        from scipy.linalg import cho_factor

        XtX = self.X.T @ self.X
        XtX_cho = cho_factor(XtX)

        # SEM-RE α block: precompute the λ-independent terms of BᵀB once so
        # the sweep uses the closed form instead of rebuilding a dense NT × N
        # matrix each iteration (O(N²·NT) → O(N²)).
        sem_BtB_M1 = sem_BtB_M2 = None
        if self.model_type == "sem" and self.W_sparse is not None:
            sem_BtB_M1, sem_BtB_M2 = _sem_re_unit_aggregated_terms(
                self.W_sparse, self.unit_idx, self.N
            )

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
            sem_BtB_M1=sem_BtB_M1,
            sem_BtB_M2=sem_BtB_M2,
        )

    def _assemble_idata(
        self,
        chain_results: list[dict],
        *,
        log_likelihood: bool = False,
    ) -> az.InferenceData:
        """Convert chain output dicts to InferenceData.

        Parameters
        ----------
        chain_results : list of dict
            One dict per chain, each containing parameter trace arrays.

        Returns
        -------
        az.InferenceData
        """
        spatial_param = self._spatial_param_name()

        # Stack chain results
        posterior_samples = {}
        for key in [spatial_param, "sigma", "sigma_alpha"]:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)  # (chains, n_keep)

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack(
            [c["beta"] for c in chain_results], axis=0
        )  # (chains, n_keep, k)

        # alpha has shape (n_keep, N) per chain
        posterior_samples["alpha"] = np.stack(
            [c["alpha"] for c in chain_results], axis=0
        )  # (chains, n_keep, N)

        # Feature names for coords
        coords = {
            "coefficient": self.feature_names,
            "unit": list(range(self.N)),
        }
        dims = {
            "beta": ["coefficient"],
            "alpha": ["unit"],
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
        )

    def _spatial_param_name(self) -> str:
        return "lam"
