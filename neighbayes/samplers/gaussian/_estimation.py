"""GibbsEstimation base class for Gaussian spatial Gibbs samplers.

Orchestrates chain running, InferenceData assembly, and method
dispatch for the 3-block Gaussian Gibbs sampler (β, σ², ρ/λ).

Two execution backends are supported:

- ``gibbs_method="jax"`` (default): Full-JIT Gibbs with
  slice sampling for ρ/λ.  Requires JAX and equinox.
  Falls back to ``"numpy"`` when JAX is not installed.
- ``gibbs_method="numpy"``: Python-loop Gibbs with adaptive
  slice sampling for ρ/λ.  No JAX dependency.

Subclasses implement model-specific logic:
- ``_spatial_param_name()``: "rho" or "lam"
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod

import numpy as np
import scipy.sparse as sp

from ..._lazy_deps import az
from ..._logdet._refit import DEFAULT_PAD_SD
from ..._logdet._warmup import WarmupJacobian

_log = logging.getLogger(__name__)

from .._utils._idata import gibbs_to_inference_data
from ._chain_runner import run_chains
from ._core import (
    GaussianGibbsCache,
    GaussianGibbsPriors,
    _initialize_gaussian_gibbs,
    run_gaussian_chain,
)


class GibbsEstimation:
    """Base class for Gaussian spatial Gibbs sampler configuration and execution.

    Encapsulates the data, priors, cache, and chain-running logic for
    the 3-block Gibbs sampler (β, σ², ρ/λ).  Subclasses provide
    model-specific details (SAR vs SEM, collapsed vs un-collapsed ρ/λ).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDM/SDEM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable for arrays of rho values.
    feature_names : list of str
        Names for the columns of X (for InferenceData coords).
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        Wy: np.ndarray | None,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str,
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
        logdet_refit: bool = False,
        logdet_refit_pad_sd: float = DEFAULT_PAD_SD,
        logdet_aaa_check: bool = False,
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
        self.W_eigs = W_eigs
        self.logdet_method = logdet_method
        self.T = int(T)
        self.logdet_refit = bool(logdet_refit)
        self.logdet_refit_pad_sd = float(logdet_refit_pad_sd)
        self.logdet_aaa_check = bool(logdet_aaa_check)
        self.warmup_jacobian = None
        self.n, self.k = X.shape

    def __getstate__(self):
        """Pickle without the warmup log-determinant.

        NumPy chains run in worker processes that receive ``self`` through the
        chain closures.  Workers never adapt, and the warmup log-determinant holds
        a factorization context and the weights again, so it stays behind.
        """
        state = self.__dict__.copy()
        state["warmup_jacobian"] = None
        return state

    def fit(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = -1,
        progressbar: bool = True,
        gibbs_method: str = "jax",
        slice_width: float | None = None,
        chain_method: str | None = None,
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
            Number of parallel workers for the NumPy path.
            ``-1`` uses all CPUs.  When ``n_jobs=1``, chains run
            sequentially with progress bars.  When ``n_jobs>1``
            (or ``-1``), chains run in parallel via ``joblib``.
            Ignored for the JAX path (use ``chain_method`` instead).
        progressbar : bool, default True
            Show per-chain progress bars.
        gibbs_method : str, default "jax"
            Execution backend: ``"jax"`` for full-JIT Gibbs with
            slice sampling for ρ/λ (default, falls back to ``"numpy"``
            when JAX is not installed), or ``"numpy"`` for Python-loop
            Gibbs with adaptive slice sampling.
        slice_width : float or None, default None
            Initial step-out width for the ρ/λ slice sampler on the JAX
            path.  If None, defaults to ``(rho_upper - rho_lower) * 0.1``.
            Ignored when ``gibbs_method="numpy"`` (the NumPy path adapts
            its own slice width).
        chain_method : str or None, default None
            How to run multiple chains for the JAX path.  Chains always run
            in parallel, vectorized under ``jax.vmap`` on one device, so
            ``"vectorized"`` is the only value.  Ignored for the NumPy path
            (use ``n_jobs`` to control parallelism instead).

        Returns
        -------
        az.InferenceData
            With ``posterior``, ``log_likelihood``, and ``observed_data``
            groups.
        """
        # Default chain_method for JAX path
        if chain_method is None:
            chain_method = "vectorized" if gibbs_method == "jax" else None

        if gibbs_method == "jax":
            return self._fit_jax(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                thin=thin,
                n_jobs=n_jobs,
                progressbar=progressbar,
                slice_width=slice_width,
                chain_method=chain_method,
            )

        # ── NumPy path (default) ──
        # Build cache
        cache = self._build_cache()
        self.warmup_jacobian = None  # per-run result; never carry one fit into the next

        spatial_param = self._spatial_param_name()
        _log.info(f"Gibbs sampling ({chains} chains, 3-block: β, σ², {spatial_param})")
        t_start = time.time()

        # Derive per-chain seeds
        from .._utils._seeds import spawn_chain_seeds

        # extra=1 for the scouting phase (index 0), chains at indices 1..n
        all_seeds = spawn_chain_seeds(random_seed, chains, extra=1)
        # Each chain scouts on its own stream, spawned from the seed the scouting
        # phase reserves, so the chains stay distinct through the first half of
        # warmup and resume from distinct states.
        scouting_seeds = all_seeds[0].spawn(chains)
        seeds = all_seeds[1:]  # list[SeedSequence]

        parallel = n_jobs != 1

        def _chain_fn(init_by_chain, phase_draws, phase_tune, *, scouting=False):
            def _run_one_chain(chain_id, seed, progress_manager=None, chain_id_kw=None):
                # A scouting phase consumed the base seed, so the phase that
                # follows it must not reuse it.  Runs without a refit keep the
                # original seeding exactly.
                if scouting:
                    rng = np.random.default_rng(scouting_seeds[chain_id])
                else:
                    rng = np.random.default_rng(seed)
                init = init_by_chain[chain_id]
                if init is None:
                    init = _initialize_gaussian_gibbs(
                        self.y, self.X, cache.XtX_cho, self.priors, rng
                    )
                return run_gaussian_chain(
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
                    return_state=scouting,
                    # Scouting draws are discarded except for ρ; computing a
                    # pointwise log-likelihood for them would add an O(n·k) pass
                    # per iteration and an (iters × n) array per chain to pickle
                    # back from every worker.
                    store_log_lik=not scouting,
                )

            return _run_one_chain

        inits: list = [None] * chains
        tune_remaining = tune

        # ── Optional warmup phase A: run half of warmup, then rebuild the
        # Jacobian interpolant on the ρ range the chains actually found.  The
        # refit is frozen here, before any retained draw, so the kernel that
        # produces the posterior is fixed — the same discipline as step-size or
        # slice-width adaptation.  The remaining warmup runs under the refit
        # interpolant, which doubles as a check that the window holds.
        wj = self._warmup_jacobian()
        if cache.logdet_fn is None and wj is None:
            raise RuntimeError(
                "No log-determinant evaluator: the model skipped building one for a "
                "warmup adaptation that is not going to happen. This is a wiring bug: "
                "sampler_builds_evaluators and GibbsEstimation._warmup_jacobian disagree."
            )
        if wj is not None:
            # Warmup starts on a deliberately cheap interpolant (see
            # WarmupJacobian.initial).  Its draws are discarded, so it only has to
            # steer the chains to the right neighborhood.
            start = wj.initial(tune)
            cache.logdet_fn, cache.logdet_vec_fn = start.scalar_fn, start.vec_fn
        if wj is not None and wj.adapts(tune):
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
                model_type=self.model_type,
            )
            inits = [c["_final_state"] for c in warm]
            tune_remaining = tune - tune_a
            # Chains start at ρ = 0 and walk to the posterior, so the first
            # part of phase A is transient.  Including it would stretch the
            # window back to the initial value and defeat the refit; the
            # second half is what the chains have actually settled on.
            adapted = wj.adapt(
                np.concatenate([c[spatial_param][tune_a // 2 :] for c in warm])
            )
            if adapted is not None:
                cache.logdet_fn, cache.logdet_vec_fn = adapted.scalar_fn, adapted.vec_fn
                if wj.window is not None:
                    # The interpolant is only valid on its interval (a Chebyshev
                    # series diverges outside it), so the support follows the refit.
                    cache.rho_lower, cache.rho_upper = adapted.rho_min, adapted.rho_max

        chain_results = run_chains(
            chain_fn=_chain_fn(inits, draws, tune_remaining),
            n_chains=chains,
            seeds=seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=parallel,
            draws=draws,
            tune=tune_remaining,
            model_type=self.model_type,
            # A scouting phase just spawned this pool with the same worker
            # count; respawning it would re-pay the process launch and
            # re-pickle y/X/W for every worker.
            reuse_workers=inits[0] is not None,
        )

        # Assemble InferenceData
        idata = self._assemble_idata(chain_results)
        self._record_refit(idata, chain_results, spatial_param)
        elapsed = time.time() - t_start
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        return idata

    def _fit_jax(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        random_seed: int | None = None,
        thin: int = 1,
        n_jobs: int = 1,
        progressbar: bool = True,
        slice_width: float | None = None,
        chain_method: str = "vectorized",
    ) -> az.InferenceData:
        """Run JAX JIT Gibbs chains and assemble InferenceData.

        Uses slice sampling for the ρ/λ update, enabling full JIT
        compilation of the Gibbs step.

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
        n_jobs : int, default 1
            Ignored: the chains run vectorized under ``jax.vmap``.
        progressbar : bool, default True
            Show per-chain progress bars.
        slice_width : float or None, default None
            Initial step-out width for the ρ/λ slice sampler.  If None,
            defaults to ``(rho_upper - rho_lower) * 0.1``.
        chain_method : str, default "vectorized"
            Chains always run in parallel, vectorized under ``jax.vmap`` on
            one device.  ``"parallel"`` is not supported.

        Returns
        -------
        az.InferenceData
        """
        from ._jax import run_chains_jax_gibbs_vectorized

        if chain_method == "parallel":
            raise NotImplementedError(
                "chain_method='parallel' is not supported for the JAX path. "
                "Use chain_method='vectorized' for JAX-native parallelism."
            )
        if chain_method != "vectorized":
            raise ValueError(
                f"Unknown chain_method {chain_method!r}; use 'vectorized'."
            )

        self.warmup_jacobian = None
        # Build JAX-native logdet function
        # The refit path carries the interpolant as traced state, and the step
        # then ignores any closed-over evaluator — so building one would be a
        # full precompute (a Cholesky factorization per node) thrown away.
        # The refit and the AAA node check pool ρ across chains halfway through
        # warmup, inside the vectorized runner.
        wj = self._warmup_jacobian()
        param_fn = params0 = refit_hook = None
        if wj is not None:
            param_fn = wj.param_fn()
            start = wj.initial(tune, jax=True)
            params0 = start.params
            # The post-chain pointwise log-likelihood needs a NumPy evaluator of
            # the same fit, and the model built none.
            self.logdet_vec_fn = start.vec_fn
            if wj.adapts(tune):
                refit_hook = self._jax_adaptation_hook(wj)
            logdet_jax = None
        elif self.logdet_vec_fn is None:
            raise RuntimeError(
                "No log-determinant evaluator: the model skipped building one for a "
                "warmup adaptation that is not going to happen. This is a wiring bug: "
                "sampler_builds_evaluators and GibbsEstimation._warmup_jacobian disagree."
            )
        else:
            logdet_jax = self._build_logdet_jax()
        # A refit that replaces the interpolant must also replace the evaluator
        # the post-chain pointwise log-likelihood uses, and that evaluator is
        # passed to the runner before the refit happens — so pass a late-binding
        # indirection rather than the function itself.
        self._active_logdet_vec_fn = self.logdet_vec_fn
        logdet_vec_fn = (
            self.logdet_vec_fn
            if refit_hook is None
            else (lambda a: self._active_logdet_vec_fn(a))
        )

        spatial_param = self._spatial_param_name()
        method_str = " (vectorized)"
        _log.info(
            f"JAX Gibbs sampling{method_str} ({chains} chains, slice, "
            f"3-block: β, σ², {spatial_param})"
        )
        t_start = time.time()

        # ── Vectorized path: jax.vmap ──
        # Derive per-chain seeds
        from .._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]

        # Build cache for initialization
        cache = self._build_cache()

        # Initialize per-chain states
        inits = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            init = _initialize_gaussian_gibbs(
                self.y,
                self.X,
                cache.XtX_cho,
                self.priors,
                rng,
            )
            inits.append(init)

        chain_results = run_chains_jax_gibbs_vectorized(
            y=self.y,
            X=self.X,
            W_sparse=self.W_sparse,
            Wy=self.Wy,
            logdet_jax=logdet_jax,
            logdet_vec_fn=logdet_vec_fn,
            priors=self.priors,
            inits=inits,
            draws=draws,
            tune=tune,
            thin=thin,
            jax_seeds=seeds,
            model_type=self.model_type,
            slice_width=slice_width,
            progressbar=progressbar,
            logdet_param_fn=param_fn,
            logdet_params=params0,
            refit_hook=refit_hook,
        )

        # Assemble InferenceData
        idata = self._assemble_idata(chain_results)
        self._record_refit(idata, chain_results, spatial_param)
        elapsed = time.time() - t_start
        _log.info(
            f"Sampling {chains} chains for {tune} tune and {draws} draw "
            f"iterations ({chains * tune:,} + {chains * draws:,} draws total) "
            f"took {elapsed:.0f} seconds."
        )
        return idata

    # ------------------------------------------------------------------
    # Warmup adaptation of the log-determinant
    # ------------------------------------------------------------------

    def _warmup_jacobian(self) -> WarmupJacobian | None:
        """The warmup log-determinant for this run, or ``None`` when nothing adapts.

        Construction is lazy, with no factorization until evaluators are built, so
        this is cheap to call unconditionally.
        """
        if self.W_sparse is None or not (self.logdet_refit or self.logdet_aaa_check):
            return None
        wj = WarmupJacobian.for_sampler(
            self.W_sparse,
            self.logdet_method,
            T=self.T,
            rho_min=self.priors.rho_lower,
            rho_max=self.priors.rho_upper,
            refit=self.logdet_refit,
            refit_pad_sd=self.logdet_refit_pad_sd,
            aaa_check=self.logdet_aaa_check,
        )
        if not wj.active:
            return None
        self.warmup_jacobian = wj
        return wj

    @property
    def refit_window(self):
        """The refit window of the most recent fit, or ``None``."""
        wj = self.warmup_jacobian
        return None if wj is None else wj.window

    @property
    def aaa_check(self):
        """The AAA node check of the most recent fit, or ``None``."""
        wj = self.warmup_jacobian
        return None if wj is None else wj.check

    def _jax_adaptation_hook(self, wj: WarmupJacobian):
        """The hook a JAX runner calls halfway through warmup, with ρ pooled across chains.

        Returns ``(params, rho_min, rho_max)`` for the adapted fit, or ``None`` to keep
        the installed one.  The parameters keep the shape of the warmup fit's, so the
        compiled step is reused.  The retained draws are produced under the adapted
        fit, so their pointwise log-likelihood switches to its NumPy evaluator.
        """

        def _hook(pooled_rho):
            adapted = wj.adapt(pooled_rho, jax=True)
            if adapted is None:
                return None
            self._active_logdet_vec_fn = adapted.vec_fn
            return adapted.params, adapted.rho_min, adapted.rho_max

        return _hook

    def _record_refit(self, idata, chain_results, spatial_param: str) -> None:
        """Attach the warmup adaptation to ``idata``; see :meth:`WarmupJacobian.record`."""
        if self.warmup_jacobian is not None:
            self.warmup_jacobian.record(
                idata, np.concatenate([c[spatial_param] for c in chain_results])
            )

    def _build_logdet_jax(self) -> callable:
        """Build a JAX-native logdet callable for the JAX Gibbs path.

        Uses ``make_logdet_jax_fn`` from ``neighbayes.logdet`` with the
        model's eigenvalues (if available) or sparse W matrix.

        The panel Jacobian is ``T·log|I_N − ρW|``, applied by passing the
        per-period ``N×N`` weights with ``T=self.T``.  For panels the sampler
        receives the ``NT×NT`` block-diagonal lag matrix (``I_T ⊗ W``) as
        ``self.W_sparse`` — whose determinant *already* carries the ``T``
        replication — so the per-period block ``W[:N, :N]`` is extracted first to
        avoid a ``T²`` double-count (``W_eigs`` is already length ``N``; a
        cross-section has ``T=1`` and the slice is a no-op).

        Returns
        -------
        callable
            JAX-native logdet function ``(rho) -> jax.numpy.ndarray``.
        """
        from ..._logdet import make_logdet_jax_fn

        # Use eigenvalues if available (fastest for JAX path)
        if self.W_eigs is not None:
            W_input = self.W_eigs
        else:
            W = self.W_sparse
            n_units = W.shape[0] // self.T  # per-period unit count
            W_input = W[:n_units, :n_units] if self.T > 1 else W

        return make_logdet_jax_fn(
            W=W_input,
            method=self.logdet_method,
            rho_min=self.priors.rho_lower,
            rho_max=self.priors.rho_upper,
            T=self.T,
        )

    def _build_cache(self) -> GaussianGibbsCache:
        """Build the GibbsCache from model data."""
        from scipy.linalg import cho_factor

        XtX = self.X.T @ self.X
        XtX_cho = cho_factor(XtX)

        # Precompute Wy for all model types (SAR uses it directly,
        # SEM/SDEM use it for sigma2 and collapsed density)
        Wy = self.Wy if self.Wy is not None else self.W_sparse @ self.y

        # Precompute y-side inner products for ALL model types.
        # SAR/SDM need them for the O(k) collapsed ρ density (Phase 1 perf:
        # eliminates O(nk) per slice evaluation).  SEM/SDEM need them for
        # both the collapsed λ density and the β block (Phase 2 perf:
        # eliminates O(nk²) per Gibbs iteration).
        yty = float(self.y @ self.y)
        yTWy = float(self.y @ Wy)
        WyTWy = float(Wy @ Wy)
        XTy = self.X.T @ self.y  # (k,)
        XTWy = self.X.T @ Wy  # (k,)

        # WX-side quantities only needed for SEM/SDEM
        WX = None
        XtWX = None
        WXtWX = None
        WXTy = None
        WXTWy = None
        if self.model_type in ("sem", "sdem"):
            WX = self.W_sparse @ self.X  # (n, k)
            XtWX = self.X.T @ WX  # (k, k)
            WXtWX = WX.T @ WX  # (k, k)
            WXTy = WX.T @ self.y  # (k,)
            WXTWy = WX.T @ Wy  # (k,)

        return GaussianGibbsCache(
            XtX=XtX,
            XtX_cho=XtX_cho,
            logdet_fn=self.logdet_fn,
            logdet_vec_fn=self.logdet_vec_fn,
            rho_lower=self.priors.rho_lower,
            rho_upper=self.priors.rho_upper,
            model_type=self.model_type,
            Wy=Wy,
            W_sparse=self.W_sparse,
            WX=WX,
            XtWX=XtWX,
            WXtWX=WXtWX,
            yty=yty,
            yTWy=yTWy,
            WyTWy=WyTWy,
            XTy=XTy,
            XTWy=XTWy,
            WXTy=WXTy,
            WXTWy=WXTWy,
        )

    def _assemble_idata(
        self,
        chain_results: list[dict],
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
        for key in [spatial_param, "sigma"]:
            arrays = [c[key] for c in chain_results]
            posterior_samples[key] = np.stack(arrays, axis=0)  # (chains, n_keep)

        # Also expose sigma² so downstream consumers (e.g. bridge sampling) can
        # evaluate the PyMC model logp, which treats sigma² as the free RV.
        posterior_samples["sigma2"] = posterior_samples["sigma"] ** 2

        # beta has shape (n_keep, k) per chain
        posterior_samples["beta"] = np.stack(
            [c["beta"] for c in chain_results], axis=0
        )  # (chains, n_keep, k)

        # Feature names for coords
        coords = {"coefficient": self.feature_names}
        dims = {"beta": ["coefficient"]}

        # Log-likelihood: shape (chains, n_keep, n)
        log_lik = np.stack(
            [c["log_lik"] for c in chain_results], axis=0
        )  # (chains, n_keep, n)

        # Sample stats: per-draw joint log-likelihood and acceptance rate.
        # ``lp`` is the sum of the pointwise log-likelihood (which already
        # includes the Jacobian correction for SAR/SEM); broadcasting the
        # per-chain ``mh_accept_rate`` scalar across draws gives ArviZ a
        # uniform ``(chain, draw)``-shaped stat without per-step tracking.
        n_keep = log_lik.shape[1]
        lp = log_lik.sum(axis=-1)  # (chains, n_keep)
        accept_per_chain = np.array(
            [c.get("mh_accept_rate", 1.0) for c in chain_results],
            dtype=np.float64,
        )
        acceptance_rate = np.broadcast_to(
            accept_per_chain[:, None], (len(chain_results), n_keep)
        ).copy()
        sample_stats = {"lp": lp, "acceptance_rate": acceptance_rate}

        idata = gibbs_to_inference_data(
            posterior_samples=posterior_samples,
            log_likelihood={"obs": log_lik},
            observed_data={"obs": self.y},
            coords=coords,
            dims=dims,
            sample_stats=sample_stats,
        )

        return idata

    @abstractmethod
    def _spatial_param_name(self) -> str:
        """Return the name of the spatial parameter ('rho' or 'lam')."""
        ...


class GaussianSARGibbs(GibbsEstimation):
    """Gibbs sampler for SAR/SDM Gaussian models.

    3-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    ρ (collapsed slice sampling).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - rho*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    model_type : str
        "sar" or "sdm".
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
        Wy: np.ndarray,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str = "sar",
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
        logdet_refit: bool = False,
        logdet_refit_pad_sd: float = DEFAULT_PAD_SD,
        logdet_aaa_check: bool = False,
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
            model_type=model_type,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            T=T,
            logdet_refit=logdet_refit,
            logdet_refit_pad_sd=logdet_refit_pad_sd,
            logdet_aaa_check=logdet_aaa_check,
        )

    def _spatial_param_name(self) -> str:
        return "rho"


class GaussianSEMGibbs(GibbsEstimation):
    """Gibbs sampler for SEM/SDEM Gaussian models.

    3-block sampler: β (conjugate normal), σ² (conjugate Inv-Γ),
    λ (conditional slice sampling).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix (for SDEM, this is [X, WX]).
    W_sparse : csr_matrix of shape (n, n)
        Row-standardized spatial weights matrix.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    logdet_fn : callable
        log|I - lam*W| callable (numpy scalar).
    logdet_vec_fn : callable
        Vectorized logdet callable.
    feature_names : list of str
        Names for the columns of X.
    model_type : str
        "sem" or "sdem".
    W_eigs : ndarray or None
        Real eigenvalues of W (for JAX logdet).
    logdet_method : str or None
        Logdet method for JAX path (auto-selected when None).
    T : int, default 1
        Panel time-period count.
    """

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        W_sparse: sp.csr_matrix,
        priors: GaussianGibbsPriors,
        logdet_fn: callable,
        logdet_vec_fn: callable,
        feature_names: list[str],
        model_type: str = "sem",
        W_eigs: np.ndarray | None = None,
        logdet_method: str | None = None,
        T: int = 1,
        logdet_refit: bool = False,
        logdet_refit_pad_sd: float = DEFAULT_PAD_SD,
        logdet_aaa_check: bool = False,
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
            model_type=model_type,
            W_eigs=W_eigs,
            logdet_method=logdet_method,
            T=T,
            logdet_refit=logdet_refit,
            logdet_refit_pad_sd=logdet_refit_pad_sd,
            logdet_aaa_check=logdet_aaa_check,
        )

    def _spatial_param_name(self) -> str:
        return "lam"
