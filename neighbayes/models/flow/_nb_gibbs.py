"""Shared reduced-form PG-Gibbs driver for the NB flow models.

The four NB SAR flow classes (cross-section/panel x unrestricted/separable)
build identical caches, priors, and chain plumbing around the two
``negbin_reduced`` flow chain kernels; this module holds the single driver
they all delegate to from ``_fit_gibbs``.
"""

from __future__ import annotations

import numpy as np

from ..._lazy_deps import xr


def run_negbin_flow_gibbs(
    model,
    *,
    separable: bool,
    model_type: str,
    omega_size: int,
    T: int = 1,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int | None = None,
    progressbar: bool = True,
    n_jobs: int = -1,
    gibbs_backend: str = "numpy",
    krylov_reuse: bool = True,
    log_likelihood: bool = False,
) -> xr.DataTree:
    """Run the reduced-form PG-Gibbs sampler for an NB SAR flow model.

    Builds the cache, priors, and per-chain initial states from ``model``
    attributes, dispatches to :func:`run_chain_unrestricted` (3-rho) or
    :func:`run_chain_separable` (2-rho Kronecker), assembles the
    posterior into an :class:`xarray.DataTree`, and stores it on
    ``model._idata``.

    Parameters
    ----------
    model : FlowModel or FlowPanelModel
        NB flow model instance providing ``_X``, ``_y_int_vec``,
        ``_W_sparse``, ``_Wd``/``_Wo``/``_Ww``, ``_n``, ``priors``, and
        (unrestricted only) ``restrict_positive``.
    separable : bool
        If True use the separable Kronecker kernel (``rho_w = -rho_d rho_o``,
        box prior); otherwise the unrestricted 3-rho kernel.
    model_type : str
        Progress-display label passed to :func:`run_chains`.
    omega_size : int
        Length of the Polya-Gamma latent vector (number of flow
        observations; ``N_f * T`` for panels).
    T : int, default 1
        Number of panel periods sharing the per-period system matrix.
    """
    from ...samplers._utils._idata import gibbs_to_inference_data
    from ...samplers.gaussian._chain_runner import run_chains
    from ...samplers.negbin_reduced._flow import (
        FlowReducedGibbsCache,
        FlowReducedGibbsPriors,
        FlowReducedGibbsState,
        run_chain_unrestricted,
    )
    from ...samplers.negbin_reduced._flow_structured import (
        classify_flow_design,
        run_chain_separable_structured,
    )

    X = model._X
    y = model._y_int_vec.astype(np.float64)
    k = X.shape[1]
    W_csc = model._W_sparse.tocsc()

    # --- Build cache ---
    cache_kwargs: dict = dict(
        # The separable sampler works from the n×n W alone; building the
        # N×N Kronecker weights (W⊗W has nnz(W)² entries) is unrestricted-only.
        Wd=None if separable else model._Wd,
        Wo=None if separable else model._Wo,
        Ww=None if separable else model._Ww,
        W_csc=W_csc,
        n=model._n,
        separable=separable,
        rho_lower=model.priors.get("rho_lower", -0.999),
        rho_upper=model.priors.get("rho_upper", 0.999),
        T=T,
        krylov_reuse=krylov_reuse,
    )
    if not separable:
        cache_kwargs["positive"] = model.restrict_positive
    cache = FlowReducedGibbsCache(**cache_kwargs)

    # --- Build priors ---
    pv = model._flow_count_priors()
    priors = FlowReducedGibbsPriors(
        beta_mu=pv["beta_mu"],
        beta_sigma=pv["beta_sigma"],
        alpha_sigma=pv["alpha_sigma"],
        alpha_nu=pv["alpha_nu"],
        alpha_fixed=pv["alpha_fixed"],
        rho_lower=model.priors.get("rho_lower", -0.999),
        rho_upper=model.priors.get("rho_upper", 0.999),
    )

    # --- Build init state (per-chain, via closure) ---
    def _make_init(rng: np.random.Generator) -> FlowReducedGibbsState:
        beta0 = rng.normal(0.0, 0.1, size=k)
        if separable:
            rho_d0 = rng.uniform(-0.1, 0.1)
            rho_o0 = rng.uniform(-0.1, 0.1)
            rho_w0 = None
        else:
            rho_lo = 0.0 if model.restrict_positive else -0.1
            rho_d0 = rng.uniform(rho_lo, 0.1)
            rho_o0 = rng.uniform(rho_lo, 0.1)
            rho_w0 = rng.uniform(0.0 if model.restrict_positive else -0.05, 0.05)
        return FlowReducedGibbsState(
            beta=beta0,
            rho_d=rho_d0,
            rho_o=rho_o0,
            rho_w=rho_w0,
            alpha=1.0 if pv["alpha_fixed"] is None else pv["alpha_fixed"],
            omega=np.ones(omega_size, dtype=np.float64) * 0.5,
        )

    # Rank-one / full-rank split of the flow design, shared by every chain.
    struct = classify_flow_design(X, model._n, T) if separable else None

    # --- Chain function ---
    def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
        rng = np.random.default_rng(seed)
        init = _make_init(rng)
        common = dict(
            y=y,
            X=X,
            priors=priors,
            cache=cache,
            init=init,
            draws=draws,
            tune=tune,
            thin=1,
            rng=rng,
            chain_id=chain_id,
            progress_manager=progress_manager,
            store_log_lik=log_likelihood,
        )
        if separable:
            # Structured sweep (cross-section and panel): n × n solves only,
            # rank-one design columns integrated out of the ρ updates
            # (samplers/negbin_reduced/_flow_structured.py).  It replaces the
            # Krylov-basis kernel, whose cross-sweep basis reuse biased ρ.
            return run_chain_separable_structured(
                y,
                X,
                W_csc,
                model._n,
                priors,
                init,
                draws,
                tune,
                T=T,
                rho_lower=priors.rho_lower,
                rho_upper=priors.rho_upper,
                rng=rng,
                chain_id=chain_id,
                progress_manager=progress_manager,
                store_log_lik=log_likelihood,
                struct=struct,
            )
        return run_chain_unrestricted(
            Wd=model._Wd, Wo=model._Wo, Ww=model._Ww, **common
        )

    # --- Run chains ---
    if gibbs_backend == "jax" and T > 1 and not separable:
        raise NotImplementedError(
            "The unrestricted JAX flow Gibbs kernel is cross-section only; fit "
            "unrestricted panels with gibbs_backend='numpy'."
        )
    if gibbs_backend == "jax":
        from ...samplers._utils._seeds import seed_sequence_to_int, spawn_chain_seeds

        child_seeds = spawn_chain_seeds(random_seed, chains)
        seeds = [seed_sequence_to_int(s) for s in child_seeds]
        # One init per chain (same closure the NumPy path uses).
        inits = [_make_init(np.random.default_rng(s)) for s in seeds]

        if separable:
            # JAX port of the structured sweep (cross-section and panel).
            from ...samplers.negbin_reduced._flow_structured_jax import (
                run_chains_jax_flow_structured,
            )

            chain_results = run_chains_jax_flow_structured(
                y,
                W_csc,
                model._n,
                priors,
                inits,
                draws,
                tune,
                struct=struct,
                jax_seeds=seeds,
                store_log_lik=log_likelihood,
            )
        else:
            from ...samplers.negbin_reduced._flow_jax import run_chains_jax_flow

            chain_results = run_chains_jax_flow(
                y=y,
                X=X,
                Wd=model._Wd,
                Wo=model._Wo,
                Ww=model._Ww,
                priors=priors,
                inits=inits,
                draws=draws,
                tune=tune,
                positive=model.restrict_positive,
                n_cycles=cache.n_rho_omega_cycles,
                jax_seeds=seeds,
                progressbar=progressbar,
                krylov_reuse=krylov_reuse,
                store_log_lik=log_likelihood,
            )
    else:
        from ...samplers._utils._seeds import spawn_chain_seeds

        np_seeds = (
            spawn_chain_seeds(random_seed, chains) if random_seed is not None else None
        )
        chain_results = run_chains(
            chain_fn=_chain_fn,
            n_chains=chains,
            seeds=np_seeds,
            n_jobs=n_jobs,
            progressbar=progressbar,
            parallel=n_jobs != 1,
            draws=draws,
            tune=tune,
            model_type=model_type,
        )

    # --- Assemble DataTree ---
    posterior_samples = {
        "rho_d": np.stack([c["rho_d"] for c in chain_results], axis=0),
        "rho_o": np.stack([c["rho_o"] for c in chain_results], axis=0),
        "rho_w": np.stack([c["rho_w"] for c in chain_results], axis=0),
        "beta": np.stack([c["beta"] for c in chain_results], axis=0),
        "alpha": np.stack([c["alpha"] for c in chain_results], axis=0),
    }
    ll = None
    if log_likelihood:
        ll = {"obs": np.stack([c["log_lik"] for c in chain_results], axis=0)}
    coords = {"coefficient": list(model._feature_names)}
    dims = {"beta": ["coefficient"]}

    model._idata = gibbs_to_inference_data(
        posterior_samples=posterior_samples,
        log_likelihood=ll,
        observed_data={"obs": model._y_int_vec},
        coords=coords,
        dims=dims,
    )
    return model._idata
