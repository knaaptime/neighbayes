"""Shared auxiliary-mixture Gibbs driver for the Poisson flow models.

Mirrors :mod:`._nb_gibbs` for the Poisson likelihood.  The two differ only in
the augmentation: where the NB driver seeds a Pólya–Gamma latent vector, this
one lets the sampler build its own ragged augmented design (one row per
observation, plus one more per strictly positive count), so there is no
``omega_size`` to pass.  There is likewise no ``alpha`` in the posterior —
Poisson has no free dispersion parameter.
"""

from __future__ import annotations

import numpy as np

from ..._lazy_deps import az


def run_poisson_flow_gibbs(
    model,
    *,
    separable: bool,
    model_type: str,
    T: int = 1,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    random_seed: int | None = None,
    progressbar: bool = True,
    n_jobs: int = -1,
    log_likelihood: bool = False,
) -> az.InferenceData:
    """Run the reduced-form auxiliary-mixture sampler for a Poisson flow model.

    Parameters
    ----------
    model : FlowModel
        Poisson flow model instance providing ``_X``, ``_y_int_vec``,
        ``_W_sparse``, ``_Wd``/``_Wo``/``_Ww``, ``_n``, ``priors``, and
        (unrestricted only) ``restrict_positive``.
    separable : bool
        If True use the separable Kronecker kernel (``rho_w = -rho_d rho_o``);
        otherwise the unrestricted 3-ρ kernel.
    model_type : str
        Progress-display label.
    """
    from ...samplers._utils._idata import gibbs_to_inference_data
    from ...samplers._utils._seeds import spawn_chain_seeds
    from ...samplers.gaussian._chain_runner import run_chains
    from ...samplers.negbin_reduced._flow import FlowReducedGibbsCache
    from ...samplers.poisson_reduced._core import (
        FlowPoissonGibbsState,
        run_chain_separable,
        run_chain_unrestricted,
    )
    from .._base._shared import gelman_default_beta_prior
    from ..priors import FlowReducedGibbsPriors

    X = model._X
    y = model._y_int_vec.astype(np.float64)
    k = X.shape[1]
    W_csc = model._W_sparse.tocsc()

    cache_kwargs: dict = dict(
        Wd=model._Wd,
        Wo=model._Wo,
        Ww=model._Ww,
        W_csc=W_csc,
        n=model._n,
        separable=separable,
        rho_lower=model.priors.get("rho_lower", -0.999),
        rho_upper=model.priors.get("rho_upper", 0.999),
        T=T,
    )
    if not separable:
        cache_kwargs["positive"] = model.restrict_positive
    cache = FlowReducedGibbsCache(**cache_kwargs)

    default_beta_mu, default_beta_sigma = gelman_default_beta_prior(
        model._y, X, list(model._feature_names)
    )
    priors = FlowReducedGibbsPriors(
        beta_mu=model.priors.get("beta_mu", default_beta_mu),
        beta_sigma=model.priors.get("beta_sigma", default_beta_sigma),
        rho_lower=model.priors.get("rho_lower", -0.999),
        rho_upper=model.priors.get("rho_upper", 0.999),
    )

    def _make_init(rng: np.random.Generator) -> FlowPoissonGibbsState:
        from ...samplers.poisson_reduced._augment import (
            build_augmented_index,
            draw_augmentation,
        )

        beta0 = rng.normal(0.0, 0.1, size=k)
        if separable:
            rho_d0, rho_o0, rho_w0 = (
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.1, 0.1),
                None,
            )
        else:
            lo = 0.0 if model.restrict_positive else -0.1
            rho_d0, rho_o0 = rng.uniform(lo, 0.1), rng.uniform(lo, 0.1)
            rho_w0 = rng.uniform(0.0 if model.restrict_positive else -0.05, 0.05)
        design = build_augmented_index(y)
        s0, om0 = draw_augmentation(y, X @ beta0, design, rng=rng)
        return FlowPoissonGibbsState(
            beta=beta0, rho_d=rho_d0, rho_o=rho_o0, rho_w=rho_w0, s=s0, omega=om0
        )

    def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
        rng = np.random.default_rng(seed)
        runner = run_chain_separable if separable else run_chain_unrestricted
        return runner(
            y,
            X,
            cache,
            priors,
            draws,
            tune,
            thin=1,
            rng=rng,
            init=_make_init(rng),
            store_log_lik=log_likelihood,
        )

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

    posterior_samples = {
        "rho_d": np.stack([c["rho_d"] for c in chain_results], axis=0),
        "rho_o": np.stack([c["rho_o"] for c in chain_results], axis=0),
        "rho_w": np.stack([c["rho_w"] for c in chain_results], axis=0),
        "beta": np.stack([c["beta"] for c in chain_results], axis=0),
    }
    ll = None
    if log_likelihood:
        ll = {"obs": np.stack([c["log_lik"] for c in chain_results], axis=0)}

    model._idata = gibbs_to_inference_data(
        posterior_samples=posterior_samples,
        log_likelihood=ll,
        observed_data={"obs": model._y_int_vec},
        coords={"coefficient": list(model._feature_names)},
        dims={"beta": ["coefficient"]},
    )
    return model._idata
