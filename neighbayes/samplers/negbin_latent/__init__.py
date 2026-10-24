"""Gibbs sampler for the latent SAR and SEM negative binomial (:mod:`._core`).

Registers ``("count_latent", "cross_section")``, which the latent
:class:`~neighbayes.models.SARNegBin`, :class:`~neighbayes.models.SEMNegBin`
and :class:`~neighbayes.models.SDEMNegBin` share, on the NumPy and JAX
(:mod:`._jax`) backends.
"""

from .._registry import register
from ._core import run_chain

__all__ = ["run_chain"]


def _run_latent_negbin(
    model,
    *,
    draws,
    tune,
    chains,
    random_seed,
    thin,
    n_jobs,
    progressbar,
    backend,
    slice_width=0.4,
    log_likelihood=False,
):
    """Registry runner: a thin adapter over the model's ``_fit_gibbs``."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        backend=backend,
        slice_width=slice_width,
        log_likelihood=log_likelihood,
    )


register(
    "count_latent",
    "cross_section",
    run=_run_latent_negbin,
    backends={"numpy", "jax"},
    options={"slice_width"},
    skips_log_likelihood=True,
)
