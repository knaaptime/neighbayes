"""Gibbs sampler for the latent SAR and SEM logit and probit (:mod:`._core`).

Registers ``("binary_latent", "cross_section")``, which the latent
:class:`~neighbayes.models.SARLogit`, :class:`~neighbayes.models.SEMLogit`,
:class:`~neighbayes.models.SARProbit` and :class:`~neighbayes.models.SEMProbit`
and their SDEMs share, on the NumPy and JAX (:mod:`._jax`) backends; the
logit's Kolmogorov scales come from ``augax``.
"""

from .._registry import register
from ._core import LatentBinaryChain

__all__ = ["LatentBinaryChain"]


def _run_latent_binary(
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
    "binary_latent",
    "cross_section",
    run=_run_latent_binary,
    backends={"numpy", "jax"},
    options={"slice_width"},
    skips_log_likelihood=True,
)
