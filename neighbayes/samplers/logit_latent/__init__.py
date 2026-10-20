"""Gibbs sampler for the latent SAR and SEM logit (:mod:`._core`).

Registers ``("binary_latent", "cross_section")``, which the latent
:class:`~neighbayes.models.SARLogit` and :class:`~neighbayes.models.SEMLogit`
share, on the NumPy and JAX (:mod:`._jax`) backends; the Kolmogorov scales come
from ``augax``.
"""

from .._registry import register
from ._core import run_chain

__all__ = ["run_chain"]


def _run_latent_logit(
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
    if log_likelihood:
        raise NotImplementedError(
            "The latent logit's pointwise likelihood is an orthant probability "
            "of the correlated utilities, with no closed form; it stores none."
        )
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
    )


register(
    "binary_latent",
    "cross_section",
    run=_run_latent_logit,
    backends={"numpy", "jax"},
    options={"slice_width"},
    skips_log_likelihood=True,
)
