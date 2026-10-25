"""Gibbs samplers for the latent SAR hurdle and zero-inflated NB (:mod:`._core`).

Registers ``("hurdle_latent", "cross_section")`` and ``("zinb_latent",
"cross_section")``, which ``SARHurdleNegBin(latent=True)`` and
``SARZINB(latent=True)`` use, on the NumPy and JAX (:mod:`._jax`) backends.
"""

from .._registry import register
from ._core import make_chain

__all__ = ["make_chain"]


def _run_latent_two_part(
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


for _likelihood in ("hurdle_latent", "zinb_latent"):
    register(
        _likelihood,
        "cross_section",
        run=_run_latent_two_part,
        backends={"numpy", "jax"},
        options={"slice_width"},
        skips_log_likelihood=True,
    )
