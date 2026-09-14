"""ZINB Gibbs sampler package."""

from .._registry import register
from ._core import (
    ZINBGibbsCache,
    ZINBGibbsPriors,
    ZINBGibbsState,
    _sample_z,
    run_zinb_chain,
)

__all__ = [
    "ZINBGibbsCache",
    "ZINBGibbsPriors",
    "ZINBGibbsState",
    "_sample_z",
    "run_zinb_chain",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry — zero-inflated SAR NegBin (9-block Pólya-Gamma)
# ---------------------------------------------------------------------------
#
# Gibbs-only (no NUTS build).  Both backends fit the *same* reduced-form ZINB
# (reduced-form SAR-logit selection + reduced-form SAR-NB count): "numpy"
# (CHOLMOD) and "jax" (thread-parallel, sparsax-KLU + pgjax).  ``auto``
# prefers jax (fastest for cross-section).


def _run_zinb_gibbs(
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
    timeout=None,
):
    """Registry runner for the zero-inflated SAR NegBin 9-block Gibbs."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        backend=backend,
        timeout=timeout,
    )


register(
    "zinb",
    "cross_section",
    run=_run_zinb_gibbs,
    backends={"numpy", "jax"},
    auto_backend="jax",
    options={"timeout"},
)
