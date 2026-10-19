"""Pólya–Gamma Gibbs sampler for the SAR and SEM logit with random effects.

Serves the random-effects binary models ``SARLogitRE``
(``η = ρWη + Xβ + ν``) and ``SEMLogitRE`` (``η = Xβ + (I−λW)⁻¹ν``) under the
registry key ``"binary_re"``.  The canonical reduced-form ``SARLogit``
lives in :mod:`neighbayes.samplers.logit_reduced` (key ``"binary"``).
"""

from ...models.priors import LogitGibbsPriors, LogitREGibbsPriors, SEMLogitGibbsPriors
from .._registry import register
from ._core import (
    LogitGibbsCache,
    LogitGibbsState,
    SEMLogitGibbsCache,
    SEMLogitGibbsState,
    run_chain,
    run_chain_sem,
)

__all__ = [
    "LogitGibbsCache",
    "LogitGibbsPriors",
    "LogitGibbsState",
    "LogitREGibbsPriors",
    "SEMLogitGibbsCache",
    "SEMLogitGibbsPriors",
    "SEMLogitGibbsState",
    "run_chain",
    "run_chain_sem",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry
# ---------------------------------------------------------------------------
#
# SARLogitRE and SEMLogitRE are both structural cross-section binary
# (Bernoulli) models; the SAR-vs-SEM distinction is handled by MRO dispatch of
# ``_fit_gibbs`` (each model owns its Pólya-Gamma orchestration).  ``auto``
# prefers the ``jax`` path (fastest for cross-section); the CHOLMOD NumPy path
# is available via ``backend="numpy"``.  The reduced-form ``SARLogit``
# (key ``"binary"``) is registered separately in ``logit_reduced``.


def _run_binary_gibbs(
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
    return_eta=False,
    krylov_degree=0,
    krylov_dmax=0.4,
    log_likelihood=False,
):
    """Registry runner for cross-section binary (logit) Pólya-Gamma Gibbs."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        backend=backend,
        return_eta=return_eta,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        log_likelihood=log_likelihood,
    )


register(
    "binary_re",
    "cross_section",
    run=_run_binary_gibbs,
    backends={"jax", "numpy"},
    auto_backend="jax",
    options={
        "return_eta",
        "krylov_degree",
        "krylov_dmax",
    },
    skips_log_likelihood=True,
)
