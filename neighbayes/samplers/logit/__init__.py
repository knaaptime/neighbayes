"""Pólya–Gamma Gibbs sampler for structural-form SAR-logit and SEM-logit.

Serves the *structural* (latent-field) binary models ``SARLogitStructural``
(``η = ρWη + Xβ + ν``) and ``SEMLogit`` (``η = Xβ + (I−λW)⁻¹ν``) under the
registry key ``"binary_structural"``.  The canonical reduced-form ``SARLogit``
lives in :mod:`neighbayes.samplers.logit_reduced` (key ``"binary"``).
"""

from .._registry import register
from ._core import (
    LogitGibbsCache,
    LogitGibbsPriors,
    LogitGibbsState,
    SEMLogitGibbsCache,
    SEMLogitGibbsPriors,
    SEMLogitGibbsState,
    run_chain,
    run_chain_sem,
)
from ._jax import run_chain_jax, run_chain_jax_sem

__all__ = [
    "LogitGibbsCache",
    "LogitGibbsPriors",
    "LogitGibbsState",
    "SEMLogitGibbsCache",
    "SEMLogitGibbsPriors",
    "SEMLogitGibbsState",
    "run_chain",
    "run_chain_jax",
    "run_chain_sem",
    "run_chain_jax_sem",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry
# ---------------------------------------------------------------------------
#
# SARLogitStructural and SEMLogit are both structural cross-section binary
# (Bernoulli) models; the SAR-vs-SEM distinction is handled by MRO dispatch of
# ``_fit_gibbs`` (each model owns its Pólya-Gamma orchestration).  ``auto``
# prefers the ``jax`` path (fastest for cross-section); the CHOLMOD NumPy path
# is available via ``gibbs_backend="numpy"``.  The reduced-form ``SARLogit``
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
    pg_n_terms=25,
    n_probes=5,
    lanczos_deg=15,
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
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        log_likelihood=log_likelihood,
    )


register(
    "binary_structural",
    "cross_section",
    run=_run_binary_gibbs,
    backends={"jax", "numpy"},
    auto_backend="jax",
    options={
        "return_eta",
        "pg_n_terms",
        "n_probes",
        "lanczos_deg",
        "krylov_degree",
        "krylov_dmax",
    },
    skips_log_likelihood=True,
)
