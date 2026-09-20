"""Negative Binomial (Pólya-Gamma) Gibbs samplers.

This subpackage provides Pólya-Gamma augmented Gibbs samplers for
spatial count models (NegBin), with both NumPy and JAX backends.

Public API
----------
GibbsState
    State container for the cross-sectional PG Gibbs sampler.
GibbsPriors
    Prior hyperparameters for the cross-sectional PG Gibbs sampler.
GibbsCache
    Precomputed cache for the cross-sectional PG Gibbs sampler.
run_chain
    Run a single cross-sectional PG Gibbs chain.
JAXGibbsState
    JAX-compatible state container (equinox Module).
"""

from .._registry import register
from ._core import GibbsCache, GibbsPriors, GibbsState, JAXGibbsState, run_chain

__all__ = [
    "GibbsState",
    "GibbsPriors",
    "GibbsCache",
    "run_chain",
    "JAXGibbsState",
    "FlowGibbsState",
    "FlowGibbsPriors",
    "FlowGibbsCache",
    "FlowGibbsSliceState",
    "FlowGibbsCacheNS",
    "FlowGibbsSliceStateNS",
    "run_flow_chain_separable",
    "run_flow_chain_nonseparable",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry — structural-form SAR NegBin (Pólya-Gamma)
# ---------------------------------------------------------------------------
#
# Gibbs-only (no NUTS build).  ``auto`` prefers the ``jax`` path (fastest for
# cross-section); the CHOLMOD NumPy path is available via gibbs_backend="numpy".


def _run_count_structural_gibbs(
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
    """Registry runner for structural-form SAR NegBin Pólya-Gamma Gibbs."""
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
    "count_structural",
    "cross_section",
    run=_run_count_structural_gibbs,
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
