"""Pólya–Gamma Gibbs sampler for the reduced-form SAR Negative Binomial.

This sampler targets the canonical spatial-econometric SAR-NB model

.. math::

    y_i \\sim \\mathrm{NegBin}(\\mu_i, \\alpha), \\qquad
    \\mu = \\exp\\{(I - \\rho W)^{-1} X\\beta\\}

— that is, spatial dependence enters only through the *mean propagator*
:math:`(I - \\rho W)^{-1}` and overdispersion is handled entirely by the
NB dispersion parameter :math:`\\alpha`.  There is **no** latent spatial
random effect (no :math:`\\sigma`).  Contrast with
:mod:`neighbayes.samplers.negbin`, which samples the structural-form
model :math:`\\eta = \\rho W \\eta + X\\beta + \\nu,\\; \\nu \\sim N(0,
\\sigma^2 I)`.

The reduced form is the form most commonly written down in the spatial
econometrics literature (LeSage & Pace 2009), and is the preferred
default when you do not have substantive reason to model an unobserved
spatially-smoothed latent field.
"""

from .._registry import register
from ._core import (
    ReducedGibbsCache,
    ReducedGibbsPriors,
    ReducedGibbsState,
    ReducedKrylovBasis,
    run_chain,
)
from ._flow import (
    FlowReducedGibbsCache,
    FlowReducedGibbsPriors,
    FlowReducedGibbsState,
    run_chain_separable,
    run_chain_unrestricted,
)
from ._jax import run_chains_jax_reduced

__all__ = [
    "ReducedGibbsCache",
    "ReducedGibbsPriors",
    "ReducedGibbsState",
    "run_chain",
    "FlowReducedGibbsCache",
    "FlowReducedGibbsPriors",
    "FlowReducedGibbsState",
    "run_chain_separable",
    "run_chain_unrestricted",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry — reduced-form SAR NegBin (Pólya-Gamma)
# ---------------------------------------------------------------------------
#
# SARNegBin also supports NUTS (``sampler="nuts"`` builds a reduced-form PyMC
# model); the base dispatcher routes that path.  ``auto`` prefers the ``jax``
# path (fastest for cross-section); the CHOLMOD NumPy path is available via
# gibbs_backend="numpy".


def _run_count_reduced_gibbs(
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
    init_jitter=0.1,
    slice_width=0.4,
    krylov_degree=12,
    krylov_dmax=0.4,
    n_rho_omega_cycles=1,
    krylov_reuse=True,
    timeout=None,
    log_likelihood=False,
):
    """Registry runner for reduced-form SAR NegBin Pólya-Gamma Gibbs."""
    return model._fit_gibbs(
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        thin=thin,
        n_jobs=n_jobs,
        progressbar=progressbar,
        backend=backend,
        init_jitter=init_jitter,
        slice_width=slice_width,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
        n_rho_omega_cycles=n_rho_omega_cycles,
        krylov_reuse=krylov_reuse,
        timeout=timeout,
        log_likelihood=log_likelihood,
    )


register(
    "count",
    "cross_section",
    run=_run_count_reduced_gibbs,
    backends={"jax", "numpy"},
    auto_backend="jax",
    options={
        "init_jitter",
        "slice_width",
        "krylov_degree",
        "krylov_dmax",
        "n_rho_omega_cycles",
        "krylov_reuse",
        "timeout",
    },
    skips_log_likelihood=True,
)
