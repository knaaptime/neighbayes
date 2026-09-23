r"""Pólya–Gamma Gibbs sampler for the reduced-form SAR-logit.

This sampler targets the canonical binary spatial model

.. math::

    y_i \sim \mathrm{Bernoulli}(\sigma(\eta_i)), \qquad
    \eta = (I - \rho W)^{-1} X\beta

— the binary analogue of the reduced-form SAR Negative-Binomial: the spatial lag
enters the *linear predictor*, so the ``|I − ρW|`` Jacobian cancels when β is
marginalized out and the system is *linear* in ρ.  That makes the ρ conditional
Krylov-accelerable and the sweep arithmetic-heavy.  It reuses the reduced-NB
machinery almost verbatim — the shift-invert Krylov basis, the CHOLMOD
normal-equations solver (NumPy) / sparsax sparse-LU (JAX), and the thread-parallel
chain runner.

Differences from the count model: the Pólya–Gamma draw uses h = 1 (Bernoulli),
the working response is κ/ω with κ = y − ½ (no ``log α`` offset), and there is no
dispersion parameter α (hence no α slice).

Contrast with :mod:`neighbayes.samplers.logit`, which samples the *structural*
latent-field SAR-logit / SEM-logit (``η = ρWη + Xβ + ν``).
"""

from .._registry import register
from ._core import ReducedLogitGibbsState, run_chain
from ._jax import run_chains_jax_reduced_logit

__all__ = [
    "ReducedLogitGibbsState",
    "run_chain",
    "run_chains_jax_reduced_logit",
]


# ---------------------------------------------------------------------------
# Gibbs registry entry — reduced-form SAR-logit (Pólya-Gamma)
# ---------------------------------------------------------------------------
#
# Gibbs-only (no NUTS build).  The canonical ``SARLogit`` model.  ``auto``
# prefers the ``jax`` thread-parallel path (fastest for cross-section models);
# the NumPy (CHOLMOD) path is available via ``gibbs_backend="numpy"``.


def _run_binary_reduced_gibbs(
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
    krylov_reuse=True,
    timeout=None,
    log_likelihood=False,
):
    """Registry runner for the reduced-form SAR-logit Pólya-Gamma Gibbs."""
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
        krylov_reuse=krylov_reuse,
        timeout=timeout,
        log_likelihood=log_likelihood,
    )


register(
    "binary",
    "cross_section",
    run=_run_binary_reduced_gibbs,
    backends={"jax", "numpy"},
    auto_backend="jax",
    options={
        "init_jitter",
        "slice_width",
        "krylov_degree",
        "krylov_dmax",
        "krylov_reuse",
        "timeout",
    },
    skips_log_likelihood=True,
)
