"""Bayesian inference for spatial impact measures.

This module provides **post-processing helpers** that convert posterior
draws of model parameters into posterior distributions of direct,
indirect, and total impacts.  The actual *computation* of the impact
matrices :math:`(I-\\rho W)^{-1}(\\beta_{1k} I + \\beta_{2k} W)` lives
in the model classes (e.g. ``SAR._compute_impacts_posterior``,
``SDM._compute_impacts_posterior``); this module summarizes
those draws with means, credible intervals, and Bayesian p-values.
This is the fully Bayesian analog of the simulation-based approach in
:cite:t:`lesage2009IntroductionSpatial` and the asymptotic variance
formulas in :cite:t:`arbia2020TestingImpact`.

The key idea is that for each MCMC draw :math:`g = 1, \\dots, G`, we
compute the impact measures from the parameter values
:math:`(\\rho^{(g)}, \\beta^{(g)}, \\sigma^{(g)})`, yielding a posterior
distribution over each impact.  Posterior means, 95% credible intervals,
and Bayesian p-values are then computed from these draws.

For SAR and SDM models, the impact measures involve the spatial multiplier
:math:`S(\\rho) = (I - \\rho W)^{-1}`, which is evaluated efficiently
using the pre-computed eigenvalues of :math:`W`.  For SLX and SDEM
models (no :math:`\\rho`), the impacts are linear in :math:`\\beta`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _chunked_eig_means(
    rho_draws: np.ndarray,
    eigs: np.ndarray,
    weights: np.ndarray | None = None,
    chunk: int = 1024,
) -> np.ndarray:
    """Compute ``mean_j(weights_j / (1 - rho_g * eigs_j))`` per draw, chunked.

    The naive vectorized form ``1 / (1 - rho[:, None] * eigs[None, :])`` is
    correct but allocates a ``(G, n)`` intermediate, which dominates memory
    for large panels (e.g., ``G=8000``, ``n=14000`` ≈ 900 MB).  Chunking
    over draws keeps peak memory at ``O(chunk * n)`` while preserving the
    same vectorized inner kernel.

    Parameters
    ----------
    rho_draws : np.ndarray, shape (G,)
        Posterior draws of the spatial autoregressive parameter.
    eigs : np.ndarray, shape (n,)
        Eigenvalues of the spatial weights matrix ``W`` (complex or real).
    weights : np.ndarray, shape (n,), optional
        Per-eigenvalue weights (complex or real).  Pass ``eigs`` to compute
        ``mean(eigs / (1 - rho*eigs))`` (the diagonal of ``S @ W``); pass
        ``None`` for uniform weights (the diagonal of ``S``).
    chunk : int, default 1024
        Number of draws processed per inner iteration.  Tune for memory.

    Returns
    -------
    np.ndarray, shape (G,)
        The per-draw scalar mean (float64).  For complex inputs the result
        is the real part, which is exact since the underlying quantity is
        the trace of a real matrix.
    """
    G = rho_draws.shape[0]
    n = eigs.shape[0]
    out = np.empty(G, dtype=np.float64)
    if weights is None:
        for i in range(0, G, chunk):
            block = 1.0 / (1.0 - rho_draws[i : i + chunk, None] * eigs[None, :])
            out[i : i + chunk] = block.mean(axis=1).real
    else:
        for i in range(0, G, chunk):
            block = weights[None, :] / (
                1.0 - rho_draws[i : i + chunk, None] * eigs[None, :]
            )
            out[i : i + chunk] = (block.sum(axis=1) / n).real
    return out


def _compute_bayesian_pvalue(samples: np.ndarray) -> np.ndarray:
    """Compute two-sided Bayesian p-value for each column of *samples*.

    The Bayesian p-value tests :math:`H_0: \\text{effect} = 0`:

    .. math::
        p = 2 \\times \\min\\bigl(P(\\text{effect} > 0 \\mid \\text{data}),\\;
        P(\\text{effect} < 0 \\mid \\text{data})\\bigr)

    Parameters
    ----------
    samples : np.ndarray
        Posterior draws, shape ``(G, k)`` or ``(G,)``.

    Returns
    -------
    np.ndarray
        Bayesian p-values, same shape as columns of *samples*.
    """
    if samples.ndim == 1:
        frac_above = np.mean(samples > 0)
        frac_below = np.mean(samples < 0)
        return np.array([2.0 * min(frac_above, frac_below)])
    frac_above = np.mean(samples > 0, axis=0)
    frac_below = np.mean(samples < 0, axis=0)
    return 2.0 * np.minimum(frac_above, frac_below)


def _compute_ci(samples: np.ndarray) -> List[Tuple[float, float]]:
    """Compute 95% equal-tailed credible intervals for each column.

    Parameters
    ----------
    samples : np.ndarray
        Posterior draws, shape ``(G, k)``.

    Returns
    -------
    list of tuple
        One ``(2.5%, 97.5%)`` interval per column.
    """
    if samples.ndim == 1:
        return [
            (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        ]
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return [(float(lo[j]), float(hi[j])) for j in range(samples.shape[1])]


def _build_effects_dataframe(
    direct_samples: np.ndarray,
    indirect_samples: np.ndarray,
    total_samples: np.ndarray,
    feature_names: list[str],
    model_type: str,
) -> pd.DataFrame:
    """Build a summary DataFrame from posterior draws of impacts.

    Each row corresponds to one covariate.  Columns contain posterior
    means, 95% credible-interval bounds, and Bayesian p-values for
    direct, indirect, and total effects.

    Parameters
    ----------
    direct_samples : np.ndarray
        Posterior draws of direct effects, shape ``(G, k)``.
    indirect_samples : np.ndarray
        Posterior draws of indirect effects, shape ``(G, k)``.
    total_samples : np.ndarray
        Posterior draws of total effects, shape ``(G, k)``.
    feature_names : list of str
        Names of the covariates.
    model_type : str
        Label identifying the model type (e.g. ``"SAR"``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by feature names with columns:

        ==============  =====================================================
        Column          Description
        ==============  =====================================================
        direct          Posterior mean of direct effect
        direct_ci_lower Lower bound of 95% credible interval (2.5%)
        direct_ci_upper Upper bound of 95% credible interval (97.5%)
        direct_pvalue   Two-sided Bayesian p-value
        indirect        Posterior mean of indirect effect
        indirect_ci_lower  Lower bound of 95% credible interval
        indirect_ci_upper  Upper bound of 95% credible interval
        indirect_pvalue   Two-sided Bayesian p-value
        total           Posterior mean of total effect
        total_ci_lower  Lower bound of 95% credible interval
        total_ci_upper  Upper bound of 95% credible interval
        total_pvalue    Two-sided Bayesian p-value
        ==============  =====================================================

        The DataFrame has a ``model_type`` attribute storing *model_type*
        and an ``n_draws`` attribute storing the number of posterior draws.
    """
    direct_mean = np.mean(direct_samples, axis=0)
    indirect_mean = np.mean(indirect_samples, axis=0)
    total_mean = np.mean(total_samples, axis=0)

    direct_ci = _compute_ci(direct_samples)
    indirect_ci = _compute_ci(indirect_samples)
    total_ci = _compute_ci(total_samples)

    direct_pvalue = _compute_bayesian_pvalue(direct_samples)
    indirect_pvalue = _compute_bayesian_pvalue(indirect_samples)
    total_pvalue = _compute_bayesian_pvalue(total_samples)

    data = {
        "direct": direct_mean,
        "direct_ci_lower": [ci[0] for ci in direct_ci],
        "direct_ci_upper": [ci[1] for ci in direct_ci],
        "direct_pvalue": direct_pvalue,
        "indirect": indirect_mean,
        "indirect_ci_lower": [ci[0] for ci in indirect_ci],
        "indirect_ci_upper": [ci[1] for ci in indirect_ci],
        "indirect_pvalue": indirect_pvalue,
        "total": total_mean,
        "total_ci_lower": [ci[0] for ci in total_ci],
        "total_ci_upper": [ci[1] for ci in total_ci],
        "total_pvalue": total_pvalue,
    }

    df = pd.DataFrame(data, index=feature_names)
    df.index.name = "variable"
    df.attrs["model_type"] = model_type
    df.attrs["n_draws"] = direct_samples.shape[0]
    return df
