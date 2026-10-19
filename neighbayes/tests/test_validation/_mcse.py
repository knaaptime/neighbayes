"""Monte-Carlo error estimation for the toolbox-validation comparisons.

arviz's rank-normalized ESS fails outright on the toolbox's grid-valued rho
draws (rank ties at the 0.001 sampling grid) and on heavily autocorrelated
single chains, so the ESS here is computed directly: Geyer's initial
positive sequence on each chain's autocovariance (FFT-based), summed over
chains.  MCSEs follow from the plan's standard — posterior sd over sqrt(ESS).
"""

from __future__ import annotations

import numpy as np


def _ess_1d(x: np.ndarray) -> float:
    """Effective sample size of one chain, Geyer initial positive sequence."""
    n = x.size
    if n < 2:
        return float(n)
    xc = x - x.mean()
    var0 = np.dot(xc, xc) / n
    if var0 <= 0.0:
        return float(n)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n].real / n
    rho = acov / var0
    # tau = -1 + 2 * sum of positive adjacent pairs (rho_{2j} + rho_{2j+1}),
    # truncated at the first non-positive pair (skip the monotone correction:
    # it only shrinks tau, and a larger tau is the conservative direction for
    # a tolerance).
    tau = -1.0
    for j in range(0, n - 1, 2):
        pair = rho[j] + rho[j + 1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    tau = max(tau, 1.0)
    return float(n / tau)


def ess(draws) -> float:
    """Effective sample size, summed over chains (first axis = chain)."""
    x = np.asarray(draws, dtype=float)
    if x.ndim == 1:
        return _ess_1d(x)
    return float(sum(_ess_1d(chain) for chain in x))


def mcse_mean(draws) -> float:
    """Monte-Carlo error of the pooled posterior mean."""
    flat = np.asarray(draws, dtype=float).ravel()
    return float(np.std(flat, ddof=1) / np.sqrt(max(ess(draws), 1.0)))


def mcse_sd(draws) -> float:
    """Monte-Carlo error of the posterior sd, sd / sqrt(2 ESS)."""
    flat = np.asarray(draws, dtype=float).ravel()
    return float(np.std(flat, ddof=1) / np.sqrt(2.0 * max(ess(draws), 1.0)))
