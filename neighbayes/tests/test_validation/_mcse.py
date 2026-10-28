"""Monte-Carlo error estimation for the toolbox-validation comparisons.

arviz's rank-normalized ESS fails outright on the toolbox's grid-valued rho
draws (rank ties at the 0.001 sampling grid) and on heavily autocorrelated
single chains, so the ESS here is computed directly: Geyer's initial
positive sequence on each chain's autocovariance (FFT-based), summed over
chains.  MCSEs follow from the plan's standard — posterior sd over sqrt(ESS) —
and the assertions below apply it: a difference between the two sides within a
multiple of their Monte-Carlo errors joined in quadrature.
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


# ---------------------------------------------------------------------------
# Comparisons: neighbayes draws (chain, draw) against the toolbox's (draw,)
# ---------------------------------------------------------------------------


def assert_mean_matches(label, nb_draws, tb_draws, multiplier=4.0):
    """|Δ posterior mean| within ``multiplier`` × joined Monte-Carlo error."""
    tol = multiplier * np.hypot(mcse_mean(nb_draws), mcse_mean(tb_draws))
    diff = abs(float(np.mean(nb_draws)) - float(np.mean(tb_draws)))
    assert diff <= tol, (
        f"{label}: |Δ posterior mean| = {diff:.4f} > {tol:.4f} "
        f"(neighbayes {np.mean(nb_draws):.4f} [ESS {ess(nb_draws):.0f}], "
        f"toolbox {np.mean(tb_draws):.4f} [ESS {ess(tb_draws):.0f}])"
    )


def assert_sd_matches(label, nb_draws, tb_draws, multiplier=4.0):
    """|Δ posterior sd| within ``multiplier`` × joined Monte-Carlo error."""
    tol = multiplier * np.hypot(mcse_sd(nb_draws), mcse_sd(tb_draws))
    diff = abs(float(np.std(nb_draws)) - float(np.std(tb_draws)))
    assert diff <= tol, (
        f"{label}: |Δ posterior sd| = {diff:.4f} > {tol:.4f} "
        f"(neighbayes {np.std(nb_draws):.4f}, toolbox {np.std(tb_draws):.4f})"
    )


def _corr_mcse(a, b) -> tuple[float, float]:
    """Correlation of paired draws and its Monte-Carlo error.

    The error is the iid one, (1 - r²)/sqrt(m), at the smaller of the two
    series' effective sizes m.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    m = max(min(ess(a), ess(b)), 4.0)
    return r, (1.0 - r * r) / np.sqrt(m)


def assert_corr_matches(label, nb_pair, tb_pair, multiplier=4.0):
    """Posterior correlation of two parameters, within joined Monte-Carlo error."""
    r_nb, se_nb = _corr_mcse(*nb_pair)
    r_tb, se_tb = _corr_mcse(*tb_pair)
    tol = multiplier * np.hypot(se_nb, se_tb)
    assert abs(r_nb - r_tb) <= tol, (
        f"{label}: |Δ posterior correlation| = {abs(r_nb - r_tb):.3f} > {tol:.3f} "
        f"(neighbayes {r_nb:+.3f}, toolbox {r_tb:+.3f})"
    )
