"""Shared numpy Clenshaw recurrence for Chebyshev-in-ρ series.

Single home for the numpy evaluation of a Chebyshev series
``Σ_j coeffs[j] T_j(x(ρ))`` where ``x`` maps ``[rho_min, rho_max] → [-1, 1]``.
Both the ``chol_cheb`` evaluator (:mod:`neighbayes._logdet._chol_cheb`)
and the logdet factories (:mod:`neighbayes._logdet._factories`) evaluate the
same series with the same recurrence; keeping it here removes the duplicate
inline copies.  JAX / PyTensor evaluators are intentionally *not* routed
through this module — they live co-located with their array library.

This is a leaf module (numpy only) so any logdet submodule may import it
without risking an import cycle.
"""

from __future__ import annotations

import numpy as np


def clenshaw_scalar(coeffs, r, rmin_cb, rmax_cb, T=1):
    """Evaluate a Chebyshev series at scalar ``r`` via Clenshaw recurrence."""
    r = float(r)
    x = (2.0 * r - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)
    m = len(coeffs)
    if m == 0:
        return 0.0
    if m == 1:
        return float(coeffs[0])
    b_next = 0.0
    b_curr = float(coeffs[m - 1])
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + float(coeffs[k])
        b_next = b_curr
        b_curr = b_new
    val = float(coeffs[0]) + x * b_curr - b_next
    return val if T == 1 else T * val


def clenshaw_vec(coeffs, rho_arr, rmin_cb, rmax_cb, T=1):
    """Evaluate a Chebyshev series at an array of ρ via Clenshaw recurrence."""
    rho_arr = np.asarray(rho_arr, dtype=np.float64)
    x = (2.0 * rho_arr - rmax_cb - rmin_cb) / (rmax_cb - rmin_cb)
    m = len(coeffs)
    if m == 0:
        return np.zeros_like(rho_arr, dtype=np.float64)
    if m == 1:
        return np.full_like(rho_arr, coeffs[0], dtype=np.float64)
    b_next = np.zeros_like(x, dtype=np.float64)
    b_curr = np.full_like(x, coeffs[m - 1], dtype=np.float64)
    for k in range(m - 2, 0, -1):
        b_new = 2.0 * x * b_curr - b_next + coeffs[k]
        b_next = b_curr
        b_curr = b_new
    val = coeffs[0] + x * b_curr - b_next
    return val if T == 1 else T * val
