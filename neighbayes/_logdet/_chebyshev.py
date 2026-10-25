"""Chebyshev polynomial approximation of log|I - ρW| (``cheb`` on ``eig``).

Exact values at Chebyshev nodes from the eigendecomposition of ``W``, then
DCT-I for the coefficients.  The node and order helpers here are shared with
the Cholesky and LU Chebyshev interpolants.
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Chebyshev nodes, orders and coefficients
# ---------------------------------------------------------------------------


def chebyshev_gauss_nodes(order: int, lo: float, hi: float):
    """Chebyshev-Gauss nodes ``cos((2k-1)\u03c0/(2\u00b7order))`` mapped to ``[lo, hi]``.

    Returns ``(mapped_nodes, nodes_cos)`` for ``k = 1..order``.  Pair with
    :func:`chebyshev_coeffs_dct1` — the single source for the node/DCT-I
    combination used by every Chebyshev-in-\u03c1 logdet surrogate.
    """
    k = np.arange(1, order + 1)
    nodes_cos = np.cos((2 * k - 1) * np.pi / (2 * order))
    return 0.5 * (hi - lo) * nodes_cos + 0.5 * (hi + lo), nodes_cos


def bernstein_rho(rho_min: float, rho_max: float) -> float:
    """Bernstein-ellipse parameter for ``log|I - \u03c1W|`` on ``[rho_min, rho_max]``.

    ``J(\u03c1) = \u03a3 log(1 - \u03c1\u03bb\u1d62)`` is analytic on the interval with logarithmic
    singularities at ``\u03c1 = 1/\u03bb\u1d62``.  For row-standardized ``W`` the extreme
    eigenvalues are ``\u03bb_max = 1`` (Perron) and ``\u03bb_min \u2248 -1``, so the nearest
    singularities sit at ``\u03c1 = \u00b11``.  Mapping the interval to ``[-1, 1]`` sends
    a singularity at ``s`` to ``t = (2s - a - b)/(b - a)``, and the Chebyshev
    truncation error decays as ``\u03c1_B^{-m}`` with

        \u03c1_B = |t| + \u221a(t\u00b2 - 1)

    for the nearest singularity :cite:p:`trefethen2013`.  ``\u03c1_B`` grows as the
    interval retreats from ``\u00b11``, which is what makes a narrow interval \u2014 a
    post-warmup range, say \u2014 cheap to interpolate on.

    Returns ``inf`` for a degenerate (zero-width) interval.
    """
    a = float(rho_min)
    b = float(rho_max)
    if b <= a:
        return float("inf")
    out = float("inf")
    for s in (1.0, -1.0):
        t = abs((2.0 * s - a - b) / (b - a))
        if t <= 1.0:
            return 1.0  # singularity inside the interval \u2014 no convergence
        out = min(out, t + np.sqrt(t * t - 1.0))
    return float(out)


# Empirical error model for the Cholesky-Chebyshev interpolant, fitted by least
# squares over rook lattices n \u2208 {484, 2500, 10000, 22500} \u00d7 six intervals \u00d7
# five orders (103 observations, residual sd 0.29 log10 units, worst
# under-prediction 4\u00d7):
#
#     log|e| = log C + \u03b1\u00b7log n - s\u00b7m\u00b7log \u03c1_B,   C = 0.034, \u03b1 = 0.92, s = 1.12
#
# ``\u03b1 \u2248 1`` because ``J(\u03c1)`` itself is O(n).  The fitted rate ``s = 1.12``
# exceeds the Bernstein exponent of 1, i.e. the asymptotic bound is mildly
# conservative here; we use ``s = 1`` in the order rule below so the safety
# margin is kept rather than fitted away.
_CHEB_ERR_C = 0.034
_CHEB_ERR_ALPHA = 0.92

#: Default accuracy target, **relative** to the scale of ``J(\u03c1)`` itself, which
#: is ``O(n)``.  An absolute target would be the wrong invariant: it would make
#: the interpolant needlessly loose on small problems, where the factorizations
#: are cheap anyway, and needlessly tight on large ones.  Because the fitted
#: error also grows as ``n^0.92``, a relative target leaves the selected order
#: nearly independent of ``n`` \u2014 matching the n-independence of the lookup table
#: this replaced \u2014 while still tracking the interval.
DEFAULT_CHEB_RTOL = 1e-9


def cheb_order_for_tolerance(
    rho_min: float,
    rho_max: float,
    n: int,
    tol: float | None = None,
    rtol: float = DEFAULT_CHEB_RTOL,
    floor: int = 4,
    cap: int | None = None,
) -> int:
    """Chebyshev order meeting an error target on ``[rho_min, rho_max]``.

    Inverts the error model above for ``m``:

        m = ln(C \u00b7 n^\u03b1 / tol) / ln \u03c1_B

    Replaces a width-keyed lookup table, which could not distinguish intervals
    by their distance to the ``\u03c1 = \u00b11`` singularities and so returned the same
    order for the applied default ``[0.1, 0.8]`` and for a post-warmup window an
    order of magnitude narrower.

    Parameters
    ----------
    rho_min, rho_max : float
        The \u03c1 approximation interval.
    n : int
        Matrix dimension; sets the scale of ``J(\u03c1)`` and hence of the error.
    tol : float, optional
        Target maximum **absolute** error in log-units.  When ``None`` (the
        default) it is taken as ``rtol \u00b7 n``.
    rtol : float, default 1e-9
        Target error relative to the ``O(n)`` scale of ``J``, used when ``tol``
        is ``None``.
    floor, cap : int
        Clamps on the returned order.

    Returns
    -------
    int
        Number of Chebyshev nodes (= exact factorizations at setup).
    """
    # Node cap.  ``NEIGHBAYES_LOGDET_NODE_CAP`` forces every interpolant to the
    # same ceiling, which is how a probe-matched comparison is set up: on a wide
    # interval each rule is clipped to the shared budget, while on a narrow
    # post-warmup interval both ask for far less than the cap and adapt freely.
    if cap is None:
        _c = os.getenv("NEIGHBAYES_LOGDET_NODE_CAP")
        cap = int(_c) if _c else 200

    rho_b = bernstein_rho(rho_min, rho_max)
    if not np.isfinite(rho_b) or rho_b <= 1.0:
        return int(cap)
    n_eff = max(float(n), 1.0)
    if tol is None:
        tol = float(rtol) * n_eff
    scale = _CHEB_ERR_C * n_eff**_CHEB_ERR_ALPHA
    m = np.log(max(scale / max(float(tol), 1e-16), 1.0)) / np.log(rho_b)
    return int(np.clip(int(np.ceil(m)), floor, cap))


def cheb_tail_error(coeffs: np.ndarray) -> float:
    """A-posteriori truncation-error estimate from the Chebyshev coefficient tail.

    For a geometrically converging series the truncation error is dominated by
    the first omitted term, which the last retained coefficients bound.  Taking
    the largest of the final two is the standard Chebfun-style estimate; unlike
    the a-priori order rule it needs no fitted constant and no assumption about
    where the singularities lie.

    It is a **diagnostic, not a gate**.  The bound only holds once the series
    has entered its asymptotic decay, so at low order it is badly conservative:
    on a rook lattice at n = 10,000 over ``[0.55, 0.65]`` it reports 2.8e-3 for
    an interpolant whose true maximum error is 5.2e-6, while at order 13 over
    the same interval it reports 4.6e-12 against a true 2.3e-12.  Use it to
    confirm resolution at moderate-to-high order, and the a-priori rule
    :func:`cheb_order_for_tolerance` to choose the order.
    """
    c = np.abs(np.asarray(coeffs, dtype=np.float64))
    if c.size <= 2:
        return float(c[-1]) if c.size else 0.0
    return float(c[-2:].max())


def chebyshev_coeffs_dct1(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients from values at Chebyshev-Gauss nodes (DCT-I).

    ``values[k-1] = f(x_k)`` at the nodes from :func:`chebyshev_gauss_nodes`;
    returns ``coeffs`` such that ``f(x) \u2248 \u03a3_j coeffs[j] T_j(x)``.
    """
    values = np.asarray(values, dtype=np.float64)
    order = len(values)
    k = np.arange(1, order + 1)
    coeffs = np.zeros(order, dtype=np.float64)
    for j in range(order):
        scale = 2.0 / order if j > 0 else 1.0 / order
        coeffs[j] = scale * np.sum(
            values * np.cos(j * (2 * k - 1) * np.pi / (2 * order))
        )
    return coeffs
