r"""Pointwise log-likelihood of a latent model, each latent value integrated out.

A latent model's observation :math:`y_i` depends on the parameters only through
its latent value :math:`\eta_i` (the log-mean, or the logit's utility), and the
latent values are coupled by the filter: :math:`A\eta - c = \varepsilon` with
:math:`\varepsilon_j` i.i.d. from the family's disturbance (log-gamma or
logistic) and :math:`c` the systematic part.  The pointwise likelihood the
leave-one-out estimate needs is

.. math::

    p(y_i \mid \theta, \eta_{-i})
        = \int p(y_i \mid \eta_i)\, p(\eta_i \mid \eta_{-i}, \theta)\, d\eta_i,

the integrated importance sampling of Li, Qiu, Zhang and Feng (2016):
integrating out :math:`\eta_i` leaves weights :math:`1/p(y_i \mid \theta,
\eta_{-i})`, which PSIS-LOO handles, where the likelihood conditional on
:math:`\eta_i` gives degenerate ones (Merkle, Furr and Rabe-Hesketh 2019).
The conditional :math:`p(\eta_i \mid \eta_{-i}, \theta) \propto \prod_j
f(\varepsilon_j)` runs over the equations column :math:`i` of :math:`A`
touches, and is log-concave; a Newton step finds its mode and a
double-exponential rule at the Laplace scale integrates it, over the line or,
for the logit's threshold, the half-line.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

#: The disturbance families: the negative binomial's log-gamma, the logit's logistic.
NEGBIN, LOGIT = 0, 1

_H = 0.1
_U_FULL = np.arange(-35, 36) * _H  # sinh-sinh nodes on the line
_U_HALF = np.arange(-45, 36) * _H  # exp-sinh nodes on the half-line


@njit(cache=True)
def _logf(e, fam, alpha):
    """Log density of a disturbance, up to a constant, and two derivatives."""
    if fam == NEGBIN:
        ex = np.exp(e)
        return alpha * e - alpha * ex, alpha * (1.0 - ex), -alpha * ex
    a = abs(e)
    t = np.tanh(0.5 * e)
    return -a - 2.0 * np.log1p(np.exp(-a)), -t, -0.5 * (1.0 - t * t)


@njit(cache=True)
def _cond(t, i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha):
    """``log p(η_i = t | η_{-i})`` up to a constant, and two derivatives in ``t``."""
    g = g1 = g2 = 0.0
    for p in range(indptr[i], indptr[i + 1]):
        a = ones[p] - par * wvals[p]
        v, d1, d2 = _logf(e[rows[p]] + a * (t - eta_i), fam, alpha)
        g += v
        g1 += a * d1
        g2 += a * a * d2
    return g, g1, g2


@njit(cache=True)
def _obs(t, y, fam):
    """``log p(y | η = t)`` for the negative binomial's Poisson draw, and derivatives."""
    if fam == NEGBIN:
        et = np.exp(t)
        return y * t - et, y - et, -et
    return 0.0, 0.0, 0.0


@njit(cache=True)
def _mode(i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha, y, with_obs):
    """Mode and Laplace scale of the conditional (times the observation's term)."""
    t = eta_i
    g2 = -1.0
    for _ in range(50):
        _, g1, g2 = _cond(t, i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha)
        if with_obs:
            _, o1, o2 = _obs(t, y, fam)
            g1 += o1
            g2 += o2
        step = -g1 / g2
        step = min(max(step, -2.0), 2.0)
        t += step
        if abs(step) < 1e-10:
            break
    return t, 1.0 / np.sqrt(-g2)


@njit(cache=True)
def _log_integral(
    i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha, y, with_obs, side
):
    """``log ∫ exp(conditional [+ observation])`` over the line (``side=0``) or a half-line."""
    mode, scale = _mode(
        i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha, y, with_obs
    )
    half = side != 0
    u_nodes = _U_HALF if half else _U_FULL
    if half:
        # The half-line's own scale: the mode when it lies inside, else the width.
        inside = mode * side > 0
        scale = scale + (abs(mode) if inside else 0.0)
    vals = np.empty(u_nodes.size)
    top = -np.inf
    for q in range(u_nodes.size):
        u = u_nodes[q]
        s = 0.5 * np.pi * np.sinh(u)
        if half:
            x = np.exp(s)
            jac = 0.5 * np.pi * np.cosh(u) * x
            t = side * scale * x
        else:
            x = np.sinh(s)
            jac = 0.5 * np.pi * np.cosh(u) * np.cosh(s)
            t = mode + scale * x
        if not np.isfinite(t) or jac == 0.0:
            vals[q] = -np.inf
            continue
        g, _, _ = _cond(t, i, eta_i, e, indptr, rows, ones, wvals, par, fam, alpha)
        if with_obs:
            o, _, _ = _obs(t, y, fam)
            g += o
        v = g + np.log(jac)
        vals[q] = v
        if v > top:
            top = v
    acc = 0.0
    for q in range(u_nodes.size):
        acc += np.exp(vals[q] - top)
    return top + np.log(acc * _H * scale)


@njit(cache=True, parallel=True)
def integrated_loglik(latent, e, y, indptr, rows, ones, wvals, par, alpha, fam, lgy):
    """``log p(y_i | θ, η_{-i})`` for every draw (rows) and observation (columns).

    ``latent`` and ``e = A η - c`` are ``(D, n)``; ``indptr``, ``rows``,
    ``ones`` and ``wvals`` are the columns of ``I`` and ``W`` (so column ``i``
    of ``A`` is ``ones - par * wvals`` over its rows); ``par`` and ``alpha``
    are ``(D,)``; ``lgy`` is ``log y!``.
    """
    D, n = latent.shape
    out = np.empty((D, n))
    for flat in prange(D * n):
        d = flat // n
        i = flat - d * n
        args = (i, latent[d, i], e[d], indptr, rows, ones, wvals, par[d], fam, alpha[d])
        z = _log_integral(*args, y[i], False, 0)
        if fam == NEGBIN:
            out[d, i] = _log_integral(*args, y[i], True, 0) - z - lgy[i]
        else:
            side = 1 if y[i] > 0 else -1
            out[d, i] = _log_integral(*args, y[i], False, side) - z
    return out
