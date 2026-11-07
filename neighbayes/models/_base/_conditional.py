r"""Pointwise log-likelihood of a Gaussian spatial model: one observation given the rest.

A spatial likelihood does not factorize, so the term leave-one-out needs is
:math:`\log p(y_i \mid y_{-i}, \theta)` (Bürkner, Gabry and Vehtari 2021, Prop. 1):
with :math:`Q` the precision of :math:`y` and :math:`g = Q (y - \mu)`,

.. math::

    \log p(y_i \mid y_{-i}, \theta)
        = -\tfrac12 \log 2\pi + \tfrac12 \log Q_{ii} - \frac{g_i^2}{2 Q_{ii}}.

The innovations are :math:`e = D y - h`, :math:`e \sim N(0, \sigma^2 I)`, so
:math:`Q = \sigma^{-2} D' \Omega D` and :math:`g = \sigma^{-2} D' \Omega e`: matrix-vector
products, no solve. Within a period :math:`D` is the filter
:math:`A = I - \sum_k \rho_k W_k` (one weights matrix, or a flow's three); a dynamic
panel adds :math:`-L` on the next period, the lagged terms. :math:`\Omega = \Omega_T
\otimes \Omega_N`: :math:`\Omega_T = I_T - c\,\iota\iota'`, where :math:`c` is 0 with no
unit effects, :math:`1/T` with fixed effects (their flat-prior limit, whose joint is
singular but whose conditionals are proper for :math:`T \ge 2`), and
:math:`\sigma_\alpha^2 / (\sigma^2 + T \sigma_\alpha^2)` with random effects integrated
out; :math:`\Omega_N` is the within-period projector with time effects and the
identity otherwise.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)


class LinearOperator:
    """``sum_a coef_a M_a`` over weights matrices ``M_a``, one coefficient row per draw."""

    def __init__(self, mats, coefs):
        self.mats = [sp.csr_matrix(m) for m in mats]
        self.coefs = np.atleast_2d(np.asarray(coefs, dtype=np.float64))
        if self.coefs.shape[1] != len(self.mats):
            self.coefs = self.coefs.T
        assert self.coefs.shape[1] == len(self.mats)

    @classmethod
    def filter(cls, W, par, n_draws):
        """``I - sum_k par_k W_k``; ``W`` one matrix or a list, ``par`` (draws,) or (draws, K)."""
        Ws = list(W) if isinstance(W, (list, tuple)) else [W]
        par = np.asarray(par, dtype=np.float64).reshape(n_draws, len(Ws))
        N = Ws[0].shape[0]
        return cls(
            [sp.identity(N, format="csr")] + Ws,
            np.hstack([np.ones((n_draws, 1)), -par]),
        )

    def transpose_apply(self, E):
        """``M' E`` for ``E`` of shape (draws, periods, N)."""
        S, T, N = E.shape
        out = np.zeros_like(E)
        flat = E.reshape(S * T, N).T
        for m, coef in zip(self.mats, self.coefs.T):
            out += coef[:, None, None] * np.asarray(m.T @ flat).T.reshape(S, T, N)
        return out

    def gram_diagonal(self, time_effects):
        """``diag(M' Omega_N M)``, shape (draws, N)."""
        N = self.mats[0].shape[0]
        d = np.zeros((self.coefs.shape[0], N))
        for a, ma in enumerate(self.mats):
            for b, mb in enumerate(self.mats):
                cross = np.asarray(ma.multiply(mb).sum(axis=0)).ravel()
                d += np.outer(self.coefs[:, a] * self.coefs[:, b], cross)
        if time_effects:
            columns = sum(
                np.outer(coef, np.asarray(m.sum(axis=0)).ravel())
                for m, coef in zip(self.mats, self.coefs.T)
            )
            d -= columns**2 / N
        return d


def _project(E, c, time_effects):
    if time_effects:
        E = E - E.mean(axis=2, keepdims=True)
    return E - c * E.sum(axis=1, keepdims=True)


def conditional_moments(
    e, sigma2, own, T=1, c=0.0, time_effects=False, next_period=None
):
    """``(g, q)`` of shape (draws, N*T): ``Q (y - mu)`` and ``diag(Q)``."""
    e = np.asarray(e, dtype=np.float64)
    S, n = e.shape
    N = n // T
    sigma2 = np.asarray(sigma2, dtype=np.float64).reshape(S, 1, 1)
    c = np.broadcast_to(np.asarray(c, dtype=np.float64), (S,)).reshape(S, 1, 1)
    E = _project(e.reshape(S, T, N), c, time_effects)
    G = own.transpose_apply(E)
    d = np.broadcast_to(own.gram_diagonal(time_effects)[:, None, :], (S, T, N)) * (
        1.0 - c
    )
    if next_period is not None:
        if np.any(c != 0):
            raise ValueError("A next-period operator needs no unit effects (c = 0).")
        later = next_period.transpose_apply(E[:, 1:, :])
        G = G.copy()
        G[:, :-1, :] -= later
        d = d.copy()
        d[:, :-1, :] += next_period.gram_diagonal(time_effects)[:, None, :]
    return (G / sigma2).reshape(S, n), (d / sigma2).reshape(S, n)


def normal_conditional_loglik(g, q):
    return -_HALF_LOG_2PI + 0.5 * np.log(q) - 0.5 * g * g / q


#: sinh nodes on the line: x = x* + w sinh(u), trapezoid in u with step 0.1.
_U = np.arange(-70, 71) * 0.1


@njit(parallel=True)
def _student_t_kernel(E, indptr, rows, ident, opvals, par, sigma, nu):
    """``log p(y_i | y_-i)`` for i.i.d. Student-t innovations ``e = D y - h``.

    ``y_i`` enters the innovations of column ``i`` of ``D``; the conditional is
    their product over its integral along ``y_i``, whose normalizing constants
    cancel.  The integral runs on sinh nodes about the Gaussian-weighted centre.
    """
    S, T, N = E.shape
    K = opvals.shape[0]
    nnz = rows.shape[0]
    half = 0.5 * (nu + 1.0)
    out = np.empty((S, T, N))
    for s in prange(S):
        d = ident.copy()
        for k in range(K):
            for p in range(nnz):
                d[p] -= par[s, k] * opvals[k, p]
        scale = nu * sigma[s] * sigma[s]
        vals = np.empty(_U.size)
        for t in range(T):
            for i in range(N):
                lo, hi = indptr[i], indptr[i + 1]
                num, sdd, sde = 0.0, 0.0, 0.0
                for p in range(lo, hi):
                    ek = E[s, t, rows[p]]
                    num -= half * np.log1p(ek * ek / scale)
                    sdd += d[p] * d[p]
                    sde += d[p] * ek
                centre = -sde / sdd
                width = sigma[s] / np.sqrt(sdd)
                top = -np.inf
                for j in range(_U.size):
                    x = centre + width * np.sinh(_U[j])
                    f = 0.0
                    for p in range(lo, hi):
                        z = E[s, t, rows[p]] + d[p] * x
                        f -= half * np.log1p(z * z / scale)
                    vals[j] = f + np.log(width * np.cosh(_U[j]))
                    if vals[j] > top:
                        top = vals[j]
                acc = 0.0
                for j in range(_U.size):
                    acc += np.exp(vals[j] - top)
                out[s, t, i] = num - top - np.log(acc * 0.1)
    return out


def student_t_conditional_loglik(e, par, sigma, nu, W, T=1):
    """``log p(y_i | y_-i, theta)`` under i.i.d. Student-t innovations, shape (draws, N*T).

    ``e`` are the innovations of the filter ``I - sum_k par_k W_k``, with any sampled
    effects already removed; there is no closed form, so fixed effects, whose flat
    prior would have to be integrated with the innovations, are not supported.
    """
    e = np.asarray(e, dtype=np.float64)
    S, n = e.shape
    N = n // T
    Ws = [sp.csr_matrix(w) for w in (W if isinstance(W, (list, tuple)) else [W])]
    pattern = sp.identity(N, format="csc") + sum(abs(w) for w in Ws).tocsc()
    pattern.sort_indices()
    indptr = pattern.indptr.astype(np.int64)
    rows = pattern.indices.astype(np.int64)
    cols = np.repeat(np.arange(N), np.diff(indptr))
    ident = (rows == cols).astype(np.float64)
    opvals = np.vstack([np.asarray(w[rows, cols]).ravel() for w in Ws])
    par = np.asarray(par, dtype=np.float64).reshape(S, len(Ws))
    sigma = np.asarray(sigma, dtype=np.float64).reshape(S)
    out = _student_t_kernel(
        e.reshape(S, T, N), indptr, rows, ident, opvals, par, sigma, float(nu)
    )
    return out.reshape(S, n)


def gaussian_conditional_loglik(
    e, par, sigma2, W, T=1, c=0.0, time_effects=False, next_period=None
):
    """``log p(y_i | y_-i, theta)``, shape (draws, N*T).

    ``e`` has shape (draws, N*T), stacked by period, then unit; ``par`` has shape
    (draws,) or (draws, K) for the K matrices in ``W``; ``c`` is a scalar or has shape
    (draws,); ``next_period`` is the :class:`LinearOperator` ``L`` of a dynamic panel.
    """
    S = np.asarray(e).shape[0]
    own = LinearOperator.filter(W, par, S)
    g, q = conditional_moments(e, sigma2, own, T, c, time_effects, next_period)
    return normal_conditional_loglik(g, q)
