r"""Gibbs sampler for the latent SAR and SEM negative binomial.

With :math:`A = I - \rho W` (SAR) or :math:`I - \lambda W` (SEM) and
:math:`S = A^{-1}`, the model is

.. math::
    y_i \mid \eta_i \sim \mathrm{Poisson}(e^{\eta_i}), \qquad
    \text{SAR: } A\eta = X\beta + \varepsilon, \qquad
    \text{SEM: } A(\eta - X\beta) = \varepsilon,

with :math:`\varepsilon_i = \log\gamma_i`, :math:`\gamma_i \sim
\mathrm{Gamma}(\alpha, \alpha)` iid, so that :math:`\rho = 0`
(:math:`\lambda = 0`) is the NB2 of :class:`~neighbayes.models.NegBin`: the
multiplier propagates the negative binomial's own gamma disturbance.  The state
carries the log-means :math:`\eta`; given them the disturbances are the linear
map :math:`\varepsilon = A\eta - X\beta` (SAR) or :math:`A(\eta - X\beta)`
(SEM), and the joint density

.. math::
    \sum_i (y_i\eta_i - e^{\eta_i}) + \log|A|
    + \sum_j \alpha(\varepsilon_j - e^{\varepsilon_j})
    + n\{\alpha\log\alpha - \log\Gamma(\alpha)\}

is log-concave in :math:`\eta`, in :math:`\beta` and in the spatial parameter.
Each sweep leaves the posterior invariant:

1. :math:`\eta_i \mid \cdot`, one coordinate at a time: a slice (Neal 2003) on
   a log-concave density that touches column ``i`` of :math:`A` only.
2. :math:`\rho \mid \eta, u`: a slice along :math:`\beta = u - \rho g`, with
   :math:`g` the least-squares coefficient of :math:`W\eta` on :math:`X`, which
   moves :math:`\varepsilon` only along the part of :math:`W\eta` that
   :math:`X` does not explain; the shift has unit Jacobian.  The SEM's
   :math:`\lambda` is drawn with :math:`\beta` held, the two being nearly
   orthogonal.
3. :math:`\beta \mid \eta, \rho, \alpha`: a gamma regression with log link,
   drawn by independence Metropolis–Hastings from a multivariate *t* at its
   Laplace approximation (a function of the conditioning values only).
4. :math:`\alpha \mid \varepsilon`: a slice on :math:`\log\alpha` under the
   NB models' Half-Student-*t* prior.

Steps 1–4 condition on :math:`\eta`, which the counts pin only loosely when
they are small; :math:`\alpha` and the spatial parameter then creep.  Two
moves interweave the parametrization in which the disturbances are held
instead (Yu and Meng 2011), each exact:

5. :math:`\alpha` with the standardized disturbances
   :math:`z = (\varepsilon - m_\alpha)/s_\alpha` held, :math:`m_\alpha` and
   :math:`s_\alpha` their mean and sd: a slice on :math:`\log\alpha`, the log-means
   following in closed form from :math:`S X\beta` and :math:`S\mathbf 1`.
6. the spatial parameter with :math:`\varepsilon` held: one random-walk
   Metropolis proposal, the log-means following by one solve.

The solves are BiCGSTAB from the nearby solution at hand.  There is no
Pólya–Gamma step and no added scale: the only disturbance is the negative
binomial's own.
"""

from __future__ import annotations

from typing import Callable, Literal

import numba
import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln, polygamma

from ...models.priors import ReducedGibbsPriors
from .._utils._slice import SliceWidthState, slice_sample_1d, slice_sample_1d_adaptive

#: Degrees of freedom of the β proposal.
_BETA_DF = 8.0
#: Stepping-out budget of the η slices.
_ETA_MAX_STEPS = 32
#: Relative residual of the filter solves.
_SOLVE_RTOL = 1e-12
#: Iteration cap of a filter solve before it falls back on a direct solve.
_SOLVE_MAXITER = 500
#: Target acceptance of the spatial parameter's interweaving proposal.
_PAR_ACCEPT = 0.44


def filter_columns(
    W: sp.spmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column structure of ``A = I - rho W``: ``A``'s values are ``ones - rho * w``.

    Returns the CSC ``indptr`` and row ``indices`` of the pattern of
    ``I + W``, with each entry's identity value ``ones`` and weight ``w``.
    """
    W = sp.coo_matrix(W)
    n = W.shape[0]
    rows = np.r_[W.row, np.arange(n)]
    cols = np.r_[W.col, np.arange(n)]
    # One complex matrix keeps the two value arrays on the same pattern.
    vals = np.r_[W.data.astype(np.complex128), np.full(n, 1j)]
    P = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    P.sum_duplicates()
    P.sort_indices()
    return (
        P.indptr.astype(np.int64),
        P.indices.astype(np.int64),
        np.ascontiguousarray(P.data.imag),
        np.ascontiguousarray(P.data.real),
    )


@numba.njit(cache=True)
def _eta_logf(t, i, d0, y_i, alpha, eps, indptr, rows, aval):
    """Log density of ``eta_i = t`` given the rest, up to a constant."""
    d = t - d0
    acc = y_i * t - np.exp(t)
    for p in range(indptr[i], indptr[i + 1]):
        e = eps[rows[p]] + aval[p] * d
        acc += alpha * (e - np.exp(e))
    return acc


@numba.njit(cache=True)
def _eta_sweep(eta, eps, y, alpha, indptr, rows, aval, rng):
    """One Gibbs pass over the log-means, updating ``eps`` alongside.

    The slice width of ``eta_i`` is a multiple of its conditional sd at
    ``e^eta_i = y_i`` and ``e^eps = 1``, which does not depend on ``eta_i``.
    """
    for i in range(eta.size):
        x0 = eta[i]
        info = y[i]
        for p in range(indptr[i], indptr[i + 1]):
            info += alpha * aval[p] * aval[p]
        w = 2.5 / np.sqrt(info)
        z = _eta_logf(x0, i, x0, y[i], alpha, eps, indptr, rows, aval)
        z -= rng.standard_exponential()
        left = x0 - w * rng.random()
        right = left + w
        j = int(_ETA_MAX_STEPS * rng.random())
        k = _ETA_MAX_STEPS - 1 - j
        while (
            j > 0 and _eta_logf(left, i, x0, y[i], alpha, eps, indptr, rows, aval) > z
        ):
            left -= w
            j -= 1
        while (
            k > 0 and _eta_logf(right, i, x0, y[i], alpha, eps, indptr, rows, aval) > z
        ):
            right += w
            k -= 1
        while True:
            x1 = left + rng.random() * (right - left)
            if _eta_logf(x1, i, x0, y[i], alpha, eps, indptr, rows, aval) > z:
                break
            if x1 < x0:
                left = x1
            else:
                right = x1
        d = x1 - x0
        for p in range(indptr[i], indptr[i + 1]):
            eps[rows[p]] += aval[p] * d
        eta[i] = x1


@numba.njit(cache=True)
def _filter_apply(indptr, indices, data, par, v, out):
    """``out = (I - par W) v`` for CSR ``W``."""
    for i in range(v.size):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            acc += data[p] * v[indices[p]]
        out[i] = v[i] - par * acc


@numba.njit(cache=True)
def _bicgstab(indptr, indices, data, par, b, x, rtol, maxiter):
    """Solve ``(I - par W) x = b`` from the start ``x``, in place; ``True`` if converged."""
    n = b.size
    r = np.empty(n)
    _filter_apply(indptr, indices, data, par, x, r)
    r = b - r
    bnorm = np.sqrt(b @ b)
    if bnorm == 0.0:
        x[:] = 0.0
        return True
    tol = rtol * bnorm
    if np.sqrt(r @ r) <= tol:
        return True
    rhat = r.copy()
    p = np.zeros(n)
    v = np.zeros(n)
    s = np.empty(n)
    t = np.empty(n)
    rho_old = alpha = omega = 1.0
    for _ in range(maxiter):
        rho_new = rhat @ r
        if rho_new == 0.0:
            return False
        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        _filter_apply(indptr, indices, data, par, p, v)
        alpha = rho_new / (rhat @ v)
        s = r - alpha * v
        if np.sqrt(s @ s) <= tol:
            x += alpha * p
            return True
        _filter_apply(indptr, indices, data, par, s, t)
        tt = t @ t
        if tt == 0.0:
            return False
        omega = (t @ s) / tt
        x += alpha * p + omega * s
        r = s - omega * t
        if np.sqrt(r @ r) <= tol:
            return True
        rho_old = rho_new
    return False


def _filter_solve(
    W: sp.csr_matrix, par: float, b: np.ndarray, x0: np.ndarray
) -> np.ndarray:
    """``(I - par W)^{-1} b`` by BiCGSTAB from ``x0``, directly if it stalls."""
    x = np.array(x0, dtype=np.float64)
    if _bicgstab(W.indptr, W.indices, W.data, par, b, x, _SOLVE_RTOL, _SOLVE_MAXITER):
        return x
    A = (sp.identity(b.size, format="csc") - par * W).tocsc()
    return sp.linalg.spsolve(A, b)


def _gamma_loglik(e: np.ndarray, alpha: float) -> float:
    """``sum alpha (e - e^e)``: the disturbances' log density, less its constant."""
    return alpha * float(np.sum(e - np.exp(e)))


def _alpha_log_density(n: int, s: float, log_a: float, priors) -> float:
    """``log p(eps | alpha) + log p(alpha) + log alpha`` at ``alpha = e^log_a``.

    ``s`` is ``sum(eps - e^eps)``, the disturbances' sufficient statistic.
    """
    a = np.exp(log_a)
    nu, sigma = priors.alpha_nu, priors.alpha_sigma
    return (
        n * (a * log_a - gammaln(a))
        + a * s
        - 0.5 * (nu + 1.0) * np.log1p(a * a / (nu * sigma * sigma))
        + log_a
    )


def _gamma_moments(alpha: float) -> tuple[float, float]:
    """Mean and sd of ``log gamma`` for ``gamma ~ Gamma(alpha, alpha)``."""
    return float(digamma(alpha) - np.log(alpha)), float(np.sqrt(polygamma(1, alpha)))


def _beta_laplace(c, Z, alpha, b0, B0inv):
    """Mode and negative Hessian of ``beta | c`` for ``eps = c - Z beta``.

    Newton from the least-squares fit of ``c`` less the disturbances' mean,
    with step halving, so the result depends on ``(c, Z, alpha)`` only.
    """
    m, s = _gamma_moments(alpha)
    G = Z.T @ Z / s**2 + B0inv
    b = np.linalg.solve(G, Z.T @ (c - m) / s**2 + B0inv @ b0)

    def logp(x):
        r = x - b0
        return _gamma_loglik(c - Z @ x, alpha) - 0.5 * r @ B0inv @ r

    lp = logp(b)
    for _ in range(20):
        ee = np.exp(c - Z @ b)
        grad = alpha * (Z.T @ (ee - 1.0)) - B0inv @ (b - b0)
        H = alpha * (Z.T * ee) @ Z + B0inv
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:
            cand = b + t * step
            lc = logp(cand)
            if lc >= lp or t < 1e-6:
                break
            t *= 0.5
        b, lp_old, lp = cand, lp, lc
        if abs(lp - lp_old) < 1e-10 * (1.0 + abs(lp)):
            break
    ee = np.exp(c - Z @ b)
    return b, alpha * (Z.T * ee) @ Z + B0inv


def _beta_step(beta, c, Z, alpha, b0, B0inv, rng):
    """``beta | c`` by independence MH from a *t* at the Laplace approximation."""
    k = beta.size
    mode, H = _beta_laplace(c, Z, alpha, b0, B0inv)
    L = np.linalg.cholesky(H)

    def logp(x):
        r = x - b0
        return _gamma_loglik(c - Z @ x, alpha) - 0.5 * r @ B0inv @ r

    def logq(x):
        z = L.T @ (x - mode)
        return -0.5 * (_BETA_DF + k) * np.log1p(z @ z / _BETA_DF)

    scale = np.sqrt(rng.chisquare(_BETA_DF) / _BETA_DF)
    prop = mode + np.linalg.solve(L.T, rng.standard_normal(k)) / scale
    log_ratio = logp(prop) - logp(beta) + logq(beta) - logq(prop)
    if np.log(rng.random()) < log_ratio:
        return prop
    return beta


def _alpha_step(alpha, e, priors, rng):
    """``alpha | eps``: slice on ``log alpha``."""
    n, s = e.size, float(np.sum(e - np.exp(e)))
    la, _ = slice_sample_1d(
        log_density=lambda la: _alpha_log_density(n, s, la, priors),
        x0=np.log(alpha),
        lower=-10.0,
        upper=10.0,
        w=0.5,
        rng=rng,
    )
    return float(np.exp(la))


def _poisson(y: np.ndarray, eta: np.ndarray) -> float:
    return float(y @ eta - np.exp(eta).sum())


def _alpha_interweave(alpha, eta, eps, sxb, s1, y, priors, rng):
    """``alpha`` with ``z = (eps - m_alpha) / s_alpha`` held; returns ``(alpha, eta)``.

    With ``eps' = m' + b (eps - m)``, ``b = s'/s``, the log-means move to
    ``sxb + (m' - b m) S1 + b (eta - sxb)``, where ``sxb`` is ``S X beta``
    (SAR) or ``X beta`` (SEM).  The Jacobian of ``eps`` in ``z`` is
    ``s_alpha^n``.
    """
    n = eps.size
    m, s = _gamma_moments(alpha)
    z = (eps - m) / s
    dev = eta - sxb

    def move(la):
        mm, ss = _gamma_moments(np.exp(la))
        b = ss / s
        return mm + ss * z, sxb + (mm - b * m) * s1 + b * dev, ss

    def logp(la):
        e, et, ss = move(la)
        s = float(np.sum(e - np.exp(e)))
        return _poisson(y, et) + _alpha_log_density(n, s, la, priors) + n * np.log(ss)

    la, _ = slice_sample_1d(
        log_density=logp, x0=np.log(alpha), lower=-10.0, upper=10.0, w=0.5, rng=rng
    )
    return float(np.exp(la)), move(la)[1]


def initial_state(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """``eta = log(y + 1/2)``, ``beta`` its least-squares fit, ``alpha`` from the rest."""
    eta = np.log(y + 0.5)
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    var = float(np.var(eta - X @ beta))
    return eta, beta, float(np.clip(1.0 / max(var, 1e-2), 0.2, 50.0))


def run_chain(
    *,
    y: np.ndarray,
    X: np.ndarray,
    W: sp.csr_matrix,
    process: Literal["sar", "sem"],
    priors: ReducedGibbsPriors,
    logdet_fn: Callable[[float], float],
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator,
    chain_id: int = 0,
    progress_manager=None,
    slice_width: float = 0.4,
) -> dict:
    """One chain; returns ``{"rho" | "lam": ..., "beta": ..., "alpha": ...}``."""
    n, k = X.shape
    y = np.asarray(y, dtype=np.float64)
    W = sp.csr_matrix(W, dtype=np.float64)
    W.sort_indices()
    indptr, rows, ones, wvals = filter_columns(W)
    b0 = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,)).copy()
    B0inv = np.diag(
        1.0 / np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,)) ** 2
    )
    lo, hi = float(priors.rho_lower), float(priors.rho_upper)
    fixed = priors.alpha_fixed
    WX = np.asarray(W @ X)
    XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
    row_std = bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0))
    ones_n = np.ones(n)
    width = SliceWidthState(w=slice_width)
    sar = process == "sar"

    par = 0.0
    eta, beta, alpha = initial_state(y, X)
    if fixed is not None:
        alpha = float(fixed)
    sxb = X @ beta
    s1 = ones_n.copy()
    step = 0.1

    n_keep = draws // thin
    par_draws = np.empty(n_keep)
    beta_draws = np.empty((n_keep, k))
    alpha_draws = np.empty(n_keep)
    for it in range(tune + draws):
        # 1. The log-means, given the disturbances' map.
        if sar:
            eps = eta - par * (W @ eta) - X @ beta
        else:
            r = eta - X @ beta
            eps = r - par * (W @ r)
        _eta_sweep(eta, eps, y, alpha, indptr, rows, ones - par * wvals, rng)

        # 2. The spatial parameter, given eta.
        Weta = W @ eta
        if sar:
            # beta = u - rho g keeps eps = e0 - rho d.
            g = XtX_inv_Xt @ Weta
            u = beta + par * g
            e0 = eta - X @ u
            d = Weta - X @ g

            def logp(r):
                b = u - r * g - b0
                return (
                    logdet_fn(r)
                    + _gamma_loglik(e0 - r * d, alpha)
                    - 0.5 * b @ B0inv @ b
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=width, rng=rng
            )
            beta = u - par * g
            c, Z = eta - par * Weta, X
        else:
            # beta = u + bhat(lam), bhat(lam) the least-squares fit of
            # A eta - m on A X: the disturbances keep their level as lam moves.
            m = _gamma_moments(alpha)[0]
            G0, G1, G2 = X.T @ X, X.T @ WX + WX.T @ X, WX.T @ WX
            h0 = X.T @ (eta - m)
            h1 = X.T @ Weta + WX.T @ (eta - m)
            h2 = WX.T @ Weta

            def bhat(lam):
                return np.linalg.solve(
                    G0 - lam * G1 + lam * lam * G2, h0 - lam * h1 + lam * lam * h2
                )

            u = beta - bhat(par)

            def logp(lam):
                b = u + bhat(lam)
                e = eta - lam * Weta - (X - lam * WX) @ b
                return (
                    logdet_fn(lam)
                    + _gamma_loglik(e, alpha)
                    - 0.5 * (b - b0) @ B0inv @ (b - b0)
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=width, rng=rng
            )
            beta = u + bhat(par)
            c, Z = eta - par * Weta, X - par * WX

        # 3-4. beta and alpha, given eta.
        beta = _beta_step(beta, c, Z, alpha, b0, B0inv, rng)
        eps = c - Z @ beta
        if fixed is None:
            alpha = _alpha_step(alpha, eps, priors, rng)

            # 5. alpha with the standardized disturbances held.
            s1 = ones_n / (1.0 - par) if row_std else _filter_solve(W, par, ones_n, s1)
            if sar:
                sxb = _filter_solve(W, par, X @ beta, sxb)
            else:
                sxb = X @ beta
            alpha, eta = _alpha_interweave(alpha, eta, eps, sxb, s1, y, priors, rng)

        # 6. The spatial parameter with the disturbances held.
        if sar:
            xi = eta - par * (W @ eta)
            cur, base = eta, np.zeros(n)
        else:
            xi = (eta - X @ beta) - par * (W @ (eta - X @ beta))
            cur, base = eta - X @ beta, X @ beta
        prop = par + step * rng.standard_normal()
        accepted = False
        if lo < prop < hi:
            new = _filter_solve(W, prop, xi, cur)
            if np.log(rng.random()) < _poisson(y, base + new) - _poisson(y, base + cur):
                par, eta, accepted = prop, base + new, True
        if it < tune:
            step *= np.exp((float(accepted) - _PAR_ACCEPT) / np.sqrt(it + 1.0))

        if it >= tune and (it - tune) % thin == 0:
            idx = (it - tune) // thin
            if idx < n_keep:
                par_draws[idx] = par
                beta_draws[idx] = beta
                alpha_draws[idx] = alpha
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    return {
        "rho" if sar else "lam": par_draws,
        "beta": beta_draws,
        "alpha": alpha_draws,
    }
