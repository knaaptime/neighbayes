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

A panel stacks ``T`` periods time-first, each with its own disturbances and the
same filter, :math:`A\eta_t = Z_t\beta + c + \varepsilon_t` (SAR), so the
Jacobian is :math:`T\log|A|` and every step runs over the stack.  Period
effects are columns of :math:`Z` with their own prior.  Random unit effects
:math:`c_g \sim N(\mu, \sigma_c^2)` join the systematic part inside the
filter, as in the Gaussian SAR with random effects; each enters only its own
unit's equations, so

7. :math:`c_g \mid \cdot` is one-dimensional and log-concave: independence
   Metropolis–Hastings from a *t* at its mode, all units at once, and
   :math:`\sigma_c` a slice on :math:`\log\sigma_c` under a half-*t*(3) prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numba
import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln, polygamma

from ...models.priors import ReducedGibbsPriors
from .._utils._latent_loglik import POISSON, TRUNCATED, UNOBSERVED, log_positive
from .._utils._slice import SliceWidthState, slice_sample_1d, slice_sample_1d_adaptive
from .._utils._sweeps import run_sweeps

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
def _obs_logf(t, y, kind):
    """``log p(y | eta = t) + log y!`` for the count's observation ``kind``."""
    if kind == UNOBSERVED:
        return 0.0
    v = y * t - np.exp(t)
    if kind == TRUNCATED:
        v -= log_positive(t)
    return v


def _obs_loglik(y: np.ndarray, eta: np.ndarray, kind: np.ndarray) -> float:
    """``sum log p(y | eta) + log y!`` over the observed log-means."""
    on = kind != UNOBSERVED
    v = float(y[on] @ eta[on] - np.exp(eta[on]).sum())
    trunc = kind == TRUNCATED
    if trunc.any():
        t = eta[trunc]
        lp = t - 0.5 * np.exp(t)
        big = t >= -30.0
        lp[big] = np.log(-np.expm1(-np.exp(t[big])))
        v -= float(lp.sum())
    return v


@numba.njit(cache=True)
def _eta_logf(t, i, d0, y_i, kind_i, alpha, eps, indptr, rows, aval):
    """Log density of ``eta_i = t`` given the rest, up to a constant."""
    d = t - d0
    acc = _obs_logf(t, y_i, kind_i)
    for p in range(indptr[i], indptr[i + 1]):
        e = eps[rows[p]] + aval[p] * d
        acc += alpha * (e - np.exp(e))
    return acc


@numba.njit(cache=True)
def _eta_sweep(eta, eps, y, kind, alpha, indptr, rows, aval, rng):
    """One Gibbs pass over the log-means, updating ``eps`` alongside.

    The slice width of ``eta_i`` is a multiple of its conditional sd at
    ``e^eta_i = y_i`` and ``e^eps = 1``, which does not depend on ``eta_i``.
    """
    for i in range(eta.size):
        x0 = eta[i]
        y_i, k_i = y[i], kind[i]
        info = y_i if k_i != UNOBSERVED else 0.0
        for p in range(indptr[i], indptr[i + 1]):
            info += alpha * aval[p] * aval[p]
        w = 2.5 / np.sqrt(info)
        z = _eta_logf(x0, i, x0, y_i, k_i, alpha, eps, indptr, rows, aval)
        z -= rng.standard_exponential()
        left = x0 - w * rng.random()
        right = left + w
        j = int(_ETA_MAX_STEPS * rng.random())
        k = _ETA_MAX_STEPS - 1 - j
        while (
            j > 0
            and _eta_logf(left, i, x0, y_i, k_i, alpha, eps, indptr, rows, aval) > z
        ):
            left -= w
            j -= 1
        while (
            k > 0
            and _eta_logf(right, i, x0, y_i, k_i, alpha, eps, indptr, rows, aval) > z
        ):
            right += w
            k -= 1
        while True:
            x1 = left + rng.random() * (right - left)
            if _eta_logf(x1, i, x0, y_i, k_i, alpha, eps, indptr, rows, aval) > z:
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


def _lag(W: sp.csr_matrix, v: np.ndarray) -> np.ndarray:
    """``(I_T ⊗ W) v`` for a time-first stack ``v`` of shape ``(T, n)`` or ``(T, n, k)``."""
    T, n = v.shape[:2]
    flat = np.moveaxis(v, 0, 1).reshape(n, -1)
    return np.moveaxis(np.asarray(W @ flat).reshape((n, T) + v.shape[2:]), 0, 1)


def _solve_periods(W, par, B, X0):
    """``(I_T ⊗ A)^{-1}`` applied to each period's row of ``B`` (``(T, n)``)."""
    return np.stack([_filter_solve(W, par, B[t], X0[t]) for t in range(B.shape[0])])


@dataclass(frozen=True)
class UnitEffects:
    """Random unit effects ``c_g ~ N(mu, sigma^2)``, ``sigma ~ half-t(3, scale)``.

    ``group`` maps each row of the time-first stack to its unit.
    """

    group: np.ndarray
    n_groups: int
    mu: float
    sd_scale: float


def _unit_effect_step(c, base, alpha, fx: UnitEffects, sigma, rng):
    """``c | base``, ``base = eps + c[group]``: independence MH from a *t* at the mode.

    Each ``c_g`` has the log-concave conditional
    ``sum_r alpha (b_r - c - e^{b_r - c}) - (c - mu)^2 / (2 sigma^2)``.
    """
    G = fx.n_groups
    n_obs = np.bincount(fx.group, minlength=G).astype(np.float64)
    S = np.bincount(fx.group, weights=np.exp(base), minlength=G)
    B = np.bincount(fx.group, weights=base, minlength=G)
    prec = 1.0 / sigma**2

    def logp(x):
        return alpha * (B - n_obs * x - np.exp(-x) * S) - 0.5 * prec * (x - fx.mu) ** 2

    mode = c.copy()
    for _ in range(30):
        grad = alpha * (np.exp(-mode) * S - n_obs) - prec * (mode - fx.mu)
        hess = alpha * np.exp(-mode) * S + prec
        step = np.clip(grad / hess, -2.0, 2.0)
        mode += step
        if np.max(np.abs(step)) < 1e-10:
            break
    sd = 1.0 / np.sqrt(alpha * np.exp(-mode) * S + prec)

    def logq(x):
        z = (x - mode) / sd
        return -0.5 * (_BETA_DF + 1.0) * np.log1p(z * z / _BETA_DF)

    prop = mode + sd * rng.standard_t(_BETA_DF, size=G)
    log_ratio = logp(prop) - logp(c) + logq(c) - logq(prop)
    return np.where(np.log(rng.random(G)) < log_ratio, prop, c)


def _unit_sd_step(sigma, c, fx: UnitEffects, rng):
    """``sigma_c | c``: slice on ``log sigma`` under the half-t(3, scale) prior."""
    ss, G = float(np.sum((c - fx.mu) ** 2)), c.size

    def logp(ls):
        s2 = np.exp(2.0 * ls)
        return (
            -G * ls - 0.5 * ss / s2 - 2.0 * np.log1p(s2 / (3.0 * fx.sd_scale**2)) + ls
        )

    ls, _ = slice_sample_1d(logp, np.log(sigma), -10.0, 10.0, w=0.5, rng=rng)
    return float(np.exp(ls))


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


def _alpha_interweave(alpha, eta, eps, sxb, s1, y, kind, priors, rng):
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
        return (
            _obs_loglik(y, et, kind)
            + _alpha_log_density(n, s, la, priors)
            + n * np.log(ss)
        )

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


class LatentNegBinChain:
    """One chain's state and sweep (see the module).

    ``y`` and ``X`` stack ``T`` periods of ``n`` units time-first, ``W`` being
    the ``n x n`` weights.  ``kind`` says what each count observes
    (:data:`POISSON` everywhere by default); a two-part model sets it between
    sweeps.  :func:`run_chain` runs one alone.
    """

    def __init__(
        self,
        *,
        y: np.ndarray,
        X: np.ndarray,
        W: sp.csr_matrix,
        process: Literal["sar", "sem"],
        priors: ReducedGibbsPriors,
        logdet_fn: Callable[[float], float],
        rng: np.random.Generator,
        slice_width: float = 0.4,
        T: int = 1,
        unit_effects: Optional[UnitEffects] = None,
        kind: Optional[np.ndarray] = None,
    ):
        nT, k = X.shape
        n = nT // T
        self.sar = process == "sar"
        if unit_effects is not None and not self.sar:
            raise NotImplementedError(
                "Random unit effects are implemented for the SAR."
            )
        self.T, self.n, self.k = T, n, k
        self.rng = rng
        self.priors = priors
        self.logdet_fn = logdet_fn
        self.y = np.asarray(y, dtype=np.float64).reshape(T, n)
        self.kind = (
            np.full((T, n), POISSON, dtype=np.int8)
            if kind is None
            else np.asarray(kind, dtype=np.int8).reshape(T, n)
        )
        self.X = X
        W = sp.csr_matrix(W, dtype=np.float64)
        W.sort_indices()
        self.W = W
        self.cols = filter_columns(W)
        self.b0 = np.broadcast_to(
            np.asarray(priors.beta_mu, dtype=np.float64), (k,)
        ).copy()
        self.B0inv = np.diag(
            1.0 / np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,)) ** 2
        )
        self.lo, self.hi = float(priors.rho_lower), float(priors.rho_upper)
        self.fixed = priors.alpha_fixed
        self.WX = _lag(W, X.reshape(T, n, k)).reshape(nT, k)
        self.XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
        self.row_std = bool(np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0))
        self.ones_n = np.ones((T, n))
        self.width = SliceWidthState(w=slice_width)

        self.par = 0.0
        eta, self.beta, self.alpha = initial_state(self.y.reshape(-1), X)
        self.eta = eta.reshape(T, n)
        if self.fixed is not None:
            self.alpha = float(self.fixed)
        self.fx = unit_effects
        fx = unit_effects
        self.c = np.zeros(fx.n_groups) if fx is not None else None
        self.sigma_c = fx.sd_scale if fx is not None else None
        self.sxb = (X @ self.beta).reshape(T, n)
        self.s1 = self.ones_n.copy()
        self.step = 0.1
        self.it = 0

    def _effects(self):
        """The unit effects on the stack, ``(T, n)``."""
        fx = self.fx
        return 0.0 if fx is None else self.c[fx.group].reshape(self.T, self.n)

    def sweep(self, tuning: bool) -> None:
        """One Gibbs sweep."""
        T, n, X, W, rng, sar = self.T, self.n, self.X, self.W, self.rng, self.sar
        y, kind, priors, fx = self.y, self.kind, self.priors, self.fx
        b0, B0inv, lo, hi = self.b0, self.B0inv, self.lo, self.hi
        indptr, rows, ones, wvals = self.cols
        par, eta, beta, alpha = self.par, self.eta, self.beta, self.alpha
        effects = self._effects

        def lag(v):
            return _lag(W, v)

        # 1. The log-means, given the disturbances' map, period by period.
        Xb = (X @ beta).reshape(T, n)
        if sar:
            eps = eta - par * lag(eta) - Xb - effects()
        else:
            r = eta - Xb
            eps = r - par * lag(r)
        aval = ones - par * wvals
        for t in range(T):
            _eta_sweep(eta[t], eps[t], y[t], kind[t], alpha, indptr, rows, aval, rng)

        # 2. The spatial parameter, given eta.
        Weta = lag(eta)
        logdet_fn = self.logdet_fn
        logdet = (lambda r: T * logdet_fn(r)) if T > 1 else logdet_fn
        if sar:
            # beta = u - rho g keeps eps = e0 - rho d.
            g = self.XtX_inv_Xt @ Weta.reshape(-1)
            u = beta + par * g
            e0 = (eta - effects()).reshape(-1) - X @ u
            d = Weta.reshape(-1) - X @ g

            def logp(r):
                b = u - r * g - b0
                return (
                    logdet(r) + _gamma_loglik(e0 - r * d, alpha) - 0.5 * b @ B0inv @ b
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=self.width, rng=rng
            )
            beta = u - par * g
            target, Z = (eta - par * Weta - effects()).reshape(-1), X
        else:
            # beta = u + bhat(lam), bhat(lam) the least-squares fit of
            # A eta - m on A X: the disturbances keep their level as lam moves.
            WX = self.WX
            m = _gamma_moments(alpha)[0]
            ev, We = eta.reshape(-1), Weta.reshape(-1)
            G0, G1, G2 = X.T @ X, X.T @ WX + WX.T @ X, WX.T @ WX
            h0 = X.T @ (ev - m)
            h1 = X.T @ We + WX.T @ (ev - m)
            h2 = WX.T @ We

            def bhat(lam):
                return np.linalg.solve(
                    G0 - lam * G1 + lam * lam * G2, h0 - lam * h1 + lam * lam * h2
                )

            u = beta - bhat(par)

            def logp(lam):
                b = u + bhat(lam)
                e = ev - lam * We - (X - lam * WX) @ b
                return (
                    logdet(lam)
                    + _gamma_loglik(e, alpha)
                    - 0.5 * (b - b0) @ B0inv @ (b - b0)
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=self.width, rng=rng
            )
            beta = u + bhat(par)
            target, Z = ev - par * We, X - par * WX

        # 3-4. beta, the unit effects and alpha, given eta.
        beta = _beta_step(beta, target, Z, alpha, b0, B0inv, rng)
        eps = target - Z @ beta
        self.par, self.beta = par, beta
        if fx is not None:
            base = eps + self.c[fx.group]
            self.c = _unit_effect_step(self.c, base, alpha, fx, self.sigma_c, rng)
            self.sigma_c = _unit_sd_step(self.sigma_c, self.c, fx, rng)
            eps = base - self.c[fx.group]
        if self.fixed is None:
            alpha = _alpha_step(alpha, eps, priors, rng)

            # 5. alpha with the standardized disturbances held.
            if self.row_std:
                self.s1 = self.ones_n / (1.0 - par)
            else:
                self.s1 = _solve_periods(W, par, self.ones_n, self.s1)
            Xb = (X @ beta).reshape(T, n)
            if sar:
                self.sxb = _solve_periods(W, par, Xb + effects(), self.sxb)
            else:
                self.sxb = Xb
            alpha, eta_v = _alpha_interweave(
                alpha,
                eta.reshape(-1),
                eps,
                self.sxb.reshape(-1),
                self.s1.reshape(-1),
                y.reshape(-1),
                kind.reshape(-1),
                priors,
                rng,
            )
            eta = eta_v.reshape(T, n)
        self.alpha = alpha

        # 6. The spatial parameter with the disturbances held.
        Xb = (X @ beta).reshape(T, n)
        if sar:
            xi = eta - par * lag(eta)
            cur, base_eta = eta, np.zeros((T, n))
        else:
            xi = (eta - Xb) - par * lag(eta - Xb)
            cur, base_eta = eta - Xb, Xb
        prop = par + self.step * rng.standard_normal()
        accepted = False
        if lo < prop < hi:
            new = _solve_periods(W, prop, xi, cur)
            yv, kv = y.reshape(-1), kind.reshape(-1)
            log_ratio = _obs_loglik(yv, (base_eta + new).reshape(-1), kv) - (
                _obs_loglik(yv, (base_eta + cur).reshape(-1), kv)
            )
            if np.log(rng.random()) < log_ratio:
                par, eta, accepted = prop, base_eta + new, True
        if tuning:
            self.step *= np.exp(
                (float(accepted) - _PAR_ACCEPT) / np.sqrt(self.it + 1.0)
            )
        self.par, self.eta = par, eta
        self.it += 1

    def draw(self, store_latent: bool) -> dict:
        """The current values, named as :func:`run_chain` returns them."""
        out = {
            "rho" if self.sar else "lam": self.par,
            "beta": self.beta,
            "alpha": self.alpha,
        }
        if self.fx is not None:
            out["unit_effect"] = self.c
            out["unit_effect_sd"] = self.sigma_c
        if store_latent:
            out["latent"] = self.eta.reshape(-1)
        return out


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
    store_latent: bool = False,
    T: int = 1,
    unit_effects: Optional[UnitEffects] = None,
    kind: Optional[np.ndarray] = None,
) -> dict:
    """One chain; returns ``{"rho" | "lam": ..., "beta": ..., "alpha": ...}``.

    ``y`` and ``X`` stack ``T`` periods of ``n`` units time-first, ``W`` being
    the ``n x n`` weights; ``unit_effects`` (SAR only) adds ``"unit_effect"``
    and ``"unit_effect_sd"``; ``kind`` is what each count observes
    (:data:`POISSON` by default).  ``store_latent`` adds the log-means at every
    kept draw, ``"latent"``, flattened time-first.
    """
    chain = LatentNegBinChain(
        y=y,
        X=X,
        W=W,
        process=process,
        priors=priors,
        logdet_fn=logdet_fn,
        rng=rng,
        slice_width=slice_width,
        T=T,
        unit_effects=unit_effects,
        kind=kind,
    )
    return run_sweeps(
        chain,
        draws=draws,
        tune=tune,
        thin=thin,
        chain_id=chain_id,
        progress_manager=progress_manager,
        store_latent=store_latent,
    )
