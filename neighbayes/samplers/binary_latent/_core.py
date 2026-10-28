r"""Gibbs sampler for the latent SAR and SEM logit and probit.

With :math:`A = I - \rho W` (SAR) or :math:`I - \lambda W` (SEM), the model is

.. math::
    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    \text{SAR: } A y^* = X\beta + \varepsilon, \qquad
    \text{SEM: } A (y^* - X\beta) = \varepsilon,

with :math:`\varepsilon_i` iid standard logistic (``link="logit"``) or
standard normal (``"probit"``), so that :math:`\rho = 0` (:math:`\lambda = 0`)
is the logit or the probit.  The logistic is a normal scale mixture,
:math:`\varepsilon_i \mid m_i \sim N(0, m_i^2)` with :math:`m_i = 2K_i` and
:math:`K_i` Kolmogorov distributed; the probit's scales are :math:`m_i = 1`.
Given the scales ``m`` (``mix_scale``) the latent utilities are Gaussian with
precision :math:`Q = A^\top P A`, :math:`P = \mathrm{diag}(m^{-2})`.  Each
sweep is exact (Albert & Chib 1993 for the latent utilities, Holmes & Held
2006 for the scales):

1. :math:`y^* \mid y, \beta, \rho, m`: Gaussian truncated to the orthant of
   ``y``, one coordinate at a time (Geweke 1991), from the precision ``Q`` and
   linear term ``b`` of the joint (no solve).
2. :math:`m \mid \varepsilon`: :func:`augax.ks_sample_numpy` (the logit only).
3. :math:`\rho \mid y^*, m`: slice on the density with :math:`\beta`
   integrated out.  For the SAR :math:`X^\top P X` does not involve
   :math:`\rho`, so each evaluation is a quadratic plus the log-determinant;
   for the SEM it is a :math:`k \times k` solve plus the log-determinant.
4. :math:`\beta \mid \rho, y^*, m`: Gaussian.

There is no Pólya–Gamma step: ``y`` is a threshold of :math:`y^*`, not a
Bernoulli draw given it, and a PG layer would add a second, local logistic
disturbance, whose :math:`\rho = 0` member is not the logit.

A zero-inflated count's zero leaves its activation unknown: its utility's
side of zero is drawn with it, the positive side weighted by the probability
:math:`q_i` that an active unit counts zero (``sign`` 0, ``logq``).

A SAR may carry random unit effects inside its filter, :math:`A y^* = X\beta
+ c_{g(i)} + \varepsilon`, :math:`c_g \sim N(\mu, \sigma_c^2)`: they offset
steps 1–4, and given the scales they are Gaussian (step 5), :math:`\sigma_c`
a slice under its half-*t* prior.

6. The scale move (Liu & Wu 1999): :math:`(y^*, \beta, c) \to g\,(y^*,
   \beta, c)` keeps every sign, so the data, and maps :math:`\varepsilon` to
   :math:`g\varepsilon`; drawn from its conditional under the Haar measure
   :math:`dg/g`, with :math:`d` coordinates scaled,

   .. math::
       \log \pi(g) = (d - 1)\log g - \tfrac{g^2}{2}\varepsilon^\top P
       \varepsilon - \tfrac12 (g\beta - b_0)^\top B_0^{-1} (g\beta - b_0)
       - \sum_g \frac{(g c_g - \mu)^2}{2\sigma_c^2},

   it is exact: :math:`g^2 \sim \mathrm{Gamma}(d/2, Q/2)` when the prior
   means are zero, a slice on :math:`\log g` otherwise.  It moves the
   utilities' scale, which the conditional steps cross slowly when
   :math:`\rho` is near one.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import numba
import numpy as np
import scipy.sparse as sp

from ...models.priors import BinaryGibbsPriors
from .._utils._slice import SliceWidthState, slice_sample_1d, slice_sample_1d_adaptive
from ..negbin_latent._core import UnitEffects, _unit_sd_step


@numba.njit(cache=True)
def _rtnorm_lower(a, rng):
    """``z ~ N(0, 1)`` truncated to ``z > a`` (Robert 1995 above a = 0.45).

    A non-finite bound has no draw, and would never accept one: it raises.
    """
    if not np.isfinite(a):
        raise ValueError("truncated-normal bound is not finite")
    if a < 0.45:
        while True:
            z = rng.standard_normal()
            if z > a:
                return z
    # The optimal rate; far out, where a * a would overflow, it is a.
    alpha = 0.5 * (a + np.sqrt(a * a + 4.0)) if a < 1e150 else a
    while True:
        z = a + rng.standard_exponential() / alpha
        if rng.random() <= np.exp(-0.5 * (z - alpha) ** 2):
            return z


@numba.njit(cache=True)
def _log_ndtr(x):
    """``log Phi(x)``, by its asymptotic series far in the lower tail."""
    if x > -20.0:
        return np.log(0.5 * math.erfc(-x / np.sqrt(2.0)))
    r = 1.0 / (x * x)
    return (
        -0.5 * x * x
        - np.log(-x)
        - 0.5 * np.log(2.0 * np.pi)
        + np.log1p(-r + 3.0 * r * r - 15.0 * r * r * r)
    )


@numba.njit(cache=True)
def _orthant_sweep(ystar, sign, logq, pneg, indptr, indices, data, b, rng):
    """One Gibbs pass over the latent utilities.

    ``ystar | rest`` has density proportional to
    ``exp(-ystar' Q ystar / 2 + b' ystar)`` on the orthant ``sign * ystar > 0``;
    ``Q`` is the CSR triple ``(indptr, indices, data)``.  Where ``sign`` is 0
    the positive side is weighted by ``exp(logq)`` and the side drawn first;
    ``pneg`` receives its probability of the negative side.
    """
    for i in range(ystar.size):
        qii = 0.0
        acc = b[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j == i:
                qii = data[p]
            else:
                acc -= data[p] * ystar[j]
        sd = 1.0 / np.sqrt(qii)
        mean = acc / qii
        side = sign[i]
        if side == 0:
            log_odds = _log_ndtr(mean / sd) + logq[i] - _log_ndtr(-mean / sd)
            pneg[i] = 1.0 / (1.0 + np.exp(log_odds))
            side = -1 if rng.random() < pneg[i] else 1
        if side > 0:
            ystar[i] = mean + sd * _rtnorm_lower(-mean / sd, rng)
        else:
            ystar[i] = mean - sd * _rtnorm_lower(mean / sd, rng)


def _chol_solve(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.linalg.solve(L, v))


def _unit_effect_step(e, p, fx: UnitEffects, sigma, rng):
    """``c | e``, ``e = c[group] + eps`` with ``eps_r ~ N(0, 1 / p_r)``: Gaussian."""
    G = fx.n_groups
    prec = np.bincount(fx.group, weights=p, minlength=G) + 1.0 / sigma**2
    mean = np.bincount(fx.group, weights=p * e, minlength=G) + fx.mu / sigma**2
    return mean / prec + rng.standard_normal(G) / np.sqrt(prec)


class LatentBinaryChain:
    """One chain's state and sweep (see the module).

    ``link`` is ``"logit"`` or ``"probit"``.  ``sign`` is each utility's side of zero (``+1``, ``-1``, or ``0`` for
    unknown, which needs ``logq`` set before each sweep); by default it is
    that of ``y``.  ``unit_effects`` adds a SAR's random unit effects.
    :func:`~neighbayes.samplers._utils._sweeps.run_sweeps` runs one.
    """

    def __init__(
        self,
        *,
        y: np.ndarray,
        X: np.ndarray,
        W: sp.csr_matrix,
        process: Literal["sar", "sem"],
        priors: BinaryGibbsPriors,
        logdet_fn: Callable[[float], float],
        rng: np.random.Generator,
        link: Literal["logit", "probit"] = "logit",
        slice_width: float = 0.4,
        sign: Optional[np.ndarray] = None,
        unit_effects: Optional[UnitEffects] = None,
    ):
        n, k = X.shape
        self.sar = process == "sar"
        self.logit = link == "logit"
        if unit_effects is not None and not self.sar:
            raise NotImplementedError(
                "Random unit effects are implemented for the SAR."
            )
        self.n, self.k, self.rng = n, k, rng
        self.sign = (
            np.where(y > 0, 1, -1).astype(np.int8)
            if sign is None
            else np.asarray(sign, dtype=np.int8)
        )
        self.logq = np.zeros(n)
        self.pneg = np.zeros(n)
        self.X = X
        self.W = sp.csr_matrix(W)
        self.eye = sp.identity(n, format="csr")
        self.logdet_fn = logdet_fn
        self.b0 = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
        self.B0inv = np.diag(
            1.0 / np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,)) ** 2
        )
        self.B0inv_b0 = self.B0inv @ self.b0
        self.lo, self.hi = float(priors.rho_lower), float(priors.rho_upper)
        self.WX = np.asarray(self.W @ X)
        self.width = SliceWidthState(w=slice_width)

        # Start: the utilities on their side of zero, the scales at the
        # logit's prior mean (E[2K] = sqrt(2 pi) log 2) or the probit's 1, no
        # dependence.
        self.par = 0.0
        self.beta = self.b0.copy()
        self.ystar = 0.5 * np.where(self.sign != 0, self.sign, 1).astype(np.float64)
        self.m = np.full(n, np.sqrt(2.0 * np.pi) * np.log(2.0) if self.logit else 1.0)
        self.fx = unit_effects
        if unit_effects is not None:
            self.c = np.full(unit_effects.n_groups, float(unit_effects.mu))
            self.sigma_c = float(unit_effects.sd_scale)
        # The scale move's dimension, and whether its draw is a gamma.
        self.scale_dim = n + k + (unit_effects.n_groups if unit_effects else 0)
        self.scale_gamma = not np.any(self.b0) and (
            unit_effects is None or unit_effects.mu == 0.0
        )

    def sweep(self, tuning: bool) -> None:
        """One Gibbs sweep."""
        X, W, rng, k = self.X, self.W, self.rng, self.k
        par, beta, ystar, m = self.par, self.beta, self.ystar, self.m
        logdet_fn, B0inv, B0inv_b0 = self.logdet_fn, self.B0inv, self.B0inv_b0
        lo, hi = self.lo, self.hi

        A = (self.eye - par * W).tocsr()
        p = 1.0 / (m * m)
        Q = (A.T @ sp.diags(p) @ A).tocsr()
        Q.sort_indices()
        offset = self.c[self.fx.group] if self.fx is not None else 0.0
        Xb = X @ beta + offset
        b = A.T @ (p * Xb) if self.sar else Q @ Xb
        _orthant_sweep(
            ystar, self.sign, self.logq, self.pneg, Q.indptr, Q.indices, Q.data, b, rng
        )

        if self.logit:
            from augax import ks_sample_numpy

            eps = A @ ystar - Xb if self.sar else A @ (ystar - Xb)
            m = ks_sample_numpy(eps, rng)
            p = 1.0 / (m * m)
        z = W @ ystar
        # The SAR's response, less its unit effects.
        yt = ystar - offset
        py, pz = p * yt, p * z

        if self.sar:
            # r = y* - rho z; m(rho) = X'P r + B0^-1 b0 = c0 - rho c1.
            Binv = X.T @ (p[:, None] * X) + B0inv
            L = np.linalg.cholesky(Binv)
            c0 = X.T @ py + B0inv_b0
            c1 = X.T @ pz
            Bc0, Bc1 = _chol_solve(L, c0), _chol_solve(L, c1)
            q00, q01, q11 = c0 @ Bc0, c0 @ Bc1, c1 @ Bc1
            yPy, yPz, zPz = yt @ py, yt @ pz, z @ pz

            def logp(r):
                return (
                    logdet_fn(r)
                    - 0.5 * (yPy - 2.0 * r * yPz + r * r * zPz)
                    + 0.5 * (q00 - 2.0 * r * q01 + r * r * q11)
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=self.width, rng=rng
            )
            mean = Bc0 - par * Bc1
        else:
            # X~ = X - lam WX, r = y* - lam z: every term is quadratic in lam.
            WX = self.WX
            G0 = X.T @ (p[:, None] * X)
            G1 = X.T @ (p[:, None] * WX)
            G1 = G1 + G1.T
            G2 = WX.T @ (p[:, None] * WX)
            h0, h1, h2 = X.T @ py, X.T @ pz + WX.T @ py, WX.T @ pz
            s0, s1, s2 = ystar @ py, 2.0 * (ystar @ pz), z @ pz

            def blocks(lam):
                Binv = G0 - lam * G1 + lam * lam * G2 + B0inv
                return np.linalg.cholesky(
                    Binv
                ), h0 - lam * h1 + lam * lam * h2 + B0inv_b0

            def logp(lam):
                L, mm = blocks(lam)
                w = np.linalg.solve(L, mm)
                return (
                    logdet_fn(lam)
                    - 0.5 * (s0 - lam * s1 + lam * lam * s2)
                    + 0.5 * (w @ w)
                    - np.sum(np.log(np.diag(L)))
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=self.width, rng=rng
            )
            L, mm = blocks(par)
            mean = _chol_solve(L, mm)

        # beta ~ N(mean, Binv^-1): mean + L^-T z.
        self.beta = mean + np.linalg.solve(L.T, rng.standard_normal(k))
        self.par, self.m = par, m

        # 5. The unit effects, given the scales, and their sd.
        offset = 0.0
        if self.fx is not None:
            e = ystar - par * z - X @ self.beta
            self.c = _unit_effect_step(e, p, self.fx, self.sigma_c, rng)
            self.sigma_c = _unit_sd_step(self.sigma_c, self.c, self.fx, rng)
            offset = self.c[self.fx.group]

        # 6. The scale move.
        if self.sar:
            e = ystar - par * z - X @ self.beta - offset
        else:
            e = (ystar - X @ self.beta) - par * (z - self.WX @ self.beta)
        g = self._scale(float(e @ (p * e)))
        self.ystar = ystar * g
        self.beta = self.beta * g
        if self.fx is not None:
            self.c = self.c * g

    def _scale(self, epe: float) -> float:
        """The scale move's ``g`` (see the module), ``epe`` being ``eps' P eps``."""
        d, beta, B0inv, rng = self.scale_dim, self.beta, self.B0inv, self.rng
        fx = self.fx
        if self.scale_gamma:
            Q = epe + beta @ B0inv @ beta
            if fx is not None:
                Q += float(self.c @ self.c) / self.sigma_c**2
            return float(np.sqrt(rng.gamma(0.5 * d, 2.0 / Q)))
        b0 = self.b0

        def logp(lg):
            g = np.exp(lg)
            r = g * beta - b0
            v = d * lg - 0.5 * g * g * epe - 0.5 * r @ B0inv @ r
            if fx is not None:
                v -= 0.5 * float(np.sum((g * self.c - fx.mu) ** 2)) / self.sigma_c**2
            return v

        lg, _ = slice_sample_1d(logp, 0.0, -10.0, 10.0, w=2.0 / np.sqrt(d), rng=rng)
        return float(np.exp(lg))

    def draw(self, store_latent: bool) -> dict:
        """The current values: ``rho`` or ``lam``, and ``beta``.

        Unit effects add ``unit_effect`` and ``unit_effect_sd``; ``store_latent``
        adds the utilities, ``latent``.
        """
        out = {"rho" if self.sar else "lam": self.par, "beta": self.beta}
        if self.fx is not None:
            out["unit_effect"] = self.c
            out["unit_effect_sd"] = self.sigma_c
        if store_latent:
            out["latent"] = self.ystar
        return out
