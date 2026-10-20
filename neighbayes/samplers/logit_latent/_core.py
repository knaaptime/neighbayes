r"""Gibbs sampler for the latent SAR and SEM logit.

With :math:`A = I - \rho W` (SAR) or :math:`I - \lambda W` (SEM), the model is

.. math::
    y_i = \mathbf 1\{y^*_i > 0\}, \qquad
    \text{SAR: } A y^* = X\beta + \varepsilon, \qquad
    \text{SEM: } A (y^* - X\beta) = \varepsilon,

with :math:`\varepsilon_i` iid standard logistic, so that :math:`\rho = 0`
(:math:`\lambda = 0`) is the logit.  The logistic is a normal scale mixture,
:math:`\varepsilon_i \mid m_i \sim N(0, m_i^2)` with :math:`m_i = 2K_i` and
:math:`K_i` Kolmogorov distributed, so given the scales ``m`` (``mix_scale``)
the latent utilities are Gaussian with precision :math:`Q = A^\top P A`,
:math:`P = \mathrm{diag}(m^{-2})`.  Each sweep is exact (Albert & Chib 1993 for
the latent utilities, Holmes & Held 2006 for the scales):

1. :math:`y^* \mid y, \beta, \rho, m`: Gaussian truncated to the orthant of
   ``y``, one coordinate at a time (Geweke 1991), from the precision ``Q`` and
   linear term ``b`` of the joint (no solve).
2. :math:`m \mid \varepsilon`: :func:`augax.ks_sample_numpy`.
3. :math:`\rho \mid y^*, m`: slice on the density with :math:`\beta`
   integrated out.  For the SAR :math:`X^\top P X` does not involve
   :math:`\rho`, so each evaluation is a quadratic plus the log-determinant;
   for the SEM it is a :math:`k \times k` solve plus the log-determinant.
4. :math:`\beta \mid \rho, y^*, m`: Gaussian.

There is no Pólya–Gamma step: ``y`` is a threshold of :math:`y^*`, not a
Bernoulli draw given it, and a PG layer would add a second, local logistic
disturbance, whose :math:`\rho = 0` member is not the logit.
"""

from __future__ import annotations

from typing import Callable, Literal

import numba
import numpy as np
import scipy.sparse as sp

from ...models.priors import LogitGibbsPriors
from .._utils._slice import SliceWidthState, slice_sample_1d_adaptive


@numba.njit(cache=True)
def _rtnorm_lower(a, rng):
    """``z ~ N(0, 1)`` truncated to ``z > a`` (Robert 1995 above a = 0.45)."""
    if a < 0.45:
        while True:
            z = rng.standard_normal()
            if z > a:
                return z
    alpha = 0.5 * (a + np.sqrt(a * a + 4.0))
    while True:
        z = a + rng.standard_exponential() / alpha
        if rng.random() <= np.exp(-0.5 * (z - alpha) ** 2):
            return z


@numba.njit(cache=True)
def _orthant_sweep(ystar, sign, indptr, indices, data, b, rng):
    """One Gibbs pass over the latent utilities.

    ``ystar | rest`` has density proportional to
    ``exp(-ystar' Q ystar / 2 + b' ystar)`` on the orthant ``sign * ystar > 0``;
    ``Q`` is the CSR triple ``(indptr, indices, data)``.
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
        if sign[i] > 0:
            ystar[i] = mean + sd * _rtnorm_lower(-mean / sd, rng)
        else:
            ystar[i] = mean - sd * _rtnorm_lower(mean / sd, rng)


def _chol_solve(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.linalg.solve(L, v))


def run_chain(
    *,
    y: np.ndarray,
    X: np.ndarray,
    W: sp.csr_matrix,
    process: Literal["sar", "sem"],
    priors: LogitGibbsPriors,
    logdet_fn: Callable[[float], float],
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator,
    chain_id: int = 0,
    progress_manager=None,
    slice_width: float = 0.4,
) -> dict:
    """One chain; returns ``{"rho" | "lam": (n_keep,), "beta": (n_keep, k)}``."""
    from augax import ks_sample_numpy

    n, k = X.shape
    sign = np.where(y > 0, 1, -1).astype(np.int8)
    W = sp.csr_matrix(W)
    eye = sp.identity(n, format="csr")
    b0 = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    B0inv = np.diag(
        1.0 / np.broadcast_to(np.asarray(priors.beta_sigma, float), (k,)) ** 2
    )
    B0inv_b0 = B0inv @ b0
    lo, hi = float(priors.rho_lower), float(priors.rho_upper)
    WX = np.asarray(W @ X)
    width = SliceWidthState(w=slice_width)

    # Logit start: the utilities on their side of zero, the prior mean of the
    # scales (E[2K] = sqrt(2 pi) log 2), no dependence.
    par = 0.0
    beta = b0.copy()
    ystar = 0.5 * sign.astype(np.float64)
    m = np.full(n, np.sqrt(2.0 * np.pi) * np.log(2.0))

    n_keep = draws // thin
    par_draws = np.empty(n_keep)
    beta_draws = np.empty((n_keep, k))
    for it in range(tune + draws):
        A = (eye - par * W).tocsr()
        p = 1.0 / (m * m)
        Q = (A.T @ sp.diags(p) @ A).tocsr()
        Q.sort_indices()
        Xb = X @ beta
        b = A.T @ (p * Xb) if process == "sar" else Q @ Xb
        _orthant_sweep(ystar, sign, Q.indptr, Q.indices, Q.data, b, rng)

        eps = A @ ystar - Xb if process == "sar" else A @ (ystar - Xb)
        m = ks_sample_numpy(eps, rng)
        p = 1.0 / (m * m)
        z = W @ ystar
        py, pz = p * ystar, p * z

        if process == "sar":
            # r = y* - rho z; m(rho) = X'P r + B0^-1 b0 = c0 - rho c1.
            Binv = X.T @ (p[:, None] * X) + B0inv
            L = np.linalg.cholesky(Binv)
            c0 = X.T @ py + B0inv_b0
            c1 = X.T @ pz
            Bc0, Bc1 = _chol_solve(L, c0), _chol_solve(L, c1)
            q00, q01, q11 = c0 @ Bc0, c0 @ Bc1, c1 @ Bc1
            yPy, yPz, zPz = ystar @ py, ystar @ pz, z @ pz

            def logp(r):
                return (
                    logdet_fn(r)
                    - 0.5 * (yPy - 2.0 * r * yPz + r * r * zPz)
                    + 0.5 * (q00 - 2.0 * r * q01 + r * r * q11)
                )

            par, _, _, _ = slice_sample_1d_adaptive(
                logp, par, lo, hi, width_state=width, rng=rng
            )
            mean = Bc0 - par * Bc1
        else:
            # X~ = X - lam WX, r = y* - lam z: every term is quadratic in lam.
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
                logp, par, lo, hi, width_state=width, rng=rng
            )
            L, mm = blocks(par)
            mean = _chol_solve(L, mm)

        # beta ~ N(mean, Binv^-1): mean + L^-T z.
        beta = mean + np.linalg.solve(L.T, rng.standard_normal(k))

        if it >= tune and (it - tune) % thin == 0:
            idx = (it - tune) // thin
            if idx < n_keep:
                par_draws[idx] = par
                beta_draws[idx] = beta
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)

    return {"rho" if process == "sar" else "lam": par_draws, "beta": beta_draws}
