r"""Log marginal likelihood of Gaussian spatial models by deterministic quadrature.

For OLS, SLX, SAR, SDM, SEM and SDEM with Gaussian errors, the marginal
likelihood

.. math::

    p(y) = \int\!\!\int\!\!\int p(y \mid \beta, \sigma^2, \rho)\,
           p(\beta)\, p(\sigma^2)\, p(\rho)\; d\beta\, d\sigma^2\, d\rho

reduces to a two-dimensional integral that can be computed to quadrature
precision, with no Monte Carlo error.  The model's own priors are used
unchanged: β ~ N(μ₀, diag(s²)), σ² ~ InvGamma(a, b), ρ ~ Uniform(lo, hi).

**β, analytically.**  With A = I − ρW, the filtered response ``Ay``
(SAR/SDM) or ``Ay`` with filtered design ``AX`` (SEM/SDEM) is Gaussian given
(ρ, σ²) once β is integrated out:

.. math::

    p(y \mid \rho, \sigma^2) = |A|\; \mathcal N\!\left(r;\, 0,\,
        \sigma^2 I + G G^\top\right),
    \qquad r = y^* - X^*\mu_0,\quad G = X^* \operatorname{diag}(s).

``GGᵀ`` has rank k, so with the thin SVD ``G = U S Vᵀ`` the covariance's
log-determinant is ``(n−k) log σ² + Σ log(σ² + s_j²)`` and the quadratic form
is ``(rᵀr − |Uᵀr|²)/σ² + Σ (Uᵀr)_j² / (σ² + s_j²)``.  For SAR, SDM, OLS and SLX
the design does not depend on ρ, so ``U`` and ``S`` are computed once; for
SEM/SDEM they come from a k×k eigendecomposition at each ρ node.  Either way an
evaluation at fixed ρ costs O(k), plus the log-determinant.

**σ² and ρ, numerically.**  The σ² integral is taken in ``t = log σ²`` by
adaptive Gauss–Kronrod quadrature around its mode; the ρ integral likewise,
with the mode passed as a breakpoint so a sharply peaked posterior is not
stepped over.  Both are done on the log scale relative to their maxima.

The log-determinant is the model's own evaluator, the same one the Gibbs and
NUTS paths use, so this and bridge sampling estimate the same quantity.

References
----------
LeSage, J. P., & Parent, O. (2007). Bayesian model averaging for spatial
econometric models. *Geographical Analysis*, 39(3), 241–267.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammaln

__all__ = ["quadrature_log_marginal_likelihood"]

_LOG_2PI = math.log(2.0 * math.pi)


class _GaussianEvidence:
    """``log p(y | ρ, σ²)`` with β integrated out, for one fitted model."""

    def __init__(self, model):
        jacobian_param = getattr(model, "_jacobian_param", "__missing__")
        if jacobian_param not in (None, "rho", "lam"):
            raise ValueError(
                f"{type(model).__name__} is not a Gaussian cross-sectional model; "
                "quadrature marginal likelihoods cover OLS, SLX, SAR, SDM, SEM "
                "and SDEM."
            )
        if getattr(model, "_likelihood", "gaussian") != "gaussian":
            raise ValueError(
                f"{type(model).__name__} does not have a Gaussian likelihood."
            )
        if getattr(model, "robust", False):
            raise ValueError(
                "Quadrature marginal likelihoods need Gaussian errors: with "
                "Student-t errors β cannot be integrated out in closed form. "
                "Use method='bridge' for robust models."
            )

        y = np.asarray(model._y, dtype=np.float64)
        Z = np.asarray(model._design_matrix(), dtype=np.float64)
        priors = model._gaussian_priors(Z, model._design_names())
        n, k = Z.shape
        mu = np.asarray(priors["beta_mu"], dtype=np.float64) * np.ones(k)
        s = np.asarray(priors["beta_sigma"], dtype=np.float64) * np.ones(k)

        self.n, self.k = n, k
        self.a0 = float(priors["sigma2_alpha"])
        self.b0 = float(priors["sigma2_beta"])
        self.kind = jacobian_param

        a = y - Z @ mu
        if jacobian_param is None:
            self.lo = self.hi = 0.0
            self.logdet_fn: Callable[[float], float] | None = None
            c = np.zeros(n)
        else:
            self.lo = float(priors[f"{jacobian_param}_lower"])
            self.hi = float(priors[f"{jacobian_param}_upper"])
            self.logdet_fn = model._logdet_numpy_fn
            Wy = np.asarray(model._Wy, dtype=np.float64)
            if jacobian_param == "rho":
                c = Wy
            else:
                WZ = np.asarray(model._spatial_lag(Z), dtype=np.float64)
                c = Wy - WZ @ mu

        # r(ρ) = a − ρ c in every case.
        self.aa, self.ac, self.cc = float(a @ a), float(a @ c), float(c @ c)

        if jacobian_param == "lam":
            # X*(λ) = Z − λWZ: keep the λ-independent pieces of X*ᵀX* and X*ᵀr.
            self.s = s
            self.ZtZ = Z.T @ Z
            self.ZtWZ = Z.T @ WZ
            self.WZtWZ = WZ.T @ WZ
            self.Zta, self.Ztc = Z.T @ a, Z.T @ c
            self.WZta, self.WZtc = WZ.T @ a, WZ.T @ c
        else:
            # X* = Z for every ρ: one thin SVD of G = Z diag(s).
            U, sv, _ = np.linalg.svd(Z * s, full_matrices=False)
            self.s2_fixed = sv * sv
            self.Uta, self.Utc = U.T @ a, U.T @ c

    def _projection(self, rho: float) -> tuple[np.ndarray, np.ndarray]:
        """``(s_j², Uᵀr)`` at spatial parameter ``rho``."""
        if self.kind != "lam":
            return self.s2_fixed, self.Uta - rho * self.Utc
        lam = rho
        XtX = self.ZtZ - lam * (self.ZtWZ + self.ZtWZ.T) + lam * lam * self.WZtWZ
        Xtr = self.Zta - lam * (self.Ztc + self.WZta) + lam * lam * self.WZtc
        s2, V = np.linalg.eigh(self.s[:, None] * XtX * self.s[None, :])
        s2 = np.maximum(s2, 1e-300)
        # Uᵀr = S⁻¹ Vᵀ Gᵀr with Gᵀr = diag(s) X*ᵀr.
        return s2, (V.T @ (self.s * Xtr)) / np.sqrt(s2)

    def log_integrand_t(self, rho: float) -> Callable[[np.ndarray], np.ndarray]:
        """``t ↦ log p(y | ρ, σ²=eᵗ) + log p(σ²) + t`` (the dσ² = eᵗ dt Jacobian)."""
        n, k, a0, b0 = self.n, self.k, self.a0, self.b0
        rr = self.aa - 2.0 * rho * self.ac + rho * rho * self.cc
        s2, z = self._projection(rho)
        z2 = z * z
        resid_perp = max(rr - float(z2.sum()), 0.0)
        logdet_A = 0.0 if self.logdet_fn is None else float(self.logdet_fn(rho))
        const = logdet_A - 0.5 * n * _LOG_2PI + a0 * math.log(b0) - gammaln(a0)

        def f(t):
            t = np.asarray(t, dtype=np.float64)
            sig2 = np.exp(t)
            dens = np.add.outer(sig2, s2)  # (..., k)
            log_c = (n - k) * t + np.log(dens).sum(axis=-1)
            quad = resid_perp / sig2 + (z2 / dens).sum(axis=-1)
            return const - 0.5 * (log_c + quad) - a0 * t - b0 / sig2

        return f


def _log_integrate(f: Callable, lo: float, hi: float, mode_guess: float, epsrel: float):
    """``log ∫ exp(f)`` over ``[lo, hi]`` and its absolute error on the log scale.

    Finds the mode, integrates ``exp(f − f_max)`` with the mode as a breakpoint.
    """
    res = optimize.minimize_scalar(
        lambda x: -float(f(x)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10 * max(1.0, abs(hi - lo))},
    )
    x_star = float(res.x)
    # A bounded search can settle on a local optimum at the edge of a narrow
    # interval; the caller's guess is a second candidate.
    if lo < mode_guess < hi and float(f(mode_guess)) > float(f(x_star)):
        x_star = mode_guess
    f_max = float(f(x_star))
    if not np.isfinite(f_max):
        return -np.inf, 0.0, x_star

    def g(x):
        v = float(f(x)) - f_max
        return math.exp(v) if np.isfinite(v) else 0.0

    points = [x_star] if lo < x_star < hi else None
    val, err = integrate.quad(
        g, lo, hi, points=points, epsabs=0.0, epsrel=epsrel, limit=500
    )
    if val <= 0.0:
        return -np.inf, 0.0, x_star
    return f_max + math.log(val), err / val, x_star


def quadrature_log_marginal_likelihood(
    model, *, epsrel: float = 1e-10, return_diagnostics: bool = False
):
    """Exact log marginal likelihood of a Gaussian spatial model by quadrature.

    Integrates β analytically and σ², ρ by adaptive quadrature under the
    model's own priors, so the result carries no Monte Carlo error and needs
    no posterior draws: the model does not have to be fit.

    Parameters
    ----------
    model
        An OLS, SLX, SAR, SDM, SEM or SDEM model with Gaussian errors
        (``robust=False``).
    epsrel : float, default 1e-10
        Relative tolerance of each one-dimensional quadrature.
    return_diagnostics : bool, default False
        Also return a dict with the quadrature error estimate and the mode of
        the spatial parameter's marginal posterior.

    Returns
    -------
    float or dict
        ``log p(y)``; with ``return_diagnostics=True``, a dict with keys
        ``logml``, ``abserr`` (estimated absolute error of ``logml``),
        ``mode`` and ``method`` (``"quadrature"``).
    """
    ev = _GaussianEvidence(model)

    # Scale for the σ² search: the OLS residual variance brackets the mode.
    s2_guess = max(ev.aa / ev.n, 1e-12)
    t_lo, t_hi = math.log(s2_guess) - 30.0, math.log(s2_guess) + 30.0

    inner_err = [0.0]
    t_guess = [math.log(s2_guess)]

    def log_inner(rho: float) -> float:
        f = ev.log_integrand_t(rho)
        val, err, t_star = _log_integrate(f, t_lo, t_hi, t_guess[0], epsrel)
        if np.isfinite(val):
            t_guess[0] = t_star
            inner_err[0] = max(inner_err[0], err)
        return val

    if ev.kind is None:
        logml, err = log_inner(0.0), inner_err[0]
        mode = None
    else:
        lo, hi = ev.lo, ev.hi
        # Integrating the Uniform(lo, hi) prior density 1/(hi − lo) over [lo, hi].
        outer, outer_err, mode = _log_integrate(log_inner, lo, hi, 0.0, epsrel)
        logml = outer - math.log(hi - lo)
        err = outer_err + inner_err[0]

    if return_diagnostics:
        return {
            "logml": float(logml),
            "abserr": float(err),
            "mode": mode,
            "method": "quadrature",
        }
    return float(logml)
