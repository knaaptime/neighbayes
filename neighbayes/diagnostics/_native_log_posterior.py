"""Native log-posterior for Gibbs-fitted Gaussian spatial models.

Bridge sampling needs one thing from a model: an unnormalized log-posterior it
can evaluate at arbitrary parameter vectors, plus the transform that maps stored
(constrained) draws into the unconstrained space that function lives in.
:func:`~.bayesfactor.compile_log_posterior` obtains both from PyMC's
``compile_logp``.

That is free for a NUTS fit, which already has the graph, but a Gibbs fit has
none — so the PyMC route builds and compiles an entire PyTensor graph purely as
a density oracle, costing roughly two seconds per model before bridge sampling
starts.  Every term in that graph is already available to the Gibbs path in
closed form: the priors come from ``_gaussian_priors``, the likelihood is a
Gaussian (or Student-t) quadratic form, and the Jacobian is the same
log-determinant surrogate the sampler itself used.  This module assembles them
directly.

The output must match PyMC's *absolutely*, not up to a constant: Bayes factors
compare marginal likelihoods across different models, so every normalizing
constant survives into the comparison.  ``test_native_log_posterior.py`` pins
the two against each other draw by draw, at machine precision for the Gaussian
likelihood.

Robust (Student-t) models agree to ~6e-10 per observation instead, and the
residual is not in the algebra: it is the difference between :func:`scipy.
special.gammaln` and PyTensor's own log-gamma at the same arguments, with
scipy's being the more accurate of the two.  It shows up as a constant offset,
so it shifts a log marginal likelihood by ``n * 6e-10`` — some eight orders of
magnitude below bridge sampling's own Monte-Carlo error.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.special import expit, gammaln, log_expit

__all__ = ["native_log_posterior"]


def _normal_logp(eps: np.ndarray, sigma2: float) -> float:
    """``sum(logpdf(eps; 0, sqrt(sigma2)))``, constants included."""
    n = eps.size
    return float(
        -0.5 * n * np.log(2.0 * np.pi)
        - 0.5 * n * np.log(sigma2)
        - 0.5 * float(eps @ eps) / sigma2
    )


def _studentt_logp(eps: np.ndarray, sigma2: float, nu: float) -> float:
    """``sum(logpdf(eps; nu, 0, sqrt(sigma2)))``, constants included."""
    n = eps.size
    sigma = np.sqrt(sigma2)
    return float(
        n
        * (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sigma)
        )
        - 0.5 * (nu + 1.0) * float(np.sum(np.log1p((eps / sigma) ** 2 / nu)))
    )


def native_log_posterior(
    model,
) -> Optional[tuple[Callable, list[str], dict, Callable]]:
    """Build the :func:`compile_log_posterior` 4-tuple without a PyMC graph.

    Parameters
    ----------
    model
        A fitted Gaussian spatial model (OLS, SLX, SAR, SDM, SEM, SDEM).

    Returns
    -------
    tuple or None
        ``(log_posterior_fn, param_names, param_info, to_unconstrained)`` with
        exactly the semantics of :func:`compile_log_posterior`, or ``None`` when
        *model* falls outside the supported family — callers should then fall
        back to the PyMC route rather than treat this as an error.
    """
    jacobian_param = getattr(model, "_jacobian_param", "__missing__")
    if jacobian_param not in (None, "rho", "lam"):
        return None
    for attr in ("_gaussian_priors", "_design_matrix", "_design_names", "_y"):
        if not hasattr(model, attr):
            return None

    y = np.asarray(model._y, dtype=np.float64)
    Z = np.asarray(model._design_matrix(), dtype=np.float64)
    priors = model._gaussian_priors(Z, model._design_names())

    beta_mu = np.asarray(priors["beta_mu"], dtype=np.float64) * np.ones(Z.shape[1])
    beta_sd = np.asarray(priors["beta_sigma"], dtype=np.float64) * np.ones(Z.shape[1])
    ig_alpha = float(priors["sigma2_alpha"])
    ig_beta = float(priors["sigma2_beta"])
    k = Z.shape[1]

    robust = bool(getattr(model, "robust", False))
    nu = float(model._nu) if robust else 0.0

    # --- spatial parameter, its Uniform prior, and the Jacobian potential ----
    if jacobian_param is None:
        spatial_name = None
        lo = hi = 0.0
        logdet_fn = None
        Wy = None
        WZ = None
    else:
        spatial_name = jacobian_param
        lo = float(priors[f"{jacobian_param}_lower"])
        hi = float(priors[f"{jacobian_param}_upper"])
        logdet_fn = model._logdet_numpy_fn
        Wy = np.asarray(model._Wy, dtype=np.float64)
        if jacobian_param == "lam":
            # eps = (y - λWy) - (Z - λWZ)β, matching _build_pymc_model_lam.
            cached = getattr(model, "_WZ_cache", None)
            if cached is None:
                cached = np.asarray(model._spatial_lag(Z), dtype=np.float64)
                model._WZ_cache = cached
            WZ = cached

    # PyMC's ``value_vars`` follow declaration order, and every branch of
    # ``_build_pymc_model`` declares the spatial parameter first, then beta,
    # then sigma2.  The flat layout below must agree with that ordering or the
    # bridge estimator would read the blocks transposed.
    names: list[str] = []
    shapes: dict[str, tuple] = {}
    sizes: dict[str, int] = {}
    if spatial_name is not None:
        names.append(f"{spatial_name}_interval__")
        shapes[names[-1]] = ()
        sizes[names[-1]] = 1
    names.append("beta")
    shapes["beta"] = (k,)
    sizes["beta"] = k
    names.append("sigma2_log__")
    shapes["sigma2_log__"] = ()
    sizes["sigma2_log__"] = 1

    def log_posterior(theta_flat: np.ndarray) -> float:
        theta_flat = np.asarray(theta_flat, dtype=np.float64).ravel()
        offset = 0
        total = 0.0

        if spatial_name is not None:
            u = float(theta_flat[0])
            offset = 1
            s = expit(u)
            val = lo + (hi - lo) * s
            # Uniform log-density (−log(hi−lo)) plus the interval transform's
            # log-Jacobian (log((hi−lo)·s·(1−s))); the width cancels exactly.
            total += float(log_expit(u) + log_expit(-u))
        else:
            val = 0.0

        beta = theta_flat[offset : offset + k]
        offset += k
        u_s2 = float(theta_flat[offset])
        sigma2 = np.exp(u_s2)

        # beta ~ Normal(beta_mu, beta_sd), identity transform.
        z = (beta - beta_mu) / beta_sd
        total += float(
            -0.5 * k * np.log(2.0 * np.pi)
            - np.sum(np.log(beta_sd))
            - 0.5 * float(z @ z)
        )

        # sigma2 ~ InverseGamma(alpha, beta) with a log transform.  The
        # transform's log-Jacobian is log(sigma2), which absorbs one power of
        # the -(alpha+1) exponent.
        total += float(
            ig_alpha * np.log(ig_beta)
            - gammaln(ig_alpha)
            - ig_alpha * u_s2
            - ig_beta / sigma2
        )

        # Likelihood, in the same algebraic form each PyMC branch uses.
        if jacobian_param is None:
            eps = y - Z @ beta
        elif jacobian_param == "rho":
            eps = y - (val * Wy + Z @ beta)
        else:
            eps = (y - val * Wy) - (Z - val * WZ) @ beta

        total += (
            _studentt_logp(eps, sigma2, nu) if robust else _normal_logp(eps, sigma2)
        )

        # pm.Potential("jacobian", logdet(·)) — the same surrogate, bounds and
        # method the sampler itself used, so the two paths cannot disagree
        # about the log-determinant.
        if logdet_fn is not None:
            total += float(logdet_fn(val))

        return float(total)

    def constrained_to_unconstrained(posterior) -> np.ndarray:
        n_total = posterior.sizes["chain"] * posterior.sizes["draw"]
        blocks = []
        if spatial_name is not None:
            arr = np.asarray(posterior[spatial_name].values).reshape(n_total, -1)
            blocks.append(np.log((arr - lo) / (hi - arr)))
        blocks.append(np.asarray(posterior["beta"].values).reshape(n_total, -1))
        s2 = np.asarray(posterior["sigma2"].values).reshape(n_total, -1)
        blocks.append(np.log(s2))
        return np.hstack(blocks)

    return (
        log_posterior,
        names,
        {"shapes": shapes, "sizes": sizes},
        constrained_to_unconstrained,
    )
