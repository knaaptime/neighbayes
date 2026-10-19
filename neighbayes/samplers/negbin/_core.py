"""Shared pieces of the Pólya–Gamma negative binomial samplers.

The reduced-form, panel, ZINB, hurdle and flow NB samplers draw the dispersion
α with :func:`_sample_alpha` (a slice on ``log α`` under the Half-Student-t
prior), carrying η and α in :class:`GibbsState`, and store the pointwise
log-likelihood with :func:`_nb_loglik_pointwise`.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...models.priors import ReducedGibbsPriors
from .._utils._slice import slice_sample_1d


@dataclass
class GibbsState:
    """Mutable state carried through one Gibbs sweep (Python-loop path).

    All arrays are numpy arrays; scalars are Python floats.  The reduced-form,
    panel, ZINB and flow samplers carry their η and α in it to
    :func:`_sample_alpha`.
    """

    eta: np.ndarray  # (n,) latent index
    beta: np.ndarray  # (k,) regression coefficients
    sigma2: float  # residual variance σ²
    rho: float  # spatial autoregressive parameter
    alpha: float  # NB dispersion parameter
    omega: np.ndarray  # (n,) PG auxiliary variables


def _sample_alpha(
    state: GibbsState,
    y: np.ndarray,
    priors: ReducedGibbsPriors,
    *,
    rng: np.random.Generator,
    cell_mask: np.ndarray | None = None,
    y_tab: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Block 6: Draw α | y, η — 1-D slice on log(α).

    Log-density on the log(α) scale:
        log p(log α | y, η) = log α + Σ_i log NB(y_i | exp(η_i), α) + log p(α)

    where log α is the Jacobian from the change of variables α = exp(log α),
    and p(α) is the Half-Student-t(ν_α, σ_α) prior — the same prior the
    PyMC/NUTS builders place via ``pm.HalfStudentT`` and the same kernel the
    reduced-form JAX sampler uses, so a model's posterior does not depend on
    ``backend``.

    Parameters
    ----------
    state : GibbsState
        Current state (uses alpha, eta).
    y : ndarray of shape (n,)
        Integer response vector.
    priors : ReducedGibbsPriors
        Prior hyperparameters (alpha_sigma, alpha_nu).
    rng : numpy.random.Generator
        Random state.
    cell_mask : ndarray of bool, shape (n,), optional
        Live cells; masked cells drop out of every term (the tabulated
        ``Σ gammaln(y+α)`` runs over live counts only).
    y_tab : (y_vals, y_counts), optional
        Pre-tabulated ``np.unique(y_live, return_counts=True)`` of the live
        cells, to hoist the per-call sort out of a Gibbs loop; must align
        with ``y`` after ``cell_mask`` is applied.

    Returns
    -------
    alpha_new : float
        New draw of α, or ``priors.alpha_fixed`` when the priors hold α fixed.
    """
    fixed = getattr(priors, "alpha_fixed", None)
    if fixed is not None:
        return float(fixed)
    alpha_sigma = priors.alpha_sigma
    alpha_nu = priors.alpha_nu
    eta = state.eta
    log_alpha = np.log(state.alpha)

    # α-independent pieces of Σ log NB(y | μ, α), computed once per draw:
    #   Σ log NB = Σ gammaln(y+α) − N·gammaln(α) + Σ y·η − Σ (y+α)·log(μ+α)
    #              + N·α·log α       (dropping −Σ log y!, constant in α)
    # Σ gammaln(y+α) runs over the distinct counts only, and log μ = η exactly,
    # so each evaluation is one log over the data instead of two gammaln and
    # three logs (the dominant cost of the α block on large flow data).
    from scipy.special import gammaln

    y_arr = np.asarray(y, dtype=np.float64)
    if cell_mask is not None:
        y_arr = y_arr[cell_mask]
        eta = eta[cell_mask]
    if y_tab is None:
        y_vals, y_counts = np.unique(y_arr, return_counts=True)
    else:
        y_vals, y_counts = y_tab
    n_obs = y_arr.size
    y_dot_eta = float(y_arr @ eta)
    mu = np.exp(eta)

    def log_density(log_a: float) -> float:
        """Log-density on the log(α) scale."""
        alpha = np.exp(log_a)
        if alpha <= 0:
            return -np.inf

        log_mu_a = np.log(mu + alpha)
        total_log_lik = (
            float(y_counts @ gammaln(y_vals + alpha))
            - n_obs * gammaln(alpha)
            + y_dot_eta
            - float(y_arr @ log_mu_a)
            - alpha * float(log_mu_a.sum())
            + n_obs * alpha * log_a
        )

        # Half-Student-t(nu, sigma) prior on alpha:
        #   p(alpha) ∝ (1 + alpha^2 / (nu * sigma^2))^{-(nu+1)/2}
        # Matches the PyMC/NUTS path (pm.HalfStudentT) and the reduced-form JAX
        # path.  A HalfNormal here would be a hard wall at alpha ≈ 4*sigma,
        # which prevents the NB from representing near-equidispersed
        # (Poisson-like) data at all.
        log_prior = (
            -0.5
            * (alpha_nu + 1.0)
            * np.log1p((alpha * alpha) / (alpha_nu * alpha_sigma * alpha_sigma))
        )

        # Jacobian: d(alpha)/d(log_alpha) = alpha, so log|J| = log(alpha) = log_a
        return log_a + total_log_lik + log_prior

    # Slice sample on log(alpha) with bounds
    log_alpha_new, _ = slice_sample_1d(
        log_density=log_density,
        x0=log_alpha,
        lower=-10.0,  # alpha > exp(-10) ≈ 4.5e-5
        upper=10.0,  # alpha < exp(10) ≈ 22026
        w=0.5,
        rng=rng,
    )

    return np.exp(log_alpha_new)


def _nb_loglik_pointwise(
    y: np.ndarray,
    eta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute pointwise NB log-likelihood.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Integer response.
    eta : ndarray of shape (n,)
        Latent log-mean.
    alpha : float
        NB dispersion parameter.

    Returns
    -------
    log_lik : ndarray of shape (n,)
        Pointwise log-likelihood values.

    Notes
    -----
    This is the **full NB2 log-pmf**, including the ``-log Γ(y+1)``
    normalizing term.  That term is draw-independent, so it does not
    affect the α slice sampler (whose internal densities deliberately
    omit it), but it is essential here: without it every stored NB
    log-likelihood is shifted by ``+log(y!)``, which corrupts pointwise
    LOO/WAIC and makes NB elpd incomparable with the Poisson site (which
    stores the full pmf).
    """
    from scipy.special import gammaln

    mu = np.exp(eta)
    return (
        gammaln(y + alpha)
        - gammaln(alpha)
        - gammaln(y + 1.0)
        + y * np.log(mu / (mu + alpha))
        + alpha * np.log(alpha / (mu + alpha))
    )
