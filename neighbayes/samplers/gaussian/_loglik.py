"""Pointwise log-likelihood for Gaussian spatial models.

Unlike the NB Gibbs sampler (where the Jacobian cancels in the
non-centred parameterization), Gaussian SAR/SEM/SDM/SDEM models
require the spatial Jacobian log|I - ρW| (or log|I - λW|)
as part of the complete log-likelihood.

The convention used throughout neighbayes: distribute the Jacobian
equally across observations:

    ℓᵢ = Gaussian_part + log|I - ρW| / n

This ensures Σᵢ ℓᵢ equals the total log-likelihood used for sampling.

NumPy-path logdet uses the model's existing ``logdet_fn`` callable.
JAX-path logdet uses a JAX-native callable (any method: eigenvalue,
Chebyshev, or trace polynomial) for JIT-compatible evaluation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# SAR / SDM pointwise log-likelihood (spatial lag on y)
# ---------------------------------------------------------------------------


def sar_pointwise_loglik_numpy(
    y: np.ndarray,
    X: np.ndarray,
    Wy: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    rho: float,
    logdet_fn: callable,
    n: int,
) -> np.ndarray:
    """Complete pointwise log-likelihood for SAR/SDM including Jacobian.

    NumPy version for the Python-loop Gibbs path.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma : float
        Residual standard deviation.
    rho : float
        Spatial autoregressive parameter.
    logdet_fn : callable
        log|I - rho*W| callable.
    n : int
        Number of observations.

    Returns
    -------
    ll : ndarray of shape (n,)
        Per-observation log-likelihood (including Jacobian/n).
    """
    mu = rho * Wy + X @ beta
    resid = y - mu
    ll = -0.5 * (resid / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    jacobian = logdet_fn(rho)
    ll += jacobian / n
    return ll


def sar_pointwise_loglik_jax(
    y,
    X,
    Wy,
    beta,
    sigma,
    rho,
    logdet_jax,
    n,
):
    """Complete pointwise log-likelihood for SAR/SDM including Jacobian.

    JAX version for the JAX-JIT Gibbs path. Uses a JAX-native logdet
    function (any method: eigenvalue, Chebyshev, or trace polynomial)
    for JIT-compatible log|I - ρW| evaluation.

    Parameters
    ----------
    y : jax.Array of shape (n,)
        Response vector.
    X : jax.Array of shape (n, k)
        Design matrix.
    Wy : jax.Array of shape (n,)
        W @ y (precomputed).
    beta : jax.Array of shape (k,)
        Regression coefficients.
    sigma : jax.Array (scalar)
        Residual standard deviation.
    rho : jax.Array (scalar)
        Spatial autoregressive parameter.
    logdet_jax : callable
        JAX-native log|I - rho*W| callable.
    n : int
        Number of observations.

    Returns
    -------
    ll : jax.Array of shape (n,)
        Per-observation log-likelihood (including Jacobian/n).
    """
    import jax.numpy as jnp

    mu = rho * Wy + X @ beta
    resid = y - mu
    ll = -0.5 * (resid / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
    jacobian = logdet_jax(rho)
    ll += jacobian / n
    return ll


# ---------------------------------------------------------------------------
# SEM / SDEM pointwise log-likelihood (spatial error)
# ---------------------------------------------------------------------------


def sem_pointwise_loglik_numpy(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    beta: np.ndarray,
    sigma: float,
    lam: float,
    logdet_fn: callable,
    n: int,
) -> np.ndarray:
    """Complete pointwise log-likelihood for SEM/SDEM including Jacobian.

    NumPy version for the Python-loop Gibbs path.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Sparse spatial weights matrix.
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma : float
        Residual standard deviation.
    lam : float
        Spatial error parameter.
    logdet_fn : callable
        log|I - lam*W| callable.
    n : int
        Number of observations.

    Returns
    -------
    ll : ndarray of shape (n,)
        Per-observation log-likelihood (including Jacobian/n).
    """
    resid = y - X @ beta
    eps = resid - lam * (W_sparse @ resid)
    ll = -0.5 * (eps / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    jacobian = logdet_fn(lam)
    ll += jacobian / n
    return ll


def sem_pointwise_loglik_jax(
    y,
    X,
    W_dense,
    beta,
    sigma,
    lam,
    logdet_jax,
    n,
):
    """Complete pointwise log-likelihood for SEM/SDEM including Jacobian.

    JAX version for the JAX-JIT Gibbs path. Uses a JAX-native logdet
    function (any method) for JIT-compatible log|I - λW| evaluation.

    Parameters
    ----------
    y : jax.Array of shape (n,)
        Response vector.
    X : jax.Array of shape (n, k)
        Design matrix.
    W_dense : jax.Array of shape (n, n)
        Dense spatial weights matrix (JAX needs dense for matmul).
    beta : jax.Array of shape (k,)
        Regression coefficients.
    sigma : jax.Array (scalar)
        Residual standard deviation.
    lam : jax.Array (scalar)
        Spatial error parameter.
    logdet_jax : callable
        JAX-native log|I - lam*W| callable.
    n : int
        Number of observations.

    Returns
    -------
    ll : jax.Array of shape (n,)
        Per-observation log-likelihood (including Jacobian/n).
    """
    import jax.numpy as jnp

    resid = y - X @ beta
    eps = resid - lam * (W_dense @ resid)
    ll = -0.5 * (eps / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
    jacobian = logdet_jax(lam)
    ll += jacobian / n
    return ll


# ---------------------------------------------------------------------------
# OLS / SLX pointwise log-likelihood (no Jacobian)
# ---------------------------------------------------------------------------


def ols_pointwise_loglik_numpy(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Pointwise log-likelihood for OLS/SLX (no Jacobian).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    beta : ndarray of shape (k,)
        Regression coefficients.
    sigma : float
        Residual standard deviation.

    Returns
    -------
    ll : ndarray of shape (n,)
        Per-observation log-likelihood.
    """
    resid = y - X @ beta
    return -0.5 * (resid / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Vectorized post-chain computation
# ---------------------------------------------------------------------------


def _eps_log_density(
    eps: np.ndarray, sigma: np.ndarray, nu: float | None = None
) -> np.ndarray:
    """Pointwise Normal(0, σ) or Student-t(ν, 0, σ) log-density of residuals."""
    if nu is None:
        return -0.5 * (eps / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
    from scipy.special import gammaln

    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * np.log(nu * np.pi)
        - np.log(sigma)
        - 0.5 * (nu + 1.0) * np.log1p((eps / sigma) ** 2 / nu)
    )


def sar_pointwise_loglik_vectorized(
    rho_draws: np.ndarray,
    beta_draws: np.ndarray,
    sigma_draws: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    Wy: np.ndarray,
    logdet_vec_fn: callable,
    n: int,
    nu: float | None = None,
) -> np.ndarray:
    """Vectorized pointwise LL for SAR/SDM over all posterior draws.

    Uses a vectorized numpy logdet function for batch evaluation over
    all rho draws at once.

    Parameters
    ----------
    rho_draws : ndarray of shape (n_keep,)
        Posterior draws of ρ.
    beta_draws : ndarray of shape (n_keep, k)
        Posterior draws of β.
    sigma_draws : ndarray of shape (n_keep,)
        Posterior draws of σ.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    Wy : ndarray of shape (n,)
        W @ y (precomputed).
    logdet_vec_fn : callable
        Vectorized logdet function from ``make_logdet_numpy_vec_fn()``.
        Accepts array of ρ values, returns array of logdet values.
    n : int
        Number of observations.
    nu : float or None, default None
        Student-t degrees of freedom for robust models; ``None`` for Normal.

    Returns
    -------
    ll : ndarray of shape (n_keep, n)
        Per-observation log-likelihood for each posterior draw.
    """
    mu = rho_draws[:, None] * Wy[None, :] + (beta_draws @ X.T)  # (n_keep, n)
    resid = y[None, :] - mu
    ll_gauss = _eps_log_density(resid, sigma_draws[:, None], nu)
    jacobian = logdet_vec_fn(rho_draws)  # (n_keep,)
    ll_total = ll_gauss + jacobian[:, None] / n  # (n_keep, n)
    return ll_total


def sem_pointwise_loglik_vectorized(
    lam_draws: np.ndarray,
    beta_draws: np.ndarray,
    sigma_draws: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    logdet_vec_fn: callable,
    n: int,
    nu: float | None = None,
) -> np.ndarray:
    """Vectorized pointwise LL for SEM/SDEM over all posterior draws.

    Parameters
    ----------
    lam_draws : ndarray of shape (n_keep,)
        Posterior draws of λ.
    beta_draws : ndarray of shape (n_keep, k)
        Posterior draws of β.
    sigma_draws : ndarray of shape (n_keep,)
        Posterior draws of σ.
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix of shape (n, n)
        Sparse spatial weights matrix.
    logdet_vec_fn : callable
        Vectorized logdet function.
    n : int
        Number of observations.
    nu : float or None, default None
        Student-t degrees of freedom for robust models; ``None`` for Normal.

    Returns
    -------
    ll : ndarray of shape (n_keep, n)
        Per-observation log-likelihood for each posterior draw.
    """
    # Compute residuals for each draw
    resid_all = y[None, :] - (beta_draws @ X.T)  # (n_keep, n)

    # Apply spatial filter: eps = (I - lam*W) @ resid
    # For each draw, eps_g = resid_g - lam_g * (W @ resid_g)
    W_resid = (W_sparse @ resid_all.T).T  # (n_keep, n)
    eps = resid_all - lam_draws[:, None] * W_resid  # (n_keep, n)

    ll_gauss = _eps_log_density(eps, sigma_draws[:, None], nu)
    jacobian = logdet_vec_fn(lam_draws)  # (n_keep,)
    ll_total = ll_gauss + jacobian[:, None] / n  # (n_keep, n)
    return ll_total
