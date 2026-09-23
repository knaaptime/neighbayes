r"""Native blackjax NUTS for Gaussian spatial models — a pure-JAX fast path.

Prototype of "spatial at the speed of linear": sample the **joint**
``(β, σ², ρ)`` posterior of a Gaussian SAR directly with gradient-based NUTS,
bypassing the PyMC graph.  Because ``Wy = W @ y`` is precomputed as data, the
SAR residual

    e(β, ρ) = y − ρ·Wy − Xβ

costs one extra vector op over the OLS residual ``y − Xβ``; the Jacobian term
``log|I − ρW|`` is the *only* spatial addition to the log-density, supplied as a
pluggable differentiable callable ``logdet_fn(ρ)`` (eigenvalue, chol-cheb
Clenshaw, or SLQ — all differentiable via the fixed :mod:`neighbayes._logdet`
jax closures).  Its ρ-gradient — the resolvent trace ``−tr(W(I−ρW)⁻¹)`` — flows
through ``jax.grad`` automatically, so no per-leapfrog solves are needed.

Backend-neutrality
------------------
:func:`make_sar_joint_logdensity` returns a plain ``logdensity_fn(theta)`` over
a flat unconstrained vector.  It takes ``logdet_fn`` as an argument and touches
only ``jax.numpy`` — so a numpyro ``potential_fn``, a nutpie custom-JAX driver,
or blackjax all consume it unchanged.  :func:`run_chain_blackjax_gaussian` is
one such driver (blackjax window adaptation + NUTS); it is not the only possible
one.

Unconstrained parametrisation
-----------------------------
NUTS runs on ``ℝ^d``; the model parameters are mapped there with explicit
log-Jacobians:

* ``β``      — identity (already unconstrained), ``k`` entries;
* ``σ²``     — ``σ² = exp(τ)``, ``log|J| = τ``;
* ``ρ``      — ``ρ = lower + (upper−lower)·sigmoid(u)`` (spatial only),
  ``log|J| = log(upper−lower) + logsigmoid(u) + logsigmoid(−u)``.

Set ``spatial=False`` to drop ``ρ`` and the logdet entirely — that yields the
OLS ``(β, σ²)`` joint sampler used as the linear-model throughput yardstick from
the same code path.

Priors match the Gibbs / PyMC-NUTS paths (Normal ``β``, InverseGamma ``σ²``,
Uniform ``ρ``) so the samplers target the same posterior and the shootout is
apples-to-apples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..._jax_dispatch import ensure_x64


def _check_blackjax_available() -> None:
    """Raise ImportError if JAX or blackjax is not installed."""
    import importlib.util

    for pkg in ("jax", "blackjax"):
        if importlib.util.find_spec(pkg) is None:
            raise ImportError(
                f"{pkg} is required for the native blackjax spatial-NUTS path. "
                f"Install with: pip install {pkg}"
            )


# ---------------------------------------------------------------------------
# Backend-neutral joint log-density
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointLogDensity:
    """A built joint log-density and the helpers to interpret its samples.

    Attributes
    ----------
    logdensity_fn : callable
        ``theta (d,) -> scalar`` unnormalized joint log-density on the
        unconstrained parameters.  Pure ``jax.numpy``; differentiable and
        JIT-compatible.
    to_constrained : callable
        ``theta (..., d) -> dict`` mapping unconstrained draws back to
        ``{"beta", "sigma2"[, "rho"]}`` in natural coordinates.
    init_position : np.ndarray, shape (d,)
        A reasonable unconstrained starting point.
    dim : int
        Dimension ``d`` of the unconstrained space.
    spatial : bool
        Whether ρ (and the logdet term) is included.
    """

    logdensity_fn: object
    to_constrained: object
    init_position: np.ndarray
    dim: int
    spatial: bool


def make_sar_joint_logdensity(
    y,
    X,
    Wy,
    priors,
    logdet_fn=None,
    *,
    spatial: bool = True,
    rho_lower: float = -1.0,
    rho_upper: float = 1.0,
) -> JointLogDensity:
    """Build the joint ``(β, σ²[, ρ])`` unconstrained log-density for SAR / OLS.

    Parameters
    ----------
    y : array, shape (n,)
    X : array, shape (n, k)
    Wy : array, shape (n,) or None
        ``W @ y``, precomputed.  Required when ``spatial=True``.
    priors : GaussianGibbsPriors-like
        Provides ``beta_mu``, ``beta_sigma``, ``sigma2_alpha``, ``sigma2_beta``
        (and, for spatial, ``rho_lower``/``rho_upper`` if not passed here).
    logdet_fn : callable or None
        Differentiable ``ρ -> log|I − ρW|``.  Required when ``spatial=True``.
    spatial : bool, default True
        Include ρ and the logdet term.  ``False`` → OLS ``(β, σ²)`` yardstick.
    rho_lower, rho_upper : float
        Bounds for the ρ transform (spatial only).

    Returns
    -------
    JointLogDensity
    """
    import jax
    import jax.numpy as jnp
    from jax.nn import log_sigmoid
    from jax.scipy.special import gammaln

    ensure_x64()

    y_j = jnp.asarray(np.asarray(y, dtype=np.float64))
    X_j = jnp.asarray(np.asarray(X, dtype=np.float64))
    n, k = X_j.shape

    if spatial:
        if Wy is None or logdet_fn is None:
            raise ValueError("spatial=True requires both Wy and logdet_fn.")
        Wy_j = jnp.asarray(np.asarray(Wy, dtype=np.float64))

    # Prior hyperparameters as JAX constants (broadcast β prior to length k).
    beta_mu = jnp.broadcast_to(
        jnp.asarray(np.asarray(priors.beta_mu, dtype=np.float64)), (k,)
    )
    beta_sigma = jnp.broadcast_to(
        jnp.asarray(np.asarray(priors.beta_sigma, dtype=np.float64)), (k,)
    )
    a0 = jnp.float64(priors.sigma2_alpha)
    b0 = jnp.float64(priors.sigma2_beta)
    lo = jnp.float64(rho_lower)
    hi = jnp.float64(rho_upper)
    log_width = jnp.log(hi - lo)

    half_log_2pi = 0.5 * jnp.log(2.0 * jnp.pi)

    def logdensity_fn(theta):
        beta = theta[:k]
        tau = theta[k]
        sigma2 = jnp.exp(tau)

        if spatial:
            u = theta[k + 1]
            s = jax.nn.sigmoid(u)
            rho = lo + (hi - lo) * s
            resid = y_j - rho * Wy_j - X_j @ beta
            jac_rho = log_width + log_sigmoid(u) + log_sigmoid(-u)
            logdet = logdet_fn(rho)
        else:
            resid = y_j - X_j @ beta
            jac_rho = 0.0
            logdet = 0.0

        ss = jnp.dot(resid, resid)

        # Gaussian log-likelihood (including the log|I-ρW| Jacobian for SAR).
        loglik = (
            logdet
            - n * half_log_2pi
            - 0.5 * n * tau  # -0.5 n log σ²
            - 0.5 * ss / sigma2
        )

        # Priors.
        log_prior_beta = jnp.sum(
            -half_log_2pi
            - jnp.log(beta_sigma)
            - 0.5 * ((beta - beta_mu) / beta_sigma) ** 2
        )
        # InverseGamma(a0, b0) on σ².
        log_prior_sigma2 = (
            a0 * jnp.log(b0) - gammaln(a0) - (a0 + 1.0) * tau - b0 / sigma2
        )
        # Uniform(lo, hi) on ρ (constant within bounds; enforced by transform).
        log_prior_rho = -log_width if spatial else 0.0

        # Jacobians of the unconstrained transforms.
        jac_sigma2 = tau

        return (
            loglik
            + log_prior_beta
            + log_prior_sigma2
            + log_prior_rho
            + jac_sigma2
            + jac_rho
        )

    def to_constrained(theta):
        theta = jnp.asarray(theta)
        beta = theta[..., :k]
        sigma2 = jnp.exp(theta[..., k])
        out = {"beta": np.asarray(beta), "sigma2": np.asarray(sigma2)}
        if spatial:
            s = jax.nn.sigmoid(theta[..., k + 1])
            out["rho"] = np.asarray(lo + (hi - lo) * s)
        return out

    # Initial position: OLS-ish β, τ = log Var(resid at β=0)? use log Var(y),
    # ρ at the interval midpoint (u = 0).
    dim = k + 1 + (1 if spatial else 0)
    init = np.zeros(dim, dtype=np.float64)
    init[k] = float(np.log(max(np.var(np.asarray(y, dtype=np.float64)), 1e-6)))
    # u=0 → ρ at midpoint already (init already zero).

    return JointLogDensity(
        logdensity_fn=logdensity_fn,
        to_constrained=to_constrained,
        init_position=init,
        dim=dim,
        spatial=spatial,
    )


# ---------------------------------------------------------------------------
# blackjax NUTS driver
# ---------------------------------------------------------------------------


@dataclass
class BlackjaxResult:
    """Samples + diagnostics from :func:`run_chain_blackjax_gaussian`.

    Arrays are shaped ``(chains, draws, ...)``.
    """

    beta: np.ndarray  # (chains, draws, k)
    sigma2: np.ndarray  # (chains, draws)
    rho: np.ndarray | None  # (chains, draws) or None (OLS)
    num_integration_steps: np.ndarray  # (chains, draws) leapfrog steps per draw
    acceptance_rate: np.ndarray  # (chains, draws)
    step_size: float  # adapted step size (mean over chains)
    num_divergent: int
    warmup_time: float
    sampling_time: float
    method: str


def run_chain_blackjax_gaussian(
    y,
    X,
    W_sparse,
    priors,
    *,
    spatial: bool = True,
    logdet_method: str | None = None,
    eigs: np.ndarray | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 1,
    seed: int = 0,
    target_accept: float = 0.8,
    rho_lower: float | None = None,
    rho_upper: float | None = None,
) -> BlackjaxResult:
    """Sample a Gaussian SAR (or OLS, ``spatial=False``) with native blackjax NUTS.

    Builds ``Wy`` and a differentiable ``logdet_fn`` (via
    :func:`neighbayes._logdet.make_logdet_jax_fn`), assembles the joint
    unconstrained log-density, runs blackjax window adaptation for ``tune`` steps
    then NUTS for ``draws``.  Returns constrained draws and per-draw NUTS
    diagnostics.

    ``logdet_method`` selects the logdet surrogate ("eigenvalue", "cheb_cholesky",
    "slq", ...); ``eigs`` (if given) is passed through so the eigenvalue method
    skips its own eigendecomposition.  ``rho_lower/rho_upper`` default to the
    priors' bounds.
    """
    _check_blackjax_available()
    import blackjax
    import jax
    import jax.numpy as jnp

    ensure_x64()

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if rho_lower is None:
        rho_lower = float(getattr(priors, "rho_lower", -1.0))
    if rho_upper is None:
        rho_upper = float(getattr(priors, "rho_upper", 1.0))

    logdet_fn = None
    Wy = None
    if spatial:
        W_csr = sp.csr_matrix(W_sparse)
        Wy = W_csr @ y
        from ..._logdet import make_logdet_jax_fn

        # Default: hand W to make_logdet_jax_fn.  Optionally pass precomputed
        # eigenvalues to skip a redundant eigendecomposition for the reference
        # arm — but only *real* spectra, since the 1-D input path casts to
        # float64 (a genuinely complex spectrum is left as W so it stays
        # complex128 inside make_logdet_jax_fn).
        W_input = W_csr
        if logdet_method == "eigenvalue" and eigs is not None:
            eigs_arr = np.asarray(eigs)
            if not (np.iscomplexobj(eigs_arr) and np.abs(eigs_arr.imag).max() > 1e-9):
                W_input = np.real(eigs_arr).astype(np.float64)
        logdet_fn = make_logdet_jax_fn(
            W_input,
            method=logdet_method,
            rho_min=rho_lower,
            rho_max=rho_upper,
        )

    jld = make_sar_joint_logdensity(
        y,
        X,
        Wy,
        priors,
        logdet_fn,
        spatial=spatial,
        rho_lower=rho_lower,
        rho_upper=rho_upper,
    )
    logdensity_fn = jld.logdensity_fn

    def _one_step(kernel):
        @jax.jit
        def step(state, key):
            state, info = kernel(key, state)
            return state, (state.position, info)

        return step

    beta_chains, sigma2_chains, rho_chains = [], [], []
    nsteps_chains, accept_chains = [], []
    step_sizes = []
    n_divergent = 0
    warmup_time = 0.0
    sampling_time = 0.0

    key = jax.random.PRNGKey(seed)
    for c in range(chains):
        key, wkey, skey = jax.random.split(key, 3)
        init_pos = jnp.asarray(jld.init_position)

        # Window adaptation → (step_size, inverse_mass_matrix).
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, target_acceptance_rate=target_accept
        )
        t0 = time.perf_counter()
        (last_state, parameters), _ = warmup.run(wkey, init_pos, num_steps=tune)
        # Block on the adapted step size so the warmup timing is real.
        step_sizes.append(float(jax.block_until_ready(parameters["step_size"])))
        warmup_time += time.perf_counter() - t0

        kernel = blackjax.nuts(logdensity_fn, **parameters).step
        step = _one_step(kernel)

        keys = jax.random.split(skey, draws)
        t0 = time.perf_counter()
        _, (positions, infos) = jax.lax.scan(step, last_state, keys)
        positions = jax.block_until_ready(positions)
        sampling_time += time.perf_counter() - t0

        constrained = jld.to_constrained(positions)
        beta_chains.append(constrained["beta"])
        sigma2_chains.append(constrained["sigma2"])
        if spatial:
            rho_chains.append(constrained["rho"])
        nsteps_chains.append(np.asarray(infos.num_integration_steps))
        accept_chains.append(np.asarray(infos.acceptance_rate))
        n_divergent += int(np.asarray(infos.is_divergent).sum())

    return BlackjaxResult(
        beta=np.stack(beta_chains),
        sigma2=np.stack(sigma2_chains),
        rho=np.stack(rho_chains) if spatial else None,
        num_integration_steps=np.stack(nsteps_chains),
        acceptance_rate=np.stack(accept_chains),
        step_size=float(np.mean(step_sizes)),
        num_divergent=n_divergent,
        warmup_time=warmup_time,
        sampling_time=sampling_time,
        method=(logdet_method or "auto") if spatial else "ols",
    )
