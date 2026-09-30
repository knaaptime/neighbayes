r"""JAX-accelerated full-JIT Gibbs sampler for Gaussian spatial models.

Composes one partially collapsed Gibbs sweep (σ², ρ/λ, β) into a single
``@eqx.filter_jit``-compiled function, eliminating Python→JAX
dispatch overhead entirely.

Architecture
------------
Each sweep draws, in this order:

- **σ²** from its conjugate inverse-gamma conditional given β and ρ/λ.
- **ρ/λ** by Neal (2003) stepping-out slice sampling on
  :math:`p(\rho \mid \sigma^2, y)`, with β integrated out under its Normal
  prior, using a JAX-native ``log|I-ρW|``.
- **β** from its conjugate normal conditional given the new ρ/λ and σ².

β is redrawn straight after the ρ/λ draw that marginalized it (van Dyk &
Park, 2008), so each stored triple is a draw from the joint posterior.

With :math:`P = X^{*\top}X^*/\sigma^2 + \Lambda_0^{-1}` and
:math:`b = X^{*\top}y^*/\sigma^2 + \Lambda_0^{-1}\mu_0`, the spatial density is

.. math::

    \log p(\rho \mid \sigma^2, y) = \log|I - \rho W|
    - \frac{y^{*\top} y^*}{2\sigma^2}
    - \tfrac12 \log|P| + \tfrac12 b^\top P^{-1} b + \text{const}.

For SAR/SDM, :math:`y^* = y - \rho W y` and :math:`X^* = X`, so :math:`P`
is fixed within a sweep and the density is the log-determinant plus a
quadratic in ρ.  For SEM/SDEM, :math:`y^* = (I - \lambda W)y` and
:math:`X^* = (I - \lambda W)X`, so each evaluation is a k×k Cholesky.

Slice sampling
~~~~~~~~~~~~~~
The ρ/λ update uses Neal's (2003) stepping-out slice sampler on the
spatial log-density, carrying a persistent interval in the sampler
state across sweeps for better ESS per sample.  No gradient, proposal,
or Metropolis correction is required — every slice step accepts.

Limitations
-----------
- O(k³) dense Cholesky limits scalability to k ≤ ~2000.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3),
705–767.

van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers.
*Journal of the American Statistical Association*, 103(482), 790–796.

LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial
Econometrics*. CRC Press.
"""

from __future__ import annotations

import numpy as np

from .._utils._jax_utils import (
    check_jax_available as _check_jax_available_impl,
)


def _check_jax_available() -> None:
    """Raise ImportError if JAX or equinox is not installed."""
    _check_jax_available_impl(require_equinox=True)


# ---------------------------------------------------------------------------
# JAX Gaussian Gibbs state (equinox Module)
# ---------------------------------------------------------------------------

from ..._jax_dispatch import ensure_x64
from .._utils._jax_base import make_jax_state_class

JAXGaussianGibbsState = make_jax_state_class(
    "JAXGaussianGibbsState",
    (
        "beta",
        "sigma2",
        "rho",
        "slice_w",
        "slice_L",
        "slice_R",
        # The Jacobian interpolant and the ρ support travel in the state rather
        # than in the step's closure so a warmup-adaptive refit can substitute
        # them between the two scan phases without recompiling — the same
        # mechanism the slice width already uses.  ``logdet_params`` is an
        # opaque, method-specific pytree of fixed-shape arrays (see
        # :meth:`neighbayes._logdet._refit.LogdetRefitter.jax_params`); it is
        # ``None`` when the run uses a fixed interpolant, in which case the
        # step falls back to its closed-over evaluator.
        "logdet_params",
        "rho_lo",
        "rho_hi",
        # Student-t mixing variances vᵢ; all ones (and unused) for Gaussian
        # errors, so every model shares one state layout.
        "v",
    ),
)


# ---------------------------------------------------------------------------
# Precomputed ρ-independent inner products
# ---------------------------------------------------------------------------


def _precompute_gibbs_constants(y, X, Wy, W_sparse, is_sar: bool):
    """Compute all ρ-independent inner products as float64 JAX arrays.

    These are bound into the JIT closure so each density evaluation
    avoids touching any n-sized array.  For SAR/SDM, only the y-side
    quantities are needed (X_eff = X).  For SEM/SDEM, the WX-side
    quantities (cross-products with WX) are also required.

    Parameters
    ----------
    y : ndarray of shape (n,)
    X : ndarray of shape (n, k)
    Wy : ndarray of shape (n,)
        ``W @ y`` (precomputed by caller).
    W_sparse : scipy.sparse matrix or None
        Sparse W; used to form ``WX = W @ X`` for SEM/SDEM only.
    is_sar : bool
        If True, return zero placeholders for the WX-side quantities
        (so the JIT step has a static signature).

    Returns
    -------
    dict[str, jax.numpy.ndarray]
        Keys: ``Wy_jax, WX_jax, yty, yTWy, WyTWy, XTy, XTWy, WXTy,
        WXTWy, XtWX, WXtWX``.
    """
    import jax.numpy as jnp

    n, k = X.shape
    y64 = np.asarray(y, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)
    Wy64 = np.asarray(Wy, dtype=np.float64)

    yty = float(y64 @ y64)
    yTWy = float(y64 @ Wy64)
    WyTWy = float(Wy64 @ Wy64)
    XTy = X64.T @ y64
    XTWy = X64.T @ Wy64

    if is_sar:
        WX64 = np.zeros((n, k), dtype=np.float64)
        WXTy = np.zeros(k, dtype=np.float64)
        WXTWy = np.zeros(k, dtype=np.float64)
        XtWX = np.zeros((k, k), dtype=np.float64)
        WXtWX = np.zeros((k, k), dtype=np.float64)
    else:
        WX64 = np.asarray(W_sparse @ X64, dtype=np.float64)
        WXTy = WX64.T @ y64
        WXTWy = WX64.T @ Wy64
        XtWX = X64.T @ WX64
        WXtWX = WX64.T @ WX64

    return {
        "Wy_jax": jnp.asarray(Wy64, dtype=jnp.float64),
        "WX_jax": jnp.asarray(WX64, dtype=jnp.float64),
        "yty": jnp.float64(yty),
        "yTWy": jnp.float64(yTWy),
        "WyTWy": jnp.float64(WyTWy),
        "XTy": jnp.asarray(XTy, dtype=jnp.float64),
        "XTWy": jnp.asarray(XTWy, dtype=jnp.float64),
        "WXTy": jnp.asarray(WXTy, dtype=jnp.float64),
        "WXTWy": jnp.asarray(WXTWy, dtype=jnp.float64),
        "XtWX": jnp.asarray(XtWX, dtype=jnp.float64),
        "WXtWX": jnp.asarray(WXtWX, dtype=jnp.float64),
    }


# ---------------------------------------------------------------------------
# JIT-compiled Gibbs step builder
# ---------------------------------------------------------------------------


def _make_gaussian_gibbs_step(
    y_jax,
    X_jax,
    Wy_jax,
    WX_jax,
    n,
    k,
    logdet_jax,
    XtX_jax,
    # Precomputed ρ-independent inner products (closure constants)
    yty,
    yTWy,
    WyTWy,
    XTy,
    XTWy,
    WXTy,
    WXTWy,
    XtWX,
    WXtWX,
    priors,
    model_type: str,
    logdet_param_fn=None,
    nu: float | None = None,
):
    """Build a JIT-compiled Gaussian Gibbs step with data bound into the closure.

    Creates a ``@eqx.filter_jit``-compiled function that performs one
    complete 3-block Gibbs sweep (β, σ², ρ/λ) in a single XLA kernel
    call, eliminating all Python→JAX dispatch overhead.

    The ρ/λ update is Neal (2003) slice sampling with persistent-interval
    reuse: it explores the full conditional thoroughly and every step
    accepts (no Metropolis correction).

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Response vector (JAX array).
    X_jax : jax.numpy.ndarray of shape (n, k)
        Design matrix (JAX array).
    Wy_jax : jax.numpy.ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).  None for SEM/SDEM.
    W_dense_jax : jax.numpy.ndarray of shape (n, n) or None
        Dense W matrix (for SEM/SDEM residual filtering).  None for
        SAR/SDM.
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".
    nu : float or None, default None
        Student-t degrees of freedom for robust errors.  When set, each sweep
        first draws the scale-mixture variances ``v`` and weights every
        cross-product by ``1/v``; ``None`` gives Gaussian errors.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key) -> (new_state, accept)

        where ``state`` is a ``JAXGaussianGibbsState`` and ``key`` is a
        JAX PRNG key.  ``accept`` is always ``True`` (slice sampling has
        no rejection step); it is retained so callers can accumulate a
        (trivially unit) acceptance rate.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    ensure_x64()

    is_sar = model_type in ("sar", "sdm")

    # Convert constants to JAX arrays
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    sigma2_alpha_jax = jnp.float64(priors.sigma2_alpha)
    sigma2_beta_jax = jnp.float64(priors.sigma2_beta)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    @eqx.filter_jit
    def gibbs_step(state, key):
        """One partially collapsed Gibbs sweep: σ² → ρ/λ (slice) → β.

        Parameters
        ----------
        state : JAXGaussianGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.

        Returns
        -------
        new_state : JAXGaussianGibbsState
            Updated state.
        accept : bool
            Always ``True`` (slice sampling has no rejection step).
        """
        rho = state.rho  # holds λ for SEM/SDEM

        # The interpolant and the ρ support are read from the state, so a
        # warmup refit can swap them between scan phases without a retrace.
        # With no parameterized evaluator the state carries the prior bounds
        # and ``logdet_params is None``, and this reduces to the closure form.
        rho_lo = state.rho_lo
        rho_hi = state.rho_hi
        if logdet_param_fn is None:
            _logdet_of = logdet_jax
        else:

            def _logdet_of(param_val):
                return logdet_param_fn(param_val, state.logdet_params)

        # Partially collapsed sweep: [v | β, σ², ρ →] σ² | β, ρ →
        # ρ | σ² (β integrated out under its Normal prior) → β | ρ, σ².  β must
        # be redrawn straight after the ρ draw that marginalized it, so the
        # stored triple is a draw from the joint posterior.
        if is_sar:
            resid = y_jax - rho * Wy_jax - X_jax @ state.beta
        else:
            # SEM filtered residual: (I-λW)(y - Xβ) = (y-λWy) - (X-λWX)β
            resid = (y_jax - rho * Wy_jax) - (X_jax - rho * WX_jax) @ state.beta

        if nu is None:
            key_beta, key_sigma2, key_rho = jax.random.split(key, 3)
            v_new = state.v
            ss = resid @ resid
            XtX_m, yty_m, yTWy_m, WyTWy_m = XtX_jax, yty, yTWy, WyTWy
            XTy_m, XTWy_m = XTy, XTWy
            XtWX_m, WXtWX_m, WXTy_m, WXTWy_m = XtWX, WXtWX, WXTy, WXTWy
        else:
            key_beta, key_sigma2, key_rho, key_v = jax.random.split(key, 4)
            # vᵢ | · ~ InvGamma((ν+1)/2, (ν + εᵢ²/σ²)/2)
            g = jax.random.gamma(key_v, 0.5 * (nu + 1.0), shape=(n,))
            v_new = 0.5 * (nu + resid * resid / state.sigma2) / g
            w = 1.0 / v_new
            ss = w @ (resid * resid)
            # Every cross-product weighted by Ω = diag(1/v): O(n·k²) per sweep.
            Xw = X_jax * w[:, None]
            wy, wWy = w * y_jax, w * Wy_jax
            XtX_m = Xw.T @ X_jax
            yty_m, yTWy_m, WyTWy_m = y_jax @ wy, Wy_jax @ wy, Wy_jax @ wWy
            XTy_m, XTWy_m = Xw.T @ y_jax, Xw.T @ Wy_jax
            if is_sar:
                XtWX_m, WXtWX_m, WXTy_m, WXTWy_m = XtWX, WXtWX, WXTy, WXTWy
            else:
                WXw = WX_jax * w[:, None]
                XtWX_m = Xw.T @ WX_jax
                WXtWX_m = WXw.T @ WX_jax
                WXTy_m = WXw.T @ y_jax
                WXTWy_m = WXw.T @ Wy_jax

        # ── Block 1: σ² | β, ρ/λ[, v], y — conjugate InverseGamma draw ──
        # Prior σ² ~ InverseGamma(α, β), matching the NUTS path.
        a_post = sigma2_alpha_jax + jnp.float64(n / 2.0)
        b_post = sigma2_beta_jax + 0.5 * ss
        sigma2_inv = jax.random.gamma(key_sigma2, a_post) / b_post
        sigma2_new = jnp.maximum(1.0 / sigma2_inv, 1e-10)

        def _in_support(param_val):
            return jnp.where(
                (param_val >= rho_lo) & (param_val <= rho_hi), 0.0, -jnp.inf
            )

        # ── Block 2: ρ/λ | σ², y — slice sampling, β integrated out ──
        if is_sar:
            # P = XᵀX/σ² + Λ₀⁻¹ does not depend on ρ: factor once per sweep and
            # reduce b(ρ)ᵀP⁻¹b(ρ), b(ρ) = b0 − ρ b1, to a quadratic in ρ.
            P = XtX_m / sigma2_new + beta_prior_prec
            L_P = jnp.linalg.cholesky(P)
            b0 = XTy_m / sigma2_new + beta_mu_jax / beta_sigma2_jax
            b1 = XTWy_m / sigma2_new
            m0 = jax.scipy.linalg.cho_solve((L_P, True), b0)
            m1 = jax.scipy.linalg.cho_solve((L_P, True), b1)
            c00, c01, c11 = b0 @ m0, b0 @ m1, b1 @ m1

            def log_density_spatial(param_val):
                r_dot_r = (
                    yty_m - 2.0 * param_val * yTWy_m + (param_val * param_val) * WyTWy_m
                )
                quad = c00 - 2.0 * param_val * c01 + (param_val * param_val) * c11
                return (
                    _logdet_of(param_val)
                    - 0.5 * r_dot_r / sigma2_new
                    + 0.5 * quad
                    + _in_support(param_val)
                )

        else:

            def _sem_terms(param_val):
                p2 = param_val * param_val
                XtX_star = XtX_m - param_val * (XtWX_m + XtWX_m.T) + p2 * WXtWX_m
                Xty_star = XTy_m - param_val * (XTWy_m + WXTy_m) + p2 * WXTWy_m
                yty_star = yty_m - 2.0 * param_val * yTWy_m + p2 * WyTWy_m
                L_P = jnp.linalg.cholesky(XtX_star / sigma2_new + beta_prior_prec)
                b = Xty_star / sigma2_new + beta_mu_jax / beta_sigma2_jax
                return L_P, b, yty_star

            def log_density_spatial(param_val):
                L_P, b, yty_star = _sem_terms(param_val)
                quad = b @ jax.scipy.linalg.cho_solve((L_P, True), b)
                logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_P)))
                return (
                    _logdet_of(param_val)
                    - 0.5 * yty_star / sigma2_new
                    - 0.5 * logdet_P
                    + 0.5 * quad
                    + _in_support(param_val)
                )

        # Slice sampling: uses persistent interval for better ESS.
        from .._utils._jax_slice import jax_slice_sample_1d_adaptive

        # Always pass persistent interval (JAX arrays, never None).
        # jax_slice_sample_1d_adaptive checks whether x0 lies inside
        # [L_prev, R_prev] to decide whether to attempt reuse.
        rho_new, _, L_final, R_final, _ = jax_slice_sample_1d_adaptive(
            log_density_spatial,
            rho,
            rho_lo,
            rho_hi,
            key=key_rho,
            w=state.slice_w,
            L_prev=state.slice_L,
            R_prev=state.slice_R,
        )

        # ── Block 3: β | ρ, σ², y — conjugate normal, reusing P's factor ──
        if is_sar:
            m_beta = m0 - rho_new * m1
        else:
            L_P, b, _ = _sem_terms(rho_new)
            m_beta = jax.scipy.linalg.cho_solve((L_P, True), b)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + jax.scipy.linalg.solve_triangular(
            L_P.T, z_beta, lower=False
        )

        # Slice sampling always "accepts" (no MH step)
        accept = jnp.bool_(True)

        new_state = JAXGaussianGibbsState(
            beta=beta_new,
            sigma2=sigma2_new,
            rho=rho_new,
            logdet_params=state.logdet_params,  # swapped in Python between phases
            rho_lo=rho_lo,
            rho_hi=rho_hi,
            slice_w=state.slice_w,  # width adapted in Python between phases
            slice_L=L_final,
            slice_R=R_final,
            v=v_new,
        )
        return new_state, accept

    return gibbs_step


# ---------------------------------------------------------------------------
# jax.lax.scan-based chain runner
# ---------------------------------------------------------------------------


def _run_chain_jax_gibbs_scanned(
    gibbs_step,
    init_state,
    key,
    n_iters,
):
    """Run a single chain using ``jax.lax.scan``.

    All outputs are JAX arrays — no Python side effects.  This
    function is compatible with ``jax.vmap`` for vectorized
    multi-chain execution.

    Parameters
    ----------
    gibbs_step : callable
        JIT-compiled step function with signature
        ``(state, key) -> (new_state, accept)``.
    init_state : JAXGaussianGibbsState
        Initial state.
    key : jax.random.PRNGKey
        JAX random key.
    n_iters : int
        Number of iterations to run.

    Returns
    -------
    final_state : JAXGaussianGibbsState
        State after ``n_iters`` steps.
    final_key : jax.random.PRNGKey
        PRNG key after ``n_iters`` splits, so chunked runs can resume
        the chain deterministically.
    rhos : jax.numpy.ndarray of shape (n_iters,)
        Trace of ρ/λ values.
    betas : jax.numpy.ndarray of shape (n_iters, k)
        Trace of β values.
    sigma2s : jax.numpy.ndarray of shape (n_iters,)
        Trace of σ² values.
    accept_rate : jax.numpy.float64
        Fraction of slice steps accepted (always ``1.0``).
    """
    import jax
    import jax.numpy as jnp

    def scan_body(carry, _):
        state, key, accept_sum = carry
        key, step_key = jax.random.split(key)
        state, accept = gibbs_step(state, step_key)
        return (
            (state, key, accept_sum + accept),
            (state.rho, state.beta, state.sigma2, accept),
        )

    (final_state, final_key, total_accept), (rhos, betas, sigma2s, accepts) = (
        jax.lax.scan(
            scan_body,
            (init_state, key, jnp.float64(0.0)),
            None,
            length=n_iters,
        )
    )

    accept_rate = total_accept / jnp.float64(n_iters)
    return final_state, final_key, rhos, betas, sigma2s, accept_rate


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def run_chain_jax_gaussian(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    Wy: np.ndarray | None,
    logdet_jax,
    logdet_vec_fn,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    rng=None,
    model_type: str = "sar",
    slice_width: float | None = None,
    progressbar: bool = True,
    chain_id: int = 0,
    progress_manager: object | None = None,
    logdet_param_fn=None,
    logdet_params=None,
    nu: float | None = None,
):
    """Run one chain of the full-JIT JAX Gaussian Gibbs sampler.

    Creates a JIT-compiled Gibbs step function and runs it in a Python
    loop, handling warmup, thinning, step-size/width adaptation, and
    storage.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).  None for SEM/SDEM.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    logdet_vec_fn : callable
        Vectorized numpy logdet callable for post-chain LL computation.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    init : GaussianGibbsState
        Initial state (from OLS warm start).
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int, default 1
        Keep every thin-th draw.
    rng : numpy.random.Generator, optional
        Random state.
    model_type : str, default "sar"
        One of "sar", "sem", "sdm", "sdem".
    slice_width : float or None, default None
        Initial step-out width for slice sampling.  If None, defaults
        to ``(rho_upper - rho_lower) * 0.1`` (10% of the support).
    progressbar : bool, default True
        Show progress bar for this chain.
    chain_id : int, default 0
        Chain index (0-based) for progress bar display.
    progress_manager : object or None
        ``GibbsProgressBarManager`` instance. If provided,
        ``update()`` is called after each iteration.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho`` (or ``lam``), ``beta``,
        ``sigma``, and ``log_lik``.  Each array has shape
        ``(n_keep, ...)`` where n_keep = draws // thin.
        Also includes ``mh_accept_rate`` (float).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from ._loglik import (
        sar_pointwise_loglik_vectorized,
        sem_pointwise_loglik_vectorized,
    )

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws
    is_sar = model_type in ("sar", "sdm")

    # Default slice width: 10% of the support range
    rho_range = priors.rho_upper - priors.rho_lower
    if slice_width is None:
        slice_width = rho_range * 0.1

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)

    if Wy is None:
        Wy_np = np.asarray(W_sparse @ np.asarray(y, dtype=np.float64))
    else:
        Wy_np = np.asarray(Wy, dtype=np.float64)
    _consts = _precompute_gibbs_constants(
        y=y, X=X, Wy=Wy_np, W_sparse=W_sparse, is_sar=is_sar
    )

    # Initialize JAX state

    state = JAXGaussianGibbsState(
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        sigma2=jnp.float64(init.sigma2),
        rho=jnp.float64(init.rho),
        logdet_params=logdet_params,
        rho_lo=jnp.float64(priors.rho_lower),
        rho_hi=jnp.float64(priors.rho_upper),
        slice_w=jnp.float64(slice_width),
        # Initialize persistent interval to support bounds (no prior info)
        slice_L=jnp.float64(priors.rho_lower),
        slice_R=jnp.float64(priors.rho_upper),
        v=jnp.ones(n, dtype=jnp.float64),
    )

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=jnp.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)

    # ── Slice-width adaptation ──
    adapted_slice_width = slice_width

    # Build the JIT-compiled step function
    gibbs_step = _make_gaussian_gibbs_step(
        y_jax=y_jax,
        X_jax=X_jax,
        Wy_jax=_consts["Wy_jax"],
        WX_jax=_consts["WX_jax"],
        n=n,
        k=k,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        yty=_consts["yty"],
        yTWy=_consts["yTWy"],
        WyTWy=_consts["WyTWy"],
        XTy=_consts["XTy"],
        XTWy=_consts["XTWy"],
        WXTy=_consts["WXTy"],
        WXTWy=_consts["WXTWy"],
        XtWX=_consts["XtWX"],
        WXtWX=_consts["WXtWX"],
        priors=priors,
        model_type=model_type,
        logdet_param_fn=logdet_param_fn,
        nu=nu,
    )

    key = jax.random.PRNGKey(rng.integers(2**31))

    # ── Phase 1: Warmup via jax.lax.scan ──
    if tune > 0:
        key, warmup_key = jax.random.split(key)
        state, _, _, _, _, _ = _run_chain_jax_gibbs_scanned(
            gibbs_step,
            state,
            warmup_key,
            tune,
        )

        # Adapt the slice width in Python (no recompilation needed).
        # Slice sampling always "accepts", so we simply shrink the width
        # mildly after warmup for efficiency.
        adapted_slice_width = slice_width * 0.95  # mild shrink
        state = eqx.tree_at(
            lambda s: s.slice_w, state, jnp.float64(adapted_slice_width)
        )

        # Update progress bar for warmup phase
        if progress_manager is not None:
            for i in range(tune):
                progress_manager.update(chain_id, i, tuning=True)
            progress_manager.refresh()

    # ── Phase 2: Post-warmup draws via jax.lax.scan ──
    key, draw_key = jax.random.split(key)
    state, _, rhos, betas, sigma2s, draw_accept_rate = _run_chain_jax_gibbs_scanned(
        gibbs_step,
        state,
        draw_key,
        draws,
    )

    # Apply thinning and convert to NumPy
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    rho_samples = np.asarray(rhos[thin_slice])
    beta_samples = np.asarray(betas[thin_slice])
    sigma_samples = np.sqrt(np.asarray(sigma2s[thin_slice]))

    # Update progress bar for draw phase
    if progress_manager is not None:
        for i in range(tune, total_iters):
            progress_manager.update(chain_id, i, tuning=False)
        progress_manager.refresh()

    # ── Post-chain: vectorized pointwise log-likelihood ──
    # Compute after the chain completes using stored posterior draws,
    # avoiding per-draw logdet calls inside the JIT step.
    if is_sar:
        log_lik = sar_pointwise_loglik_vectorized(
            rho_draws=rho_samples,
            beta_draws=beta_samples,
            sigma_draws=sigma_samples,
            y=y,
            X=X,
            Wy=Wy,
            logdet_vec_fn=logdet_vec_fn,
            n=n,
            nu=nu,
        )
    else:
        log_lik = sem_pointwise_loglik_vectorized(
            lam_draws=rho_samples,
            beta_draws=beta_samples,
            sigma_draws=sigma_samples,
            y=y,
            X=X,
            W_sparse=W_sparse,
            logdet_vec_fn=logdet_vec_fn,
            n=n,
            nu=nu,
        )

    # Name the spatial parameter appropriately
    param_name = "rho" if is_sar else "lam"
    result = {
        param_name: rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "log_lik": log_lik,
        "mh_accept_rate": float(draw_accept_rate),
    }

    return result


# ---------------------------------------------------------------------------
# Vectorized multi-chain runner (jax.vmap)
# ---------------------------------------------------------------------------


def run_chains_jax_gibbs_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    Wy: np.ndarray | None,
    logdet_jax,
    logdet_vec_fn,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    model_type: str = "sar",
    slice_width: float | None = None,
    progressbar: bool = True,
    logdet_param_fn=None,
    logdet_params=None,
    refit_hook=None,
    log_likelihood: bool = True,
    nu: float | None = None,
) -> list[dict]:
    """Run multiple JAX Gibbs chains via ``jax.vmap``.

    All chains run in parallel on a single device via vectorized
    map.  This avoids Python multiprocessing entirely and is the
    recommended way to run multiple JAX chains.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix.
    Wy : ndarray of shape (n,) or None
        W @ y (precomputed, for SAR/SDM).  None for SEM/SDEM.
    logdet_jax : callable
        JAX-native logdet function.
    logdet_vec_fn : callable
        Vectorized numpy logdet callable for post-chain LL computation.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    inits : list of GaussianGibbsState
        Per-chain initial states.
    draws : int
        Number of post-warmup draws per chain.
    tune : int
        Number of warmup draws per chain.
    thin : int, default 1
        Keep every thin-th draw.
    jax_seeds : list of int, optional
        Per-chain JAX PRNG seeds.
    model_type : str, default "sar"
        One of "sar", "sem", "sdm", "sdem".
    slice_width : float or None, default None
        Initial step-out width for slice sampling.  If None, defaults
        to ``(rho_upper - rho_lower) * 0.1``.
    progressbar : bool, default True
        Show per-chain progress bars.

    Returns
    -------
    list of dict
        One dict per chain, each containing posterior sample arrays
        and ``mh_accept_rate``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from ._loglik import (
        sar_pointwise_loglik_vectorized,
        sem_pointwise_loglik_vectorized,
    )

    chains = len(inits)
    n, k = X.shape
    is_sar = model_type in ("sar", "sdm")

    # Default slice width: 10% of the support range
    rho_range = priors.rho_upper - priors.rho_lower
    if slice_width is None:
        slice_width = rho_range * 0.1

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)

    if Wy is None:
        Wy_np = np.asarray(W_sparse @ np.asarray(y, dtype=np.float64))
    else:
        Wy_np = np.asarray(Wy, dtype=np.float64)
    _consts = _precompute_gibbs_constants(
        y=y, X=X, Wy=Wy_np, W_sparse=W_sparse, is_sar=is_sar
    )

    # Build the JIT-compiled step function (shared across all chains)
    gibbs_step = _make_gaussian_gibbs_step(
        y_jax=y_jax,
        X_jax=X_jax,
        Wy_jax=_consts["Wy_jax"],
        WX_jax=_consts["WX_jax"],
        n=n,
        k=k,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        yty=_consts["yty"],
        yTWy=_consts["yTWy"],
        WyTWy=_consts["WyTWy"],
        XTy=_consts["XTy"],
        XTWy=_consts["XTWy"],
        WXTy=_consts["WXTy"],
        WXTWy=_consts["WXTWy"],
        XtWX=_consts["XtWX"],
        WXtWX=_consts["WXtWX"],
        priors=priors,
        model_type=model_type,
        logdet_param_fn=logdet_param_fn,
        nu=nu,
    )

    # Convert NumPy initial states to JAX states, then batch into a
    # vmappable pytree by stacking each leaf.
    jax_inits = [
        JAXGaussianGibbsState(
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            sigma2=jnp.float64(init.sigma2),
            rho=jnp.float64(init.rho),
            logdet_params=logdet_params,
            rho_lo=jnp.float64(priors.rho_lower),
            rho_hi=jnp.float64(priors.rho_upper),
            slice_w=jnp.float64(slice_width),
            slice_L=jnp.float64(priors.rho_lower),
            slice_R=jnp.float64(priors.rho_upper),
            v=jnp.ones(n, dtype=jnp.float64),
        )
        for init in inits
    ]
    init_states = jax.tree.map(lambda *a: jnp.stack(a), *jax_inits)

    # Batch PRNG keys
    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(jax_seeds[0])
    keys = jax.random.split(master_key, chains)

    from .._utils._progress import GibbsProgressBarManager

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type=model_type,
    ) as pm:
        # Record start times for all chains (they run simultaneously via vmap)
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        # Chunk both phases into ~20 segments so the progress bar
        # advances smoothly and Python regains control between
        # segments.  Chunk sizes are Python constants so each kernel
        # JIT-compiles once and is reused across chunks.
        warmup_chunk = max(1, tune // 20) if tune > 0 else 1
        draws_chunk = max(1, draws // 20) if draws > 0 else 1

        # ── Phase 1: Warmup via vmap (chunked) ──
        if tune > 0:
            warmup_keys = keys

            def _warmup_chunk(state, key, n):
                return _run_chain_jax_gibbs_scanned(gibbs_step, state, key, n)

            warmup_vmap = jax.jit(
                lambda s, k: jax.vmap(
                    lambda s_, k_: _warmup_chunk(s_, k_, warmup_chunk)
                )(s, k)
            )

            state = init_states
            chunk_keys = warmup_keys
            iter_done = 0
            # Warmup ρ, kept only when a refit will consume it.
            refit_at = tune // 2 if refit_hook is not None else None
            warm_rho: list[np.ndarray] = []
            while iter_done < tune:
                step = min(warmup_chunk, tune - iter_done)
                if step == warmup_chunk:
                    state, chunk_keys, rhos_w, _, _, _ = warmup_vmap(state, chunk_keys)
                else:
                    state, chunk_keys, rhos_w, _, _, _ = jax.vmap(
                        lambda s_, k_: _warmup_chunk(s_, k_, step)
                    )(state, chunk_keys)
                jax.block_until_ready(state.rho)
                if refit_at is not None:
                    warm_rho.append(np.asarray(rhos_w))
                iter_done += step
                if pm is not None:
                    for c in range(chains):
                        pm.update(c, iter_done - 1, tuning=True)

                # ── Warmup Jacobian refit ──
                # Rebuild the interpolant on the ρ range the chains found, then
                # substitute it into the state.  Because the interpolant and the
                # ρ support are traced state rather than closure constants, and
                # the new arrays are zero-padded to the same capacity, the
                # compiled step is reused — no retrace.  Everything after this
                # point, including the rest of warmup, runs under the refit
                # interpolant, so the kernel is frozen well before the first
                # retained draw.
                if refit_at is not None and iter_done >= refit_at:
                    pooled = np.concatenate(warm_rho, axis=0)  # (iters, chains)
                    # Chains start at ρ = 0; the early transient would stretch
                    # the window back to the initial value, so use the tail.
                    pooled = pooled[len(pooled) // 2 :].ravel()
                    refit = refit_hook(pooled)
                    refit_at = None
                    warm_rho = []
                    if refit is not None:
                        new_params, lo, hi = refit
                        import equinox as eqx

                        bcast = jax.tree.map(
                            lambda a: jnp.broadcast_to(a, (chains, *jnp.shape(a))),
                            new_params,
                        )
                        lo_b = jnp.full((chains,), lo, dtype=jnp.float64)
                        hi_b = jnp.full((chains,), hi, dtype=jnp.float64)
                        state = eqx.tree_at(
                            lambda st: (
                                st.rho,
                                st.logdet_params,
                                st.rho_lo,
                                st.rho_hi,
                                st.slice_w,
                                st.slice_L,
                                st.slice_R,
                            ),
                            state,
                            (
                                # A chain may have stepped outside the window in
                                # the chunk after the draws that defined it.
                                jnp.clip(state.rho, lo_b, hi_b),
                                bcast,
                                lo_b,
                                hi_b,
                                # Rescale the slice width and reset the
                                # persistent interval to the new, much narrower
                                # support.
                                jnp.full((chains,), (hi - lo) * 0.1, jnp.float64),
                                lo_b,
                                hi_b,
                            ),
                        )

            final_states = state
        else:
            final_states = init_states

        # ── Phase 2: Post-warmup draws via vmap (chunked) ──
        draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

        def _draws_chunk(state, key, n):
            return _run_chain_jax_gibbs_scanned(gibbs_step, state, key, n)

        draws_vmap = jax.jit(
            lambda s, k: jax.vmap(lambda s_, k_: _draws_chunk(s_, k_, draws_chunk))(
                s, k
            )
        )

        state = final_states
        chunk_keys = draw_keys
        rho_chunks: list[np.ndarray] = []
        beta_chunks: list[np.ndarray] = []
        sigma2_chunks: list[np.ndarray] = []
        draw_accept_sum = jnp.zeros(chains, dtype=jnp.float64)
        iter_done = 0
        while iter_done < draws:
            step = min(draws_chunk, draws - iter_done)
            if step == draws_chunk:
                state, chunk_keys, rhos_c, betas_c, sigma2s_c, chunk_rate = draws_vmap(
                    state, chunk_keys
                )
            else:
                state, chunk_keys, rhos_c, betas_c, sigma2s_c, chunk_rate = jax.vmap(
                    lambda s_, k_: _draws_chunk(s_, k_, step)
                )(state, chunk_keys)
            rho_chunks.append(np.asarray(rhos_c))
            beta_chunks.append(np.asarray(betas_c))
            sigma2_chunks.append(np.asarray(sigma2s_c))
            draw_accept_sum = draw_accept_sum + chunk_rate * jnp.float64(step)
            iter_done += step
            if pm is not None:
                for c in range(chains):
                    pm.update(c, tune + iter_done - 1, tuning=False)

        rhos = np.concatenate(rho_chunks, axis=1)
        betas = np.concatenate(beta_chunks, axis=1)
        sigma2s = np.concatenate(sigma2_chunks, axis=1)
        accept_rates = draw_accept_sum / jnp.float64(draws)

        if pm is not None:
            pm.refresh()

    # Convert to NumPy and assemble per-chain results
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        rho_c = np.asarray(rhos[c][thin_slice])
        beta_c = np.asarray(betas[c][thin_slice])
        sigma_c = np.sqrt(np.asarray(sigma2s[c][thin_slice]))

        # Pointwise log-likelihood, one value per draw and observation, only
        # when requested: at n = 250,000 it is 16 GB and ~10 s of a fit.
        if not log_likelihood:
            log_lik = None
        elif is_sar:
            log_lik = sar_pointwise_loglik_vectorized(
                rho_draws=rho_c,
                beta_draws=beta_c,
                sigma_draws=sigma_c,
                y=y,
                X=X,
                Wy=Wy,
                logdet_vec_fn=logdet_vec_fn,
                n=n,
                nu=nu,
            )
        else:
            log_lik = sem_pointwise_loglik_vectorized(
                lam_draws=rho_c,
                beta_draws=beta_c,
                sigma_draws=sigma_c,
                y=y,
                X=X,
                W_sparse=W_sparse,
                logdet_vec_fn=logdet_vec_fn,
                n=n,
                nu=nu,
            )

        param_name = "rho" if is_sar else "lam"
        results.append(
            {
                param_name: rho_c,
                "beta": beta_c,
                "sigma": sigma_c,
                "log_lik": log_lik,
                "mh_accept_rate": float(accept_rates[c]),
            }
        )

    return results
