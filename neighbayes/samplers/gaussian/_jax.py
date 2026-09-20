r"""JAX-accelerated full-JIT Gibbs sampler for Gaussian spatial models.

Composes the 3-block Gibbs step (β, σ², ρ/λ) into a single
``@eqx.filter_jit``-compiled function, eliminating Python→JAX
dispatch overhead entirely.

Architecture
------------
The sampler uses:

- **β draw**: Conjugate normal via Cholesky factorization.
  O(k³) but fast for k ≤ ~2000.
- **σ² draw**: Conjugate inverse-Gamma (direct, no solve needed).
- **ρ/λ draw**: Neal (2003) stepping-out slice sampling with
  persistent-interval reuse and JAX-native logdet ``log|I-ρW|``
  (eigenvalue, Chebyshev, or trace polynomial).  The slice density is
  the collapsed (SAR) or conditional (SEM) log-density with β and σ²
  integrated out.

For SAR/SDM, the ρ log-density is *collapsed* (β and σ² integrated out):

.. math::

    \log p(\rho \mid y) = \log|I - \rho W|
    - \frac{n-k}{2}\log\text{RSS}(\rho) + \text{const}

where RSS(ρ) uses the Woodbury form:

.. math::

    \text{RSS}(\rho) = r^\top r - (X^\top r)^\top (X^\top X)^{-1}(X^\top r),
    \quad r = y - \rho W y

For SEM/SDEM, the λ log-density is also *collapsed* (β and σ² integrated out):

.. math::

    \log p(\lambda \mid y) = \log|I - \lambda W|
    - \frac{1}{2}\log|X^{*\top} X^*|
    - \frac{n-k}{2}\log\text{RSS}(\lambda) + \text{const}

where :math:`y^* = (I - \lambda W)y`, :math:`X^* = (I - \lambda W)X`, and
:math:`\text{RSS}(\lambda) = y^{*\top} y^* - y^{*\top} X^* (X^{*\top} X^*)^{-1} X^{*\top} y^*`.
The extra term :math:`-\frac{1}{2}\log|X^{*\top} X^*|` appears because :math:`X^*`
depends on :math:`\lambda` (unlike SAR where :math:`X` is fixed).

Slice sampling
~~~~~~~~~~~~~~
The ρ/λ update uses Neal's (2003) stepping-out slice sampler on the
collapsed log-density, carrying a persistent interval in the sampler
state across sweeps for better ESS per sample.  No gradient, proposal,
or Metropolis correction is required — every slice step accepts.

Limitations
-----------
- O(k³) dense Cholesky limits scalability to k ≤ ~2000.

References
----------
Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3),
705–767.

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
    XtX_cho_jax,
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
    XtX_cho_jax : tuple of (jax.numpy.ndarray, bool)
        Cholesky factor of X^T X from ``jax.scipy.linalg.cho_factor``.
    priors : GaussianGibbsPriors
        Prior hyperparameters.
    model_type : str
        One of "sar", "sem", "sdm", "sdem".

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
        """One complete 3-block Gibbs sweep: β → σ² → ρ/λ (slice).

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
        sigma2 = state.sigma2
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

        key_beta, key_sigma2, key_rho = jax.random.split(key, 3)

        # ── Block 1: β | ρ, σ², y — conjugate normal ──
        if is_sar:
            r = y_jax - rho * Wy_jax
            X_eff = X_jax
            XtX_eff = XtX_jax
            # X*'y* = Xᵀ(y - ρWy) = XTy - ρ·XTWy — O(k) from precomputed
            Xty_eff = XTy - rho * XTWy
        else:
            # SEM: transform y and X by (I - λW) using PRECOMPUTED Wy/WX
            r = y_jax - rho * Wy_jax
            X_eff = X_jax - rho * WX_jax
            # Quadratic-in-ρ form for X*'X* — no O(nk²) needed
            XtX_eff = XtX_jax - rho * (XtWX + XtWX.T) + (rho * rho) * WXtWX
            # Quadratic-in-ρ form for X*'y* — avoids O(nk) matmul
            Xty_eff = XTy - rho * (XTWy + WXTy) + (rho * rho) * WXTWy

        Sigma_beta_inv = XtX_eff / sigma2 + beta_prior_prec
        rhs_beta = beta_mu_jax / beta_sigma2_jax + Xty_eff / sigma2
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = jnp.linalg.solve(L_beta.T, jnp.linalg.solve(L_beta, rhs_beta))
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + jnp.linalg.solve(L_beta.T, z_beta)

        # ── Block 2: σ² | β, ρ/λ, y — conjugate InverseGamma draw ──
        # Prior: σ² ~ InverseGamma(α, β).  Full conditional:
        #   σ² | rest ~ InverseGamma(α + n/2, β + ss/2)
        # Matches the InverseGamma prior used by the NUTS path so the two
        # samplers target identical posteriors (LeSage 2009 convention).
        if is_sar:
            resid = y_jax - rho * Wy_jax - X_jax @ beta_new
        else:
            # SEM filtered residual: (I-ρW)(y - Xβ) = (y-ρWy) - (X-ρWX)β
            resid = r - X_eff @ beta_new

        ss = resid @ resid
        a_post = sigma2_alpha_jax + jnp.float64(n / 2.0)
        b_post = sigma2_beta_jax + 0.5 * ss
        sigma2_inv = jax.random.gamma(key_sigma2, a_post) / b_post
        sigma2_new = jnp.maximum(1.0 / sigma2_inv, 1e-10)

        # ── Block 3: ρ/λ — slice sampling ──
        # Build the collapsed log-density used by the slice sampler.
        # All n-dependent inner products are *precomputed* closure constants,
        # so each density evaluation is O(k³) — no O(nk) or O(n²k) work.
        if is_sar:

            def log_density_spatial(param_val):
                # r'r and X'r in closed form (quadratic / linear in param_val)
                r_dot_r = yty - 2.0 * param_val * yTWy + (param_val * param_val) * WyTWy
                Xtr = XTy - param_val * XTWy
                rss = r_dot_r - Xtr @ jax.scipy.linalg.cho_solve(XtX_cho_jax, Xtr)
                rss = jnp.maximum(rss, 1e-300)
                logdet = _logdet_of(param_val)
                log_prior = jnp.where(
                    (param_val >= rho_lo) & (param_val <= rho_hi),
                    0.0,
                    -jnp.inf,
                )
                return logdet - 0.5 * (n - k) * jnp.log(rss) + log_prior

        else:

            def log_density_spatial(param_val):
                rho_sq = param_val * param_val
                yty_star = yty - 2.0 * param_val * yTWy + rho_sq * WyTWy
                Xty_star = XTy - param_val * (XTWy + WXTy) + rho_sq * WXTWy
                XtX_star = XtX_jax - param_val * (XtWX + XtWX.T) + rho_sq * WXtWX

                L_XtX = jnp.linalg.cholesky(XtX_star)
                XtX_star_inv_Xty = jax.scipy.linalg.cho_solve((L_XtX, True), Xty_star)
                rss = yty_star - Xty_star @ XtX_star_inv_Xty
                rss = jnp.maximum(rss, 1e-300)

                logdet = _logdet_of(param_val)
                logdet_XtX = 2.0 * jnp.sum(jnp.log(jnp.diag(L_XtX)))

                log_prior = jnp.where(
                    (param_val >= rho_lo) & (param_val <= rho_hi),
                    0.0,
                    -jnp.inf,
                )
                return (
                    logdet - 0.5 * logdet_XtX - 0.5 * (n - k) * jnp.log(rss) + log_prior
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
    XtX_cho_jax = jax.scipy.linalg.cho_factor(XtX_jax)

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
        XtX_cho_jax=XtX_cho_jax,
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
    XtX_cho_jax = jax.scipy.linalg.cho_factor(XtX_jax)

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
        XtX_cho_jax=XtX_cho_jax,
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
