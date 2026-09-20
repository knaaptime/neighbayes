r"""JAX-accelerated full-JIT Gibbs sampler for SAR-logit.

Composes all Gibbs blocks (PG ω, η, β, ρ slice) into a single
``@jax.jit``-compiled function, eliminating Python→JAX dispatch overhead
entirely.

Architecture
------------
The sampler uses:

- **PG sampling**: Exact Pólya-Gamma draws via :func:`make_pg_draw`
  (pgjax's Devroye sampler when installed; numpy C extension fallback).
  Since h = 1 for binary logit, draws are exact with no truncation bias.
- **η and β draws**: Dense Cholesky factorization via ``jnp.linalg.cholesky``
  and ``jnp.linalg.solve``.  O(n³) but fast for n ≤ ~2000.
- **ρ/λ draw**: 1-D slice sampler on the doubly-collapsed log-density.
  For SAR-logit, the density marginalizes out both η and β via a
  multi-RHS Cholesky solve.  For SEM-logit, the density marginalizes
  out η but conditions on β.  Slice sampling avoids the expensive
  gradient computation that MALA requires (autodiff through Cholesky
  + solve is ~3× the forward cost).


References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.
"""

from __future__ import annotations

import numpy as np

from ..._jax_dispatch import ensure_x64
from .._utils._jax_utils import (
    build_w_bcoo as _build_w_bcoo,
)
from .._utils._jax_utils import (
    check_jax_available as _check_jax_available_impl,
)
from .._utils._jax_utils import (
    make_pg_draw,
)
from .._utils._spatial_normal import (
    _sparsax_factor_ops_available,
    build_precision_krylov_basis_jax,
    eval_precision_logdet_from_basis_jax,
    eval_precision_solve_from_basis_jax,
)
from ._core import JAXLogitGibbsState

# Pólya-Gamma for the ω ~ PG(1, η) block.  pgjax's exact Devroye sampler is
# on-device (no host round-trip) and exact for any h; falls back to the numpy
# C extension via pure_callback when pgjax is unavailable.  h = 1 (Bernoulli).
_draw_pg1 = make_pg_draw()


def _check_jax_available() -> None:
    """Raise ImportError if JAX or equinox is not installed."""
    _check_jax_available_impl(require_equinox=True)


def _make_gibbs_step_with_data(
    y_jax,
    X_jax,
    W_bcoo,
    Wt_bcoo,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    XtX_jax,
    priors,
    n_probes,
    lanczos_deg,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
):
    """Build a JIT-compiled Gibbs step with data bound into the closure.

    This function creates a ``@jax.jit``-compiled function that performs
    one complete Gibbs sweep (ω, η, β, ρ slice) in a single XLA kernel
    call, eliminating all Python→JAX dispatch overhead.

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Binary response vector (JAX array).
    X_jax : jax.numpy.ndarray of shape (n, k)
        Design matrix (JAX array).
    W_bcoo : jax.experimental.sparse.BCOO of shape (n, n)
        Row-standardized W as a sparse BCOO matrix (for ``W @ x``).
    Wt_bcoo : jax.experimental.sparse.BCOO of shape (n, n)
        Transpose ``Wᵀ`` as a sparse BCOO matrix (for ``Wᵀ @ x``).
    n : int
        Number of spatial units.
    k : int
        Number of regression coefficients.
    W_sym_dense : jax.numpy.ndarray of shape (n, n) or None
        Dense (W + W^T).  Only needed for the dense-Cholesky fallback.
    WtW_dense : jax.numpy.ndarray of shape (n, n) or None
        Dense W^T W.  Only needed for the dense-Cholesky fallback.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key, slice_width) -> (new_state, accept)

        where ``state`` is a :class:`JAXLogitGibbsState`,
        ``key`` is a JAX PRNG key, and ``slice_width`` is a ``jnp.float64``
        stepping-out width for the ρ slice sampler.  ``accept`` is always
        ``1.0`` (slice sampling always accepts).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from .._utils._jax_slice import jax_slice_sample_1d

    ensure_x64()

    use_sparsax = sparsax_pattern is not None

    # Sparse W matvecs — never densify W: W @ x and Wᵀ @ x go through BCOO.
    def W_matvec(x):
        return W_bcoo @ x

    def Wt_matvec(x):
        return Wt_bcoo @ x

    # Dense (W+Wᵀ) and WᵀW are only needed to assemble the dense P in the
    # no-sparsax fallback; skip building them on the sparse path.
    if not use_sparsax:
        W_sym = jnp.asarray(W_sym_dense, dtype=jnp.float64)
        WtW = jnp.asarray(WtW_dense, dtype=jnp.float64)
    # Prior hyperparameters
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)
    1.0 / beta_sigma2_jax  # diagonal of V₀⁻¹
    V0_inv_b0 = beta_mu_jax / beta_sigma2_jax  # V₀⁻¹ b₀

    # For logit: h = 1 always, kappa = y - 0.5
    kappa = y_jax - 0.5

    # Precompute W^T X for the β-marginalized density (sparse, O(nnz·k))
    WtX = Wt_bcoo @ X_jax  # (n, k)

    # ── sparsax setup (optional sparse SPD Cholesky path) ──
    if use_sparsax:
        _Ai = jnp.asarray(sparsax_pattern["Ai"], dtype=jnp.int32)
        _Aj = jnp.asarray(sparsax_pattern["Aj"], dtype=jnp.int32)
        _W_sym_vals = jnp.asarray(sparsax_pattern["W_sym_vals"], dtype=jnp.float64)
        _WtW_vals = jnp.asarray(sparsax_pattern["WtW_vals"], dtype=jnp.float64)
        _diag_idx = jnp.asarray(sparsax_pattern["diag_idx"], dtype=jnp.int32)
        _nnz = len(sparsax_pattern["Ai"])
        _n_static = int(sparsax_pattern["n"])
        # Factor-once closures: η-draw and each ρ-density eval do exactly ONE
        # numeric factorization via sparsax 0.4 sample_gaussian / factor_solve
        # (0.3 fallback inside).
        from .._utils._sparsax_utils import make_sparsax_ops

        _eta_sample, _solve_logdet = make_sparsax_ops(_Ai, _Aj, _n_static)

        def _assemble_Ax(omega, rho_val):
            """Assemble COO values for P = I + diag(ω) − ρ(W+Wᵀ) + ρ²WᵀW."""
            Ax = -rho_val * _W_sym_vals + rho_val**2 * _WtW_vals
            diag_vals = jnp.zeros(_nnz, dtype=jnp.float64)
            diag_vals = diag_vals.at[_diag_idx].set(1.0 + omega)
            return Ax + diag_vals

    @eqx.filter_jit
    def gibbs_step(state, key, slice_width, return_steps=False):
        """One complete Gibbs sweep: ω → η → β → ρ (slice).

        Parameters
        ----------
        state : JAXLogitGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.
        slice_width : jax.numpy.float64
            Stepping-out width for the ρ slice sampler.
        return_steps : bool, default False
            Also return the ρ slice's ``(left, right)`` step-out counts.

        Returns
        -------
        new_state : JAXLogitGibbsState
            Updated state.
        accept : jax.numpy.float64
            Always 1.0 (slice sampling always accepts).
        steps : tuple of jax.numpy scalars
            Step-out counts; returned only when ``return_steps``.
        """
        eta = state.eta
        beta = state.beta
        rho = state.rho

        key_omega, key_eta, key_beta, key_rho = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — on-device (pgjax Devroye, else exp) ──
        omega_new = _draw_pg1(jnp.ones(n), eta, key_omega)

        # ── Block 2: η | ω, ρ, β — Cholesky solve (σ² = 1) ──
        # P = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (no 1/σ² scaling)
        Xbeta = X_jax @ beta
        # RHS: Xbeta - ρ W'Xbeta + κ  (σ² = 1); W'Xbeta = (WᵀX)β reuses WtX.
        rhs = Xbeta - rho * (WtX @ beta) + kappa

        if use_sparsax:
            # Sparse SPD Cholesky via sparsax — mean + draw ~ N(P⁻¹ rhs, P⁻¹)
            # from ONE factorization.
            Ax = _assemble_Ax(omega_new, rho)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = _eta_sample(Ax, rhs, z_eta)
        else:
            P_diag = jnp.ones(n) + omega_new
            P = jnp.diag(P_diag) - rho * W_sym + rho**2 * WtW
            P = P + 1e-6 * jnp.eye(n)  # regularization for numerical stability

            L = jnp.linalg.cholesky(P)
            P_inv_rhs = cho_solve((L, True), rhs)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, ρ — conjugate normal (σ² = 1) ──
        A_rho_eta = eta_new - rho * W_matvec(eta_new)
        # Σ_β⁻¹ = Λ₀⁻¹ + X^TX  (no 1/σ² scaling)
        Sigma_beta_inv = beta_prior_prec + XtX_jax
        # rhs = Λ₀⁻¹μ₀ + X^T A_ρη  (no 1/σ² scaling)
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_jax.T @ A_rho_eta
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: ρ — 1-D slice sampler (doubly-collapsed density) ──
        #
        # The doubly-collapsed log-density (η AND β integrated out) is:
        #
        #   log p(ρ | ω, y) = log|I - ρW| - ½ log|P_η|
        #                     - ½ log|Σ_β*⁻¹| + ½ κ' P_η⁻¹ κ
        #                     + ½ m_β*' Σ_β*⁻¹ m_β*
        #
        # Slice sampling avoids the expensive gradient computation that
        # MALA requires (autodiff through Cholesky + solve is ~3× the
        # forward cost).  Each slice candidate needs only a forward
        # evaluation — no backward pass.
        #
        # ── Krylov basis (sparsax factor-reuse path) ──
        # When krylov_degree > 0 and the sparsax factor-reuse primitives
        # are available, factor P(ρ_c) ONCE at the slice centre ρ_c = ρ,
        # build the shift-invert Krylov basis via a fori_loop of
        # solve_factor calls (zero refactors), and evaluate every candidate
        # within krylov_dmax via a Horner recurrence in pure JAX.  Candidates
        # outside the radius fall back to a direct factor_solve.  This is
        # the JAX-native analog of the NumPy KrylovPrecisionBasis.
        _use_krylov = (
            use_sparsax and krylov_degree > 0 and _sparsax_factor_ops_available()
        )
        if _use_krylov:
            # G = ∂P/∂ρ|_{ρ_c} = G1 − 2ρ_c G2  (COO values at the pattern)
            _G_vals_c = _W_sym_vals - 2.0 * rho * _WtW_vals
            # Seed with the ρ-independent RHS columns [κ, X, WtX]
            # so P(ρ)⁻¹u(ρ) = P(ρ)⁻¹X − ρ·P(ρ)⁻¹WtX is a linear combo of
            # two Horner evaluations against the same basis.
            _rhs_seed = jnp.column_stack([kappa, X_jax, WtX])  # (n, 1 + 2k)
            _Ax_c = _assemble_Ax(omega_new, rho)
            (
                _token,
                _V_stack,
                _ld_coefs,
                _safe_dmax,
            ) = build_precision_krylov_basis_jax(
                _Ai,
                _Aj,
                _Ax_c,
                _G_vals_c,
                _WtW_vals,
                _rhs_seed,
                n=_n_static,
                degree=krylov_degree,
                dmax=krylov_dmax,
            )

        def log_density_rho(rho_val):
            """Doubly-collapsed log-density of ρ (η and β integrated out).

            Uses one Cholesky of P_η with (k+1) right-hand sides
            to obtain both the κ quadratic form and the β-marginal
            terms.
            """
            # u = X - ρ W^T X  — (n, k)
            u = X_jax - rho_val * WtX

            # Multi-RHS: P_η [z | M] = [κ | u]
            rhs_stack = jnp.column_stack([kappa, u])  # (n, k+1)

            if use_sparsax:
                if _use_krylov:
                    # Krylov-accelerated path: Horner eval against the basis
                    # built at ρ_c, plus first-order logdet trace correction.
                    drho = rho_val - rho
                    within = jnp.abs(drho) <= _safe_dmax
                    # Horner in pure JAX (no sparsax calls):
                    sol_all = eval_precision_solve_from_basis_jax(_V_stack, drho)
                    # sol_all columns: [P(ρ)⁻¹κ, P(ρ)⁻¹X, P(ρ)⁻¹WtX]
                    z_kry = sol_all[:, 0]
                    PX = sol_all[:, 1 : 1 + k]
                    PWtX = sol_all[:, 1 + k :]
                    # u(ρ) = X − ρ·WtX  →  P(ρ)⁻¹u = PX − ρ·PWtX
                    M_kry = PX - rho_val * PWtX
                    log_det_P_kry = eval_precision_logdet_from_basis_jax(
                        _ld_coefs, drho
                    )
                    # Direct solve for candidates outside the Krylov radius.
                    # `jnp.where` evaluates both operands, so this factorization
                    # runs regardless and the basis is pure added work — see the
                    # note in ``samplers/negbin/_jax.py`` for why the fallback
                    # cannot simply be dropped.
                    Ax_r = _assemble_Ax(omega_new, rho_val)
                    sol_dir, log_det_P_dir = _solve_logdet(Ax_r, rhs_stack)
                    z_vec = jnp.where(within, z_kry, sol_dir[:, 0])
                    M_mat = jnp.where(within, M_kry, sol_dir[:, 1:])
                    log_det_P = jnp.where(within, log_det_P_kry, log_det_P_dir)
                else:
                    Ax_r = _assemble_Ax(omega_new, rho_val)
                    # Multi-RHS solve + logdet from one factorization (factor_solve).
                    sol, log_det_P = _solve_logdet(Ax_r, rhs_stack)  # sol: (n, k+1)
                    z_vec = sol[:, 0]
                    M_mat = sol[:, 1:]
            else:
                # P_η = I + diag(ω) - ρ(W+W^T) + ρ²W^TW  (σ² = 1)
                P_diag_r = jnp.ones(n) + omega_new
                P_r = jnp.diag(P_diag_r) - rho_val * W_sym + rho_val**2 * WtW
                P_r = P_r + 1e-6 * jnp.eye(n)

                L_r = jnp.linalg.cholesky(P_r)
                sol = cho_solve((L_r, True), rhs_stack)  # (n, k+1)
                log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
                z_vec = sol[:, 0]
                M_mat = sol[:, 1:]

            # Σ_β*⁻¹ = X^TX + V₀⁻¹ - u^T M  — (k, k)
            Sig_beta_inv = XtX_jax + beta_prior_prec - u.T @ M_mat
            Sig_beta_inv = Sig_beta_inv + 1e-10 * jnp.eye(k)  # regularize

            L_sig = jnp.linalg.cholesky(Sig_beta_inv)
            log_det_Sig_inv = 2.0 * jnp.sum(jnp.log(jnp.diag(L_sig)))

            # m_β* = Σ_β* (u^T z + V₀⁻¹ b₀)
            rhs_b = u.T @ z_vec + V0_inv_b0
            m_b = cho_solve((L_sig, True), rhs_b)

            # Quadratic forms
            quad_kappa = kappa @ z_vec
            quad_beta = rhs_b @ m_b

            logdet_W = logdet_jax(rho_val)

            log_prior = jnp.where(
                (rho_val >= rho_lower_jax) & (rho_val <= rho_upper_jax), 0.0, -jnp.inf
            )
            return (
                logdet_W
                - 0.5 * log_det_P
                - 0.5 * log_det_Sig_inv
                + 0.5 * quad_kappa
                + 0.5 * quad_beta
                + log_prior
            )

        # ── Slice sampling for ρ (shared JAX helper) ──
        rho_new, _, steps_left, steps_right = jax_slice_sample_1d(
            log_density_rho,
            rho,
            rho_lower_jax,
            rho_upper_jax,
            key=key_rho,
            w=slice_width,
            return_steps=True,
        )

        new_state = JAXLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            rho=rho_new,
            omega=omega_new,
        )
        if return_steps:
            return new_state, jnp.float64(1.0), (steps_left, steps_right)
        return new_state, jnp.float64(1.0)  # slice always accepts

    return gibbs_step


def _logit_loglik_pointwise_jax(y, eta):
    """Pointwise logit log-likelihood for storage (host-side, returns NumPy).

    Delegates to the single NumPy source in :mod:`..logit._core`: this runs
    off-device on already-materialized draws, so there is nothing for JAX to
    do here and no reason to keep a second copy of the formula.
    """
    from ._core import _logit_loglik_pointwise

    return _logit_loglik_pointwise(
        np.asarray(y, dtype=np.float64), np.asarray(eta, dtype=np.float64)
    )


def run_chain_jax(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
):
    """Run one chain of the full-JIT JAX Gibbs sampler for SAR-logit.

    Creates a JIT-compiled Gibbs step function and runs it in a Python
    loop, handling warmup, thinning, and storage.  The ρ update uses a
    1-D slice sampler on the doubly-collapsed log-density (η and β
    integrated out).

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix (used to extract W_dense).
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T).
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.
    priors : LogitGibbsPriors
        Prior hyperparameters.
    init : LogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    pg_n_terms : int
        Ignored (kept for API compatibility).  PG draws now use the
        exact sum-of-exponentials method which does not require
        truncation.
    n_probes : int
        Ignored (kept for API compatibility).
    lanczos_deg : int
        Ignored (kept for API compatibility).
    progress_manager : optional
        Progress bar manager.
    chain_id : int, default 0
        Chain identifier for progress reporting.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0; slice sampling
        always accepts).
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)

    # Sparse W as BCOO — never densify W for the O(nnz) matvecs.
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    # Initialize state
    state = JAXLogitGibbsState(
        eta=jnp.asarray(init.eta, dtype=jnp.float64),
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        rho=jnp.float64(init.rho),
        omega=jnp.asarray(init.omega, dtype=jnp.float64),
    )

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Build the JIT-compiled step function
    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    # Warmup the JIT function (first call triggers compilation)
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    slice_width = jnp.float64(0.2)
    _ = gibbs_step(state, warmup_key, slice_width)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key, slice_width)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = float(state.rho)
                beta_samples[idx] = np.asarray(state.beta)
                eta_np = np.asarray(state.eta)
                eta_norm_samples[idx] = float(eta_np @ eta_np)
                if return_eta:
                    eta_samples[idx] = eta_np
                if store_log_lik:
                    log_lik_samples[idx] = _logit_loglik_pointwise_jax(y, eta_np)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    if progress_manager is not None:
        progress_manager.refresh()

    result = {
        "rho": rho_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
        "mh_accept_rate": 1.0,  # slice always accepts
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ===========================================================================
# SEM-logit JAX Gibbs sampler
# ===========================================================================


def _make_gibbs_step_with_data_sem(
    y_jax,
    X_jax,
    W_bcoo,
    Wt_bcoo,
    n,
    k,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    n_probes,
    lanczos_deg,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
):
    """Build a JIT-compiled SEM-logit Gibbs step with data bound into the closure.

    Same 4-block structure as SAR-logit (ω, η, β, λ) but with SEM-specific
    rhs and β block formulas.

    Key differences from SAR-logit:
    - η rhs: A_λ'A_λXβ + κ  (not A_λ'Xβ + κ)
    - β block: uses X* = A_λX, η* = A_λη  (not X, A_ρη)
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    from .._utils._jax_slice import jax_slice_sample_1d
    from ._core import JAXSEMLogitGibbsState

    ensure_x64()

    use_sparsax = sparsax_pattern is not None

    # Sparse W matvecs — never densify W.  (W+Wᵀ)@x and (WᵀW)@x compose the
    # base W @ x / Wᵀ @ x BCOO matvecs, so no dense W_sym/WtW is materialized.
    def W_matvec(x):
        return W_bcoo @ x

    def Wt_matvec(x):
        return Wt_bcoo @ x

    def Wsym_matvec(x):
        return W_matvec(x) + Wt_matvec(x)

    def WtW_matvec(x):
        return Wt_matvec(W_matvec(x))

    # Dense (W+Wᵀ) and WᵀW are only needed to assemble the dense P in the
    # no-sparsax fallback; skip building them on the sparse path.
    if not use_sparsax:
        W_sym = jnp.asarray(W_sym_dense, dtype=jnp.float64)
        WtW = jnp.asarray(WtW_dense, dtype=jnp.float64)
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    lam_lower_jax = jnp.float64(priors.lam_lower)
    lam_upper_jax = jnp.float64(priors.lam_upper)

    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)
    beta_mu_jax / beta_sigma2_jax  # V₀⁻¹ b₀

    kappa = y_jax - 0.5

    # ── sparsax setup (optional sparse SPD Cholesky path) ──
    if use_sparsax:
        _Ai = jnp.asarray(sparsax_pattern["Ai"], dtype=jnp.int32)
        _Aj = jnp.asarray(sparsax_pattern["Aj"], dtype=jnp.int32)
        _W_sym_vals = jnp.asarray(sparsax_pattern["W_sym_vals"], dtype=jnp.float64)
        _WtW_vals = jnp.asarray(sparsax_pattern["WtW_vals"], dtype=jnp.float64)
        _diag_idx = jnp.asarray(sparsax_pattern["diag_idx"], dtype=jnp.int32)
        _nnz = len(sparsax_pattern["Ai"])
        _n_static = int(sparsax_pattern["n"])
        # Factor-once closures (sparsax 0.4 sample_gaussian / factor_solve).
        from .._utils._sparsax_utils import make_sparsax_ops

        _eta_sample, _solve_logdet = make_sparsax_ops(_Ai, _Aj, _n_static)

        def _assemble_Ax(omega, lam_val):
            """Assemble COO values for P = I + diag(ω) − λ(W+Wᵀ) + λ²WᵀW."""
            Ax = -lam_val * _W_sym_vals + lam_val**2 * _WtW_vals
            diag_vals = jnp.zeros(_nnz, dtype=jnp.float64)
            diag_vals = diag_vals.at[_diag_idx].set(1.0 + omega)
            return Ax + diag_vals

    @eqx.filter_jit
    def gibbs_step(state, key, slice_width, return_steps=False):
        """One complete SEM-logit Gibbs sweep: ω → η → β → λ (slice).

        With ``return_steps`` it also returns the λ slice's ``(left, right)``
        step-out counts.
        """
        eta = state.eta
        beta = state.beta
        lam = state.lam

        key_omega, key_eta, key_beta, key_lam = jax.random.split(key, 4)

        # ── Block 1: ω ~ PG(1, η) — on-device (pgjax Devroye, else exp) ──
        omega_new = _draw_pg1(jnp.ones(n), eta, key_omega)

        # ── Block 2: η | ω, β, λ — SEM-specific rhs ──
        # P = I + diag(ω) - λ(W+W^T) + λ²W^TW
        Xbeta = X_jax @ beta
        # SEM rhs: A_λ'A_λXβ + κ = Xβ - λ(W+W')Xβ + λ²W'WXβ + κ
        WsymXbeta = Wsym_matvec(Xbeta)
        WtWXbeta = WtW_matvec(Xbeta)
        rhs = Xbeta - lam * WsymXbeta + lam**2 * WtWXbeta + kappa

        if use_sparsax:
            # Mean + draw ~ N(P⁻¹ rhs, P⁻¹) from ONE factorization.
            Ax = _assemble_Ax(omega_new, lam)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = _eta_sample(Ax, rhs, z_eta)
        else:
            P_diag = jnp.ones(n) + omega_new
            P = jnp.diag(P_diag) - lam * W_sym + lam**2 * WtW
            P = P + 1e-6 * jnp.eye(n)

            L = jnp.linalg.cholesky(P)
            P_inv_rhs = cho_solve((L, True), rhs)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, λ — SEM-style transformed data ──
        # X* = (I - λW)X,  η* = (I - λW)η
        A_lam_eta = eta_new - lam * W_matvec(eta_new)
        X_star = X_jax - lam * W_matvec(X_jax)
        eta_star = A_lam_eta

        XstXs = X_star.T @ X_star
        Sigma_beta_inv = beta_prior_prec + XstXs
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_star.T @ eta_star
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: λ — 1-D slice sampler (η-collapsed density) ──
        #
        # The collapsed log-density (η integrated out, β conditioned) is:
        #
        #   log p(λ | β, ω, y) = log|I - λW| - ½ log|P_η|
        #                       + ½ rhs^T P_η⁻¹ rhs - ½ Xβ^T A_λ^T A_λ Xβ
        #
        # Slice sampling avoids the expensive gradient computation that
        # MALA requires (autodiff through Cholesky + solve is ~3× the
        # forward cost).

        # Hoist the λ-independent matvecs out of the slice density: Xβ and the
        # (W+Wᵀ)Xβ, WᵀWXβ products are constant across every stepping-out /
        # shrink evaluation, so a single set of sparse matvecs per sweep
        # replaces recomputing them dozens of times.
        Xbeta_r = X_jax @ beta_new
        WsymXbeta_r = Wsym_matvec(Xbeta_r)
        WtWXbeta_r = WtW_matvec(Xbeta_r)

        # ── Krylov basis (sparsax factor-reuse path, SEM) ──
        # The SEM RHS is quadratic in λ: rhs = Xβ − λ·W_sym·Xβ + λ²·WtW·Xβ + κ.
        # Seed with [κ, Xβ, W_sym·Xβ, WtW·Xβ] and reconstruct the solve as a
        # quadratic combination of Horner evaluations.
        _use_krylov_sem = (
            use_sparsax and krylov_degree > 0 and _sparsax_factor_ops_available()
        )
        if _use_krylov_sem:
            _lam_c = lam
            _G_vals_c_sem = _W_sym_vals - 2.0 * _lam_c * _WtW_vals
            _rhs_seed_sem = jnp.column_stack(
                [
                    kappa,
                    Xbeta_r,
                    WsymXbeta_r,
                    WtWXbeta_r,
                ]
            )  # (n, 4)
            _Ax_c_sem = _assemble_Ax(omega_new, _lam_c)
            (
                _token_sem,
                _V_stack_sem,
                _ld_coefs_sem,
                _safe_dmax_sem,
            ) = build_precision_krylov_basis_jax(
                _Ai,
                _Aj,
                _Ax_c_sem,
                _G_vals_c_sem,
                _WtW_vals,
                _rhs_seed_sem,
                n=_n_static,
                degree=krylov_degree,
                dmax=krylov_dmax,
            )

        def log_density_lam(lam_val):
            """Collapsed log-density of λ (η integrated out, β conditioned).

            Uses one Cholesky of P_r to obtain log|P_r| and the
            quadratic form exactly.
            """
            rhs_r = Xbeta_r - lam_val * WsymXbeta_r + lam_val**2 * WtWXbeta_r + kappa

            if use_sparsax:
                if _use_krylov_sem:
                    dlam = lam_val - _lam_c
                    within = jnp.abs(dlam) <= _safe_dmax_sem
                    sol_all = eval_precision_solve_from_basis_jax(_V_stack_sem, dlam)
                    # sol_all: [P(λ)⁻¹κ, P(λ)⁻¹Xβ, P(λ)⁻¹W_symXβ, P(λ)⁻¹WtWXβ]
                    Pkappa = sol_all[:, 0]
                    PXbeta = sol_all[:, 1:2]
                    PWsymXbeta = sol_all[:, 2:3]
                    PWtWXbeta = sol_all[:, 3:]
                    # rhs = Xβ − λ·W_symXβ + λ²·WtWXβ + κ
                    m_kry = (
                        PXbeta - lam_val * PWsymXbeta + lam_val**2 * PWtWXbeta
                    ).ravel() + Pkappa
                    log_det_P_kry = eval_precision_logdet_from_basis_jax(
                        _ld_coefs_sem, dlam
                    )
                    quad_kry = rhs_r @ m_kry
                    # Direct solve outside the radius (see the SAR-logit site).
                    Ax_r = _assemble_Ax(omega_new, lam_val)
                    m_dir, log_det_P_dir = _solve_logdet(Ax_r, rhs_r)
                    quad_dir = rhs_r @ m_dir
                    m = jnp.where(within, m_kry, m_dir)
                    log_det_P = jnp.where(within, log_det_P_kry, log_det_P_dir)
                    quad_r = jnp.where(within, quad_kry, quad_dir)
                else:
                    Ax_r = _assemble_Ax(omega_new, lam_val)
                    # Solve + logdet from one factorization (factor_solve).
                    m, log_det_P = _solve_logdet(Ax_r, rhs_r)
                    quad_r = rhs_r @ m
            else:
                P_diag_r = jnp.ones(n) + omega_new
                P_r = jnp.diag(P_diag_r) - lam_val * W_sym + lam_val**2 * WtW
                P_r = P_r + 1e-6 * jnp.eye(n)

                L_r = jnp.linalg.cholesky(P_r)
                log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
                v = solve_triangular(L_r, rhs_r, lower=True)
                quad_r = v @ v

            logdet_W = logdet_jax(lam_val)

            # SEM correction: -½Xβ'A_λ'A_λXβ
            # = ½λXβ'(W+W')Xβ - ½λ²Xβ'W'WXβ + const
            xbeta_correction = 0.5 * lam_val * (
                Xbeta_r @ WsymXbeta_r
            ) - 0.5 * lam_val**2 * (Xbeta_r @ WtWXbeta_r)

            log_prior = jnp.where(
                (lam_val >= lam_lower_jax) & (lam_val <= lam_upper_jax), 0.0, -jnp.inf
            )
            return (
                logdet_W - 0.5 * log_det_P + 0.5 * quad_r + xbeta_correction + log_prior
            )

        # ── Slice sampling for λ (shared JAX helper) ──
        lam_new, _, steps_left, steps_right = jax_slice_sample_1d(
            log_density_lam,
            lam,
            lam_lower_jax,
            lam_upper_jax,
            key=key_lam,
            w=slice_width,
            return_steps=True,
        )

        new_state = JAXSEMLogitGibbsState(
            eta=eta_new,
            beta=beta_new,
            lam=lam_new,
            omega=omega_new,
        )
        if return_steps:
            return new_state, jnp.float64(1.0), (steps_left, steps_right)
        return new_state, jnp.float64(1.0)  # slice always accepts

    return gibbs_step


def run_chain_jax_sem(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    init,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
):
    """Run one chain of the full-JIT JAX Gibbs sampler for SEM-logit.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Binary response vector.
    X : ndarray of shape (n, k)
        Design matrix.
    W_sparse : csr_matrix
        Spatial weights matrix.
    W_sym_dense : jax.numpy.ndarray of shape (n, n)
        Dense (W + W^T).
    WtW_dense : jax.numpy.ndarray of shape (n, n)
        Dense W^T W.
    logdet_jax : callable
        JAX-native function ``(lam) -> jax.numpy.ndarray`` computing
        log|I - lam*W|.
    priors : SEMLogitGibbsPriors
        Prior hyperparameters.
    init : SEMLogitGibbsState
        Initial state.
    draws : int
        Number of post-warmup draws.
    tune : int
        Number of warmup draws.
    thin : int
        Keep every thin-th draw.
    return_eta : bool
        If True, store the full latent field η.
    rng : numpy.random.Generator, optional
        Random state.
    pg_n_terms : int
        Ignored (kept for API compatibility).  PG draws now use the
        exact sum-of-exponentials method which does not require
        truncation.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``lam``, ``beta``, ``log_lik``,
        ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0; slice sampling
        always accepts).
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from ._core import JAXSEMLogitGibbsState

    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    # Convert data to JAX arrays
    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)

    # Sparse W as BCOO — never densify W for the O(nnz) matvecs.
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    # Initialize state
    state = JAXSEMLogitGibbsState(
        eta=jnp.asarray(init.eta, dtype=jnp.float64),
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        lam=jnp.float64(init.lam),
        omega=jnp.asarray(init.omega, dtype=jnp.float64),
    )

    # Pre-allocate storage
    lam_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Build the JIT-compiled step function
    gibbs_step = _make_gibbs_step_with_data_sem(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    # Warmup the JIT function
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    slice_width = jnp.float64(0.2)
    _ = gibbs_step(state, warmup_key, slice_width)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key, slice_width)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                lam_samples[idx] = float(state.lam)
                beta_samples[idx] = np.asarray(state.beta)
                eta_np = np.asarray(state.eta)
                eta_norm_samples[idx] = float(eta_np @ eta_np)
                if return_eta:
                    eta_samples[idx] = eta_np
                if store_log_lik:
                    log_lik_samples[idx] = _logit_loglik_pointwise_jax(y, eta_np)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    if progress_manager is not None:
        progress_manager.refresh()

    result = {
        "lam": lam_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
        "mh_accept_rate": 1.0,  # slice always accepts
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ===========================================================================
# Multi-chain runners
# ===========================================================================
#
# Chains run in parallel threads, each an ordinary ``jax.jit`` program sharing
# one compiled Gibbs step (see ``run_chains_chunked``).  ``jax.pmap`` and
# ``jax.vmap`` are both slower here; ``run_chains_in_threads`` records why.


def _logit_loglik_pointwise_jax_op(y_jax, eta):
    """Pointwise logit log-likelihood as a pure-JAX op (vmap-safe).

    The one copy of this formula that must stay JAX-native: it runs inside
    the jitted, vmapped Gibbs step.  Keep it equal to
    :func:`..logit._core._logit_loglik_pointwise`.
    """
    import jax.numpy as jnp

    return y_jax * eta - jnp.logaddexp(0.0, eta)


def _stack_chain_inits(inits, state_cls, scalar_field):
    """Convert a list of per-chain states into one vmap-able pytree.

    ``scalar_field`` names the scalar parameter (``"rho"`` or ``"lam"``)
    so the helper works for both SAR and SEM state classes.
    """
    import jax.numpy as jnp

    jax_inits = [
        state_cls(
            eta=jnp.asarray(init.eta, dtype=jnp.float64),
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            omega=jnp.asarray(init.omega, dtype=jnp.float64),
            **{scalar_field: jnp.float64(getattr(init, scalar_field))},
        )
        for init in inits
    ]
    import jax

    return jax.tree.map(lambda *a: jnp.stack(a), *jax_inits)


def run_chains_jax_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progressbar: bool = True,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    store_log_lik: bool = True,
) -> list[dict]:
    """Run multiple SAR-logit Gibbs chains in parallel.

    Chains run in parallel threads, each an ordinary ``jax.jit`` program, and
    the Gibbs step is compiled once (see
    :func:`.._utils._jax_utils.run_chains_chunked`).

    Parameters
    ----------
    y, X, W_sparse, W_sym_dense, WtW_dense, logdet_jax, priors :
        Same meaning as in :func:`run_chain_jax`.
    inits : list of LogitGibbsState
        Per-chain initial states.  Length determines the number of
        chains.
    draws, tune, thin :
        Same meaning as in :func:`run_chain_jax`.
    jax_seeds : list of int, optional
        Per-chain JAX PRNG seeds.  Defaults to ``range(chains)``.
    pg_n_terms, n_probes, lanczos_deg :
        Same meaning as in :func:`run_chain_jax`.
    progressbar : bool, default True
        Show per-chain progress bars.  Because all chains advance in
        lock-step under vmap, the bar is updated in bulk at the end of
        each phase.

    Returns
    -------
    list of dict
        One dict per chain with keys ``rho``, ``beta``, ``log_lik``,
        ``eta_norm``, ``mh_accept_rate``.

    Notes
    -----
    ``return_eta=True`` is not supported by this runner; use the
    sequential :func:`run_chain_jax` path if you need the full latent
    field.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    gibbs_step = _make_gibbs_step_with_data(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        XtX_jax=XtX_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        sparsax_pattern=sparsax_pattern,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    init_states = _stack_chain_inits(inits, JAXLogitGibbsState, "rho")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._progress import GibbsProgressBarManager

    draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

    def _sweep(carry, key, tuning):
        # The slice starts at width 0.2, as in the NumPy sampler, and adapts
        # during warmup.
        state, width = carry
        state, _, (steps_left, steps_right) = gibbs_step(
            state, key, width, return_steps=True
        )
        width = adapt_slice_width(width, steps_left, steps_right, tuning)
        trace = (state.rho, state.beta, state.eta @ state.eta)
        if store_log_lik:
            trace += (_logit_loglik_pointwise_jax_op(y_jax, state.eta),)
        return (state, width), trace

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sar_logit",
    ) as pm:
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        def _progress(i, tuning):
            if pm is not None:
                for c in range(chains):
                    pm.update(c, i, tuning=tuning)

        _, traces = run_chains_chunked(
            _sweep,
            [
                (jax.tree.map(lambda a, c=c: a[c], init_states), jnp.float64(0.2))
                for c in range(chains)
            ],
            list(warmup_keys),
            list(draw_keys),
            tune=tune,
            draws=draws,
            on_chunk=_progress,
        )

    # Thin and pack as per-chain dicts
    rhos, betas, eta_norms = traces[:3]
    log_liks = traces[3] if store_log_lik else None
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "rho": rhos[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy() if store_log_lik else None,
                "mh_accept_rate": 1.0,
            }
        )
    return results


def run_chains_jax_sem_vectorized(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse,
    W_sym_dense,
    WtW_dense,
    logdet_jax,
    priors,
    inits: list,
    draws: int,
    tune: int,
    thin: int = 1,
    jax_seeds: list[int] | None = None,
    pg_n_terms: int = 25,
    n_probes: int = 5,
    lanczos_deg: int = 15,
    progressbar: bool = True,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
    store_log_lik: bool = True,
) -> list[dict]:
    """Run multiple SEM-logit Gibbs chains in parallel.

    See :func:`run_chains_jax_vectorized` for the SAR-logit analogue
    and shared design rationale.  Returns per-chain dicts keyed on
    ``lam`` rather than ``rho``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    from ._core import JAXSEMLogitGibbsState

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    W_bcoo, Wt_bcoo = _build_w_bcoo(W_sparse)

    gibbs_step = _make_gibbs_step_with_data_sem(
        y_jax=y_jax,
        X_jax=X_jax,
        W_bcoo=W_bcoo,
        Wt_bcoo=Wt_bcoo,
        n=n,
        k=k,
        W_sym_dense=W_sym_dense,
        WtW_dense=WtW_dense,
        logdet_jax=logdet_jax,
        priors=priors,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        sparsax_pattern=sparsax_pattern,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    init_states = _stack_chain_inits(inits, JAXSEMLogitGibbsState, "lam")

    if jax_seeds is None:
        jax_seeds = list(range(chains))
    master_key = jax.random.PRNGKey(int(jax_seeds[0]))
    warmup_keys = jax.random.split(master_key, chains)

    from .._utils._jax_slice import adapt_slice_width
    from .._utils._jax_utils import run_chains_chunked
    from .._utils._progress import GibbsProgressBarManager

    draw_keys = jax.random.split(jax.random.fold_in(master_key, 1), chains)

    def _sweep(carry, key, tuning):
        # The slice starts at width 0.2, as in the NumPy sampler, and adapts
        # during warmup.
        state, width = carry
        state, _, (steps_left, steps_right) = gibbs_step(
            state, key, width, return_steps=True
        )
        width = adapt_slice_width(width, steps_left, steps_right, tuning)
        trace = (state.lam, state.beta, state.eta @ state.eta)
        if store_log_lik:
            trace += (_logit_loglik_pointwise_jax_op(y_jax, state.eta),)
        return (state, width), trace

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sem_logit",
    ) as pm:
        if pm is not None:
            for c in range(chains):
                pm.start_chain(c)

        def _progress(i, tuning):
            if pm is not None:
                for c in range(chains):
                    pm.update(c, i, tuning=tuning)

        _, traces = run_chains_chunked(
            _sweep,
            [
                (jax.tree.map(lambda a, c=c: a[c], init_states), jnp.float64(0.2))
                for c in range(chains)
            ],
            list(warmup_keys),
            list(draw_keys),
            tune=tune,
            draws=draws,
            on_chunk=_progress,
        )

    lams, betas, eta_norms = traces[:3]
    log_liks = traces[3] if store_log_lik else None
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "lam": lams[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy() if store_log_lik else None,
                "mh_accept_rate": 1.0,
            }
        )
    return results
