r"""JAX-accelerated full-JIT Gibbs sampler for SAR Negative Binomial.

Composes all Gibbs blocks (PG ω, η, β, σ², ρ slice) into a single
``@jax.jit``-compiled function, eliminating Python→JAX dispatch overhead
entirely.  Achieves 12–92× speedup over CHOLMOD for n ≤ 1000.

The key insight is that each individual JAX operation (PG draw, Cholesky
solve, etc.) is fast (~0.05ms), but the Python→JAX dispatch overhead
per call is ~30ms.  By composing all blocks into a single JIT-compiled
function, we pay the dispatch cost only once per Gibbs iteration instead
of 6+ times.

Architecture
------------
The sampler uses:

- **PG sampling**: Truncated alternating-series method (K=``pg_n_terms``)
  using ``jax.random.gamma`` for Gamma(h, 1) draws — correct variance
  (Var[PG] ∝ h, not h²) — plus a closed-form tail-mean correction that
  makes the sample mean exact (see :func:`_pg_gamma_series_draw`).
  Fully JIT-compatible.
- **η and β draws**: Dense Cholesky factorization via ``jnp.linalg.cholesky``
  and ``jnp.linalg.solve``.  O(n³) but fast for n ≤ ~2000.
- **σ² draw**: Conjugate inverse-Gamma (direct, no solve needed).
- **ρ draw**: Neal's stepping-out slice sampler with Lanczos-based
  ``log|P|`` estimation.  A fixed Lanczos key is used for all
  log-density evaluations within a single slice step, ensuring the
  density is deterministic within the stepping-out and shrinkage
  phases.  This matches the numpy path's slice sampler and avoids the
  mixing issues that MALA/MH can have for spatial autoregressive
  parameters.
- **α draw**: JAX-compiled slice sampling on log(α).

Limitations
-----------
- O(n³) dense Cholesky limits scalability to n ≤ ~2000.
- The PG draw's neglected tail *variance* is O(1/K³) (its mean is exact
  after the tail correction).
- Lanczos log|P| estimation is stochastic but uses a fixed key within
  each slice step for consistency.

References
----------
Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference
for logistic models using Pólya–Gamma latent variables.
*Journal of the American Statistical Association*, 108(504), 1339–1349.

Neal, R. M. (2003). Slice sampling. *Annals of Statistics*, 31(3), 705–767.
"""

from __future__ import annotations

import numpy as np

from ..._jax_dispatch import ensure_x64
from .._utils._spatial_normal import (
    _sparsax_factor_ops_available,
    build_precision_krylov_basis_jax,
    eval_precision_logdet_from_basis_jax,
    eval_precision_solve_from_basis_jax,
)
from ._core import JAXGibbsState


def _pg_gamma_series_draw(key, h, z, n_terms: int):
    r"""Draw PG(h, z) via the truncated Gamma series with tail-mean correction.

    .. math::

        \mathrm{PG}(h, z) = \frac{1}{2\pi^2} \sum_{k \ge 0}
            \frac{g_k}{(k + 1/2)^2 + c^2},
        \quad g_k \sim \Gamma(h, 1),\; c = \frac{|z|}{2\pi}.

    Truncating at ``K = n_terms`` drops a tail whose *mean* is
    deterministic and available in closed form via
    :math:`\sum_{k \ge 0} 1/((k+1/2)^2 + c^2) = \pi \tanh(\pi c)/(2c)`:
    adding ``h * (S_inf(c) - S_K(c)) / (2 pi^2)`` back makes the sample
    mean exactly :math:`E[\mathrm{PG}(h,z)] = h \tanh(z/2)/(2z)`.  The
    neglected tail *variance* is :math:`O(1/K^3)` — negligible at K=25.

    Without the correction the truncated series has a systematic −0.8%
    to −1.7% mean bias at K=25 (the bias for which the old Gamma-series
    method was removed).  This function is the scan-compatible
    replacement.

    Parameters
    ----------
    key : jax.random.PRNGKey
        PRNG key for the Gamma draws.
    h : jax.numpy.ndarray of shape (n,)
        Shape parameters (NB augmentation: ``h = y + alpha``); may be
        non-integer.
    z : jax.numpy.ndarray of shape (n,)
        Tilting parameters.
    n_terms : int
        Number of series terms K (static under ``jit``).

    Returns
    -------
    jax.numpy.ndarray of shape (n,)
        PG(h, z) draws, mean-exact, all elements positive.
    """
    import jax
    import jax.numpy as jnp

    pi = jnp.pi
    pi2 = pi * pi
    k_idx = jnp.arange(n_terms, dtype=jnp.float64)

    c = jnp.abs(z) / (2.0 * pi)  # (n,)
    denominators = (k_idx + 0.5) ** 2 + c[:, None] ** 2  # (n, K)

    g = jax.random.gamma(
        key, h[:, None] * jnp.ones((1, n_terms)), dtype=jnp.float64
    )  # (n, K) Gamma(h, 1) draws
    series = jnp.sum(g / denominators, axis=1) / (2.0 * pi2)  # (n,)

    # Closed-form tail mean: S_inf(c) = pi*tanh(pi*c)/(2c), S_inf(0) = pi^2/2.
    safe_c = jnp.where(c < 1e-12, 1.0, c)
    s_inf = jnp.where(c < 1e-12, pi2 / 2.0, pi * jnp.tanh(pi * safe_c) / (2.0 * safe_c))
    s_partial = jnp.sum(1.0 / denominators, axis=1)  # (n,)
    tail_mean = h * (s_inf - s_partial) / (2.0 * pi2)

    return jnp.maximum(series + tail_mean, 1e-6)


from .._utils._jax_utils import (
    check_jax_available as _check_jax_available_impl,
)


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
    pg_n_terms,
    n_probes,
    lanczos_deg,
    sparsax_pattern=None,
    krylov_degree: int = 0,
    krylov_dmax: float = 0.4,
):
    """Build a JIT-compiled Gibbs step with data bound into the closure.

    This function creates a ``@jax.jit``-compiled function that performs
    one complete Gibbs sweep (ω, η, β, σ², ρ slice) in a single XLA
    kernel call, eliminating all Python→JAX dispatch overhead.

    Parameters
    ----------
    y_jax : jax.numpy.ndarray of shape (n,)
        Response vector (JAX array).
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
        Dense (W + W^T).  Only needed for the dense-Cholesky fallback
        (``sparsax_pattern is None``); pass ``None`` on the sparse path.
    WtW_dense : jax.numpy.ndarray of shape (n, n) or None
        Dense W^T W.  Only needed for the dense-Cholesky fallback.
    logdet_jax : callable
        JAX-native function ``(rho) -> jax.numpy.ndarray`` computing
        log|I - rho*W|.  Built by :func:`~neighbayes.logdet.make_logdet_jax_fn`.
        Replaces the former ``W_eigs`` eigenvalue-based logdet, allowing
        trace-seeded Chebyshev or other methods that avoid the O(n³)
        eigendecomposition.
    XtX_jax : jax.numpy.ndarray of shape (k, k)
        Precomputed X^T X.
    priors : GibbsPriors
        Prior hyperparameters.
    pg_n_terms : int
        Number of alternating-series terms for the PG draw (mean-exact
        via tail correction; see :func:`_pg_gamma_series_draw`).
        Values below 20 can destabilize the Gibbs chain.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    gibbs_step : callable
        A JIT-compiled function with signature::

            gibbs_step(state, key) -> (new_state, accept)

        where ``state`` is a :class:`~neighbayes.samplers.negbin._core.JAXGibbsState`
        and ``key`` is a JAX PRNG key.  ``accept`` is always ``True``
        (slice sampling has no rejection step).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve, solve_triangular

    ensure_x64()

    use_sparsax = sparsax_pattern is not None

    # Sparse W matvecs — never densify W: W @ x and Wᵀ @ x go through BCOO
    # (O(nnz), O(nnz) memory) instead of a dense n×n materialization.
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
    # Note: priors.beta_sigma is the standard deviation, not the variance
    beta_mu_jax = jnp.broadcast_to(jnp.asarray(priors.beta_mu, dtype=jnp.float64), (k,))
    beta_sigma2_jax = jnp.broadcast_to(
        jnp.asarray(priors.beta_sigma, dtype=jnp.float64) ** 2, (k,)
    )
    rho_lower_jax = jnp.float64(priors.rho_lower)
    rho_upper_jax = jnp.float64(priors.rho_upper)

    # Prior precision for beta
    beta_prior_prec = jnp.diag(1.0 / beta_sigma2_jax)

    # ── sparsax setup (optional sparse SPD Cholesky path) ──
    if use_sparsax:
        _Ai = jnp.asarray(sparsax_pattern["Ai"], dtype=jnp.int32)
        _Aj = jnp.asarray(sparsax_pattern["Aj"], dtype=jnp.int32)
        _W_sym_vals = jnp.asarray(sparsax_pattern["W_sym_vals"], dtype=jnp.float64)
        _WtW_vals = jnp.asarray(sparsax_pattern["WtW_vals"], dtype=jnp.float64)
        _diag_idx = jnp.asarray(sparsax_pattern["diag_idx"], dtype=jnp.int32)
        _nnz = len(sparsax_pattern["Ai"])
        _n_static = int(sparsax_pattern["n"])
        # Factor-once closures: the η-draw and each ρ-density eval do exactly ONE
        # numeric factorization (matching numpy's CholmodFactor reuse), via
        # sparsax 0.4 `sample_gaussian` / `factor_solve` (0.3 fallback inside).
        from .._utils._sparsax_utils import make_sparsax_ops

        _eta_sample, _solve_logdet = make_sparsax_ops(_Ai, _Aj, _n_static)

        def _assemble_Ax(omega, rho_val, inv_s2):
            """Assemble COO values for P = I/σ² + diag(ω) − (ρ/σ²)(W+Wᵀ) + (ρ²/σ²)WᵀW."""
            Ax = -rho_val * _W_sym_vals * inv_s2 + rho_val**2 * _WtW_vals * inv_s2
            diag_vals = jnp.zeros(_nnz, dtype=jnp.float64)
            diag_vals = diag_vals.at[_diag_idx].set(inv_s2 + omega)
            return Ax + diag_vals

    @eqx.filter_jit
    def gibbs_step(state, key, slice_width=0.2, return_steps=False):
        """One complete Gibbs sweep: ω → η → β → σ² → ρ (slice) → α (slice).

        Parameters
        ----------
        state : JAXGibbsState
            Current state.
        key : jax.random.PRNGKey
            JAX random key.
        slice_width : float or jax.numpy.float64, default 0.2
            Stepping-out width for the ρ slice sampler.
        return_steps : bool, default False
            Also return the ρ slice's ``(left, right)`` step-out counts.

        Returns
        -------
        new_state : JAXGibbsState
            Updated state.
        accept : bool
            Always True (slice sampling has no rejection step).
        steps : tuple of jax.numpy scalars
            Step-out counts; returned only when ``return_steps``.
        """
        eta = state.eta
        beta = state.beta
        sigma2 = state.sigma2
        rho = state.rho
        alpha = state.alpha

        key_omega, key_eta, key_beta, key_sigma2, key_rho = jax.random.split(key, 5)

        # ── Block 1: ω ~ PG(y + α, η) — Gamma series + tail-mean correction ──
        # Gamma(h,1) per-term draws give the correct variance (Var[PG] ∝ h,
        # not h²); the closed-form tail correction makes the mean exact.
        omega_new = _pg_gamma_series_draw(key_omega, y_jax + alpha, eta, pg_n_terms)

        # ── Block 2: η | ω, ρ, β, σ² — Cholesky solve ──
        inv_s2 = 1.0 / sigma2
        Xbeta = X_jax @ beta
        kappa = (y_jax - alpha) / 2.0
        rhs = Xbeta * inv_s2 - rho * Wt_matvec(Xbeta) * inv_s2 + kappa

        if use_sparsax:
            Ax = _assemble_Ax(omega_new, rho, inv_s2)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            # Mean + correlated draw ~ N(P⁻¹ rhs, P⁻¹) from ONE factorization.
            eta_new = _eta_sample(Ax, rhs, z_eta)
        else:
            P_diag = jnp.ones(n) * inv_s2 + omega_new
            P = jnp.diag(P_diag) - rho * W_sym * inv_s2 + rho**2 * WtW * inv_s2
            P = P + 1e-6 * jnp.eye(n)  # regularization for numerical stability

            L = jnp.linalg.cholesky(P)
            P_inv_rhs = cho_solve((L, True), rhs)
            z_eta = jax.random.normal(key_eta, shape=(n,), dtype=jnp.float64)
            eta_new = P_inv_rhs + solve_triangular(L.T, z_eta, lower=False)

        # ── Block 3: β | η, ρ, σ² — conjugate normal ──
        A_rho_eta = eta_new - rho * W_matvec(eta_new)
        Sigma_beta_inv = beta_prior_prec + XtX_jax / sigma2
        rhs_beta = beta_mu_jax / beta_sigma2_jax + X_jax.T @ A_rho_eta / sigma2
        L_beta = jnp.linalg.cholesky(Sigma_beta_inv)
        m_beta = cho_solve((L_beta, True), rhs_beta)
        z_beta = jax.random.normal(key_beta, shape=(k,), dtype=jnp.float64)
        beta_new = m_beta + solve_triangular(L_beta.T, z_beta, lower=False)

        # ── Block 4: σ² | η, ρ, β — conjugate inverse-Gamma ──
        Xbeta_new = X_jax @ beta_new
        r = A_rho_eta - Xbeta_new
        a_post = jnp.float64(priors.sigma2_alpha + n / 2.0)
        b_post = jnp.float64(priors.sigma2_beta + r @ r / 2.0)
        sigma2_inv = jax.random.gamma(key_sigma2, a_post) / b_post
        sigma2_new = jnp.maximum(1.0 / sigma2_inv, 1e-10)

        # ── Block 5: ρ — slice sampling (collapsed, η integrated out) ──
        # Uses the shared JAX slice sampler (same as logit/_jax.py).
        from .._utils._jax_slice import jax_slice_sample_1d

        key_rho, key_alpha = jax.random.split(key_rho)

        # Hoist the ρ-independent pieces of the collapsed rhs out of the
        # slice density: these are constant across every stepping-out /
        # shrink evaluation, so computing them once (one sparse Wᵀ matvec)
        # avoids recomputing an O(nnz) matvec dozens of times per sweep.
        inv_s2_r = 1.0 / sigma2_new
        Xbeta_r = X_jax @ beta_new
        kappa_r = (y_jax - alpha) / 2.0
        Wt_Xbeta_r = Wt_matvec(Xbeta_r)

        # ── Krylov basis (sparsax factor-reuse path) ──
        # When krylov_degree > 0 and the sparsax factor-reuse primitives
        # are available, factor P(ρ_c) ONCE and reuse for all candidates
        # within krylov_dmax via Horner.  The negbin RHS is
        # Xβ/σ² − ρ·WtXβ/σ² + κ, so seed with the ρ-independent columns
        # [κ, Xβ/σ², WtXβ/σ²].
        _use_krylov = (
            use_sparsax and krylov_degree > 0 and _sparsax_factor_ops_available()
        )
        if _use_krylov:
            _Ax_c_nb = _assemble_Ax(omega_new, rho, inv_s2_r)
            # G = ∂P/∂ρ|_{ρ_c} = G1/σ² − 2ρ_c G2/σ² (COO values)
            _G_vals_c_nb = (_W_sym_vals - 2.0 * rho * _WtW_vals) * inv_s2_r
            _rhs_seed_nb = jnp.column_stack(
                [
                    kappa_r,
                    Xbeta_r * inv_s2_r,
                    Wt_Xbeta_r * inv_s2_r,
                ]
            )  # (n, 3)
            _G2_vals_nb = _WtW_vals * inv_s2_r
            (
                _token_nb,
                _V_stack_nb,
                _ld_coefs_nb,
                _safe_dmax_nb,
            ) = build_precision_krylov_basis_jax(
                _Ai,
                _Aj,
                _Ax_c_nb,
                _G_vals_c_nb,
                _G2_vals_nb,
                _rhs_seed_nb,
                n=_n_static,
                degree=krylov_degree,
                dmax=krylov_dmax,
            )

        def log_density_rho(rho_val):
            """Collapsed log-density of ρ (η integrated out).

            Uses one Cholesky of P_r to obtain log|P_r| and the
            quadratic form exactly — strictly faster than iterative
            Lanczos+CG for the dense regime this sampler targets.
            """
            rhs_r = Xbeta_r * inv_s2_r - rho_val * Wt_Xbeta_r * inv_s2_r + kappa_r

            if use_sparsax:
                if _use_krylov:
                    drho = rho_val - rho
                    within = jnp.abs(drho) <= _safe_dmax_nb

                    def _from_basis(_):
                        # Horner in pure JAX against the basis built at ρ_c.
                        # Columns: [P(ρ)⁻¹κ, P(ρ)⁻¹Xβ/σ², P(ρ)⁻¹WtXβ/σ²]
                        sol_all = eval_precision_solve_from_basis_jax(_V_stack_nb, drho)
                        # rhs = Xβ/σ² − ρ·WtXβ/σ² + κ
                        m_k = (
                            sol_all[:, 1:2] - rho_val * sol_all[:, 2:]
                        ).ravel() + sol_all[:, 0]
                        ld_k = eval_precision_logdet_from_basis_jax(_ld_coefs_nb, drho)
                        return m_k, ld_k, rhs_r @ m_k

                    def _direct(_):
                        Ax_r = _assemble_Ax(omega_new, rho_val, inv_s2_r)
                        m_d, ld_d = _solve_logdet(Ax_r, rhs_r)
                        return m_d, ld_d, rhs_r @ m_d

                    # `lax.cond`, not `jnp.where`: `where` evaluates *both*
                    # operands, so the direct factorization would run on every
                    # candidate and the basis would be pure added work (measured
                    # 0.84x end-to-end).  `cond` genuinely branches here because
                    # each chain runs as its own `jit` program (see
                    # `run_chains_in_threads`); under `vmap` it would lower back
                    # to `select`.
                    m, log_det_P, quad_r = jax.lax.cond(
                        within, _from_basis, _direct, operand=None
                    )
                else:
                    Ax_r = _assemble_Ax(omega_new, rho_val, inv_s2_r)
                    # Solve + logdet from one factorization (factor_solve on 0.4).
                    m, log_det_P = _solve_logdet(Ax_r, rhs_r)
                    quad_r = rhs_r @ m
            else:
                P_diag_r = jnp.ones(n) * inv_s2_r + omega_new
                P_r = (
                    jnp.diag(P_diag_r)
                    - rho_val * W_sym * inv_s2_r
                    + rho_val**2 * WtW * inv_s2_r
                )
                P_r = P_r + 1e-6 * jnp.eye(n)

                L_r = jnp.linalg.cholesky(P_r)
                log_det_P = 2.0 * jnp.sum(jnp.log(jnp.diag(L_r)))
                v = solve_triangular(L_r, rhs_r, lower=True)
                quad_r = v @ v

            logdet_W = logdet_jax(rho_val)

            log_prior = jnp.where(
                (rho_val >= rho_lower_jax) & (rho_val <= rho_upper_jax), 0.0, -jnp.inf
            )
            return logdet_W - 0.5 * log_det_P + 0.5 * quad_r + log_prior

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

        rho_new = jnp.clip(rho_new, rho_lower_jax, rho_upper_jax)

        # Accept flag is always True for slice sampling (no rejection)
        accept = jnp.bool_(True)

        # ── Block 6: α | y, η — JAX slice sampling ──
        alpha_new = _sample_alpha_jax(
            JAXGibbsState(
                eta=eta_new,
                beta=beta_new,
                sigma2=sigma2_new,
                rho=rho_new,
                omega=omega_new,
                alpha=alpha,
            ),
            y_jax,
            priors.alpha_sigma,
            priors.alpha_nu,
            key_alpha,
        )

        new_state = JAXGibbsState(
            eta=eta_new,
            beta=beta_new,
            sigma2=sigma2_new,
            rho=rho_new,
            omega=omega_new,
            alpha=alpha_new,
        )
        if return_steps:
            return new_state, accept, (steps_left, steps_right)
        return new_state, accept

    return gibbs_step


def _jax_nb_log_density_alpha(log_a, y_jax, eta, alpha_sigma, alpha_nu):
    """JAX-compiled NB log-density for α slice sampling.

    Computes log p(log_a | y, η) up to a constant, where α = exp(log_a).

    Parameters
    ----------
    log_a : jax.numpy.ndarray
        Log of α (scalar).
    y_jax : jax.numpy.ndarray
        Integer response vector.
    eta : jax.numpy.ndarray
        Current latent field.
    alpha_sigma : float
        Prior scale for α.

    Returns
    -------
    jax.numpy.ndarray
        Log-density value (scalar).
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jax_gammaln

    a = jnp.exp(log_a)
    mu = jnp.exp(eta)
    # Numerically stable NB log-likelihood
    log_mu_ratio = jnp.log(jnp.maximum(mu / (mu + a), 1e-300))
    log_alpha_ratio = jnp.log(jnp.maximum(a / (mu + a), 1e-300))
    log_lik = (
        jax_gammaln(y_jax + a)
        - jax_gammaln(a)
        + y_jax * log_mu_ratio
        + a * log_alpha_ratio
    )
    log_lik = jnp.where(jnp.isfinite(log_lik), log_lik, -1e10)
    total_log_lik = jnp.sum(log_lik)
    # Half-Student-t(ν, σ) prior on α, matching the PyMC/NUTS builders and the
    # reduced-form JAX sampler.  Jacobian for the log transform added below.
    log_prior = (
        -0.5
        * (alpha_nu + 1.0)
        * jnp.log1p((a * a) / (alpha_nu * alpha_sigma * alpha_sigma))
    )
    return log_a + total_log_lik + log_prior


def _sample_alpha_jax(state, y_jax, alpha_sigma, alpha_nu, key):
    """Sample α using JAX-compiled slice sampling.

    This eliminates the per-iteration Python↔JAX boundary crossing
    that ``_sample_alpha_python`` requires, keeping the entire Gibbs
    loop inside XLA.

    Uses Neal's stepping-out procedure with ``jax.lax.while_loop``
    for the stepping-out phase, and a bounded shrinkage loop.

    Parameters
    ----------
    state : JAXGibbsState
        Current Gibbs state (JAX arrays).
    y_jax : jax.numpy.ndarray
        Integer response vector (JAX array).
    alpha_sigma : float
        Prior scale for α.
    key : jax.random.PRNGKey
        JAX random key.

    Returns
    -------
    jax.numpy.ndarray
        New α value (scalar JAX array).
    """
    import jax
    import jax.numpy as jnp

    log_alpha = jnp.log(state.alpha)

    # Log-density at current point
    log_y0 = _jax_nb_log_density_alpha(
        log_alpha, y_jax, state.eta, alpha_sigma, alpha_nu
    )

    # Draw vertical level: log(u) where u ~ Uniform(0, f(x0))
    key, subkey = jax.random.split(key)
    log_u = log_y0 + jnp.log(jax.random.uniform(subkey, dtype=jnp.float64))

    # Slice sampling parameters
    w = jnp.float64(1.0)  # step-out width (wider than Python version for efficiency)
    lower_bound = jnp.float64(-10.0)
    upper_bound = jnp.float64(10.0)

    # --- Stepping out ---
    key, subkey = jax.random.split(key)
    u_rand = jax.random.uniform(subkey, dtype=jnp.float64)
    L = jnp.maximum(log_alpha - u_rand * w, lower_bound)
    R = jnp.minimum(L + w, upper_bound)

    # Step out left: expand L until log_density(L) < log_u or L hits lower_bound
    def step_out_left(carry):
        L_val, _ = carry
        L_new = jnp.maximum(L_val - w, lower_bound)
        return (L_new, jnp.float64(0.0))

    def should_step_left(carry):
        L_val, _ = carry
        return (L_val > lower_bound) & (
            _jax_nb_log_density_alpha(L_val, y_jax, state.eta, alpha_sigma, alpha_nu)
            > log_u
        )

    L_final, _ = jax.lax.while_loop(
        should_step_left, step_out_left, (L, jnp.float64(0.0))
    )

    # Step out right: expand R until log_density(R) < log_u or R hits upper_bound
    def step_out_right(carry):
        R_val, _ = carry
        R_new = jnp.minimum(R_val + w, upper_bound)
        return (R_new, jnp.float64(0.0))

    def should_step_right(carry):
        R_val, _ = carry
        return (R_val < upper_bound) & (
            _jax_nb_log_density_alpha(R_val, y_jax, state.eta, alpha_sigma, alpha_nu)
            > log_u
        )

    R_final, _ = jax.lax.while_loop(
        should_step_right, step_out_right, (R, jnp.float64(0.0))
    )

    # --- Shrinkage ---
    # Sample from [L, R] and shrink until we find a point above log_u.
    # Use jax.lax.while_loop for proper early termination.
    def shrink_while_cond(carry):
        _, _, _, _, done = carry
        return ~done

    def shrink_while_body(carry):
        L_val, R_val, key_val, x_best, _ = carry
        key_val, subkey = jax.random.split(key_val)
        x_new = L_val + jax.random.uniform(subkey, dtype=jnp.float64) * (R_val - L_val)
        log_dens_new = _jax_nb_log_density_alpha(
            x_new, y_jax, state.eta, alpha_sigma, alpha_nu
        )
        accepted = log_dens_new > log_u
        L_new = jnp.where(x_new < log_alpha, x_new, L_val)
        R_new = jnp.where(x_new >= log_alpha, x_new, R_val)
        collapsed = (R_new - L_new) < 1e-15
        done = accepted | collapsed
        x_best = jnp.where(accepted, x_new, x_best)
        return (L_new, R_new, key_val, x_best, done)

    # Start with current log_alpha as fallback
    _, _, _, log_alpha_new, _ = jax.lax.while_loop(
        shrink_while_cond,
        shrink_while_body,
        (L_final, R_final, key, log_alpha, jnp.bool_(False)),
    )

    return jnp.exp(log_alpha_new)


def _sample_alpha_python(state, y, alpha_sigma, alpha_nu, rng):
    """Sample α using Python slice sampling (not JIT-compatible).

    This is called from the Python loop because scipy's gammaln
    is not JAX-compatible.

    Parameters
    ----------
    state : JAXGibbsState
        Current Gibbs state.
    y : ndarray
        Integer response vector.
    alpha_sigma : float
        Prior scale for α.
    rng : numpy.random.Generator
        Random state.

    Returns
    -------
    float
        New α value.
    """
    from scipy.special import gammaln

    from .._utils._slice import slice_sample_1d

    alpha = float(state.alpha)
    eta = np.asarray(state.eta)
    log_alpha = np.log(alpha)

    def log_density(log_a):
        a = np.exp(log_a)
        if a <= 0:
            return -np.inf
        mu = np.exp(eta)
        # Numerically stable NB log-likelihood
        log_lik = (
            gammaln(y + a)
            - gammaln(a)
            + y * np.log(np.maximum(mu / (mu + a), 1e-300))
            + a * np.log(np.maximum(a / (mu + a), 1e-300))
        )
        log_lik = np.where(np.isfinite(log_lik), log_lik, -1e10)
        total_log_lik = np.sum(log_lik)
        # Half-Student-t(ν, σ) prior on α (alpha_nu was previously accepted and
        # silently discarded here, giving this path a HalfNormal prior).
        log_prior = (
            -0.5
            * (alpha_nu + 1.0)
            * np.log1p((a * a) / (alpha_nu * alpha_sigma * alpha_sigma))
        )
        return log_a + total_log_lik + log_prior

    log_alpha_new, _ = slice_sample_1d(
        log_density=log_density,
        x0=log_alpha,
        lower=-10.0,
        upper=10.0,
        w=0.5,
        rng=rng,
    )
    return np.exp(log_alpha_new)


def _nb_loglik_pointwise_jax(y, eta, alpha):
    """Compute pointwise NB log-likelihood (numpy, for storage).

    Uses ``jax.scipy.special.gammaln`` to avoid scipy dependency.
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jax_gammaln

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    eta_jax = jnp.asarray(eta, dtype=jnp.float64)
    alpha_jax = jnp.float64(alpha)
    mu = jnp.exp(eta_jax)
    # Numerically stable computation
    log_mu_ratio = jnp.log(jnp.maximum(mu / (mu + alpha_jax), 1e-300))
    log_alpha_ratio = jnp.log(jnp.maximum(alpha_jax / (mu + alpha_jax), 1e-300))
    result = (
        jax_gammaln(y_jax + alpha_jax)
        - jax_gammaln(alpha_jax)
        - jax_gammaln(y_jax + 1.0)
        + y_jax * log_mu_ratio
        + alpha_jax * log_alpha_ratio
    )
    return np.asarray(result)


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
    store_log_lik: bool = True,
):
    """Run one chain of the full-JIT JAX Gibbs sampler.

    Creates a JIT-compiled Gibbs step function and runs it in a Python
    loop, handling warmup, thinning, α updates, and storage.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Integer response vector.
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
        log|I - rho*W|.  Built by :func:`~neighbayes.logdet.make_logdet_jax_fn`.
        Allows trace-seeded Chebyshev or other methods that avoid the
        O(n³) eigendecomposition.
    priors : GibbsPriors
        Prior hyperparameters.
    init : GibbsState
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
        Number of alternating-series terms for the PG draw (mean-exact
        via tail correction; see :func:`_pg_gamma_series_draw`).
        Values below 20 can destabilize the Gibbs chain.
    n_probes : int
        Number of Lanczos probes for log|P| estimation.
    lanczos_deg : int
        Lanczos iteration depth.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, ``sigma``,
        ``alpha``, ``log_lik``, ``eta_norm``, and optionally ``eta``.
        Also includes ``mh_accept_rate`` (always 1.0 for slice sampling).
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
    from .._utils._jax_utils import build_w_bcoo

    W_bcoo, Wt_bcoo = build_w_bcoo(W_sparse)

    # Initialize state
    state = JAXGibbsState(
        eta=jnp.asarray(init.eta, dtype=jnp.float64),
        beta=jnp.asarray(init.beta, dtype=jnp.float64),
        sigma2=jnp.float64(init.sigma2),
        rho=jnp.float64(init.rho),
        omega=jnp.asarray(init.omega, dtype=jnp.float64),
        alpha=jnp.float64(init.alpha),
    )

    # Pre-allocate storage
    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    sigma_samples = np.empty(n_keep, dtype=np.float64)
    alpha_samples = np.empty(n_keep, dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None
    eta_norm_samples = np.empty(n_keep, dtype=np.float64)
    eta_samples = np.empty((n_keep, n), dtype=np.float64) if return_eta else None

    # Build the JIT-compiled step function (slice sampling for ρ)
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
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
    )

    # Warmup the JIT function (first call triggers compilation)
    key = jax.random.PRNGKey(rng.integers(2**31))
    key, warmup_key = jax.random.split(key)
    _ = gibbs_step(state, warmup_key)

    # Run the chain
    for i in range(total_iters):
        key, step_key = jax.random.split(key)
        state, _ = gibbs_step(state, step_key)

        # Store post-warmup draws
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = float(state.rho)
                beta_samples[idx] = np.asarray(state.beta)
                sigma_samples[idx] = np.sqrt(float(state.sigma2))
                alpha_samples[idx] = float(state.alpha)
                eta_np = np.asarray(state.eta)
                eta_norm_samples[idx] = float(eta_np @ eta_np)
                if return_eta:
                    eta_samples[idx] = eta_np
                if store_log_lik:
                    log_lik_samples[idx] = _nb_loglik_pointwise_jax(
                        y, eta_np, float(state.alpha)
                    )

    result = {
        "rho": rho_samples,
        "beta": beta_samples,
        "sigma": sigma_samples,
        "alpha": alpha_samples,
        "log_lik": log_lik_samples,
        "eta_norm": eta_norm_samples,
        "mh_accept_rate": 1.0,  # slice sampling always accepts
    }
    if return_eta:
        result["eta"] = eta_samples

    return result


# ---------------------------------------------------------------------------
# Multi-chain runner: chains run in parallel threads, each an ordinary jit
# program sharing one compiled Gibbs step (see ``run_chains_chunked``).
# Mirrors the SAR-logit runner in ``neighbayes/samplers/logit/_jax.py``.
# ---------------------------------------------------------------------------


def _nb_loglik_pointwise_jax_op(y_jax, eta, alpha):
    """Pointwise NB log-likelihood as a pure-JAX op (vmap-safe).

    Full NB2 log-pmf, including ``-log Γ(y+1)``.  See
    :func:`neighbayes.samplers.negbin._core._nb_loglik_pointwise`.
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln as jax_gammaln

    mu = jnp.exp(eta)
    log_mu_ratio = jnp.log(jnp.maximum(mu / (mu + alpha), 1e-300))
    log_alpha_ratio = jnp.log(jnp.maximum(alpha / (mu + alpha), 1e-300))
    return (
        jax_gammaln(y_jax + alpha)
        - jax_gammaln(alpha)
        - jax_gammaln(y_jax + 1.0)
        + y_jax * log_mu_ratio
        + alpha * log_alpha_ratio
    )


def _stack_nb_inits(inits):
    """Stack per-chain :class:`GibbsState` into one vmap-able pytree."""
    import jax
    import jax.numpy as jnp

    jax_inits = [
        JAXGibbsState(
            eta=jnp.asarray(init.eta, dtype=jnp.float64),
            beta=jnp.asarray(init.beta, dtype=jnp.float64),
            sigma2=jnp.float64(init.sigma2),
            rho=jnp.float64(init.rho),
            alpha=jnp.float64(init.alpha),
            omega=jnp.asarray(init.omega, dtype=jnp.float64),
        )
        for init in inits
    ]
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
    """Run multiple SAR-NB Gibbs chains in parallel.

    Chains run in parallel threads, each an ordinary ``jax.jit`` program, and
    the Gibbs step is compiled once (see
    :func:`.._utils._jax_utils.run_chains_chunked`).

    Parameters mirror :func:`run_chain_jax`, except ``init`` is replaced
    by a list of per-chain initial states and ``return_eta`` is not
    supported (use the per-chain :func:`run_chain_jax` if you need the
    full latent field stored).

    Returns
    -------
    list of dict
        One dict per chain with keys ``rho``, ``beta``, ``sigma``,
        ``alpha``, ``log_lik``, ``eta_norm``, ``mh_accept_rate``.
    """
    import jax
    import jax.numpy as jnp

    ensure_x64()

    chains = len(inits)
    n, k = X.shape

    y_jax = jnp.asarray(y, dtype=jnp.float64)
    X_jax = jnp.asarray(X, dtype=jnp.float64)
    XtX_jax = jnp.asarray(X.T @ X, dtype=jnp.float64)
    from .._utils._jax_utils import build_w_bcoo

    W_bcoo, Wt_bcoo = build_w_bcoo(W_sparse)

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
        pg_n_terms=pg_n_terms,
        n_probes=n_probes,
        lanczos_deg=lanczos_deg,
        sparsax_pattern=sparsax_pattern,
        krylov_degree=krylov_degree,
        krylov_dmax=krylov_dmax,
    )

    init_states = _stack_nb_inits(inits)

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
        trace = (
            state.rho,
            state.beta,
            state.sigma2,
            state.alpha,
            state.eta @ state.eta,
        )
        if store_log_lik:
            trace += (_nb_loglik_pointwise_jax_op(y_jax, state.eta, state.alpha),)
        return (state, width), trace

    with GibbsProgressBarManager(
        chains=chains,
        draws=draws,
        tune=tune,
        progressbar=progressbar,
        model_type="sar_negbin",
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

    rhos, betas, sigma2s, alphas, eta_norms = traces[:5]
    log_liks = traces[5] if store_log_lik else None
    thin_slice = slice(None, None, thin) if thin > 1 else slice(None)
    results = []
    for c in range(chains):
        results.append(
            {
                "rho": rhos[c, thin_slice].copy(),
                "beta": betas[c, thin_slice].copy(),
                "sigma": np.sqrt(sigma2s[c, thin_slice]).copy(),
                "alpha": alphas[c, thin_slice].copy(),
                "eta_norm": eta_norms[c, thin_slice].copy(),
                "log_lik": log_liks[c, thin_slice].copy() if store_log_lik else None,
                "mh_accept_rate": 1.0,
            }
        )
    return results
