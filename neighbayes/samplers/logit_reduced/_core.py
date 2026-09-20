r"""NumPy reduced-form SAR-logit Pólya–Gamma Gibbs sampler.

Binary analogue of the reduced-form SAR Negative-Binomial sampler
(:mod:`..negbin_reduced._core`): the spatial lag enters the *linear predictor*,

.. math::

    \eta = (I - \rho W)^{-1} X\beta, \qquad y_i \sim \mathrm{Bernoulli}(\sigma(\eta_i)),

so — exactly as in the reduced-form count model — the :math:`|I - \rho W|`
Jacobian cancels when :math:`\beta` is marginalized out and the ρ conditional is
Krylov-accelerable.  This module reuses the reduced-NB machinery almost verbatim
(the shift-invert Krylov basis, the CHOLMOD normal-equations solver, and the
adaptive slice sampler), differing only in the Pólya–Gamma augmentation:

* the shape is **h = 1** (Bernoulli) instead of :math:`y + \alpha`;
* the working response is :math:`s = \kappa/\omega` with :math:`\kappa = y - \tfrac12`
  (no ``log α`` offset);
* there is **no dispersion parameter α** (hence no α slice).

Contrast with :mod:`..logit` (structural latent-field SAR/SEM-logit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._spatial_normal import CholmodFactor
from ..negbin_reduced._core import (
    _KRYLOV_DMAX_DEFAULT,
    ReducedGibbsCache,
    ReducedKrylovBasis,
    _build_krylov_basis,
    _CholmodNormalEqSolver,
    _eval_U_from_basis,
    _make_solver,
    _prior_precision_and_mean,
    make_sar_solver,
)


@dataclass
class ReducedLogitGibbsState:
    """Mutable state for one reduced-form SAR-logit Gibbs chain.

    Attributes
    ----------
    beta : ndarray, shape (k,)
        Regression coefficients.
    rho : float
        Spatial autoregressive parameter.
    omega : ndarray, shape (n,)
        Pólya–Gamma auxiliary variables (PG(1, η)).
    """

    beta: np.ndarray
    rho: float
    omega: np.ndarray


# The pointwise Bernoulli-logit log-likelihood is shared with the structural
# sampler — re-exported here so this module's own callers (and its tests) keep
# importing it from where they always have.
from ..logit._core import _logit_loglik_pointwise  # noqa: E402


def _sample_omega(eta: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Block 1: draw :math:`\\omega \\sim \\mathrm{PG}(1, \\eta)` (Bernoulli)."""
    return sample_polyagamma(np.ones_like(eta), np.clip(eta, -20.0, 20.0), rng=rng)


def _rho_log_density_marginal(
    rho: float,
    omega: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    W_csc: sp.csc_matrix,
    n: int,
    V0_inv_diag: np.ndarray,
    mu0: np.ndarray,
    rho_lower: float,
    rho_upper: float,
    *,
    basis: Optional[ReducedKrylovBasis] = None,
    krylov_dmax: float = _KRYLOV_DMAX_DEFAULT,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
    intercept_col: int = 0,
) -> float:
    r"""β-marginalized conditional log-density of ρ for the reduced logit.

    Identical Gaussian core to :func:`..negbin_reduced._core._rho_log_density_marginal`,
    with the Bernoulli working response :math:`s = \kappa/\omega`
    (:math:`\kappa = y - \tfrac12`) and no ``log α`` offset.  The
    ``log|I − ρW|`` Jacobian cancels under β-marginalization, so the density is
    a plain Gaussian normalizing constant.  Candidates outside the Krylov radius
    fall back to a direct factorization of ``A_rho``.
    """
    if rho <= rho_lower or rho >= rho_upper:
        return -np.inf

    use_basis = (
        basis is not None
        and basis.degree > 0
        and abs(rho - basis.rho_basis) <= min(krylov_dmax, basis.safe_dmax)
    )
    if use_basis:
        U = _eval_U_from_basis(basis, rho - basis.rho_basis)
    else:
        try:
            # Outside the Krylov radius: factor A_rho at this candidate rather
            # than running a Chebyshev/CG solve off W's spectral bounds.
            U = _make_solver(rho, W_csc, n, cholmod_solver=cholmod_solver).solve(X)
        except (RuntimeError, ValueError):
            return -np.inf

    # --- Intercept reparameterization: δ₀ = β₀/(1−ρ) ---
    reparam = intercept_col >= 0 and abs(rho) > 1e-8
    if reparam:
        scale = 1.0 - rho
        U = U.copy()
        U[:, intercept_col] = 1.0
        V0_inv_diag_rp = V0_inv_diag.copy()
        V0_inv_diag_rp[intercept_col] = V0_inv_diag[intercept_col] * scale * scale
        mu0_rp = mu0.copy()
        mu0_rp[intercept_col] = mu0[intercept_col] / scale
    else:
        V0_inv_diag_rp = V0_inv_diag
        mu0_rp = mu0

    # Bernoulli working response: κ = y − ½, s = κ/ω (no log-α offset).
    s = (y - 0.5) / omega
    r = s - U @ mu0_rp

    Uw = U * omega[:, None]
    M = U.T @ Uw
    k = M.shape[0]
    M.flat[:: k + 1] += V0_inv_diag_rp
    v = Uw.T @ r

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return -np.inf

    w = solve_triangular(L, v, lower=True)
    quad_pen = float(w @ w)
    rOr = float(np.dot(r, omega * r))
    log_det_M = 2.0 * float(np.sum(np.log(np.diag(L))))

    result = -0.5 * log_det_M - 0.5 * (rOr - quad_pen)
    if reparam:
        result += np.log(scale)
    if not np.isfinite(result):
        return -np.inf
    return result


def _sample_beta(
    beta_current: np.ndarray,
    Xtilde: np.ndarray,
    omega: np.ndarray,
    y: np.ndarray,
    priors,
    *,
    rng: np.random.Generator,
    rho: float = 0.0,
    intercept_col: int = 0,
) -> np.ndarray:
    r"""Block 2: conjugate Gaussian draw for :math:`\beta` (Bernoulli).

    Same posterior as the reduced-NB β draw with :math:`\kappa = y - \tfrac12`
    and no ``ω log α`` term:
    :math:`\Sigma_\beta^{-1} = \tilde X^\top \Omega \tilde X + V_0^{-1}`,
    :math:`m_\beta = \Sigma_\beta(\tilde X^\top \kappa + V_0^{-1}\mu_0)`.
    """
    k = Xtilde.shape[1]
    kappa = y - 0.5
    V0_inv_diag, mu0, _ = _prior_precision_and_mean(priors, k)

    reparam = intercept_col >= 0 and abs(rho) > 1e-8
    if reparam:
        scale = 1.0 - rho
        Xtilde_rp = Xtilde.copy()
        Xtilde_rp[:, intercept_col] = 1.0
        V0_inv_diag_rp = V0_inv_diag.copy()
        V0_inv_diag_rp[intercept_col] = V0_inv_diag[intercept_col] * scale * scale
        mu0_rp = mu0.copy()
        mu0_rp[intercept_col] = mu0[intercept_col] / scale
    else:
        Xtilde_rp = Xtilde
        V0_inv_diag_rp = V0_inv_diag
        mu0_rp = mu0

    Xt_omega = Xtilde_rp * omega[:, None]
    Sigma_beta_inv = Xt_omega.T @ Xtilde_rp
    Sigma_beta_inv.flat[:: k + 1] += V0_inv_diag_rp
    rhs = Xtilde_rp.T @ kappa + V0_inv_diag_rp * mu0_rp

    try:
        L = np.linalg.cholesky(Sigma_beta_inv)
    except np.linalg.LinAlgError:
        Sigma_beta_inv.flat[:: k + 1] += 1e-10
        try:
            L = np.linalg.cholesky(Sigma_beta_inv)
        except np.linalg.LinAlgError:
            return beta_current

    w = solve_triangular(L, rhs, lower=True)
    m_beta = solve_triangular(L.T, w, lower=False)
    z = rng.standard_normal(k)
    delta = solve_triangular(L.T, z, lower=False)
    result = m_beta + delta
    if reparam:
        result[intercept_col] *= scale
    return result


def _sample_rho(
    state: ReducedLogitGibbsState,
    cache: ReducedGibbsCache,
    y: np.ndarray,
    X: np.ndarray,
    priors,
    *,
    rng: np.random.Generator,
    sweep_idx: int,
    tune: int,
    basis: Optional[ReducedKrylovBasis] = None,
    cholmod_solver: Optional[_CholmodNormalEqSolver] = None,
    intercept_col: int = 0,
) -> tuple[float, float]:
    """Block 3: 1-D adaptive slice on ρ with β marginalized."""
    n, k = X.shape
    rho_lower = cache.rho_lower
    rho_upper = cache.rho_upper
    V0_inv_diag, mu0, _ = _prior_precision_and_mean(priors, k)

    def log_density(rho: float) -> float:
        return _rho_log_density_marginal(
            rho=rho,
            omega=state.omega,
            y=y,
            X=X,
            W_csc=cache.W_csc,
            n=n,
            V0_inv_diag=V0_inv_diag,
            mu0=mu0,
            rho_lower=rho_lower,
            rho_upper=rho_upper,
            basis=basis,
            krylov_dmax=cache.krylov_dmax,
            cholmod_solver=cholmod_solver,
            intercept_col=intercept_col,
        )

    if cache.rho_adaptive_width and cache.rho_slice_width_state is not None:
        width_state = cache.rho_slice_width_state
        log_dens_x0 = log_density(state.rho)
        rho_new, log_density_new, steps_left, steps_right = slice_sample_1d_adaptive(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            width_state=width_state,
            rng=rng,
            log_density_x0=log_dens_x0,
        )
        if sweep_idx < tune:
            update_slice_width(width_state, steps_left, steps_right)
    else:
        rho_new, log_density_new = slice_sample_1d(
            log_density=log_density,
            x0=state.rho,
            lower=rho_lower,
            upper=rho_upper,
            w=0.2,
            rng=rng,
        )
    return rho_new, log_density_new


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors,
    cache: ReducedGibbsCache,
    init: ReducedLogitGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = True,
) -> dict[str, np.ndarray]:
    """Run one chain of the reduced-form SAR-logit PG-Gibbs sampler.

    Returns
    -------
    dict[str, np.ndarray]
        Posterior samples with keys ``rho``, ``beta``, and pointwise
        ``log_lik`` (each indexed by post-warmup draw).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, k = X.shape
    total_iters = tune + draws
    n_keep = draws // thin if thin > 0 else draws

    rho_samples = np.empty(n_keep, dtype=np.float64)
    beta_samples = np.empty((n_keep, k), dtype=np.float64)
    log_lik_samples = np.empty((n_keep, n), dtype=np.float64) if store_log_lik else None

    state = ReducedLogitGibbsState(
        beta=np.asarray(init.beta, dtype=np.float64).copy(),
        rho=float(init.rho),
        omega=np.asarray(init.omega, dtype=np.float64).copy(),
    )

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    use_krylov = cache.krylov_degree > 0
    krylov_degree = cache.krylov_degree

    # Detect intercept column for the δ₀ = β₀/(1−ρ) reparameterization.
    intercept_col = -1
    for _j in range(k):
        if np.all(X[:, _j] == 1.0):
            intercept_col = _j
            break

    # Build the CHOLMOD normal-equations solver once per chain (worker-side).
    cholmod_solver: _CholmodNormalEqSolver | None = None
    if (
        cache.cholmod_pattern is not None
        and cache.W_sym is not None
        and cache.WtW is not None
    ):
        cholmod_factor = CholmodFactor(cache.cholmod_pattern)
        cholmod_solver = make_sar_solver(
            cholmod_factor=cholmod_factor,
            W_csc=cache.W_csc,
            W_sym=cache.W_sym,
            WtW=cache.WtW,
            n=n,
        )

    _n_cycles = cache.n_rho_omega_cycles

    # Per-chain Krylov basis cache for reuse across sweeps.
    _prev_basis = None
    _prev_rho = None

    for i in range(total_iters):
        # --- Build Krylov basis at current ρ (or factorize for legacy) ---
        if use_krylov:
            if (
                cache.krylov_reuse
                and _prev_basis is not None
                and abs(state.rho - _prev_rho) < cache.krylov_reuse_threshold
            ):
                basis = _prev_basis
            else:
                try:
                    basis = _build_krylov_basis(
                        state.rho,
                        X,
                        cache.W_csc,
                        n,
                        degree=krylov_degree,
                        cholmod_solver=cholmod_solver,
                    )
                except (RuntimeError, ValueError):
                    state.rho = 0.0
                    basis = _build_krylov_basis(
                        0.0,
                        X,
                        cache.W_csc,
                        n,
                        degree=krylov_degree,
                        cholmod_solver=cholmod_solver,
                    )
                _prev_basis = basis
                _prev_rho = state.rho

            # η = U(ρ) @ β — use Horner when basis was reused at a different ρ.
            if abs(state.rho - basis.rho_basis) < 1e-12:
                eta = basis.V_stack[0] @ state.beta
            else:
                _drho_eta = state.rho - basis.rho_basis
                eta = _eval_U_from_basis(basis, _drho_eta) @ state.beta
        else:
            try:
                solver = _make_solver(
                    state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            except (RuntimeError, ValueError):
                state.rho = 0.0
                solver = _make_solver(
                    0.0, cache.W_csc, n, cholmod_solver=cholmod_solver
                )
            eta = solver.solve(X @ state.beta)
            basis = None

        # --- (ω, ρ, β) cycles ---
        state.omega = _sample_omega(eta, rng=rng)
        Xtilde = None

        for _cycle in range(_n_cycles):
            state.rho, _ = _sample_rho(
                state=state,
                cache=cache,
                y=y,
                X=X,
                priors=priors,
                rng=rng,
                sweep_idx=i,
                tune=tune,
                basis=basis,
                cholmod_solver=cholmod_solver,
                intercept_col=intercept_col,
            )

            # Xtilde = (I − ρW)⁻¹X at the new ρ (Krylov eval or direct solve).
            _lam_at_max = 1.0 - state.rho * cache.W_eig_max
            _lam_at_min = 1.0 - state.rho * cache.W_eig_min
            _lam_min = min(_lam_at_max, _lam_at_min)
            _lam_max = max(_lam_at_max, _lam_at_min)
            if _lam_min <= 0:
                state.rho = 0.0
                Xtilde = X.copy()
            elif basis is not None:
                drho = state.rho - basis.rho_basis
                if abs(drho) <= min(cache.krylov_dmax, basis.safe_dmax):
                    Xtilde = _eval_U_from_basis(basis, drho)
                else:
                    Xtilde = _make_solver(
                        state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                    ).solve(X)
            else:
                Xtilde = _make_solver(
                    state.rho, cache.W_csc, n, cholmod_solver=cholmod_solver
                ).solve(X)

            state.beta = _sample_beta(
                beta_current=state.beta,
                Xtilde=Xtilde,
                omega=state.omega,
                y=y,
                priors=priors,
                rng=rng,
                rho=state.rho,
                intercept_col=intercept_col,
            )

            eta = Xtilde @ state.beta

            if _cycle < _n_cycles - 1:
                state.omega = _sample_omega(eta, rng=rng)

        # --- Store post-warmup draw ---
        if i >= tune and (i - tune) % thin == 0:
            idx = (i - tune) // thin
            if idx < n_keep:
                rho_samples[idx] = state.rho
                beta_samples[idx] = state.beta
                if store_log_lik:
                    log_lik_samples[idx] = _logit_loglik_pointwise(y, eta)

        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)

    return {
        "rho": rho_samples,
        "beta": beta_samples,
        "log_lik": log_lik_samples,
    }
