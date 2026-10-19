"""Pólya–Gamma Gibbs sampler for the SAR and SEM logit with random effects.

SAR:  ``y ~ Bernoulli(logit⁻¹(η))``, ``η = ρWη + Xβ + ν``, ``ν ~ N(0, σ²I)``.
SEM:  ``η = Xβ + u``, ``u = λWu + ν``.

Both are one Gaussian latent model, ``Aη ~ N(Dβ, σ²I)`` with ``A = I − rW``:
``D = X`` for SAR (``r = ρ``) and ``D = AX`` for SEM (``r = λ``, from
``A(η − Xβ) = ν``).  The random effects' sd σ has a half-t prior
(``LogitREGibbsPriors.sd_nu``, ``sd_scale``).  It is identified only through the
spatial structure — ν propagates through the filter, the logistic noise does
not — so it is weakly identified in a cross-section, and honestly so.

The logistic likelihood is augmented with Pólya–Gamma variables (Polson, Scott
& Windle 2013), ``ω ~ PG(1, η)``, which leaves the Gaussian working likelihood
``exp(κη − ½ηᵀΩη)``, ``κ = y − ½``.  The latent field's precision given the
data is ``P = AᵀA/σ² + Ω``.

**Sweep** (partially collapsed Gibbs, then two moves on the exact likelihood):

1. ``ω | η``.
2. ``r | ω, σ`` and then ``σ | ω, r``, each by slice sampling with η *and* β
   integrated out (:func:`_collapsed_parts`).
3. ``β | ω, r, σ`` with η integrated out, from the same collapsed moments.
4. ``η | ω, r, β, σ``.
5. The ridge move (:func:`_ridge_move`): σ and β together, with the
   standardized effects and the attenuated ``β/g(σ)`` held.
6. The scale move (:func:`_scale_move`): ``(η, β, σ) → eᵗ(η, β, σ)``.

Steps 2–4 draw ``(r, σ, β, η) | ω`` as one block, so η and β are never used
between the r and σ updates and their redraw.  The sampler this replaced drew
η and β before ρ and reused them after it, which left its posterior for the
intercept too narrow.

Steps 5 and 6 condition on no ω, which is redrawn before its next use, so they
leave the sweep valid.  They are what lets σ mix.  In a cross-section the
likelihood barely distinguishes small σ from large σ with a larger β (the
logit's attenuation), so σ's posterior has a long tail.  Given ω the tail is
sticky: a large σ means a large ``|η|``, small Pólya–Gamma weights and pseudo-
data ``κ/ω`` that themselves favour a large σ, and a chain that wanders there
stays.  On a 35 × 35 SEM lattice (σ = 0.5), steps 1–4 alone gave σ an R-hat of
1.3–1.5; with both moves, an effective sample two to four times NUTS's in under
a third of its run time, and NUTS's tail mass.

References
----------
Liu, J. S., & Sabatti, C. (2000). Generalised Gibbs sampler and multigrid
Monte Carlo for Bayesian computation. *Biometrika*, 87(2), 353–369.

Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for
logistic models using Pólya–Gamma latent variables. *JASA*, 108(504),
1339–1349.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve, solve_triangular

from ...models.priors import LogitREGibbsPriors, SEMLogitGibbsPriors
from .._utils._base import GibbsBaseState
from .._utils._group_effects import _slice_log_sd
from .._utils._polyagamma import sample_polyagamma
from .._utils._slice import (
    SliceWidthState,
    slice_sample_1d,
    slice_sample_1d_adaptive,
    update_slice_width,
)
from .._utils._sparsax_utils import CachedSparseSolver
from .._utils._spatial_normal import (
    CholmodFactor,
    KrylovPrecisionBasis,
    build_precision_krylov_basis,
    eval_precision_logdet_from_basis,
    eval_precision_solve_from_basis,
    sample_spatial_normal,
)

# ---------------------------------------------------------------------------
# State and cache
# ---------------------------------------------------------------------------


@dataclass
class LogitGibbsState(GibbsBaseState):
    """Mutable state of the SAR-logit sweep."""

    eta: np.ndarray  # (n,) latent log-odds
    omega: np.ndarray  # (n,) PG auxiliary variables
    sigma: float = 1.0  # sd of the random effects ν

    def to_jax(self) -> "JAXLogitGibbsState":
        """Convert to a JAX-compatible :class:`JAXLogitGibbsState`."""
        import jax.numpy as jnp

        return JAXLogitGibbsState(
            eta=jnp.asarray(self.eta, dtype=jnp.float64),
            beta=jnp.asarray(self.beta, dtype=jnp.float64),
            rho=jnp.float64(self.rho),
            omega=jnp.asarray(self.omega, dtype=jnp.float64),
            sigma=jnp.float64(self.sigma),
        )


# JAX-compatible state class (equinox.Module when available, stub otherwise).
from .._utils._jax_base import make_jax_state_class  # noqa: E402

JAXLogitGibbsState = make_jax_state_class(
    "JAXLogitGibbsState",
    ("eta", "beta", "rho", "omega", "sigma"),
    numpy_state_cls=LogitGibbsState,
)


class LogitGibbsCache(NamedTuple):
    """Precomputed data that doesn't change across sweeps (σ-free pieces)."""

    W_sparse: sp.csr_matrix
    XtX: np.ndarray  # (k, k) = XᵀX
    logdet_fn: Callable[[float], float]  # log|I − ρW|
    rho_lower: float
    rho_upper: float
    cholmod_factor: CholmodFactor
    W_sym: sp.csr_matrix  # W + Wᵀ
    WtW: sp.csr_matrix  # WᵀW
    WtX: np.ndarray  # (n, k) = WᵀX, ρ-constant
    lanczos_n_probes: int = 10
    lanczos_deg: int = 30
    # --- Krylov basis reuse for the ρ slice -------------------------------
    # When krylov_degree > 0, the slice step factors P(ρ_c) once and reuses
    # the basis for all candidates within krylov_dmax.  Only beneficial on
    # high-fill-in graphs (queen contiguity, knn); on ring/lattices with
    # minimal fill-in CHOLMOD is already cheap, which the threshold guards.
    krylov_degree: int = 0
    krylov_dmax: float = 0.4
    krylov_min_n: int = 400
    rho_adaptive_width: bool = True
    rho_slice_width_state: SliceWidthState | None = None


@dataclass
class SEMLogitGibbsState:
    """Mutable state of the SEM-logit sweep."""

    eta: np.ndarray  # (n,) latent log-odds
    beta: np.ndarray  # (k,) regression coefficients
    lam: float  # spatial error parameter
    omega: np.ndarray  # (n,) PG auxiliary variables
    sigma: float = 1.0  # sd of the random effects ν

    def to_jax(self) -> "JAXSEMLogitGibbsState":
        """Convert to a JAX-compatible :class:`JAXSEMLogitGibbsState`."""
        import jax.numpy as jnp

        return JAXSEMLogitGibbsState(
            eta=jnp.asarray(self.eta, dtype=jnp.float64),
            beta=jnp.asarray(self.beta, dtype=jnp.float64),
            lam=jnp.float64(self.lam),
            omega=jnp.asarray(self.omega, dtype=jnp.float64),
            sigma=jnp.float64(self.sigma),
        )


JAXSEMLogitGibbsState = make_jax_state_class(
    "JAXSEMLogitGibbsState",
    ("eta", "beta", "lam", "omega", "sigma"),
    numpy_state_cls=SEMLogitGibbsState,
)


class SEMLogitGibbsCache(NamedTuple):
    """Precomputed data for the SEM-logit sweep (σ-free pieces)."""

    W_sparse: sp.csr_matrix
    XtX: np.ndarray  # (k, k) = XᵀX
    logdet_fn: Callable[[float], float]  # log|I − λW|
    lam_lower: float
    lam_upper: float
    cholmod_factor: CholmodFactor
    W_sym: sp.csr_matrix  # W + Wᵀ
    WtW: sp.csr_matrix  # WᵀW
    lanczos_n_probes: int = 10
    lanczos_deg: int = 30
    krylov_degree: int = 0
    krylov_dmax: float = 0.4
    krylov_min_n: int = 400
    lam_adaptive_width: bool = True
    lam_slice_width_state: SliceWidthState | None = None


# ---------------------------------------------------------------------------
# The shared collapsed algebra
# ---------------------------------------------------------------------------


class _Design(NamedTuple):
    """``D(r)`` and the pieces the collapsed density needs, polynomial in r.

    ``u(r) = AᵀD = Σ_j r^j U_j`` and ``DᵀD(r) = Σ_j r^j G_j``: SAR has
    ``D = X``, ``u = X − rWᵀX``; SEM has ``D = AX``, ``u = AᵀAX = X − r(W+Wᵀ)X
    + r²WᵀWX`` and ``DᵀD = XᵀAᵀAX``.
    """

    U: tuple  # n × k coefficient arrays of u(r)
    G: tuple  # k × k coefficient arrays of DᵀD(r)

    def u(self, r):
        return sum(r**j * Uj for j, Uj in enumerate(self.U))

    def DtD(self, r):
        return sum(r**j * Gj for j, Gj in enumerate(self.G))


def _sar_design(X, cache):
    return _Design(U=(X, -cache.WtX), G=(cache.XtX,))


def _sem_design(X, cache):
    WsX, WtWX = cache.W_sym @ X, cache.WtW @ X
    return _Design(U=(X, -WsX, WtWX), G=(X.T @ X, -(X.T @ WsX), X.T @ WtWX))


def _beta_prior(priors, k):
    mu = np.broadcast_to(np.asarray(priors.beta_mu, dtype=np.float64), (k,))
    sd = np.broadcast_to(np.asarray(priors.beta_sigma, dtype=np.float64), (k,))
    return mu, sd


def _sd_log_prior(s, priors):
    """The half-t(``sd_nu``, ``sd_scale``) log density of σ, up to a constant."""
    nu, scale = float(priors.sd_nu), float(priors.sd_scale)
    return -0.5 * (nu + 1.0) * np.log1p(s * s / (nu * scale * scale))


def _logit_loglik(y, eta):
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _precision(base, W_sym, WtW, r, sigma):
    """``P = base + (−r(W+Wᵀ) + r²WᵀW)/σ²``; ``base`` holds ``I/σ² + diag(ω)``."""
    s2 = sigma**2
    return base - (r / s2) * W_sym + (r * r / s2) * WtW


def _collapsed_parts(r, sigma, omega, kappa, design, cache, priors, krylov=None,
                     lanczos_rng=None):  # fmt: skip
    r"""``log p(r, σ | ω)`` up to a constant, with η and β integrated out.

    With ``P = AᵀA/σ² + Ω``, ``z = P⁻¹κ``, ``M = P⁻¹u``:
    ``Σ_β⁻¹ = DᵀD/σ² + V₀⁻¹ − uᵀM/σ⁴``, ``r_β = uᵀz/σ² + V₀⁻¹b₀`` and

    .. math::

        \log|A| - n\log\sigma - \tfrac12\log|P| - \tfrac12\log|\Sigma_\beta^{-1}|
        + \tfrac12\kappa^\top z + \tfrac12 r_\beta^\top\Sigma_\beta r_\beta .

    ``krylov`` (a basis built at this σ) reuses one factorization across r.
    Returns ``(log density, chol(Σ_β⁻¹), r_β)``, or ``(-inf, None, None)``.
    """
    n = kappa.size
    s2 = sigma**2
    k = design.G[0].shape[0]
    beta_mu, beta_sd = _beta_prior(priors, k)
    V0_inv = 1.0 / beta_sd**2
    u = design.u(r)
    if krylov is not None and abs(r - krylov.rho_basis) <= krylov.safe_dmax:
        dr = r - krylov.rho_basis
        sol = eval_precision_solve_from_basis(krylov, dr)
        z = sol[:, 0]
        M = sum(
            r**j * sol[:, 1 + j * k : 1 + (j + 1) * k] for j in range(len(design.U))
        )
        log_det_P = eval_precision_logdet_from_basis(krylov, dr, rng=lanczos_rng)
    else:
        f = cache.cholmod_factor
        base = sp.eye(n, format="csr") / s2 + sp.diags(omega, format="csr")
        f.factorize(_precision(base, cache.W_sym, cache.WtW, r, sigma))
        log_det_P = f.logdet()
        sol = f.solve(np.column_stack([kappa, u]))
        z, M = sol[:, 0], sol[:, 1:]
    Sig_inv = design.DtD(r) / s2 - (u.T @ M) / s2**2
    Sig_inv[np.diag_indices(k)] += V0_inv
    try:
        Lb = np.linalg.cholesky(Sig_inv)
    except np.linalg.LinAlgError:
        return -np.inf, None, None
    rhs_b = (u.T @ z) / s2 + beta_mu * V0_inv
    w = solve_triangular(Lb, rhs_b, lower=True)
    val = (
        cache.logdet_fn(r)
        - n * np.log(sigma)
        - 0.5 * log_det_P
        - float(np.sum(np.log(np.diag(Lb))))
        + 0.5 * float(kappa @ z)
        + 0.5 * float(w @ w)
    )
    return (val if np.isfinite(val) else -np.inf), Lb, rhs_b


def _krylov_basis(r, sigma, omega, kappa, design, cache, rng):
    """A shift-invert Krylov basis of ``P(r)`` at this σ, seeded with
    ``[κ, U_0, U_1, ...]`` so ``P⁻¹u(r)`` is a polynomial combination."""
    n = kappa.size
    if cache.krylov_degree <= 0 or n < cache.krylov_min_n:
        return None
    s2 = sigma**2
    return build_precision_krylov_basis(
        rho_c=float(r),
        base=sp.eye(n, format="csr") / s2 + sp.diags(omega, format="csr"),
        G1=cache.W_sym / s2,
        G2=cache.WtW / s2,
        rhs=np.column_stack([kappa, *design.U]),
        degree=cache.krylov_degree,
        cholmod_factor=cache.cholmod_factor,
        n_probes=cache.lanczos_n_probes,
        lanczos_deg=cache.lanczos_deg,
        rng=rng,
        dmax=cache.krylov_dmax,
    )


def _sample_r_sigma_beta(r, sigma, omega, y, X, design, cache, priors, bounds,
                         width, *, rng, tuning):  # fmt: skip
    """Blocks 2–3: r | ω, σ, then σ | ω, r (η and β integrated out), then β.

    Returns ``(r, σ, β)``.
    """
    kappa = y - 0.5
    lanczos_rng = np.random.default_rng(rng.integers(2**31))
    krylov: KrylovPrecisionBasis | None = _krylov_basis(
        r, sigma, omega, kappa, design, cache, lanczos_rng
    )
    r, _, sl, sr = slice_sample_1d_adaptive(
        log_density=lambda v: _collapsed_parts(
            v, sigma, omega, kappa, design, cache, priors, krylov, lanczos_rng
        )[0],
        x0=r,
        lower=bounds[0],
        upper=bounds[1],
        width_state=width,
        rng=rng,
    )
    if tuning:
        update_slice_width(width, sl, sr)

    def sigma_density(ls):
        s = np.exp(ls)
        val = _collapsed_parts(r, s, omega, kappa, design, cache, priors)[0]
        return val + _sd_log_prior(s, priors) + ls

    sigma = _slice_log_sd(sigma_density, sigma, float(priors.sd_scale), 0.5, rng)

    _, Lb, rhs_b = _collapsed_parts(r, sigma, omega, kappa, design, cache, priors)
    k = rhs_b.size
    mean = cho_solve((Lb, True), rhs_b)
    beta = mean + solve_triangular(Lb, rng.standard_normal(k), lower=True, trans="T")
    return float(r), float(sigma), beta


#: The logit attenuation ``g(σ) = √(1 + cσ²)``: integrating a normal effect of
#: sd σ out of a logit scales its coefficients by about ``1/g(σ)``
#: (``logistic(x) ≈ Φ(x√(π/8))``), so the data pin ``β/g(σ)``.
_ATTENUATION = np.pi / 8.0


def _attenuation(s):
    return np.sqrt(1.0 + _ATTENUATION * s * s)


def _ridge_move(eta, m, beta, sigma, y, priors, rng):
    r"""Block 5: σ with the standardized effects and ``β̃ = β/g(σ)`` held.

    ``η = m + d`` splits the latent field into its β part ``m = A⁻¹Dβ``
    (linear in β) and the effects' part ``d = A⁻¹ν`` (linear in σ).  Holding
    ``ν/σ`` and ``β̃``, a new σ' gives ``β' = g(σ')β̃`` and
    ``η' = (g(σ')/g(σ))m + (σ'/σ)d``: σ and β move together along the ridge
    where the data hold the attenuated coefficients.  It is a Gibbs step in
    the ``(σ, β̃, ν/σ)`` parameterization on the exact Bernoulli likelihood,
    whose density carries the Jacobian ``g(σ')ᵏ`` of ``β = g(σ)β̃``.
    Returns ``(η, β, σ)``.
    """
    k = beta.size
    mu, sd = _beta_prior(priors, k)
    g0 = _attenuation(sigma)
    d = eta - m
    bt = beta / g0

    def at(s):
        g = _attenuation(s)
        return (g / g0) * m + (s / sigma) * d, g * bt, g

    def log_density(ls):
        s = np.exp(ls)
        e, b, g = at(s)
        val = (
            _logit_loglik(y, e)
            - 0.5 * float(np.sum(((b - mu) / sd) ** 2))
            + k * np.log(g)
            + _sd_log_prior(s, priors)
            + ls
        )
        return val if np.isfinite(val) else -np.inf

    s = _slice_log_sd(log_density, sigma, float(priors.sd_scale), 0.3, rng)
    e, b, _ = at(s)
    return e, b, s


def _scale_move(eta, beta, sigma, y, priors, rng):
    r"""Block 6: rescale the latent scale jointly, ``(η, β, σ) → eᵗ(η, β, σ)``.

    With the standardized effects held, η is linear in ``(β, σ)``, so the
    scaled state has latent field ``eᵗη``.  ``t`` is drawn by a slice step on
    ``log p(y | eᵗη) + log p(eᵗβ) + log p(eᵗσ) + (k + 1)t``, a generalized
    Gibbs move on the scale group (Liu & Sabatti 2000) whose Jacobian is the
    last term.  Without it the ridge move alone leaves chains in σ's tail.
    Returns ``(η, β, σ)``.
    """
    k = beta.size
    mu, sd = _beta_prior(priors, k)

    def log_density(t):
        e = np.exp(t)
        val = (
            _logit_loglik(y, e * eta)
            - 0.5 * float(np.sum(((e * beta - mu) / sd) ** 2))
            + _sd_log_prior(e * sigma, priors)
            + (k + 1) * t
        )
        return val if np.isfinite(val) else -np.inf

    # Bounds far outside any reachable scale keep the step translation-
    # equivariant, which the generalized Gibbs move needs.
    t, _ = slice_sample_1d(log_density, 0.0, lower=-30.0, upper=30.0, w=0.2, rng=rng)
    e = float(np.exp(t))
    return e * eta, e * beta, e * sigma


def _sample_omega(eta: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Block 1: ω | η ~ PG(1, η)."""
    return sample_polyagamma(np.ones(len(eta)), eta, rng=rng)


def _sample_eta(r, sigma, beta, omega, y, design, cache, *, rng):
    """Block 4: η | ω, r, β, σ ~ N(P⁻¹(uβ/σ² + κ), P⁻¹), ``u = AᵀD``."""
    n = y.size
    s2 = sigma**2
    base = sp.eye(n, format="csr") / s2 + sp.diags(omega, format="csr")
    P = _precision(base, cache.W_sym, cache.WtW, r, sigma)
    rhs = design.u(r) @ beta / s2 + (y - 0.5)
    return sample_spatial_normal(P, rhs, rng=rng, cached_factor=cache.cholmod_factor).x


def _logit_loglik_pointwise(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Pointwise Bernoulli-logit log-likelihood ``y η − log(1 + e^η)``."""
    return y * eta - np.logaddexp(0.0, eta)


def _run(y, X, priors, cache, init, draws, tune, thin, return_eta, rng,
         progress_manager, chain_id, store_log_lik, *, sem):  # fmt: skip
    rng = np.random.default_rng() if rng is None else rng
    n, k = X.shape
    name = "lam" if sem else "rho"
    design = _sem_design(X, cache) if sem else _sar_design(X, cache)
    bounds = (
        (priors.lam_lower, priors.lam_upper)
        if sem
        else (priors.rho_lower, priors.rho_upper)
    )
    width = (
        cache.lam_slice_width_state if sem else cache.rho_slice_width_state
    ) or SliceWidthState(w=0.2)
    adapt = cache.lam_adaptive_width if sem else cache.rho_adaptive_width
    n_keep = draws // thin if thin > 0 else draws
    out = {
        name: np.empty(n_keep),
        "beta": np.empty((n_keep, k)),
        "sigma": np.empty(n_keep),
        "log_lik": np.empty((n_keep, n)) if store_log_lik else None,
        "eta_norm": np.empty(n_keep),
    }
    if return_eta:
        out["eta"] = np.empty((n_keep, n))
    r, eta, beta, sigma = (
        getattr(init, name),
        init.eta.copy(),
        init.beta.copy(),
        float(init.sigma),
    )
    # The β part of η, A⁻¹Dβ: Xβ for SEM, a sparse solve for SAR.
    solver = None if sem else CachedSparseSolver([sp.csr_matrix(cache.W_sparse)], n)
    for i in range(tune + draws):
        omega = _sample_omega(eta, rng=rng)
        r, sigma, beta = _sample_r_sigma_beta(
            r, sigma, omega, y, X, design, cache, priors, bounds, width,
            rng=rng, tuning=adapt and i < tune,
        )  # fmt: skip
        eta = _sample_eta(r, sigma, beta, omega, y, design, cache, rng=rng)
        m = X @ beta if sem else solver.solve([-r], X @ beta)
        eta, beta, sigma = _ridge_move(eta, m, beta, sigma, y, priors, rng)
        eta, beta, sigma = _scale_move(eta, beta, sigma, y, priors, rng)
        if i >= tune and (i - tune) % thin == 0:
            j = (i - tune) // thin
            if j < n_keep:
                out[name][j] = r
                out["beta"][j] = beta
                out["sigma"][j] = sigma
                if store_log_lik:
                    out["log_lik"][j] = _logit_loglik_pointwise(y, eta)
                out["eta_norm"][j] = float(eta @ eta)
                if return_eta:
                    out["eta"][j] = eta
        if progress_manager is not None:
            progress_manager.update(chain_id, i, tuning=i < tune)
    return out


def run_chain(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: LogitREGibbsPriors,
    cache: LogitGibbsCache,
    init: LogitGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
) -> dict:
    """One chain of the SAR-logit Gibbs sampler.

    Returns ``rho``, ``beta``, ``sigma``, ``log_lik``, ``eta_norm`` and, with
    ``return_eta``, ``eta``.
    """
    return _run(
        y, X, priors, cache, init, draws, tune, thin, return_eta, rng,
        progress_manager, chain_id, store_log_lik, sem=False,
    )  # fmt: skip


def run_chain_sem(
    y: np.ndarray,
    X: np.ndarray,
    W_sparse: sp.csr_matrix,
    priors: SEMLogitGibbsPriors,
    cache: SEMLogitGibbsCache,
    init: SEMLogitGibbsState,
    draws: int,
    tune: int,
    thin: int = 1,
    return_eta: bool = False,
    rng=None,
    progress_manager=None,
    chain_id: int = 0,
    store_log_lik: bool = True,
) -> dict:
    """One chain of the SEM-logit Gibbs sampler.

    Returns ``lam``, ``beta``, ``sigma``, ``log_lik``, ``eta_norm`` and, with
    ``return_eta``, ``eta``.
    """
    return _run(
        y, X, priors, cache, init, draws, tune, thin, return_eta, rng,
        progress_manager, chain_id, store_log_lik, sem=True,
    )  # fmt: skip
