r"""Exact-likelihood Gibbs for the separable NB flow model (cross-section and panel).

The model of :mod:`._flow_structured`: ``y_od ~ NB2(μ_od, α)``, ``log μ = η =
A⁻¹Xβ`` with ``A = L_o ⊗ L_d``, ``L_k = I_n − ρ_k W``, ``T`` periods stacked
time-first sharing ``ρ``, ``β`` and ``α``.  The reduced form has no latent
field: ``η`` is a function of ``k + 3`` parameters, so the sweep slices the
exact NB2 likelihood, with no Pólya–Gamma layer.  At small ``α`` the
Pólya–Gamma likelihood given ``ω`` is far more precise than the NB2 one (the
ratio ``Σ E[ω]`` to the Fisher information was 71 on the corrected LA panel),
and chains that condition on ``ω`` crawl; this sweep has no such regime.

**Coordinates.**  ``a = −log(1 − ρ_d)``, ``b = −log(1 − ρ_o)`` and ``θ = mβ``,
``m = e^{a+b} = 1/((1 − ρ_d)(1 − ρ_o))``.  With a row-standardized ``W`` the
mean level of ``η`` is about ``β₀m``, so holding ``θ`` holds the level while
``(a, b)`` move, and the ρ–β ridge runs close to the ``(a, b)`` axes.  The
Jacobian of ``(ρ_d, ρ_o, β)`` in ``(a, b, θ)`` is ``e^{−(k+1)(a+b)}``.

**Sweep.**

1. ``θ | a, b, α``: ``η = Uθ`` with ``U = A⁻¹X/m``, linear, so each evaluation
   is one pass over the flows.  A slice along each eigenvector of the Fisher
   information ``UᵀDU + P₀/m²``, ``D = μα/(μ + α)``, in random order; the
   eigenvectors are re-estimated at the current state in the first warmup
   sweeps and every :data:`_FISHER_EVERY` after, and fixed after warmup.
2. ``(a, b)`` with ``θ`` sheared along its regression on them: a slice along
   ``(d, Gd)`` for each eigenvector ``d`` of the warmup covariance of
   ``(a, b)``, ``G`` the regression coefficients of ``θ`` on ``(a, b)`` in the
   same draws.  Rescaling alone leaves the coefficients of origin and
   destination attributes correlated with ``a`` or ``b`` (0.7–0.9 in
   simulated flows), and a move of ``(a, b)`` with ``θ`` held then crawls; the
   shear raised ρ's effective draws from 3% to 40% of the sweeps there.  The
   covariance is estimated from the draws after the first fifth of warmup and
   refreshed at two, three and four fifths of it.  Each evaluation is the
   two-sided solve of :mod:`._flow_structured`.
3. ``α | η``: a slice on ``log α``.

Each slice of blocks 1 and 2 shares its stepping-out budget between the sides
at random (Neal 2003), so it stays reversible when the budget binds.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .._utils._slice import slice_along
from .._utils._spatial_normal import FACTORIZATION_ERRORS
from ..negbin._core import GibbsState as _StructuralState
from ..negbin._core import _nb_loglik_pointwise, _sample_alpha
from ._core import ReducedGibbsPriors
from ._flow_structured import (
    FlowDesignStructure,
    _FilterSolver,
    _low_rank,
    _prior_arrays,
    classify_flow_design,
)

#: Warmup sweeps between re-estimates of the θ directions (and the first few,
#: which all re-estimate).
_FISHER_EVERY = 20
_FISHER_FIRST = 5
#: Fifths of warmup: the (a, b, θ) covariance starts after the first and the
#: directions of block 2 refresh at the others.
_AB_START = 1
_AB_REFRESH = (2, 3, 4)
#: Draws the (a, b, θ) covariance needs before it sets directions.
_AB_MIN_DRAWS = 20
#: Initial slice widths of θ and of (a, b).
_THETA_WIDTH = 0.1
_AB_WIDTH = 0.2


def _nb_ll(y: np.ndarray, eta: np.ndarray, alpha: float) -> float:
    """``Σ log NB2(y | e^η, α)`` up to terms constant in ``η``."""
    return float(np.sum(y * eta - (y + alpha) * np.logaddexp(eta, np.log(alpha))))


class ExactFlowChain:
    """One chain's state and sweep (see the module).

    ``y`` stacks ``T`` periods of ``n²`` flows time-first; ``struct`` is the
    design's :func:`classify_flow_design`.  ``tune`` sets the warmup schedule
    of the (a, b) directions.
    """

    def __init__(
        self,
        y: np.ndarray,
        struct: FlowDesignStructure,
        W_csc: sp.csc_matrix,
        priors,
        init,
        tune: int,
        rng: np.random.Generator,
        rho_lower: float = -0.999,
        rho_upper: float = 0.999,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.y_tab = np.unique(self.y, return_counts=True)
        self.struct = struct
        self.n, self.T, self.k = struct.n, struct.T, struct.k
        self.mu0, self.prec0 = _prior_arrays(priors, self.k)
        self.core_priors = ReducedGibbsPriors(
            beta_mu=priors.beta_mu,
            beta_sigma=priors.beta_sigma,
            alpha_sigma=priors.alpha_sigma,
            alpha_nu=priors.alpha_nu,
            alpha_fixed=getattr(priors, "alpha_fixed", None),
            rho_lower=rho_lower,
            rho_upper=rho_upper,
        )
        self.rng = rng
        self.tune = int(tune)
        self.solve_o, self.solve_d = _FilterSolver(W_csc), _FilterSolver(W_csc)
        self.ab_lo, self.ab_hi = -np.log1p(-rho_lower), -np.log1p(-rho_upper)
        # Periods whose full-rank columns coincide share one sum and its solve.
        self.full_keys = [tuple(int(i) for i in struct.f_idx[t]) for t in range(self.T)]
        self.distinct_keys = sorted(set(self.full_keys))

        self.a = -np.log1p(-float(init.rho_d))
        self.b = -np.log1p(-float(init.rho_o))
        self.theta = np.asarray(init.beta, dtype=np.float64) * np.exp(self.a + self.b)
        self.alpha = float(init.alpha)
        self.th_dirs, self.th_widths = np.eye(self.k), np.full(self.k, _THETA_WIDTH)
        self.ab_dirs, self.ab_widths = np.eye(2), np.full(2, _AB_WIDTH)
        self.ab_shear = np.zeros((self.k, 2))
        self.warm_draws: list[np.ndarray] = []
        self.eta = self._eta_at(self.a, self.b, self.theta)
        self.it = 0

    # --- the design through the filter -------------------------------------
    def _factors(self, a: float, b: float):
        """``(L_o, L_d)`` factored at ``ρ_o = 1 − e^{−b}``, ``ρ_d = 1 − e^{−a}``."""
        return self.solve_o.at(-np.expm1(-b)), self.solve_d.at(-np.expm1(-a))

    def _columns(self) -> np.ndarray:
        """``U = A⁻¹X/m`` at the current ``(a, b)``, one row per coefficient."""
        s, n, T = self.struct, self.n, self.T
        Lo, Ld = self._factors(self.a, self.b)
        A, B = Lo.solve(s.U), Ld.solve(s.V)
        Ufull = [Ld.solve(Lo.solve(Fi).T).T for Fi in s.F]
        cols = np.empty((self.k, T, n, n))
        for t in range(T):
            for j, c in enumerate(s.cheap_cols):
                cols[c, t] = np.outer(A[:, s.u_idx[t, j]], B[:, s.v_idx[t, j]])
            for j, c in enumerate(s.full_cols):
                cols[c, t] = Ufull[s.f_idx[t, j]]
        return cols.reshape(self.k, -1) * np.exp(-(self.a + self.b))

    def _eta_at(self, a: float, b: float, theta: np.ndarray) -> np.ndarray:
        """``η = A⁻¹Xθ/m`` at ``(a, b)``, flat over the periods."""
        s = self.struct
        cc, fc = s.cheap_cols, s.full_cols
        Lo, Ld = self._factors(a, b)
        A, B = Lo.solve(s.U), Ld.solve(s.V)
        full = {}
        for key in self.distinct_keys:
            Bf = np.zeros((self.n, self.n))
            for tj, i in zip(theta[fc], key):
                Bf += tj * s.F[i]
            full[key] = Ld.solve(Lo.solve(Bf).T).T
        eta = np.stack(
            [
                full[self.full_keys[t]]
                + (
                    _low_rank(A, B, s.u_idx[t], s.v_idx[t], theta[cc])
                    if len(cc)
                    else 0.0
                )
                for t in range(self.T)
            ]
        )
        return eta.ravel() * np.exp(-(a + b))

    def _log_prior(self, theta: np.ndarray, s: float) -> float:
        """The Normal prior of ``β = θ e^{−s}``."""
        return -0.5 * float(np.sum(self.prec0 * (theta * np.exp(-s) - self.mu0) ** 2))

    # --- the sweep -----------------------------------------------------------
    def sweep(self, tuning: bool) -> None:
        """One sweep (see the module)."""
        rng, y, it = self.rng, self.y, self.it
        theta, alpha, eta = self.theta, self.alpha, self.eta
        s = self.a + self.b

        # 1. θ | a, b, α along the Fisher information's eigenvectors.
        cols = self._columns()
        if tuning and (it < _FISHER_FIRST or it % _FISHER_EVERY == 0):
            w = alpha * np.exp(-np.logaddexp(0.0, np.log(alpha) - eta))
            H = (cols * w) @ cols.T
            H[np.diag_indices(self.k)] += self.prec0 * np.exp(-2.0 * s)
            lam, V = np.linalg.eigh(H)
            self.th_dirs, self.th_widths = V, 2.0 / np.sqrt(np.maximum(lam, 1e-12))
        UD = self.th_dirs.T @ cols
        del cols
        f = _nb_ll(y, eta, alpha) + self._log_prior(theta, s)
        for j in rng.permutation(self.k):
            d, ud = self.th_dirs[:, j], UD[j]
            t, f, _ = slice_along(
                lambda t, d=d, ud=ud, e=eta, th=theta: (
                    _nb_ll(y, e + t * ud, alpha) + self._log_prior(th + t * d, s),
                    None,
                ),
                f,
                None,
                self.th_widths[j],
                rng,
            )
            theta, eta = theta + t * d, eta + t * ud
        del UD

        # 2. (a, b) with θ sheared along G, the Jacobian e^{-(k+1)(a+b)}.
        k1 = self.k + 1

        def g(z):
            a, b, th = z[0], z[1], z[2:]
            if not (self.ab_lo < a < self.ab_hi and self.ab_lo < b < self.ab_hi):
                return -np.inf, None
            try:
                e = self._eta_at(a, b, th)
            except FACTORIZATION_ERRORS:
                return -np.inf, None  # ρ at the edge of its support
            val = _nb_ll(y, e, alpha) + self._log_prior(th, a + b) - k1 * (a + b)
            return (val, e) if np.isfinite(val) else (-np.inf, None)

        z = np.r_[self.a, self.b, theta]
        gz = f - k1 * s
        for j in rng.permutation(2):
            d = self.ab_dirs[:, j]
            v = np.r_[d, self.ab_shear @ d]
            t, gz, eta = slice_along(
                lambda t, z=z, v=v: g(z + t * v), gz, eta, self.ab_widths[j], rng
            )
            z = z + t * v
        self.a, self.b, theta = float(z[0]), float(z[1]), z[2:]
        if tuning and it >= _AB_START * self.tune // 5:
            self.warm_draws.append(z)
            refresh = {q * self.tune // 5 for q in _AB_REFRESH}
            if it in refresh and len(self.warm_draws) > _AB_MIN_DRAWS:
                C = np.cov(np.asarray(self.warm_draws).T)
                C_ab = C[:2, :2] + 1e-10 * np.eye(2)
                self.ab_shear = np.linalg.solve(C_ab, C[:2, 2:]).T
                lam, self.ab_dirs = np.linalg.eigh(C_ab)
                self.ab_widths = 2.0 * np.sqrt(np.maximum(lam, 1e-12))

        # 3. α | η.
        state = _StructuralState(
            eta=eta, beta=theta, rho=0.0, alpha=alpha, omega=np.empty(0)
        )
        self.alpha = _sample_alpha(
            state, y, self.core_priors, rng=rng, y_tab=self.y_tab
        )
        self.theta, self.eta = theta, eta
        self.it += 1

    def draw(self) -> dict:
        """The current ``rho_d``, ``rho_o``, ``rho_w``, ``beta`` and ``alpha``."""
        rho_d, rho_o = -np.expm1(-self.a), -np.expm1(-self.b)
        return {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": -rho_d * rho_o,
            "beta": self.theta * np.exp(-(self.a + self.b)),
            "alpha": self.alpha,
        }


def run_chain_separable_exact(
    y: np.ndarray,
    X: np.ndarray | None,
    W_csc: sp.csc_matrix,
    n: int,
    priors,
    init,
    draws: int,
    tune: int,
    *,
    T: int = 1,
    rho_lower: float = -0.999,
    rho_upper: float = 0.999,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    chain_id: int = 0,
    progress_manager: object | None = None,
    store_log_lik: bool = False,
    struct: FlowDesignStructure | None = None,
) -> dict[str, np.ndarray]:
    """One chain of the exact separable NB flow sampler.

    ``y`` and ``X`` are stacked time-first over ``T`` periods of ``n²`` flows.
    Pass ``struct`` (from :func:`classify_flow_design`) to share the
    classification across chains; ``X`` is then unused.  Returns ``rho_d,
    rho_o, rho_w, beta, alpha, log_lik``.
    """
    rng = np.random.default_rng() if rng is None else rng
    struct = classify_flow_design(X, n, T) if struct is None else struct
    chain = ExactFlowChain(
        y, struct, W_csc, priors, init, tune, rng, rho_lower, rho_upper
    )
    n_keep = draws // thin if thin > 0 else draws
    out = {
        "rho_d": np.empty(n_keep),
        "rho_o": np.empty(n_keep),
        "rho_w": np.empty(n_keep),
        "beta": np.empty((n_keep, struct.k)),
        "alpha": np.empty(n_keep),
        "log_lik": np.empty((n_keep, chain.y.size)) if store_log_lik else None,
    }
    for it in range(tune + draws):
        chain.sweep(tuning=it < tune)
        if it >= tune and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                for name, value in chain.draw().items():
                    out[name][j] = value
                if store_log_lik:
                    out["log_lik"][j] = _nb_loglik_pointwise(
                        chain.y, chain.eta, chain.alpha
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)
    return out
