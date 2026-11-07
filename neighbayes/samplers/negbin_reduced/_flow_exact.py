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

**Levels** (:class:`FlowLevels`), outside the filter: per origin, destination,
pair or period, ``η_{t,ij} = (A⁻¹X_tβ)_{ij} + γ_i + κ_j + d_ij + τ_t`` with any
of them present (a :class:`LevelSet` each).  Fixed levels have a fixed Normal
prior or none (flat); random ones are ``N(μ, τ²)`` with τ learned under a
half-t(3) prior.  The doubly constrained gravity model with random balancing
factors is the random origin and destination levels; the fixed-effects panel
is the fixed pair and period levels.  A fixed set after the first holds its
first level at its prior mean, so the sets share one level.  A flat set's
units with no count in any live cell are dropped from the likelihood, their
levels having no proper posterior, as in the conditional likelihood; a
dropped unit says nothing about the other parameters, its level being free.

Between blocks 1 and 2 the sweep draws every level of a set at once, each from
its cells' log-concave conditional by independence Metropolis–Hastings from a
*t* at the mode; then, between consecutive random sets, ``c`` in ``(γ + c,
κ − c)``, which leaves ``η`` unchanged, exactly from its Gaussian conditional;
then the random sets' sds by slices on ``log τ``.

The levels absorb the means of everything else in ``η`` over their cells, so
with the levels held a move of ``θ`` or ``(a, b)`` that shifts those means
crawls (ρ's effective draws 0.1% of the sweeps in a simulated doubly
constrained flow).  Both blocks therefore carry the levels along.  Block 1's
move along ``d`` shifts the levels by minus the means of the filtered column
``Ud`` that they absorb, so ``η`` moves by the rest of it only (with pair
levels, its within-pair variation: the within transform, done in the
sampler), and its directions are those of the Fisher information of that rest
plus the levels' prior.  Block 2 holds each level's total, the level plus the
mean of the filtered part over its cells, a translation of the levels with
unit Jacobian.  ρ's effective draws rose to 7–12%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.special import expit, logsumexp

from .._utils._slice import slice_along, slice_sample_1d
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
#: Newton steps to a level's conditional mode, the largest step, and the
#: degrees of freedom of its proposal.
_LEVEL_NEWTON = 30
_LEVEL_MAX_STEP = 2.0
_LEVEL_DF = 8.0


def _nb_ll(y: np.ndarray, eta: np.ndarray, alpha: float, live=1.0) -> float:
    """``Σ log NB2(y | e^η, α)`` over the live cells, up to terms constant in ``η``."""
    return float(
        np.sum(live * (y * eta - (y + alpha) * np.logaddexp(eta, np.log(alpha))))
    )


#: The axes of the ``(..., T, n, n)`` flow stack each kind of level sums over.
_LEVEL_AXES = {
    "origin": (-3, -1),
    "destination": (-3, -2),
    "pair": (-3,),
    "period": (-2, -1),
}


@dataclass(frozen=True)
class LevelSet:
    """One set of levels added to the log-mean after the filter.

    ``kind`` is ``"origin"``, ``"destination"``, ``"pair"`` or ``"period"``.
    Random levels are ``N(mu, τ²)``, τ learned under a half-t(3, ``scale``)
    prior; fixed ones ``N(mu, scale²)``, flat when ``scale`` is infinite.
    """

    kind: str
    random: bool = False
    mu: float = 0.0
    scale: float = np.inf

    @property
    def axes(self) -> tuple[int, ...]:
        return _LEVEL_AXES[self.kind]

    @property
    def flat(self) -> bool:
        return not self.random and np.isinf(self.scale)


@dataclass(frozen=True)
class FlowLevels:
    """A model's level sets, in order (see the module)."""

    sets: tuple[LevelSet, ...]

    def baseline(self, i: int) -> bool:
        """Whether set ``i`` holds its first level at its mean: a fixed set after the first."""
        return i > 0 and not self.sets[i].random


def _grow(v: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """A level array broadcast against the ``(..., T, n, n)`` stack."""
    return np.expand_dims(v, axes)


def _live_mean(F: np.ndarray, axes: tuple[int, ...], M: np.ndarray) -> np.ndarray:
    """The mean of ``F`` over ``axes``, live cells (``M``) only."""
    return np.sum(F * M, axis=axes) / np.maximum(np.sum(M, axis=axes), 1.0)


def normalize_cell_mask(cell_mask, T: int, n: int):
    """Validate a cell mask and flatten it to ``(T·n·n,)`` bool, or ``None``.

    Accepts a bool array shaped ``(T, n, n)``, ``(T·n·n,)`` or ``(n, n)``
    (broadcast over periods); ``True`` marks a live cell.  The flat order is
    C-contiguous time-first, matching the stacked response.
    """
    if cell_mask is None:
        return None
    m = np.asarray(cell_mask)
    if m.dtype != bool:
        m = m.astype(bool)
    N = T * n * n
    if m.shape == (T, n, n):
        return m.reshape(N)
    if m.shape == (N,):
        return m
    if m.shape == (n, n):
        return np.broadcast_to(m, (T, n, n)).reshape(N).copy()
    raise ValueError(
        f"cell_mask must be bool of shape {(T, n, n)}, {(N,)} or {(n, n)}; "
        f"got {m.shape}"
    )


def live_cells(y, T, n, levels, cell_mask=None) -> np.ndarray:
    """The ``(T, n, n)`` 0/1 mask of the cells in the likelihood.

    ``cell_mask`` (``True`` live) less the cells of every flat level set's
    units with no count in a live cell: their levels have no proper posterior,
    and with the level free they say nothing about the rest.
    """
    Y = np.asarray(y, dtype=np.float64).reshape(T, n, n)
    M = np.ones((T, n, n))
    if cell_mask is not None:
        M = M * np.asarray(cell_mask, dtype=np.float64).reshape(T, n, n)
    for st in levels.sets if levels is not None else ():
        if st.flat:
            M = M * ~_grow(np.sum(Y * M, axis=st.axes) == 0, st.axes)
    return M


def _level_draw(base, y, M, alpha, current, mu, prec, axes, rng):
    """Every level of a set at once, given the rest of the log-mean ``base``.

    Level ``l`` adds to the cells its ``axes`` sum over, and its conditional
    ``Σ [y(b + l) − (y + α) log(e^{b+l} + α)] − prec (l − μ)²/2`` over its live
    cells (``M``) is log-concave.  Each is drawn by independence
    Metropolis–Hastings from a *t* at its mode, which Newton reaches from a
    start that ignores the current value, so the proposal depends on the
    conditioning values only.  A level without live cells keeps its value.
    """
    la = np.log(alpha)
    yM, aM = y * M, (y + alpha) * M
    sy = yM.sum(axis=axes)
    live = M.sum(axis=axes) > 0

    def logp(v):
        e = base + _grow(v, axes)
        ll = (yM * e - aM * np.logaddexp(e, la)).sum(axis=axes)
        return ll - 0.5 * prec * (v - mu) ** 2

    with np.errstate(divide="ignore"):
        start = np.log(sy + 0.5) - logsumexp(np.where(M > 0, base, -np.inf), axis=axes)
    x = np.where(live, start, mu)
    for _ in range(_LEVEL_NEWTON):
        p = expit(base + _grow(x, axes) - la)
        grad = sy - (aM * p).sum(axis=axes) - prec * (x - mu)
        hess = (aM * p * (1.0 - p)).sum(axis=axes) + prec
        step = np.where(
            live, np.clip(grad / np.where(live, hess, 1.0), -_LEVEL_MAX_STEP,
                          _LEVEL_MAX_STEP), 0.0,
        )  # fmt: skip
        x = x + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(base + _grow(x, axes) - la)
    sd = 1.0 / np.sqrt(np.where(live, (aM * p * (1.0 - p)).sum(axis=axes) + prec, 1.0))

    def logq(v):
        z = (v - x) / sd
        return -0.5 * (_LEVEL_DF + 1.0) * np.log1p(z * z / _LEVEL_DF)

    prop = x + sd * rng.standard_t(_LEVEL_DF, size=x.shape)
    log_ratio = logp(prop) - logp(current) + logq(current) - logq(prop)
    accept = live & (np.log(rng.random(x.shape)) < log_ratio)
    return np.where(accept, prop, current)


def _sd_draw(sigma, x, mu, scale, rng):
    """The sd of ``x ~ N(mu, sigma²)``: a slice on ``log sigma``, half-t(3, scale)."""
    ss, G = float(np.sum((x - mu) ** 2)), x.size

    def logp(ls):
        s2 = np.exp(2.0 * ls)
        return -G * ls - 0.5 * ss / s2 - 2.0 * np.log1p(s2 / (3.0 * scale**2)) + ls

    ls, _ = slice_sample_1d(logp, np.log(sigma), -10.0, 10.0, w=0.5, rng=rng)
    return float(np.exp(ls))


class ExactFlowChain:
    """One chain's state and sweep (see the module).

    ``y`` stacks ``T`` periods of ``n²`` flows time-first; ``struct`` is the
    design's :func:`classify_flow_design`.  ``tune`` sets the warmup schedule
    of the (a, b) directions; ``levels`` adds levels after the filter, and
    ``cell_mask`` (``T × n × n``, True live) removes cells from the likelihood.
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
        levels: Optional[FlowLevels] = None,
        cell_mask: Optional[np.ndarray] = None,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.struct = struct
        self.n, self.T, self.k = struct.n, struct.T, struct.k
        T, n = self.T, self.n
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
        self.full_keys = [tuple(int(i) for i in struct.f_idx[t]) for t in range(T)]
        self.distinct_keys = sorted(set(self.full_keys))

        self.levels = levels if levels is not None else FlowLevels(())
        sets = self.levels.sets
        M = live_cells(self.y, T, n, self.levels, cell_mask)
        self.live = M
        live_y = self.y[M.ravel() > 0]
        self.live_flat = M.ravel() > 0
        self.y_tab = np.unique(live_y, return_counts=True)
        shapes = {"origin": (n,), "destination": (n,), "pair": (n, n), "period": (T,)}
        self.lev = [np.full(shapes[st.kind], st.mu) for st in sets]
        self.tau = [st.scale if st.random else np.inf for st in sets]
        self.unit_live = [np.sum(M, axis=st.axes) > 0 for st in sets]

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
        """``U = A⁻¹X/m`` at the current ``(a, b)``, ``(k, T, n, n)``."""
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
        return cols * np.exp(-(self.a + self.b))

    def _eta_at(self, a: float, b: float, theta: np.ndarray) -> np.ndarray:
        """``η = A⁻¹Xθ/m`` plus the levels at ``(a, b)``, flat over the periods."""
        return (self._filtered_at(a, b, theta) + self._level_sum(self.lev)).ravel()

    def _filtered_at(self, a: float, b: float, theta: np.ndarray) -> np.ndarray:
        """``A⁻¹Xθ/m`` at ``(a, b)``, ``(T, n, n)``."""
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
        return eta * np.exp(-(a + b))

    # --- the levels ----------------------------------------------------------
    def _level_sum(self, lev) -> np.ndarray:
        """The levels' part of ``η``, ``(T, n, n)`` (0 without levels)."""
        out = np.zeros((self.T, self.n, self.n))
        for st, v in zip(self.levels.sets, lev):
            out = out + _grow(v, st.axes)
        return out

    def _level_parts(self, F: np.ndarray) -> list[np.ndarray]:
        """The parts of ``F`` (``(..., T, n, n)``) each level set absorbs.

        Each set in turn takes the live-cell means of what the sets before it
        left; a set holding a baseline passes its first unit's part to the
        first set, so its baseline level stays put.
        """
        sets, parts, rest = self.levels.sets, [], F
        for st in sets:
            part = _live_mean(rest, st.axes, self.live)
            parts.append(part)
            rest = rest - _grow(part, st.axes)
        for i, st in enumerate(sets):
            if self.levels.baseline(i):
                lead = parts[i].shape[: parts[i].ndim - self.lev[i].ndim]
                c = parts[i].reshape(lead + (-1,))[..., 0]
                parts[i] = parts[i] - np.expand_dims(
                    c, tuple(range(-self.lev[i].ndim, 0))
                )
                parts[0] = parts[0] + np.expand_dims(
                    c, tuple(range(-self.lev[0].ndim, 0))
                )
        return parts

    def _level_precisions(self) -> list[float]:
        """Each set's prior precision: ``1/τ²`` random, ``1/scale²`` fixed (0 flat)."""
        return [
            1.0 / self.tau[i] ** 2 if st.random else 1.0 / st.scale**2
            for i, st in enumerate(self.levels.sets)
        ]

    def _level_log_prior(self, lev) -> float:
        """The levels' Normal priors at their current sds (0 for flat sets)."""
        out = 0.0
        for st, v, prec in zip(self.levels.sets, lev, self._level_precisions()):
            if prec > 0:
                out -= 0.5 * prec * float(np.sum((v - st.mu) ** 2))
        return out

    def _level_moves(self, eta: np.ndarray) -> np.ndarray:
        """The levels, the shifts between random sets and their sds; returns ``η``."""
        sets, rng, alpha = self.levels.sets, self.rng, self.alpha
        E = eta.reshape(self.T, self.n, self.n)
        Y = self.y.reshape(E.shape)
        precs = self._level_precisions()
        for i, st in enumerate(sets):
            base = E - _grow(self.lev[i], st.axes)
            new = _level_draw(
                base, Y, self.live, alpha, self.lev[i], st.mu, precs[i], st.axes, rng
            )
            if self.levels.baseline(i):
                new.reshape(-1)[0] = st.mu
            self.lev[i] = new
            E = base + _grow(new, st.axes)
        for i in range(len(sets) - 1):
            si, sj = sets[i], sets[i + 1]
            if not (si.random and sj.random):
                continue
            # c in (l_i + c, l_j − c), η unchanged: a Gaussian conditional.
            pi, pj = precs[i], precs[i + 1]
            prec = self.lev[i].size * pi + self.lev[i + 1].size * pj
            mean = (
                pi * np.sum(si.mu - self.lev[i]) + pj * np.sum(self.lev[i + 1] - sj.mu)
            ) / prec
            c = mean + rng.standard_normal() / np.sqrt(prec)
            self.lev[i], self.lev[i + 1] = self.lev[i] + c, self.lev[i + 1] - c
        for i, st in enumerate(sets):
            if st.random:
                self.tau[i] = _sd_draw(self.tau[i], self.lev[i], st.mu, st.scale, rng)
        return E.ravel()

    def _log_prior(self, theta: np.ndarray, s: float) -> float:
        """The Normal prior of ``β = θ e^{−s}``."""
        return -0.5 * float(np.sum(self.prec0 * (theta * np.exp(-s) - self.mu0) ** 2))

    # --- the sweep -----------------------------------------------------------
    def sweep(self, tuning: bool) -> None:
        """One sweep (see the module)."""
        rng, y, it, live = self.rng, self.y, self.it, self.live.ravel()
        theta, alpha, eta = self.theta, self.alpha, self.eta
        s = self.a + self.b
        sets = self.levels.sets

        # 1. θ | a, b, α along the Fisher information's eigenvectors, the
        # levels following the parts of the moved column they absorb.
        k = self.k
        cols = self._columns()
        parts = self._level_parts(cols)
        for st, part in zip(sets, parts):
            cols = cols - _grow(part, st.axes)
        cols = cols.reshape(k, -1)
        parts = [part.reshape(k, -1) for part in parts]
        if tuning and (it < _FISHER_FIRST or it % _FISHER_EVERY == 0):
            w = alpha * np.exp(-np.logaddexp(0.0, np.log(alpha) - eta)) * live
            H = (cols * w) @ cols.T
            for part, prec in zip(parts, self._level_precisions()):
                H += prec * (part @ part.T)
            H[np.diag_indices(k)] += self.prec0 * np.exp(-2.0 * s)
            lam, V = np.linalg.eigh(H)
            self.th_dirs, self.th_widths = V, 2.0 / np.sqrt(np.maximum(lam, 1e-12))
        UD = self.th_dirs.T @ cols
        LD = [-(self.th_dirs.T @ part) for part in parts]
        del cols, parts
        lev = [v.ravel() for v in self.lev]

        def f_theta(t, d, ud, ld, e, th, lv):
            val = _nb_ll(y, e + t * ud, alpha, live) + self._log_prior(th + t * d, s)
            moved = [
                (v + t * dv).reshape(o.shape) for v, dv, o in zip(lv, ld, self.lev)
            ]
            return val + self._level_log_prior(moved), None

        f = _nb_ll(y, eta, alpha, live) + self._log_prior(theta, s)
        f += self._level_log_prior(self.lev)
        for j in rng.permutation(k):
            d, ud, ld = self.th_dirs[:, j], UD[j], [D[j] for D in LD]
            t, f, _ = slice_along(
                lambda t, a=(d, ud, ld, eta, theta, lev): f_theta(t, *a),
                f,
                None,
                self.th_widths[j],
                rng,
            )
            theta, eta = theta + t * d, eta + t * ud
            lev = [v + t * dv for v, dv in zip(lev, ld)]
        del UD, LD
        self.lev = [v.reshape(o.shape) for v, o in zip(lev, self.lev)]
        if sets:
            eta = self._level_moves(eta)

        # 2. (a, b) with θ sheared along G, the Jacobian e^{-(k+1)(a+b)}, each
        # level's total held.
        k1 = k + 1
        F0 = eta.reshape(self.T, self.n, self.n) - self._level_sum(self.lev)
        hold = [v + part for v, part in zip(self.lev, self._level_parts(F0))]

        def g(z):
            a, b, th = z[0], z[1], z[2:]
            if not (self.ab_lo < a < self.ab_hi and self.ab_lo < b < self.ab_hi):
                return -np.inf, None
            try:
                F = self._filtered_at(a, b, th)
            except FACTORIZATION_ERRORS:
                return -np.inf, None  # ρ at the edge of its support
            lz = [h - part for h, part in zip(hold, self._level_parts(F))]
            e = (F + self._level_sum(lz)).ravel()
            val = _nb_ll(y, e, alpha, live) + self._log_prior(th, a + b) - k1 * (a + b)
            val += self._level_log_prior(lz)
            return (val, (e, lz)) if np.isfinite(val) else (-np.inf, None)

        z = np.r_[self.a, self.b, theta]
        gz = _nb_ll(y, eta, alpha, live) + self._log_prior(theta, s) - k1 * s
        gz += self._level_log_prior(self.lev)
        aux = (eta, self.lev)
        for j in rng.permutation(2):
            d = self.ab_dirs[:, j]
            v = np.r_[d, self.ab_shear @ d]
            t, gz, aux = slice_along(
                lambda t, z=z, v=v: g(z + t * v), gz, aux, self.ab_widths[j], rng
            )
            z = z + t * v
        eta, self.lev = aux
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

        # 3. α | η, over the live cells.
        state = _StructuralState(
            eta=eta, beta=theta, rho=0.0, alpha=alpha, omega=np.empty(0)
        )
        self.alpha = _sample_alpha(
            state,
            y,
            self.core_priors,
            rng=rng,
            cell_mask=self.live_flat,
            y_tab=self.y_tab,
        )
        self.theta, self.eta = theta, eta
        self.it += 1

    def draw(self) -> dict:
        """The current ``rho_d``, ``rho_o``, ``rho_w``, ``beta``, ``alpha`` and levels.

        Each level set adds ``{kind}_effect`` (NaN for dropped units) and, if
        random, ``{kind}_effect_sd``.
        """
        rho_d, rho_o = -np.expm1(-self.a), -np.expm1(-self.b)
        out = {
            "rho_d": rho_d,
            "rho_o": rho_o,
            "rho_w": -rho_d * rho_o,
            "beta": self.theta * np.exp(-(self.a + self.b)),
            "alpha": self.alpha,
        }
        for i, st in enumerate(self.levels.sets):
            out[f"{st.kind}_effect"] = np.where(self.unit_live[i], self.lev[i], np.nan)
            if st.random:
                out[f"{st.kind}_effect_sd"] = self.tau[i]
        return out

    def log_lik(self) -> np.ndarray:
        """Each cell's NB2 log-likelihood at the current state, 0 off the live cells."""
        return np.where(
            self.live_flat, _nb_loglik_pointwise(self.y, self.eta, self.alpha), 0.0
        )


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
    levels: Optional[FlowLevels] = None,
    cell_mask: Optional[np.ndarray] = None,
    keep_pair_draws: bool = True,
) -> dict[str, np.ndarray]:
    """One chain of the exact separable NB flow sampler.

    ``y`` and ``X`` are stacked time-first over ``T`` periods of ``n²`` flows.
    Pass ``struct`` (from :func:`classify_flow_design`) to share the
    classification across chains; ``X`` is then unused.  Returns ``rho_d,
    rho_o, rho_w, beta, alpha, log_lik``, and each level set's draws
    (:meth:`ExactFlowChain.draw`).  Without ``keep_pair_draws`` the pair
    levels come back as ``pair_effect_summary``, their posterior ``{"mean",
    "sd"}`` over every post-warmup sweep.
    """
    rng = np.random.default_rng() if rng is None else rng
    struct = classify_flow_design(X, n, T) if struct is None else struct
    chain = ExactFlowChain(
        y, struct, W_csc, priors, init, tune, rng, rho_lower, rho_upper, levels,
        cell_mask,
    )  # fmt: skip
    n_keep = draws // thin if thin > 0 else draws
    first = chain.draw()
    summary = "pair_effect" if not keep_pair_draws and "pair_effect" in first else None
    out = {
        name: np.empty((n_keep,) + np.shape(value))
        for name, value in first.items()
        if name != summary
    }
    out["log_lik"] = np.empty((n_keep, chain.y.size)) if store_log_lik else None
    if summary is not None:
        mean, m2 = np.zeros_like(first[summary]), np.zeros_like(first[summary])
    for it in range(tune + draws):
        chain.sweep(tuning=it < tune)
        if it >= tune:
            d = chain.draw()
            if summary is not None:
                # Running moments over every post-warmup sweep (Welford).
                v, cnt = d.pop(summary), it - tune + 1
                delta = v - mean
                mean = mean + delta / cnt
                m2 = m2 + delta * (v - mean)
            if (it - tune) % thin == 0 and (j := (it - tune) // thin) < n_keep:
                for name, value in d.items():
                    out[name][j] = value
                if store_log_lik:
                    out["log_lik"][j] = chain.log_lik()
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)
    if summary is not None:
        out["pair_effect_summary"] = {
            "mean": mean,
            "sd": np.sqrt(m2 / max(draws - 1, 1)),
        }
    return out
