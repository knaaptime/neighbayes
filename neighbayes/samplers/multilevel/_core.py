r"""Gibbs sampler for the Gaussian spatial multilevel model.

The model has levels ``ℓ = 0`` (the units, ``θ_0 ≡ y``) through ``L``.  Each
level has a graph ``W_ℓ``, covariates ``X_ℓ``, a noise sd ``σ_ℓ`` and a
process, and each level's effect enters the equation of the level below
through ``c_ℓ = Δ_ℓ θ_{ℓ+1}`` (``Δ_ℓ`` maps a level-ℓ row to its parent;
``c_L = 0``)::

    lag:    θ_ℓ = ρ_ℓ W_ℓ θ_ℓ + X_ℓ β_ℓ + c_ℓ + ε_ℓ
    error:  θ_ℓ = X_ℓ β_ℓ + c_ℓ + v_ℓ,   v_ℓ = λ_ℓ W_ℓ v_ℓ + ε_ℓ
    none:   θ_ℓ = X_ℓ β_ℓ + c_ℓ + ε_ℓ
    ε_ℓ ~ N(0, σ_ℓ² I)

Every residual ``e_ℓ`` is linear in the stacked linear unknowns
``z = (β_0, β_1, θ_1, …, β_L, θ_L)`` and affine in the level's own
autoregressive parameter, so given the ρ's and σ's, ``z`` is jointly Gaussian
with a sparse precision ``P`` whose values are a quadratic polynomial in each
ρ.  The sweep is a partially collapsed Gibbs sampler (van Dyk & Park 2008):

1. ``σ_0² | z, ρ_0`` — conjugate inverse gamma.
2. ``ρ_0 | σ's, ρ_ℓ's`` with ``z`` integrated out.  For a lag at the units
   ``P`` is free of ``ρ_0`` and the log density is ``log|A_0|`` plus a
   quadratic whose coefficients take two solves with ``P``; for an error
   process ``P`` depends on ``λ_0`` and is refactored per evaluation.
3. The upper levels' ``ρ_ℓ`` and ``σ_ℓ``, by one of four schemes
   (``parametrization``):

   * ``"collapsed"`` (default): each from its conditional with ``z``
     integrated out, ``log|A_ℓ| − J_ℓ log σ_ℓ − ½ log|P| + ½ qᵀP⁻¹q`` — one
     CHOLMOD refactorization per evaluation.  Run before step 4.
   * ``"centred"``: given ``z``, ``σ_ℓ`` under its half-t prior (Huang &
     Wand 2013 mixture) and ``ρ_ℓ`` by slice (``log|A_ℓ|`` plus a
     quadratic).  Run after step 4.
   * ``"noncentred"``: with ``ε̃_ℓ = e_ℓ/σ_ℓ`` held, ``θ_ℓ`` is rebuilt from
     ``(ρ_ℓ, σ_ℓ, ε̃_ℓ)`` and the likelihood is level ``ℓ−1``'s equation;
     ``ρ_ℓ`` moves by slice with no Jacobian, ``σ_ℓ`` exactly (``θ_ℓ`` is
     affine in it).  One ``J_ℓ`` solve per evaluation.  Run after step 4.
   * ``"interweave"``: the centred then the non-centred moves (Yu & Meng
     2011).
4. ``z | everything`` — one CHOLMOD draw, straight after the collapsed moves.

The collapsed moves need the Gaussian likelihood; the centred, non-centred
and interwoven ones need only the conditional of ``z`` and carry over to
count likelihoods.  On a three-level lag model with 16 units per group
(n = 576) the collapsed scheme gave 7–14× the ESS of interweaving for the
upper levels' ρ and σ and the best ESS per second; non-centred alone mixed
worst, as expected when each group's data are informative.

This module is the NumPy reference.  :mod:`._jax` compiles the same sweep,
adds a dense Schur-complement path for a small top level and caches each
slice's starting density, and compiles once per model structure.

Drawing all ``β`` jointly with the effects removes the ridge between the
units' intercept and the mean of the effects, which a ``β | θ`` /
``θ | β`` alternation crosses slowly.

References
----------
van Dyk, D. A., & Park, T. (2008). Partially collapsed Gibbs samplers.
*JASA*, 103(482), 790–796.

Yu, Y., & Meng, X.-L. (2011). To center or not to center: that is not the
question — an ancillarity–sufficiency interweaving strategy (ASIS) for
boosting MCMC efficiency. *JCGS*, 20(3), 531–570.

Huang, A., & Wand, M. P. (2013). Simple marginally noninformative prior
distributions for covariance matrices. *Bayesian Analysis*, 8(2), 439–452.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .._utils._group_effects import noncentered_effect_sd
from .._utils._slice import SliceWidthState, slice_sample_1d_adaptive
from .._utils._spatial_normal import CholmodFactor, NotPositiveDefiniteError

PROCESSES = ("lag", "error", "none")
PARAMETRIZATIONS = ("collapsed", "interweave", "centred", "noncentred")


@dataclass
class LevelSpec:
    """One level as the sampler sees it (level 0 is the units).

    Attributes
    ----------
    X : ndarray of shape (J, k)
        Design, Durbin columns included; ``k`` may be 0 above the units.
    W : csr_matrix of shape (J, J) or None
        Row-standardized graph; ``None`` only for ``process="none"``.
    process : {"lag", "error", "none"}
    parent : ndarray of shape (J,) or None
        Index of each row's group at the next level; ``None`` at the top.
    beta_mu, beta_sigma : ndarray of shape (k,)
        Normal prior on ``β_ℓ``.
    rho_lower, rho_upper : float
        Uniform prior support of ``ρ_ℓ`` (or ``λ_ℓ``).
    logdet : callable or None
        ``ρ ↦ log|I − ρW_ℓ|``.
    logdet_method : str or None
        The method behind ``logdet``, so the JAX backend builds the same one.
    logdet_jax : tuple or None
        ``(kind, params)`` from :func:`neighbayes._logdet._jax.logdet_jax_params`; built on
        demand by the JAX backend when absent.
    """

    X: np.ndarray
    W: sp.csr_matrix | None
    process: str
    parent: np.ndarray | None
    beta_mu: np.ndarray
    beta_sigma: np.ndarray
    rho_lower: float = -1.0
    rho_upper: float = 1.0
    logdet: Callable[[float], float] | None = None
    logdet_method: str | None = None
    logdet_jax: tuple | None = None

    @property
    def spatial(self) -> bool:
        return self.process in ("lag", "error")


@dataclass
class MultilevelGibbsPriors:
    """Variance priors shared with the NUTS build.

    ``σ_0² ~ Inv-Γ(sigma2_alpha, sigma2_beta)``; ``σ_ℓ ~ half-t_ν(0, A)`` for
    ``ℓ ≥ 1`` with ``ν = sigma_nu`` and ``A = sigma_scale``.
    """

    sigma2_alpha: float = 2.0
    sigma2_beta: float = 1.0
    sigma_nu: float = 3.0
    sigma_scale: float = 1.0


# ---------------------------------------------------------------------------
# Structure: the stacked linear unknowns and the polynomial precision
# ---------------------------------------------------------------------------


def _embed(nrows: int, dim: int, blocks) -> sp.csr_matrix:
    """Place ``(column_offset, matrix)`` blocks side by side in ``nrows × dim``."""
    rows, cols, vals = [], [], []
    for off, M in blocks:
        M = sp.coo_matrix(M)
        rows.append(M.row)
        cols.append(M.col + off)
        vals.append(M.data)
    if not rows:
        return sp.csr_matrix((nrows, dim))
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nrows, dim),
    )


def incidence(parent: np.ndarray, n_parent: int) -> sp.csr_matrix:
    """``Δ``: the 0/1 matrix mapping each row to its parent group."""
    J = len(parent)
    return sp.csr_matrix(
        (np.ones(J), (np.arange(J), np.asarray(parent))), shape=(J, n_parent)
    )


class MultilevelStructure:
    """Layout of ``z`` and the parameter-free pieces of every residual.

    ``z = (β_0, β_1, θ_1, …, β_L, θ_L)``.  For ``ℓ ≥ 1`` the residual is
    ``e_ℓ = (R0_ℓ + ρ_ℓ R1_ℓ) z``; at the units it is ``e_0 = A_0 y − C z``
    (lag), ``A_0 (y − C z)`` (error) or ``y − C z`` (none), with
    ``C z = X_0 β_0 + Δ_0 θ_1``.  The precision of ``z`` given the ρ's and
    σ's is

        P = Σ_ℓ (M0_ℓ + ρ_ℓ M1_ℓ + ρ_ℓ² M2_ℓ)/σ_ℓ² + (N0 + λ_0 N1 + λ_0² N2)/σ_0² + Λ

    and each term's values are stored aligned to the CSC structure of one
    union pattern, so CHOLMOD analyzes it once per chain.
    """

    def __init__(self, y: np.ndarray, levels: list[LevelSpec]):
        if len(levels) < 2:
            raise ValueError(
                "A multilevel model needs at least one level above the units."
            )
        self.y = np.asarray(y, dtype=np.float64)
        self.levels = levels
        self.L = L = len(levels) - 1
        self.J = [len(self.y)] + [lv.X.shape[0] for lv in levels[1:]]
        self.k = [lv.X.shape[1] for lv in levels]

        # --- layout of z ---
        self.off_b: dict[int, int] = {}
        self.off_t: dict[int, int] = {}
        pos = 0
        self.off_b[0] = pos
        pos += self.k[0]
        for ell in range(1, L + 1):
            self.off_b[ell] = pos
            pos += self.k[ell]
            self.off_t[ell] = pos
            pos += self.J[ell]
        self.dim = dim = pos

        # Δ_ℓ (J_ℓ × J_{ℓ+1}) for ℓ = 0..L−1.
        self.D = [incidence(levels[m].parent, self.J[m + 1]) for m in range(L)]

        # --- residual pieces for ℓ ≥ 1 ---
        self.R0: dict[int, sp.csr_matrix] = {}
        self.R1: dict[int, sp.csr_matrix | None] = {}
        for ell in range(1, L + 1):
            lv = levels[ell]
            J = self.J[ell]
            blocks = [(self.off_t[ell], sp.eye(J, format="csr"))]
            if self.k[ell]:
                blocks.append((self.off_b[ell], -lv.X))
            if ell < L:
                blocks.append((self.off_t[ell + 1], -self.D[ell]))
            R0 = _embed(J, dim, blocks)
            if lv.process == "lag":
                R1 = _embed(J, dim, [(self.off_t[ell], -lv.W)])
            elif lv.process == "error":
                R1 = sp.csr_matrix(-(lv.W @ R0))
            else:
                R1 = None
            self.R0[ell], self.R1[ell] = R0, R1

        # --- the units' design C (n × dim) ---
        blocks = [(self.off_t[1], self.D[0])]
        if self.k[0]:
            blocks.insert(0, (self.off_b[0], levels[0].X))
        self.C = _embed(self.J[0], dim, blocks)
        W0 = levels[0].W
        self.Wy = (W0 @ self.y) if W0 is not None else None
        self.Cty = self.C.T @ self.y
        self.Ctw = self.C.T @ self.Wy if levels[0].process == "lag" else None
        self.WC = sp.csr_matrix(W0 @ self.C) if levels[0].process == "error" else None

        # --- precision terms on one pattern ---
        terms: list[sp.spmatrix] = []
        self._level_terms: dict[int, tuple] = {}
        for ell in range(1, L + 1):
            R0, R1 = self.R0[ell], self.R1[ell]
            M0 = (R0.T @ R0).tocsr()
            if R1 is None:
                trio = (M0, None, None)
            else:
                M1 = (R0.T @ R1 + R1.T @ R0).tocsr()
                M2 = (R1.T @ R1).tocsr()
                trio = (M0, M1, M2)
            self._level_terms[ell] = trio
            terms.extend(t for t in trio if t is not None)
        CtC = (self.C.T @ self.C).tocsr()
        if levels[0].process == "error":
            CtWC = (self.C.T @ self.WC).tocsr()
            unit_trio = (CtC, -(CtWC + CtWC.T).tocsr(), (self.WC.T @ self.WC).tocsr())
        else:
            unit_trio = (CtC, None, None)
        terms.extend(t for t in unit_trio if t is not None)

        lam_diag = np.zeros(dim)
        q_mu = np.zeros(dim)
        for ell, lv in enumerate(levels):
            if self.k[ell]:
                sl = slice(self.off_b[ell], self.off_b[ell] + self.k[ell])
                prec = 1.0 / np.asarray(lv.beta_sigma, dtype=np.float64) ** 2
                lam_diag[sl] = prec
                q_mu[sl] = prec * np.asarray(lv.beta_mu, dtype=np.float64)
        self.q_mu = q_mu

        S = sp.csr_matrix((dim, dim))
        for t in terms:
            S = S + abs(t)
        S = (S + sp.diags(np.asarray(S.sum(axis=1)).ravel() + 1.0)).tocsc()
        S.sort_indices()
        self.pattern = S
        rows = S.indices
        cols = np.repeat(np.arange(dim), np.diff(S.indptr))

        def on_pattern(M):
            if M is None:
                return None
            return np.asarray(sp.csr_matrix(M)[rows, cols], dtype=np.float64).ravel()

        self._vals = {
            ell: tuple(on_pattern(t) for t in trio)
            for ell, trio in self._level_terms.items()
        }
        self._unit_vals = tuple(on_pattern(t) for t in unit_trio)
        diag_mask = rows == cols
        self._lam_vals = np.where(diag_mask, lam_diag[cols], 0.0)

        # --- pieces for the non-centred moves (ℓ ≥ 1) ---
        # θ_ℓ enters level ℓ−1's residual as −G D θ_ℓ with G D = GD0 + ρ GD1.
        self._parent_K: dict[int, tuple] = {}
        self._parent_GD: dict[int, tuple] = {}
        for ell in range(1, L + 1):
            m = ell - 1
            D = self.D[m]
            GD0 = D
            GD1 = (
                sp.csr_matrix(-(levels[m].W @ D))
                if levels[m].process == "error"
                else None
            )
            K0 = (GD0.T @ GD0).tocsr()
            if GD1 is None:
                K = (K0, None, None)
            else:
                K = (K0, (GD0.T @ GD1 + GD1.T @ GD0).tocsr(), (GD1.T @ GD1).tocsr())
            self._parent_GD[ell] = (GD0, GD1)
            self._parent_K[ell] = K

    # -- slices of z -------------------------------------------------------

    def beta(self, z: np.ndarray, ell: int) -> np.ndarray:
        return z[self.off_b[ell] : self.off_b[ell] + self.k[ell]]

    def theta(self, z: np.ndarray, ell: int) -> np.ndarray:
        return z[self.off_t[ell] : self.off_t[ell] + self.J[ell]]

    def set_theta(self, z: np.ndarray, ell: int, value: np.ndarray) -> None:
        z[self.off_t[ell] : self.off_t[ell] + self.J[ell]] = value

    # -- precision and its pieces -----------------------------------------

    def precision(self, rho: np.ndarray, sig2: np.ndarray) -> sp.csc_matrix:
        """``P`` at the given ρ's and σ²'s (``rho[0]`` used only for an error)."""
        vals = self._lam_vals.copy()
        u0, u1, u2 = self._unit_vals
        r0 = float(rho[0])
        unit = u0 if u1 is None else u0 + r0 * (u1 + r0 * u2)
        vals += unit / sig2[0]
        for ell, (v0, v1, v2) in self._vals.items():
            r = float(rho[ell])
            term = v0 if v1 is None else v0 + r * (v1 + r * v2)
            vals += term / sig2[ell]
        P = self.pattern
        return sp.csc_matrix((vals, P.indices, P.indptr), shape=P.shape)

    def new_factor(self) -> CholmodFactor:
        """A CHOLMOD factor analyzed on the union pattern (one per chain)."""
        return CholmodFactor(self.pattern)

    def unit_filtered(self, lam: float) -> np.ndarray:
        """``A_0ᵀA_0 y`` for an error process at the units."""
        W = self.levels[0].W
        u = self.y - lam * self.Wy
        return u - lam * (W.T @ u)

    # -- residuals -----------------------------------------------------------

    def unit_resid(self, z: np.ndarray, rho0: float) -> np.ndarray:
        """``e_0`` at the current state."""
        proc = self.levels[0].process
        Cz = self.C @ z
        if proc == "lag":
            return self.y - rho0 * self.Wy - Cz
        u = self.y - Cz
        if proc == "error":
            return u - rho0 * (self.levels[0].W @ u)
        return u

    def level_parts(
        self, z: np.ndarray, ell: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(R0 z, R1 z)``: ``e_ℓ = R0 z + ρ_ℓ R1 z``."""
        R1 = self.R1[ell]
        return self.R0[ell] @ z, (R1 @ z if R1 is not None else None)

    def level_resid(self, z: np.ndarray, ell: int, rho: float) -> np.ndarray:
        p, s = self.level_parts(z, ell)
        return p if s is None else p + rho * s

    def resid(self, z: np.ndarray, ell: int, rho: np.ndarray) -> np.ndarray:
        if ell == 0:
            return self.unit_resid(z, float(rho[0]))
        return self.level_resid(z, ell, float(rho[ell]))

    # -- the parent likelihood as a quadratic in θ_ℓ -----------------------

    def parent_quadratic(
        self, z: np.ndarray, ell: int, rho: np.ndarray
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """``(K, h)`` with level ``ℓ−1``'s ``‖e‖² = θ_ℓᵀKθ_ℓ − 2hᵀθ_ℓ + const``."""
        m = ell - 1
        r = float(rho[m])
        GD0, GD1 = self._parent_GD[ell]
        K0, K1, K2 = self._parent_K[ell]
        GD = GD0 if GD1 is None else (GD0 + r * GD1).tocsr()
        K = K0 if K1 is None else (K0 + r * K1 + (r * r) * K2).tocsr()
        theta = self.theta(z, ell)
        s = self.resid(z, m, rho) + GD @ theta
        return K, GD.T @ s


# ---------------------------------------------------------------------------
# Per-level solves with A_ℓ(ρ) = I − ρW_ℓ
# ---------------------------------------------------------------------------


class _FilterSolver:
    """Solves with ``I − ρW`` for one level, refactored only when ρ changes."""

    def __init__(self, W: sp.csr_matrix):
        self._W = sp.csc_matrix(W)
        self._I = sp.eye(W.shape[0], format="csc")
        self._rho: float | None = None
        self._lu = None

    def solve(self, rho: float, rhs: np.ndarray) -> np.ndarray:
        from scipy.sparse.linalg import splu

        rho = float(rho)
        if rho != self._rho:
            self._lu = splu((self._I - rho * self._W).tocsc())
            self._rho = rho
        return self._lu.solve(np.asarray(rhs, dtype=np.float64))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MultilevelState:
    """Mutable chain state.  ``rho[0]`` and ``sig2[0]`` belong to the units."""

    z: np.ndarray
    rho: np.ndarray  # (L+1,), 0 where a level has no process
    sig2: np.ndarray  # (L+1,)
    aux: np.ndarray  # (L+1,), Huang–Wand mixture variables (index 0 unused)


def initialize(
    st: MultilevelStructure,
    priors: MultilevelGibbsPriors,
    rng: np.random.Generator,
) -> MultilevelState:
    """Start from least squares at the units and nested group means above."""
    levels, L = st.levels, st.L
    z = np.zeros(st.dim)
    X0 = levels[0].X
    if st.k[0]:
        beta0, *_ = np.linalg.lstsq(X0, st.y, rcond=None)
        z[st.off_b[0] : st.off_b[0] + st.k[0]] = beta0
        r = st.y - X0 @ beta0
    else:
        r = st.y - st.y.mean()
    # θ_ℓ: the mean of the level below's values within each group.
    below = r
    for ell in range(1, L + 1):
        D = st.D[ell - 1]
        counts = np.asarray(D.sum(axis=0)).ravel()
        means = (D.T @ below) / np.maximum(counts, 1.0)
        st.set_theta(z, ell, means)
        below = means
    rho = np.zeros(L + 1)
    for ell, lv in enumerate(levels):
        if lv.spatial:
            half = 0.2 * min(abs(lv.rho_lower), abs(lv.rho_upper), 1.0)
            rho[ell] = rng.uniform(-half, half)
    sig2 = np.empty(L + 1)
    e0 = st.unit_resid(z, rho[0])
    sig2[0] = max(float(e0 @ e0) / len(e0), 1e-6 * max(np.var(st.y), 1e-12))
    for ell in range(1, L + 1):
        e = st.level_resid(z, ell, rho[ell])
        sig2[ell] = max(float(e @ e) / len(e), 1e-4 * sig2[0])
    return MultilevelState(z=z, rho=rho, sig2=sig2, aux=np.ones(L + 1))


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _sample_sigma2_units(
    st: MultilevelStructure,
    state: MultilevelState,
    priors: MultilevelGibbsPriors,
    rng: np.random.Generator,
) -> None:
    e = st.unit_resid(state.z, state.rho[0])
    a = priors.sigma2_alpha + 0.5 * len(e)
    b = priors.sigma2_beta + 0.5 * float(e @ e)
    state.sig2[0] = 1.0 / rng.gamma(a, 1.0 / b)


def _unit_q(st: MultilevelStructure, rho0: float, sig2_0: float) -> np.ndarray:
    """The linear term of ``z``'s full conditional at ``ρ_0``."""
    proc = st.levels[0].process
    if proc == "lag":
        Ct = st.Cty - rho0 * st.Ctw
    elif proc == "error":
        Ct = st.C.T @ st.unit_filtered(rho0)
    else:
        Ct = st.Cty
    return st.q_mu + Ct / sig2_0


def _sample_rho_units(
    st: MultilevelStructure,
    state: MultilevelState,
    factor: CholmodFactor,
    width: SliceWidthState | None,
    rng: np.random.Generator,
) -> None:
    """``ρ_0`` with ``z`` integrated out; ``z`` must be redrawn after it."""
    lv0 = st.levels[0]
    s2 = state.sig2[0]
    y, Wy = st.y, st.Wy

    if lv0.process == "lag":
        factor.factorize(st.precision(state.rho, state.sig2))
        a = st.q_mu + st.Cty / s2
        b = st.Ctw / s2
        Pa, Pb = factor.solve(a), factor.solve(b)
        c0 = -0.5 * (y @ y) / s2 + 0.5 * (a @ Pa)
        c1 = (y @ Wy) / s2 - (a @ Pb)
        c2 = -0.5 * (Wy @ Wy) / s2 + 0.5 * (b @ Pb)

        def log_density(r):
            return lv0.logdet(r) + c0 + r * (c1 + r * c2)

        state.rho[0], _, _, _ = slice_sample_1d_adaptive(
            log_density,
            state.rho[0],
            lower=lv0.rho_lower,
            upper=lv0.rho_upper,
            rng=rng,
            width_state=width,
        )
        return

    if lv0.process == "error":
        rho = state.rho.copy()

        def log_density(lam):
            rho[0] = lam
            q = _unit_q(st, lam, s2)
            u = y - lam * Wy
            try:
                factor.factorize(st.precision(rho, state.sig2))
                quad = 0.5 * (q @ factor.solve(q)) - 0.5 * factor.logdet()
            except NotPositiveDefiniteError:
                return -np.inf  # λ at the edge of its support
            return lv0.logdet(lam) - 0.5 * (u @ u) / s2 + quad

        state.rho[0], _, _, _ = slice_sample_1d_adaptive(
            log_density,
            state.rho[0],
            lower=lv0.rho_lower,
            upper=lv0.rho_upper,
            rng=rng,
            width_state=width,
        )


def log_marginal(
    st: MultilevelStructure,
    rho: np.ndarray,
    sig2: np.ndarray,
    factor: CholmodFactor | None = None,
) -> float:
    """``log p(y | ρ's, σ's)`` with ``z`` integrated out, up to a constant.

    ``Σ log|A_ℓ| − Σ J_ℓ log σ_ℓ − ½ log|P| − ½‖t_0‖²/σ_0² + ½ qᵀP⁻¹q``, the
    sum over levels with a process for the log-determinants and over all
    levels for the scales; ``t_0`` is ``A_0 y`` (or ``y`` with no process).
    """
    factor = factor if factor is not None else st.new_factor()
    factor.factorize(st.precision(rho, sig2))
    q = _unit_q(st, float(rho[0]), float(sig2[0]))
    lv0 = st.levels[0]
    t0 = st.y - rho[0] * st.Wy if lv0.spatial else st.y
    val = (
        -0.5 * factor.logdet() + 0.5 * (q @ factor.solve(q)) - 0.5 * (t0 @ t0) / sig2[0]
    )
    for ell, lv in enumerate(st.levels):
        val -= 0.5 * st.J[ell] * np.log(sig2[ell])
        if lv.spatial:
            val += lv.logdet(float(rho[ell]))
    return float(val)


def _draw_z(
    st: MultilevelStructure,
    state: MultilevelState,
    factor: CholmodFactor,
    rng: np.random.Generator,
) -> None:
    """``z | everything``: one CHOLMOD draw."""
    factor.factorize(st.precision(state.rho, state.sig2))
    state.z = factor.sample(_unit_q(st, state.rho[0], state.sig2[0]), rng=rng)


def _collapsed_level(
    st: MultilevelStructure,
    state: MultilevelState,
    ell: int,
    priors: MultilevelGibbsPriors,
    factor: CholmodFactor,
    width: SliceWidthState | None,
    rng: np.random.Generator,
) -> None:
    """``ρ_ℓ`` then ``σ_ℓ`` from their conditionals with ``z`` integrated out.

    With ``z`` Gaussian given the ρ's and σ's, the marginal log density of a
    level's parameters is ``log|A_ℓ| − J_ℓ log σ_ℓ − ½ log|P| + ½ qᵀP⁻¹q``
    plus terms free of them (``q`` depends on neither); each evaluation is one
    CHOLMOD refactorization.  ``z`` must be redrawn after these moves.
    """
    from .._utils._group_effects import _slice_log_sd

    lv = st.levels[ell]
    q = _unit_q(st, state.rho[0], state.sig2[0])
    rho, sig2 = state.rho.copy(), state.sig2.copy()

    def marginal():
        try:
            factor.factorize(st.precision(rho, sig2))
            val = -0.5 * factor.logdet() + 0.5 * (q @ factor.solve(q))
        except NotPositiveDefiniteError:
            return -np.inf  # ρ or σ at the edge of its support
        return val if np.isfinite(val) else -np.inf

    if lv.spatial:

        def log_density_rho(r):
            rho[ell] = r
            return lv.logdet(r) + marginal()

        state.rho[ell], _, _, _ = slice_sample_1d_adaptive(
            log_density_rho,
            state.rho[ell],
            lower=lv.rho_lower,
            upper=lv.rho_upper,
            rng=rng,
            width_state=width,
        )
        rho[ell] = state.rho[ell]

    nu, A, J = priors.sigma_nu, priors.sigma_scale, st.J[ell]

    def log_density_sd(ls):
        sd = np.exp(ls)
        sig2[ell] = sd * sd
        return (
            -J * ls
            + marginal()
            - 0.5 * (nu + 1.0) * np.log1p(sd * sd / (nu * A * A))
            + ls
        )

    sd = _slice_log_sd(log_density_sd, float(np.sqrt(state.sig2[ell])), A, 0.5, rng)
    state.sig2[ell] = sd * sd


def _centred_level(
    st: MultilevelStructure,
    state: MultilevelState,
    ell: int,
    priors: MultilevelGibbsPriors,
    width: SliceWidthState | None,
    rng: np.random.Generator,
) -> None:
    """``σ_ℓ`` (Huang–Wand) then ``ρ_ℓ`` (slice), given ``z``."""
    lv = st.levels[ell]
    nu, A = priors.sigma_nu, priors.sigma_scale
    p, s = st.level_parts(state.z, ell)
    e = p if s is None else p + state.rho[ell] * s
    state.aux[ell] = 1.0 / rng.gamma(
        0.5 * (nu + 1.0), 1.0 / (nu / state.sig2[ell] + 1.0 / A**2)
    )
    state.sig2[ell] = 1.0 / rng.gamma(
        0.5 * (len(e) + nu), 1.0 / (0.5 * float(e @ e) + nu / state.aux[ell])
    )
    if s is None:
        return
    s2 = state.sig2[ell]
    pp, ps, ss = float(p @ p), float(p @ s), float(s @ s)

    def log_density(r):
        return lv.logdet(r) - 0.5 * (pp + r * (2.0 * ps + r * ss)) / s2

    state.rho[ell], _, _, _ = slice_sample_1d_adaptive(
        log_density,
        state.rho[ell],
        lower=lv.rho_lower,
        upper=lv.rho_upper,
        rng=rng,
        width_state=width,
    )


def _noncentred_level(
    st: MultilevelStructure,
    state: MultilevelState,
    ell: int,
    priors: MultilevelGibbsPriors,
    solver: _FilterSolver | None,
    width: SliceWidthState | None,
    rng: np.random.Generator,
) -> None:
    """``ρ_ℓ`` then ``σ_ℓ`` with ``ε̃_ℓ = e_ℓ/σ_ℓ`` held; ``θ_ℓ`` rebuilt."""
    lv = st.levels[ell]
    z = state.z
    sd = float(np.sqrt(state.sig2[ell]))
    eps = st.level_resid(z, ell, state.rho[ell]) / sd
    # X_ℓβ_ℓ + c_ℓ: the part of θ_ℓ's equation that the move holds.
    mean = np.zeros(st.J[ell])
    if st.k[ell]:
        mean += lv.X @ st.beta(z, ell)
    if ell < st.L:
        mean += st.D[ell] @ st.theta(z, ell + 1)
    K, h = st.parent_quadratic(z, ell, state.rho)
    s2 = state.sig2[ell - 1]

    def parts(r):
        """``(a, b)`` with ``θ_ℓ = a + σ_ℓ b`` at ``ρ_ℓ = r``."""
        if lv.process == "lag":
            return solver.solve(r, mean), solver.solve(r, eps)
        if lv.process == "error":
            return mean, solver.solve(r, eps)
        return mean, eps

    def loglik(theta):
        return -0.5 * (theta @ (K @ theta) - 2.0 * (h @ theta)) / s2

    if lv.spatial:

        def log_density(r):
            try:
                a, b = parts(r)
            except RuntimeError:
                return -np.inf
            val = loglik(a + sd * b)
            return val if np.isfinite(val) else -np.inf

        state.rho[ell], _, _, _ = slice_sample_1d_adaptive(
            log_density,
            state.rho[ell],
            lower=lv.rho_lower,
            upper=lv.rho_upper,
            rng=rng,
            width_state=width,
        )

    a, b = parts(state.rho[ell])
    new_sd = noncentered_effect_sd(
        sd,
        float(b @ (K @ b)) / s2,
        float((h - K @ a) @ b) / s2,
        priors.sigma_scale,
        rng,
        nu=priors.sigma_nu,
    )
    state.sig2[ell] = new_sd * new_sd
    st.set_theta(z, ell, a + new_sd * b)


# ---------------------------------------------------------------------------
# Chain runner
# ---------------------------------------------------------------------------


def unit_loglik_pointwise(
    st: MultilevelStructure, z: np.ndarray, rho0: float, sig2_0: float
) -> np.ndarray:
    """Pointwise log-likelihood of ``y`` given the effects, Jacobian spread over n."""
    e = st.unit_resid(z, rho0)
    n = len(e)
    ll = -0.5 * e * e / sig2_0 - 0.5 * np.log(2.0 * np.pi * sig2_0)
    lv0 = st.levels[0]
    if lv0.spatial:
        ll = ll + lv0.logdet(rho0) / n
    return ll


def run_multilevel_chain(
    st: MultilevelStructure,
    priors: MultilevelGibbsPriors,
    draws: int,
    tune: int,
    *,
    thin: int = 1,
    rng: np.random.Generator | None = None,
    parametrization: str = "collapsed",
    store_theta: bool = True,
    store_log_lik: bool = False,
    chain_id: int = 0,
    progress_manager: object | None = None,
) -> dict[str, np.ndarray]:
    """Run one chain; returns traces keyed by level index.

    Keys: ``rho`` (n_keep, L+1), ``sigma`` (n_keep, L+1), ``beta_{ℓ}``
    (n_keep, k_ℓ), ``theta_{ℓ}`` (n_keep, J_ℓ) when ``store_theta``, and
    ``log_lik`` (n_keep, n) when ``store_log_lik``.
    """
    if parametrization not in PARAMETRIZATIONS:
        raise ValueError(
            f"parametrization must be one of {PARAMETRIZATIONS}, got {parametrization!r}"
        )
    if rng is None:
        rng = np.random.default_rng()
    levels, L = st.levels, st.L
    collapsed = parametrization == "collapsed"
    centred = parametrization in ("interweave", "centred")
    noncentred = parametrization in ("interweave", "noncentred")

    state = initialize(st, priors, rng)
    factor = st.new_factor()
    widths_c = [
        SliceWidthState(w=0.1 * (lv.rho_upper - lv.rho_lower)) if lv.spatial else None
        for lv in levels
    ]
    widths_nc = [
        SliceWidthState(w=0.1 * (lv.rho_upper - lv.rho_lower)) if lv.spatial else None
        for lv in levels
    ]
    solvers = [
        _FilterSolver(lv.W) if (ell >= 1 and lv.spatial) else None
        for ell, lv in enumerate(levels)
    ]

    n_keep = draws // thin
    out: dict[str, np.ndarray] = {
        "rho": np.empty((n_keep, L + 1)),
        "sigma": np.empty((n_keep, L + 1)),
    }
    for ell in range(L + 1):
        if st.k[ell]:
            out[f"beta_{ell}"] = np.empty((n_keep, st.k[ell]))
        if ell >= 1 and store_theta:
            out[f"theta_{ell}"] = np.empty((n_keep, st.J[ell]))
    if store_log_lik:
        out["log_lik"] = np.empty((n_keep, st.J[0]))

    for it in range(tune + draws):
        _sample_sigma2_units(st, state, priors, rng)
        if levels[0].spatial:
            _sample_rho_units(st, state, factor, widths_c[0], rng)
        if collapsed:
            for ell in range(1, L + 1):
                _collapsed_level(st, state, ell, priors, factor, widths_c[ell], rng)
        _draw_z(st, state, factor, rng)
        if centred:
            for ell in range(1, L + 1):
                _centred_level(st, state, ell, priors, widths_c[ell], rng)
        if noncentred:
            for ell in range(1, L + 1):
                _noncentred_level(
                    st, state, ell, priors, solvers[ell], widths_nc[ell], rng
                )

        if it >= tune and (it - tune) % thin == 0:
            j = (it - tune) // thin
            if j < n_keep:
                out["rho"][j] = state.rho
                out["sigma"][j] = np.sqrt(state.sig2)
                for ell in range(L + 1):
                    if st.k[ell]:
                        out[f"beta_{ell}"][j] = st.beta(state.z, ell)
                    if ell >= 1 and store_theta:
                        out[f"theta_{ell}"][j] = st.theta(state.z, ell)
                if store_log_lik:
                    out["log_lik"][j] = unit_loglik_pointwise(
                        st, state.z, state.rho[0], state.sig2[0]
                    )
        if progress_manager is not None:
            progress_manager.update(chain_id, it, tuning=it < tune)
    return out
