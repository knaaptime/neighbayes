r"""Resolvent-gradient sampler for the unrestricted Gaussian flow model.

Model: :math:`A(\rho)\,y = X\beta + \varepsilon`, ``A = I_N - W_F``,
:math:`W_F = \rho_d(I\otimes W) + \rho_o(W\otimes I) + \rho_w(W\otimes W)`,
:math:`\varepsilon \sim N(0, \sigma^2 I_N)`, ``N = n²``.

The conditional log-posterior for the flow parameters ``ρ`` is

.. math::

    \log p(\rho \mid \beta, \sigma^2)
      = \log|A(\rho)| - \frac{1}{2\sigma^2}\lVert A(\rho)y - X\beta\rVert^2
        + \log\pi(\rho),

where — crucially — ``A(ρ)y = y - ρ_d\,W_d y - ρ_o\,W_o y - ρ_w\,W_w y`` is *linear*
in ``ρ`` using the precomputed spatial lags ``W_k y``, so the data term and its
gradient are **exact and O(N)**.  Only the ``log|A|`` Jacobian and its gradient are
non-trivial; these come from the scalable resolvent-Kronecker estimator
(:mod:`neighbayes._logdet._flow_resolvent`), whose relative error *improves* with
``N`` — replacing the noise-amplified ``"traces"`` value method.

``β`` and ``σ²`` have the usual conjugate Gibbs updates given ``ρ`` (``A(ρ)y`` is a
cheap linear transform of the data), so the sampler is MALA-on-``ρ`` within Gibbs.

.. note::
   Per-step this needs a handful of Kronecker GMRES solves; it is intended for the
   GPU path (batched ``n x n`` matmuls) where it becomes competitive.  The exactness
   of the ``ρ`` gradient/target coupling is unit-tested against finite differences of
   the exact log-determinant; end-to-end ρ-ESS/sec is a GPU benchmark (see the plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from ..._logdet._flow_resolvent import (
    FlowKron,
    flow_logdet_value,
    flow_logdet_value_and_grad,
)


def resolve_flow_logdet(W, method: str = "auto"):
    """Pick the unrestricted flow log-determinant backend.

    ``"auto"`` uses the exact trace-moment method
    (:class:`~neighbayes._logdet._flow_kron_traces.FlowKronTraceLogdet`) when ``W``
    is undirected (D-symmetrizable, so its spectrum is real and the Chebyshev
    basis converges up to the stability wall), and the stochastic resolvent
    estimator (``"jax"``) for directed ``W``.  ``"kron_traces"`` forces the exact
    method (Taylor basis for directed ``W``, which is slow near the wall).

    Returns ``(logdet_value_and_grad or None, resolved_method)``; ``None`` means
    each chain's target builds its own resolvent estimator.
    """
    if method == "auto":
        from ..._logdet._config import _is_symmetric_W

        method = "kron_traces" if _is_symmetric_W(W) else "jax"
    if method == "kron_traces":
        from ..._logdet._flow_kron_traces import FlowKronTraceLogdet

        return FlowKronTraceLogdet(W), method
    if method not in ("jax", "numpy"):
        raise ValueError(
            f"logdet_method must be 'auto', 'kron_traces', 'jax' or 'numpy', got {method!r}"
        )
    return None, method


def _default_logdet_value_and_grad(kron, probes):
    """Resolvent value+grad closure sharing frozen probes across a chain.

    Uses relaxed GMRES tolerance (``1e-6``) since the Hutchinson trace estimator
    already carries stochastic error ~1/√(N·P); tightening GMRES below the
    stochastic noise floor wastes iterations without improving accuracy.
    """

    def _fn(rho_d, rho_o, rho_w):
        return flow_logdet_value_and_grad(
            kron, rho_d, rho_o, rho_w, probes=probes, tol=1e-6
        )

    return _fn


def _jax_logdet_value_and_grad(kron, probes, n_quad=8):
    """JAX-native value+grad closure — sparsax solves, JIT-compiled via equinox.

    All P probes are solved in a single batched sparsax call, and the Hutchinson
    accumulation + ray integration are done in JAX.  The entire function is
    JIT-compiled, eliminating Python overhead per MALA step.
    """
    from ..._logdet._flow_resolvent import _make_flow_kron_jax

    fn = _make_flow_kron_jax(kron, probes, n_quad=n_quad)

    def _fn(rho_d, rho_o, rho_w):
        import jax.numpy as jnp

        v, g = fn(jnp.float64(rho_d), jnp.float64(rho_o), jnp.float64(rho_w))
        return float(v), np.asarray(g)

    return _fn


@dataclass
class FlowResolventTarget:
    """ρ-conditional log-posterior and gradient for the unrestricted flow model.

    Parameters
    ----------
    W : array or sparse
        The ``n x n`` (directed, row-standardized) weights matrix.
    y : ndarray (N,)
        Flow observations (``N = n²``).
    X : ndarray (N, k)
        Design matrix.
    logdet_value_and_grad : callable, optional
        ``(rho_d, rho_o, rho_w) -> (value, grad3)`` for ``log|A|``.  Defaults to the
        resolvent estimator with a fixed frozen-probe set (deterministic field).
        Inject an exact implementation for testing.
    rho_bound : float
        Reject ρ with ``|ρ_d|+|ρ_o|+|ρ_w| >= rho_bound`` or ``sum(ρ) >= rho_bound``
        (a conservative row-stochastic stability wall).
    positive : bool
        Additionally reject any ``ρ_k < 0`` (the model-level
        ``restrict_positive=True`` prior: flat on the positive simplex).
    """

    W: object
    y: np.ndarray
    X: np.ndarray
    T: int = 1
    logdet_value_and_grad: object = None
    rho_bound: float = 0.999
    positive: bool = False
    n_probes: int = 48
    seed: int | np.random.SeedSequence = 0
    logdet_method: str = "jax"
    n_quad: int = 8

    def __post_init__(self):
        self.kron = FlowKron(self.W)
        self.n = self.kron.n
        self.Nf = self.kron.N  # per-period flow count = n²
        self.T = int(self.T)
        self.Ntot = self.Nf * self.T
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.y.shape[0] != self.Ntot:
            raise ValueError(
                f"y has length {self.y.shape[0]}, expected Nf*T = {self.Ntot}."
            )
        # Per-period spatial lags W_k y (stacked over T); make the data term
        # linear in ρ.  The panel operator is block-diagonal I_T ⊗ (I_N − W_F),
        # so each period's block gets the cross-sectional W_k.
        self.L = np.column_stack(
            [
                self._period_lag(self.y, self.kron.matvec_Wd),
                self._period_lag(self.y, self.kron.matvec_Wo),
                self._period_lag(self.y, self.kron.matvec_Ww),
            ]
        )
        self.XtX = self.X.T @ self.X
        # Cached Cholesky factor of XtX (used by draw_beta_sigma2).
        self._XtX_cho = scipy.linalg.cho_factor(self.XtX, lower=True)
        self._last_ld_val = 0.0  # cached per-period log|A| for the Jacobian
        self._cached_rho = None  # cache key for logdet reuse after Gibbs
        self._cached_ld_val = 0.0
        self._cached_ld_grad = np.zeros(3)
        if self.logdet_value_and_grad is None:
            rng = np.random.default_rng(self.seed)
            probes = rng.choice([-1.0, 1.0], size=(self.Nf, self.n_probes)).astype(
                np.float64
            )
            if self.logdet_method == "jax":
                self.logdet_value_and_grad = _jax_logdet_value_and_grad(
                    self.kron, probes, n_quad=self.n_quad
                )
            else:
                self.logdet_value_and_grad = _default_logdet_value_and_grad(
                    self.kron, probes
                )

    def _period_lag(self, vec, fn):
        """Apply the per-period flow operator ``fn`` to each length-``Nf`` block."""
        if self.T == 1:
            return fn(vec)
        out = np.empty_like(vec)
        for t in range(self.T):
            s = slice(t * self.Nf, (t + 1) * self.Nf)
            out[s] = fn(vec[s])
        return out

    # -- data helpers ------------------------------------------------------
    def Ay(self, rho):
        """``A(ρ) y = y - L ρ`` (linear in ρ; per-period for panels)."""
        return self.y - self.L @ np.asarray(rho, dtype=np.float64)

    def residual(self, rho, beta):
        """Pointwise Gaussian residual ``r = A(ρ)y − Xβ`` (length ``Ntot``)."""
        return self.Ay(rho) - self.X @ np.asarray(beta, dtype=np.float64)

    def in_bounds(self, rho) -> bool:
        rho = np.asarray(rho, dtype=np.float64)
        if self.positive and np.any(rho < 0.0):
            return False
        return bool(np.abs(rho).sum() < self.rho_bound and rho.sum() < self.rho_bound)

    # -- ρ-conditional target ---------------------------------------------
    def logpost_and_grad(self, rho, beta, sigma2):
        """Return ``(logp, grad3)`` of ``log p(ρ | β, σ²)`` (flat prior in-bounds).

        ``log|A_panel| = T·log|I_N − W_F|`` so the log-determinant value/gradient
        are scaled by ``T`` (``T=1`` for the cross-section).  The per-period
        ``log|A|`` value is cached in ``self._last_ld_val`` so the sampler can attach
        the change-of-variables Jacobian to the ``log_likelihood`` group.

        Caches the logdet ``(value, grad)`` keyed on ρ so that
        :meth:`logpost_cached` can reuse it when only β/σ² have changed
        (after a Gibbs update), avoiding a redundant resolvent computation.
        """
        rho = np.asarray(rho, dtype=np.float64)
        if not self.in_bounds(rho):
            self._cached_rho = None
            return -np.inf, np.zeros(3)
        r = self.Ay(rho) - self.X @ beta
        ld_val, ld_grad = self.logdet_value_and_grad(rho[0], rho[1], rho[2])
        self._last_ld_val = float(ld_val)
        # Cache logdet for reuse after Gibbs β/σ² updates (ρ unchanged)
        self._cached_rho = rho.copy()
        self._cached_ld_val = ld_val
        self._cached_ld_grad = np.asarray(ld_grad, dtype=np.float64)
        logp = self.T * ld_val - 0.5 * float(r @ r) / sigma2
        # d/dρ_k [-||r||²/2σ²] = (rᵀ L_k)/σ²  since ∂r/∂ρ_k = -L_k
        grad = self.T * self._cached_ld_grad + (self.L.T @ r) / sigma2
        return logp, grad

    def logpost_cached(self, rho, beta, sigma2):
        """Recompute ``logp, grad`` using the cached logdet (ρ must be unchanged).

        After a Gibbs β/σ² update the logdet ``(value, grad)`` is identical
        because it depends only on ρ.  This method avoids the expensive
        resolvent computation, cutting the per-iteration GMRES solves in half.
        """
        rho = np.asarray(rho, dtype=np.float64)
        r = self.Ay(rho) - self.X @ beta
        ld_val = self._cached_ld_val
        ld_grad = self._cached_ld_grad
        self._last_ld_val = float(ld_val)
        logp = self.T * ld_val - 0.5 * float(r @ r) / sigma2
        grad = self.T * ld_grad + (self.L.T @ r) / sigma2
        return logp, grad

    # -- conjugate Gibbs updates for β, σ² given ρ ------------------------
    def draw_beta_sigma2(self, rho, rng, a0=1e-3, b0=1e-3):
        """Draw ``(β, σ²)`` from their conjugate conditionals given ρ."""
        e = self.Ay(rho)
        bhat = scipy.linalg.cho_solve(self._XtX_cho, self.X.T @ e)
        resid = e - self.X @ bhat
        sse = float(resid @ resid)
        # σ² | ρ  (inverse-gamma with weak prior)
        a_n = a0 + self.Ntot / 2.0
        b_n = b0 + 0.5 * sse
        sigma2 = 1.0 / rng.gamma(a_n, 1.0 / b_n)
        # β | ρ, σ²  (Normal): bhat + sqrt(σ²) · (L')⁻¹ z
        z = rng.standard_normal(self.X.shape[1])
        beta = bhat + np.sqrt(sigma2) * scipy.linalg.solve_triangular(
            self._XtX_cho[0], z, lower=True, trans="T"
        )
        return beta, sigma2


def _lag_matrix(kron: FlowKron, M: np.ndarray, which: str, T: int = 1) -> np.ndarray:
    """Apply ``W_k`` to each column of ``M`` (shape ``(Nf*T, k)``), per period."""
    M = np.asarray(M, dtype=np.float64)
    fn = {"d": kron.matvec_Wd, "o": kron.matvec_Wo, "w": kron.matvec_Ww}[which]
    Nf = kron.N
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        if T == 1:
            out[:, j] = fn(col)
        else:
            for t in range(T):
                s = slice(t * Nf, (t + 1) * Nf)
                out[s, j] = fn(col[s])
    return out


@dataclass
class SEMFlowResolventTarget:
    """λ-conditional log-posterior/gradient for the unrestricted SEM flow model.

    Model: ``y = Xβ + u``, ``A(λ) u = ε`` with ``A = I_N − W_F(λ)``,
    ``ε ~ N(0, σ² I_N)``.  The whitened residual is ``r = A(y − Xβ) = ỹ − X̃β`` with
    ``ỹ = Ay``, ``X̃ = AX`` — both cheap linear-in-λ transforms via the precomputed
    lags ``W_k y`` and ``W_k X``.  ``β`` is a GLS draw on ``(X̃, ỹ)``; the log-det
    ``log|A|`` and its gradient come from the resolvent estimator (same as the SAR
    flow).  Implements the ``logpost_and_grad`` / ``draw_beta_sigma2`` interface so it
    reuses :func:`run_flow_resolvent_gibbs`.
    """

    W: object
    y: np.ndarray
    X: np.ndarray
    T: int = 1
    logdet_value_and_grad: object = None
    rho_bound: float = 0.999
    positive: bool = False
    n_probes: int = 48
    seed: int | np.random.SeedSequence = 0
    logdet_method: str = "jax"
    n_quad: int = 8

    def __post_init__(self):
        self.kron = FlowKron(self.W)
        self.n = self.kron.n
        self.Nf = self.kron.N
        self.T = int(self.T)
        self.Ntot = self.Nf * self.T
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.y.shape[0] != self.Ntot:
            raise ValueError(
                f"y has length {self.y.shape[0]}, expected Nf*T = {self.Ntot}."
            )
        # Per-period lags of y (Ntot,3) and of each design column (Ntot,k).
        y2 = self.y[:, None]
        self.L_y = np.column_stack(
            [
                _lag_matrix(self.kron, y2, "d", self.T).ravel(),
                _lag_matrix(self.kron, y2, "o", self.T).ravel(),
                _lag_matrix(self.kron, y2, "w", self.T).ravel(),
            ]
        )
        self.WdX = _lag_matrix(self.kron, self.X, "d", self.T)
        self.WoX = _lag_matrix(self.kron, self.X, "o", self.T)
        self.WwX = _lag_matrix(self.kron, self.X, "w", self.T)
        self._last_ld_val = 0.0  # cached per-period log|A| for the Jacobian
        self._cached_rho = None  # cache key for logdet reuse after Gibbs
        self._cached_ld_val = 0.0
        self._cached_ld_grad = np.zeros(3)
        if self.logdet_value_and_grad is None:
            rng = np.random.default_rng(self.seed)
            probes = rng.choice([-1.0, 1.0], size=(self.Nf, self.n_probes)).astype(
                np.float64
            )
            if self.logdet_method == "jax":
                self.logdet_value_and_grad = _jax_logdet_value_and_grad(
                    self.kron, probes, n_quad=self.n_quad
                )
            else:
                self.logdet_value_and_grad = _default_logdet_value_and_grad(
                    self.kron, probes
                )

    def in_bounds(self, rho) -> bool:
        rho = np.asarray(rho, dtype=np.float64)
        if self.positive and np.any(rho < 0.0):
            return False
        return bool(np.abs(rho).sum() < self.rho_bound and rho.sum() < self.rho_bound)

    def _whiten(self, rho):
        """Return ``(ỹ, X̃) = (Ay, AX)`` at the given λ (linear in λ)."""
        rho = np.asarray(rho, dtype=np.float64)
        ytil = self.y - self.L_y @ rho
        Xtil = self.X - (rho[0] * self.WdX + rho[1] * self.WoX + rho[2] * self.WwX)
        return ytil, Xtil

    def residual(self, rho, beta):
        """Pointwise whitened residual ``r = A(λ)(y − Xβ)`` (length ``Ntot``)."""
        ytil, Xtil = self._whiten(rho)
        return ytil - Xtil @ np.asarray(beta, dtype=np.float64)

    def logpost_and_grad(self, rho, beta, sigma2):
        rho = np.asarray(rho, dtype=np.float64)
        if not self.in_bounds(rho):
            self._cached_rho = None
            return -np.inf, np.zeros(3)
        ytil, Xtil = self._whiten(rho)
        r = ytil - Xtil @ beta
        ld_val, ld_grad = self.logdet_value_and_grad(rho[0], rho[1], rho[2])
        self._last_ld_val = float(ld_val)
        # Cache logdet for reuse after Gibbs β/σ² updates (ρ unchanged)
        self._cached_rho = rho.copy()
        self._cached_ld_val = ld_val
        self._cached_ld_grad = np.asarray(ld_grad, dtype=np.float64)
        logp = self.T * ld_val - 0.5 * float(r @ r) / sigma2
        # W_k e = W_k y - W_k X β ;  d/dλ_k[-||r||²/2σ²] = (rᵀ W_k e)/σ²
        WkE = np.column_stack(
            [
                self.L_y[:, 0] - self.WdX @ beta,
                self.L_y[:, 1] - self.WoX @ beta,
                self.L_y[:, 2] - self.WwX @ beta,
            ]
        )
        grad = self.T * self._cached_ld_grad + (WkE.T @ r) / sigma2
        return logp, grad

    def logpost_cached(self, rho, beta, sigma2):
        """Recompute ``logp, grad`` using the cached logdet (ρ must be unchanged)."""
        rho = np.asarray(rho, dtype=np.float64)
        ytil, Xtil = self._whiten(rho)
        r = ytil - Xtil @ beta
        ld_val = self._cached_ld_val
        ld_grad = self._cached_ld_grad
        self._last_ld_val = float(ld_val)
        logp = self.T * ld_val - 0.5 * float(r @ r) / sigma2
        WkE = np.column_stack(
            [
                self.L_y[:, 0] - self.WdX @ beta,
                self.L_y[:, 1] - self.WoX @ beta,
                self.L_y[:, 2] - self.WwX @ beta,
            ]
        )
        grad = self.T * ld_grad + (WkE.T @ r) / sigma2
        return logp, grad

    def draw_beta_sigma2(self, rho, rng, a0=1e-3, b0=1e-3):
        ytil, Xtil = self._whiten(rho)
        XtX = Xtil.T @ Xtil
        # Cholesky-based solve: faster and more stable than forming inv(XtX).
        L = np.linalg.cholesky(XtX)  # XtX = L L'
        bhat = scipy.linalg.cho_solve((L, True), Xtil.T @ ytil)
        resid = ytil - Xtil @ bhat
        sse = float(resid @ resid)
        sigma2 = 1.0 / rng.gamma(a0 + self.Ntot / 2.0, 1.0 / (b0 + 0.5 * sse))
        # β draw: bhat + sqrt(σ²) · (L')⁻¹ z  (cov = σ² (L L')⁻¹ = σ² (L')⁻¹ L⁻¹)
        z = rng.standard_normal(self.X.shape[1])
        beta = bhat + np.sqrt(sigma2) * scipy.linalg.solve_triangular(
            L, z, lower=True, trans="T"
        )
        return beta, sigma2


def run_flow_resolvent_gibbs(
    target: FlowResolventTarget,
    *,
    draws: int = 1000,
    tune: int = 500,
    step_size: float = 5e-4,
    target_accept: float = 0.57,
    rho_init=None,
    seed: int = 0,
    compute_log_likelihood: bool = True,
    progressbar: bool = True,
    chain_idx: int = 0,
    progress_manager=None,
):
    """MALA-on-ρ within Gibbs for the unrestricted Gaussian flow model.

    Returns a dict of stacked posterior draws for ``rho_d, rho_o, rho_w, beta, sigma``,
    plus ``log_abs_det`` (per-draw ``log|A|``) and — when ``compute_log_likelihood`` —
    ``loglik`` (per-draw pointwise log-likelihood *including* the change-of-variables
    Jacobian, spread uniformly), so the caller can build a ``log_likelihood`` group
    for LOO/WAIC.  The MALA step size is adapted during tuning toward ``target_accept``
    (≈0.57 is optimal for MALA).  This is the reference driver; for large ``N`` run the
    GMRES solves on GPU.
    """
    rng = np.random.default_rng(seed)
    rho = np.zeros(3) if rho_init is None else np.asarray(rho_init, dtype=np.float64)
    beta, sigma2 = target.draw_beta_sigma2(rho, rng)
    logp, grad = target.logpost_and_grad(rho, beta, sigma2)

    out = {k: [] for k in ("rho_d", "rho_o", "rho_w", "beta", "sigma", "log_abs_det")}
    if compute_log_likelihood:
        out["loglik"] = []
    _half_log_2pi = 0.5 * np.log(2.0 * np.pi)
    log_eps = np.log(step_size)
    accepted_window = 0
    window = 0
    for it in range(tune + draws):
        eps = np.exp(log_eps)
        # Save cached logdet so we can restore it if the MALA proposal is
        # rejected (logpost_and_grad overwrites _cached_ld_val with the
        # proposed ρ's logdet).
        _saved_ld_val = target._cached_ld_val
        _saved_ld_grad = target._cached_ld_grad
        _saved_cached_rho = target._cached_rho
        # --- MALA proposal on ρ ---
        prop = rho + eps * grad + np.sqrt(2 * eps) * rng.standard_normal(3)
        logp_p, grad_p = target.logpost_and_grad(prop, beta, sigma2)
        accept = 0.0
        if np.isfinite(logp_p):
            q_fwd = -np.sum((prop - rho - eps * grad) ** 2) / (4 * eps)
            q_bwd = -np.sum((rho - prop - eps * grad_p) ** 2) / (4 * eps)
            log_alpha = (logp_p - logp) + (q_bwd - q_fwd)
            accept = min(1.0, np.exp(min(0.0, log_alpha)))
            if np.log(rng.uniform()) < log_alpha:
                rho, logp, grad = prop, logp_p, grad_p
            else:
                # Proposal rejected: restore the cache to the current ρ's logdet.
                target._cached_ld_val = _saved_ld_val
                target._cached_ld_grad = _saved_ld_grad
                target._cached_rho = _saved_cached_rho
        # --- step-size adaptation (during tuning only) ---
        if it < tune:
            accepted_window += accept
            window += 1
            if window == 50:
                rate = accepted_window / window
                log_eps += 0.5 * (rate - target_accept)  # nudge toward target
                accepted_window = 0
                window = 0
        # --- Gibbs β, σ² ---
        beta, sigma2 = target.draw_beta_sigma2(rho, rng)
        # ρ unchanged after Gibbs β/σ² update — reuse cached logdet (50% fewer GMRES solves)
        logp, grad = target.logpost_cached(rho, beta, sigma2)
        if it >= tune:
            out["rho_d"].append(rho[0])
            out["rho_o"].append(rho[1])
            out["rho_w"].append(rho[2])
            out["beta"].append(beta.copy())
            out["sigma"].append(np.sqrt(sigma2))
            # log|A| for this draw (cached by the last logpost_and_grad call above).
            ld_val = target._last_ld_val
            out["log_abs_det"].append(target.T * ld_val)
            if compute_log_likelihood:
                # Pointwise Gaussian log-density of the whitened/filtered residual,
                # plus the joint change-of-variables Jacobian T·log|A| spread evenly
                # across all Ntot observations (matches the PyMC-path convention).
                r = target.residual(rho, beta)
                ll = (
                    -_half_log_2pi
                    - 0.5 * np.log(sigma2)
                    - 0.5 * (r * r) / sigma2
                    + (target.T * ld_val) / target.Ntot
                )
                out["loglik"].append(ll)
        # --- progress bar ---
        if progress_manager is not None:
            progress_manager.update(chain_idx, it, tuning=it < tune)
    return {k: np.asarray(v) for k, v in out.items()}


def _sample_flow_chains(
    target_cls,
    W,
    y,
    X,
    *,
    param_prefix: str,
    T: int,
    draws: int,
    tune: int,
    chains: int,
    step_size: float,
    n_probes: int,
    coord_names,
    logdet_value_and_grad,
    random_seed,
    compute_log_likelihood: bool = True,
    progressbar: bool = True,
    n_jobs: int = -1,
    logdet_method: str = "auto",
    n_quad: int = 8,
    positive: bool = False,
):
    """Run ``chains`` MALA-within-Gibbs chains for a flow target → InferenceData.

    ``param_prefix`` is ``"rho"`` (SAR flow) or ``"lam"`` (SEM flow); λ and ρ are
    otherwise interchangeable (same resolvent log-det, same sampler).  The returned
    ``InferenceData`` carries the per-draw Jacobian ``log|A|`` in ``sample_stats`` and
    — when ``compute_log_likelihood`` — a pointwise ``log_likelihood`` group (Gaussian
    density + change-of-variables Jacobian) so ``az.loo`` / ``az.waic`` work directly.

    When ``parallel=True``, chains are dispatched via ``run_chains`` (joblib
    process-based parallelism with shared-memory progress bars), matching the
    cross-sectional Gibbs sampler infrastructure.
    """
    import arviz as az

    from ._chain_runner import run_chains

    k = np.asarray(X).shape[1]
    if coord_names is None:
        coord_names = [f"x{j}" for j in range(k)]

    from .._utils._seeds import spawn_chain_seeds

    seeds = spawn_chain_seeds(random_seed, chains)

    if logdet_value_and_grad is None:
        # Built once and shared by every chain: the exact method's trace moments
        # are the one-time cost.
        logdet_value_and_grad, logdet_method = resolve_flow_logdet(W, logdet_method)

    def _chain_fn(chain_id, seed, progress_manager=None, chain_id_kw=0):
        target = target_cls(
            W,
            y,
            X,
            T=T,
            logdet_value_and_grad=logdet_value_and_grad,
            positive=positive,
            n_probes=n_probes,
            seed=seed,
            logdet_method=logdet_method,
            n_quad=n_quad,
        )
        return run_flow_resolvent_gibbs(
            target,
            draws=draws,
            tune=tune,
            step_size=step_size,
            seed=seed,
            compute_log_likelihood=compute_log_likelihood,
            progressbar=progressbar,
            chain_idx=chain_id,
            progress_manager=progress_manager,
        )

    posts = run_chains(
        chain_fn=_chain_fn,
        n_chains=chains,
        seeds=seeds,
        n_jobs=n_jobs,
        progressbar=progressbar,
        parallel=n_jobs != 1,
        draws=draws,
        tune=tune,
        model_type="flow",
    )

    def _stack(key):
        return np.stack([p[key] for p in posts], axis=0)

    posterior = {
        f"{param_prefix}_d": _stack("rho_d"),
        f"{param_prefix}_o": _stack("rho_o"),
        f"{param_prefix}_w": _stack("rho_w"),
        "sigma": _stack("sigma"),
        "beta": _stack("beta"),
    }
    # Per-draw Jacobian log|A| (= T·log|I_N − W_F|) is always attached so the
    # change-of-variables correction is available on the arviz object.
    sample_stats = {"log_abs_det": _stack("log_abs_det")}
    log_likelihood = {"obs": _stack("loglik")} if compute_log_likelihood else None
    return az.from_dict(
        posterior=posterior,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
        coords={"coefficient": list(coord_names)},
        dims={"beta": ["coefficient"]},
    )


def sample_flow_resolvent(
    W,
    y,
    X,
    *,
    T: int = 1,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    step_size: float = 5e-4,
    n_probes: int = 48,
    n_quad: int = 8,
    coord_names=None,
    logdet_value_and_grad=None,
    random_seed=None,
    compute_log_likelihood: bool = True,
    progressbar: bool = True,
    n_jobs: int = -1,
    logdet_method: str = "auto",
    restrict_positive: bool = False,
):
    """Sample the unrestricted **SAR** flow posterior → ``arviz.InferenceData``.

    Builds a :class:`FlowResolventTarget` from ``(W, y, X)`` and runs ``chains``
    MALA-within-Gibbs chains, packaging ``rho_d, rho_o, rho_w, beta, sigma``.  ``T>1``
    handles the panel (stacked over ``T`` periods; log-det scaled by ``T``).
    ``logdet_method`` selects the log-det backend (see :func:`resolve_flow_logdet`;
    ``"auto"`` is exact for undirected ``W``); ``logdet_value_and_grad`` overrides it
    with any ``(ρ_d, ρ_o, ρ_w) -> (value, grad)`` callable.  With ``restrict_positive`` the flat ρ prior is
    truncated to ``ρ_k ≥ 0`` (the model-level positivity constraint).  The per-draw
    Jacobian ``log|A|`` is attached in ``sample_stats``; with
    ``compute_log_likelihood`` (default) a pointwise ``log_likelihood`` group
    (Gaussian density + Jacobian) is added for LOO/WAIC — set it ``False`` to skip
    the ``(chains, draws, N)`` allocation at very large ``N``.
    """
    return _sample_flow_chains(
        FlowResolventTarget,
        W,
        y,
        X,
        param_prefix="rho",
        T=T,
        draws=draws,
        tune=tune,
        chains=chains,
        step_size=step_size,
        n_probes=n_probes,
        coord_names=coord_names,
        logdet_value_and_grad=logdet_value_and_grad,
        random_seed=random_seed,
        compute_log_likelihood=compute_log_likelihood,
        progressbar=progressbar,
        n_jobs=n_jobs,
        logdet_method=logdet_method,
        n_quad=n_quad,
        positive=restrict_positive,
    )


def sample_sem_flow_resolvent(
    W,
    y,
    X,
    *,
    T: int = 1,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    step_size: float = 5e-4,
    n_probes: int = 48,
    n_quad: int = 8,
    coord_names=None,
    logdet_value_and_grad=None,
    random_seed=None,
    compute_log_likelihood: bool = True,
    progressbar: bool = True,
    n_jobs: int = -1,
    logdet_method: str = "auto",
    restrict_positive: bool = False,
):
    """Sample the unrestricted **SEM** flow posterior → ``arviz.InferenceData``.

    Same resolvent log-det and sampler as the SAR flow; the data term uses the
    whitened residual ``A(y−Xβ)`` and a GLS ``β`` draw.  ``T>1`` handles the panel.
    With ``restrict_positive`` the flat λ prior is truncated to ``λ_k ≥ 0``.
    Packages ``lam_d, lam_o, lam_w, beta, sigma`` with the per-draw Jacobian ``log|A|``
    in ``sample_stats`` and (default) a pointwise ``log_likelihood`` group for LOO/WAIC.
    """
    return _sample_flow_chains(
        SEMFlowResolventTarget,
        W,
        y,
        X,
        param_prefix="lam",
        T=T,
        draws=draws,
        tune=tune,
        chains=chains,
        step_size=step_size,
        n_probes=n_probes,
        coord_names=coord_names,
        logdet_value_and_grad=logdet_value_and_grad,
        random_seed=random_seed,
        compute_log_likelihood=compute_log_likelihood,
        progressbar=progressbar,
        n_jobs=n_jobs,
        logdet_method=logdet_method,
        n_quad=n_quad,
        positive=restrict_positive,
    )


def _poly_terms(degree: int):
    """Exponent tuples ``(a, b, c)`` for all 3-variable monomials up to ``degree``."""
    terms = [(0, 0, 0)]
    for total in range(1, int(degree) + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                terms.append((a, b, total - a - b))
    return terms


def _poly_design(R, terms, mu, sd):
    """Centered/scaled monomial design matrix for points ``R`` (shape ``(M, 3)``)."""
    Z = (np.asarray(R, dtype=np.float64) - mu) / sd
    cols = [(Z[:, 0] ** a) * (Z[:, 1] ** b) * (Z[:, 2] ** c) for (a, b, c) in terms]
    return np.column_stack(cols)


def attach_flow_log_abs_det(
    idata,
    W,
    *,
    T: int = 1,
    param_names=("rho_d", "rho_o", "rho_w"),
    n_probes: int = 16,
    n_quad: int = 6,
    seed: int = 0,
    n_surrogate: int = 64,
    surrogate_degree: int = 3,
    exact_max_n: int = 1200,
):
    """Attach the per-draw flow Jacobian ``T·log|A(ρ)|`` to ``idata.sample_stats``.

    A **diagnostic** for the count (Negative-Binomial) flow models, whose discrete
    sampling density carries no ``|A|`` change-of-variables term — so, unlike the
    Gaussian flow models, this value must *not* enter the pointwise
    ``log_likelihood`` (that would bias LOO/WAIC).  It is instead exposed under
    ``sample_stats["log_abs_det"]`` for inspection and reporting, matching the
    ``log_abs_det`` the Gaussian resolvent sampler already records.

    The per-node ``log|A(ρ)|`` uses the right tool for the scale: an **exact
    eigenvalue** double-sum when ``n ≤ exact_max_n`` (cheap and exact for small W —
    all the cross-sectional flows and every test), and the scalable **resolvent**
    value (frozen probes) only for the large directed W where eigendecomposition is
    infeasible.  Because ``log|A(ρ)|`` is *analytic* in the three flow parameters and
    the posterior ρ occupy a tiny compact region, when there are more draws than nodes
    the value is evaluated at ``n_surrogate`` representative ρ-nodes and a low-degree
    polynomial surrogate (``surrogate_degree``) is fit and predicted at every draw —
    collapsing thousands of per-draw evaluations to a few dozen, with surrogate error
    far below the value's own accuracy.  ``param_names`` selects the posterior flow
    parameters (``rho_*`` for SAR, ``lam_*`` for SEM).  Written in place; ``idata`` is
    returned.
    """
    import xarray as xr

    post = idata.posterior
    # Best-effort diagnostic: models without the three spatial-filter parameters
    # (e.g. the aspatial NegBinFlow, where A = I so log|A| = 0) have nothing to
    # report, so skip rather than raise.
    if not all(name in post.data_vars for name in param_names):
        return idata
    rd = np.asarray(post[param_names[0]].values)  # (chain, draw)
    ro = np.asarray(post[param_names[1]].values)
    rw = np.asarray(post[param_names[2]].values)
    shape = rd.shape[:2]
    R = np.column_stack([rd.ravel(), ro.ravel(), rw.ravel()])  # (M_all, 3)
    m_all = R.shape[0]

    kron = FlowKron(W)
    if kron.n <= int(exact_max_n):
        # Exact eigenvalue double-sum: log|A| = Σ_ij log|1 − (ρ_o λ_i + ρ_d λ_j
        # + ρ_w λ_i λ_j)|.  Directed W → complex spectrum; |·| is the modulus.
        Wd = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=np.float64)
        lam = np.linalg.eigvals(Wd)
        li, lj = lam[:, None], lam[None, :]
        lij = li * lj

        def _value_at(points):
            out = np.empty(len(points), dtype=np.float64)
            for i, (a, b, c) in enumerate(points):
                out[i] = float(
                    np.sum(np.log(np.abs(1.0 - (b * li + a * lj + c * lij))))
                )
            return out
    else:
        from ..._jax_dispatch import _sparsax_available

        rng = np.random.default_rng(seed)
        probes = rng.choice([-1.0, 1.0], size=(kron.N, int(n_probes))).astype(
            np.float64
        )

        if _sparsax_available():
            # JAX-native path: JIT-compiled, reuses sparsax's cached analysis
            from ..._logdet._flow_resolvent import _make_flow_kron_jax

            jax_fn = _make_flow_kron_jax(kron, probes, n_quad=n_quad)

            def _value_at(points):
                import jax.numpy as jnp

                results = []
                for p in points:
                    v, _ = jax_fn(
                        jnp.float64(p[0]), jnp.float64(p[1]), jnp.float64(p[2])
                    )
                    results.append(float(v))
                return np.array(results)
        else:

            def _value_at(points):
                return np.array(
                    [
                        flow_logdet_value(
                            kron,
                            float(p[0]),
                            float(p[1]),
                            float(p[2]),
                            probes=probes,
                            n_quad=n_quad,
                        )
                        for p in points
                    ]
                )

    if m_all <= int(n_surrogate):
        # Few enough draws to evaluate directly.
        vals = _value_at(R)
    else:
        # Fit a smooth polynomial surrogate on a spread-out node subset, predict all.
        terms = _poly_terms(surrogate_degree)
        mu = R.mean(0)
        sd = R.std(0)
        sd[sd == 0.0] = 1.0
        node_idx = np.linspace(0, m_all - 1, int(n_surrogate)).astype(int)
        nodes = R[node_idx]
        node_vals = _value_at(nodes)
        coef, *_ = np.linalg.lstsq(
            _poly_design(nodes, terms, mu, sd), node_vals, rcond=None
        )
        vals = _poly_design(R, terms, mu, sd) @ coef

    lad = (T * vals).reshape(shape)
    da = xr.DataArray(lad, dims=("chain", "draw"), name="log_abs_det")
    if "sample_stats" in idata.groups():
        idata.sample_stats["log_abs_det"] = da
    else:
        idata.add_groups({"sample_stats": xr.Dataset({"log_abs_det": da})})
    return idata
