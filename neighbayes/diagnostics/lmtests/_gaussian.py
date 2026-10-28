r"""The Gaussian LM tests as score tests of one model.

Every Gaussian test is a score test of the general nesting model

.. math::
    A\mathbf y = Z\delta + \mathbf u, \qquad B\mathbf u = \varepsilon,
    \qquad \varepsilon \sim N(0, \sigma^2 I),

with :math:`A = I - \rho W`, :math:`B = I - \lambda W` and :math:`Z = [X,
WX]`, whose :math:`WX` coefficients are :math:`\theta`, at a null that fixes
some of :math:`(\theta, \rho, \lambda)` at zero and fits the rest.  With
:math:`\varepsilon = B(A\mathbf y - Z\delta)`, :math:`G_A = WA^{-1}` and
:math:`G_B = WB^{-1}`, the scores are

.. math::
    g_\delta = Z^\top B^\top\varepsilon/\sigma^2, \quad
    g_\rho = -\mathrm{tr}\,G_A + (BW\mathbf y)^\top\varepsilon/\sigma^2, \quad
    g_\lambda = -\mathrm{tr}\,G_B + (W\mathbf u)^\top\varepsilon/\sigma^2, \quad
    g_{\sigma^2} = -\frac{n}{2\sigma^2} + \frac{\varepsilon^\top\varepsilon}{2\sigma^4},

and, :math:`A` and :math:`B` commuting (both are polynomials in :math:`W`), the
expected information is (Anselin 1988)

.. math::
    J_{\delta\delta} &= (BZ)^\top BZ/\sigma^2, \quad
    J_{\delta\rho} = (BZ)^\top m/\sigma^2, \quad m = BG_AZ\delta, \quad
    J_{\delta\lambda} = J_{\delta\sigma^2} = 0,\\
    J_{\rho\rho} &= \mathrm{tr}(G_A^\top G_A + G_A^2) + m^\top m/\sigma^2, \quad
    J_{\rho\lambda} = \mathrm{tr}(G_A^\top G_B + G_AG_B), \quad
    J_{\lambda\lambda} = \mathrm{tr}(G_B^\top G_B + G_B^2),\\
    J_{\rho\sigma^2} &= \mathrm{tr}\,G_A/\sigma^2, \quad
    J_{\lambda\sigma^2} = \mathrm{tr}\,G_B/\sigma^2, \quad
    J_{\sigma^2\sigma^2} = n/(2\sigma^4).

The scores are evaluated at every posterior draw of the null fit and the
information at its posterior mean.  At a draw the scores of the null's own
parameters are not zero, as they are at the restricted maximum, so a tested
score correlated with them carries their posterior spread, which an
information concentrated on them leaves out: the lag score through the part of
:math:`WZ\delta` in the span of :math:`Z` (the intercept's :math:`\beta_0 W\mathbf
1`, for a row-standardized :math:`W`), every score at a spatial null through
the spatial parameter.  Each test therefore uses the Neyman-orthogonal score
against all the parameters the null estimates, :math:`F` (Doğan, Taşpınar and
Bera 2021),

.. math::
    g^*_T = g_T - J_{TF}J_{FF}^{-1}g_F, \qquad
    J^*_{TT} = J_{TT} - J_{TF}J_{FF}^{-1}J_{FT},

which is the score at the maximum to first order, with the efficient
information.  A robust test also removes the directions :math:`R` that its
alternative leaves at zero (Bera and Yoon 1993):

.. math::
    g_{T\cdot R} = g^*_T - J^*_{TR}{J^*_{RR}}^{-1}g^*_R, \qquad
    V = J^*_{TT} - J^*_{TR}{J^*_{RR}}^{-1}J^*_{RT}.

A test robust only to parameters the null estimates is the plain test.  The
statistic is the test at the posterior-mean parameters, :math:`g^*(\bar\theta)'
V^{-1} g^*(\bar\theta)`, the analogue of the score test at the estimate: the
posterior mean of the per-draw statistic exceeds it by the score's posterior
variance in the metric of :math:`V`, which is not part of the test and
matters where :math:`V` is small (a robust test against a nearly collinear
direction).

Panels
------
A panel stacks its periods, :math:`W` acting as :math:`I_T \otimes W`.  The
fixed-effects designs are the Lee and Yu (2010) transformed likelihood, which
the demeaned data give exactly: every trace is the per-period trace times the
Jacobian multiplier (:math:`T - 1` with unit effects), time effects remove the
direction :math:`\mathbf 1` (the traces of :math:`JAJB` with :math:`J = I -
\mathbf 1\mathbf 1^\top/N`, and :math:`W` followed by the cross-sectional
demeaning), and :math:`n` is the number of transformed observations.  Random
unit effects :math:`\alpha_i \sim N(0, \sigma_\alpha^2)` are coefficients on
the unit dummies whose score and information carry their prior, as at the
posterior mode the penalized score is what vanishes.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .core import _get_posterior_draws, _resolve_X_for_beta, _safe_inv


@dataclass
class _Design:
    """The data and dimensions of a Gaussian null, cross-section or panel."""

    y: np.ndarray  # (n,)
    X: np.ndarray  # (n, k_x), the columns beta multiplies besides WX
    WX: np.ndarray  # (n, k_wx)
    durbin: bool  # whether beta covers WX
    W: sp.csr_matrix  # (N, N)
    N: int
    T: int
    m: float  # trace multiplier: the periods the Jacobian counts
    shift: float  # coefficient of the time effects' -log(1 - rho) term
    n_eff: float  # observations the variance counts
    unit_idx: np.ndarray | None = None  # random unit effects' units
    lags: tuple[str, ...] = ()  # dynamic lag coefficients, last in X

    @property
    def time_fe(self) -> bool:
        return self.shift != 0.0

    def lag(self, v: np.ndarray, transpose: bool = False) -> np.ndarray:
        """``(I_T ⊗ W) v`` on the observation axis (last), demeaned under time effects."""
        Wm = self.W.T if transpose else self.W
        lead = v.shape[:-1]
        r = v.reshape(-1, self.N)
        out = np.asarray(Wm @ r.T).T
        if self.time_fe:
            out = out - out.mean(axis=1, keepdims=True)
        return out.reshape(*lead, self.T * self.N)

    def unit_sum(self, v: np.ndarray) -> np.ndarray:
        """``D^T v`` for the unit dummies ``D``, on the observation axis (last)."""
        lead = v.shape[:-1]
        return v.reshape(*lead, self.T, self.N).sum(axis=-2)


def _design(model, beta: np.ndarray) -> _Design:
    """The null's design; a dynamic panel's lag terms are columns of ``X``.

    A dynamic panel conditions on its first period: ``y`` and ``X`` are the
    later ``T - 1`` periods, and ``phi y_{t-1}`` (with ``theta W y_{t-1}`` in
    the dynamic SDM) are predetermined regressors.
    """
    post = model.inference_data.posterior
    dynamic = getattr(model, "_design", None) in ("dynamic", "restricted")
    if getattr(model, "_design", None) == "restricted":
        raise NotImplementedError(
            "The restricted dynamic SDM ties the lagged spillover to -rho phi; "
            "its LM tests are not derived."
        )
    if dynamic:
        y = np.asarray(model._y_dyn, dtype=np.float64)
        X_all, WX = model._X_dyn, model._WX_dyn
    else:
        y = np.asarray(model._y, dtype=np.float64)
        X_all = model._X
        WX = getattr(model, "_WX", np.empty((y.size, 0)))
    n = y.size
    WX = np.asarray(WX, dtype=np.float64)
    if WX.ndim != 2:
        WX = np.empty((n, 0))
    k_wx = WX.shape[1]
    if dynamic:
        k = X_all.shape[1]
        durbin = beta.shape[1] == k + k_wx and k_wx > 0
        X = np.asarray(X_all, dtype=np.float64)
    else:
        Zb = _resolve_X_for_beta(model, beta)
        if Zb.shape != (n, beta.shape[1]):
            raise NotImplementedError(
                f"The Gaussian LM tests need beta over X or [X, WX]; "
                f"{type(model).__name__}'s posterior has {beta.shape[1]} "
                "coefficients."
            )
        durbin = k_wx > 0 and np.array_equal(Zb[:, Zb.shape[1] - k_wx :], WX)
        X = Zb[:, : Zb.shape[1] - k_wx] if durbin else Zb
    W = sp.csr_matrix(model._W_sparse, dtype=np.float64)
    lags: tuple[str, ...] = ()
    panel = getattr(model, "_is_panel", False) is True
    if dynamic:
        N = int(model._N)
        T = n // N
        m, shift, n_eff = float(T), 0.0, float(n)
        lags = ("phi",) + (("theta",) if "theta" in post else ())
        cols = [model._y_lag] + ([model._Wy_lag] if "theta" in post else [])
        X = np.hstack([X, np.column_stack(cols)])
    elif panel:
        N, T = int(model._N), int(model._T)
        m, shift = float(model._jacobian_T), float(model._jacobian_shift)
        n_eff = float(model._n_effective)
    else:
        N, T, m, shift, n_eff = n, 1, 1.0, 0.0, float(n)
    if N * T != n or W.shape != (N, N):
        raise NotImplementedError(
            f"The Gaussian LM tests need a static design; {type(model).__name__} "
            f"has {n} observations and a {W.shape[0]}-unit W."
        )
    unit_idx = None
    if panel and not dynamic and "unit_effect" in post:
        unit_idx = np.asarray(model._unit_idx)
    return _Design(
        y,
        np.ascontiguousarray(X),
        WX,
        durbin,
        W,
        N,
        T,
        m,
        shift,
        n_eff,
        unit_idx,
        lags,
    )


def resolvent_traces(W, r: float) -> dict:
    r"""Exact traces of :math:`G = (I - rW)^{-1}W` and its products with :math:`W`.

    Returns ``tr_G``, ``tr_G2`` (:math:`\mathrm{tr}\,G^2`), ``tr_GtG``
    (:math:`\|G\|_F^2`), ``tr_WG`` and ``tr_WtG``.  One sparse factorization of
    :math:`I - rW` is reused for chunked column solves :math:`G_{:,J} =
    (I - rW)^{-1}W_{:,J}`, so the traces are exact and eigenvalue-free (an
    eigendecomposition gives the transpose traces only for a normal
    :math:`W`, which a row-standardized one generally is not).
    """
    W = sp.csr_matrix(W)
    n = W.shape[0]
    W_csc = W.tocsc()
    solve = sp.linalg.splu((sp.identity(n, format="csc") - r * W_csc).tocsc()).solve
    out = dict.fromkeys(("tr_G", "tr_G2", "tr_GtG", "tr_WG", "tr_WtG"), 0.0)
    chunk = max(1, min(512, n))
    for j0 in range(0, n, chunk):
        j1 = min(j0 + chunk, n)
        diag = (j0 + np.arange(j1 - j0), np.arange(j1 - j0))
        Wc = W_csc[:, j0:j1].toarray()
        Gc = solve(Wc)
        WGc = np.asarray(W @ Gc)
        out["tr_G"] += float(Gc[diag].sum())
        out["tr_GtG"] += float(np.sum(Gc * Gc))
        out["tr_WtG"] += float(np.sum(Wc * Gc))
        out["tr_WG"] += float(WGc[diag].sum())
        out["tr_G2"] += float(solve(WGc)[diag].sum())
    return out


def _traces(W, r: float, time_fe: bool) -> dict:
    r"""The traces of :func:`resolvent_traces` (``G = W`` at ``r = 0``).

    Under time effects each :math:`\mathrm{tr}(AB)` is :math:`\mathrm{tr}(JAJB)
    = \mathrm{tr}(AB) - (\mathbf 1^\top AB\mathbf 1 + \mathbf 1^\top BA\mathbf
    1)/N + (\mathbf 1^\top A\mathbf 1)(\mathbf 1^\top B\mathbf 1)/N^2` and
    :math:`\mathrm{tr}\,G` is :math:`\mathrm{tr}(JG)`.
    """
    W = sp.csr_matrix(W)
    n = W.shape[0]
    if r == 0.0:
        tr_W2 = float(W.multiply(W.T).sum())
        tr_WtW = float(W.multiply(W).sum())
        out = {
            "tr_G": float(W.diagonal().sum()),
            "tr_G2": tr_W2,
            "tr_GtG": tr_WtW,
            "tr_WG": tr_W2,
            "tr_WtG": tr_WtW,
        }
    else:
        out = resolvent_traces(W, r)
    if not time_fe:
        return out
    one = np.ones(n)
    w1, wt1 = np.asarray(W @ one), np.asarray(W.T @ one)
    if r == 0.0:
        g1, gt1 = w1, wt1
    else:
        A = (sp.identity(n, format="csc") - r * W).tocsc()
        g1 = sp.linalg.spsolve(A, w1)
        gt1 = np.asarray(W.T @ sp.linalg.spsolve(A.T.tocsc(), one))
    s_g, s_w = float(g1.sum()), float(w1.sum())

    def jj(tr_AB, A1, At1, B1, Bt1, sA, sB):
        return tr_AB - (At1 @ B1 + Bt1 @ A1) / n + sA * sB / n**2

    return {
        "tr_G": out["tr_G"] - s_g / n,
        "tr_G2": jj(out["tr_G2"], g1, gt1, g1, gt1, s_g, s_g),
        "tr_GtG": jj(out["tr_GtG"], gt1, g1, g1, gt1, s_g, s_g),
        "tr_WG": jj(out["tr_WG"], w1, wt1, g1, gt1, s_w, s_g),
        "tr_WtG": jj(out["tr_WtG"], wt1, w1, g1, gt1, s_w, s_g),
    }


@dataclass
class GaussianScores:
    """Per-draw scores and posterior-mean information of one Gaussian null fit.

    The parameter order is ``delta`` (``k_x`` columns of ``X``, then ``k_wx``
    of ``WX``), the ``k_u`` random unit effects, ``rho``, ``lam``, ``sigma2``;
    ``free`` indexes the ones the null estimates.
    """

    k_x: int
    k_wx: int
    k_u: int
    free: np.ndarray
    scores: np.ndarray  # (draws, p)
    info: np.ndarray  # (p, p)
    score_bar: np.ndarray  # (p,), at the posterior-mean parameters

    def index(self, name: str) -> list[int]:
        p = self.k_x + self.k_wx + self.k_u
        return {
            "theta": list(range(self.k_x, self.k_x + self.k_wx)),
            "rho": [p],
            "lam": [p + 1],
        }[name]


_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def gaussian_scores(model) -> GaussianScores:
    """Scores and information of a fitted Gaussian null, computed once per posterior.

    The null is read from the posterior: ``rho`` or ``lam`` present means the
    null estimates it, ``beta`` covering ``[X, WX]`` that it estimates
    ``theta``, and ``unit_effect`` that it has random unit effects.
    """
    idata = model.inference_data
    cached = _CACHE.get(model)
    if cached is not None and cached[0] is idata:
        return cached[1]

    post = idata.posterior
    beta = _get_posterior_draws(idata, "beta")
    d = _design(model, beta)
    D = beta.shape[0]
    y, W = d.y, d.W
    k_x, k_wx = d.X.shape[1], d.WX.shape[1]
    Z = np.hstack([d.X, d.WX])
    p_z = k_x + k_wx
    k_b = k_x - len(d.lags)  # the columns of X that beta multiplies
    lag_draws = [_get_posterior_draws(idata, name).reshape(D, 1) for name in d.lags]
    wx_draws = beta[:, k_b:] if d.durbin else np.zeros((D, k_wx))
    delta = np.hstack([beta[:, :k_b], *lag_draws, wx_draws])
    has_rho, has_lam = "rho" in post, "lam" in post
    if has_rho and has_lam:
        raise NotImplementedError(
            "The Gaussian LM tests take a null with one spatial parameter."
        )
    rho = _get_posterior_draws(idata, "rho").reshape(-1) if has_rho else np.zeros(D)
    lam = _get_posterior_draws(idata, "lam").reshape(-1) if has_lam else np.zeros(D)
    sigma2 = _get_posterior_draws(idata, "sigma").reshape(-1) ** 2
    units = d.unit_idx is not None
    k_u = d.N if units else 0
    if units:
        alpha = _get_posterior_draws(idata, "unit_effect")
        s2_alpha = _get_posterior_draws(idata, "unit_effect_sd").reshape(-1) ** 2
    # The last row is the posterior mean, at which the statistic is evaluated.
    r_bar, l_bar = float(rho.mean()), float(lam.mean())
    d_bar, s2 = delta.mean(axis=0), float(sigma2.mean())
    delta, rho, lam = np.vstack([delta, d_bar]), np.r_[rho, r_bar], np.r_[lam, l_bar]
    sigma2 = np.r_[sigma2, s2]
    if units:
        alpha = np.vstack([alpha, alpha.mean(axis=0)])
        s2_alpha = np.r_[s2_alpha, s2_alpha.mean()]

    Wy = d.lag(y)
    WWy = d.lag(Wy)
    u = y[None, :] - rho[:, None] * Wy[None, :] - delta @ Z.T
    if units:
        u = u - alpha[:, d.unit_idx]
    Wu = d.lag(u)
    eps = u - lam[:, None] * Wu
    Bt_eps = eps - lam[:, None] * d.lag(eps, transpose=True)

    def jacobian_trace(par):
        """``-d/dr`` of the Jacobian ``m log|I - rW| - shift log(1 - r)``."""
        grad = np.asarray(model._logdet_grad_numpy_vec_fn(par))
        return -d.m * grad - d.shift / (1.0 - par)

    # At zero the Jacobian's derivative needs no logdet: m tr(W) - shift.
    tr_0 = d.m * float(W.diagonal().sum()) - d.shift
    tr_GA = jacobian_trace(rho) if has_rho else tr_0
    tr_GB = jacobian_trace(lam) if has_lam else tr_0
    BWy = Wy[None, :] - lam[:, None] * WWy[None, :]
    blocks = [Bt_eps @ Z / sigma2[:, None]]
    if units:
        blocks.append(d.unit_sum(Bt_eps) / sigma2[:, None] - alpha / s2_alpha[:, None])
    blocks += [
        (-tr_GA + np.sum(BWy * eps, axis=1) / sigma2)[:, None],
        (-tr_GB + np.sum(Wu * eps, axis=1) / sigma2)[:, None],
        (-d.n_eff / (2.0 * sigma2) + np.sum(eps * eps, axis=1) / (2.0 * sigma2**2))[
            :, None
        ],
    ]
    scores = np.hstack(blocks)

    # Information at the posterior mean.

    def B(v):
        return v - l_bar * d.lag(v)

    BZ = B(Z.T).T
    mean_y = Z @ d_bar
    if units:
        mean_y = mean_y + alpha[-1][d.unit_idx]
    if has_rho:
        lu = sp.linalg.splu((sp.identity(d.N, format="csc") - r_bar * W).tocsc())
        mean_y = lu.solve(mean_y.reshape(d.T, d.N).T).T.reshape(-1)
    mvec = B(d.lag(mean_y))
    w_tr = _traces(W, 0.0, d.time_fe)
    if has_rho:
        tA = tAB = _traces(W, r_bar, d.time_fe)
        tB = w_tr
    elif has_lam:
        tB = tAB = _traces(W, l_bar, d.time_fe)
        tA = w_tr
    else:
        tA = tB = tAB = w_tr
    p_u = p_z + k_u
    p = p_u + 3
    J = np.zeros((p, p))
    J[:p_z, :p_z] = BZ.T @ BZ / s2
    J[:p_z, p_u] = J[p_u, :p_z] = BZ.T @ mvec / s2
    if units:
        Bn = (sp.identity(d.N) - l_bar * W).toarray()
        J[:p_z, p_z:p_u] = d.unit_sum(BZ.T) @ Bn / s2
        J[p_z:p_u, :p_z] = J[:p_z, p_z:p_u].T
        J[p_z:p_u, p_z:p_u] = d.T * (Bn.T @ Bn) / s2 + np.eye(d.N) / s2_alpha[-1]
        J[p_z:p_u, p_u] = J[p_u, p_z:p_u] = Bn.T @ d.unit_sum(mvec) / s2
    J[p_u, p_u] = d.m * (tA["tr_GtG"] + tA["tr_G2"]) + mvec @ mvec / s2
    J[p_u, p_u + 1] = J[p_u + 1, p_u] = d.m * (tAB["tr_WtG"] + tAB["tr_WG"])
    J[p_u + 1, p_u + 1] = d.m * (tB["tr_GtG"] + tB["tr_G2"])
    J[p_u, p_u + 2] = J[p_u + 2, p_u] = d.m * tA["tr_G"] / s2
    J[p_u + 1, p_u + 2] = J[p_u + 2, p_u + 1] = d.m * tB["tr_G"] / s2
    J[p_u + 2, p_u + 2] = d.n_eff / (2.0 * s2**2)

    free = list(range(k_x)) + (list(range(k_x, p_z)) if d.durbin else [])
    free += list(range(p_z, p_u))
    free += ([p_u] if has_rho else []) + ([p_u + 1] if has_lam else []) + [p_u + 2]
    out = GaussianScores(k_x, k_wx, k_u, np.asarray(free), scores[:-1], J, scores[-1])
    _CACHE[model] = (idata, out)
    return out


def efficient_lm(
    g: np.ndarray,
    J: np.ndarray,
    free,
    T: list[int],
    R: list[int],
    *,
    test_name: str,
) -> tuple[np.ndarray, float]:
    """Per-draw LM of the efficient (and robust) score, and the statistic.

    ``g`` holds one row per draw and a last row at the posterior-mean
    parameters, ``J`` the information there; ``free`` indexes the
    parameters the null estimates, ``T`` the tested ones and ``R`` those the
    test is robust to.
    """
    F = np.asarray(free)
    J_FF_inv = _safe_inv(J[np.ix_(F, F)], f"J_FF ({test_name})")

    def orth(S, U):
        """Scores of ``S`` and their information with ``U``, net of ``F``."""
        g_S = g[:, S] - g[:, F] @ (J_FF_inv @ J[np.ix_(F, S)])
        J_SU = J[np.ix_(S, U)] - J[np.ix_(S, F)] @ J_FF_inv @ J[np.ix_(F, U)]
        return g_S, J_SU

    g_T, V = orth(T, T)
    if R:
        g_R, J_RR = orth(R, R)
        _, J_TR = orth(T, R)
        coef = _safe_inv(J_RR, f"J_RR ({test_name})") @ J_TR.T
        g_T = g_T - g_R @ coef
        V = V - J_TR @ coef
    V_inv = _safe_inv(V, f"V ({test_name})")
    LM = np.einsum("di,ij,dj->d", g_T, V_inv, g_T)
    return LM[:-1], float(LM[-1])


def orthogonal_lm(
    model, tested: tuple[str, ...], robust: tuple[str, ...] = (), *, test_name: str
) -> tuple[np.ndarray, float, int]:
    """The efficient (and robust) score test: per-draw LM, statistic and df.

    ``tested`` and ``robust`` name directions among ``"theta"`` (the ``WX``
    block), ``"rho"`` and ``"lam"``.  The statistic is the test at the
    posterior-mean parameters.
    """
    gs = gaussian_scores(model)
    free = set(gs.free.tolist())
    T = [i for name in tested for i in gs.index(name)]
    if not T:
        raise ValueError(
            "Model has no WX columns. The WX test requires at least one "
            "spatially lagged covariate. Ensure the model was constructed "
            "with a W matrix and durbin_vars."
        )
    if free & set(T):
        raise ValueError(
            f"{test_name} tests {', '.join(tested)} = 0, which "
            f"{type(model).__name__} estimates."
        )
    # A direction the null estimates is already orthogonalized away.
    R = [i for name in robust for i in gs.index(name) if i not in free]
    # The posterior-mean score rides along as the last row.
    g = np.vstack([gs.scores, gs.score_bar])
    LM, statistic = efficient_lm(g, gs.info, gs.free, T, R, test_name=test_name)
    return LM, statistic, len(T)
