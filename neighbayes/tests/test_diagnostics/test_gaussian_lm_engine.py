"""The Gaussian LM tests' score engine (:mod:`neighbayes.diagnostics.lmtests._gaussian`).

The scores must be the gradient of the general nesting model's
log-likelihood, the information its expected outer product, and every suite
test must hold its size at its own null when the intercept is large and the
covariates spatially smooth, where a score that carries the posterior spread
of the null's parameters (the defect these tests guard against) is inflated
many times over.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.diagnostics.lmtests import bayesian_lm_lag_test
from neighbayes.diagnostics.lmtests._gaussian import _CACHE, gaussian_scores
from neighbayes.tests.test_diagnostics._lm_mocks import gaussian_mock

N = 30


def _weights(seed=0):
    rng = np.random.default_rng(seed)
    adj = rng.random((N, N)) < 0.15
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    for i in range(N):
        if not adj[i].any():
            adj[i, (i + 1) % N] = adj[(i + 1) % N, i] = True
    return adj / adj.sum(1, keepdims=True)


WD = _weights()
W = sp.csr_matrix(WD)
X = np.c_[np.ones(N), np.random.default_rng(1).standard_normal((N, 2))]
WX = np.asarray(W @ X[:, 1:])
Z = np.hstack([X, WX])
KZ = Z.shape[1]


def _loglik(y, delta, rho, lam, s2):
    A = np.eye(N) - rho * WD
    B = np.eye(N) - lam * WD
    eps = B @ (A @ y - Z @ delta)
    return (
        -0.5 * N * np.log(2 * np.pi * s2)
        + np.linalg.slogdet(A)[1]
        + np.linalg.slogdet(B)[1]
        - eps @ eps / (2 * s2)
    )


def _fd_score(y, delta, rho, lam, s2, h=1e-6):
    theta = np.r_[delta, rho, lam, s2]

    def f(t):
        return _loglik(y, t[:KZ], t[KZ], t[KZ + 1], t[KZ + 2])

    eye = np.eye(theta.size) * h
    return np.array([(f(theta + e) - f(theta - e)) / (2 * h) for e in eye])


def _mock(y, beta, rho=None, lam=None, s2=1.0):
    m = gaussian_mock(X, W)
    m._y, m._X, m._WX, m._W_sparse = y, X, WX, W
    post = {"beta": np.asarray(beta)[None, None, :], "sigma": np.array([[np.sqrt(s2)]])}
    if rho is not None:
        post["rho"] = np.array([[rho]])
    if lam is not None:
        post["lam"] = np.array([[lam]])
    m.inference_data = az.from_dict({"posterior": post})
    return m


B3 = [1.0, 0.3, -0.2]
B5 = [1.0, 0.3, -0.2, 0.4, 0.1]
NULLS = {
    "OLS": dict(beta=B3),
    "SLX": dict(beta=B5),
    "SAR": dict(beta=B3, rho=0.35),
    "SEM": dict(beta=B3, lam=0.45),
    "SDM": dict(beta=B5, rho=-0.3),
    "SDEM": dict(beta=B5, lam=0.5),
}


@pytest.mark.parametrize("null", NULLS)
def test_scores_are_the_loglikelihood_gradient(null):
    c = NULLS[null]
    y = np.random.default_rng(2).standard_normal(N) + 2.0
    gs = gaussian_scores(_mock(y, c["beta"], c.get("rho"), c.get("lam"), s2=1.7))
    beta = np.asarray(c["beta"])
    delta = beta if beta.size == KZ else np.r_[beta, 0.0, 0.0]
    ref = _fd_score(y, delta, c.get("rho", 0.0), c.get("lam", 0.0), 1.7)
    np.testing.assert_allclose(gs.scores[0], ref, rtol=1e-6, atol=1e-6)


def test_a_direction_the_null_estimates_is_refused():
    y = np.random.default_rng(3).standard_normal(N)
    with pytest.raises(ValueError, match="estimates"):
        bayesian_lm_lag_test(_mock(y, B3, rho=0.2))


@pytest.mark.slow
@pytest.mark.parametrize("null", ["OLS", "SAR", "SEM"])
def test_information_is_the_scores_expected_outer_product(null):
    """Monte Carlo ``E[g g']`` at the truth, against the closed form."""
    c = NULLS[null]
    delta = np.r_[c["beta"], 0.0, 0.0]
    rho, lam, s2 = c.get("rho", 0.0), c.get("lam", 0.0), 1.7
    A = np.eye(N) - rho * WD
    B = np.eye(N) - lam * WD
    rng = np.random.default_rng(4)
    sims = 4000
    G = np.empty((sims, KZ + 3))
    for s in range(sims):
        eps = rng.standard_normal(N) * np.sqrt(s2)
        y = np.linalg.solve(A, Z @ delta + np.linalg.solve(B, eps))
        G[s] = _fd_score(y, delta, rho, lam, s2, h=1e-5)
    J_mc = G.T @ G / sims
    _CACHE.clear()
    J = gaussian_scores(
        _mock(np.zeros(N), c["beta"], c.get("rho"), c.get("lam"), s2)
    ).info
    scale = np.sqrt(np.outer(np.diag(J), np.diag(J)))
    # Each standardized entry has MC sd ~ 1.4/sqrt(sims) ~ 0.022.
    assert np.max(np.abs(J - J_mc) / scale) < 0.1


def _smooth_data(null, seed, n_side=30):
    from neighbayes import dgp

    Wg = dgp.simulate_sar(n_side=n_side, seed=0)["W_sparse"].tocsr()
    n = Wg.shape[0]
    rng = np.random.default_rng(seed)
    filt = sp.linalg.splu((sp.identity(n) - 0.8 * Wg).tocsc())
    Zs = filt.solve(rng.standard_normal((n, 2)))
    Zs = (Zs - Zs.mean(0)) / Zs.std(0)
    Xs = np.c_[np.ones(n), Zs]
    mean = Xs @ np.array([3.0, 1.0, -0.5])
    if null in ("SLX", "SDM", "SDEM"):
        mean = mean + np.asarray(Wg @ Zs) @ np.array([0.5, 0.3])
    eps = rng.standard_normal(n)
    A = sp.linalg.splu((sp.identity(n) - 0.4 * Wg).tocsc())
    if null in ("SAR", "SDM"):
        y = A.solve(mean + eps)
    elif null in ("SEM", "SDEM"):
        y = mean + A.solve(eps)
    else:
        y = mean + eps
    return y, Xs, Wg


@pytest.mark.slow
@pytest.mark.parametrize("null", ["OLS", "SLX", "SAR", "SEM", "SDM", "SDEM"])
def test_suites_hold_their_size_at_their_null(null):
    """Mean LM/df near one at the null; the defect gave 3 to 105."""
    import neighbayes.models as M

    reps = 6
    stats = []
    for seed in range(reps):
        y, Xs, Wg = _smooth_data(null, seed)
        m = getattr(M, null)(y=y, X=Xs, W=Wg)
        kw = dict(
            draws=400, tune=300, chains=2, progressbar=False, random_seed=seed, n_jobs=1
        )
        if null in ("OLS", "SLX"):
            kw["cores"] = 1
        m.fit(**kw)
        df = m.spatial_diagnostics()
        stats.append(df["statistic"] / df["df"])
    mean = np.mean(stats, axis=0)
    assert np.all(mean < 2.5), dict(zip(stats[0].index, np.round(mean, 2)))


# ---------------------------------------------------------------------------
# Panels: pooled, Lee-Yu fixed effects, random effects
# ---------------------------------------------------------------------------

NP, TP = 12, 4
WN = _weights(seed=5)[:NP, :NP]
WN = np.where(WN.sum(1, keepdims=True) > 0, WN, np.roll(np.eye(NP), 1, axis=1))
WN = (WN > 0) / (WN > 0).sum(1, keepdims=True)
WNS = sp.csr_matrix(WN)
I_T = np.eye(TP)
XP = np.c_[np.ones(NP * TP), np.random.default_rng(6).standard_normal((NP * TP, 2))]
WXP = np.kron(I_T, WN) @ XP[:, 1:]


def _demean(v, unit, time):
    r = v.reshape(TP, NP, *v.shape[1:])
    if unit:
        r = r - r.mean(0, keepdims=True)
    if time:
        r = r - r.mean(1, keepdims=True)
    return r.reshape(v.shape)


def _panel_parts(unit, time):
    """Demeaned design, Jacobian multiplier and shift, effective n."""
    m = TP - 1 if unit else TP
    return (
        _demean(XP, unit, time),
        _demean(WXP, unit, time),
        m,
        float(m) if time else 0.0,
        (NP - int(time)) * (TP - int(unit)),
    )


def _panel_loglik(
    y, X, WX, delta, rho, lam, s2, m, shift, n_eff, time, alpha=None, s2a=None
):
    """The Lee-Yu transformed (or random-effects) log-likelihood, densely."""
    J = np.eye(NP) - np.ones((NP, NP)) / NP
    L = np.kron(I_T, J @ WN) if time else np.kron(I_T, WN)
    Zp = np.hstack([X, WX])
    u = y - rho * (L @ y) - Zp @ delta
    if alpha is not None:
        u = u - np.tile(alpha, TP)
    eps = u - lam * (L @ u)
    A, B = np.eye(NP) - rho * WN, np.eye(NP) - lam * WN
    ll = (
        -0.5 * n_eff * np.log(2 * np.pi * s2)
        + m * (np.linalg.slogdet(A)[1] + np.linalg.slogdet(B)[1])
        - shift * (np.log(1 - rho) + np.log(1 - lam))
        - eps @ eps / (2 * s2)
    )
    if alpha is not None:
        ll -= 0.5 * alpha @ alpha / s2a
    return ll


def _panel_mock(
    y, X, WX, beta, m, shift, n_eff, rho=None, lam=None, s2=1.0, alpha=None, s2a=None
):
    from neighbayes.tests.test_diagnostics._lm_mocks import panel_mock

    mo = panel_mock(X, WNS, NP, TP)
    mo._y, mo._X, mo._WX, mo._W_sparse = y, X, WX, WNS
    mo._jacobian_T, mo._jacobian_shift, mo._n_effective = m, shift, n_eff
    post = {"beta": np.asarray(beta)[None, None, :], "sigma": np.array([[np.sqrt(s2)]])}
    if rho is not None:
        post["rho"] = np.array([[rho]])
    if lam is not None:
        post["lam"] = np.array([[lam]])
    if alpha is not None:
        post["unit_effect"] = alpha[None, None, :]
        post["unit_effect_sd"] = np.array([[np.sqrt(s2a)]])
    mo.inference_data = az.from_dict({"posterior": post})
    return mo


PANELS = {
    "pooled": (False, False, False),
    "unit FE": (True, False, False),
    "two-way FE": (True, True, False),
    "time FE": (False, True, False),
    "random effects": (False, False, True),
}


@pytest.mark.parametrize("spatial", ["rho", "lam"])
@pytest.mark.parametrize("panel", PANELS)
def test_panel_scores_are_the_loglikelihood_gradient(panel, spatial):
    unit, time, re = PANELS[panel]
    X, WX, m, shift, n_eff = _panel_parts(unit, time)
    rng = np.random.default_rng(7)
    y = _demean(rng.standard_normal(NP * TP) + 2.0, unit, time)
    beta = np.array([1.0, 0.3, -0.2])
    delta = np.r_[beta, 0.0, 0.0]
    par = {"rho": 0.3} if spatial == "rho" else {"lam": 0.4}
    alpha = rng.standard_normal(NP) * 0.5 if re else None
    s2a = 0.7 if re else None
    gs = gaussian_scores(
        _panel_mock(
            y, X, WX, beta, m, shift, n_eff, s2=1.3, alpha=alpha, s2a=s2a, **par
        )
    )
    rho, lam = par.get("rho", 0.0), par.get("lam", 0.0)
    k = delta.size
    if re:
        theta = np.r_[delta, alpha, rho, lam, 1.3]

        def f(t):
            return _panel_loglik(
                y,
                X,
                WX,
                t[:k],
                t[k + NP],
                t[k + NP + 1],
                t[k + NP + 2],
                m,
                shift,
                n_eff,
                time,
                t[k : k + NP],
                s2a,
            )
    else:
        theta = np.r_[delta, rho, lam, 1.3]

        def f(t):
            return _panel_loglik(
                y, X, WX, t[:k], t[k], t[k + 1], t[k + 2], m, shift, n_eff, time
            )

    h = 1e-6
    ref = np.array(
        [(f(theta + e) - f(theta - e)) / (2 * h) for e in np.eye(theta.size) * h]
    )
    np.testing.assert_allclose(gs.scores[0], ref, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("spatial", ["rho", "lam"])
@pytest.mark.parametrize("panel", ["unit FE", "two-way FE"])
def test_panel_information_is_the_scores_expected_outer_product(panel, spatial):
    """Monte Carlo ``E[g g']`` of the transformed likelihood, with effects in the DGP."""
    unit, time, _ = PANELS[panel]
    X, WX, m, shift, n_eff = _panel_parts(unit, time)
    beta = np.array([1.0, 0.3, -0.2])
    delta = np.r_[beta, 0.0, 0.0]
    rho, lam = (0.3, 0.0) if spatial == "rho" else (0.0, 0.4)
    A = np.kron(I_T, np.eye(NP) - rho * WN)
    B = np.kron(I_T, np.eye(NP) - lam * WN)
    Zr = np.hstack([XP, WXP])
    rng = np.random.default_rng(8)
    sims, h = 3000, 1e-5
    theta = np.r_[delta, rho, lam, 1.3]
    G = np.empty((sims, theta.size))
    for s in range(sims):
        eps = rng.standard_normal(NP * TP) * np.sqrt(1.3)
        fe = np.tile(rng.standard_normal(NP), TP)
        if time:
            fe = fe + np.repeat(rng.standard_normal(TP), NP)
        yd = _demean(
            np.linalg.solve(A, Zr @ delta + fe + np.linalg.solve(B, eps)), unit, time
        )

        def f(t):
            return _panel_loglik(
                yd, X, WX, t[:5], t[5], t[6], t[7], m, shift, n_eff, time
            )

        G[s] = [(f(theta + e) - f(theta - e)) / (2 * h) for e in np.eye(theta.size) * h]
    J_mc = G.T @ G / sims
    _CACHE.clear()
    par = {"rho": rho} if spatial == "rho" else {"lam": lam}
    J = gaussian_scores(
        _panel_mock(
            _demean(np.zeros(NP * TP), unit, time),
            X,
            WX,
            beta,
            m,
            shift,
            n_eff,
            s2=1.3,
            **par,
        )
    ).info
    keep = np.diag(J) > 1e-9
    scale = np.sqrt(np.outer(np.diag(J), np.diag(J)))[np.ix_(keep, keep)]
    assert np.max(np.abs(J - J_mc)[np.ix_(keep, keep)] / scale) < 0.12


def _panel_data(null, design, seed, n_side=10, T=5):
    from neighbayes import dgp

    Wg = dgp.simulate_sar(n_side=n_side, seed=0)["W_sparse"].tocsr()
    N = Wg.shape[0]
    rng = np.random.default_rng(seed)
    filt = sp.linalg.splu((sp.identity(N) - 0.8 * Wg).tocsc())
    A = sp.linalg.splu((sp.identity(N) - 0.4 * Wg).tocsc())
    effects = rng.standard_normal(N) if design != "pooled" else np.zeros(N)
    ys, Xs = [], []
    for _ in range(T):
        Zt = filt.solve(rng.standard_normal((N, 2)))
        Zt = (Zt - Zt.mean(0)) / Zt.std(0)
        Xt = np.c_[np.ones(N), Zt]
        mean = Xt @ np.array([3.0, 1.0, -0.5]) + effects
        eps = rng.standard_normal(N)
        if null == "SAR":
            ys.append(A.solve(mean + eps))
        elif null == "SEM":
            ys.append(mean + A.solve(eps))
        else:
            ys.append(mean + eps)
        Xs.append(Xt)
    return np.concatenate(ys), np.vstack(Xs), Wg, N, T


@pytest.mark.slow
@pytest.mark.parametrize("design", ["pooled", "fixed", "random"])
@pytest.mark.parametrize("null", ["OLS", "SAR", "SEM"])
def test_panel_suites_hold_their_size_at_their_null(null, design):
    """Mean LM/df near one; the SAR panel's LM-Error was 165 to 20,000."""
    import neighbayes.models as M

    effects = {"pooled": {}, "fixed": {"unit_effects": "fixed"}}
    effects["random"] = {"unit_effects": "random"}
    stats = []
    for seed in range(4):
        y, Xs, Wg, N, T = _panel_data(null, design, seed)
        m = getattr(M, null)(y=y, X=Xs, W=Wg, N=N, T=T, **effects[design])
        m.fit(
            draws=400, tune=300, chains=2, progressbar=False, random_seed=seed, n_jobs=1
        )
        df = m.spatial_diagnostics()
        stats.append(df["statistic"] / df["df"])
    mean = np.mean(stats, axis=0)
    assert np.all(mean < 2.5), dict(zip(stats[0].index, np.round(mean, 2)))
