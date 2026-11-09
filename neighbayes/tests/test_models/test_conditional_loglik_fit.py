"""After a fit, the stored pointwise term is ``log p(y_i | y_-i, theta)`` per draw."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import integrate, stats

from neighbayes.models import SAR, SEM

QUICK = dict(
    draws=20,
    tune=20,
    chains=1,
    random_seed=0,
    progressbar=False,
    idata_kwargs={"log_likelihood": True},
)
SIDE = 4
N = SIDE * SIDE
T = 3


def _W():
    rows, cols = [], []
    for i in range(SIDE):
        for j in range(SIDE):
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < SIDE and 0 <= b < SIDE:
                    rows.append(SIDE * i + j)
                    cols.append(SIDE * a + b)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    return sp.csr_matrix(sp.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W)


def _dense(y, mu, Sigma):
    out = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        coef = np.linalg.solve(Sigma[np.ix_(keep, keep)], Sigma[keep, i])
        m = mu[i] + coef @ (y[keep] - mu[keep])
        v = Sigma[i, i] - coef @ Sigma[keep, i]
        out.append(stats.norm(m, np.sqrt(v)).logpdf(y[i]))
    return np.array(out)


def _draws(model, *names, k=3):
    post = model.inference_data.posterior
    return [
        post[v].values.reshape(post.sizes["chain"] * post.sizes["draw"], -1)[:k]
        for v in names
    ]


def _stored(model, k=3):
    ll = model.inference_data.log_likelihood["obs"].values
    return ll.reshape(-1, ll.shape[-1])[:k]


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), np.column_stack([np.ones(n), rng.normal(size=n)])


def test_cross_section_sar():
    W = _W()
    y, X = _data(N)
    m = SAR(y=y, X=X, W=W)
    m.fit(**QUICK)
    rho, beta, sigma = _draws(m, "rho", "beta", "sigma")
    for s, ours in enumerate(_stored(m)):
        Ainv = np.linalg.inv(np.eye(N) - rho[s, 0] * W.toarray())
        dense = _dense(y, Ainv @ X @ beta[s], sigma[s, 0] ** 2 * Ainv @ Ainv.T)
        np.testing.assert_allclose(ours, dense, atol=1e-8)
    assert "log_joint" in m.inference_data.children


def test_random_effects_sar_integrates_the_effects():
    W = _W()
    y, X = _data(N * T, 1)
    m = SAR(y=y, X=X, W=W, N=N, T=T, unit_effects="random")
    m.fit(**QUICK)
    rho, beta, sigma, sd = _draws(m, "rho", "beta", "sigma", "unit_effect_sd")
    for s, ours in enumerate(_stored(m)):
        block = np.kron(np.eye(T), np.linalg.inv(np.eye(N) - rho[s, 0] * W.toarray()))
        cov = sigma[s, 0] ** 2 * np.eye(N * T) + sd[s, 0] ** 2 * np.kron(
            np.ones((T, T)), np.eye(N)
        )
        dense = _dense(m._y, block @ m._X @ beta[s], block @ cov @ block.T)
        np.testing.assert_allclose(ours, dense, atol=1e-8)


def test_random_effects_sem_conditions_on_the_effects():
    W = _W()
    y, X = _data(N * T, 2)
    m = SEM(y=y, X=X, W=W, N=N, T=T, unit_effects="random")
    m.fit(**QUICK)
    lam, beta, sigma, alpha = _draws(m, "lam", "beta", "sigma", "unit_effect")
    for s, ours in enumerate(_stored(m)):
        Binv = np.linalg.inv(np.eye(N) - lam[s, 0] * W.toarray())
        block = np.kron(np.eye(T), Binv)
        mu = m._X @ beta[s] + np.tile(alpha[s], T)
        dense = _dense(m._y, mu, sigma[s, 0] ** 2 * block @ block.T)
        np.testing.assert_allclose(ours, dense, atol=1e-8)


def test_dynamic_sar_includes_the_next_period():
    W = _W()
    y, X = _data(N * T, 3)
    m = SAR(y=y, X=X, W=W, N=N, T=T, dynamic=True)
    m.fit(**{**QUICK, "sampler": "nuts"})
    rho, phi, beta, sigma = _draws(m, "rho", "phi", "beta", "sigma")
    Td = T - 1
    for s, ours in enumerate(_stored(m)):
        A = np.eye(N) - rho[s, 0] * W.toarray()
        D = np.kron(np.eye(Td), A) - np.kron(np.eye(Td, k=-1), phi[s, 0] * np.eye(N))
        h = m._X_dyn @ beta[s] + np.concatenate(
            [phi[s, 0] * m._y_lag[:N], np.zeros(N * (Td - 1))]
        )
        Dinv = np.linalg.inv(D)
        dense = _dense(m._y_dyn, Dinv @ h, sigma[s, 0] ** 2 * Dinv @ Dinv.T)
        np.testing.assert_allclose(ours, dense, atol=1e-8)


def test_robust_sar_matches_quadrature():
    W = _W()
    y, X = _data(N, 4)
    m = SAR(y=y, X=X, W=W, robust=True)
    m.fit(**{**QUICK, "sampler": "nuts"})
    rho, beta, sigma = _draws(m, "rho", "beta", "sigma", k=1)
    A = np.eye(N) - rho[0, 0] * W.toarray()
    ours = _stored(m, k=1)[0]
    t = stats.t(m._nu, scale=sigma[0, 0])

    def log_joint(v, i):
        yy = y.copy()
        yy[i] = v
        return t.logpdf(A @ yy - X @ beta[0]).sum()

    for i in range(0, N, 5):
        peak = log_joint(y[i], i)
        mass, _ = integrate.quad(
            lambda v: np.exp(log_joint(v, i) - peak),
            -np.inf,
            np.inf,
            epsabs=1e-13,
            epsrel=1e-11,
            limit=400,
        )
        assert abs(ours[i] + np.log(mass)) < 1e-6


def test_robust_fixed_effects_warn_and_keep_the_stored_term():
    W = _W()
    y, X = _data(N * T, 5)
    m = SAR(y=y, X=X, W=W, N=N, T=T, unit_effects="fixed", robust=True)
    with pytest.warns(UserWarning, match="not leave-one-out"):
        m.fit(**{**QUICK, "sampler": "nuts"})
    assert "log_joint" not in m.inference_data.children


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("process", ["SAR", "SEM"])
def test_multilevel_stores_each_unit_given_the_others(process, backend):
    from neighbayes import dgp
    from neighbayes.models import Level, Multilevel
    from neighbayes.models._base._conditional import gaussian_conditional_loglik

    if backend == "jax":
        pytest.importorskip("jax")
    kind = {"SAR": "lag", "SEM": "error"}[process]
    d = dgp.simulate_multilevel(
        n_side=6,
        blocks=(2,),
        processes=(process, process),
        rhos=(0.4, 0.5),
        sigmas=(1.0, 0.6),
        seed=0,
    )
    lv = d["levels"][0]
    m = Multilevel(
        Level(y=d["y"], X=d["X"], W=d["W_sparse"], process=process),
        Level(X=lv["X"], W=lv["W_sparse"], groups=d["parent"]),
    )
    m.fit(
        draws=10,
        tune=10,
        chains=1,
        random_seed=0,
        progressbar=False,
        backend=backend,
        store_theta=True,
        idata_kwargs={"log_likelihood": True},
    )
    post = m.inference_data.posterior
    name = {"SAR": "rho_0", "SEM": "lam_0"}[process]
    par = post[name].values.reshape(-1)
    beta = post["beta_0"].values.reshape(par.size, -1)
    theta = post["theta_1"].values.reshape(par.size, -1)
    sigma = post["sigma_0"].values.reshape(-1)
    W = sp.csr_matrix(d["W_sparse"])
    y = np.asarray(d["y"], dtype=float)
    u = y[None, :] - beta @ np.asarray(d["X"]).T - theta[:, np.asarray(d["parent"])]
    if kind == "lag":
        e = u - par[:, None] * (W @ y)[None, :]
    else:
        e = u - par[:, None] * np.asarray(W @ u.T).T
    expected = gaussian_conditional_loglik(e, par, sigma**2, W)
    stored = m.inference_data.log_likelihood["obs"].values.reshape(par.size, -1)
    np.testing.assert_allclose(stored, expected, atol=1e-8)
    assert m.inference_data["log_joint"]["obs"].shape == (1, 10)


@pytest.mark.parametrize(
    "cls_name, panel",
    [("SARTobit", False), ("SEMTobit", False), ("SDMTobit", False), ("SARTobit", True)],
)
def test_tobit_conditions_on_each_draws_latent_outcome(cls_name, panel):
    import neighbayes.models as M

    W = _W()
    n = N * T if panel else N
    y, X = _data(n, 6)
    y = np.maximum(y + 0.3, 0.0)
    kw = dict(N=N, T=T) if panel else {}
    m = getattr(M, cls_name)(y=y, X=X, W=W, censoring=0.0, **kw)
    m.fit(**{**QUICK, "sampler": "nuts"})
    post = m.inference_data.posterior
    name = "rho" if "rho" in post else "lam"
    par, beta, sigma, gap = _draws(m, name, "beta", "sigma", "y_cens_gap")
    Z = np.hstack([m._X, m._WX]) if m._has_wx_in_beta else m._X
    cens = m._censored_idx
    assert cens.size > 0
    for s, ours in enumerate(_stored(m)):
        Binv = np.kron(
            np.eye(n // N), np.linalg.inv(np.eye(N) - par[s, 0] * W.toarray())
        )
        mu = Binv @ Z @ beta[s] if name == "rho" else Z @ beta[s]
        Sigma = sigma[s, 0] ** 2 * Binv @ Binv.T
        y_star = m._y.astype(float).copy()
        y_star[cens] = 0.0 - gap[s]
        for i in range(n):
            keep = np.arange(n) != i
            coef = np.linalg.solve(Sigma[np.ix_(keep, keep)], Sigma[keep, i])
            mi = mu[i] + coef @ (y_star[keep] - mu[keep])
            vi = Sigma[i, i] - coef @ Sigma[keep, i]
            want = (
                stats.norm(mi, np.sqrt(vi)).logcdf(0.0)
                if i in cens
                else stats.norm(mi, np.sqrt(vi)).logpdf(m._y[i])
            )
            assert abs(ours[i] - want) < 1e-8
    assert "log_joint" in m.inference_data.children


def _flow_case(n=4, periods=1, seed=7):
    from neighbayes.dgp import simulate_sar_flow

    out = simulate_sar_flow(
        n=n,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=[1.0],
        beta_o=[0.5],
        sigma=1.0,
        seed=seed,
    )
    X = np.vstack([out["X"]] * periods)
    y = np.random.default_rng(seed).normal(size=n * n * periods)
    return y, X, out


@pytest.mark.parametrize(
    "cls_name, kw",
    [
        ("SARFlow", {}),
        ("SEMFlow", {}),
        ("OLSFlow", {}),
        ("SARFlow", {"separable": True}),
    ],
)
def test_flow_stores_each_flow_given_the_others(cls_name, kw):
    import neighbayes.models.flow as F

    y, X, out = _flow_case()
    m = getattr(F, cls_name)(y, X, out["G"], col_names=out["col_names"], **kw)
    sampler = "nuts" if cls_name == "OLSFlow" or kw else None
    m.fit(**{**QUICK, **({"sampler": sampler} if sampler else {})})
    post = m.inference_data.posterior
    prefix = next((p for p in ("rho", "lam") if f"{p}_d" in post), None)
    W = sp.csr_matrix(m._W_sparse).toarray()
    I = np.eye(m._n)
    Ws = [np.kron(I, W), np.kron(W, I), np.kron(W, W)]
    beta, sigma = _draws(m, "beta", "sigma")
    pars = _draws(m, *[f"{prefix}_{k}" for k in "dow"]) if prefix else None
    for s, ours in enumerate(_stored(m)):
        F_ = np.eye(m._N) - (
            sum(p[s, 0] * Wk for p, Wk in zip(pars, Ws)) if prefix else 0
        )
        Finv = np.linalg.inv(F_)
        mu = Finv @ m._X @ beta[s] if prefix == "rho" else m._X @ beta[s]
        np.testing.assert_allclose(
            ours, _dense(m._y, mu, sigma[s, 0] ** 2 * Finv @ Finv.T), atol=1e-8
        )


def test_flow_panel_with_pair_effects_integrates_them():
    import neighbayes.models.flow as F

    periods = 2
    y, X, out = _flow_case(n=3, periods=periods, seed=8)
    m = F.SARFlow(
        y, X, out["G"], col_names=out["col_names"], T=periods, pair_effects="fixed"
    )
    m.fit(**QUICK)
    W = sp.csr_matrix(m._W_sparse).toarray()
    I = np.eye(m._n)
    Ws = [np.kron(I, W), np.kron(W, I), np.kron(W, W)]
    pars = _draws(m, "rho_d", "rho_o", "rho_w")
    beta, sigma = _draws(m, "beta", "sigma")
    n = m._N * periods
    for s, ours in enumerate(_stored(m)):
        A = np.eye(m._N) - sum(p[s, 0] * Wk for p, Wk in zip(pars, Ws))
        block = np.kron(np.eye(periods), np.linalg.inv(A))
        cov = sigma[s, 0] ** 2 * np.eye(n) + 1e6 * np.kron(
            np.ones((periods, periods)), np.eye(m._N)
        )
        dense = _dense(y, block @ m._X @ beta[s], block @ cov @ block.T)
        np.testing.assert_allclose(ours, dense, atol=1e-3)
