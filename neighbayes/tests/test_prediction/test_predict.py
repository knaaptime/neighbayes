"""Tests for ``predict`` and ``predict_in_sample`` on the Gaussian models."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes.dgp import simulate_sar
from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX

_SPATIAL = {OLS: None, SLX: None, SAR: "rho", SDM: "rho", SEM: "lam", SDEM: "lam"}
_LAG = {SAR, SDM}
_WX = {SLX, SDM, SDEM}


def _renormalized(W, idx):
    """Rows and columns ``idx`` of ``W``, row-standardized again."""
    Ws = sp.csr_matrix(W)[idx][:, idx]
    rs = np.asarray(Ws.sum(axis=1)).ravel()
    inv = np.divide(1.0, rs, out=np.zeros_like(rs), where=rs > 0)
    return sp.csr_matrix(sp.diags(inv) @ Ws)


@pytest.fixture(scope="module")
def split():
    """6x6 SAR sample, union rook W, and an interleaved held-out set."""
    gdf = simulate_sar(
        n_side=6,
        rho=0.5,
        beta=np.array([1.0, 2.0]),
        sigma=1.0,
        seed=0,
        create_gdf=True,
        geometry_type="polygon",
    )
    W = sp.csr_matrix(Graph.build_contiguity(gdf, rook=True).transform("r").sparse)
    oos = np.array([3, 10, 17, 22, 30, 35])
    S = np.setdiff1d(np.arange(len(gdf)), oos)
    return gdf, W, oos, S


def _stub_posterior(model, cls, k, G=4, seed=0):
    rng = np.random.default_rng(seed)
    post = {
        "beta": rng.normal(0.5, 0.4, size=(1, G, k)),
        "sigma": rng.uniform(0.7, 1.3, size=(1, G)),
    }
    if _SPATIAL[cls]:
        post[_SPATIAL[cls]] = rng.uniform(-0.3, 0.8, size=(1, G))
    model._idata = az.from_dict({"posterior": post})
    return post


def _dense_moments(cls, W, X, beta, theta, sigma):
    """Marginal mean and covariance of y on graph W at one draw."""
    n = W.shape[0]
    Z = np.hstack([X, W @ X[:, 1:]]) if cls in _WX else X
    Ai = np.linalg.inv(np.eye(n) - theta * W.toarray())
    mu = Ai @ Z @ beta if cls in _LAG else Z @ beta
    return mu, sigma**2 * Ai @ Ai.T


@pytest.mark.parametrize("cls", list(_SPATIAL))
def test_predict_equals_dense_goldberger(split, cls):
    """BP at each draw equals μ_O + Σ_OS Σ_SS⁻¹ (y_S − μ_S) on the union graph."""
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = cls(formula="y ~ X_1", data=train, W=_renormalized(W, S))
    k = 3 if cls in _WX else 2
    post = _stub_posterior(model, cls, k)
    out = model.predict(gdf.iloc[oos], W, oos=oos, random_seed=0).predictions

    y = gdf["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), gdf["X_1"].to_numpy(float)])
    for g in range(post["sigma"].shape[1]):
        theta = post[_SPATIAL[cls]][0, g] if _SPATIAL[cls] else 0.0
        mu, Sig = _dense_moments(
            cls, W, X, post["beta"][0, g], theta, post["sigma"][0, g]
        )
        bp = mu[oos] + Sig[np.ix_(oos, S)] @ np.linalg.solve(
            Sig[np.ix_(S, S)], y[S] - mu[S]
        )
        np.testing.assert_allclose(out["y_bp"].values[0, g], bp, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            out["y_tc"].values[0, g], mu[oos], rtol=1e-9, atol=1e-11
        )
    assert out["y"].shape == (1, post["sigma"].shape[1], oos.size)
    np.testing.assert_array_equal(out["obs_new"].values, gdf.index[oos])


@pytest.mark.parametrize("cls", list(_SPATIAL))
def test_predict_in_sample_equals_dense_loo(split, cls):
    gdf, W, _, _ = split
    model = cls(formula="y ~ X_1", data=gdf, W=W)
    k = 3 if cls in _WX else 2
    post = _stub_posterior(model, cls, k, G=3, seed=1)
    out = model.predict_in_sample(random_seed=0).predictions

    y = gdf["y"].to_numpy(float)
    n = len(y)
    X = np.column_stack([np.ones(n), gdf["X_1"].to_numpy(float)])
    for g in range(3):
        theta = post[_SPATIAL[cls]][0, g] if _SPATIAL[cls] else 0.0
        mu, Sig = _dense_moments(
            cls, W, X, post["beta"][0, g], theta, post["sigma"][0, g]
        )
        loo = np.empty(n)
        for i in range(n):
            r = np.delete(np.arange(n), i)
            loo[i] = mu[i] + Sig[i, r] @ np.linalg.solve(
                Sig[np.ix_(r, r)], y[r] - mu[r]
            )
        np.testing.assert_allclose(out["y_bp"].values[0, g], loo, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(out["y_tc"].values[0, g], mu, rtol=1e-9, atol=1e-11)


def test_predict_matrix_mode_matches_formula_mode(split):
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    Wt = _renormalized(W, S)
    f = SDM(formula="y ~ X_1", data=train, W=Wt)
    X_train = np.column_stack([np.ones(len(S)), train["X_1"].to_numpy(float)])
    m = SDM(y=train["y"].to_numpy(float), X=X_train, W=Wt)
    _stub_posterior(f, SDM, 3)
    _stub_posterior(m, SDM, 3)
    X_new = np.column_stack([np.ones(oos.size), gdf["X_1"].to_numpy(float)[oos]])
    a = f.predict(gdf.iloc[oos], W, oos=oos).predictions["y_bp"].values
    b = m.predict(X_new, W, oos=oos).predictions["y_bp"].values
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_predict_replays_formula_transforms(split):
    """A stateful transform is evaluated with the training data's state."""
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = OLS(formula="y ~ center(X_1)", data=train, W=_renormalized(W, S))
    post = _stub_posterior(model, OLS, 2)
    out = model.predict(gdf.iloc[oos], W, oos=oos).predictions
    x = gdf["X_1"].to_numpy(float)[oos] - train["X_1"].mean()
    expected = post["beta"][0, :, :1] + post["beta"][0, :, 1:] * x[None, :]
    np.testing.assert_allclose(out["y_tc"].values[0], expected, rtol=1e-12)


def test_predict_default_oos_is_appended(split):
    gdf, W, _, _ = split
    n = len(gdf)
    last = np.arange(n - 4, n)
    S = np.arange(n - 4)
    model = SAR(formula="y ~ X_1", data=gdf.iloc[S], W=_renormalized(W, S))
    _stub_posterior(model, SAR, 2)
    a = model.predict(gdf.iloc[last], W).predictions["y_bp"].values
    b = model.predict(gdf.iloc[last], W, oos=last).predictions["y_bp"].values
    np.testing.assert_array_equal(a, b)


def test_predict_validation(split):
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = SAR(formula="y ~ X_1", data=train, W=_renormalized(W, S))
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(gdf.iloc[oos], W, oos=oos)
    _stub_posterior(model, SAR, 2)
    with pytest.raises(TypeError, match="DataFrame"):
        model.predict(np.ones((oos.size, 2)), W, oos=oos)
    with pytest.raises(ValueError, match="oos has"):
        model.predict(gdf.iloc[oos], W, oos=oos[:-1])
    with pytest.raises(ValueError):
        model.predict(gdf.iloc[oos], _renormalized(W, S), oos=oos)
    with pytest.raises(ValueError, match="thin"):
        model.predict(gdf.iloc[oos], W, oos=oos, thin=0)


def test_predict_rejects_robust(split):
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = SAR(formula="y ~ X_1", data=train, W=_renormalized(W, S), robust=True)
    _stub_posterior(model, SAR, 2)
    with pytest.raises(NotImplementedError, match="robust"):
        model.predict(gdf.iloc[oos], W, oos=oos)
    with pytest.raises(NotImplementedError, match="robust"):
        model.predict_in_sample()


def test_predict_thin(split):
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = SEM(formula="y ~ X_1", data=train, W=_renormalized(W, S))
    _stub_posterior(model, SEM, 2, G=6)
    full = model.predict(gdf.iloc[oos], W, oos=oos).predictions["y_bp"].values
    thin = model.predict(gdf.iloc[oos], W, oos=oos, thin=2).predictions["y_bp"].values
    np.testing.assert_array_equal(thin, full[:, ::2])


def _block_holdout(gdf, rep):
    """A 4x4 held-out block on a 10x10 lattice, placed by ``rep``."""
    rows, cols = np.divmod(np.arange(len(gdf)), 10)
    r0, c0 = 1 + 4 * (rep % 2), 1 + 4 * ((rep // 2) % 2)
    oos = np.flatnonzero(
        (rows >= r0) & (rows < r0 + 4) & (cols >= c0) & (cols < c0 + 4)
    )
    return oos, np.setdiff1d(np.arange(len(gdf)), oos)


def _simulate(cls, rep):
    from neighbayes.dgp import simulate_sem

    sim, kw = (
        (simulate_sar, {"rho": 0.6}) if cls is SAR else (simulate_sem, {"lam": 0.6})
    )
    gdf = sim(
        n_side=10,
        beta=np.array([1.0, 2.0]),
        sigma=1.0,
        seed=100 + rep,
        create_gdf=True,
        geometry_type="polygon",
        contiguity="rook",
        **kw,
    )
    W = sp.csr_matrix(Graph.build_contiguity(gdf, rook=True).transform("r").sparse)
    return gdf, W


def _covered(model, gdf, W, oos, seed):
    y = model.predict(gdf.iloc[oos], W, oos=oos, random_seed=seed).predictions["y"]
    lo, hi = np.quantile(y.values.reshape(-1, oos.size), [0.05, 0.95], axis=0)
    truth = gdf["y"].to_numpy(float)[oos]
    return (truth >= lo) & (truth <= hi)


@pytest.mark.parametrize("cls", [SAR, SEM])
def test_predictive_interval_coverage_at_true_parameters(cls):
    """With the DGP's parameters as the posterior, 90% intervals are nominal.

    This isolates the conditional itself from estimation error.
    """
    hits = []
    for rep in range(60):
        gdf, W = _simulate(cls, rep)
        oos, S = _block_holdout(gdf, rep)
        model = cls(
            formula="y ~ X_1",
            data=gdf.iloc[S].reset_index(drop=True),
            W=_renormalized(W, S),
        )
        G = 2000
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta": np.tile([1.0, 2.0], (1, G, 1)),
                    "sigma": np.ones((1, G)),
                    _SPATIAL[cls]: np.full((1, G), 0.6),
                }
            }
        )
        hits.append(_covered(model, gdf, W, oos, rep))
    coverage = np.concatenate(hits).mean()  # 960 predictions
    assert 0.87 <= coverage <= 0.93, coverage


@pytest.mark.slow
@pytest.mark.parametrize("cls", [SAR, SEM])
def test_predictive_interval_coverage(cls):
    """90% intervals for a held-out block reach nominal coverage after fitting.

    Measured 2026-09-30: SAR 0.918, SEM 0.902.  The fit uses the training
    units' row-renormalized W (Goulard et al. 2017, eq. 8), which biases the
    posterior when many training units lose neighbors: with a scattered
    12% holdout instead of a block, SAR's rho fell from 0.57 to 0.42, sigma
    rose from 1.05 to 1.20, and coverage reached 0.95-0.96.
    """
    hits = []
    for rep in range(25):
        gdf, W = _simulate(cls, rep)
        oos, S = _block_holdout(gdf, rep)
        model = cls(
            formula="y ~ X_1",
            data=gdf.iloc[S].reset_index(drop=True),
            W=_renormalized(W, S),
        )
        model.fit(draws=300, tune=300, chains=2, random_seed=rep, progressbar=False)
        hits.append(_covered(model, gdf, W, oos, rep))
    coverage = np.concatenate(hits).mean()  # 400 predictions
    assert 0.85 <= coverage <= 0.95, coverage


def test_predict_new_unit_changes_fitted_wx(split):
    """SLX: a fitted unit's WX row is recomputed on the union graph."""
    gdf, W, oos, S = split
    train = gdf.iloc[S].reset_index(drop=True)
    model = SLX(formula="y ~ X_1", data=train, W=_renormalized(W, S))
    post = _stub_posterior(model, SLX, 3)
    X_new = gdf.iloc[oos].copy()
    X_new2 = X_new.copy()
    X_new2["X_1"] = X_new2["X_1"] + 5.0
    a = model.predict(X_new, W, oos=oos).predictions["y_tc"].values
    b = model.predict(X_new2, W, oos=oos).predictions["y_tc"].values
    # Moving a new unit's covariate moves the predictions of new units that
    # neighbor it through the lagged term, by beta_WX * (W Δx).
    dx = np.zeros(len(gdf))
    dx[oos] = 5.0
    shift = (W @ dx)[oos][None, :] * post["beta"][0, :, 2:3] + 5.0 * post["beta"][
        0, :, 1:2
    ]
    np.testing.assert_allclose((b - a)[0], shift, rtol=1e-10, atol=1e-12)
    assert isinstance(X_new, pd.DataFrame)
