"""Exact log marginal likelihood by quadrature for Gaussian spatial models.

The quadrature path integrates β out through a rank-k reduction of the
marginal covariance.  These tests check it against a brute-force reference
that forms the full ``n × n`` covariance at every node and integrates σ² and
the spatial parameter with nested adaptive quadrature, and check that the
Bayes-factor dispatcher uses it without requiring a fit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln

from neighbayes.diagnostics import bayes_factor_compare_models, log_marginal_likelihood
from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.tests.helpers import W_to_graph, make_rook_W

MODELS = {"OLS": OLS, "SLX": SLX, "SAR": SAR, "SDM": SDM, "SEM": SEM, "SDEM": SDEM}


def _data(seed=0, side=3):
    rng = np.random.default_rng(seed)
    Wd = make_rook_W(side)
    n = Wd.shape[0]
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = np.linalg.solve(np.eye(n) - 0.4 * Wd, X @ [1.0, 2.0] + rng.standard_normal(n))
    return y, X, W_to_graph(Wd)


def _make(name, y, X, G):
    Model = MODELS[name]
    if name in ("OLS", "SLX"):
        return Model(y=y, X=X, W=G)
    return Model(y=y, X=X, W=G, logdet_method="eigenvalue")


def _brute_force_logml(model):
    """Dense covariance at every node; nested adaptive quadrature over (ρ, log σ²)."""
    y = model._y
    Z = model._design_matrix()
    pr = model._gaussian_priors(Z, model._design_names())
    n, k = Z.shape
    mu = np.full(k, 1.0) * pr["beta_mu"]
    L0 = np.diag((np.full(k, 1.0) * pr["beta_sigma"]) ** 2)
    a0, b0 = pr["sigma2_alpha"], pr["sigma2_beta"]
    kind = model._jacobian_param
    Wd = model._W_sparse.toarray()

    def log_f(t, val):
        A = np.eye(n) - val * Wd
        Xs = A @ Z if kind == "lam" else Z
        r = A @ y - Xs @ mu
        C = math.exp(t) * np.eye(n) + Xs @ L0 @ Xs.T
        ll = (
            np.linalg.slogdet(A)[1]
            - 0.5 * n * math.log(2 * math.pi)
            - 0.5 * np.linalg.slogdet(C)[1]
            - 0.5 * r @ np.linalg.solve(C, r)
        )
        return ll + a0 * math.log(b0) - gammaln(a0) - a0 * t - b0 * math.exp(-t)

    shift = log_f(math.log(np.var(y)), 0.0)

    def inner(val):
        return integrate.quad(
            lambda t: math.exp(log_f(t, val) - shift), -20, 15, limit=200, epsrel=1e-11
        )[0]

    if kind is None:
        return shift + math.log(inner(0.0))
    lo, hi = pr[f"{kind}_lower"], pr[f"{kind}_upper"]
    total = integrate.quad(inner, lo, hi, limit=200, epsrel=1e-11)[0]
    return shift + math.log(total) - math.log(hi - lo)


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_dense_brute_force(name):
    y, X, G = _data()
    model = _make(name, y, X, G)
    got = log_marginal_likelihood(model)
    np.testing.assert_allclose(got, _brute_force_logml(model), rtol=0, atol=1e-7)


def test_compare_models_needs_no_fit():
    y, X, G = _data(seed=1, side=4)
    models = {name: _make(name, y, X, G) for name in ("SAR", "SEM", "SDM")}
    log_bf, diag = bayes_factor_compare_models(
        models, method="quadrature", log=True, return_diagnostics=True
    )
    logml = {name: log_marginal_likelihood(m) for name, m in models.items()}
    np.testing.assert_allclose(log_bf.loc["SAR", "SEM"], logml["SAR"] - logml["SEM"])
    assert all(d["method"] == "quadrature" for d in diag.values())
    assert all(d["abserr"] < 1e-6 for d in diag.values())


def test_robust_model_rejected():
    y, X, G = _data()
    with pytest.raises(ValueError, match="Gaussian errors"):
        log_marginal_likelihood(SAR(y=y, X=X, W=G, robust=True))


def test_unknown_kwarg_rejected():
    y, X, G = _data()
    with pytest.raises(TypeError, match="unexpected keyword"):
        bayes_factor_compare_models(
            [_make("SAR", y, X, G)], method="quadrature", tol1=1
        )


@pytest.mark.slow
def test_agrees_with_bridge_sampling():
    """Bridge sampling estimates the same quantity, up to its own error."""
    y, X, G = _data(seed=2, side=12)
    models = {}
    for name in ("SAR", "SEM"):
        m = MODELS[name](y=y, X=X, W=G)
        m.fit(draws=10000, tune=1000, chains=4, random_seed=1, progressbar=False)
        models[name] = m
    _, diag = bayes_factor_compare_models(
        models, method="bridge", return_diagnostics=True, random_state=0
    )
    for name, m in models.items():
        np.testing.assert_allclose(
            log_marginal_likelihood(m), diag[name]["logml"], atol=0.05
        )
