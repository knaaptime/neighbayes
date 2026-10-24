"""The flow LM tests (:mod:`neighbayes.diagnostics.lmtests.flow`).

The scores must be the gradient of the lag-flow model's log-likelihood at its
aspatial null, cross-section and Lee-Yu panels alike, the information their
expected outer product, and each suite must hold its size at the null when the
intercept is large.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.diagnostics.lmtests import bayesian_lm_flow_joint_test
from neighbayes.diagnostics.lmtests.flow import _flow_scores

n, T = 5, 3
N = n * n


def _weights():
    rng = np.random.default_rng(0)
    adj = rng.random((n, n)) < 0.4
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    for i in range(n):
        if not adj[i].any():
            adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = True
    return adj / adj.sum(1, keepdims=True)


WN = _weights()
W = sp.csr_matrix(WN)
WK = [np.kron(np.eye(n), WN), np.kron(WN, np.eye(n)), np.kron(WN, WN)]


class _Model:
    """A flow fit as the tests read it (weak-referenceable, unlike a namespace)."""

    def __init__(self, y, X, design, beta, s2, extra=None):
        periods, m, shift, n_eff = _dims(design)
        self._y, self._X, self._W_sparse = y, X, W
        self._jacobian_T, self._n_effective = m, n_eff
        if design is not None:
            self._T, self._flow_jacobian_shift = periods, shift
        post = {"beta": beta[None, None, :], "sigma": np.array([[np.sqrt(s2)]])}
        post.update(extra or {})
        self.inference_data = az.from_dict({"posterior": post})


def _dims(design):
    """Periods, Jacobian multiplier, time-effects shift and effective n."""
    if design is None:
        return 1, 1.0, 0.0, N
    unit, time = design in (1, 3), design in (2, 3)
    m = T - 1 if unit else T
    return T, m, (m if time else 0.0), (N - time) * (T - unit)


def _demean(v, design):
    v = v.reshape(T, N, -1).copy()
    if design in (1, 3):
        v = v - v.mean(axis=0, keepdims=True)
    if design in (2, 3):
        v = v - v.mean(axis=1, keepdims=True)
    return v.reshape(T * N, -1).squeeze()


def _data(design, rng, intercept=3.0):
    periods = _dims(design)[0]
    X = np.c_[np.ones(periods * N), rng.standard_normal((periods * N, 2))]
    beta = np.array([intercept, 0.8, -0.4])
    if design:
        X, beta = _demean(X, design), np.r_[0.0, beta[1:]]
    return X, beta


def _loglik(y, X, beta, rho, s2, design):
    """The lag-flow log-likelihood, Lee-Yu transformed for a panel."""
    periods, m, shift, n_eff = _dims(design)
    A = np.eye(N) - sum(r * w for r, w in zip(rho, WK))
    e = []
    for t in range(periods):
        yt = y[t * N : (t + 1) * N]
        lag = sum(r * (w @ yt) for r, w in zip(rho, WK))
        if shift:
            lag = lag - lag.mean()
        e.append(yt - lag - X[t * N : (t + 1) * N] @ beta)
    e = np.concatenate(e)
    jac = m * np.linalg.slogdet(A)[1] - shift * np.log(1 - sum(rho))
    return jac - 0.5 * n_eff * np.log(2 * np.pi * s2) - e @ e / (2 * s2)


DESIGNS = [None, 0, 1, 2, 3]
IDS = ["cross-section", "pooled", "unit FE", "time FE", "two-way FE"]


@pytest.mark.parametrize("design", DESIGNS, ids=IDS)
def test_scores_are_the_loglikelihood_gradient(design):
    rng = np.random.default_rng(1)
    X, beta = _data(design, rng)
    y = X @ beta + X[:, 1] ** 2 + rng.standard_normal(X.shape[0])
    if design:
        y = _demean(y, design)
    s2 = 1.3
    g, J, free, k = _flow_scores(_Model(y, X, design, beta, s2))
    theta = np.r_[beta, 0.0, 0.0, 0.0, s2]

    def f(t):
        return _loglik(y, X, t[:k], t[k : k + 3], t[k + 3], design)

    h = 1e-6
    fd = np.array([(f(theta + e) - f(theta - e)) / (2 * h) for e in np.eye(k + 4) * h])
    # A demeaned intercept is a zero column: no likelihood direction.
    sl = slice(1, None) if design else slice(None)
    np.testing.assert_allclose(g[-1, sl], fd[sl], atol=1e-6)


def test_a_spatial_or_count_flow_is_refused():
    rng = np.random.default_rng(2)
    X, beta = _data(None, rng)
    y = X @ beta + rng.standard_normal(N)
    sar = _Model(y, X, None, beta, 1.0, {"rho_d": np.array([[0.2]])})
    with pytest.raises(ValueError, match="estimates rho_d"):
        bayesian_lm_flow_joint_test(sar)
    nb = _Model(y, X, None, beta, 1.0)
    nb.inference_data = az.from_dict(
        {"posterior": {"beta": beta[None, None, :], "alpha": np.array([[2.0]])}}
    )
    with pytest.raises(ValueError, match="not Gaussian"):
        bayesian_lm_flow_joint_test(nb)


@pytest.mark.slow
@pytest.mark.parametrize("design", [None, 1, 2, 3], ids=IDS[:1] + IDS[2:])
def test_information_is_the_scores_expected_outer_product(design):
    rng = np.random.default_rng(3)
    X, beta = _data(design, rng)
    s2 = 0.7
    G = []
    for _ in range(5000):
        y = X @ beta + np.sqrt(s2) * rng.standard_normal(X.shape[0])
        if design:
            y = _demean(y, design)
        g, J, _, _ = _flow_scores(_Model(y, X, design, beta, s2))
        G.append(g[-1])
    G = np.array(G)
    sl = slice(1, None) if design else slice(None)
    C, J = np.cov(G.T)[sl, sl], J[sl, sl]
    se = np.sqrt((np.outer(np.diag(C), np.diag(C)) + C**2) / len(G))
    assert (np.abs(C - J) / se).max() < 4.5


@pytest.mark.slow
@pytest.mark.parametrize("effects", [None, 1, 2, 3])
def test_suites_hold_their_size_at_their_null(effects):
    """Mean LM/df near one at the OLSFlow null with a large intercept."""
    from neighbayes.dgp import simulate_sar_flow
    from neighbayes.dgp.flows import _simulate_sar_flow_panel_fe
    from neighbayes.models.flow import OLSFlow

    stats = []
    for seed in range(12):
        if effects is None:
            d = simulate_sar_flow(
                n=10, rho_d=0.0, rho_o=0.0, rho_w=0.0, alpha=3.0,
                distribution="normal", seed=seed,
            )  # fmt: skip
            m = OLSFlow(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
        else:
            d = _simulate_sar_flow_panel_fe(
                n=8, T=4, rho_d=0.0, rho_o=0.0, rho_w=0.0, distribution="normal",
                sigma_alpha=0.5 if effects in (1, 3) else 0.0,
                time_effect_sd=1.0 if effects in (2, 3) else 0.0, seed=seed,
            )  # fmt: skip
            m = OLSFlow(
                d["y"], d["X"], d["G"], T=4, col_names=d["col_names"],
                pair_effects="fixed" if effects in (1, 3) else None,
                time_effects=effects in (2, 3),
            )  # fmt: skip
        m.fit(
            draws=400, tune=400, chains=1, progressbar=False, random_seed=seed, cores=1
        )
        df = m.spatial_diagnostics()
        stats.append(df["statistic"] / df["df"])
    mean = np.mean(stats, axis=0)
    assert np.all(mean < 2.5), dict(zip(stats[0].index, np.round(mean, 2)))
