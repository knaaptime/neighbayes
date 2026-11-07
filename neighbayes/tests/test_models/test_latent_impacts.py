"""Outcome-scale impacts of the latent models.

The expected count of the latent negative binomial has a closed form in the
multiplier's entries, :math:`\\mathbb{E}\\,y_i = e^{m_i}\\prod_j
\\Gamma(\\alpha + M_{ij}) / (\\Gamma(\\alpha)\\alpha^{M_{ij}})`, the density
of the latent logit's propagated disturbance is the Fourier inverse of
:math:`\\prod_j \\pi M_{ij}t / \\sinh(\\pi M_{ij}t)`, and the latent probit's is
normal with variance :math:`\\sum_j M_{ij}^2`.  The negative binomial's effects
are that closed form, interpolated across the draws, and the binary models'
Monte Carlo estimates must match theirs, on a small lattice, for SAR, SEM and
SDEM.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.special import gammaln

import neighbayes.models as M
from neighbayes import dgp
from neighbayes.models.cross_section._latent import latent_outcome_effects
from neighbayes.models.cross_section._latent_negbin import latent_count_effects

PAR, ALPHA = 0.6, 0.7
CASES = [
    (M.SARNegBin, dgp.simulate_sar_negbin, "nb"),
    (M.SEMNegBin, dgp.simulate_sem_negbin, "nb"),
    (M.SDEMNegBin, dgp.simulate_sdem_negbin, "nb"),
    (M.SARLogit, dgp.simulate_sar_logit, "logit"),
    (M.SEMLogit, dgp.simulate_sem_logit, "logit"),
    (M.SDEMLogit, dgp.simulate_sdem_logit, "logit"),
    (M.SARProbit, dgp.simulate_sar_probit, "probit"),
    (M.SEMProbit, dgp.simulate_sem_probit, "probit"),
    (M.SDEMProbit, dgp.simulate_sdem_probit, "probit"),
]


def _logistic_sum_density(weights, x):
    """Density at ``x`` of ``Σ_j weights_j ε_j``, ε_j i.i.d. standard logistic."""
    a = np.pi * weights[np.abs(weights) > 1e-12]

    def cf(t):
        return np.prod(a * t / np.sinh(a * t)) if t else 1.0

    return quad(lambda t: np.cos(t * x) * cf(t), 0, np.inf, limit=400)[0] / np.pi


def _model_at(cls, simulate, family):
    """A latent model of ``cls`` holding one posterior draw."""
    d = simulate(n_side=5, seed=1)
    latent = {"latent": True} if cls in (M.SARNegBin, M.SARLogit) else {}
    m = cls(y=d["y"], X=d["X"], W=d["W_sparse"], **latent)
    X, _ = m._latent_design
    beta = np.r_[0.3, np.random.default_rng(0).normal(0, 0.5, X.shape[1] - 1)]
    post = {m._latent_param: np.array([[PAR]]), "beta": beta[None, None, :]}
    if family == "nb":
        post["alpha"] = np.array([[ALPHA]])
    m._idata = az.from_dict({"posterior": post})
    return m, X, beta


def _exact_effects(m, X, beta, family):
    W = m._W_sparse.toarray()
    n = W.shape[0]
    Mx = np.linalg.inv(np.eye(n) - PAR * W)
    sar = m._latent_process == "sar"
    mean = Mx @ (X @ beta) if sar else X @ beta
    if family == "nb":
        logc = (gammaln(ALPHA + Mx) - gammaln(ALPHA) - Mx * np.log(ALPHA)).sum(1)
        w = np.exp(mean + logc)
    elif family == "probit":
        sd = np.sqrt((Mx**2).sum(1))
        w = np.exp(-0.5 * (mean / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)
    else:
        w = np.array([_logistic_sum_density(Mx[i], -mean[i]) for i in range(n)])
    if sar:
        ni = m._beta_nonintercept_indices
        return np.mean(w * np.diag(Mx)) * beta[ni], np.mean(w * Mx.sum(1)) * beta[ni]
    if getattr(m, "_has_wx_in_beta", False):
        k = m._X.shape[1]
        b1, b2 = beta[:k][list(m._beta_wx_column_indices)], beta[k:]
        direct = np.mean(w) * b1 + np.mean(w * np.diag(W)) * b2
        return direct, np.mean(w) * b1 + np.mean(w * W.sum(1)) * b2
    ni = m._beta_nonintercept_indices
    return np.mean(w) * beta[ni], np.mean(w) * beta[ni]


@pytest.mark.parametrize(
    "cls, simulate, family", CASES, ids=[c[0].__name__ for c in CASES]
)
def test_outcome_effects_match_the_exact_ones(cls, simulate, family):
    m, X, beta = _model_at(cls, simulate, family)
    ex_direct, ex_total = _exact_effects(m, X, beta, family)
    direct, indirect, total = m._latent_outcome_effects(
        **({} if family == "nb" else {"n_mc": 20000})
    )
    np.testing.assert_allclose(direct[0], ex_direct, rtol=0.01)
    np.testing.assert_allclose(total[0], ex_total, rtol=0.01)
    np.testing.assert_allclose(indirect[0], ex_total - ex_direct, rtol=0.01, atol=1e-12)


@pytest.mark.parametrize(
    "cls, simulate, family", CASES, ids=[c[0].__name__ for c in CASES]
)
def test_impacts_take_the_outcome_scale(cls, simulate, family):
    m, _, _ = _model_at(cls, simulate, family)
    outcome = "count" if family == "nb" else "probability"
    df = m.impacts(scale=outcome)
    assert df.attrs["scale"] == outcome and np.isfinite(df.to_numpy(float)).all()
    index = m.impacts(scale=m._index_scale)
    assert index.attrs["scale"] == m._index_scale
    with pytest.raises(ValueError, match="scale must be"):
        m.impacts(scale="nonsense")


# ---------------------------------------------------------------------------
# The negative binomial's closed form, where simulation fails
# ---------------------------------------------------------------------------


def _lattice(side=6):
    m = _model_at(M.SARNegBin, dgp.simulate_sar_negbin, "nb")[0]
    d = dgp.simulate_sar_negbin(n_side=side, seed=1)
    return sp.csr_matrix(d["W_sparse"]) if side != 5 else m._W_sparse.tocsr()


def _exact_count_effects(W, par, alpha, beta, offset, periods=1):
    """Direct and total effects of each draw on the stack, by dense inverses."""
    N = W.shape[0]
    Wd = W.toarray()
    direct, total = [], []
    for r, a, b, off in zip(par, alpha, beta, offset):
        S = np.linalg.inv(np.eye(N) - r * Wd)
        logc = (gammaln(a + S) - gammaln(a) - S * np.log(a)).sum(1)
        w = np.exp((off.reshape(periods, N) @ S.T) + logc)
        direct.append(np.mean(w * np.diag(S)) * b[1])
        total.append(np.mean(w * S.sum(1)) * b[1])
    return np.array(direct), np.array(total)


@pytest.mark.parametrize("periods", [1, 3])
def test_count_effects_are_exact_at_small_alpha(periods):
    """Small alpha and large rho: 128 simulations read 16-100% low here."""
    W = _lattice()
    N = W.shape[0]
    rng = np.random.default_rng(0)
    G = 30
    par = rng.uniform(0.5, 0.95, G)
    alpha = np.exp(rng.uniform(np.log(0.03), np.log(0.5), G))
    beta = np.column_stack([np.full(G, 0.5), rng.normal(0.5, 0.05, G)])
    X = np.column_stack([np.ones(N * periods), rng.standard_normal(N * periods)])
    offset = beta @ X.T
    direct, total = latent_count_effects(
        par=par, alpha=alpha, beta=beta, offset=offset, W=W, sar=True, ni=[1],
        periods=periods,
    )  # fmt: skip
    ex_d, ex_t = _exact_count_effects(W, par, alpha, beta, offset, periods)
    np.testing.assert_allclose(direct[:, 0], ex_d, rtol=1e-8)
    np.testing.assert_allclose(total[:, 0], ex_t, rtol=1e-8)


def test_count_effects_take_a_held_alpha():
    W = _lattice()
    N = W.shape[0]
    rng = np.random.default_rng(1)
    par = rng.uniform(0.2, 0.6, 8)
    alpha = np.full(8, 1.5)
    beta = np.column_stack([np.zeros(8), np.ones(8)])
    offset = beta @ np.column_stack([np.ones(N), rng.standard_normal(N)]).T
    direct, total = latent_count_effects(
        par=par, alpha=alpha, beta=beta, offset=offset, W=W, sar=True, ni=[1]
    )
    ex_d, ex_t = _exact_count_effects(W, par, alpha, beta, offset)
    np.testing.assert_allclose(direct[:, 0], ex_d, rtol=1e-8)
    np.testing.assert_allclose(total[:, 0], ex_t, rtol=1e-8)


def test_fitted_counts_are_the_closed_form():
    m, X, beta = _model_at(M.SARNegBin, dgp.simulate_sar_negbin, "nb")
    Wd = m._W_sparse.toarray()
    S = np.linalg.inv(np.eye(Wd.shape[0]) - PAR * Wd)
    logc = (gammaln(ALPHA + S) - gammaln(ALPHA) - S * np.log(ALPHA)).sum(1)
    np.testing.assert_allclose(
        m._fitted_mean_from_posterior_latent(), np.exp(S @ X @ beta + logc), rtol=1e-10
    )


def test_draws_with_no_expected_count_are_left_out():
    """At ``S_ij <= -alpha`` the moment is infinite; the rest of the posterior stands."""
    W = _lattice()
    N = W.shape[0]
    par = np.array([0.4, 0.3, -0.8])
    alpha = np.array([1.0, 1.2, 0.02])
    beta = np.column_stack([np.zeros(3), np.ones(3)])
    offset = beta @ np.column_stack([np.ones(N), np.linspace(-1, 1, N)]).T
    with pytest.warns(UserWarning, match="1 of 3 posterior draws"):
        direct, total = latent_count_effects(
            par=par, alpha=alpha, beta=beta, offset=offset, W=W, sar=True, ni=[1]
        )
    ex_d, ex_t = _exact_count_effects(W, par[:2], alpha[:2], beta[:2], offset[:2])
    np.testing.assert_allclose(direct[:, 0], ex_d, rtol=1e-8)
    np.testing.assert_allclose(total[:, 0], ex_t, rtol=1e-8)
    with pytest.raises(ValueError, match="No posterior draw"):
        latent_count_effects(
            par=par[2:], alpha=alpha[2:], beta=beta[2:], offset=offset[2:], W=W,
            sar=True, ni=[1],
        )  # fmt: skip


def test_outcome_effects_do_not_depend_on_the_chunking(monkeypatch):
    """Each draw owns its random stream, so batches and threads change nothing."""
    from neighbayes.models.cross_section import _latent

    W = _lattice()
    N = W.shape[0]
    rng = np.random.default_rng(2)
    par = rng.uniform(0.1, 0.6, 7)
    beta = np.column_stack([np.zeros(7), np.ones(7)])
    offset = beta @ np.column_stack([np.ones(N), rng.standard_normal(N)]).T

    def run():
        return latent_outcome_effects(
            par=par, beta=beta, offset=offset, extra={}, W=W, sar=True, ni=[1],
            noise=lambda r, size, params: r.logistic(size=size),
            weight=M.SARLogit._latent_outcome_weight, n_mc=8, seed=3,
        )  # fmt: skip

    first = run()
    monkeypatch.setattr(_latent, "_OUTCOME_CHUNK", 2)
    second = run()
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
