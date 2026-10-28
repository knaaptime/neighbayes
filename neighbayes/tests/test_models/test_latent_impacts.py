"""Outcome-scale impacts of the latent models.

The expected count of the latent negative binomial has a closed form in the
multiplier's entries, :math:`\\mathbb{E}\\,y_i = e^{m_i}\\prod_j
\\Gamma(\\alpha + M_{ij}) / (\\Gamma(\\alpha)\\alpha^{M_{ij}})`, the density
of the latent logit's propagated disturbance is the Fourier inverse of
:math:`\\prod_j \\pi M_{ij}t / \\sinh(\\pi M_{ij}t)`, and the latent probit's is
normal with variance :math:`\\sum_j M_{ij}^2`.  The Monte Carlo estimates must
match all three on a small lattice, for SAR, SEM and SDEM.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

import neighbayes.models as M
from neighbayes import dgp

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
    direct, indirect, total = m._latent_outcome_effects(n_mc=20000)
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
