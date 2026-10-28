"""Fast build/method/recovery tests for non-spatial Logit and NegBin."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from neighbayes.models import Logit, NegBin
from neighbayes.models.priors import LogitPriors, NegBinPriors
from neighbayes.tests.helpers import W_to_graph, make_line_W

# ---------------------------------------------------------------------------
# Logit
# ---------------------------------------------------------------------------


def _binary_data(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([0.1, 0.7, -0.5])
    p = 1.0 / (1.0 + np.exp(-X @ beta))
    y = (rng.uniform(size=n) < p).astype(float)
    return y, X, beta


def test_logit_build_pymc_model():
    y, X, _ = _binary_data()
    m = Logit(y=y, X=X)
    pymc_model = m._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)
    assert "beta" in pymc_model.named_vars
    assert "obs" in pymc_model.named_vars


def test_logit_rejects_non_binary_y():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(5), rng.normal(size=5)])
    with pytest.raises(ValueError, match="binary"):
        Logit(y=np.array([0.0, 1.0, 2.0, 0.0, 1.0]), X=X)


def test_logit_rejects_robust():
    y, X, _ = _binary_data()
    with pytest.raises(NotImplementedError, match="robust"):
        Logit(y=y, X=X, robust=True)


def test_logit_priors_typed():
    y, X, _ = _binary_data()
    m = Logit(y=y, X=X, priors=LogitPriors(beta_sigma=5.0))
    assert m.priors["beta_sigma"] == 5.0


def test_logit_accepts_optional_W():
    y, X, _ = _binary_data()
    W = W_to_graph(make_line_W(len(y)))
    m = Logit(y=y, X=X, W=W)
    assert m._W_sparse is not None


def test_logit_fit_recovers_coefficients():
    y, X, beta_true = _binary_data(seed=1, n=400)
    m = Logit(y=y, X=X)
    idata = m.fit(draws=400, tune=400, chains=2, progressbar=False, random_seed=42)
    beta_post = idata.posterior["beta"].mean(("chain", "draw")).values
    assert np.allclose(beta_post, beta_true, atol=0.4)


def test_logit_impacts_raises():
    y, X, _ = _binary_data()
    m = Logit(y=y, X=X)
    with pytest.raises(NotImplementedError, match="no spatial structure"):
        m._compute_impacts_posterior()


# ---------------------------------------------------------------------------
# NegBin
# ---------------------------------------------------------------------------


def _count_data(seed: int = 0, n: int = 300, alpha_true: float = 2.0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([0.5, 0.4, -0.3])
    mu = np.exp(X @ beta)
    y = rng.negative_binomial(alpha_true, alpha_true / (alpha_true + mu)).astype(float)
    return y, X, beta, alpha_true


def test_negbin_build_pymc_model():
    y, X, _, _ = _count_data()
    m = NegBin(y=y, X=X)
    pymc_model = m._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)
    assert "beta" in pymc_model.named_vars
    assert "alpha" in pymc_model.named_vars
    assert "obs" in pymc_model.named_vars


def test_negbin_rejects_noninteger_or_negative_y():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(4), rng.normal(size=4)])
    with pytest.raises(ValueError, match="integer-valued"):
        NegBin(y=np.array([0.0, 1.2, 2.0, 1.0]), X=X)
    with pytest.raises(ValueError, match="non-negative"):
        NegBin(y=np.array([0.0, 1.0, -1.0, 2.0]), X=X)


def test_negbin_rejects_robust():
    y, X, _, _ = _count_data()
    with pytest.raises(NotImplementedError, match="robust"):
        NegBin(y=y, X=X, robust=True)


def test_negbin_priors_typed():
    y, X, _, _ = _count_data()
    m = NegBin(y=y, X=X, priors=NegBinPriors(alpha_nu=5.0, alpha_sigma=1.5))
    assert m.priors["alpha_nu"] == 5.0
    assert m.priors["alpha_sigma"] == 1.5


def test_negbin_fit_recovers_coefficients_and_alpha():
    y, X, beta_true, alpha_true = _count_data(seed=2, n=500, alpha_true=2.0)
    m = NegBin(y=y, X=X)
    idata = m.fit(draws=500, tune=500, chains=2, progressbar=False, random_seed=7)
    beta_post = idata.posterior["beta"].mean(("chain", "draw")).values
    alpha_post = float(idata.posterior["alpha"].mean())
    assert np.allclose(beta_post, beta_true, atol=0.25)
    assert 0.5 * alpha_true < alpha_post < 2.0 * alpha_true


def test_negbin_impacts_raises():
    y, X, _, _ = _count_data()
    m = NegBin(y=y, X=X)
    with pytest.raises(NotImplementedError, match="no spatial structure"):
        m._compute_impacts_posterior()
