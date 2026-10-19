"""The SAR and SEM logit with random effects: learned σ, Gibbs against NUTS.

Neither model builds a PyMC model of its own (both are Gibbs-only), so the
reference here is written out: ``η = (I − ρW)⁻¹(Xβ + σz)`` (SAR) or
``η = Xβ + (I − λW)⁻¹σz`` (SEM), ``z ~ N(0, I)``, σ ~ half-Cauchy(1), under the
same priors the Gibbs sampler resolves.  The comparison guards the sweep's
validity: an earlier sweep reused η and β drawn before the ρ update, and left
the intercept's posterior a third too narrow.

It compares what the data identify: r, the attenuated ``β/g(σ)`` and log σ.
β and σ apart trade off along a ridge, and their means and sds are set by a
long tail of σ that NUTS itself samples with divergences and a few hundred
effective draws; two NUTS seeds disagree there as much as NUTS and Gibbs do.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from neighbayes.dgp import simulate_sar_logit, simulate_sem_logit
from neighbayes.models import SARLogit, SEMLogit
from neighbayes.samplers.logit._core import _attenuation

BACKENDS = ["numpy"] if sys.platform.startswith("win") else ["numpy", "jax"]


def _nuts_reference(model, y, X, W, sem):
    import pymc as pm
    import pytensor.tensor as pt

    from neighbayes._ops import SparseSARSolveOp

    bm, bs = model._resolved_beta_prior(link="logit")
    b = model._logdet_bounds
    par = "lam" if sem else "rho"
    with pm.Model():
        r = pm.Uniform(par, b.rho_min, b.rho_max)
        beta = pm.Normal("beta", mu=bm, sigma=bs, shape=X.shape[1])
        sd = pm.HalfCauchy("unit_effect_sd", beta=1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=len(y))
        if sem:
            eta = pt.dot(X, beta) + SparseSARSolveOp(W)(r, sd * z)
        else:
            eta = SparseSARSolveOp(W)(r, pt.dot(X, beta) + sd * z)
        pm.Bernoulli("obs", logit_p=eta, observed=y)
        return pm.sample(
            draws=2000, tune=2000, chains=4, random_seed=2, progressbar=False,
            target_accept=0.95,
        ).posterior  # fmt: skip


def test_posterior_carries_the_effect_sd():
    d = simulate_sar_logit(n_side=6, unit_effects="random", seed=0)
    m = SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"], unit_effects="random")
    post = m.fit(
        draws=20, tune=20, chains=1, random_seed=0, progressbar=False
    ).posterior
    assert "unit_effect_sd" in post and np.all(post["unit_effect_sd"].values > 0)


def test_effect_sd_prior_keys_are_accepted():
    d = simulate_sem_logit(n_side=6, unit_effects="random", seed=0)
    m = SEMLogit(
        y=d["y"], X=d["X"], W=d["W_sparse"], unit_effects="random",
        priors={"unit_effect_sd_nu": 3.0, "unit_effect_sd_scale": 0.5},
    )  # fmt: skip
    m.fit(draws=10, tune=10, chains=1, random_seed=0, progressbar=False)


def _identified(post, par):
    """``r``, ``β/g(σ)`` and ``log σ``, flattened over chains."""
    s = post["unit_effect_sd"].values.reshape(-1)
    b = post["beta"].values.reshape(s.size, -1)
    return {
        par: post[par].values.reshape(-1),
        "attenuated_beta": b / _attenuation(s)[:, None],
        "log_unit_effect_sd": np.log(s),
    }


@pytest.mark.slow
@pytest.mark.parametrize("sem", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_gibbs_matches_nuts(sem, backend):
    sim, cls = (simulate_sem_logit, SEMLogit) if sem else (simulate_sar_logit, SARLogit)
    par = "lam" if sem else "rho"
    d = sim(n_side=12, beta=[0.3, 1.0], unit_effects="random", seed=0, **{par: 0.5})
    y, X, W = d["y"], d["X"], d["W_sparse"]
    m = cls(y=y, X=X, W=W, unit_effects="random")
    gibbs = m.fit(
        draws=3000, tune=1000, chains=4, random_seed=1, progressbar=False,
        backend=backend,
    ).posterior  # fmt: skip
    a_all, b_all = (
        _identified(gibbs, par),
        _identified(_nuts_reference(m, y, X, W, sem), par),
    )
    for var, a in a_all.items():
        b = b_all[var]
        assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.2 * b.std(0)), var
        np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.2, err_msg=var)
