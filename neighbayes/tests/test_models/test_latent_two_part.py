"""The latent SAR hurdle and zero-inflated NB: designs, simulators, fits, summaries.

Both sweeps were checked exact by Geweke's (2004) joint-distribution test (the
parameters' marginals reproduce their priors when each sweep alternates with a
fresh draw of the latent state and counts); these tests pin the wiring.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

import neighbayes.models as M
from neighbayes import dgp
from neighbayes.models.cross_section._latent_two_part import (
    _draw_truncated_poisson,
    _truncated_mean,
)

MODELS = {
    "zinb": (M.SARZINB, dgp.simulate_sar_zinb),
    "hurdle": (M.SARHurdleNegBin, dgp.simulate_sar_hurdle_negbin),
}
QUICK = dict(draws=40, tune=40, chains=2, progressbar=False, random_seed=0)


def _model(which, n_side=5, seed=1, **kw):
    cls, sim = MODELS[which]
    d = sim(n_side=n_side, seed=seed, latent=True, **kw)
    m = cls(y=d["y"], X=d["X"], X_zero=d["X_zero"], W=d["W_sparse"], latent=True)
    return d, m


@pytest.mark.parametrize("which", list(MODELS))
def test_designs(which):
    cls, sim = MODELS[which]
    d, m = _model(which)
    assert m._design == "latent"
    assert m._gibbs_key == (f"{which}_latent", "cross_section")
    assert cls(y=d["y"], X=d["X"], W=d["W_sparse"])._design == "static"
    panel = dict(y=np.r_[d["y"], d["y"]], X=np.r_[d["X"], d["X"]], W=d["W_sparse"])
    assert cls(**panel, N=25, T=2, latent=True)._design == "latent"
    with pytest.raises(NotImplementedError, match="Gibbs sampling"):
        m._build_pymc_model_latent()


@pytest.mark.parametrize("which", list(MODELS))
def test_simulators_draw_the_latent_forms(which):
    d, _ = _model(which, n_side=12, seed=3)
    on = d["eta_zero"] > 0
    np.testing.assert_array_equal(d["d"], on.astype(float))
    assert np.all(d["y"][~on] == 0)
    if which == "hurdle":
        assert np.all(d["y"][on] >= 1)
    t = MODELS[which][1](n_side=12, seed=3, latent=True, target_pi=0.3)
    assert abs(np.mean(t["d"]) - 0.3) < 0.1


PANEL = dict(unit_effects="random", time_effects=True)


def _panel_model(which, n_side=5, T=3, seed=1, **kw):
    cls, sim = MODELS[which]
    d = sim(n_side=n_side, T=T, seed=seed, latent=True, **PANEL, **kw)
    m = cls(
        y=d["y"],
        X=d["X"],
        X_zero=d["X_zero"],
        W=d["W_sparse"],
        N=n_side**2,
        T=T,
        latent=True,
        **PANEL,
    )
    return d, m


@pytest.mark.parametrize("which", list(MODELS))
def test_panel_simulators_draw_the_latent_forms(which):
    d, _ = _panel_model(which, seed=3)
    on = d["eta_zero"] > 0
    np.testing.assert_array_equal(d["d"], on.astype(float))
    assert np.all(d["y"][~on] == 0)
    if which == "hurdle":
        assert np.all(d["y"][on] >= 1)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize("which", list(MODELS))
def test_panel_fit_and_summaries(which, backend):
    """Count effects in both models; the hurdle's zero part takes its own."""
    d, m = _panel_model(which)
    idata = m.fit(**QUICK, backend=backend, idata_kwargs={"log_likelihood": True})
    post = idata.posterior
    zero = which == "hurdle"
    for suffix, present in (("", True), ("_zero", zero)):
        names = {
            f"time_effect{suffix}",
            f"unit_effect{suffix}",
            f"unit_effect_sd{suffix}",
        }
        assert (names <= set(post.data_vars)) == present, suffix
    assert post["time_effect"].shape == (2, 40, 2)
    assert post["unit_effect"].shape == (2, 40, 25)
    n = d["y"].size
    ll = idata.log_likelihood["obs"].values
    assert ll.shape == (2, 40, n) and np.all(np.isfinite(ll)) and np.all(ll <= 0)
    az.loo(idata)
    assert m.activation_probabilities(draws=5).shape == (n,)
    assert m.posterior_predictive(n_draws=3, random_seed=1).shape == (3, n)
    for equation, outcome in (("zero", "probability"), ("count", "count")):
        assert np.isfinite(m.impacts(equation=equation).to_numpy(float)).all()
        out = m.impacts(equation=equation, scale=outcome)
        assert np.isfinite(out.to_numpy(float)).all()
    if which == "zinb":
        za = m.zero_attribution()
        assert za["structural_prob"].shape == (np.sum(d["y"] == 0),)


def test_truncated_poisson():
    rng = np.random.default_rng(0)
    for lam in (1e-6, 0.3, 4.0):
        y = _draw_truncated_poisson(rng, np.full(200_000, lam))
        assert y.min() >= 1
        assert abs(y.mean() - _truncated_mean(np.array([lam]))[0]) < 0.01


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize("which", list(MODELS))
def test_fit_and_summaries(which, backend):
    d, m = _model(which)
    idata = m.fit(**QUICK, backend=backend, idata_kwargs={"log_likelihood": True})
    post = idata.posterior
    assert {"rho", "rho_zero", "alpha", "beta", "beta_zero"} <= set(post.data_vars)
    assert post["beta"].shape == (2, 40, 2) and post["beta_zero"].shape == (2, 40, 2)
    ll = idata.log_likelihood["obs"].values
    assert ll.shape == (2, 40, 25) and np.all(np.isfinite(ll)) and np.all(ll <= 0)
    az.loo(idata)
    n = d["y"].size
    assert m.activation_probabilities(draws=5).shape == (n,)
    assert m.fitted_values().shape == (n,)
    pp = m.posterior_predictive(n_draws=7, random_seed=1)
    assert pp.shape == (7, n) and np.all(pp >= 0)
    for equation, outcome in (("zero", "probability"), ("count", "count")):
        link = m.impacts(equation=equation)
        assert link.attrs["scale"] == ("logodds" if equation == "zero" else "logmean")
        out = m.impacts(equation=equation, scale=outcome)
        assert out.attrs["scale"] == outcome
        assert np.isfinite(out.to_numpy(float)).all()
    with pytest.raises(ValueError, match="scale must be"):
        m.impacts(equation="zero", scale="count")
    if which == "zinb":
        za = m.zero_attribution()
        assert za["structural_prob"].shape == za["zero_indices"].shape
        assert np.all((za["structural_prob"] >= 0) & (za["structural_prob"] <= 1))
    else:
        assert np.all(m.positive_mean(draws=5) >= 1.0)
        if backend == "numpy":
            assert np.all(pp[:, d["y"] == 0] >= 0)


def test_reduced_form_reports_the_link_scale_only():
    cls, sim = MODELS["hurdle"]
    d = sim(n_side=5, seed=1)
    m = cls(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    assert m.impacts(equation="zero", scale="logodds").attrs["scale"] == "logodds"
    with pytest.raises(NotImplementedError, match="link scale"):
        m.impacts(equation="zero", scale="probability")


def test_hurdle_zero_part_is_the_latent_logit():
    """The hurdle's parts share nothing: its zero part is SARLogit(latent=True) on d."""
    d, m = _model("hurdle", n_side=8, seed=5)
    kw = dict(draws=1500, tune=500, chains=2, progressbar=False, backend="numpy")
    hurdle = m.fit(**kw, random_seed=0).posterior
    logit = (
        M.SARLogit(y=d["d"], X=d["X_zero"], W=d["W_sparse"], latent=True)
        .fit(**kw, random_seed=1)
        .posterior
    )
    for a, b in (("rho_zero", "rho"), ("beta_zero", "beta")):
        x, z = hurdle[a].values, logit[b].values
        se = np.sqrt(x.var(axis=(0, 1)) / az.ess(hurdle)[a].values.clip(50))
        diff = np.abs(x.mean(axis=(0, 1)) - z.mean(axis=(0, 1)))
        assert np.all(diff < 4 * np.sqrt(2) * se), (a, diff, se)


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize("which", list(MODELS))
def test_recovery(which, backend):
    d, m = _model(which, n_side=20, seed=104, rho=0.5, rho_zero=0.4, alpha=3.0)
    post = m.fit(
        draws=1500,
        tune=1000,
        chains=2,
        progressbar=False,
        random_seed=0,
        backend=backend,
    ).posterior
    truth = d["params_true"]
    for name in ("rho", "rho_zero", "alpha", "beta", "beta_zero"):
        v = post[name].values.reshape(-1, *post[name].shape[2:])
        z = (v.mean(0) - np.atleast_1d(truth[name])) / v.std(0)
        assert np.all(np.abs(z) < 3.5), (name, z)
