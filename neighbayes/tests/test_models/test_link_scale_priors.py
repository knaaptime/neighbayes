"""Gelman et al. (2008) default priors on each family's link scale.

The default ``beta`` prior scales with the outcome: ``sd(y)`` for Gaussian
models, the unit scale of the linear predictor for log, logit and probit
links.  Count and binary models used to take the Gaussian formula on the count
or 0/1 scale, which centres a log-link intercept on ``mean(y)``.  The Tobit
models used fixed-scale priors and a proper prior on the censored gap, a
second density on data the regression already describes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logit, ndtri

from neighbayes.models._base._shared import gelman_default_beta_prior


def _design(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(2.0, 3.0, n)
    return np.column_stack([np.ones(n), x]), ["intercept", "x"], x


@pytest.mark.parametrize(
    "link, centre, unit",
    [
        ("log", lambda y: np.log(y.mean()), 1.0),
        ("logit", lambda y: logit(y.mean()), 1.0),
        ("probit", lambda y: ndtri(y.mean()), 1 / 1.6),
    ],
)
def test_link_scale_defaults(link, centre, unit):
    X, names, x = _design()
    rng = np.random.default_rng(1)
    y = rng.poisson(40.0, 400) if link == "log" else rng.binomial(1, 0.3, 400)
    mu, sd = gelman_default_beta_prior(y, X, names, link=link)
    np.testing.assert_allclose(mu, [centre(y), 0.0])
    np.testing.assert_allclose(sd, [2.5 * unit, 2.5 * unit / x.std()])


def test_log_link_is_free_of_the_count_scale():
    """Scaling the counts moves only the intercept's centre, by log of the factor."""
    X, names, _ = _design()
    y = np.random.default_rng(2).poisson(5.0, 400) + 1
    mu1, sd1 = gelman_default_beta_prior(y, X, names, link="log")
    mu2, sd2 = gelman_default_beta_prior(1000 * y, X, names, link="log")
    np.testing.assert_allclose(sd1, sd2)
    np.testing.assert_allclose(mu2 - mu1, [np.log(1000), 0.0])


def test_unknown_link_raises():
    X, names, _ = _design()
    with pytest.raises(ValueError, match="link"):
        gelman_default_beta_prior(np.ones(400), X, names, link="cloglog")


# -- Tobit -------------------------------------------------------------------


def _tobit(scale=1.0, seed=3):
    from neighbayes.dgp.nonlinear import simulate_sar_tobit

    return simulate_sar_tobit(
        n_side=12,
        rho=0.4,
        beta=np.array([0.5, 1.5]) * scale,
        sigma=scale,
        seed=seed,
    )


def test_tobit_priors_are_gaussian_and_gap_is_flat():
    from neighbayes.models import SARTobit

    d = _tobit()
    m = SARTobit(y=d["y"], X=d["X"], W=d["W_graph"])
    model = m._build_pymc_model()
    free = {v.name: v for v in model.free_RVs}
    assert "sigma2" in free and "sigma" in [v.name for v in model.deterministics]
    gap = free["y_cens_gap"]
    assert gap.owner.op.__class__.__name__.startswith("HalfFlat")
    mu, sd = m._resolved_beta_prior(m._X, list(m._model_coords()["coefficient"]))
    beta = model.named_vars["beta"]
    np.testing.assert_allclose(beta.owner.inputs[-1].eval(), sd)


@pytest.mark.slow
def test_tobit_estimates_are_free_of_the_scale_of_y():
    """A proper gap prior bent the fit once y outgrew its scale; a flat one does not."""
    from neighbayes.models import SARTobit

    fits = []
    for scale in (1.0, 50.0):
        d = _tobit(scale)
        post = (
            SARTobit(y=d["y"], X=d["X"], W=d["W_graph"])
            .fit(draws=600, tune=600, chains=2, random_seed=0, progressbar=False)
            .posterior
        )
        fits.append(
            np.r_[post["beta"].mean(("chain", "draw")).values, post["sigma"].mean()]
            / scale
        )
    np.testing.assert_allclose(fits[0], fits[1], atol=0.06)


# -- count flows -----------------------------------------------------------


def _nb_flow():
    from neighbayes.dgp import generate_negbin_flow_data

    return generate_negbin_flow_data(
        n=5, k=2, rho_d=0.1, rho_o=0.1, rho_w=0.0, alpha=5.0, seed=42
    )


def test_nb_flow_nuts_and_gibbs_resolve_the_same_priors(monkeypatch):
    import neighbayes.samplers.negbin_reduced._flow as nbf
    from neighbayes.models.flow._flow import SARNegBinFlowSeparable

    d = _nb_flow()
    m = SARNegBinFlowSeparable(d["y_vec"], d["X"], d["G"], logdet_method="eigenvalue")
    pv = m._flow_count_priors()
    np.testing.assert_allclose(pv["beta_sigma"][1:], 2.5 / m._X[:, 1:].std(0))

    model = m._build_pymc_model()
    beta = model.named_vars["beta"]
    np.testing.assert_allclose(beta.owner.inputs[-1].eval(), pv["beta_sigma"])
    np.testing.assert_allclose(beta.owner.inputs[-2].eval(), pv["beta_mu"])

    captured = {}

    class _Stop(Exception):
        pass

    def _capture(**kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(nbf, "FlowReducedGibbsPriors", _capture)
    with pytest.raises(_Stop):
        m.fit(sampler="gibbs", draws=5, tune=5, chains=1, progressbar=False)
    np.testing.assert_array_equal(captured["beta_sigma"], pv["beta_sigma"])
    np.testing.assert_array_equal(captured["beta_mu"], pv["beta_mu"])
    assert captured["alpha_sigma"] == pv["alpha_sigma"]
    assert captured["alpha_nu"] == pv["alpha_nu"]


def test_nb_flow_panel_alpha_matches_gibbs():
    """The panel NUTS build uses the half-t on alpha that the Gibbs path uses."""
    from neighbayes.dgp.flows import generate_panel_negbin_flow_data
    from neighbayes.models.flow_panel import SARNegBinFlowSeparablePanel

    d = generate_panel_negbin_flow_data(n=5, T=3, seed=1)
    m = SARNegBinFlowSeparablePanel(
        y=d["y"], W=d["G"], X=d["X"], T=3, col_names=d["col_names"]
    )
    alpha = m._build_pymc_model().named_vars["alpha"]
    assert "StudentT" in type(alpha.owner.op).__name__
