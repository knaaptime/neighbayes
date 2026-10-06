"""Reduced-form SAR hurdle NB: SARHurdleNB, the hurdle panels and the flow hurdle.

Fast tests pin the truncated-NB pieces, the DGPs, construction, and the API
of every class on short chains.  The ``slow`` tests check the augmentation
against quadrature, the binary half against ``SARLogit`` fit on ``d``, and
the Gibbs samplers against NUTS and against each other.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from neighbayes import dgp
from neighbayes.dgp.hurdle import truncated_negative_binomial
from neighbayes.models import HurdleNBPanel, SARHurdleNB, SARHurdleNBPanel
from neighbayes.samplers.hurdle._truncated import (
    draw_missed_zeros,
    hurdle_loglik_pointwise,
    log_nb_zero,
    truncated_nb_loglik_pointwise,
)

QUICK = dict(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)


def _xs(n_side=8, seed=0, **kw):
    return dgp.simulate_sar_hurdle(n_side=n_side, seed=seed, **kw)


def _panel(seed=1, **kw):
    return dgp.simulate_panel_sar_hurdle(
        N=25, T=4, n_side=5, seed=seed, unit_effect_sd=0.5, time_effect_sd=0.3,
        sel_unit_effect_sd=0.5, sel_time_effect_sd=0.3, **kw,
    )  # fmt: skip


def _agree(a, b, var, mean_tol=0.2, sd_tol=0.15):
    x = a[var].values.reshape(-1, *a[var].shape[2:])
    r = b[var].values.reshape(-1, *b[var].shape[2:])
    assert np.all(np.abs(x.mean(0) - r.mean(0)) < mean_tol * r.std(0)), var
    np.testing.assert_allclose(x.std(0), r.std(0), rtol=sd_tol, err_msg=var)


def _agree_mc(a, b, var, z=4.0, sd_tol=0.3, min_ess=400):
    """Two posteriors agree within Monte Carlo error.

    Means within ``z`` combined MC standard errors (``sd / sqrt(ESS)``); sds
    within ``sd_tol`` where both chains have at least ``min_ess`` effective
    draws (slowly mixing components are compared on means only).  Positive
    scale parameters (``alpha`` and the effect sds), heavy-tailed on small
    data, are compared on the log scale.
    """
    import arviz as az

    tf = np.log if var in ("alpha", "group_sd", "sel_group_sd") else (lambda v: v)
    x = tf(a[var].values.reshape(a[var].shape[0], a[var].shape[1], -1))
    r = tf(b[var].values.reshape(b[var].shape[0], b[var].shape[1], -1))
    for j in range(x.shape[-1]):
        xa, ra = x[..., j], r[..., j]
        ex, er = float(az.ess(xa)), float(az.ess(ra))
        se = np.sqrt(xa.var() / ex + ra.var() / er)
        assert abs(xa.mean() - ra.mean()) < z * se, (var, j, xa.mean(), ra.mean(), se)
        if min(ex, er) >= min_ess:
            assert abs(xa.std() / ra.std() - 1.0) < sd_tol, (var, j)


# ---------------------------------------------------------------------------
# Truncated NB pieces
# ---------------------------------------------------------------------------


class TestTruncated:
    def test_truncated_pmf_normalizes(self):
        y = np.arange(1, 4000, dtype=float)
        for mu, a in [(0.3, 2.0), (5.0, 0.5), (20.0, 3.0)]:
            ll = truncated_nb_loglik_pointwise(y, np.full(y.size, np.log(mu)), a)
            assert abs(np.exp(logsumexp(ll)) - 1.0) < 1e-8

    def test_missed_zeros_are_geometric(self):
        rng = np.random.default_rng(0)
        eta = np.full(200_000, np.log(0.7))
        p0 = np.exp(log_nb_zero(eta[:1], 2.0))[0]
        m = draw_missed_zeros(eta, 2.0, rng)
        assert m.min() == 0
        np.testing.assert_allclose(m.mean(), p0 / (1 - p0), rtol=0.02)

    def test_hurdle_loglik(self):
        y = np.array([0.0, 3.0])
        eta_b, eta_c = np.array([0.4, -0.2]), np.array([1.0, 1.0])
        ll = hurdle_loglik_pointwise(y, eta_b, eta_c, 2.0)
        assert np.isclose(ll[0], -np.logaddexp(0.0, 0.4))
        tnb = truncated_nb_loglik_pointwise(np.array([3.0]), np.array([1.0]), 2.0)
        assert np.isclose(ll[1], -np.logaddexp(0.0, 0.2) + tnb[0])


class TestDGP:
    def test_cross_section(self):
        from neighbayes.dgp.utils import spatial_filter_factor

        d = _xs(lam=0.5, target_pi=0.6)
        expected = spatial_filter_factor(d["W_sel_sparse"], 0.5)(
            d["Z"] @ d["params_true"]["gamma"]
        )
        np.testing.assert_allclose(d["eta_bin"], expected, atol=1e-10)
        assert np.all((d["y"] > 0) == (d["d"] == 1))

    def test_truncated_draws_positive(self):
        y = truncated_negative_binomial(
            np.random.default_rng(0), np.full(5000, 0.05), 0.5
        )
        assert y.min() >= 1

    def test_panel(self):
        d = _panel(effect_corr=0.6)
        assert d["y"].shape == (100,) and np.all((d["y"] > 0) == (d["d"] == 1))
        assert {"sel_unit_effect", "sel_time_effect", "lam", "gamma"} <= set(
            d["params_true"]
        )


# ---------------------------------------------------------------------------
# Generic machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effects", ["groups", "periods", "intercept"])
def test_level_direction_shifts_eta(effects):
    from neighbayes.samplers._utils._glm_equation import MaterializedEquation
    from neighbayes.samplers._utils._group_effects import GroupEffects
    from neighbayes.samplers.count_panel import CountPanelFE
    from neighbayes.samplers.count_panel._filters import SARFilter

    d = _panel()
    N, T = 25, 4
    rng = np.random.default_rng(0)
    groups = GroupEffects(np.tile(np.arange(N), T), N, 0.0, 1.0)
    D = np.kron(np.eye(T), np.ones((N, 1)))
    X = d["X"] if effects == "intercept" else d["X"][:, 1:]
    fe = {
        "groups": CountPanelFE(groups=groups),
        "periods": CountPanelFE(D_tau=D),
        "intercept": CountPanelFE(),
    }[effects]
    k = X.shape[1] + fe.n_tau
    c = rng.normal(size=N) if effects == "groups" else None
    eq = MaterializedEquation(
        X, SARFilter(d["W_sparse"], T, -0.9, 0.9), np.zeros(k), np.ones(k), fe,
        theta=rng.normal(size=k), c=c, params={"rho": 0.3}, rng=rng,
    )  # fmt: skip
    v, log_prior, apply = eq.level_direction()
    before = eq.eta.copy()
    apply(0.7)
    np.testing.assert_allclose(eq.eta, eq._eta_of(eq.U), atol=1e-10)
    np.testing.assert_allclose(eq.eta, before + 0.7 * v, atol=1e-10)


# ---------------------------------------------------------------------------
# SARHurdleNB
# ---------------------------------------------------------------------------


class TestSARHurdleNB:
    def test_api(self):
        d = _xs()
        m = SARHurdleNB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
        idata = m.fit(idata_kwargs={"log_likelihood": True}, **QUICK)
        assert {"rho", "lam", "beta", "gamma", "alpha"} <= set(
            idata.posterior.data_vars
        )
        assert idata["log_likelihood"]["obs"].shape[-1] == d["y"].size
        assert m.corridor_probabilities(draws=5).shape == d["y"].shape
        assert np.all(m.conditional_mean(draws=5) >= 1.0)
        pp = m.posterior_predictive(n_draws=4, random_seed=0)
        assert pp.shape == (4, d["y"].size) and pp.min() >= 0
        for eq in ("count", "selection", "hurdle"):
            assert len(m.spatial_effects(eq)) == 1

    @pytest.mark.parametrize("sampler", ["gibbs", "nuts"])
    def test_alpha_fixed(self, sampler):
        d = _xs()
        m = SARHurdleNB(y=d["y"], X=d["X"], W=d["W_graph"], priors={"alpha_fixed": 7.0})
        post = m.fit(sampler=sampler, **QUICK).posterior
        assert np.all(post["alpha"].values == 7.0)

    def test_names_and_formula(self):
        d = _xs()
        df = pd.DataFrame({"y": d["y"], "x1": d["X"][:, 1], "z1": d["Z"][:, 1]})
        m = SARHurdleNB(formula="y ~ x1", data=df, sel_formula="~ z1", W=d["W_graph"])
        assert m._sel_feature_names == ["Intercept", "z1"]
        post = m.fit(**QUICK).posterior
        assert list(post["gamma"].coords["sel_coefficient"].values) == [
            "Intercept",
            "z1",
        ]

    def test_guards(self):
        d = _xs()
        with pytest.raises(ValueError, match="positive count"):
            SARHurdleNB(y=np.zeros_like(d["y"]), X=d["X"], W=d["W_graph"])
        with pytest.raises(ValueError, match="integer"):
            SARHurdleNB(y=d["y"] + 0.5, X=d["X"], W=d["W_graph"])


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SARHurdleNBPanel, HurdleNBPanel])
@pytest.mark.parametrize("effects", [0, 1, 2, 3])
@pytest.mark.parametrize("sampler", ["gibbs", "nuts"])
def test_hurdle_panel_api(cls, effects, sampler):
    d = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = cls(
            y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=4, effects=effects
        )
        post = m.fit(sampler=sampler, **QUICK).posterior
    names = set(post.data_vars)
    assert {"beta", "gamma", "alpha"} <= names
    assert ("rho" in names) == (cls is SARHurdleNBPanel)
    for pre in ("", "sel_"):
        assert (pre + "group_effect" in names) == (effects in (1, 3))
        assert (pre + "time_effect" in names) == (effects in (2, 3))
    assert m.posterior_predictive(n_draws=2, random_seed=0).shape == (2, 100)


def test_hurdle_panel_registry_and_storage():
    from neighbayes.samplers._registry import resolve

    assert resolve("hurdle", "panel") is not None
    assert resolve("hurdle", "cross_section") is not None
    d = _panel()
    m = SARHurdleNBPanel(
        y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=4, effects=1
    )
    idata = m.fit(store_group_effects=False, **QUICK)
    assert "group_effect" not in idata.posterior.data_vars
    assert {"group_effect_summary", "sel_group_effect_summary"} <= set(idata.children)
    assert m.corridor_probabilities(draws=3).shape == (100,)


def test_hurdle_panel_absorbs_and_warns():
    d = _panel()
    trend = np.repeat(np.arange(4.0), 25)  # varies only over time
    Z = np.column_stack([d["Z"], np.tile(np.arange(25.0), 4), trend])
    with pytest.warns(
        UserWarning, match="absorbed by the period effects in the binary"
    ):
        m = SARHurdleNBPanel(
            y=d["y"], X=d["X"], Z=Z, W=d["W_sparse"], N=25, T=4, effects=3
        )
    # Unit effects absorb nothing; the period effects absorb the trend.
    assert m._Z.shape[1] == 3
    y = d["y"].copy()
    y.reshape(4, 25)[:, 0] = 0  # unit 0 never positive
    with pytest.warns(UserWarning, match="never switch"):
        SARHurdleNBPanel(y=y, X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=4, effects=1)


def test_hurdle_panel_learned_sds_and_mundlak():
    d = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBPanel(
            y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=4, effects=1,
            mundlak=True,
        )  # fmt: skip
    assert m._feature_names[-1].endswith("_mean")
    assert m._sel_feature_names[-1].endswith("_mean")
    for sampler in ("gibbs", "nuts"):
        post = m.fit(sampler=sampler, **QUICK).posterior
        assert {"group_sd", "sel_group_sd"} <= set(post.data_vars)
    assert len(m.spatial_effects("selection")) == 1  # the mean is not an effect


# ---------------------------------------------------------------------------
# Slow: exactness
# ---------------------------------------------------------------------------


def _quadrature(y, prior_sd=2.5, a_sigma=2.5, a_nu=3.0):
    bg = np.linspace(-12.0, 6.0, 901)
    lg = np.linspace(-8.0, 7.0, 301)
    LP = np.empty((bg.size, lg.size))
    for j, la in enumerate(lg):
        a = np.exp(la)
        prior = la - 0.5 * (a_nu + 1) * np.log1p(a * a / (a_nu * a_sigma**2))
        for i, b in enumerate(bg):
            LP[i, j] = truncated_nb_loglik_pointwise(y, np.full(y.size, b), a).sum()
        LP[:, j] += prior - 0.5 * (bg / prior_sd) ** 2
    W = np.exp(LP - logsumexp(LP))
    wb, wa, A = W.sum(1), W.sum(0), np.exp(lg)
    mb, ma = wb @ bg, wa @ A
    return (mb, np.sqrt(wb @ (bg - mb) ** 2)), (ma, np.sqrt(wa @ (A - ma) ** 2))


@pytest.mark.slow
@pytest.mark.parametrize("mu", [0.3, 1.0, 5.0])
def test_truncated_count_matches_quadrature(mu):
    """Intercept-only count half through the full kernel vs a 2-D quadrature.

    At ``μ = 0.3`` the positives are mostly ones and ``(β₀, α)`` lie on a
    ridge; the joint level/α slice is what lets the chain cover it.
    """
    from neighbayes.samplers.count_panel import CountPanelFE
    from neighbayes.samplers.count_panel._filters import NoFilter
    from neighbayes.samplers.hurdle._generic import HurdlePriors, run_chain

    rng = np.random.default_rng(42)
    y = truncated_negative_binomial(rng, np.full(200, mu), 2.0).astype(float)
    y = np.concatenate([y, np.zeros(50)])  # the binary half needs both outcomes
    one = np.ones((y.size, 1))
    priors = HurdlePriors(
        gamma_mu=np.zeros(1), gamma_sigma=np.full(1, 2.5),
        theta_mu=np.zeros(1), theta_sigma=np.full(1, 2.5),
    )  # fmt: skip
    outs = [
        run_chain(
            y,
            one,
            one,
            NoFilter(),
            NoFilter(),
            CountPanelFE(),
            CountPanelFE(),
            priors,
            6000,
            1500,
            rng=np.random.default_rng(s),
        )  # fmt: skip
        for s in range(4)
    ]
    b = np.concatenate([o["beta"][:, 0] for o in outs])
    a = np.concatenate([o["alpha"] for o in outs])
    (qb, sb), (qa, sa) = _quadrature(y[y > 0])
    assert abs(b.mean() - qb) < 0.1 * sb and abs(b.std() / sb - 1) < 0.1
    assert abs(a.mean() - qa) < 0.1 * sa and abs(a.std() / sa - 1) < 0.15


@pytest.mark.slow
def test_binary_half_matches_sar_logit():
    """With unlinked halves the binary half is exactly SARLogit fit on d."""
    from neighbayes.models import SARLogit

    d = _xs(n_side=15, seed=3, lam=0.4)
    m = SARHurdleNB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
    lo, hi = m._bounds("lam")
    h = m.fit(draws=3000, tune=1000, chains=4, progressbar=False, random_seed=1)
    logit = SARLogit(
        y=d["d"], X=d["Z"], W=d["W_graph"], priors={"rho_lower": lo, "rho_upper": hi}
    )
    r = logit.fit(draws=3000, tune=1000, chains=4, progressbar=False, random_seed=2)
    hp, rp = h.posterior, r.posterior
    x, ref = hp["lam"].values.ravel(), rp["rho"].values.ravel()
    assert abs(x.mean() - ref.mean()) < 0.15 * ref.std()
    np.testing.assert_allclose(x.std(), ref.std(), rtol=0.15)
    g, gr = hp["gamma"].values.reshape(-1, 2), rp["beta"].values.reshape(-1, 2)
    assert np.all(np.abs(g.mean(0) - gr.mean(0)) < 0.15 * gr.std(0))
    np.testing.assert_allclose(g.std(0), gr.std(0), rtol=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("alpha_fixed", [None, 5.0])
def test_sarhurdle_gibbs_matches_nuts(alpha_fixed):
    d = _xs(n_side=10, seed=4)
    priors = {} if alpha_fixed is None else {"alpha_fixed": alpha_fixed}
    m = SARHurdleNB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"], priors=priors)
    g = m.fit(draws=4000, tune=1000, chains=4, progressbar=False, random_seed=1)
    n = m.fit(
        sampler="nuts", draws=2000, tune=1500, chains=4, progressbar=False,
        random_seed=2, target_accept=0.95,
    )  # fmt: skip
    for var in ("rho", "lam", "beta", "gamma") + (
        ("alpha",) if alpha_fixed is None else ()
    ):
        _agree(g.posterior, n.posterior, var)


@pytest.mark.slow
def test_hurdle_panel_gibbs_matches_nuts():
    d = _panel(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBPanel(
            y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=4, effects=3
        )
    g = m.fit(draws=4000, tune=1000, chains=4, progressbar=False, random_seed=1)
    n = m.fit(
        sampler="nuts", draws=2000, tune=1500, chains=4, progressbar=False,
        random_seed=2, target_accept=0.95,
    )  # fmt: skip
    for var in ("rho", "lam", "beta", "gamma", "alpha", "time_effect",
                "sel_time_effect", "group_effect", "sel_group_effect"):  # fmt: skip
        _agree(g.posterior, n.posterior, var, mean_tol=0.25, sd_tol=0.2)


# ---------------------------------------------------------------------------
# Flow hurdle
# ---------------------------------------------------------------------------


def _flow(T=1, seed=2, n=6, **kw):
    return dgp.generate_hurdle_flow_data_separable(
        n=n, T=T, seed=seed, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4,
        alpha=3.0, **kw,
    )  # fmt: skip


def _flow_panel_data(n=5, T=3, seed=5):
    return _flow(
        T=T, n=n, seed=seed, pair_effect_sd=0.5, time_effect_sd=0.3,
        sel_pair_effect_sd=0.5, sel_time_effect_sd=0.3,
    )  # fmt: skip


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax), "nuts"]
)
def test_flow_hurdle_cross_section(backend):
    from neighbayes.models import SARHurdleNBFlowSeparable

    d = _flow()
    m = SARHurdleNBFlowSeparable(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
    kw = {"sampler": "nuts"} if backend == "nuts" else {"gibbs_backend": backend}
    idata = m.fit(idata_kwargs={"log_likelihood": True}, **kw, **QUICK)
    names = set(idata.posterior.data_vars)
    assert {"lam_d", "lam_o", "rho_d", "rho_o", "beta", "gamma", "alpha"} <= names
    assert idata["log_likelihood"]["obs"].shape[-1] == d["y_vec"].size
    assert m.spatial_effects("hurdle", draws=5).shape[0] > 0
    assert m.conditional_mean(draws=3).shape == (36,)


@pytest.mark.parametrize("effects", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax), "nuts"]
)
def test_flow_hurdle_panel(effects, backend):
    from neighbayes.models import SARHurdleNBFlowSeparablePanel

    d = _flow_panel_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=3, col_names=d["col_names"], effects=effects
        )
        kw = {"sampler": "nuts"} if backend == "nuts" else {"gibbs_backend": backend}
        post = m.fit(**kw, **QUICK).posterior
    names = set(post.data_vars)
    for pre in ("", "sel_"):
        assert (pre + "group_effect" in names) == (effects in (1, 3))
        assert (pre + "group_sd" in names) == (effects in (1, 3))
        assert (pre + "time_effect" in names) == (effects in (2, 3))
    assert m.posterior_predictive(n_draws=2, random_seed=0).shape == (2, 75)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_flow_hurdle_alpha_fixed_and_summaries(backend):
    from neighbayes.models import SARHurdleNBFlowSeparablePanel

    d = _flow_panel_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=3, col_names=d["col_names"], effects=1,
            priors={"alpha_fixed": 4.0},
        )  # fmt: skip
        idata = m.fit(gibbs_backend=backend, store_group_effects=False, **QUICK)
    assert np.all(idata.posterior["alpha"].values == 4.0)
    assert {"group_effect_summary", "sel_group_effect_summary"} <= set(idata.children)


def test_flow_hurdle_guards():
    from neighbayes.models import (
        SARHurdleNBFlowSeparable,
        SARHurdleNBFlowSeparablePanel,
    )

    d = _flow()
    with pytest.raises(ValueError, match="cross-section"):
        SARHurdleNBFlowSeparable(d["y_vec"], d["X"], d["G"], effects=1)
    with pytest.raises(ValueError, match="rows"):
        SARHurdleNBFlowSeparable(d["y_vec"], d["X"], d["G"], Z=np.ones((5, 1)))
    with pytest.raises(ValueError, match="positive count"):
        SARHurdleNBFlowSeparable(np.zeros_like(d["y_vec"]), d["X"], d["G"])
    m = SARHurdleNBFlowSeparablePanel(
        d["y_vec"], d["X"], d["G"], T=1, Z=d["X"][:, :2], col_names=d["col_names"]
    )
    m.fit(**QUICK)
    with pytest.raises(NotImplementedError, match="flow design"):
        m.spatial_effects("selection")


@pytest.mark.slow
def test_flow_hurdle_structured_matches_generic():
    """The structured n × n sweep against the materialized-U kernel (pair effects)."""
    from neighbayes.models import SARHurdleNBFlowSeparablePanel
    from neighbayes.samplers.count_panel._filters import FlowSeparableFilter
    from neighbayes.samplers.hurdle._generic import HurdlePriors, run_chain

    # Enough positive weeks per pair that the count half is identified; on a
    # tiny panel (~2 positive weeks per pair) both samplers mix too slowly
    # for a tight comparison.
    d = _flow(
        T=6, n=8, seed=7, target_pi=0.7, pair_effect_sd=0.5, time_effect_sd=0.3,
        sel_pair_effect_sd=0.5, sel_time_effect_sd=0.3,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=6, col_names=d["col_names"], effects=3
        )
    s = m.fit(
        draws=4000, tune=1000, chains=4, progressbar=False, random_seed=1
    ).posterior
    pr = m._hurdle_flow_priors()
    fp, sp_ = m._count_fe_priors(), m._sel_fe_priors()
    fe_c, fe_b = m._count_fe_spec(), m._sel_fe_spec()
    nt = fe_c.n_tau
    priors = HurdlePriors(
        gamma_mu=np.concatenate([pr.gamma_mu, np.full(nt, sp_["time_effect_mu"])]),
        gamma_sigma=np.concatenate([pr.gamma_sigma, np.full(nt, sp_["time_effect_sigma"])]),
        theta_mu=np.concatenate([pr.beta_mu, np.full(nt, fp["time_effect_mu"])]),
        theta_sigma=np.concatenate([pr.beta_sigma, np.full(nt, fp["time_effect_sigma"])]),
        alpha_sigma=pr.alpha_sigma, alpha_nu=pr.alpha_nu,
    )  # fmt: skip
    filt = lambda: FlowSeparableFilter(m._W_sparse, 6, -0.999, 0.999)  # noqa: E731
    outs = [
        run_chain(
            m._y,
            m._X,
            m._Z,
            filt(),
            filt(),
            fe_c,
            fe_b,
            priors,
            4000,
            1000,
            rng=np.random.default_rng(10 + c),
        )  # fmt: skip
        for c in range(4)
    ]
    for var, key in (("rho_d", "rho_d"), ("lam_d", "lam_d"), ("alpha", "alpha")):
        x, r = s[var].values.ravel(), np.concatenate([o[key] for o in outs])
        assert abs(x.mean() - r.mean()) < 0.2 * r.std(), var
        np.testing.assert_allclose(x.std(), r.std(), rtol=0.15, err_msg=var)
    for var in ("beta", "gamma"):
        x = s[var].values.reshape(-1, s[var].shape[-1])
        r = np.concatenate([o[var] for o in outs])
        assert np.all(np.abs(x.mean(0) - r.mean(0)) < 0.2 * r.std(0)), var
        np.testing.assert_allclose(x.std(0), r.std(0), rtol=0.15, err_msg=var)


@pytest.mark.slow
def test_flow_hurdle_matches_nuts():
    from neighbayes.models import SARHurdleNBFlowSeparable

    d = _flow(n=7, seed=3)
    m = SARHurdleNBFlowSeparable(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
    g = m.fit(draws=4000, tune=1000, chains=4, progressbar=False, random_seed=1)
    n = m.fit(
        sampler="nuts", draws=2000, tune=1500, chains=4, progressbar=False,
        random_seed=2, target_accept=0.95,
    )  # fmt: skip
    for var in ("rho_d", "rho_o", "lam_d", "lam_o", "beta", "gamma"):
        _agree(g.posterior, n.posterior, var)
    # α is heavy-tailed on a few small positive counts; compare it on the log scale.
    la, lr = np.log(g.posterior["alpha"].values), np.log(n.posterior["alpha"].values)
    assert abs(la.mean() - lr.mean()) < 0.2 * lr.std()
    np.testing.assert_allclose(la.std(), lr.std(), rtol=0.15)


@pytest.mark.slow
@pytest.mark.requires_jax
def test_flow_hurdle_jax_matches_numpy():
    from neighbayes.models import SARHurdleNBFlowSeparablePanel

    d = _flow_panel_data(n=6, T=4, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=4, col_names=d["col_names"], effects=3
        )
    fit = dict(draws=3000, tune=1000, chains=4, progressbar=False)
    a = m.fit(gibbs_backend="numpy", random_seed=1, n_jobs=4, **fit).posterior
    b = m.fit(gibbs_backend="jax", random_seed=2, **fit).posterior
    for var in ("rho_d", "rho_o", "lam_d", "lam_o", "beta", "gamma", "alpha",
                "time_effect", "sel_time_effect", "group_sd", "sel_group_sd"):  # fmt: skip
        _agree_mc(b, a, var)


@pytest.mark.slow
@pytest.mark.requires_jax
def test_flow_hurdle_panel_recovers_sparse_flows():
    """Sparse flows with pair effects in both halves: the hurdle recovers both.

    About three positive weeks per pair.  Under a fixed, wide prior on the pair
    effects this gave α ≈ 1.5 against a true 3 and ρ several sds off (an
    incidental-parameter effect, with ρ drifting toward 1 on some seeds); the
    pooled effects with a learned sd recover them.
    """
    import arviz as az

    from neighbayes.models import SARHurdleNBFlowSeparablePanel

    n, T = 20, 8
    d = _flow(
        T=T, n=n, seed=8, lam_d=0.3, lam_o=0.2, target_pi=0.4, pair_effect_sd=1.0,
        time_effect_sd=0.3, sel_pair_effect_sd=1.0, sel_time_effect_sd=0.3,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARHurdleNBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=T, col_names=d["col_names"], effects=3
        )
    post = m.fit(
        draws=3000, tune=1000, chains=4, progressbar=False, random_seed=1,
        gibbs_backend="jax",
    ).posterior  # fmt: skip
    truth = {"lam_d": 0.3, "lam_o": 0.2, "rho_d": 0.3, "rho_o": 0.2, "alpha": 3.0}
    for var, value in truth.items():
        x = post[var].values
        assert abs(x.mean() - value) < 3.0 * x.std(), var
        assert float(az.rhat(x)) < 1.05, var
    keep = m._sel_keep if m._sel_keep is not None else np.arange(len(d["gamma"]))
    g = post["gamma"].values
    for j, value in enumerate(np.asarray(d["gamma"])[keep]):
        assert abs(g[..., j].mean() - value) < 3.0 * g[..., j].std(), j
        assert float(az.rhat(g[..., j])) < 1.05, j
