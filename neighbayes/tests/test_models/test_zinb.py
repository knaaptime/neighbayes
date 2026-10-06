"""Zero-inflated NB models: SARZINB, the ZINB panels and the flow ZINB.

Fast tests pin construction, the bugs fixed on 2026-10-03 (prior keys rejected,
the DGP's selection noise, the JAX α support), ``alpha_fixed`` on every sampler,
and the API of every class on short chains.  The ``slow`` tests check the Gibbs
samplers against NUTS and against each other.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from neighbayes import dgp
from neighbayes.models import (
    SARZINB,
    SARZINBFlowSeparable,
    SARZINBFlowSeparablePanel,
    SARZINBPanel,
    ZINBPanel,
)

QUICK = dict(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
A = 23.0


def _xs(n_side=7, seed=0):
    return dgp.simulate_sar_zinb(n_side=n_side, seed=seed)


def _panel(seed=1):
    return dgp.simulate_panel_sar_zinb(
        N=25, T=3, n_side=5, seed=seed, unit_effect_sd=0.5, time_effect_sd=0.3
    )


def _flow(T=1, seed=2, **kw):
    return dgp.generate_zinb_flow_data_separable(n=4, T=T, seed=seed, **kw)


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
# Bugs fixed 2026-10-03
# ---------------------------------------------------------------------------


class TestFixes:
    def test_documented_prior_keys_are_accepted(self):
        d = _xs()
        priors = {
            "gamma_mu": 0.0,
            "gamma_sigma": 2.0,
            "lam_lower": -0.5,
            "lam_upper": 0.9,
            "rho_lower": -0.5,
            "alpha_sigma": 1.5,
            "alpha_nu": 4.0,
            "alpha_fixed": A,
        }
        m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"], priors=priors)
        pv = m._zinb_priors()
        assert pv.lam_lower == -0.5 and pv.alpha_fixed == A
        assert np.allclose(pv.gamma_sigma, 2.0)

    def test_zinb_dgp_selection_is_reduced_form(self):
        from neighbayes.dgp.utils import spatial_filter_factor

        d = dgp.simulate_sar_zinb(n_side=6, seed=3, lam=0.3)
        expected = spatial_filter_factor(d["W_sparse"], 0.3)(
            d["Z"] @ d["params_true"]["gamma"]
        )
        np.testing.assert_allclose(d["eta_sel"], expected, atol=1e-10)

    def test_logit_dgp_reduced_form_option(self):
        from neighbayes.dgp.utils import spatial_filter_factor

        d = dgp.simulate_sar_logit(n_side=6, seed=4, rho=0.3, structural=False)
        expected = spatial_filter_factor(d["W_sparse"], 0.3)(
            d["X"] @ d["params_true"]["beta"]
        )
        np.testing.assert_allclose(d["eta_true"], expected, atol=1e-10)

    def test_target_pi_reports_the_generating_gamma(self):
        """``target_pi`` once shifted η by a constant but γ₀ inside the filter."""
        from neighbayes.dgp.flows import _graph_to_csr
        from neighbayes.dgp.utils import spatial_filter_factor

        d = dgp.simulate_sar_zinb(n_side=6, seed=3, lam=0.5, target_pi=0.4)
        expected = spatial_filter_factor(d["W_sel_sparse"], 0.5)(
            d["Z"] @ d["params_true"]["gamma"]
        )
        np.testing.assert_allclose(d["eta_sel"], expected, atol=1e-10)

        f = dgp.generate_zinb_flow_data_separable(
            n=6, T=2, lam_d=0.4, lam_o=0.3, target_pi=0.6, seed=3
        )
        W = _graph_to_csr(f["G"]).toarray()
        eye = np.eye(W.shape[0])
        A = np.kron(eye - 0.3 * W, eye - 0.4 * W)
        Xt = f["X"].reshape(2, A.shape[0], -1)
        expected = np.concatenate([np.linalg.solve(A, x @ f["gamma"]) for x in Xt])
        np.testing.assert_allclose(f["eta_sel"], expected, atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.requires_jax
    def test_jax_alpha_support_matches_numpy(self):
        """The JAX α slice once stopped at e⁴ ≈ 55; near-Poisson data needs more."""
        rng = np.random.default_rng(5)
        d = dgp.simulate_sar_zinb(n_side=15, seed=5, alpha=400.0, rng=rng)
        m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
        idata = m.fit(
            draws=600, tune=600, chains=2, progressbar=False, gibbs_backend="jax"
        )
        assert float(idata.posterior["alpha"].max()) > 60.0


# ---------------------------------------------------------------------------
# SARZINB
# ---------------------------------------------------------------------------


class TestSARZINB:
    @pytest.mark.parametrize(
        "kw",
        [
            dict(gibbs_backend="numpy", n_jobs=1),
            pytest.param(dict(gibbs_backend="jax"), marks=pytest.mark.requires_jax),
            dict(sampler="nuts"),
        ],
    )
    def test_alpha_fixed_on_every_sampler(self, kw):
        d = _xs()
        m = SARZINB(
            y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"], priors={"alpha_fixed": A}
        )
        assert np.allclose(m.fit(**QUICK, **kw).posterior["alpha"].values, A)

    def test_nuts_model(self):
        d = _xs()
        m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
        model = m._build_pymc_model()
        assert {"lam", "gamma", "rho", "beta", "alpha", "obs"} <= set(model.named_vars)

    def test_selection_names_and_formula(self):
        d = _xs()
        Zdf = pd.DataFrame(d["Z"], columns=["Intercept", "zq"])
        m = SARZINB(y=d["y"], X=d["X"], Z=Zdf, W=d["W_graph"])
        assert m._sel_feature_names == ["Intercept", "zq"]
        frame = pd.DataFrame({"y": d["y"], "x": d["X"][:, 1], "zq": d["Z"][:, 1]})
        m2 = SARZINB("y ~ x", data=frame, W=d["W_graph"], sel_formula="~ zq")
        assert m2._sel_feature_names == ["Intercept", "zq"]
        np.testing.assert_allclose(m2._Z, d["Z"])
        with pytest.raises(ValueError, match="not both"):
            SARZINB("y ~ x", data=frame, W=d["W_graph"], Z=d["Z"], sel_formula="~ zq")

    def test_posterior_summaries(self):
        d = _xs()
        m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
        m.fit(gibbs_backend="numpy", n_jobs=1, **QUICK)
        n = d["y"].size
        pp = m.posterior_predictive(n_draws=4, random_seed=0)
        assert pp.shape == (4, n) and np.all(pp >= 0)
        pi = m.corridor_probabilities(draws=5)
        assert pi.shape == (n,) and np.all((pi > 0) & (pi < 1))
        za = m.zero_attribution(draws=5)
        n0 = int(np.sum(d["y"] == 0))
        assert za["structural_prob"].shape == (n0,)
        np.testing.assert_allclose(za["structural_prob"] + za["sampling_prob"], 1.0)
        assert m.spatial_effects(equation="selection").shape[0] == 1


# ---------------------------------------------------------------------------
# ZINB panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SARZINBPanel, ZINBPanel])
@pytest.mark.parametrize("effects", [0, 1, 2, 3])
@pytest.mark.parametrize("sampler", ["gibbs", "nuts"])
def test_zinb_panel_api(cls, effects, sampler):
    d = _panel()
    m = cls(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=25, T=3, effects=effects)
    kw = dict(QUICK, sampler=sampler)
    if sampler == "gibbs":
        kw["n_jobs"] = 1
    post = m.fit(**kw).posterior
    assert ("group_effect" in post) == (effects in (1, 3))
    assert ("time_effect" in post) == (effects in (2, 3))
    assert ("lam" in post) == (cls is SARZINBPanel)
    assert post["gamma"].shape[-1] == 2
    assert m.posterior_predictive(n_draws=2, random_seed=0).shape == (2, 75)
    assert m.spatial_effects().shape[0] == 1
    assert m.spatial_effects(equation="selection").shape[0] == 1


def test_zinb_panel_registry_and_storage():
    from neighbayes.samplers._registry import resolve

    assert resolve("zinb", "panel") is not None
    d = _panel()
    m = SARZINBPanel(
        y=d["y"],
        X=d["X"],
        Z=d["Z"],
        W=d["W_sparse"],
        N=25,
        T=3,
        effects=1,
        priors={"alpha_fixed": A},
    )
    idata = m.fit(store_group_effects=False, thin=2, n_jobs=1, **QUICK)
    assert idata.posterior["beta"].shape[1] == QUICK["draws"] // 2
    assert idata["group_effect_summary"]["mean"].shape == (25,)
    assert np.allclose(idata.posterior["alpha"].values, A)
    assert np.all(np.isfinite(m.zero_attribution(draws=3)["structural_prob"]))


# ---------------------------------------------------------------------------
# Flow ZINB
# ---------------------------------------------------------------------------


def _flow_model(cls, d, **kw):
    if cls is SARZINBFlowSeparable:
        return cls(d["y_vec"], d["X"], d["G"], col_names=d["col_names"], **kw)
    return cls(d["y"], d["X"], d["G"], T=3, col_names=d["col_names"], **kw)


@pytest.mark.parametrize(
    "backend",
    ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax), "nuts"],
)
def test_flow_zinb_cross_section(backend):
    d = _flow()
    m = _flow_model(SARZINBFlowSeparable, d, priors={"alpha_fixed": A})
    kw = (
        dict(QUICK, sampler="nuts")
        if backend == "nuts"
        else dict(QUICK, gibbs_backend=backend, n_jobs=1)
    )
    post = m.fit(**kw).posterior
    assert {"lam_d", "lam_o", "lam_w", "rho_d", "rho_o", "gamma"} <= set(post.data_vars)
    assert np.allclose(post["alpha"].values, A)
    assert m.posterior_predictive(n_draws=2, random_seed=0).shape == (2, 16)
    assert m.spatial_effects().shape[0] > 0
    assert m.spatial_effects(equation="selection").shape[0] > 0


@pytest.mark.parametrize("effects", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_flow_zinb_panel(effects, backend):
    d = _flow(T=3, pair_effect_sd=0.4, time_effect_sd=0.3)
    X = np.column_stack([d["X"], np.random.default_rng(0).normal(size=d["X"].shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARZINBFlowSeparablePanel(
            d["y"],
            X,
            d["G"],
            T=3,
            col_names=list(d["col_names"]) + ["z"],
            effects=effects,
        )
        post = m.fit(gibbs_backend=backend, n_jobs=1, **QUICK).posterior
    assert ("group_effect" in post) == (effects in (1, 3))
    assert ("group_sd" in post) == (effects in (1, 3))
    assert ("time_effect" in post) == (effects in (2, 3))
    assert m.zero_attribution(draws=3)["structural_prob"].size == int(
        np.sum(d["y"] == 0)
    )


def test_flow_zinb_guards():
    d = _flow()
    with pytest.raises(ValueError, match="cross-section"):
        _flow_model(SARZINBFlowSeparable, d, effects=1)
    Z = d["X"][:, :3]
    m = _flow_model(SARZINBFlowSeparable, d, Z=Z)
    m.fit(n_jobs=1, **QUICK)
    with pytest.raises(NotImplementedError, match="Z=None"):
        m.spatial_effects(equation="selection")


# ---------------------------------------------------------------------------
# Slow: exactness
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize("alpha_fixed", [None, 15.0])
def test_sarzinb_gibbs_matches_nuts(backend, alpha_fixed):
    d = dgp.simulate_sar_zinb(n_side=14, seed=3, rho=0.4, lam=0.3)
    priors = None if alpha_fixed is None else {"alpha_fixed": alpha_fixed}
    m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"], priors=priors)
    g = m.fit(
        draws=3000, tune=1000, chains=4, progressbar=False, random_seed=1,
        gibbs_backend=backend, n_jobs=4,
    ).posterior  # fmt: skip
    n = m.fit(
        sampler="nuts", draws=1500, tune=1500, chains=4, progressbar=False,
        random_seed=2,
    ).posterior  # fmt: skip
    for var in ("rho", "lam", "beta", "gamma", "alpha"):
        if var == "alpha" and alpha_fixed is not None:
            continue
        _agree(g, n, var)


@pytest.mark.slow
def test_generic_kernel_matches_sarzinb():
    from neighbayes.samplers.count_panel import CountPanelFE
    from neighbayes.samplers.count_panel._filters import SARFilter
    from neighbayes.samplers.zinb._generic import ZINBPriors, run_chain

    d = dgp.simulate_sar_zinb(n_side=14, seed=3, rho=0.4, lam=0.3)
    m = SARZINB(y=d["y"], X=d["X"], Z=d["Z"], W=d["W_graph"])
    ref = m.fit(
        draws=3000, tune=1000, chains=4, progressbar=False, random_seed=1, n_jobs=4
    ).posterior
    pv = m._zinb_priors()
    pri = ZINBPriors(
        gamma_mu=pv.gamma_mu,
        gamma_sigma=pv.gamma_sigma,
        theta_mu=pv.beta_mu,
        theta_sigma=pv.beta_sigma,
    )
    W = m._W_sparse
    outs = [
        run_chain(
            m._y,
            m._X,
            m._Z,
            SARFilter(W, 1, pv.rho_lower, pv.rho_upper),
            SARFilter(W, 1, pv.lam_lower, pv.lam_upper),
            CountPanelFE(),
            pri,
            3000,
            1000,
            rng=np.random.default_rng(s),
        )  # fmt: skip
        for s in range(4)
    ]
    for var in ("rho", "lam", "beta", "gamma", "alpha"):
        x = np.concatenate([o[var] for o in outs])
        r = ref[var].values.reshape(-1, *ref[var].shape[2:])
        assert np.all(np.abs(x.mean(0) - r.mean(0)) < 0.2 * r.std(0)), var


@pytest.mark.slow
def test_zinb_panel_gibbs_matches_nuts():
    d = dgp.simulate_panel_sar_zinb(
        N=49, T=4, n_side=7, seed=2, unit_effect_sd=0.6, time_effect_sd=0.3,
        fe_x_corr=0.5,
    )  # fmt: skip
    m = SARZINBPanel(
        y=d["y"], X=d["X"], Z=d["Z"], W=d["W_sparse"], N=49, T=4, effects=3
    )
    g = m.fit(
        draws=4000, tune=1000, chains=4, progressbar=False, random_seed=1, n_jobs=4
    ).posterior
    n = m.fit(
        sampler="nuts", draws=1500, tune=1500, chains=4, progressbar=False,
        random_seed=2, target_accept=0.95,
    ).posterior  # fmt: skip
    for var in ("rho", "lam", "beta", "gamma", "time_effect", "alpha"):
        _agree(g, n, var)


@pytest.mark.slow
@pytest.mark.requires_jax
def test_flow_zinb_panel_pair_effects_identify_selection():
    """With pair effects a pair's other weeks pin its count distribution.

    On a sparse flow cross-section the selection equation is not identified (NB
    with a smaller α explains the zeros; λ's posterior is its prior, and NUTS
    diverges).  The panel with pair effects recovers it, and the chain mixes.
    """
    import arviz as az

    n, T = 20, 8
    d = dgp.generate_zinb_flow_data_separable(
        n=n, T=T, seed=8, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4,
        alpha=3.0, lam_d=0.3, lam_o=0.2, target_pi=0.7, pair_effect_sd=1.5,
        time_effect_sd=0.3,
    )  # fmt: skip
    names = d["col_names"]
    idx = [names.index(c) for c in ("intercept", "dest_x0", "orig_x0", "log_distance")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = SARZINBFlowSeparablePanel(
            d["y"], d["X"], d["G"], T=T, col_names=names, Z=d["X"][:, idx], effects=3
        )
    post = m.fit(
        draws=3000, tune=1000, chains=4, progressbar=False, random_seed=1,
        gibbs_backend="jax",
    ).posterior  # fmt: skip
    truth = {"lam_d": 0.3, "lam_o": 0.2, "alpha": 3.0}
    for var, value in truth.items():
        x = post[var].values
        assert abs(x.mean() - value) < 2.5 * x.std(), var
        assert float(az.rhat(x)) < 1.05, var
    g = post["gamma"].values
    for j, value in enumerate(d["gamma"][idx]):
        assert abs(g[..., j].mean() - value) < 2.5 * g[..., j].std(), j
        assert float(az.rhat(g[..., j])) < 1.05, j


@pytest.mark.slow
@pytest.mark.requires_jax
def test_flow_zinb_jax_matches_numpy():
    # Data on which the selection half is identified and mixes: on a tiny,
    # sparse panel λ's ESS is ~1% of draws and the comparison is only noise.
    d = dgp.generate_zinb_flow_data_separable(
        n=8, T=6, seed=5, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4,
        alpha=3.0, pair_effect_sd=0.5, time_effect_sd=0.3, target_pi=0.7,
    )  # fmt: skip
    m = SARZINBFlowSeparablePanel(
        d["y"], d["X"], d["G"], T=6, col_names=d["col_names"], effects=3
    )
    fit = dict(draws=3000, tune=1000, chains=4, progressbar=False)
    a = m.fit(gibbs_backend="numpy", random_seed=1, n_jobs=4, **fit).posterior
    b = m.fit(gibbs_backend="jax", random_seed=2, **fit).posterior
    # Against Monte Carlo error: the selection half still mixes slowest.
    for var in ("rho_d", "rho_o", "beta", "time_effect", "alpha", "group_sd",
                "lam_d", "lam_o", "gamma"):  # fmt: skip
        _agree_mc(b, a, var)
