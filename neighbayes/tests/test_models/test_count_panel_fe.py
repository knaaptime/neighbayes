"""NB count panels with pooled unit/pair effects and period effects.

Fast tests cover construction, the absorbed-column and identification rules,
storage and the API on short chains.  The ``slow`` tests check recovery against
DGPs whose effects correlate with ``X`` (so a pooled fit is visibly biased),
agreement of the structured and materialized separable flow kernels, and of
Gibbs with NUTS.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neighbayes.dgp.flows import (
    generate_panel_negbin_flow_data,
    generate_panel_negbin_flow_data_separable,
)
from neighbayes.dgp.panel_count import simulate_panel_sar_negbin
from neighbayes.models import (
    NegBinFlowPanel,
    NegBinPanel,
    SARNegBinFlowPanel,
    SARNegBinFlowSeparablePanel,
    SARNegBinPanel,
)

FLOW_CLASSES = [SARNegBinFlowPanel, SARNegBinFlowSeparablePanel, NegBinFlowPanel]
PANEL_CLASSES = [SARNegBinPanel, NegBinPanel]
QUICK = dict(draws=20, tune=20, chains=1, progressbar=False, random_seed=0, n_jobs=1)


def _flow(n=4, T=3, seed=0, **kw):
    return generate_panel_negbin_flow_data(n=n, T=T, seed=seed, **kw)


def _flow_model(cls, data, T, **kw):
    return cls(
        y=data["y"], X=data["X"], W=data["G"], T=T, col_names=data["col_names"], **kw
    )


def _place(cls, data, N, T, **kw):
    import pandas as pd

    X = pd.DataFrame(data["X"], columns=["Intercept", "x"])
    return cls(y=data["y"], X=X, W=data["W_sparse"], N=N, T=T, **kw)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("cls", FLOW_CLASSES)
    def test_pair_effects_keep_time_invariant_columns(self, cls):
        """Pooled pair effects absorb nothing; the intercept anchors the level."""
        data = _flow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _flow_model(cls, data, 3, effects=1)
        assert list(m._feature_names) == list(data["col_names"])
        assert m._row_group.shape == (data["y"].size,)
        assert m._D_tau is None
        fp = m._count_fe_priors()
        assert fp["group_effect_mu"] == 0.0 and fp["group_effect_sd_scale"] == 1.0

    def test_period_effects_absorb_time_only_columns(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        trend = np.repeat(np.arange(3.0), 16)
        X = np.column_stack([d["X"], trend])
        with pytest.warns(UserWarning, match="absorbed by the period effects"):
            two_way = SARNegBinPanel(
                y=d["y"], X=X, W=d["W_sparse"], N=16, T=3, effects=3
            )
        # Beside unit effects the intercept stays as the baseline level.
        assert two_way._X.shape[1] == 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time_only = SARNegBinPanel(
                y=d["y"], X=X, W=d["W_sparse"], N=16, T=3, effects=2
            )
        assert time_only._X.shape[1] == 1  # the intercept goes too

    @pytest.mark.parametrize("cls", FLOW_CLASSES)
    def test_period_effects_layout(self, cls):
        data = _flow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            two_way = _flow_model(cls, data, 3, effects=3)
            time_only = _flow_model(cls, data, 3, effects=2)
        # Beside pair effects the first period is the baseline; without them
        # every period gets an effect (the intercept is absorbed).
        assert two_way._D_tau.shape == (data["y"].size, 2)
        assert time_only._D_tau.shape == (data["y"].size, 3)
        assert two_way._period_labels == ["t1", "t2"]
        assert time_only._period_labels == ["t0", "t1", "t2"]

    def test_effects_names_and_ints_agree(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        a = _place(SARNegBinPanel, d, 16, 3, effects="two_way")
        b = _place(SARNegBinPanel, d, 16, 3, effects=3)
        assert a.model == b.model == 3
        assert a.effects == "two_way"

    def test_time_invariant_design_identifies_rho_under_unit_effects(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        X = np.column_stack([np.ones(48), np.tile(np.arange(16.0), 3)])
        m = SARNegBinPanel(y=d["y"], X=X, W=d["W_sparse"], N=16, T=3, effects=1)
        assert m._X.shape[1] == 2

    def test_spatial_model_with_nothing_left_raises(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        with pytest.raises(ValueError, match="not identified"):
            SARNegBinPanel(
                y=d["y"], X=np.ones((48, 1)), W=d["W_sparse"], N=16, T=3, effects=2
            )

    def test_aspatial_model_without_slopes_is_allowed(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        m = NegBinPanel(
            y=d["y"], X=np.ones((48, 1)), W=d["W_sparse"], N=16, T=3, effects=2
        )
        assert m._X.shape[1] == 0

    def test_mundlak_adds_unit_means(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        m = _place(SARNegBinPanel, d, 16, 3, effects=1, mundlak=True)
        assert m._feature_names == ["Intercept", "x", "x_mean"]
        np.testing.assert_allclose(
            m._X[:16, 2], d["X"][:, 1].reshape(3, 16).mean(axis=0)
        )
        assert m._nonintercept_indices == [1]
        with pytest.raises(ValueError, match="mundlak"):
            _place(SARNegBinPanel, d, 16, 3, effects=2, mundlak=True)

    def test_all_zero_groups_warn(self):
        d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=0)
        y = d["y"].copy()
        y[np.tile(np.arange(16), 3) == 0] = 0
        y[np.tile(np.arange(16), 3) == 5] = 0
        X = np.column_stack([np.ones(48), d["X"][:, 1]])
        with pytest.warns(UserWarning, match="2 of 16 groups"):
            NegBinPanel(y=y, X=X, W=d["W_sparse"], N=16, T=3, effects=1)

    def test_unit_effects_need_two_periods(self):
        d = simulate_panel_sar_negbin(N=16, T=1, n_side=4, seed=0)
        with pytest.raises(ValueError, match="T >= 2"):
            _place(NegBinPanel, d, 16, 1, effects=1)

    def test_unrestricted_fixed_effects_require_numpy_backend(self):
        data = _flow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _flow_model(SARNegBinFlowPanel, data, 3, effects=1)
        with pytest.raises(NotImplementedError, match="NumPy"):
            m.fit(gibbs_backend="jax", **QUICK)


# ---------------------------------------------------------------------------
# Short fits: every class, every effects mode, both samplers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", FLOW_CLASSES)
@pytest.mark.parametrize("effects", [0, 1, 2, 3])
def test_flow_fit_api(cls, effects):
    data = _flow()
    X = np.column_stack(
        [data["X"], np.random.default_rng(0).normal(size=data["X"].shape[0])]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = cls(
            y=data["y"],
            X=X,
            W=data["G"],
            T=3,
            col_names=list(data["col_names"]) + ["z"],
            effects=effects,
        )
        idata = m.fit(sampler="gibbs", **QUICK)
    post = idata.posterior
    assert ("group_effect" in post) == (effects in (1, 3))
    assert ("group_sd" in post) == (effects in (1, 3))
    assert ("time_effect" in post) == (effects in (2, 3))
    if effects in (1, 3):
        assert post["group_effect"].shape[-1] == 16
    assert np.all(np.isfinite(post["beta"].values))
    pp = m.posterior_predictive(n_draws=2, random_seed=0)
    assert pp.shape == (2, data["y"].size)
    assert m.spatial_effects().shape[0] > 0


@pytest.mark.parametrize("cls", PANEL_CLASSES)
@pytest.mark.parametrize("effects", [0, 1, 2, 3])
def test_panel_fit_api(cls, effects):
    d = simulate_panel_sar_negbin(N=25, T=3, n_side=5, seed=1)
    m = _place(cls, d, 25, 3, effects=effects)
    idata = m.fit(**QUICK)  # Gibbs, the default, as for the Gaussian panels
    assert m._pymc_model is None
    post = idata.posterior
    assert ("group_effect" in post) == (effects in (1, 3))
    assert ("group_sd" in post) == (effects in (1, 3))
    assert ("time_effect" in post) == (effects in (2, 3))
    assert ("rho" in post) == (cls is SARNegBinPanel)
    assert "alpha" in post
    assert m.posterior_predictive(n_draws=2, random_seed=0).shape == (2, 75)
    assert m.spatial_effects().shape[0] == 1
    assert np.all(np.isfinite(m._fitted_mean_from_posterior()))


@pytest.mark.parametrize("cls", PANEL_CLASSES)
def test_panel_nuts_builds_fixed_effects(cls):
    d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=2)
    m = _place(cls, d, 16, 3, effects=3)
    model = m._build_pymc_model()
    assert "group_effect" in model.named_vars
    assert "time_effect" in model.named_vars
    assert model.named_vars["group_effect"].shape.eval()[0] == 16


def test_nuts_on_request():
    d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=3)
    m = _place(NegBinPanel, d, 16, 3, effects=1)
    idata = m.fit(
        sampler="nuts", draws=20, tune=20, chains=1, progressbar=False, random_seed=0
    )
    assert m._pymc_model is not None
    assert {"group_effect", "group_sd"} <= set(idata.posterior.data_vars)


def test_pooled_aspatial_gibbs_runs_on_request():
    d = simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=5)
    m = _place(NegBinPanel, d, 16, 3, effects=0)
    idata = m.fit(sampler="gibbs", **QUICK)
    assert m._pymc_model is None
    assert {"alpha", "beta"} <= set(idata.posterior.data_vars)


def test_gibbs_runs_through_the_registry():
    from neighbayes.samplers._registry import resolve

    entry = resolve("count", "panel")
    assert entry is not None and entry.backends == {"numpy"}
    d = simulate_panel_sar_negbin(N=16, T=4, n_side=4, seed=6)
    m = _place(SARNegBinPanel, d, 16, 4, effects=1)
    idata = m.fit(thin=2, **QUICK)
    assert idata.posterior["beta"].shape[1] == QUICK["draws"] // 2
    with pytest.raises(TypeError, match="unsupported keyword"):
        m.fit(not_an_option=1, **QUICK)


def test_group_effect_summary_storage():
    d = simulate_panel_sar_negbin(N=25, T=3, n_side=5, seed=4)
    m = _place(SARNegBinPanel, d, 25, 3, effects=1)
    idata = m.fit(store_group_effects=False, **QUICK)
    assert "group_effect" not in idata.posterior
    summary = idata["group_effect_summary"]
    assert summary["mean"].shape == (25,)
    assert np.all(summary["sd"].values >= 0)
    with pytest.raises(RuntimeError, match="store_group_effects=True"):
        m.posterior_predictive(n_draws=2)
    assert np.all(np.isfinite(m._fitted_mean_from_posterior()))


@pytest.mark.requires_jax
@pytest.mark.parametrize("effects", [1, 2, 3])
@pytest.mark.parametrize("store", [True, False])
def test_separable_fixed_effects_jax_backend(effects, store):
    data = _flow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _flow_model(SARNegBinFlowSeparablePanel, data, 3, effects=effects)
        idata = m.fit(
            gibbs_backend="jax",
            store_group_effects=store,
            idata_kwargs={"log_likelihood": True},
            draws=20,
            tune=20,
            chains=2,
            progressbar=False,
            random_seed=0,
        )
    post = idata.posterior
    assert ("time_effect" in post) == (effects in (2, 3))
    assert ("group_sd" in post) == (effects in (1, 3))
    if effects in (1, 3):
        if store:
            assert post["group_effect"].shape == (2, 20, 16)
        else:
            assert idata["group_effect_summary"]["sd"].shape == (16,)
    assert np.all(np.isfinite(idata.log_likelihood["obs"].values))


def test_log_likelihood_storage():
    data = _flow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _flow_model(SARNegBinFlowSeparablePanel, data, 3, effects=3)
        idata = m.fit(idata_kwargs={"log_likelihood": True}, **QUICK)
    assert idata.log_likelihood["obs"].shape[-1] == data["y"].size
    assert np.all(np.isfinite(idata.log_likelihood["obs"].values))


# ---------------------------------------------------------------------------
# Slow: recovery and cross-checks
# ---------------------------------------------------------------------------


def _covers(draws, truth, k=3.0):
    draws = np.asarray(draws).reshape(-1, *np.shape(truth))
    return np.all(np.abs(draws.mean(0) - truth) < k * draws.std(0))


@pytest.mark.slow
@pytest.mark.recovery
def test_panel_two_way_recovery_beats_pooled():
    N, T, rho, beta = 225, 6, 0.4, np.array([0.5, 0.6])
    d = simulate_panel_sar_negbin(
        N=N, T=T, n_side=15, rho=rho, beta=beta, alpha=3.0, seed=11,
        unit_effect_sd=0.8, time_effect_sd=0.3, fe_x_corr=0.7,
    )  # fmt: skip
    fit = dict(draws=600, tune=400, chains=2, progressbar=False, random_seed=3)
    # The unit effects correlate with x, so the pooled-effects model needs the
    # Mundlak means to keep the slope unbiased.
    model = _place(SARNegBinPanel, d, N, T, effects=3, mundlak=True)
    fe = model.fit(**fit)
    pooled = _place(SARNegBinPanel, d, N, T, effects=0).fit(**fit)
    slope_fe = fe.posterior["beta"].values[..., model._feature_names.index("x")]
    slope_pooled = pooled.posterior["beta"].values[..., 1]
    assert _covers(fe.posterior["rho"].values, rho)
    assert _covers(slope_fe, beta[1])
    # The unit effects load on X's persistent part, so pooling is biased.
    assert abs(slope_pooled.mean() - beta[1]) > 4 * slope_pooled.std()


@pytest.mark.slow
@pytest.mark.recovery
def test_flow_separable_pair_effects_recovery():
    rho_d, rho_o = 0.4, 0.3
    data = generate_panel_negbin_flow_data_separable(
        n=8, T=6, rho_d=rho_d, rho_o=rho_o, seed=21, alpha=3.0,
        beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4,
        pair_effect_sd=0.7, time_effect_sd=0.3, fe_x_corr=0.6,
    )  # fmt: skip
    m = _flow_model(SARNegBinFlowSeparablePanel, data, 6, effects=3)
    idata = m.fit(draws=600, tune=400, chains=2, progressbar=False, random_seed=5)
    post = idata.posterior
    assert _covers(post["rho_d"].values, rho_d)
    assert _covers(post["rho_o"].values, rho_o)
    names = list(m._feature_names)
    for side, truth in (("dest_x0", 0.4), ("orig_x0", 0.4)):
        assert _covers(post["beta"].values[..., names.index(side)], truth)


@pytest.mark.slow
def test_structured_and_materialized_separable_kernels_agree():
    """The NB separable panel's structured FE sweep vs the generic kernel."""
    from neighbayes.samplers.count_panel._filters import FlowSeparableFilter

    data = generate_panel_negbin_flow_data_separable(
        n=6, T=4, seed=31, pair_effect_sd=0.5, time_effect_sd=0.3, fe_x_corr=0.5
    )
    m = _flow_model(SARNegBinFlowSeparablePanel, data, 4, effects=3)
    fit = dict(draws=1500, tune=500, chains=2, progressbar=False, n_jobs=1)
    structured = m.fit(random_seed=1, attach_log_abs_det=False, **fit)
    pv = m._flow_count_priors()
    generic = m._run_count_panel_gibbs(
        FlowSeparableFilter(m._W_sparse, 4, -0.999, 0.999),
        beta_mu=pv["beta_mu"],
        beta_sigma=pv["beta_sigma"],
        draws=1500,
        tune=500,
        chains=2,
        random_seed=2,
        progressbar=False,
        n_jobs=1,
        log_likelihood=False,
        store_group_effects=True,
        model_type="check",
        separable_flow=True,
    )
    for var in ("rho_d", "rho_o", "beta", "time_effect", "alpha"):
        a = structured.posterior[var].values
        b = generic.posterior[var].values
        sd = a.reshape(-1, *a.shape[2:]).std(0)
        diff = np.abs(a.mean((0, 1)) - b.mean((0, 1)))
        assert np.all(diff < 0.25 * sd), (var, diff, sd)


@pytest.mark.slow
@pytest.mark.requires_jax
def test_separable_fixed_effects_jax_matches_numpy():
    data = generate_panel_negbin_flow_data_separable(
        n=6, T=4, seed=31, pair_effect_sd=0.5, time_effect_sd=0.3, fe_x_corr=0.5
    )
    m = _flow_model(SARNegBinFlowSeparablePanel, data, 4, effects=3)
    fit = dict(
        draws=2000, tune=600, chains=2, progressbar=False, attach_log_abs_det=False
    )
    a = m.fit(gibbs_backend="numpy", random_seed=1, n_jobs=1, **fit)
    b = m.fit(gibbs_backend="jax", random_seed=2, **fit)
    for var in ("rho_d", "rho_o", "beta", "time_effect", "alpha"):
        x, y = a.posterior[var].values, b.posterior[var].values
        sd = x.reshape(-1, *x.shape[2:]).std(0)
        assert np.all(np.abs(x.mean((0, 1)) - y.mean((0, 1))) < 0.25 * sd), var


@pytest.mark.slow
@pytest.mark.parametrize("alpha_fixed", [None, 30.0])
def test_unit_effects_gibbs_matches_nuts(alpha_fixed):
    d = simulate_panel_sar_negbin(
        N=64, T=4, n_side=8, rho=0.0, alpha=3.0, seed=41,
        unit_effect_sd=0.7, time_effect_sd=0.3, fe_x_corr=0.5,
    )  # fmt: skip
    priors = None if alpha_fixed is None else {"alpha_fixed": alpha_fixed}
    gibbs = _place(NegBinPanel, d, 64, 4, effects=3, priors=priors).fit(
        sampler="gibbs", draws=2000, tune=500, chains=2, progressbar=False,
        random_seed=1, n_jobs=1,
    )  # fmt: skip
    nuts = _place(NegBinPanel, d, 64, 4, effects=3, priors=priors).fit(
        sampler="nuts", draws=1000, tune=1000, chains=2, progressbar=False,
        random_seed=1,
    )  # fmt: skip
    for var in ("beta", "time_effect"):
        a = gibbs.posterior[var].values.reshape(-1, gibbs.posterior[var].shape[-1])
        b = nuts.posterior[var].values.reshape(-1, nuts.posterior[var].shape[-1])
        assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.2 * b.std(0)), var
        np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.15)
