"""Spatial multilevel model: construction, the sampler's algebra, and exactness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import multivariate_normal

from neighbayes import dgp
from neighbayes.models import Level, SpatialMultilevel
from neighbayes.samplers.multilevel import MultilevelStructure, log_marginal


def _data(n_side=8, blocks=(2, 2), processes=None, seed=0, **kw):
    L = len(blocks)
    return dgp.simulate_spatial_multilevel(
        n_side=n_side,
        blocks=blocks,
        processes=processes or ("lag",) * (L + 1),
        rhos=kw.pop("rhos", (0.4, 0.5, 0.4)[: L + 1]),
        sigmas=kw.pop("sigmas", (1.0, 0.6, 0.6)[: L + 1]),
        seed=seed,
        **kw,
    )


def _model(d, processes=None, durbin=None, **kw):
    f = d["frames"]
    processes = processes or ("lag",) * len(f)
    durbin = durbin or (False,) * len(f)
    levels = [
        Level(
            "y ~ x1 + x2", data=f[0], W=d["W"], process=processes[0], durbin=durbin[0]
        )
    ]
    for ell in range(1, len(f)):
        levels.append(
            Level(
                "~ z1",
                data=f[ell],
                W=d["levels"][ell - 1]["W"],
                key=f"g{ell}",
                process=processes[ell],
                durbin=durbin[ell],
            )
        )
    return SpatialMultilevel(*levels, **kw)


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


def test_dgp_nests_and_spans_parents():
    d = _data(n_side=8, blocks=(2, 2))
    u, l1, l2 = d["frames"]
    assert (len(u), len(l1), len(l2)) == (64, 16, 4)
    # every unit's g2 is its g1's parent
    np.testing.assert_array_equal(l1["g2"].to_numpy()[u["g1"]], u["g2"])
    # the level-1 graph links groups with different parents
    W1 = d["levels"][0]["W_sparse"].tocoo()
    par = d["levels"][0]["parent"]
    assert np.any(par[W1.row] != par[W1.col])


# ---------------------------------------------------------------------------
# The sampler's algebra against a dense reduced form
# ---------------------------------------------------------------------------


def _dense_log_marginal(m, rho, sig2):
    """``log p(y | ρ, σ)`` from the reduced form, β integrated under its prior."""
    mean = cov = None
    for ell in range(m.L, -1, -1):
        r = m._levels[ell]
        J = len(r.ids)
        mu_b, sd_b = m._beta_prior(ell)
        m_l = r.X @ mu_b if r.X.shape[1] else np.zeros(J)
        V_l = (r.X * sd_b**2) @ r.X.T if r.X.shape[1] else np.zeros((J, J))
        if ell < m.L:
            D = np.zeros((J, len(mean)))
            D[np.arange(J), r.parent] = 1.0
            m_l = m_l + D @ mean
            V_l = V_l + D @ cov @ D.T
        if r.process == "none":
            mean, cov = m_l, V_l + sig2[ell] * np.eye(J)
            continue
        S = np.linalg.inv(np.eye(J) - rho[ell] * r.W.toarray())
        if r.process == "lag":
            mean, cov = S @ m_l, S @ (V_l + sig2[ell] * np.eye(J)) @ S.T
        else:
            mean, cov = m_l, V_l + sig2[ell] * S @ S.T
    return multivariate_normal(mean, cov).logpdf(m._y)


@pytest.mark.parametrize(
    "processes, durbin",
    [
        (("lag", "lag", "lag"), (False, False, False)),
        (("error", "lag", "error"), (False, False, False)),
        (("lag", "error", "none"), (False, True, False)),
        (("none", "lag", "lag"), (True, False, True)),
    ],
)
def test_collapsed_marginal_matches_dense_reduced_form(processes, durbin):
    d = _data(n_side=8, blocks=(2, 2), seed=3)
    m = _model(d, processes, durbin)
    st = MultilevelStructure(m._y, m._level_specs())
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(4):
        rho = rng.uniform(-0.6, 0.8, size=3)
        sig2 = rng.uniform(0.2, 1.5, size=3)
        vals.append((log_marginal(st, rho, sig2), _dense_log_marginal(m, rho, sig2)))
    sparse_diff = np.diff([v[0] for v in vals])
    dense_diff = np.diff([v[1] for v in vals])
    np.testing.assert_allclose(sparse_diff, dense_diff, rtol=1e-8, atol=1e-8)


def test_precision_matches_its_residual_definition():
    d = _data(n_side=8, blocks=(2, 2), seed=4)
    m = _model(d, ("error", "lag", "error"))
    st = MultilevelStructure(m._y, m._level_specs())
    rho, sig2 = np.array([0.3, -0.4, 0.6]), np.array([0.7, 0.5, 1.3])
    P = st.precision(rho, sig2).toarray()
    # direct: Σ R_ℓᵀR_ℓ/σ_ℓ² + (A_0C)ᵀ(A_0C)/σ_0² + Λ
    A0C = st.C - rho[0] * (m._levels[0].W @ st.C)
    direct = (A0C.T @ A0C).toarray() / sig2[0]
    for ell in (1, 2):
        R = st.R0[ell] + rho[ell] * st.R1[ell]
        direct += (R.T @ R).toarray() / sig2[ell]
    for ell in range(3):
        k = st.k[ell]
        if k:
            sl = slice(st.off_b[ell], st.off_b[ell] + k)
            idx = np.arange(sl.start, sl.stop)
            direct[idx, idx] += 1.0 / m._beta_prior(ell)[1] ** 2
    np.testing.assert_allclose(P, direct, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Construction and linking
# ---------------------------------------------------------------------------


def test_keys_resolve_through_the_units_when_a_level_has_no_data():
    d = _data(n_side=8, blocks=(2, 2), seed=1)
    u, l1, l2 = d["frames"]
    a = _model(d)
    b = SpatialMultilevel(
        Level("y ~ x1 + x2", data=u, W=d["W"]),
        Level(W=d["levels"][0]["W"], key="g1"),  # no data: no covariates
        Level("~ z1", data=l2, W=d["levels"][1]["W"], key="g2"),
    )
    np.testing.assert_array_equal(a._levels[1].parent, b._levels[1].parent)
    assert b._levels[1].X.shape == (16, 0)


def test_graph_ids_order_the_groups_and_reindex_the_data():
    d = _data(n_side=8, blocks=(2,), seed=2)
    u, l1 = d["frames"]
    W1 = d["levels"][0]["W"]
    shuffled = l1.sample(frac=1.0, random_state=0)
    m = SpatialMultilevel(
        Level("y ~ x1 + x2", data=u, W=d["W"]),
        Level("~ z1", data=shuffled, W=W1, key="g1"),
    )
    assert m._levels[1].ids == list(W1.unique_ids)
    np.testing.assert_allclose(m._levels[1].X[:, 0], l1["z1"].to_numpy())


def test_matrix_mode_with_groups():
    d = _data(n_side=8, blocks=(2,), seed=2)
    lv = d["levels"][0]
    m = SpatialMultilevel(
        Level(y=d["y"], X=d["X"], W=d["W_sparse"]),
        Level(X=lv["X"], W=lv["W_sparse"], groups=d["parent"]),
    )
    np.testing.assert_array_equal(m._levels[0].parent, d["parent"])
    assert m._levels[1].names == ["x0"]


def test_upper_intercepts_are_dropped():
    d = _data(n_side=8, blocks=(2,), seed=2)
    u, l1 = d["frames"]
    m = SpatialMultilevel(
        Level("y ~ x1", data=u, W=d["W"]),
        Level("~ z1", data=l1, W=d["levels"][0]["W"], key="g1"),
    )
    assert m._levels[1].names == ["z1"]
    X = np.column_stack([np.ones(16), l1["z1"]])
    with pytest.warns(UserWarning, match="constant"):
        m2 = SpatialMultilevel(
            Level("y ~ x1", data=u, W=d["W"]),
            Level(X=X, W=d["levels"][0]["W_sparse"], groups=u["g1"].to_numpy()),
        )
    assert m2._levels[1].X.shape == (16, 1)


def test_linking_errors():
    d = _data(n_side=8, blocks=(2, 2), seed=1)
    u, l1, l2 = d["frames"]
    W1, W2 = d["levels"][0]["W"], d["levels"][1]["W"]
    with pytest.raises(ValueError, match="key"):
        SpatialMultilevel(Level("y ~ x1", data=u, W=d["W"]), Level(W=W1))
    with pytest.raises(ValueError, match="neither"):
        SpatialMultilevel(Level("y ~ x1", data=u, W=d["W"]), Level(W=W1, key="nope"))
    bad = u.copy()
    bad.loc[0, "g2"] = (bad.loc[0, "g2"] + 1) % 4  # one unit leaves its parent
    with pytest.raises(ValueError, match="not nested"):
        SpatialMultilevel(
            Level("y ~ x1", data=bad, W=d["W"]),
            Level(W=W1, key="g1"),
            Level(W=W2, key="g2"),
        )
    with pytest.raises(ValueError, match="needs W"):
        SpatialMultilevel(Level("y ~ x1", data=u, W=d["W"]), Level(key="g1"))
    with pytest.raises(ValueError, match="at least one level"):
        SpatialMultilevel(Level("y ~ x1", data=u, W=d["W"]))


def test_level_without_graph_when_process_is_none():
    d = _data(n_side=8, blocks=(2,), seed=2)
    u, l1 = d["frames"]
    m = SpatialMultilevel(
        Level("y ~ x1", data=u, W=d["W"]),
        Level("~ z1", data=l1, key="g1", process="none"),
    )
    idata = m.fit(
        draws=30, tune=20, chains=1, progressbar=False, n_jobs=1, random_seed=0
    )
    assert "rho_1" not in idata.posterior and "sigma_1" in idata.posterior


# ---------------------------------------------------------------------------
# Sampling (short chains)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    "parametrization", ["collapsed", "interweave", "centred", "noncentred"]
)
def test_parametrizations_run_and_name_by_level(parametrization, backend):
    d = _data(n_side=8, blocks=(2, 2), seed=5)
    m = _model(d, ("lag", "error", "lag"), (False, True, False))
    idata = m.fit(
        draws=40, tune=20, chains=2, progressbar=False, n_jobs=1, random_seed=1,
        parametrization=parametrization, gibbs_backend=backend,
    )  # fmt: skip
    post = idata.posterior
    for v in ("rho_0", "beta_0", "sigma_0", "lam_1", "beta_1", "sigma_1",
              "theta_1", "rho_2", "beta_2", "sigma_2", "theta_2"):  # fmt: skip
        assert v in post, v
        assert np.all(np.isfinite(post[v].values)), v
    assert list(post["beta_1"].coords["coefficient_1"].values) == ["z1", "W*z1"]
    assert post["theta_2"].shape == (2, 40, 4)
    s = m.summary()
    assert "theta_1" not in " ".join(s.index)


# ---------------------------------------------------------------------------
# JAX backend: compilation and precision
# ---------------------------------------------------------------------------


def test_jax_compiles_once_per_structure(monkeypatch):
    """A refit, another run length and new data of the same shape reuse the sweep."""
    import neighbayes.samplers._utils._jax_utils as U

    monkeypatch.setattr(U, "_SWEEPS", {})
    calls = []
    real = U._compile_chunk
    monkeypatch.setattr(
        U, "_compile_chunk", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    kw = dict(chains=2, progressbar=False, gibbs_backend="jax", store_theta=False)
    m = _model(_data(n_side=8, blocks=(2, 2), seed=12))
    m.fit(draws=30, tune=10, random_seed=0, **kw)
    m.fit(draws=30, tune=10, random_seed=1, **kw)
    m.fit(draws=300, tune=40, random_seed=2, **kw)
    _model(_data(n_side=8, blocks=(2, 2), seed=13)).fit(
        draws=30, tune=10, random_seed=3, **kw
    )
    assert len(calls) == 1


def test_jax_chunking_does_not_change_draws():
    m = _model(_data(n_side=8, blocks=(2,), seed=14))
    kw = dict(chains=1, progressbar=False, gibbs_backend="jax", random_seed=5, tune=20)
    short = m.fit(draws=50, **kw).posterior["rho_1"].values
    long = m.fit(draws=300, **kw).posterior["rho_1"].values
    np.testing.assert_array_equal(short[0], long[0, :50])


def test_jax_fit_builds_float64_params_when_x64_starts_off():
    """Regression: params cached before x64 was enabled came out float32.

    The backend dispatch now enables float64 before the model builds anything.
    """
    import jax

    m = _model(_data(n_side=8, blocks=(2, 2), seed=15))
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        m.fit(draws=5, tune=5, chains=1, progressbar=False, gibbs_backend="jax")
        assert jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", was or True)
    kind, params = m._logdet_jax_cache[1]
    assert kind == "eig"
    assert all(np.asarray(a).dtype == np.float64 for a in params)


def test_store_theta_and_log_likelihood():
    d = _data(n_side=8, blocks=(2,), seed=6)
    m = _model(d)
    idata = m.fit(
        draws=20, tune=10, chains=2, progressbar=False, n_jobs=1, random_seed=0,
        store_theta=False, idata_kwargs={"log_likelihood": True},
    )  # fmt: skip
    assert "theta_1" not in idata.posterior
    assert idata.log_likelihood["obs"].shape == (2, 20, 64)
    with pytest.raises(RuntimeError, match="store_theta"):
        m.posterior_predictive()


def test_unknown_gibbs_option_raises():
    m = _model(_data(n_side=8, blocks=(2,), seed=6))
    with pytest.raises(TypeError, match="unsupported"):
        m.fit(draws=5, tune=5, chains=1, progressbar=False, slice_width=0.1)
    with pytest.raises(ValueError, match="parametrization"):
        m.fit(draws=5, tune=5, chains=1, progressbar=False, n_jobs=1,
              parametrization="sideways")  # fmt: skip


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_lag3():
    d = _data(n_side=12, blocks=(2, 3), seed=7)
    m = _model(d)
    m.fit(draws=60, tune=40, chains=2, progressbar=False, n_jobs=1, random_seed=2)
    return m


def test_unit_effects_totals_follow_the_multiplier(fitted_lag3):
    m = fitted_lag3
    _, s = m.spatial_effects(level=0, return_posterior_samples=True)
    rho0 = m._draws("rho_0")
    beta = m._draws("beta_0")[:, 1:]
    np.testing.assert_allclose(s["total"], beta / (1.0 - rho0[:, None]), rtol=1e-8)


def test_composed_totals_multiply_the_filters(fitted_lag3):
    m = fitted_lag3
    _, s = m.spatial_effects(level=2, return_posterior_samples=True)
    mult = np.prod([1.0 / (1.0 - m._draws(f"rho_{ell}")) for ell in range(3)], axis=0)
    np.testing.assert_allclose(
        s["total"][:, 0], m._draws("beta_2")[:, 0] * mult, rtol=1e-8
    )
    # within the level: β / (1 − ρ_2)
    _, w = m.spatial_effects(level="g2", on="level", return_posterior_samples=True)
    np.testing.assert_allclose(
        w["total"][:, 0],
        m._draws("beta_2")[:, 0] / (1.0 - m._draws("rho_2")),
        rtol=1e-8,
    )


def test_probe_trace_estimates_the_exact_one(fitted_lag3):
    m = fitted_lag3
    _, exact = m.spatial_effects(level=1, return_posterior_samples=True, max_draws=20)
    _, probe = m.spatial_effects(
        level=1, return_posterior_samples=True, max_draws=20, exact_max=0,
        n_probes=4000, random_seed=0,
    )  # fmt: skip
    np.testing.assert_allclose(probe["direct"], exact["direct"], rtol=0.05)
    np.testing.assert_allclose(probe["total"], exact["total"], rtol=1e-10)


def test_fitted_and_predictive_shapes(fitted_lag3):
    m = fitted_lag3
    assert m.fitted_values().shape == (144,)
    assert m.posterior_predictive(max_draws=7, random_seed=0).shape == (7, 144)


# ---------------------------------------------------------------------------
# NUTS build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "processes", [("lag", "lag", "error"), ("error", "none", "lag")]
)
def test_pymc_model_builds_with_the_gibbs_names(processes):
    m = _model(_data(n_side=8, blocks=(2, 2), seed=8), processes)
    model = m._build_pymc_model()
    names = set(model.named_vars)
    for v in m._param_names():
        assert v in names, v
    assert np.isfinite(model.compile_logp()(model.initial_point()))


# ---------------------------------------------------------------------------
# Exactness and recovery (slow)
# ---------------------------------------------------------------------------


def _flat(post, var):
    x = post[var].values
    return x.reshape(-1, *x.shape[2:])


@pytest.mark.slow
@pytest.mark.parametrize(
    "processes, durbin",
    [
        (("lag", "lag", "lag"), (False, False, False)),
        (("lag", "error", "none"), (False, False, False)),
        (("error", "lag"), (False, False)),
        (("none", "lag"), (False, True)),
    ],
)
def test_gibbs_matches_nuts(processes, durbin):
    L = len(processes) - 1
    d = _data(n_side=16, blocks=(2, 2)[:L], processes=processes, seed=11)
    m = _model(d, processes, durbin)
    names = m._param_names()
    nuts = m.fit(
        sampler="nuts", draws=2500, tune=2500, chains=4, target_accept=0.95,
        progressbar=False, random_seed=2,
    ).posterior  # fmt: skip
    ref = {v: _flat(nuts, v) for v in names}
    for par in ("collapsed", "interweave"):
        post = m.fit(
            draws=5000, tune=1000, chains=4, progressbar=False, random_seed=1,
            parametrization=par,
        ).posterior  # fmt: skip
        for v in names:
            a, b = _flat(post, v), ref[v]
            if v == "beta_0":
                # The intercept is heavy-tailed (its spread grows as the top
                # level's ρ → 1, where NUTS diverges): compare quantiles.
                qa = np.quantile(a, [0.25, 0.5, 0.75], axis=0)
                qb = np.quantile(b, [0.25, 0.5, 0.75], axis=0)
                assert np.all(np.abs(qa - qb) < 0.15 * b.std(0)), (par, v)
                continue
            assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.15 * b.std(0)), (par, v)
            np.testing.assert_allclose(
                a.std(0), b.std(0), rtol=0.12, err_msg=f"{par} {v}"
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "processes, durbin",
    [
        (("lag", "lag", "lag"), (False, False, False)),
        (("error", "error", "none"), (False, True, False)),
    ],
)
def test_jax_matches_numpy(processes, durbin):
    """The two backends sample one posterior (means within Monte Carlo error)."""
    import arviz as az

    d = _data(n_side=16, blocks=(2, 2), processes=processes, seed=11)
    m = _model(d, processes, durbin)
    kw = dict(draws=4000, tune=1000, chains=4, progressbar=False, store_theta=False)
    ref = m.fit(gibbs_backend="numpy", random_seed=1, **kw)
    for par in ("collapsed", "interweave"):
        got = m.fit(gibbs_backend="jax", parametrization=par, random_seed=2, **kw)
        for v in m._param_names():
            a, b = _flat(got.posterior, v), _flat(ref.posterior, v)
            se = np.hypot(
                np.atleast_1d(az.mcse(got, var_names=[v])[v].values),
                np.atleast_1d(az.mcse(ref, var_names=[v])[v].values),
            )
            assert np.all(np.abs(a.mean(0) - b.mean(0)) < 4.0 * se), (par, v)


@pytest.mark.slow
def test_recovery():
    d = _data(n_side=36, blocks=(3, 3), seed=21, rhos=(0.4, 0.5, 0.5),
              sigmas=(1.0, 0.5, 0.5))  # fmt: skip
    m = _model(d)
    post = m.fit(
        draws=1500, tune=500, chains=4, progressbar=False, random_seed=3
    ).posterior
    truth = d["params"]
    checks = {
        "rho_0": truth["rhos"][0],
        "rho_1": truth["rhos"][1],
        "sigma_0": truth["sigmas"][0],
        "sigma_1": truth["sigmas"][1],
    }
    for v, t in checks.items():
        x = _flat(post, v)
        assert abs(x.mean() - t) < 3 * x.std(), v
    b0 = _flat(post, "beta_0")
    for j in (1, 2):
        assert abs(b0[:, j].mean() - truth["betas"][0][j]) < 3 * b0[:, j].std()
    b1 = _flat(post, "beta_1")[:, 0]
    assert abs(b1.mean() - truth["betas"][1][0]) < 3 * b1.std()
    # effects: posterior means track the truth
    th = _flat(post, "theta_1").mean(0)
    assert np.corrcoef(th, d["levels"][0]["theta"])[0, 1] > 0.8


def test_unit_variance_prior_scale_is_within_group():
    """σ₀²'s default scale leaves out the upper levels' variance.

    ``Var(y)`` counts every upper level, and an Inv-Γ(a, b) prior shifts the
    posterior mean of σ₀² by about ``2b/n``: with strong upper levels that was
    3–4 posterior sd.  The default is the pooled variance within level-1 groups.
    """
    from neighbayes.models.priors import MultilevelPriors

    d = _data(n_side=8, blocks=(2, 2), sigmas=(1.0, 3.0, 3.0))
    m = _model(d)
    y = np.asarray(d["y"], dtype=float)
    g = d["frames"][0]["g1"].to_numpy()
    within = sum(((y[g == k] - y[g == k].mean()) ** 2).sum() for k in np.unique(g))
    expected = within / (y.size - np.unique(g).size)
    scale = m._variance_priors()["sigma2_beta"]
    assert scale == pytest.approx(expected)
    assert scale < np.var(y)  # the upper levels' variance is excluded
    pinned = _model(d, priors=MultilevelPriors(sigma2_beta=2.5))
    assert pinned._variance_priors()["sigma2_beta"] == 2.5
