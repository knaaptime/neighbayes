"""The exact-likelihood separable NB flow sampler (``negbin_reduced._flow_exact``).

Its structured algebra must reproduce dense Kronecker solves, and one sweep from
a draw of the joint must return draws from it: the parameters from their
priors, the counts from the NB2 given them, on both backends, with and without
levels after the filter: random origin and destination levels; fixed pair and
period levels (the fixed-effects panel); fixed origin, destination and period
levels.  Fixed levels take a proper prior here, a flat one having no draws.
The posterior against an independent Pólya–Gamma reference is
``test_flow_separable_posterior.py``.

The counts' Poisson draws are exact for log-means below 40; a clipped
log-mean would generate them from another model than the one the sampler
targets.  The draws therefore redraw the rare replicate (probability near
10⁻⁶ under these priors, from the half-t tail of the level sds) whose
log-mean passes 40: conditioning on so rare an event moves the paired
differences' means far less than the checks resolve.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.samplers.negbin_reduced._flow import (
    FlowReducedGibbsPriors,
    FlowReducedGibbsState,
)
from neighbayes.samplers.negbin_reduced._flow_exact import (
    ExactFlowChain,
    FlowLevels,
    LevelSet,
    run_chain_separable_exact,
)
from neighbayes.samplers.negbin_reduced._flow_structured import classify_flow_design

N = 3
LO, HI = -0.5, 0.6
W3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float) / 2.0
#: The level designs the checks run.
LEVELS = {
    None: None,
    "origin": FlowLevels((LevelSet("origin", True, 0.5, 0.25),)),
    "both": FlowLevels(
        (LevelSet("origin", True, 0.5, 0.25), LevelSet("destination", True, -0.3, 0.3))
    ),
    "pair+period": FlowLevels(
        (LevelSet("pair", False, 0.3, 0.6), LevelSet("period", False, 0.0, 0.5))
    ),
    "random pair+period": FlowLevels(
        (LevelSet("pair", True, 0.3, 0.4), LevelSet("period", False, 0.0, 0.5))
    ),
    "od+period": FlowLevels(
        (
            LevelSet("origin", False, 0.3, 0.6),
            LevelSet("destination", False, 0.0, 0.5),
            LevelSet("period", False, 0.0, 0.5),
        )
    ),
}
#: The ``(T, n, n)`` axes each kind of level is constant over.
_AXES = {"origin": (0, 2), "destination": (0, 1), "pair": (0,), "period": (1, 2)}


def _absorbed(X, T, levels):
    """Which columns of ``X`` lie in the span of the level sets."""
    rest = X.reshape(T, N, N, -1)
    for st in levels.sets if levels is not None else ():
        rest = rest - rest.mean(axis=_AXES[st.kind], keepdims=True)
    return np.abs(rest).max(axis=(0, 1, 2)) < 1e-10


def _case(T, rng, levels=None):
    """``(X, struct, priors)``: log-distance, an intercept, origin and destination
    attributes and, in a panel, a time-varying origin attribute, a time-varying
    distance term and period dummies unless period levels take their place;
    less the columns the level sets absorb."""
    xo, xd = rng.standard_normal(N), rng.standard_normal(N)
    xo_t = rng.standard_normal((T, N))
    dist = np.log1p(np.abs(rng.standard_normal((N, N))))
    periods = levels is not None and any(st.kind == "period" for st in levels.sets)
    blocks = []
    for t in range(T):
        cols = [
            dist,
            np.ones((N, N)),
            np.outer(xo, np.ones(N)),
            np.outer(np.ones(N), xd),
        ]
        if T > 1:
            cols += [np.outer(xo_t[t], np.ones(N)), dist * (1.0 + 0.5 * t)]
            if not periods:
                cols += [np.full((N, N), float(t == s)) for s in range(1, T)]
        blocks.append(np.stack([c.ravel() for c in cols], 1))
    X = np.vstack(blocks)
    X = X[:, ~_absorbed(X, T, levels)]
    k = X.shape[1]
    priors = FlowReducedGibbsPriors(
        beta_mu=np.zeros(k), beta_sigma=np.full(k, 0.5), alpha_sigma=2.5,
        alpha_nu=3.0, alpha_fixed=None, rho_lower=LO, rho_upper=HI,
    )  # fmt: skip
    return X, classify_flow_design(X, N, T), priors


def _eta(X, T, rho_d, rho_o, beta):
    """``η = (L_o ⊗ L_d)⁻¹ Xβ`` per period, densely."""
    A = np.kron(np.eye(N) - rho_o * W3, np.eye(N) - rho_d * W3)
    return np.concatenate(
        [np.linalg.solve(A, X[t * N * N : (t + 1) * N * N] @ beta) for t in range(T)]
    )


def _level_sum(T, levels, lev):
    """The levels' part of the log-mean, flat over the periods."""
    out = np.zeros((T, N, N))
    for st, v in zip(levels.sets if levels is not None else (), lev):
        out = out + np.expand_dims(v, _AXES[st.kind])
    return out.ravel()


def _half_t(rng, scale):
    """A half-t(3, scale) draw within the samplers' ``(e^-10, e^10)`` bounds."""
    while True:
        v = scale * abs(rng.standard_t(3.0))
        if np.exp(-10.0) < v < np.exp(10.0):
            return v


def _joint_params(rng, X, T, priors, levels):
    """The parameters from their priors, and the log-means."""
    p = {"rho_d": rng.uniform(LO, HI), "rho_o": rng.uniform(LO, HI)}
    p["beta"] = priors.beta_mu + priors.beta_sigma * rng.standard_normal(X.shape[1])
    p["alpha"] = _half_t(rng, 2.5)
    p["lev"], p["tau"] = [], []
    shapes = {"origin": (N,), "destination": (N,), "pair": (N, N), "period": (T,)}
    for i, st in enumerate(levels.sets if levels is not None else ()):
        sd = _half_t(rng, st.scale) if st.random else st.scale
        v = st.mu + sd * rng.standard_normal(shapes[st.kind])
        if levels.baseline(i):
            v.reshape(-1)[0] = st.mu
        p["lev"].append(v)
        p["tau"].append(sd if st.random else np.inf)
    eta = _eta(X, T, p["rho_d"], p["rho_o"], p["beta"])
    return p, eta + _level_sum(T, levels, p["lev"])


def _joint_draw(rng, X, T, priors, levels=None):
    """``(params, y)``: the parameters from their priors, the counts from the NB2."""
    for _ in range(3):
        p, eta = _joint_params(rng, X, T, priors, levels)
        if eta.max() < 40.0:
            y = rng.poisson(rng.gamma(p["alpha"], np.exp(eta) / p["alpha"]))
            return p, y.astype(np.float64)
    raise AssertionError("the test priors let log-means overflow repeatedly")


def _init(p, k):
    return FlowReducedGibbsState(
        beta=p.get("beta", np.zeros(k)), rho_d=p.get("rho_d", 0.0),
        rho_o=p.get("rho_o", 0.0), rho_w=None, alpha=p.get("alpha", 1.0), omega=None,
    )  # fmt: skip


#: Fixed slice directions off the axes, for the invariance checks.
_AB_DIRS = np.linalg.qr(np.array([[1.0, 0.6], [0.3, 1.0]]))[0]
_AB_WIDTHS = np.array([0.4, 0.3])


def _theta_dirs(k):
    return np.linalg.qr(np.random.default_rng(5).standard_normal((k, k)))[0]


def _shear(k):
    return np.random.default_rng(6).normal(0.0, 0.5, (k, 2))


def _diffs(p, d, levels):
    """The paired differences a sweep must leave centred at zero."""
    out = [d["rho_d"] - p["rho_d"], d["rho_o"] - p["rho_o"]]
    out += [np.log(d["alpha"] / p["alpha"]), *(d["beta"] - p["beta"])]
    for i, st in enumerate(levels.sets if levels is not None else ()):
        new, old = np.ravel(d[f"{st.kind}_effect"]), np.ravel(p["lev"][i])
        out += [new.mean() - old.mean(), new[-1] - old[-1]]
        if st.random:
            out += [np.log(d[f"{st.kind}_effect_sd"] / p["tau"][i])]
    return out


def _z(d):
    return d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))


@pytest.mark.parametrize("T", [1, 2])
def test_columns_and_eta_match_dense(T):
    rng = np.random.default_rng(0)
    X, struct, priors = _case(T, rng)
    p, y = _joint_draw(rng, X, T, priors)
    chain = ExactFlowChain(
        y, struct, sp.csc_matrix(W3), priors, _init(p, X.shape[1]), 10, rng, LO, HI
    )
    rho_d, rho_o, beta = p["rho_d"], p["rho_o"], p["beta"]
    np.testing.assert_allclose(chain.eta, _eta(X, T, rho_d, rho_o, beta), atol=1e-12)
    m = 1.0 / ((1.0 - rho_d) * (1.0 - rho_o))
    dense = np.column_stack([_eta(X, T, rho_d, rho_o, e) for e in np.eye(X.shape[1])])
    cols = chain._columns().reshape(X.shape[1], -1)
    np.testing.assert_allclose(cols, dense.T / m, atol=1e-12)
    draw = chain.draw()
    assert draw["rho_w"] == pytest.approx(-rho_d * rho_o)
    np.testing.assert_allclose(draw["beta"], beta)


@pytest.mark.parametrize("levels", ["both", "pair+period", "od+period"])
def test_levels_enter_outside_the_filter(levels):
    rng = np.random.default_rng(2)
    lv = LEVELS[levels]
    X, struct, priors = _case(3, rng, lv)
    p, y = _joint_draw(rng, X, 3, priors, lv)
    chain = ExactFlowChain(
        y, struct, sp.csc_matrix(W3), priors, _init(p, X.shape[1]), 10, rng, LO, HI, lv
    )
    chain.lev = [np.copy(v) for v in p["lev"]]
    eta = _eta(X, 3, p["rho_d"], p["rho_o"], p["beta"]) + _level_sum(3, lv, p["lev"])
    np.testing.assert_allclose(chain._eta_at(chain.a, chain.b, chain.theta), eta)


def test_flat_levels_drop_units_without_counts():
    """A flat pair level with no count in any period leaves the likelihood."""
    rng = np.random.default_rng(4)
    flat = FlowLevels((LevelSet("pair"), LevelSet("period", False, 0.0, 2.5)))
    X, struct, priors = _case(3, rng, flat)
    _, y = _joint_draw(rng, X, 3, priors, LEVELS["pair+period"])
    Y = y.reshape(3, N, N)
    Y[:, 0, 1] = 0.0
    chain = ExactFlowChain(
        Y.ravel(), struct, sp.csc_matrix(W3), priors, _init({}, X.shape[1]), 10, rng,
        LO, HI, flat,
    )  # fmt: skip
    dead = Y.sum(axis=0) == 0
    assert np.array_equal(chain.live[0] == 0, dead)
    chain.sweep(tuning=True)
    d = chain.draw()
    assert np.isnan(d["pair_effect"][0, 1])
    assert np.all(np.isfinite(d["pair_effect"][~dead]))
    assert d["period_effect"][0] == 0.0
    assert np.all(chain.log_lik().reshape(3, N, N)[:, dead] == 0.0)


def _run(backend, y, struct, priors, levels, **kw):
    """Two chains of 20 + 30 sweeps on ``backend``."""
    inits = [_init({}, struct.k) for _ in range(2)]
    W = sp.csc_matrix(W3)
    if backend == "numpy":
        return [
            run_chain_separable_exact(
                y,
                None,
                W,
                N,
                priors,
                i,
                30,
                20,
                T=struct.T,
                rho_lower=LO,
                rho_upper=HI,
                rng=np.random.default_rng(c),
                struct=struct,
                levels=levels,
                **kw,
            )  # fmt: skip
            for c, i in enumerate(inits)
        ]
    from neighbayes.samplers.negbin_reduced._flow_exact_jax import (
        run_chains_jax_flow_exact,
    )

    return run_chains_jax_flow_exact(
        y, W, N, priors, inits, 30, 20, struct=struct, jax_seeds=[1, 2],
        levels=levels, **kw,
    )  # fmt: skip


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
@pytest.mark.parametrize("levels", [None, "both", "pair+period"])
def test_runs_and_stores_the_pointwise_likelihood(backend, levels):
    rng = np.random.default_rng(1)
    lv = LEVELS[levels]
    X, struct, priors = _case(3, rng, lv)
    _, y = _joint_draw(rng, X, 3, priors, lv)
    k = X.shape[1]
    out = _run(backend, y, struct, priors, lv, store_log_lik=True)
    for r in out:
        assert r["beta"].shape == (30, k) and r["log_lik"].shape == (30, y.size)
        assert np.all(np.isfinite(r["log_lik"]))
        assert np.all((LO < r["rho_d"]) & (r["rho_d"] < HI))
        np.testing.assert_allclose(r["rho_w"], -r["rho_d"] * r["rho_o"])
        for st in lv.sets if lv is not None else ():
            assert r[f"{st.kind}_effect"].shape[0] == 30
            if st.random:
                assert np.all(r[f"{st.kind}_effect_sd"] > 0)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_pair_summary_is_the_moments_of_the_draws(backend):
    """Without the pair draws, the same chains report their running moments."""
    rng = np.random.default_rng(7)
    lv = LEVELS["pair+period"]
    X, struct, priors = _case(3, rng, lv)
    _, y = _joint_draw(rng, X, 3, priors, lv)
    full = _run(backend, y, struct, priors, lv)
    summ = _run(backend, y, struct, priors, lv, keep_pair_draws=False)
    for f, s in zip(full, summ):
        assert "pair_effect" not in s
        np.testing.assert_allclose(s["beta"], f["beta"])
        np.testing.assert_allclose(s["period_effect"], f["period_effect"])
        draws = f["pair_effect"]
        np.testing.assert_allclose(
            s["pair_effect_summary"]["mean"], draws.mean(0), atol=1e-10
        )
        np.testing.assert_allclose(
            s["pair_effect_summary"]["sd"], draws.std(0, ddof=1), atol=1e-10
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "T, levels",
    [(1, None), (2, None), (1, "origin"), (1, "both"), (2, "both"),
     (3, "pair+period"), (3, "random pair+period"), (3, "od+period")],
)  # fmt: skip
def test_sweep_is_invariant(T, levels):
    """One NumPy sweep from a draw of the joint, along fixed directions."""
    rng = np.random.default_rng(10 + T)
    lv = LEVELS[levels]
    X, struct, priors = _case(T, rng, lv)
    k = X.shape[1]
    W = sp.csc_matrix(W3)
    diffs = []
    for _ in range(30_000):
        p, y = _joint_draw(rng, X, T, priors, lv)
        chain = ExactFlowChain(y, struct, W, priors, _init(p, k), 100, rng, LO, HI, lv)
        if lv is not None:
            chain.lev = [np.copy(v) for v in p["lev"]]
            chain.tau = list(p["tau"])
            chain.eta = chain._eta_at(chain.a, chain.b, chain.theta)
        chain.th_dirs, chain.th_widths = _theta_dirs(k), np.full(k, 0.5)
        chain.ab_dirs, chain.ab_widths = _AB_DIRS, _AB_WIDTHS
        chain.ab_shear = _shear(k)
        chain.sweep(tuning=False)
        diffs.append(_diffs(p, chain.draw(), lv))
    z = _z(np.asarray(diffs))
    assert np.all(np.abs(z) < 4.0), z


def _jax_sweep_z(T, reps, seed, levels=None, prior_scale=1.0):
    """z-scores of one JAX sweep's paired differences, ``reps`` draws by ``vmap``.

    ``prior_scale`` multiplies the sampler's prior precision on β, away from
    the one the draws come from, for the power check.
    """
    import jax
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes.samplers.negbin_reduced._flow_exact_jax import (
        _exact_fns,
        _exact_parts,
        start_state,
        trace_names,
    )

    ensure_x64()
    rng = np.random.default_rng(seed)
    lv = LEVELS[levels]
    X, struct, priors = _case(T, rng, lv)
    k = X.shape[1]
    static, base = _exact_parts(
        np.ones(N * N * T), sp.csc_matrix(W3), struct, priors, 100, lv
    )
    base["prec0"] = base["prec0"] * prior_scale
    sweep = _exact_fns(*static)
    pad = 128  # the padded distinct counts, one length for every draw
    in_data = {key: None for key in base}
    in_data.update(y=0, y_vals=0, y_counts=0)
    template = start_state(struct, _init({}, k), lv)
    n_sets = len(template["lev"])
    in_state = {key: None for key in template}
    in_state.update(theta=0, a=0, b=0, alpha=0, lev=(0,) * n_sets, tau=(0,) * n_sets)
    names = ("rho_d", "rho_o", "beta", "alpha") + trace_names(lv)
    run = jax.jit(
        jax.vmap(
            lambda s, key, d: sweep(s, key, jnp.bool_(False), d)[1][: len(names)],
            in_axes=(in_state, 0, in_data),
        )
    )
    R = 10_000
    diffs = []
    for chunk in range(reps // R):
        draws = [_joint_draw(rng, X, T, priors, lv) for _ in range(R)]
        y = np.stack([d[1] for d in draws])
        tabs = [np.unique(v, return_counts=True) for v in y]
        data = {
            **base,
            "y": jnp.asarray(y),
            "y_vals": jnp.asarray([np.pad(v, (0, pad - v.size)) for v, _ in tabs]),
            "y_counts": jnp.asarray(
                [np.pad(c.astype(float), (0, pad - c.size)) for _, c in tabs]
            ),
        }
        P = [d[0] for d in draws]
        rd, ro = np.array([q["rho_d"] for q in P]), np.array([q["rho_o"] for q in P])
        a, b = -np.log1p(-rd), -np.log1p(-ro)
        beta = np.array([q["beta"] for q in P])
        tau = np.array([[t if np.isfinite(t) else 1.0 for t in q["tau"]] for q in P])
        state = {
            **template,
            "theta": jnp.asarray(beta * np.exp(a + b)[:, None]),
            "a": jnp.asarray(a),
            "b": jnp.asarray(b),
            "alpha": jnp.asarray([q["alpha"] for q in P]),
            "lev": tuple(
                jnp.asarray(np.stack([q["lev"][i] for q in P])) for i in range(n_sets)
            ),
            "tau": tuple(jnp.asarray(tau[:, i]) for i in range(n_sets)),
            "it": jnp.int32(1000),
            "th_dirs": jnp.asarray(_theta_dirs(k)),
            "th_w": jnp.full(k, 0.5),
            "ab_dirs": jnp.asarray(_AB_DIRS),
            "ab_w": jnp.asarray(_AB_WIDTHS),
            "ab_shear": jnp.asarray(_shear(k)),
        }
        keys = jax.random.split(jax.random.PRNGKey(seed * 100 + chunk), R)
        out = dict(zip(names, (np.asarray(v) for v in run(state, keys, data))))
        diffs += [
            _diffs(P[r], {key: v[r] for key, v in out.items()}, lv) for r in range(R)
        ]
    return _z(np.asarray(diffs))


@pytest.mark.slow
@pytest.mark.requires_jax
@pytest.mark.parametrize(
    "T, levels",
    [(1, None), (2, None), (1, "origin"), (2, "both"), (3, "pair+period"),
     (3, "random pair+period"), (3, "od+period")],
)  # fmt: skip
def test_jax_sweep_is_invariant(T, levels):
    z = _jax_sweep_z(T, 200_000, seed=30 + T, levels=levels)
    assert np.all(np.abs(z) < 4.0), z


@pytest.mark.slow
@pytest.mark.requires_jax
def test_jax_invariance_check_has_power():
    """The same check fails when the sampler's prior on β is not the draws'."""
    z = _jax_sweep_z(1, 100_000, seed=33, prior_scale=0.5)
    assert np.max(np.abs(z)) > 10.0, z
