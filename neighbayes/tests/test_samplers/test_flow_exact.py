"""The exact-likelihood separable NB flow sampler (``negbin_reduced._flow_exact``).

Its structured algebra must reproduce dense Kronecker solves, and one sweep from
a draw of the joint must return draws from it: the parameters from their
priors, the counts from the NB2 given them, on both backends.  The posterior
against an independent Pólya–Gamma reference is
``test_flow_separable_posterior.py``.
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
    run_chain_separable_exact,
)
from neighbayes.samplers.negbin_reduced._flow_structured import classify_flow_design

N = 3
LO, HI = -0.5, 0.8
W3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float) / 2.0


def _case(T, rng):
    """``(X, struct, priors)``: intercept, origin and destination attributes and
    log-distance, with period dummies in a panel."""
    xo, xd = rng.standard_normal(N), rng.standard_normal(N)
    dist = np.log1p(np.abs(rng.standard_normal((N, N))))
    blocks = []
    for t in range(T):
        cols = [
            np.ones((N, N)),
            np.outer(xo, np.ones(N)),
            np.outer(np.ones(N), xd),
            dist,
        ]
        cols += [np.full((N, N), float(t == s)) for s in range(1, T)]
        blocks.append(np.stack([c.ravel() for c in cols], 1))
    X = np.vstack(blocks)
    k = X.shape[1]
    priors = FlowReducedGibbsPriors(
        beta_mu=np.zeros(k), beta_sigma=np.full(k, 0.7), alpha_sigma=2.5,
        alpha_nu=3.0, alpha_fixed=None, rho_lower=LO, rho_upper=HI,
    )  # fmt: skip
    return X, classify_flow_design(X, N, T), priors


def _eta(X, T, rho_d, rho_o, beta):
    """``η = (L_o ⊗ L_d)⁻¹ Xβ`` per period, densely."""
    A = np.kron(np.eye(N) - rho_o * W3, np.eye(N) - rho_d * W3)
    return np.concatenate(
        [np.linalg.solve(A, X[t * N * N : (t + 1) * N * N] @ beta) for t in range(T)]
    )


def _joint_draw(rng, X, T, priors):
    """``(rho_d, rho_o, beta, alpha, y)`` from the priors and the NB2."""
    rho_d, rho_o = rng.uniform(LO, HI, 2)
    beta = priors.beta_mu + priors.beta_sigma * rng.standard_normal(X.shape[1])
    while True:  # the sampler bounds log alpha to [-10, 10]
        alpha = 2.5 * abs(rng.standard_t(3.0))
        if np.exp(-10.0) < alpha < np.exp(10.0):
            break
    mu = np.exp(np.clip(_eta(X, T, rho_d, rho_o, beta), -30.0, 30.0))
    y = rng.poisson(rng.gamma(alpha, mu / alpha)).astype(np.float64)
    return rho_d, rho_o, beta, alpha, y


#: Fixed slice directions off the axes, for the invariance checks.
_AB_DIRS = np.linalg.qr(np.array([[1.0, 0.6], [0.3, 1.0]]))[0]
_AB_WIDTHS = np.array([0.4, 0.3])


def _theta_dirs(k):
    return np.linalg.qr(np.random.default_rng(5).standard_normal((k, k)))[0]


def _shear(k):
    return np.random.default_rng(6).normal(0.0, 0.5, (k, 2))


def _z(d):
    return d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))


@pytest.mark.parametrize("T", [1, 2])
def test_columns_and_eta_match_dense(T):
    rng = np.random.default_rng(0)
    X, struct, priors = _case(T, rng)
    rho_d, rho_o, beta, alpha, y = _joint_draw(rng, X, T, priors)
    init = FlowReducedGibbsState(
        beta=beta, rho_d=rho_d, rho_o=rho_o, rho_w=None, alpha=alpha, omega=None
    )
    chain = ExactFlowChain(y, struct, sp.csc_matrix(W3), priors, init, 10, rng, LO, HI)
    np.testing.assert_allclose(chain.eta, _eta(X, T, rho_d, rho_o, beta), atol=1e-12)
    m = 1.0 / ((1.0 - rho_d) * (1.0 - rho_o))
    dense = np.column_stack([_eta(X, T, rho_d, rho_o, e) for e in np.eye(X.shape[1])])
    np.testing.assert_allclose(chain._columns(), dense.T / m, atol=1e-12)
    draw = chain.draw()
    assert draw["rho_w"] == pytest.approx(-rho_d * rho_o)
    np.testing.assert_allclose(draw["beta"], beta)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_runs_and_stores_the_pointwise_likelihood(backend):
    rng = np.random.default_rng(1)
    X, struct, priors = _case(2, rng)
    *_, y = _joint_draw(rng, X, 2, priors)
    k = X.shape[1]
    inits = [
        FlowReducedGibbsState(
            beta=np.zeros(k), rho_d=0.0, rho_o=0.0, rho_w=None, alpha=1.0, omega=None
        )
        for _ in range(2)
    ]
    W = sp.csc_matrix(W3)
    if backend == "numpy":
        out = [
            run_chain_separable_exact(
                y,
                None,
                W,
                N,
                priors,
                i,
                30,
                20,
                T=2,
                rho_lower=LO,
                rho_upper=HI,
                rng=np.random.default_rng(c),
                store_log_lik=True,
                struct=struct,
            )  # fmt: skip
            for c, i in enumerate(inits)
        ]
    else:
        from neighbayes.samplers.negbin_reduced._flow_exact_jax import (
            run_chains_jax_flow_exact,
        )

        out = run_chains_jax_flow_exact(
            y, W, N, priors, inits, 30, 20, struct=struct, jax_seeds=[1, 2],
            store_log_lik=True,
        )  # fmt: skip
    for r in out:
        assert r["beta"].shape == (30, k) and r["log_lik"].shape == (30, y.size)
        assert np.all(np.isfinite(r["log_lik"]))
        assert np.all((LO < r["rho_d"]) & (r["rho_d"] < HI))
        np.testing.assert_allclose(r["rho_w"], -r["rho_d"] * r["rho_o"])


@pytest.mark.slow
@pytest.mark.parametrize("T", [1, 2])
def test_sweep_is_invariant(T):
    """One NumPy sweep from a draw of the joint, along fixed directions."""
    rng = np.random.default_rng(10 + T)
    X, struct, priors = _case(T, rng)
    k = X.shape[1]
    W = sp.csc_matrix(W3)
    diffs = []
    for _ in range(30_000):
        rho_d, rho_o, beta, alpha, y = _joint_draw(rng, X, T, priors)
        init = FlowReducedGibbsState(
            beta=beta, rho_d=rho_d, rho_o=rho_o, rho_w=None, alpha=alpha, omega=None
        )
        chain = ExactFlowChain(y, struct, W, priors, init, 100, rng, LO, HI)
        chain.th_dirs, chain.th_widths = _theta_dirs(k), np.full(k, 0.5)
        chain.ab_dirs, chain.ab_widths = _AB_DIRS, _AB_WIDTHS
        chain.ab_shear = _shear(k)
        chain.sweep(tuning=False)
        d = chain.draw()
        diffs.append(
            [d["rho_d"] - rho_d, d["rho_o"] - rho_o, np.log(d["alpha"] / alpha)]
            + list(d["beta"] - beta)
        )
    z = _z(np.asarray(diffs))
    assert np.all(np.abs(z) < 4.0), z


def _jax_sweep_z(T, reps, seed, prior_scale=1.0):
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
    )

    ensure_x64()
    rng = np.random.default_rng(seed)
    X, struct, priors = _case(T, rng)
    k = X.shape[1]
    static, base = _exact_parts(
        np.ones(N * N * T), sp.csc_matrix(W3), struct, priors, 100
    )
    base["prec0"] = base["prec0"] * prior_scale
    sweep = _exact_fns(*static)
    pad = 128  # the padded distinct counts, one length for every draw
    in_data = {key: None for key in base}
    in_data.update(y=0, y_vals=0, y_counts=0)
    in_state = dict.fromkeys(("theta", "a", "b", "alpha"), 0)
    in_state.update(
        dict.fromkeys(
            (
                "it",
                "th_dirs",
                "th_w",
                "ab_dirs",
                "ab_w",
                "ab_shear",
                "w_cnt",
                "w_mean",
                "w_M2",
            )
        )
    )
    run = jax.jit(
        jax.vmap(
            lambda s, key, d: sweep(s, key, jnp.bool_(False), d)[1][:4],
            in_axes=(in_state, 0, in_data),
        )
    )
    R = 10_000
    diffs = []
    for chunk in range(reps // R):
        draws = [_joint_draw(rng, X, T, priors) for _ in range(R)]
        rd, ro, beta, alpha = (np.array([d[i] for d in draws]) for i in range(4))
        y = np.stack([d[4] for d in draws])
        tabs = [np.unique(v, return_counts=True) for v in y]
        data = {
            **base,
            "y": jnp.asarray(y),
            "y_vals": jnp.asarray([np.pad(v, (0, pad - v.size)) for v, _ in tabs]),
            "y_counts": jnp.asarray(
                [np.pad(c.astype(float), (0, pad - c.size)) for _, c in tabs]
            ),
        }
        a, b = -np.log1p(-rd), -np.log1p(-ro)
        state = {
            "theta": jnp.asarray(beta * np.exp(a + b)[:, None]),
            "a": jnp.asarray(a),
            "b": jnp.asarray(b),
            "alpha": jnp.asarray(alpha),
            "it": jnp.int32(1000),
            "th_dirs": jnp.asarray(_theta_dirs(k)),
            "th_w": jnp.full(k, 0.5),
            "ab_dirs": jnp.asarray(_AB_DIRS),
            "ab_w": jnp.asarray(_AB_WIDTHS),
            "ab_shear": jnp.asarray(_shear(k)),
            "w_cnt": jnp.int32(0),
            "w_mean": jnp.zeros(k + 2),
            "w_M2": jnp.zeros((k + 2, k + 2)),
        }
        keys = jax.random.split(jax.random.PRNGKey(seed * 100 + chunk), R)
        rd1, ro1, b1, al1 = (np.asarray(v) for v in run(state, keys, data))
        diffs.append(
            np.column_stack([rd1 - rd, ro1 - ro, np.log(al1 / alpha), b1 - beta])
        )
    return _z(np.concatenate(diffs))


@pytest.mark.slow
@pytest.mark.requires_jax
@pytest.mark.parametrize("T", [1, 2])
def test_jax_sweep_is_invariant(T):
    z = _jax_sweep_z(T, 200_000, seed=30 + T)
    assert np.all(np.abs(z) < 4.0), z


@pytest.mark.slow
@pytest.mark.requires_jax
def test_jax_invariance_check_has_power():
    """The same check fails when the sampler's prior on β is not the draws'."""
    z = _jax_sweep_z(1, 100_000, seed=33, prior_scale=0.5)
    assert np.max(np.abs(z)) > 10.0, z
