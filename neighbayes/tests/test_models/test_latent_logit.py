"""The latent SAR logit (``SARLogit(latent=True)``), SEM logit and SDEM logit.

The sampler is exact, so at ``n = 3`` its posterior must match the exact one,
computed by product Gauss–Legendre quadrature over the latent utilities (the
esda_triad harness); at ``n = 900`` it recovers the DGP and concentrates on
zero dependence when there is none, and the logit's decision tree names the
form that generated the data.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit

from neighbayes import dgp
from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite
from neighbayes.models import Logit, SARLogit, SDEMLogit, SEMLogit

pytest.importorskip("augax")

QUICK = dict(draws=40, tune=40, chains=1, progressbar=False, random_seed=0, n_jobs=1)


@pytest.fixture(scope="module")
def sar_fit():
    d = dgp.simulate_sar_logit(n_side=8, rho=0.4, seed=1, latent=True)
    m = SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sem_fit():
    d = dgp.simulate_sem_logit(n_side=8, lam=0.4, seed=2)
    m = SEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sdem_fit():
    d = dgp.simulate_sdem_logit(n_side=8, lam=0.4, seed=3)
    m = SDEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


def test_designs():
    d = dgp.simulate_sar_logit(n_side=4, seed=0)
    latent = SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    assert latent._design == "latent"
    assert latent._gibbs_key == ("binary_latent", "cross_section")
    assert SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"])._design == "static"
    sem = SEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"])
    assert sem._gibbs_key == ("binary_latent", "cross_section")
    with pytest.raises(NotImplementedError, match="latent SEM already"):
        SEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    sdem = SDEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"])
    assert sdem._gibbs_key == ("binary_latent", "cross_section")
    assert sdem._latent_design[0].shape == (16, 3)
    with pytest.raises(NotImplementedError, match="latent SDEM already"):
        SDEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    with pytest.raises(ValueError, match="latent must be True or False"):
        SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent="yes")
    with pytest.raises(NotImplementedError, match="a panel"):
        SEMLogit(
            y=np.r_[d["y"], d["y"]], X=np.r_[d["X"], d["X"]], W=d["W_sparse"], N=16, T=2
        )


def test_simulators_draw_the_latent_forms():
    d = dgp.simulate_sar_logit(n_side=5, seed=3, latent=True)
    assert set(np.unique(d["y"])) <= {0.0, 1.0}
    np.testing.assert_array_equal(d["y"], (d["eta_true"] > 0).astype(float))
    e = dgp.simulate_sem_logit(n_side=5, seed=3)
    np.testing.assert_array_equal(e["y"], (e["eta_true"] > 0).astype(float))
    assert e["params_true"]["lam"] == 0.35
    s = dgp.simulate_sdem_logit(n_side=5, seed=3)
    np.testing.assert_array_equal(s["y"], (s["eta_true"] > 0).astype(float))
    np.testing.assert_array_equal(s["params_true"]["theta"], [0.5])


@pytest.mark.parametrize(
    "fit, name, k",
    [("sar_fit", "rho", 2), ("sem_fit", "lam", 2), ("sdem_fit", "lam", 3)],
)
def test_fit_summaries(request, fit, name, k):
    m = request.getfixturevalue(fit)
    post = m.inference_data.posterior
    assert {name, "beta"} <= set(post.data_vars)
    assert post["beta"].shape[-1] == k
    assert "log_likelihood" not in m.inference_data.children
    p = m.fitted_probabilities()
    assert p.shape == (64,) and np.all((p >= 0) & (p <= 1))
    assert m.impacts().shape[0] == 1


def test_sem_impacts_have_no_spillover(sem_fit):
    df = sem_fit.impacts()
    assert float(df["indirect"].iloc[0]) == 0.0


def test_sdem_impacts_are_the_durbin_coefficients(sdem_fit):
    """Direct ``beta``, indirect ``theta`` (row-standardized ``W``)."""
    df = sdem_fit.impacts()
    beta = sdem_fit.inference_data.posterior["beta"].values.reshape(-1, 3)
    assert float(df["direct"].iloc[0]) == pytest.approx(beta[:, 1].mean())
    assert float(df["indirect"].iloc[0]) == pytest.approx(beta[:, 2].mean())


def test_refusals(sar_fit):
    with pytest.raises(NotImplementedError, match="orthant probability"):
        sar_fit.fit(idata_kwargs={"log_likelihood": True}, **QUICK)
    with pytest.raises(NotImplementedError, match="Gibbs sampling"):
        sar_fit.fit(sampler="nuts", draws=5, tune=5, chains=1, progressbar=False)
    with pytest.raises(NotImplementedError, match="latent index"):
        sar_fit.impacts(scale="probability")


@pytest.mark.requires_jax
@pytest.mark.parametrize(
    "cls, kw", [(SARLogit, {"latent": True}), (SEMLogit, {}), (SDEMLogit, {})]
)
def test_jax_backend_runs(cls, kw):
    d = dgp.simulate_sar_logit(n_side=8, rho=0.4, seed=4, latent=True)
    post = (
        cls(y=d["y"], X=d["X"], W=d["W_sparse"], **kw)
        .fit(backend="jax", **QUICK)
        .posterior
    )
    assert np.all(np.isfinite(post["beta"].values))


def test_colour_classes_are_conditionally_independent():
    """No two utilities of one colour share a nonzero of Q = A'PA."""
    from neighbayes.samplers.logit_latent._jax import colour_classes

    W = dgp.simulate_sar_logit(n_side=9, seed=0)["W_sparse"]
    colour = colour_classes(W)
    B = (sp.identity(W.shape[0]) + (W != 0)).astype(bool)
    Q = (B.T @ B).tocoo()
    off = Q.row != Q.col
    assert np.all(colour[Q.row[off]] != colour[Q.col[off]])


@pytest.mark.requires_jax
@pytest.mark.parametrize("a", [-3.0, 0.0, 2.0, 8.0, 29.0, 35.0, 60.0])
def test_truncated_normal_draws(a):
    """Inversion below the switch and rejection above it match the truncated normal."""
    import jax
    import jax.numpy as jnp

    from neighbayes.samplers.logit_latent import _jax as J

    jax.config.update("jax_enable_x64", True)
    stats = pytest.importorskip("scipy.stats")
    n = 200_000
    av = jnp.full(n, a)
    if a <= J._TN_INVERT_MAX:
        z = J._tn_invert(av, jax.random.uniform(jax.random.key(0), (n,)))
    else:
        z = J._tn_beyond(jax.random.key(0), av, jnp.ones(n, bool))
    z = np.asarray(z)
    ref = stats.truncnorm(a, np.inf)
    assert np.all(z > a)
    assert abs(z.mean() - ref.mean()) < 5 * ref.std() / np.sqrt(n)
    assert z.std() == pytest.approx(ref.std(), rel=0.02)


def test_no_suite_for_the_latent_designs(sar_fit, sem_fit, sdem_fit):
    assert get_diagnostic_suite(sar_fit) is None
    assert get_diagnostic_suite(sem_fit) is None
    assert get_diagnostic_suite(sdem_fit) is None


# ---------------------------------------------------------------------------
# Exactness at n = 3
# ---------------------------------------------------------------------------

W3 = np.array([[0, 1, 1], [1, 0, 0], [2, 1, 0]], float)
W3 /= W3.sum(1, keepdims=True)
X3 = np.array([0.8, -1.1, 0.4])
Y3 = np.array([1.0, 0.0, 1.0])
BOUND = 0.9


def _exact_moments(process, n_gl=40, n_beta=61, n_par=81):
    """Posterior mean and sd of ``beta`` and the spatial parameter, exactly.

    The likelihood integrates the utilities' density over the orthant of
    ``y`` by product Gauss–Legendre quadrature, ``y* = s t / (1 - t)``; the
    posterior is a trapezoid sum on a ``(beta, par)`` grid, under the
    ``N(0, 1)`` and ``U(-0.9, 0.9)`` priors the fits use.
    """
    t, wt = np.polynomial.legendre.leggauss(n_gl)
    t, wt = (t + 1) / 2, wt / 2
    T = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    WT = np.prod(np.stack(np.meshgrid(wt, wt, wt, indexing="ij"), -1).reshape(-1, 3), 1)
    ystar = (2 * Y3 - 1) * T / (1 - T)
    weight = WT * np.prod(1 / (1 - T) ** 2, 1)
    bg = np.linspace(-4, 4, n_beta)
    pg = np.linspace(-BOUND, BOUND, n_par)
    lp = np.empty((n_beta, n_par))
    for j, par in enumerate(pg):
        A = np.eye(3) - par * W3
        logdet = np.log(abs(np.linalg.det(A)))
        for i, b in enumerate(bg):
            if process == "sar":
                eps = ystar @ A.T - X3 * b
            else:
                eps = (ystar - X3 * b) @ A.T
            dens = np.sum(weight * np.prod(expit(eps) * expit(-eps), 1))
            lp[i, j] = np.log(dens) + logdet - 0.5 * b * b
    trap = lambda g: np.r_[0.5, np.ones(g.size - 2), 0.5]  # noqa: E731
    post = np.exp(lp - lp.max()) * trap(bg)[:, None] * trap(pg)[None, :]
    post /= post.sum()
    out = {}
    for name, grid, marg in (("beta", bg, post.sum(1)), ("par", pg, post.sum(0))):
        mean = marg @ grid
        out[name] = (mean, np.sqrt(marg @ grid**2 - mean**2))
    return out


_BACKENDS = ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]


@pytest.mark.slow
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("process", ["sar", "sem"])
def test_posterior_is_exact_at_n3(process, backend):
    import arviz as az

    exact = _exact_moments(process)
    priors = {"beta_mu": 0.0, "beta_sigma": 1.0}
    W = sp.csr_matrix(W3)
    if process == "sar":
        priors.update(rho_lower=-BOUND, rho_upper=BOUND)
        model, name = (
            SARLogit(y=Y3, X=X3[:, None], W=W, latent=True, priors=priors),
            "rho",
        )
    else:
        priors.update(lam_lower=-BOUND, lam_upper=BOUND)
        model, name = SEMLogit(y=Y3, X=X3[:, None], W=W, priors=priors), "lam"
    post = model.fit(
        draws=10_000,
        tune=1000,
        chains=4,
        progressbar=False,
        random_seed=1,
        n_jobs=4,
        backend=backend,
    ).posterior
    for key, draws in (
        ("beta", post["beta"].values[..., 0]),
        ("par", post[name].values),
    ):
        mean, sd = exact[key]
        mcse = float(az.mcse(az.convert_to_dataset({key: draws}))[key])
        assert abs(draws.mean() - mean) < 4 * mcse, (key, draws.mean(), mean, mcse)
        assert draws.std() == pytest.approx(sd, rel=0.03), key


# ---------------------------------------------------------------------------
# Recovery at n = 900
# ---------------------------------------------------------------------------

BETA = np.array([0.3, 1.0])


@pytest.mark.slow
@pytest.mark.parametrize(
    "process, truth",
    [("sar", 0.4), ("sem", 0.4), ("sdem", 0.4), ("sar", 0.0)],
    ids=str,
)
def test_recovery(process, truth):
    coef = BETA
    if process == "sar":
        d = dgp.simulate_sar_logit(
            n_side=30, rho=truth, beta=BETA, seed=101, latent=True
        )
        model, name = SARLogit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True), "rho"
    elif process == "sem":
        d = dgp.simulate_sem_logit(n_side=30, lam=truth, beta=BETA, seed=205)
        model, name = SEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
    else:
        d = dgp.simulate_sdem_logit(n_side=30, lam=truth, beta=BETA, seed=307)
        model, name = SDEMLogit(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
        coef = np.r_[BETA, d["params_true"]["theta"]]
    post = model.fit(
        draws=1500, tune=750, chains=2, progressbar=False, random_seed=0, n_jobs=2
    ).posterior
    par = post[name].values
    assert abs(par.mean() - truth) < 3 * par.std()
    beta = post["beta"].values.reshape(-1, coef.size)
    assert np.all(np.abs(beta.mean(0) - coef) < 3 * beta.std(0))


# ---------------------------------------------------------------------------
# The decision tree, from fitted logits
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "simulate, leaf",
    [
        (
            lambda: dgp.simulate_sar_logit(n_side=30, rho=0.0, beta=BETA, seed=10),
            "Logit",
        ),
        (
            lambda: dgp.simulate_sar_logit(n_side=30, rho=0.6, beta=BETA, seed=10),
            "SARLogit",
        ),
        (
            lambda: dgp.simulate_sem_logit(n_side=30, lam=0.6, beta=BETA, seed=10),
            "SEMLogit",
        ),
        (
            lambda: dgp.simulate_sdem_logit(
                n_side=30, lam=0.6, beta=BETA, theta=np.array([-1.0]), seed=10
            ),
            "SDEMLogit",
        ),
    ],
    ids=["null", "reduced SAR", "latent SEM", "SDEM"],
)
def test_decision_tree_names_the_generating_form(simulate, leaf):
    d = simulate()
    m = Logit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(draws=500, tune=500, chains=2, progressbar=False, random_seed=0, cores=1)
    assert m.spatial_diagnostics_decision(format="model") == leaf
