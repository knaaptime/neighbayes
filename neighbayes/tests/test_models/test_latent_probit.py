"""The SAR probit (latent, and ``latent=False``), the SEM probit and the SDEM probit.

The samplers are exact, so at ``n = 3`` their posteriors must match the exact
ones: the latent models' likelihood is a trivariate normal orthant
probability, the reduced form's a product of normal distribution functions.
At ``n = 900`` they recover the DGP and concentrate on zero dependence when
there is none.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal

from neighbayes import dgp
from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite
from neighbayes.models import Probit, SARProbit, SDEMProbit, SEMProbit

QUICK = dict(draws=40, tune=40, chains=1, progressbar=False, random_seed=0, n_jobs=1)


@pytest.fixture(scope="module")
def sar_fit():
    d = dgp.simulate_sar_probit(n_side=8, rho=0.4, seed=1)
    m = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def reduced_fit():
    d = dgp.simulate_sar_probit(n_side=8, rho=0.4, seed=1, latent=False)
    m = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=False)
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sem_fit():
    d = dgp.simulate_sem_probit(n_side=8, lam=0.4, seed=2)
    m = SEMProbit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sdem_fit():
    d = dgp.simulate_sdem_probit(n_side=8, lam=0.4, seed=3)
    m = SDEMProbit(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


def test_designs():
    d = dgp.simulate_sar_probit(n_side=4, seed=0)
    latent = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"])
    assert latent._design == "latent"
    assert latent._gibbs_key == ("binary_latent", "cross_section")
    assert latent._link == "probit"
    reduced = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=False)
    assert reduced._design == "static"
    assert reduced._gibbs_key == ("binary", "cross_section")
    for cls in (SEMProbit, SDEMProbit):
        m = cls(y=d["y"], X=d["X"], W=d["W_sparse"])
        assert m._gibbs_key == ("binary_latent", "cross_section")
        name = cls.__name__.removesuffix("Probit")
        with pytest.raises(NotImplementedError, match=f"latent {name} already"):
            cls(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    assert SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)._design == (
        "latent"
    )
    assert Probit(y=d["y"], X=d["X"], W=d["W_sparse"])._link == "probit"


def test_simulators_draw_the_latent_forms():
    d = dgp.simulate_sar_probit(n_side=5, seed=3)
    np.testing.assert_array_equal(d["y"], (d["eta_true"] > 0).astype(float))
    r = dgp.simulate_sar_probit(n_side=5, seed=3, latent=False)
    assert set(np.unique(r["y"])) <= {0.0, 1.0}
    e = dgp.simulate_sem_probit(n_side=5, seed=3)
    np.testing.assert_array_equal(e["y"], (e["eta_true"] > 0).astype(float))
    s = dgp.simulate_sdem_probit(n_side=5, seed=3)
    np.testing.assert_array_equal(s["y"], (s["eta_true"] > 0).astype(float))
    np.testing.assert_array_equal(s["params_true"]["theta"], [0.5])
    a = dgp.simulate_probit(n_side=5, seed=3)
    assert a["params_true"]["rho"] == 0.0
    with pytest.raises(ValueError, match="reduced form has none"):
        dgp.simulate_sar_probit(n_side=5, latent=False, err_hetero=True)


@pytest.mark.parametrize(
    "fit, name, k",
    [
        ("sar_fit", "rho", 2),
        ("reduced_fit", "rho", 2),
        ("sem_fit", "lam", 2),
        ("sdem_fit", "lam", 3),
    ],
)
def test_fit_summaries(request, fit, name, k):
    m = request.getfixturevalue(fit)
    post = m.inference_data.posterior
    assert {name, "beta"} <= set(post.data_vars)
    assert post["beta"].shape[-1] == k
    p = m.fitted_probabilities()
    assert p.shape == (64,) and np.all((p >= 0) & (p <= 1))
    for scale in ("index", "probability"):
        df = m.impacts(scale=scale)
        assert df.shape[0] == 1 and df.attrs["scale"] == scale
        assert np.isfinite(df.to_numpy(float)).all()


def test_sem_impacts_have_no_spillover(sem_fit):
    assert float(sem_fit.impacts()["indirect"].iloc[0]) == 0.0


def test_refusals(sar_fit):
    with pytest.raises(NotImplementedError, match="Gibbs sampling"):
        sar_fit.fit(sampler="nuts", draws=5, tune=5, chains=1, progressbar=False)
    with pytest.raises(ValueError, match="scale must be"):
        sar_fit.impacts(scale="logodds")


def test_suites(sar_fit, reduced_fit, sem_fit):
    assert get_diagnostic_suite(sar_fit) is None
    assert get_diagnostic_suite(sem_fit) is None
    assert get_diagnostic_suite(reduced_fit).name == "SARProbit"


@pytest.mark.requires_jax
@pytest.mark.parametrize(
    "cls, kw",
    [
        (SARProbit, {}),
        (SARProbit, {"latent": False}),
        (SEMProbit, {}),
        (SDEMProbit, {}),
    ],
)
def test_jax_backend_runs(cls, kw):
    d = dgp.simulate_sar_probit(n_side=8, rho=0.4, seed=4)
    post = (
        cls(y=d["y"], X=d["X"], W=d["W_sparse"], **kw)
        .fit(backend="jax", **QUICK)
        .posterior
    )
    assert np.all(np.isfinite(post["beta"].values))


def test_probit_does_not_need_augax(monkeypatch):
    """The probit's scales are fixed at one: no Kolmogorov draw."""
    import builtins

    real = builtins.__import__

    def no_augax(name, *args, **kwargs):
        if name == "augax":
            raise ImportError("augax blocked")
        return real(name, *args, **kwargs)

    d = dgp.simulate_sar_probit(n_side=5, seed=5)
    monkeypatch.setattr(builtins, "__import__", no_augax)
    post = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"]).fit(**QUICK).posterior
    assert np.all(np.isfinite(post["rho"].values))


# ---------------------------------------------------------------------------
# Exactness at n = 3
# ---------------------------------------------------------------------------

W3 = np.array([[0, 1, 1], [1, 0, 0], [2, 1, 0]], float)
W3 /= W3.sum(1, keepdims=True)
X3 = np.array([0.8, -1.1, 0.4])
Y3 = np.array([1.0, 0.0, 1.0])
BOUND = 0.9


def _exact_moments(process, n_beta=61, n_par=81):
    """Posterior mean and sd of ``beta`` and the spatial parameter, exactly.

    The latent models' likelihood is the probability that the normal
    utilities fall in the orthant of ``y``; the reduced form's is
    :math:`\\prod_i \\Phi((2y_i - 1)\\eta_i)`.  The posterior is a trapezoid
    sum on a ``(beta, par)`` grid, under the ``N(0, 1)`` and ``U(-0.9, 0.9)``
    priors the fits use.
    """
    D = np.diag(2 * Y3 - 1)
    bg = np.linspace(-4, 4, n_beta)
    pg = np.linspace(-BOUND, BOUND, n_par)
    lp = np.empty((n_beta, n_par))
    for j, par in enumerate(pg):
        S = np.linalg.inv(np.eye(3) - par * W3)
        cov = D @ S @ S.T @ D
        for i, b in enumerate(bg):
            if process == "reduced":
                lp[i, j] = log_ndtr((2 * Y3 - 1) * (S @ X3 * b)).sum()
            else:
                mean = S @ X3 * b if process == "sar" else X3 * b
                # P(D y* > 0) = P(-D y* < 0), -D y* ~ N(-D mean, D S S' D).
                prob = multivariate_normal(-D @ mean, cov).cdf(np.zeros(3))
                lp[i, j] = np.log(prob)
            lp[i, j] -= 0.5 * b * b
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
@pytest.mark.parametrize("process", ["sar", "sem", "reduced"])
def test_posterior_is_exact_at_n3(process, backend):
    import arviz as az

    exact = _exact_moments(process)
    priors = {"beta_mu": 0.0, "beta_sigma": 1.0}
    W = sp.csr_matrix(W3)
    if process == "sem":
        priors.update(lam_lower=-BOUND, lam_upper=BOUND)
        model, name = SEMProbit(y=Y3, X=X3[:, None], W=W, priors=priors), "lam"
    else:
        priors.update(rho_lower=-BOUND, rho_upper=BOUND)
        model = SARProbit(
            y=Y3, X=X3[:, None], W=W, priors=priors, latent=process == "sar"
        )
        name = "rho"
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
    [("sar", 0.4), ("reduced", 0.4), ("sem", 0.4), ("sdem", 0.4), ("sar", 0.0)],
    ids=str,
)
def test_recovery(process, truth):
    coef = BETA
    if process in ("sar", "reduced"):
        latent = process == "sar"
        d = dgp.simulate_sar_probit(
            n_side=30, rho=truth, beta=BETA, seed=101, latent=latent
        )
        model = SARProbit(y=d["y"], X=d["X"], W=d["W_sparse"], latent=latent)
        name = "rho"
    elif process == "sem":
        d = dgp.simulate_sem_probit(n_side=30, lam=truth, beta=BETA, seed=205)
        model, name = SEMProbit(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
    else:
        d = dgp.simulate_sdem_probit(n_side=30, lam=truth, beta=BETA, seed=307)
        model, name = SDEMProbit(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
        coef = np.r_[BETA, d["params_true"]["theta"]]
    post = model.fit(
        draws=1500, tune=750, chains=2, progressbar=False, random_seed=0, n_jobs=2
    ).posterior
    par = post[name].values
    assert abs(par.mean() - truth) < 3 * par.std()
    beta = post["beta"].values.reshape(-1, coef.size)
    assert np.all(np.abs(beta.mean(0) - coef) < 3 * beta.std(0))
