"""The latent SAR (``SARNegBin(latent=True)``), SEM and SDEM negative binomial.

The sampler is exact, so at ``n = 3`` its posterior must match the exact one,
computed by product Gauss–Legendre quadrature over the inverse CDF of the
gamma disturbances; at ``n = 900`` it recovers the DGP and concentrates on zero
dependence when there is none.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import gammaincinv

from neighbayes import dgp
from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite
from neighbayes.models import SARNegBin, SDEMNegBin, SEMNegBin

QUICK = dict(draws=40, tune=40, chains=1, progressbar=False, random_seed=0, n_jobs=1)


@pytest.fixture(scope="module")
def sar_fit():
    d = dgp.simulate_sar_negbin(n_side=8, rho=0.4, seed=1, latent=True)
    m = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sem_fit():
    d = dgp.simulate_sem_negbin(n_side=8, lam=0.4, seed=2)
    m = SEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


@pytest.fixture(scope="module")
def sdem_fit():
    d = dgp.simulate_sdem_negbin(n_side=8, lam=0.4, seed=3)
    m = SDEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(**QUICK)
    return m


def test_designs():
    d = dgp.simulate_sar_negbin(n_side=4, seed=0)
    latent = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    assert latent._design == "latent"
    assert latent._gibbs_key == ("count_latent", "cross_section")
    assert SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])._design == "static"
    for cls, name in ((SEMNegBin, "SEM"), (SDEMNegBin, "SDEM")):
        m = cls(y=d["y"], X=d["X"], W=d["W_sparse"])
        assert m._gibbs_key == ("count_latent", "cross_section")
        with pytest.raises(NotImplementedError, match=f"latent {name} already"):
            cls(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
    assert SDEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])._latent_design[0].shape == (
        16,
        3,
    )
    with pytest.raises(ValueError, match="latent must be True or False"):
        SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], latent="yes")
    panel = dict(y=np.r_[d["y"], d["y"]], X=np.r_[d["X"], d["X"]], W=d["W_sparse"])
    with pytest.raises(NotImplementedError, match="cross-section only"):
        SARNegBin(**panel, N=16, T=2, latent=True)
    with pytest.raises(NotImplementedError, match="a panel"):
        SEMNegBin(**panel, N=16, T=2)


def test_simulators_draw_the_latent_forms():
    d = dgp.simulate_sar_negbin(n_side=5, seed=3, latent=True)
    np.testing.assert_allclose(d["mu"], np.exp(d["eta_true"]))
    assert d["params_true"]["rho"] == 0.5
    e = dgp.simulate_sem_negbin(n_side=5, seed=3)
    assert e["params_true"]["lam"] == 0.5
    s = dgp.simulate_sdem_negbin(n_side=5, seed=3)
    np.testing.assert_array_equal(s["params_true"]["theta"], [0.5])
    with pytest.raises(NotImplementedError, match="latent=True"):
        dgp.simulate_sar_negbin(n_side=4, T=2, latent=True)
    with pytest.raises(NotImplementedError, match="as a panel"):
        dgp.simulate_sem_negbin(n_side=4, T=2)


@pytest.mark.parametrize(
    "fit, name, k",
    [("sar_fit", "rho", 2), ("sem_fit", "lam", 2), ("sdem_fit", "lam", 3)],
)
def test_fit_summaries(request, fit, name, k):
    m = request.getfixturevalue(fit)
    post = m.inference_data.posterior
    assert {name, "beta", "alpha"} <= set(post.data_vars)
    assert post["beta"].shape[-1] == k
    assert "log_likelihood" not in m.inference_data.children
    mu = m.fitted_values()
    assert mu.shape == (64,) and np.all(mu > 0)
    assert m.impacts().shape[0] == 1


def test_sem_impacts_have_no_spillover(sem_fit):
    assert float(sem_fit.impacts()["indirect"].iloc[0]) == 0.0


def test_sdem_impacts_are_the_durbin_coefficients(sdem_fit):
    df = sdem_fit.impacts()
    beta = sdem_fit.inference_data.posterior["beta"].values.reshape(-1, 3)
    assert float(df["direct"].iloc[0]) == pytest.approx(beta[:, 1].mean())
    assert float(df["indirect"].iloc[0]) == pytest.approx(beta[:, 2].mean())


def test_refusals(sar_fit):
    with pytest.raises(NotImplementedError, match="no closed form"):
        sar_fit.fit(idata_kwargs={"log_likelihood": True}, **QUICK)
    with pytest.raises(NotImplementedError, match="Gibbs sampling"):
        sar_fit.fit(sampler="nuts", draws=5, tune=5, chains=1, progressbar=False)
    with pytest.raises(NotImplementedError, match="log-mean"):
        sar_fit.impacts(scale="count")


def test_alpha_fixed_is_held():
    d = dgp.simulate_sem_negbin(n_side=6, seed=4)
    m = SEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], priors={"alpha_fixed": 3.0})
    post = m.fit(**QUICK).posterior
    np.testing.assert_array_equal(post["alpha"].values, 3.0)


@pytest.mark.requires_jax
@pytest.mark.parametrize(
    "cls, kw", [(SARNegBin, {"latent": True}), (SEMNegBin, {}), (SDEMNegBin, {})]
)
def test_jax_backend_runs(cls, kw):
    d = dgp.simulate_sar_negbin(n_side=8, rho=0.4, seed=4, latent=True)
    post = (
        cls(y=d["y"], X=d["X"], W=d["W_sparse"], **kw)
        .fit(backend="jax", **QUICK)
        .posterior
    )
    assert np.all(np.isfinite(post["beta"].values))
    assert np.all(post["alpha"].values > 0)


def test_no_suite_for_the_latent_designs(sar_fit, sem_fit, sdem_fit):
    for m in (sar_fit, sem_fit, sdem_fit):
        assert get_diagnostic_suite(m) is None


# ---------------------------------------------------------------------------
# Exactness at n = 3
# ---------------------------------------------------------------------------

W3 = np.array([[0, 1, 1], [1, 0, 0], [2, 1, 0]], float)
W3 /= W3.sum(1, keepdims=True)
X3 = np.array([0.8, -1.1, 0.4])
Y3 = np.array([3.0, 0.0, 1.0])
BOUND = 0.9
#: A near half-normal(1) prior on alpha, so that the alpha grid is finite.
ALPHA_PRIOR = {"alpha_nu": 30.0, "alpha_sigma": 1.0}


def _exact_moments(process, n_q=16, n_beta=51, n_par=37, n_la=36):
    """Posterior mean and sd of ``beta``, the spatial parameter and ``alpha``.

    The likelihood integrates the Poisson counts over the gamma disturbances
    by product Gauss–Legendre quadrature on their inverse CDF; the posterior
    is a trapezoid sum on a ``(beta, par, log alpha)`` grid, under the
    ``N(0, 1)``, ``U(-0.9, 0.9)`` and :data:`ALPHA_PRIOR` priors the fits use.
    """
    nu, sig = ALPHA_PRIOR["alpha_nu"], ALPHA_PRIOR["alpha_sigma"]
    t, wt = np.polynomial.legendre.leggauss(n_q)
    t, wt = (t + 1) / 2, wt / 2
    U = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)
    lw = np.log(np.prod(np.stack(np.meshgrid(wt, wt, wt, indexing="ij"), -1), -1))
    lw = lw.reshape(-1)
    bg = np.linspace(-3.5, 4.5, n_beta)
    pg = np.linspace(-BOUND, BOUND, n_par)
    lag = np.linspace(-4.0, 1.8, n_la)
    lp = np.empty((n_beta, n_par, n_la))
    for kk, la in enumerate(lag):
        a = np.exp(la)
        eps = np.log(gammaincinv(a, U) / a)
        prior = -0.5 * (nu + 1) * np.log1p(a * a / (nu * sig * sig)) + la
        for j, par in enumerate(pg):
            S = np.linalg.inv(np.eye(3) - par * W3)
            lat = eps @ S.T
            mean = S @ X3 if process == "sar" else X3
            for i, b in enumerate(bg):
                eta = lat + mean * b
                m = eta @ Y3 - np.exp(eta).sum(1) + lw
                top = m.max()
                lp[i, j, kk] = top + np.log(np.exp(m - top).sum()) - 0.5 * b * b + prior
    trap = lambda g: np.r_[0.5, np.ones(g.size - 2), 0.5]  # noqa: E731
    post = np.exp(lp - lp.max())
    post *= trap(bg)[:, None, None] * trap(pg)[None, :, None] * trap(lag)[None, None]
    post /= post.sum()
    out = {}
    for name, grid, axes in (
        ("beta", bg, (1, 2)),
        ("par", pg, (0, 2)),
        ("alpha", np.exp(lag), (0, 1)),
    ):
        marg = post.sum(axes)
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
    priors = {"beta_mu": 0.0, "beta_sigma": 1.0, **ALPHA_PRIOR}
    W = sp.csr_matrix(W3)
    if process == "sar":
        priors.update(rho_lower=-BOUND, rho_upper=BOUND)
        model = SARNegBin(y=Y3, X=X3[:, None], W=W, latent=True, priors=priors)
        name = "rho"
    else:
        priors.update(lam_lower=-BOUND, lam_upper=BOUND)
        model, name = SEMNegBin(y=Y3, X=X3[:, None], W=W, priors=priors), "lam"
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
        ("alpha", post["alpha"].values),
    ):
        mean, sd = exact[key]
        mcse = float(az.mcse(az.convert_to_dataset({key: draws}))[key])
        assert abs(draws.mean() - mean) < 4 * mcse, (key, draws.mean(), mean, mcse)
        assert draws.std() == pytest.approx(sd, rel=0.03), key


# ---------------------------------------------------------------------------
# Recovery at n = 900
# ---------------------------------------------------------------------------

BETA = np.array([1.0, 0.6])
ALPHA = 2.0


@pytest.mark.slow
@pytest.mark.parametrize(
    "process, truth",
    [("sar", 0.4), ("sem", 0.4), ("sdem", 0.4), ("sar", 0.0)],
    ids=str,
)
def test_recovery(process, truth):
    coef = BETA
    kw = dict(n_side=30, beta=BETA, alpha=ALPHA)
    if process == "sar":
        d = dgp.simulate_sar_negbin(rho=truth, seed=101, latent=True, **kw)
        model = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], latent=True)
        name = "rho"
    elif process == "sem":
        d = dgp.simulate_sem_negbin(lam=truth, seed=205, **kw)
        model, name = SEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
    else:
        d = dgp.simulate_sdem_negbin(lam=truth, seed=307, **kw)
        model, name = SDEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"]), "lam"
        coef = np.r_[BETA, d["params_true"]["theta"]]
    post = model.fit(
        draws=1500, tune=750, chains=2, progressbar=False, random_seed=0, n_jobs=2
    ).posterior
    for draws, true in ((post[name].values, truth), (post["alpha"].values, ALPHA)):
        assert abs(draws.mean() - true) < 3 * draws.std()
    beta = post["beta"].values.reshape(-1, coef.size)
    assert np.all(np.abs(beta.mean(0) - coef) < 3 * beta.std(0))


# ---------------------------------------------------------------------------
# The decision tree, from fitted negative binomials
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "simulate, leaf",
    [
        (
            lambda: dgp.simulate_sar_negbin(n_side=30, rho=0.0, beta=BETA, seed=10),
            "NegBin",
        ),
        (
            lambda: dgp.simulate_sar_negbin(n_side=30, rho=0.5, beta=BETA, seed=10),
            "SARNegBin",
        ),
        (
            lambda: dgp.simulate_sar_negbin(
                n_side=30, rho=0.5, beta=BETA, seed=10, latent=True
            ),
            "SARNegBin(latent=True)",
        ),
        (
            lambda: dgp.simulate_sem_negbin(n_side=30, lam=0.5, beta=BETA, seed=10),
            "SEMNegBin",
        ),
        (
            lambda: dgp.simulate_sdem_negbin(
                n_side=30, lam=0.5, beta=BETA, theta=np.array([-0.6]), seed=10
            ),
            "SDEMNegBin",
        ),
    ],
    ids=["null", "reduced SAR", "latent SAR", "latent SEM", "SDEM"],
)
def test_decision_tree_names_the_generating_form(simulate, leaf):
    from neighbayes.models import NegBin

    d = simulate()
    m = NegBin(y=d["y"], X=d["X"], W=d["W_sparse"])
    m.fit(draws=500, tune=500, chains=2, progressbar=False, random_seed=0, cores=1)
    assert m.spatial_diagnostics_decision(format="model") == leaf
