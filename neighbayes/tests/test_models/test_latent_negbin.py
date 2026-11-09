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
    assert SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"])._design == "latent"
    reduced = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], latent=False)
    assert reduced._design == "static"
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
    m = SARNegBin(**panel, N=16, T=2, latent=True, unit_effects="random")
    assert (m._design, m._is_panel, m._latent_units, m._latent_periods) == (
        "latent",
        True,
        True,
        False,
    )
    with pytest.raises(NotImplementedError, match="random unit effects, not fixed"):
        SARNegBin(**panel, N=16, T=2, latent=True, unit_effects="fixed")
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
    p = dgp.simulate_sar_negbin(n_side=4, T=3, latent=True, rho=0.5, seed=3)
    assert p["y"].shape == (48,) and p["params_true"]["rho"] == 0.5
    with pytest.raises(NotImplementedError, match="as a panel"):
        dgp.simulate_sem_negbin(n_side=4, T=2)


def test_small_alpha_disturbances_do_not_underflow():
    # log(rng.gamma(alpha, 1 / alpha)) is -inf for 2.4% of draws at alpha = 0.005
    from scipy.special import digamma, polygamma

    from neighbayes.samplers._utils._latent_loglik import log_gamma_noise

    alpha = 0.005
    e = log_gamma_noise(np.random.default_rng(0), alpha, 400_000)
    assert np.isfinite(e).all()
    sd = np.sqrt(polygamma(1, alpha) / e.size)
    assert abs(e.mean() - (digamma(alpha) - np.log(alpha))) < 5 * sd
    # A -inf disturbance made the filter solve raise "singular or too
    # ill-conditioned"; 144 units draw at least one with probability 0.97.
    assert np.isfinite(
        dgp.simulate_sar_negbin(n_side=12, alpha=alpha, seed=0)["eta_true"]
    ).all()
    assert np.isfinite(
        dgp.simulate_sar_negbin(n_side=12, T=3, alpha=alpha, seed=0)["mu"]
    ).all()


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
    with pytest.raises(NotImplementedError, match="Gibbs sampling"):
        sar_fit.fit(sampler="nuts", draws=5, tune=5, chains=1, progressbar=False)


def test_alpha_fixed_is_held():
    d = dgp.simulate_sem_negbin(n_side=6, seed=4)
    m = SEMNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], priors={"alpha_fixed": 3.0})
    post = m.fit(**QUICK).posterior
    np.testing.assert_array_equal(post["alpha"].values, 3.0)


@pytest.mark.requires_jax
def test_jax_log_mean_step_reaches_a_flat_zero_conditional():
    """One JAX log-mean step from exact draws of a deep zero's conditional.

    At alpha = 0.05 the conditional of a zero count's log-mean, its disturbance
    and its four neighbours' near -20, is nearly flat for about 100 units below
    its mode.  The step must leave it invariant and decorrelate: a *t* at the
    mode alone gave corr(x0, x1) = 0.986, and the package fit stalled.
    """
    import jax
    import jax.numpy as jnp

    from neighbayes.samplers._utils._latent_loglik import POISSON
    from neighbayes.samplers.negbin_latent._jax import _eta_mh

    alpha, M = 0.05, 50_000
    Av = np.r_[1.0, np.full(4, -0.1)]
    rest = np.full(5, -20.0)
    t = np.linspace(-1500.0, 40.0, 400_001)
    e = rest + Av * t[:, None]
    logf = -np.exp(t) + alpha * np.sum(e - np.exp(e), axis=1)
    p = np.exp(logf - logf.max())
    x0 = np.interp(np.random.default_rng(0).random(M), np.cumsum(p) / p.sum(), t)
    x1 = np.asarray(
        _eta_mh(
            jnp.asarray(x0),
            jnp.zeros(M),
            jnp.full(M, POISSON, jnp.int8),
            jnp.asarray(rest + Av * x0[:, None]),
            jnp.asarray(np.broadcast_to(Av, (M, 5))),
            alpha,
            jax.random.PRNGKey(1),
            jnp.ones(M, bool),
        )
    )
    for g in (x1 - x0, x1**2 - x0**2):
        assert abs(g.mean()) < 4.0 * g.std() / np.sqrt(M)
    assert np.corrcoef(x0, x1)[0, 1] < 0.85


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


@pytest.mark.validation
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
            lambda: dgp.simulate_sar_negbin(
                n_side=30, rho=0.5, beta=BETA, seed=10, latent=False
            ),
            "SARNegBin(latent=False)",
        ),
        (
            lambda: dgp.simulate_sar_negbin(n_side=30, rho=0.5, beta=BETA, seed=10),
            "SARNegBin",
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


# ---------------------------------------------------------------------------
# The latent panel: unit and period effects inside the filter
# ---------------------------------------------------------------------------

PANEL_KW = dict(latent=True, unit_effects="random", time_effects=True)


def _panel(n_side=5, T=3, seed=1, **kw):
    d = dgp.simulate_sar_negbin(n_side=n_side, T=T, seed=seed, **PANEL_KW, **kw)
    m = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], N=n_side**2, T=T, **PANEL_KW)
    return d, m


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_panel_fit(backend):
    d, m = _panel()
    idata = m.fit(
        **{**QUICK, "chains": 2}, backend=backend, idata_kwargs={"log_likelihood": True}
    )
    post = idata.posterior
    assert {
        "rho",
        "beta",
        "alpha",
        "time_effect",
        "unit_effect",
        "unit_effect_sd",
    } <= set(post.data_vars)
    assert post["beta"].shape[-1] == 2 and post["time_effect"].shape[-1] == 2
    assert post["unit_effect"].shape[-1] == 25
    ll = idata.log_likelihood["obs"].values
    assert ll.shape == (2, 40, 75) and np.all(np.isfinite(ll))
    assert m.fitted_values().shape == (75,)
    for scale in ("logmean", "count"):
        assert np.isfinite(m.impacts(scale=scale).to_numpy(float)).all()


def test_panel_logdet_counts_each_period_once():
    """The sampler multiplies one period's log|A| by T (not the effects' T - 1)."""
    _, m = _panel(T=4)
    ld = np.linalg.slogdet(np.eye(25) - 0.4 * m._W_sparse.toarray())[1]
    assert m._latent_logdet_fn(0.4) == pytest.approx(ld, rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_panel_recovery(backend):
    d, m = _panel(n_side=15, T=5, seed=4, rho=0.5, alpha=3.0, unit_effect_sd=0.5)
    m.fit(
        draws=1000,
        tune=800,
        chains=2,
        progressbar=False,
        random_seed=0,
        backend=backend,
    )
    post = m.inference_data.posterior
    truth = d["params_true"]
    for name, value in (("rho", 0.5), ("alpha", 3.0)):
        v = post[name].values.ravel()
        assert abs(v.mean() - value) < 3 * v.std(), name
    sd = post["unit_effect_sd"].values.ravel()
    assert abs(sd.mean() - truth["unit_effect"].std()) < 3 * sd.std()
    tau = post["time_effect"].values.reshape(-1, 4)
    np.testing.assert_array_less(
        np.abs(tau.mean(0) - truth["time_effect"][1:]), 3 * tau.std(0) + 1e-9
    )


def test_panel_period_effects_carry_the_time_effects_term():
    """Period effects inside the filter add Lee & Yu's -T log(1 - rho) (T), else 0."""
    _, m = _panel(T=4)
    assert m._latent_level_shift == 4.0
    assert m._latent_panel_kwargs()["level_shift"] == 4.0
    kw = dict(latent=True, unit_effects="random")
    d = dgp.simulate_sar_negbin(n_side=5, T=4, seed=1, **kw)
    m = SARNegBin(y=d["y"], X=d["X"], W=d["W_sparse"], N=25, T=4, **kw)
    assert m._latent_level_shift == 0.0


@pytest.mark.validation
def test_level_shift_sweep_is_invariant():
    """One sweep from a draw of the shifted target leaves it invariant.

    rho is drawn from the chain's prior times (1 - rho)^-m, the rest from their
    priors, then the disturbances, log-means and counts given them; one sweep
    from that state must return draws from the same joint, so the paired
    differences between its output and input average zero.
    """
    from neighbayes.models.priors import ReducedGibbsPriors
    from neighbayes.samplers._utils._latent_loglik import log_gamma_noise
    from neighbayes.samplers.negbin_latent._core import LatentNegBinChain
    from neighbayes.tests.helpers import make_rook_W

    rng = np.random.default_rng(31)
    side, T, shift, lo, hi = 3, 4, 4.0, 0.0, 0.6
    n = side * side
    Wd = make_rook_W(side)
    X = np.column_stack(
        [rng.standard_normal(n * T), np.kron(np.eye(T), np.ones((n, 1)))]
    )
    b0, bs = np.r_[0.6, np.full(T, 0.5)], np.r_[0.5, np.full(T, 0.5)]
    priors = ReducedGibbsPriors(beta_mu=b0, beta_sigma=bs, rho_lower=lo, rho_upper=hi)
    chain = LatentNegBinChain(
        y=np.zeros(n * T), X=X, W=sp.csr_matrix(Wd), process="sar", priors=priors,
        logdet_fn=lambda r: float(np.linalg.slogdet(np.eye(n) - r * Wd)[1]),
        rng=rng, T=T, level_shift=shift,
    )  # fmt: skip
    e = 1.0 - shift
    a, b = (1.0 - lo) ** e, (1.0 - hi) ** e
    diffs = []
    for _ in range(30_000):
        rho = 1.0 - (a + rng.random() * (b - a)) ** (1.0 / e)
        beta = b0 + bs * rng.standard_normal(b0.size)
        while True:  # the sampler bounds log alpha to [-10, 10]
            alpha = 2.5 * abs(rng.standard_t(3.0))
            if np.exp(-10.0) < alpha < np.exp(10.0):
                break
        eps = log_gamma_noise(rng, alpha, (T, n))
        eta = np.linalg.solve(
            np.eye(n) - rho * Wd, ((X @ beta).reshape(T, n) + eps).T
        ).T
        chain.y = rng.poisson(np.exp(eta)).astype(np.float64)
        chain.par, chain.beta, chain.alpha, chain.eta = rho, beta.copy(), alpha, eta
        chain.sweep(tuning=False)
        diffs.append([chain.par - rho, np.log(chain.alpha) - np.log(alpha)])
    d = np.asarray(diffs)
    z = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(d)))
    assert np.all(np.abs(z) < 4.0), z


# ---------------------------------------------------------------------------
# The joint move (step 8): rho, log alpha and the level, eta_P and z_Z held
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [None, "eigenvalue", "chol_cheb", "lu_cheb", "chol_aaa", "lu_aaa", "stoch_cheb"],
)
def test_zero_set_logdet_is_the_zero_sites_filter(method):
    """``log|I - rho W_ZZ|`` over each period's zero sites, isolated ones included."""
    from neighbayes.samplers.negbin_latent._core import zero_set_logdet
    from neighbayes.tests.helpers import make_rook_W

    Wd = make_rook_W(8)
    zero = np.random.default_rng(0).random((2, 64)) < 0.7
    fn = zero_set_logdet(sp.csr_matrix(Wd), zero, method, -0.99, 0.99)
    for r in (-0.9, 0.2, 0.95):
        exact = sum(
            np.linalg.slogdet(np.eye(z.sum()) - r * Wd[np.ix_(z, z)])[1] for z in zero
        )
        assert fn(r) == pytest.approx(exact, abs=1e-6)
    assert zero_set_logdet(sp.csr_matrix(Wd), np.zeros((1, 64), bool))(0.5) == 0.0


def test_joint_axes_and_level_direction():
    """The level is the intercept, or the sum of the period dummies without one."""
    from neighbayes.samplers.negbin_latent._core import joint_axes

    x = np.random.default_rng(0).standard_normal(6)
    axes, v = joint_axes(np.column_stack([np.ones(6), x]), False)
    assert axes == (0, 1, 2) and np.allclose(v, [1.0, 0.0])
    dummies = np.kron(np.eye(2), np.ones((3, 1)))
    axes, v = joint_axes(np.column_stack([x, dummies]), True)
    assert axes == (0, 2) and np.allclose(v, [0.0, 1.0, 1.0])
    axes, v = joint_axes(x[:, None], False)
    assert axes == (0, 1) and np.all(v == 0.0)


def _joint_draw(rng, case, X, Wd, priors, sar=True, T=1, fx=None, shift=0.0):
    """A draw from the joint of the parameters, the log-means and the counts.

    ``case`` sets what the counts observe: ``"poisson"``, the hurdle's count
    part (positives truncated at zero, zeros unobserved) or the ZINB's (the
    inactive units unobserved).  Returns ``(params, y, kind)``, ``(T, n)``.
    """
    from scipy.stats import poisson

    from neighbayes.samplers._utils._latent_loglik import (
        POISSON,
        TRUNCATED,
        UNOBSERVED,
        log_gamma_noise,
    )

    n = Wd.shape[0]
    lo, hi = priors.rho_lower, priors.rho_upper
    if shift:  # the prior times (1 - rho)^-shift, by inversion
        e = 1.0 - shift
        a, b = (1.0 - lo) ** e, (1.0 - hi) ** e
        rho = 1.0 - (a + rng.random() * (b - a)) ** (1.0 / e)
    else:
        rho = rng.uniform(lo, hi)
    b0, bs = np.asarray(priors.beta_mu), np.asarray(priors.beta_sigma)
    beta = b0 + bs * rng.standard_normal(b0.size)
    while True:  # the sampler bounds log alpha to [-10, 10]
        alpha = 2.5 * abs(rng.standard_t(3.0))
        if np.exp(-10.0) < alpha < np.exp(10.0):
            break
    systematic = (X @ beta).reshape(T, n)
    p = {"par": rho, "beta": beta, "alpha": alpha}
    if fx is not None:
        p["sigma_c"] = fx.sd_scale * abs(rng.standard_t(3.0))
        p["c"] = fx.mu + p["sigma_c"] * rng.standard_normal(fx.n_groups)
        systematic = systematic + p["c"][fx.group].reshape(T, n)
    eps = log_gamma_noise(rng, alpha, (T, n))
    S = np.linalg.inv(np.eye(n) - rho * Wd)
    p["eta"] = (systematic + eps) @ S.T if sar else systematic + eps @ S.T
    lam = np.exp(np.minimum(p["eta"], 30.0))
    if case == "hurdle":
        pos = rng.random((T, n)) < 0.5
        u = rng.uniform(np.exp(-lam), 1.0)
        y = np.where(pos, np.maximum(poisson.ppf(u, lam), 1.0), 0.0)
        kind = np.where(pos, TRUNCATED, UNOBSERVED)
    elif case == "zinb":
        active = rng.random((T, n)) < 0.7
        y = np.where(active, rng.poisson(lam), 0.0)
        kind = np.where(active, POISSON, UNOBSERVED)
    else:
        y, kind = rng.poisson(lam).astype(np.float64), np.full((T, n), POISSON)
    return p, y, kind


def _joint_setup(name, rng):
    """``(X, W, priors, sar, T, unit effects, level shift, case)`` of a 3 x 3 case."""
    from neighbayes.models.priors import ReducedGibbsPriors
    from neighbayes.samplers.negbin_latent._core import UnitEffects
    from neighbayes.tests.helpers import make_rook_W

    n, panel = 9, name == "panel"
    T = 2 if panel else 1
    x = rng.standard_normal(n * T)
    if panel:  # the level is the period dummies' sum
        X = np.column_stack([x, np.kron(np.eye(T), np.ones((n, 1)))])
        fx = UnitEffects(
            group=np.tile(np.arange(n), T), n_groups=n, mu=0.0, sd_scale=0.5
        )
    else:
        X, fx = np.column_stack([np.ones(n), x]), None
    k = X.shape[1]
    priors = ReducedGibbsPriors(
        beta_mu=np.full(k, 0.6),
        beta_sigma=np.full(k, 0.5),
        rho_lower=0.0,
        rho_upper=0.6,
    )
    case = name if name in ("hurdle", "zinb") else "poisson"
    return X, make_rook_W(3), priors, name != "sem", T, fx, 2.0 if panel else 0.0, case


#: Fixed directions off the axes and widths for the invariance checks.
_JOINT_DIRS = np.linalg.qr(np.random.default_rng(5).standard_normal((3, 3)))[0]
_JOINT_WIDTHS = np.array([0.15, 0.4, 0.3])


def _one_move_z(name, reps, seed, zero_logdet=None):
    """z-scores of one joint move's paired differences, from exact draws.

    Of the spatial parameter, log alpha, the level of X beta and the mean
    log-mean; each chain is built on its draw's counts, so the held sets are
    its zeros and positives.
    """
    from neighbayes.samplers.negbin_latent._core import LatentNegBinChain

    rng = np.random.default_rng(seed)
    X, Wd, priors, sar, T, fx, shift, case = _joint_setup(name, rng)
    n = Wd.shape[0]

    def logdet(r):
        return float(np.linalg.slogdet(np.eye(n) - r * Wd)[1])

    diffs = []
    for _ in range(reps):
        p, y, kind = _joint_draw(rng, case, X, Wd, priors, sar, T, fx, shift)
        chain = LatentNegBinChain(
            y=y.ravel(), X=X, W=sp.csr_matrix(Wd), process="sar" if sar else "sem",
            priors=priors, logdet_fn=logdet, rng=rng, T=T, unit_effects=fx,
            kind=kind.ravel(), level_shift=shift, zero_logdet=zero_logdet,
        )  # fmt: skip
        chain.joint_dirs, chain.joint_widths = _JOINT_DIRS, _JOINT_WIDTHS
        for key, value in p.items():
            setattr(chain, key, np.copy(value) if np.ndim(value) else value)
        chain._joint_move()
        diffs.append(
            [
                chain.par - p["par"],
                np.log(chain.alpha / p["alpha"]),
                np.mean(X @ (chain.beta - p["beta"])),
                np.mean(chain.eta - p["eta"]),
            ]
        )
    d = np.asarray(diffs)
    return d.mean(0) / (d.std(0, ddof=1) / np.sqrt(reps))


@pytest.mark.validation
@pytest.mark.parametrize("name", ["sar", "sem", "panel", "hurdle", "zinb"])
def test_joint_move_is_invariant(name):
    """One joint move from a draw of the joint returns draws from it.

    The panel has unit effects, period dummies for its level and the
    time-effects term; the hurdle's and ZINB's count parts observe the
    positives truncated and the active units only.
    """
    z = _one_move_z(name, 30_000, seed=11)
    assert np.all(np.abs(z) < 4.0), z


@pytest.mark.validation
def test_joint_move_check_has_power():
    """The same check fails when the move's Jacobian omits ``log|A_ZZ|``."""
    z = _one_move_z("panel", 30_000, seed=13, zero_logdet=lambda r: 0.0)
    assert np.max(np.abs(z)) > 5.0, z


@pytest.mark.validation
@pytest.mark.requires_jax
@pytest.mark.parametrize("process", ["sar", "sem"])
def test_jax_sweep_is_invariant(process):
    """One JAX sweep from draws of the joint, 40,000 at once by ``vmap``.

    Each replicate carries its own counts, zero mask and zero-set series, so
    the masked solves and the slices' stages run on every zero pattern.
    """
    import jax
    import jax.numpy as jnp

    from neighbayes._jax_dispatch import ensure_x64
    from neighbayes._logdet._chebyshev import (
        chebyshev_coeffs_dct1,
        chebyshev_gauss_nodes,
    )
    from neighbayes._logdet._jax import logdet_jax_params, logdet_param_evaluator
    from neighbayes.samplers.negbin_latent import _jax as nb_jax
    from neighbayes.samplers.negbin_reduced._jax import (
        _build_sparse_ctx,
        _sparse_solve_parts,
    )

    ensure_x64()
    rng = np.random.default_rng(21)
    X, Wd, priors, sar, T, fx, shift, case = _joint_setup(process, rng)
    n, R = Wd.shape[0], 10_000
    W = sp.csr_matrix(Wd)
    lu_solve, arrays = _sparse_solve_parts(_build_sparse_ctx(W, n))
    arrays = {key: arrays[key] for key in ("Ai", "Aj", "eye", "w")}
    kind, params = logdet_jax_params(W, "eigenvalue", 0.0, 0.6)
    consts = nb_jax._latent_data(np.ones(n), X, W, priors, params, arrays)
    sweep = nb_jax._make_sweep(
        n, 2, process, int(consts["classes"].shape[0]), False, True,
        logdet_param_evaluator(kind), lu_solve, False,
    )  # fmt: skip
    nodes, _ = chebyshev_gauss_nodes(nb_jax._ZERO_LOGDET_ORDER, 0.0, 0.6)
    in_consts = {key: None for key in consts}
    in_consts.update(y=0, zmask=0, zero_logdet=(0, None, None))
    in_state = {
        key: 0
        for key in nb_jax.start_state(np.ones(n), X, 1, priors, 0.4, None, (0, 1, 2))
    }
    in_state["it"] = None
    run = jax.jit(
        jax.vmap(
            lambda s, key, d: sweep(s, key, jnp.bool_(False), d)[0],
            in_axes=(in_state, 0, in_consts),
        )
    )
    diffs = []
    for chunk in range(4):
        draws = [_joint_draw(rng, case, X, Wd, priors, sar) for _ in range(R)]
        p = {key: np.stack([d[0][key] for d in draws]) for key in draws[0][0]}
        y = np.stack([d[1] for d in draws])
        coeffs = []
        for z in y[:, 0] == 0:
            lam = np.linalg.eigvals(Wd[np.ix_(z, z)])
            coeffs.append(
                chebyshev_coeffs_dct1(
                    [np.sum(np.log(np.abs(1.0 - r * lam))) for r in nodes]
                )
            )
        d = {
            **consts,
            "y": jnp.asarray(y),
            "zmask": jnp.asarray(y == 0, dtype=jnp.float64),
            "zero_logdet": (jnp.asarray(np.array(coeffs)), *consts["zero_logdet"][1:]),
        }
        state = {
            "par": jnp.asarray(p["par"]),
            "beta": jnp.asarray(p["beta"]),
            "alpha": jnp.asarray(p["alpha"]),
            "eta": jnp.asarray(p["eta"]),
            "sxb": jnp.zeros((R, 1, n)),
            "s1": jnp.ones((R, 1, n)),
            "step": jnp.full(R, 0.1),
            "slice_width": jnp.full(R, 0.4),
            "it": jnp.float64(1000.0),
            "joint_dirs": jnp.broadcast_to(_JOINT_DIRS, (R, 3, 3)),
            "joint_widths": jnp.broadcast_to(_JOINT_WIDTHS, (R, 3)),
            "joint_start": jnp.zeros((R, 1, n)),
        }
        out = run(state, jax.random.split(jax.random.PRNGKey(chunk), R), d)
        diffs.append(
            np.column_stack(
                [
                    np.asarray(out["par"]) - p["par"],
                    np.log(np.asarray(out["alpha"]) / p["alpha"]),
                    (np.asarray(out["beta"]) - p["beta"]) @ X.mean(0),
                    np.asarray(out["eta"]).mean((1, 2)) - p["eta"].mean((1, 2)),
                ]
            )
        )
    diffs = np.concatenate(diffs)
    z = diffs.mean(0) / (diffs.std(0, ddof=1) / np.sqrt(len(diffs)))
    assert np.all(np.abs(z) < 4.0), z
