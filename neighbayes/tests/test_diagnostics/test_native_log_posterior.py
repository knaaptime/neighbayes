"""Equivalence pins for the PyMC-free bridge-sampling log-posterior.

:func:`~neighbayes.diagnostics._native_log_posterior.native_log_posterior`
replaces a compiled PyTensor graph with closed-form algebra, so it has to
reproduce :func:`~neighbayes.diagnostics.bayesfactor.compile_log_posterior`
*absolutely*, not up to a constant: Bayes factors compare marginal likelihoods
across models, so any offset survives into the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.diagnostics._native_log_posterior import native_log_posterior
from neighbayes.diagnostics.bayesfactor import compile_log_posterior
from neighbayes.models.cross_section.ols import OLS
from neighbayes.models.cross_section.sar import SAR
from neighbayes.models.cross_section.sdem import SDEM
from neighbayes.models.cross_section.sdm import SDM
from neighbayes.models.cross_section.sem import SEM
from neighbayes.tests.helpers import make_rook_W


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    W = sp.csr_matrix(make_rook_W(5).astype(float))
    n = W.shape[0]
    X = rng.normal(size=(n, 2))
    y = sp.linalg.spsolve(
        sp.eye(n, format="csc") - 0.5 * W,
        1.0 + 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=n),
    )
    return y, X, W


def _fit(cls, data, **kwargs):
    y, X, W = data
    ctor = dict(y=y, X=X, **kwargs)
    if cls is not OLS:
        ctor["W"] = W
    model = cls(**ctor)
    sampler = "nuts" if kwargs.get("robust") or cls is OLS else "gibbs"
    model.fit(
        draws=120, tune=80, chains=2, sampler=sampler, progressbar=False, random_seed=1
    )
    return model


@pytest.mark.parametrize("cls", [SAR, SDM, SEM, SDEM])
class TestMatchesPyMC:
    def test_parameter_names_and_order_match(self, cls, data):
        """A different order would make the bridge estimator read blocks
        transposed, silently."""
        model = _fit(cls, data)
        _, pymc_names, pymc_info, _ = compile_log_posterior(model.pymc_model)
        _, native_names, native_info, _ = native_log_posterior(model)
        assert native_names == pymc_names
        assert native_info == pymc_info

    def test_unconstrained_draws_match(self, cls, data):
        model = _fit(cls, data)
        _, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
        _, _, _, native_to_u = native_log_posterior(model)
        post = model.inference_data.posterior
        assert np.allclose(native_to_u(post), pymc_to_u(post), rtol=0, atol=1e-12)

    def test_log_posterior_matches_absolutely(self, cls, data):
        """Machine precision, not "up to a constant" — constants matter here."""
        model = _fit(cls, data)
        pymc_fn, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
        native_fn, _, _, _ = native_log_posterior(model)
        theta = pymc_to_u(model.inference_data.posterior)[:60]
        diff = np.array([native_fn(t) - pymc_fn(t) for t in theta])
        assert np.abs(diff).max() < 1e-9


@pytest.mark.parametrize("cls", [SAR, SEM])
def test_robust_matches_within_gammaln_accuracy(cls, data):
    """Student-t agrees to ~6e-10 per observation.

    The residual is not in the algebra: it is ``scipy.special.gammaln`` versus
    PyTensor's log-gamma at the same arguments, with scipy's being the more
    accurate.  It appears as a constant offset, so it moves a log marginal
    likelihood by ``n * 6e-10`` — far below bridge sampling's Monte-Carlo error.
    """
    model = _fit(cls, data, robust=True)
    pymc_fn, _, _, pymc_to_u = compile_log_posterior(model.pymc_model)
    native_fn, _, _, _ = native_log_posterior(model)
    theta = pymc_to_u(model.inference_data.posterior)[:60]
    diff = np.array([native_fn(t) - pymc_fn(t) for t in theta])
    n = model._y.shape[0]
    assert np.abs(diff).max() < n * 1e-8
    # Constant offset, so it cancels in any ratio of posterior densities.
    assert diff.std() < 1e-12


def test_unsupported_family_returns_none():
    """``None`` means "no closed form here", so callers fall back to PyMC
    rather than treating it as a failure."""

    class NotAGaussianModel:
        pass

    assert native_log_posterior(NotAGaussianModel()) is None


def test_bridge_uses_native_path_without_building_a_pymc_graph(data):
    """The point of the exercise: a Gibbs fit must not need a PyTensor graph."""
    from neighbayes.diagnostics.bayesfactor import bayes_factor_compare_models

    models = [_fit(cls, data) for cls in (SAR, SEM)]
    assert all(m._pymc_model is None for m in models)

    bayes_factor_compare_models(models, model_labels=["SAR", "SEM"], method="bridge")

    assert all(m._pymc_model is None for m in models), (
        "bridge sampling built a PyMC graph; the native log-posterior was bypassed"
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

N_UNITS, N_PERIODS = 25, 3


@pytest.fixture(scope="module")
def panel_data():
    rng = np.random.default_rng(3)
    W = sp.csr_matrix(make_rook_W(5).astype(float))
    W = sp.csr_matrix(sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel()) @ W)
    n = N_UNITS * N_PERIODS
    X = rng.normal(size=(n, 2))
    alpha = np.tile(rng.normal(scale=0.7, size=N_UNITS), N_PERIODS)
    mean = 1.0 + 2.0 * X[:, 0] - X[:, 1] + alpha + rng.normal(scale=0.5, size=n)
    W_T = sp.kron(sp.identity(N_PERIODS), W, format="csc")
    y = sp.linalg.spsolve(sp.identity(n, format="csc") - 0.3 * W_T, mean)
    return y, X, W


def _fit_panel(cls, panel_data, draws=120, **kwargs):
    y, X, W = panel_data
    model = cls(y=y, X=X, W=W, N=N_UNITS, T=N_PERIODS, **kwargs)
    model.fit(
        draws=draws,
        tune=200,
        chains=4,
        progressbar=False,
        random_seed=1,
    )
    return model


@pytest.mark.parametrize(
    "cls, effects",
    [
        (SAR, dict(unit_effects="fixed")),
        (SAR, dict(time_effects=True)),
        (SAR, dict(unit_effects="fixed", time_effects=True)),
        (SEM, dict(unit_effects="fixed")),
        (SEM, dict(unit_effects="fixed", time_effects=True)),
        (SDM, dict(unit_effects="fixed", time_effects=True)),
        (OLS, dict(unit_effects="fixed")),
    ],
)
def test_fixed_effects_match_pymc_absolutely(cls, effects, panel_data):
    """The degrees-of-freedom and time-effects terms vary with the parameters,
    so missing them would not cancel in a Bayes factor."""
    model = _fit_panel(cls, panel_data, **effects)
    pymc_fn, _, _, pymc_to_u = compile_log_posterior(model._build_pymc_model())
    native_fn, _, _, _ = native_log_posterior(model)
    theta = pymc_to_u(model.inference_data.posterior)[:60]
    diff = np.array([native_fn(t) - pymc_fn(t) for t in theta])
    assert np.abs(diff).max() < 1e-8


def test_dynamic_panel_takes_the_pymc_route(panel_data):
    """The native density has no time lag, so it must not claim this model."""
    from neighbayes.diagnostics.bayesfactor import bayes_factor_compare_models

    model = _fit_panel(SAR, panel_data, dynamic=True)
    assert native_log_posterior(model) is None
    bf = bayes_factor_compare_models(
        {"a": model, "b": model}, method="bridge", log=True, random_state=0
    )
    assert np.isfinite(bf.loc["a", "b"])


def _dense_covariance(model, value, sigma2, sd):
    """Covariance of the outcome (or filtered outcome) given the parameters."""
    N, T = model._N, model._T
    n = N * T
    Z_unit = np.tile(np.eye(N), (T, 1))  # stacked by period, then unit
    if model._jacobian_param == "lam":
        A = np.eye(N) - value * model._W_sparse.toarray()
        B = np.kron(np.eye(T), A)
        return sigma2 * np.linalg.inv(B.T @ B) + sd**2 * Z_unit @ Z_unit.T
    return sigma2 * np.eye(n) + sd**2 * Z_unit @ Z_unit.T


def _dense_outcome(model, value):
    if model._jacobian_param == "rho":
        return model._y - value * model._Wy
    return model._y


@pytest.mark.parametrize("cls", [OLS, SAR, SEM, SDEM])
def test_random_effects_density_matches_a_dense_gaussian(cls, panel_data):
    """Integrating the effects out must leave the exact marginal density."""
    from scipy import stats

    model = _fit_panel(cls, panel_data, unit_effects="random")
    fn, names, info, to_u = native_log_posterior(model)
    assert names[-2:] == ["sigma2_log__", "unit_effect_sd_log__"]
    assert sum(info["sizes"].values()) == to_u(model.inference_data.posterior).shape[1]

    Z = model._design_matrix()
    from neighbayes.models.panel._re import _re_prior_values

    pv = _re_prior_values(model, Z)
    k = Z.shape[1]
    spatial = model._jacobian_param is not None
    for theta in to_u(model.inference_data.posterior)[:8]:
        offset = 1 if spatial else 0
        u = theta[0] if spatial else 0.0
        value = 2.0 * np.exp(u) / (1.0 + np.exp(u)) - 1.0 if spatial else 0.0
        beta = theta[offset : offset + k]
        s2, sd = np.exp(theta[offset + k]), np.exp(theta[offset + k + 1])

        expected = stats.multivariate_normal(
            Z @ beta, _dense_covariance(model, value, s2, sd)
        ).logpdf(_dense_outcome(model, value))
        expected += stats.norm(pv["beta_mu"], pv["beta_sigma"]).logpdf(beta).sum()
        expected += stats.invgamma(pv["sigma2_alpha"], scale=pv["sigma2_beta"]).logpdf(
            s2
        ) + np.log(s2)
        expected += stats.halfcauchy(scale=pv["unit_effect_sd_scale"]).logpdf(sd)
        expected += np.log(sd)
        if spatial:
            # Uniform(-1, 1) on the interval transform, and the Jacobian.
            expected += np.log(0.5) + np.log(2.0 * np.exp(u) / (1.0 + np.exp(u)) ** 2)
        if model._jacobian_param == "rho":
            # A SEM's covariance already carries the determinant.
            expected += float(model._logdet_numpy_fn(value))

        assert fn(theta) == pytest.approx(expected, abs=1e-7)


def test_random_effects_with_student_t_errors_refuse_clearly(panel_data):
    y, X, W = panel_data
    model = SAR(
        y=y, X=X, W=W, N=N_UNITS, T=N_PERIODS, unit_effects="random", robust=True
    )
    with pytest.raises(NotImplementedError, match="closed form"):
        native_log_posterior(model)


def _exact_random_effects_logml(model, m=14, width=8.0):
    """log p(y) by quadrature over the spatial parameter, sigma and sigma_alpha.

    Beta integrates out in closed form through a dense Gaussian.  Slow by design:
    it is the reference, and it knows nothing of the native density's algebra.
    """
    from scipy import stats
    from scipy.special import logsumexp

    from neighbayes.models.panel._re import _re_prior_values

    Z = model._design_matrix()
    pv = _re_prior_values(model, Z)
    mu = np.asarray(pv["beta_mu"]) * np.ones(Z.shape[1])
    V = np.diag((np.asarray(pv["beta_sigma"]) * np.ones(Z.shape[1])) ** 2)
    post = model.inference_data.posterior
    x, w = np.polynomial.legendre.leggauss(m)

    def grid(draws, lower):
        lo = max(draws.mean() - width * draws.std(), lower)
        hi = draws.mean() + width * draws.std()
        return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), np.log(0.5 * (hi - lo) * w)

    param = model._jacobian_param
    values, log_wv = grid(post[param].values.ravel(), -0.999)
    sigmas, log_ws = grid(post["sigma"].values.ravel(), 1e-6)
    sds, log_wd = grid(post["unit_effect_sd"].values.ravel(), 1e-6)
    prior_s2 = stats.invgamma(pv["sigma2_alpha"], scale=pv["sigma2_beta"])
    prior_sd = stats.halfcauchy(scale=pv["unit_effect_sd_scale"])
    design_cov = Z @ V @ Z.T

    terms = []
    for value, lwv in zip(values, log_wv):
        outcome = _dense_outcome(model, value)
        # A SEM's covariance already carries the determinant.
        logdet = float(model._logdet_numpy_fn(value)) if param == "rho" else 0.0
        for s, lws in zip(sigmas, log_ws):
            s2 = s * s
            log_s = prior_s2.logpdf(s2) + np.log(2.0 * s)
            for sd, lwd in zip(sds, log_wd):
                cov = _dense_covariance(model, value, s2, sd) + design_cov
                ll = stats.multivariate_normal(Z @ mu, cov).logpdf(outcome)
                terms.append(
                    ll
                    + logdet
                    + np.log(0.5)
                    + log_s
                    + prior_sd.logpdf(sd)
                    + lwv
                    + lws
                    + lwd
                )
    return float(logsumexp(terms))


@pytest.mark.parametrize("cls", [SAR, SEM])
def test_bridge_recovers_the_exact_marginal_likelihood(cls, panel_data):
    """Tolerance 0.25: bridge's proposal fits log sigma_alpha's tail imperfectly,
    which moved the full-size estimates by up to 0.14 and exceeds its reported
    MCSE.  A shared offset of that size is immaterial to any Bayes factor."""
    from neighbayes.diagnostics.bayesfactor import bayes_factor_compare_models

    model = _fit_panel(cls, panel_data, draws=5000, unit_effects="random")
    _, diagnostics = bayes_factor_compare_models(
        {"a": model, "b": model},
        method="bridge",
        return_diagnostics=True,
        log=True,
        random_state=0,
    )
    assert diagnostics["a"]["logml"] == pytest.approx(
        _exact_random_effects_logml(model), abs=0.25
    )
    assert model._pymc_model is None
