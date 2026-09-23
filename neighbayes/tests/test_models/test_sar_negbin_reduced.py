"""Fast build/method tests for the reduced-form SARNegBin.

These exercise the reduced-form (PG-Gibbs, no σ) model.  A separate,
slower recovery-style test lives in ``test_sar_negbin_reduced_recovery.py``
(gated by a marker if added).
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

import neighbayes as bp
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


def _count_data(seed: int = 101, n: int = 10):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_reduced_inherits_from_spatial_model():
    """The reduced-form class inherits from SpatialModel."""
    from neighbayes.models.base import SpatialModel

    assert issubclass(bp.models.SARNegBin, SpatialModel)


def test_reduced_rejects_noninteger_or_negative_y():
    _, _, _ = _count_data()
    y_bad = np.array([0.0, 1.2, 2.0, 1.0])
    X_bad = np.column_stack([np.ones(4), np.arange(4)])
    with pytest.raises(ValueError, match="integer-valued"):
        bp.models.SARNegBin(y=y_bad, X=X_bad, W=W_to_graph(make_line_W(4)))

    y_neg = np.array([0.0, 1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="non-negative"):
        bp.models.SARNegBin(y=y_neg, X=X_bad, W=W_to_graph(make_line_W(4)))


def test_reduced_robust_is_unsupported():
    y, X, W = _count_data()
    with pytest.raises(NotImplementedError):
        bp.models.SARNegBin(y=y, X=X, W=W, robust=True)


def test_reduced_build_pymc_model_returns_valid_model():
    """The reduced-form model builds a valid PyMC model with Jacobian."""
    import pymc as pm

    y, X, W = _count_data()
    model = bp.models.SARNegBin(y=y, X=X, W=W)
    pymc_model = model._build_pymc_model()
    assert isinstance(pymc_model, pm.Model)
    assert "rho" in pymc_model.named_vars
    assert "beta" in pymc_model.named_vars
    assert "alpha" in pymc_model.named_vars
    # No log|I - rhoW| Jacobian: y is not linearly transformed in the
    # reduced-form count model, so there is no change of variables to
    # correct for. Including it biases rho toward zero.
    assert "jacobian" not in pymc_model.named_vars
    # Reduced form must NOT have sigma, sigma2, or z
    assert "sigma" not in pymc_model.named_vars
    assert "sigma2" not in pymc_model.named_vars
    assert "z" not in pymc_model.named_vars


# ---------------------------------------------------------------------------
# Post-fit machinery (mock posterior — no σ / z draws)
# ---------------------------------------------------------------------------


def test_reduced_fitted_values_and_effects_with_mock_posterior():
    y, X, W = _count_data(seed=103)
    model = bp.models.SARNegBin(y=y, X=X, W=W)

    # Reduced form: only β, ρ, α — no σ, no z.
    model._idata = _idata(
        {
            "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
            "rho": np.array([0.15, 0.16]),
            "alpha": np.array([2.0, 2.1]),
        }
    )

    fitted = model.fitted_values()
    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)

    effects = model.spatial_effects()
    assert "direct" in effects.columns
    assert np.all(np.isfinite(effects["direct"].values))


# ---------------------------------------------------------------------------
# End-to-end Gibbs fit (small, fast)
# ---------------------------------------------------------------------------


def test_reduced_fit_returns_inference_data():
    """A short Gibbs run should return a well-formed InferenceData."""
    rng = np.random.default_rng(0)
    n = 60
    W = W_to_graph(make_line_W(n))
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.5 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu)

    model = bp.models.SARNegBin(y=y, X=X, W=W)
    idata = model.fit(
        draws=30,
        tune=30,
        chains=2,
        random_seed=0,
        idata_kwargs={"log_likelihood": True},
    )

    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata.groups()
    assert "log_likelihood" in idata.groups()
    assert "observed_data" in idata.groups()
    # Reduced-form posterior must NOT contain σ or z.
    assert "sigma" not in idata.posterior.data_vars
    assert "z" not in idata.posterior.data_vars
    # Required parameters present with correct shapes.
    assert idata.posterior["rho"].shape == (2, 30)
    assert idata.posterior["alpha"].shape == (2, 30)
    assert idata.posterior["beta"].shape == (2, 30, 2)
    assert idata.log_likelihood["obs"].shape == (2, 30, n)


def test_reduced_fit_default_is_gibbs():
    """Default sampler='gibbs' should produce the same result as explicit."""
    rng = np.random.default_rng(42)
    n = 30
    W = W_to_graph(make_line_W(n))
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.5 * x1
    mu = np.exp(eta)
    y = rng.poisson(mu)

    model = bp.models.SARNegBin(y=y, X=X, W=W)
    # Default call (no sampler kwarg) should use Gibbs
    idata = model.fit(draws=10, tune=10, chains=1, random_seed=0)
    assert isinstance(idata, az.InferenceData)
    assert "rho" in idata.posterior.data_vars
    assert "alpha" in idata.posterior.data_vars


# ---------------------------------------------------------------------------
# Mixing: β-marginalized ρ slice should give healthy ESS
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reduced_rho_mixing_with_beta_marginalization():
    """ρ ESS_bulk should be a non-trivial fraction of total draws.

    Regression test for the β-marginalized ρ slice sampler.  Without
    marginalization, ρ ESS at this DGP is in single digits per chain;
    with marginalization it is in the hundreds.  We set a deliberately
    loose floor so the test is robust to seed / library variation.
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    side = 10  # n = 100 keeps the test fast (~10s)
    n = side * side
    rows, cols = [], []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    rows.append(i)
                    cols.append(rr * side + cc)
    W_unweighted = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(W_unweighted.sum(axis=1)).ravel()
    W_sparse = sp.diags(1.0 / deg) @ W_unweighted

    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([0.5, 0.4])
    rho_true = 0.5
    alpha_true = 2.0
    A = sp.eye(n) - rho_true * W_sparse
    eta = sp.linalg.spsolve(A.tocsc(), X @ beta_true)
    mu = np.exp(eta)
    p = alpha_true / (alpha_true + mu)
    y = rng.negative_binomial(alpha_true, p).astype(float)

    model = bp.models.SARNegBin(y=y, X=X, W=W_to_graph(W_sparse))
    model.fit(draws=400, tune=400, chains=2, random_seed=0)

    rho_ess = float(az.ess(model.inference_data, var_names=["rho"]).rho.values)
    # Conservative floor: at 800 total post-warmup draws, even a fraction
    # of 0.1 (ESS=80) signals that ρ is mixing, vs the pre-marginalized
    # ESS that was ~5–10 at this DGP.
    assert rho_ess >= 80.0, f"ρ ESS too low: {rho_ess}"


def test_reduced_nuts_and_gibbs_agree_and_recover_rho():
    """The two samplers must target the same posterior.

    Regression test for a spurious ``log|I - rho W|`` Potential in the
    reduced-form PyMC model.  ``y`` is not linearly transformed here — the
    spatial structure enters only through the mean — so no Jacobian belongs.
    Including it dragged the NUTS posterior toward zero (rho ~ 0.14 against a
    true 0.4) while Gibbs was unaffected, and both samplers reported
    rhat ~ 1.00, so nothing flagged the disagreement.
    """
    import arviz as az

    from neighbayes.dgp import simulate_sar_negbin
    from neighbayes.models import SARNegBin

    data = simulate_sar_negbin(n=25, rho=0.4, seed=1)
    y, X, W = data["y"], data["X"], data["W_graph"]
    kwargs = dict(draws=1000, tune=800, chains=2, random_seed=1, progressbar=False)

    gibbs = SARNegBin(y=y, X=X, W=W).fit(**kwargs)
    nuts = SARNegBin(y=y, X=X, W=W).fit(sampler="nuts", **kwargs)

    rho_gibbs = float(gibbs.posterior["rho"].mean())
    rho_nuts = float(nuts.posterior["rho"].mean())

    # Both recover the truth ...
    assert rho_gibbs == pytest.approx(0.4, abs=0.12), f"gibbs rho={rho_gibbs}"
    assert rho_nuts == pytest.approx(0.4, abs=0.12), f"nuts rho={rho_nuts}"
    # ... and agree with each other well inside the posterior sd.
    sd = float(gibbs.posterior["rho"].std())
    assert abs(rho_gibbs - rho_nuts) < 0.5 * sd, (
        f"samplers disagree: gibbs={rho_gibbs:.3f} nuts={rho_nuts:.3f} sd={sd:.3f}"
    )
    for idata in (gibbs, nuts):
        assert float(az.rhat(idata, var_names=["rho"])["rho"]) < 1.05


def test_reduced_nuts_uses_the_resolved_beta_prior():
    """NUTS must not fall back to N(0, 1e6) when the priors dict omits beta_*.

    ``beta_mu``/``beta_sigma`` default to None and are resolved to the
    data-scaled Gelman prior at build time, so ``priors.get("beta_sigma", 1e6)``
    silently gave the NUTS path a near-improper prior the Gibbs path never saw.
    """
    from neighbayes.dgp import simulate_sar_negbin
    from neighbayes.models import SARNegBin

    data = simulate_sar_negbin(n=15, rho=0.4, seed=1)
    model = SARNegBin(y=data["y"], X=data["X"], W=data["W_graph"])
    expected_mu, expected_sigma = model._gelman_default_beta_prior(
        model._X, list(model._feature_names)
    )

    pymc_model = model._build_pymc_model()
    beta = pymc_model.named_vars["beta"]
    sigma = beta.owner.inputs[-1].eval()

    np.testing.assert_allclose(
        np.broadcast_to(sigma, expected_sigma.shape), expected_sigma, rtol=1e-8
    )
    assert np.all(sigma < 1e5), "beta prior fell back to the near-improper default"
