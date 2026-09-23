"""Tests for bridge sampling and Bayes factor comparison.

Tests follow the R bridgesampling package's validation approach:
1. Analytical normalizing constant for a 2D standard normal
2. BIC-to-BF conversion
3. Posterior model probabilities
4. Bridge sampling with log_posterior function
5. ESS weighting
6. Two-phase convergence
7. Repetitions
"""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from neighbayes.diagnostics.bayesfactor import (
    _bridge_logml,
    _nearest_pos_def,
    _run_iterative_scheme,
    bayes_factor_compare_models,
    bic_to_bf,
    compile_log_posterior,
    post_prob,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_2d_normal_idata(n_samples=10000, seed=42):
    """Create InferenceData from a 2D standard normal distribution.

    The analytical log marginal likelihood is log(2π) ≈ 1.8379.
    This matches the R bridgesampling package's Example 1.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n_samples)
    # Reshape to (1, n_samples, 2) for ArviZ
    posterior_dict = {
        "x1": samples[:, 0].reshape(1, -1),
        "x2": samples[:, 1].reshape(1, -1),
    }
    idata = az.from_dict(posterior=posterior_dict)
    return idata, samples


def _make_2d_normal_log_posterior():
    """Return the true log-posterior for a 2D standard normal.

    log p(x) = -0.5 * x^T x  (unnormalized; the normalizing constant
    is what bridge sampling estimates).
    """

    def log_posterior(theta_flat):
        return -0.5 * np.sum(theta_flat**2)

    return log_posterior


def _make_simple_linear_idata(n=30, k=2, seed=42):
    """Create InferenceData for a simple linear regression model."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([1.0, 2.0])
    y = X @ beta_true + rng.normal(scale=0.5, size=n)

    # Fake posterior draws
    n_draws = 2000
    beta_samples = rng.normal(loc=beta_true, scale=0.1, size=(n_draws, k))
    sigma_samples = np.abs(rng.normal(loc=0.5, scale=0.05, size=n_draws))

    idata = az.from_dict(
        posterior={
            "beta": beta_samples.reshape(1, n_draws, k),
            "sigma": sigma_samples.reshape(1, n_draws),
        },
        log_likelihood={
            "obs": rng.normal(size=(1, n_draws, n)),  # placeholder
        },
        observed_data={"y": y},
    )
    return idata


# ---------------------------------------------------------------------------
# Test: _nearest_pos_def
# ---------------------------------------------------------------------------


class TestNearestPosDef:
    def test_already_pd(self):
        A = np.eye(3)
        result = _nearest_pos_def(A)
        np.testing.assert_allclose(result, A, atol=1e-10)

    def test_indefinite(self):
        # A matrix with a negative eigenvalue
        A = np.array([[1, 2], [2, 1]], dtype=float)
        eigvals = np.linalg.eigvals(A)
        assert np.any(eigvals < 0), "Test matrix should be indefinite"
        result = _nearest_pos_def(A)
        result_eigvals = np.linalg.eigvals(result)
        assert np.all(result_eigvals > 0), "Result should be positive definite"

    def test_singular(self):
        A = np.array([[1, 1], [1, 1]], dtype=float)
        result = _nearest_pos_def(A)
        result_eigvals = np.linalg.eigvals(result)
        assert np.all(result_eigvals > 0), "Result should be positive definite"


# ---------------------------------------------------------------------------
# Test: _run_iterative_scheme
# ---------------------------------------------------------------------------


class TestIterativeScheme:
    def test_2d_standard_normal(self):
        """Test against the analytical normalizing constant of a 2D standard normal.

        The true log marginal likelihood is log(2π) ≈ 1.8379.
        """
        rng = np.random.default_rng(42)
        n_post = 5000
        n_prop = 5000

        # Posterior samples from N(0, I_2)
        post_samples = rng.multivariate_normal(np.zeros(2), np.eye(2), size=n_post)
        # Proposal samples from N(0, I_2) — same distribution
        prop_samples = rng.multivariate_normal(np.zeros(2), np.eye(2), size=n_prop)

        # q11: log unnormalized posterior at posterior samples
        q11 = -0.5 * np.sum(post_samples**2, axis=1)
        # q12: log proposal density at posterior samples
        q12 = multivariate_normal.logpdf(post_samples, mean=np.zeros(2), cov=np.eye(2))
        # q21: log unnormalized posterior at proposal samples
        q21 = -0.5 * np.sum(prop_samples**2, axis=1)
        # q22: log proposal density at proposal samples
        q22 = multivariate_normal.logpdf(prop_samples, mean=np.zeros(2), cov=np.eye(2))

        result = _run_iterative_scheme(
            q11=q11,
            q12=q12,
            q21=q21,
            q22=q22,
            r0=1.0,
            tol=1e-10,
            maxiter=1000,
            criterion="r",
            neff=None,
            use_neff=False,
        )

        # The true logml is log(2π) ≈ 1.8379
        true_logml = np.log(2 * np.pi)
        np.testing.assert_allclose(
            result["logml"],
            true_logml,
            rtol=0.05,
            err_msg=f"logml={result['logml']:.4f}, expected={true_logml:.4f}",
        )
        assert result["converged"], "Iterative scheme should converge"
        assert result["niter"] < 100, (
            f"Should converge quickly, got {result['niter']} iterations"
        )


# ---------------------------------------------------------------------------
# Test: _bridge_logml
# ---------------------------------------------------------------------------


class TestBridgeLogml:
    def test_2d_normal_with_log_posterior(self):
        """Bridge sampling with a true log_posterior function on 2D standard normal."""
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        logml = _bridge_logml(idata, log_posterior=log_post, random_state=42)

        true_logml = np.log(2 * np.pi)
        np.testing.assert_allclose(
            logml,
            true_logml,
            rtol=0.05,
            err_msg=f"logml={logml:.4f}, expected={true_logml:.4f}",
        )

    def test_2d_normal_diagnostics(self):
        """Bridge sampling returns diagnostics when requested."""
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        diag = _bridge_logml(
            idata, log_posterior=log_post, return_diagnostics=True, random_state=42
        )

        assert isinstance(diag, dict)
        assert "logml" in diag
        assert "iterations" in diag
        assert "mcse_logml" in diag
        assert "converged" in diag
        assert diag["method"] == "bridge"
        assert diag["converged"]

    def test_2d_normal_with_repetitions(self):
        """Bridge sampling with repetitions returns median logml."""
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        diag = _bridge_logml(
            idata,
            log_posterior=log_post,
            return_diagnostics=True,
            repetitions=3,
            random_state=42,
        )

        true_logml = np.log(2 * np.pi)
        np.testing.assert_allclose(diag["logml"], true_logml, rtol=0.05)
        assert diag["repetitions"] == 3
        assert "logml_reps" in diag
        assert len(diag["logml_reps"]) == 3

    def test_ess_weighting(self):
        """Bridge sampling with use_neff=True uses ESS instead of nominal N."""
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        diag_neff = _bridge_logml(
            idata,
            log_posterior=log_post,
            return_diagnostics=True,
            use_neff=True,
            random_state=42,
        )
        diag_no_neff = _bridge_logml(
            idata,
            log_posterior=log_post,
            return_diagnostics=True,
            use_neff=False,
            random_state=42,
        )

        # Both should converge and give reasonable estimates
        true_logml = np.log(2 * np.pi)
        np.testing.assert_allclose(diag_neff["logml"], true_logml, rtol=0.05)
        np.testing.assert_allclose(diag_no_neff["logml"], true_logml, rtol=0.05)
        # ESS should be reported when use_neff=True
        assert diag_neff["neff"] is not None


# ---------------------------------------------------------------------------
# Test: bic_to_bf
# ---------------------------------------------------------------------------


class TestBicToBf:
    def test_basic(self):
        bic1, bic2, bic3 = 100, 95, 110
        result = bic_to_bf([bic1, bic2, bic3], denominator=bic1)
        np.testing.assert_allclose(result[0], 1.0)
        assert result[1] > 1  # model 2 has lower BIC, should be favored
        assert result[2] < 1  # model 3 has higher BIC

    def test_log(self):
        bic_values = [100, 95, 110]
        result = bic_to_bf(bic_values, denominator=100, log=True)
        # log(BF) = (BIC_denom - BIC_model) / 2
        expected = np.array([0, 2.5, -5.0])
        np.testing.assert_allclose(result, expected)

    def test_default_denominator(self):
        result = bic_to_bf([100, 95])
        np.testing.assert_allclose(result[0], 1.0)


# ---------------------------------------------------------------------------
# Test: post_prob
# ---------------------------------------------------------------------------


class TestPostProb:
    def test_uniform_prior(self):
        probs = post_prob([-20.8, -18.0, -19.0], model_names=["H0", "H1", "H2"])
        assert isinstance(probs, pd.Series)
        np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-10)
        # H1 has highest logml, should have highest probability
        assert probs["H1"] > probs["H0"]
        assert probs["H1"] > probs["H2"]

    def test_custom_prior(self):
        probs = post_prob(
            [-20.8, -18.0], model_names=["H0", "H1"], prior_prob=[0.8, 0.2]
        )
        np.testing.assert_allclose(probs.sum(), 1.0, rtol=1e-10)

    def test_equal_logml(self):
        probs = post_prob([-20.0, -20.0], model_names=["A", "B"])
        np.testing.assert_allclose(probs["A"], probs["B"], rtol=1e-10)
        np.testing.assert_allclose(probs["A"], 0.5, rtol=1e-10)

    def test_invalid_prior(self):
        with pytest.raises(ValueError, match="non-negative"):
            post_prob([-20, -18], prior_prob=[-0.5, 1.5])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="match length"):
            post_prob([-20, -18], prior_prob=[0.5])


# ---------------------------------------------------------------------------
# Test: bayes_factor_compare_models
# ---------------------------------------------------------------------------


class TestBayesFactorCompareModels:
    def test_bic_method_with_idata(self):
        """BIC method should work with InferenceData (no model object needed)."""
        idata = _make_simple_linear_idata()
        df = bayes_factor_compare_models([idata], method="bic", model_labels=["OLS"])
        assert df.shape == (1, 1)
        assert df.loc["OLS", "OLS"] == 1.0

    def test_bic_metric_alias_with_idata(self):
        """Backward compatibility: metric='bic' should behave like method='bic'."""
        idata = _make_simple_linear_idata()
        df = bayes_factor_compare_models([idata], model_labels=["OLS"], metric="bic")
        assert df.shape == (1, 1)
        assert df.loc["OLS", "OLS"] == 1.0

    def test_method_metric_conflict_raises(self):
        """Passing conflicting method and metric should raise a clear error."""
        idata = _make_simple_linear_idata()
        with pytest.raises(ValueError, match="both `method` and `metric`"):
            bayes_factor_compare_models([idata], method="bic", metric="bridge")

    def test_bridge_requires_model_object(self):
        """Bridge method raises ValueError when InferenceData is passed without model object."""
        idata, _ = _make_2d_normal_idata(n_samples=1000, seed=42)
        with pytest.raises(
            ValueError, match="InferenceData.*bridge sampling requires a fitted model"
        ):
            bayes_factor_compare_models([idata], method="bridge", model_labels=["M1"])

    def test_bridge_with_log_posterior_direct(self):
        """Bridge method with explicit log_posterior on 2D normal (direct _bridge_logml call)."""
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        logml = _bridge_logml(idata, log_posterior=log_post, random_state=42)
        true_logml = np.log(2 * np.pi)
        np.testing.assert_allclose(logml, true_logml, rtol=0.05)

    def test_bridge_with_model_object(self):
        """Bridge method with a fitted model object auto-compiles log_posterior."""
        import pymc as pm

        rng = np.random.default_rng(42)
        n = 30
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(scale=0.5, size=n)

        # Create and fit a simple model
        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0, sigma=100, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=10)
            mu = pm.math.dot(X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(
                draws=500, tune=500, chains=2, random_seed=42, progressbar=False
            )

        # Create a mock model object with .inference_data and .pymc_model
        class MockModel:
            def __init__(self, pymc_model, idata):
                self._pymc_model = pymc_model
                self._idata = idata

            @property
            def pymc_model(self):
                return self._pymc_model

            @property
            def inference_data(self):
                return self._idata

        mock_model = MockModel(model, idata)
        df = bayes_factor_compare_models(
            {"OLS": mock_model},
            method="bridge",
            random_state=42,
        )
        assert df.shape == (1, 1)
        assert df.loc["OLS", "OLS"] == 1.0

        # Also test with return_diagnostics to verify constrained_to_unconstrained works
        df2, diag = bayes_factor_compare_models(
            {"OLS": mock_model},
            method="bridge",
            random_state=42,
            return_diagnostics=True,
        )
        assert "OLS" in diag
        assert "logml" in diag["OLS"]
        # logml should be a reasonable finite number (not astronomically large)
        assert np.isfinite(diag["OLS"]["logml"])
        assert abs(diag["OLS"]["logml"]) < 1e6  # sanity check

    def test_return_diagnostics(self):
        idata, _ = _make_2d_normal_idata(n_samples=10000, seed=42)
        log_post = _make_2d_normal_log_posterior()

        diag = _bridge_logml(
            idata, log_posterior=log_post, return_diagnostics=True, random_state=42
        )
        assert isinstance(diag, dict)
        assert "logml" in diag
        assert diag["method"] == "bridge"
        assert diag["converged"]

    def test_unknown_method(self):
        idata, _ = _make_2d_normal_idata(n_samples=1000, seed=42)
        with pytest.raises(ValueError, match="Unknown method"):
            bayes_factor_compare_models([idata], method="unknown")

    def test_dict_input_bic(self):
        """BIC method accepts a dict of InferenceData."""
        idata = _make_simple_linear_idata()
        df = bayes_factor_compare_models(
            {"OLS": idata},
            method="bic",
        )
        assert df.shape == (1, 1)
        assert "OLS" in df.index

    def test_dict_labels_override(self):
        """Explicit model_labels override dict keys."""
        idata = _make_simple_linear_idata()
        df = bayes_factor_compare_models(
            {"OLS": idata},
            method="bic",
            model_labels=["Custom"],
        )
        assert "Custom" in df.index
        assert "OLS" not in df.index

    def test_bic_without_observed_data(self):
        """BIC method infers n_obs from log_likelihood when observed_data is missing."""
        idata = _make_simple_linear_idata()
        # Remove observed_data to simulate the case where it's absent
        del idata["observed_data"]
        df = bayes_factor_compare_models(
            {"OLS": idata},
            method="bic",
        )
        assert df.shape == (1, 1)
        assert np.isfinite(df.iloc[0, 0])

    def test_bic_with_model_object(self):
        """BIC method accepts fitted model objects and uses model._y for n_obs."""
        idata = _make_simple_linear_idata()
        # Remove observed_data to force fallback to model._y
        del idata["observed_data"]

        class FakeModel:
            inference_data = idata
            pymc_model = None
            _y = np.zeros(30)  # n_obs = 30

        df = bayes_factor_compare_models(
            {"OLS": FakeModel()},
            method="bic",
        )
        assert df.shape == (1, 1)
        assert np.isfinite(df.iloc[0, 0])

    def test_unfitted_model_raises(self):
        """Model without inference_data raises ValueError."""

        class UnfittedModel:
            inference_data = None
            pymc_model = None

        with pytest.raises(ValueError, match="no inference_data"):
            bayes_factor_compare_models([UnfittedModel()], method="bridge")


# ---------------------------------------------------------------------------
# Test: compile_log_posterior (requires PyMC)
# ---------------------------------------------------------------------------


class TestCompileLogPosterior:
    @pytest.fixture
    def pymc_model(self):
        """Create a simple PyMC model for testing."""
        import pymc as pm

        rng = np.random.default_rng(42)
        n = 30
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + rng.normal(scale=0.5, size=n)

        with pm.Model() as model:
            beta = pm.Normal("beta", mu=0, sigma=100, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=10)
            mu = pm.math.dot(X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        return model

    def test_returns_callable(self, pymc_model):
        logp_fn, names, info, to_unconstrained = compile_log_posterior(pymc_model)
        assert callable(logp_fn)
        assert isinstance(names, list)
        assert len(names) > 0
        assert "shapes" in info
        assert "sizes" in info
        assert callable(to_unconstrained)

    def test_evaluates_at_theta(self, pymc_model):
        logp_fn, names, info, to_unconstrained = compile_log_posterior(pymc_model)
        # Build a flat theta vector
        total_size = sum(info["sizes"][n] for n in names)
        theta = np.zeros(total_size)
        result = logp_fn(theta)
        assert np.isfinite(result), f"log_posterior returned {result}"

    def test_different_theta_gives_different_logp(self, pymc_model):
        logp_fn, names, info, to_unconstrained = compile_log_posterior(pymc_model)
        total_size = sum(info["sizes"][n] for n in names)
        theta1 = np.zeros(total_size)
        theta2 = np.ones(total_size)
        lp1 = logp_fn(theta1)
        lp2 = logp_fn(theta2)
        assert lp1 != lp2, "Different theta should give different log-posterior"

    def test_constrained_to_unconstrained(self, pymc_model):
        """Test that constrained_to_unconstrained correctly transforms samples."""
        import pymc as pm

        logp_fn, names, info, to_unconstrained = compile_log_posterior(pymc_model)

        # Create fake posterior samples in constrained space
        # The model has: beta (shape=2, no transform), sigma (HalfNormal, LogTransform)
        n_chains, n_draws = 2, 100
        rng = np.random.default_rng(42)
        beta_samples = rng.normal(size=(n_chains, n_draws, 2))
        sigma_samples = np.abs(
            rng.normal(size=(n_chains, n_draws))
        )  # constrained (positive)

        posterior = az.from_dict(
            posterior={
                "beta": beta_samples,
                "sigma": sigma_samples.reshape(n_chains, n_draws, 1),
            }
        ).posterior

        # Transform to unconstrained space
        unconstrained = to_unconstrained(posterior)
        assert unconstrained.shape == (n_chains * n_draws, 3)  # 2 beta + 1 sigma_log__

        # sigma_log__ should be log(sigma)
        sigma_log_from_transform = unconstrained[:, 2]
        sigma_log_expected = np.log(sigma_samples.ravel())
        np.testing.assert_allclose(
            sigma_log_from_transform, sigma_log_expected, rtol=1e-10
        )

        # beta should be unchanged (identity transform)
        beta_from_transform = unconstrained[:, :2]
        beta_expected = beta_samples.reshape(-1, 2)
        np.testing.assert_allclose(beta_from_transform, beta_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test: ModelComparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    def _make_fake_model(self, idata):
        class _FakeModel:
            inference_data = None
            pymc_model = None
            _y = np.zeros(30)

        m = _FakeModel()
        m.inference_data = idata
        return m

    def test_dict_input_and_repr(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        cmp = ModelComparison({"OLS": m, "OLS2": m}, method="bic")
        assert cmp.labels == ["OLS", "OLS2"]
        assert "n_models=2" in repr(cmp)

    def test_list_with_labels(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        cmp = ModelComparison([m, m], labels=["A", "B"], method="bic")
        assert cmp.labels == ["A", "B"]

    def test_list_without_labels_autogen(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        cmp = ModelComparison([m], method="bic")
        assert cmp.labels == ["model_0"]

    def test_dict_with_labels_raises(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        with pytest.raises(ValueError, match="dict keys or the `labels`"):
            ModelComparison({"A": m}, labels=["B"])

    def test_label_length_mismatch_raises(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        with pytest.raises(ValueError, match="must match length"):
            ModelComparison([m, m], labels=["only_one"])

    def test_bad_type_raises(self):
        from neighbayes.diagnostics import ModelComparison

        with pytest.raises(TypeError, match="must be a list"):
            ModelComparison(42)

    def test_bic_workflow(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m1 = self._make_fake_model(idata)
        m2 = self._make_fake_model(idata)
        cmp = ModelComparison({"M1": m1, "M2": m2}, method="bic")

        bf = cmp.bayes_factors
        assert bf.shape == (2, 2)
        assert list(bf.index) == ["M1", "M2"]
        # Identical models => BF ~ 1
        np.testing.assert_allclose(bf.to_numpy(), np.ones((2, 2)), atol=1e-6)

        logmls = cmp.log_marginal_likelihoods
        assert isinstance(logmls, pd.Series)
        assert list(logmls.index) == ["M1", "M2"]

        probs = cmp.posterior_probabilities()
        assert isinstance(probs, pd.Series)
        np.testing.assert_allclose(probs.sum(), 1.0)
        np.testing.assert_allclose(probs.to_numpy(), [0.5, 0.5], atol=1e-6)

    def test_compute_is_cached(self):
        from neighbayes.diagnostics import ModelComparison

        idata = _make_simple_linear_idata()
        m = self._make_fake_model(idata)
        cmp = ModelComparison({"M1": m}, method="bic")
        _ = cmp.bayes_factors
        cached = cmp._bf_df
        _ = cmp.posterior_probabilities()
        assert cmp._bf_df is cached
