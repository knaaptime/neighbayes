"""Fast build/method tests for SARLogit and SEMLogit."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes import dgp
from neighbayes.models.cross_section.sar_logit import SARLogit
from neighbayes.models.cross_section.sar_logit_structural import SARLogitStructural
from neighbayes.models.cross_section.sem_logit import SEMLogit
from neighbayes.models.priors import SARLogitPriors, SEMLogitPriors
from neighbayes.tests.helpers import W_to_graph, make_line_W


def _binary_data(seed: int = 101):
    """Create a small binary dataset for testing."""
    rng = np.random.default_rng(seed)
    n = 20
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    eta = 0.3 + 0.6 * x1
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, probs).astype(float)
    W = W_to_graph(make_line_W(n))
    return y, X, W


def _idata(vars_dict: dict[str, np.ndarray]) -> az.InferenceData:
    payload = {k: np.asarray(v)[None, ...] for k, v in vars_dict.items()}
    return az.from_dict(posterior=payload)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    """SARLogit construction and validation."""

    def test_construct_matrix_mode(self):
        """Model can be constructed in matrix mode."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        assert model._y.shape == (20,)
        assert model._X.shape == (20, 2)

    def test_rejects_non_binary_y(self):
        """Model should reject y with values outside {0, 1}."""
        _, X, W = _binary_data()
        y_bad = np.array([0.0, 0.5, 1.0, 2.0] * 5)
        with pytest.raises(ValueError, match="binary"):
            SARLogit(y=y_bad, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True should raise NotImplementedError."""
        y, X, W = _binary_data()
        with pytest.raises(NotImplementedError, match="robust"):
            SARLogit(y=y, X=X, W=W, robust=True)

    def test_priors_cls(self):
        """Model should use SARLogitPriors."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        assert model._priors_cls is SARLogitPriors

    def test_spatial_params(self):
        """Model should declare rho as spatial parameter."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        assert model._spatial_params == ("rho",)
        assert model._jacobian_param == "rho"
        assert model._model_type == "sar_logit"


# ---------------------------------------------------------------------------
# _build_pymc_model test
# ---------------------------------------------------------------------------


class TestBuildPyMCModel:
    """_build_pymc_model should raise NotImplementedError."""

    def test_raises(self):
        """Gibbs-only model should not build a PyMC model."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="Gibbs"):
            model._build_pymc_model()


# ---------------------------------------------------------------------------
# fit() kwarg rejection test
# ---------------------------------------------------------------------------


class TestFitKwargRejection:
    """fit() should reject NUTS-specific kwargs."""

    def test_rejects_nuts_sampler(self):
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(nuts_sampler="numpyro")

    def test_rejects_target_accept(self):
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(target_accept=0.95)


# ---------------------------------------------------------------------------
# fitted_probabilities test
# ---------------------------------------------------------------------------


class TestFittedProbabilities:
    """fitted_probabilities should return probabilities in [0, 1]."""

    def test_output_range(self):
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        n = X.shape[0]
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        probs = model.fitted_probabilities()
        assert probs.shape == (n,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.all(np.isfinite(probs))


# ---------------------------------------------------------------------------
# spatial_effects test
# ---------------------------------------------------------------------------


class TestSpatialEffects:
    """spatial_effects should compute direct/indirect/total impacts."""

    def test_probability_scale_eigen(self):
        """``scale='probability'`` returns response-scale impacts via eigen path."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        df_logodds = model.spatial_effects(scale="logodds")
        df_prob = model.spatial_effects(scale="probability", method="eigen")
        # Probability-scale effects are bounded above by log-odds effects in
        # absolute value because the multiplier p*(1-p) ≤ 1/4 < 1.
        means_lo = df_logodds["direct"].to_numpy()
        means_pr = df_prob["direct"].to_numpy()
        assert np.all(np.abs(means_pr) <= np.abs(means_lo) + 1e-9)
        assert np.all(np.isfinite(means_pr))
        assert df_prob.attrs.get("scale") == "probability"

    def test_probability_scale_sparse_matches_eigen(self):
        """Sparse Hutchinson path agrees with eigen path within MC tolerance."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        df_eig = model.spatial_effects(scale="probability", method="eigen")
        df_spr = model.spatial_effects(scale="probability", method="sparse")
        # Hutchinson with 20 probes is noisy; check sign + order of magnitude.
        e = df_eig["direct"].to_numpy()
        s = df_spr["direct"].to_numpy()
        np.testing.assert_allclose(s, e, rtol=0.3, atol=1e-3)

    def test_probability_scale_invalid_scale(self):
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "rho": np.array([0.15, 0.16]),
            }
        )
        with pytest.raises(ValueError, match="logodds.*probability"):
            model.spatial_effects(scale="bogus")


# ---------------------------------------------------------------------------
# DGP integration test
# ---------------------------------------------------------------------------


class TestDGPIntegration:
    """simulate_sar_logit should produce valid data."""

    def test_dgp_output(self):
        out = dgp.simulate_sar_logit(n=5, rho=0.3, seed=42)
        assert "y" in out
        assert "X" in out
        assert "W_sparse" in out
        assert "params_true" in out
        n_obs = out["y"].shape[0]
        assert n_obs == 25  # 5x5 grid
        assert set(np.unique(out["y"])) <= {0.0, 1.0}
        assert out["params_true"]["rho"] == 0.3

    def test_model_from_dgp(self):
        """Model should accept data from simulate_sar_logit."""
        out = dgp.simulate_sar_logit(n=5, rho=0.3, seed=42)
        model = SARLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        assert model._y.shape == (25,)


# ---------------------------------------------------------------------------
# SEMLogit tests
# ---------------------------------------------------------------------------


class TestSEMConstruction:
    """SEMLogit construction and validation."""

    def test_construct_matrix_mode(self):
        """Model can be constructed in matrix mode."""
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        assert model._y.shape == (20,)
        assert model._X.shape == (20, 2)

    def test_rejects_non_binary_y(self):
        """Model should reject y with values outside {0, 1}."""
        _, X, W = _binary_data()
        y_bad = np.array([0.0, 0.5, 1.0, 2.0] * 5)
        with pytest.raises(ValueError, match="binary"):
            SEMLogit(y=y_bad, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True should raise NotImplementedError."""
        y, X, W = _binary_data()
        with pytest.raises(NotImplementedError, match="robust"):
            SEMLogit(y=y, X=X, W=W, robust=True)

    def test_priors_cls(self):
        """Model should use SEMLogitPriors."""
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        assert model._priors_cls is SEMLogitPriors

    def test_spatial_params(self):
        """Model should declare lam as spatial parameter."""
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        assert model._spatial_params == ("lam",)
        assert model._jacobian_param == "lam"
        assert model._model_type == "sem_logit"


class TestSEMBuildPyMCModel:
    """_build_pymc_model should raise NotImplementedError."""

    def test_raises(self):
        """Gibbs-only model should not build a PyMC model."""
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="Gibbs"):
            model._build_pymc_model()


class TestSEMFitKwargRejection:
    """fit() should reject NUTS-specific kwargs."""

    def test_rejects_nuts_sampler(self):
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(nuts_sampler="numpyro")

    def test_rejects_target_accept(self):
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(target_accept=0.95)


class TestSEMFittedProbabilities:
    """fitted_probabilities should return probabilities in [0, 1]."""

    def test_output_range(self):
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        n = X.shape[0]
        model._idata = _idata(
            {
                "beta": np.stack([np.array([0.2, 0.7]), np.array([0.21, 0.71])]),
                "lam": np.array([0.15, 0.16]),
            }
        )
        probs = model.fitted_probabilities()
        assert probs.shape == (n,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.all(np.isfinite(probs))


class TestSEMSpatialEffects:
    """spatial_effects should compute direct/indirect/total impacts."""


class TestSEMDGPIntegration:
    """simulate_sem_logit should produce valid data."""

    def test_dgp_output(self):
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        assert "y" in out
        assert "X" in out
        assert "W_sparse" in out
        assert "params_true" in out
        n_obs = out["y"].shape[0]
        assert n_obs == 25  # 5x5 grid
        assert set(np.unique(out["y"])) <= {0.0, 1.0}
        assert out["params_true"]["lam"] == 0.3

    def test_model_from_dgp(self):
        """Model should accept data from simulate_sem_logit."""
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        model = SEMLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        assert model._y.shape == (25,)


class TestSEMFitIntegration:
    """SEMLogit.fit() should produce valid InferenceData."""

    def test_fit_small(self):
        """Fit on small DGP data should return valid InferenceData."""
        out = dgp.simulate_sem_logit(n=5, lam=0.3, seed=42)
        model = SEMLogit(y=out["y"], X=out["X"], W=out["W_graph"])
        idata = model.fit(draws=50, tune=50, chains=2, random_seed=42)
        assert "lam" in idata.posterior
        assert "beta" in idata.posterior
        assert idata.posterior["lam"].shape[0] == 2  # chains
        assert idata.posterior["beta"].shape[-1] == 2  # k coefficients


class TestProgressForwarding:
    """Progress callbacks should be forwarded into logit chain runners."""

    def test_sar_fit_forwards_progress_manager(self, monkeypatch):
        """SARLogit.fit should pass progress_manager through to run_chain."""
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        seen: dict[str, object] = {}

        def _fake_run_chain(
            y,
            X,
            W_sparse,
            priors,
            cache,
            init,
            draws,
            tune,
            thin=1,
            return_eta=False,
            rng=None,
            progress_manager=None,
            chain_id=0,
            store_log_lik=True,
        ):
            seen["progress_manager"] = progress_manager
            seen["chain_id"] = chain_id
            n_keep = draws // thin if thin > 0 else draws
            n, k = X.shape
            return {
                "rho": np.zeros(n_keep),
                "beta": np.zeros((n_keep, k)),
                "log_lik": np.zeros((n_keep, n)),
                "eta_norm": np.zeros(n_keep),
            }

        monkeypatch.setattr(
            "neighbayes.models.cross_section.sar_logit.run_chain",
            _fake_run_chain,
        )

        model.fit(
            draws=4,
            tune=2,
            chains=1,
            random_seed=0,
            n_jobs=1,
            progressbar=True,
            gibbs_backend="numpy",
        )

        assert seen["progress_manager"] is not None
        assert seen["chain_id"] == 0

    def test_sem_fit_forwards_progress_manager(self, monkeypatch):
        """SEMLogit.fit should pass progress_manager through to run_chain_sem."""
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        seen: dict[str, object] = {}

        def _fake_run_chain_sem(
            y,
            X,
            W_sparse,
            priors,
            cache,
            init,
            draws,
            tune,
            thin=1,
            return_eta=False,
            rng=None,
            progress_manager=None,
            chain_id=0,
            store_log_lik=True,
        ):
            seen["progress_manager"] = progress_manager
            seen["chain_id"] = chain_id
            n_keep = draws // thin if thin > 0 else draws
            n, k = X.shape
            return {
                "lam": np.zeros(n_keep),
                "beta": np.zeros((n_keep, k)),
                "log_lik": np.zeros((n_keep, n)),
                "eta_norm": np.zeros(n_keep),
            }

        monkeypatch.setattr(
            "neighbayes.models.cross_section.sem_logit.run_chain_sem",
            _fake_run_chain_sem,
        )

        model.fit(
            draws=4,
            tune=2,
            chains=1,
            random_seed=0,
            n_jobs=1,
            progressbar=True,
            gibbs_backend="numpy",
        )

        assert seen["progress_manager"] is not None
        assert seen["chain_id"] == 0


# ---------------------------------------------------------------------------
# JAX vmap (gibbs_backend="jax") path
# ---------------------------------------------------------------------------


jax = pytest.importorskip("jax")


class TestJaxVectorizedFit:
    """The jax_dense path should run all chains together via jax.vmap."""

    def test_sar_jax_dense_runs_and_shapes(self):
        y, X, W = _binary_data()
        model = SARLogit(y=y, X=X, W=W)
        idata = model.fit(
            draws=20,
            tune=10,
            chains=2,
            random_seed=0,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        assert rho.shape == (2, 20)
        assert beta.shape == (2, 20, X.shape[1])
        # Distinct chains should not be byte-identical.
        assert not np.allclose(rho[0], rho[1])

    def test_sar_jax_dense_rejects_return_eta(self):
        # return_eta is a structural-only option (the reduced-form SARLogit has
        # no latent field); it lives on SARLogitStructural's jax_dense path.
        y, X, W = _binary_data()
        model = SARLogitStructural(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="return_eta"):
            model.fit(
                draws=4,
                tune=2,
                chains=1,
                random_seed=0,
                n_jobs=1,
                progressbar=False,
                gibbs_backend="jax",
                return_eta=True,
            )

    def test_sem_jax_dense_runs_and_shapes(self):
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        idata = model.fit(
            draws=20,
            tune=10,
            chains=2,
            random_seed=0,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        lam = idata.posterior["lam"].values
        beta = idata.posterior["beta"].values
        assert lam.shape == (2, 20)
        assert beta.shape == (2, 20, X.shape[1])
        assert not np.allclose(lam[0], lam[1])

    def test_sem_jax_dense_rejects_return_eta(self):
        y, X, W = _binary_data()
        model = SEMLogit(y=y, X=X, W=W)
        with pytest.raises(NotImplementedError, match="return_eta"):
            model.fit(
                draws=4,
                tune=2,
                chains=1,
                random_seed=0,
                n_jobs=1,
                progressbar=False,
                gibbs_backend="jax",
                return_eta=True,
            )
