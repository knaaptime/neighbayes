"""Parameter recovery tests for SARNegBinStructural (Gibbs sampler).

These tests verify that the Pólya–Gamma Gibbs sampler recovers known
parameters from simulated SAR-NB data. They are marked ``slow`` and
``recovery`` because they run MCMC.

Run with::

    pytest -m recovery              # recovery tests only
    pytest -m slow                   # all slow tests
    pytest test_sar_negbin_latent.py # this file only
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes import dgp
from neighbayes.models import SARNegBin, SARNegBinStructural
from neighbayes.tests.helpers import W_to_graph, _as_sparse, make_rook_W

# ---------------------------------------------------------------------------
# Fast build / validation tests (not marked slow)
# ---------------------------------------------------------------------------


class TestSARNegBinStructuralBuild:
    """Construction and validation tests (no sampling)."""

    def test_build_model(self):
        """SARNegBinStructural constructs without error."""
        n = 10
        W = W_to_graph(make_rook_W(3))  # 9 units
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.poisson(2, size=n).astype(float)
        model = SARNegBinStructural(y=y[:9], X=X[:9], W=W)
        assert model is not None

    def test_rejects_noninteger_y(self):
        """Non-integer y raises ValueError."""
        n = 10
        W = W_to_graph(make_rook_W(3))
        y = np.array([0.5, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 0.0])
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(ValueError, match="integer-valued"):
            SARNegBinStructural(y=y, X=X, W=W)

    def test_rejects_negative_y(self):
        """Negative y raises ValueError."""
        W = W_to_graph(make_rook_W(3))
        y = np.array([0, 1, -1, 2, 0, 3, 1, 2, 0], dtype=float)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(ValueError, match="non-negative"):
            SARNegBinStructural(y=y, X=X, W=W)

    def test_rejects_robust(self):
        """robust=True raises NotImplementedError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        with pytest.raises(NotImplementedError, match="robust"):
            SARNegBinStructural(y=y, X=X, W=W, robust=True)

    def test_rejects_nuts_kwargs(self):
        """NUTS-specific kwargs raise TypeError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        model = SARNegBinStructural(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="nuts_sampler"):
            model.fit(draws=10, nuts_sampler="blackjax")

    def test_rejects_target_accept(self):
        """target_accept kwarg raises TypeError."""
        W = W_to_graph(make_rook_W(3))
        y = np.ones(9)
        X = np.column_stack([np.ones(9), np.random.randn(9)])
        model = SARNegBinStructural(y=y, X=X, W=W)
        with pytest.raises(TypeError, match="target_accept"):
            model.fit(draws=10, target_accept=0.95)


# ---------------------------------------------------------------------------
# Parameter recovery tests (marked slow/recovery)
# ---------------------------------------------------------------------------

# Larger grid for parameter recovery — n=1024 gives enough information
# for the Gibbs sampler to recover parameters with tight tolerances.
# (With only n=100 the NB dispersion ``alpha`` is weakly identified and
# the sampler can drift toward ``alpha ≈ 0``, which breaks the
# Pólya–Gamma draw used in the omega update.)
SIDE = 32  # 1024 cross-sectional units
RHO_TRUE = 0.4
BETA_TRUE = np.array([0.5, 0.8])
ALPHA_TRUE = 2.0
SIGMA2_TRUE = 0.5  # structural-form residual variance
DRAWS = 1000
TUNE = 1000
CHAINS = 2


@pytest.fixture(scope="module")
def sar_nb_data():
    """Simulated SAR-NB data from the structural-form DGP (n=1024)."""
    rng = np.random.default_rng(42)
    W_dense = make_rook_W(SIDE)
    W_graph = W_to_graph(W_dense)
    return dgp.simulate_sar_negbin(
        W=_as_sparse(W_graph),
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        alpha=ALPHA_TRUE,
        sigma2=SIGMA2_TRUE,
        rng=rng,
    )


@pytest.mark.slow
@pytest.mark.recovery
class TestSARNegBinStructuralRecovery:
    """Parameter recovery tests for the Gibbs sampler.

    Uses n=1024 (32×32 rook grid) with 1000 draws × 2 chains so that
    posterior summaries are precise enough for meaningful tolerance checks.
    """

    def test_fit_returns_idata(self, sar_nb_data):
        """fit() returns a DataTree with the expected groups.

        The pointwise log-likelihood is stored only on request (as in PyMC),
        so the fit asks for it.
        """
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        assert "posterior" in idata
        assert "log_likelihood" in idata
        assert "observed_data" in idata
        assert "rho" in idata.posterior
        assert "beta" in idata.posterior
        assert "sigma" in idata.posterior
        assert "alpha" in idata.posterior

    def test_posterior_shapes(self, sar_nb_data):
        """Posterior arrays have correct shapes."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]
        n_chains = CHAINS
        n_draws = DRAWS

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=n_draws,
            tune=TUNE,
            chains=n_chains,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        assert idata.posterior["rho"].shape == (n_chains, n_draws)
        assert idata.posterior["beta"].shape == (n_chains, n_draws, X.shape[1])
        assert idata.posterior["sigma"].shape == (n_chains, n_draws)
        assert idata.posterior["alpha"].shape == (n_chains, n_draws)

    def test_rho_recovery(self, sar_nb_data):
        """Posterior mean for rho is within 0.2 of the true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        rho_mean = float(idata.posterior["rho"].mean())
        # The collapsed ρ update integrates out η, giving much better
        # mixing than the conditional update.  With n=1024 and
        # sigma2=0.5, the posterior mean should be within 0.2 of truth.
        assert abs(rho_mean - RHO_TRUE) < 0.2, (
            f"rho_mean={rho_mean:.3f} too far from rho_true={RHO_TRUE}"
        )

    def test_alpha_recovery(self, sar_nb_data):
        """Posterior mean for alpha is within 1.5 of the true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        assert alpha_mean > 0, f"alpha_mean={alpha_mean:.3f} should be positive"
        assert abs(alpha_mean - ALPHA_TRUE) < 1.5, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )

    def test_beta_recovery(self, sar_nb_data):
        """Posterior mean for slope coefficient is within 0.4 of true value."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        # Check slope (beta[1]) is within 0.4 of true value
        assert abs(beta_mean[1] - BETA_TRUE[1]) < 0.4, (
            f"slope={beta_mean[1]:.3f} too far from true={BETA_TRUE[1]}"
        )

    def test_spatial_effects(self, sar_nb_data):
        """spatial_effects() returns a DataFrame with expected columns."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        model.fit(
            draws=200, tune=200, chains=1, random_seed=42, n_jobs=1, progressbar=False
        )

        effects = model.spatial_effects()
        assert "direct" in effects.columns
        assert "indirect" in effects.columns
        assert "total" in effects.columns
        assert np.all(np.isfinite(effects["direct"].values))

    def test_return_eta(self, sar_nb_data):
        """return_eta=True stores the latent field in the posterior."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]
        n = len(y)

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=50,
            tune=50,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            return_eta=True,
            gibbs_backend="numpy",
        )

        assert "eta" in idata.posterior
        assert idata.posterior["eta"].shape[-1] == n

    def test_thinning(self, sar_nb_data):
        """thin=2 halves the number of stored draws."""
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=100,
            tune=100,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            thin=2,
        )

        # With thin=2, we keep 100/2 = 50 draws
        assert idata.posterior["rho"].shape[1] == 50


# ---------------------------------------------------------------------------
# JAX vmap (gibbs_backend="jax") path
# ---------------------------------------------------------------------------


jax = pytest.importorskip("jax")


@pytest.mark.slow
class TestSARNegBinStructuralJaxVectorized:
    """The jax_dense path should run all chains together via jax.vmap."""

    def test_jax_dense_runs_and_shapes(self, sar_nb_data):
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=40,
            tune=20,
            chains=2,
            random_seed=0,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        rho = idata.posterior["rho"].values
        beta = idata.posterior["beta"].values
        sigma = idata.posterior["sigma"].values
        alpha = idata.posterior["alpha"].values
        assert rho.shape == (2, 40)
        assert beta.shape == (2, 40, X.shape[1])
        assert sigma.shape == (2, 40)
        assert alpha.shape == (2, 40)
        # Distinct chains should not be byte-identical.
        assert not np.allclose(rho[0], rho[1])

    def test_jax_dense_rejects_return_eta(self, sar_nb_data):
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]
        model = SARNegBinStructural(y=y, X=X, W=W)
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


@pytest.mark.slow
@pytest.mark.recovery
@pytest.mark.requires_jax
class TestSARNegBinStructuralJaxDenseRecovery:
    """Parameter recovery through the ``jax_dense`` path.

    Tombstone for the truncated Gamma-series PG mean bias: without the
    closed-form tail-mean correction, the K=25 series under-draws omega
    by ~1%, which accumulates across Gibbs iterations and drags alpha
    toward collapse.
    """

    def test_alpha_rho_recovery_jax_dense(self, sar_nb_data):
        pytest.importorskip("jax")
        y = sar_nb_data["y"]
        X = sar_nb_data["X"]
        W = sar_nb_data["W_graph"]

        model = SARNegBinStructural(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=-1,
            progressbar=False,
            gibbs_backend="numpy",
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        rho_mean = float(idata.posterior["rho"].mean())
        # Same tolerances as the factorize-path recovery tests above.
        assert abs(alpha_mean - ALPHA_TRUE) < 1.5, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )
        assert abs(rho_mean - RHO_TRUE) < 0.2, (
            f"rho_mean={rho_mean:.3f} too far from rho_true={RHO_TRUE}"
        )


# ---------------------------------------------------------------------------
# Reduced-form SARNegBin recovery
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sar_nb_reduced_data():
    """SAR-NB data from the *reduced-form* DGP (sigma2=0, no latent noise)."""
    rng = np.random.default_rng(99)
    W_dense = make_rook_W(SIDE)
    W_graph = W_to_graph(W_dense)
    return dgp.simulate_sar_negbin(
        W=_as_sparse(W_graph),
        rho=RHO_TRUE,
        beta=BETA_TRUE,
        alpha=ALPHA_TRUE,
        sigma2=0.0,  # reduced form: deterministic latent
        rng=rng,
    )


@pytest.mark.slow
@pytest.mark.recovery
class TestSARNegBinReducedRecovery:
    """Parameter recovery for the reduced-form SARNegBin (Gibbs sampler).

    The DGP uses ``sigma2=0`` (no latent noise), matching the reduced-form
    model specification: ``eta = (I - rho W)^{-1} X beta``.
    """

    def test_recovers_rho(self, sar_nb_reduced_data):
        y = sar_nb_reduced_data["y"]
        X = sar_nb_reduced_data["X"]
        W = sar_nb_reduced_data["W_graph"]

        model = SARNegBin(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        rho_mean = float(idata.posterior["rho"].mean())
        assert abs(rho_mean - RHO_TRUE) < 0.2, (
            f"rho_mean={rho_mean:.3f} too far from rho_true={RHO_TRUE}"
        )

    def test_recovers_alpha(self, sar_nb_reduced_data):
        y = sar_nb_reduced_data["y"]
        X = sar_nb_reduced_data["X"]
        W = sar_nb_reduced_data["W_graph"]

        model = SARNegBin(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        alpha_mean = float(idata.posterior["alpha"].mean())
        assert alpha_mean > 0, f"alpha_mean={alpha_mean:.3f} should be positive"
        assert abs(alpha_mean - ALPHA_TRUE) < 1.5, (
            f"alpha_mean={alpha_mean:.3f} too far from alpha_true={ALPHA_TRUE}"
        )

    def test_recovers_beta(self, sar_nb_reduced_data):
        y = sar_nb_reduced_data["y"]
        X = sar_nb_reduced_data["X"]
        W = sar_nb_reduced_data["W_graph"]

        model = SARNegBin(y=y, X=X, W=W)
        idata = model.fit(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        beta_mean = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        assert abs(beta_mean[1] - BETA_TRUE[1]) < 0.4, (
            f"slope={beta_mean[1]:.3f} too far from true={BETA_TRUE[1]}"
        )
