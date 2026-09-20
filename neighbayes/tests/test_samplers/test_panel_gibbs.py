"""Tests for panel FE Gibbs sampler integration.

Covers:
- SARPanelFE and SEMPanelFE Gibbs dispatch
- Valid InferenceData output from panel Gibbs
- Panel Gibbs vs NUTS posterior agreement (slow)
- Edge cases (robust raises, intercept drop)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    W_to_graph,
    make_line_W,
    make_panel_sar_data,
    make_panel_sem_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel_W():
    """Create a small row-standardized W for panel tests."""
    W_dense = make_line_W(PANEL_N)
    W_graph = W_to_graph(W_dense)
    return W_dense, W_graph


# ---------------------------------------------------------------------------
# Fast integration tests
# ---------------------------------------------------------------------------


class TestPanelGibbsDispatch:
    """Panel models should dispatch sampler='gibbs' to _fit_gibbs."""

    def test_sar_panel_fe_gibbs_produces_idata(self):
        """SARPanelFE.fit(sampler='gibbs') returns valid InferenceData."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        assert "posterior" in idata.groups()
        assert "rho" in idata.posterior.data_vars
        assert "beta" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert "log_likelihood" in idata.groups()
        # Beta should have length equal to non-intercept columns
        assert idata.posterior["beta"].shape[-1] == X.shape[1] - 1

    def test_sem_panel_fe_gibbs_produces_idata(self):
        """SEMPanelFE.fit(sampler='gibbs') returns valid InferenceData."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=50,
            tune=20,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        assert "posterior" in idata.groups()
        assert "lam" in idata.posterior.data_vars
        assert "beta" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert "log_likelihood" in idata.groups()
        assert idata.posterior["beta"].shape[-1] == X.shape[1] - 1

    def test_sar_panel_fe_gibbs_multiple_chains(self):
        """SARPanelFE Gibbs works with multiple chains."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=30,
            tune=10,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert idata.posterior["rho"].shape[0] == 2  # 2 chains

    def test_sem_panel_fe_gibbs_multiple_chains(self):
        """SEMPanelFE Gibbs works with multiple chains."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=30,
            tune=10,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert idata.posterior["lam"].shape[0] == 2  # 2 chains


class TestPanelGibbsEdgeCases:
    """Edge cases for panel Gibbs."""

    def test_sar_panel_fe_robust_raises(self):
        """Gibbs should raise for robust SARPanelFE."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(
            y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1, robust=True
        )
        with pytest.raises(NotImplementedError, match="robust"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                progressbar=False,
            )

    def test_sem_panel_fe_robust_raises(self):
        """Gibbs should raise for robust SEMPanelFE."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(
            y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1, robust=True
        )
        with pytest.raises(NotImplementedError, match="robust"):
            model.fit(
                sampler="gibbs",
                draws=10,
                tune=5,
                chains=1,
                progressbar=False,
            )

    def test_sar_panel_fe_gibbs_thinning(self):
        """Thinning should reduce kept draws."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=60,
            tune=10,
            thin=2,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        # 60 draws with thin=2 → 30 kept
        assert idata.posterior["rho"].shape[1] == 30

    def test_sem_panel_fe_gibbs_thinning(self):
        """Thinning should reduce kept draws."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=60,
            tune=10,
            thin=2,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        assert idata.posterior["lam"].shape[1] == 30


# ---------------------------------------------------------------------------
# Slow recovery tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPanelGibbsRecovery:
    """Panel Gibbs should recover true parameters."""

    def test_sar_panel_fe_gibbs_recovers_rho(self):
        """SARPanelFE Gibbs posterior mean of rho near true value."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        rho_true = 0.4
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=rho_true,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=500,
            tune=300,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        rho_hat = float(idata.posterior["rho"].mean())
        assert abs(rho_hat - rho_true) < 0.25, (
            f"SARPanelFE Gibbs rho: expected ≈{rho_true}, got {rho_hat:.3f}"
        )

    def test_sem_panel_fe_gibbs_recovers_lam(self):
        """SEMPanelFE Gibbs posterior mean of lam near true value."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        lam_true = 0.3
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=lam_true,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=500,
            tune=300,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )
        lam_hat = float(idata.posterior["lam"].mean())
        assert abs(lam_hat - lam_true) < 0.25, (
            f"SEMPanelFE Gibbs lam: expected ≈{lam_true}, got {lam_hat:.3f}"
        )


@pytest.mark.slow
class TestPanelGibbsVsNUTS:
    """Panel Gibbs and NUTS posteriors should agree."""

    def test_sar_panel_fe_gibbs_vs_nuts(self):
        """SARPanelFE Gibbs and NUTS means should be close."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )

        model_nuts = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=300,
            tune=300,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
        )

        model_gibbs = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=300,
            tune=300,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["rho", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            assert abs(mean_nuts - mean_gibbs) < 0.2, (
                f"{param}: NUTS={mean_nuts:.3f}, Gibbs={mean_gibbs:.3f}"
            )

        beta_nuts = idata_nuts.posterior["beta"].mean(dim=["chain", "draw"]).values
        beta_gibbs = idata_gibbs.posterior["beta"].mean(dim=["chain", "draw"]).values
        assert np.allclose(beta_nuts, beta_gibbs, atol=0.2), (
            f"beta: NUTS={beta_nuts}, Gibbs={beta_gibbs}"
        )

    def test_sem_panel_fe_gibbs_vs_nuts(self):
        """SEMPanelFE Gibbs and NUTS means should be close."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )

        model_nuts = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata_nuts = model_nuts.fit(
            sampler="nuts",
            draws=300,
            tune=300,
            chains=2,
            random_seed=42,
            target_accept=0.9,
            progressbar=False,
        )

        model_gibbs = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata_gibbs = model_gibbs.fit(
            sampler="gibbs",
            draws=300,
            tune=300,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
        )

        for param in ["lam", "sigma"]:
            mean_nuts = float(idata_nuts.posterior[param].mean())
            mean_gibbs = float(idata_gibbs.posterior[param].mean())
            assert abs(mean_nuts - mean_gibbs) < 0.2, (
                f"{param}: NUTS={mean_nuts:.3f}, Gibbs={mean_gibbs:.3f}"
            )

        beta_nuts = idata_nuts.posterior["beta"].mean(dim=["chain", "draw"]).values
        beta_gibbs = idata_gibbs.posterior["beta"].mean(dim=["chain", "draw"]).values
        assert np.allclose(beta_nuts, beta_gibbs, atol=0.2), (
            f"beta: NUTS={beta_nuts}, Gibbs={beta_gibbs}"
        )


# ---------------------------------------------------------------------------
# JAX path tests
# ---------------------------------------------------------------------------


class TestPanelGibbsJAX:
    """JAX JIT path for panel Gibbs."""

    def test_sar_panel_fe_jax_produces_idata(self):
        """SARPanelFE Gibbs with gibbs_backend='jax' returns valid InferenceData."""
        pytest.importorskip("jax")
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=30,
            tune=10,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "posterior" in idata.groups()
        assert "rho" in idata.posterior.data_vars
        assert "beta" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert idata.posterior["beta"].shape[-1] == X.shape[1] - 1

    def test_sem_panel_fe_jax_produces_idata(self):
        """SEMPanelFE Gibbs with gibbs_backend='jax' returns valid InferenceData."""
        pytest.importorskip("jax")
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=30,
            tune=10,
            chains=1,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
        )
        assert "posterior" in idata.groups()
        assert "lam" in idata.posterior.data_vars
        assert "beta" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars
        assert idata.posterior["beta"].shape[-1] == X.shape[1] - 1

    def test_sar_panel_fe_jax_vectorized_chains(self):
        """SARPanelFE JAX Gibbs with vectorized chains."""
        pytest.importorskip("jax")
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sar_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            rho=0.4,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SARPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=20,
            tune=10,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
            chain_method="vectorized",
        )
        assert idata.posterior["rho"].shape[0] == 2  # 2 chains

    def test_sem_panel_fe_jax_vectorized_chains(self):
        """SEMPanelFE JAX Gibbs with vectorized chains."""
        pytest.importorskip("jax")
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(42)
        W_dense, W_graph = _make_panel_W()
        y, X, _ = make_panel_sem_data(
            rng,
            W_dense,
            PANEL_N,
            PANEL_T,
            lam=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.8,
        )
        model = SEMPanelFE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1)
        idata = model.fit(
            sampler="gibbs",
            draws=20,
            tune=10,
            chains=2,
            random_seed=42,
            n_jobs=1,
            progressbar=False,
            gibbs_backend="jax",
            chain_method="vectorized",
        )
        assert idata.posterior["lam"].shape[0] == 2  # 2 chains
