"""Tests for panel flow models.

This file targets the first implementation slice:
- FlowPanelModel behavior through SARFlowPanel
- SARFlowPanel construction and demeaning
- SARFlowSeparablePanel model build
- Parameter recovery tests for all 4 panel flow variants
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models.flow_panel import (
    OLSFlowPanel,
    SARFlowPanel,
    SARFlowSeparablePanel,
)


def _flow_test_graph(n: int):
    """Return a small Graph for tests via neighbayes's DGP machinery."""
    from neighbayes.dgp.flows import generate_flow_data

    return generate_flow_data(n=n, seed=0)["G"]


def _panel_flow_stack(n: int, T: int, k: int, seed: int = 0):
    """Build a noise-only panel-flow stack via neighbayes's panel DGP."""
    from neighbayes.dgp.flows import generate_panel_flow_data

    out = generate_panel_flow_data(
        n=n,
        T=T,
        k=k,
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=np.zeros(k),
        beta_o=np.zeros(k),
        sigma=1.0,
        sigma_alpha=0.0,
        seed=seed,
    )
    return out["y"], out["X"], out["col_names"]


class TestFlowPanelModelConstruction:
    def test_sar_flow_panel_builds(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=10)

        model = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )

        assert model._n == n
        assert model._T == T
        assert model._N_flow == n * n
        assert model._y.shape == (n * n * T,)
        assert model._X.shape[0] == n * n * T
        assert model._Wd_y.shape == (n * n * T,)

    def test_pair_fixed_effect_demeaning_zero_mean_by_pair(self):
        n = 4
        T = 4
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=11)

        model = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=1,
        )

        y2 = model._y.reshape(T, n * n)
        np.testing.assert_allclose(y2.mean(axis=0), 0.0, atol=1e-10)

    def test_invalid_y_length_raises(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=12)

        with pytest.raises(ValueError, match=r"n\^2\*T"):
            SARFlowPanel(
                y=y[:-1],
                W=G,
                X=X,
                T=T,
                col_names=col_names,
            )


class TestSeparablePanelModelBuild:
    def test_sar_flow_separable_panel_builds_pymc_model(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=13)

        model = SARFlowSeparablePanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )

        pm_model = model._build_pymc_model()
        assert pm_model is not None
        assert "rho_w" in pm_model.named_vars


@pytest.mark.slow
def test_sar_flow_panel_fit_smoke():
    """Minimal posterior smoke test for the unrestricted panel flow model."""
    n = 4
    T = 2
    G = _flow_test_graph(n)
    y, X, col_names = _panel_flow_stack(n=n, T=T, k=1, seed=15)

    model = SARFlowPanel(
        y=y,
        W=G,
        X=X,
        T=T,
        col_names=col_names,
        effects=0,
        restrict_positive=True,
    )
    idata = model.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
    assert "rho_d" in idata.posterior
    assert "rho_o" in idata.posterior
    assert "rho_w" in idata.posterior
    assert "beta" in idata.posterior
    assert "sigma" in idata.posterior


# ---------------------------------------------------------------------------
# Parameter recovery tests (slow — deselected by default)
# ---------------------------------------------------------------------------

# Panel flow dimensions — larger n for reliable recovery with the resolvent sampler
PANEL_FLOW_N = 8  # 64 O-D pairs per period
PANEL_FLOW_T = 5  # 5 periods → 320 total obs

# True parameter values
PF_RHO_D_TRUE = 0.25
PF_RHO_O_TRUE = 0.25
PF_RHO_W_TRUE = 0.10
PF_BETA_D_TRUE = np.array([1.0, -0.5])
PF_BETA_O_TRUE = np.array([0.5, 0.3])
PF_SIGMA_TRUE = 1.0
PF_SIGMA_ALPHA_TRUE = 0.5

# Separable-model true values — asymmetric so rho_d ≠ rho_o (breaks swap symmetry)
# rho_w = -0.4*0.3 = -0.12 provides clear identification signal
PF_RHO_D_SEP_TRUE = 0.40
PF_RHO_O_SEP_TRUE = 0.30

# Tolerances — tightened for meaningful recovery
ABS_TOL_RHO = 0.15
ABS_TOL_BETA = 0.30
ABS_TOL_SIGMA = 0.25


@pytest.mark.slow
class TestSARFlowPanelRecovery:
    """SARFlowPanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_panel(self):
        from neighbayes.dgp.flows import generate_panel_flow_data

        G = _flow_test_graph(PANEL_FLOW_N)
        out = generate_panel_flow_data(
            n=PANEL_FLOW_N,
            T=PANEL_FLOW_T,
            G=G,
            rho_d=PF_RHO_D_TRUE,
            rho_o=PF_RHO_O_TRUE,
            rho_w=PF_RHO_W_TRUE,
            beta_d=PF_BETA_D_TRUE,
            beta_o=PF_BETA_O_TRUE,
            sigma=PF_SIGMA_TRUE,
            sigma_alpha=PF_SIGMA_ALPHA_TRUE,
            seed=42,
        )
        # Default panel-flow DGP is lognormal; SARFlowPanel has Gaussian
        # likelihood, so fit on np.log(y) (== eta).
        model = SARFlowPanel(
            y=np.log(out["y"]),
            W=G,
            X=out["X"],
            T=PANEL_FLOW_T,
            col_names=out["col_names"],
            effects=0,
            restrict_positive=True,
        )
        idata = model.fit(
            draws=1000,
            tune=1000,
            chains=2,
            progressbar=False,
            random_seed=42,
            n_probes=48,
            n_quad=8,
        )
        return idata

    def test_sar_flow_panel_recovers_rho_d(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PF_RHO_D_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_d: expected ≈{PF_RHO_D_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_rho_o(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PF_RHO_O_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_o: expected ≈{PF_RHO_O_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_rho_w(self, fitted_panel):
        rho_hat = float(fitted_panel.posterior["rho_w"].mean())
        assert abs(rho_hat - PF_RHO_W_TRUE) < ABS_TOL_RHO, (
            f"SARFlowPanel rho_w: expected ≈{PF_RHO_W_TRUE}, got {rho_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_sigma(self, fitted_panel):
        sigma_hat = float(fitted_panel.posterior["sigma"].mean())
        assert abs(sigma_hat - PF_SIGMA_TRUE) < ABS_TOL_SIGMA, (
            f"SARFlowPanel sigma: expected ≈{PF_SIGMA_TRUE}, got {sigma_hat:.3f}"
        )

    def test_sar_flow_panel_recovers_beta(self, fitted_panel):
        # DGP design layout is [alpha, intra, beta_d (k cols), beta_o (k cols),
        # ..., gamma_dist]; recover the beta_d / beta_o blocks only.
        k = len(PF_BETA_D_TRUE)
        beta_hat = fitted_panel.posterior["beta"].mean(dim=("chain", "draw")).values
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, PF_BETA_D_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowPanel beta_d: expected ≈{PF_BETA_D_TRUE}, got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, PF_BETA_O_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowPanel beta_o: expected ≈{PF_BETA_O_TRUE}, got {beta_o_hat}"
        )


@pytest.mark.slow
class TestSARFlowSeparablePanelRecovery:
    """SARFlowSeparablePanel posterior means should be close to true DGP values."""

    @pytest.fixture(scope="class")
    def fitted_separable_panel(self):
        from neighbayes.dgp.flows import generate_panel_flow_data_separable

        G = _flow_test_graph(PANEL_FLOW_N)
        out = generate_panel_flow_data_separable(
            n=PANEL_FLOW_N,
            T=PANEL_FLOW_T,
            G=G,
            rho_d=PF_RHO_D_SEP_TRUE,
            rho_o=PF_RHO_O_SEP_TRUE,
            beta_d=PF_BETA_D_TRUE,
            beta_o=PF_BETA_O_TRUE,
            sigma=PF_SIGMA_TRUE,
            sigma_alpha=PF_SIGMA_ALPHA_TRUE,
            seed=42,
        )
        # Default panel-flow DGP is lognormal; fit on np.log(y).
        model = SARFlowSeparablePanel(
            y=np.log(out["y"]),
            W=G,
            X=out["X"],
            T=PANEL_FLOW_T,
            col_names=out["col_names"],
            effects=0,
        )
        idata = model.fit(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            progressbar=False,
            random_seed=42,
        )
        return idata

    def test_separable_panel_recovers_rho_d(self, fitted_separable_panel):
        rho_hat = float(fitted_separable_panel.posterior["rho_d"].mean())
        assert abs(rho_hat - PF_RHO_D_SEP_TRUE) < ABS_TOL_RHO, (
            f"SARFlowSeparablePanel rho_d: expected ≈{PF_RHO_D_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_separable_panel_recovers_rho_o(self, fitted_separable_panel):
        rho_hat = float(fitted_separable_panel.posterior["rho_o"].mean())
        assert abs(rho_hat - PF_RHO_O_SEP_TRUE) < ABS_TOL_RHO, (
            f"SARFlowSeparablePanel rho_o: expected ≈{PF_RHO_O_SEP_TRUE}, got {rho_hat:.3f}"
        )

    def test_separable_panel_recovers_beta(self, fitted_separable_panel):
        # Same design layout as SARFlowPanel: [alpha, intra, beta_d (k),
        # beta_o (k), ..., gamma_dist].
        k = len(PF_BETA_D_TRUE)
        beta_hat = (
            fitted_separable_panel.posterior["beta"].mean(dim=("chain", "draw")).values
        )
        beta_d_hat = beta_hat[2 : 2 + k]
        beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
        assert np.allclose(beta_d_hat, PF_BETA_D_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowSeparablePanel beta_d: expected ≈{PF_BETA_D_TRUE}, "
            f"got {beta_d_hat}"
        )
        assert np.allclose(beta_o_hat, PF_BETA_O_TRUE, atol=ABS_TOL_BETA), (
            f"SARFlowSeparablePanel beta_o: expected ≈{PF_BETA_O_TRUE}, "
            f"got {beta_o_hat}"
        )


def _ols_panel_flow_data(n: int, T: int, beta_d, beta_o, sigma, seed=0):
    """Stack T panel-flow periods with rho = 0 via neighbayes's panel DGP."""
    from neighbayes.dgp.flows import generate_panel_flow_data

    out = generate_panel_flow_data(
        n=n,
        T=T,
        k=len(beta_d),
        rho_d=0.0,
        rho_o=0.0,
        rho_w=0.0,
        beta_d=beta_d,
        beta_o=beta_o,
        sigma=sigma,
        sigma_alpha=0.0,
        seed=seed,
    )
    return out["G"], out["y"], out["X"], out["col_names"]


class TestOLSFlowPanelEffects:
    """Tests for the new non-spatial OLSFlowPanel gravity baseline."""

    def test_olsflow_panel_builds_and_skips_logdet_precompute(self):
        n = 4
        T = 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=20)

        model = OLSFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        # Trace / separable precomputation must be skipped.
        assert model._traces is None
        assert model._separable_logdet_fn is None
        assert model._n == n
        assert model._T == T

    def test_olsflow_panel_posterior_predictive_shape(self):
        n = 4
        T = 2
        G, y, X, col_names = _ols_panel_flow_data(
            n=n,
            T=T,
            beta_d=[1.0],
            beta_o=[0.5],
            sigma=0.5,
            seed=1,
        )
        model = OLSFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        y_rep = model.posterior_predictive(n_draws=5, random_seed=3)
        assert y_rep.shape == (5, n * n * T)
        assert np.all(np.isfinite(y_rep))

    @pytest.mark.parametrize("fe_mode", [1, 2])
    def test_olsflow_panel_fe_modes_smoke(self, fe_mode):
        """Pair FE (1) and time FE (2) wire through `_demean_panel` correctly."""
        n = 4
        T = 3
        G, y, X, col_names = _ols_panel_flow_data(
            n=n,
            T=T,
            beta_d=[0.5],
            beta_o=[0.3],
            sigma=0.2,
            seed=fe_mode,
        )
        model = OLSFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=fe_mode,
        )
        model.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0)
        post = model._idata.posterior
        assert "beta" in post.data_vars
        assert "sigma" in post.data_vars
        assert "rho_d" not in post.data_vars
        assert "rho_o" not in post.data_vars


# ---------------------------------------------------------------------------
# Pointwise log-likelihood (with Jacobian correction)
# ---------------------------------------------------------------------------


class TestFlowPanelLogLikelihood:
    """Verify ``log_likelihood`` group is attached for panel flow fits and
    usable for ``az.loo`` / ``az.compare``."""

    def _check_loo(self, idata, n_obs):
        import arviz as az

        assert hasattr(idata, "log_likelihood")
        assert "obs" in idata.log_likelihood.data_vars
        ll = idata.log_likelihood["obs"].values
        assert ll.shape[2] == n_obs
        assert np.isfinite(ll).all()
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd_loo)

    def test_sar_flow_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        idata = m.fit(
            draws=20,
            tune=20,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata, n_obs=n * n * T)

    def test_sar_flow_separable_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = SARFlowSeparablePanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        idata = m.fit(
            draws=20,
            tune=20,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata, n_obs=n * n * T)

    def test_ols_flow_panel_loglik(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        m = OLSFlowPanel(y=y, W=G, X=X, T=T, col_names=col_names, effects=0)
        idata = m.fit(
            draws=20,
            tune=20,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        self._check_loo(idata, n_obs=n * n * T)

    def test_compare_flow_panel_models(self):
        import arviz as az

        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=0)
        kw = dict(
            draws=30,
            tune=30,
            chains=1,
            progressbar=False,
            random_seed=0,
            idata_kwargs={"log_likelihood": True},
        )
        m_sar = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        m_sep = SARFlowSeparablePanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        m_ols = OLSFlowPanel(y=y, W=G, X=X, T=T, col_names=col_names, effects=0)
        idata_sar = m_sar.fit(**kw)
        idata_sep = m_sep.fit(**kw)
        idata_ols = m_ols.fit(**kw)
        comp = az.compare({"sar": idata_sar, "sep": idata_sep, "ols": idata_ols})
        assert "rank" in comp.columns
        assert len(comp) == 3


class TestFlowPanelSamplerDispatch:
    """Flow panel models should dispatch sampler='gibbs'/'nuts' correctly."""

    def test_sar_flow_panel_sampler_gibbs(self):
        """Unrestricted SARFlowPanel with sampler='gibbs' uses the resolvent sampler."""
        n, T = 4, 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=50)
        m = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
            restrict_positive=True,
        )
        idata = m.fit(
            draws=20,
            tune=20,
            chains=1,
            progressbar=False,
            random_seed=0,
            sampler="gibbs",
        )
        assert "beta" in idata.posterior.data_vars
        assert "rho_d" in idata.posterior.data_vars
        assert "rho_o" in idata.posterior.data_vars
        assert "rho_w" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars

    def test_sar_flow_panel_sampler_nuts(self):
        """Unrestricted SARFlowPanel with sampler='nuts' uses PyMC NUTS."""
        n, T = 4, 3
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=51)
        m = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
            restrict_positive=True,
        )
        idata = m.fit(
            draws=10,
            tune=10,
            chains=1,
            progressbar=False,
            random_seed=0,
            sampler="nuts",
        )
        assert "beta" in idata.posterior.data_vars
        assert "rho_d" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars

    def test_sar_flow_separable_panel_sampler_nuts(self):
        """Separable panel with sampler='nuts' uses PyMC NUTS (default)."""
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=52)
        m = SARFlowSeparablePanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
        )
        idata = m.fit(
            draws=10,
            tune=10,
            chains=1,
            progressbar=False,
            random_seed=0,
            sampler="nuts",
        )
        assert "beta" in idata.posterior.data_vars
        assert "rho_d" in idata.posterior.data_vars
        assert "sigma" in idata.posterior.data_vars

    def test_invalid_sampler_raises(self):
        n, T = 4, 2
        G = _flow_test_graph(n)
        y, X, col_names = _panel_flow_stack(n=n, T=T, k=2, seed=53)
        m = SARFlowPanel(
            y=y,
            W=G,
            X=X,
            T=T,
            col_names=col_names,
            effects=0,
            restrict_positive=True,
        )
        with pytest.raises(ValueError, match="sampler must be"):
            m.fit(
                draws=10,
                tune=10,
                chains=1,
                progressbar=False,
                random_seed=0,
                sampler="mcmc",
            )
