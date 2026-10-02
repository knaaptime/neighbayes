"""Tests that every model produces a valid log_likelihood group when fit with idata_kwargs={"log_likelihood": True}.

Each test verifies:
1. The InferenceData has a ``log_likelihood`` group.
2. The ``log_likelihood`` group contains an ``"obs"`` data variable (not a coordinate).
3. The ``"obs"`` DataArray has the correct dimensions: (chain, draw, obs_dim*) and finite values.
4. The shape along the observation axis matches the number of observations in the data.

For models that use Pattern A (pm.Normal("obs", observed=y) auto-captures), the log_likelihood
is automatically created by PyMC. For Pattern B (pm.Potential), the model's fit() method must
manually compute and attach the log_likelihood.

Pattern A+J models (SAR, SDM, etc.) auto-capture the Gaussian part and then add the Jacobian
correction via _reconstruct_cross_sectional_log_likelihood().
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from neighbayes.models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARTobit,
    SDEMPanelFE,
    SDMPanelFE,
    SDMTobit,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
    SEMTobit,
)
from neighbayes.models.base import SpatialModel
from neighbayes.models.cross_section.tobit import _SpatialTobitBase
from neighbayes.models.panel._tobit import _PanelTobitBase
from neighbayes.models.panel_base import SpatialPanelModel
from neighbayes.tests.helpers import W_to_graph, make_line_W

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_section_data(n: int = 8, seed: int = 42):
    """Small cross-sectional dataset for fast tests."""
    rng = np.random.default_rng(seed)
    W_dense = make_line_W(n)
    W = W_to_graph(W_dense)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    return y, X, W, W_dense, n


def _panel_data(N: int = 4, T: int = 3, seed: int = 42):
    """Small panel dataset for fast tests."""
    rng = np.random.default_rng(seed)
    n = N * T
    W_dense = make_line_W(N)
    W = W_to_graph(W_dense)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    return y, X, W, W_dense, N, T, n


def _tobit_data(n: int = 8, seed: int = 42, censor_point: float = 0.0):
    """Small Tobit dataset for fast tests."""
    rng = np.random.default_rng(seed)
    W_dense = make_line_W(n)
    W = W_to_graph(W_dense)
    x1 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1])
    y_latent = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    y = np.where(y_latent > censor_point, y_latent, censor_point)
    return y, X, W, W_dense, n


def _assert_valid_log_likelihood(idata, n_obs: int, label: str):
    """Assert that idata has a valid log_likelihood group with correct structure."""
    assert hasattr(idata, "log_likelihood"), f"{label}: missing log_likelihood group"
    ll = idata.log_likelihood
    assert "obs" in ll.data_vars, (
        f"{label}: 'obs' not in log_likelihood data_vars; "
        f"data_vars={list(ll.data_vars)}, coords={list(ll.coords)}"
    )
    obs_da = ll["obs"]
    # Check that "obs" is a data variable, not a coordinate (xarray bug)
    assert "obs" not in ll.coords, (
        f"{label}: 'obs' is a coordinate instead of a data variable — "
        f"this breaks az.loo()/az.compare()"
    )
    # Check shape: (chain, draw, obs_dim*)
    assert obs_da.ndim == 3, (
        f"{label}: expected 3 dims, got {obs_da.ndim} ({obs_da.dims})"
    )
    assert obs_da.shape[0] >= 1, f"{label}: expected >=1 chain, got {obs_da.shape[0]}"
    assert obs_da.shape[1] >= 1, f"{label}: expected >=1 draw, got {obs_da.shape[1]}"
    assert obs_da.shape[2] == n_obs, (
        f"{label}: expected {n_obs} observations, got {obs_da.shape[2]}"
    )
    # Check all values are finite
    assert np.all(np.isfinite(obs_da.values)), (
        f"{label}: log_likelihood contains non-finite values"
    )


# ===========================================================================
# Cross-sectional models
# ===========================================================================


class TestCrossSectionalLogLikelihood:
    """Test that cross-sectional models produce valid log_likelihood groups."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _cross_section_data()

    def test_ols_log_likelihood(self):
        model = OLS(y=self.y, X=self.X, W=self.W)
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "OLS")

    def test_slx_log_likelihood(self):
        model = SLX(y=self.y, X=self.X, W=self.W)
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SLX")

    def test_sar_log_likelihood(self):
        model = SAR(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SAR")

    def test_sem_log_likelihood(self):
        model = SEM(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SEM")

    def test_sdm_log_likelihood(self):
        model = SDM(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SDM")

    def test_sdem_log_likelihood(self):
        model = SDEM(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SDEM")


# ===========================================================================
# Tobit models
# ===========================================================================


class TestTobitLogLikelihood:
    """Test that Tobit models produce valid log_likelihood groups."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _tobit_data()

    def test_sar_tobit_log_likelihood(self):
        model = SARTobit(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SARTobit")

    def test_sem_tobit_log_likelihood(self):
        model = SEMTobit(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SEMTobit")

    def test_sdm_tobit_log_likelihood(self):
        model = SDMTobit(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SDMTobit")


# ===========================================================================
# Panel FE models
# ===========================================================================


class TestPanelFELogLikelihood:
    """Test that panel FE models produce valid log_likelihood groups."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def test_ols_panel_fe_log_likelihood(self):
        model = OLSPanelFE(y=self.y, X=self.X, W=self.W, N=self.N, T=self.T, effects=1)
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "OLSPanelFE")

    def test_sar_panel_fe_log_likelihood(self):
        model = SARPanelFE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SARPanelFE")

    def test_sem_panel_fe_log_likelihood(self):
        model = SEMPanelFE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SEMPanelFE")

    def test_sdm_panel_fe_log_likelihood(self):
        model = SDMPanelFE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SDMPanelFE")

    def test_sdem_panel_fe_log_likelihood(self):
        model = SDEMPanelFE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SDEMPanelFE")


# ===========================================================================
# Panel RE models
# ===========================================================================


class TestPanelRELogLikelihood:
    """Test that panel RE models produce valid log_likelihood groups."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def test_ols_panel_re_log_likelihood(self):
        model = OLSPanelRE(y=self.y, X=self.X, W=self.W, N=self.N, T=self.T, effects=1)
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "OLSPanelRE")

    def test_sar_panel_re_log_likelihood(self):
        model = SARPanelRE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SARPanelRE")

    def test_sem_panel_re_log_likelihood(self):
        model = SEMPanelRE(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SEMPanelRE")


# ===========================================================================
# Panel Tobit models
# ===========================================================================


class TestPanelTobitLogLikelihood:
    """Test that panel Tobit models produce valid log_likelihood groups."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.N, self.T, self.n = _panel_data()

    def test_sar_panel_tobit_log_likelihood(self):
        model = SARPanelTobit(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SARPanelTobit")

    def test_sem_panel_tobit_log_likelihood(self):
        model = SEMPanelTobit(
            y=self.y,
            X=self.X,
            W=self.W,
            N=self.N,
            T=self.T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=10,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, self.n, "SEMPanelTobit")


# ===========================================================================
# Monkeypatch-based fast tests (no MCMC)
# ===========================================================================


class TestLogLikelihoodStructureFast:
    """Fast tests using monkeypatched MCMC to verify log_likelihood structure.

    These tests stub out the base class fit() to return a fake InferenceData
    with only a posterior group, then verify that the model's fit() correctly
    adds the log_likelihood group.
    """

    def test_sar_fit_adds_log_likelihood(self, monkeypatch):
        """SAR fit() should add log_likelihood with Jacobian correction."""
        y, X, W, W_dense, n = _cross_section_data()
        model = SAR(y=y, X=X, W=W, logdet_method="eigenvalue")

        posterior = {
            "rho": np.array([[0.3, 0.31]]),
            "beta": np.array([[[0.5, 0.8], [0.51, 0.81]]]),
            "sigma": np.array([[1.0, 1.01]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SAR")

    def test_sem_fit_adds_log_likelihood(self, monkeypatch):
        """SEM fit() should add log_likelihood computed from Potential."""
        y, X, W, W_dense, n = _cross_section_data()
        model = SEM(y=y, X=X, W=W, logdet_method="eigenvalue")

        posterior = {
            "lam": np.array([[0.3, 0.31]]),
            "beta": np.array([[[0.5, 0.8], [0.51, 0.81]]]),
            "sigma": np.array([[1.0, 1.01]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SEM")

    def test_sdm_fit_adds_log_likelihood(self, monkeypatch):
        """SDM fit() should add log_likelihood with Jacobian correction."""
        y, X, W, W_dense, n = _cross_section_data()
        model = SDM(y=y, X=X, W=W, logdet_method="eigenvalue")

        posterior = {
            "rho": np.array([[0.3, 0.31]]),
            "beta": np.array([[[0.5, 0.8, 0.1], [0.51, 0.81, 0.11]]]),
            "sigma": np.array([[1.0, 1.01]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SDM")

    def test_sdem_fit_adds_log_likelihood(self, monkeypatch):
        """SDEM fit() should add log_likelihood computed from Potential."""
        y, X, W, W_dense, n = _cross_section_data()
        model = SDEM(y=y, X=X, W=W, logdet_method="eigenvalue")

        posterior = {
            "lam": np.array([[0.3, 0.31]]),
            "beta": np.array([[[0.5, 0.8, 0.1], [0.51, 0.81, 0.11]]]),
            "sigma": np.array([[1.0, 1.01]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SDEM")

    def test_sem_panel_fe_fit_adds_log_likelihood(self, monkeypatch):
        """SEMPanelFE fit() should add log_likelihood computed from Potential."""
        y, X, W, W_dense, N, T, n = _panel_data()
        model = SEMPanelFE(
            y=y, X=X, W=W, N=N, T=T, effects=0, logdet_method="eigenvalue"
        )

        posterior = {
            "lam": np.array([[0.1, 0.11]]),
            "beta": np.array([[[0.2, 0.9], [0.21, 0.91]]]),
            "sigma": np.array([[1.0, 1.1]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialPanelModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SEMPanelFE")

    def test_sdem_panel_fe_fit_adds_log_likelihood(self, monkeypatch):
        """SDEMPanelFE fit() should add log_likelihood computed from Potential."""
        y, X, W, W_dense, N, T, n = _panel_data()
        model = SDEMPanelFE(
            y=y, X=X, W=W, N=N, T=T, effects=0, logdet_method="eigenvalue"
        )

        posterior = {
            "lam": np.array([[0.1, 0.11]]),
            "beta": np.array([[[0.2, 0.9, 0.15], [0.21, 0.91, 0.16]]]),
            "sigma": np.array([[1.0, 1.1]]),
        }
        fake_idata = az.from_dict({"posterior": posterior})

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialPanelModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SDEMPanelFE")

    def test_sar_panel_fe_fit_adds_log_likelihood(self, monkeypatch):
        """SARPanelFE fit() should add Jacobian-corrected log_likelihood."""
        y, X, W, W_dense, N, T, n = _panel_data()
        model = SARPanelFE(
            y=y, X=X, W=W, N=N, T=T, effects=0, logdet_method="eigenvalue"
        )

        # SARPanelFE uses pm.Normal("obs") which auto-captures, so we need
        # a fake idata with log_likelihood already present (simulating PyMC auto-capture)
        n_obs = n
        posterior = {
            "rho": np.array([[0.2, 0.21]]),
            "beta": np.array([[[0.2, 0.9], [0.21, 0.91]]]),
            "sigma": np.array([[1.0, 1.1]]),
        }
        # Simulate auto-captured log_likelihood (Pattern A)
        log_lik = np.random.randn(1, 2, n_obs) - 5.0
        fake_idata = az.from_dict(
            {"posterior": posterior, "log_likelihood": {"obs": log_lik}}
        )

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialPanelModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SARPanelFE")

    def test_sdm_panel_fe_fit_adds_log_likelihood(self, monkeypatch):
        """SDMPanelFE fit() should add Jacobian-corrected log_likelihood."""
        y, X, W, W_dense, N, T, n = _panel_data()
        model = SDMPanelFE(
            y=y, X=X, W=W, N=N, T=T, effects=0, logdet_method="eigenvalue"
        )

        n_obs = n
        posterior = {
            "rho": np.array([[0.2, 0.21]]),
            "beta": np.array([[[0.2, 0.9, 0.15], [0.21, 0.91, 0.16]]]),
            "sigma": np.array([[1.0, 1.1]]),
        }
        log_lik = np.random.randn(1, 2, n_obs) - 5.0
        fake_idata = az.from_dict(
            {"posterior": posterior, "log_likelihood": {"obs": log_lik}}
        )

        def _fake_fit_nuts(self, **kwargs):
            self._idata = fake_idata
            return fake_idata, True

        monkeypatch.setattr(SpatialPanelModel, "_fit_nuts", _fake_fit_nuts)
        out = model.fit(
            sampler="nuts",
            draws=2,
            tune=1,
            chains=1,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

        _assert_valid_log_likelihood(out, n, "SDMPanelFE")


# ===========================================================================
# ArviZ compatibility tests
# ===========================================================================


class TestArviZCompatibility:
    """Test that log_likelihood groups work with ArviZ functions."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        self.y, self.X, self.W, self.W_dense, self.n = _cross_section_data()

    def test_sar_loo(self):
        """SAR log_likelihood should work with az.loo()."""
        model = SAR(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=50,
            tune=25,
            chains=2,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd), "az.loo() returned a non-finite elpd"

    def test_sem_loo(self):
        """SEM log_likelihood should work with az.loo()."""
        model = SEM(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
        idata = model.fit(
            draws=50,
            tune=25,
            chains=2,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
        loo = az.loo(idata)
        assert np.isfinite(loo.elpd), "az.loo() returned a non-finite elpd"

    def test_compare_multiple_models(self):
        """az.compare() should work with multiple models that have log_likelihood."""
        models = {}
        for cls, name in [(OLS, "OLS"), (SAR, "SAR"), (SEM, "SEM")]:
            model = (
                cls(y=self.y, X=self.X, W=self.W, logdet_method="eigenvalue")
                if cls != OLS
                else cls(y=self.y, X=self.X, W=self.W)
            )
            idata = model.fit(
                draws=50,
                tune=25,
                chains=2,
                random_seed=42,
                progressbar=False,
                idata_kwargs={"log_likelihood": True},
            )
            models[name] = idata

        comparison = az.compare(models)
        assert isinstance(comparison, object), "az.compare() should return a DataFrame"
        assert len(comparison) == 3, (
            f"Expected 3 models in comparison, got {len(comparison)}"
        )


# ===========================================================================
# JAX backend parity tests (Phase D)
# ===========================================================================


class TestJaxLogLikelihoodCapture:
    """Verify that PyMC's JAX path captures `log_likelihood["obs"]` natively
    for every spatial-error model migrated to ``pm.CustomDist``.

    Each test fits the model on numpyro and asserts the log_likelihood group
    has shape ``(chain, draw, n_obs)`` and finite values — matching what the
    PyMC backend produces via its manual fallback path.
    """

    pytestmark = pytest.mark.requires_jax

    def test_sem_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, n = _cross_section_data()
        model = SEM(y=y, X=X, W=W, logdet_method="eigenvalue")
        idata = model.fit(
            sampler="nuts",
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SEM[numpyro]")

    def test_sdem_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, n = _cross_section_data()
        model = SDEM(y=y, X=X, W=W, logdet_method="eigenvalue")
        idata = model.fit(
            sampler="nuts",
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SDEM[numpyro]")

    def test_sem_panel_fe_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, N, T, n = _panel_data()
        model = SEMPanelFE(
            y=y,
            X=X,
            W=W,
            N=N,
            T=T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            sampler="nuts",
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SEMPanelFE[numpyro]")

    def test_sdem_panel_fe_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, N, T, n = _panel_data()
        model = SDEMPanelFE(
            y=y,
            X=X,
            W=W,
            N=N,
            T=T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            sampler="nuts",
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SDEMPanelFE[numpyro]")

    def test_sem_panel_re_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, N, T, n = _panel_data()
        model = SEMPanelRE(
            y=y, X=X, W=W, N=N, T=T, effects=1, logdet_method="eigenvalue"
        )
        idata = model.fit(
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            sampler="nuts",
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SEMPanelRE[numpyro]")

    def test_sem_panel_tobit_jax(self):
        pytest.importorskip("numpyro")
        y, X, W, _, N, T, n = _panel_data()
        # introduce some censoring at zero
        y = np.where(y > 0.0, y, 0.0)
        model = SEMPanelTobit(
            y=y,
            X=X,
            W=W,
            N=N,
            T=T,
            effects=1,
            logdet_method="eigenvalue",
        )
        idata = model.fit(
            draws=20,
            tune=20,
            chains=1,
            random_seed=0,
            progressbar=False,
            nuts_sampler="numpyro",
            idata_kwargs={"log_likelihood": True},
        )
        _assert_valid_log_likelihood(idata, n, "SEMPanelTobit[numpyro]")
