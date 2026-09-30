"""Tests for the spatial-MCMC adequacy diagnostic.

Implements the post-fit checks of Wolf, Anselin & Arribas-Bel (2018,
Geographical Analysis 50:97-119).  Tests use synthetic ArviZ
InferenceData stand-ins (no MCMC) so they run in milliseconds.
"""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pytest

from neighbayes.diagnostics import SpatialMCMCReport, spatial_mcmc_diagnostic
from neighbayes.tests.helpers import make_idata as _make_idata


class _FakeModel:
    """Minimal stand-in exposing the ``inference_data`` attribute."""

    def __init__(self, idata):
        self.inference_data = idata


def _ar1_chain(
    n_chain: int, n_draw: int, phi: float, mean: float, scale: float, seed: int
):
    """Generate AR(1) draws so we can dial autocorrelation."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_chain, n_draw))
    for c in range(n_chain):
        x = mean
        for t in range(n_draw):
            x = mean + phi * (x - mean) + rng.normal(0.0, scale * np.sqrt(1 - phi**2))
            out[c, t] = x
    return out


class TestSpatialMCMCDiagnostic:
    def test_well_mixed_chain_is_adequate(self):
        # iid normal draws → ESS ≈ nominal, no warnings
        rng = np.random.default_rng(0)
        idata = _make_idata({"rho": rng.normal(0.3, 0.05, size=(4, 1500))})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = spatial_mcmc_diagnostic(_FakeModel(idata))
        assert isinstance(report, SpatialMCMCReport)
        assert report.adequate
        assert report.warnings_triggered == []
        assert report.ess_bulk["rho"] > 1000
        assert report.yield_pct["rho"] > 50.0
        assert report.hpdi_drift_pct["rho"] < 5.0
        assert report.nominal_size == 4 * 1500

    def test_low_ess_triggers_warning(self):
        # heavily autocorrelated, short chain → low ESS
        draws = _ar1_chain(2, 400, phi=0.98, mean=0.7, scale=0.05, seed=1)
        idata = _make_idata({"rho": draws})
        with pytest.warns(UserWarning, match="ESS|yield|HPDI"):
            report = spatial_mcmc_diagnostic(_FakeModel(idata), min_ess=1000)
        assert not report.adequate
        assert any("rho" in w for w in report.warnings_triggered)
        assert report.ess_bulk["rho"] < 1000

    def test_emit_warnings_false_silent(self):
        draws = _ar1_chain(2, 400, phi=0.98, mean=0.7, scale=0.05, seed=2)
        idata = _make_idata({"rho": draws})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = spatial_mcmc_diagnostic(_FakeModel(idata), emit_warnings=False)
        assert not report.adequate
        assert len(report.warnings_triggered) >= 1

    def test_lam_parameter_detected(self):
        rng = np.random.default_rng(3)
        idata = _make_idata({"lam": rng.normal(0.4, 0.05, size=(4, 1500))})
        report = spatial_mcmc_diagnostic(_FakeModel(idata), emit_warnings=False)
        assert "lam" in report.parameters
        assert "lam" in report.ess_bulk

    def test_flow_params_autodetected(self):
        rng = np.random.default_rng(31)
        idata = _make_idata(
            {
                "rho_d": rng.normal(0.3, 0.05, size=(2, 800)),
                "rho_o": rng.normal(0.2, 0.05, size=(2, 800)),
                "rho_w": rng.normal(0.1, 0.05, size=(2, 800)),
            }
        )
        report = spatial_mcmc_diagnostic(_FakeModel(idata), emit_warnings=False)
        assert set(report.parameters) == {"rho_d", "rho_o", "rho_w"}
        assert report.adequate

    def test_extra_params_reported_but_not_gating(self):
        # rho good; beta bad — should still be adequate (extras don't gate)
        rng = np.random.default_rng(4)
        good_rho = rng.normal(0.3, 0.05, size=(4, 1500))
        bad_beta = _ar1_chain(2, 200, phi=0.99, mean=1.0, scale=0.1, seed=4)[
            :, :, None
        ]  # shape (chain, draw, 1)
        # Pad to (4, 1500, 1) by tiling
        bad_beta_full = np.tile(bad_beta, (2, int(np.ceil(1500 / 200)), 1))[:, :1500, :]
        idata = az.from_dict({"posterior": {"rho": good_rho, "beta": bad_beta_full}})
        report = spatial_mcmc_diagnostic(
            _FakeModel(idata), extra_params=["beta"], emit_warnings=False
        )
        assert report.adequate  # only rho gates adequacy
        assert "beta" in report.ess_bulk

    def test_unfit_model_raises(self):
        class Empty:
            inference_data = None

        with pytest.raises(RuntimeError, match="not been fit"):
            spatial_mcmc_diagnostic(Empty())

    def test_no_spatial_param_raises(self):
        rng = np.random.default_rng(5)
        idata = _make_idata({"sigma": rng.normal(1.0, 0.1, size=(2, 500))})
        with pytest.raises(ValueError, match="rho.*lam"):
            spatial_mcmc_diagnostic(_FakeModel(idata))

    def test_no_spatial_param_with_extras_ok(self):
        rng = np.random.default_rng(6)
        idata = _make_idata({"sigma": rng.normal(1.0, 0.1, size=(2, 500))})
        report = spatial_mcmc_diagnostic(
            _FakeModel(idata), extra_params=["sigma"], emit_warnings=False
        )
        assert "sigma" in report.parameters
        assert report.adequate  # no spatial params to gate

    def test_threshold_overrides(self):
        rng = np.random.default_rng(7)
        idata = _make_idata({"rho": rng.normal(0.3, 0.05, size=(2, 500))})
        # impossibly tight ESS bound → must fail
        report = spatial_mcmc_diagnostic(
            _FakeModel(idata), min_ess=1e9, emit_warnings=False
        )
        assert not report.adequate
        assert any("ESS" in w for w in report.warnings_triggered)

    def test_to_frame_shape_and_columns(self):
        rng = np.random.default_rng(8)
        idata = _make_idata(
            {
                "rho_d": rng.normal(0.3, 0.05, size=(4, 1500)),
                "rho_o": rng.normal(0.2, 0.05, size=(4, 1500)),
                "rho_w": rng.normal(0.1, 0.05, size=(4, 1500)),
            }
        )
        report = spatial_mcmc_diagnostic(_FakeModel(idata), emit_warnings=False)
        df = report.to_frame()
        # one row per spatial param, in the order returned by parameters
        assert list(df.index) == report.parameters
        assert df.index.name == "parameter"
        # column order matches _FRAME_COLUMNS
        assert list(df.columns) == list(SpatialMCMCReport._FRAME_COLUMNS)
        # values mirror the dataclass dicts
        for name in report.parameters:
            assert df.loc[name, "ess_bulk"] == report.ess_bulk[name]
            assert df.loc[name, "r_hat"] == report.r_hat[name]
            assert bool(df.loc[name, "adequate"]) is report.adequate_by_param[name]
        # well-mixed → all adequate True
        assert df["adequate"].all()
        assert report.adequate

    def test_to_frame_per_param_adequate_isolates_failures(self):
        # Construct a posterior where rho_d is clean but rho_o has terrible
        # autocorrelation, so only rho_o should fail.
        rng = np.random.default_rng(9)
        good = rng.normal(0.3, 0.05, size=(4, 1500))
        bad = _ar1_chain(4, 1500, phi=0.999, mean=0.2, scale=0.05, seed=10)
        idata = _make_idata({"rho_d": good, "rho_o": bad})
        report = spatial_mcmc_diagnostic(_FakeModel(idata), emit_warnings=False)
        df = report.to_frame()
        assert bool(df.loc["rho_d", "adequate"]) is True
        assert bool(df.loc["rho_o", "adequate"]) is False
        # scalar rollup matches: not adequate overall when any param fails
        assert not report.adequate
        assert report.adequate_by_param == {"rho_d": True, "rho_o": False}
