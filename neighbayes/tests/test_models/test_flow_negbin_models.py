"""Tests for Negative Binomial flow and panel-flow model variants."""

from __future__ import annotations

import pytest

from neighbayes.models.flow._flow import (
    NegBinFlow,
    SARNegBinFlow,
    SARNegBinFlowSeparable,
)
from neighbayes.models.flow_panel._panel import (
    NegBinFlowPanel,
    SARNegBinFlowPanel,
    SARNegBinFlowSeparablePanel,
)


def _small_negbin_flow(seed: int = 0):
    from neighbayes.dgp.flows import generate_negbin_flow_data

    return generate_negbin_flow_data(n=6, seed=seed)


def _small_panel_negbin_flow(seed: int = 0):
    from neighbayes.dgp.flows import generate_panel_negbin_flow_data

    return generate_panel_negbin_flow_data(n=5, T=3, seed=seed)


class TestNegativeBinomialFlowConstruction:
    def test_negbin_sar_flow_builds(self):
        data = _small_negbin_flow(seed=1)
        model = SARNegBinFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_sar_flow_separable_builds(self):
        data = _small_negbin_flow(seed=2)
        model = SARNegBinFlowSeparable(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_flow_builds(self):
        data = _small_negbin_flow(seed=3)
        model = NegBinFlow(
            data["y_vec"],
            data["X"],
            data["G"],
            col_names=data["col_names"],
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars


class TestNegativeBinomialPanelFlowConstruction:
    def test_negbin_panel_builds(self):
        data = _small_panel_negbin_flow(seed=4)
        model = SARNegBinFlowPanel(
            y=data["y"],
            W=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            effects=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_separable_builds(self):
        data = _small_panel_negbin_flow(seed=5)
        model = SARNegBinFlowSeparablePanel(
            y=data["y"],
            W=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            effects=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_aspatial_builds(self):
        data = _small_panel_negbin_flow(seed=6)
        model = NegBinFlowPanel(
            y=data["y"],
            W=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            effects=0,
        )
        pm_model = model._build_pymc_model()
        assert "alpha" in pm_model.named_vars

    def test_negbin_panel_pair_effects_build(self):
        """Pair effects are pooled parameters (see test_count_panel_fe.py), not demeaning."""
        data = _small_panel_negbin_flow(seed=7)
        model = SARNegBinFlowPanel(
            y=data["y"],
            W=data["G"],
            X=data["X"],
            T=3,
            col_names=data["col_names"],
            effects=3,
        )
        assert list(model._feature_names) == list(data["col_names"])
        pm_model = model._build_pymc_model()
        assert {"alpha", "group_effect", "group_sd", "time_effect"} <= set(
            pm_model.named_vars
        )


def _check_beta_recovery(
    idata, beta_d_true, beta_o_true, gamma_dist_true, k=2, tol=0.4
):
    """Assert beta coefficient recovery for flow models.

    Design matrix layout (flow_design_matrix_with_orig, k_d=k_o=k):
      [0] intercept      → DGP true = 0
      [1] intra indicator→ DGP true = 0
      [2:2+k] beta_d     → checked here
      [2+k:2+2k] beta_o  → checked here
      [-1] log_dist (gamma_dist) → checked here
    """
    import numpy as np

    beta_hat = idata.posterior["beta"].mean(dim=["chain", "draw"]).values
    beta_d_hat = beta_hat[2 : 2 + k]
    beta_o_hat = beta_hat[2 + k : 2 + 2 * k]
    gamma_dist_hat = float(beta_hat[-1])

    for i, (true_val, hat_val) in enumerate(zip(beta_d_true, beta_d_hat)):
        assert abs(hat_val - true_val) < tol, (
            f"beta_d[{i}] recovery failed: true={true_val:.3f}, hat={hat_val:.3f}"
        )
    for i, (true_val, hat_val) in enumerate(zip(beta_o_true, beta_o_hat)):
        assert abs(hat_val - true_val) < tol, (
            f"beta_o[{i}] recovery failed: true={true_val:.3f}, hat={hat_val:.3f}"
        )
    assert abs(gamma_dist_hat - gamma_dist_true) < tol, (
        f"gamma_dist recovery failed: true={gamma_dist_true:.3f}, hat={gamma_dist_hat:.3f}"
    )


@pytest.mark.slow
class TestNegativeBinomialFlowRecovery:
    """Parameter recovery checks for NB flow variants (deselected by default)."""

    def test_negbin_sar_flow_recovers_all_params(self):
        from neighbayes.dgp.flows import generate_negbin_flow_data

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.2, 0.1
        beta_d_true, beta_o_true = [1.5, -0.8], [0.7, 1.2]
        gamma_dist_true = -0.5
        alpha_true = 2.0

        out = generate_negbin_flow_data(
            n=15,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            rho_w=rho_w_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=42,
        )
        model = SARNegBinFlow(
            out["y_vec"],
            out["X"],
            out["G"],
            col_names=out["col_names"],
        )
        idata = model.fit(
            sampler="gibbs",
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        rho_w_hat = float(idata.posterior["rho_w"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        assert abs(rho_d_hat - rho_d_true) < 0.20, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.20, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(rho_w_hat - rho_w_true) < 0.20, (
            f"rho_w: {rho_w_hat:.3f} vs {rho_w_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)

    def test_negbin_sar_flow_separable_recovers_all_params(self):
        from neighbayes.dgp.flows import generate_negbin_flow_data_separable

        rho_d_true, rho_o_true = 0.4, 0.3
        beta_d_true, beta_o_true = [1.2, -0.6], [0.9, 1.1]
        gamma_dist_true = -0.5
        alpha_true = 1.8

        out = generate_negbin_flow_data_separable(
            n=15,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=43,
        )
        model = SARNegBinFlowSeparable(
            out["y_vec"],
            out["X"],
            out["G"],
            col_names=out["col_names"],
        )
        idata = model.fit(
            sampler="gibbs",
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=43,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        # Separable NB recovery exhibits materially higher Monte-Carlo variability
        # than the unrestricted NB flow recovery case on this small synthetic sample.
        assert abs(rho_d_hat - rho_d_true) < 0.45, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.30, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true, tol=0.8)


@pytest.mark.slow
class TestNegativeBinomialPanelFlowRecovery:
    """Parameter recovery checks for panel NB flow variants."""

    def test_negbin_panel_sar_recovers_all_params(self):
        from neighbayes.dgp.flows import generate_panel_negbin_flow_data

        rho_d_true, rho_o_true, rho_w_true = 0.25, 0.2, 0.1
        beta_d_true, beta_o_true = [1.4, -0.7], [0.8, 1.1]
        gamma_dist_true = -0.5
        alpha_true = 2.2

        out = generate_panel_negbin_flow_data(
            n=7,
            T=5,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            rho_w=rho_w_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=44,
        )
        model = SARNegBinFlowPanel(
            y=out["y"],
            W=out["G"],
            X=out["X"],
            T=5,
            col_names=out["col_names"],
            effects=0,
        )
        idata = model.fit(
            sampler="gibbs",
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=44,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        rho_w_hat = float(idata.posterior["rho_w"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        assert abs(rho_d_hat - rho_d_true) < 0.25, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.25, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(rho_w_hat - rho_w_true) < 0.25, (
            f"rho_w: {rho_w_hat:.3f} vs {rho_w_true}"
        )
        # alpha is weakly identified on this sample (52% zeros, heavy tail):
        # the exact-likelihood MLE on this realization is ~4.0, so the full
        # posterior mean legitimately sits well above the DGP value of 2.2.
        assert abs(alpha_hat - alpha_true) < 3.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)

    def test_negbin_panel_sar_separable_recovers_all_params(self):
        from neighbayes.dgp.flows import generate_panel_negbin_flow_data_separable

        rho_d_true, rho_o_true = 0.4, 0.3
        beta_d_true, beta_o_true = [1.3, -0.5], [0.6, 1.0]
        gamma_dist_true = -0.5
        alpha_true = 1.7

        # n=10/T=6 (NT=600): at NT≈250 this realization has a competing joint
        # mode (rho_o≈0 with beta_d absorbing the origin-side signal) that
        # beats the DGP truth in exact likelihood — the (rho_o, beta) split is
        # simply not identified there.
        out = generate_panel_negbin_flow_data_separable(
            n=10,
            T=6,
            rho_d=rho_d_true,
            rho_o=rho_o_true,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=45,
        )
        model = SARNegBinFlowSeparablePanel(
            y=out["y"],
            W=out["G"],
            X=out["X"],
            T=6,
            col_names=out["col_names"],
            effects=0,
        )
        idata = model.fit(
            sampler="gibbs",
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=45,
            progressbar=False,
        )

        rho_d_hat = float(idata.posterior["rho_d"].mean())
        rho_o_hat = float(idata.posterior["rho_o"].mean())
        alpha_hat = float(idata.posterior["alpha"].mean())

        assert abs(rho_d_hat - rho_d_true) < 0.35, (
            f"rho_d: {rho_d_hat:.3f} vs {rho_d_true}"
        )
        assert abs(rho_o_hat - rho_o_true) < 0.30, (
            f"rho_o: {rho_o_hat:.3f} vs {rho_o_true}"
        )
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )

        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true, tol=0.8)


@pytest.mark.slow
class TestNegBinFlowAspatialRecovery:
    """Recovery for the aspatial NegBinFlow (no spatial parameters)."""

    def test_negbin_flow_recovers_beta_and_alpha(self):
        from neighbayes.dgp.flows import generate_negbin_flow_data

        beta_d_true, beta_o_true = [1.5, -0.8], [0.7, 1.2]
        gamma_dist_true = -0.5
        alpha_true = 2.0

        out = generate_negbin_flow_data(
            n=15,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=42,
        )
        model = NegBinFlow(
            out["y_vec"],
            out["X"],
            out["G"],
            col_names=out["col_names"],
        )
        idata = model.fit(
            sampler="gibbs",
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

        alpha_hat = float(idata.posterior["alpha"].mean())
        assert abs(alpha_hat - alpha_true) < 1.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )
        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)


@pytest.mark.slow
class TestNegBinFlowPanelAspatialRecovery:
    """Recovery for the aspatial NegBinFlowPanel (no spatial parameters)."""

    def test_negbin_flow_panel_recovers_beta_and_alpha(self):
        from neighbayes.dgp.flows import generate_panel_negbin_flow_data

        beta_d_true, beta_o_true = [1.4, -0.7], [0.8, 1.1]
        gamma_dist_true = -0.5
        alpha_true = 2.2

        out = generate_panel_negbin_flow_data(
            n=7,
            T=5,
            rho_d=0.0,
            rho_o=0.0,
            rho_w=0.0,
            beta_d=beta_d_true,
            beta_o=beta_o_true,
            gamma_dist=gamma_dist_true,
            alpha=alpha_true,
            seed=42,
        )
        model = NegBinFlowPanel(
            y=out["y"],
            W=out["G"],
            X=out["X"],
            T=5,
            col_names=out["col_names"],
            effects=0,
        )
        idata = model.fit(
            draws=1500,
            tune=1500,
            chains=2,
            random_seed=42,
            progressbar=False,
        )

        alpha_hat = float(idata.posterior["alpha"].mean())
        assert abs(alpha_hat - alpha_true) < 3.5, (
            f"alpha: {alpha_hat:.3f} vs {alpha_true}"
        )
        _check_beta_recovery(idata, beta_d_true, beta_o_true, gamma_dist_true)


class TestFlowGibbsBackend:
    """Flow models default to ``gibbs_backend="auto"``, resolved per configuration."""

    def test_public_fits_default_to_auto(self):
        import inspect

        from neighbayes.models.flow_panel._hurdle import SARHurdleNBFlowSeparablePanel
        from neighbayes.models.flow_panel._panel import SARZINBFlowSeparablePanel

        for cls in (
            SARNegBinFlow,
            SARNegBinFlowSeparable,
            NegBinFlow,
            SARNegBinFlowPanel,
            SARNegBinFlowSeparablePanel,
            SARZINBFlowSeparablePanel,
            SARHurdleNBFlowSeparablePanel,
        ):
            default = inspect.signature(cls.fit).parameters["gibbs_backend"].default
            assert default == "auto", cls.__name__

    def test_auto_takes_jax_only_where_a_kernel_exists(self):
        pytest.importorskip("jax")
        data = _small_negbin_flow(seed=3)
        model = SARNegBinFlow(data["y_vec"], data["X"], data["G"])
        assert model._resolve_gibbs_backend("auto", jax=True) == "jax"
        assert model._resolve_gibbs_backend("auto", jax=False) == "numpy"
        assert model._resolve_gibbs_backend("numpy", jax=True) == "numpy"

    def test_explicit_jax_without_a_kernel_raises(self):
        """The aspatial flow used to ignore ``"jax"`` and run NumPy."""
        data = _small_negbin_flow(seed=4)
        model = NegBinFlow(data["y_vec"], data["X"], data["G"])
        with pytest.raises(ValueError, match="not supported"):
            model.fit(draws=5, tune=5, chains=1, progressbar=False, gibbs_backend="jax")

    def test_unrestricted_panel_auto_runs_numpy(self):
        """The unrestricted JAX kernel is cross-section only, so a panel stays NumPy."""
        data = _small_panel_negbin_flow(seed=5)
        model = SARNegBinFlowPanel(
            data["y"], data["X"], data["G"], T=3, col_names=data["col_names"]
        )
        assert (
            model._resolve_gibbs_backend(
                "auto", jax=not model._count_effects and model._T == 1
            )
            == "numpy"
        )
        with pytest.raises(ValueError, match="not supported"):
            model.fit(draws=5, tune=5, chains=1, progressbar=False, gibbs_backend="jax")
