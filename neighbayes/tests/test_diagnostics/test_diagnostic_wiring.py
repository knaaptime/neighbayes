"""Static wiring checks for model-specific diagnostic methods."""

from __future__ import annotations

import numpy as np

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX


def test_cross_sectional_models_no_frequentist_spec_tests():
    """Cross-sectional models should not expose frequentist spatial spec tests."""
    for cls in [OLS, SLX, SAR, SDM, SEM, SDEM]:
        assert not hasattr(cls, "lm_error_test")
        assert not hasattr(cls, "lm_lag_test")
        assert not hasattr(cls, "moran_test")
        assert not hasattr(cls, "spatial_specification_tests")
        assert not hasattr(cls, "diagnostics")
        assert not hasattr(cls, "regression_diagnostics")
        assert not hasattr(cls, "heteroskedasticity_diagnostics")
        assert not hasattr(cls, "autocorrelation_diagnostics")
        assert not hasattr(cls, "outlier_diagnostics")

    assert not hasattr(SAR, "lm_rho_test")
    assert not hasattr(SAR, "lm_rho_robust_test")
    assert not hasattr(SDM, "lm_rho_test")
    assert not hasattr(SDM, "lm_rho_robust_test")
    assert not hasattr(SEM, "wald_error_test")
    assert not hasattr(SEM, "lr_ratio_test")
    assert not hasattr(SDEM, "wald_error_test")


def test_panel_models_no_frequentist_diagnostics():
    """Panel models should not expose frequentist diagnostic methods."""
    for cls in [OLS, SAR, SEM]:
        assert not hasattr(cls, "diagnostics")
        assert not hasattr(cls, "regression_diagnostics")
        assert not hasattr(cls, "heteroskedasticity_diagnostics")
        assert not hasattr(cls, "autocorrelation_diagnostics")
        assert not hasattr(cls, "outlier_diagnostics")
        assert not hasattr(cls, "panel_diagnostics")
        assert not hasattr(cls, "hausman_test")


from neighbayes.models import SDEM, SDM, SLX, NegBinFlow
from neighbayes.models.flow import OLSFlow, SARFlow
from neighbayes.tests.helpers import flow_form

negbin_flow = flow_form(NegBinFlow)


EXPECTED_REGISTRIES = {
    OLS: [
        "LM-Lag",
        "LM-Error",
        "LM-SDM-Joint",
        "LM-SLX-Error-Joint",
        "Robust-LM-Lag",
        "Robust-LM-Error",
    ],
    SAR: ["LM-Error", "LM-WX", "Robust-LM-WX", "Robust-LM-Error"],
    SEM: ["LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX"],
    SLX: ["LM-Lag", "LM-Error", "Robust-LM-Lag-SDM", "Robust-LM-Error-SDEM"],
    SDM: ["LM-Error-SDM", "Robust-LM-Error-SDM"],
    SDEM: ["LM-Lag-SDEM", "Robust-LM-Lag-SDEM"],
    OLSFlow: [
        "LM-Flow-Destination",
        "LM-Flow-Origin",
        "LM-Flow-OD",
        "LM-Flow-Joint",
        "Robust-LM-Flow-Destination",
        "Robust-LM-Flow-Origin",
        "Robust-LM-Flow-OD",
    ],
}


def test_a_flow_panel_takes_the_panel_flow_suite():
    from neighbayes.dgp import simulate_sar_flow
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite

    d = simulate_sar_flow(n=3, seed=0, distribution="normal")
    y, X = np.tile(d["y_vec"], 2), np.tile(d["X"], (2, 1))
    panel = OLSFlow(y, X, d["G"], T=2, col_names=d["col_names"])
    labels = [label for _, label in get_diagnostic_suite(panel).tests]
    assert labels == [f"Panel-{x}" for x in EXPECTED_REGISTRIES[OLSFlow]]


def test_fitted_spatial_and_count_flows_have_no_suite():
    """The flow tests take the Gaussian aspatial null; no other flow has a suite."""
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite
    from neighbayes.models.flow import NegBinFlow, SARNegBinFlow, SEMFlow

    for cls in (SARFlow, SEMFlow, NegBinFlow, SARNegBinFlow):
        assert get_diagnostic_suite(cls) is None, cls.__name__
    assert get_diagnostic_suite(negbin_flow) is None


def test_spatial_diagnostics_registries_match_expected():
    """Each model class should expose the expected ordered list of LM tests."""
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite

    for cls, expected_labels in EXPECTED_REGISTRIES.items():
        suite = get_diagnostic_suite(cls)
        assert suite is not None, f"No suite registered for {cls.__name__}"
        actual_labels = [label for _, label in suite.tests]
        assert actual_labels == expected_labels, (
            f"{cls.__name__}: expected {expected_labels}, got {actual_labels}"
        )


def _panel(cls, **design):
    """A small panel instance of ``cls``: a panel's suite depends on its design."""
    from neighbayes.tests.helpers import W_to_graph, make_line_W

    rng = np.random.default_rng(0)
    N, T = 6, 3
    X = np.column_stack([np.ones(N * T), rng.standard_normal(N * T)])
    W = W_to_graph(make_line_W(N))
    return cls(y=rng.standard_normal(N * T), X=X, W=W, N=N, T=T, **design)


_OLS_PANEL = [
    "Panel-LM-Lag",
    "Panel-LM-Error",
    "Panel-LM-SDM-Joint",
    "Panel-LM-SLX-Error-Joint",
    "Panel-Robust-LM-Lag",
    "Panel-Robust-LM-Error",
]
_FE = {"unit_effects": "fixed"}
_RE = {"unit_effects": "random"}
EXPECTED_PANEL_REGISTRIES = [
    (OLS, _FE, _OLS_PANEL),
    (SAR, _FE, ["Panel-LM-Error", "Panel-LM-WX", "Panel-Robust-LM-WX"]),
    (SEM, _FE, ["Panel-LM-Lag", "Panel-LM-WX"]),
    (
        SLX,
        _FE,
        [
            "Panel-LM-Lag",
            "Panel-LM-Error",
            "Panel-Robust-LM-Lag-SDM",
            "Panel-Robust-LM-Error-SDEM",
        ],
    ),
    (SDM, _FE, ["Panel-LM-Error-SDM"]),
    (SDEM, _FE, ["Panel-LM-Lag-SDEM"]),
    (OLS, _RE, _OLS_PANEL),
    (SAR, _RE, ["Panel-LM-Error", "Panel-LM-WX", "Panel-Robust-LM-WX"]),
    (SEM, _RE, ["Panel-LM-Lag", "Panel-LM-WX"]),
    (SDEM, _RE, ["Panel-LM-Lag-SDEM"]),
]


def test_panel_diagnostics_registries_match_expected():
    """A panel's suite follows its design, not the cross-section's."""
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite

    for cls, design, expected_labels in EXPECTED_PANEL_REGISTRIES:
        suite = get_diagnostic_suite(_panel(cls, **design))
        assert suite is not None, f"No suite registered for a {cls.__name__} panel"
        actual_labels = [label for _, label in suite.tests]
        assert actual_labels == expected_labels, (
            f"{cls.__name__} {design}: expected {expected_labels}, got {actual_labels}"
        )


def test_count_suites_follow_the_design():
    """A count cross-section has its reduced-form SAR suite; the panels have none."""
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite
    from neighbayes.models import NegBin, SARNegBin
    from neighbayes.tests.helpers import W_to_graph, make_line_W

    rng = np.random.default_rng(0)
    N, T = 6, 3
    X = np.column_stack([np.ones(N * T), rng.standard_normal(N * T)])
    y = rng.poisson(2.0, size=N * T)
    W = W_to_graph(make_line_W(N))
    xs = dict(y=y[:N], X=X[:N], W=W)

    def labels(model):
        return [label for _, label in get_diagnostic_suite(model).tests]

    assert labels(SARNegBin(**xs)) == ["LM-Error", "LM-WX"]
    assert labels(NegBin(**xs)) == [
        "LM-Lag",
        "LM-Error",
        "Robust-LM-Lag",
        "Robust-LM-Error",
        "LM-SARMA",
        "LM-WX",
        "Moran-Working",
    ]
    for cls in (NegBin, SARNegBin):
        panel = cls(y=y, X=X, W=W, N=N, T=T, unit_effects="random")
        assert get_diagnostic_suite(panel) is None


def test_slx_uses_slx_specific_lm_tests():
    """SLX uses Koley-Bera SLX-specific LM tests, not raw OLS LM tests."""
    from neighbayes.diagnostics.lmtests.registry import get_diagnostic_suite

    suite = get_diagnostic_suite(SLX)
    labels = [label for _, label in suite.tests]
    # Must not be empty placeholders
    assert len(labels) >= 2


def test_flow_models_expose_decision_api():
    """FlowModel, cross-section and panel alike, exposes spatial_diagnostics_decision."""
    from neighbayes.models.flow import FlowModel

    assert callable(FlowModel.spatial_diagnostics_decision)


def test_flow_decision_tree_dispatch():
    """get_flow_spec and get_panel_flow_spec dispatch to correct tree roots."""
    import pandas as pd

    from neighbayes.diagnostics import _decision_trees as _dt

    # OLSFlow: joint sig -> SARFlow
    spec = _dt.get_flow_spec("OLSFlow")

    def sig_all(name):
        return name == "LM-Flow-Joint"

    d, _ = _dt.evaluate(spec, sig_all)
    assert d == "SARFlow"

    # OLSFlow: no sig -> OLSFlow
    d2, _ = _dt.evaluate(spec, lambda name: False)
    assert d2 == "OLSFlow"

    # Panel variants
    spec_p = _dt.get_panel_flow_spec("OLSFlow")
    dp, _ = _dt.evaluate(spec_p, lambda name: name == "Panel-LM-Flow-Joint")
    assert dp == "SARFlow"
