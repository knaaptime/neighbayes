"""Fast behavior tests verifying frequentist panel diagnostics have been removed."""

from __future__ import annotations

from neighbayes.models import OLS, SAR, SLX
from neighbayes.models.panel._fe import OLSPanelFE, SARPanelFE, SLXPanelFE
from neighbayes.models.panel._re import OLSPanelRE


def test_panel_models_no_longer_expose_frequentist_diagnostics():
    """Panel models should not expose frequentist diagnostic methods."""
    for cls in [OLSPanelFE, OLSPanelRE, SARPanelFE, SLXPanelFE]:
        assert not hasattr(cls, "diagnostics"), (
            f"{cls.__name__} still has diagnostics()"
        )
        assert not hasattr(cls, "regression_diagnostics"), (
            f"{cls.__name__} still has regression_diagnostics()"
        )
        assert not hasattr(cls, "heteroskedasticity_diagnostics"), (
            f"{cls.__name__} still has heteroskedasticity_diagnostics()"
        )
        assert not hasattr(cls, "autocorrelation_diagnostics"), (
            f"{cls.__name__} still has autocorrelation_diagnostics()"
        )
        assert not hasattr(cls, "outlier_diagnostics"), (
            f"{cls.__name__} still has outlier_diagnostics()"
        )
        assert not hasattr(cls, "panel_diagnostics"), (
            f"{cls.__name__} still has panel_diagnostics()"
        )

    assert not hasattr(OLSPanelFE, "hausman_test"), (
        "OLSPanelFE still has hausman_test()"
    )
