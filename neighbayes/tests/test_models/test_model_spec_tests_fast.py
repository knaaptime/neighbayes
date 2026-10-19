"""Fast tests verifying that frequentist spatial specification tests have been removed.

The stats module provided purely frequentist diagnostics (LM, Wald, LR, Moran)
that operated on OLS/MLE point estimates rather than Bayesian posteriors.  These
have been removed in favor of the Bayesian LM tests in ``diagnostics.bayesian_diagnostics``.
"""

from __future__ import annotations

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX


def test_cross_sectional_models_no_longer_expose_frequentist_spec_tests():
    """Cross-sectional models should not expose frequentist spatial spec tests."""
    for cls in [OLS, SLX, SAR, SDM, SEM, SDEM]:
        assert not hasattr(cls, "spatial_specification_tests"), (
            f"{cls.__name__} still has spatial_specification_tests"
        )
        assert not hasattr(cls, "lm_error_test"), (
            f"{cls.__name__} still has lm_error_test"
        )
        assert not hasattr(cls, "lm_lag_test"), f"{cls.__name__} still has lm_lag_test"
        assert not hasattr(cls, "moran_test"), f"{cls.__name__} still has moran_test"

    assert not hasattr(SAR, "lm_rho_test")
    assert not hasattr(SAR, "lm_rho_robust_test")
    assert not hasattr(SDM, "lm_rho_test")
    assert not hasattr(SDM, "lm_rho_robust_test")
    assert not hasattr(SEM, "wald_error_test")
    assert not hasattr(SEM, "lr_ratio_test")
    assert not hasattr(SDEM, "wald_error_test")


def test_panel_models_no_longer_expose_frequentist_spec_tests():
    """Panel models should not expose frequentist spatial spec tests."""
    for cls in [OLS, SAR, SEM]:
        assert not hasattr(cls, "spatial_specification_tests"), (
            f"{cls.__name__} still has spatial_specification_tests"
        )

    assert not hasattr(OLS, "lm_error_test")
    assert not hasattr(OLS, "lm_sar_test")
    assert not hasattr(OLS, "lm_joint_test")
    assert not hasattr(SAR, "lm_error_conditional_test")
    assert not hasattr(SAR, "lr_sar_test")
    assert not hasattr(SEM, "lm_sar_conditional_test")
    assert not hasattr(SEM, "lr_error_test")
