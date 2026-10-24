"""Diagnostic suites — named registries of Bayesian LM specification tests.

A :class:`DiagnosticSuite` is a frozen, named tuple of
``(test_callable, display_label)`` pairs.  Model classes are mapped to
their diagnostic suite via the :data:`DIAGNOSTICS_REGISTRY` in
:mod:`neighbayes.diagnostics.lmtests.registry`, which
:meth:`SpatialModel.spatial_diagnostics` looks up at runtime.

Each test callable lazily imports the corresponding function from
:mod:`neighbayes.diagnostics.lmtests` on first call.  This keeps the
suites import-cheap and avoids circular imports between the models
and diagnostics packages.

The suites here are the single source of truth for "which tests apply
to which model".  Adding a new test family to a model means:

1.  Implement the test function in ``lmtests/`` and export it from
    the package ``__init__``.
2.  Add an entry (or build a new :class:`DiagnosticSuite`) here.
3.  Register the suite in :data:`DIAGNOSTICS_REGISTRY` in
    :mod:`neighbayes.diagnostics.lmtests.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

_MODULE = "neighbayes.diagnostics.lmtests"

# Type alias for the (callable, label) pairs stored in a suite.
TestEntry = tuple[Callable[[object], object], str]


def _lazy_test(name: str) -> Callable[[object], object]:
    """Return a thunk that lazily imports ``name`` from :mod:`lmtests`.

    The lazy import dodges circular dependencies: models declare suites
    at class-construction time but the LM test functions transitively
    import from the models package via ArviZ utilities.
    """

    def _fn(model):  # pragma: no cover - trivial dispatch
        return getattr(import_module(_MODULE), name)(model)

    _fn.__name__ = f"_lazy_{name}"
    _fn.__qualname__ = _fn.__name__
    return _fn


def _suite(*entries: tuple[str, str]) -> tuple[TestEntry, ...]:
    """Build an immutable tuple of (lazy-test, label) pairs."""
    return tuple((_lazy_test(fn_name), label) for fn_name, label in entries)


@dataclass(frozen=True)
class DiagnosticSuite:
    """Named, immutable registry of Bayesian LM specification tests.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. ``"OLS"`` or ``"SAR-Panel"``.
    tests : tuple[tuple[Callable, str], ...]
        Sequence of ``(test_callable, display_label)`` pairs that
        :meth:`SpatialModel.spatial_diagnostics` will execute in order.
    """

    name: str
    tests: tuple[TestEntry, ...]


# ---------------------------------------------------------------------------
# Cross-sectional suites
# ---------------------------------------------------------------------------

OLS_SUITE = DiagnosticSuite(
    name="OLS",
    tests=_suite(
        ("bayesian_lm_lag_test", "LM-Lag"),
        ("bayesian_lm_error_test", "LM-Error"),
        ("bayesian_lm_sdm_joint_test", "LM-SDM-Joint"),
        ("bayesian_lm_slx_error_joint_test", "LM-SLX-Error-Joint"),
        ("bayesian_robust_lm_lag_test", "Robust-LM-Lag"),
        ("bayesian_robust_lm_error_test", "Robust-LM-Error"),
    ),
)

SLX_SUITE = DiagnosticSuite(
    name="SLX",
    tests=_suite(
        ("bayesian_lm_lag_test", "LM-Lag"),
        ("bayesian_lm_error_test", "LM-Error"),
        ("bayesian_robust_lm_lag_sdm_test", "Robust-LM-Lag-SDM"),
        ("bayesian_robust_lm_error_sdem_test", "Robust-LM-Error-SDEM"),
    ),
)

SAR_SUITE = DiagnosticSuite(
    name="SAR",
    tests=_suite(
        ("bayesian_lm_error_from_sar_test", "LM-Error"),
        ("bayesian_lm_wx_test", "LM-WX"),
        ("bayesian_robust_lm_wx_test", "Robust-LM-WX"),
        ("bayesian_robust_lm_error_sar_test", "Robust-LM-Error"),
    ),
)

SEM_SUITE = DiagnosticSuite(
    name="SEM",
    tests=_suite(
        ("bayesian_lm_lag_test", "LM-Lag"),
        ("bayesian_lm_wx_sem_test", "LM-WX"),
        ("bayesian_robust_lm_lag_sem_test", "Robust-LM-Lag"),
        ("bayesian_robust_lm_wx_sem_test", "Robust-LM-WX"),
    ),
)

SDM_SUITE = DiagnosticSuite(
    name="SDM",
    tests=_suite(
        ("bayesian_lm_error_sdm_test", "LM-Error-SDM"),
        ("bayesian_robust_lm_error_sdm_test", "Robust-LM-Error-SDM"),
    ),
)

SDEM_SUITE = DiagnosticSuite(
    name="SDEM",
    tests=_suite(
        ("bayesian_lm_lag_sdem_test", "LM-Lag-SDEM"),
        ("bayesian_robust_lm_lag_sdem_test", "Robust-LM-Lag-SDEM"),
    ),
)

# Tobit variants — the Tobit SAR and SDM specifications use ``bayesian_lm_error_test``
# (which subtracts a fitted-value residual) rather than the SAR/SDM-null
# variants, matching the pre-suite behavior preserved verbatim here.

SAR_TOBIT_SUITE = DiagnosticSuite(
    name="SARTobit",
    tests=_suite(
        ("bayesian_lm_error_test", "LM-Error"),
        ("bayesian_lm_wx_test", "LM-WX"),
        ("bayesian_robust_lm_wx_test", "Robust-LM-WX"),
        ("bayesian_robust_lm_error_sar_test", "Robust-LM-Error"),
    ),
)

# The logit and NB at the GLM null: the latent SAR's and SEM's score tests
# (``cross_sectional_glm``), the WX score, and the Moran diagnostic on the
# Pólya–Gamma working residuals.
_GLM_NULL_TESTS = (
    ("bayesian_glm_lm_lag_test", "LM-Lag"),
    ("bayesian_glm_lm_error_test", "LM-Error"),
    ("bayesian_glm_robust_lm_lag_test", "Robust-LM-Lag"),
    ("bayesian_glm_robust_lm_error_test", "Robust-LM-Error"),
    ("bayesian_glm_lm_sarma_test", "LM-SARMA"),
    ("bayesian_glm_lm_wx_test", "LM-WX"),
    ("bayesian_glm_moran_working_test", "Moran-Working"),
)
# At a reduced-form SAR null the error and WX scores are uncorrelated with
# every other score, so their robust forms are these.
_GLM_SAR_TESTS = (
    ("bayesian_glm_lm_error_test", "LM-Error"),
    ("bayesian_glm_lm_wx_test", "LM-WX"),
)

LOGIT_SUITE = DiagnosticSuite(name="Logit", tests=_suite(*_GLM_NULL_TESTS))
NEGBIN_SUITE = DiagnosticSuite(name="NegBin", tests=_suite(*_GLM_NULL_TESTS))
SAR_LOGIT_SUITE = DiagnosticSuite(name="SARLogit", tests=_suite(*_GLM_SAR_TESTS))
SAR_NEGBIN_SUITE = DiagnosticSuite(name="SARNegBin", tests=_suite(*_GLM_SAR_TESTS))

SDM_TOBIT_SUITE = DiagnosticSuite(
    name="SDMTobit",
    tests=_suite(
        ("bayesian_lm_error_test", "LM-Error"),
        ("bayesian_robust_lm_error_sdm_test", "Robust-LM-Error"),
    ),
)


# ---------------------------------------------------------------------------
# Flow (origin-destination) suites
# ---------------------------------------------------------------------------

FLOW_SUITE = DiagnosticSuite(
    name="Flow",
    tests=_suite(
        ("bayesian_lm_flow_destination_test", "LM-Flow-Destination"),
        ("bayesian_lm_flow_origin_test", "LM-Flow-Origin"),
        ("bayesian_lm_flow_od_test", "LM-Flow-OD"),
        ("bayesian_lm_flow_joint_test", "LM-Flow-Joint"),
        ("bayesian_robust_lm_flow_destination_test", "Robust-LM-Flow-Destination"),
        ("bayesian_robust_lm_flow_origin_test", "Robust-LM-Flow-Origin"),
        ("bayesian_robust_lm_flow_od_test", "Robust-LM-Flow-OD"),
    ),
)


# ---------------------------------------------------------------------------
# Panel suites (fixed effects and pooled — also reused by random effects
# and most dynamic-panel variants where the test list is identical).
# ---------------------------------------------------------------------------

OLS_PANEL_SUITE = DiagnosticSuite(
    name="OLS-Panel",
    tests=_suite(
        ("bayesian_panel_lm_lag_test", "Panel-LM-Lag"),
        ("bayesian_panel_lm_error_test", "Panel-LM-Error"),
        ("bayesian_panel_lm_sdm_joint_test", "Panel-LM-SDM-Joint"),
        ("bayesian_panel_lm_slx_error_joint_test", "Panel-LM-SLX-Error-Joint"),
        ("bayesian_panel_robust_lm_lag_test", "Panel-Robust-LM-Lag"),
        ("bayesian_panel_robust_lm_error_test", "Panel-Robust-LM-Error"),
    ),
)

SAR_PANEL_SUITE = DiagnosticSuite(
    name="SAR-Panel",
    tests=_suite(
        ("bayesian_panel_lm_error_test", "Panel-LM-Error"),
        ("bayesian_panel_lm_wx_test", "Panel-LM-WX"),
        ("bayesian_panel_robust_lm_wx_test", "Panel-Robust-LM-WX"),
    ),
)

SEM_PANEL_SUITE = DiagnosticSuite(
    name="SEM-Panel",
    tests=_suite(
        ("bayesian_panel_lm_lag_test", "Panel-LM-Lag"),
        ("bayesian_panel_lm_wx_sem_test", "Panel-LM-WX"),
    ),
)

SDM_PANEL_SUITE = DiagnosticSuite(
    name="SDM-Panel",
    tests=_suite(
        ("bayesian_panel_lm_error_sdm_test", "Panel-LM-Error-SDM"),
    ),
)

SDEM_PANEL_SUITE = DiagnosticSuite(
    name="SDEM-Panel",
    tests=_suite(
        ("bayesian_panel_lm_lag_sdem_test", "Panel-LM-Lag-SDEM"),
    ),
)

SLX_PANEL_SUITE = DiagnosticSuite(
    name="SLX-Panel",
    tests=_suite(
        ("bayesian_panel_lm_lag_test", "Panel-LM-Lag"),
        ("bayesian_panel_lm_error_test", "Panel-LM-Error"),
        ("bayesian_panel_robust_lm_lag_sdm_test", "Panel-Robust-LM-Lag-SDM"),
        ("bayesian_panel_robust_lm_error_sdem_test", "Panel-Robust-LM-Error-SDEM"),
    ),
)


# Dynamic-panel-only variants — these differ from the FE/RE menus above and
# must remain separate to preserve existing behavior.

SEM_PANEL_DYNAMIC_SUITE = DiagnosticSuite(
    name="SEM-Panel-Dynamic",
    tests=_suite(
        ("bayesian_panel_lm_lag_sdem_test", "Panel-LM-Lag-SDEM"),
        ("bayesian_panel_robust_lm_lag_sdem_test", "Panel-Robust-LM-Lag-SDEM"),
    ),
)

SLX_PANEL_DYNAMIC_SUITE = DiagnosticSuite(
    name="SLX-Panel-Dynamic",
    tests=_suite(
        ("bayesian_panel_robust_lm_lag_sdm_test", "Panel-Robust-LM-Lag-SDM"),
        ("bayesian_panel_robust_lm_error_sdem_test", "Panel-Robust-LM-Error-SDEM"),
    ),
)


# ---------------------------------------------------------------------------
# Flow-panel suite
# ---------------------------------------------------------------------------

FLOW_PANEL_SUITE = DiagnosticSuite(
    name="Flow-Panel",
    tests=_suite(
        ("bayesian_panel_lm_flow_destination_test", "Panel-LM-Flow-Destination"),
        ("bayesian_panel_lm_flow_origin_test", "Panel-LM-Flow-Origin"),
        ("bayesian_panel_lm_flow_od_test", "Panel-LM-Flow-OD"),
        ("bayesian_panel_lm_flow_joint_test", "Panel-LM-Flow-Joint"),
        (
            "bayesian_panel_robust_lm_flow_destination_test",
            "Panel-Robust-LM-Flow-Destination",
        ),
        ("bayesian_panel_robust_lm_flow_origin_test", "Panel-Robust-LM-Flow-Origin"),
        ("bayesian_panel_robust_lm_flow_od_test", "Panel-Robust-LM-Flow-OD"),
    ),
)


__all__ = [
    "DiagnosticSuite",
    "OLS_SUITE",
    "SLX_SUITE",
    "SAR_SUITE",
    "SEM_SUITE",
    "SDM_SUITE",
    "SDEM_SUITE",
    "SAR_TOBIT_SUITE",
    "LOGIT_SUITE",
    "NEGBIN_SUITE",
    "SAR_LOGIT_SUITE",
    "SAR_NEGBIN_SUITE",
    "SDM_TOBIT_SUITE",
    "FLOW_SUITE",
    "OLS_PANEL_SUITE",
    "SAR_PANEL_SUITE",
    "SEM_PANEL_SUITE",
    "SDM_PANEL_SUITE",
    "SDEM_PANEL_SUITE",
    "SLX_PANEL_SUITE",
    "SEM_PANEL_DYNAMIC_SUITE",
    "SLX_PANEL_DYNAMIC_SUITE",
    "FLOW_PANEL_SUITE",
]
