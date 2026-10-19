"""Registry mapping model classes to their diagnostic test suites.

This module decouples model classes from their diagnostic test assignments.
Instead of each model class storing ``_spatial_diagnostics_tests`` as a
class attribute (which creates circular-import risk between the models
and diagnostics packages), the mapping lives here as a standalone dict.

Model classes look up their suite via :func:`get_diagnostic_suite`, which
falls back to MRO traversal so that subclasses inherit their parent's
suite unless they override it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .suites import DiagnosticSuite

if TYPE_CHECKING:
    from ...models.base import SpatialModel

# ---------------------------------------------------------------------------
# Registry: model class → DiagnosticSuite
# ---------------------------------------------------------------------------
# Populated lazily to avoid circular imports.  The ``_register`` function
# is called at the bottom of this module after all suite definitions are
# available, and the model classes are referenced by *fully-qualified
# dotted path* so that importing this module does not trigger model imports.
# ---------------------------------------------------------------------------

_REGISTRY: dict[tuple[str, str | None], DiagnosticSuite | None] = {}


def _register(
    class_path: str, suite: DiagnosticSuite | None, *, variant: str | None = None
) -> None:
    """Register a model class (by dotted path) to a diagnostic suite.

    ``variant`` is ``"panel"`` or ``"dynamic"`` for the suite of a class's
    panels (see :func:`_variant`); a ``None`` suite records that they have
    none, rather than the class's.
    """
    _REGISTRY[(class_path, variant)] = suite


def _variant(model) -> str | None:
    """A model instance's suite variant: ``None``, ``"panel"`` or ``"dynamic"``."""
    if isinstance(model, type) or not getattr(model, "_is_panel", False):
        return None
    if getattr(model, "_design", None) in ("dynamic", "restricted"):
        return "dynamic"
    return "panel"


def get_diagnostic_suite(model: "SpatialModel | type") -> DiagnosticSuite | None:
    """Return the :class:`DiagnosticSuite` for *model*, or ``None``.

    Looks up the model class in the registry: for an instance, under its
    design's variant first (a panel's suite), then under the class itself.
    A class passed directly gets its cross-sectional suite.  If the exact class
    is not found, walks the MRO to find the first registered ancestor.  Returns
    ``None`` when no suite is registered for any class in the MRO.
    """
    cls = model if isinstance(model, type) else type(model)
    variant = _variant(model)
    keys = (variant, None) if variant is not None else (None,)
    for klass in cls.__mro__:
        fqn = f"{klass.__module__}.{klass.__qualname__}"
        for v in keys:
            if (fqn, v) in _REGISTRY:
                return _REGISTRY[(fqn, v)]
    return None


# ---------------------------------------------------------------------------
# Cross-sectional models
# ---------------------------------------------------------------------------

from .suites import (  # noqa: E402
    FLOW_INTRA_SUITE,
    FLOW_PANEL_SUITE,
    FLOW_SUITE,
    LOGIT_SUITE,
    NEGBIN_SUITE,
    OLS_SUITE,
    SAR_SUITE,
    SAR_TOBIT_SUITE,
    SDEM_SUITE,
    SDM_SUITE,
    SDM_TOBIT_SUITE,
    SEM_SUITE,
    SLX_SUITE,
)

# Cross-sectional
_register("neighbayes.models.cross_section.ols.OLS", OLS_SUITE)
_register("neighbayes.models.cross_section.sar.SAR", SAR_SUITE)
_register("neighbayes.models.cross_section.sem.SEM", SEM_SUITE)
_register("neighbayes.models.cross_section.slx.SLX", SLX_SUITE)
_register("neighbayes.models.cross_section.sdm.SDM", SDM_SUITE)
_register("neighbayes.models.cross_section.sdem.SDEM", SDEM_SUITE)

# Tobit
_register("neighbayes.models.cross_section.tobit.SARTobit", SAR_TOBIT_SUITE)
_register("neighbayes.models.cross_section.tobit.SEMTobit", SEM_SUITE)
_register("neighbayes.models.cross_section.tobit.SDMTobit", SDM_TOBIT_SUITE)

# NegBin: the count panels have no suite.
_register("neighbayes.models.cross_section.sar_negbin.SARNegBin", NEGBIN_SUITE)
_register("neighbayes.models.cross_section.sar_negbin.SARNegBin", None, variant="panel")

# Non-spatial GLM baselines
_register("neighbayes.models.cross_section.logit.Logit", LOGIT_SUITE)
_register("neighbayes.models.cross_section.negbin.NegBin", NEGBIN_SUITE)
_register("neighbayes.models.cross_section.negbin.NegBin", None, variant="panel")

# Flow (cross-sectional)
_register("neighbayes.models.flow._flow.SARFlow", FLOW_SUITE)
_register("neighbayes.models.flow._flow.SARFlowSeparable", FLOW_SUITE)
_register("neighbayes.models.flow._flow.SARNegBinFlow", FLOW_SUITE)
_register("neighbayes.models.flow._flow.SARNegBinFlowSeparable", FLOW_SUITE)
_register("neighbayes.models.flow._flow.OLSFlow", FLOW_INTRA_SUITE)
_register("neighbayes.models.flow._flow.NegBinFlow", FLOW_INTRA_SUITE)
_register("neighbayes.models.flow._flow.SEMFlow", FLOW_SUITE)
_register("neighbayes.models.flow._flow.SEMFlowSeparable", FLOW_SUITE)

# ---------------------------------------------------------------------------
# Panel models
# ---------------------------------------------------------------------------

from .suites import (  # noqa: E402
    OLS_PANEL_SUITE,
    SAR_PANEL_SUITE,
    SDEM_PANEL_SUITE,
    SDM_PANEL_SUITE,
    SEM_PANEL_DYNAMIC_SUITE,
    SEM_PANEL_SUITE,
    SLX_PANEL_DYNAMIC_SUITE,
    SLX_PANEL_SUITE,
)

# The Gaussian processes' panels: static and random effects, then dynamic
_CS = "neighbayes.models.cross_section"
for _path, _panel, _dynamic in (
    (f"{_CS}.ols.OLS", OLS_PANEL_SUITE, OLS_PANEL_SUITE),
    (f"{_CS}.sar.SAR", SAR_PANEL_SUITE, SAR_PANEL_SUITE),
    (f"{_CS}.sem.SEM", SEM_PANEL_SUITE, SEM_PANEL_DYNAMIC_SUITE),
    (f"{_CS}.sdm.SDM", SDM_PANEL_SUITE, SDM_PANEL_SUITE),
    (f"{_CS}.sdem.SDEM", SDEM_PANEL_SUITE, SDEM_PANEL_SUITE),
    (f"{_CS}.slx.SLX", SLX_PANEL_SUITE, SLX_PANEL_DYNAMIC_SUITE),
):
    _register(_path, _panel, variant="panel")
    _register(_path, _dynamic, variant="dynamic")

# Pooled Tobit panels
_register(f"{_CS}.tobit.SARTobit", SAR_PANEL_SUITE, variant="panel")
_register(f"{_CS}.tobit.SEMTobit", SEM_PANEL_DYNAMIC_SUITE, variant="panel")

# Flow Panel
_register("neighbayes.models.flow_panel._panel.SARFlowPanelFE", FLOW_PANEL_SUITE)
_register(
    "neighbayes.models.flow_panel._panel.SARFlowSeparablePanelFE", FLOW_PANEL_SUITE
)
_register("neighbayes.models.flow_panel._panel.OLSFlowPanelFE", FLOW_PANEL_SUITE)
_register("neighbayes.models.flow_panel._panel.SEMFlowPanelFE", FLOW_PANEL_SUITE)
_register(
    "neighbayes.models.flow_panel._panel.SEMFlowSeparablePanelFE", FLOW_PANEL_SUITE
)
