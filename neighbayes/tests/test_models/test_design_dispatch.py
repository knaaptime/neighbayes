"""The public classes resolve their design arguments.

ADR 0002: a public class names a process and a family; panel structure,
effects, dynamics, and the flow restriction are arguments.  A Gaussian, Tobit,
count, logit or probit class implements every design itself; the flow
families' public classes still dispatch to implementation classes.  These tests pin the table from public call
to design (or implementation class) and effects code, and the combinations that
must raise.
"""

from __future__ import annotations

import numpy as np
import pytest

import neighbayes.models as m
from neighbayes.dgp.utils import rook_grid_weights


def _route(cls, **kwargs):
    """Where a call goes, without constructing, and its effects code.

    The implementation class for a dispatching public class, else the design
    the class resolves.
    """
    if "_dispatch" in cls.__dict__:
        impl, kw = cls.__dict__["_dispatch"].__func__(cls, dict(kwargs))
        return impl.__name__, kw.get("effects")
    design, kw = cls._resolve_design(dict(kwargs))
    return design, kw.get("effects")


PANEL = {"N": 16, "T": 3}

ROUTES = [
    # Gaussian processes: the design
    (m.SAR, {}, "static", None),
    (m.SAR, {"T": 1}, "static", None),
    (m.SAR, PANEL, "static", 0),
    (m.SAR, {**PANEL, "unit_effects": "fixed"}, "static", 1),
    (m.SAR, {**PANEL, "time_effects": True}, "static", 2),
    (m.SAR, {**PANEL, "unit_effects": "fixed", "time_effects": True}, "static", 3),
    (m.SAR, {**PANEL, "unit_effects": "random"}, "random", None),
    (m.SAR, {**PANEL, "dynamic": True}, "dynamic", 0),
    (m.SAR, {**PANEL, "dynamic": True, "time_effects": True}, "dynamic", 2),
    (m.OLS, {**PANEL, "unit_effects": "random"}, "random", None),
    (m.SEM, {**PANEL, "unit_effects": "random"}, "random", None),
    (m.SDEM, {**PANEL, "unit_effects": "random"}, "random", None),
    (m.SLX, {**PANEL, "unit_effects": "fixed"}, "static", 1),
    (m.SDM, {**PANEL, "dynamic": True}, "dynamic", 0),
    (m.SDM, {**PANEL, "dynamic": "restricted"}, "restricted", 0),
    (m.SAR, {"time_col": "year", "unit_col": "id"}, "static", 0),
    # Count families: the design
    (m.SARNegBin, {}, "static", None),
    (m.SARNegBin, PANEL, "panel", 0),
    (
        m.SARNegBin,
        {**PANEL, "unit_effects": "random", "time_effects": True},
        "panel",
        3,
    ),
    (m.NegBin, {**PANEL, "unit_effects": "random"}, "panel", 1),
    (m.ZINB, {**PANEL, "time_effects": True}, "panel", 2),
    (m.SARZINB, {**PANEL, "unit_effects": "random"}, "panel", 1),
    (m.HurdleNegBin, PANEL, "panel", 0),
    (m.SARHurdleNegBin, {**PANEL, "unit_effects": "random"}, "panel", 1),
    # Logit, Tobit and probit families
    (m.SARLogit, {}, "static", None),
    (m.SARLogit, {"latent": True}, "latent", None),
    (m.SEMLogit, {}, "static", None),
    (m.SDEMLogit, {}, "static", None),
    (m.Logit, {}, "static", None),
    (m.SARTobit, PANEL, "static", None),
    (m.SEMTobit, PANEL, "static", None),
    (m.SEMProbit, {"group_effects": "random", "groups": [0, 1]}, "random", None),
    # Flows
    (m.SARFlow, {}, "SARFlow", None),
    (m.SARFlow, {"separable": True}, "SARFlowSeparable", None),
    (m.SARFlow, {"T": 2, "pair_effects": "fixed"}, "SARFlowPanelFE", 1),
    (
        m.SARFlow,
        {"T": 2, "separable": True, "time_effects": True},
        "SARFlowSeparablePanelFE",
        2,
    ),
    (m.SEMFlow, {"separable": True}, "SEMFlowSeparable", None),
    (m.OLSFlow, {"T": 2}, "OLSFlowPanelFE", 0),
    (m.NegBinFlow, {"T": 2, "pair_effects": "random"}, "NegBinFlowPanelRE", 1),
    (m.SARNegBinFlow, {"separable": True}, "SARNegBinFlowSeparable", None),
    (m.SARNegBinFlow, {"T": 2, "pair_effects": "random"}, "SARNegBinFlowPanelRE", 1),
    (m.SARPoissonFlow, {"separable": True}, "SARPoissonFlowSeparable", None),
    (m.SARZINBFlow, {"separable": True}, "SARZINBFlowSeparable", None),
    (
        m.SARHurdleNegBinFlow,
        {"separable": True, "T": 2, "pair_effects": "random", "time_effects": True},
        "SARHurdleNegBinFlowSeparablePanelRE",
        3,
    ),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "impl", "effects"),
    ROUTES,
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_routes(cls, kwargs, impl, effects):
    assert _route(cls, **kwargs) == (impl, effects)


REFUSALS = [
    (m.SARLogit, {"unit_effects": "random"}, NotImplementedError, "no unit effects"),
    (
        m.SEMProbit,
        {"group_effects": "random", "groups": [0], "T": 2},
        NotImplementedError,
        "group effects only",
    ),
    (m.SEMProbit, {"groups": [0, 1]}, NotImplementedError, "latent SEM probit"),
    (m.SEMProbit, {"group_effects": "random"}, ValueError, "group_col"),
    (m.ZINB, {}, NotImplementedError, "panels only"),
    (m.HurdleNegBin, {}, NotImplementedError, "panels only"),
    (m.SARZINB, {"unit_effects": "random"}, NotImplementedError, "only in a panel"),
    (m.SARNegBin, {"unit_effects": "random"}, NotImplementedError, "only in a panel"),
    (m.SARNegBin, {**PANEL, "dynamic": True}, NotImplementedError, "no dynamic"),
    (m.NegBin, {"unit_effects": "random"}, NotImplementedError, "only in a panel"),
    (m.SAR, {"unit_effects": "fixed"}, NotImplementedError, "only in a panel"),
    (m.SAR, {"time_effects": True}, ValueError, "needs a panel"),
    (m.SAR, {"dynamic": True}, ValueError, "needs a panel"),
    (m.SAR, {"unit_col": "id"}, ValueError, "unit_col needs time_col"),
    (m.SAR, {**PANEL, "dynamic": True, "unit_effects": "fixed"}, ValueError, "Nickell"),
    (m.SAR, {**PANEL, "dynamic": "restricted"}, NotImplementedError, "restricted"),
    (
        m.SAR,
        {**PANEL, "unit_effects": "random", "time_effects": True},
        NotImplementedError,
        "no time effects",
    ),
    (
        m.SDM,
        {**PANEL, "unit_effects": "random"},
        NotImplementedError,
        "random unit effects",
    ),
    (m.SAR, {**PANEL, "unit_effects": "both"}, ValueError, "unit_effects must be"),
    (
        m.SARNegBin,
        {**PANEL, "unit_effects": "fixed"},
        NotImplementedError,
        "random unit effects",
    ),
    (m.SARLogit, PANEL, NotImplementedError, "a panel"),
    (m.SDMTobit, PANEL, NotImplementedError, "a panel"),
    (m.SARTobit, {**PANEL, "time_effects": True}, NotImplementedError, "no effects"),
    (m.SARFlow, {"unit_effects": "fixed"}, TypeError, "pair_effects"),
    (m.SARFlow, {"pair_effects": "fixed"}, NotImplementedError, "only in a panel"),
    (m.SARFlow, {"separable": "yes"}, ValueError, "separable must be"),
    (m.OLSFlow, {"separable": True}, ValueError, "no spatial dependence"),
    (
        m.SARNegBinFlow,
        {"T": 2, "pair_effects": "fixed"},
        NotImplementedError,
        "fixed pair effects",
    ),
    (m.SARZINBFlow, {}, NotImplementedError, "separable=True"),
    (m.SARPoissonFlow, {"T": 2}, NotImplementedError, "panel"),
]


@pytest.mark.parametrize(
    ("cls", "kwargs", "error", "match"),
    REFUSALS,
    ids=lambda v: v.__name__ if isinstance(v, type) else None,
)
def test_refusals(cls, kwargs, error, match):
    with pytest.raises(error, match=match):
        _route(cls, **kwargs)


@pytest.fixture(scope="module")
def panel_data():
    W, G = rook_grid_weights(4)
    rng = np.random.default_rng(0)
    n, T = 16, 3
    X = np.column_stack([np.ones(n * T), rng.normal(size=n * T)])
    return {"y": rng.normal(size=n * T), "X": X, "W": G, "N": n, "T": T}


@pytest.mark.parametrize(
    "design, expected",
    [
        ({"unit_effects": "fixed", "time_effects": True}, "static"),
        ({"unit_effects": "random"}, "random"),
        ({"dynamic": True}, "dynamic"),
    ],
)
def test_a_panel_is_the_public_class(panel_data, design, expected):
    """One class serves every design: a panel is an instance of it, not of a variant."""
    model = m.SAR(**panel_data, **design)
    assert type(model) is m.SAR
    assert model._design == expected
    assert not isinstance(model, m.SEM)


def test_fixed_effects_panel_records_its_effects(panel_data):
    model = m.SAR(**panel_data, unit_effects="fixed", time_effects=True)
    assert model.effects == "two_way"


def test_a_count_cross_section_is_the_public_class(panel_data):
    n = panel_data["N"]
    y = np.random.default_rng(1).poisson(2.0, size=n)
    model = m.SARNegBin(y=y, X=panel_data["X"][:n], W=panel_data["W"])
    assert type(model) is m.SARNegBin
    assert model._design == "static"


@pytest.mark.parametrize("cls", [m.SARNegBin, m.ZINB, m.SARHurdleNegBin])
def test_a_count_panel_is_the_public_class(panel_data, cls):
    y = np.random.default_rng(1).poisson(2.0, size=panel_data["N"] * panel_data["T"])
    model = cls(**{**panel_data, "y": y}, unit_effects="random", time_effects=True)
    assert type(model) is cls
    assert model._design == "panel"
    assert model.effects == "two_way"


def test_cross_section_is_the_public_class(panel_data):
    n = panel_data["N"]
    model = m.SAR(y=panel_data["y"][:n], X=panel_data["X"][:n], W=panel_data["W"])
    assert type(model) is m.SAR
