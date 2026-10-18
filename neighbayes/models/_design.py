"""Design dispatch for the public model classes.

A public class names a process and a family (ADR 0002): ``SAR``, ``SARNegBin``,
``SEMLogit``, ``SARNegBinFlow``.  Panel structure, effects, dynamics, and the
flow restriction are constructor arguments:

``time_col`` / ``T``
    Giving time makes the model a panel; ``T=1`` is a cross-section.
``unit_effects`` / ``pair_effects``
    ``None``, ``"fixed"``, or ``"random"``; flows index their effects by pair.
``time_effects``
    ``True`` adds fixed time effects (panels only).
``dynamic``
    ``True`` adds the time lag :math:`y_{t-1}`; ``"restricted"`` is the
    restricted dynamic SDM (panels only).
``separable``
    ``True`` pins :math:`\\rho_w = -\\rho_o\\rho_d` (flows only).

Each facade's ``_dispatch`` maps those arguments to the class that implements
the combination and to that class's own keyword arguments, and raises for a
combination the package does not implement.  :class:`FacadeMeta` runs it when
the public class is called and registers the implementation as a virtual
subclass, so ``isinstance(model, SAR)`` holds for a SAR panel.
"""

from __future__ import annotations

import importlib
from abc import ABCMeta

_UNIT_VALUES = (None, "fixed", "random")
_DYNAMIC_VALUES = (False, True, "restricted")

_NICKELL = (
    "Dynamic panels take no unit effects: within-demeaning correlates the "
    "demeaned time lag with the demeaned disturbance (Nickell 1981)."
)


class FacadeMeta(ABCMeta):
    """Metaclass that routes a public class's construction through ``_dispatch``.

    Only a class whose own namespace defines ``_dispatch`` dispatches; its
    subclasses and the implementation classes construct normally.
    """

    def __call__(cls, *args, **kwargs):
        dispatch = cls.__dict__.get("_dispatch")
        if dispatch is None:
            return super().__call__(*args, **kwargs)
        impl, kwargs = dispatch.__func__(cls, dict(kwargs))
        if impl is cls:
            return super().__call__(*args, **kwargs)
        cls.register(impl)
        return impl(*args, **kwargs)


def _load(path: str) -> type:
    """Import ``"subpackage.module:Class"`` relative to ``neighbayes.models``."""
    module, name = path.split(":")
    return getattr(importlib.import_module(f"neighbayes.models.{module}"), name)


def _pop_design(kw: dict, *, pair: bool = False) -> tuple:
    """Remove and validate the effects and dynamics arguments."""
    index = "pair_effects" if pair else "unit_effects"
    other = "unit_effects" if pair else "pair_effects"
    if "effects" in kw:
        raise TypeError(
            f"effects= is not an argument; pass {index}= and time_effects= instead."
        )
    if other in kw:
        raise TypeError(
            f"{'Flow' if pair else 'This'} models index their effects with "
            f"{index!r}, not {other!r}."
        )
    unit = kw.pop(index, None)
    time = kw.pop("time_effects", False)
    dynamic = kw.pop("dynamic", False)
    if unit not in _UNIT_VALUES:
        raise ValueError(f"{index} must be None, 'fixed', or 'random'; got {unit!r}.")
    if not isinstance(time, bool):
        raise ValueError(f"time_effects must be True or False; got {time!r}.")
    if dynamic not in _DYNAMIC_VALUES:
        raise ValueError(
            f"dynamic must be False, True, or 'restricted'; got {dynamic!r}."
        )
    return unit, time, dynamic


def _is_panel(kw: dict) -> bool:
    """A model is a panel when time is given; ``T=1`` is a cross-section."""
    if kw.get("time_col") is not None:
        return True
    T = kw.get("T")
    if T is None and kw.get("N") is not None:
        raise ValueError(
            "N and T are required for a matrix-mode panel; got N without T."
        )
    if T is None or int(T) == 1:
        kw.pop("T", None)
        kw.pop("N", None)
        kw.pop("time_col", None)
        if kw.pop("unit_col", None) is not None:
            raise ValueError("unit_col needs time_col: a panel is declared by time.")
        return False
    return True


def _effects_code(unit: bool, time: bool) -> int:
    """The implementations' effects code: 0 none, 1 unit/pair, 2 time, 3 both."""
    return int(unit) + 2 * int(time)


def _cross_section_only(name: str, time: bool, dynamic) -> None:
    if time:
        raise ValueError(f"{name}: time_effects needs a panel (pass time_col or T).")
    if dynamic:
        raise ValueError(f"{name}: dynamic needs a panel (pass time_col or T).")


def _need(spec: dict, key: str, name: str, what: str) -> type:
    path = spec.get(key)
    if path is None:
        raise NotImplementedError(f"{name} is not implemented with {what}.")
    return _load(path)


# ---------------------------------------------------------------------------
# Gaussian processes: OLS, SLX, SAR, SEM, SDM, SDEM
# ---------------------------------------------------------------------------

_GAUSSIAN = {
    "OLS": {
        "fe": "panel._fe:OLSPanelFE",
        "re": "panel._re:OLSPanelRE",
        "dynamic": "panel._dynamic:OLSPanelDynamic",
    },
    "SLX": {
        "fe": "panel._fe:SLXPanelFE",
        "dynamic": "panel._dynamic:SLXPanelDynamic",
    },
    "SAR": {
        "fe": "panel._fe:SARPanelFE",
        "re": "panel._re:SARPanelRE",
        "dynamic": "panel._dynamic:SARPanelDynamic",
    },
    "SEM": {
        "fe": "panel._fe:SEMPanelFE",
        "re": "panel._re:SEMPanelRE",
        "dynamic": "panel._dynamic:SEMPanelDynamic",
    },
    "SDM": {
        "fe": "panel._fe:SDMPanelFE",
        "dynamic": "panel._dynamic:SDMPanelDynamic",
        "restricted": "panel._dynamic:SDMPanelDynamicRestricted",
    },
    "SDEM": {
        "fe": "panel._fe:SDEMPanelFE",
        "re": "panel._re:SDEMPanelRE",
        "dynamic": "panel._dynamic:SDEMPanelDynamic",
    },
}


def gaussian(cls, kw: dict):
    """Gaussian processes: fixed- or random-effects and dynamic panels."""
    name = cls.__name__
    spec = _GAUSSIAN[name]
    unit, time, dynamic = _pop_design(kw)
    if not _is_panel(kw):
        _cross_section_only(name, time, dynamic)
        if unit is not None:
            raise NotImplementedError(
                f"{name} takes unit effects only in a panel (pass time_col or T)."
            )
        return cls, kw
    if dynamic:
        if unit is not None:
            raise ValueError(_NICKELL)
        key = "restricted" if dynamic == "restricted" else "dynamic"
        impl = _need(spec, key, name, f"dynamic={dynamic!r}")
        kw["effects"] = _effects_code(False, time)
        return impl, kw
    if unit == "random":
        if time:
            raise NotImplementedError(
                f"{name} with random unit effects takes no time effects."
            )
        return _need(spec, "re", name, "random unit effects"), kw
    kw["effects"] = _effects_code(unit == "fixed", time)
    return _need(spec, "fe", name, "a panel"), kw


# ---------------------------------------------------------------------------
# Logit, count, and Tobit families
# ---------------------------------------------------------------------------

# "xs": the facade implements the cross-section itself (True) or names why it
# does not (a message); "xs_random": the cross-section SAR (or SEM) with random
# effects; "panel": the panel, whose unit effects are random and time effects
# fixed in the count families.
_DISCRETE = {
    "Logit": {"xs": True},
    "SARLogit": {"xs": True, "xs_random": "cross_section.sar_logit_re:SARLogitRE"},
    "SEMLogit": {
        "xs": "The latent SEM logit is not implemented; pass unit_effects='random' "
        "for the SEM logit with random effects",
        "xs_random": "cross_section.sem_logit:SEMLogitRE",
    },
    "NegBin": {"xs": True, "panel": "panel._count:NegBinPanelRE"},
    "SARNegBin": {
        "xs": True,
        "xs_random": "cross_section.sar_negbin_re:SARNegBinRE",
        "panel": "panel._count:SARNegBinPanelRE",
    },
    "ZINB": {
        "xs": "ZINB is implemented for panels only (pass time_col or T)",
        "panel": "panel._zinb:ZINBPanelRE",
    },
    "SARZINB": {"xs": True, "panel": "panel._zinb:SARZINBPanelRE"},
    "HurdleNegBin": {
        "xs": "HurdleNegBin is implemented for panels only (pass time_col or T)",
        "panel": "panel._hurdle:HurdleNegBinPanelRE",
    },
    "SARHurdleNegBin": {"xs": True, "panel": "panel._hurdle:SARHurdleNegBinPanelRE"},
}


def discrete(cls, kw: dict):
    """Logit and count families: random effects in a cross-section, count panels."""
    name = cls.__name__
    spec = _DISCRETE[name]
    unit, time, dynamic = _pop_design(kw)
    if not _is_panel(kw):
        _cross_section_only(name, time, dynamic)
        if unit == "random":
            return _need(spec, "xs_random", name, "random unit effects"), kw
        if unit is not None:
            raise NotImplementedError(f"{name} takes no fixed unit effects.")
        if spec["xs"] is not True:
            raise NotImplementedError(spec["xs"] + ".")
        return cls, kw
    if dynamic:
        raise NotImplementedError(f"{name} has no dynamic panel.")
    if unit == "fixed":
        raise NotImplementedError(
            f"{name} panels take random unit effects, not fixed ones."
        )
    kw["effects"] = _effects_code(unit == "random", time)
    return _need(spec, "panel", name, "a panel"), kw


_TOBIT = {
    "SARTobit": "panel._tobit:SARTobitPanel",
    "SEMTobit": "panel._tobit:SEMTobitPanel",
    "SDMTobit": None,
}


def tobit(cls, kw: dict):
    """Tobit processes: a panel without effects."""
    name = cls.__name__
    unit, time, dynamic = _pop_design(kw)
    if unit is not None or time or dynamic:
        raise NotImplementedError(f"{name} takes no effects and no dynamics.")
    if not _is_panel(kw):
        return cls, kw
    return _need({"panel": _TOBIT[name]}, "panel", name, "a panel"), kw


def sem_probit(cls, kw: dict):
    """The SEM probit with random group effects (LeSage and Pace 2009, §10.6)."""
    unit, time, dynamic = _pop_design(kw)
    group_effects = kw.pop("group_effects", None)
    if unit is not None or time or dynamic or _is_panel(kw):
        raise NotImplementedError("SEMProbit takes group effects only.")
    if group_effects != "random":
        raise NotImplementedError(
            "The latent SEM probit is not implemented; pass group_col (or groups) "
            "and group_effects='random' for the SEM probit with random group effects."
        )
    if not {"group_col", "groups", "group_sizes"} & kw.keys():
        raise ValueError(
            "Group effects need group_col (formula mode), or groups or group_sizes "
            "(matrix mode)."
        )
    return _load("cross_section.sem_probit:SEMProbitRE"), kw


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

# Per family: the pair-effects treatment its panels take, and the implementations
# indexed by (separable, panel).  A missing entry is not implemented.
_FLOW = {
    "OLSFlow": {
        "pairs": "fixed",
        "aspatial": True,
        (False, True): "flow_panel._panel:OLSFlowPanelFE",
    },
    "SARFlow": {
        "pairs": "fixed",
        (True, False): "flow._flow:SARFlowSeparable",
        (False, True): "flow_panel._panel:SARFlowPanelFE",
        (True, True): "flow_panel._panel:SARFlowSeparablePanelFE",
    },
    "SEMFlow": {
        "pairs": "fixed",
        (True, False): "flow._flow:SEMFlowSeparable",
        (False, True): "flow_panel._panel:SEMFlowPanelFE",
        (True, True): "flow_panel._panel:SEMFlowSeparablePanelFE",
    },
    "NegBinFlow": {
        "pairs": "random",
        "aspatial": True,
        (False, True): "flow_panel._panel:NegBinFlowPanelRE",
    },
    "SARNegBinFlow": {
        "pairs": "random",
        (True, False): "flow._flow:SARNegBinFlowSeparable",
        (False, True): "flow_panel._panel:SARNegBinFlowPanelRE",
        (True, True): "flow_panel._panel:SARNegBinFlowSeparablePanelRE",
    },
    "SARPoissonFlow": {
        "pairs": None,
        (True, False): "flow._flow:SARPoissonFlowSeparable",
    },
    "SARZINBFlow": {
        "pairs": "random",
        "pure": True,
        (True, False): "flow_panel._panel:SARZINBFlowSeparable",
        (True, True): "flow_panel._panel:SARZINBFlowSeparablePanelRE",
    },
    "SARHurdleNegBinFlow": {
        "pairs": "random",
        "pure": True,
        (True, False): "flow_panel._hurdle:SARHurdleNegBinFlowSeparable",
        (True, True): "flow_panel._hurdle:SARHurdleNegBinFlowSeparablePanelRE",
    },
}


def flow(cls, kw: dict):
    """Flow models: the separable restriction, flow panels, and pair effects."""
    name = cls.__name__
    spec = _FLOW[name]
    pair, time, dynamic = _pop_design(kw, pair=True)
    separable = kw.pop("separable", False)
    if not isinstance(separable, bool):
        raise ValueError(f"separable must be True or False; got {separable!r}.")
    if separable and spec.get("aspatial"):
        raise ValueError(f"{name} has no spatial dependence to restrict.")
    if dynamic:
        raise NotImplementedError(f"{name} has no dynamic panel.")
    panel = _is_panel(kw)
    if not panel:
        _cross_section_only(name, time, dynamic)
        if pair is not None:
            raise NotImplementedError(
                f"{name} takes pair effects only in a panel (pass T)."
            )
        if not separable and not spec.get("pure"):
            return cls, kw
    elif pair is not None and pair != spec["pairs"]:
        raise NotImplementedError(f"{name} panels take no {pair} pair effects.")
    path = spec.get((separable, panel))
    if path is None:
        what = ("the separable" if separable else "the unrestricted") + (
            " panel" if panel else " cross-section"
        )
        hint = " (pass separable=True)" if not separable and spec.get("pure") else ""
        raise NotImplementedError(f"{name} is not implemented as {what}{hint}.")
    if panel:
        kw["effects"] = _effects_code(pair is not None, time)
    return _load(path), kw
