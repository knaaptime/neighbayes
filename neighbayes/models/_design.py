"""Design dispatch for the public model classes.

A public class names a process and a family (ADR 0002): ``SAR``, ``SARNegBin``,
``SEMProbit``, ``SARNegBinFlow``.  Panel structure, effects, dynamics, and the
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

The Gaussian, Tobit, count, logit and probit classes implement every design
themselves (see :mod:`._mixins._designs`, :mod:`.cross_section.tobit` and
:mod:`.cross_section.sem_probit`).  The flow families are facades: each one's
``_dispatch`` maps those arguments to the class that implements the
combination and to that class's own keyword arguments, and raises for a
combination the package does not implement.
:class:`FacadeMeta` runs it when the public class is called and registers the
implementation as a virtual subclass, so ``isinstance(model, SARFlow)`` holds
for a SARFlow panel.
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


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

# Per family: the pair-effects treatment its panels take, and the implementations
# indexed by (separable, panel).  A missing entry is not implemented.
_FLOW = {
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
