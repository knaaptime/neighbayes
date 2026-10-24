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

Every class implements its designs itself (see :mod:`._mixins._designs`,
:mod:`.cross_section.tobit` and :mod:`.cross_section.sem_probit`); this module
holds the argument validation they share.  :class:`FacadeMeta` routes a
class's construction through a ``_dispatch`` table where one is defined.
"""

from __future__ import annotations

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
