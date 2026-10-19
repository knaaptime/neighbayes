"""One class per Gaussian process: the panel design is a constructor argument.

A Gaussian process class (OLS, SLX, SAR, SEM, SDM, SDEM) implements every design
it supports:

``"static"``
    The cross-section, and the static panel with or without fixed effects.
``"random"``
    The panel with random unit effects (:mod:`..panel._re`).
``"dynamic"``, ``"restricted"``
    The dynamic panel (:mod:`..panel._dynamic`); ``"restricted"`` is the
    restricted dynamic SDM and otherwise reuses the dynamic implementations.

A method that differs by design is implemented as ``_<method>_<design>``
(``_build_pymc_model_random``, ``_fit_dynamic``); :class:`GaussianDesigns`
calls the one for the model's design, and a design without its own
implementation uses the shared one.  Each class lists the designs it
implements in ``_DESIGNS``, with the attributes each sets where they differ
from the class's.
"""

from __future__ import annotations

import inspect

from .._design import (
    _NICKELL,
    _cross_section_only,
    _effects_code,
    _is_panel,
    _pop_design,
)
from ..base import SpatialModel

#: Designs that reuse another design's implementations.
_FALLBACK = {"restricted": "dynamic"}


def _by_design(name: str):
    """A method that calls ``_<name>_<design>``, or the shared ``name``."""

    def method(self, *args, **kwargs):
        return self._implementation(name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"GaussianDesigns.{name}"
    return method


class GaussianDesigns:
    """Design dispatch for the Gaussian process classes (see the module)."""

    _design: str = "static"
    #: design → the instance attributes it sets where they differ from the
    #: class's.  Only listed designs are implemented.
    _DESIGNS: dict[str, dict] = {"static": {}}

    def __init__(
        self,
        formula=None,
        data=None,
        y=None,
        X=None,
        W=None,
        *,
        unit_effects=None,
        time_effects=False,
        dynamic=False,
        centered=False,
        **kwargs,
    ):
        design, kwargs = self._resolve_design(
            dict(
                kwargs,
                unit_effects=unit_effects,
                time_effects=time_effects,
                dynamic=dynamic,
            )
        )
        if centered and design != "random":
            raise ValueError(
                "centered applies to random unit effects (unit_effects='random')."
            )
        self._design = design
        for attr, value in self._DESIGNS[design].items():
            setattr(self, attr, value)
        if design == "random":
            self.centered = bool(centered)
        super().__init__(formula, data, y, X, W, **kwargs)
        if (
            design == "static"
            and self._is_panel
            and self._gibbs_key == ("gaussian", "cross_section")
        ):
            self._gibbs_key = ("gaussian", "panel_fe")
        init = getattr(self, f"_init_{design}", None)
        if init is not None:
            init()

    def _implementation(self, name: str):
        """The bound method ``name`` dispatches to under this model's design."""
        stem = name.lstrip("_")
        for d in (self._design, _FALLBACK.get(self._design)):
            impl = getattr(self, f"_{stem}_{d}", None) if d is not None else None
            if impl is not None:
                return impl
        return getattr(super(GaussianDesigns, self), name)

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """The design the arguments describe, and the base class's arguments."""
        name = cls.__name__
        unit, time, dynamic = _pop_design(kw)
        if not _is_panel(kw):
            _cross_section_only(name, time, dynamic)
            if unit is not None:
                raise NotImplementedError(
                    f"{name} takes unit effects only in a panel (pass time_col or T)."
                )
            return "static", kw
        if dynamic:
            if unit is not None:
                raise ValueError(_NICKELL)
            design = "restricted" if dynamic == "restricted" else "dynamic"
            if design not in cls._DESIGNS:
                raise NotImplementedError(
                    f"{name} is not implemented with dynamic={dynamic!r}."
                )
            kw["effects"] = _effects_code(False, time)
            return design, kw
        if unit == "random":
            if time:
                raise NotImplementedError(
                    f"{name} with random unit effects takes no time effects."
                )
            if "random" not in cls._DESIGNS:
                raise NotImplementedError(
                    f"{name} is not implemented with random unit effects."
                )
            return "random", kw
        kw["effects"] = _effects_code(unit == "fixed", time)
        return "static", kw

    _build_pymc_model = _by_design("_build_pymc_model")
    _fit_gibbs = _by_design("_fit_gibbs")
    _fitted_mean_from_posterior = _by_design("_fitted_mean_from_posterior")
    _compute_impacts_posterior = _by_design("_compute_impacts_posterior")
    _model_coords = _by_design("_model_coords")
    _beta_names = _by_design("_beta_names")
    fit = _by_design("fit")
    fit.__doc__ = SpatialModel.fit.__doc__
    fit.__wrapped__ = SpatialModel.fit


def _constructor_signature() -> inspect.Signature:
    """The design arguments, then the base class's: what ``**kwargs`` takes.

    Set on the class: the facade metaclass's ``__call__`` would otherwise hide
    the constructor's arguments from :func:`inspect.signature`.
    """
    own = list(inspect.signature(GaussianDesigns.__init__).parameters.values())[1:]
    base = inspect.signature(SpatialModel.__init__).parameters
    params = [p for p in own if p.kind is not p.VAR_KEYWORD]
    names = {p.name for p in params}
    params += [
        p
        for name, p in base.items()
        if p.kind is p.KEYWORD_ONLY and name not in names and name != "effects"
    ]
    return inspect.Signature(params)


GaussianDesigns.__signature__ = _constructor_signature()
