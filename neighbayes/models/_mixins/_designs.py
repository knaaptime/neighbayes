"""One class per process: the design is a constructor argument.

A process class implements every design it supports.  A method that differs by
design is implemented as ``_<method>_<design>`` (``_build_pymc_model_random``,
``_fit_gibbs_panel``); the family's dispatcher calls the one for the model's
design, and a design without its own implementation uses the shared one.  Each
class lists the designs it implements in ``_DESIGNS``, with the attributes each
sets where they differ from the class's.

The Gaussian processes (OLS, SLX, SAR, SEM, SDM, SDEM), :class:`GaussianDesigns`:

``"static"``
    The cross-section, and the static panel with or without fixed effects.
``"random"``
    The panel with random unit effects (:mod:`..panel._re`).
``"dynamic"``, ``"restricted"``
    The dynamic panel (:mod:`..panel._dynamic`); ``"restricted"`` is the
    restricted dynamic SDM and otherwise reuses the dynamic implementations.

The count families (NegBin, SARNegBin, SEMNegBin, SDEMNegBin, ZINB, SARZINB,
HurdleNegBin, SARHurdleNegBin), :class:`CountDesigns`:

``"static"``
    The cross-section: SARNegBin's reduced form, and the latent SEM of
    SEMNegBin and SDEMNegBin.
``"latent"``
    ``SARNegBin(latent=True)``, the latent SAR, which propagates the negative
    binomial's gamma disturbance (:mod:`..cross_section._latent_negbin`).
``"panel"``
    The panel, pooled or with random unit effects and fixed time effects
    (:mod:`..panel._count`, :mod:`..panel._zinb`, :mod:`..panel._hurdle`).

The binary families (Logit, SARLogit, SEMLogit, SDEMLogit),
:class:`BinaryDesigns`:

``"static"``
    The cross-section: SARLogit's reduced form, and the latent SEM of SEMLogit
    and SDEMLogit (a SEM has no reduced form to distinguish, ADR 0003).
``"latent"``
    ``SARLogit(latent=True)``, the latent SAR, which propagates the logistic
    disturbance (:mod:`..cross_section._latent_logit`).

The flow families (OLSFlow, SARFlow, SEMFlow), :class:`FlowDesigns`; each
design is a cross-section (``T=1``) or a panel with pair and time effects:

``"static"``
    The unrestricted filter, three free spatial parameters (none for the
    aspatial OLSFlow).
``"separable"``
    ``separable=True``: ``ρ_w = -ρ_d ρ_o``, so the filter factors as
    ``L_o ⊗ L_d``.
"""

from __future__ import annotations

import inspect

import numpy as np

from .._design import (
    _NICKELL,
    _cross_section_only,
    _effects_code,
    _is_panel,
    _pop_design,
)
from ..base import SpatialModel
from ._flow_shared import FlowSharedMethods


def _latent_flag(cls, latent, suffix: str) -> bool:
    """Validate ``latent=``: ``True`` only where the class implements the design."""
    if not isinstance(latent, bool):
        raise ValueError(f"latent must be True or False; got {latent!r}.")
    if latent and "latent" not in cls._DESIGNS:
        name = cls.__name__
        raise NotImplementedError(
            f"{name} has no latent=True design"
            + (
                f" (it is the latent {name.removesuffix(suffix)} already)."
                if hasattr(cls, "_latent_process")
                else "."
            )
        )
    return latent


def _by_design(name: str):
    """A method that calls ``_<name>_<design>``, or the shared ``name``."""

    def method(self, *args, **kwargs):
        return self._implementation(name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = f"DesignDispatch.{name}"
    return method


class DesignDispatch:
    """Design bookkeeping and method dispatch (see the module)."""

    _design: str = "static"
    #: design → the instance attributes it sets where they differ from the
    #: class's.  Only listed designs are implemented.
    _DESIGNS: dict[str, dict] = {"static": {}}
    #: Designs that reuse another design's implementations.
    _FALLBACK: dict[str, str] = {}

    def _set_design(self, design: str) -> None:
        self._design = design
        for attr, value in self._DESIGNS[design].items():
            setattr(self, attr, value)

    def _implementation(self, name: str):
        """The bound method ``name`` dispatches to under this model's design."""
        stem = name.lstrip("_")
        for d in (self._design, self._FALLBACK.get(self._design)):
            impl = getattr(self, f"_{stem}_{d}", None) if d is not None else None
            if impl is not None:
                return impl
        impl = getattr(super(DesignDispatch, self), name, None)
        if impl is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement {name}() with design "
                f"{self._design!r}."
            )
        return impl


class GaussianDesigns(DesignDispatch):
    """Design dispatch for the Gaussian process classes (see the module)."""

    _FALLBACK = {"restricted": "dynamic"}

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
        self._set_design(design)
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


class BinaryDesigns(DesignDispatch):
    """Design dispatch for the binary families (see the module).

    The response is validated as binary; there is no panel, and a
    cross-section takes no unit effects.  ``latent=True`` selects the latent
    SAR where the class implements it.
    """

    _likelihood: str = "binary"

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
        latent=False,
        **kwargs,
    ):
        design, kwargs = self._resolve_design(
            dict(
                kwargs,
                unit_effects=unit_effects,
                time_effects=time_effects,
                dynamic=dynamic,
                latent=latent,
            )
        )
        if kwargs.get("robust"):
            raise NotImplementedError(
                f"robust=True is not supported for {type(self).__name__}."
            )
        self._set_design(design)
        super().__init__(formula, data, y, X, W, **kwargs)
        if not np.isin(self._y, (0.0, 1.0)).all():
            raise ValueError("y must be binary with values in {0, 1}.")
        init = getattr(self, f"_init_{design}", None)
        if init is not None:
            init()

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """The design the arguments describe, and the base class's arguments."""
        name = cls.__name__
        unit, time, dynamic = _pop_design(kw)
        latent = kw.pop("latent", False)
        if _is_panel(kw):
            raise NotImplementedError(f"{name} is not implemented with a panel.")
        _cross_section_only(name, time, dynamic)
        if unit is not None:
            # One draw per observation is a disturbance, not a unit effect.
            raise NotImplementedError(f"{name} takes no unit effects.")
        latent = _latent_flag(cls, latent, "Logit")
        return ("latent" if latent else "static"), kw

    _build_pymc_model = _by_design("_build_pymc_model")
    _fit_gibbs = _by_design("_fit_gibbs")
    _fitted_mean_from_posterior = _by_design("_fitted_mean_from_posterior")
    fitted_probabilities = _by_design("fitted_probabilities")
    fitted_probabilities.__doc__ = (
        """Fitted ``P(y = 1)`` at the posterior-mean parameters."""
    )


#: The zero part's design and weights, taken by the two-part count families.
_ZERO_PART_ARGS = ("X_zero", "formula_zero", "W_zero")


class CountDesigns(DesignDispatch):
    """Design dispatch for the count families (see the module).

    The response is validated as counts for every design; a two-part family
    (``_zero_part``) also takes the zero part's design and weights, which it
    builds before the panel's effects drop the columns they absorb.
    ``latent=True`` selects the latent SAR of a cross-section where the class
    implements it.
    """

    _likelihood: str = "count"
    #: Whether the family has a zero part (ZINB, hurdle).
    _zero_part: bool = False
    #: Whether the family needs a positive count (the hurdle's count part).
    _needs_positive: bool = False

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
        mundlak=False,
        latent=False,
        **kwargs,
    ):
        zero = (
            {k: kwargs.pop(k, None) for k in _ZERO_PART_ARGS}
            if self._zero_part
            else None
        )
        design, kwargs = self._resolve_design(
            dict(
                kwargs,
                unit_effects=unit_effects,
                time_effects=time_effects,
                dynamic=dynamic,
                latent=latent,
            )
        )
        effects = kwargs.pop("effects", 0)
        if mundlak and design != "panel":
            raise ValueError("mundlak=True needs a panel with unit effects.")
        if kwargs.get("robust"):
            raise NotImplementedError(
                f"robust=True is not supported for {type(self).__name__}."
            )
        self._set_design(design)
        super().__init__(formula, data, y, X, W, **kwargs)
        self._init_counts()
        if zero is not None:
            self._init_zero_part(data, **zero)
        if design == "panel":
            self._init_panel(effects, mundlak)
        else:
            init = getattr(self, f"_init_{design}", None)
            if init is not None:
                init()

    def _init_counts(self) -> None:
        """Validate the response as counts: ``_y_int`` and a float ``_y``."""
        from ._count_panel import require_counts

        name = type(self).__name__
        self._y_int = require_counts(self._y_raw, name).reshape(-1)
        if self._needs_positive and not np.any(self._y_int > 0):
            raise ValueError(f"{name} needs at least one positive count.")
        self._y = self._y_int.astype(np.float64)
        if self._W_sparse is not None:
            self._Wy = self._sparse_panel_lag(self._y)

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """The design the arguments describe, and the base class's arguments."""
        name = cls.__name__
        unit, time, dynamic = _pop_design(kw)
        latent = _latent_flag(cls, kw.pop("latent", False), "NegBin")
        if latent and _is_panel(kw):
            raise NotImplementedError(
                f"{name}(latent=True) is implemented for the cross-section only."
            )
        if not _is_panel(kw):
            _cross_section_only(name, time, dynamic)
            if unit is not None:
                # One draw per observation is a disturbance, not a unit effect.
                raise NotImplementedError(
                    f"{name} takes unit effects only in a panel (pass time_col or T)."
                )
            if "static" not in cls._DESIGNS:
                raise NotImplementedError(
                    f"{name} is implemented for panels only (pass time_col or T)."
                )
            return ("latent" if latent else "static"), kw
        if dynamic:
            raise NotImplementedError(f"{name} has no dynamic panel.")
        if unit == "fixed":
            raise NotImplementedError(
                f"{name} panels take random unit effects, not fixed ones."
            )
        if "panel" not in cls._DESIGNS:
            raise NotImplementedError(f"{name} is not implemented with a panel.")
        kw["effects"] = _effects_code(unit == "random", time)
        return "panel", kw

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__signature__ = _constructor_signature(
            CountDesigns.__init__, _ZERO_PART_ARGS if cls._zero_part else ()
        )

    _build_pymc_model = _by_design("_build_pymc_model")
    _fit_gibbs = _by_design("_fit_gibbs")
    _fitted_mean_from_posterior = _by_design("_fitted_mean_from_posterior")
    _compute_impacts_posterior = _by_design("_compute_impacts_posterior")
    _model_coords = _by_design("_model_coords")
    _part_etas = _by_design("_part_etas")
    posterior_predictive = _by_design("posterior_predictive")
    posterior_predictive.__doc__ = """Posterior-predictive counts, shape ``(n_draws, n_obs)``.

        Parameters
        ----------
        n_draws : int, optional
            Posterior draws to simulate from; all of them by default.
        random_seed : int, optional
            Seed for the simulation.
        """


class FlowDesigns(DesignDispatch):
    """Design dispatch for the flow families (see the module).

    ``separable=True`` selects the separable design.  ``T`` makes a panel, the
    cross-section being one period, with ``pair_effects`` (the family's
    ``_pair_effects`` treatment) and ``time_effects``.
    """

    #: The pair effects the family's panels take.
    _pair_effects: str = "fixed"

    def __init__(
        self,
        y,
        X,
        W,
        *,
        pair_effects=None,
        time_effects=False,
        dynamic=False,
        separable=False,
        **kwargs,
    ):
        design, kwargs = self._resolve_design(
            dict(
                kwargs,
                pair_effects=pair_effects,
                time_effects=time_effects,
                dynamic=dynamic,
                separable=separable,
            )
        )
        self._set_design(design)
        super().__init__(y, X, W, **kwargs)
        init = getattr(self, f"_init_{design}", None)
        if init is not None:
            init()

    @classmethod
    def _resolve_design(cls, kw: dict) -> tuple[str, dict]:
        """The design the arguments describe, and the base class's arguments."""
        name = cls.__name__
        pair, time, dynamic = _pop_design(kw, pair=True)
        separable = kw.pop("separable", False)
        if not isinstance(separable, bool):
            raise ValueError(f"separable must be True or False; got {separable!r}.")
        if dynamic:
            raise NotImplementedError(f"{name} has no dynamic panel.")
        design = "separable" if separable else "static"
        if design not in cls._DESIGNS:
            if separable:
                raise ValueError(f"{name} has no spatial dependence to restrict.")
            raise NotImplementedError(
                f"{name} is implemented as the separable flow only "
                "(pass separable=True)."
            )
        if not _is_panel(kw):
            _cross_section_only(name, time, dynamic)
            if pair is not None:
                raise NotImplementedError(
                    f"{name} takes pair effects only in a panel (pass T)."
                )
            return design, kw
        if pair is not None and pair != cls._pair_effects:
            raise NotImplementedError(f"{name} panels take no {pair} pair effects.")
        kw["effects"] = _effects_code(pair is not None, time)
        return design, kw

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from ..flow._flow import FlowModel

        cls.__signature__ = _constructor_signature(
            FlowDesigns.__init__, base=FlowModel.__init__
        )

    _build_pymc_model = _by_design("_build_pymc_model")
    _compute_impacts_posterior = _by_design("_compute_impacts_posterior")
    _compute_jacobian_log_det = _by_design("_compute_jacobian_log_det")
    _simulate_y_rep = _by_design("_simulate_y_rep")
    fit = _by_design("fit")
    fit.__doc__ = FlowSharedMethods.fit.__doc__
    fit.__wrapped__ = FlowSharedMethods.fit


def _constructor_signature(
    init, extra: tuple[str, ...] = (), base=None
) -> inspect.Signature:
    """The design arguments, ``extra``, then the base class's: what ``**kwargs`` takes.

    ``base`` is the base class's ``__init__`` (``SpatialModel``'s by default).
    Set on the class: the facade metaclass's ``__call__`` would otherwise hide
    the constructor's arguments from :func:`inspect.signature`.
    """
    own = list(inspect.signature(init).parameters.values())[1:]
    base = inspect.signature(base or SpatialModel.__init__).parameters
    params = [p for p in own if p.kind is not p.VAR_KEYWORD]
    params += [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
        for name in extra
    ]
    names = {p.name for p in params}
    params += [
        p
        for name, p in base.items()
        if p.kind is p.KEYWORD_ONLY and name not in names and name != "effects"
    ]
    return inspect.Signature(params)


GaussianDesigns.__signature__ = _constructor_signature(GaussianDesigns.__init__)
BinaryDesigns.__signature__ = _constructor_signature(BinaryDesigns.__init__)
CountDesigns.__signature__ = _constructor_signature(CountDesigns.__init__)
