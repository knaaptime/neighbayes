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

The flow families, :class:`FlowDesigns` (OLSFlow, SARFlow, SEMFlow) and
:class:`FlowCountDesigns` (NegBinFlow, SARNegBinFlow, SARPoissonFlow,
SARZINBFlow, SARHurdleNegBinFlow); each design is a cross-section (``T=1``)
or a panel with pair and time effects:

``"static"``
    The unrestricted filter, three free spatial parameters (none for the
    aspatial OLSFlow and NegBinFlow).
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
    #: Whether the class's latent design is implemented as a panel too.
    _latent_panel: bool = False
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
        panel = "effects" in kwargs
        effects = kwargs.pop("effects", 0)
        if mundlak and not panel:
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
        if panel:
            # The count effects' bookkeeping, shared by the latent panel.
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
        if latent and _is_panel(kw) and not cls._latent_panel:
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
        if not latent and "panel" not in cls._DESIGNS:
            raise NotImplementedError(f"{name} is not implemented with a panel.")
        kw["effects"] = _effects_code(unit == "random", time)
        return ("latent" if latent else "panel"), kw

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
    #: Whether the family is implemented as a panel.
    _has_panel: bool = True

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
        self._construct(y, X, W, **kwargs)
        init = getattr(self, f"_init_{design}", None)
        if init is not None:
            init()

    def _construct(self, y, X, W, **kwargs) -> None:
        """Build the flow model from the base class's arguments."""
        super().__init__(y, X, W, **kwargs)

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
        if not cls._has_panel:
            raise NotImplementedError(f"{name} is not implemented with a panel.")
        if pair is not None and pair != cls._pair_effects:
            raise NotImplementedError(f"{name} panels take no {pair} pair effects.")
        kw["effects"] = _effects_code(pair is not None, time)
        return design, kw

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_DESIGNS" not in cls.__dict__:  # a design layer, not a family
            return
        from ..flow._flow import FlowModel

        # ``fit`` reads as the family's own (its first design's), not the shared NUTS fit.
        impl = next(
            (
                getattr(cls, f"_fit_{d}")
                for d in cls._DESIGNS
                if hasattr(cls, f"_fit_{d}")
            ),
            None,
        )
        if impl is not None and "fit" not in cls.__dict__:
            fit = _by_design("fit")
            fit.__doc__, fit.__wrapped__ = impl.__doc__, impl
            cls.fit = fit

        cls.__signature__ = _constructor_signature(
            FlowDesigns.__init__, cls._signature_extra(), base=FlowModel.__init__
        )

    @classmethod
    def _signature_extra(cls) -> tuple[str, ...]:
        """Constructor arguments beyond the design's and the base class's."""
        return ()

    _build_pymc_model = _by_design("_build_pymc_model")
    _compute_impacts_posterior = _by_design("_compute_impacts_posterior")
    _compute_jacobian_log_det = _by_design("_compute_jacobian_log_det")
    _simulate_y_rep = _by_design("_simulate_y_rep")
    fit = _by_design("fit")
    fit.__doc__ = FlowSharedMethods.fit.__doc__
    fit.__wrapped__ = FlowSharedMethods.fit


class FlowCountDesigns(FlowDesigns):
    """Design dispatch for the count flow families (see the module).

    The response is validated as counts.  A panel's pair effects are random
    and, with its period effects, are parameters of the count model rather than
    a within transform (``mundlak=True`` adds the pair means of the
    time-varying columns).  A two-part family (``_zero_part``) also takes the
    zero part's design ``X_zero`` and weights ``W_zero``.
    """

    _pair_effects = "random"
    #: Whether the family has a zero part (ZINB, hurdle).
    _zero_part: bool = False
    #: Whether the family needs a positive count (the hurdle's count part).
    _needs_positive: bool = False

    def _construct(self, y, X, W, *, effects=0, mundlak=False, **kwargs) -> None:
        from ._count_panel import require_counts

        zero = (
            {k: kwargs.pop(k, None) for k in ("X_zero", "W_zero")}
            if self._zero_part
            else None
        )
        name = type(self).__name__
        y_arr = require_counts(y, name)
        if self._needs_positive and not np.any(y_arr > 0):
            raise ValueError(f"{name} needs at least one positive count.")
        # The effects are parameters of the count model, not a within transform.
        super()._construct(y_arr.astype(np.float64), X, W, **kwargs)
        if zero is not None:
            # Before the count effects drop the columns they absorb from X.
            self._init_flow_zero_part(**zero)
        self._finish_count_init(y_arr, effects, mundlak)

    def _init_flow_zero_part(self, X_zero=None, W_zero=None) -> None:
        """The zero part's design ``_Z`` (default ``X``) and weights (default ``W``)."""
        import pandas as pd

        from ...graph import _weights_to_csr

        self._sel_is_X = X_zero is None
        if X_zero is None:
            self._Z = np.asarray(self._X, dtype=np.float64).copy()
            self._sel_feature_names = list(self._feature_names)
        else:
            Z_arr = np.asarray(X_zero, dtype=np.float64)
            Z_arr = Z_arr[:, None] if Z_arr.ndim == 1 else Z_arr
            if Z_arr.shape[0] != self._N * self._T:
                raise ValueError(
                    f"X_zero must have n²·T = {self._N * self._T} rows, "
                    f"got {Z_arr.shape[0]}."
                )
            self._sel_feature_names = (
                [str(c) for c in X_zero.columns]
                if isinstance(X_zero, pd.DataFrame)
                else [f"z{j}" for j in range(Z_arr.shape[1])]
            )
            self._Z = Z_arr
        if W_zero is None:
            self._W_zero_sparse, self._same_W = self._W_sparse, True
        else:
            self._W_zero_sparse, self._same_W = _weights_to_csr(W_zero), False
            if self._W_zero_sparse.shape != self._W_sparse.shape:
                raise ValueError("W_zero must be n × n, like W.")
        self._is_sel_row_std = True

    @classmethod
    def _signature_extra(cls) -> tuple[str, ...]:
        return (("mundlak",) if cls._has_panel else ()) + (
            ("X_zero", "W_zero") if cls._zero_part else ()
        )


def _constructor_signature(
    init, extra: tuple[str, ...] = (), base=None
) -> inspect.Signature:
    """The design arguments, ``extra``, then the base class's: what ``**kwargs`` takes.

    ``base`` is the base class's ``__init__`` (``SpatialModel``'s by default);
    a ``*args`` in ``init`` stands for its positional arguments.  Set on the
    class, so that :func:`inspect.signature` (and the API docs) show every
    argument the class takes rather than the design layer's ``**kwargs``.
    """
    own = list(inspect.signature(init).parameters.values())[1:]
    base = inspect.signature(base or SpatialModel.__init__).parameters
    params = []
    for p in own:
        if p.kind is p.VAR_POSITIONAL:
            params += [
                q for q in list(base.values())[1:] if q.kind is q.POSITIONAL_OR_KEYWORD
            ]
        elif p.kind is not p.VAR_KEYWORD:
            params.append(p)
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
