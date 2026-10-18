"""Design dispatch for the public simulators.

Each public simulator draws from the public model class it is named for and
takes the same design arguments (ADR 0002): ``T`` (with ``N``) makes a panel,
``unit_effects`` / ``pair_effects`` and ``time_effects`` add effects,
``dynamic`` adds the time lag, and ``separable`` restricts a flow.  The scale
of drawn effects is ``unit_effect_sd`` (``pair_effect_sd`` for flows) and
``time_effect_sd``, each defaulting to 0.5 when its effects are requested.

Fixed and random unit effects draw the same data, effects from
:math:`N(0, \\sigma^2)`: the distinction is the model's, not the data's.
Defaults match the class defaults, so ``simulate_sar_logit()`` draws the
reduced form that ``SARLogit()`` fits.
"""

from __future__ import annotations

import functools
import importlib
import inspect

_DESIGN = {
    "T": None,
    "N": None,
    "unit_effects": None,
    "pair_effects": None,
    "time_effects": False,
    "dynamic": False,
    "separable": False,
    "group_effects": None,
    "unit_effect_sd": None,
    "pair_effect_sd": None,
    "time_effect_sd": None,
}
# Switches the design arguments replace, and what replaces each.
_RETIRED = {
    "sigma_alpha": "unit_effects= (pair_effects= for flows) with unit_effect_sd=",
    "sigma2": "unit_effects='random' with unit_effect_sd=",
    "structural": "unit_effects='random'",
}
_DEFAULT_SD = 0.5
_NICKELL = (
    "Dynamic panels take no unit effects: within-demeaning correlates the "
    "demeaned time lag with the demeaned disturbance (Nickell 1981)."
)


def simulator(route):
    """Wrap a simulator so its design arguments dispatch through ``route``.

    ``route(fn, design, kwargs)`` returns ``(target, kwargs)``, the function to
    call and its keyword arguments.  Positional arguments bind to the wrapped
    function's parameter names first.
    """

    def wrap(fn):
        sig = inspect.signature(fn)
        positional = [
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        @functools.wraps(fn)
        def public(*args, **kwargs):
            if len(args) > len(positional):
                raise TypeError(
                    f"{fn.__name__}() takes at most {len(positional)} positional "
                    "arguments."
                )
            for key, instead in _RETIRED.items():
                if key in kwargs:
                    raise TypeError(
                        f"{fn.__name__}(): {key}= is replaced by {instead}."
                    )
            kwargs.update(zip(positional, args))
            design = {k: kwargs.pop(k) for k in list(kwargs) if k in route.accepts}
            extra = {k for k in kwargs if k in _DESIGN} - set(sig.parameters)
            if extra:
                raise TypeError(f"{fn.__name__}() does not take {sorted(extra)}.")
            target, kw = route(fn, design, kwargs)
            return target(**kw)

        params = [
            p
            for name, p in sig.parameters.items()
            if p.kind is not p.VAR_KEYWORD and name not in _RETIRED
        ]
        var_kw = [p for p in sig.parameters.values() if p.kind is p.VAR_KEYWORD]
        extra = [
            inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=_DESIGN[k])
            for k in _DESIGN
            if k in route.accepts and k not in sig.parameters
        ]
        public.__signature__ = sig.replace(parameters=params + extra + var_kw)
        return public

    return wrap


def accepts(*names):
    """Declare the design arguments a route takes."""

    def mark(route):
        route.accepts = frozenset(names)
        return route

    return mark


def _private(module, name):
    return getattr(importlib.import_module(f"neighbayes.dgp.{module}"), name)


def _validate(design, index="unit_effects"):
    unit = design.get(index)
    if unit not in (None, "fixed", "random"):
        raise ValueError(f"{index} must be None, 'fixed', or 'random'; got {unit!r}.")
    time = design.get("time_effects", False)
    if not isinstance(time, bool):
        raise ValueError(f"time_effects must be True or False; got {time!r}.")
    dynamic = design.get("dynamic", False)
    if dynamic not in (False, True, "restricted"):
        raise ValueError(
            f"dynamic must be False, True, or 'restricted'; got {dynamic!r}."
        )
    return unit, time, dynamic


def _panel(design, kw):
    """Move ``T``/``N`` into ``kw`` for a panel; ``T`` absent or 1 is a cross-section."""
    T, N = design.get("T"), design.get("N")
    if T is None and N is not None:
        raise ValueError("N and T are required for a panel; got N without T.")
    if T is None or int(T) == 1:
        return False
    kw["T"] = int(T)
    if N is not None:
        kw["N"] = int(N)
    return True


def _scale(design, key, on, what):
    """The effect scale: ``design[key]`` (default 0.5) when ``on``, else 0."""
    sd = design.get(key)
    if sd is not None and not on:
        raise ValueError(f"{key} needs {what}.")
    return (_DEFAULT_SD if sd is None else float(sd)) if on else 0.0


def _with_N(target, kw):
    """Size a panel from ``n_side`` when that is all the call gives.

    Panels take ``N``; ``n_side`` sets it to ``n_side ** 2`` and, for a target
    that does not take ``n_side`` itself, builds the grid weights.
    """
    params = inspect.signature(target).parameters
    n_side = kw.get("n_side")
    if n_side is not None and "N" not in kw:
        kw["N"] = int(n_side) ** 2
    if "n_side" in kw and "n_side" not in params:
        kw.pop("n_side")
        if n_side is not None and kw.get("W") is None and kw.get("gdf") is None:
            from .utils import resolve_weights

            kw["W"] = resolve_weights(
                n_side=n_side, contiguity=kw.get("contiguity", "queen")
            )[0]
    return kw


def _no_panel_only(name, time, dynamic):
    if time or dynamic:
        raise ValueError(f"{name}: time_effects and dynamic need a panel (pass T).")


# ---------------------------------------------------------------------------
# Gaussian processes
# ---------------------------------------------------------------------------

_GAUSSIAN = {
    "simulate_ols": "ols",
    "simulate_slx": "slx",
    "simulate_sar": "sar",
    "simulate_sem": "sem",
    "simulate_sdm": "sdm",
    "simulate_sdem": "sdem",
}


@accepts(
    "T",
    "N",
    "unit_effects",
    "time_effects",
    "dynamic",
    "unit_effect_sd",
    "time_effect_sd",
)
def gaussian(fn, design, kw):
    """OLS, SLX, SAR, SEM, SDM, SDEM: panels with effects or the time lag."""
    name = fn.__name__
    proc = _GAUSSIAN[name]
    unit, time, dynamic = _validate(design)
    if not _panel(design, kw):
        if unit is not None:
            raise NotImplementedError(
                f"{name} takes unit effects only in a panel (pass T)."
            )
        _no_panel_only(name, time, dynamic)
        _scale(design, "unit_effect_sd", False, "unit_effects")
        _scale(design, "time_effect_sd", False, "time_effects")
        return fn, kw
    if dynamic:
        if unit is not None:
            raise ValueError(_NICKELL)
        if dynamic == "restricted" and proc != "sdm":
            raise NotImplementedError(f"{name} has no restricted dynamic panel.")
        suffix = "_restricted" if dynamic == "restricted" else ""
        target = _private("panel_dynamic", f"_simulate_{proc}_panel_dynamic{suffix}")
    else:
        target = _private("panel_fe", f"_simulate_{proc}_panel_fe")
    kw["sigma_alpha"] = _scale(
        design, "unit_effect_sd", unit is not None, "unit_effects"
    )
    kw["time_effect_sd"] = _scale(design, "time_effect_sd", time, "time_effects")
    return target, _with_N(target, kw)


# ---------------------------------------------------------------------------
# Tobit, logit, probit, and counts
# ---------------------------------------------------------------------------

_TOBIT = {
    "simulate_sar_tobit": "_simulate_sar_tobit_panel",
    "simulate_sem_tobit": "_simulate_sem_tobit_panel",
}


@accepts("T", "N")
def tobit(fn, design, kw):
    """Tobit processes: a panel without effects."""
    name = fn.__name__
    if not _panel(design, kw):
        return fn, kw
    if name not in _TOBIT:
        raise NotImplementedError(f"{name} is not implemented as a panel.")
    target = _private("panel_tobit", _TOBIT[name])
    kw["sigma_alpha"] = 0.0
    return target, _with_N(target, kw)


@accepts("unit_effects")
def logit(fn, design, kw):
    """SAR and SEM logit: the reduced form, or random effects."""
    name = fn.__name__
    unit, _, _ = _validate(design)
    if unit == "fixed":
        raise NotImplementedError(f"{name} takes no fixed unit effects.")
    if name == "simulate_sem_logit":
        if unit != "random":
            raise NotImplementedError(
                "The latent SEM logit is not implemented; pass unit_effects='random' "
                "for the SEM logit with random effects."
            )
        return fn, kw
    kw["structural"] = unit == "random"
    return fn, kw


_COUNT = {
    "simulate_sar_negbin": ("panel_count", "_simulate_sar_negbin_panel_re"),
    "simulate_sar_zinb": ("panel_count", "_simulate_sar_zinb_panel_re"),
    "simulate_sar_hurdle_negbin": ("hurdle", "_simulate_sar_hurdle_negbin_panel_re"),
}


@accepts("T", "N", "unit_effects", "time_effects", "unit_effect_sd", "time_effect_sd")
def count(fn, design, kw):
    """Count families: random effects in a cross-section (NB only), count panels."""
    name = fn.__name__
    unit, time, _ = _validate(design)
    if unit == "fixed":
        raise NotImplementedError(f"{name} takes random unit effects, not fixed ones.")
    if not _panel(design, kw):
        _no_panel_only(name, time, False)
        _scale(design, "time_effect_sd", False, "time_effects")
        if unit == "random":
            if name != "simulate_sar_negbin":
                raise NotImplementedError(
                    f"{name} takes random unit effects only in a panel (pass T)."
                )
            kw["sigma2"] = _scale(design, "unit_effect_sd", True, "unit_effects") ** 2
        else:
            _scale(design, "unit_effect_sd", False, "unit_effects")
        return fn, kw
    target = _private(*_COUNT[name])
    kw["unit_effect_sd"] = _scale(
        design, "unit_effect_sd", unit is not None, "unit_effects"
    )
    kw["time_effect_sd"] = _scale(design, "time_effect_sd", time, "time_effects")
    return target, _with_N(target, kw)


@accepts("group_effects")
def sem_probit(fn, design, kw):
    """The SEM probit with random group effects."""
    if design.get("group_effects") != "random":
        raise NotImplementedError(
            "The latent SEM probit is not implemented; pass group_effects='random' "
            "for the SEM probit with random group effects."
        )
    return fn, kw


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

# Per simulator: the pair-effects treatment its panels take, and the targets by
# (separable, panel); None means the wrapped function itself.
_FLOW = {
    "simulate_sar_flow": (
        "fixed",
        {
            (False, False): None,
            (True, False): "_simulate_sar_flow_separable",
            (False, True): "_simulate_sar_flow_panel_fe",
            (True, True): "_simulate_sar_flow_separable_panel_fe",
        },
    ),
    "simulate_sem_flow": (
        "fixed",
        {
            (False, False): None,
            (True, False): "_simulate_sem_flow_separable",
            (False, True): "_simulate_sem_flow_panel_fe",
            (True, True): "_simulate_sem_flow_separable_panel_fe",
        },
    ),
    "simulate_sar_negbin_flow": (
        "random",
        {
            (False, False): None,
            (True, False): "_simulate_sar_negbin_flow_separable",
            (False, True): "_simulate_sar_negbin_flow_panel_re",
            (True, True): "_simulate_sar_negbin_flow_separable_panel_re",
        },
    ),
    "simulate_sar_poisson_flow": (
        None,
        {(False, False): None, (True, False): "_simulate_sar_poisson_flow_separable"},
    ),
    "simulate_sar_zinb_flow": ("random", {(True, False): None, (True, True): None}),
    "simulate_sar_hurdle_negbin_flow": (
        "random",
        {(True, False): None, (True, True): None},
    ),
}


@accepts(
    "T", "separable", "pair_effects", "time_effects", "pair_effect_sd", "time_effect_sd"
)
def flow(fn, design, kw):
    """Flow simulators: the separable restriction, flow panels, and pair effects."""
    name = fn.__name__
    pairs, targets = _FLOW[name]
    pair, time, _ = _validate(design, index="pair_effects")
    separable = design.get("separable", False)
    if not isinstance(separable, bool):
        raise ValueError(f"separable must be True or False; got {separable!r}.")
    panel = _panel(design, kw)
    if not panel and (pair is not None or time):
        raise NotImplementedError(
            f"{name} takes pair and time effects only in a panel (pass T)."
        )
    if not panel:
        _scale(design, "pair_effect_sd", False, "pair_effects")
        _scale(design, "time_effect_sd", False, "time_effects")
    if panel and pair is not None and pair != pairs:
        raise NotImplementedError(f"{name} panels take no {pair} pair effects.")
    if (separable, panel) not in targets:
        what = ("the separable" if separable else "the unrestricted") + (
            " panel" if panel else " cross-section"
        )
        hint = " (pass separable=True)" if (True, panel) in targets else ""
        raise NotImplementedError(f"{name} is not implemented as {what}{hint}.")
    private = targets[(separable, panel)]
    target = fn if private is None else _private("flows", private)
    if panel:
        sd = _scale(design, "pair_effect_sd", pair is not None, "pair_effects")
        kw["sigma_alpha" if pairs == "fixed" else "pair_effect_sd"] = sd
        kw["time_effect_sd"] = _scale(design, "time_effect_sd", time, "time_effects")
    return target, kw


def aspatial(spatial, zeros, *, panel_only=False, xs_random=True):
    """An aspatial simulator: ``spatial`` with its dependence parameters at zero.

    ``panel_only`` mirrors a class implemented for panels only, and
    ``xs_random=False`` one that takes random effects only in a panel.
    """

    def build(stub):
        @functools.wraps(stub)
        def public(*args, **kwargs):
            clash = set(zeros) & set(kwargs)
            if clash:
                raise TypeError(
                    f"{stub.__name__}() has no spatial dependence; got {sorted(clash)}."
                )
            if kwargs.get("separable"):
                raise ValueError(
                    f"{stub.__name__}() has no spatial dependence to restrict."
                )
            cross_section = kwargs.get("T") in (None, 1)
            if panel_only and cross_section:
                raise NotImplementedError(
                    f"{stub.__name__}() is implemented for panels only (pass T)."
                )
            if (
                not xs_random
                and cross_section
                and kwargs.get("unit_effects") is not None
            ):
                raise NotImplementedError(
                    f"{stub.__name__}() takes unit effects only in a panel (pass T)."
                )
            return spatial(*args, **zeros, **kwargs)

        sig = inspect.signature(spatial)
        public.__signature__ = sig.replace(
            parameters=[p for n, p in sig.parameters.items() if n not in zeros]
        )
        return public

    return build
