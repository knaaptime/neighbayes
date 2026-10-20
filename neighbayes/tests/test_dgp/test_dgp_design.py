"""The public simulators take the same design arguments as the model classes.

Each simulator draws from the class it is named for (ADR 0002): ``T`` makes a
panel, ``unit_effects`` / ``pair_effects`` and ``time_effects`` add effects,
``dynamic`` adds the time lag, and ``separable`` restricts a flow.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import neighbayes.dgp as d


def test_one_simulator_per_public_class():
    import neighbayes.models as m

    classes = {n for n in m.__all__ if n != "Level"}
    assert len(d.__all__) == len(classes) == 29


def test_signature_shows_design_arguments():
    params = inspect.signature(d.simulate_sar).parameters
    for name in ("T", "N", "unit_effects", "time_effects", "dynamic", "unit_effect_sd"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    flow = inspect.signature(d.simulate_sar_flow).parameters
    assert {"separable", "pair_effects", "T"} <= set(flow)


def test_panel_unit_effects_scale():
    none = d.simulate_sar(n_side=4, T=3, seed=1)
    fixed = d.simulate_sar(n_side=4, T=3, seed=1, unit_effects="fixed")
    random = d.simulate_sar(
        n_side=4, T=3, seed=1, unit_effects="random", unit_effect_sd=0.2
    )
    assert none["params_true"]["unit_effect_sd"] == 0.0
    assert fixed["params_true"]["unit_effect_sd"] == 0.5
    assert random["params_true"]["unit_effect_sd"] == 0.2
    assert none["y"].shape == (48,)


def test_time_effects_change_the_draw_only_when_requested():
    base = d.simulate_sem(n_side=4, T=3, seed=1)
    again = d.simulate_sem(n_side=4, T=3, seed=1, time_effects=False)
    timed = d.simulate_sem(n_side=4, T=3, seed=1, time_effects=True)
    np.testing.assert_array_equal(base["y"], again["y"])
    assert not np.allclose(base["y"], timed["y"])


def test_dynamic_and_restricted_dynamic():
    assert d.simulate_sdm(n_side=4, T=3, seed=1, dynamic="restricted")["y"].shape == (
        48,
    )
    assert d.simulate_ols(n_side=4, T=3, seed=1, dynamic=True)["y"].shape == (48,)


def test_count_unit_effects_need_a_panel():
    p = d.simulate_sar_negbin(
        n_side=4, T=3, seed=1, unit_effects="random", time_effects=True
    )
    assert p["y"].shape == (48,)
    with pytest.raises(NotImplementedError, match="only in a panel"):
        d.simulate_sar_negbin(n_side=5, seed=1, unit_effects="random")


def test_logit_takes_no_unit_effects():
    for fn in (d.simulate_sar_logit, d.simulate_logit):
        with pytest.raises(NotImplementedError, match="no unit effects"):
            fn(n_side=4, seed=3, unit_effects="random")


def test_flow_routes():
    assert d.simulate_sar_flow(n=5, seed=1, separable=True)["y_vec"].shape == (25,)
    panel = d.simulate_sar_flow(
        n=5, seed=1, T=2, pair_effects="fixed", time_effects=True
    )
    assert panel["y"].shape == (50,)
    nb = d.simulate_sar_negbin_flow(
        n=5, seed=1, T=2, separable=True, pair_effects="random"
    )
    assert nb["y"].shape == (50,)
    assert d.simulate_sar_zinb_flow(n=5, seed=1, separable=True)["y"].shape == (25,)
    assert d.simulate_ols_flow(n=5, seed=1)["y_vec"].shape == (25,)


def test_aspatial_simulators():
    assert d.simulate_negbin(n_side=4, seed=1)["y"].shape == (16,)
    assert d.simulate_zinb(n_side=4, T=2, seed=1)["y"].shape == (32,)


REFUSALS = [
    (
        d.simulate_sar,
        {"N": 16, "T": 3, "sigma_alpha": 0.5},
        TypeError,
        "unit_effect_sd",
    ),
    (d.simulate_sar_logit, {"structural": True}, TypeError, "latent=True"),
    (d.simulate_sar_negbin, {"sigma2": 0.5}, TypeError, "unit_effect_sd"),
    (d.simulate_sar, {"N": 16}, ValueError, "N and T are required"),
    (d.simulate_sar, {"unit_effects": "fixed"}, NotImplementedError, "only in a panel"),
    (
        d.simulate_sar,
        {"n_side": 4, "unit_effect_sd": 0.3},
        ValueError,
        "needs unit_effects",
    ),
    (
        d.simulate_sar,
        {"n_side": 4, "T": 3, "dynamic": True, "unit_effects": "fixed"},
        ValueError,
        "Nickell",
    ),
    (
        d.simulate_sar,
        {"n_side": 4, "T": 3, "dynamic": "restricted"},
        NotImplementedError,
        "restricted",
    ),
    (d.simulate_sem_probit, {"n_side": 3}, NotImplementedError, "latent SEM probit"),
    (d.simulate_zinb, {"n_side": 4}, NotImplementedError, "panels only"),
    (d.simulate_negbin, {"rho": 0.3}, TypeError, "no spatial dependence"),
    (d.simulate_ols_flow, {"separable": True}, ValueError, "no spatial dependence"),
    (
        d.simulate_sar_flow,
        {"n": 5, "pair_effects": "fixed"},
        NotImplementedError,
        "only in a panel",
    ),
    (
        d.simulate_sar_negbin_flow,
        {"n": 5, "T": 2, "pair_effects": "fixed"},
        NotImplementedError,
        "fixed pair effects",
    ),
    (d.simulate_sar_zinb_flow, {"n": 5}, NotImplementedError, "separable=True"),
    (d.simulate_sar_poisson_flow, {"n": 5, "T": 2}, NotImplementedError, "panel"),
    (d.simulate_sdm_tobit, {"n_side": 4, "T": 3}, NotImplementedError, "panel"),
]


@pytest.mark.parametrize(
    ("fn", "kwargs", "error", "match"),
    REFUSALS,
    ids=lambda v: getattr(v, "__name__", None),
)
def test_refusals(fn, kwargs, error, match):
    with pytest.raises(error, match=match):
        fn(**kwargs)
