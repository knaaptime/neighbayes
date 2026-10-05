"""Every JAX Gibbs family compiles once per model structure.

A second model of the same structure, with different data and a different run
length, must reuse the compiled programs of the first: compilation costs
seconds, and a simulation study would otherwise pay it once per replicate.
XLA compilations are counted through JAX's public monitoring events.  The
multilevel model has its own check in ``test_multilevel.py``.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="JAX not installed"
)

_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


def _cases():
    from neighbayes import dgp
    from neighbayes import models as M
    from neighbayes.tests.helpers import (
        W_to_graph,
        _as_sparse,
        make_rook_W,
        make_sar_logit_data,
        make_sar_logit_structural_data,
        make_sem_logit_data,
    )

    Wd = make_rook_W(6)
    W = W_to_graph(Wd)
    beta = np.array([0.3, 1.0])
    out = {}
    for name, sim, cls in (
        ("SAR", dgp.simulate_sar, M.SAR),
        ("SEM", dgp.simulate_sem, M.SEM),
    ):
        d = sim(n_side=6, seed=1)
        out[name] = (
            d["y"],
            lambda y, d=d, cls=cls: cls(y=y, X=d["X"], W=d["W_sparse"]),
        )
    dp = dgp.simulate_panel_sar_fe(N=36, T=3, n_side=6, seed=2)
    out["SARPanelFE"] = (
        dp["y"],
        lambda y: M.SARPanelFE(y=y, X=dp["X"], W=dp["W_sparse"], N=36, T=3, effects=1),
    )
    for name, make, cls, kw in (
        ("SARLogit", make_sar_logit_data, M.SARLogit, {"rho": 0.4}),
        (
            "SARLogitStructural",
            make_sar_logit_structural_data,
            M.SARLogitStructural,
            {"rho": 0.4},
        ),
        ("SEMLogit", make_sem_logit_data, M.SEMLogit, {"lam": 0.4}),
    ):
        y, X = make(np.random.default_rng(3), Wd, beta=beta, **kw)
        out[name] = (y, lambda y, X=X, cls=cls: cls(y=y, X=X, W=W))
    for name, cls, s2 in (
        ("SARNegBin", M.SARNegBin, 0.0),
        ("SARNegBinStructural", M.SARNegBinStructural, 0.5),
    ):
        dn = dgp.simulate_sar_negbin(
            W=_as_sparse(W), rho=0.4, beta=np.array([0.5, 0.8]), alpha=2.0,
            sigma2=s2, rng=np.random.default_rng(4),
        )  # fmt: skip
        out[name] = (dn["y"], lambda y, dn=dn, cls=cls: cls(y=y, X=dn["X"], W=W))
    dz = dgp.simulate_sar_zinb(n_side=6, seed=6)
    out["SARZINB"] = (dz["y"], lambda y: M.SARZINB(y=y, X=dz["X"], W=dz["W_sparse"]))
    fd = dgp.generate_negbin_flow_data(
        n=5, k=2, rho_d=0.1, rho_o=0.1, rho_w=0.0, alpha=5.0, seed=42
    )
    out["SARNegBinFlow"] = (
        fd["y_vec"],
        lambda y: M.SARNegBinFlow(y, fd["X"], fd["G"]),
    )
    out["SARNegBinFlowSeparable"] = (
        fd["y_vec"],
        lambda y: M.SARNegBinFlowSeparable(y, fd["X"], fd["G"]),
    )
    pf = dgp.generate_panel_negbin_flow_data_separable(
        n=5, T=3, rho_d=0.3, rho_o=0.2, seed=21, alpha=3.0, beta_d=[0.4, 0.4],
        beta_o=[0.4, 0.4], gamma_dist=-0.4, pair_effect_sd=0.5, time_effect_sd=0.3,
    )  # fmt: skip
    out["SARNegBinFlowSeparablePanel"] = (
        pf["y"],
        lambda y: M.SARNegBinFlowSeparablePanel(
            y=y, X=pf["X"], W=pf["G"], T=3, col_names=pf["col_names"], effects=3
        ),
    )
    hd = dgp.generate_hurdle_flow_data_separable(
        n=6, T=1, seed=2, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4,
        alpha=3.0,
    )  # fmt: skip
    out["SARHurdleNBFlowSeparable"] = (
        hd["y_vec"],
        lambda y: M.SARHurdleNBFlowSeparable(
            y, hd["X"], hd["G"], col_names=hd["col_names"]
        ),
    )
    zd = dgp.generate_zinb_flow_data_separable(
        n=4, T=3, seed=2, pair_effect_sd=0.4, time_effect_sd=0.3
    )
    out["SARZINBFlowSeparablePanel"] = (
        zd["y"],
        lambda y: M.SARZINBFlowSeparablePanel(
            y, zd["X"], zd["G"], T=3, col_names=zd["col_names"], effects=3
        ),
    )
    return out


_CASES = (
    "SAR",
    "SEM",
    "SARPanelFE",
    "SARLogit",
    "SARLogitStructural",
    "SEMLogit",
    "SARNegBin",
    "SARNegBinStructural",
    "SARZINB",
    "SARNegBinFlow",
    "SARNegBinFlowSeparable",
    "SARNegBinFlowSeparablePanel",
    "SARHurdleNBFlowSeparable",
    "SARZINBFlowSeparablePanel",
)


def _other_data(y):
    """Data of the same shape and kind as ``y``, with different values."""
    rng = np.random.default_rng(99)
    y = np.asarray(y, dtype=np.float64)
    if np.isin(y, (0.0, 1.0)).all():
        return rng.permutation(y)
    if np.array_equal(y, np.round(y)):
        return y + rng.integers(0, 3, size=y.shape)
    return y + rng.normal(0.0, 0.1, size=y.shape)


@pytest.fixture
def compile_count():
    """A callable returning the number of XLA compilations so far."""
    import jax.monitoring

    events = []

    def listener(name, _secs, **_kw):
        if name == _COMPILE_EVENT:
            events.append(name)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        yield lambda: len(events)
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)


@pytest.mark.parametrize("name", _CASES)
def test_refit_does_not_recompile(name, compile_count):
    y, build = _cases()[name]
    kw = dict(chains=2, random_seed=1, progressbar=False, gibbs_backend="jax")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(y).fit(draws=10, tune=10, **kw)
        first = compile_count()
        assert first > 0, "the counter saw no compilation on the first fit"
        build(_other_data(y)).fit(draws=17, tune=13, **kw)
    assert compile_count() == first
