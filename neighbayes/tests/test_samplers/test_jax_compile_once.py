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
    from functools import partial

    from neighbayes import dgp
    from neighbayes import models as M
    from neighbayes.tests.helpers import (
        W_to_graph,
        _as_sparse,
        make_rook_W,
        make_sar_logit_data,
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
    dp = dgp.simulate_sar(N=36, T=3, n_side=6, seed=2, unit_effects="fixed")
    out["SARPanelFE"] = (
        dp["y"],
        lambda y: M.SAR(
            unit_effects="fixed", y=y, X=dp["X"], W=dp["W_sparse"], N=36, T=3
        ),
    )
    y, X = make_sar_logit_data(np.random.default_rng(3), Wd, rho=0.4, beta=beta)
    out["SARLogit"] = (y, lambda y, X=X: M.SARLogit(y=y, X=X, W=W))
    dn = dgp.simulate_sar_negbin(W=_as_sparse(W), rho=0.4, beta=np.array([0.5, 0.8]), alpha=2.0, rng=np.random.default_rng(4))  # fmt: skip
    out["SARNegBin"] = (dn["y"], lambda y: M.SARNegBin(y=y, X=dn["X"], W=W))
    dz = dgp.simulate_sar_zinb(n_side=6, seed=6)
    out["SARZINB"] = (dz["y"], lambda y: M.SARZINB(y=y, X=dz["X"], W=dz["W_sparse"]))
    dl = dgp.simulate_sar_logit(n_side=6, seed=7, latent=True)
    out["SARLogitLatent"] = (
        dl["y"],
        lambda y: M.SARLogit(y=y, X=dl["X"], W=dl["W_sparse"], latent=True),
    )
    de = dgp.simulate_sem_logit(n_side=6, seed=8)
    out["SEMLogit"] = (de["y"], lambda y: M.SEMLogit(y=y, X=de["X"], W=de["W_sparse"]))
    dnl = dgp.simulate_sar_negbin(n_side=6, seed=9, latent=True)
    out["SARNegBinLatent"] = (
        dnl["y"],
        lambda y: M.SARNegBin(y=y, X=dnl["X"], W=dnl["W_sparse"], latent=True),
    )
    dne = dgp.simulate_sem_negbin(n_side=6, seed=10)
    out["SEMNegBin"] = (
        dne["y"],
        lambda y: M.SEMNegBin(y=y, X=dne["X"], W=dne["W_sparse"]),
    )
    fd = dgp.simulate_sar_negbin_flow(
        n=5, k=2, rho_d=0.1, rho_o=0.1, rho_w=0.0, alpha=5.0, seed=42
    )
    out["SARNegBinFlow"] = (
        fd["y_vec"],
        lambda y: M.SARNegBinFlow(y, fd["X"], fd["G"]),
    )
    out["SARNegBinFlowSeparable"] = (
        fd["y_vec"],
        lambda y: M.SARNegBinFlow(y, fd["X"], fd["G"], separable=True),
    )
    pf = dgp.simulate_sar_negbin_flow(n=5, T=3, rho_d=0.3, rho_o=0.2, seed=21, alpha=3.0, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4, separable=True, pair_effects="random", pair_effect_sd=0.5, time_effects=True, time_effect_sd=0.3)  # fmt: skip
    out["SARNegBinFlowSeparablePanelRE"] = (
        pf["y"],
        lambda y: M.SARNegBinFlow(
            separable=True,
            pair_effects="random",
            time_effects=True,
            y=y,
            X=pf["X"],
            W=pf["G"],
            T=3,
            col_names=pf["col_names"],
        ),
    )
    hd = dgp.simulate_sar_hurdle_negbin_flow(n=6, T=1, seed=2, beta_d=[0.4, 0.4], beta_o=[0.4, 0.4], gamma_dist=-0.4, alpha=3.0, separable=True)  # fmt: skip
    out["SARHurdleNegBinFlowSeparable"] = (
        hd["y_vec"],
        lambda y: M.SARHurdleNegBinFlow(
            y, hd["X"], hd["G"], separable=True, col_names=hd["col_names"]
        ),
    )
    zd = dgp.simulate_sar_zinb_flow(
        n=4,
        T=3,
        seed=2,
        separable=True,
        pair_effects="random",
        pair_effect_sd=0.4,
        time_effects=True,
        time_effect_sd=0.3,
    )
    out["SARZINBFlowSeparablePanelRE"] = (
        zd["y"],
        lambda y: M.SARZINBFlow(
            y,
            zd["X"],
            zd["G"],
            separable=True,
            pair_effects="random",
            time_effects=True,
            T=3,
            col_names=zd["col_names"],
        ),
    )
    return out


_CASES = (
    "SAR",
    "SEM",
    "SARPanelFE",
    "SARLogit",
    "SARNegBin",
    "SARZINB",
    "SARLogitLatent",
    "SEMLogit",
    "SARNegBinLatent",
    "SEMNegBin",
    "SARNegBinFlow",
    "SARNegBinFlowSeparable",
    "SARNegBinFlowSeparablePanelRE",
    "SARHurdleNegBinFlowSeparable",
    "SARZINBFlowSeparablePanelRE",
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
    kw = dict(chains=2, random_seed=1, progressbar=False, backend="jax")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(y).fit(draws=10, tune=10, **kw)
        first = compile_count()
        assert first > 0, "the counter saw no compilation on the first fit"
        build(_other_data(y)).fit(draws=17, tune=13, **kw)
    assert compile_count() == first
