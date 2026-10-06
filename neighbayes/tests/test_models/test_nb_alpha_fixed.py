"""``priors={"alpha_fixed": a}`` holds the NB2 dispersion on every sampler.

Each fit must return ``alpha`` equal to the fixed value in every draw, and the
PyMC models must record it as a constant, not a free variable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neighbayes import dgp
from neighbayes.dgp.flows import (
    generate_negbin_flow_data,
    generate_panel_negbin_flow_data,
)
from neighbayes.models import (
    NegBin,
    NegBinFlow,
    NegBinFlowPanel,
    NegBinPanel,
    SARNegBin,
    SARNegBinFlow,
    SARNegBinFlowPanel,
    SARNegBinFlowSeparable,
    SARNegBinFlowSeparablePanel,
    SARNegBinPanel,
    SARNegBinStructural,
)
from neighbayes.tests.helpers import W_to_graph, make_rook_W

A = 37.0
SHORT = dict(draws=15, tune=15, chains=1, progressbar=False, random_seed=0)


def _held(idata):
    alpha = idata.posterior["alpha"].values
    return np.allclose(alpha, A)


def _cross_section():
    d = dgp.simulate_sar_negbin(n_side=8, rho=0.3, seed=1)
    return d, W_to_graph(make_rook_W(8))


def _flow():
    return generate_negbin_flow_data(n=5, seed=2)


def _flow_panel():
    return generate_panel_negbin_flow_data(n=4, T=3, seed=3)


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_sar_negbin(backend):
    d, W = _cross_section()
    m = SARNegBin(y=d["y"], X=d["X"], W=W, priors={"alpha_fixed": A})
    assert _held(m.fit(gibbs_backend=backend, n_jobs=1, **SHORT))


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("jax", marks=pytest.mark.requires_jax)]
)
def test_sar_negbin_structural(backend):
    d, W = _cross_section()
    m = SARNegBinStructural(y=d["y"], X=d["X"], W=W, priors={"alpha_fixed": A})
    assert _held(m.fit(gibbs_backend=backend, n_jobs=1, **SHORT))


@pytest.mark.parametrize(
    "cls,backend",
    [
        (NegBinFlow, "numpy"),
        (SARNegBinFlow, "numpy"),
        pytest.param(SARNegBinFlow, "jax", marks=pytest.mark.requires_jax),
        (SARNegBinFlowSeparable, "numpy"),
        pytest.param(SARNegBinFlowSeparable, "jax", marks=pytest.mark.requires_jax),
    ],
)
def test_flow_cross_section(cls, backend):
    d = _flow()
    m = cls(
        d["y_vec"], d["X"], d["G"], col_names=d["col_names"], priors={"alpha_fixed": A}
    )
    kw = {} if cls is NegBinFlow else {"gibbs_backend": backend}
    assert _held(m.fit(sampler="gibbs", n_jobs=1, **kw, **SHORT))


@pytest.mark.parametrize(
    "cls,effects,backend",
    [
        (SARNegBinFlowPanel, 0, "numpy"),
        (SARNegBinFlowPanel, 3, "numpy"),
        (SARNegBinFlowSeparablePanel, 0, "numpy"),
        (SARNegBinFlowSeparablePanel, 3, "numpy"),
        pytest.param(
            SARNegBinFlowSeparablePanel, 3, "jax", marks=pytest.mark.requires_jax
        ),
        (NegBinFlowPanel, 0, "numpy"),
        (NegBinFlowPanel, 3, "numpy"),
    ],
)
def test_flow_panels(cls, effects, backend):
    d = _flow_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = cls(
            y=d["y"],
            X=d["X"],
            W=d["G"],
            T=3,
            col_names=d["col_names"],
            effects=effects,
            priors={"alpha_fixed": A},
        )
        kw = {} if cls is NegBinFlowPanel else {"gibbs_backend": backend}
        idata = (
            m.fit(n_jobs=1, attach_log_abs_det=False, **kw, **SHORT)
            if kw
            else m.fit(n_jobs=1, **SHORT)
        )
    assert _held(idata)


@pytest.mark.parametrize("cls", [SARNegBinPanel, NegBinPanel])
@pytest.mark.parametrize("effects", [0, 3])
def test_panels(cls, effects):
    d = dgp.simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=4)
    m = cls(
        y=d["y"], X=d["X"], W=d["W_sparse"], N=16, T=3, effects=effects,
        priors={"alpha_fixed": A},
    )  # fmt: skip
    assert _held(m.fit(n_jobs=1, **SHORT))


@pytest.mark.parametrize(
    "build",
    [
        lambda: NegBin(
            y=np.random.default_rng(0).poisson(2.0, 50).astype(float),
            X=np.ones((50, 1)),
            priors={"alpha_fixed": A},
        ),
        lambda: SARNegBin(
            y=_cross_section()[0]["y"],
            X=_cross_section()[0]["X"],
            W=_cross_section()[1],
            priors={"alpha_fixed": A},
        ),
        lambda: SARNegBinFlowSeparable(
            _flow()["y_vec"], _flow()["X"], _flow()["G"], priors={"alpha_fixed": A}
        ),
        lambda: SARNegBinPanel(
            y=dgp.simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=4)["y"],
            X=dgp.simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=4)["X"],
            W=dgp.simulate_panel_sar_negbin(N=16, T=3, n_side=4, seed=4)["W_sparse"],
            N=16,
            T=3,
            priors={"alpha_fixed": A},
        ),
    ],
)
def test_pymc_records_alpha_as_constant(build):
    model = build()._build_pymc_model()
    assert "alpha" in model.named_vars
    assert "alpha" not in {rv.name for rv in model.free_RVs}
    assert float(model.named_vars["alpha"].eval()) == A


def test_pymc_fit_holds_alpha():
    y = np.random.default_rng(1).poisson(2.0, 60).astype(float)
    m = NegBin(y=y, X=np.ones((60, 1)), priors={"alpha_fixed": A})
    assert _held(m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0))


@pytest.mark.parametrize("bad", [0.0, -2.0, np.inf])
def test_alpha_fixed_must_be_positive(bad):
    d = _flow()
    m = SARNegBinFlowSeparable(d["y_vec"], d["X"], d["G"], priors={"alpha_fixed": bad})
    with pytest.raises(ValueError, match="alpha_fixed"):
        m.fit(sampler="gibbs", **SHORT)
