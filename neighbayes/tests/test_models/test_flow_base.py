"""The flow base class: one ``FlowModel``, a cross-section being one period."""

from __future__ import annotations

import numpy as np
import pytest

import neighbayes.models as M
from neighbayes.dgp import simulate_sar_flow

T = 3


@pytest.fixture(scope="module")
def flows():
    d = simulate_sar_flow(
        n=4, rho_d=0.2, rho_o=0.1, rho_w=0.0, distribution="normal", seed=0
    )
    rng = np.random.default_rng(1)
    y = np.concatenate(
        [d["y_vec"] + rng.normal(0, 0.5, d["y_vec"].size) for _ in range(T)]
    )
    X = np.vstack([d["X"]] * T)
    X[:, 2:] += rng.normal(0, 0.5, X[:, 2:].shape)  # attributes vary by period
    return d, y, X


def test_a_cross_section_is_one_period(flows):
    d, _, _ = flows
    m = M.SARFlow(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
    assert (m._T, m._is_panel, m._N, m.effects) == (1, False, 16, 0)
    np.testing.assert_array_equal(m._y, d["y_vec"])
    np.testing.assert_array_equal(m._X, d["X"])


def test_a_panel_stacks_periods(flows):
    d, y, X = flows
    m = M.SARFlow(y, X, d["G"], T=T, pair_effects="fixed", col_names=d["col_names"])
    assert (m._T, m._is_panel, m._N, m.effects) == (T, True, 16, 1)
    assert m._y.shape == (16 * T,)


def test_fixed_effects_need_a_panel(flows):
    d, _, _ = flows
    with pytest.raises(NotImplementedError, match="only in a panel"):
        M.SARFlow(d["y_vec"], d["X"], d["G"], pair_effects="fixed")


def test_y_is_checked_against_n_and_T(flows):
    d, y, X = flows
    with pytest.raises(ValueError, match="y must have shape"):
        M.SARFlow(y[:-1], X, d["G"], T=T)


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("cls", [M.OLSFlow, M.SARFlow, M.SEMFlow])
def test_the_gaussian_nuts_models_take_robust(flows, cls, panel):
    d, y, X = flows
    kw = dict(T=T) if panel else {}
    yy, XX = (y, X) if panel else (d["y_vec"], d["X"])
    m = cls(
        yy,
        XX,
        d["G"],
        robust=True,
        separable=cls is M.SEMFlow,
        col_names=d["col_names"],
        **kw,
    )
    assert m.robust


@pytest.mark.parametrize(
    "cls, panel", [(M.SEMFlow, False), (M.SEMFlow, True), (M.NegBinFlow, True)]
)
def test_robust_needs_a_student_t_model(flows, cls, panel):
    """Only the Gaussian NUTS models with Student-t errors take ``robust=True``."""
    d, y, X = flows
    if cls is M.NegBinFlow:
        y = np.round(np.exp(y - y.max())).astype(int)
    kw = dict(T=T) if panel else {}
    yy, XX = (y, X) if panel else (d["y_vec"], d["X"])
    with pytest.raises(NotImplementedError, match="robust=True"):
        cls(yy, XX, d["G"], robust=True, col_names=d["col_names"], **kw)


def test_a_robust_resolvent_flow_samples_by_nuts(flows):
    d, y, X = flows
    m = M.SARFlow(y, X, d["G"], T=T, robust=True, col_names=d["col_names"])
    with pytest.raises(NotImplementedError, match="no Gibbs sampler"):
        m.fit(sampler="gibbs", draws=5, tune=5, chains=1, progressbar=False)
    m.fit(draws=20, tune=20, chains=1, progressbar=False, random_seed=0, cores=1)
    assert m._pymc_model is not None  # the NUTS path builds the PyMC model
    assert "rho_d" in m.inference_data.posterior


def test_a_panel_reads_the_panel_tree(flows):
    from neighbayes.diagnostics import _decision_trees as dt

    d, y, X = flows
    xs = M.OLSFlow(d["y_vec"], d["X"], d["G"], col_names=d["col_names"])
    panel = M.OLSFlow(y, X, d["G"], T=T, col_names=d["col_names"])
    assert xs._flow_spec_fn is dt.get_flow_spec
    assert panel._flow_spec_fn is dt.get_panel_flow_spec


@pytest.mark.parametrize("cls", [M.SARFlow, M.SEMFlow])
def test_one_class_per_family(flows, cls):
    """Every design of a Gaussian flow family is the family's own class."""
    d, y, X = flows
    names = d["col_names"]
    models = [
        cls(d["y_vec"], d["X"], d["G"], col_names=names),
        cls(d["y_vec"], d["X"], d["G"], separable=True, col_names=names),
        cls(y, X, d["G"], T=T, pair_effects="fixed", col_names=names),
        cls(y, X, d["G"], T=T, separable=True, time_effects=True, col_names=names),
    ]
    assert {type(m) for m in models} == {cls}
    assert [m._design for m in models] == ["static", "separable"] * 2
    assert "separable=True" in repr(models[1])
    assert "T=3, effects=2" in repr(models[3])


@pytest.mark.parametrize("separable", [False, True])
def test_sem_flow_posterior_predictive(flows, separable):
    """SEMFlow predicts from its ``lam_*`` draws (it read ``rho_*`` before)."""
    d, y, X = flows
    m = M.SEMFlow(y, X, d["G"], T=T, separable=separable, col_names=d["col_names"])
    m.fit(draws=10, tune=10, chains=1, progressbar=False, random_seed=0, cores=1)
    assert m.posterior_predictive(n_draws=3, random_seed=0).shape == (3, 16 * T)
