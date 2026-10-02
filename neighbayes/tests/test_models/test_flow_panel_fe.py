"""Flow panels under fixed effects: absorbed columns, the NUTS Jacobian, effects.

Pair effects absorb every column fixed within a pair (intercept, intra
indicator, log distance).  Those are dropped before sampling, as the ordinary
panels drop theirs, and effects read beta back in the full design layout.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from neighbayes.dgp.flows import generate_panel_flow_data
from neighbayes.models.flow_panel import OLSFlowPanel, SARFlowPanel


def _data(effects_sigma=1.0):
    return generate_panel_flow_data(
        n=6,
        T=3,
        seed=0,
        distribution="normal",
        beta_d=[1.0, -0.5],
        beta_o=[0.5, 0.3],
        sigma_alpha=effects_sigma,
    )


@pytest.mark.parametrize(
    "effects, absorbed",
    [
        (0, []),
        (1, ["intercept", "intra_indicator", "log_distance"]),
        (2, ["intercept"]),
    ],
)
def test_absorbed_columns_are_dropped(effects, absorbed):
    d = _data()
    m = OLSFlowPanel(
        y=d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=effects
    )
    assert m._design_feature_names == list(d["col_names"])
    assert [c for c in d["col_names"] if c not in m._feature_names] == absorbed
    assert m._X.shape[1] == len(m._feature_names)
    assert np.all(np.abs(m._X).max(axis=0) > 0)


def test_time_invariant_attribute_warns():
    d = _data()
    X = d["X"].copy()
    N = 36
    j = d["col_names"].index("dest_x0")
    X[:, j] = np.tile(X[:N, j], 3)  # same in every period
    with pytest.warns(UserWarning, match="dest_x0"):
        OLSFlowPanel(y=d["y"], X=X, W=d["G"], T=3, col_names=d["col_names"], effects=1)


def test_beta_layout_scatters_with_nan():
    d = _data()
    m = OLSFlowPanel(
        y=d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=1
    )

    k = len(m._feature_names)
    draws = np.arange(2 * k, dtype=float).reshape(1, 2, k)

    class _V:
        values = draws

    full = m._beta_layout({"beta": _V()})
    assert full.shape == (2, len(m._design_feature_names))
    for j, name in enumerate(m._design_feature_names):
        if name in m._feature_names:
            np.testing.assert_array_equal(
                full[:, j], draws[0, :, m._feature_names.index(name)]
            )
        else:
            assert np.isnan(full[:, j]).all()


def test_flow_logdet_op_gradient():
    import pytensor
    import pytensor.tensor as pt

    from neighbayes._logdet._flow_kron_traces import FlowKronTraceLogdet
    from neighbayes._ops import FlowLogdetOp

    W = sp.csr_matrix(_data()["G"].sparse)
    r = pt.dvector("r")
    value, _ = FlowLogdetOp(FlowKronTraceLogdet(W))(r[0], r[1], r[2])
    f = pytensor.function([r], [value, pytensor.grad(value, r)])
    x0 = np.array([0.3, 0.2, 0.1])
    _, g = f(x0)
    h = 1e-6
    fd = [(f(x0 + h * e)[0] - f(x0 - h * e)[0]) / (2 * h) for e in np.eye(3)]
    np.testing.assert_allclose(g, fd, rtol=1e-6)
    # Exact against the dense determinant.
    n = W.shape[0]
    I = np.eye(n)
    Wd = W.toarray()
    A = (
        np.eye(n * n)
        - 0.3 * np.kron(I, Wd)
        - 0.2 * np.kron(Wd, I)
        - 0.1 * np.kron(Wd, Wd)
    )
    np.testing.assert_allclose(f(x0)[0], np.linalg.slogdet(A)[1], rtol=1e-10)


def test_sar_flow_panel_nuts_includes_jacobian():
    d = _data()
    m = SARFlowPanel(
        y=d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=1
    )
    model = m._build_pymc_model()
    assert "jacobian" in [p.name for p in model.potentials]


def test_separable_effects_beyond_general_radius():
    """ρ_w = −ρ_d ρ_o converges for |ρ_d|, |ρ_o| < 1, past |ρ_d|+|ρ_o|+|ρ_w| = 1."""
    from neighbayes.models.flow._flow import _flow_effect_sums, _FlowEffectMoments

    W = sp.csr_matrix(_data()["G"].sparse)
    rd, ro = 0.7, 0.6
    sums = _flow_effect_sums(_FlowEffectMoments(W), [rd], [ro], [-rd * ro])
    n = W.shape[0]
    I = np.eye(n)
    Wd = W.toarray()
    A = (
        np.eye(n * n)
        - rd * np.kron(I, Wd)
        - ro * np.kron(Wd, I)
        + rd * ro * np.kron(Wd, Wd)
    )
    Ainv = np.linalg.inv(A)
    dest_total = sum(Ainv @ np.kron(np.ones(n), I[j]) for j in range(n)).sum() / n**2
    np.testing.assert_allclose(sums[0, 0, 0], dest_total, rtol=1e-10)
    with pytest.raises(ValueError, match="draw reaches"):
        _flow_effect_sums(_FlowEffectMoments(W), [0.7], [0.6], [0.1])


def test_gaussian_flow_priors_are_shared_and_data_scaled():
    """NUTS and the resolvent sampler resolve the same data-scaled priors."""
    from neighbayes.samplers.gaussian._flow_resolvent import FlowResolventTarget

    d = _data()
    m = SARFlowPanel(
        y=d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=1
    )
    pv = m._flow_gaussian_priors()
    assert pv["sigma2_alpha"] == 2.0
    np.testing.assert_allclose(pv["sigma2_beta"], np.var(m._y))
    tgt = FlowResolventTarget(
        m._W_sparse,
        m._y,
        m._X,
        T=3,
        logdet_value_and_grad=lambda *r: (0.0, np.zeros(3)),
        **pv,
    )
    np.testing.assert_array_equal(tgt.beta_sigma, pv["beta_sigma"])
    np.testing.assert_array_equal(tgt.beta_mu, pv["beta_mu"])
    # The defaults scale with y: rescaling y rescales every prior scale.
    m2 = SARFlowPanel(
        y=1000 * d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=1
    )
    pv2 = m2._flow_gaussian_priors()
    np.testing.assert_allclose(pv2["beta_sigma"], 1000 * pv["beta_sigma"])
    np.testing.assert_allclose(pv2["sigma2_beta"], 1e6 * pv["sigma2_beta"])
    model = m._build_pymc_model()
    assert "sigma2" in [v.name for v in model.free_RVs]
    assert "sigma" in [v.name for v in model.deterministics]


def test_old_sigma_sigma_key_is_rejected():
    d = _data()
    m = OLSFlowPanel(
        y=d["y"],
        X=d["X"],
        W=d["G"],
        T=3,
        col_names=d["col_names"],
        priors={"sigma_sigma": 10.0},
    )
    with pytest.raises(ValueError, match="sigma_sigma"):
        m._build_pymc_model()


@pytest.mark.slow
def test_sar_flow_panel_nuts_and_gibbs_agree():
    """Same likelihood, same priors: the two samplers share a posterior."""
    d = generate_panel_flow_data(
        n=8,
        T=3,
        seed=1,
        distribution="normal",
        beta_d=[1.0, -0.5],
        beta_o=[0.5, 0.3],
        rho_d=0.3,
        rho_o=0.2,
        rho_w=0.1,
        sigma_alpha=1.0,
    )
    fits = {}
    for sampler in ("nuts", "gibbs"):
        m = SARFlowPanel(
            y=d["y"], X=d["X"], W=d["G"], T=3, col_names=d["col_names"], effects=1
        )
        fits[sampler] = m.fit(
            sampler=sampler,
            draws=1500,
            tune=1000,
            chains=2,
            random_seed=0,
            progressbar=False,
        ).posterior
    for name in ("rho_d", "rho_o", "rho_w", "sigma"):
        a, b = (float(fits[s][name].mean()) for s in ("nuts", "gibbs"))
        sd = float(fits["nuts"][name].std())
        assert abs(a - b) < 0.25 * sd, (name, a, b, sd)
