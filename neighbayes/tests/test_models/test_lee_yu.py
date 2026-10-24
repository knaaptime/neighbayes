"""Lee & Yu (2010) fixed-effects correction for Gaussian panels.

Demeaning a panel spends degrees of freedom.  The likelihood of the demeaned
data has ``N(T-1)`` effective observations under unit effects, a Jacobian of
``(T-1)·log|I - ρW|``, and, under time effects with a row-standardized ``W``,
a ``-m·log(1-ρ)`` term.  Without the correction σ² is shrunk by ``(T-1)/T``,
which at ``T = 3`` is a third.  These tests pin the bookkeeping and the
recovery on both backends, and on the flow panels.
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.models import SAR, SEM
from neighbayes.tests.helpers import W_to_graph, make_line_W, make_panel_sar_data


def _fit_sigma2(model, sampler):
    idata = model.fit(
        sampler=sampler,
        draws=800,
        tune=500,
        chains=1,
        random_seed=0,
        progressbar=False,
    )
    return float((idata.posterior["sigma"] ** 2).mean())


@pytest.mark.parametrize(
    "unit, time, jac_T, shift, n_eff",
    [
        # (units, T) = (20, 4); n = 80.
        (None, False, 4, 0.0, 80),
        ("fixed", False, 3, 0.0, 60),
        (None, True, 4, 4.0, 76),
        ("fixed", True, 3, 3.0, 57),
    ],
)
@pytest.mark.parametrize("cls", [SAR, SEM])
def test_fe_dimensions(cls, unit, time, jac_T, shift, n_eff):
    N, T = 20, 4
    W = make_line_W(N)
    y, X, _ = make_panel_sar_data(np.random.default_rng(0), W, N, T, rho=0.3)
    m = cls(y=y, X=X, W=W_to_graph(W), N=N, T=T, unit_effects=unit, time_effects=time)
    assert m._jacobian_T == jac_T
    assert m._jacobian_shift == shift
    assert m._n_effective == n_eff


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["nuts", "gibbs"])
def test_unit_fe_recovers_sigma2_at_short_T(sampler):
    """At ``T = 3`` the uncorrected posterior centres σ² near 2/3."""
    N, T = 300, 3
    W = make_line_W(N)
    y, X, _ = make_panel_sar_data(
        np.random.default_rng(1), W, N, T, rho=0.4, sigma=1.0, sigma_alpha=1.0
    )
    m = SAR(unit_effects="fixed", y=y, X=X, W=W_to_graph(W), N=N, T=T)
    assert abs(_fit_sigma2(m, sampler) - 1.0) < 0.17


def _flow_panel(n, T, rho_d, rho_o, seed):
    """Gaussian flow panel with pair effects, from the package DGP."""
    from neighbayes.dgp import simulate_sar_flow

    d = simulate_sar_flow(
        n=n,
        T=T,
        seed=seed,
        distribution="normal",
        rho_d=rho_d,
        rho_o=rho_o,
        rho_w=-rho_d * rho_o,
        beta_d=[1.0, -0.5],
        beta_o=[0.5, 0.3],
        pair_effects="fixed",
        pair_effect_sd=1.0,
    )
    return d["y"], d["X"], d["G"], d["col_names"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, separable",
    [("OLSFlow", False), ("SARFlow", True), ("SARFlow", False)],
    ids=["OLSFlow", "SARFlow separable", "SARFlow"],
)
def test_flow_pair_fe_recovers_sigma2_at_short_T(name, separable):
    import neighbayes.models as M

    cls = getattr(M, name)
    kw = {"separable": True} if separable else {}
    rho = (0.0, 0.0) if name == "OLSFlow" else (0.3, 0.2)
    y, X, G, names = _flow_panel(12, 3, *rho, seed=3)
    m = cls(y=y, X=X, W=G, T=3, col_names=names, pair_effects="fixed", **kw)
    assert m._jacobian_T == 2 and m._n_effective == 2 * 144
    idata = m.fit(draws=800, tune=500, chains=1, random_seed=0, progressbar=False)
    sigma2 = float((idata.posterior["sigma"] ** 2).mean())
    assert abs(sigma2 - 1.0) < 0.22
