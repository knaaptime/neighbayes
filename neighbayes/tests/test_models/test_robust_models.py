"""Tests for robust (Student-t) error distribution across model classes.

Each test verifies that:
1. ``robust=True`` builds and samples without error
2. ``nu`` (degrees of freedom) is a fixed hyperparameter, not sampled
3. ``robust=False`` (default) still works (backward compatibility)
4. SARProbit raises ``NotImplementedError`` when ``robust=True``

Run with::

    pytest tests/test_robust_models.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neighbayes.dgp import simulate_sar, simulate_sem
from neighbayes.models import (
    OLS,
    SAR,
    SDEM,
    SDM,
    SEM,
    SLX,
    OLSPanelFE,
    OLSPanelRE,
    SARPanelFE,
    SARPanelRE,
    SARPanelTobit,
    SARProbit,
    SARTobit,
    SDEMPanelFE,
    SDMPanelFE,
    SDMTobit,
    SEMPanelFE,
    SEMPanelRE,
    SEMPanelTobit,
    SEMTobit,
    SLXPanelFE,
)
from neighbayes.tests.helpers import (
    PANEL_N,
    PANEL_T,
    SAMPLE_KWARGS,
    W_to_graph,
    _as_sparse,
    make_line_W,
    make_rook_W,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

# Minimal sampling for build/smoke tests — just need posterior to exist
QUICK_KWARGS = dict(
    tune=100, draws=100, chains=2, random_seed=42, progressbar=False, sampler="nuts"
)

SIDE = 6  # 36 cross-sectional units


# ---------------------------------------------------------------------------
# Cross-sectional model tests
# ---------------------------------------------------------------------------


class TestRobustCrossSectional:
    """Robust (Student-t) error distribution for cross-sectional models."""

    @pytest.fixture
    def sar_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(
            W=_as_sparse(W), rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        return out["y"], out["X"], W

    @pytest.fixture
    def sem_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sem(
            W=_as_sparse(W), lam=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        return out["y"], out["X"], W

    def test_ols_robust_builds_and_samples(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = OLS(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior, "nu is fixed, not sampled"
        assert "beta" in idata.posterior

    def test_ols_default_still_works(self, sar_data):
        y, X, W = sar_data
        W_graph = W_to_graph(W)
        model = OLS(y=y, X=X, W=W_graph)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior, (
            "nu should NOT be in posterior when robust=False"
        )
        assert "beta" in idata.posterior

    @pytest.mark.parametrize(
        "cls,data_fixture,extra_var",
        [
            (OLS, "sar_data", "beta"),
            (SAR, "sar_data", "rho"),
            (SLX, "sar_data", "beta"),
            (SEM, "sem_data", "lam"),
            (SDM, "sar_data", "rho"),
            (SDEM, "sem_data", "lam"),
        ],
        ids=lambda v: getattr(v, "__name__", str(v)),
    )
    def test_robust_builds_and_samples(self, request, cls, data_fixture, extra_var):
        y, X, W = request.getfixturevalue(data_fixture)
        W_graph = W_to_graph(W)
        model = cls(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled
        assert extra_var in idata.posterior


# ---------------------------------------------------------------------------
# Panel FE model tests
# ---------------------------------------------------------------------------


class TestRobustPanelFE:
    """Robust (Student-t) error distribution for panel FE models."""

    @pytest.fixture
    def panel_data(self, rng):
        from neighbayes.dgp import simulate_panel_sar_fe

        W = make_line_W(PANEL_N)
        W_graph = W_to_graph(W)
        out = simulate_panel_sar_fe(
            N=PANEL_N,
            T=PANEL_T,
            rho=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.5,
            rng=rng,
            W=_as_sparse(W),
        )
        return out["y"], out["X"], W_graph

    def test_ols_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = OLSPanelFE(
            y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1, robust=True
        )
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sar_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = SARPanelFE(
            y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1, robust=True
        )
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sem_panel_fe_robust(self, panel_data):
        y, X, W_graph = panel_data
        model = SEMPanelFE(
            y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, effects=1, robust=True
        )
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled


# ---------------------------------------------------------------------------
# Panel RE model tests
# ---------------------------------------------------------------------------


class TestRobustPanelRE:
    """Robust (Student-t) error distribution for panel RE models."""

    @pytest.fixture
    def panel_re_data(self, rng):
        from neighbayes.dgp import simulate_panel_sar_re

        W = make_line_W(PANEL_N)
        W_graph = W_to_graph(W)
        out = simulate_panel_sar_re(
            N=PANEL_N,
            T=PANEL_T,
            rho=0.3,
            beta=np.array([1.0, 2.0]),
            sigma=0.5,
            rng=rng,
            W=_as_sparse(W),
        )
        return out["y"], out["X"], W_graph

    def test_ols_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = OLSPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sar_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = SARPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sem_panel_re_robust(self, panel_re_data):
        y, X, W_graph = panel_re_data
        model = SEMPanelRE(y=y, X=X, W=W_graph, N=PANEL_N, T=PANEL_T, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled


# ---------------------------------------------------------------------------
# Tobit model tests
# ---------------------------------------------------------------------------


class TestRobustTobit:
    """Robust (Student-t) error distribution for Tobit models."""

    @pytest.fixture
    def sar_tobit_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(
            W=_as_sparse(W), rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        # Apply censoring at 0
        y = np.maximum(out["y"], 0.0)
        return y, out["X"], W

    @pytest.fixture
    def sem_tobit_data(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sem(
            W=_as_sparse(W), lam=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        y = np.maximum(out["y"], 0.0)
        return y, out["X"], W

    def test_sar_tobit_robust(self, sar_tobit_data):
        y, X, W = sar_tobit_data
        W_graph = W_to_graph(W)
        model = SARTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sem_tobit_robust(self, sem_tobit_data):
        y, X, W = sem_tobit_data
        W_graph = W_to_graph(W)
        model = SEMTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled

    def test_sdm_tobit_robust(self, sar_tobit_data):
        y, X, W = sar_tobit_data
        W_graph = W_to_graph(W)
        model = SDMTobit(y=y, X=X, W=W_graph, robust=True)
        idata = model.fit(**QUICK_KWARGS)
        assert "nu" not in idata.posterior  # fixed, not sampled


# ---------------------------------------------------------------------------
# SARProbit — should raise NotImplementedError
# ---------------------------------------------------------------------------


class TestRobustSpatialProbit:
    """SARProbit should raise NotImplementedError when robust=True."""

    def test_spatial_probit_raises(self, rng):
        W = make_line_W(6)
        W_graph = W_to_graph(W)
        m = 6  # number of regions
        region_ids = np.repeat(np.arange(m), 3)  # 3 obs per region
        n = m * 3
        X = rng.standard_normal((n, 2))
        y = rng.binomial(1, 0.5, size=n).astype(float)
        model = SARProbit(y=y, X=X, W=W_graph, region_ids=region_ids, robust=True)
        with pytest.raises(
            NotImplementedError, match="Robust.*not supported.*SARProbit"
        ):
            model._build_pymc_model()


# ---------------------------------------------------------------------------
# Nu prior parameter test
# ---------------------------------------------------------------------------


class TestNuPriorParameters:
    """``nu`` is a fixed hyperparameter (LeSage), not a sampled quantity."""

    def test_default_nu_is_lesage_rval(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(
            W=_as_sparse(W), rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        model = OLS(y=out["y"], X=out["X"], W=W_to_graph(W), robust=True)
        assert model._nu == 4.0

    def test_custom_nu(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(
            W=_as_sparse(W), rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        model = OLS(
            y=out["y"], X=out["X"], W=W_to_graph(W), robust=True, priors={"nu": 10.0}
        )
        assert model._nu == 10.0
        idata = model.fit(**QUICK_KWARGS)
        # nu is fixed, so it must NOT appear as a sampled variable
        assert "nu" not in idata.posterior

    def test_nu_must_exceed_two(self, rng):
        W = make_rook_W(SIDE)
        out = simulate_sar(
            W=_as_sparse(W), rho=0.5, beta=np.array([1.0, 2.0]), sigma=0.8, rng=rng
        )
        model = OLS(
            y=out["y"], X=out["X"], W=W_to_graph(W), robust=True, priors={"nu": 1.5}
        )
        with pytest.raises(ValueError, match="must be > 2"):
            _ = model._nu
