"""Fast unit tests for neighbayes.models.panel_base helpers and input validation.

These tests exercise _demean_panel, resolve_W, _as_dense_W,
SpatialPanelModel.__init__ (formula mode, error paths), and
spatial_diagnostics_decision branches without running MCMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes.models import OLS, SAR, SDEM, SDM, SEM, SLX
from neighbayes.models._base._shared import resolve_W
from neighbayes.models.panel_base import (
    _as_dense_W,
    _demean_panel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rook_W(n: int = 4) -> np.ndarray:
    """Row-standardized rook W on a line of n units."""
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1.0
        if i < n - 1:
            W[i, i + 1] = 1.0
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def _W_to_graph(W_dense: np.ndarray) -> Graph:
    """Convert dense W to a libpysal Graph."""
    n = W_dense.shape[0]
    focal, neighbor, weight = [], [], []
    for i in range(n):
        for j in range(n):
            if W_dense[i, j] != 0:
                focal.append(i)
                neighbor.append(j)
                weight.append(W_dense[i, j])
    return Graph.from_arrays(
        np.array(focal, int), np.array(neighbor, int), np.array(weight, float)
    ).transform("r")


# ---------------------------------------------------------------------------
# _demean_panel
# ---------------------------------------------------------------------------


class TestDemeanPanel:
    """Tests for the _demean_panel helper."""

    @pytest.fixture
    def panel_data(self):
        rng = np.random.default_rng(42)
        N, T, k = 5, 4, 2
        y = rng.standard_normal(N * T)
        X = rng.standard_normal((N * T, k))
        return y, X, N, T

    def test_model_0_pooled(self, panel_data):
        y, X, N, T = panel_data
        y_d, X_d = _demean_panel(y, X, N, T, effects=0)
        np.testing.assert_array_equal(y_d, y)
        np.testing.assert_array_equal(X_d, X)

    def test_model_1_unit_fe(self, panel_data):
        y, X, N, T = panel_data
        y_d, X_d = _demean_panel(y, X, N, T, effects=1)
        # Demeaned data should have zero within-unit mean
        y_d_2d = y_d.reshape(T, N)
        for i in range(N):
            np.testing.assert_allclose(y_d_2d[:, i].mean(), 0, atol=1e-12)

    def test_model_2_time_fe(self, panel_data):
        y, X, N, T = panel_data
        y_d, X_d = _demean_panel(y, X, N, T, effects=2)
        # Demeaned data should have zero within-period mean
        y_d_2d = y_d.reshape(T, N)
        for t in range(T):
            np.testing.assert_allclose(y_d_2d[t, :].mean(), 0, atol=1e-12)

    def test_model_3_two_way_fe(self, panel_data):
        y, X, N, T = panel_data
        y_d, X_d = _demean_panel(y, X, N, T, effects=3)
        # Two-way demeaned: subtract unit mean, time mean, add grand mean
        y2 = y.reshape(T, N)
        y_i = y2.mean(axis=0, keepdims=True)
        y_t = y2.mean(axis=1, keepdims=True)
        y_g = y2.mean()
        expected = y2 - y_i - y_t + y_g
        np.testing.assert_allclose(y_d, expected.ravel(), atol=1e-12)

    def test_invalid_model_raises(self, panel_data):
        y, X, N, T = panel_data
        with pytest.raises(ValueError, match="effects must be one of"):
            _demean_panel(y, X, N, T, effects=5)

    def test_unit_fe_with_T1_raises(self):
        y = np.ones(5)
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="Unit fixed effects"):
            _demean_panel(y, X, N=5, T=1, effects=1)

    def test_two_way_fe_with_T1_raises(self):
        y = np.ones(5)
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="Unit fixed effects"):
            _demean_panel(y, X, N=5, T=1, effects=3)

    def test_time_fe_with_T1_ok(self):
        """effects=2 (time FE) should work with T=1 (degenerate but valid)."""
        y = np.arange(5, dtype=float)
        X = np.arange(10, dtype=float).reshape(5, 2)
        y_d, X_d = _demean_panel(y, X, N=5, T=1, effects=2)
        # With T=1, time demeaning subtracts the single period mean
        np.testing.assert_allclose(y_d, y - y.mean(), atol=1e-12)


# ---------------------------------------------------------------------------
# resolve_W
# ---------------------------------------------------------------------------


class TestParsePanelW:
    """Tests for resolve_W validation."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_accepts_graph(self, W_graph):
        W_csr, row_std = resolve_W(W_graph, n=4, T=3)
        assert W_csr.shape == (4, 4)
        assert row_std is True

    def test_accepts_sparse_matrix(self):
        W_dense = _rook_W(4)
        W_sp = sp.csr_matrix(W_dense)
        W_csr, row_std = resolve_W(W_sp, n=4, T=3)
        assert W_csr.shape == (4, 4)

    def test_accepts_block_diagonal_sparse(self):
        W_dense = _rook_W(4)
        W_block = sp.kron(sp.eye(3, format="csr"), sp.csr_matrix(W_dense))
        W_csr, row_std = resolve_W(W_block, n=4, T=3)
        # Should accept N*T x N*T shape
        assert W_csr.shape[0] == 12

    def test_rejects_legacy_libpysal_W(self):
        """A mock object with .sparse and .transform should raise TypeError."""
        W_dense = _rook_W(4)

        class FakeLegacyW:
            sparse = sp.csr_matrix(W_dense)
            transform = "r"

        with pytest.raises(TypeError, match="legacy libpysal.weights.W"):
            resolve_W(FakeLegacyW(), n=4, T=3)

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="W must be a libpysal.graph.Graph"):
            resolve_W(np.ones((4, 4)), n=4, T=3)

    def test_rejects_non_square(self):
        W_rect = sp.csr_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError, match="W must be a square"):
            resolve_W(W_rect, n=4, T=3)

    def test_rejects_wrong_size(self):
        W_dense = _rook_W(4)
        W_sp = sp.csr_matrix(W_dense)
        with pytest.raises(ValueError, match="W has shape"):
            resolve_W(W_sp, n=5, T=3)

    def test_warns_non_row_standardized(self):
        W = sp.csr_matrix(np.ones((4, 4)))  # Not row-standardized
        with pytest.warns(UserWarning, match="row-standardized"):
            resolve_W(W, n=4, T=3)


# ---------------------------------------------------------------------------
# _as_dense_W
# ---------------------------------------------------------------------------


class TestAsDenseW:
    """Tests for _as_dense_W conversion."""

    def test_graph_to_dense(self):
        W_graph = _W_to_graph(_rook_W(3))
        result = _as_dense_W(W_graph, N=3, T=2)
        assert result.shape == (6, 6)

    def test_sparse_to_dense(self):
        W_sp = sp.csr_matrix(_rook_W(3))
        result = _as_dense_W(W_sp, N=3, T=2)
        assert result.shape == (6, 6)

    def test_ndarray_to_dense(self):
        W_arr = _rook_W(3)
        result = _as_dense_W(W_arr, N=3, T=2)
        assert result.shape == (6, 6)

    def test_block_diagonal_passthrough(self):
        W_dense = _rook_W(3)
        W_block = np.kron(np.eye(2), W_dense)
        result = _as_dense_W(W_block, N=3, T=2)
        np.testing.assert_array_equal(result, W_block)

    def test_wrong_shape_raises(self):
        W = np.ones((5, 5))
        with pytest.raises(ValueError, match="W has shape"):
            _as_dense_W(W, N=3, T=2)


# ---------------------------------------------------------------------------
# SpatialPanelModel.__init__ error paths
# ---------------------------------------------------------------------------


class TestSpatialPanelModelInit:
    """Test SpatialPanelModel constructor validation (no MCMC)."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_W_required(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        with pytest.raises(ValueError, match="W is required"):
            OLS(y=np.ones(12), X=np.ones((12, 2)), W=None, N=4, T=3)

    def test_formula_mode_requires_data(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        with pytest.raises(ValueError, match="data is required"):
            OLS(formula="y ~ x1", data=None, W=W_graph, unit_col="u", time_col="t")

    def test_chebyshev_init_does_not_eagerly_compute_eigs(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(0)

        def _boom(*args, **kwargs):
            raise RuntimeError("eigs should be lazy for panel chebyshev init")

        monkeypatch.setattr(np.linalg, "eigvals", _boom)

        model = OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
            logdet_method="chebyshev",
        )
        assert "_W_eigs" not in model.__dict__

    def test_sparse_grid_init_does_not_eagerly_compute_eigs(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(0)

        def _boom(*args, **kwargs):
            raise RuntimeError("eigs should be lazy for panel sparse_grid init")

        monkeypatch.setattr(np.linalg, "eigvals", _boom)

        model = OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
            logdet_method="eigenvalue",
        )
        assert "_W_eigs" not in model.__dict__

    def test_formula_mode_requires_unit_and_time_cols(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        df = pd.DataFrame(
            {
                "y": [1, 2, 3, 4],
                "x1": [5, 6, 7, 8],
                "u": [1, 1, 2, 2],
                "t": [1, 2, 1, 2],
            }
        )
        with pytest.raises(ValueError, match="unit_col and time_col"):
            OLS(formula="y ~ x1", data=df, W=W_graph, time_col="t")

    def test_matrix_mode_requires_N_and_T(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        with pytest.raises(ValueError, match="N and T are required"):
            OLS(y=np.ones(12), X=np.ones((12, 2)), W=W_graph, N=4)
        with pytest.raises(ValueError, match="N and T are required"):
            OLS(y=np.ones(12), X=np.ones((12, 2)), W=W_graph, T=3)

    def test_matrix_mode_NT_mismatch(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        with pytest.raises(ValueError, match="N\\*T must equal"):
            OLS(y=np.ones(12), X=np.ones((12, 2)), W=W_graph, N=5, T=3)

    def test_formula_mode_balanced_panel_check(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        # 3 rows but N=2, T=2 => 4 expected
        df = pd.DataFrame(
            {"y": [1, 2, 3], "x1": [4, 5, 6], "u": [1, 1, 2], "t": [1, 2, 1]}
        )
        with pytest.raises(ValueError, match="balanced panel"):
            OLS(formula="y ~ x1", data=df, W=W_graph, unit_col="u", time_col="t")

    def test_matrix_mode_creates_model(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        model = OLS(
            y=np.random.default_rng(0).standard_normal(12),
            X=np.random.default_rng(1).standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        assert model._N == 4
        assert model._T == 3
        assert model._y.shape == (12,)

    def test_formula_mode_creates_model(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(12),
                "x1": rng.standard_normal(12),
                "x2": rng.standard_normal(12),
                "unit": np.repeat(range(4), 3),
                "time": np.tile(range(3), 4),
            }
        )
        model = OLS(
            formula="y ~ x1 + x2", data=df, W=W_graph, unit_col="unit", time_col="time"
        )
        assert model._N == 4
        assert model._T == 3
        assert model._panel_index is not None

    def test_durbin_vars_subsets_lagged_columns(self, W_graph):
        from neighbayes.models.panel._fe import SDMPanelFE

        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "y": rng.standard_normal(12),
                "x1": rng.standard_normal(12),
                "x2": rng.standard_normal(12),
                "unit": np.repeat(range(4), 3),
                "time": np.tile(range(3), 4),
            }
        )
        full = SDM(
            formula="y ~ x1 + x2",
            data=df,
            W=W_graph,
            unit_col="unit",
            time_col="time",
        )
        # By default both x1 and x2 are lagged (intercept excluded).
        assert full._wx_feature_names == ["x1", "x2"]

        subset = SDM(
            formula="y ~ x1 + x2",
            data=df,
            W=W_graph,
            unit_col="unit",
            time_col="time",
            durbin_vars=["x1"],
        )
        assert subset._wx_feature_names == ["x1"]
        assert subset._WX.shape == (12, 1)

    def test_durbin_vars_unknown_raises(self, W_graph):
        from neighbayes.models.panel._fe import SDMPanelFE

        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="durbin_vars contains names not found"):
            SDM(
                y=rng.standard_normal(12),
                X=rng.standard_normal((12, 2)),
                W=W_graph,
                N=4,
                T=3,
                durbin_vars=["nonexistent"],
            )


class TestPanelDiagnosticsDecision:
    """Test spatial_diagnostics_decision branches for panel models."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    @pytest.fixture
    def panel_model(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(0)
        return OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )

    def test_ols_panel_fe_only_lag(self, panel_model, monkeypatch):
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.9]}, index=["Panel-LM-Lag", "Panel-LM-Error"]
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SAR"
        )

    def test_ols_panel_fe_only_error(self, panel_model, monkeypatch):
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.9, 0.001]}, index=["Panel-LM-Lag", "Panel-LM-Error"]
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SEM"
        )

    def test_ols_panel_fe_neither(self, panel_model, monkeypatch):
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.9, 0.9]}, index=["Panel-LM-Lag", "Panel-LM-Error"]
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "OLS"
        )

    def test_ols_panel_fe_both_with_robust_lag(self, panel_model, monkeypatch):
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001, 0.9]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag",
                "Panel-Robust-LM-Error",
            ],
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SAR"
        )

    def test_ols_panel_fe_both_with_robust_error(self, panel_model, monkeypatch):
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.9, 0.001]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag",
                "Panel-Robust-LM-Error",
            ],
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SEM"
        )

    def test_ols_panel_fe_both_robust_both(self, panel_model, monkeypatch):
        # Both naive AND both robust significant: route to the dominant
        # single-channel panel model via the robust p-value tie-break.
        # SARARPanel is intentionally unreachable from a panel-OLS fit;
        # the user must escalate by fitting SARPanelFE (or SEMPanelFE)
        # and re-running diagnostics from there.
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001, 0.001]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag",
                "Panel-Robust-LM-Error",
            ],
        )
        monkeypatch.setattr(panel_model, "spatial_diagnostics", lambda: df)
        assert (
            panel_model.spatial_diagnostics_decision(alpha=0.05, format="model")
            == "SAR"
        )

    def test_sdm_panel_fe_error_sdm(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import SDMPanelFE

        rng = np.random.default_rng(0)
        model = SDM(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame({"p_value": [0.001]}, index=["Panel-LM-Error-SDM"])
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "GNS"

    def test_sdem_panel_fe_lag_sdem(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import SDEMPanelFE

        rng = np.random.default_rng(0)
        model = SDEM(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame({"p_value": [0.001]}, index=["Panel-LM-Lag-SDEM"])
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "GNS"

    def test_slx_panel_fe_robust_lag_sdm(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import SLXPanelFE

        rng = np.random.default_rng(0)
        model = SLX(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001, 0.9]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag-SDM",
                "Panel-Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_slx_panel_fe_robust_error_sdem(self, W_graph, monkeypatch):
        from neighbayes.models.panel._fe import SLXPanelFE

        rng = np.random.default_rng(0)
        model = SLX(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.9, 0.001]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag-SDM",
                "Panel-Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_slx_panel_fe_robust_both_lag_wins(self, W_graph, monkeypatch):
        """When both robust tests fire, the smaller-p side wins (Lag-SDM here)."""
        from neighbayes.models.panel._fe import SLXPanelFE

        rng = np.random.default_rng(0)
        model = SLX(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.0001, 0.01]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag-SDM",
                "Panel-Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_slx_panel_fe_robust_both_error_wins(self, W_graph, monkeypatch):
        """When both robust tests fire, the smaller-p side wins (Error-SDEM here)."""
        from neighbayes.models.panel._fe import SLXPanelFE

        rng = np.random.default_rng(0)
        model = SLX(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.01, 0.0001]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-Error",
                "Panel-Robust-LM-Lag-SDM",
                "Panel-Robust-LM-Error-SDEM",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"

    def test_slx_panel_fe_neither_robust(self, W_graph, monkeypatch):
        """When neither robust test fires, SLX is the terminal model."""
        from neighbayes.models.panel._fe import SLXPanelFE

        rng = np.random.default_rng(0)
        model = SLX(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.9, 0.9]},
            index=["Panel-Robust-LM-Lag-SDM", "Panel-Robust-LM-Error-SDEM"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SLX"

    def test_sar_panel_fe_error_and_wx(self, W_graph, monkeypatch):
        """SAR panel: LM-Error significant, Robust-LM-WX significant → SDM."""
        from neighbayes.models.panel._fe import SARPanelFE

        rng = np.random.default_rng(0)
        model = SAR(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001, 0.001]},
            index=["Panel-LM-Error", "Panel-LM-WX", "Panel-Robust-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDM"

    def test_sem_panel_fe_lag_and_wx(self, W_graph, monkeypatch):
        """SEM panel: LM-Lag significant, LM-WX significant → SDEM."""
        from neighbayes.models.panel._fe import SEMPanelFE

        rng = np.random.default_rng(0)
        model = SEM(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        import pandas as pd

        df = pd.DataFrame(
            {"p_value": [0.001, 0.001]},
            index=["Panel-LM-Lag", "Panel-LM-WX"],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"


# ---------------------------------------------------------------------------
# Phase 3: _batch_sparse_lag parity and memory-warning tests
# ---------------------------------------------------------------------------


class TestBatchSparseLag:
    """Tests that _batch_sparse_lag produces the same result as the dense path.

    The expected value is computed via the full (N*T)×(N*T) Kronecker product
    so the test directly validates the Phase-3 sparse replacement.
    """

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    @pytest.fixture
    def model(self, W_graph):
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(7)
        return OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )

    def _dense_lag(self, model, resid):
        """Reference: full Kronecker dense W_NT @ resid.T then transpose."""
        from neighbayes.models.panel_base import _as_dense_W

        W_dense = _as_dense_W(model._W_sparse, model._N, model._T)
        return (W_dense @ resid.T).T

    def test_single_draw_parity(self, model):
        rng = np.random.default_rng(0)
        resid = rng.standard_normal((1, model._N * model._T))
        expected = self._dense_lag(model, resid)
        result = model._batch_sparse_lag(resid)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_draws_parity(self, model):
        rng = np.random.default_rng(1)
        s = 50
        resid = rng.standard_normal((s, model._N * model._T))
        expected = self._dense_lag(model, resid)
        result = model._batch_sparse_lag(resid)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_explicit_T_eff_matches_default(self, model):
        rng = np.random.default_rng(2)
        resid = rng.standard_normal((20, model._N * model._T))
        result_default = model._batch_sparse_lag(resid)
        result_explicit = model._batch_sparse_lag(resid, T_eff=model._T)
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_dynamic_T_eff_parity(self, model):
        """T_eff=T-1 path used in dynamic panel log-likelihoods."""
        rng = np.random.default_rng(3)
        T_eff = model._T - 1
        resid = rng.standard_normal((30, model._N * T_eff))
        W_dyn_dense = np.kron(np.eye(T_eff), model._W_sparse.toarray())
        expected = (W_dyn_dense @ resid.T).T
        result = model._batch_sparse_lag(resid, T_eff=T_eff)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_output_shape(self, model):
        rng = np.random.default_rng(4)
        s, n = 16, model._N * model._T
        resid = rng.standard_normal((s, n))
        assert model._batch_sparse_lag(resid).shape == (s, n)


class TestWDenseMemoryWarning:
    """_W_dense should emit a ResourceWarning for large panel matrices."""

    def test_small_panel_no_warning(self):
        """Small panels (well below 100 MB threshold) emit no warning."""
        import warnings

        W_graph = _W_to_graph(_rook_W(4))
        from neighbayes.models.panel._fe import OLSPanelFE

        rng = np.random.default_rng(0)
        model = OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        # Access _W_dense — no warning expected for tiny N, T
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            _ = model._W_dense  # should not raise

    def test_large_panel_triggers_warning(self):
        """Matrices exceeding the threshold trigger ResourceWarning."""
        import warnings

        W_graph = _W_to_graph(_rook_W(4))
        from neighbayes.models.panel._fe import OLSPanelFE
        from neighbayes.models.panel_base import SpatialPanelModel

        rng = np.random.default_rng(0)
        model = OLS(
            y=rng.standard_normal(12),
            X=rng.standard_normal((12, 2)),
            W=W_graph,
            N=4,
            T=3,
        )
        # Force the threshold to 0 so any materialization triggers the warning.
        original = SpatialPanelModel._DENSE_W_WARN_BYTES
        try:
            SpatialPanelModel._DENSE_W_WARN_BYTES = 0
            with pytest.warns(ResourceWarning, match="dense panel weight matrix"):
                _ = model._W_dense
        finally:
            SpatialPanelModel._DENSE_W_WARN_BYTES = original
