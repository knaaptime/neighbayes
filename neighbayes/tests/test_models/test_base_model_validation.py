"""Fast unit tests for neighbayes.models.base input validation and decision tree.

Tests _parse_W error paths, SpatialModel.__init__ formula mode,
durbin_vars validation, and spatial_diagnostics_decision branches
without running MCMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from libpysal.graph import Graph

from neighbayes.models._base._shared import _parse_W

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
# _parse_W
# ---------------------------------------------------------------------------


class TestParseW:
    """Tests for _parse_W validation."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_accepts_graph(self, W_graph):
        W_csr, is_row_std = _parse_W(W_graph, n=4)
        assert sp.issparse(W_csr)
        assert W_csr.shape == (4, 4)

    def test_accepts_sparse_matrix(self):
        W_sp = sp.csr_matrix(_rook_W(4))
        W_csr, is_row_std = _parse_W(W_sp, n=4)
        assert sp.issparse(W_csr)
        assert W_csr.shape == (4, 4)

    def test_rejects_legacy_libpysal_W(self):
        """A mock object with .sparse and .transform should raise TypeError."""

        class FakeLegacyW:
            sparse = sp.csr_matrix(_rook_W(4))
            transform = "r"

        with pytest.raises(TypeError, match="legacy libpysal.weights.W"):
            _parse_W(FakeLegacyW(), n=4)

    def test_rejects_wrong_type(self):
        with pytest.raises(TypeError, match="W must be a libpysal.graph.Graph"):
            _parse_W(np.ones((4, 4)), n=4)

    def test_rejects_non_square(self):
        W_rect = sp.csr_matrix(np.ones((3, 4)))
        with pytest.raises(ValueError, match="W must be a square matrix"):
            _parse_W(W_rect, n=4)

    def test_rejects_wrong_size(self):
        W_sp = sp.csr_matrix(_rook_W(4))
        with pytest.raises(ValueError, match="W has shape"):
            _parse_W(W_sp, n=5)

    def test_warns_non_row_standardized(self):
        W = sp.csr_matrix(np.ones((4, 4)))  # Not row-standardized
        with pytest.warns(UserWarning, match="row-standardized"):
            _parse_W(W, n=4)


class TestIsolates:
    """Isolates — units with no neighbor inside the bandwidth — are valid.

    Row-standardizing leaves an isolate's row summing to 0 rather than 1
    because there is nothing to divide by.  That is correct output, not a
    standardization failure, and must not be warned about or routed onto the
    O(n³) eigenvalue path.
    """

    @staticmethod
    def _W_with_isolates(side=4, isolates=(3,)):
        W = _rook_W(side).astype(float)
        for i in isolates:
            W[i, :] = 0.0
            W[:, i] = 0.0
        rs = W.sum(1)
        conn = rs != 0
        W[conn] = W[conn] / rs[conn][:, None]
        return sp.csr_matrix(W)

    def test_isolates_do_not_warn(self, recwarn):
        W = self._W_with_isolates()
        _parse_W(W, n=W.shape[0])
        assert not [w for w in recwarn if "row-standardized" in str(w.message)]

    def test_isolates_keep_the_row_standardized_flag(self):
        """False here would divert a valid, common W onto the eigenvalue path."""
        W = self._W_with_isolates()
        _, row_std = _parse_W(W, n=W.shape[0])
        assert row_std is True

    def test_all_zero_W_still_warns(self):
        W = sp.csr_matrix((4, 4))
        with pytest.warns(UserWarning, match="row-standardized"):
            _parse_W(W, n=4)

    def test_partially_standardized_W_still_warns(self):
        """A row summing to neither 0 nor 1 is a real failure."""
        W = _rook_W(4).astype(float)
        rs = W.sum(1)
        W = W / rs[:, None]
        W[2, :] *= 3.0  # row 2 now sums to 3
        with pytest.warns(UserWarning, match="row-standardized"):
            _parse_W(sp.csr_matrix(W), n=W.shape[0])

    @pytest.mark.parametrize("rho", [0.2, 0.5, 0.8, 0.95])
    def test_multiplier_closed_forms_match_brute_force(self, rho):
        """The isolate correction must be exact, not approximate.

        An isolate's row of ``(I - ρW)⁻¹`` is ``e_i`` (row sum 1, not
        ``1/(1-ρ)``); for ``(I - ρW)⁻¹W`` it is zero.  Without the correction
        the shortcut overstates total impacts, increasingly so as ρ → 1.
        """
        from neighbayes.models.cross_section.sar import SAR

        W = self._W_with_isolates()
        n = W.shape[0]
        rng = np.random.default_rng(0)
        model = SAR(y=rng.normal(size=n), X=rng.normal(size=(n, 1)), W=W)
        assert model._n_isolates == 1

        Wd = W.toarray()
        M = np.linalg.inv(np.eye(n) - rho * Wd)
        draws = np.array([rho])
        assert np.allclose(model._batch_mean_row_sum(draws), M.sum(1).mean())
        assert np.allclose(model._batch_mean_row_sum_MW(draws), (M @ Wd).sum(1).mean())
        assert np.allclose(model._multiplier_row_sums(rho), M.sum(1))

    def test_no_isolates_path_is_unchanged(self):
        from neighbayes.models.cross_section.sar import SAR

        W = _rook_W(4).astype(float)
        W = sp.csr_matrix(W / W.sum(1)[:, None])
        n = W.shape[0]
        rng = np.random.default_rng(0)
        model = SAR(y=rng.normal(size=n), X=rng.normal(size=(n, 1)), W=W)
        assert model._n_isolates == 0
        rho = np.array([0.2, 0.6, 0.9])
        assert np.allclose(model._batch_mean_row_sum(rho), 1.0 / (1.0 - rho))
        assert np.allclose(model._batch_mean_row_sum_MW(rho), 1.0 / (1.0 - rho))


# ---------------------------------------------------------------------------
# SpatialModel.__init__ error paths
# ---------------------------------------------------------------------------


class TestSpatialModelInit:
    """Test SpatialModel constructor validation (no MCMC)."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    def test_formula_mode_requires_data(self, W_graph):
        from neighbayes.models.cross_section.ols import OLS

        with pytest.raises(ValueError, match="data must be provided"):
            OLS(formula="y ~ x1", data=None, W=W_graph)

    def test_matrix_mode_creates_model(self, W_graph):
        from neighbayes.models.cross_section.ols import OLS

        rng = np.random.default_rng(0)
        model = OLS(y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph)
        assert model._y.shape == (4,)

    def test_no_formula_no_matrices_raises(self, W_graph):
        from neighbayes.models.cross_section.ols import OLS

        with pytest.raises(ValueError, match="Provide either"):
            OLS(W=W_graph)

    def test_durbin_vars_unknown_raises(self, W_graph):
        from neighbayes.models.cross_section.slx import SLX

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="durbin_vars contains names not found"):
            SLX(
                y=rng.standard_normal(4),
                X=rng.standard_normal((4, 2)),
                W=W_graph,
                durbin_vars=["nonexistent"],
            )


# ---------------------------------------------------------------------------
# spatial_diagnostics_decision for cross-sectional models
# ---------------------------------------------------------------------------


class TestCrossSectionalDiagnosticsDecision:
    """Test spatial_diagnostics_decision branches for cross-sectional models."""

    @pytest.fixture
    def W_graph(self):
        return _W_to_graph(_rook_W(4))

    @pytest.mark.parametrize(
        "cls_name, pvalues, expected",
        [
            # WX alone at a SAR fit adds WX though its robust test is silent:
            # the robust test guards against an error term LM-Error rules out.
            (
                "sar.SAR",
                {"LM-WX": 0.031, "Robust-LM-WX": 0.082, "LM-Error": 0.12},
                "SDM",
            ),
            ("sar.SAR", {"LM-WX": 0.5, "LM-Error": 0.01}, "SARAR"),
            ("sar.SAR", {"LM-WX": 0.5, "LM-Error": 0.5}, "SAR"),
            # Both plain and both robust tests reject: the stronger robust test.
            (
                "sar.SAR",
                {
                    "LM-WX": 0.01,
                    "LM-Error": 0.01,
                    "Robust-LM-WX": 0.02,
                    "Robust-LM-Error": 0.001,
                },
                "SARAR",
            ),
            # Both plain tests reject and neither robust one does: the stronger.
            (
                "ols.OLS",
                {
                    "LM-Lag": 0.01,
                    "LM-Error": 0.01,
                    "Robust-LM-Lag": 0.3,
                    "Robust-LM-Error": 0.2,
                },
                "SEM",
            ),
            ("sem.SEM", {"LM-Lag": 0.5, "LM-WX": 0.01}, "SDEM"),
            ("ols.OLS", {"LM-Lag": 0.5, "LM-Error": 0.5, "LM-WX": 0.01}, "SLX"),
            (
                "slx.SLX",
                {
                    "LM-Lag": 0.01,
                    "LM-Error": 0.01,
                    "Robust-LM-Lag-SDM": 0.01,
                    "Robust-LM-Error-SDEM": 0.5,
                },
                "SDM",
            ),
            ("sdm.SDM", {"LM-Error-SDM": 0.01}, "GNS"),
            ("sdem.SDEM", {"LM-Lag-SDEM": 0.5}, "SDEM"),
            ("tobit.SARTobit", {"LM-WX": 0.01, "LM-Error": 0.5}, "SDMTobit"),
            ("tobit.SEMTobit", {"LM-Lag": 0.01, "LM-WX": 0.5}, "SARARTobit"),
            ("tobit.SDMTobit", {"LM-Error-SDM": 0.01}, "GNSTobit"),
        ],
    )
    def test_decision(self, W_graph, monkeypatch, cls_name, pvalues, expected):
        import importlib

        module, name = cls_name.rsplit(".", 1)
        cls = getattr(
            importlib.import_module(f"neighbayes.models.cross_section.{module}"), name
        )
        rng = np.random.default_rng(0)
        extra = {"censoring": 0.0} if name.endswith("Tobit") else {}
        model = cls(
            y=rng.standard_normal(4), X=rng.standard_normal((4, 2)), W=W_graph, **extra
        )
        df = pd.DataFrame({"p_value": list(pvalues.values())}, index=list(pvalues))
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert (
            model.spatial_diagnostics_decision(alpha=0.05, format="model") == expected
        )

    def test_panel_decision_breaks_ties_on_the_panel_tests(self, W_graph, monkeypatch):
        from neighbayes.models import SEM

        rng = np.random.default_rng(0)
        model = SEM(
            y=rng.standard_normal(8), X=rng.standard_normal((8, 2)), W=W_graph, N=4, T=2
        )
        df = pd.DataFrame(
            {"p_value": [0.01, 0.01, 0.4, 0.2]},
            index=[
                "Panel-LM-Lag",
                "Panel-LM-WX",
                "Panel-Robust-LM-Lag",
                "Panel-Robust-LM-WX",
            ],
        )
        monkeypatch.setattr(model, "spatial_diagnostics", lambda: df)
        assert model.spatial_diagnostics_decision(alpha=0.05, format="model") == "SDEM"


# ---------------------------------------------------------------------------
# Output format tests for spatial_diagnostics_decision
# ---------------------------------------------------------------------------


class TestDecisionOutputFormats:
    """format= kwarg for spatial_diagnostics_decision (ascii, graphviz, model)."""

    @pytest.fixture
    def fitted_ols(self):
        from neighbayes.models.cross_section.ols import OLS

        rng = np.random.default_rng(0)
        return OLS(
            y=rng.standard_normal(4),
            X=rng.standard_normal((4, 2)),
            W=_W_to_graph(_rook_W(4)),
        )

    @pytest.fixture
    def diag_df(self):
        return pd.DataFrame(
            {"p_value": [0.001, 0.9, 0.001, 0.9]},
            index=["LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error"],
        )

    def test_format_model_returns_string(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="model")
        assert isinstance(result, str)
        assert result == "SAR"

    def test_format_ascii_returns_tree_string(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="ascii")
        assert isinstance(result, str)
        assert "← SELECTED" in result
        assert "SAR" in result
        # The chosen leaf line must contain the marker
        chosen_lines = [ln for ln in result.splitlines() if "SELECTED" in ln]
        assert len(chosen_lines) == 1
        assert "[SAR]" in chosen_lines[0]

    def test_format_graphviz_returns_digraph(self, fitted_ols, diag_df, monkeypatch):
        graphviz = pytest.importorskip("graphviz")
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision(format="graphviz")
        assert isinstance(result, graphviz.Digraph)
        src = result.source
        assert "SAR" in src

    def test_format_graphviz_default(self, fitted_ols, diag_df, monkeypatch):
        """graphviz is the default when the package is available."""
        pytest.importorskip("graphviz")
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        result = fitted_ols.spatial_diagnostics_decision()
        # graphviz.Digraph has a .source attribute
        assert hasattr(result, "source")

    def test_graphviz_fallback_when_unavailable(self, fitted_ols, diag_df, monkeypatch):
        """If graphviz is not installed, warn and fall back to ASCII."""
        from neighbayes.diagnostics import _decision_trees as _dt

        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        # Force `find_spec("graphviz")` to return None
        monkeypatch.setattr(
            _dt.importlib.util,
            "find_spec",
            lambda name: (
                None
                if name == "graphviz"
                else __import__("importlib").util.find_spec(name)
            ),
        )
        with pytest.warns(UserWarning, match="graphviz package is not installed"):
            result = fitted_ols.spatial_diagnostics_decision(format="graphviz")
        assert isinstance(result, str)
        assert "← SELECTED" in result

    def test_invalid_format_raises(self, fitted_ols, diag_df, monkeypatch):
        monkeypatch.setattr(fitted_ols, "spatial_diagnostics", lambda: diag_df)
        with pytest.raises(ValueError, match="unknown format"):
            fitted_ols.spatial_diagnostics_decision(format="bogus")

    def test_sar_ascii_error_branch_drawn_per_channel(self):
        """The SAR tree tests LM-Error under both LM-WX edges, and the
        renderer draws each subtree in full rather than as a back-reference."""
        from neighbayes.diagnostics import _decision_trees as _dt

        spec = _dt.get_spec("SAR")

        def sig(name):
            return {
                "LM-WX": False,
                "LM-Error": True,
                "Robust-LM-WX": False,
            }.get(name, False)

        d, p = _dt.evaluate(spec, sig)
        result = _dt.render_ascii(
            spec,
            p,
            d,
            p_values={"LM-WX": 0.5, "LM-Error": 0.01, "Robust-LM-WX": 0.5},
            alpha=0.05,
        )
        lines = result.splitlines()
        # Full node expansions of LM-Error: lines that contain " LM-Error"
        # (preceded by space — to exclude the substring match in
        # "Robust-LM-Error", which is itself an inner gate inside each
        # error_branch subtree).
        full_expansions = [
            ln
            for ln in lines
            if " LM-Error" in ln and "see above" not in ln and "→" not in ln
        ]
        assert len(full_expansions) == 2, (
            f"Expected two full LM-Error expansions (one per channel), "
            f"got {len(full_expansions)}:\n{result}"
        )
        # No back-references should remain — the subtrees are independent.
        back_refs = [ln for ln in lines if "→" in ln and "LM-Error" in ln]
        assert len(back_refs) == 0, (
            f"Expected no back-references to LM-Error, got {len(back_refs)}:\n{result}"
        )
