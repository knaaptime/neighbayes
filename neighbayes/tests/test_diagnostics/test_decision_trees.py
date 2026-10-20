"""Fast unit tests for neighbayes.diagnostics._decision_trees.

Tests TreeNode, evaluate, render_ascii, render, get_spec, and get_panel_spec
covering branches not exercised by the integration tests.
"""

from __future__ import annotations

import pytest

from neighbayes.diagnostics._decision_trees import (
    TreeNode,
    evaluate,
    get_panel_spec,
    get_spec,
    render,
    render_ascii,
)

# ---------------------------------------------------------------------------
# TreeNode basics
# ---------------------------------------------------------------------------


class TestTreeNode:
    def test_simple_tree(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        assert tree.kind == "test"
        assert tree.name == "LM-Lag"
        assert tree.if_true == "SAR"
        assert tree.if_false == "OLS"

    def test_predicate_node(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        assert tree.predicate_id == "lag_le_error"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_leaf_string(self):
        result, path = evaluate("OLS", sig_lookup=lambda _: False)
        assert result == "OLS"
        assert path == []

    def test_single_test_sig(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Lag")
        assert result == "SAR"
        assert len(path) == 1
        assert path[0][1] is True

    def test_single_test_not_sig(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "OLS"
        assert path[0][1] is False

    def test_nested_tree(self):
        inner = TreeNode(kind="test", name="LM-Error", if_true="SEM", if_false="OLS")
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false=inner)
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Error")
        assert result == "SEM"
        assert len(path) == 2

    def test_predicate_node(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: False,
            predicate_lookup={"lag_le_error": lambda: True},
        )
        assert result == "SAR"

    def test_predicate_false(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="lag_le_error",
        )
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: False,
            predicate_lookup={"lag_le_error": lambda: False},
        )
        assert result == "SEM"

    def test_predicate_missing_id_raises(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id=None,
        )
        with pytest.raises(ValueError, match="predicate node missing predicate_id"):
            evaluate(tree, sig_lookup=lambda _: False)

    def test_predicate_missing_lookup_raises(self):
        tree = TreeNode(
            kind="predicate",
            name="p comparison",
            if_true="SAR",
            if_false="SEM",
            predicate_id="missing_key",
        )
        with pytest.raises(KeyError):
            evaluate(tree, sig_lookup=lambda _: False, predicate_lookup={})

    def test_unknown_node_kind_raises(self):
        tree = TreeNode(kind="bad", name="x", if_true="A", if_false="B")
        with pytest.raises(ValueError, match="unknown node kind"):
            evaluate(tree, sig_lookup=lambda _: False)


# ---------------------------------------------------------------------------
# render_ascii
# ---------------------------------------------------------------------------


class TestRenderAscii:
    def test_leaf(self):
        result = render_ascii("OLS", [], "OLS")
        assert "OLS" in result
        assert "SELECTED" in result

    def test_simple_tree(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda n: n == "LM-Lag")
        text = render_ascii(tree, path, result)
        assert "SAR" in text
        assert "SELECTED" in text


# ---------------------------------------------------------------------------
# render dispatch
# ---------------------------------------------------------------------------


class TestRenderDispatch:
    def test_model_format(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        output = render(tree, path, result, fmt="model")
        assert output == "SAR"

    def test_ascii_format(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        output = render(tree, path, result, fmt="ascii")
        assert isinstance(output, str)
        assert "SAR" in output

    def test_unknown_format_raises(self):
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        with pytest.raises(ValueError, match="unknown format"):
            render(tree, path, result, fmt="bad_format")

    def test_graphviz_fallback_warns(self, monkeypatch):
        """When graphviz is not installed, should warn and fall back to ASCII."""
        from neighbayes.diagnostics import _decision_trees as _dt

        monkeypatch.setattr(_dt, "graphviz_available", lambda: False)
        tree = TreeNode(kind="test", name="LM-Lag", if_true="SAR", if_false="OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        with pytest.warns(UserWarning, match="graphviz package is not installed"):
            output = render(tree, path, result, fmt="graphviz")
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# get_spec / get_panel_spec
# ---------------------------------------------------------------------------


class TestGetSpec:
    def test_ols_spec(self):
        tree = get_spec("OLS")
        assert isinstance(tree, TreeNode)

    def test_sar_spec(self):
        tree = get_spec("SAR")
        assert isinstance(tree, TreeNode)

    def test_sem_spec(self):
        tree = get_spec("SEM")
        assert isinstance(tree, TreeNode)

    def test_slx_spec(self):
        tree = get_spec("SLX")
        assert isinstance(tree, TreeNode)

    def test_sdm_spec(self):
        tree = get_spec("SDM")
        assert isinstance(tree, TreeNode)

    def test_sdem_spec(self):
        tree = get_spec("SDEM")
        assert isinstance(tree, TreeNode)

    def test_unknown_returns_string(self):
        result = get_spec("UnknownModel")
        assert result == "UnknownModel"

    def test_ols_tree_evaluate_all_sig(self):
        """All naive + robust significant → SAR via robust p-value tie-break.

        ``SARAR`` is intentionally unreachable from the OLS tree because
        its proper null is a fitted SAR (or SEM) model.
        """
        tree = get_spec("OLS")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "robust_lag_pval_le_error_pval": lambda: True,
                "lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SAR"

    def test_ols_tree_evaluate_none_sig(self):
        """Walk the OLS tree with no tests significant → OLS."""
        tree = get_spec("OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "OLS"

    def test_ols_tree_only_lag_sig(self):
        """Only LM-Lag significant → SAR."""
        tree = get_spec("OLS")

        def lookup(name):
            return name == "LM-Lag"

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SAR"

    def test_ols_tree_only_error_sig(self):
        """Only LM-Error significant → SEM."""
        tree = get_spec("OLS")

        def lookup(name):
            return name == "LM-Error"

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SEM"

    def test_ols_tree_robust_lag_only(self):
        """Both naive sig, robust lag only → SAR."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error", "Robust-LM-Lag")

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SAR"

    def test_ols_tree_robust_error_only(self):
        """Both naive sig, robust error only → SEM."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error", "Robust-LM-Error")

        result, path = evaluate(tree, sig_lookup=lookup)
        assert result == "SEM"

    def test_ols_tree_robust_both(self):
        """Both naive and both robust sig → robust p-value tie-break (SAR/SEM).

        The OLS tree never reaches SARAR because SARAR's correct null is
        a fitted SAR (or SEM); the user must escalate by fitting that
        intermediate model and re-running diagnostics.
        """
        tree = get_spec("OLS")

        def lookup(name):
            return name in (
                "LM-Lag",
                "LM-Error",
                "Robust-LM-Lag",
                "Robust-LM-Error",
            )

        # Robust-LM-Lag wins the tie-break -> SAR.
        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"robust_lag_pval_le_error_pval": lambda: True},
        )
        assert result == "SAR"

        # Robust-LM-Error wins the tie-break -> SEM.
        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"robust_lag_pval_le_error_pval": lambda: False},
        )
        assert result == "SEM"

    def test_ols_tree_robust_neither_predicate(self):
        """Both naive sig, neither robust → predicate fallback."""
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error")

        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"lag_pval_le_error_pval": lambda: True},
        )
        assert result == "SAR"

    def test_ols_tree_robust_neither_predicate_false(self):
        tree = get_spec("OLS")

        def lookup(name):
            return name in ("LM-Lag", "LM-Error")

        result, path = evaluate(
            tree,
            sig_lookup=lookup,
            predicate_lookup={"lag_pval_le_error_pval": lambda: False},
        )
        assert result == "SEM"

    @pytest.mark.parametrize(
        "significant, leaf",
        [
            ((), "Logit"),
            (("Robust-LM-Lag",), "SARLogit"),
            (("Robust-LM-Lag", "LM-Error"), "SARLogit(latent=True)"),
            (("Robust-LM-Lag", "LM-Error", "Robust-LM-Error"), "SDEMLogit"),
            (("LM-Error", "Robust-LM-Error"), "SEMLogit"),
            (("LM-Error",), "SEMLogit or SARLogit(latent=True)"),
        ],
    )
    def test_logit_tree_reads_the_tests_as_a_pair(self, significant, leaf):
        """The spillover and error tests, read jointly, name the logit's form."""
        result, _ = evaluate(get_spec("Logit"), sig_lookup=lambda n: n in significant)
        assert result == leaf


class TestGetPanelSpec:
    def test_ols_fe_spec(self):
        tree = get_panel_spec("OLS")
        assert isinstance(tree, TreeNode)

    def test_sar_fe_spec(self):
        tree = get_panel_spec("SAR")
        assert isinstance(tree, TreeNode)

    def test_sem_fe_spec(self):
        tree = get_panel_spec("SEM")
        assert isinstance(tree, TreeNode)

    def test_slx_fe_spec(self):
        tree = get_panel_spec("SLX")
        assert isinstance(tree, TreeNode)

    def test_sdm_fe_spec(self):
        tree = get_panel_spec("SDM")
        assert isinstance(tree, TreeNode)

    def test_sdem_fe_spec(self):
        tree = get_panel_spec("SDEM")
        assert isinstance(tree, TreeNode)

    def test_ols_re_spec(self):
        tree = get_panel_spec("OLS")
        assert isinstance(tree, TreeNode)

    def test_unknown_returns_string(self):
        result = get_panel_spec("UnknownPanel")
        assert result == "UnknownPanel"

    def test_panel_ols_all_sig(self):
        # All naive + robust significant: route via the robust p-value
        # tie-break to the dominant single-channel panel model.  SARAR is
        # intentionally unreachable from a panel-OLS fit.
        tree = get_panel_spec("OLS")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "panel_robust_lag_pval_le_error_pval": lambda: True,
                "panel_lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SAR"

    def test_panel_ols_none_sig(self):
        tree = get_panel_spec("OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert "OLS" in result


class TestGetPanelDynamicSpec:
    """Dynamic panel decision trees reuse static panel structure with 'Dynamic' suffix."""

    def test_ols_dynamic_spec(self):
        tree = get_panel_spec("OLS")
        assert isinstance(tree, TreeNode)

    def test_sar_dynamic_spec(self):
        tree = get_panel_spec("SAR")
        assert isinstance(tree, TreeNode)

    def test_sem_dynamic_spec(self):
        tree = get_panel_spec("SEM")
        assert isinstance(tree, TreeNode)

    def test_slx_dynamic_spec(self):
        tree = get_panel_spec("SLX")
        assert isinstance(tree, TreeNode)

    def test_sdm_dynamic_spec(self):
        tree = get_panel_spec("SDM")
        assert isinstance(tree, TreeNode)

    def test_sdem_dynamic_spec(self):
        tree = get_panel_spec("SDEM")
        assert isinstance(tree, TreeNode)

    def test_dynamic_ols_all_sig(self):
        tree = get_panel_spec("OLS")
        result, path = evaluate(
            tree,
            sig_lookup=lambda _: True,
            predicate_lookup={
                "panel_robust_lag_pval_le_error_pval": lambda: True,
                "panel_lag_pval_le_error_pval": lambda: True,
            },
        )
        assert result == "SAR"

    def test_dynamic_ols_none_sig(self):
        tree = get_panel_spec("OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "OLS"


class TestGetPanelTobitSpec:
    """Panel Tobit decision trees mirror cross-sectional Tobit specs."""

    def test_sar_tobit_spec(self):
        tree = get_panel_spec("SARTobit")
        assert isinstance(tree, TreeNode)

    def test_sem_tobit_spec(self):
        tree = get_panel_spec("SEMTobit")
        assert isinstance(tree, TreeNode)

    def test_sar_tobit_all_sig(self):
        tree = get_panel_spec("SARTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        assert result == "SDMTobit"

    def test_sar_tobit_none_sig(self):
        tree = get_panel_spec("SARTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SARTobit"

    def test_sem_tobit_all_sig(self):
        tree = get_panel_spec("SEMTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: True)
        assert result == "SDEMTobit"

    def test_sem_tobit_none_sig(self):
        tree = get_panel_spec("SEMTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SEMTobit"
