"""Fast unit tests for neighbayes.diagnostics._decision_trees.

Tests TreeNode, evaluate, render_ascii, render, get_spec, and get_panel_spec
covering branches not exercised by the integration tests.
"""

from __future__ import annotations

import pytest

from neighbayes.diagnostics._decision_trees import (
    PREDICATE_PAIRS,
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

    @pytest.mark.parametrize("family", ["Logit", "Probit", "NegBin"])
    @pytest.mark.parametrize(
        "significant, leaf",
        [
            ((), "{f}"),
            (("Robust-LM-Lag",), "SAR{f}(latent=False)"),
            (("Robust-LM-Lag", "LM-Error"), "SAR{f}"),
            (("Robust-LM-Lag", "LM-Error", "Robust-LM-Error"), "SDEM{f}"),
            (("LM-Error", "Robust-LM-Error"), "SEM{f}"),
            (("LM-Error",), "SEM{f} or SAR{f}"),
        ],
    )
    def test_glm_tree_reads_the_tests_as_a_pair(self, family, significant, leaf):
        """The spillover and error tests, read jointly, name the GLM's form."""
        result, _ = evaluate(get_spec(family), sig_lookup=lambda n: n in significant)
        assert result == leaf.format(f=family)


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

    def test_panel_ols_none_sig(self):
        tree = get_panel_spec("OLS")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert "OLS" in result


class TestGetPanelTobitSpec:
    """Panel Tobit decision trees mirror cross-sectional Tobit specs."""

    def test_sar_tobit_spec(self):
        tree = get_panel_spec("SARTobit")
        assert isinstance(tree, TreeNode)

    def test_sem_tobit_spec(self):
        tree = get_panel_spec("SEMTobit")
        assert isinstance(tree, TreeNode)

    def test_sar_tobit_none_sig(self):
        tree = get_panel_spec("SARTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SARTobit"

    def test_sem_tobit_none_sig(self):
        tree = get_panel_spec("SEMTobit")
        result, path = evaluate(tree, sig_lookup=lambda _: False)
        assert result == "SEMTobit"


# ---------------------------------------------------------------------------
# The classical rule, at every tree that tests two directions
# ---------------------------------------------------------------------------

# (spec getter, model type, plain a, plain b, robust a, robust b, leaf a, leaf b)
_PAIR_TREES = [
    ("OLS", "LM-Lag", "LM-Error", "Robust-LM-Lag", "Robust-LM-Error", "SAR", "SEM"),
    ("SAR", "LM-WX", "LM-Error", "Robust-LM-WX", "Robust-LM-Error", "SDM", "SARAR"),
    ("SEM", "LM-Lag", "LM-WX", "Robust-LM-Lag", "Robust-LM-WX", "SARAR", "SDEM"),
    (
        "SLX",
        "LM-Lag",
        "LM-Error",
        "Robust-LM-Lag-SDM",
        "Robust-LM-Error-SDEM",
        "SDM",
        "SDEM",
    ),
]
_TREES = [
    pytest.param(getter, null + family, labels, (a + family, b + family), id=label)
    for null, *names, a, b in _PAIR_TREES
    for getter, prefix, family, label in (
        (get_spec, "", "", null),
        (get_panel_spec, "Panel-", "", f"Panel-{null}"),
        (get_spec, "", "Tobit", f"{null}Tobit"),
        (get_panel_spec, "Panel-", "Tobit", f"Panel-{null}Tobit"),
    )
    if not (family and null in ("OLS", "SLX"))
    for labels in [[prefix + name for name in names]]
]

# p-values of (plain a, plain b, robust a, robust b), and the leaf they reach.
_CASES = [
    pytest.param((0.5, 0.5, 0.5, 0.5), "stay", id="neither"),
    pytest.param((0.01, 0.5, 0.5, 0.5), "a", id="a-alone-robust-silent"),
    pytest.param((0.01, 0.5, 0.01, 0.5), "a", id="a-alone"),
    pytest.param((0.5, 0.01, 0.5, 0.5), "b", id="b-alone-robust-silent"),
    pytest.param((0.5, 0.5, 0.01, 0.01), "stay", id="robust-only"),
    pytest.param((0.01, 0.01, 0.01, 0.5), "a", id="both-robust-a"),
    pytest.param((0.01, 0.01, 0.5, 0.01), "b", id="both-robust-b"),
    pytest.param((0.01, 0.01, 0.001, 0.01), "a", id="both-both-a-stronger"),
    pytest.param((0.01, 0.01, 0.01, 0.001), "b", id="both-both-b-stronger"),
    pytest.param((0.01, 0.01, 0.2, 0.4), "a", id="both-neither-a-stronger"),
    pytest.param((0.01, 0.01, 0.4, 0.2), "b", id="both-neither-b-stronger"),
]


def _walk(spec, p):
    predicates = {
        key: (lambda a=a, b=b: p[a] <= p[b]) for key, (a, b) in PREDICATE_PAIRS.items()
    }
    return evaluate(spec, lambda name: p.get(name, 1.0) < 0.05, predicates)[0]


@pytest.mark.parametrize("getter, null, tests, leaves", _TREES)
@pytest.mark.parametrize("pvalues, expected", _CASES)
def test_two_direction_trees_follow_the_classical_rule(
    getter, null, tests, leaves, pvalues, expected
):
    """A lone plain rejection adds its direction; the robust pair breaks ties."""
    leaf = {"stay": null, "a": leaves[0], "b": leaves[1]}[expected]
    assert _walk(getter(null), dict(zip(tests, pvalues))) == leaf


@pytest.mark.parametrize(
    "getter, null, test, leaf",
    [
        (get_spec, "SDM", "LM-Error-SDM", "GNS"),
        (get_spec, "SDEM", "LM-Lag-SDEM", "GNS"),
        (get_spec, "SDMTobit", "LM-Error-SDM", "GNSTobit"),
        (get_panel_spec, "SDM", "Panel-LM-Error-SDM", "GNS"),
        (get_panel_spec, "SDEM", "Panel-LM-Lag-SDEM", "GNS"),
    ],
)
def test_one_direction_trees_read_their_one_test(getter, null, test, leaf):
    assert _walk(getter(null), {test: 0.01}) == leaf
    assert _walk(getter(null), {test: 0.5}) == null


_OLS_CASES = [
    pytest.param({"LM-WX": 0.01}, "SLX", id="wx-alone"),
    # The lag and WX: Koley and Bera's pair, whose lag test is LM-Error here.
    pytest.param(
        {"LM-Lag": 0.01, "LM-Error": 0.5, "LM-WX": 0.01, "Robust-LM-WX": 0.01},
        "SLX",
        id="lag-wx-wx-robust",
    ),
    pytest.param(
        {"LM-Lag": 0.01, "LM-Error": 0.2, "LM-WX": 0.01, "Robust-LM-WX": 0.5},
        "SAR",
        id="lag-wx-lag-stronger",
    ),
    pytest.param(
        {"LM-Lag": 0.01, "LM-Error": 0.6, "LM-WX": 0.01, "Robust-LM-WX": 0.2},
        "SLX",
        id="lag-wx-wx-stronger",
    ),
    pytest.param({"LM-Error": 0.02, "LM-WX": 0.01}, "SLX", id="error-wx-wx-stronger"),
    pytest.param(
        {"LM-Error": 0.001, "LM-WX": 0.01}, "SEM", id="error-wx-error-stronger"
    ),
    # All three: the lag and the error settle it, as in the paper's Figure 2;
    # the SAR or SEM fit then tests WX.
    pytest.param(
        {
            "LM-Lag": 0.01,
            "LM-Error": 0.01,
            "LM-WX": 0.001,
            "Robust-LM-Lag": 0.5,
            "Robust-LM-Error": 0.01,
        },
        "SEM",
        id="all-three",
    ),
]


@pytest.mark.parametrize("prefix", ["", "Panel-"])
@pytest.mark.parametrize("pvalues, leaf", _OLS_CASES)
def test_the_ols_tree_reads_wx_too(prefix, pvalues, leaf):
    """The OLS fit leaves WX untested as well (Koley & Bera 2024)."""
    spec = get_panel_spec("OLS") if prefix else get_spec("OLS")
    p = {prefix + name: value for name, value in pvalues.items()}
    assert _walk(spec, p) == leaf
