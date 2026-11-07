"""Declarative decision-tree specs and renderers for spatial model selection.

This module provides:

- A small ``TreeNode`` dataclass that represents the LM-test decision trees
  as data, one per fitted model type.
- ``evaluate(node, sig_lookup, predicate_lookup)`` which walks the tree and
  returns the recommended model name plus the traversed path.
- ``render_ascii`` and ``render_graphviz`` which render the tree (with the
  traversed path highlighted) as either an indented string or a
  ``graphviz.Digraph``.

The same specs back both the cross-sectional (``models/base.py``) and the
panel (``models/panel_base.py``) decision methods.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Union

__all__ = [
    "TreeNode",
    "evaluate",
    "render_ascii",
    "render_graphviz",
    "graphviz_available",
    "get_spec",
    "get_panel_spec",
    "get_flow_spec",
    "get_panel_flow_spec",
]


# ---------------------------------------------------------------------------
# Tree primitives
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A single node in a decision tree.

    Parameters
    ----------
    kind : {"test", "predicate"}
        ``"test"`` means evaluate the named Bayesian LM test against alpha.
        ``"predicate"`` means evaluate a custom callable (used for the
        OLS robust-pair tie-break).
    name : str
        For ``kind="test"``, the test name as it appears in the diagnostics
        DataFrame (e.g. ``"LM-Lag"``). For ``kind="predicate"``, a human
        readable label (used as the node label and edge labels).
    if_true : TreeNode or str
        Subtree to follow when the test is significant (or predicate is
        true). A ``str`` is treated as a leaf with that model name.
    if_false : TreeNode or str
        Subtree to follow otherwise.
    predicate_id : str, optional
        For ``kind="predicate"``, an identifier looked up in the
        ``predicate_lookup`` dict passed to :func:`evaluate`.
    """

    kind: str  # "test" | "predicate"
    name: str
    if_true: Union["TreeNode", str]
    if_false: Union["TreeNode", str]
    predicate_id: str | None = None
    # Auto-assigned during ``evaluate`` for stable rendering identifiers.
    _node_id: str = field(default="", repr=False)


def _assign_ids(
    node: Union[TreeNode, str], counter: list[int], _visited: set | None = None
) -> None:
    if _visited is None:
        _visited = set()
    if isinstance(node, str):
        return
    obj_id = id(node)
    if obj_id in _visited:
        return  # shared node: ID already assigned on first encounter
    _visited.add(obj_id)
    node._node_id = f"n{counter[0]}"
    counter[0] += 1
    _assign_ids(node.if_true, counter, _visited)
    _assign_ids(node.if_false, counter, _visited)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    root: Union[TreeNode, str],
    sig_lookup: Callable[[str], bool],
    predicate_lookup: dict[str, Callable[[], bool]] | None = None,
) -> tuple[str, list[tuple[str, bool]]]:
    """Walk the tree and return the recommended model and traversed path.

    Parameters
    ----------
    root : TreeNode or str
        Root of the decision tree (or a bare leaf string).
    sig_lookup : callable
        Maps a test name to a boolean indicating significance.
    predicate_lookup : dict, optional
        Maps a ``predicate_id`` to a zero-argument callable returning bool.

    Returns
    -------
    decision : str
        Recommended model name.
    path : list of (node_id, branch_taken)
        Sequence of decisions made. ``branch_taken`` is ``True`` if the
        ``if_true`` branch was followed, else ``False``.
    """
    _assign_ids(root, [0])
    predicate_lookup = predicate_lookup or {}
    path: list[tuple[str, bool]] = []
    node: Union[TreeNode, str] = root
    while not isinstance(node, str):
        if node.kind == "test":
            branch = bool(sig_lookup(node.name))
        elif node.kind == "predicate":
            if node.predicate_id is None:
                raise ValueError("predicate node missing predicate_id")
            branch = bool(predicate_lookup[node.predicate_id]())
        else:
            raise ValueError(f"unknown node kind: {node.kind}")
        path.append((node._node_id, branch))
        node = node.if_true if branch else node.if_false
    return node, path


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------


def _node_label(node: TreeNode) -> str:
    if node.kind == "predicate":
        return node.name
    return node.name


def render_ascii(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> str:
    """Render the decision tree as an indented ASCII string.

    The traversed path is highlighted (``*`` marker on the chosen edge),
    and the chosen leaf is annotated with ``← SELECTED``.  Shared subtree
    objects (DAG nodes with multiple parents) are expanded once; subsequent
    references emit a short back-pointer line instead of repeating the
    subtree.
    """
    p_values = p_values or {}
    path_dict = dict(path)

    lines: list[str] = []
    rendered_nodes: set[str] = set()  # node_ids already fully expanded

    def render(
        node: Union[TreeNode, str],
        prefix: str,
        is_last: bool,
        edge_label: str | None,
        on_path: bool,
    ) -> None:
        connector = "" if edge_label is None else ("└── " if is_last else "├── ")
        marker = " *" if on_path else ""
        if isinstance(node, str):
            tag = " ← SELECTED" if on_path and node == decision else ""
            edge = f"<{edge_label}> " if edge_label else ""
            lines.append(f"{prefix}{connector}{edge}[{node}]{marker}{tag}")
            return

        # Internal node
        label = _node_label(node)
        annot = ""
        if on_path and node.kind == "test" and node.name in p_values:
            pv = p_values[node.name]
            annot = f"  (p={pv:.4f}, alpha={alpha})"
        edge_str = f"<{edge_label}> " if edge_label else ""

        # Shared-node deduplication: if this node was already fully rendered,
        # emit a back-reference instead of re-expanding the subtree.
        if node._node_id in rendered_nodes:
            lines.append(f"{prefix}{connector}{edge_str}→ {label}{marker}  (see above)")
            return

        rendered_nodes.add(node._node_id)
        lines.append(f"{prefix}{connector}{edge_str}{label}{marker}{annot}")

        # Determine child path membership
        new_prefix = prefix + ("    " if is_last else "│   ") if edge_label else ""
        children: list[tuple[str, Union[TreeNode, str]]] = [
            ("sig" if node.kind == "test" else "true", node.if_true),
            ("not sig" if node.kind == "test" else "false", node.if_false),
        ]
        taken = path_dict.get(node._node_id)
        for i, (lbl, child) in enumerate(children):
            child_on_path = on_path and (
                (taken is True and lbl in ("sig", "true"))
                or (taken is False and lbl in ("not sig", "false"))
            )
            render(child, new_prefix, i == len(children) - 1, lbl, child_on_path)

    if isinstance(root, str):
        return f"[{root}] ← SELECTED"
    render(root, "", True, None, on_path=True)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphviz renderer
# ---------------------------------------------------------------------------


def graphviz_available() -> bool:
    """Return True if the optional ``graphviz`` package is importable."""
    return importlib.util.find_spec("graphviz") is not None


def render_graphviz(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
    title: str | None = None,
    theme: str | Any = "default",
) -> Any:
    """Render the decision tree as a ``graphviz.Digraph``.

    The traversed path is drawn with thicker, colored edges and filled
    nodes; the chosen leaf is highlighted.  Three built-in themes are
    provided (``"default"``, ``"minimal"``, ``"dark"``) and users may
    supply a custom :class:`~neighbayes.diagnostics._decision_style.GraphTheme`
    object for full control.

    Requires the optional ``graphviz`` Python package.
    """
    from ._decision_style import get_theme

    graphviz = importlib.import_module("graphviz")
    p_values = p_values or {}
    path_dict = dict(path)
    t = get_theme(theme)

    dot = graphviz.Digraph()
    for k, v in t.graph_attrs().items():
        dot.attr("graph", **{k: v})
    if title:
        dot.attr(
            label=title,
            labelloc="t",
            fontsize=t.title_fontsize,
            fontcolor=t.title_fontcolor,
        )

    if isinstance(root, str):
        style = t.leaf_chosen.as_dict()
        dot.node("leaf", label=root, **style)
        return dot

    leaf_counter = [0]
    emitted_nodes: set[str] = set()

    def _node_style(
        node: TreeNode, on_path: bool, is_chosen: bool = False
    ) -> dict[str, str]:
        if is_chosen:
            base = t.leaf_chosen.as_dict()
        elif isinstance(node, str):
            base = t.leaf_other.as_dict()
        elif node.kind == "test":
            base = t.test_node.as_dict()
        elif node.kind == "predicate":
            base = t.predicate_node.as_dict()
        else:
            base = {}
        if on_path and t.node_on_path is not None:
            override = t.node_on_path.as_dict()
            override.pop("shape", None)  # preserve base shape (ellipse / diamond / box)
            base = {**base, **override}
        return base

    def _edge_style(on_path: bool) -> dict[str, str]:
        return t.edge_on_path.as_dict() if on_path else t.edge_off_path.as_dict()

    def add(
        node: Union[TreeNode, str],
        parent_id: str | None,
        edge_label: str | None,
        on_path: bool,
    ) -> None:
        if isinstance(node, str):
            leaf_id = f"leaf{leaf_counter[0]}"
            leaf_counter[0] += 1
            is_chosen = on_path and node == decision
            style = _node_style(node, on_path, is_chosen=is_chosen)
            dot.node(leaf_id, label=node, **style)
            if parent_id is not None:
                estyle = _edge_style(on_path)
                dot.edge(parent_id, leaf_id, label=edge_label or "", **estyle)
            return

        label = _node_label(node)
        if on_path and node.kind == "test" and node.name in p_values:
            label = f"{label}\np={p_values[node.name]:.3f}"
        already_emitted = node._node_id in emitted_nodes
        if not already_emitted:
            emitted_nodes.add(node._node_id)
            style = _node_style(node, on_path)
            dot.node(node._node_id, label=label, **style)
        if parent_id is not None:
            estyle = _edge_style(on_path)
            dot.edge(parent_id, node._node_id, label=edge_label or "", **estyle)

        if not already_emitted:
            taken = path_dict.get(node._node_id)
            true_label = "sig" if node.kind == "test" else "true"
            false_label = "not sig" if node.kind == "test" else "false"
            add(node.if_true, node._node_id, true_label, on_path and taken is True)
            add(node.if_false, node._node_id, false_label, on_path and taken is False)

    add(root, None, None, on_path=True)
    return dot


# ---------------------------------------------------------------------------
# Public dispatch helper
# ---------------------------------------------------------------------------


def render(
    root: Union[TreeNode, str],
    path: list[tuple[str, bool]],
    decision: str,
    p_values: dict[str, float] | None = None,
    alpha: float = 0.05,
    fmt: str = "graphviz",
    title: str | None = None,
    theme: str | Any = "default",
) -> Any:
    """Dispatch to the requested renderer, with graphviz availability check.

    If ``fmt == "graphviz"`` and the ``graphviz`` package is not installed,
    a :class:`UserWarning` is issued and the ASCII rendering is returned.
    """
    if fmt == "model":
        return decision
    if fmt == "ascii":
        return render_ascii(root, path, decision, p_values=p_values, alpha=alpha)
    if fmt == "graphviz":
        if not graphviz_available():
            warnings.warn(
                "graphviz package is not installed; falling back to ASCII "
                "rendering. Install with `pip install graphviz` (and the "
                "system 'graphviz' binary) to enable graph output.",
                UserWarning,
                stacklevel=3,
            )
            return render_ascii(root, path, decision, p_values=p_values, alpha=alpha)
        return render_graphviz(
            root,
            path,
            decision,
            p_values=p_values,
            alpha=alpha,
            title=title,
            theme=theme,
        )
    raise ValueError(
        f"unknown format {fmt!r}; expected 'graphviz', 'ascii', or 'model'"
    )


# ---------------------------------------------------------------------------
# Gaussian and Tobit decision tree specs
# ---------------------------------------------------------------------------
#
# Every fitted model whose tree can extend it in two directions reads its four
# tests as Anselin, Bera, Florax and Yoon (1996) read theirs at the OLS fit,
# the rule ``spreg.stge_classic`` implements.  A plain LM test is valid when
# the other direction is absent; a robust one is valid either way and pays for
# it in power.  So the robust pair is consulted only when both plain tests
# reject:
#
# - neither plain test rejects: keep the fitted model;
# - one rejects: add that direction;
# - both reject: add the direction whose robust test rejects; if both or
#   neither do, the one with the smaller robust p-value.
#
# The OLS fit leaves a third direction, WX, and its tree is the one in the
# paper's Figure 2 (see ``_ols_spec``).  The SDM and SDEM leave one direction,
# and their one test decides.  SARAR and GNS are where the tests point, though
# neighbayes does not fit them.


@dataclass(frozen=True)
class _Direction:
    """A direction a fitted model leaves at zero, as its tree tests it."""

    test: str  # plain LM test
    robust: str  # the same test, robust to the other direction
    model: str  # the model that adds the direction


# The two directions each model's tree tests, as its diagnostic suite labels
# them.  Panel suites carry the same labels behind a ``Panel-`` prefix.
_DIRECTION_PAIRS: dict[str, tuple[_Direction, _Direction]] = {
    "OLS": (
        _Direction("LM-Lag", "Robust-LM-Lag", "SAR"),
        _Direction("LM-Error", "Robust-LM-Error", "SEM"),
    ),
    "SAR": (
        _Direction("LM-WX", "Robust-LM-WX", "SDM"),
        _Direction("LM-Error", "Robust-LM-Error", "SARAR"),
    ),
    "SEM": (
        _Direction("LM-Lag", "Robust-LM-Lag", "SARAR"),
        _Direction("LM-WX", "Robust-LM-WX", "SDEM"),
    ),
    "SLX": (
        _Direction("LM-Lag", "Robust-LM-Lag-SDM", "SDM"),
        _Direction("LM-Error", "Robust-LM-Error-SDEM", "SDEM"),
    ),
}

# At the OLS fit the lag score is the error score plus β̂ᵀ times the WX score,
# and the error and WX scores are uncorrelated.  So Koley and Bera's lag robust
# to WX is the error test, and the error and WX tests robust to each other are
# the plain ones; the suite runs none of them twice, and the tree reads LM-Error
# where it needs the lag robust to WX.
_OLS_LAG_WX = ("LM-Error", "Robust-LM-WX")
_OLS_ERROR_WX = ("LM-Error", "LM-WX")

# The one direction the SDM and SDEM leave: (test, model).
_SINGLE_DIRECTION: dict[str, tuple[str, str]] = {
    "SDM": ("LM-Error-SDM", "GNS"),
    "SDEM": ("LM-Lag-SDEM", "GNS"),
}

# ``p(a) <= p(b)`` for the robust pair of every two-direction tree, keyed by
# the predicate id its tie-break nodes carry.
_PREDICATES = {
    **{null: (a.robust, b.robust) for null, (a, b) in _DIRECTION_PAIRS.items()},
    "OLS-lag-WX": _OLS_LAG_WX,
    "OLS-error-WX": _OLS_ERROR_WX,
}
PREDICATE_PAIRS: dict[str, tuple[str, str]] = {
    prefix + key: (prefix + a, prefix + b)
    for key, (a, b) in _PREDICATES.items()
    for prefix in ("", "Panel-")
}


def _test(name: str, if_true, if_false) -> TreeNode:
    return TreeNode(kind="test", name=name, if_true=if_true, if_false=if_false)


def _stronger(a: str, b: str, to_a: str, to_b: str, key: str) -> TreeNode:
    """``to_a`` when test ``a`` has the smaller p-value, else ``to_b``."""
    return TreeNode(
        kind="predicate",
        name=f"{a} p <= {b} p",
        if_true=to_a,
        if_false=to_b,
        predicate_id=key,
    )


def _settle(a: _Direction, b: _Direction, key: str, prefix: str, family: str):
    """Both plain tests reject: the robust pair picks, the stronger on a tie."""
    ra, rb = prefix + a.robust, prefix + b.robust
    to_a, to_b = a.model + family, b.model + family
    # Each tie-break is built anew, so the renderers draw both.
    return _test(
        ra,
        _test(rb, _stronger(ra, rb, to_a, to_b, prefix + key), to_a),
        _test(rb, to_b, _stronger(ra, rb, to_a, to_b, prefix + key)),
    )


def _pair_spec(null: str, *, prefix: str = "", family: str = "") -> TreeNode:
    """The tree of a model that leaves two directions untested.

    ``prefix`` is the suite's test-label prefix (``"Panel-"`` for panels) and
    ``family`` the leaves' suffix (``"Tobit"`` for the Tobit classes).
    """
    a, b = _DIRECTION_PAIRS[null]
    to_a, to_b = a.model + family, b.model + family
    return _test(
        prefix + a.test,
        _test(prefix + b.test, _settle(a, b, null, prefix, family), to_a),
        _test(prefix + b.test, to_b, null + family),
    )


def _ols_spec(*, prefix: str = "") -> TreeNode:
    """The OLS tree: the lag, the error process and ``WX``.

    One rejection adds its direction (SAR, SEM, SLX).  The lag and the error
    together go to their robust pair, whether or not ``WX`` also rejects; the
    SAR or SEM fit then tests ``WX``.  The lag and ``WX`` (without the error) go
    to Koley and Bera's robust pair, whose lag test is LM-Error here: it has not
    rejected, so ``WX`` goes ahead if Robust LM-WX rejects, and otherwise the
    smaller p-value of the two decides.  The error and ``WX`` go to the smaller
    p-value.
    """
    lag, error = _DIRECTION_PAIRS["OLS"]
    wx = prefix + "LM-WX"
    lag_robust, wx_robust = (prefix + name for name in _OLS_LAG_WX)
    lag_wx = _test(
        wx_robust,
        "SLX",
        TreeNode(
            kind="predicate",
            name=f"{lag_robust} (the lag robust to WX) p <= {wx_robust} p",
            if_true="SAR",
            if_false="SLX",
            predicate_id=prefix + "OLS-lag-WX",
        ),
    )
    error_wx = _stronger(
        *(prefix + name for name in _OLS_ERROR_WX),
        "SEM",
        "SLX",
        prefix + "OLS-error-WX",
    )
    return _test(
        prefix + lag.test,
        _test(
            prefix + error.test,
            _settle(lag, error, "OLS", prefix, ""),
            _test(wx, lag_wx, "SAR"),
        ),
        _test(prefix + error.test, _test(wx, error_wx, "SEM"), _test(wx, "SLX", "OLS")),
    )


def _single_spec(null: str, *, prefix: str = "", family: str = "") -> TreeNode:
    """The tree of a model that leaves one direction untested."""
    name, model = _SINGLE_DIRECTION[null]
    return TreeNode(
        kind="test",
        name=prefix + name,
        if_true=model + family,
        if_false=null + family,
    )


def _glm_spec(family: str) -> TreeNode:
    """A GLM's tests, read as a pair (:mod:`.lmtests.cross_sectional_glm`).

    ``family`` is the aspatial class (``"Logit"``, ``"Probit"``, ``"NegBin"``),
    whose SAR, SEM and SDEM classes the leaves name: the unmarked SAR is the
    latent one and the reduced form is marked ``latent=False`` (ADR 0004).  At
    the GLM null the latent SAR's
    score splits into the spillover term (``Robust-LM-Lag``, the reduced-form
    SAR's score) and the error term (``LM-Error``, the latent SEM's), and
    ``Robust-LM-Error`` tests whether they move in the proportion one
    :math:`\rho` implies:

    - spillover alone → the reduced-form SAR;
    - both, in proportion → the latent SAR;
    - both, out of proportion (opposite signs among them) → the SDEM,
      separate parameters for the covariates' spillover and the clustering;
    - error alone, out of proportion → the latent SEM; in proportion, the data
      cannot tell the latent SEM from the latent SAR, whose spillover would
      have stayed undetectable.

    The full test (``LM-Lag``) and ``LM-SARMA`` are reported beside the tree.
    """
    sem, sdem = f"SEM{family}", f"SDEM{family}"
    sar, latent_sar = f"SAR{family}(latent=False)", f"SAR{family}"
    return TreeNode(
        kind="test",
        name="Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-Error",
                if_true=sdem,
                if_false=latent_sar,
            ),
            if_false=sar,
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-Error",
                if_true=sem,
                if_false=f"{sem} or {latent_sar}",
            ),
            if_false=family,
        ),
    )


_CROSS_SECTIONAL_SPECS: dict[str, Callable[[], TreeNode]] = {
    "Logit": partial(_glm_spec, "Logit"),
    "Probit": partial(_glm_spec, "Probit"),
    "NegBin": partial(_glm_spec, "NegBin"),
    "OLS": _ols_spec,
    **{null: partial(_pair_spec, null) for null in ("SAR", "SEM", "SLX")},
    **{null: partial(_single_spec, null) for null in _SINGLE_DIRECTION},
    "SARTobit": partial(_pair_spec, "SAR", family="Tobit"),
    "SEMTobit": partial(_pair_spec, "SEM", family="Tobit"),
    "SDMTobit": partial(_single_spec, "SDM", family="Tobit"),
}


def get_spec(model_type: str) -> TreeNode | str:
    """Return the cross-sectional decision tree for ``model_type``.

    Falls back to a single leaf with the model name if no spec exists.
    """
    factory = _CROSS_SECTIONAL_SPECS.get(model_type)
    if factory is None:
        return model_type
    return factory()


# Static, random-effects and dynamic panels share these trees: a leaf names the
# process, and the panel design comes from the model the diagnostics ran on.
_PANEL_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLS": partial(_ols_spec, prefix="Panel-"),
    **{
        null: partial(_pair_spec, null, prefix="Panel-")
        for null in ("SAR", "SEM", "SLX")
    },
    **{
        null: partial(_single_spec, null, prefix="Panel-") for null in _SINGLE_DIRECTION
    },
    "SARTobit": partial(_pair_spec, "SAR", prefix="Panel-", family="Tobit"),
    "SEMTobit": partial(_pair_spec, "SEM", prefix="Panel-", family="Tobit"),
}


def get_panel_spec(model_type: str) -> TreeNode | str:
    """Return the panel decision tree for ``model_type``.

    The model type is the class name: a Gaussian process (``"SAR"``, for its
    fixed-effects, random-effects and dynamic panels alike) or a Tobit
    (``"SARTobit"``, for its pooled panel).  Falls back to a leaf if not found.
    """
    factory = _PANEL_SPECS.get(model_type)
    return model_type if factory is None else factory()


# ---------------------------------------------------------------------------
# Flow (origin-destination) decision tree specs
# ---------------------------------------------------------------------------


def _ols_flow_spec() -> TreeNode:
    """OLSFlow decision tree.

    The joint test is the gatekeeper: any spatial dependence across the three
    flow directions (destination, origin, network) recommends SARFlow; no
    signal recommends staying with OLSFlow.
    """
    return TreeNode(
        kind="test",
        name="LM-Flow-Joint",
        if_true="SARFlow",
        if_false="OLSFlow",
    )


def _panel_ols_flow_spec() -> TreeNode:
    """OLSFlow panel decision tree (panel mirror of ``_ols_flow_spec``)."""
    return TreeNode(
        kind="test",
        name="Panel-LM-Flow-Joint",
        if_true="SARFlow",
        if_false="OLSFlow",
    )


_FLOW_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLSFlow": _ols_flow_spec,
}

_PANEL_FLOW_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLSFlow": _panel_ols_flow_spec,
}


def get_flow_spec(model_type: str) -> TreeNode | str:
    """Return the flow decision tree for ``model_type``.

    Falls back to a single leaf with the model name if no spec is defined.
    """
    factory = _FLOW_SPECS.get(model_type)
    if factory is None:
        return model_type
    return factory()


def get_panel_flow_spec(model_type: str) -> TreeNode | str:
    """Return the panel-flow decision tree for ``model_type``.

    Falls back to a single leaf with the model name if no spec is defined.
    """
    factory = _PANEL_FLOW_SPECS.get(model_type)
    if factory is None:
        return model_type
    return factory()
