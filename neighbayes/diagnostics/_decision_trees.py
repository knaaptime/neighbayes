"""Declarative decision-tree specs and renderers for spatial model selection.

This module provides:

- A small ``TreeNode`` dataclass that represents the Koley & Bera / stge_kb
  decision trees as data, parameterized per starting model type.
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
            lines.append(f"{prefix}{connector}[{node}]{marker}{tag}")
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
# Cross-sectional decision tree specs
# ---------------------------------------------------------------------------


def _ols_spec() -> TreeNode:
    """OLS decision tree (Koley & Bera 2024, stge_kb).

    Reachable terminals: ``OLS``, ``SAR``, ``SEM``.

    ``SARAR`` is intentionally **not** a terminal here.  SARAR's correct
    null model is SAR (or SEM), not OLS, so a Bayesian specification test
    for the SARAR alternative cannot be conducted from an OLS fit alone.
    When the OLS tests indicate joint lag+error dependence, the tree
    recommends the dominant single-channel model (SAR or SEM) and the
    user is expected to fit that intermediate model and re-run
    diagnostics from there to potentially escalate to SARAR.
    """
    # Robust-vs-robust tie-break used when both naive AND both robust
    # tests fire: the smaller-p (= larger-stat) channel wins.
    robust_tie_break = TreeNode(
        kind="predicate",
        name="Robust-LM-Lag p <= Robust-LM-Error p",
        if_true="SAR",
        if_false="SEM",
        predicate_id="robust_lag_pval_le_error_pval",
    )
    # Both naive sig: robust pair branch.  We never escalate to SARAR
    # because that would require a fitted SAR/SEM null.
    both_sig = TreeNode(
        kind="test",
        name="Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error",
            if_true=robust_tie_break,
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-Error",
            if_true="SEM",
            if_false=TreeNode(
                kind="predicate",
                name="LM-Lag p <= LM-Error p",
                if_true="SAR",
                if_false="SEM",
                predicate_id="lag_pval_le_error_pval",
            ),
        ),
    )
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=both_sig,
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-Error",
            if_true="SEM",
            if_false="OLS",
        ),
    )


def _sar_spec() -> TreeNode:
    # Precondition principle: a robust LM test only refines its naive
    # precursor — never consult ``Robust-LM-WX`` unless ``LM-WX`` has
    # already fired, and never commit to a model whose only support comes
    # from a robust test.
    #
    # WX channel: if naive ``LM-WX`` from the SAR fit fires AND the robust
    # version survives the ρ-orthogonalization, the omitted block is WX
    # (→ SDM).  If naive LM-WX is silent, or fires but the robust check
    # clears it, fall through to the error channel.
    #
    # Error channel (fall-through): naive ``LM-Error`` only escalates to
    # SARAR if the SAR-null-conditional ``Robust-LM-Error`` (Schur-purged
    # for ρ) also fires; otherwise the naive signal is attributable to
    # nuisance dependence already absorbed by the SAR fit and we keep SAR.
    #
    # The error branch is constructed twice (instead of being shared) so
    # the graphviz renderer draws two distinct subtrees and the visual
    # ordering matches the strict naive→robust descent within each
    # channel.
    def _error_branch() -> TreeNode:
        return TreeNode(
            kind="test",
            name="LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-Error",
                if_true="SARAR",
                if_false="SAR",
            ),
            if_false="SAR",
        )

    return TreeNode(
        kind="test",
        name="LM-WX",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDM",
            if_false=_error_branch(),
        ),
        if_false=_error_branch(),
    )


def _sem_spec() -> TreeNode:
    # If LM-Lag fires from a SEM fit it can mean either (i) a true lag on top
    # of the error process (→ SARAR) or (ii) omitted WX masquerading as ρ
    # (→ SDEM).  ``LM-WX`` (here the SEM-null variant exposed under that
    # label, see SEM diagnostic suite) settles the ambiguity.
    #
    # Robust-after-naive precondition: a naive precursor (``LM-Lag`` or
    # ``LM-WX``) must fire before consulting its Schur-orthogonalized
    # robust counterpart.  When both naive precursors fire AND both robust
    # tests survive, ``Robust-LM-Lag`` p vs ``Robust-LM-WX`` p settles
    # which direction the SEM fit is missing the most evidence in:
    # smaller-p (larger-stat) wins.
    both_robust = TreeNode(
        kind="predicate",
        name="Robust-LM-Lag p <= Robust-LM-WX p",
        if_true="SARAR",
        if_false="SDEM",
        predicate_id="lag_sem_pval_le_wx_sem_pval",
    )
    both_naive_sig = TreeNode(
        kind="test",
        name="Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true=both_robust,
            if_false="SARAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDEM",
            if_false="SEM",
        ),
    )
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-WX",
            if_true=both_naive_sig,
            if_false=TreeNode(
                kind="test",
                name="Robust-LM-Lag",
                if_true="SARAR",
                if_false="SEM",
            ),
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-WX",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-WX",
                if_true="SDEM",
                if_false="SEM",
            ),
            if_false="SEM",
        ),
    )


def _slx_spec() -> TreeNode:
    # Precondition principle: ``Robust-LM-Lag-SDM`` only refines naive
    # ``LM-Lag``, and ``Robust-LM-Error-SDEM`` only refines naive
    # ``LM-Error``.  Never escalate the SLX fit to SDM/SDEM unless the
    # corresponding naive precursor from the SLX fit has fired.
    #
    # When both naive precursors fire, both robust refinements are
    # consulted; if both robust tests survive, the smaller-p (larger-stat)
    # side wins via ``lag_sdm_pval_le_error_sdem_pval``.  ``GNS`` is
    # intentionally not a terminal here; users wanting GNS should fit
    # it explicitly and compare via :func:`bayes_factor_compare_models`.
    both_naive_sig = TreeNode(
        kind="test",
        name="Robust-LM-Lag-SDM",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error-SDEM",
            if_true=TreeNode(
                kind="predicate",
                name="Robust-LM-Lag-SDM p <= Robust-LM-Error-SDEM p",
                if_true="SDM",
                if_false="SDEM",
                predicate_id="lag_sdm_pval_le_error_sdem_pval",
            ),
            if_false="SDM",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-Error-SDEM",
            if_true="SDEM",
            if_false="SLX",
        ),
    )
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=both_naive_sig,
            if_false=TreeNode(
                kind="test",
                name="Robust-LM-Lag-SDM",
                if_true="SDM",
                if_false="SLX",
            ),
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-Error-SDEM",
                if_true="SDEM",
                if_false="SLX",
            ),
            if_false="SLX",
        ),
    )


def _sdm_spec() -> TreeNode:
    # Robust-after-naive: only escalate to GNS when both the naive SDM
    # error score (``LM-Error-SDM``) and its Neyman-orthogonal
    # counterpart (``Robust-LM-Error-SDM``, Schur-purged for ρ at the
    # SDM posterior mean) survive.  Without the robust gate the naive
    # error signal can be a relabelling of mis-specified ρ.
    return TreeNode(
        kind="test",
        name="LM-Error-SDM",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error-SDM",
            if_true="GNS",
            if_false="SDM",
        ),
        if_false="SDM",
    )


def _sdem_spec() -> TreeNode:
    # Robust-after-naive: only escalate to GNS when both ``LM-Lag-SDEM``
    # and ``Robust-LM-Lag-SDEM`` (Schur-purged for the WX block, with
    # λ filtered at its posterior mean) fire.
    return TreeNode(
        kind="test",
        name="LM-Lag-SDEM",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Lag-SDEM",
            if_true="GNS",
            if_false="SDEM",
        ),
        if_false="SDEM",
    )


# Tobit variants: the same logic, with Tobit leaves.
def _sar_tobit_spec() -> TreeNode:
    # Tobit mirror of ``_sar_spec`` — see that docstring.  The error
    # branch is built twice (not shared) so the graphviz layout draws
    # naive→robust strictly within each channel.
    def _error_branch() -> TreeNode:
        return TreeNode(
            kind="test",
            name="LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-Error",
                if_true="SARARTobit",
                if_false="SARTobit",
            ),
            if_false="SARTobit",
        )

    return TreeNode(
        kind="test",
        name="LM-WX",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDMTobit",
            if_false=_error_branch(),
        ),
        if_false=_error_branch(),
    )


def _sem_tobit_spec() -> TreeNode:
    # Tobit mirror of ``_sem_spec`` — see that docstring.
    both_robust = TreeNode(
        kind="predicate",
        name="Robust-LM-Lag p <= Robust-LM-WX p",
        if_true="SARARTobit",
        if_false="SDEMTobit",
        predicate_id="lag_sem_pval_le_wx_sem_pval",
    )
    both_naive_sig = TreeNode(
        kind="test",
        name="Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true=both_robust,
            if_false="SARARTobit",
        ),
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-WX",
            if_true="SDEMTobit",
            if_false="SEMTobit",
        ),
    )
    return TreeNode(
        kind="test",
        name="LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="LM-WX",
            if_true=both_naive_sig,
            if_false=TreeNode(
                kind="test",
                name="Robust-LM-Lag",
                if_true="SARARTobit",
                if_false="SEMTobit",
            ),
        ),
        if_false=TreeNode(
            kind="test",
            name="LM-WX",
            if_true=TreeNode(
                kind="test",
                name="Robust-LM-WX",
                if_true="SDEMTobit",
                if_false="SEMTobit",
            ),
            if_false="SEMTobit",
        ),
    )


def _sdm_tobit_spec() -> TreeNode:
    # Tobit mirror of ``_sdm_spec`` — see that docstring.
    return TreeNode(
        kind="test",
        name="LM-Error",
        if_true=TreeNode(
            kind="test",
            name="Robust-LM-Error",
            if_true="GNSTobit",
            if_false="SDMTobit",
        ),
        if_false="SDMTobit",
    )


_CROSS_SECTIONAL_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLS": _ols_spec,
    "SAR": _sar_spec,
    "SEM": _sem_spec,
    "SLX": _slx_spec,
    "SDM": _sdm_spec,
    "SDEM": _sdem_spec,
    "SARTobit": _sar_tobit_spec,
    "SEMTobit": _sem_tobit_spec,
    "SDMTobit": _sdm_tobit_spec,
}


def get_spec(model_type: str) -> TreeNode | str:
    """Return the cross-sectional decision tree for ``model_type``.

    Falls back to a single leaf with the model name if no spec exists
    (matches the current ``return model_type`` fallback behavior).
    """
    factory = _CROSS_SECTIONAL_SPECS.get(model_type)
    if factory is None:
        return model_type
    return factory()


# ---------------------------------------------------------------------------
# Panel decision tree specs
# ---------------------------------------------------------------------------


def _panel_ols_spec() -> TreeNode:
    # Panel mirror of :func:`_ols_spec`.  ``SARAR`` is
    # intentionally not a reachable terminal: its proper null is
    # ``SAR`` (or ``SEM``), and a valid
    # specification test for SARAR cannot be conducted from a panel-OLS
    # fit alone.  The tree recommends the dominant single-channel model
    # and the user re-runs diagnostics from there.
    robust_tie_break = TreeNode(
        kind="predicate",
        name="Panel-Robust-LM-Lag p <= Panel-Robust-LM-Error p",
        if_true="SAR",
        if_false="SEM",
        predicate_id="panel_robust_lag_pval_le_error_pval",
    )
    both_sig = TreeNode(
        kind="test",
        name="Panel-Robust-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error",
            if_true=robust_tie_break,
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error",
            if_true="SEM",
            if_false=TreeNode(
                kind="predicate",
                name="Panel-LM-Lag p <= Panel-LM-Error p",
                if_true="SAR",
                if_false="SEM",
                predicate_id="panel_lag_pval_le_error_pval",
            ),
        ),
    )
    # OLSPanel{RE} doesn't expose robust tests; the evaluator will simply
    # follow `if_false` when the test isn't present (sig_lookup returns
    # False for missing tests).
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true=both_sig,
            if_false="SAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true="SEM",
            if_false="OLS",
        ),
    )


def _panel_sar_spec() -> TreeNode:
    # Panel mirror of ``_sar_spec`` — see that docstring.  Build the
    # error branch twice (not shared) so graphviz draws naive→robust
    # strictly within each channel.
    def _error_branch() -> TreeNode:
        return TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true="SARAR",
            if_false="SAR",
        )

    return TreeNode(
        kind="test",
        name="Panel-LM-WX",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-WX",
            if_true="SDM",
            if_false=_error_branch(),
        ),
        if_false=_error_branch(),
    )


def _panel_sem_spec() -> TreeNode:
    # Panel mirror of ``_sem_spec`` — see that docstring.
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-LM-WX",
            if_true="SDEM",
            if_false="SARAR",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-WX",
            if_true="SDEM",
            if_false="SEM",
        ),
    )


def _panel_slx_spec() -> TreeNode:
    # Panel mirror of ``_slx_spec`` — see that docstring.
    both_naive_sig = TreeNode(
        kind="test",
        name="Panel-Robust-LM-Lag-SDM",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error-SDEM",
            if_true=TreeNode(
                kind="predicate",
                name="Panel-Robust-LM-Lag-SDM p <= Panel-Robust-LM-Error-SDEM p",
                if_true="SDM",
                if_false="SDEM",
                predicate_id="panel_lag_sdm_pval_le_error_sdem_pval",
            ),
            if_false="SDM",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Error-SDEM",
            if_true="SDEM",
            if_false="SLX",
        ),
    )
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true=both_naive_sig,
            if_false=TreeNode(
                kind="test",
                name="Panel-Robust-LM-Lag-SDM",
                if_true="SDM",
                if_false="SLX",
            ),
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-Error",
            if_true=TreeNode(
                kind="test",
                name="Panel-Robust-LM-Error-SDEM",
                if_true="SDEM",
                if_false="SLX",
            ),
            if_false="SLX",
        ),
    )


def _panel_sdm_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Error-SDM",
        if_true="GNS",
        if_false="SDM",
    )


def _panel_sdem_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag-SDEM",
        if_true="GNS",
        if_false="SDEM",
    )


# Static and dynamic panels share these trees: a leaf names the process, and
# the panel design comes from the model the diagnostics were run on.
_PANEL_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLS": _panel_ols_spec,
    "SAR": _panel_sar_spec,
    "SEM": _panel_sem_spec,
    "SLX": _panel_slx_spec,
    "SDM": _panel_sdm_spec,
    "SDEM": _panel_sdem_spec,
}


# Panel Tobit specs mirror the cross-sectional Tobit specs.
def _panel_tobit_sar_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Error",
        if_true=TreeNode(
            kind="test",
            name="Panel-Robust-LM-WX",
            if_true="SDMTobit",
            if_false="SARARTobit",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-WX",
            if_true="SDMTobit",
            if_false=TreeNode(
                kind="test",
                name="Panel-LM-WX",
                if_true="SDMTobit",
                if_false="SARTobit",
            ),
        ),
    )


def _panel_tobit_sem_spec() -> TreeNode:
    return TreeNode(
        kind="test",
        name="Panel-LM-Lag",
        if_true=TreeNode(
            kind="test",
            name="Panel-LM-WX",
            if_true="SDEMTobit",
            if_false="SARARTobit",
        ),
        if_false=TreeNode(
            kind="test",
            name="Panel-LM-WX",
            if_true="SDEMTobit",
            if_false="SEMTobit",
        ),
    )


_PANEL_TOBIT_SPECS: dict[str, Callable[[], TreeNode]] = {
    "SARTobit": _panel_tobit_sar_spec,
    "SEMTobit": _panel_tobit_sem_spec,
}


def get_panel_spec(model_type: str) -> TreeNode | str:
    """Return the panel decision tree for ``model_type``.

    The model type is the class name: a Gaussian process (``"SAR"``, for its
    fixed-effects, random-effects and dynamic panels alike) or a Tobit
    (``"SARTobit"``, for its pooled panel).  Falls back to a leaf if not found.
    """
    # Pooled Tobit panels (SARTobit, SEMTobit)
    if "Tobit" in model_type:
        factory = _PANEL_TOBIT_SPECS.get(model_type)
        if factory is not None:
            return factory()
        return model_type

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


def _sar_flow_spec() -> TreeNode:
    """SARFlow decision tree (Neyman-orthogonalized robust marginals).

    Robust marginals are queried in sequence (destination → origin →
    network); each is tested only once the previous has been checked.  Any
    significant direction confirms the SARFlow fit is warranted.  If none
    survive the orthogonalization the simpler OLSFlow is preferred.
    """
    return TreeNode(
        kind="test",
        name="Robust-LM-Flow-Destination",
        if_true="SARFlow",
        if_false=TreeNode(
            kind="test",
            name="Robust-LM-Flow-Origin",
            if_true="SARFlow",
            if_false=TreeNode(
                kind="test",
                name="Robust-LM-Flow-OD",
                if_true="SARFlow",
                if_false="OLSFlow",
            ),
        ),
    )


def _panel_ols_flow_spec() -> TreeNode:
    """OLSFlowPanelFE decision tree (panel mirror of ``_ols_flow_spec``)."""
    return TreeNode(
        kind="test",
        name="Panel-LM-Flow-Joint",
        if_true="SARFlow",
        if_false="OLSFlow",
    )


def _panel_sar_flow_spec() -> TreeNode:
    """SARFlowPanelFE decision tree (panel mirror of ``_sar_flow_spec``)."""
    return TreeNode(
        kind="test",
        name="Panel-Robust-LM-Flow-Destination",
        if_true="SARFlow",
        if_false=TreeNode(
            kind="test",
            name="Panel-Robust-LM-Flow-Origin",
            if_true="SARFlow",
            if_false=TreeNode(
                kind="test",
                name="Panel-Robust-LM-Flow-OD",
                if_true="SARFlow",
                if_false="OLSFlow",
            ),
        ),
    )


_FLOW_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLSFlow": _ols_flow_spec,
    "SARFlow": _sar_flow_spec,
}

_PANEL_FLOW_SPECS: dict[str, Callable[[], TreeNode]] = {
    "OLSFlowPanelFE": _panel_ols_flow_spec,
    "SARFlowPanelFE": _panel_sar_flow_spec,
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
