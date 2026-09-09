"""Render a Trajectory as a DAG in Mermaid or Graphviz DOT.

Nodes are the query, the frontier branches, one per step, the evidence items
referenced by those steps, and the report. Edges are the step spine, each step
to the branch it worked on, each step to the evidence it introduced or pruned,
and the write step to the report.

Evidence nodes are labelled by id: step records carry evidence ids but the
Trajectory does not serialize the evidence pool, so urls exist only for
citations. Branches hang off the query unless ``parent_id`` is present in
``final_stats["branches"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adr.core.types import ActionType, Trajectory

_MAX_LABEL = 44


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str  # query | branch | step | evidence | report | citation


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)


def _clip(text: str, limit: int = _MAX_LABEL) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _ids(value: Any) -> list[str]:
    return [str(x) for x in value] if isinstance(value, list) else []


def build_graph(
    traj: Trajectory, *, include_evidence: bool = True, include_citations: bool = False
) -> Graph:
    graph = Graph()
    add_node = graph.nodes.append
    add_edge = graph.edges.append

    add_node(Node("Q", f"{_clip(traj.query.text)}\n[{traj.query.id}]", "query"))

    branches = traj.final_stats.get("branches") or []
    known = {str(b.get("id")) for b in branches if isinstance(b, dict)}
    for row in branches:
        if not isinstance(row, dict):
            continue
        bid = str(row.get("id"))
        add_node(Node(f"B_{bid}", f"{_clip(row.get('goal', bid))}\n[{row.get('status', '?')}]", "branch"))
        parent = row.get("parent_id")
        src = f"B_{parent}" if parent and str(parent) in known else "Q"
        add_edge(Edge(src, f"B_{bid}"))

    # Evidence is pruned if any step prunes it, regardless of when it appeared.
    pruned: set[str] = set()
    for step in traj.steps:
        pruned.update(_ids(step.extra.get("pruned_item_ids")))
        if step.action.type is ActionType.PRUNE:
            pruned.update(step.action.evidence_ids)

    seen_evidence: set[str] = set()
    write_step: str | None = None

    for step in traj.steps:
        sid = f"S{step.step}"
        detail = [_tokens(step.tokens.total_tokens) + " tok"]
        if step.latency_s:
            detail.append(f"{step.latency_s:.1f}s")
        rnd = step.extra.get("round_id")
        head = f"{step.step}. {step.action.type.value}" + (f" (round {rnd})" if rnd else "")
        add_node(Node(sid, f"{head}\n{' · '.join(detail)}", "step"))
        if step.step > 1:
            add_edge(Edge(f"S{step.step - 1}", sid))
        if step.action.type is ActionType.WRITE:
            write_step = sid

        touched = {str(step.action.subtask_id)} if step.action.subtask_id else set()
        touched.update(
            str(f.get("node_id"))
            for f in step.extra.get("frontier") or []
            if isinstance(f, dict) and f.get("node_id")
        )
        for bid in sorted(touched & known):
            add_edge(Edge(sid, f"B_{bid}", "works"))

        if not include_evidence:
            continue
        introduced = _ids(step.extra.get("new_item_ids"))
        if not introduced and step.action.type in {ActionType.SEARCH, ActionType.READ}:
            introduced = list(step.action.evidence_ids)
        for eid in introduced:
            if eid not in seen_evidence:
                seen_evidence.add(eid)
                kind = "evidence"
                mark = "pruned" if eid in pruned else "kept"
                add_node(Node(f"E_{eid}", f"{eid}\n[{mark}]", kind))
            add_edge(Edge(sid, f"E_{eid}", "found"))
        for eid in sorted(set(_ids(step.extra.get("pruned_item_ids"))) | set(
            step.action.evidence_ids if step.action.type is ActionType.PRUNE else []
        )):
            if eid in seen_evidence:
                add_edge(Edge(sid, f"E_{eid}", "prune", dashed=True))

    if traj.report is not None:
        n_cit = len(traj.report.citations)
        add_node(
            Node("R", f"report\n{len(traj.report.article)} chars · {n_cit} citations", "report")
        )
        add_edge(Edge(write_step or "Q", "R"))
        if include_citations:
            for i, url in enumerate(traj.report.citations, start=1):
                add_node(Node(f"C{i}", _clip(url), "citation"))
                add_edge(Edge("R", f"C{i}", dashed=True))
    return graph


# ── renderers ─────────────────────────────────────────────────────────
_MERMAID_SHAPE = {
    "query": ('(["', '"])'),
    "branch": ('["', '"]'),
    "step": ('["', '"]'),
    "evidence": ('("', '")'),
    "report": ('[/"', '"/]'),
    "citation": ('("', '")'),
}

_MERMAID_CLASSES = """    classDef query fill:#1f2937,stroke:#111827,color:#f9fafb
    classDef branch fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef step fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef kept fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef pruned fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-dasharray:3 3
    classDef report fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef citation fill:#fff,stroke:#d1d5db,color:#374151"""


def _mermaid_label(text: str) -> str:
    return text.replace('"', "#quot;").replace("\n", "<br/>")


def to_mermaid(graph: Graph, *, direction: str = "TD") -> str:
    lines = [f"flowchart {direction}"]
    for node in graph.nodes:
        open_s, close_s = _MERMAID_SHAPE[node.kind]
        lines.append(f"    {node.id}{open_s}{_mermaid_label(node.label)}{close_s}")
    for edge in graph.edges:
        arrow = "-.->" if edge.dashed else "-->"
        mid = f"|{_mermaid_label(edge.label)}|" if edge.label else ""
        lines.append(f"    {edge.src} {arrow}{mid} {edge.dst}")
    lines.append(_MERMAID_CLASSES)
    for node in graph.nodes:
        cls = node.kind
        if node.kind == "evidence":
            cls = "pruned" if "[pruned]" in node.label else "kept"
        lines.append(f"    class {node.id} {cls}")
    return "\n".join(lines) + "\n"


_DOT_ATTRS = {
    "query": 'shape=oval, style=filled, fillcolor="#1f2937", fontcolor="#f9fafb"',
    "branch": 'shape=box, style="rounded,filled", fillcolor="#dbeafe"',
    "step": 'shape=box, style=filled, fillcolor="#f3f4f6"',
    "report": 'shape=note, style=filled, fillcolor="#fef3c7"',
    "citation": 'shape=ellipse, style=filled, fillcolor="#ffffff"',
}
_DOT_EVIDENCE = {
    "kept": 'shape=ellipse, style=filled, fillcolor="#dcfce7"',
    "pruned": 'shape=ellipse, style="filled,dashed", fillcolor="#fee2e2"',
}


def to_dot(graph: Graph, *, direction: str = "TB") -> str:
    lines = [f"digraph trajectory {{\n    rankdir={direction};", '    node [fontname="Helvetica", fontsize=10];']
    for node in graph.nodes:
        if node.kind == "evidence":
            attrs = _DOT_EVIDENCE["pruned" if "[pruned]" in node.label else "kept"]
        else:
            attrs = _DOT_ATTRS[node.kind]
        label = node.label.replace('"', '\\"').replace("\n", "\\n")
        lines.append(f'    "{node.id}" [label="{label}", {attrs}];')
    for edge in graph.edges:
        bits = ["style=dashed"] if edge.dashed else []
        if edge.label:
            bits.append(f'label="{edge.label}"')
        suffix = f' [{", ".join(bits)}]' if bits else ""
        lines.append(f'    "{edge.src}" -> "{edge.dst}"{suffix};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def render(
    traj: Trajectory,
    *,
    fmt: str = "mermaid",
    include_evidence: bool = True,
    include_citations: bool = False,
    direction: str | None = None,
) -> str:
    graph = build_graph(traj, include_evidence=include_evidence, include_citations=include_citations)
    if fmt == "mermaid":
        return to_mermaid(graph, direction=direction or "TD")
    if fmt == "dot":
        return to_dot(graph, direction=direction or "TB")
    raise ValueError(f"Unknown format {fmt!r}; expected 'mermaid' or 'dot'")
