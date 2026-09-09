"""Trajectory -> DAG rendering."""

from __future__ import annotations

from adr.core.types import (
    ActionType,
    OrchestratorAction,
    Query,
    Report,
    StepRecord,
    TokenUsage,
    Trajectory,
)
from adr.eval.dag import build_graph, render, to_dot, to_mermaid


def _traj() -> Trajectory:
    query = Query(id="chip", text="why is there a chip shortage", dataset="deep_research_gym")
    return Trajectory(
        query=query,
        steps=[
            StepRecord(
                step=1,
                action=OrchestratorAction(type=ActionType.PRUNE, evidence_ids=["e2"]),
                tokens=TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
                latency_s=10.0,
                extra={
                    "round_id": 1,
                    "new_item_ids": ["e1", "e2"],
                    "kept_item_ids": ["e1"],
                    "pruned_item_ids": ["e2"],
                    "frontier": [{"node_id": "n1", "status": "open"}],
                },
            ),
            StepRecord(
                step=2,
                action=OrchestratorAction(type=ActionType.WRITE, report_draft="# R"),
                tokens=TokenUsage(total_tokens=5900),
            ),
            StepRecord(step=3, action=OrchestratorAction(type=ActionType.TERMINATE)),
        ],
        report=Report(article="# R", citations=["https://a.example/1"]),
        final_stats={
            "branches": [
                {"id": "n1", "goal": "goal one", "status": "active"},
                {"id": "n2", "goal": "goal two", "status": "open", "parent_id": "n1"},
            ]
        },
    )


def test_graph_shape() -> None:
    g = build_graph(_traj())
    nodes = {n.id: n for n in g.nodes}
    assert set(nodes) == {"Q", "B_n1", "B_n2", "S1", "S2", "S3", "E_e1", "E_e2", "R"}
    assert nodes["E_e2"].label.endswith("[pruned]")
    assert nodes["E_e1"].label.endswith("[kept]")

    edges = {(e.src, e.dst, e.label) for e in g.edges}
    assert ("Q", "B_n1", "") in edges
    assert ("B_n1", "B_n2", "") in edges, "parent_id must nest branches"
    assert ("S1", "S2", "") in edges and ("S2", "S3", "") in edges
    assert ("S1", "B_n1", "works") in edges
    assert ("S1", "E_e1", "found") in edges
    assert ("S2", "R", "") in edges, "report hangs off the write step"
    assert any(e.dashed and e.dst == "E_e2" for e in g.edges)


def test_toggles() -> None:
    bare = build_graph(_traj(), include_evidence=False)
    assert not [n for n in bare.nodes if n.kind == "evidence"]

    cited = build_graph(_traj(), include_citations=True)
    assert [n.label for n in cited.nodes if n.kind == "citation"] == ["https://a.example/1"]


def test_mermaid_and_dot_are_wellformed() -> None:
    g = build_graph(_traj(), include_citations=True)

    mermaid = to_mermaid(g, direction="LR")
    assert mermaid.startswith("flowchart LR")
    assert "S1 -.->|prune| E_e2" in mermaid
    assert "class E_e2 pruned" in mermaid
    assert "<br/>" in mermaid and "\\n" not in mermaid

    dot = to_dot(g)
    assert dot.startswith("digraph trajectory {") and dot.rstrip().endswith("}")
    assert dot.count("->") == len(g.edges)
    assert '"S1" -> "E_e2" [style=dashed, label="prune"];' in dot


def test_render_dispatch() -> None:
    traj = _traj()
    assert render(traj, fmt="mermaid").startswith("flowchart TD")
    assert render(traj, fmt="dot").startswith("digraph")
    try:
        render(traj, fmt="svg")
    except ValueError as exc:
        assert "svg" in str(exc)
    else:
        raise AssertionError("unknown format must raise")
