"""The full-tree replay check against a hand-built live writer input.

The expected string follows the live legacy path of the fork's
DeepResearchSkill, written out by hand rather than through writer_context:
learnings with their sources first, in discovery order (roots, then each
root's children depth first), then the first level's contexts followed by
each root's sub-call contexts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("gpt_researcher.skills.deep_research")

from adr.replay.check import check_full_replay, first_difference, trajectory_files  # noqa: E402


def _node(nid, parent, depth, learning, citation=None):
    return {
        "node_id": nid,
        "parent_node_id": parent,
        "tree_depth": depth,
        "research_goal": f"goal {nid}",
        "learnings": [learning],
        "citations": {learning: citation} if citation else {},
        "context": f"context of {nid}",
        "cost": {"tokens_input": 10},
    }


# d2b2, in the order the live run appends nodes: both roots, then r1's
# children, then r2's.
NODES = [
    _node("r1", None, 1, "L r1", "https://a"),
    _node("r2", None, 1, "L r2"),
    _node("r1c0", "r1", 2, "L r1c0", "https://b"),
    _node("r1c1", "r1", 2, "L r1c1"),
    _node("r2c0", "r2", 2, "L r2c0"),
    _node("r2c1", "r2", 2, "L r2c1"),
]
LIVE = "\n".join([
    "L r1 [Source: https://a]", "L r2", "L r1c0 [Source: https://b]", "L r1c1", "L r2c0", "L r2c1",
    "context of r1", "context of r2",
    "context of r1c0", "context of r1c1",
    "context of r2c0", "context of r2c1",
])


def _write(tmp_path: Path, final_context: str, nodes=NODES, name="57.json") -> Path:
    run = tmp_path / "run" / "gpt_researcher"
    run.mkdir(parents=True, exist_ok=True)
    p = run / name
    p.write_text(json.dumps({"query": "q", "nodes": nodes, "final_context": final_context}))
    return p


def test_full_replay_reproduces_the_live_input(tmp_path):
    r = check_full_replay(_write(tmp_path, LIVE))
    assert r.ok, r
    assert r.n_nodes == 6


def test_reordered_learnings_are_caught(tmp_path):
    lines = LIVE.split("\n")
    lines[0], lines[1] = lines[1], lines[0]  # what list(set(...)) used to do
    r = check_full_replay(_write(tmp_path, "\n".join(lines)))
    assert not r.ok
    assert r.first_diff == 3  # "L r" is shared
    assert "L r2" in r.live_snippet


def test_missing_node_cache_and_unfinished_runs(tmp_path):
    assert "GR_NODE_CACHE" in check_full_replay(_write(tmp_path, LIVE, nodes=[])).reason
    assert "did not finish" in check_full_replay(_write(tmp_path, "  ")).reason


def test_unknown_pack_is_rejected(tmp_path):
    assert "not replayable" in check_full_replay(_write(tmp_path, LIVE), pack="balanced").reason


def test_first_difference():
    assert first_difference("abc", "abc") is None
    assert first_difference("abc", "abd") == 2
    assert first_difference("ab", "abc") == 2


def test_trajectory_files_expands_run_dirs(tmp_path):
    a = _write(tmp_path, LIVE, name="1.json")
    b = _write(tmp_path, LIVE, name="2.json")
    assert trajectory_files([tmp_path / "run"]) == [a, b]
    assert trajectory_files([a]) == [a]


def test_script_exit_codes(tmp_path, capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_replay_exact.py"
    spec = importlib.util.spec_from_file_location("check_replay_exact", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = _write(tmp_path, LIVE, name="1.json")
    assert mod.main([str(good)]) == 0
    bad = _write(tmp_path, LIVE.replace("context of r2c1", "context of r2c1!"), name="2.json")
    assert mod.main([str(tmp_path / "run")]) == 1
    assert "1/2 identical" in capsys.readouterr().out
