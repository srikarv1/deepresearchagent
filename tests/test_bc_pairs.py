"""bc_pairs: replaying RandomizedPolicy rounds into prompt/completion rows.

Uses a synthetic rollout corpus laid out exactly like `adr rollouts` writes
it. Needs the fork checkout with gpt_researcher/orchestration/serialize.py
(RandomizedPolicy branch); skipped otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.rollouts.fork import fork_available, load_fork_module

pytestmark = pytest.mark.skipif(not fork_available(), reason="gpt-researcher fork with orchestration/serialize.py not found")

from adr.rollouts.bc_pairs import (  # noqa: E402
    build_bc_pairs,
    discover_run_dirs,
    load_rollout,
    rows_for_trajectory,
    select_top,
)
from adr.rollouts.reward import RewardConfig  # noqa: E402

FEATS = ["rho", "nov", "red", "cov", "tok", "depth", "age", "src"]


def _raw_trajectory(query: str, *, policy: str = "random", terminate_last: bool = False) -> dict:
    """Two rounds. Round 1: A, B new; keep A. Round 2: C new; A retained; keep A, C."""
    ev = {
        "A": {"item_id": "A", "content": "alpha text " * 30, "source_url": "https://a.gov/x", "tree_depth": 1, "retrieval_round": 1},
        "B": {"item_id": "B", "content": "beta text " * 30, "source_url": "https://b.com/y", "tree_depth": 1, "retrieval_round": 1},
        "C": {"item_id": "C", "content": "gamma text " * 30, "source_url": "https://c.org/z", "tree_depth": 2, "retrieval_round": 2},
    }
    params = {"keep_ratio": 0.5, "temperature": 0.1}
    r1_meta = {
        "policy": policy, "params": params, "tokens_used": 1000, "token_budget": 50000,
        "feature_names": FEATS,
        "features": {"A": [0.9, 1.0, 0.0, 0.5, 75, 1, 0, 2], "B": [0.2, 1.0, 0.0, 0.0, 75, 1, 0, 0]},
        "sources": {"A": "a.gov", "B": "b.com"},
        "frontier_stats": {"n1": {"gap": 0.7, "n_items": 2.0}, "n2": {"gap": 0.4, "n_items": 0.0}},
        "phi_prev": 0.0, "phi_after": 0.4, "terminate_reason": "",
    }
    r2_meta = {
        "policy": policy, "params": params, "tokens_used": 3000, "token_budget": 50000,
        "feature_names": FEATS,
        "features": {"A": [0.9, 1.0, 0.0, 0.5, 75, 1, 1, 2], "C": [0.7, 0.3, 0.0, 0.5, 75, 2, 0, 1]},
        "sources": {"A": "a.gov", "C": "c.org"},
        "frontier_stats": {"n3": {"gap": 0.2, "n_items": 0.0}},
        "phi_prev": 0.4, "phi_after": 0.6, "terminate_reason": "bernoulli" if terminate_last else "",
    }
    return {
        "query": query, "query_id": "abc", "subquestions": ["s1", "s2"], "num_rounds": 2,
        "rounds": [
            {"round_id": 1, "new_item_ids": ["A", "B"], "retained_ids": ["A"],
             "decision": {"type": "continue", "kept_item_ids": ["A"], "pruned_item_ids": ["B"],
                          "branch_allocation": {"n1": 0.75, "n2": 0.25}, "policy": policy, "meta": r1_meta},
             "frontier": [{"node_id": "n1", "subquery": "branch one", "parent_subquery": "", "status": "open"},
                          {"node_id": "n2", "subquery": "branch two", "parent_subquery": "", "status": "open"}]},
            {"round_id": 2, "new_item_ids": ["C"], "retained_ids": ["A", "C"],
             "decision": {"type": "terminate", "kept_item_ids": ["A", "C"], "pruned_item_ids": [],
                          "branch_allocation": {"n3": 1.0}, "policy": policy, "meta": r2_meta},
             "frontier": [{"node_id": "n3", "subquery": "branch three", "parent_subquery": "", "status": "completed"}]},
        ],
        "evidence": ev, "report": "report", "total_tokens": 4000, "total_latency": 120.0,
    }


def _write_run(corpus: Path, qid: str, seed: int, *, quality: float | None, tokens: int = 40000,
               policy: str = "random", report: bool = True) -> Path:
    run = corpus / f"{qid}-s{seed}"
    (run / "trajectories").mkdir(parents=True)
    (run / "gpt_researcher").mkdir()
    (run / "metrics").mkdir()
    (run / "config.yaml").write_text("budget:\n  max_tokens: 100000\n  max_latency_s: 600.0\n")
    harness = {
        "query": {"id": qid, "text": f"question {qid}", "dataset": "deep_research_bench"},
        "report": {"article": "# report"} if report else None,
        "error": None if report else "no_report",
        "final_stats": {"usage": {"total_tokens": tokens}, "wall_s": 120.0},
    }
    (run / "trajectories" / f"{qid}.json").write_text(json.dumps(harness))
    (run / "gpt_researcher" / f"{qid}.json").write_text(json.dumps(_raw_trajectory(f"question {qid}", policy=policy)))
    if quality is not None:
        (run / "metrics" / "race_raw_results.jsonl").write_text(json.dumps({"id": int(qid), "overall_score": quality}) + "\n")
    return run


def test_discover_and_load(tmp_path: Path):
    corpus = tmp_path / "corpus"
    _write_run(corpus, "51", 0, quality=0.5)
    _write_run(corpus, "51", 1, quality=0.6)
    dirs = discover_run_dirs([corpus])
    assert [d.name for d in dirs] == ["51-s0", "51-s1"]
    assert discover_run_dirs([dirs[0]]) == [dirs[0]]
    recs = load_rollout(dirs[0])
    assert len(recs) == 1 and recs[0].query_id == "51" and recs[0].policy == "random"
    assert recs[0].params == {"keep_ratio": 0.5, "temperature": 0.1}


def test_rows_replay_state_and_action(tmp_path: Path):
    corpus = tmp_path / "corpus"
    run = _write_run(corpus, "51", 0, quality=0.5)
    rec = load_rollout(run)[0]
    rows = rows_for_trajectory(rec, final_round_terminate=False)
    assert len(rows) == 2

    r1, r2 = rows
    assert r1["round_id"] == 1 and r2["round_id"] == 2
    # Round 1 state: both items new, features rendered at 2dp / ints, sources, gaps.
    assert "Root question: question 51" in r1["prompt"]
    assert "Round 1 | depth 1 | tokens used 1000 | context budget 50000 | open branches 2" in r1["prompt"]
    assert "A | 0.90 1.00 0.00 0.50 75 1 0 2 | new | a.gov | alpha text" in r1["prompt"]
    assert "B | 0.20 1.00 0.00 0.00 75 1 0 0 | new | b.com | beta text" in r1["prompt"]
    assert "n1 | 0.70 | branch one" in r1["prompt"] and "n2 | 0.40 | branch two" in r1["prompt"]
    assert r1["completion"] == "KEEP: A\nALLOC: n1=0.75 n2=0.25\nDECISION: CONTINUE"
    # Round 2: A retained (listed first), C new, depth from the new item.
    assert "Round 2 | depth 2" in r2["prompt"]
    a_line = r2["prompt"].index("  A | ")
    c_line = r2["prompt"].index("  C | ")
    assert a_line < c_line and "| retained | a.gov |" in r2["prompt"]
    assert "B |" not in r2["prompt"]  # pruned last round, not in this round's pool
    # Policy said continue (no terminate_reason) even though the run ended.
    assert r2["completion"] == "KEEP: ALL\nALLOC: n3=1.00\nDECISION: CONTINUE"
    assert r1["messages"][0]["role"] == "system" and r1["messages"][2]["content"] == r1["completion"]


def test_final_round_relabelled_terminate_by_default(tmp_path: Path):
    run = _write_run(tmp_path / "c", "51", 0, quality=0.5)
    rows = rows_for_trajectory(load_rollout(run)[0])
    assert rows[0]["completion"].endswith("DECISION: CONTINUE")
    assert rows[1]["completion"].endswith("DECISION: TERMINATE")
    assert rows[1]["meta"]["terminate"] is True


def test_prompt_is_order_independent():
    ser = load_fork_module("serialize")
    items = [
        ser.StateItem("B", [0.2, 1, 0, 0, 75, 1, 0, 0], True, "b"),
        ser.StateItem("A", [0.9, 1, 0, 0.5, 75, 1, 1, 2], False, "a"),
    ]
    fr = [ser.StateFrontier("n2", "two", 0.4), ser.StateFrontier("n1", "one", 0.7)]
    kw = dict(root_query="q", subquestions=[], round_id=2, tree_depth=1, tokens_used=0, token_budget=None, feature_names=FEATS)
    p1 = ser.serialize_state(items=items, frontier=fr, **kw)
    p2 = ser.serialize_state(items=list(reversed(items)), frontier=list(reversed(fr)), **kw)
    assert p1 == p2
    assert p1.index("  A | ") < p1.index("  B | ")  # retained before new
    assert p1.index("n1 |") < p1.index("n2 |")
    act = ser.serialize_action(kept_ids=["B", "A"], pool_ids=["B", "A", "C"], branch_allocation={"n2": 1, "n1": 3}, terminate=False)
    assert act == "KEEP: A B\nALLOC: n1=0.75 n2=0.25\nDECISION: CONTINUE"


def test_select_top_keeps_min_per_query(tmp_path: Path):
    corpus = tmp_path / "corpus"
    recs = []
    for qid, seeds_q in (("51", [0.9, 0.8, 0.1]), ("52", [0.2, 0.15, 0.1])):
        for s, q in enumerate(seeds_q):
            recs += load_rollout(_write_run(corpus, qid, s, quality=q))
    from adr.rollouts.reward import reward_for_run

    for r in recs:
        r.reward = reward_for_run(r.run_dir, r.query_id, cfg=RewardConfig(shaping=False), raw_trajectory=r.raw, harness_trajectory=r.harness)
    chosen, cut = select_top(recs, top_frac=0.34, min_per_query=1)  # top 2 of 6 are both query 51
    ids = {r.run_id for r in chosen}
    assert {"51-s0", "51-s1"} <= ids and "52-s0" in ids  # 52 kept via min_per_query
    assert len(ids) == 3
    chosen2, _ = select_top(recs, top_frac=1.0, min_per_query=1, max_per_query=1)
    assert {r.run_id for r in chosen2} == {"51-s0", "52-s0"}


def test_build_bc_pairs_end_to_end(tmp_path: Path):
    corpus = tmp_path / "corpus"
    _write_run(corpus, "51", 0, quality=0.7, tokens=30000)
    _write_run(corpus, "51", 1, quality=0.3, tokens=90000)
    _write_run(corpus, "52", 0, quality=None)                    # unscored -> dropped
    _write_run(corpus, "53", 0, quality=0.9, report=False)       # failed run -> dropped
    _write_run(corpus, "54", 0, quality=0.9, policy="topk")      # baseline -> dropped
    out = tmp_path / "bc" / "bc_pairs.jsonl"
    stats = build_bc_pairs([corpus], out, top_frac=0.5, min_per_query=1)

    assert stats.n_run_dirs == 5 and stats.n_loaded == 5
    assert stats.n_dropped == {"no_report": 1, "policy!=random": 1, "no_quality_score": 1}
    assert stats.n_selected == 1 and stats.n_queries == 1
    assert stats.n_rows == 2  # two rounds of 51-s0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {r["run_id"] for r in rows} == {"51-s0"}
    assert rows[0]["reward"] == pytest.approx(0.7 - 0.1 * 0.3 - 0.1 * 0.2 + 0.6)  # Q - tok - lat + (Phi_T - Phi_0)
    rewards = [json.loads(l) for l in out.with_name("bc_pairs_rewards.jsonl").read_text().splitlines()]
    assert {r["run_id"]: r["selected"] for r in rewards} == {"51-s0": True, "51-s1": False}
    stats_json = json.loads(out.with_name("bc_pairs_stats.json").read_text())
    assert stats_json["n_rows"] == 2 and stats_json["reward_config"]["lambda_tok"] == 0.1
