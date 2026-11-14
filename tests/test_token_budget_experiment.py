"""Token-budget experiment: config wiring, Flex queue accounting and the table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from adr.agents.gpt_researcher import orchestration_env

REPO = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("run_config", ["gym_sol_pilot_d2_budget150.yaml", "gpt_researcher_bench_budget150.yaml"])
def test_drg_and_drb_run_the_same_agent_under_150k(run_config):
    """DRG and DRB differ only in the retriever: Sol (on Flex), depth 2 /
    breadth 4, the same stopping rule and the auto report reserve."""
    run = yaml.safe_load((REPO / "configs" / run_config).read_text())
    agent = yaml.safe_load((REPO / run["agent"]["config"]).read_text())
    env = orchestration_env(agent["orchestration"])
    assert env["GR_STOP_TOTAL_TOKENS"] == "150000"
    assert env["GR_REPORT_RESERVE_TOKENS"] == "auto"  # estimated from the evidence
    assert agent["depth"] == 2 and agent["breadth"] == 4 and agent["concurrency"] == 1
    # The arm is chosen by the driver; the config must not pin a policy via env.
    assert "GR_ORCHESTRATOR" not in (agent.get("env") or {})
    assert all(agent["env"][k] == "openai:gpt-6.1-sol-1gtjfzlv8i" for k in ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM"))
    assert agent["env"]["EMBEDDING"] == "openai:text-embedding-3-small-v1-651fgh"
    assert run["dataset"]["limit"] is None  # the query list comes from scripts/sol_budget.sh


def test_scripts_point_at_the_budget150_configs():
    for script in ("sol_budget.sh", "sol_budget_judge.sh"):
        text = (REPO / "scripts" / script).read_text()
        for cfg in ("configs/gym_sol_pilot_d2_budget150.yaml", "configs/gpt_researcher_bench_budget150.yaml"):
            assert cfg in text and (REPO / cfg).exists(), (script, cfg)


def _call(start, attempt_start, duration, tier="flex", inp=1000, out=100, model="gpt-6.1-sol-1gtjfzlv8i", attempts=1):
    return {"model": model, "service_tier": tier, "start": start, "attempt_start": attempt_start,
            "end": attempt_start + duration, "duration_s": duration, "input_tokens": inp,
            "output_tokens": out, "attempts": attempts}


def test_queue_intervals_count_refusals_and_excess_only_for_flex_sol_calls():
    q = _script("flex_queue_adjust")
    coef = (2.0, 0.0, 0.01)  # 2s + 10s per 1K output tokens -> 3s for 100 tokens
    calls = [
        _call(0, 0, 3.0),                       # on time
        _call(10, 70, 3.0, attempts=3),         # refused for 60s, then on time
        _call(100, 100, 33.0),                  # queued 30s after being accepted
        _call(200, 200, 500.0, tier="default"), # standard tier: never adjusted
        _call(300, 300, 500.0, model="pilot-bc-teacher", tier="default"),
    ]
    intervals, refused, queued = q.queue_intervals(calls, coef)
    assert refused == pytest.approx(60.0)
    assert queued == pytest.approx(30.0)
    assert q.union_length(intervals) == pytest.approx(90.0)


def test_overlapping_waits_are_counted_once():
    q = _script("flex_queue_adjust")
    assert q.union_length([(0, 10), (5, 12), (20, 25)]) == pytest.approx(17.0)
    assert q.union_length([]) == 0.0


def _log_record(start, capacity_wait, accepted_duration, tier="flex", inp=1000, out=100,
                model="gpt-6.1-sol-1gtjfzlv8i", capacity_retries=0, status="ok"):
    """A final_stats.llm_calls record as the fork's LLMCallLog writes it."""
    duration = capacity_wait + accepted_duration
    return {"model": model, "usage_tag": None, "tier_requested": tier, "tier_served": tier,
            "start": start, "end": start + duration, "duration_s": duration, "attempts": 1,
            "capacity_retries": capacity_retries, "capacity_wait_s": capacity_wait, "timeouts": 0,
            "input_tokens": inp, "output_tokens": out, "status": status, "error": None}


def test_call_log_records_are_read_like_the_older_jsonl_log():
    q = _script("flex_queue_adjust")
    coef = (2.0, 0.0, 0.01)
    calls = [q.from_call_log(r) for r in (
        _log_record(10, 60.0, 3.0, capacity_retries=4),  # refused for 60s, then on time
        _log_record(100, 0.0, 33.0),                     # queued 30s after being accepted
        _log_record(200, 0.0, 900.0, status="failed"),   # failed: nothing to compare
        _log_record(300, 0.0, 500.0, tier=None, model="pilot-bc-teacher"),
    )]
    assert calls[0]["attempt_start"] == 70 and calls[0]["duration_s"] == 3.0 and calls[0]["attempts"] == 5
    intervals, refused, queued = q.queue_intervals(calls, coef)
    assert refused == pytest.approx(60.0)
    assert queued == pytest.approx(30.0)
    assert q.union_length(intervals) == pytest.approx(90.0)


def _write_run(arm_dir: Path, qid: str, *, tokens: int, wall: float, quality: float, calls: list[dict],
               budget: dict | None, new_items=(20, 15), jsonl: bool = False):
    run = arm_dir / f"{qid}-s0"
    (run / "metrics").mkdir(parents=True)
    (run / "trajectories").mkdir()
    official = {"deep_research_gym": {
        "quality": {"n": 1, "average_normalized_score": quality,
                    "per_criterion": {c: {"average_rating": 6.0} for c in ("Clarity", "Breadth", "Depth", "Support")}},
        "kpr": {"n": 1, "average_support_rate": 50.0},
        "citation": {"n": 1, "average_citation_score": 70.0},
    }}
    (run / "metrics" / "summary.json").write_text(json.dumps({
        "mean_tokens": tokens, "mean_wall_s": wall, "mean_n_searches": 20, "official": official}))
    stats = {"token_stop": budget} if budget else {}
    if not jsonl:
        stats["llm_calls"] = calls
    (run / "trajectories" / f"{qid}.json").write_text(json.dumps({
        "report": "text", "final_stats": stats,
        "steps": [{"extra": {"new_item_ids": ["x"] * n}} for n in new_items]}))
    if jsonl:
        (arm_dir / "llm_calls").mkdir(exist_ok=True)
        (arm_dir / "llm_calls" / f"{qid}-s0.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))


def test_queue_adjust_and_table_end_to_end(tmp_path, capsys, monkeypatch):
    root = tmp_path / "budget"
    on_time = 2.23998987110841 - 4.4760121998408825e-06 * 1000 + 0.01822918253254161 * 100
    under = {"stop_total_tokens": 150000, "stopped": True, "ended_above_target": False}
    over = {"stop_total_tokens": 150000, "stopped": True, "ended_above_target": True}
    for qid, quality in (("1", 60.0), ("2", 62.0)):
        # none: an older run with the jsonl call log, still read.
        _write_run(root / "none", qid, tokens=149000, wall=400.0, quality=quality, budget=under,
                   calls=[_call(1000, 1000, on_time)], jsonl=True)
        # bc waited 100s on refusals; its wall time includes that wait.
        _write_run(root / "bc", qid, tokens=152000, wall=500.0, quality=quality + 3, budget=over,
                   calls=[_log_record(1000, 100.0, on_time, capacity_retries=3)])
    # A run starved of evidence is reported, and dropped on request.
    _write_run(root / "bc", "3", tokens=90000, wall=200.0, quality=40.0, budget=under,
               calls=[_log_record(1000, 0.0, on_time)], new_items=(0, 0))

    q = _script("flex_queue_adjust")
    monkeypatch.setattr(sys, "argv", ["flex_queue_adjust.py", str(root)])
    q.main()
    adjusted = json.loads((root / "queue_adjusted.json").read_text())["runs"]
    assert adjusted["none"]["1-s0"]["queue_s"] == 0.0
    assert adjusted["bc"]["1-s0"]["queue_s"] == pytest.approx(100.0, abs=0.2)
    assert adjusted["bc"]["1-s0"]["adjusted_wall_s"] == pytest.approx(400.0, abs=0.2)
    assert adjusted["bc"]["1-s0"]["n_refusals"] == 3

    t = _script("budget_table")
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["budget_table.py", "drg", "--root", str(root), "--drop-starved"])
    t.main()
    out = capsys.readouterr().out
    assert "bc: 1 run(s) starved of evidence, left out: 3" in out
    bc_row = next(line for line in out.splitlines() if line.startswith("| PILOT (BC) |"))
    cells = [c.strip() for c in bc_row.strip("|").split("|")]
    assert cells[1] == "2"                      # n, starved run dropped
    assert cells[7] == "64.0"                   # Overall
    assert cells[9] == "152.0"                  # Tokens (K)
    assert cells[10] == "400.0"                 # Latency with the Flex wait removed
    assert cells[-2:] == ["2/2", "2/2"]         # stopped by the budget, ended over it
    assert "| Overall | +3.00 |" in out
