import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("budget_table", Path(__file__).parents[1] / "scripts" / "budget_table.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def call(tag, tokens):
    return {"usage_tag": tag, "input_tokens": tokens, "output_tokens": 0}


CALLS = [call("deep_research_plan", 10)] + [
    c for _ in range(4) for c in (call("subquery_generation", 5), call("deep_process_results", 15))
] + [call("report_generation", 30)]


def test_replay_matches_the_run_at_a_large_budget():
    assert bt.replay_tokens(CALLS, 10_000) == 120


def test_replay_stops_before_the_subquery_that_would_not_fit():
    # plan 10 + 2 sub-queries of 20 = 50; a third needs 50 + 20 + report 30 = 100 > 95
    assert bt.replay_tokens(CALLS, 95) == 80


def test_replay_always_runs_one_subquery():
    assert bt.replay_tokens(CALLS, 1) == 60


def test_model_seconds_counts_overlapping_calls_once_and_drops_capacity_wait():
    calls = [{"start": 0, "end": 30, "capacity_wait_s": 0}, {"start": 10, "end": 40, "capacity_wait_s": 0},
             {"start": 100, "end": 160, "capacity_wait_s": 20}]
    assert bt.model_seconds(calls) == 40 + 40
