"""R(tau) (PILOT Eq. 14 / 17) and the RACE per-query plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.eval.deep_research_bench import read_race_raw_results
from adr.rollouts.reward import (
    RewardConfig,
    compute_reward,
    load_kpr_goal,
    load_quality,
    load_run_budget,
    normalize_by_query,
    reward_for_run,
    shaping_from_rounds,
)


def _rounds(pairs):
    return [{"decision": {"meta": {"phi_prev": a, "phi_after": b}}} for a, b in pairs]


def test_compute_reward_matches_equation():
    cfg = RewardConfig(alpha_q=1.0, alpha_g=0.5, lambda_tok=0.2, lambda_lat=0.1, q_min=0.0,
                       gamma=1.0, token_budget=100_000, latency_budget_s=1000.0)
    r = compute_reward(quality=0.6, goal=0.4, tokens=50_000, latency_s=250.0,
                       rounds=_rounds([(0.0, 0.3), (0.3, 0.5)]), cfg=cfg)
    assert r.quality_floored == pytest.approx(0.6)
    assert r.tokens_norm == pytest.approx(0.5) and r.latency_norm == pytest.approx(0.25)
    assert r.shaping == pytest.approx(0.5)  # telescopes to Phi_T - Phi_0
    assert r.total == pytest.approx(0.6 + 0.2 - 0.1 - 0.025 + 0.5)
    assert r.has_quality and r.n_rounds == 2


def test_quality_floor_removes_cheap_bad_reports():
    cfg = RewardConfig(q_min=0.3, lambda_tok=1.0, token_budget=100, latency_budget_s=1.0, shaping=False)
    below = compute_reward(quality=0.25, goal=0.0, tokens=1, latency_s=0.0, rounds=[], cfg=cfg)
    above = compute_reward(quality=0.5, goal=0.0, tokens=1, latency_s=0.0, rounds=[], cfg=cfg)
    assert below.quality_floored == 0.0 and below.total == pytest.approx(-0.01)
    assert above.quality_floored == pytest.approx(0.2)


def test_quality_accepts_0_100_scale_and_missing():
    cfg = RewardConfig(token_budget=1, latency_budget_s=1.0, lambda_tok=0, lambda_lat=0, shaping=False)
    assert compute_reward(quality=45.0, goal=0, tokens=0, latency_s=0, rounds=[], cfg=cfg).quality == pytest.approx(0.45)
    r = compute_reward(quality=None, goal=0, tokens=0, latency_s=0, rounds=[], cfg=cfg)
    assert r.has_quality is False and r.quality_floored == 0.0


def test_shaping_ignores_rounds_without_phi():
    total, n = shaping_from_rounds(_rounds([(0.1, 0.4)]) + [{"decision": {"meta": {}}}], gamma=0.9)
    assert n == 1 and total == pytest.approx(0.9 * 0.4 - 0.1)


def test_read_race_raw_results_skips_errors(tmp_path: Path):
    p = tmp_path / "raw_results.jsonl"
    p.write_text(
        json.dumps({"id": 51, "overall_score": 0.42, "insight": 0.5}) + "\n"
        + json.dumps({"id": 52, "error": "LLM failed"}) + "\n"
        + "not json\n"
    )
    rows = read_race_raw_results(p)
    assert rows == {"51": {"overall_score": 0.42, "insight": 0.5}}
    assert read_race_raw_results(tmp_path / "missing.jsonl") == {}


def _run_dir(tmp_path: Path, qid: str = "51", quality: float | None = 0.6) -> Path:
    run = tmp_path / f"{qid}-s0"
    (run / "metrics").mkdir(parents=True)
    (run / "config.yaml").write_text("budget:\n  max_tokens: 200000\n  max_latency_s: 600.0\n")
    if quality is not None:
        (run / "metrics" / "race_raw_results.jsonl").write_text(json.dumps({"id": int(qid), "overall_score": quality}) + "\n")
    return run


def test_load_quality_from_run_copy_then_summary(tmp_path: Path):
    run = _run_dir(tmp_path, quality=0.61)
    assert load_quality(run, "51") == pytest.approx(0.61)
    assert load_quality(run, "99") is None

    run2 = _run_dir(tmp_path, qid="52", quality=None)
    (run2 / "metrics" / "summary.json").write_text(json.dumps(
        {"official": {"deep_research_bench": {"race": {"per_query": {"52": {"overall_score": 0.33}}}}}}
    ))
    assert load_quality(run2, "52") == pytest.approx(0.33)


def test_reward_for_run_uses_run_budget_and_harness_cost(tmp_path: Path):
    run = _run_dir(tmp_path, quality=0.5)
    assert load_run_budget(run) == (200000, 600.0)
    harness = {"final_stats": {"usage": {"total_tokens": 100000}, "wall_s": 300.0}}
    raw = {"rounds": _rounds([(0.0, 0.2)])}
    r = reward_for_run(run, "51", cfg=RewardConfig(lambda_tok=0.1, lambda_lat=0.1),
                       raw_trajectory=raw, harness_trajectory=harness)
    assert r.tokens == 100000 and r.tokens_norm == pytest.approx(0.5)
    assert r.latency_norm == pytest.approx(0.5)
    assert r.total == pytest.approx(0.5 - 0.05 - 0.05 + 0.2)

    # Explicit budgets override what the rollout ran under.
    r2 = reward_for_run(run, "51", cfg=RewardConfig(token_budget=50000, latency_budget_s=150.0, shaping=False),
                        raw_trajectory=raw, harness_trajectory=harness)
    assert r2.tokens_norm == pytest.approx(2.0) and r2.latency_norm == pytest.approx(2.0)


def test_quality_first_cost_is_hinge_plus_tiebreak_and_shaping_reported_not_added():
    cfg = RewardConfig(cost_mode="hinge", cost_tiebreak=0.1, shaping_in_return=False,
                       lambda_tok=1.0, lambda_lat=1.0, token_budget=100_000, latency_budget_s=1000.0)
    under = compute_reward(quality=0.6, goal=0.0, tokens=50_000, latency_s=500.0,
                           rounds=_rounds([(0.0, 0.7)]), cfg=cfg)
    over = compute_reward(quality=0.6, goal=0.0, tokens=150_000, latency_s=500.0, rounds=[], cfg=cfg)
    assert under.shaping == pytest.approx(0.7)
    assert under.cost_term == pytest.approx(0.1 * 0.5)               # only the tie-break
    assert under.total == pytest.approx(0.6 - 0.05)                  # shaping not in the total
    assert over.cost_term == pytest.approx(1.0 * 0.5 + 0.1 * 1.5)    # 50% over budget


def test_query_z_compares_rollouts_within_a_query():
    cfg = RewardConfig(quality_norm="query_z", alpha_q=1.0, alpha_g=0.5, cost_mode="hinge",
                       shaping_in_return=False, token_budget=100, latency_budget_s=1.0)

    def score(q, g):
        return compute_reward(quality=q, goal=g, tokens=10, latency_s=0.1, rounds=[], cfg=cfg)

    # Query A is easy (all high), query B hard (all low): absolute quality would
    # rank every A run above every B run; z-scores rank within each query.
    a_hi, a_lo, b_hi, b_lo = score(0.62, 0.9), score(0.58, 0.5), score(0.45, 0.6), score(0.41, 0.2)
    assert a_lo.total == 0.0  # nothing added before normalization
    normalize_by_query([("A", a_hi), ("A", a_lo), ("B", b_hi), ("B", b_lo)], cfg)
    assert a_hi.quality_z == pytest.approx(1.0) and b_lo.quality_z == pytest.approx(-1.0)
    assert b_hi.total == pytest.approx(a_hi.total) and b_hi.total > a_lo.total
    assert a_hi.total == pytest.approx(1.0 + 0.5 * 1.0)


def test_query_z_is_a_noop_for_the_original_reward():
    cfg = RewardConfig(token_budget=100, latency_budget_s=1.0, shaping=False)
    r = compute_reward(quality=0.6, goal=0.0, tokens=0, latency_s=0.0, rounds=[], cfg=cfg)
    before = r.total
    normalize_by_query([("A", r)], cfg)
    assert r.total == before and r.quality_z is None


def test_load_kpr_goal_from_gym_summary(tmp_path: Path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "summary.json").write_text(json.dumps({"official": {"deep_research_gym": {
        "kpr": {"average_support_rate": 40.0, "per_query": {"7": {"support_rate": 75.0}}}}}}))
    assert load_kpr_goal(tmp_path, "7") == pytest.approx(0.75)
    assert load_kpr_goal(tmp_path, "8") == pytest.approx(0.40)
    assert load_kpr_goal(tmp_path / "missing", "7") == 0.0


def test_unknown_reward_mode_raises():
    with pytest.raises(ValueError):
        RewardConfig(cost_mode="quadratic")
