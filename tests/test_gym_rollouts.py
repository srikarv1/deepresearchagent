"""DeepResearchGym randomized rollouts: pinned split, rollout config, reward quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from adr.datasets.loader import load_queries
from adr.datasets.splits import GYM_SPLITS_PATH, ids_for_split, load_splits, split_of
from adr.rollouts.driver import is_evaluated, plan_rollouts, run_id_for
from adr.rollouts.reward import RewardConfig, compute_reward, load_quality
from adr.runner.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from build_gym_splits import N_PINNED_TEST, POOL, build_gym_splits  # noqa: E402

GYM_ROLLOUTS = ROOT / "configs" / "rollouts_gym_random.yaml"


def test_committed_gym_split_matches_the_builder_and_mirrors_drb():
    built = build_gym_splits()
    committed = json.loads(GYM_SPLITS_PATH.read_text(encoding="utf-8"))
    assert committed == built, "run python scripts/build_gym_splits.py"
    assert committed["counts"] == {"train": 30, "val": 10, "test": 10}
    assert committed["n"] == POOL == 50 and committed["seed"] == 17
    first = [q.id for q in load_queries("deep_research_gym", limit=N_PINNED_TEST)]
    assert committed["pinned_test"] == first
    assert set(first) <= set(committed["test"])
    parts = load_splits("deep_research_gym")
    all_ids = parts["train"] + parts["val"] + parts["test"]
    assert len(all_ids) == len(set(all_ids)) == 50


def test_gym_split_filters_queries_and_tags_metadata():
    train = load_queries("deep_research_gym", split="train")
    assert len(train) == 30
    assert all(q.metadata.get("split") == "train" for q in train)
    assert ids_for_split("deep_research_gym", "test") == set(load_splits("deep_research_gym")["test"])
    assert split_of("deep_research_gym", "879779") == "test"
    # Queries outside the 50-query pool are not assigned a split.
    assert split_of("deep_research_gym", "does-not-exist") is None


def test_gym_rollout_config_plans_over_the_train_split(tmp_path: Path):
    specs, cfg = plan_rollouts(config=GYM_ROLLOUTS, out=tmp_path, n_seeds=2, seed_base=0)
    assert cfg["dataset"]["name"] == "deep_research_gym"
    assert cfg["agent"]["config"].endswith("gpt_researcher_gym.yaml")
    train = set(load_splits("deep_research_gym")["train"])
    assert {s.query_id for s in specs} == train
    assert len(specs) == 60
    assert all(s.run_id == run_id_for(s.query_id, s.seed) for s in specs)


def test_load_quality_reads_gym_per_query_and_reward_scales_it(tmp_path: Path):
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "metrics" / "summary.json").write_text(json.dumps({
        "official": {"deep_research_gym": {"quality": {
            "average_normalized_score": 55.3,
            "per_query_normalized": {"879779": 61.5, "923549": 49.1},
        }}}
    }))
    assert load_quality(run, "879779") == pytest.approx(61.5)
    assert load_quality(run, "000000") is None
    cfg = RewardConfig(token_budget=1, latency_budget_s=1.0, lambda_tok=0, lambda_lat=0, shaping=False)
    r = compute_reward(quality=load_quality(run, "879779"), goal=0, tokens=0, latency_s=0, rounds=[], cfg=cfg)
    assert r.quality == pytest.approx(0.615)


def test_is_evaluated_is_bench_aware(tmp_path: Path):
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    assert is_evaluated(run, "deep_research_gym") is False
    (run / "metrics" / "summary.json").write_text(json.dumps({
        "official": {"deep_research_gym": {"official": True, "quality": {"n": 1}}}
    }))
    assert is_evaluated(run, "deep_research_gym") is True
    assert is_evaluated(run, "deep_research_bench") is False
    # A judge that ran but produced no scores must not count as evaluated.
    (run / "metrics" / "summary.json").write_text(json.dumps({
        "official": {"deep_research_gym": {"official": False, "reason": "No Gym metric produced scores"}}
    }))
    assert is_evaluated(run, "deep_research_gym") is False
    # DRB keeps its per-query RACE copy as evidence.
    (run / "metrics" / "race_raw_results.jsonl").write_text("")
    assert is_evaluated(run) is True
