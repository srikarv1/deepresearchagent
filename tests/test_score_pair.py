"""Scoring a question/report pair the harness did not produce."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.datasets.loader import load_queries
from adr.eval.compare import compare_summaries
from adr.eval.importers import (
    resolve_question,
    trajectories_from_drb_jsonl,
    trajectory_from_pair,
    write_trajectories,
)
from adr.runner.experiment import evaluate_run_dir


def _drb_prompt(query_id: str) -> str:
    return load_queries("deep_research_bench", query_ids=[query_id])[0].text


REPORT = "# R\n\nBody citing https://example.com/a and https://example.com/b twice https://example.com/a\n"


def test_pair_becomes_scoreable_trajectory():
    traj = trajectory_from_pair(
        question="why is there a chip shortage", report=REPORT, query_id="51"
    )
    assert traj.query.id == "51"
    assert traj.query.dataset == "deep_research_bench"
    assert traj.report.article == REPORT
    # URLs are deduped in order so citation counts are not inflated by repeats.
    assert traj.report.citations == ["https://example.com/a", "https://example.com/b"]


def test_resolve_question_from_benchmark_id():
    query_id, question = resolve_question(dataset="deep_research_bench", query_id="51")
    assert query_id == "51"
    assert question.startswith("From 2020 to 2050, how many elderly people")


def test_resolve_question_matches_text_back_to_id():
    query_id, question = resolve_question(
        dataset="deep_research_bench", question=_drb_prompt("51")
    )
    assert query_id == "51"
    assert question == _drb_prompt("51")


def test_resolve_offbenchmark_question_gets_a_slug():
    query_id, question = resolve_question(
        dataset="deep_research_bench", question="Does creatine help with cognition?"
    )
    assert query_id == "does-creatine-help-with-cognition"
    assert question.startswith("Does creatine")


def test_unknown_id_without_question_is_an_error():
    with pytest.raises(ValueError, match="not in deep_research_bench"):
        resolve_question(dataset="deep_research_bench", query_id="not-a-real-id")


def test_resolve_requires_something():
    with pytest.raises(ValueError):
        resolve_question(dataset="deep_research_bench")


def test_local_metrics_on_an_imported_pair(tmp_path: Path):
    traj = trajectory_from_pair(
        question="why is there a chip shortage", report=REPORT, query_id="51"
    )
    write_trajectories(tmp_path, [traj])
    summary = evaluate_run_dir(tmp_path, official_benches=[])

    assert summary["n_queries"] == 1
    assert summary["n_with_report"] == 1
    assert summary["mean_n_citations"] == 2
    assert summary["mean_article_chars"] == len(REPORT)
    # No agent ran, so cost is genuinely zero rather than unmeasured.
    assert summary["mean_tokens"] == 0
    assert summary["scores"] == {}
    assert (tmp_path / "reports" / "51.md").exists()


def test_roundtrip_through_drb_jsonl(tmp_path: Path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"id": 51, "prompt": "p one", "article": "a one"})
        + "\n"
        + json.dumps({"id": 52, "prompt": "p two", "article": ""})
        + "\n"
    )
    trajectories = trajectories_from_drb_jsonl(path)
    assert [t.query.id for t in trajectories] == ["51", "52"]
    assert trajectories[0].query.dataset == "deep_research_bench"
    assert trajectories[1].report.article == ""


def test_compare_separates_quality_from_cost(tmp_path: Path):
    baseline = tmp_path / "a.json"
    candidate = tmp_path / "b.json"
    baseline.write_text(json.dumps({
        "n_queries": 2,
        "mean_tokens": 1000,
        "mean_wall_s": 20.0,
        "official": {"deep_research_bench": {"race": {"scores": {"overall_score": 0.55}}}},
    }))
    candidate.write_text(json.dumps({
        "n_queries": 2,
        "mean_tokens": 700,
        "mean_wall_s": 15.0,
        "official": {"deep_research_bench": {"race": {"scores": {"overall_score": 0.585}}}},
    }))

    out = compare_summaries(baseline, candidate)
    assert out["quality_deltas"]["race_overall_score"] == pytest.approx(0.035)
    assert out["cost_deltas"]["mean_tokens"] == pytest.approx(-300)
    assert out["percent_change"]["mean_tokens"] == pytest.approx(-30.0)
    assert out["percent_change"]["mean_wall_s"] == pytest.approx(-25.0)
    assert "race_overall_score" not in out["cost_deltas"]
