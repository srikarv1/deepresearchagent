"""Aggregation must reproduce the upstream formulas exactly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.eval.repos import find_deep_research_bench, find_deep_research_gym
from adr.eval.scoring import (
    aggregate_gym_citation,
    aggregate_gym_kpr,
    aggregate_gym_quality,
    headline_scores,
    parse_key_value_report,
)

RACE_TEXT = """Comprehensiveness: 0.4110
Insight: 0.4051
Instruction Following: 0.4621
Readability: 0.4172
Overall Score: 0.4218
"""

FACT_TEXT = """total_citations: 28.07
total_valid_citations: 24.51
valid_rate: 0.8731742073387959
"""


def test_parse_race_and_fact_files(tmp_path: Path):
    race = tmp_path / "race_result.txt"
    race.write_text(RACE_TEXT)
    parsed = parse_key_value_report(race)
    assert parsed["overall_score"] == pytest.approx(0.4218)
    assert parsed["instruction_following"] == pytest.approx(0.4621)

    fact = tmp_path / "fact_result.txt"
    fact.write_text(FACT_TEXT)
    assert parse_key_value_report(fact)["valid_rate"] == pytest.approx(0.8731742073387959)


def test_parse_missing_file_is_empty(tmp_path: Path):
    assert parse_key_value_report(tmp_path / "nope.txt") == {}


def test_quality_normalization_matches_upstream_formula():
    # sum=36 over 6 criteria -> 36 / 60 * 100 = 60
    results = {
        "q1": {
            "scores": {
                "Clarity": [7, "j"],
                "Depth": [6, "j"],
                "Balance": [5, "j"],
                "Breadth": [8, "j"],
                "Support": [4, "j"],
                "Insightfulness": [6, "j"],
            }
        }
    }
    agg = aggregate_gym_quality(results)
    assert agg["average_normalized_score"] == pytest.approx(60.0)
    assert agg["per_criterion"]["Breadth"]["average_rating"] == pytest.approx(8.0)
    assert agg["per_criterion"]["Breadth"]["normalized_average"] == pytest.approx(80.0)
    assert agg["n"] == 1


def test_quality_accepts_dict_shaped_scores():
    results = {"q1": {"scores": {"Clarity": {"rating": 10, "justification": "j"}}}}
    assert aggregate_gym_quality(results)["average_normalized_score"] == pytest.approx(100.0)


def test_kpr_rates_are_per_query_means():
    results = {
        "a": {"labels": {"1": ["Supported", "j"], "2": ["Omitted", "j"]}},
        "b": {"labels": {"1": ["Supported", "j"], "2": ["Contradicted", "j"],
                          "3": ["Supported", "j"], "4": ["Supported", "j"]}},
    }
    agg = aggregate_gym_kpr(results)
    # query a: 50%, query b: 75% -> mean 62.5
    assert agg["average_support_rate"] == pytest.approx(62.5)
    assert agg["average_contradicted_rate"] == pytest.approx(12.5)
    assert agg["per_query"]["b"]["n_key_points"] == 4


def test_citation_scaled_to_percent():
    results = {"a": {"score": 0.5}, "b": {"score": 1.0}}
    assert aggregate_gym_citation(results)["average_citation_score"] == pytest.approx(75.0)


def test_citation_unpacks_upstream_results_and_average(tmp_path: Path):
    from adr.eval.deep_research_gym import _citation

    # Upstream eval_citation_async returns (per_query_results, average_score).
    (tmp_path / "eval_citation_async.py").write_text(
        "async def evaluate_folder_async(report_dir, model):\n"
        "    return {'a': {'score': 0.5}, 'b': {'score': 1.0}}, 0.75\n",
        encoding="utf-8",
    )
    out = _citation(tmp_path, tmp_path, tmp_path, "judge", None)
    assert out["n"] == 2
    assert out["average_citation_score"] == pytest.approx(75.0)
    saved = json.loads((tmp_path / "faithfullness_judge.json").read_text(encoding="utf-8"))
    assert saved == {"a": {"score": 0.5}, "b": {"score": 1.0}}


def test_headline_scores_flattens_both_benches():
    flat = headline_scores(
        {
            "deep_research_bench": {
                "race": {"scores": {"overall_score": 0.42, "insight": 0.4}},
                "fact": {"scores": {"valid_rate": 0.87}},
            },
            "deep_research_gym": {
                "quality": {"average_normalized_score": 60.0,
                            "per_criterion": {"Support": {"average_rating": 4.0}}},
                "kpr": {"average_support_rate": 64.5},
                "citation": {"average_citation_score": 71.2},
            },
        }
    )
    assert flat["race_overall_score"] == pytest.approx(0.42)
    assert flat["fact_valid_rate"] == pytest.approx(0.87)
    assert flat["gym_quality"] == pytest.approx(60.0)
    assert flat["gym_quality_support"] == pytest.approx(4.0)
    assert flat["gym_average_support_rate"] == pytest.approx(64.5)
    assert flat["gym_citation_score"] == pytest.approx(71.2)


def test_headline_scores_tolerates_empty():
    assert headline_scores({}) == {}
    assert headline_scores({"deep_research_gym": {"quality": None, "kpr": None}}) == {}


# ── Checks against real upstream artifacts, when the checkouts are present ────

_gym = find_deep_research_gym()
_drb = find_deep_research_bench()


@pytest.mark.skipif(not _gym.ok, reason="DeepResearchGym checkout not available")
def test_kpr_aggregator_on_real_upstream_results():
    """Reproduce upstream's own numbers from its committed result files."""
    published = sorted(_gym.path.glob("results/**/evaluation_results_kpr_*.json"))
    if not published:
        pytest.skip("no committed KPR results in this checkout")
    for path in published:
        agg = aggregate_gym_kpr(json.loads(path.read_text(encoding="utf-8")))
        assert agg["n"] > 0
        total = (
            agg["average_support_rate"]
            + agg["average_omitted_rate"]
            + agg["average_contradicted_rate"]
        )
        assert total == pytest.approx(100.0, abs=0.01), path


@pytest.mark.skipif(not _drb.ok, reason="DeepResearch Bench checkout not available")
def test_race_parser_on_real_upstream_results():
    published = sorted(_drb.path.glob("results/race/*/race_result.txt"))
    if not published:
        pytest.skip("no committed RACE results in this checkout")
    for path in published:
        scores = parse_key_value_report(path)
        assert "overall_score" in scores
        assert 0.0 <= scores["overall_score"] <= 1.0
