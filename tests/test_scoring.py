"""Aggregation must reproduce the upstream formulas exactly."""

from __future__ import annotations

from pathlib import Path

import pytest

from adr.eval.repos import find_deep_research_bench
from adr.eval.scoring import headline_scores, parse_key_value_report

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


def test_headline_scores_flattens_both_benches():
    flat = headline_scores(
        {
            "deep_research_bench": {
                "race": {"scores": {"overall_score": 0.42, "insight": 0.4}},
                "fact": {"scores": {"valid_rate": 0.87}},
            },
            "browsecomp_plus": {"accuracy": 0.31, "recall": 0.58},
        }
    )
    assert flat["race_overall_score"] == pytest.approx(0.42)
    assert flat["race_insight"] == pytest.approx(0.4)
    assert flat["fact_valid_rate"] == pytest.approx(0.87)
    assert flat["bcp_accuracy"] == pytest.approx(0.31)
    assert flat["bcp_recall"] == pytest.approx(0.58)


def test_headline_scores_tolerates_empty():
    assert headline_scores({}) == {}
    assert headline_scores({"deep_research_bench": {"race": None, "fact": None}}) == {}


# ── Checks against real upstream artifacts, when the checkouts are present ────

_drb = find_deep_research_bench()


@pytest.mark.skipif(not _drb.ok, reason="DeepResearch Bench checkout not available")
def test_race_parser_on_real_upstream_results():
    published = sorted(_drb.path.glob("results/race/*/race_result.txt"))
    if not published:
        pytest.skip("no committed RACE results in this checkout")
    for path in published:
        scores = parse_key_value_report(path)
        assert "overall_score" in scores
        assert 0.0 <= scores["overall_score"] <= 1.0
