"""BrowseComp-Plus harness judge: official grader prompt + evidence recall."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.core.types import Query, Report, Trajectory
from adr.eval.browsecomp_plus import (
    aggregate,
    evidence_docids,
    format_query,
    parse_grade,
    recall_for,
    run_browsecomp_plus,
)
from adr.eval.browsecomp_plus_prompts import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE_NO_GET_DOCUMENT,
)
from adr.eval.scoring import headline_scores
from adr.llm.mock import MockLLM


def _traj(qid: str, article: str, *, answer: str, evidence: list[str], retrieved: list[str]) -> Trajectory:
    t = Trajectory(
        query=Query(
            id=qid,
            text=f"question {qid}",
            dataset="browsecomp_plus",
            language="en",
            metadata={
                "answer": answer,
                "evidence_docs": [{"docid": d, "url": f"http://x/{d}"} for d in evidence],
            },
        ),
        report=Report(article=article, citations=[]),
    )
    t.final_stats = {"retrieved_docids": retrieved, "usage": {"n_search_calls": 2, "n_fetch_calls": 0}}
    return t


def test_parse_grade_accepts_bold_and_plain():
    g = parse_grade("extracted_final_answer: Isar\ncorrect: yes\nconfidence: 90")
    assert g["correct"] is True and g["confidence"] == 90 and g["parse_ok"] is True
    g = parse_grade("**correct:** no\n**confidence:** 10")
    assert g["correct"] is False and g["parse_ok"] is True
    assert parse_grade("garbage")["parse_ok"] is False


def test_recall_is_intersection_over_evidence():
    assert recall_for(["1", "2"], ["1", "3", "4"]) == pytest.approx(1 / 3)
    assert recall_for([], ["1"]) == 0.0
    assert recall_for(["1"], []) is None


def test_evidence_docids_from_metadata():
    t = _traj("1", "a", answer="x", evidence=["74874", "2882"], retrieved=[])
    assert evidence_docids(t) == ["2882", "74874"]


def test_aggregate_accuracy_and_recall():
    grades = [
        {"correct": True, "parse_ok": True, "recall": 1.0},
        {"correct": False, "parse_ok": True, "recall": 0.5},
        {"correct": False, "parse_ok": False, "recall": None},
    ]
    agg = aggregate(grades)
    assert agg["n"] == 3 and agg["n_correct"] == 1
    assert agg["accuracy"] == pytest.approx(33.3333, abs=1e-3)
    assert agg["n_recall"] == 2
    assert agg["recall"] == pytest.approx(75.0)


def test_run_browsecomp_plus_grades_and_writes(tmp_path: Path):
    trajs = [
        _traj(
            "1",
            "Explanation: ...\nExact Answer: 1988-96\nConfidence: 90%",
            answer="1988-96",
            evidence=["2882", "74874"],
            retrieved=["74874", "999"],
        ),
        _traj(
            "3",
            "Exact Answer: wrong\nConfidence: 40%",
            answer="right",
            evidence=["5"],
            retrieved=["5"],
        ),
    ]
    judge = MockLLM(
        replies=[
            "extracted_final_answer: 1988-96\ncorrect: yes\nconfidence: 90",
            "extracted_final_answer: wrong\ncorrect: no\nconfidence: 40",
        ]
    )
    out = run_browsecomp_plus(
        trajs,
        run_dir=tmp_path / "run",
        model_name="gpt_researcher",
        judge_cfg={"model": "mock", "concurrency": 1},
        llm=judge,
    )
    assert out["official"] is True
    assert out["n"] == 2 and out["n_correct"] == 1
    assert out["accuracy"] == 50.0
    # query 1: 1/2 evidence hit; query 3: 1/1
    assert out["recall"] == pytest.approx(75.0)
    assert headline_scores({"browsecomp_plus": out}) == {
        "bcp_accuracy": 50.0,
        "bcp_recall": pytest.approx(75.0),
    }
    prompt = judge.calls[0][0].content
    assert "[correct_answer]: 1988-96" in prompt
    assert "Exact Answer: 1988-96" in prompt
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics" / "browsecomp_plus" / "gpt_researcher.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {r["id"] for r in rows} == {"1", "3"}
    assert (tmp_path / "run" / "exports" / "browsecomp_plus" / "gpt_researcher" / "1.json").exists()
    assert (tmp_path / "run" / "exports" / "browsecomp_plus" / "ground_truth.jsonl").exists()


def test_run_browsecomp_plus_missing_answers(tmp_path: Path):
    t = Trajectory(
        query=Query(id="1", text="q", dataset="browsecomp_plus"),
        report=Report(article="a"),
    )
    out = run_browsecomp_plus([t], run_dir=tmp_path, model_name="m", llm=MockLLM())
    assert out["official"] is False
    assert "gold answer" in out["reason"]


def test_official_query_template_wraps_raw_question():
    raw = "Which river runs through Munich?"
    wrapped = format_query(raw, "QUERY_TEMPLATE_NO_GET_DOCUMENT")
    assert wrapped == QUERY_TEMPLATE_NO_GET_DOCUMENT.format(Question=raw)
    assert "You are a deep research agent" in wrapped
    assert "using the search tool provided" in wrapped
    assert "get_document" not in wrapped
    assert format_query(raw) == raw
    with pytest.raises(ValueError, match="Unknown query template"):
        format_query(raw, "NOPE")


def test_grader_sees_raw_question_not_the_query_template():
    assert "[question]: {question}" in GRADER_TEMPLATE
    assert "You are a deep research agent" not in GRADER_TEMPLATE
