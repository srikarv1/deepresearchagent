"""Checks the DeepResearch Bench pipeline without calling the LLM backend.

A stub stands in for ``deepresearch_bench_race.py``. It records the arguments it
was handed and writes a ``race_result.txt`` in the upstream format, which lets us
assert on the invocation contract and on reading scores back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.eval.deep_research_bench import run_deep_research_bench
from adr.eval.importers import trajectory_from_pair
from adr.eval.scoring import headline_scores

RACE_STUB = """
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("target_model")
parser.add_argument("--raw_data_dir")
parser.add_argument("--cleaned_data_dir", default=None)
parser.add_argument("--max_workers", type=int)
parser.add_argument("--query_file")
parser.add_argument("--output_dir")
parser.add_argument("--only_en", action="store_true")
parser.add_argument("--only_zh", action="store_true")
parser.add_argument("--skip_cleaning", action="store_true")
args = parser.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
Path(args.raw_data_dir, "_stub_call.json").write_text(json.dumps({
    "target_model": args.target_model,
    "raw_data_dir": args.raw_data_dir,
    "max_workers": args.max_workers,
    "query_file": args.query_file,
    "output_dir": args.output_dir,
    "only_en": args.only_en,
    "only_zh": args.only_zh,
    "judge_key_seen": bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    "target_rows": sum(
        1 for _ in open(Path(args.raw_data_dir, args.target_model + ".jsonl"), encoding="utf-8")
    ),
}))
(out / "race_result.txt").write_text(
    "Comprehensiveness: 0.5120\\n"
    "Insight: 0.4880\\n"
    "Instruction Following: 0.5400\\n"
    "Readability: 0.5010\\n"
    "Overall Score: 0.5102\\n"
)
"""


@pytest.fixture
def fake_bench(tmp_path: Path) -> Path:
    root = tmp_path / "deep_research_bench"
    (root / "data" / "prompt_data").mkdir(parents=True)
    (root / "data" / "test_data" / "raw_data").mkdir(parents=True)
    (root / "deepresearch_bench_race.py").write_text(RACE_STUB, encoding="utf-8")
    (root / "data" / "prompt_data" / "query.jsonl").write_text(
        json.dumps({"id": 51, "topic": "Finance", "language": "en", "prompt": "q"}) + "\n",
        encoding="utf-8",
    )
    return root


def _traj():
    return trajectory_from_pair(
        question="What are the investment philosophies of Buffett and Munger?",
        report="# Report\n\nBody with a source https://example.com/a\n",
        query_id="51",
        dataset="deep_research_bench",
    )


def test_race_invocation_and_score_parseback(fake_bench: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    monkeypatch.delenv("JINA_API_KEY", raising=False)

    result = run_deep_research_bench(
        [_traj()],
        run_dir=tmp_path / "run",
        model_name="pilot",
        third_party_dir=fake_bench,
        language="en",
        workers=3,
    )

    assert result["official"] is True, result
    assert result["race"]["log"]["ok"], result["race"]["log"]

    call = json.loads(
        (fake_bench / "data" / "test_data" / "raw_data" / "_stub_call.json").read_text()
    )
    assert call["target_model"] == "pilot"
    assert call["only_en"] is True and call["only_zh"] is False
    assert call["max_workers"] == 3
    assert call["judge_key_seen"] is True
    assert call["target_rows"] == 1
    assert Path(call["query_file"]).as_posix().endswith("data/prompt_data/query.jsonl")

    assert result["race"]["scores"]["overall_score"] == pytest.approx(0.5102)
    assert result["race"]["scores"]["comprehensiveness"] == pytest.approx(0.5120)

    # FACT needs a scraping key, so it should be skipped rather than crash.
    assert result["fact"]["skipped"] is True
    assert "JINA_API_KEY" in result["fact"]["reason"]

    flat = headline_scores({"deep_research_bench": result})
    assert flat["race_overall_score"] == pytest.approx(0.5102)
    assert flat["race_readability"] == pytest.approx(0.5010)


def test_missing_judge_key_is_reported_not_silent(fake_bench: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "openai")
    result = run_deep_research_bench(
        [_traj()], run_dir=tmp_path / "run", model_name="pilot", third_party_dir=fake_bench
    )
    assert result["official"] is False
    assert "OPENAI_API_KEY" in result["reason"]
    assert Path(result["export"]).exists()


def test_missing_repo_is_reported(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ADR_DRB_DIR", str(tmp_path / "nope"))
    result = run_deep_research_bench(
        [_traj()], run_dir=tmp_path / "run", model_name="pilot", third_party_dir=tmp_path / "nope"
    )
    assert result["official"] is False
    assert "not found" in (result["reason"] or "")


def test_export_prompt_matches_query_file_for_reference_pairing(tmp_path: Path):
    """RACE joins target to reference by exact prompt string, so preserve it."""
    from adr.datasets.loader import load_queries
    from adr.eval.exporters import export_deep_research_bench

    query = load_queries("deep_research_bench", language="en", limit=1)[0]
    traj = trajectory_from_pair(
        question=query.text, report="body", query_id=query.id, dataset="deep_research_bench"
    )
    dest = export_deep_research_bench([traj], tmp_path / "m.jsonl")
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt"] == query.text
    assert row["id"] == int(query.id)
