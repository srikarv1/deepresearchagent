import json

from adr.core.types import Query, Report, Trajectory
from adr.eval.exporters import (
    export_browsecomp_plus,
    export_browsecomp_plus_ground_truth,
    export_deep_research_bench,
    export_deep_research_gym,
)


def _traj(qid: str, text: str, article: str, dataset: str) -> Trajectory:
    return Trajectory(
        query=Query(id=qid, text=text, dataset=dataset, language="en"),
        report=Report(article=article, citations=["https://example.com"]),
    )


def test_drb_jsonl_matches_official_schema(tmp_path):
    dest = tmp_path / "model.jsonl"
    export_deep_research_bench(
        [_traj("51", "What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger? ", "article body", "deep_research_bench")],
        dest,
    )
    row = json.loads(dest.read_text(encoding="utf-8").splitlines()[0])
    assert row == {
        "id": 51,
        "prompt": "What are the investment philosophies of Duan Yongping, Warren Buffett, and Charlie Munger? ",
        "article": "article body",
    }


def test_gym_qa_pair_files(tmp_path):
    dest = tmp_path / "gym"
    export_deep_research_gym(
        [_traj("923549", "why is there a chip shortage", "long report", "deep_research_gym")],
        dest,
    )
    assert (dest / "923549.q").read_text(encoding="utf-8").strip() == "why is there a chip shortage"
    assert (dest / "923549.a").read_text(encoding="utf-8").strip() == "long report"


def test_browsecomp_plus_run_files_match_official_schema(tmp_path):
    ok = _traj("1", "An African author...", "Exact Answer: 1988-96", "browsecomp_plus")
    ok.query.metadata["answer"] = "1988-96"
    ok.final_stats = {
        "usage": {"n_search_calls": 3, "n_fetch_calls": 7},
        "retrieved_docids": ["74874", "2882", "74874"],
    }
    failed = Trajectory(
        query=Query(id="3", text="The player...", dataset="browsecomp_plus", language="en"),
        error="RuntimeError: boom",
    )

    dest = export_browsecomp_plus(
        [ok, failed], tmp_path / "gpt_researcher", model_name="gpt_researcher"
    )
    row = json.loads((dest / "1.json").read_text(encoding="utf-8"))
    assert row["query_id"] == "1"
    assert row["status"] == "completed"
    assert row["tool_call_counts"] == {"search": 3, "fetch": 7}
    assert row["retrieved_docids"] == ["2882", "74874"]
    assert row["result"][-1] == {"type": "output_text", "output": "Exact Answer: 1988-96"}
    assert row["metadata"]["model"] == "gpt_researcher"

    row = json.loads((dest / "3.json").read_text(encoding="utf-8"))
    assert row["status"] == "failed"
    assert row["retrieved_docids"] == []
    assert row["result"][-1]["output"] == ""

    gt = export_browsecomp_plus_ground_truth([ok, failed], tmp_path / "ground_truth.jsonl")
    lines = [json.loads(line) for line in gt.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"query_id": "1", "query": "An African author...", "answer": "1988-96"}
    assert lines[1]["answer"] == ""
