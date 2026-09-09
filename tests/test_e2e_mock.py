from pathlib import Path

import yaml

import json

from adr.agents.deep_research import DeepResearchAgent
from adr.agents.pilot import PilotAgent
from adr.eval.compare import compare_summaries
from adr.runner.config import load_config
from adr.runner.experiment import run_experiment


def test_smoke_run_writes_gym_exports_and_metrics(tmp_path: Path):
    cfg = load_config(
        "configs/default.yaml",
        {
            "output_dir": str(tmp_path),
            "run_name": "e2e",
            "concurrency": 1,
            "dataset": {
                "name": "deep_research_gym",
                "query_ids": ["923549", "879779"],
                "limit": 2,
            },
            "agent": {"name": "fixture"},
            "llm": {"provider": "mock"},
            "search": {"backend": "mock"},
        },
    )
    manifest = run_experiment(cfg)
    run_dir = manifest.run_dir
    assert (run_dir / "exports" / "deep_research_gym" / manifest.run_id / "923549.a").exists()
    assert (run_dir / "exports" / "deep_research_gym" / manifest.run_id / "923549.q").exists()
    assert (run_dir / "reports" / "923549.md").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_queries"] == 2
    assert summary["n_with_report"] == 2
    assert summary["mean_n_citations"] >= 1


def test_drb_export_from_fixture(tmp_path: Path):
    cfg = load_config(
        "configs/default.yaml",
        {
            "output_dir": str(tmp_path),
            "run_name": "drb",
            "concurrency": 1,
            "dataset": {"name": "deep_research_bench", "language": "en", "limit": 1},
            "agent": {"name": "fixture"},
        },
    )
    manifest = run_experiment(cfg)
    export = manifest.run_dir / "exports" / "deep_research_bench" / f"{manifest.run_id}.jsonl"
    assert export.exists()
    row = json.loads(export.read_text(encoding="utf-8").splitlines()[0])
    assert set(row) == {"id", "prompt", "article"}
    assert row["article"]


def test_unimplemented_agents_are_registered_but_empty():
    assert DeepResearchAgent().name == "deep_research"
    assert PilotAgent().name == "pilot"


def test_compare_two_summaries(tmp_path: Path):
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text('{"n_queries": 2, "mean_tokens": 100, "mean_latency_s": 10}\n')
    right.write_text('{"n_queries": 2, "mean_tokens": 80, "mean_latency_s": 12}\n')
    out = compare_summaries(left, right)
    assert out["deltas_right_minus_left"]["mean_tokens"] == -20
