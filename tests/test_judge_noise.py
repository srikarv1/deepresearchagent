"""Writer and judge noise measurement without any LLM or judge API.

A stub stands in for ``deepresearch_bench_race.py``: it scores each report by
its length, so the expected per-query numbers are exact. A fake researcher
class stands in for the fork's writer.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from adr.eval.judge_noise import (
    NoiseQuery,
    branch_effect_se,
    load_noise_queries,
    pooled_within_sd,
    race_slot,
    rewrite_report,
    summarize,
)

RACE_STUB = """
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("target_model")
parser.add_argument("--raw_data_dir")
parser.add_argument("--max_workers", type=int)
parser.add_argument("--query_file")
parser.add_argument("--output_dir")
parser.add_argument("--only_en", action="store_true")
parser.add_argument("--only_zh", action="store_true")
parser.add_argument("--skip_cleaning", action="store_true")
args = parser.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
calls = Path(args.raw_data_dir, "_calls.txt")
calls.write_text((calls.read_text() if calls.exists() else "") + args.target_model + "\\n")
rows = [json.loads(l) for l in open(Path(args.raw_data_dir, args.target_model + ".jsonl")) if l.strip()]
with open(out / "raw_results.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps({"id": r["id"], "overall_score": len(r["article"]) / 1000}) + "\\n")
(out / "race_result.txt").write_text("Overall Score: 0.5\\n")
"""


@pytest.fixture
def fake_bench(tmp_path: Path) -> Path:
    root = tmp_path / "deep_research_bench"
    (root / "data" / "prompt_data").mkdir(parents=True)
    (root / "data" / "test_data" / "raw_data").mkdir(parents=True)
    (root / "deepresearch_bench_race.py").write_text(RACE_STUB, encoding="utf-8")
    (root / "data" / "prompt_data" / "query.jsonl").write_text(
        json.dumps({"id": 51, "language": "en", "prompt": "q51"}) + "\n", encoding="utf-8"
    )
    return root


def _write_run(run_dir: Path, qid: str, *, report: str = "r" * 400, context: str = "evidence") -> None:
    (run_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_dir / "gpt_researcher").mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectories" / f"{qid}.json").write_text(json.dumps({
        "query": {"id": qid, "text": f"q{qid}", "dataset": "deep_research_bench"},
        "report": {"article": report},
    }))
    (run_dir / "gpt_researcher" / f"{qid}.json").write_text(json.dumps({"final_context": context}))


# ── statistics ───────────────────────────────────────────────────────────────


def test_pooled_within_sd_ignores_between_query_spread():
    sd, dof = pooled_within_sd([[1.0, 3.0], [10.0, 10.0, 10.0]])
    assert dof == 3
    assert sd == pytest.approx(math.sqrt(2.0 / 3.0))


def test_pooled_within_sd_skips_singletons_and_empty():
    assert pooled_within_sd([[1.0], []]) == (None, 0)
    sd, dof = pooled_within_sd([[0.2], [0.4, 0.6]])
    assert dof == 1
    assert sd == pytest.approx(math.sqrt(0.02))


def test_branch_effect_se_by_tree_shape():
    assert branch_effect_se(1.0, "d2b4") == pytest.approx(0.5)
    assert branch_effect_se(1.0, "d3b4") == pytest.approx(math.sqrt(2 / 27))


def test_summarize_derives_writer_noise():
    s = summarize(
        {"1": [0.40, 0.44], "2": [0.50, 0.54]},
        {"1": [0.41, 0.42], "2": [0.51, 0.52]},
        n_train_queries=16,
    )
    assert s["sigma_total"] == pytest.approx(math.sqrt(0.0016 / 2 * 2 / 2))
    assert s["sigma_judge"] == pytest.approx(math.sqrt(0.0001 / 2 * 2 / 2))
    assert s["sigma_writer"] == pytest.approx(math.sqrt(s["sigma_total"] ** 2 - s["sigma_judge"] ** 2))
    d2 = s["branch_effect_se"]["d2b4"]
    assert d2["per_query"] == pytest.approx(s["sigma_total"] / 2)
    assert d2["pooled"] == pytest.approx(s["sigma_total"] / 8)
    assert s["per_query"]["1"]["writer_range"] == pytest.approx(0.04)


def test_summarize_writer_noise_floors_at_zero():
    s = summarize({"1": [0.5, 0.5]}, {"1": [0.4, 0.6]})
    assert s["sigma_writer"] == 0.0


# ── run directory ────────────────────────────────────────────────────────────


def test_load_noise_queries_reads_context_and_reports_skips(tmp_path: Path):
    run = tmp_path / "run"
    _write_run(run, "51", context="ctx51")
    _write_run(run, "52", context="")
    (run / "trajectories" / "53.json").write_text(json.dumps({
        "query": {"id": "53", "text": "q53", "dataset": "deep_research_bench"},
        "report": {"article": "a"},
    }))
    queries, skipped = load_noise_queries(run)
    assert [q.qid for q in queries] == ["51"]
    assert queries[0].final_context == "ctx51"
    assert queries[0].prompt == "q51"
    assert any(m.startswith("52:") for m in skipped)
    assert any(m.startswith("53:") and "keep_trajectory_files" in m for m in skipped)


def test_load_noise_queries_rejects_other_benchmarks(tmp_path: Path):
    run = tmp_path / "run"
    _write_run(run, "7")
    traj = run / "trajectories" / "7.json"
    data = json.loads(traj.read_text())
    data["query"]["dataset"] = "deep_research_gym"
    traj.write_text(json.dumps(data))
    queries, skipped = load_noise_queries(run, ["7", "8"])
    assert queries == []
    assert "deep_research_gym" in skipped[0]
    assert skipped[1].startswith("8:")


# ── writer ───────────────────────────────────────────────────────────────────


class _FakeResearcher:
    instances: list["_FakeResearcher"] = []

    def __init__(self, query: str, report_type: str):
        self.query = query
        self.report_type = report_type
        self.deep_researcher = object()
        self.context = []
        _FakeResearcher.instances.append(self)

    async def write_report(self) -> str:
        assert self.deep_researcher is None, "the live trajectory would be overwritten"
        return f"report for {self.query} from {len(self.context)} chars"


@pytest.mark.asyncio
async def test_rewrite_report_uses_final_context_and_detaches_deep_skill(monkeypatch):
    monkeypatch.setenv("GR_ANSWER_FORMAT", "browsecomp")
    q = NoiseQuery(qid="51", prompt="q51", report="old", final_context="x" * 120)
    text = await rewrite_report(_FakeResearcher, q)
    assert text == "report for q51 from 120 chars"
    r = _FakeResearcher.instances[-1]
    assert r.report_type == "deep"
    assert r.context == "x" * 120
    import os
    assert "GR_ANSWER_FORMAT" not in os.environ


# ── RACE ─────────────────────────────────────────────────────────────────────


def test_race_slot_scores_each_query_under_its_own_model_name(fake_bench: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock")
    q = NoiseQuery(qid="51", prompt="q51", report="", final_context="c")
    scores = race_slot("noise_x_w0", [(q, "a" * 250)], bench_root=fake_bench, env={})
    assert scores == {"51": pytest.approx(0.25)}
    staged = fake_bench / "data" / "test_data" / "raw_data" / "noise_x_w0.jsonl"
    assert json.loads(staged.read_text())["id"] == 51


# ── script end to end ────────────────────────────────────────────────────────


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_judge_noise.py"
    spec = importlib.util.spec_from_file_location("measure_judge_noise", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tracker:
    @staticmethod
    def reset() -> None:
        pass

    @staticmethod
    def get_totals() -> dict:
        return {"input_tokens": 10, "output_tokens": 5, "cost": 0.0}


class _GrowingResearcher(_FakeResearcher):
    """Each rewrite is 100 characters longer than the last, so the stub judge
    gives every rewrite a distinct score."""

    n = 0

    async def write_report(self) -> str:
        _GrowingResearcher.n += 1
        return "w" * (300 + 100 * _GrowingResearcher.n)


def test_script_end_to_end_with_resume(fake_bench: Path, tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "mock")
    monkeypatch.setenv("LLM_BACKEND", "openai")
    run = tmp_path / "run-1"
    _write_run(run, "51", report="r" * 400, context="evidence")
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("name: gpt_researcher\nenv:\n  SMART_LLM: anthropic:claude-opus-4-6\n")
    (run / "config.yaml").write_text(json.dumps({
        "agent": {"name": "gpt_researcher", "config": str(agent_cfg)},
        "eval": {"deep_research_bench": {"third_party_dir": str(fake_bench), "workers": 1}},
    }))

    mod = _load_script()
    monkeypatch.setattr(
        mod.GPTResearcherAgent, "_modules", lambda self: (_GrowingResearcher, _Tracker, None)
    )
    _GrowingResearcher.n = 0
    assert mod.main([str(run), "--rewrites", "2", "--rescores", "2"]) == 0

    summary = json.loads((run / "noise" / "summary.json").read_text())
    row = summary["per_query"]["51"]
    # original (0.4) + two rewrites (0.4, 0.5); the stub judge is deterministic.
    assert row["writer_scores"] == pytest.approx([0.4, 0.4, 0.5])
    assert row["judge_scores"] == pytest.approx([0.4, 0.4])
    assert summary["sigma_judge"] == pytest.approx(0.0)
    assert summary["writer"] == {"SMART_LLM": "anthropic:claude-opus-4-6"}
    assert "writer cost" in capsys.readouterr().out

    # A second invocation reuses the cached reports and RACE scores.
    calls = fake_bench / "data" / "test_data" / "raw_data" / "_calls.txt"
    n_calls = len(calls.read_text().splitlines())
    assert n_calls == 4
    assert mod.main([str(run), "--rewrites", "2", "--rescores", "2"]) == 0
    assert len(calls.read_text().splitlines()) == n_calls
    assert _GrowingResearcher.n == 2


def test_script_dry_run_calls_nothing(tmp_path: Path, monkeypatch, capsys):
    run = tmp_path / "run"
    _write_run(run, "51", context="e" * 40000)
    agent_cfg = tmp_path / "agent.yaml"
    agent_cfg.write_text("name: gpt_researcher\nenv:\n  SMART_LLM: anthropic:claude-opus-4-6\n")
    (run / "config.yaml").write_text(json.dumps({"agent": {"name": "gpt_researcher", "config": str(agent_cfg)}}))
    mod = _load_script()
    monkeypatch.setattr(mod.GPTResearcherAgent, "_modules", lambda self: pytest.fail("must not import the fork"))
    assert mod.main([str(run), "--rewrites", "4", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "4 reports" in out and "~0.04M input" in out
    assert not (run / "noise").exists()
