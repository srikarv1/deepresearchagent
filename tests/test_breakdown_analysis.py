"""P1 breakdown analysis: token split, latency split, arm discovery and output."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ba = _script("breakdown_analysis")


def _tags(**tokens: int) -> dict:
    return {tag: {"input": n, "output": 0, "cost": 0.0} for tag, n in tokens.items()}


def _write_run(arm_dir: Path, qid: str, *, tags: dict, wall: float, writing: float, rounds: list[float],
               search: float, llm: float, calls: list | None = None, report: str = "# r",
               error: str | None = None) -> None:
    run = arm_dir / f"{qid}-s0"
    (run / "trajectories").mkdir(parents=True)
    (run / "gpt_researcher").mkdir()
    fs = {
        "tokens_by_tag": tags,
        "wall_s": wall,
        "synthesis_latency_s": writing,
        "usage": {"search_latency_s": search, "llm_latency_s": llm},
    }
    if calls is not None:
        fs["llm_calls"] = calls
    (run / "trajectories" / f"{qid}.json").write_text(
        json.dumps({"report": report, "error": error, "final_stats": fs}), encoding="utf-8")
    (run / "gpt_researcher" / f"{qid}.json").write_text(
        json.dumps({"rounds": [{"round_cost": {"latency_seconds": s}} for s in rounds]}), encoding="utf-8")


# ── token split ───────────────────────────────────────────────────────────


def test_token_split_groups_add_up_to_the_total():
    split = ba.token_split({"tokens_by_tag": _tags(
        deep_research_plan=10, deep_search_query_gen=2, agent_selection=3, subquery_generation=40,
        deep_process_results=50, orchestrator=20, report_generation=30, report_conclusion=5, mystery=1)})
    assert split["tok_research"] == 105
    assert split["tok_orchestrator"] == 20
    assert split["tok_writing"] == 35
    assert split["tok_other"] == 1
    assert split["tok_total"] == 161 == sum(split[f"tok_{g}"] for g in ("research", "orchestrator", "writing", "other"))
    assert (split["tok_plan"], split["tok_subquery_gen"], split["tok_process"]) == (10, 40, 50)


def test_token_split_of_a_run_without_tags_is_zero():
    assert ba.token_split({})["tok_total"] == 0


# ── latency split ─────────────────────────────────────────────────────────


def test_latency_phases_add_up_to_wall_and_research_resources_to_research_time():
    fs = {"wall_s": 600.0, "synthesis_latency_s": 100.0,
          "usage": {"search_latency_s": 50.0, "llm_latency_s": 400.0}}
    lat = ba.latency_split(fs, rounds_latency=450.0)
    assert lat["lat_setup"] == 50.0
    assert lat["lat_setup"] + lat["lat_rounds"] + lat["lat_writing"] == lat["lat_wall"]
    assert lat["lat_research_llm"] == 300.0  # LLM time outside the report
    assert lat["lat_research_other"] == 150.0
    research_time = lat["lat_wall"] - lat["lat_writing"]
    assert lat["lat_search"] + lat["lat_research_llm"] + lat["lat_research_other"] == research_time
    assert lat["lat_llm_research"] is None  # no per-call log


def test_latency_split_with_missing_timings_leaves_columns_empty():
    lat = ba.latency_split({"wall_s": 100.0}, rounds_latency=None)
    assert lat["lat_wall"] == 100.0
    assert lat["lat_setup"] is None and lat["lat_research_llm"] is None and lat["lat_research_other"] is None


def test_call_log_gives_exact_llm_time_per_stage_and_flex_refusals():
    calls = [
        {"usage_tag": "deep_process_results", "duration_s": 10.0, "capacity_wait_s": 4.0},
        {"usage_tag": "subquery_generation", "duration_s": 5.0, "capacity_wait_s": 0.0},
        {"usage_tag": "orchestrator", "duration_s": 3.0},
        {"usage_tag": "report_generation", "duration_s": 20.0, "capacity_wait_s": 1.0},
        {"usage_tag": None, "duration_s": 2.0},
    ]
    lat = ba.latency_split({"wall_s": 50.0, "llm_calls": calls}, rounds_latency=None)
    assert lat["lat_llm_research"] == 17.0  # untagged calls count as research
    assert lat["lat_llm_orchestrator"] == 3.0
    assert lat["lat_llm_writing"] == 20.0
    assert lat["lat_llm_process"] == 10.0 and lat["lat_llm_subquery_gen"] == 5.0
    assert lat["lat_flex_refused"] == 5.0


# ── arms, loading, summaries ──────────────────────────────────────────────


@pytest.fixture
def root(tmp_path: Path) -> Path:
    tags_none = _tags(deep_research_plan=10_000, subquery_generation=60_000, deep_process_results=90_000,
                      report_generation=40_000)
    tags_prompted = _tags(deep_research_plan=10_000, subquery_generation=40_000, deep_process_results=70_000,
                          orchestrator=20_000, report_generation=40_000)
    for qid in ("1", "2"):
        _write_run(tmp_path / "none", qid, tags=tags_none, wall=700.0, writing=100.0, rounds=[300.0, 250.0],
                   search=100.0, llm=500.0)
    _write_run(tmp_path / "prompted", "1", tags=tags_prompted, wall=500.0, writing=100.0, rounds=[200.0, 150.0],
               search=40.0, llm=420.0)
    # Unfinished and failed runs are left out.
    _write_run(tmp_path / "prompted", "2", tags=tags_prompted, wall=1.0, writing=0.0, rounds=[], search=0.0,
               llm=0.0, error="boom")
    (tmp_path / "prompted" / "3-s0").mkdir()
    (tmp_path / "logs").mkdir()  # not an arm
    return tmp_path


def test_find_arms_orders_known_arms_and_skips_non_arm_dirs(root: Path):
    assert list(ba.find_arms(root)) == ["none", "prompted"]


def test_load_arm_keeps_only_finished_runs(root: Path):
    rows = ba.load_arm(root / "prompted")
    assert [r["query"] for r in rows] == ["1"]
    assert rows[0]["lat_rounds"] == 350.0


def test_restrict_to_common_and_summaries(root: Path):
    data = ba.restrict_to_common({a: ba.load_arm(p) for a, p in ba.find_arms(root).items()})
    assert {a: [r["query"] for r in rows] for a, rows in data.items()} == {"none": ["1"], "prompted": ["1"]}
    s = ba.summarize(data["prompted"])
    assert s["n"] == 1
    assert s["tok_total"] == 180_000 and s["tok_orchestrator"] == 20_000
    assert s["share_tok_writing"] == pytest.approx(40 / 180)
    assert s["share_lat_search"] == pytest.approx(40 / 500)


def test_queue_adjusted_json_fills_flex_queued(root: Path):
    (root / "queue_adjusted.json").write_text(
        json.dumps({"runs": {"none": {"1-s0": {"queue_s": 12.5}}}}), encoding="utf-8")
    rows = {r["query"]: r for r in ba.load_arm(root / "none")}
    assert rows["1"]["lat_flex_queued"] == 12.5
    assert rows["2"]["lat_flex_queued"] is None


def test_main_prints_tables_and_writes_csv_latex_json(root: Path, tmp_path: Path, capsys):
    out_csv, out_tex, out_json = tmp_path / "s.csv", tmp_path / "s.tex", tmp_path / "s.json"
    assert ba.main(["--root", str(root), "--common", "--csv", str(out_csv),
                    "--latex", str(out_tex), "--json", str(out_json)]) == 0
    printed = capsys.readouterr().out
    assert "No pruning" in printed and "Prompted orchestration" in printed
    assert "LLM seconds by stage" not in printed  # no per-call log in these runs
    assert out_csv.read_text(encoding="utf-8").count("\n") == 3  # header + 2 rows
    tex = out_tex.read_text(encoding="utf-8")
    assert tex.count(r"\begin{table}") == 3 and r"\%" in tex
    summaries = json.loads(out_json.read_text(encoding="utf-8"))
    assert summaries["none"]["tok_research"] == 160_000


def test_main_accepts_explicit_arms_and_rejects_bad_specs(root: Path, capsys):
    assert ba.main(["--arm", f"bc={root / 'none'}"]) == 0
    assert "PILOT (BC)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ba.main(["--arm", "no-equals-sign"])
    with pytest.raises(SystemExit):
        ba.main([])
