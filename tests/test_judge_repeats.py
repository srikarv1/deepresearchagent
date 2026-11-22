"""Judge repeats (P1 judge sensitivity): repeat layout, averaging and noise vs gaps."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


jr = _script("judge_repeats")
GYM = ("Clarity", "Breadth", "Depth", "Support", "KPR", "Overall", "Citation")


def _gym_official(overall: float) -> dict:
    """A deep_research_gym block that budget_table.gym_scores accepts."""
    return {"deep_research_gym": {
        "quality": {"n": 1, "average_normalized_score": overall,
                    "per_criterion": {c: {"average_rating": 6.0} for c in ("Clarity", "Breadth", "Depth", "Support")}},
        "kpr": {"n": 1, "average_support_rate": 50.0},
        "citation": {"n": 1, "average_citation_score": 70.0},
    }}


def _report_run(arm_dir: Path, qid: str, report: str = "# report", error: str | None = None) -> Path:
    run = arm_dir / f"{qid}-s0"
    (run / "trajectories").mkdir(parents=True)
    (run / "trajectories" / f"{qid}.json").write_text(
        json.dumps({"report": report, "error": error, "final_stats": {}}), encoding="utf-8")
    return run


def _judged(out: Path, arm: str, k: int, qid: str, overall: float) -> None:
    d = out / arm / f"r{k}" / f"{qid}-s0" / "metrics"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"official": _gym_official(overall)}), encoding="utf-8")


# ── discovery and copies ──────────────────────────────────────────────────


def test_report_runs_and_arms_skip_unfinished_and_the_repeats_dir(tmp_path: Path):
    _report_run(tmp_path / "none", "1")
    _report_run(tmp_path / "none", "2", report="")
    _report_run(tmp_path / "none", "3", error="boom")
    (tmp_path / "judge_repeats" / "none").mkdir(parents=True)
    assert [p.name for p in jr.report_runs(tmp_path / "none")] == ["1-s0"]
    assert list(jr.arms_under(tmp_path, None)) == ["none"]
    with pytest.raises(SystemExit):
        jr.arms_under(tmp_path, ["bc"])


def test_prepare_copy_holds_only_the_trajectory(tmp_path: Path):
    run = _report_run(tmp_path / "none", "7")
    (run / "metrics").mkdir()
    (run / "metrics" / "summary.json").write_text("{}", encoding="utf-8")
    dest = jr.repeat_dir(tmp_path / "out", "none", 2, run)
    jr.prepare_copy(run, dest)
    assert dest == tmp_path / "out" / "none" / "r2" / "7-s0"
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == ["trajectories/7.json"]


# ── run (judge faked) ─────────────────────────────────────────────────────


def test_run_judges_each_report_k_times_and_skips_finished_repeats(tmp_path: Path, monkeypatch):
    for q in ("1", "2"):
        _report_run(tmp_path / "none", q)
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.invalid/v1")
    calls = []

    def fake_run(cmd, stdout, stderr, env, cwd):
        dest = Path(cmd[cmd.index("evaluate") + 1])
        calls.append((dest.parent.name, dest.name, env["OPENAI_API_KEY"], env["OPENAI_BASE_URL"]))
        (dest / "metrics").mkdir(exist_ok=True)
        (dest / "metrics" / "summary.json").write_text(json.dumps({"official": _gym_official(60.0)}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jr.subprocess, "run", fake_run)
    args = SimpleNamespace(bench="drg", root=tmp_path, arms=["none"], out=None, config=None, jobs=1,
                           repeats=3, queries=None, dry_run=False)
    assert jr.run_cmd(args) == 0
    assert sorted((r, n) for r, n, _, _ in calls) == [(f"r{k}", f"{q}-s0") for k in (1, 2, 3) for q in ("1", "2")]
    # The judge gets the judge credentials, not Sol's.
    assert all(key == "k" and url == "https://judge.invalid/v1" for _, _, key, url in calls)
    # Rerun: everything is judged, nothing to do.
    calls.clear()
    assert jr.run_cmd(args) == 0 and calls == []


def test_run_reports_failures(tmp_path: Path, monkeypatch):
    _report_run(tmp_path / "none", "1")
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.invalid/v1")
    monkeypatch.setattr(jr.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    args = SimpleNamespace(bench="drg", root=tmp_path, arms=None, out=None, config=None, jobs=1,
                           repeats=1, queries=None, dry_run=False)
    assert jr.run_cmd(args) == 1


# ── summarize ─────────────────────────────────────────────────────────────


def test_summarize_arm_averages_repeats_and_measures_judge_noise(tmp_path: Path):
    out = tmp_path / "out"
    for k, overall in enumerate((60.0, 62.0, 64.0), start=1):
        _judged(out, "none", k, "1", overall)
    for k, overall in enumerate((70.0, 70.0), start=1):
        _judged(out, "none", k, "2", overall)
    _judged(out, "none", 1, "3", 50.0)  # one repeat only: dropped at min_repeats=2

    s = jr.summarize_arm(jr.collect(out, "none", "drg"), GYM, min_repeats=2)
    assert s["n_queries"] == 2 and s["repeats"] == [2, 3]
    assert s["per_query_mean"]["1"]["Overall"] == 62.0
    m = s["metrics"]["Overall"]
    assert m["mean"] == 66.0  # mean of 62 and 70
    assert m["judge_std"] == pytest.approx((2.0 + 0.0) / 2)  # stdev(60,62,64)=2, stdev(70,70)=0
    assert s["metrics"]["Clarity"]["judge_std"] == 0.0


def test_pairwise_gaps_compare_the_gap_with_the_judge_noise(tmp_path: Path):
    out = tmp_path / "out"
    for k, (a, b) in enumerate(((60.0, 61.0), (62.0, 63.0), (64.0, 65.0)), start=1):
        _judged(out, "none", k, "1", a)
        _judged(out, "bc", k, "1", b)
    summaries = {arm: jr.summarize_arm(jr.collect(out, arm, "drg"), GYM, 2) for arm in ("none", "bc")}
    overall = next(g for g in jr.pairwise_gaps(summaries, GYM) if g["metric"] == "Overall")
    assert overall["gap"] == pytest.approx(-1.0)
    assert overall["judge_std"] == pytest.approx(2.0)
    assert overall["gap_over_noise"] == pytest.approx(0.5)  # within the judge's noise
    clarity = next(g for g in jr.pairwise_gaps(summaries, GYM) if g["metric"] == "Clarity")
    assert clarity["gap_over_noise"] is None  # no noise, no ratio


def test_summarize_cmd_writes_outputs(tmp_path: Path, capsys):
    out = tmp_path / "judge_repeats"
    for k in (1, 2):
        _judged(out, "none", k, "1", 60.0 + k)
    csv_p, tex_p, json_p = tmp_path / "j.csv", tmp_path / "j.tex", tmp_path / "j.json"
    args = SimpleNamespace(bench="drg", root=tmp_path, arms=None, out=None, min_repeats=2,
                           csv=csv_p, latex=tex_p, json=json_p)
    assert jr.summarize_cmd(args) == 0
    printed = capsys.readouterr().out
    assert "No pruning" in printed and "61.500 ± 0.707" in printed
    assert csv_p.read_text(encoding="utf-8").count("\n") == 1 + len(GYM)
    assert r"$\pm$" in tex_p.read_text(encoding="utf-8")
    assert json.loads(json_p.read_text(encoding="utf-8"))["summaries"]["none"]["n_queries"] == 1


def test_judge_env_requires_credentials(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(jr, "REPO", tmp_path)  # no .env
    for k in ("JUDGE_API_KEY", "JUDGE_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        jr.judge_env()
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    with pytest.raises(SystemExit):  # base URL missing
        jr.judge_env()
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.invalid/v1")
    env = jr.judge_env()
    assert env["OPENAI_API_KEY"] == "k" and env["OPENAI_BASE_URL"] == "https://judge.invalid/v1"
