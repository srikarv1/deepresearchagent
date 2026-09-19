"""`adr rollouts`: planning, subprocess command/env, resume, and one real
offline rollout through the fixture agent (mock LLM + mock corpus)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.rollouts.driver import (
    ENV_POLICY,
    ENV_SEED,
    is_complete,
    plan_rollouts,
    read_manifest,
    run_id_for,
    run_rollouts,
)
from adr.runner.config import ROOT
from adr.runner.experiment import _run_id

DEFAULT = ROOT / "configs" / "default.yaml"


def test_plan_enumerates_query_x_seed(tmp_path: Path):
    specs, cfg = plan_rollouts(config=DEFAULT, out=tmp_path, n_seeds=3, seed_base=100, limit=2)
    assert len(specs) == 6
    assert {s.seed for s in specs} == {100, 101, 102}
    assert all(s.run_id == run_id_for(s.query_id, s.seed) for s in specs)
    assert all(s.run_dir == tmp_path / s.run_id for s in specs)
    assert cfg["dataset"]["name"] == "deep_research_gym"


def test_plan_honours_explicit_query_ids(tmp_path: Path):
    specs, _ = plan_rollouts(config=DEFAULT, out=tmp_path, n_seeds=1, query_ids=["923549"])
    assert [s.query_id for s in specs] == ["923549"]


def test_cmd_and_env(tmp_path: Path):
    specs, _ = plan_rollouts(config=DEFAULT, out=tmp_path, n_seeds=1, seed_base=7, limit=1)
    spec = specs[0]
    cmd = spec.cmd(DEFAULT, tmp_path, ["--limit", "1"])
    assert cmd[1:4] == ["-m", "adr.cli", "run"]
    assert "--query-ids" in cmd and cmd[cmd.index("--query-ids") + 1] == spec.query_id
    assert cmd[cmd.index("--run-id") + 1] == spec.run_id
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path)
    assert cmd[-2:] == ["--limit", "1"]
    env = spec.env("random", base_env={"PATH": "/bin"}, extra={"GR_CONTEXT_BUDGET_TOKENS": "50000"})
    assert env[ENV_POLICY] == "random" and env[ENV_SEED] == "7"
    assert env["GR_CONTEXT_BUDGET_TOKENS"] == "50000" and env["PATH"] == "/bin"


def test_run_id_override_wins_over_timestamp():
    from datetime import datetime

    assert _run_id({"run_id": "q1-s3", "run_name": "x"}, datetime(2026, 1, 1)) == "q1-s3"
    assert _run_id({"run_name": "x"}, datetime(2026, 1, 1)).endswith("-x")


def test_is_complete_requires_report(tmp_path: Path):
    run = tmp_path / "r"
    (run / "trajectories").mkdir(parents=True)
    assert not is_complete(run, "q")
    (run / "trajectories" / "q.json").write_text(json.dumps({"report": None, "error": "boom"}))
    assert not is_complete(run, "q")
    (run / "trajectories" / "q.json").write_text(json.dumps({"report": {"article": "x"}, "error": None}))
    assert is_complete(run, "q")


def test_dry_run_writes_plan_only(tmp_path: Path):
    out = tmp_path / "corpus"
    results = run_rollouts(config=DEFAULT, out=out, n_seeds=2, limit=1, dry_run=True)
    assert len(results) == 2 and all(r.status == "dry-run" for r in results)
    assert (out / "rollouts_config.json").exists()
    assert read_manifest(out) == []  # nothing recorded for a dry run


def test_real_rollout_then_resume(tmp_path: Path):
    """End-to-end through a subprocess `adr run` with the fixture agent."""
    out = tmp_path / "corpus"
    first = run_rollouts(config=DEFAULT, out=out, n_seeds=1, seed_base=5, limit=1, parallel=1, timeout_s=200)
    assert len(first) == 1
    res = first[0]
    assert res.status == "done", (res.error, Path(res.log or "").read_text() if res.log else "")
    run_dir = Path(res.run_dir)
    assert run_dir.name == run_id_for(res.query_id, 5)
    assert (run_dir / "trajectories" / f"{res.query_id}.json").exists()
    assert (run_dir / "metrics" / "summary.json").exists()
    cfg = (run_dir / "config.yaml").read_text()
    assert f"run_id: {run_dir.name}" in cfg
    rows = read_manifest(out)
    assert len(rows) == 1 and rows[0]["status"] == "done"

    second = run_rollouts(config=DEFAULT, out=out, n_seeds=1, seed_base=5, limit=1)
    assert second[0].status == "skipped"
    assert len(read_manifest(out)) == 2
