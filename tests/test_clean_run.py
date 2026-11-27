"""Offline regression checks: no model, retriever, or judge API is called."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path):
    (tmp_path / "scripts").mkdir()
    script = (REPO / "scripts/clean_run.sh").read_text().split("case $stage in\n")[0]
    bin_dir = tmp_path / ".venv/bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(sys.executable)
    fake = bin_dir / "adr"
    fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
out = pathlib.Path(args[args.index("--out") + 1])
ids = args[args.index("--query-ids") + 1].split(",")
env = dict(os.environ)
for i, arg in enumerate(args):
    if arg == "--env":
        k, v = args[i + 1].split("=", 1)
        env[k] = v
(out / "effective_env.json").write_text(json.dumps({k: v for k, v in env.items() if k.startswith("GR_ORCH_LLM_")}))
counter = out / "attempts"
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
fail = out.name == os.environ.get("FAKE_FAIL_ARM", "")
until = int(os.environ.get("FAKE_FAIL_UNTIL", "0"))
for q in ids:
    target = out / f"{q}-s0/trajectories/{q}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if fail and attempt <= until:
        target.write_text(os.environ.get("FAKE_PARTIAL", '{"report": ""}'))
    else:
        target.write_text(json.dumps({"report": "completed report"}))
sys.exit(17 if fail and attempt <= until and os.environ.get("FAKE_NONZERO", "1") == "1" else 0)
''')
    fake.chmod(0o755)
    # Do not start real power-management processes from tests.
    tools = tmp_path / "tools"
    tools.mkdir()
    caffeinate = tools / "caffeinate"
    caffeinate.write_text("#!/bin/sh\nexit 0\n")
    caffeinate.chmod(0o755)

    def run(command, **overrides):
        env = dict(os.environ, ROOT=str(tmp_path / "runs"), WRITER_MAX_INPUT_TOKENS="250000",
                   BUDGETS="200000", STOPPERS="", JOBS="8", STALL_POLL_S="0.02",
                   PATH=str(tools) + os.pathsep + os.environ["PATH"])
        env.update(overrides)
        invocation = tmp_path / "scripts/invocation.sh"
        invocation.write_text(script + "\n" + command + "\n")
        return subprocess.run(["bash", str(invocation)],
                              cwd=tmp_path, env=env, text=True, capture_output=True, timeout=15)
    return tmp_path, run


def test_baselines_do_not_inherit_the_learned_server(runner):
    root, run = runner
    result = run('generate "$ROOT" "1" "200000"',
                 ARMS="none extractive prompted prompted_scores bc dpo",
                 GR_ORCH_LLM_BASE_URL="http://gpu.invalid/v1", GR_ORCH_LLM_MODEL="wrong-model",
                 GR_ORCH_LLM_API_KEY="demo-key", GR_ORCH_LLM_PROVIDER="wrong-provider")
    assert result.returncode == 0, result.stdout + result.stderr
    for arm in ("none", "extractive", "prompted", "prompted_scores", "bc", "dpo"):
        env = json.loads((root / "runs/equal_200k" / arm / "effective_env.json").read_text())
        if arm in ("bc", "dpo"):
            assert env["GR_ORCH_LLM_BASE_URL"] == "http://gpu.invalid/v1"
            assert env["GR_ORCH_LLM_MODEL"] == f"pilot-{arm}-teacher"
            assert env["GR_ORCH_LLM_PROVIDER"] == "openai"
        else:
            assert all(v == "" for v in env.values()), (arm, env)


def test_failed_worker_retries_and_recovers(runner):
    root, run = runner
    result = run('run_arm "$ROOT/equal_200k" prompted 200000 1 "1"',
                 FAKE_FAIL_ARM="prompted", FAKE_FAIL_UNTIL="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "runs/equal_200k/prompted/attempts").read_text() == "2"


@pytest.mark.parametrize("partial", ['{"report": ""}', '{"report": "draft", "error": "failed"}', 'broken json'])
def test_existing_partial_file_is_not_a_finished_report(runner, partial):
    root, run = runner
    result = run('run_arm "$ROOT/equal_200k" prompted 200000 1 "1"',
                 FAKE_FAIL_ARM="prompted", FAKE_FAIL_UNTIL="99", FAKE_NONZERO="0", FAKE_PARTIAL=partial)
    assert result.returncode != 0
    assert (root / "runs/equal_200k/prompted/attempts").read_text() == "3"


def test_generation_returns_failure_if_an_arm_stays_incomplete(runner):
    _, run = runner
    result = run('generate "$ROOT" "1" "200000"', ARMS="none prompted",
                 FAKE_FAIL_ARM="prompted", FAKE_FAIL_UNTIL="99")
    assert result.returncode != 0


def test_dry_run_does_not_log_the_gpu_key(runner):
    _, run = runner
    result = run('run_arm "$ROOT/equal_200k" bc 200000 1 "1"', DRYRUN="1",
                 GR_ORCH_LLM_API_KEY="private-demo-value")
    assert result.returncode == 0
    assert "private-demo-value" not in result.stdout
    assert "[redacted]" in result.stdout


def _results():
    spec = importlib.util.spec_from_file_location("clean_results", REPO / "scripts/clean_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_counter_does_not_stop_at_an_analysis_heading():
    c = _results()
    body = "# Sources and Analytical Approach\nEvidence is discussed here.\n"
    assert c._body_words(body + "# References\nnot report words") == len(body.split())
    assert c._body_words({"article": body}) == len(body.split())


def test_incomplete_trajectories_are_excluded_and_retry_waits_are_flagged(tmp_path):
    c = _results()
    arm = tmp_path / "equal_200k/none"
    for q, report in [("1", "finished report"), ("2", "")]:
        run = arm / f"{q}-s0"
        (run / "trajectories").mkdir(parents=True)
        (run / "trajectories" / f"{q}.json").write_text(json.dumps({"report": report}))
    (arm / "logs").mkdir()
    (arm / "logs/1-s0.log").write_text("LLM request failed (attempt 1/3)\n")
    data = c.collect(tmp_path)[("equal", 200)]["none"]
    assert set(data) == {"1"}
    assert data["1"]["facts"]["net_retries"] == 1
