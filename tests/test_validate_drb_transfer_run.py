from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_drb_transfer_run import EXPECTED_QUERY_IDS, validate_run


BC = "pilot-bc-teacher"


def _write_run(root: Path, adapter: str = BC) -> Path:
    run_dir = root / "run"
    for name in ("trajectories", "reports", "gpt_researcher"):
        (run_dir / name).mkdir(parents=True)
    for query_id in EXPECTED_QUERY_IDS:
        harness = {
            "query": {"id": query_id},
            "error": None,
            "report": {"article": f"Report for {query_id}"},
            "final_stats": {"orchestrator_model": adapter},
        }
        raw = {
            "rounds": [
                {
                    "decision": {
                        "policy": "learned",
                        "meta": {"parse_failure": False, "error": ""},
                    }
                }
            ]
        }
        (run_dir / "trajectories" / f"{query_id}.json").write_text(json.dumps(harness))
        (run_dir / "reports" / f"{query_id}.md").write_text(f"Report for {query_id}")
        (run_dir / "gpt_researcher" / f"{query_id}.json").write_text(json.dumps(raw))
    return run_dir


def test_validator_accepts_complete_bc_run(tmp_path):
    assert validate_run(_write_run(tmp_path), BC) == []


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("harness", lambda row: row.update(error="Azure unavailable"), "harness error"),
        ("harness", lambda row: row["report"].update(article=""), "empty report"),
        (
            "harness",
            lambda row: row["final_stats"].update(orchestrator_model="pilot-dpo-teacher"),
            "adapter provenance",
        ),
        (
            "raw",
            lambda row: row["rounds"][0]["decision"]["meta"].update(parse_failure=True),
            "parse_failure",
        ),
        (
            "raw",
            lambda row: row["rounds"][0]["decision"]["meta"].update(error="tunnel down"),
            "decision error",
        ),
        ("raw", lambda row: row.update(rounds=[]), "no learned decisions"),
    ],
)
def test_validator_rejects_invalid_query(tmp_path, target, mutation, message):
    run_dir = _write_run(tmp_path)
    path = run_dir / ("trajectories" if target == "harness" else "gpt_researcher") / "61.json"
    row = json.loads(path.read_text())
    mutation(row)
    path.write_text(json.dumps(row))

    assert any(message in error for error in validate_run(run_dir, BC))


def test_validator_rejects_missing_query_and_raw_trajectory(tmp_path):
    run_dir = _write_run(tmp_path)
    (run_dir / "trajectories" / "75.json").unlink()
    (run_dir / "gpt_researcher" / "74.json").unlink()

    errors = validate_run(run_dir, BC)
    assert any("missing harness trajectories: 75" in error for error in errors)
    assert any("query 74: missing raw" in error for error in errors)


def test_validator_cli_exits_nonzero_on_failure(tmp_path):
    run_dir = _write_run(tmp_path)
    (run_dir / "reports" / "61.md").write_text("")
    script = Path("scripts/validate_drb_transfer_run.py").resolve()

    result = subprocess.run(
        [sys.executable, str(script), str(run_dir), "--adapter", BC],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "report file is empty" in result.stdout


def test_handoff_pins_experiment_and_fork_configuration():
    handoff = Path("WILL_HANDOFF.md").read_text(encoding="utf-8")

    assert "aadya/drb-bc-dpo-transfer" in handoff
    assert "aadya/drb-azure-flex" in handoff
    assert "f1abd1caeefc6c55146b39afbc2dda44eedd40d8" in handoff
    assert "pilot-bc-teacher" in handoff
    assert "pilot-dpo-teacher" in handoff
    assert "gpt-6.1-sol-1gtjfzlv8i" in handoff
    assert 'service_tier="flex"' in handoff
    assert "--run-id drb-bc-sol-d2-q61-75" in handoff
    assert "--run-id drb-dpo-sol-d2-q61-75" in handoff
