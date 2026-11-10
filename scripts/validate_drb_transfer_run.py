#!/usr/bin/env python3
"""Validate one completed DRB BC/DPO transfer run without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_QUERY_IDS = {str(query_id) for query_id in range(61, 76)}
ALLOWED_ADAPTERS = {"pilot-bc-teacher", "pilot-dpo-teacher"}


def _read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path}: cannot read valid JSON ({exc})")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path}: expected a JSON object")
        return None
    return value


def validate_run(run_dir: Path, adapter: str) -> list[str]:
    """Return validation errors; an empty list means the run is complete."""
    errors: list[str] = []
    run_dir = run_dir.expanduser().resolve()
    if adapter not in ALLOWED_ADAPTERS:
        return [f"unsupported adapter {adapter!r}; expected one of {sorted(ALLOWED_ADAPTERS)}"]
    if not run_dir.is_dir():
        return [f"run directory does not exist: {run_dir}"]

    trajectory_dir = run_dir / "trajectories"
    actual_ids = {path.stem for path in trajectory_dir.glob("*.json")}
    missing = sorted(EXPECTED_QUERY_IDS - actual_ids, key=int)
    unexpected = sorted(actual_ids - EXPECTED_QUERY_IDS)
    if missing:
        errors.append(f"missing harness trajectories: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected harness trajectories: {', '.join(unexpected)}")

    for query_id in sorted(EXPECTED_QUERY_IDS, key=int):
        harness_path = trajectory_dir / f"{query_id}.json"
        if not harness_path.is_file():
            continue
        harness = _read_json(harness_path, errors)
        if harness is None:
            continue

        recorded_id = str((harness.get("query") or {}).get("id", ""))
        if recorded_id != query_id:
            errors.append(f"query {query_id}: harness records query id {recorded_id!r}")
        if harness.get("error"):
            errors.append(f"query {query_id}: harness error: {harness['error']}")

        report = harness.get("report") or {}
        article = report.get("article", "") if isinstance(report, dict) else ""
        if not isinstance(article, str) or not article.strip():
            errors.append(f"query {query_id}: empty report in harness trajectory")
        report_path = run_dir / "reports" / f"{query_id}.md"
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError:
            errors.append(f"query {query_id}: missing report file {report_path}")
        else:
            if not report_text.strip():
                errors.append(f"query {query_id}: report file is empty")

        provenance = (harness.get("final_stats") or {}).get("orchestrator_model")
        if provenance != adapter:
            errors.append(
                f"query {query_id}: adapter provenance is {provenance!r}, expected {adapter!r}"
            )

        raw_path = run_dir / "gpt_researcher" / f"{query_id}.json"
        if not raw_path.is_file():
            errors.append(f"query {query_id}: missing raw GPT-Researcher trajectory")
            continue
        raw = _read_json(raw_path, errors)
        if raw is None:
            continue

        learned_decisions: list[dict[str, Any]] = []
        for round_index, round_row in enumerate(raw.get("rounds") or [], start=1):
            decision = round_row.get("decision") if isinstance(round_row, dict) else None
            if isinstance(decision, dict) and decision.get("policy") == "learned":
                learned_decisions.append(decision)
                meta = decision.get("meta") or {}
                if meta.get("parse_failure") is not False:
                    errors.append(f"query {query_id} round {round_index}: learned parse_failure")
                if meta.get("error"):
                    errors.append(
                        f"query {query_id} round {round_index}: learned decision error: "
                        f"{meta['error']}"
                    )
        if not learned_decisions:
            errors.append(f"query {query_id}: no learned decisions in raw trajectory")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--adapter", required=True, choices=sorted(ALLOWED_ADAPTERS))
    args = parser.parse_args()

    errors = validate_run(args.run_dir, args.adapter)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"PASS: {args.run_dir} contains all 15 DRB queries with valid learned decisions "
        f"from {args.adapter}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
