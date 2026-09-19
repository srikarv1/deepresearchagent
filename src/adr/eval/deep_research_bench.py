"""DeepResearch Bench evaluation via the upstream RACE and FACT pipelines.

RACE scores a report against a reference article, matched by exact prompt
string, so exported prompts must come from the benchmark's own query file. FACT
extracts citations, scrapes them, and validates each claim.

Both judges use an OpenAI-compatible backend (``utils/api.py``), selected by
``LLM_BACKEND``: ``openai`` (needs ``OPENAI_API_KEY``) or ``openrouter``
(default, needs ``OPENROUTER_API_KEY``). FACT additionally scrapes through Jina
and needs ``JINA_API_KEY``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.eval.exporters import export_deep_research_bench
from adr.eval.procs import run_script
from adr.eval.repos import find_deep_research_bench
from adr.eval.scoring import parse_key_value_report

SCRAPE_KEY = "JINA_API_KEY"


def _judge_key() -> tuple[str, str]:
    """Return (key_env_name, backend_name) based on LLM_BACKEND."""
    backend = os.environ.get("LLM_BACKEND", "openrouter").lower()
    if backend == "openai":
        return "OPENAI_API_KEY", backend
    return "OPENROUTER_API_KEY", backend


def run_deep_research_bench(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    third_party_dir: str | Path | None = None,
    language: str = "en",
    workers: int = 4,
    skip_cleaning: bool = False,
    run_race: bool = True,
    run_fact: bool = True,
    extra_env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Write the DRB raw file, then run RACE and FACT and read their scores back."""
    export_path = run_dir / "exports" / "deep_research_bench" / f"{model_name}.jsonl"
    export_deep_research_bench(trajectories, export_path)

    result: dict[str, Any] = {
        "bench": "deep_research_bench",
        "export": str(export_path),
        "official": False,
        "reason": None,
        "race": None,
        "fact": None,
    }

    located = find_deep_research_bench(third_party_dir)
    if not located.ok:
        result["reason"] = located.reason
        return result
    bench_root = located.path
    result["repo"] = str(bench_root)

    env = dict(extra_env or {})
    key_name, backend = _judge_key()
    if not (os.environ.get(key_name) or env.get(key_name)):
        result["reason"] = (
            f"{key_name} is not set (LLM_BACKEND={backend}). "
            f"Set {key_name}, or switch backend via LLM_BACKEND=openai|openrouter."
        )
        return result

    # The judges read the target report from the raw data directory keyed by model name.
    dest = bench_root / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(export_path.read_text(encoding="utf-8"), encoding="utf-8")
    result["staged"] = str(dest)

    query_file = bench_root / "data" / "prompt_data" / "query.jsonl"

    if run_race:
        result["race"] = _race(
            bench_root, model_name, query_file, language, workers, skip_cleaning, env, timeout_s,
            run_dir=run_dir,
        )
    if run_fact:
        result["fact"] = _fact(bench_root, model_name, dest, query_file, workers, env, timeout_s)

    scored = [
        block
        for block in (result["race"], result["fact"])
        if isinstance(block, dict) and block.get("scores")
    ]
    result["official"] = bool(scored)
    if not result["official"]:
        result["reason"] = "Neither RACE nor FACT produced scores; see the per-metric log fields"
    return result


def _race(
    bench_root: Path,
    model_name: str,
    query_file: Path,
    language: str,
    workers: int,
    skip_cleaning: bool,
    env: dict[str, str],
    timeout_s: float | None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = bench_root / "results" / "race" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # RACE appends to raw_results.jsonl and skips ids it already scored, so a
    # stale file from an earlier run of the same model name would silently
    # reuse old scores. Start clean; the per-run copy below keeps history.
    stale = out_dir / "raw_results.jsonl"
    if stale.exists():
        stale.unlink()
    args: list[Any] = [
        bench_root / "deepresearch_bench_race.py",
        model_name,
        "--raw_data_dir",
        bench_root / "data" / "test_data" / "raw_data",
        "--max_workers",
        workers,
        "--query_file",
        query_file,
        "--output_dir",
        out_dir,
    ]
    if language == "en":
        args.append("--only_en")
    elif language == "zh":
        args.append("--only_zh")
    if skip_cleaning:
        args.append("--skip_cleaning")

    log = run_script(args, cwd=bench_root, env=env, timeout_s=timeout_s)
    result_file = out_dir / "race_result.txt"
    raw_results = out_dir / "raw_results.jsonl"
    out: dict[str, Any] = {
        "scores": parse_key_value_report(result_file),
        "path": str(result_file),
        "raw_results": str(raw_results),
        "log": log,
    }
    # Per-query scores are the per-trajectory quality term Q(y, q) for PILOT
    # rewards. The upstream file lives under results/race/<model_name>/ and is
    # overwritten by the next evaluation with the same model name, so copy it
    # into the run directory and surface the mapping in the summary.
    per_query = read_race_raw_results(raw_results)
    if per_query:
        out["per_query"] = per_query
    if run_dir is not None and raw_results.exists():
        dest = Path(run_dir) / "metrics" / "race_raw_results.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(raw_results.read_text(encoding="utf-8"), encoding="utf-8")
        out["raw_results_copy"] = str(dest)
    return out


def read_race_raw_results(path: Path) -> dict[str, dict[str, float]]:
    """``{query_id: {overall_score, comprehensiveness, insight, instruction_following, readability}}``
    from RACE's ``raw_results.jsonl``. Rows carrying an ``error`` are skipped."""
    import json

    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("error") or "overall_score" not in row:
            continue
        qid = str(row.get("id"))
        rows[qid] = {
            k: float(row[k])
            for k in ("overall_score", "comprehensiveness", "insight", "instruction_following", "readability")
            if k in row and isinstance(row[k], (int, float))
        }
    return rows


def _fact(
    bench_root: Path,
    model_name: str,
    raw_path: Path,
    query_file: Path,
    workers: int,
    env: dict[str, str],
    timeout_s: float | None,
) -> dict[str, Any]:
    if not (os.environ.get(SCRAPE_KEY) or env.get(SCRAPE_KEY)):
        return {
            "skipped": True,
            "reason": f"{SCRAPE_KEY} is not set; FACT scrapes cited pages through Jina",
        }

    out_dir = bench_root / "results" / "fact" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stages: list[tuple[str, list[Any]]] = [
        (
            "extract",
            ["-m", "utils.extract", "--raw_data_path", raw_path,
             "--output_path", out_dir / "extracted.jsonl",
             "--query_data_path", query_file, "--n_total_process", workers],
        ),
        (
            "deduplicate",
            ["-m", "utils.deduplicate", "--raw_data_path", out_dir / "extracted.jsonl",
             "--output_path", out_dir / "deduplicated.jsonl",
             "--query_data_path", query_file, "--n_total_process", workers],
        ),
        (
            "scrape",
            ["-m", "utils.scrape", "--raw_data_path", out_dir / "deduplicated.jsonl",
             "--output_path", out_dir / "scraped.jsonl", "--n_total_process", workers],
        ),
        (
            "validate",
            ["-m", "utils.validate", "--raw_data_path", out_dir / "scraped.jsonl",
             "--output_path", out_dir / "validated.jsonl",
             "--query_data_path", query_file, "--n_total_process", workers],
        ),
        (
            "stat",
            ["-m", "utils.stat", "--input_path", out_dir / "validated.jsonl",
             "--output_path", out_dir / "fact_result.txt"],
        ),
    ]

    logs: list[dict[str, Any]] = []
    for name, args in stages:
        log = run_script(args, cwd=bench_root, env=env, timeout_s=timeout_s)
        logs.append({"stage": name, **log})
        if not log["ok"]:
            break

    result_file = out_dir / "fact_result.txt"
    return {
        "scores": parse_key_value_report(result_file),
        "path": str(result_file),
        "stages": logs,
    }
