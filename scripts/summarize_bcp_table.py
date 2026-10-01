#!/usr/bin/env python3
"""Collapse overnight BCP runs into a meeting table (markdown + json) and zip trajectories."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
NAME_RE = re.compile(r"bcp-overnight-(bm25|dense)-([a-z0-9]+)$")
LIBRARIANS = ("bm25", "dense")
POLICIES = ("none", "topk", "extractive", "llmlingua", "prompted")
ZIP_PARTS = ("trajectories", "labels", "gpt_researcher", "reports", "metrics")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: float | None, *, already_pct: bool = False) -> str:
    if v is None:
        return "—"
    x = float(v)
    if not already_pct and 0.0 <= x <= 1.0:
        x *= 100.0
    return f"{x:.1f}"


def _mean_flag(per_query: list[dict], key: str) -> float | None:
    vals = [1.0 if row.get(key) else 0.0 for row in per_query if key in row]
    if not vals:
        return None
    return sum(vals) / len(vals)


def empty_row(retriever: str, policy: str) -> dict:
    return {
        "retriever": retriever,
        "policy": policy,
        "run_dir": None,
        "n": None,
        "n_errors": None,
        "acc": None,
        "recall": None,
        "judge": None,
        "official_acc": False,
        "retrieved_recall_evidence": None,
        "keep_recall_evidence": None,
        "retrieved_recall_gold": None,
        "keep_recall_gold": None,
        "answer_in_report": None,
        "mean_wall_s": None,
        "mean_n_llm_calls": None,
        "mean_n_searches": None,
        "mean_tokens": None,
    }


def collect_found(runs_dir: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    if not runs_dir.exists():
        return found
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        m = NAME_RE.search(path.name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        prev = found.get(key)
        if prev is None or path.name > prev.name:
            found[key] = path
    return found


def row_from_run(retriever: str, policy: str, path: Path) -> dict:
    summary_path = path / "metrics" / "summary.json"
    local_path = path / "metrics" / "local.json"
    summary = _load(summary_path) if summary_path.exists() else {}
    local = _load(local_path) if local_path.exists() else summary
    per_query = local.get("per_query") or []
    official = (summary.get("official") or {}).get("browsecomp_plus") or {}
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    return {
        "retriever": retriever,
        "policy": policy,
        "run_dir": rel,
        "n": summary.get("n_queries") or local.get("n_queries") or len(per_query),
        "n_errors": local.get("n_errors"),
        "acc": official.get("accuracy"),
        "recall": official.get("recall"),
        "judge": official.get("judge_model"),
        "official_acc": bool(official.get("official")),
        "retrieved_recall_evidence": local.get("mean_retrieved_recall_evidence"),
        "keep_recall_evidence": local.get("mean_keep_recall_evidence"),
        "retrieved_recall_gold": local.get("mean_retrieved_recall_gold"),
        "keep_recall_gold": local.get("mean_keep_recall_gold"),
        "answer_in_report": _mean_flag(per_query, "answer_in_report"),
        "mean_wall_s": local.get("mean_wall_s"),
        "mean_n_llm_calls": local.get("mean_n_llm_calls"),
        "mean_n_searches": local.get("mean_n_searches"),
        "mean_tokens": local.get("mean_tokens"),
    }


def collect_rows(runs_dir: Path) -> list[dict]:
    found = collect_found(runs_dir)
    rows = []
    for retriever in LIBRARIANS:
        for policy in POLICIES:
            path = found.get((retriever, policy))
            if path is None:
                rows.append(empty_row(retriever, policy))
            else:
                rows.append(row_from_run(retriever, policy, path))
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "# BrowseComp-Plus overnight table",
        "",
        "Frozen gpt-researcher **depth=3, breadth=2**, test split slice.",
        "Librarians: **BM25** and **Qwen3-Embedding-8B**. Orchestrators: none / topk / extractive / llmlingua / prompted.",
        "Acc is **gpt-5-mini** grader (unofficial; not gpt-4.1, not the 830-query number).",
        "Recall (official def.) = retrieved ∩ evidence_docs. Keep-recall is after `GR_ORCHESTRATOR`.",
        "",
        "| librarian | GR_ORCHESTRATOR | n | Acc* | Recall | keep-R | gold keep-R | ans in report | min/q | LLM calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        wall = r.get("mean_wall_s")
        mins = f"{wall / 60:.1f}" if isinstance(wall, (int, float)) else "—"
        calls = r.get("mean_n_llm_calls")
        calls_s = f"{calls:.0f}" if isinstance(calls, (int, float)) else "—"
        lines.append(
            "| {retriever} | {policy} | {n} | {acc} | {rec} | {keep} | {gkeep} | {ans} | {mins} | {calls} |".format(
                retriever=r["retriever"],
                policy=r["policy"],
                n=r["n"] if r.get("n") not in (None, "") else "—",
                acc=_pct(r.get("acc"), already_pct=True) if r.get("acc") is not None else "—",
                rec=_pct(r.get("recall"), already_pct=True)
                if r.get("recall") is not None
                else _pct(r.get("retrieved_recall_evidence")),
                keep=_pct(r.get("keep_recall_evidence")),
                gkeep=_pct(r.get("keep_recall_gold")),
                ans=_pct(r.get("answer_in_report")),
                mins=mins,
                calls=calls_s,
            )
        )
    lines.extend(
        [
            "",
            "\\*Unofficial Acc. Caption as gpt-5-mini / test-slice / depth=3×breadth=2.",
            "",
            "## run dirs",
            "",
        ]
    )
    for r in rows:
        lines.append(f"- {r['retriever']}/{r['policy']}: `{r['run_dir'] or 'pending'}`")
    lines.append("")
    return "\n".join(lines)


def zip_trajectories(runs_dir: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for run in sorted(runs_dir.glob("*-bcp-overnight-*")):
            if not run.is_dir():
                continue
            for part in ZIP_PARTS:
                folder = run / part
                if not folder.exists():
                    continue
                for path in folder.rglob("*"):
                    if path.is_file():
                        zf.write(path, path.relative_to(runs_dir).as_posix())
                        n_files += 1
    return n_files


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "overnight-table" / "TABLE.md")
    p.add_argument("--zip", type=Path, default=None, help="Write trajectories zip here")
    args = p.parse_args()
    rows = collect_rows(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows)
    args.out.write_text(md, encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {args.out} and {json_path}", flush=True)
    if args.zip:
        n = zip_trajectories(args.runs, args.zip)
        print(f"zipped {n} files -> {args.zip}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
