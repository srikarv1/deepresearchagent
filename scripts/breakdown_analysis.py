#!/usr/bin/env python3
"""Where each method spends its tokens and its time (P1 breakdown analysis).

    python scripts/breakdown_analysis.py --root runs/gym_sol_pilot_d2
    python scripts/breakdown_analysis.py --arm none=runs/gym_sol_pilot_d2/none --arm bc=/data/drg50/bc
    python scripts/breakdown_analysis.py --root runs/gym_sol_budget150 --common --csv drg_split.csv --latex drg_split.tex

An arm is a directory of ``adr rollouts`` runs (``<arm>/<query>-s<seed>/``). With
``--root`` every subdirectory that holds runs is an arm, named after it.

Token split (per query, then the mean over queries), from the fork's per-stage
token counts (``final_stats.tokens_by_tag``), which add up to the run's total:

  Research       planning (deep_research_plan), search-query generation
                 (deep_search_query_gen), agent selection, sub-query generation
                 and result processing (deep_process_results)
  Orchestrator   the policy's own LLM calls (prompted, BC/DPO adapters, TypeSafe)
  Writing        the report (report_generation and its section calls)

Latency split (seconds, per query), from the harness wall clock and the fork's
round and report timings. The three phases add up to the wall time:

  Setup/plan     wall - research rounds - writing (research plan, embeddings, setup)
  Rounds         sum of the research rounds' latency (search, LLM, pruning)
  Writing        the report call

and the research part (Setup/plan + Rounds) is split by resource:

  Search wait    time waiting on the search API (FineWeb / Tavily)
  Research LLM   LLM time outside the report (planning, sub-queries, processing,
                 orchestrator)
  Other          the rest: embeddings, pruning / compression, overhead

Runs with the per-call LLM log (``final_stats.llm_calls``, budget runs) also get
exact LLM seconds per stage, the orchestrator's LLM time, and Azure Flex
waiting: refused (429 no_capacity retries, measured) and, when
``<root>/queue_adjusted.json`` from scripts/flex_queue_adjust.py exists, queued
(estimated). Older runs leave those columns empty.

All splits assume concurrency 1, as in the pilot and budget configs, so stage
times do not overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Iterable

# usage_tag -> group. Anything unknown is "other" so the groups still add up.
TAG_GROUPS = {
    "deep_research_plan": "research",
    "deep_search_query_gen": "research",
    "agent_selection": "research",
    "subquery_generation": "research",
    "deep_process_results": "research",
    "url_summary": "research",
    "orchestrator": "orchestrator",
    "report_generation": "writing",
    "report_introduction": "writing",
    "report_conclusion": "writing",
    "draft_titles": "writing",
}
# Research stages shown on their own, in pipeline order.
RESEARCH_STAGES = {
    "deep_research_plan": "plan",
    "deep_search_query_gen": "query_gen",
    "agent_selection": "agent_select",
    "subquery_generation": "subquery_gen",
    "deep_process_results": "process",
}
ARM_LABELS = {
    "none": "No pruning",
    "prompted": "Prompted orchestration",
    "extractive": "Extractive compression",
    "bc": "PILOT (BC)",
    "dpo": "PILOT (DPO)",
}
ARM_ORDER = ("none", "prompted", "extractive", "bc", "dpo")

TOKEN_COLUMNS = (
    "tok_research", "tok_orchestrator", "tok_writing", "tok_other", "tok_total",
    *(f"tok_{s}" for s in RESEARCH_STAGES.values()),
)
LATENCY_COLUMNS = (
    "lat_setup", "lat_rounds", "lat_writing", "lat_wall",
    "lat_search", "lat_research_llm", "lat_research_other",
    # per-call log only
    "lat_llm_research", "lat_llm_orchestrator", "lat_llm_writing",
    *(f"lat_llm_{s}" for s in RESEARCH_STAGES.values()),
    "lat_flex_refused", "lat_flex_queued",
)


# ── per query ─────────────────────────────────────────────────────────────


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def token_split(final_stats: dict) -> dict[str, float]:
    """Token counts per group and per research stage, from tokens_by_tag."""
    out = {c: 0.0 for c in TOKEN_COLUMNS}
    for tag, row in (final_stats.get("tokens_by_tag") or {}).items():
        n = float((row or {}).get("input", 0) or 0) + float((row or {}).get("output", 0) or 0)
        out[f"tok_{TAG_GROUPS.get(tag, 'other')}"] += n
        if tag in RESEARCH_STAGES:
            out[f"tok_{RESEARCH_STAGES[tag]}"] += n
        out["tok_total"] += n
    return out


def call_log_split(calls: list[dict]) -> dict[str, float]:
    """Exact LLM seconds per group / research stage and Flex refusal waiting,
    from the per-call log."""
    out = {c: 0.0 for c in LATENCY_COLUMNS if c.startswith("lat_llm_") or c == "lat_flex_refused"}
    for call in calls:
        dur = _num(call.get("duration_s")) or 0.0
        tag = call.get("usage_tag") or "unspecified"
        group = TAG_GROUPS.get(tag, "research")  # untagged calls happen during research
        out[f"lat_llm_{group}"] += dur
        if tag in RESEARCH_STAGES:
            out[f"lat_llm_{RESEARCH_STAGES[tag]}"] += dur
        out["lat_flex_refused"] += _num(call.get("capacity_wait_s")) or 0.0
    return out


def latency_split(final_stats: dict, rounds_latency: float | None) -> dict[str, float | None]:
    """Phase and resource split of the run's wall time (see module docstring)."""
    usage = final_stats.get("usage") or {}
    wall = _num(final_stats.get("wall_s")) or _num(final_stats.get("agent_wall_s"))
    writing = _num(final_stats.get("synthesis_latency_s"))
    search = _num(usage.get("search_latency_s"))
    llm = _num(usage.get("llm_latency_s"))
    out: dict[str, float | None] = {c: None for c in LATENCY_COLUMNS}
    out.update(lat_wall=wall, lat_writing=writing, lat_rounds=rounds_latency, lat_search=search)
    if wall is not None and writing is not None and rounds_latency is not None:
        out["lat_setup"] = max(0.0, wall - rounds_latency - writing)
    if llm is not None and writing is not None:
        out["lat_research_llm"] = max(0.0, llm - writing)
    if wall is not None and writing is not None and search is not None and out["lat_research_llm"] is not None:
        out["lat_research_other"] = max(0.0, wall - writing - search - out["lat_research_llm"])
    calls = final_stats.get("llm_calls")
    if isinstance(calls, list) and calls:
        out.update(call_log_split(calls))
    return out


def _rounds_latency(run_dir: Path, qid: str) -> float | None:
    path = run_dir / "gpt_researcher" / f"{qid}.json"
    if not path.exists():
        return None
    try:
        rounds = json.loads(path.read_text(encoding="utf-8")).get("rounds") or []
    except (OSError, json.JSONDecodeError):
        return None
    return float(sum(_num((r.get("round_cost") or {}).get("latency_seconds")) or 0.0 for r in rounds))


def _queue_adjusted(arm_dir: Path) -> dict[str, dict]:
    """``{run_name: row}`` for this arm from <root>/queue_adjusted.json, if any."""
    path = arm_dir.parent / "queue_adjusted.json"
    if not path.exists():
        return {}
    try:
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    return runs.get(arm_dir.name) or {}


def load_arm(arm_dir: Path) -> list[dict[str, Any]]:
    """One row per finished run (trajectory with a report and no error)."""
    rows = []
    queued = _queue_adjusted(arm_dir)
    for run_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir() and "-s" in p.name):
        qid = run_dir.name.rsplit("-s", 1)[0]
        traj_path = run_dir / "trajectories" / f"{qid}.json"
        if not traj_path.exists():
            continue
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not traj.get("report") or traj.get("error"):
            continue
        fs = traj.get("final_stats") or {}
        row: dict[str, Any] = {"query": qid, "run": run_dir.name}
        row.update(token_split(fs))
        row.update(latency_split(fs, _rounds_latency(run_dir, qid)))
        q = queued.get(run_dir.name) or queued.get(qid) or {}
        if _num(q.get("queue_s")) is not None:
            row["lat_flex_queued"] = float(q["queue_s"])
        rows.append(row)
    return rows


# ── arms and summaries ────────────────────────────────────────────────────


def find_arms(root: Path) -> dict[str, Path]:
    arms = {
        p.name: p for p in sorted(root.iterdir())
        if p.is_dir() and any(c.is_dir() and (c / "trajectories").is_dir() for c in p.iterdir())
    }
    return order_arms(arms)


def order_arms(arms: dict[str, Path]) -> dict[str, Path]:
    known = [a for a in ARM_ORDER if a in arms]
    return {a: arms[a] for a in known + sorted(set(arms) - set(known))}


def restrict_to_common(data: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Keep only queries every arm finished, so means compare like with like."""
    sets = [{r["query"] for r in rows} for rows in data.values()]
    common = set.intersection(*sets) if sets else set()
    return {arm: [r for r in rows if r["query"] in common] for arm, rows in data.items()}


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.mean(vals) if vals else None


def summarize(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for col in TOKEN_COLUMNS + LATENCY_COLUMNS:
        out[col] = mean(r.get(col) for r in rows)
    # Shares of the total, averaged per query (not a ratio of means).
    for group in ("research", "orchestrator", "writing"):
        out[f"share_tok_{group}"] = mean(
            r[f"tok_{group}"] / r["tok_total"] for r in rows if r.get("tok_total")
        )
    for phase in ("setup", "rounds", "writing", "search"):
        out[f"share_lat_{phase}"] = mean(
            r[f"lat_{phase}"] / r["lat_wall"] for r in rows
            if r.get(f"lat_{phase}") is not None and r.get("lat_wall")
        )
    return out


# ── output ────────────────────────────────────────────────────────────────


def _k(v: float | None) -> str:
    return "-" if v is None else f"{v / 1000:.1f}"


def _s(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.0f}%"


def _label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


TOKEN_TABLE = (
    ("Research", "tok_research", _k), ("Orchestrator", "tok_orchestrator", _k),
    ("Writing", "tok_writing", _k), ("Total", "tok_total", _k),
    ("Writing %", "share_tok_writing", _pct),
)
STAGE_TABLE = tuple((s.replace("_", " "), f"tok_{s}", _k) for s in RESEARCH_STAGES.values())
LATENCY_TABLE = (
    ("Setup/plan", "lat_setup", _s), ("Rounds", "lat_rounds", _s), ("Writing", "lat_writing", _s),
    ("Wall", "lat_wall", _s), ("Search wait", "lat_search", _s),
    ("Research LLM", "lat_research_llm", _s), ("Other", "lat_research_other", _s),
    ("Search %", "share_lat_search", _pct), ("Writing %", "share_lat_writing", _pct),
)
CALL_LOG_TABLE = (
    ("LLM research", "lat_llm_research", _s), ("LLM orchestrator", "lat_llm_orchestrator", _s),
    ("LLM writing", "lat_llm_writing", _s), ("Flex refused", "lat_flex_refused", _s),
    ("Flex queued", "lat_flex_queued", _s),
)


def render(summaries: dict[str, dict], table, title: str) -> str:
    headers = ["Method", "n", *(h for h, _, _ in table)]
    body = [[_label(a), str(s["n"]), *(fmt(s.get(col)) for _, col, fmt in table)] for a, s in summaries.items()]
    widths = [max(len(r[i]) for r in [headers, *body]) for i in range(len(headers))]
    line = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    return "\n".join([title, line(headers), *(line(r) for r in body)])


def latex(summaries: dict[str, dict], table, caption: str) -> str:
    cols = "l" + "r" * (len(table) + 1)
    head = " & ".join(["Method", "$n$", *(h.replace("%", r"\%") for h, _, _ in table)])
    rows = [
        " & ".join([_label(a), str(s["n"]), *(fmt(s.get(col)).replace("%", r"\%") for _, col, fmt in table)]) + r" \\"
        for a, s in summaries.items()
    ]
    return "\n".join([
        r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def has_call_log(summaries: dict[str, dict]) -> bool:
    return any(s.get("lat_llm_research") is not None for s in summaries.values())


def write_csv(path: Path, data: dict[str, list[dict]]) -> None:
    cols = ["arm", "query", "run", *TOKEN_COLUMNS, *LATENCY_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for arm, rows in data.items():
            for r in rows:
                w.writerow({"arm": arm, **{k: ("" if v is None else v) for k, v in r.items()}})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="directory whose subdirectories are arms")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR", help="an arm directory (repeatable)")
    ap.add_argument("--arms", nargs="+", help="with --root: only these arms")
    ap.add_argument("--common", action="store_true", help="only queries every arm finished")
    ap.add_argument("--csv", type=Path, help="write the per-query rows here")
    ap.add_argument("--latex", type=Path, help="write LaTeX tables here")
    ap.add_argument("--json", type=Path, help="write the per-arm summaries here")
    args = ap.parse_args(argv)

    arms: dict[str, Path] = {}
    if args.root:
        arms.update(find_arms(args.root))
        if args.arms:
            arms = {a: p for a, p in arms.items() if a in args.arms}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--arm takes NAME=DIR, got {spec!r}")
        arms[name] = Path(path)
    arms = order_arms(arms)
    if not arms:
        ap.error("no arms: pass --root or --arm NAME=DIR")

    data = {arm: load_arm(path) for arm, path in arms.items()}
    if args.common:
        data = restrict_to_common(data)
    counts = {arm: len(rows) for arm, rows in data.items()}
    if len(set(counts.values())) > 1:
        print(f"note: arms cover different query counts {counts}; --common compares the shared queries only\n",
              file=sys.stderr)
    summaries = {arm: summarize(rows) for arm, rows in data.items()}

    blocks = [
        (TOKEN_TABLE, "Tokens per query (K), mean over queries",
         "Token split per query: research, orchestrator and report writing (thousands of tokens)."),
        (STAGE_TABLE, "Research tokens by stage (K)", "Research tokens per query by stage (thousands of tokens)."),
        (LATENCY_TABLE, "Latency per query (s): phases add up to Wall; Search wait + Research LLM + Other = Setup/plan + Rounds",
         "Latency split per query (seconds)."),
    ]
    if has_call_log(summaries):
        blocks.append((CALL_LOG_TABLE, "LLM seconds by stage and Azure Flex waiting (per-call log)",
                       "LLM time by stage and Azure Flex waiting per query (seconds)."))
    print("\n\n".join(render(summaries, t, title) for t, title, _ in blocks))

    if args.latex:
        args.latex.write_text("\n\n".join(latex(summaries, t, cap) for t, _, cap in blocks) + "\n", encoding="utf-8")
    if args.csv:
        write_csv(args.csv, data)
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
