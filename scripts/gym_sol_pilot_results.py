#!/usr/bin/env python3
"""Per-query results and the Table-3-style pilot table for runs/gym_sol_pilot.

Writes ``runs/gym_sol_pilot/results_per_query.csv`` with one row per arm x
query id (missing runs included, so failures show up), prints completion
counts out of the pilot ids for each metric, then prints the Table 3 columns
plus Citation via scripts/table3_row.py with every arm paired against none.
Extractive compression and the Qwen orchestrator arms (learned-base,
learned-wsp) are included once their run directory exists.

    python scripts/gym_sol_pilot_results.py [--pilot gym_sol_pilot_d2]

Generation tokens are the fork's TokenTracker totals over every LLM call of the
run (planning, research, report writing and, for prompted, the orchestrator).
Judge usage is not metered by the harness and is not in these columns.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "runs" / "gym_sol_pilot"
ARMS = {
    "none": "No pruning",
    "prompted": "Prompted orchestration",
    "extractive": "Extractive compression",
    "learned-bc": "PILOT (BC)",
    "learned-dpo": "PILOT (DPO)",
    "learned-base": "Qwen2.5-7B orchestrator",
    "learned-wsp": "Qwen2.5-7B orchestrator (weight edit)",
}
QUERY_IDS = (
    "546111,442352,883054,331361,390111,313168,170566,832775,"
    "758081,757743,567339,709678,303673,114491,997518"
).split(",")
FIELDS = (
    "policy", "query_id", "report_path", "report_words",
    "quality_overall", "clarity", "breadth", "depth", "support", "balance", "insightfulness",
    "kpr_support_rate", "kpr_omitted_rate", "kpr_contradicted_rate", "citation_score", "judge_model",
    "gen_tokens_total", "gen_prompt_tokens", "gen_completion_tokens", "orchestrator_tokens",
    "gen_llm_calls", "gen_searches", "gen_wall_s", "driver_wall_s", "rounds", "retained_final",
    "evidence_total", "failure",
)


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _manifest(policy: str) -> dict[str, dict]:
    """Last driver row per query (the manifest is append-only)."""
    rows: dict[str, dict] = {}
    path = PILOT / policy / "manifest.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["query_id"])] = row
    return rows


def row_for(policy: str, qid: str, manifest: dict[str, dict]) -> dict:
    run = PILOT / policy / f"{qid}-s0"
    out: dict = {"policy": policy, "query_id": qid}
    failures: list[str] = []
    m = manifest.get(qid)
    if m is None:
        failures.append("not run")
    elif m.get("status") == "failed":
        failures.append(f"generation failed: {m.get('error')}")
    out["driver_wall_s"] = m.get("wall_s") if m else None

    traj = _json(run / "trajectories" / f"{qid}.json")
    report = run / "reports" / f"{qid}.md"
    if traj.get("report") and report.exists():
        out["report_path"] = str(report.relative_to(ROOT))
        out["report_words"] = len(report.read_text(encoding="utf-8").split())
    elif m is not None:
        failures.append("no report")

    fs = traj.get("final_stats") or {}
    usage = fs.get("usage") or {}
    out.update(
        gen_tokens_total=usage.get("total_tokens"),
        gen_prompt_tokens=usage.get("prompt_tokens"),
        gen_completion_tokens=usage.get("completion_tokens"),
        gen_llm_calls=usage.get("n_calls"),
        gen_searches=usage.get("n_search_calls"),
        gen_wall_s=fs.get("wall_s"),
        rounds=fs.get("num_rounds"),
        retained_final=fs.get("evidence_retained_final"),
        evidence_total=fs.get("evidence_total"),
    )
    orch = (fs.get("tokens_by_tag") or {}).get("orchestrator")
    if orch:
        out["orchestrator_tokens"] = int(orch["input"]) + int(orch["output"])

    gym = ((_json(run / "metrics" / "summary.json").get("official") or {}).get("deep_research_gym")) or {}
    out["judge_model"] = gym.get("judge_model")
    quality, kpr, cite = gym.get("quality") or {}, gym.get("kpr") or {}, gym.get("citation") or {}
    if quality.get("n"):
        out["quality_overall"] = quality["average_normalized_score"]
        for name, row in (quality.get("per_criterion") or {}).items():
            out[name.lower()] = row["average_rating"]
    if kpr.get("n"):
        out["kpr_support_rate"] = kpr["average_support_rate"]
        out["kpr_omitted_rate"] = kpr["average_omitted_rate"]
        out["kpr_contradicted_rate"] = kpr["average_contradicted_rate"]
    if cite.get("n"):
        out["citation_score"] = cite["average_citation_score"]
    if out.get("report_path"):
        for metric, block in (("quality", quality), ("kpr", kpr), ("citation", cite)):
            if not block.get("n"):
                failures.append(f"{metric} not judged" + (f": {block['error']}" if block.get("error") else ""))
    out["failure"] = "; ".join(failures)
    return out


def main() -> None:
    global PILOT
    if "--pilot" in sys.argv:
        PILOT = ROOT / "runs" / sys.argv[sys.argv.index("--pilot") + 1]
    # The two Sol arms always; the Qwen arms once they have been started.
    arms = {a: label for a, label in ARMS.items() if a in ("none", "prompted") or (PILOT / a).is_dir()}
    rows = []
    for arm in arms:
        manifest = _manifest(arm)
        rows.extend(row_for(arm, qid, manifest) for qid in QUERY_IDS)

    out_csv = PILOT / "results_per_query.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in FIELDS} for r in rows)
    print(f"wrote {out_csv.relative_to(ROOT)}\n")

    n = len(QUERY_IDS)
    print(f"| Completed (of {n}) | " + " | ".join(arms.values()) + " |")
    print("|---|" + "---|" * len(arms))
    for label, key in (("Report", "report_path"), ("Quality", "quality_overall"),
                       ("KPR", "kpr_support_rate"), ("Citation", "citation_score")):
        counts = [sum(1 for r in rows if r["policy"] == a and r.get(key) is not None) for a in arms]
        print(f"| {label} | " + " | ".join(str(c) for c in counts) + " |")
    print()

    corpora = [f"{label}={PILOT / arm}" for arm, label in arms.items()
               if any(r["policy"] == arm and r.get("citation_score") is not None for r in rows)]
    if not any(c.startswith(ARMS["none"] + "=") for c in corpora) or len(corpora) < 2:
        print("Table: skipped until no pruning and another arm have runs with quality, KPR and citation scores.")
        return
    table = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "table3_row.py"), *corpora,
         "--citation", "--baseline", ARMS["none"]],
        check=True, capture_output=True, text=True,
    ).stdout
    out_md = PILOT / "table3.md"
    out_md.write_text(
        f"# DeepResearchGym pilot on GPT-6.1 Sol: {PILOT.name} (Table 3 columns + Citation)\n\n"
        f"Judges: gpt-5-mini (quality, KPR, citation). Per-query rows: {out_csv.name}.\n\n" + table,
        encoding="utf-8",
    )
    print(table)
    print(f"wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
