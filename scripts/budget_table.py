#!/usr/bin/env python3
"""Results table for the token-budget experiment (scripts/sol_budget.sh).

    python scripts/budget_table.py drg
    python scripts/budget_table.py drb --baseline none
    python scripts/budget_table.py drg --root runs/gym_sol_budget150 --arms none bc dpo

One row per arm, per-query means over the runs that have every judge score:

  drg   Clarity Breadth Depth Support KPR KPC Overall Citation   (DeepResearchGym judges;
        KPC = key-point contradiction rate, lower is better)
  drb   RACE Comp. Insight Instr. Read. FACT valid           (DeepResearch Bench judges)
  both  Tokens (K)   total tokens of the run, every model
        Latency (s)  wall time with Azure Flex waiting removed, when
                     <root>/queue_adjusted.json exists (scripts/flex_queue_adjust.py)
        Model (s)    time at least one model call was running (union of the
                     call intervals in final_stats.llm_calls, Flex refusals
                     excluded); unlike wall time it does not depend on the
                     machine or network the run was generated on
        Calls        search calls
        Q per 1K t   headline score / Tokens (K)
        Stopped      runs the stopping rule stopped (final_stats.token_stop.stopped)
        Over         runs that ended above the target (final_stats.token_stop.ended_above_target);
                     0 with the hard cap (GR_STOP_HARD_CAP) unless research alone used it up
        Trimmed      runs whose report context the hard cap trimmed to fit

Then, for every other arm, per-query paired differences against ``--baseline``
over the queries both have, with a bootstrap 95% interval on the mean.

With ``JUDGE_PASS=N bash scripts/sol_budget_judge.sh`` re-judgings present
(``<root>_pass<N>``), each judge score is the median over passes per report.

``--breakdown`` needs no judge scores: per arm, where the tokens and model time
went by pipeline stage (plan, queries, selection, processing, report), research
vs. report tokens, $ cost, and the token curve the same stopping rule gives at
smaller budgets, replayed from the call log (deterministic, no API calls).

Runs whose first round, or at least half of their rounds, found no evidence
are listed as a warning (a retriever outage makes a run look cheap); ``--drop-starved`` leaves them out.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from adr.eval.latency import union_seconds

BENCHES = {
    "drg": {
        "root": "runs/gym_sol_budget150",
        "headline": "Overall",
        "columns": ("Clarity", "Breadth", "Depth", "Support", "KPR", "KPC", "Overall", "Citation"),
    },
    "drb": {
        "root": "runs/drb_sol_budget150",
        "headline": "RACE",
        "columns": ("RACE", "Comp.", "Insight", "Instr.", "Read.", "FACT valid"),
    },
}
COST_COLUMNS = ("Tokens (K)", "Latency (s)", "Model (s)", "Calls")
ARM_LABELS = {"none": "No pruning", "prompted": "Prompted orchestration", "extractive": "Extractive compression",
              "bc": "PILOT (BC)", "dpo": "PILOT (DPO)"}
ARM_ORDER = ("none", "prompted", "extractive", "bc", "dpo")
LOWER_IS_BETTER = {"Tokens (K)", "Latency (s)", "Model (s)", "Calls", "KPC"}


def gym_scores(official: dict) -> dict[str, float] | None:
    gym = official.get("deep_research_gym") or {}
    quality, kpr, cite = gym.get("quality") or {}, gym.get("kpr") or {}, gym.get("citation") or {}
    if not quality.get("n") or not kpr.get("n") or not cite.get("n"):
        return None
    crit = quality.get("per_criterion") or {}
    row = {c: float(crit[c]["average_rating"]) for c in ("Clarity", "Breadth", "Depth", "Support")}
    row["KPR"] = float(kpr["average_support_rate"])
    row["KPC"] = float(kpr.get("average_contradicted_rate") or 0.0)
    row["Overall"] = float(quality["average_normalized_score"])
    row["Citation"] = float(cite["average_citation_score"])
    return row


def drb_scores(official: dict) -> dict[str, float] | None:
    drb = official.get("deep_research_bench") or {}
    race = (drb.get("race") or {}).get("scores") or {}
    fact = (drb.get("fact") or {}).get("scores") or {}
    keys = {"RACE": "overall_score", "Comp.": "comprehensiveness", "Insight": "insight",
            "Instr.": "instruction_following", "Read.": "readability"}
    if not all(isinstance(race.get(k), (int, float)) for k in keys.values()):
        return None
    if not isinstance(fact.get("valid_rate"), (int, float)):
        return None
    row = {label: float(race[key]) for label, key in keys.items()}
    row["FACT valid"] = float(fact["valid_rate"])
    return row


def model_seconds(calls: list[dict]) -> float:
    """Time at least one model call was running (overlaps counted once), minus
    the time each call spent being refused for capacity (Flex 429s)."""
    return union_seconds([(float(c["start"]) + float(c.get("capacity_wait_s") or 0.0), float(c["end"]))
                          for c in calls if c.get("start") is not None and c.get("end") is not None])


def starved(traj: dict) -> bool:
    """True when the first research round, or at least half of the rounds,
    brought back no new evidence (the signature of a retriever outage; a single
    late round with nothing new is normal)."""
    new = [len((s.get("extra") or {}).get("new_item_ids") or []) for s in traj.get("steps") or []
           if "new_item_ids" in (s.get("extra") or {})]
    return bool(new) and (new[0] == 0 or 2 * sum(n == 0 for n in new) >= len(new))


def load_arm(arm_dir: Path, bench: str, adjusted: dict[str, dict]) -> tuple[dict[str, dict], list[str], list[str]]:
    """(rows by query, queries without all scores, starved queries)."""
    rows, unjudged, starved_ids = {}, [], []
    for run_dir in sorted(arm_dir.glob("*-s*")):
        qid = run_dir.name.split("-s")[0]
        summary_path = run_dir / "metrics" / "summary.json"
        traj_path = run_dir / "trajectories" / f"{qid}.json"
        if not summary_path.exists() or not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        if not traj.get("report"):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        scores = (gym_scores if bench == "drg" else drb_scores)(summary.get("official") or {})
        if scores is None:
            unjudged.append(qid)
            continue
        stats = traj.get("final_stats") or {}
        # token_stop: the merged fork (GR_STOP_TOTAL_TOKENS); token_budget: an
        # earlier prototype, still read so its test runs can be inspected.
        stop = stats.get("token_stop") or stats.get("token_budget") or {}
        tokens = float(summary["mean_tokens"])
        row = dict(scores)
        row["Tokens (K)"] = tokens / 1000.0
        adj = adjusted.get(run_dir.name)
        row["Latency (s)"] = float(adj["adjusted_wall_s"]) if adj else float(summary["mean_wall_s"])
        row["_raw_latency"] = float(summary["mean_wall_s"])
        row["Model (s)"] = model_seconds(stats.get("llm_calls") or [])
        search = (stats.get("latency_breakdown") or {}).get("search_active_s")
        if search is not None:
            row["_search"] = float(search)
        row["Calls"] = float(summary["mean_n_searches"])
        row["_stopped"] = 1.0 if stop.get("stopped") else 0.0
        limit = stop.get("stop_total_tokens") or stop.get("total_token_budget")
        over = stop.get("ended_above_target")
        row["_over"] = 1.0 if (limit and (over if over is not None else tokens > float(limit))) else 0.0
        row["_has_budget"] = 1.0 if limit else 0.0
        row["_trimmed"] = 1.0 if stop.get("report_context_trimmed") else 0.0
        if starved(traj):
            starved_ids.append(qid)
            row["_starved"] = 1.0
        rows[qid] = row
    return rows, unjudged, starved_ids


def bootstrap_ci(diffs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def fmt(col: str, v: float) -> str:
    if col == "Calls":
        return f"{v:.0f}"
    if col in ("RACE", "Comp.", "Insight", "Instr.", "Read.", "FACT valid"):
        return f"{v:.3f}"
    return f"{v:.1f}"


def median_over_passes(rows: dict[str, dict], extra: list[dict[str, dict]], columns) -> dict[str, dict]:
    """Replace each judge score by its median over the first pass and the
    ``JUDGE_PASS`` re-judgings (scripts/sol_budget_judge.sh) of the same report.
    Cost columns come from the run itself and are left as they are."""
    out = {}
    for qid, row in rows.items():
        row = dict(row)
        for col in columns:
            vals = [row[col]] + [p[qid][col] for p in extra if qid in p and col in p[qid]]
            row[col] = st.median(vals)
        out[qid] = row
    return out


STAGES = {"deep_research_plan": "plan", "deep_search_query_gen": "queries", "subquery_generation": "queries",
          "agent_selection": "selection", "orchestrator": "selection", "deep_process_results": "processing",
          "report_generation": "report"}


def replay_tokens(calls: list[dict], budget: int) -> float:
    """Total tokens the run would end with had the same stopping rule used
    ``budget``: the plan is always paid; before each sub-query (selection,
    generation and processing calls up to its processing call) stop if used +
    largest sub-query so far + this run's report cost would exceed the budget.
    The report cost is this run's own, so smaller budgets are slightly overstated."""
    tok = lambda c: c.get("input_tokens", 0) + c.get("output_tokens", 0)
    report = sum(tok(c) for c in calls if c.get("usage_tag") == "report_generation")
    used, units, unit = 0, [], 0
    for c in calls:
        tag = c.get("usage_tag")
        if tag == "report_generation":
            continue
        if tag == "deep_research_plan":
            used += tok(c)
            continue
        unit += tok(c)
        if tag == "deep_process_results":
            units.append(unit)
            unit = 0
    biggest = 0
    for u in units:
        if biggest and used + biggest + report > budget:
            break
        used, biggest = used + u, max(biggest, u)
    return used + report


def breakdown(root: Path, arms: list[str], budget: int) -> None:
    budgets = [b for b in (60_000, 80_000, 100_000, 120_000) if b < budget] + [budget]
    stages = ("plan", "queries", "selection", "processing", "report")
    print("| Approach | n | Tokens (K) | Research (K) | Report (K) | $ / query | "
          + " | ".join(f"{s} % tok" for s in stages) + " | " + " | ".join(f"{s} model-s" for s in stages) + " |")
    print("|" + "---|" * (6 + 2 * len(stages)))
    curves = {}
    for arm in arms:
        per = []
        for f in sorted((root / arm).glob("*-s*/trajectories/*.json")):
            stats = json.loads(f.read_text(encoding="utf-8")).get("final_stats") or {}
            calls = stats.get("llm_calls") or []
            if calls:
                per.append((stats, calls))
        if not per:
            continue
        rows = []
        for stats, calls in per:
            tok = {s: 0.0 for s in stages}
            sec = {s: 0.0 for s in stages}
            for c in calls:
                s = STAGES.get(c.get("usage_tag"))
                if s:
                    tok[s] += c.get("input_tokens", 0) + c.get("output_tokens", 0)
                    sec[s] += c.get("duration_s") or 0.0
            total = sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in calls)
            rows.append((total, tok, sec, float(stats.get("total_cost_usd") or 0.0)))
        m = lambda xs: st.mean(xs)
        cells = [ARM_LABELS.get(arm, arm), str(len(rows)), f"{m([r[0] for r in rows]) / 1e3:.1f}",
                 f"{m([r[0] - r[1]['report'] for r in rows]) / 1e3:.1f}", f"{m([r[1]['report'] for r in rows]) / 1e3:.1f}",
                 f"{m([r[3] for r in rows]):.3f}"]
        cells += [f"{m([100 * r[1][s] / r[0] for r in rows]):.1f}" for s in stages]
        cells += [f"{m([r[2][s] for r in rows]):.0f}" for s in stages]
        print("| " + " | ".join(cells) + " |")
        curves[arm] = [m([replay_tokens(c, b) for _, c in per]) / 1e3 for b in budgets]
    print("\nmodel-s sums call durations (parallel calls overlap). Token curve replayed from the call log:")
    print("| Approach | " + " | ".join(f"{b // 1000}K" for b in budgets) + " |")
    print("|" + "---|" * (1 + len(budgets)))
    for arm, curve in curves.items():
        print(f"| {ARM_LABELS.get(arm, arm)} | " + " | ".join(f"{v:.1f}" for v in curve) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=sorted(BENCHES))
    ap.add_argument("--budget", type=int, default=150000, help="token budget of the runs (default 150000); picks the run root")
    ap.add_argument("--root", type=Path, help="run root (default: the bench's root for --budget)")
    ap.add_argument("--arms", nargs="*", help="arm directories under the root (default: all, in table order)")
    ap.add_argument("--baseline", default="none", help="arm the others are paired against (default: none)")
    ap.add_argument("--breakdown", action="store_true", help="stage token/time breakdown and replayed budget curve (no judge scores needed)")
    ap.add_argument("--drop-starved", action="store_true", help="leave out runs flagged as starved of evidence")
    args = ap.parse_args()

    spec = BENCHES[args.bench]
    root = args.root or Path(spec["root"].replace("budget150", f"budget{args.budget // 1000}"))
    arms = args.arms or [a for a in ARM_ORDER if (root / a).is_dir()] + sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in ARM_ORDER and any(p.glob("*-s*")))
    if not arms:
        raise SystemExit(f"{root}: no arm directories with runs")

    if args.breakdown:
        breakdown(root, arms, args.budget)
        return

    adjusted_path = root / "queue_adjusted.json"
    adjusted_all = json.loads(adjusted_path.read_text(encoding="utf-8"))["runs"] if adjusted_path.exists() else {}
    if not adjusted_all:
        print(f"note: no {adjusted_path}; Latency is raw wall time. Run scripts/flex_queue_adjust.py {root} first.\n")

    data: dict[str, dict[str, dict]] = {}
    for arm in arms:
        rows, unjudged, starved_ids = load_arm(root / arm, args.bench, adjusted_all.get(arm) or {})
        if unjudged:
            print(f"warning: {arm}: {len(unjudged)} report(s) without all judge scores, left out: {', '.join(unjudged)}")
        if starved_ids:
            action = "left out" if args.drop_starved else "still included (use --drop-starved)"
            print(f"warning: {arm}: {len(starved_ids)} run(s) starved of evidence, {action}: {', '.join(starved_ids)}")
            if args.drop_starved:
                rows = {q: r for q, r in rows.items() if q not in starved_ids}
        passes = sorted(root.parent.glob(f"{root.name}_pass*/{arm}"))
        if passes and rows:
            rows = median_over_passes(rows, [load_arm(p, args.bench, {})[0] for p in passes], spec["columns"])
            print(f"note: {arm}: judge scores are the per-report median over {len(passes) + 1} passes")
        if rows:
            data[arm] = rows
    if not data:
        raise SystemExit("no judged runs yet")

    headline = spec["headline"]
    columns = spec["columns"] + COST_COLUMNS
    header = ("Approach", "n") + columns + ("Q per 1K t", "Stopped", "Over", "Trimmed")
    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for arm, rows in data.items():
        mean = {c: st.mean(r[c] for r in rows.values()) for c in columns}
        cells = [ARM_LABELS.get(arm, arm), str(len(rows))] + [fmt(c, mean[c]) for c in columns]
        cells.append(f"{mean[headline] / mean['Tokens (K)']:.3f}" if args.bench == "drg"
                     else f"{1000 * mean[headline] / mean['Tokens (K)']:.3f}")
        n_budget = int(sum(r["_has_budget"] for r in rows.values()))
        cells.append(f"{int(sum(r['_stopped'] for r in rows.values()))}/{n_budget}" if n_budget else "n/a")
        cells.append(f"{int(sum(r['_over'] for r in rows.values()))}/{n_budget}" if n_budget else "n/a")
        cells.append(f"{int(sum(r['_trimmed'] for r in rows.values()))}/{n_budget}" if n_budget else "n/a")
        print("| " + " | ".join(cells) + " |")
    if args.bench == "drb":
        print("\nQ per 1K t on drb is RACE x 1000 / Tokens (K).")
    if adjusted_all:
        for arm, rows in data.items():
            raw = st.mean(r["_raw_latency"] for r in rows.values())
            adj = st.mean(r["Latency (s)"] for r in rows.values())
            if raw - adj >= 0.05:
                print(f"{ARM_LABELS.get(arm, arm)}: {raw - adj:.1f}s of mean Flex waiting removed (raw wall {raw:.1f}s).")

    print("\nWhere wall time goes (mean s): model calls / search (runs recorded with search intervals) / rest")
    for arm, rows in data.items():
        wall = st.mean(r["_raw_latency"] for r in rows.values())
        model = st.mean(r["Model (s)"] for r in rows.values())
        searched = [r["_search"] for r in rows.values() if "_search" in r]
        search = f"{st.mean(searched):.1f} ({len(searched)}/{len(rows)} runs)" if searched else "not recorded"
        print(f"{ARM_LABELS.get(arm, arm)}: wall {wall:.1f} = model {model:.1f} / search {search} / rest {wall - model:.1f} incl. search")

    if args.baseline not in data:
        print(f"\n(no judged runs for the baseline arm {args.baseline!r}; skipping paired differences)")
        return
    base = data[args.baseline]
    for arm, rows in data.items():
        if arm == args.baseline:
            continue
        shared = sorted(set(base) & set(rows))
        if len(shared) < 2:
            continue
        print(f"\n{ARM_LABELS.get(arm, arm)} vs {ARM_LABELS.get(args.baseline, args.baseline)} (paired over {len(shared)} queries)")
        print("| Metric | mean diff | 95% CI | better / worse / tie |")
        print("|---|---|---|---|")
        for col in columns:
            diffs = [rows[q][col] - base[q][col] for q in shared]
            lo, hi = bootstrap_ci(diffs)
            sign = -1 if col in LOWER_IS_BETTER else 1
            better = sum(1 for d in diffs if sign * d > 1e-9)
            worse = sum(1 for d in diffs if sign * d < -1e-9)
            digits = 3 if args.bench == "drb" and col not in COST_COLUMNS else 2
            print(f"| {col} | {st.mean(diffs):+.{digits}f} | [{lo:+.{digits}f}, {hi:+.{digits}f}] | "
                  f"{better} / {worse} / {len(diffs) - better - worse} |")


if __name__ == "__main__":
    main()
