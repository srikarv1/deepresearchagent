"""Checks and result tables for scripts/clean_run.sh.

    python scripts/clean_results.py check <root> [--bench drg|drb] [--query ID]  # smoke checks; exit 1 on a hard failure
    python scripts/clean_results.py table <root> [--bench drg|drb]               # RESULTS.md, results.tex, .json, .csv
    python scripts/clean_results.py tune <tune root> --out thresholds.env         # prompted_scores thresholds

<root> holds <mode>_<budget>k/<arm>/<query>-s0 runs (mode: equal | stop) and the
judge copies <mode>_<budget>k_pass2 / _pass3 written by scripts/sol_budget_judge.sh.

Per report the score is the mean over the judge passes that produced scores
(a failed pass is skipped, never counted as 0); the judge SD is the SD across
those passes, pooled over the reports of a row. Differences against no pruning
are paired per query, with a 95% bootstrap interval. Cost uses gpt-6.1-sol at
Azure Flex rates ($1 / $5 per million input / output tokens) on every
model call except a locally served orchestrator (BC / DPO).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

PRICE_IN, PRICE_OUT = 1.0, 5.0  # $ per million tokens, gpt-6.1-sol Azure Flex
METRICS = ("Overall", "KPR", "KPC", "Citation", "Clarity", "Breadth", "Depth", "Support")
HEADLINE = ("Overall", "KPR", "KPC", "Citation")
ARM_ORDER = ("none", "extractive", "prompted", "prompted_scores", "bc", "dpo")
ARM_NAME = {"none": "No pruning", "extractive": "Extractive", "prompted": "Prompted",
            "prompted_scores": "Prompted (scores)", "bc": "PILOT (BC)", "dpo": "PILOT (DPO)"}
LOCAL_ORCH = {"bc", "dpo"}


def traj(run: Path) -> dict | None:
    f = run / "trajectories" / f"{run.name.split('-s')[0]}.json"
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def gym_scores(run: Path) -> dict | None:
    try:
        g = json.loads((run / "metrics" / "summary.json").read_text())["official"]["deep_research_gym"]
    except Exception:
        return None
    q, k, c = g.get("quality") or {}, g.get("kpr") or {}, g.get("citation") or {}
    if not (q.get("n") and k.get("n") and c.get("n")):
        return None
    crit = q.get("per_criterion") or {}
    row = {m: float(crit[m]["average_rating"]) for m in ("Clarity", "Breadth", "Depth", "Support") if m in crit}
    row.update(Overall=float(q["average_normalized_score"]), KPR=float(k["average_support_rate"]),
               KPC=float(k.get("average_contradicted_rate") or 0.0), Citation=float(c["average_citation_score"]))
    return row


def drb_scores(run: Path) -> dict | None:
    try:
        o = json.loads((run / "metrics" / "summary.json").read_text())["official"]["deep_research_bench"]
    except Exception:
        return None
    race, fact = (o.get("race") or {}).get("scores") or {}, (o.get("fact") or {}).get("scores") or {}
    keys = {"RACE": "overall_score", "Comp.": "comprehensiveness", "Insight": "insight",
            "Instr.": "instruction_following", "Read.": "readability"}
    if not all(isinstance(race.get(k), (int, float)) for k in keys.values()) or not fact.get("total_citations"):
        return None  # FACT with no checked citation = every page fetch failed: not a score
    row = {label: float(race[key]) for label, key in keys.items()}
    row["FACT"] = float(fact["valid_rate"])
    return row


# Per benchmark: score reader, metrics, headline columns, paired-delta metrics, decimals.
BENCHES = {
    "drg": {"scores": gym_scores, "name": "DeepResearchGym", "metrics": METRICS,
            "head": ("Overall", "KPR", "KPC", "Citation"), "delta": ("Overall", "KPR"),
            "labels": {"KPC": "Contr."}, "nd": {}},
    "drb": {"scores": drb_scores, "name": "DeepResearch Bench",
            "metrics": ("RACE", "Comp.", "Insight", "Instr.", "Read.", "FACT"),
            "head": ("RACE", "Comp.", "Insight", "Instr.", "Read.", "FACT"), "delta": ("RACE", "FACT"),
            "labels": {}, "nd": {"RACE": 3, "Comp.": 3, "Insight": 3, "Instr.": 3, "Read.": 3, "FACT": 2}},
}
B = BENCHES["drg"]
BENCH_KEY = "drg"


def nd(m: str) -> int:
    return B["nd"].get(m, 1)


_REF_HEADING = re.compile(r"^\s*#{1,6}\s*(references|sources|bibliography|works cited)\s*:?\s*$", re.I | re.M)


def _body_words(report) -> int | None:
    """Words before the reference list. Only a heading that is exactly References /
    Sources / Bibliography counts (the fork's counter also stops at headings such as
    "Sources and Analytical Approach"); the last such heading is used."""
    text = report.get("article", "") if isinstance(report, dict) else (report or "")
    if not text:
        return None
    heads = list(_REF_HEADING.finditer(text))
    return len((text[: heads[-1].start()] if heads else text).split())


def run_facts(run: Path, arm: str) -> dict:
    """Tokens, cost, latency, writer-context cuts and report length of one run."""
    t = traj(run) or {}
    fs = t.get("final_stats") or {}
    tags = fs.get("tokens_by_tag") or {}
    rb = fs.get("research_budget") or {}
    wc = fs.get("writer_context") or {}
    rl = fs.get("report_length") or {}
    lat = fs.get("latency_breakdown") or {}
    w_in = (tags.get("report_generation") or {}).get("input", 0)
    w_out = (tags.get("report_generation") or {}).get("output", 0)
    o_in = (tags.get("orchestrator") or {}).get("input", 0)
    o_out = (tags.get("orchestrator") or {}).get("output", 0)
    r_in = sum(v.get("input", 0) for k, v in tags.items() if k not in ("report_generation", "orchestrator"))
    r_out = sum(v.get("output", 0) for k, v in tags.items() if k not in ("report_generation", "orchestrator"))
    paid_in = r_in + w_in + (0 if arm in LOCAL_ORCH else o_in)
    paid_out = r_out + w_out + (0 if arm in LOCAL_ORCH else o_out)
    cuts = sum(int(wc.get(k) or 0) for k in ("evidence_words_cut_by_window", "evidence_words_cut_by_item_cap",
                                             "learnings_words_cut_by_share", "words_cut_by_window"))
    return {
        "has_report": bool(t.get("report")),
        "research_in": r_in, "research_out": r_out, "orch_in": o_in, "orch_out": o_out,
        "writer_in": w_in, "writer_out": w_out,
        "research_spend": rb.get("research_spend"), "budget": rb.get("research_budget"),
        "exhausted": rb.get("exhausted"), "exhausted_at_round": rb.get("exhausted_at_round"),
        "over_budget": rb.get("spend_over_budget"), "terminate_ignored": rb.get("terminate_ignored"),
        "cancelled_subqueries": rb.get("cancelled_subqueries"),
        "cost_usd": (paid_in * PRICE_IN + paid_out * PRICE_OUT) / 1e6,
        "research_wall_s": rb.get("research_wall_s"), "writer_wall_s": rb.get("writer_wall_s"),
        "wall_s": fs.get("wall_s") or lat.get("wall_s"), "capacity_wait_s": lat.get("capacity_wait_s"),
        "words_collected": wc.get("words_collected"), "words_cut": cuts,
        "writer_context_tokens": wc.get("context_tokens"),
        "report_words": _body_words(t.get("report")), "report_in_range": rl.get("in_range_first"),
        "rounds": fs.get("num_rounds"),
        "net_retries": _net_retries(run),
    }


def _net_retries(run: Path) -> int:
    """LLM retries after connection errors in this run's log; their waits inflate wall time."""
    log = run.parent / "logs" / f"{run.name}.log"
    try:
        return len(re.findall(r"LLM request failed \(attempt \d+/\d+\)", log.read_text(errors="ignore")))
    except OSError:
        return 0


def collect(root: Path) -> dict:
    """{(mode, budget_k): {arm: {qid: {"scores": [...], "facts": {...}}}}}"""
    out: dict = {}
    for d in sorted(root.glob("*_*k")):
        m = re.fullmatch(r"(equal|stop)_(\d+)k", d.name)
        if not m:
            continue
        key = (m[1], int(m[2]))
        for arm_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            arm = arm_dir.name
            for run in sorted(arm_dir.glob("*-s0")):
                qid = run.name.split("-s")[0]
                if not (traj(run) or {}).get("report"):
                    continue  # killed or unfinished run: no report, no scores, no costs
                passes = [run] + [root / f"{d.name}_pass{k}" / arm / run.name for k in (2, 3)]
                scores = [s for s in (B["scores"](p) for p in passes if p.is_dir()) if s]
                out.setdefault(key, {}).setdefault(arm, {})[qid] = {"scores": scores, "facts": run_facts(run, arm)}
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def boot(diffs, n=4000):
    r = random.Random(0)
    means = sorted(st.mean(r.choice(diffs) for _ in diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def per_report(entry, metric):
    vals = [s[metric] for s in entry["scores"] if metric in s]
    return st.mean(vals) if vals else None


def judge_sd(entries, metric):
    sds = [st.stdev([s[metric] for s in e["scores"]]) for e in entries if len(e["scores"]) >= 2]
    return (sum(x * x for x in sds) / len(sds)) ** 0.5 if sds else None


# ----------------------------------------------------------------- check
def check(root: Path, query: str | None) -> int:
    data = collect(root)
    hard, warn, lines = [], [], []
    for (mode, bk), arms in sorted(data.items()):
        for arm, runs in arms.items():
            for qid, e in runs.items():
                if query and qid != query:
                    continue
                f = e["facts"]
                tag = f"{mode}_{bk}k/{arm}/{qid}"
                if not f["has_report"]:
                    hard.append(f"{tag}: no report"); continue
                if f["words_cut"]:
                    hard.append(f"{tag}: {f['words_cut']} collected words were cut before the writer")
                if f["budget"] and f["over_budget"] is not None and f["over_budget"] > 0:
                    # A paid orchestrator's last checkpoint runs after the budget is reached; that
                    # spend gathers no evidence. Fail only if search/reading research itself (all
                    # research minus orchestration) overshoots by >5%, or the total by >12%.
                    orch = 0 if arm in LOCAL_ORCH else f["orch_in"] + 5 * f["orch_out"]
                    search_over = (f["research_spend"] or 0) - orch - f["budget"]
                    if search_over > 0.05 * f["budget"] or f["over_budget"] > 0.12 * f["budget"]:
                        hard.append(f"{tag}: research {f['over_budget']:.0f} over budget "
                                    f"({search_over:.0f} excluding orchestration)")
                    elif f["over_budget"] > 0.05 * f["budget"]:
                        warn.append(f"{tag}: {f['over_budget']:.0f} over budget, from the orchestrator's last "
                                    f"checkpoint (search/reading research {search_over + f['budget']:.0f})")
                if mode == "equal" and f["budget"] and not f["exhausted"]:
                    warn.append(f"{tag}: research finished below the budget (full research < budget; spend differs)")
                if f["report_words"] is not None and not (2700 <= f["report_words"] <= 3300):
                    warn.append(f"{tag}: report {f['report_words']} words, outside 3000 +/- 10%")
                if f["research_spend"] is None:
                    hard.append(f"{tag}: no research_budget record (fork / harness not up to date?)")
                lines.append(f"{tag}: spend {f['research_spend'] or 0:,.0f}/{f['budget'] or 0:,} "
                             f"(rounds {f['rounds']}, exhausted at {f['exhausted_at_round']}), writer in "
                             f"{f['writer_in']:,} tok, collected {f['words_collected']} words, cut {f['words_cut']}, "
                             f"report {f['report_words']} words, ${f['cost_usd']:.2f}, wall {f['wall_s']}")
    print("\n".join(lines) or "no runs found")
    for w in warn:
        print("WARN", w)
    for h in hard:
        print("FAIL", h)
    print(f"smoke check: {len(lines)} runs, {len(warn)} warnings, {len(hard)} failures")
    return 1 if hard or not lines else 0


# ----------------------------------------------------------------- table
def table(root: Path) -> int:
    data = collect(root)
    results, md, csv_rows = {}, [], []
    md.append(f"# Clean {B['name']} results\n")
    md.append("Mean over queries of the per-report mean over judge passes; +/- is the pooled judge SD. "
              "Delta = paired difference vs no pruning with 95% bootstrap interval. Cost at Flex rates. "
              "Research s / Writer s: mean over runs without network retries (retry waits are not agent time). "
              "Runs without a report (killed) are excluded everywhere.\n")
    for (mode, bk), arms in sorted(data.items(), key=lambda kv: (kv[0][0] != "equal", kv[0][1])):
        md.append(f"\n## {'Equal research spend' if mode == 'equal' else 'Stop decisions respected'}, {bk}K research budget\n")
        heads = [B["labels"].get(m, m) for m in B["head"]] + [f"d{m} vs none [95%]" for m in B["delta"]]
        md.append("| Method | " + " | ".join(heads) + " | Research spend | Writer in/out | Cost $ | Research s "
                  "| Writer s | Words cut | Report words | judged |")
        md.append("|" + "---|" * (len(heads) + 9))
        base = arms.get("none") or (data.get(("equal", bk)) or {}).get("none") or {}
        for arm in [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]:
            runs = arms[arm]
            scored = {q: e for q, e in runs.items() if e["scores"]}
            row = {"n_runs": len(runs), "n_scored": len(scored)}
            for m in B["metrics"]:
                vals = [v for v in (per_report(e, m) for e in scored.values()) if v is not None]
                row[m] = mean(vals)
                row[m + "_judge_sd"] = judge_sd(scored.values(), m)
                row[m + "_ci"] = boot(vals) if len(vals) >= 3 else None  # 95% interval of the query mean
            for m in B["delta"]:
                shared = [q for q in scored if q in base and base[q]["scores"]]
                if arm != "none" and len(shared) >= 3:
                    d = [per_report(scored[q], m) - per_report(base[q], m) for q in shared]
                    row[f"d{m}"], row[f"d{m}_ci"] = st.mean(d), boot(d)
            for k in ("research_spend", "writer_in", "writer_out", "cost_usd",
                      "words_cut", "report_words", "research_in", "research_out", "orch_in", "orch_out"):
                row[k] = mean(e["facts"][k] for e in runs.values())
            # Latency only from runs without network retries (retry waits are not agent time).
            clean = [e for e in runs.values() if not e["facts"]["net_retries"]]
            for k in ("research_wall_s", "writer_wall_s", "wall_s"):
                row[k] = mean(e["facts"][k] for e in clean)
            row["n_latency"] = len(clean)
            results[f"{mode}_{bk}k/{arm}"] = row

            def f(m):
                v, sd, n = row.get(m), row.get(m + "_judge_sd"), nd(m)
                return "--" if v is None else f"{v:.{n}f}" + (f" ±{sd:.{n}f}" if sd is not None else "")

            def dl(m):
                if f"d{m}" not in row:
                    return "--"
                lo, hi, n = *row[f"d{m}_ci"], nd(m)
                return f"{row[f'd{m}']:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]"

            def k(v, scale=1000, nd=0, suffix="K"):
                return "--" if v is None else f"{v / scale:.{nd}f}{suffix}"
            cost = "--" if row["cost_usd"] is None else f"{row['cost_usd']:.2f}"
            md.append(f"| {ARM_NAME.get(arm, arm)} | " + " | ".join([f(m) for m in B["head"]] + [dl(m) for m in B["delta"]])
                      + f" | {k(row['research_spend'])} "
                      f"| {k(row['writer_in'])}/{k(row['writer_out'], nd=1)} "
                      f"| {cost} "
                      f"| {k(row['research_wall_s'], 1, 0, '')} | {k(row['writer_wall_s'], 1, 0, '')} "
                      f"| {k(row['words_cut'], 1, 0, '')} | {k(row['report_words'], 1, 0, '')} "
                      f"| {row['n_scored']}/{row['n_runs']} |")
            for q, e in runs.items():
                r = {"setting": f"{mode}_{bk}k", "arm": arm, "query": q, "passes": len(e["scores"])}
                r.update({m: per_report(e, m) for m in B["metrics"]} if e["scores"] else {})
                r.update(e["facts"])
                csv_rows.append(r)
    (root / "RESULTS.md").write_text("\n".join(md) + "\n")
    (root / "results.tex").write_text(latex(results))
    (root / "results.json").write_text(json.dumps(results, indent=1, default=str))
    if csv_rows:
        keys = sorted({k for r in csv_rows for k in r}, key=lambda k: (k not in ("setting", "arm", "query"), k))
        with open(root / "per_query.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    print(f"wrote {root / 'RESULTS.md'}, results.tex, results.json, per_query.csv ({len(csv_rows)} runs)")
    return 0


def latex(results: dict) -> str:
    """Paper-ready table (equal spend, every budget; stop mode at the end) and a
    pgfplots figure of Overall and KPR vs budget with 95% intervals. Needs the
    paper's \\sd macro: \\newcommand{\\sd}[1]{{\\scriptsize\\,\\textcolor{gray}{$\\pm$#1}}}."""
    def cell(r, m):
        v, sd, n = r.get(m), r.get(m + "_judge_sd"), nd(m)
        return "--" if v is None else f"{v:.{n}f}" + (f"\\sd{{{sd:.{n}f}}}" if sd is not None else "")

    def num(v, scale=1000, nd=0):
        return "--" if v is None else f"{v / scale:.{nd}f}"
    settings = sorted({k.split("/")[0] for k in results}, key=lambda x: (not x.startswith("equal"), int(x.split("_")[1][:-1])))
    out = ["% generated by scripts/clean_results.py",
           "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\renewcommand{\\arraystretch}{1.15}",
           f"\\caption{{{B['name']} at research-only budgets"
           " (\\texttt{gpt-6.1-sol}, depth 3, breadth 4; writer context "
           "uncapped; 3{,}000-word reports). Equal spend: every method spends the full research budget; stop: stop "
           "decisions respected. Three gpt-5-mini judge passes per report; grey $\\pm$ is the judge standard deviation. "
           "Research spend is input + 5$\\times$output tokens of research and paid orchestration; writer input in "
           "thousands of tokens; cost per report at Azure Flex rates.}",
           f"\\label{{tab:clean-{BENCH_KEY}}}",
           "\\begin{adjustbox}{max width=\\linewidth}",
           "\\begin{tabular}{l" + "c" * (len(B["head"]) + 3) + "}", "\\toprule",
           "Approach & " + " & ".join(B["labels"].get(m, m) for m in B["head"])
           + " & Research spend (K) & Writer input (K) & Cost (\\$) \\\\"]
    ncol = len(B["head"]) + 4
    for st_ in settings:
        mode, bk = st_.split("_")
        out += ["\\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{'Equal spend' if mode == 'equal' else 'Stop respected'}, {bk[:-1]}K}}}} \\\\"]
        for arm in ARM_ORDER:
            r = results.get(f"{st_}/{arm}")
            if not r:
                continue
            cost = "--" if r.get("cost_usd") is None else f"{r['cost_usd']:.2f}"
            out.append(ARM_NAME[arm].replace("PILOT", chr(92) + "method{}") + " & "
                       + " & ".join(cell(r, m) for m in B["head"])
                       + f" & {num(r.get('research_spend'))} & {num(r.get('writer_in'))} & {cost} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    # Figure: Overall / KPR vs research budget, equal spend, 95% intervals of the query mean.
    budgets = sorted({int(k.split("/")[0].split("_")[1][:-1]) for k in results if k.startswith("equal")})
    for m in B["delta"]:
        out += [f"% {m} vs research budget (needs \\usepackage{{pgfplots}})", "\\begin{tikzpicture}",
                f"\\begin{{axis}}[width=0.48\\linewidth, height=4.2cm, xlabel={{Research budget (K)}}, ylabel={{{m}}}, "
                f"xtick={{{','.join(map(str, budgets))}}}, legend style={{font=\\tiny, at={{(1.02,1)}}, anchor=north west}}]"]
        for arm in ARM_ORDER:
            pts = []
            for b in budgets:
                r = results.get(f"equal_{b}k/{arm}")
                if r and r.get(m) is not None and r.get(m + "_ci"):
                    lo, hi = r[m + "_ci"]
                    pts.append(f"({b},{r[m]:.{nd(m) + 1}f}) +- (0,{max(hi - r[m], r[m] - lo):.{nd(m) + 1}f})")
            if pts:
                out += ["\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {" + " ".join(pts) + "};",
                        f"\\addlegendentry{{{ARM_NAME[arm]}}}"]
        out += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def _rounds(run: Path) -> list[dict]:
    f = run / "gpt_researcher" / f"{run.name.split('-s')[0]}.json"
    try:
        return json.loads(f.read_text()).get("rounds") or []
    except Exception:
        return []


def tune(root: Path, out: Path) -> int:
    """Thresholds for prompted_scores that make it keep and stop like prompted on
    the dev queries. Both ran with every round forced (GR_BUDGET_FORCE_FULL=1), so
    prompted's TERMINATE is recorded (terminate_ignored) and prompted_scores'
    item and continue scores exist for every round; any threshold can then be
    applied offline. Keep threshold: the item keep rate closest to prompted's.
    Stop threshold: the stop round closest to prompted's (never = last round + 1)."""
    ref_keep, ref_stop, n_rounds = [], [], []
    for run in sorted((root / "prompted").glob("*-s0")):
        rounds = _rounds(run)
        if not rounds:
            continue
        n_rounds.append(len(rounds))
        stop = None
        for i, r in enumerate(rounds, 1):
            d = r.get("decision") or {}
            kept, pruned = len(d.get("kept_item_ids") or []), len(d.get("pruned_item_ids") or [])
            if kept + pruned:
                ref_keep.append(kept / (kept + pruned))
            if stop is None and (d.get("type") == "terminate" or (d.get("meta") or {}).get("terminate_ignored")):
                stop = i
        ref_stop.append(stop if stop is not None else len(rounds) + 1)
    scored = []  # per run: list of (item scores, continue score) per round
    for run in sorted((root / "prompted_scores").glob("*-s0")):
        rounds = [((r.get("decision") or {}).get("meta") or {}) for r in _rounds(run)]
        if rounds:
            scored.append([(list((m.get("item_scores") or {}).values()), m.get("continue_score")) for m in rounds])
    if not ref_keep or not scored:
        print("tune: need prompted and prompted_scores dev runs with round logs", file=sys.stderr)
        return 1
    target_keep, target_stop = st.mean(ref_keep), st.mean(ref_stop)

    def keep_rate(t):
        rates = [sum(x >= t for x in items) / len(items) for run in scored for items, _ in run if items]
        return st.mean(rates) if rates else 0.0

    def stop_round(t):
        rs = []
        for run in scored:
            r = next((i for i, (_, c) in enumerate(run, 1) if c is not None and c < t), len(run) + 1)
            rs.append(r)
        return st.mean(rs)
    keep_t = min(range(0, 11), key=lambda t: (abs(keep_rate(t) - target_keep), t))
    stop_t = min(range(0, 11), key=lambda t: (abs(stop_round(t) - target_stop), t))
    out.write_text(f"GR_PROMPTED_KEEP_THRESHOLD={keep_t}\nGR_PROMPTED_STOP_THRESHOLD={stop_t}\n")
    print(f"tune: prompted keeps {target_keep:.0%} of items and stops at round {target_stop:.1f} "
          f"(runs: {len(ref_stop)}); prompted_scores at keep>={keep_t} keeps {keep_rate(keep_t):.0%}, "
          f"stop<{stop_t} stops at round {stop_round(stop_t):.1f} -> {out}")
    return 0


def main() -> int:
    global B, BENCH_KEY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "table", "tune"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--bench", choices=tuple(BENCHES), default="drg")
    ap.add_argument("--query")
    ap.add_argument("--out", type=Path, help="tune: thresholds file to write")
    a = ap.parse_args()
    B, BENCH_KEY = BENCHES[a.bench], a.bench
    if a.mode == "tune":
        return tune(a.root, a.out or a.root / "thresholds.env")
    return check(a.root, a.query) if a.mode == "check" else table(a.root)


if __name__ == "__main__":
    sys.exit(main())
