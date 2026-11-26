"""Checks and result tables for scripts/clean_run.sh.

    python scripts/clean_results.py check <root> [--query ID]   # smoke checks; exit 1 on a hard failure
    python scripts/clean_results.py table <root>                # RESULTS.md, results.json, per_query.csv

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
        "report_words": rl.get("words_final"), "report_in_range": rl.get("in_range_first"),
        "rounds": fs.get("num_rounds"),
    }


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
                passes = [run] + [root / f"{d.name}_pass{k}" / arm / run.name for k in (2, 3)]
                scores = [s for s in (gym_scores(p) for p in passes if p.is_dir()) if s]
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
                if f["budget"] and f["over_budget"] is not None and f["over_budget"] > 0.05 * f["budget"]:
                    hard.append(f"{tag}: research spend {f['over_budget']:.0f} over budget (>5%)")
                if mode == "equal" and f["budget"] and not f["exhausted"]:
                    warn.append(f"{tag}: research finished below the budget (full research < budget; spend differs)")
                if f["report_in_range"] is False:
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
    md.append("# Clean DRG results\n")
    md.append("Mean over queries of the per-report mean over judge passes; +/- is the pooled judge SD. "
              "Delta = paired difference vs no pruning with 95% bootstrap interval. Cost at Flex rates.\n")
    for (mode, bk), arms in sorted(data.items(), key=lambda kv: (kv[0][0] != "equal", kv[0][1])):
        md.append(f"\n## {'Equal research spend' if mode == 'equal' else 'Stop decisions respected'}, {bk}K research budget\n")
        md.append("| Method | Overall | KPR | KPC | Citation | dOverall vs none [95%] | dKPR vs none [95%] "
                  "| Research spend | Writer in/out | Cost $ | Research s | Writer s | Words cut | Report words | judged |")
        md.append("|" + "---|" * 15)
        base = arms.get("none") or (data.get(("equal", bk)) or {}).get("none") or {}
        for arm in [a for a in ARM_ORDER if a in arms] + [a for a in arms if a not in ARM_ORDER]:
            runs = arms[arm]
            scored = {q: e for q, e in runs.items() if e["scores"]}
            row = {"n_runs": len(runs), "n_scored": len(scored)}
            for m in METRICS:
                vals = [v for v in (per_report(e, m) for e in scored.values()) if v is not None]
                row[m] = mean(vals)
                row[m + "_judge_sd"] = judge_sd(scored.values(), m)
                row[m + "_ci"] = boot(vals) if len(vals) >= 3 else None  # 95% interval of the query mean
            for m in ("Overall", "KPR"):
                shared = [q for q in scored if q in base and base[q]["scores"]]
                if arm != "none" and len(shared) >= 3:
                    d = [per_report(scored[q], m) - per_report(base[q], m) for q in shared]
                    row[f"d{m}"], row[f"d{m}_ci"] = st.mean(d), boot(d)
            for k in ("research_spend", "writer_in", "writer_out", "cost_usd", "research_wall_s", "writer_wall_s",
                      "words_cut", "report_words", "research_in", "research_out", "orch_in", "orch_out"):
                row[k] = mean(e["facts"][k] for e in runs.values())
            results[f"{mode}_{bk}k/{arm}"] = row

            def f(m, nd=1):
                v, sd = row.get(m), row.get(m + "_judge_sd")
                return "--" if v is None else f"{v:.{nd}f}" + (f" ±{sd:.{nd}f}" if sd is not None else "")

            def dl(m):
                if f"d{m}" not in row:
                    return "--"
                lo, hi = row[f"d{m}_ci"]
                return f"{row[f'd{m}']:+.1f} [{lo:+.1f}, {hi:+.1f}]"

            def k(v, scale=1000, nd=0, suffix="K"):
                return "--" if v is None else f"{v / scale:.{nd}f}{suffix}"
            cost = "--" if row["cost_usd"] is None else f"{row['cost_usd']:.2f}"
            md.append(f"| {ARM_NAME.get(arm, arm)} | {f('Overall')} | {f('KPR')} | {f('KPC')} | {f('Citation')} "
                      f"| {dl('Overall')} | {dl('KPR')} | {k(row['research_spend'])} "
                      f"| {k(row['writer_in'])}/{k(row['writer_out'], nd=1)} "
                      f"| {cost} "
                      f"| {k(row['research_wall_s'], 1, 0, '')} | {k(row['writer_wall_s'], 1, 0, '')} "
                      f"| {k(row['words_cut'], 1, 0, '')} | {k(row['report_words'], 1, 0, '')} "
                      f"| {row['n_scored']}/{row['n_runs']} |")
            for q, e in runs.items():
                r = {"setting": f"{mode}_{bk}k", "arm": arm, "query": q, "passes": len(e["scores"])}
                r.update({m: per_report(e, m) for m in METRICS} if e["scores"] else {})
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
    def cell(r, m, nd=1):
        v, sd = r.get(m), r.get(m + "_judge_sd")
        return "--" if v is None else f"{v:.{nd}f}" + (f"\\sd{{{sd:.{nd}f}}}" if sd is not None else "")

    def num(v, scale=1000, nd=0):
        return "--" if v is None else f"{v / scale:.{nd}f}"
    settings = sorted({k.split("/")[0] for k in results}, key=lambda x: (not x.startswith("equal"), int(x.split("_")[1][:-1])))
    out = ["% generated by scripts/clean_results.py",
           "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\renewcommand{\\arraystretch}{1.15}",
           "\\caption{DeepResearchGym at research-only budgets (\\texttt{gpt-6.1-sol}, depth 3, breadth 4; writer context "
           "uncapped; 3{,}000-word reports). Equal spend: every method spends the full research budget; stop: stop "
           "decisions respected. Three gpt-5-mini judge passes per report; grey $\\pm$ is the judge standard deviation. "
           "Research spend is input + 5$\\times$output tokens of research and paid orchestration; writer input in "
           "thousands of tokens; cost per report at Azure Flex rates.}",
           "\\label{tab:clean-drg}",
           "\\begin{adjustbox}{max width=\\linewidth}",
           "\\begin{tabular}{lccccccc}", "\\toprule",
           "Approach & Overall & KPR & Contr. & Citation & Research spend (K) & Writer input (K) & Cost (\\$) \\\\"]
    for st_ in settings:
        mode, bk = st_.split("_")
        out += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{'Equal spend' if mode == 'equal' else 'Stop respected'}, {bk[:-1]}K}}}} \\\\"]
        for arm in ARM_ORDER:
            r = results.get(f"{st_}/{arm}")
            if not r:
                continue
            cost = "--" if r.get("cost_usd") is None else f"{r['cost_usd']:.2f}"
            out.append(f"{ARM_NAME[arm].replace('PILOT', chr(92) + 'method{}')} & {cell(r, 'Overall')} & {cell(r, 'KPR')} & "
                       f"{cell(r, 'KPC')} & {cell(r, 'Citation')} & {num(r.get('research_spend'))} & "
                       f"{num(r.get('writer_in'))} & {cost} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    # Figure: Overall / KPR vs research budget, equal spend, 95% intervals of the query mean.
    budgets = sorted({int(k.split("/")[0].split("_")[1][:-1]) for k in results if k.startswith("equal")})
    for m in ("Overall", "KPR"):
        out += [f"% {m} vs research budget (needs \\usepackage{{pgfplots}})", "\\begin{tikzpicture}",
                f"\\begin{{axis}}[width=0.48\\linewidth, height=4.2cm, xlabel={{Research budget (K)}}, ylabel={{{m}}}, "
                f"xtick={{{','.join(map(str, budgets))}}}, legend style={{font=\\tiny, at={{(1.02,1)}}, anchor=north west}}]"]
        for arm in ARM_ORDER:
            pts = []
            for b in budgets:
                r = results.get(f"equal_{b}k/{arm}")
                if r and r.get(m) is not None and r.get(m + "_ci"):
                    lo, hi = r[m + "_ci"]
                    pts.append(f"({b},{r[m]:.2f}) +- (0,{max(hi - r[m], r[m] - lo):.2f})")
            if pts:
                out += ["\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {" + " ".join(pts) + "};",
                        f"\\addlegendentry{{{ARM_NAME[arm]}}}"]
        out += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "table"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--query")
    a = ap.parse_args()
    return check(a.root, a.query) if a.mode == "check" else table(a.root)


if __name__ == "__main__":
    sys.exit(main())
