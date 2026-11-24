#!/usr/bin/env python3
"""Judge the same reports several times and average the scores (P1 judge sensitivity).

The judges (gpt-5-mini) are reasoning models without a fixed temperature, so
judging one report twice can give different scores. This script scores every
report K times, keeps each repeat separately, and reports per method the mean
of the K scores and how much they vary, next to the gaps between methods.

    # 1. judge (each report K times; finished repeats are skipped on rerun)
    python scripts/judge_repeats.py run drg --root runs/gym_sol_budget150 --arms none --repeats 3
    python scripts/judge_repeats.py run drb --root runs/drb_sol_budget150 --arms none --repeats 3

    # 2. averages and spread
    python scripts/judge_repeats.py summarize drg --root runs/gym_sol_budget150 --arms none
    python scripts/judge_repeats.py summarize drg ... --csv drg_judge.csv --latex drg_judge.tex

Each repeat judges a copy of the run that holds only its trajectory, under
``<out>/<arm>/r<k>/<run>/`` (default ``<out>`` = ``<root>/judge_repeats``), so
the run's own scores are never overwritten and ``adr evaluate`` writes each
repeat's ``metrics/summary.json`` separately.

Judge credentials: JUDGE_API_KEY + JUDGE_BASE_URL if set, else OPENAI_API_KEY +
OPENAI_BASE_URL, from the environment or .env. They must reach the gpt-5-mini
resource (not Sol). DRB also needs JINA_API_KEY for FACT.

Summary, per method and metric:
  mean          mean over queries of the per-query mean of the K scores
  judge std     mean over queries of the per-query std across the K repeats
                (how much one report's score moves between judgings)
  between-query std of the per-query means
Then, for each pair of methods, the gap in each metric and the gap divided by
the judge std: a ratio near or below 1 means the judge's noise alone could
explain the gap.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _budget_table():
    spec = importlib.util.spec_from_file_location("budget_table", REPO / "scripts" / "budget_table.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("budget_table", module)
    spec.loader.exec_module(module)
    return module


bt = _budget_table()

BENCHES = {
    "drg": {"official": "deep_research_gym", "config": "configs/gym_sol_pilot_d2_budget150.yaml",
            "columns": bt.BENCHES["drg"]["columns"], "scores": bt.gym_scores, "jobs": 2},
    # The DRB judge stages one file per model name, so DRB repeats run one at a time.
    "drb": {"official": "deep_research_bench", "config": "configs/gpt_researcher_bench_budget150.yaml",
            "columns": bt.BENCHES["drb"]["columns"], "scores": bt.drb_scores, "jobs": 1},
}


# ── discovery ─────────────────────────────────────────────────────────────


def report_runs(arm_dir: Path) -> list[Path]:
    """Run directories of an arm that hold a finished report."""
    runs = []
    for run_dir in sorted(arm_dir.glob("*-s*")):
        qid = run_dir.name.rsplit("-s", 1)[0]
        traj = run_dir / "trajectories" / f"{qid}.json"
        if not traj.exists():
            continue
        try:
            data = json.loads(traj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("report") and not data.get("error"):
            runs.append(run_dir)
    return runs


def arms_under(root: Path, only: list[str] | None) -> dict[str, Path]:
    arms = {p.name: p for p in sorted(root.iterdir())
            if p.is_dir() and p.name != "judge_repeats" and report_runs(p)}
    if only:
        missing = [a for a in only if a not in arms]
        if missing:
            raise SystemExit(f"no runs with reports for arm(s) {missing} under {root}")
        arms = {a: arms[a] for a in only}
    return arms


def repeat_dir(out: Path, arm: str, k: int, run_dir: Path) -> Path:
    return out / arm / f"r{k}" / run_dir.name


def scores_of(run_copy: Path, bench: str) -> dict[str, float] | None:
    path = run_copy / "metrics" / "summary.json"
    if not path.exists():
        return None
    try:
        official = json.loads(path.read_text(encoding="utf-8")).get("official") or {}
    except (OSError, json.JSONDecodeError):
        return None
    return BENCHES[bench]["scores"](official)


def prepare_copy(run_dir: Path, dest: Path) -> None:
    """A run copy holding only the trajectory, which is all `adr evaluate` reads."""
    qid = run_dir.name.rsplit("-s", 1)[0]
    (dest / "trajectories").mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "trajectories" / f"{qid}.json", dest / "trajectories" / f"{qid}.json")


# ── run ───────────────────────────────────────────────────────────────────


def judge_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            env.setdefault(key, value.strip().strip('"').strip("'"))
    if env.get("JUDGE_API_KEY"):
        if not env.get("JUDGE_BASE_URL"):
            raise SystemExit("JUDGE_BASE_URL is required with JUDGE_API_KEY")
        env["OPENAI_API_KEY"], env["OPENAI_BASE_URL"] = env["JUDGE_API_KEY"], env["JUDGE_BASE_URL"]
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("no judge credentials: set JUDGE_API_KEY + JUDGE_BASE_URL (or OPENAI_*) in .env")
    env.setdefault("CRAWL4AI_NO_CLOUD_NOTICE", "1")
    env.setdefault("PYTHONUTF8", "1")
    return env


def run_cmd(args: argparse.Namespace) -> int:
    spec = BENCHES[args.bench]
    out = args.out or args.root / "judge_repeats"
    config = args.config or REPO / spec["config"]
    env = judge_env()
    jobs = args.jobs or spec["jobs"]

    todo: list[tuple[str, int, Path, Path]] = []
    for arm, arm_dir in arms_under(args.root, args.arms).items():
        runs = report_runs(arm_dir)
        if args.queries:
            runs = [r for r in runs if r.name.rsplit("-s", 1)[0] in args.queries]
        for k in range(1, args.repeats + 1):
            for run_dir in runs:
                dest = repeat_dir(out, arm, k, run_dir)
                if scores_of(dest, args.bench) is None:
                    todo.append((arm, k, run_dir, dest))
    print(f"{time.strftime('%H:%M')} {args.bench}: {len(todo)} judging(s) to do, {jobs} at a time -> {out}")
    if args.dry_run or not todo:
        for arm, k, run_dir, _ in todo:
            print(f"  {arm} r{k} {run_dir.name}")
        return 0

    def judge(item: tuple[str, int, Path, Path]) -> bool:
        arm, k, run_dir, dest = item
        prepare_copy(run_dir, dest)
        cmd = [sys.executable, "-m", "adr.cli", "evaluate", str(dest),
               "--official", spec["official"], "-c", str(config)]
        with (dest / "judge.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        ok = proc.returncode == 0 and scores_of(dest, args.bench) is not None
        print(f"{time.strftime('%H:%M')} {'judged' if ok else 'FAILED'} {arm} r{k} {run_dir.name}"
              + ("" if ok else f" (see {dest / 'judge.log'})"), flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(judge, todo))
    failed = results.count(False)
    print(f"{time.strftime('%H:%M')} done: {len(results) - failed} judged, {failed} failed"
          + ("; rerun the same command to retry" if failed else ""))
    return 1 if failed else 0


# ── summarize ─────────────────────────────────────────────────────────────


def collect(out: Path, arm: str, bench: str) -> dict[str, list[dict[str, float]]]:
    """{query: [scores of each repeat that has them]}."""
    per_query: dict[str, list[dict[str, float]]] = {}
    for rep in sorted((out / arm).glob("r*")):
        for run_copy in sorted(rep.glob("*-s*")):
            scores = scores_of(run_copy, bench)
            if scores is not None:
                per_query.setdefault(run_copy.name.rsplit("-s", 1)[0], []).append(scores)
    return per_query


def summarize_arm(per_query: dict[str, list[dict[str, float]]], columns, min_repeats: int) -> dict:
    """Per metric: mean of per-query means, mean per-query judge std, between-query std."""
    kept = {q: reps for q, reps in per_query.items() if len(reps) >= min_repeats}
    out: dict = {"n_queries": len(kept), "repeats": sorted({len(r) for r in kept.values()}),
                 "per_query_mean": {}, "metrics": {}}
    for q, reps in kept.items():
        out["per_query_mean"][q] = {c: st.mean(r[c] for r in reps) for c in columns}
    for c in columns:
        means = [out["per_query_mean"][q][c] for q in kept]
        stds = [st.stdev([r[c] for r in reps]) for reps in kept.values() if len(reps) > 1]
        out["metrics"][c] = {
            "mean": st.mean(means) if means else None,
            "judge_std": st.mean(stds) if stds else None,
            "between_query_std": st.stdev(means) if len(means) > 1 else None,
        }
    return out


def pairwise_gaps(summaries: dict[str, dict], columns) -> list[dict]:
    """Gap in each metric for each pair of methods, over the queries both have,
    and the gap relative to the larger of the two methods' judge std."""
    rows = []
    for a, b in itertools.combinations(summaries, 2):
        common = set(summaries[a]["per_query_mean"]) & set(summaries[b]["per_query_mean"])
        if not common:
            continue
        for c in columns:
            gap = st.mean(summaries[a]["per_query_mean"][q][c] - summaries[b]["per_query_mean"][q][c]
                          for q in common)
            noise = max((summaries[x]["metrics"][c]["judge_std"] or 0.0) for x in (a, b))
            rows.append({"a": a, "b": b, "metric": c, "n": len(common), "gap": gap,
                         "judge_std": noise, "gap_over_noise": (abs(gap) / noise) if noise else None})
    return rows


def _f(v, digits=3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def summarize_cmd(args: argparse.Namespace) -> int:
    spec = BENCHES[args.bench]
    out = args.out or args.root / "judge_repeats"
    columns = spec["columns"]
    arms = [a for a in (args.arms or sorted(p.name for p in out.iterdir() if p.is_dir()))]
    summaries = {}
    for arm in arms:
        per_query = collect(out, arm, args.bench)
        if not per_query:
            print(f"note: no judged repeats for {arm} under {out / arm}", file=sys.stderr)
            continue
        summaries[arm] = summarize_arm(per_query, columns, args.min_repeats)
    if not summaries:
        print("nothing to summarize", file=sys.stderr)
        return 1

    print(f"Scores averaged over the repeats (mean ± judge std); {args.bench}, out={out}")
    width = max(len(bt.ARM_LABELS.get(a, a)) for a in summaries)
    print("Method".ljust(width) + "   n  reps  " + "  ".join(c.rjust(15) for c in columns))
    for arm, s in summaries.items():
        cells = [f"{_f(m['mean'])} ± {_f(m['judge_std'])}".rjust(15) for m in (s["metrics"][c] for c in columns)]
        print(bt.ARM_LABELS.get(arm, arm).ljust(width) + f"  {s['n_queries']:2d}  {','.join(map(str, s['repeats'])):>4}  "
              + "  ".join(cells))

    gaps = pairwise_gaps(summaries, columns)
    if gaps:
        print("\nGaps between methods vs judge noise (|gap| / judge std; <= 1 means the judge's noise alone could explain it)")
        for g in gaps:
            print(f"  {g['a']} - {g['b']}  {g['metric']:>10}: gap {g['gap']:+.3f}  judge std {g['judge_std']:.3f}"
                  f"  ratio {_f(g['gap_over_noise'], 2)}  (n={g['n']})")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "metric", "mean", "judge_std", "between_query_std", "n_queries"])
            for arm, s in summaries.items():
                for c in columns:
                    m = s["metrics"][c]
                    w.writerow([arm, c, m["mean"], m["judge_std"], m["between_query_std"], s["n_queries"]])
    if args.latex:
        head = " & ".join(["Method", *columns]) + r" \\"
        body = [" & ".join([bt.ARM_LABELS.get(a, a),
                            *(f"{_f(s['metrics'][c]['mean'])} $\\pm$ {_f(s['metrics'][c]['judge_std'])}" for c in columns)])
                + r" \\" for a, s in summaries.items()]
        args.latex.write_text("\n".join([
            r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule", head, r"\midrule", *body,
            r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps({"summaries": summaries, "gaps": gaps}, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("bench", choices=sorted(BENCHES))
        p.add_argument("--root", type=Path, required=True, help="directory whose subdirectories are arms")
        p.add_argument("--arms", nargs="+", help="only these arms")
        p.add_argument("--out", type=Path, help="where repeats go (default <root>/judge_repeats)")
    run_p, sum_p = sub.choices["run"], sub.choices["summarize"]
    run_p.add_argument("--repeats", type=int, default=3)
    run_p.add_argument("--queries", nargs="+", help="only these query ids")
    run_p.add_argument("--config", type=Path, help="run config with the eval section (default per bench)")
    run_p.add_argument("--jobs", type=int, help="judgings at a time (default 2 for drg, 1 for drb)")
    run_p.add_argument("--dry-run", action="store_true", help="list what would be judged")
    sum_p.add_argument("--min-repeats", type=int, default=2, help="skip queries with fewer judged repeats")
    sum_p.add_argument("--csv", type=Path)
    sum_p.add_argument("--latex", type=Path)
    sum_p.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    return run_cmd(args) if args.cmd == "run" else summarize_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
