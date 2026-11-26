"""Judge DeepResearch Bench reports (RACE + FACT, gpt-5-mini) K times, N at once.

The upstream DRB pipeline stages every report into fixed files of its checkout,
so two judgings in one checkout overwrite each other. Each worker therefore
uses its own copy of the checkout (ADR_DRB_DIR): --setup creates them from
third_party/deep_research_bench (with patches/deep_research_bench_api.patch
applied) under --workers-dir.

Pass 1 judges the run directory itself; pass P > 1 judges a copy in
<root>_passP (the layout of scripts/sol_budget_judge.sh), so earlier scores are
never overwritten. A report with RACE scores and FACT with checked citations is
skipped, so rerunning retries only what is missing.

    python scripts/judge_drb_parallel.py --setup -n 6                        # create 6 worker copies
    python scripts/judge_drb_parallel.py -n 6 runs/clean_drb/equal_200k ...  # judge passes 1-3
    JINA_READER=public python scripts/judge_drb_parallel.py ...              # keyless Jina reader

Judge credentials: JUDGE_API_KEY + JUDGE_BASE_URL (from the environment or .env).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/drb_sol_clean.yaml"


def done(d: Path) -> bool:
    try:
        o = json.loads((d / "metrics" / "summary.json").read_text())["official"]["deep_research_bench"]
    except Exception:
        return False
    fact = (o.get("fact") or {}).get("scores") or {}
    # FACT with no checked citation means every page fetch failed (e.g. Jina out of balance): not done.
    return bool(o.get("official")) and bool((o.get("race") or {}).get("scores")) and bool(fact.get("total_citations"))


def has_report(d: Path) -> bool:
    t = d / "trajectories" / f"{d.name.split('-s')[0]}.json"
    if not t.exists():
        return False
    rep = json.loads(t.read_text()).get("report")
    return bool((rep.get("article", "") if isinstance(rep, dict) else rep or "").strip())


def target(run: Path, root: Path, p: int) -> Path:
    if p == 1:
        return run
    dest = Path(f"{root}_pass{p}") / run.relative_to(root)
    if not dest.exists():
        shutil.copytree(run, dest, ignore=shutil.ignore_patterns("metrics", "judge.log", "exports"))
    return dest


def load_env() -> dict:
    env = dict(os.environ)
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not env.get("JUDGE_API_KEY") or not env.get("JUDGE_BASE_URL"):
        raise SystemExit("JUDGE_API_KEY and JUDGE_BASE_URL are required (the gpt-5-mini judge resource)")
    env.update(OPENAI_API_KEY=env["JUDGE_API_KEY"], OPENAI_BASE_URL=env["JUDGE_BASE_URL"],
               CLEAN_MODEL=env.get("CLEAN_MODEL", "gpt-5-mini"))
    if env.get("JINA_READER", "").lower() == "public":
        env["JINA_API_KEY"] = ""
    return env


def setup(workers_dir: Path, n: int) -> None:
    src = REPO / "third_party" / "deep_research_bench"
    if not (src / "utils" / "api.py").exists():
        raise SystemExit(f"{src} is missing: run scripts/bootstrap_third_party.sh first")
    if "RETRY_PATCH" not in (src / "utils" / "api.py").read_text():
        raise SystemExit(f"{src}/utils/api.py is not patched: run scripts/bootstrap_third_party.sh")
    for i in range(1, n + 1):
        w = workers_dir / f"w{i}"
        if w.exists():
            continue
        shutil.copytree(src.resolve(), w, symlinks=True,
                        ignore=shutil.ignore_patterns(".git"))
        print(f"created {w}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", type=Path, help="run roots holding <arm>/<task>-s0 runs")
    ap.add_argument("-n", type=int, default=4, help="parallel judgings (one worker copy each)")
    ap.add_argument("--passes", default="1,2,3")
    ap.add_argument("--workers-dir", type=Path, default=REPO.parent / "drb_workers")
    ap.add_argument("--setup", action="store_true", help="create the worker copies and exit")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--config", default=CONFIG)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.setup:
        setup(a.workers_dir, a.n)
        return
    missing = [i for i in range(1, a.n + 1) if not (a.workers_dir / f"w{i}").exists()]
    if missing:
        raise SystemExit(f"worker copies missing ({missing}): run with --setup -n {a.n}")
    base_env = load_env()
    tasks: "queue.Queue[tuple[int, Path, Path]]" = queue.Queue()
    for p in (int(x) for x in a.passes.split(",")):
        for root in a.roots:
            for run in sorted(root.glob("*/*-s*")):
                if run.is_dir() and has_report(run):
                    tasks.put((p, root, run))
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "skip": 0}

    def worker(i: int) -> None:
        env = dict(base_env, ADR_DRB_DIR=str(a.workers_dir / f"w{i}"))
        while True:
            try:
                p, root, run = tasks.get_nowait()
            except queue.Empty:
                return
            with lock:
                d = target(run, root, p)
            if done(d):
                stats["skip"] += 1
                continue
            t0 = time.time()
            with open(d / "judge.log", "w") as log:
                try:
                    r = subprocess.run([".venv/bin/adr", "evaluate", str(d), "--official", "deep_research_bench",
                                        "-c", a.config], stdout=log, stderr=subprocess.STDOUT, env=env,
                                       timeout=a.timeout)
                    ok = r.returncode == 0 and done(d)
                except subprocess.TimeoutExpired:
                    ok = False
            stats["ok" if ok else "fail"] += 1
            print(f"{time.strftime('%H:%M')} w{i} pass{p} {'judged' if ok else 'FAILED'} {d} "
                  f"({time.time() - t0:.0f}s) [ok {stats['ok']} fail {stats['fail']} skip {stats['skip']} "
                  f"left {tasks.qsize()}]", flush=True)

    print(f"{time.strftime('%H:%M')} {tasks.qsize()} judgings queued, {a.n} workers", flush=True)
    threads = [threading.Thread(target=worker, args=(i + 1,)) for i in range(a.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"{time.strftime('%H:%M')} ALL DONE {stats}", flush=True)


if __name__ == "__main__":
    main()
