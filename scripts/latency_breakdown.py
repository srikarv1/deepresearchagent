#!/usr/bin/env python3
"""Per-arm latency breakdown for `adr rollouts` runs, from files already on disk.

Raw wall time is not comparable across arms that were run on different
machines or at different times: web search, page fetching and embeddings
depend on the network, and Azure Flex adds queue time. This prints, per arm,
the mean of:

  wall      raw wall time (what the tables report today)
  model     time with at least one model call running (overlaps counted once)
  orch      the same for orchestration decisions only (usage_tag "orchestrator")
  queue     Flex capacity waiting inside model calls
  search    time with at least one search running (runs recorded with search
            intervals only; "-" otherwise, and search is then inside "other")
  other     wall time with neither a model call nor a search running

`model` and `orch` depend on the model endpoints rather than the machine that
ran the job, so they are the fair basis for comparing methods across machines.

    python scripts/latency_breakdown.py runs/gym_sol_budget150
    python scripts/latency_breakdown.py runs/drb_sol_budget150 --arms none bc dpo
    python scripts/latency_breakdown.py runs/gym_sol_budget150 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adr.eval.latency import latency_breakdown  # noqa: E402

FIELDS = ("wall_s", "llm_active_s", "orchestrator_active_s", "capacity_wait_s", "search_active_s", "other_s")
LABELS = ("wall", "model", "orch", "queue", "search", "other")


def run_breakdown(traj_path: Path) -> dict | None:
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    fs = data.get("final_stats") or {}
    if fs.get("latency_breakdown"):
        return fs["latency_breakdown"]
    calls = fs.get("llm_calls")
    wall = fs.get("agent_wall_s") or fs.get("wall_s")
    if not calls or wall is None:
        return None
    return latency_breakdown(float(wall), calls)


def arm_rows(arm_dir: Path) -> dict[str, dict]:
    rows = {}
    for traj in sorted(arm_dir.glob("*/trajectories/*.json")):
        bd = run_breakdown(traj)
        if bd is not None:
            rows[traj.parent.parent.name] = bd
    return rows


def mean_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return st.mean(vals) if vals else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="run root holding one directory per arm")
    ap.add_argument("--arms", nargs="*", help="arms to include (default: every arm with trajectories)")
    ap.add_argument("--json", type=Path, help="also write per-run and per-arm values here")
    args = ap.parse_args(argv)

    arms = args.arms or sorted(p.name for p in args.root.iterdir() if p.is_dir() and any(p.glob("*/trajectories/*.json")))
    out: dict[str, dict] = {}
    print(f"{'arm':12s} {'runs':>4s} " + " ".join(f"{l:>8s}" for l in LABELS))
    for arm in arms:
        rows = arm_rows(args.root / arm)
        if not rows:
            print(f"{arm:12s}    0  (no trajectories with a per-call LLM log)")
            continue
        means = {f: mean_or_none([r.get(f) for r in rows.values()]) for f in FIELDS}
        out[arm] = {"runs": rows, "mean": means}
        cells = " ".join(f"{'-':>8s}" if means[f] is None else f"{means[f]:8.1f}" for f in FIELDS)
        print(f"{arm:12s} {len(rows):4d} {cells}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
