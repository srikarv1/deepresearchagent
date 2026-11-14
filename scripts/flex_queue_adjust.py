#!/usr/bin/env python3
"""Take Azure Flex waiting out of the latency of `adr rollouts` runs on Sol.

Flex delays a call in two ways, and both can be read from the fork's per-call
LLM log, ``final_stats.llm_calls`` of each trajectory (fork PR #12; older runs
wrote the same information to ``<arm>/llm_calls/<run>.jsonl``, which is read
when present):

  refused   the request is rejected (429 no_capacity) and retried. The time
            spent waiting out those rejections is measured exactly
            (``capacity_wait_s``).
  queued    the accepted request waits before it is processed. This is
            estimated as the call's duration minus what standard tier takes
            for the same input and output tokens.

The standard-tier duration model is ``a + b*input_tokens + c*output_tokens``.
The default coefficients were fit on 611 standard-tier Sol calls from the
DeepResearchGym pilot (R^2 0.97); ``--fit`` refits them from the standard-tier
calls found under the given directories.

A run's queue time is the union of its calls' waiting intervals, so calls that
overlap in time are counted once. Only Flex calls are adjusted. A call's waits are placed at its start (the
log keeps the total, not when each retry happened), which matters only for how
overlapping calls are merged; with concurrency 1, as in these configs, calls
rarely overlap.

    python scripts/flex_queue_adjust.py runs/gym_sol_budget150
    python scripts/flex_queue_adjust.py runs/drb_sol_budget150 --arms none bc

Writes ``<root>/queue_adjusted.json``:
``{"model": {...}, "runs": {arm: {run: {wall_s, queue_s, adjusted_wall_s, ...}}}}``.
``scripts/budget_table.py`` reads it for the Latency column.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SOL_MODELS = ("gpt-6.1-sol",)
# seconds = a + b * input_tokens + c * output_tokens, standard tier.
DEFAULT_COEF = (2.23998987110841, -4.4760121998408825e-06, 0.01822918253254161)
# A killed attempt of the same query appends to the same log; keep the calls of
# the attempt that finished (those inside its wall-clock window, plus slack).
WINDOW_SLACK_S = 30.0


def is_sol(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in SOL_MODELS)


def read_calls(path: Path) -> list[dict]:
    calls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            calls.append(json.loads(line))
    return calls


def from_call_log(record: dict) -> dict:
    """One ``final_stats.llm_calls`` record (fork LLMCallLog) in the fields used
    below: the accepted attempt starts after the capacity wait, and its
    duration is the call's duration minus that wait."""
    wait = float(record.get("capacity_wait_s") or 0.0)
    duration = float(record.get("duration_s") or 0.0)
    return {
        "model": record.get("model") or "",
        "service_tier": record.get("tier_requested") or record.get("tier_served") or "",
        "start": float(record["start"]),
        "attempt_start": float(record["start"]) + wait,
        "end": float(record.get("end") or record["start"] + duration),
        "duration_s": max(0.0, duration - wait),
        "input_tokens": int(record.get("input_tokens") or 0),
        "output_tokens": int(record.get("output_tokens") or 0),
        # Normal attempts plus capacity rejections, as in the older jsonl log.
        "attempts": int(record.get("attempts") or 1) + int(record.get("capacity_retries") or 0),
        "ok": record.get("status", "ok") == "ok",
    }


def run_calls(run_dir: Path, arm_root: Path) -> tuple[list[dict], bool]:
    """(calls of one run, whether they came from the older jsonl log)."""
    log = arm_root / "llm_calls" / f"{run_dir.name}.jsonl"
    if log.exists():
        return read_calls(log), True
    qid = run_dir.name.split("-s")[0]
    traj = run_dir / "trajectories" / f"{qid}.json"
    if not traj.exists():
        return [], False
    records = (json.loads(traj.read_text(encoding="utf-8")).get("final_stats") or {}).get("llm_calls") or []
    return [from_call_log(r) for r in records], False


def fit(calls: list[dict]) -> tuple[tuple[float, float, float], float]:
    """Least squares on first-attempt standard-tier Sol calls; returns (coef, R^2)."""
    import numpy as np

    rows = [c for c in calls if is_sol(c["model"]) and c["service_tier"] != "flex" and c.get("attempts", 1) == 1]
    if len(rows) < 30:
        raise SystemExit(f"--fit needs at least 30 standard-tier Sol calls, found {len(rows)}")
    x = np.array([[1.0, c["input_tokens"], c["output_tokens"]] for c in rows])
    y = np.array([c["duration_s"] for c in rows])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coef
    r2 = 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return (float(coef[0]), float(coef[1]), float(coef[2])), r2


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_start is None or start > cur_end:
            if cur_start is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def queue_intervals(calls: list[dict], coef: tuple[float, float, float]) -> tuple[list[tuple[float, float]], float, float]:
    """Waiting intervals of the Flex Sol calls, and the refused / queued totals."""
    intervals, refused, queued = [], 0.0, 0.0
    for c in calls:
        if not is_sol(c["model"]) or c["service_tier"] != "flex":
            continue
        if c["attempt_start"] > c["start"]:
            intervals.append((c["start"], c["attempt_start"]))
            refused += c["attempt_start"] - c["start"]
        if not c.get("ok", True):
            continue  # failed call: no accepted attempt to compare
        expected = coef[0] + coef[1] * c["input_tokens"] + coef[2] * c["output_tokens"]
        excess = c["duration_s"] - expected
        if excess > 0:
            intervals.append((c["attempt_start"], c["attempt_start"] + excess))
            queued += excess
    return intervals, refused, queued


def adjust_run(run_dir: Path, arm_root: Path, coef: tuple[float, float, float]) -> dict | None:
    summary = run_dir / "metrics" / "summary.json"
    if not summary.exists():
        return None
    wall = float(json.loads(summary.read_text(encoding="utf-8"))["mean_wall_s"])
    calls, appended = run_calls(run_dir, arm_root)
    if not calls:
        return None
    if appended:
        end = max(c["end"] for c in calls)
        calls = [c for c in calls if c["start"] >= end - wall - WINDOW_SLACK_S]
    intervals, refused, queued = queue_intervals(calls, coef)
    queue = min(union_length(intervals), wall)
    sol = [c for c in calls if is_sol(c["model"])]
    return {
        "wall_s": round(wall, 1),
        "queue_s": round(queue, 1),
        "adjusted_wall_s": round(wall - queue, 1),
        "refused_s": round(refused, 1),
        "queued_s": round(queued, 1),
        "n_sol_calls": len(sol),
        "n_flex_calls": sum(c["service_tier"] == "flex" for c in sol),
        "n_refusals": sum(max(0, int(c.get("attempts", 1)) - 1) for c in sol if c["service_tier"] == "flex"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="run root with one directory per arm, e.g. runs/gym_sol_budget150")
    ap.add_argument("--arms", nargs="*", help="arms to adjust (default: every arm directory with runs)")
    ap.add_argument("--fit", nargs="+", type=Path, metavar="DIR",
                    help="refit the standard-tier model from the llm_calls logs under these directories")
    ap.add_argument("--out", type=Path, help="output file (default: <root>/queue_adjusted.json)")
    args = ap.parse_args()

    coef, model = DEFAULT_COEF, {"coef": list(DEFAULT_COEF), "source": "default (611 standard-tier Sol calls)"}
    if args.fit:
        calls = [c for d in args.fit for f in sorted(d.rglob("llm_calls/*.jsonl")) for c in read_calls(f)]
        for d in args.fit:
            for traj in sorted(d.rglob("trajectories/*.json")):
                stats = json.loads(traj.read_text(encoding="utf-8")).get("final_stats") or {}
                calls += [from_call_log(r) for r in stats.get("llm_calls") or [] if r.get("status") == "ok"]
        coef, r2 = fit(calls)
        model = {"coef": list(coef), "r2": round(r2, 4), "source": "fit on " + ", ".join(map(str, args.fit))}
    print(f"standard-tier model: {coef[0]:.2f}s + {coef[1] * 1e3:.4f}s per 1K input + {coef[2] * 1e3:.2f}s per 1K output tokens")

    arms = args.arms or sorted(p.name for p in args.root.iterdir() if p.is_dir() and any(p.glob("*-s*")))
    if not arms:
        raise SystemExit(f"{args.root}: no arm directory with runs")
    runs: dict[str, dict[str, dict]] = {}
    for arm in arms:
        rows = {}
        missing = 0
        for run_dir in sorted((args.root / arm).glob("*-s*")):
            row = adjust_run(run_dir, args.root / arm, coef)
            if row is None:
                missing += 1
            else:
                rows[run_dir.name] = row
        runs[arm] = rows
        if rows:
            wall = st.mean(r["wall_s"] for r in rows.values())
            queue = st.mean(r["queue_s"] for r in rows.values())
            print(f"{arm:12s} {len(rows):3d} runs  wall {wall:7.1f}s  queue {queue:6.1f}s  adjusted {wall - queue:7.1f}s  "
                  f"refusals {sum(r['n_refusals'] for r in rows.values())}"
                  + (f"  ({missing} without a call log or summary; made with a fork before PR #12?)" if missing else ""))
        else:
            print(f"{arm:12s} no run with both a call log and metrics/summary.json")

    out = args.out or args.root / "queue_adjusted.json"
    out.write_text(json.dumps({"model": model, "runs": runs}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
