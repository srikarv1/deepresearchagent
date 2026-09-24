"""Randomized rollout driver (PILOT §4.5 "Randomized rollouts").

One rollout = one (query, seed) pair run as its own ``adr run`` subprocess with
``GR_ORCHESTRATOR=random`` and ``GR_ORCH_SEED=<seed>``. Subprocesses because
gpt-researcher's token/latency trackers are process-global (the harness must
keep ``concurrency: 1`` inside a process), so parallelism has to come from
separate interpreters.

Layout under ``--out``::

    <out>/
      manifest.jsonl              one row per rollout (append-only, resumable)
      logs/<run_id>.log           subprocess stdout+stderr
      <run_id>/                   a normal adr run directory
        trajectories/<qid>.json   harness trajectory (rounds -> PRUNE steps)
        gpt_researcher/<qid>.json fork trajectory (decision meta, evidence)
        gpt_researcher/<qid>_emb.npz
        metrics/summary.json      local metrics (+ judge scores if evaluated)
        metrics/race_raw_results.jsonl   per-query RACE rows when judged

``run_id`` is ``<qid>-s<seed>`` so a rerun of the same command skips finished
rollouts. Judging (``--evaluate deep_research_bench``) runs *after* each
rollout, serialized across workers: the DRB repo stages one file per model
name, so two concurrent RACE calls would clobber each other.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from adr.datasets.loader import load_queries
from adr.runner.config import load_config

ENV_POLICY = "GR_ORCHESTRATOR"
ENV_SEED = "GR_ORCH_SEED"
MANIFEST = "manifest.jsonl"


@dataclass
class RolloutSpec:
    query_id: str
    seed: int
    run_id: str
    run_dir: Path
    log_path: Path
    dataset: str

    def cmd(self, config: Path, out: Path, extra_args: Iterable[str] = ()) -> list[str]:
        return [
            sys.executable,
            "-m",
            "adr.cli",
            "run",
            "--config",
            str(config),
            "--query-ids",
            self.query_id,
            "--run-id",
            self.run_id,
            "--output-dir",
            str(out),
            "--run-name",
            self.run_id,
            *extra_args,
        ]

    def env(self, policy: str, base_env: dict[str, str] | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(base_env if base_env is not None else os.environ)
        env[ENV_POLICY] = policy
        env[ENV_SEED] = str(self.seed)
        env.update(extra or {})
        return env


@dataclass
class RolloutResult:
    query_id: str
    seed: int
    run_id: str
    run_dir: str
    status: str            # done | skipped | failed | dry-run
    returncode: int | None = None
    wall_s: float | None = None
    evaluated: bool = False
    eval_returncode: int | None = None
    error: str | None = None
    log: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def run_id_for(query_id: str, seed: int) -> str:
    return f"{query_id}-s{seed}"


def is_complete(run_dir: Path, query_id: str) -> bool:
    """A rollout counts as done when the harness trajectory exists with a report."""
    path = run_dir / "trajectories" / f"{query_id}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("report")) and not data.get("error")


def is_evaluated(run_dir: Path) -> bool:
    return (run_dir / "metrics" / "race_raw_results.jsonl").exists()


def plan_rollouts(
    *,
    config: Path,
    out: Path,
    n_seeds: int,
    seed_base: int = 0,
    query_ids: list[str] | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> tuple[list[RolloutSpec], dict[str, Any]]:
    """Enumerate (query, seed) pairs from the run config's dataset section."""
    cfg = load_config(config)
    ds = cfg.get("dataset") or {}
    queries = load_queries(
        ds["name"],
        path=ds.get("path"),
        language=ds.get("language"),
        limit=limit if limit is not None else ds.get("limit"),
        query_ids=query_ids or (ds.get("query_ids") or None),
        split=split or ds.get("split"),
    )
    specs: list[RolloutSpec] = []
    for q in queries:
        for k in range(n_seeds):
            seed = seed_base + k
            rid = run_id_for(q.id, seed)
            specs.append(
                RolloutSpec(
                    query_id=q.id,
                    seed=seed,
                    run_id=rid,
                    run_dir=out / rid,
                    log_path=out / "logs" / f"{rid}.log",
                    dataset=str(ds["name"]),
                )
            )
    return specs, cfg


def _append_manifest(out: Path, row: RolloutResult, lock: threading.Lock) -> None:
    with lock:
        with (out / MANIFEST).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _evaluate(run_dir: Path, bench: str, log_path: Path, env: dict[str, str], timeout_s: float | None) -> int:
    cmd = [sys.executable, "-m", "adr.cli", "evaluate", str(run_dir), "--official", bench]
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout_s)
    return proc.returncode


def run_rollouts(
    *,
    config: Path,
    out: Path,
    n_seeds: int,
    seed_base: int = 0,
    parallel: int = 1,
    policy: str = "random",
    query_ids: list[str] | None = None,
    split: str | None = None,
    limit: int | None = None,
    evaluate: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_run_args: Iterable[str] = (),
    dry_run: bool = False,
    timeout_s: float | None = None,
    on_result: Callable[[RolloutResult], None] | None = None,
) -> list[RolloutResult]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    specs, cfg = plan_rollouts(
        config=config, out=out, n_seeds=n_seeds, seed_base=seed_base,
        query_ids=query_ids, split=split, limit=limit,
    )
    (out / "rollouts_config.json").write_text(
        json.dumps(
            {
                "config": str(config),
                "policy": policy,
                "n_seeds": n_seeds,
                "seed_base": seed_base,
                "parallel": parallel,
                "evaluate": evaluate,
                "extra_env": dict(extra_env or {}),
                "n_rollouts": len(specs),
                "dataset": cfg.get("dataset"),
                "budget": cfg.get("budget"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_lock = threading.Lock()
    eval_lock = threading.Lock()
    results: list[RolloutResult] = []

    def one(spec: RolloutSpec) -> RolloutResult:
        if dry_run:
            return RolloutResult(spec.query_id, spec.seed, spec.run_id, str(spec.run_dir), "dry-run",
                                 extra={"cmd": spec.cmd(config, out, extra_run_args)})
        env = spec.env(policy, extra=extra_env)
        done = is_complete(spec.run_dir, spec.query_id)
        res = RolloutResult(spec.query_id, spec.seed, spec.run_id, str(spec.run_dir),
                            "skipped" if done else "done", log=str(spec.log_path))
        if not done:
            t0 = time.perf_counter()
            with spec.log_path.open("w", encoding="utf-8") as fh:
                fh.write(f"$ {' '.join(spec.cmd(config, out, extra_run_args))}\n")
                fh.write(f"# {ENV_POLICY}={policy} {ENV_SEED}={spec.seed}\n")
                fh.flush()
                try:
                    proc = subprocess.run(
                        spec.cmd(config, out, extra_run_args),
                        stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout_s,
                    )
                    res.returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    res.returncode = -1
                    res.error = f"timeout after {timeout_s}s"
            res.wall_s = round(time.perf_counter() - t0, 2)
            if res.returncode != 0 or not is_complete(spec.run_dir, spec.query_id):
                res.status = "failed"
                res.error = res.error or f"exit {res.returncode} or no report; see {spec.log_path}"
        if evaluate and res.status in {"done", "skipped"} and not is_evaluated(spec.run_dir):
            with eval_lock:
                rc = _evaluate(spec.run_dir, evaluate, spec.log_path, env, timeout_s)
            res.eval_returncode = rc
            res.evaluated = rc == 0 and is_evaluated(spec.run_dir)
        elif evaluate:
            res.evaluated = is_evaluated(spec.run_dir)
        return res

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(one, s): s for s in specs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if not dry_run:
                _append_manifest(out, res, manifest_lock)
            if on_result:
                on_result(res)
    results.sort(key=lambda r: (r.query_id, r.seed))
    return results


def read_manifest(out: Path) -> list[dict[str, Any]]:
    path = Path(out) / MANIFEST
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
