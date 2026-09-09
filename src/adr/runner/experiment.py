from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm_asyncio

from adr.agents.base import AgentContext
from adr.agents.registry import build_agent
from adr.core.instrument import CostMeter, MeteredLLM, MeteredSearch
from adr.core.types import Budget, Query, ResearchTask, Trajectory
from adr.datasets.loader import load_queries
from adr.eval.deep_research_bench import run_deep_research_bench
from adr.eval.deep_research_gym import run_deep_research_gym
from adr.eval.exporters import export_deep_research_bench, export_deep_research_gym
from adr.eval.local_metrics import compute_local_metrics, write_local_metrics
from adr.eval.scoring import headline_scores
from adr.llm.factory import build_llm
from adr.tools.search import build_search


@dataclass
class RunManifest:
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    query_ids: list[str]
    started_at: str
    finished_at: str | None = None
    error: str | None = None


def run_experiment(config: dict[str, Any]) -> RunManifest:
    return asyncio.run(run_experiment_async(config))


async def run_experiment_async(config: dict[str, Any]) -> RunManifest:
    started = datetime.now(timezone.utc)
    run_id = _run_id(config, started)
    run_dir = Path(config.get("output_dir", "runs")) / run_id
    _prepare_run_dir(run_dir, config)

    queries = load_queries(
        config["dataset"]["name"],
        path=config["dataset"].get("path"),
        language=config["dataset"].get("language"),
        limit=config["dataset"].get("limit"),
        query_ids=config["dataset"].get("query_ids") or None,
    )
    manifest = RunManifest(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        query_ids=[q.id for q in queries],
        started_at=started.isoformat(),
    )
    llm = build_llm(config.get("llm") or {})
    search = build_search(config.get("search") or {})
    agent_cfg = config.get("agent") or {}
    agent = build_agent(agent_cfg.get("name", "fixture"), agent_cfg.get("config"))
    budget_cfg = dict(config.get("budget") or {})
    enforce_budget = bool(budget_cfg.pop("enforce", False))

    concurrency = max(1, int(config.get("concurrency", 1)))
    semaphore = asyncio.Semaphore(concurrency)
    trajectories: list[Trajectory] = []

    async def _one(query: Query) -> Trajectory:
        async with semaphore:
            task = ResearchTask(query=query, budget=Budget(**budget_cfg))
            # Metered per query so cost is measured by the harness, not self-reported.
            meter = CostMeter()
            ctx = AgentContext(
                llm=MeteredLLM(llm, meter, budget=task.budget, enforce=enforce_budget),
                search=MeteredSearch(search, meter, budget=task.budget, enforce=enforce_budget),
                extra={"config": config, "meter": meter, "run_dir": str(run_dir)},
            )
            t0 = time.perf_counter()
            try:
                traj = await agent.run(task, ctx)
            except NotImplementedError as exc:
                traj = Trajectory(query=query, error=str(exc))
            except Exception as exc:
                traj = Trajectory(query=query, error=f"{type(exc).__name__}: {exc}")
                (run_dir / "errors" / f"{query.id}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if traj.final_stats is None:
                traj.final_stats = {}
            traj.final_stats["wall_s"] = round(time.perf_counter() - t0, 4)
            traj.final_stats["usage"] = meter.snapshot()
            traj.final_stats["budget_violations"] = list(meter.violations)
            _write_query_artifacts(run_dir, traj)
            return traj

    (run_dir / "errors").mkdir(exist_ok=True)
    trajectories = list(await tqdm_asyncio.gather(*[_one(q) for q in queries]))

    metrics = compute_local_metrics(trajectories)
    write_local_metrics(run_dir / "metrics" / "local.json", metrics)

    model_name = str(agent_cfg.get("name") or "agent")
    dataset_name = config["dataset"]["name"]
    if dataset_name == "deep_research_bench":
        export_deep_research_bench(trajectories, run_dir / "exports" / "deep_research_bench" / f"{model_name}.jsonl")
    else:
        export_deep_research_gym(trajectories, run_dir / "exports" / "deep_research_gym" / model_name)

    official: dict[str, Any] = {}
    for bench in config.get("eval", {}).get("official_benches") or []:
        official[bench] = await asyncio.to_thread(_run_official, bench, trajectories, run_dir, model_name, config)

    summary = {
        "run_id": run_id,
        "agent": model_name,
        "dataset": dataset_name,
        "n_queries": len(trajectories),
        **{k: v for k, v in metrics.items() if k != "per_query"},
        "scores": headline_scores(official),
        "official": official,
    }
    (run_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    (run_dir / "manifest.json").write_text(manifest_json(manifest), encoding="utf-8")
    return manifest


def evaluate_run_dir(run_dir: Path, *, official_benches: list[str], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score an existing run (or an imported report folder) without re-running the agent."""
    run_dir = Path(run_dir)
    trajectories = _load_trajectories(run_dir)
    if not trajectories:
        raise FileNotFoundError(f"No trajectories in {run_dir / 'trajectories'}")
    metrics = compute_local_metrics(trajectories)
    write_local_metrics(run_dir / "metrics" / "local.json", metrics)
    cfg = config or {}
    model_name = ((cfg.get("agent") or {}).get("name")) or run_dir.name
    official: dict[str, Any] = {}
    for bench in official_benches:
        official[bench] = _run_official(bench, trajectories, run_dir, model_name, cfg)
    summary = {
        "run_id": run_dir.name,
        "n_queries": len(trajectories),
        **{k: v for k, v in metrics.items() if k != "per_query"},
        "scores": headline_scores(official),
        "official": official,
    }
    (run_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_official(
    bench: str,
    trajectories: list[Trajectory],
    run_dir: Path,
    model_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if bench in {"deep_research_bench", "drb"}:
        eval_cfg = _eval_file(config, "deep_research_bench")
        return run_deep_research_bench(
            trajectories,
            run_dir=run_dir,
            model_name=model_name,
            third_party_dir=eval_cfg.get("third_party_dir"),
            language=eval_cfg.get("language") or (config.get("dataset") or {}).get("language") or "en",
            workers=int(eval_cfg.get("workers", 4)),
            skip_cleaning=bool(eval_cfg.get("skip_cleaning", False)),
            run_race=bool(eval_cfg.get("run_race", True)),
            run_fact=bool(eval_cfg.get("run_fact", True)),
            timeout_s=eval_cfg.get("timeout_s"),
        )
    if bench in {"deep_research_gym", "gym"}:
        eval_cfg = _eval_file(config, "deep_research_gym")
        return run_deep_research_gym(
            trajectories,
            run_dir=run_dir,
            model_name=model_name,
            third_party_dir=eval_cfg.get("third_party_dir"),
            key_point_dir=eval_cfg.get("key_point_dir"),
            judge_model=eval_cfg.get("judge_model", "gpt-4.1-mini"),
            run_quality=bool(eval_cfg.get("run_quality", True)),
            run_kpr=bool(eval_cfg.get("run_kpr", True)),
            run_citation=bool(eval_cfg.get("run_citation", False)),
            timeout_s=eval_cfg.get("timeout_s"),
        )
    raise ValueError(f"Unknown bench {bench!r}")


def _eval_file(config: dict[str, Any], name: str) -> dict[str, Any]:
    from adr.runner.config import ROOT

    path = ROOT / "configs" / "eval" / f"{name}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extra = (config.get("eval") or {}).get(name) or {}
    if isinstance(data, dict) and isinstance(extra, dict):
        data.update(extra)
    return data if isinstance(data, dict) else {}


def _prepare_run_dir(run_dir: Path, config: dict[str, Any]) -> None:
    for part in ("queries", "reports", "trajectories", "exports", "metrics"):
        (run_dir / part).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_query_artifacts(run_dir: Path, traj: Trajectory) -> None:
    (run_dir / "queries" / f"{traj.query.id}.json").write_text(
        traj.query.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "trajectories" / f"{traj.query.id}.json").write_text(
        traj.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    if traj.report:
        (run_dir / "reports" / f"{traj.query.id}.md").write_text(traj.report.article, encoding="utf-8")


def _load_trajectories(run_dir: Path) -> list[Trajectory]:
    rows: list[Trajectory] = []
    for path in sorted((run_dir / "trajectories").glob("*.json")):
        rows.append(Trajectory.model_validate_json(path.read_text(encoding="utf-8")))
    return rows


def _run_id(config: dict[str, Any], started: datetime) -> str:
    stamp = started.strftime("%Y%m%d-%H%M%S")
    name = str(config.get("run_name") or "run")
    return f"{stamp}-{name}"


def manifest_json(manifest: RunManifest) -> str:
    payload = {
        "run_id": manifest.run_id,
        "run_dir": str(manifest.run_dir),
        "query_ids": manifest.query_ids,
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "error": manifest.error,
    }
    return json.dumps(payload, indent=2) + "\n"
