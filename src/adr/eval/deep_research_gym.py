"""DeepResearchGym evaluation via the upstream judge functions.

We dynamically import ``evaluate_folder_async`` from the upstream scripts
and call them directly, bypassing the ``__main__`` blocks which hardcode
cluster paths and accept different CLI flags than this harness needs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.eval.exporters import export_deep_research_gym
from adr.eval.repos import find_deep_research_gym, find_key_points
from adr.eval.scoring import (
    aggregate_gym_citation,
    aggregate_gym_kpr,
    aggregate_gym_quality,
)


def _import_gym_module(gym_root: Path, name: str) -> Any:
    """Dynamically import a module from the Gym checkout by file path."""
    script = gym_root / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(script))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_async(coro: Any, timeout_s: float | None = None) -> Any:
    if timeout_s is not None:
        async def _with_timeout() -> Any:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return asyncio.run(_with_timeout())
    return asyncio.run(coro)


def run_deep_research_gym(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    third_party_dir: str | Path | None = None,
    key_point_dir: str | Path | None = None,
    judge_model: str = "gpt-4.1-mini",
    run_quality: bool = True,
    run_kpr: bool = True,
    run_citation: bool = False,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Write Gym ``.q`` / ``.a`` files, then run the official judges on them."""
    export_dir = run_dir / "exports" / "deep_research_gym" / model_name
    export_deep_research_gym(trajectories, export_dir)

    result: dict[str, Any] = {
        "bench": "deep_research_gym",
        "export": str(export_dir),
        "judge_model": judge_model,
        "official": False,
        "reason": None,
        "quality": None,
        "kpr": None,
        "citation": None,
    }

    located = find_deep_research_gym(third_party_dir)
    if not located.ok:
        result["reason"] = located.reason
        return result
    gym_root = located.path
    result["repo"] = str(gym_root)

    if not os.environ.get("OPENAI_API_KEY"):
        result["reason"] = "OPENAI_API_KEY is not set; the Gym judges cannot run"
        return result

    out_dir = run_dir / "metrics" / "deep_research_gym"
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_quality:
        result["quality"] = _quality(gym_root, export_dir, out_dir, judge_model, timeout_s)
    if run_kpr:
        result["kpr"] = _kpr(
            gym_root, export_dir, out_dir, judge_model, key_point_dir, timeout_s
        )
    if run_citation:
        result["citation"] = _citation(gym_root, export_dir, out_dir, judge_model, timeout_s)

    ran = [
        block
        for block in (result["quality"], result["kpr"], result["citation"])
        if isinstance(block, dict) and block.get("n")
    ]
    result["official"] = bool(ran)
    if not result["official"]:
        result["reason"] = "No Gym metric produced scores; see the per-metric log fields"
    return result


def _quality(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    try:
        mod = _import_gym_module(gym_root, "eval_quality_async")
        # Upstream signature is (report_dir, model, num_workers=32). The report
        # dir is the folder that holds the .q/.a files, not its parent.
        results = _run_async(
            mod.evaluate_folder_async(str(export_dir), judge_model),
            timeout_s=timeout_s,
        )
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"quality_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_quality(results), "path": str(path)}


def _kpr(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    key_point_dir: str | Path | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    key_points = find_key_points(gym_root, key_point_dir)
    if key_points is None:
        return {
            "n": 0,
            "skipped": True,
            "reason": (
                "No aggregated key points found. Expected *_aggregated.json under "
                f"{gym_root / 'key_point'} or set eval.deep_research_gym.key_point_dir"
            ),
        }

    try:
        mod = _import_gym_module(gym_root, "eval_kpr_async")
        # Upstream signature is (report_dir, model, key_point_dir, num_workers=32).
        results = _run_async(
            mod.evaluate_folder_async(str(export_dir), judge_model, str(key_points)),
            timeout_s=timeout_s,
        )
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"relevance_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_kpr(results), "key_point_dir": str(key_points), "path": str(path)}


def _citation(
    gym_root: Path,
    export_dir: Path,
    out_dir: Path,
    judge_model: str,
    timeout_s: float | None,
) -> dict[str, Any]:
    try:
        mod = _import_gym_module(gym_root, "eval_citation_async")
        results = _run_async(
            mod.evaluate_folder_async(str(export_dir), judge_model),
            timeout_s=timeout_s,
        )
    except Exception as e:
        return {"n": 0, "error": str(e)}

    path = out_dir / f"faithfullness_{judge_model}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return {**aggregate_gym_citation(results), "path": str(path)}
