"""Aggregation of official judge outputs into comparable scalars.

Every formula here mirrors the aggregation in the upstream scripts so that
numbers produced by this harness line up with the published ones:

  - DeepResearch Bench: the ``race_result.txt`` / ``fact_result.txt`` files
    written by ``deepresearch_bench_race.py`` and ``utils.stat``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_NUM = re.compile(r"^\s*([A-Za-z_ ]+?)\s*:\s*([-+0-9.eE]+)\s*$")


def _label_of(value: Any) -> str | None:
    """Official results store ``(label, justification)`` tuples as JSON lists."""
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, dict):
        return str(value.get("label")) if value.get("label") else None
    if isinstance(value, str):
        return value
    return None


def _rating_of(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and value:
        try:
            return int(value[0])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict) and value.get("rating") is not None:
        try:
            return int(value["rating"])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def parse_key_value_report(path: str | Path) -> dict[str, float]:
    """Parse the ``Name: value`` files written by RACE and FACT.

    Keys are normalized to snake_case, e.g. ``Overall Score`` -> ``overall_score``.
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _NUM.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        try:
            out[key] = float(match.group(2))
        except ValueError:
            continue
    return out


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def headline_scores(official: dict[str, Any]) -> dict[str, float]:
    """Flatten the per-bench official blocks into comparable top-line numbers.

    These are the keys ``adr compare`` diffs alongside the local cost metrics.
    """
    flat: dict[str, float] = {}

    drb = official.get("deep_research_bench") or {}
    race = (drb.get("race") or {}).get("scores") or {}
    for key in ("overall_score", "comprehensiveness", "insight", "instruction_following", "readability"):
        if isinstance(race.get(key), (int, float)):
            flat[f"race_{key}"] = float(race[key])
    fact = (drb.get("fact") or {}).get("scores") or {}
    for key in ("valid_rate", "total_citations", "total_valid_citations"):
        if isinstance(fact.get(key), (int, float)):
            flat[f"fact_{key}"] = float(fact[key])

    bcp = official.get("browsecomp_plus") or {}
    for key in ("accuracy", "recall"):
        if isinstance(bcp.get(key), (int, float)):
            flat[f"bcp_{key}"] = float(bcp[key])

    return flat
