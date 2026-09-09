"""Aggregation of official judge outputs into comparable scalars.

Every formula here mirrors the aggregation in the upstream scripts so that
numbers produced by this harness line up with the published ones:

  - DeepResearchGym quality: ``analyze_results`` in ``eval_quality_async.py``
  - DeepResearchGym KPR:     ``analyze_results`` in ``eval_kpr_async.py``
  - DeepResearch Bench:      the ``race_result.txt`` / ``fact_result.txt`` files
    written by ``deepresearch_bench_race.py`` and ``utils.stat``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

GYM_QUALITY_CRITERIA = [
    "Clarity",
    "Depth",
    "Balance",
    "Breadth",
    "Support",
    "Insightfulness",
]

KPR_LABELS = ("Supported", "Omitted", "Contradicted")

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


def aggregate_gym_quality(results: dict[str, Any]) -> dict[str, Any]:
    """Mean of per-query normalized scores, plus per-criterion means.

    Per-query normalized score is ``sum(ratings) / (n_criteria * 10) * 100``,
    matching the upstream definition.
    """
    per_query: dict[str, float] = {}
    criterion_ratings: dict[str, list[int]] = {}

    for query_id, row in results.items():
        scores = (row or {}).get("scores") or {}
        ratings: list[int] = []
        for name, value in scores.items():
            rating = _rating_of(value)
            if rating is None:
                continue
            ratings.append(rating)
            criterion_ratings.setdefault(name, []).append(rating)
        if ratings:
            per_query[query_id] = sum(ratings) / (len(ratings) * 10) * 100

    mean = sum(per_query.values()) / len(per_query) if per_query else 0.0
    per_criterion = {
        name: {
            "average_rating": round(sum(vals) / len(vals), 4),
            "normalized_average": round(sum(vals) / len(vals) / 10 * 100, 4),
        }
        for name, vals in sorted(criterion_ratings.items())
        if vals
    }
    return {
        "n": len(per_query),
        "average_normalized_score": round(mean, 4),
        "per_criterion": per_criterion,
        "per_query_normalized": {k: round(v, 4) for k, v in per_query.items()},
    }


def aggregate_gym_kpr(results: dict[str, Any]) -> dict[str, Any]:
    """Mean of per-query Supported / Omitted / Contradicted rates (percentages)."""
    rates: dict[str, dict[str, float]] = {}
    for query_id, row in results.items():
        labels = (row or {}).get("labels") or {}
        counts = dict.fromkeys(KPR_LABELS, 0)
        total = 0
        for value in labels.values():
            label = _label_of(value)
            if label is None:
                continue
            total += 1
            if label in counts:
                counts[label] += 1
        if not total:
            continue
        rates[query_id] = {
            "support_rate": counts["Supported"] / total * 100,
            "omitted_rate": counts["Omitted"] / total * 100,
            "contradicted_rate": counts["Contradicted"] / total * 100,
            "n_key_points": total,
        }

    n = len(rates) or 1
    return {
        "n": len(rates),
        "average_support_rate": round(sum(r["support_rate"] for r in rates.values()) / n, 4),
        "average_omitted_rate": round(sum(r["omitted_rate"] for r in rates.values()) / n, 4),
        "average_contradicted_rate": round(
            sum(r["contradicted_rate"] for r in rates.values()) / n, 4
        ),
        "per_query": {k: {m: round(v, 4) for m, v in row.items()} for k, row in rates.items()},
    }


def aggregate_gym_citation(results: dict[str, Any]) -> dict[str, Any]:
    """Mean per-query citation faithfulness, scaled to 0-100 as upstream does."""
    scores: dict[str, float] = {}
    for query_id, row in results.items():
        if not isinstance(row, dict):
            continue
        value = row.get("score")
        if isinstance(value, (int, float)):
            scores[query_id] = float(value) * 100
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    return {
        "n": len(scores),
        "average_citation_score": round(mean, 4),
        "per_query": {k: round(v, 4) for k, v in scores.items()},
    }


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

    gym = official.get("deep_research_gym") or {}
    quality = gym.get("quality") or {}
    if isinstance(quality.get("average_normalized_score"), (int, float)):
        flat["gym_quality"] = float(quality["average_normalized_score"])
    for name, row in (quality.get("per_criterion") or {}).items():
        if isinstance(row, dict) and isinstance(row.get("average_rating"), (int, float)):
            flat[f"gym_quality_{name.lower()}"] = float(row["average_rating"])
    kpr = gym.get("kpr") or {}
    for key in ("average_support_rate", "average_omitted_rate", "average_contradicted_rate"):
        if isinstance(kpr.get(key), (int, float)):
            flat[f"gym_{key}"] = float(kpr[key])
    citation = gym.get("citation") or {}
    if isinstance(citation.get("average_citation_score"), (int, float)):
        flat["gym_citation_score"] = float(citation["average_citation_score"])

    bc = official.get("browsecomp") or {}
    for key in ("accuracy", "calibration_error"):
        if isinstance(bc.get(key), (int, float)):
            flat[f"bc_{key}"] = float(bc[key])

    return flat
