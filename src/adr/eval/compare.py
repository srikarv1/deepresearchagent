from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adr.eval.scoring import headline_scores

# Spend: unambiguously better when smaller.
_COST_KEYS = {
    "mean_tokens",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "total_tokens",
    "mean_wall_s",
    "mean_step_latency_s",
    "mean_n_llm_calls",
    "mean_n_searches",
    "mean_n_reads",
    "mean_n_steps",
}

# Shape of the trajectory and report. Direction depends on what you are testing,
# so these are reported without an implied better/worse.
_STRUCTURE_KEYS = {
    "mean_n_retained",
    "mean_n_pruned",
    "mean_prune_rate",
    "mean_n_citations",
    "mean_article_chars",
    "mean_rounds_before_answer",
    "mean_rounds_after_answer",
    "mean_searches_after_answer",
    "mean_keep_recall_evidence",
    "mean_keep_recall_gold",
    "mean_retrieved_recall_evidence",
    "mean_retrieved_recall_gold",
}


def _numbers(summary: dict[str, Any]) -> dict[str, float]:
    """Cost metrics plus flattened official judge scores."""
    out: dict[str, float] = {
        k: float(v)
        for k, v in summary.items()
        if k.startswith("mean_") and isinstance(v, (int, float))
    }
    if isinstance(summary.get("total_tokens"), (int, float)):
        out["total_tokens"] = float(summary["total_tokens"])
    out.update(headline_scores(summary.get("official") or {}))
    return out


def _pct(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return round((after - before) / abs(before) * 100, 2)


def compare_summaries(left: Path, right: Path) -> dict[str, Any]:
    """Diff two run summaries. ``right`` is treated as the new system."""
    a = json.loads(Path(left).read_text(encoding="utf-8"))
    b = json.loads(Path(right).read_text(encoding="utf-8"))
    a_nums, b_nums = _numbers(a), _numbers(b)

    shared = sorted(set(a_nums) & set(b_nums))
    deltas = {k: round(b_nums[k] - a_nums[k], 4) for k in shared}
    percent = {k: _pct(a_nums[k], b_nums[k]) for k in shared}

    return {
        "left": str(left),
        "right": str(right),
        "n_left": a.get("n_queries"),
        "n_right": b.get("n_queries"),
        "quality_deltas": {
            k: v for k, v in deltas.items() if k not in _COST_KEYS and k not in _STRUCTURE_KEYS
        },
        "cost_deltas": {k: v for k, v in deltas.items() if k in _COST_KEYS},
        "structure_deltas": {k: v for k, v in deltas.items() if k in _STRUCTURE_KEYS},
        "percent_change": percent,
        "deltas_right_minus_left": deltas,
        "left_values": {k: a_nums[k] for k in shared},
        "right_values": {k: b_nums[k] for k in shared},
        "only_left": sorted(set(a_nums) - set(b_nums)),
        "only_right": sorted(set(b_nums) - set(a_nums)),
    }
