"""Build Table 4 from a set of run directories.

Each run dir is one (dataset, orchestration policy, seed). Rows are grouped by
policy; every numeric column is reported as mean ± std over the seeds in the
group. Quality columns come from ``metrics/summary.json`` (``scores``, the
flattened judge output); efficiency columns are per-query means from
``metrics/local.json``.

Nothing here calls a model: if a run has not been judged yet its quality cells
are blank and the efficiency cells still fill in.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

POLICY_ORDER = ["none", "topk", "extractive", "llmlingua", "prompted", "pilot", "legacy"]
POLICY_LABELS = {
    "none": "No pruning",
    "topk": "Top-k similarity",
    "extractive": "Extractive compression",
    "llmlingua": "Prompt compression (LLMLingua-2)",
    "prompted": "Prompted orchestration",
    "pilot": "PILOT",
    "legacy": "Fork default (legacy)",
}

# (column label, source, key). Source "scores" reads summary.scores; "local"
# reads the mean of that key over local.json per_query rows.
QUALITY_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "deep_research_bench": [
        ("RACE overall", "race_overall_score"),
        ("RACE comp.", "race_comprehensiveness"),
        ("RACE insight", "race_insight"),
        ("RACE instr.", "race_instruction_following"),
        ("RACE read.", "race_readability"),
        ("FACT valid %", "fact_valid_rate"),
    ],
    "browsecomp": [
        ("Accuracy %", "bc_accuracy"),
        ("Calib. err %", "bc_calibration_error"),
    ],
    "deep_research_gym": [
        ("Quality", "gym_quality"),
        ("KPR support %", "gym_average_support_rate"),
    ],
}

EFFICIENCY_COLUMNS: list[tuple[str, str, float]] = [
    # label, per_query key, scale
    ("Tokens (K)", "tokens", 1 / 1000),
    ("Latency (s)", "wall_s", 1.0),
    ("LLM calls", "n_llm_calls", 1.0),
    ("Searches", "n_searches", 1.0),
    ("Prune rate", "prune_rate", 1.0),
]


@dataclass
class RunRow:
    run_dir: Path
    dataset: str
    policy: str
    seed: Any
    n_queries: int
    scores: dict[str, float]
    efficiency: dict[str, float]
    judged: bool


@dataclass
class Cell:
    values: list[float] = field(default_factory=list)

    def fmt(self, digits: int = 1) -> str:
        if not self.values:
            return "—"
        mean = statistics.fmean(self.values)
        if len(self.values) == 1:
            return f"{mean:.{digits}f}"
        sd = statistics.stdev(self.values)
        return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load_run(run_dir: Path) -> RunRow | None:
    summary_path = run_dir / "metrics" / "summary.json"
    local_path = run_dir / "metrics" / "local.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = _yaml(run_dir / "config.yaml")

    dataset = summary.get("dataset") or (config.get("dataset") or {}).get("name") or "unknown"
    policy = summary.get("orchestrator") or _orchestrator_of(config) or "legacy"
    seed = summary.get("seed", config.get("seed"))

    efficiency: dict[str, float] = {}
    if local_path.exists():
        local = json.loads(local_path.read_text(encoding="utf-8"))
        rows = [r for r in (local.get("per_query") or []) if not r.get("error")]
        for _, key, scale in EFFICIENCY_COLUMNS:
            vals = [float(r[key]) * scale for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                efficiency[key] = statistics.fmean(vals)

    scores = {k: float(v) for k, v in (summary.get("scores") or {}).items() if isinstance(v, (int, float))}
    judged = any(
        isinstance(b, dict) and b.get("official") for b in (summary.get("official") or {}).values()
    )
    return RunRow(
        run_dir=run_dir,
        dataset=str(dataset),
        policy=str(policy),
        seed=seed,
        n_queries=int(summary.get("n_queries") or 0),
        scores=scores,
        efficiency=efficiency,
        judged=judged,
    )


def collect_runs(paths: list[Path]) -> list[RunRow]:
    rows: list[RunRow] = []
    for p in paths:
        if (p / "metrics").exists():
            candidates = [p]
        else:
            candidates = sorted(c for c in p.iterdir() if c.is_dir()) if p.is_dir() else []
        for c in candidates:
            row = load_run(c)
            if row is not None:
                rows.append(row)
    return rows


def build_table(rows: list[RunRow], dataset: str | None = None) -> dict[str, Any]:
    """Group by policy and aggregate. Returns a dict with ``dataset``,
    ``columns``, ``rows`` (label, n_runs, seeds, cells) and ``notes``."""
    if dataset is None:
        datasets = sorted({r.dataset for r in rows})
        if len(datasets) != 1:
            raise ValueError(f"Runs span several datasets {datasets}; pass --dataset")
        dataset = datasets[0]
    rows = [r for r in rows if r.dataset == dataset]

    q_cols = QUALITY_COLUMNS.get(dataset, [])
    columns = [label for label, _ in q_cols] + [label for label, _, _ in EFFICIENCY_COLUMNS]

    grouped: dict[str, list[RunRow]] = {}
    for r in rows:
        grouped.setdefault(r.policy, []).append(r)

    ordered = sorted(grouped, key=lambda p: (POLICY_ORDER.index(p) if p in POLICY_ORDER else 99, p))
    out_rows = []
    notes: list[str] = []
    for policy in ordered:
        runs = grouped[policy]
        cells: list[Cell] = []
        for _, key in q_cols:
            cells.append(Cell([r.scores[key] for r in runs if key in r.scores]))
        for _, key, _ in EFFICIENCY_COLUMNS:
            cells.append(Cell([r.efficiency[key] for r in runs if key in r.efficiency]))
        unjudged = [r for r in runs if not r.judged]
        if unjudged and q_cols:
            notes.append(
                f"{POLICY_LABELS.get(policy, policy)}: {len(unjudged)}/{len(runs)} runs not judged yet "
                f"({', '.join(r.run_dir.name for r in unjudged)})"
            )
        n_q = sorted({r.n_queries for r in runs})
        if len(n_q) > 1:
            notes.append(f"{POLICY_LABELS.get(policy, policy)}: runs have different n_queries {n_q}")
        out_rows.append(
            {
                "policy": policy,
                "label": POLICY_LABELS.get(policy, policy),
                "n_runs": len(runs),
                "seeds": [r.seed for r in runs],
                "n_queries": n_q[0] if n_q else 0,
                "cells": cells,
            }
        )
    return {"dataset": dataset, "columns": columns, "rows": out_rows, "notes": notes}


def to_markdown(table: dict[str, Any]) -> str:
    cols = ["Method", "Seeds", "n"] + table["columns"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in table["rows"]:
        digits = [1] * len(row["cells"])
        # Prune rate is a fraction; show two decimals.
        if "Prune rate" in table["columns"]:
            digits[table["columns"].index("Prune rate")] = 2
        cells = [c.fmt(d) for c, d in zip(row["cells"], digits)]
        lines.append(
            "| " + " | ".join([row["label"], str(row["n_runs"]), str(row["n_queries"]), *cells]) + " |"
        )
    text = "\n".join(lines) + "\n"
    if table["notes"]:
        text += "\n" + "\n".join(f"- {n}" for n in table["notes"]) + "\n"
    return text


def _yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _orchestrator_of(config: dict[str, Any]) -> str | None:
    agent = config.get("agent") or {}
    for block in (agent.get("overrides") or {}, agent):
        env = (block.get("env") or {}) if isinstance(block, dict) else {}
        if env.get("GR_ORCHESTRATOR"):
            return str(env["GR_ORCHESTRATOR"])
    return None
