"""Trajectory reward R(tau), PILOT Eq. 14 / 17.

    R = alpha_q * Q~(y, q) + alpha_g * G(y, q)
        - lambda_tok * c_tok / B_tok - lambda_lat * c_lat / B_lat
        + sum_t [ gamma * Phi(s_{t+1}) - Phi(s_t) ]

  Q     RACE overall score in [0, 1] (per-query row from raw_results.jsonl),
        floored: Q~ = (Q - q_min) * 1[Q >= q_min]           (Eq. 17)
  G     goal satisfaction in [0, 1]; optional external file, default 0 and
        alpha_g = 0 for BC v1 (needs a sub-question judge, not built yet)
  c_tok harness-measured total tokens; c_lat harness wall clock
  Phi   coverage potential; the RandomizedPolicy logs phi_prev / phi_after per
        round, so shaping is read straight from decision meta (Eq. 15-16)

Budgets default to the run config's ``budget.max_tokens`` /
``budget.max_latency_s`` (what the rollout was run under), so the same run
can be re-scored under a different budget by overriding them here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RewardConfig:
    alpha_q: float = 1.0
    alpha_g: float = 0.0
    lambda_tok: float = 0.1
    lambda_lat: float = 0.1
    q_min: float = 0.0
    gamma: float = 1.0
    shaping: bool = True
    shaping_weight: float = 1.0
    token_budget: int | None = None
    latency_budget_s: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardConfig":
        data = dict(data or {})
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class RewardBreakdown:
    quality: float | None          # raw Q
    quality_floored: float          # Q~
    goal: float                    # G
    tokens: int
    latency_s: float
    tokens_norm: float
    latency_norm: float
    shaping: float
    total: float
    has_quality: bool
    n_rounds: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Inputs from a run directory
# --------------------------------------------------------------------------


def load_run_budget(run_dir: Path) -> tuple[int | None, float | None]:
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.exists():
        return None, None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    budget = cfg.get("budget") or {}
    tok = budget.get("max_tokens")
    lat = budget.get("max_latency_s")
    return (int(tok) if tok else None, float(lat) if lat else None)


def load_quality(run_dir: Path, query_id: str) -> float | None:
    """RACE overall score for ``query_id``: from the run's copied raw results,
    else from ``summary.json`` (``official.deep_research_bench.race.per_query``)."""
    run_dir = Path(run_dir)
    copy = run_dir / "metrics" / "race_raw_results.jsonl"
    if copy.exists():
        from adr.eval.deep_research_bench import read_race_raw_results

        rows = read_race_raw_results(copy)
        row = rows.get(str(query_id))
        if row and "overall_score" in row:
            return float(row["overall_score"])
    summary = run_dir / "metrics" / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        race = ((data.get("official") or {}).get("deep_research_bench") or {}).get("race") or {}
        per_query = race.get("per_query") or {}
        row = per_query.get(str(query_id))
        if isinstance(row, dict) and "overall_score" in row:
            return float(row["overall_score"])
        if isinstance(row, (int, float)):
            return float(row)
    return None


def load_goal(goal_file: Path | None, run_id: str, query_id: str) -> float:
    """Optional ``{run_id: G}`` or ``{query_id: G}`` JSON; missing -> 0."""
    if goal_file is None or not Path(goal_file).exists():
        return 0.0
    data = json.loads(Path(goal_file).read_text(encoding="utf-8"))
    for key in (run_id, str(query_id)):
        if key in data:
            return float(data[key])
    return 0.0


def shaping_from_rounds(rounds: list[dict[str, Any]], gamma: float) -> tuple[float, int]:
    """sum_t gamma * Phi(K_t) - Phi(K_{t-1}) using the policy's logged
    ``phi_prev`` / ``phi_after``; rounds without them contribute 0."""
    total = 0.0
    n = 0
    for snap in rounds:
        meta = ((snap.get("decision") or {}).get("meta")) or {}
        if "phi_prev" in meta and "phi_after" in meta:
            total += gamma * float(meta["phi_after"]) - float(meta["phi_prev"])
            n += 1
    return total, n


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


def compute_reward(
    *,
    quality: float | None,
    goal: float,
    tokens: int,
    latency_s: float,
    rounds: list[dict[str, Any]],
    cfg: RewardConfig,
) -> RewardBreakdown:
    q = None if quality is None else float(quality)
    if q is not None and q > 1.0:  # tolerate 0-100 inputs
        q = q / 100.0
    q_floor = 0.0 if q is None or q < cfg.q_min else (q - cfg.q_min)

    b_tok = cfg.token_budget or 1
    b_lat = cfg.latency_budget_s or 1.0
    tok_norm = float(tokens) / float(b_tok)
    lat_norm = float(latency_s) / float(b_lat)

    shaping, n_shaped = (0.0, 0)
    if cfg.shaping:
        shaping, n_shaped = shaping_from_rounds(rounds, cfg.gamma)
        shaping *= cfg.shaping_weight

    total = (
        cfg.alpha_q * q_floor
        + cfg.alpha_g * float(goal)
        - cfg.lambda_tok * tok_norm
        - cfg.lambda_lat * lat_norm
        + shaping
    )
    return RewardBreakdown(
        quality=q,
        quality_floored=q_floor,
        goal=float(goal),
        tokens=int(tokens),
        latency_s=float(latency_s),
        tokens_norm=tok_norm,
        latency_norm=lat_norm,
        shaping=shaping,
        total=float(total),
        has_quality=q is not None,
        n_rounds=len(rounds),
        detail={"n_shaped_rounds": n_shaped, "token_budget": b_tok, "latency_budget_s": b_lat},
    )


def reward_for_run(
    run_dir: Path,
    query_id: str,
    *,
    cfg: RewardConfig,
    raw_trajectory: dict[str, Any] | None,
    harness_trajectory: dict[str, Any],
    goal_file: Path | None = None,
) -> RewardBreakdown:
    """Assemble inputs from a rollout run directory and score it."""
    run_dir = Path(run_dir)
    cfg_eff = RewardConfig(**asdict(cfg))
    if cfg_eff.token_budget is None or cfg_eff.latency_budget_s is None:
        tok_b, lat_b = load_run_budget(run_dir)
        cfg_eff.token_budget = cfg_eff.token_budget or tok_b
        cfg_eff.latency_budget_s = cfg_eff.latency_budget_s or lat_b

    stats = harness_trajectory.get("final_stats") or {}
    usage = stats.get("usage") or {}
    tokens = int(usage.get("total_tokens") or 0)
    if not tokens and raw_trajectory:
        tokens = int(raw_trajectory.get("total_tokens") or 0)
    latency = float(stats.get("wall_s") or stats.get("agent_wall_s") or 0.0)
    if not latency and raw_trajectory:
        latency = float(raw_trajectory.get("total_latency") or 0.0)

    rounds = list((raw_trajectory or {}).get("rounds") or [])
    return compute_reward(
        quality=load_quality(run_dir, query_id),
        goal=load_goal(goal_file, run_dir.name, query_id),
        tokens=tokens,
        latency_s=latency,
        rounds=rounds,
        cfg=cfg_eff,
    )
