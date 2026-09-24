"""Build ``bc_pairs.jsonl`` from randomized rollouts (PILOT §4.6 Stage 1).

For every rollout run directory: load the harness trajectory (report, cost),
the fork trajectory (per-round decisions with the RandomizedPolicy's logged
features), and the RACE quality score; compute R(tau); keep the top-return
trajectories; and emit one row per orchestration round:

    prompt      serialized state s_t   (fork serialize_state, from meta.features)
    completion  serialized action a_t  (KEEP / ALLOC / DECISION)
    messages    [{system}, {user: prompt}, {assistant: completion}]  for SFT

The state is rebuilt from what the policy *saw*: pool = kept ∪ pruned ids of
that round (``new_item_ids`` marks the new ones), features and frontier gaps
from ``decision.meta``, snippets from the evidence text. It therefore matches
the online prompt exactly; ``tests/test_bc_pairs.py`` pins that.

Selection follows the paper (top decile by return) but keeps at least
``min_per_query`` trajectories per query so every query is represented, and
drops rollouts with no report or, by default, no quality score.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from adr.rollouts.fork import load_fork_module
from adr.rollouts.reward import RewardBreakdown, RewardConfig, reward_for_run


@dataclass
class RolloutRecord:
    run_dir: Path
    run_id: str
    query_id: str
    dataset: str
    harness: dict[str, Any]
    raw: dict[str, Any] | None
    reward: RewardBreakdown | None = None
    reason: str | None = None  # why it was dropped, if it was

    @property
    def policy(self) -> str | None:
        if not self.raw or not self.raw.get("rounds"):
            return None
        dec = self.raw["rounds"][0].get("decision") or {}
        return dec.get("policy")

    @property
    def params(self) -> dict[str, Any]:
        if not self.raw or not self.raw.get("rounds"):
            return {}
        meta = (self.raw["rounds"][0].get("decision") or {}).get("meta") or {}
        return dict(meta.get("params") or {})


@dataclass
class BuildStats:
    n_run_dirs: int = 0
    n_loaded: int = 0
    n_dropped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_selected: int = 0
    n_rows: int = 0
    n_queries: int = 0
    reward_min: float | None = None
    reward_max: float | None = None
    reward_cut: float | None = None
    policies: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_run_dirs": self.n_run_dirs,
            "n_loaded": self.n_loaded,
            "n_dropped": dict(self.n_dropped),
            "n_selected": self.n_selected,
            "n_rows": self.n_rows,
            "n_queries": self.n_queries,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "reward_cut": self.reward_cut,
            "policies": dict(self.policies),
        }


# --------------------------------------------------------------------------
# Discovery / loading
# --------------------------------------------------------------------------


def discover_run_dirs(roots: Iterable[Path]) -> list[Path]:
    """Every directory under ``roots`` (or the root itself) that has trajectories/."""
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if (root / "trajectories").is_dir():
            found.append(root)
            continue
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "trajectories").is_dir():
                found.append(child)
    return found


def load_rollout(run_dir: Path) -> list[RolloutRecord]:
    run_dir = Path(run_dir)
    records: list[RolloutRecord] = []
    for tpath in sorted((run_dir / "trajectories").glob("*.json")):
        harness = json.loads(tpath.read_text(encoding="utf-8"))
        qid = str((harness.get("query") or {}).get("id") or tpath.stem)
        dataset = str((harness.get("query") or {}).get("dataset") or "")
        raw_path = run_dir / "gpt_researcher" / f"{qid}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else None
        records.append(RolloutRecord(run_dir, run_dir.name, qid, dataset, harness, raw))
    return records


# --------------------------------------------------------------------------
# Per-round replay
# --------------------------------------------------------------------------


def _tree_depth_for(snap: dict[str, Any], evidence: dict[str, Any]) -> int:
    for iid in snap.get("new_item_ids") or []:
        e = evidence.get(iid)
        if e and e.get("tree_depth") is not None:
            return int(e["tree_depth"])
    return 1


def rows_for_trajectory(
    rec: RolloutRecord,
    *,
    final_round_terminate: bool = True,
    snippet_chars: int | None = None,
    include_meta: bool = True,
) -> list[dict[str, Any]]:
    """One (prompt, completion) row per round of ``rec``."""
    serialize = load_fork_module("serialize")
    features_mod = load_fork_module("features")
    raw = rec.raw or {}
    evidence: dict[str, Any] = raw.get("evidence") or {}
    rounds: list[dict[str, Any]] = raw.get("rounds") or []
    subquestions = list(raw.get("subquestions") or [])
    root_query = str(raw.get("query") or (rec.harness.get("query") or {}).get("text") or "")
    default_names = list(features_mod.FEATURE_NAMES)

    rows: list[dict[str, Any]] = []
    for idx, snap in enumerate(rounds):
        dec = snap.get("decision") or {}
        meta = dec.get("meta") or {}
        kept = list(dec.get("kept_item_ids") or [])
        pruned = list(dec.get("pruned_item_ids") or [])
        pool_ids = list(dict.fromkeys([*kept, *pruned]))
        if not pool_ids:
            continue
        feats: dict[str, list[float]] = meta.get("features") or {}
        if not feats:
            # Legacy / baseline rounds carry no features; nothing to serialize.
            continue
        names = list(meta.get("feature_names") or default_names)
        new_ids = set(snap.get("new_item_ids") or [])
        sources = meta.get("sources") or {}
        # Retained-first, then new, matching OrchestrationInput.pool order.
        ordered = [i for i in pool_ids if i not in new_ids] + [i for i in pool_ids if i in new_ids]
        items = []
        for iid in ordered:
            if iid not in feats:
                continue
            e = evidence.get(iid) or {}
            items.append(
                serialize.StateItem(
                    item_id=iid,
                    features=feats[iid],
                    is_new=iid in new_ids,
                    text=str(e.get("content") or ""),
                    source=str(sources.get(iid) or features_mod.source_domain(str(e.get("source_url") or ""))),
                )
            )
        fstats = meta.get("frontier_stats") or {}
        frontier = [
            serialize.StateFrontier(
                node_id=str(fn.get("node_id")),
                subquery=str(fn.get("subquery") or ""),
                gap=(fstats.get(str(fn.get("node_id"))) or {}).get("gap"),
            )
            for fn in (snap.get("frontier") or [])
        ]
        kwargs: dict[str, Any] = dict(
            root_query=root_query,
            subquestions=subquestions,
            round_id=int(snap.get("round_id") or idx + 1),
            tree_depth=_tree_depth_for(snap, evidence),
            tokens_used=int(meta.get("tokens_used") or 0),
            token_budget=meta.get("token_budget"),
            items=items,
            frontier=frontier,
            feature_names=names,
        )
        if snippet_chars is not None:
            kwargs["snippet_chars"] = snippet_chars
        prompt = serialize.serialize_state(**kwargs)

        is_last = idx == len(rounds) - 1
        if meta.get("policy") == "random" and "terminate_reason" in meta:
            terminate = bool(meta.get("terminate_reason"))
        else:
            terminate = str(dec.get("type") or "") == "terminate"
        if final_round_terminate and is_last:
            terminate = True
        completion = serialize.serialize_action(
            kept_ids=kept,
            pool_ids=[it.item_id for it in items],
            branch_allocation=dict(dec.get("branch_allocation") or {}),
            terminate=terminate,
        )
        row: dict[str, Any] = {
            "query_id": rec.query_id,
            "dataset": rec.dataset,
            "run_id": rec.run_id,
            "round_id": kwargs["round_id"],
            "n_rounds": len(rounds),
            "reward": rec.reward.total if rec.reward else None,
            "prompt": prompt,
            "completion": completion,
            "messages": [
                {"role": "system", "content": serialize.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
        }
        if include_meta:
            row["meta"] = {
                "policy": dec.get("policy"),
                "params": meta.get("params"),
                "n_pool": len(items),
                "n_kept": len(kept),
                "terminate": terminate,
                "reward_breakdown": rec.reward.as_dict() if rec.reward else None,
            }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Selection + build
# --------------------------------------------------------------------------


def select_top(
    records: list[RolloutRecord],
    *,
    top_frac: float = 0.1,
    min_per_query: int = 1,
    max_per_query: int | None = None,
) -> tuple[list[RolloutRecord], float | None]:
    scored = [r for r in records if r.reward is not None]
    if not scored:
        return [], None
    scored.sort(key=lambda r: r.reward.total, reverse=True)  # type: ignore[union-attr]
    n_top = max(1, int(math.ceil(top_frac * len(scored))))
    cut = scored[n_top - 1].reward.total  # type: ignore[union-attr]
    chosen: dict[str, RolloutRecord] = {r.run_id: r for r in scored[:n_top]}

    by_query: dict[str, list[RolloutRecord]] = defaultdict(list)
    for r in scored:
        by_query[r.query_id].append(r)
    for qid, rs in by_query.items():
        have = [r for r in rs if r.run_id in chosen]
        for r in rs:
            if len(have) >= min_per_query:
                break
            if r.run_id not in chosen:
                chosen[r.run_id] = r
                have.append(r)
        if max_per_query is not None:
            keep = sorted(have, key=lambda r: r.reward.total, reverse=True)[:max_per_query]  # type: ignore[union-attr]
            for r in have:
                if r not in keep:
                    chosen.pop(r.run_id, None)
    out = sorted(chosen.values(), key=lambda r: (r.query_id, -r.reward.total))  # type: ignore[union-attr]
    return out, cut


def build_bc_pairs(
    roots: list[Path],
    out_path: Path,
    *,
    reward_cfg: RewardConfig | None = None,
    top_frac: float = 0.1,
    min_per_query: int = 1,
    max_per_query: int | None = None,
    require_quality: bool = True,
    require_policy: str | None = "random",
    final_round_terminate: bool = True,
    snippet_chars: int | None = None,
    goal_file: Path | None = None,
) -> BuildStats:
    reward_cfg = reward_cfg or RewardConfig()
    stats = BuildStats()
    run_dirs = discover_run_dirs(roots)
    stats.n_run_dirs = len(run_dirs)

    records: list[RolloutRecord] = []
    for rd in run_dirs:
        for rec in load_rollout(rd):
            stats.n_loaded += 1
            if rec.harness.get("error") or not rec.harness.get("report"):
                stats.n_dropped["no_report"] += 1
                continue
            if rec.raw is None or not rec.raw.get("rounds"):
                stats.n_dropped["no_fork_trajectory"] += 1
                continue
            if require_policy and rec.policy != require_policy:
                stats.n_dropped[f"policy!={require_policy}"] += 1
                continue
            rec.reward = reward_for_run(
                rd, rec.query_id, cfg=reward_cfg, raw_trajectory=rec.raw,
                harness_trajectory=rec.harness, goal_file=goal_file,
            )
            if require_quality and not rec.reward.has_quality:
                stats.n_dropped["no_quality_score"] += 1
                continue
            stats.policies[str(rec.policy)] += 1
            records.append(rec)

    if records:
        totals = [r.reward.total for r in records]  # type: ignore[union-attr]
        stats.reward_min, stats.reward_max = min(totals), max(totals)

    selected, cut = select_top(records, top_frac=top_frac, min_per_query=min_per_query, max_per_query=max_per_query)
    stats.reward_cut = cut
    stats.n_selected = len(selected)
    stats.n_queries = len({r.query_id for r in selected})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in selected:
            for row in rows_for_trajectory(rec, final_round_terminate=final_round_terminate, snippet_chars=snippet_chars):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
    stats.n_rows = n_rows

    # Sidecars: per-trajectory rewards for every loaded rollout (selected or
    # not) so the selection is auditable, plus build stats.
    rewards_path = out_path.with_name(out_path.stem + "_rewards.jsonl")
    with rewards_path.open("w", encoding="utf-8") as fh:
        chosen_ids = {r.run_id for r in selected}
        for rec in sorted(records, key=lambda r: (r.query_id, r.run_id)):
            fh.write(
                json.dumps(
                    {
                        "run_id": rec.run_id,
                        "query_id": rec.query_id,
                        "selected": rec.run_id in chosen_ids,
                        "params": rec.params,
                        **(rec.reward.as_dict() if rec.reward else {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    out_path.with_name(out_path.stem + "_stats.json").write_text(
        json.dumps({**stats.as_dict(), "reward_config": reward_cfg.__dict__}, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats
