"""Reward table for ReplayGym: Gym quality and key-point recall of the report the
agent's own writer produces from each reachable plan, with replicates.

Research is replayed exactly from the cached tree, so the only noise left in a
plan's score is the writer's and the judge's. A replicate is a fresh report plus
a fresh judgment of it. Keeping every replicate lets training average them and
lets ``gate_stats`` measure that noise before any policy is trained.

Layout under ``root``:
    <qid>.json               header + per-plan replicates (atomic writes)
    reports/<qid>/<tag>.md   every report that was judged (reused after a restart)
    judge/<qid>/<attempt>/   Gym judge work dirs
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from adr.replay.plans import FIXED_RULES, PlanSpace
from adr.replay.tree import MAX_CONTEXT_WORDS, WRITER_CONTEXT_RULE, ResearchTree

TABLE_VERSION = 1
SCORE_KEYS = ("quality", "kpr")                     # Gym scale, 0-100
METRICS = {"kpr": ("kpr",), "quality": ("quality",), "mean": ("quality", "kpr")}
MIN_REPORT_CHARS = 2000                               # same floor as adr.replay.synth
# Header fields that define what a score means; a table never mixes them.
IDENTITY = ("version", "context_rule", "pack", "max_words", "writer", "judge_model")

Synthesize = Callable[[str, dict[str, str], int], dict[str, str]]
Judge = Callable[[str, str, dict[str, str], Path], dict[str, dict[str, Any]]]


class TableMismatch(RuntimeError):
    """The table on disk was built under another writer, judge or context rule."""


class IncompleteTable(RuntimeError):
    """Some plan has fewer replicates than required."""


def table_header(qid: str, tree: ResearchTree, *, pack: str, max_words: int, judge_model: str) -> dict[str, Any]:
    """Call after ``synth.apply_agent_env()`` so the writer models are the live ones."""
    return {
        "version": TABLE_VERSION,
        "qid": qid,
        "query": tree.query,
        "context_rule": WRITER_CONTEXT_RULE,
        "pack": pack,
        "max_words": int(max_words),
        "writer": {k: os.environ.get(k, "") for k in ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM")},
        "judge_model": judge_model,
        "full_tokens": int(tree.tokens(tree.all_nodes())),
        "n_nodes": len(tree.nodes),
    }


@dataclass
class RewardTable:
    path: Path
    header: dict[str, Any]
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def file(root: Path, qid: str) -> Path:
        return Path(root) / f"{qid}.json"

    @classmethod
    def load(cls, root: Path, qid: str) -> RewardTable | None:
        path = cls.file(root, qid)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, dict(data["header"]), dict(data.get("plans") or {}))

    @classmethod
    def open(cls, root: Path, header: dict[str, Any]) -> RewardTable:
        tab = cls.load(root, header["qid"])
        if tab is None:
            return cls(cls.file(root, header["qid"]), dict(header))
        tab.check(header, IDENTITY)
        return tab

    def check(self, want: dict[str, Any], fields: tuple[str, ...] = IDENTITY) -> None:
        diff = {k: (self.header.get(k), want.get(k)) for k in fields if k in want and self.header.get(k) != want.get(k)}
        if diff:
            raise TableMismatch(f"{self.path} was built under other settings (table, requested): {diff}")

    def n_reps(self, key: str) -> int:
        return len((self.plans.get(key) or {}).get("reps") or [])

    def add_rep(self, key: str, nodes: frozenset[str], tokens: int, scores: dict[str, Any]) -> None:
        entry = self.plans.setdefault(key, {"nodes": sorted(nodes), "tokens": int(tokens), "reps": []})
        if entry["nodes"] != sorted(nodes):
            raise RuntimeError(f"{self.path}: plan key {key} maps to two different sub-trees")
        entry["reps"].append(scores)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"header": self.header, "plans": self.plans}, indent=0), encoding="utf-8")
        os.replace(tmp, self.path)

    def scores(self, space: PlanSpace, min_reps: int = 1) -> dict[str, np.ndarray]:
        """``SCORE_KEYS`` -> (n_plans, R) in [0, 1], R = the smallest replicate count."""
        keys = space.keys
        counts = [self.n_reps(k) for k in keys]
        r = min(counts) if counts else 0
        if r < min_reps:
            short = sum(c < min_reps for c in counts)
            raise IncompleteTable(f"{space.qid}: {short}/{len(keys)} plans have fewer than {min_reps} replicates")
        for k, plan in zip(keys, space.plans):
            if self.plans[k]["nodes"] != sorted(plan):
                raise RuntimeError(f"{self.path}: plan key {k} maps to two different sub-trees")
        return {
            m: np.asarray([[float(rep[m]) / 100.0 for rep in self.plans[k]["reps"][:r]] for k in keys], np.float64)
            for m in SCORE_KEYS
        }


def _valid(sc: dict[str, Any] | None) -> bool:
    if not sc:
        return False
    for m in SCORE_KEYS:
        v = sc.get(m)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0.0 <= v <= 100.0:
            return False
    return True


def _synthesize(query: str, contexts: dict[str, str], concurrency: int) -> dict[str, str]:
    from adr.replay.synth import synthesize_many

    return asyncio.run(synthesize_many(query, contexts, concurrency))


def _judge(qid: str, query: str, reports: dict[str, str], work_dir: Path) -> dict[str, dict[str, Any]]:
    from adr.replay.judge import judge_reports

    return judge_reports(qid, query, reports, work_dir)


def fill_table(
    tree: ResearchTree,
    space: PlanSpace,
    *,
    root: Path,
    reps: int,
    judge_model: str,
    pack: str = "tail",
    max_words: int = MAX_CONTEXT_WORDS,
    concurrency: int = 2,
    judge_attempts: int = 3,
    synthesize: Synthesize | None = None,
    judge: Judge | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Give every plan of ``space`` ``reps`` scored replicates.

    One pass adds at most one replicate per plan, so an interrupted build still
    leaves a usable single-replicate table. Resumable: complete plans are
    skipped and reports already on disk are judged instead of rewritten.
    Configuration errors from the judge (no Gym checkout, no key points) raise.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    synthesize = synthesize or _synthesize
    judge = judge or _judge
    qid, keys = space.qid, space.keys
    tab = RewardTable.open(root, table_header(qid, tree, pack=pack, max_words=max_words, judge_model=judge_model))
    rep_dir = Path(root) / "reports" / qid
    rep_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()

    for _ in range(reps):
        todo = [i for i, k in enumerate(keys) if tab.n_reps(k) < reps]
        if not todo:
            break
        before = sum(tab.n_reps(k) for k in keys)
        tags = {f"{keys[i]}_r{tab.n_reps(keys[i])}": i for i in todo}
        reports: dict[str, str] = {}
        contexts: dict[str, str] = {}
        for tag, i in tags.items():
            f = rep_dir / f"{tag}.md"
            text = f.read_text(encoding="utf-8") if f.exists() else ""
            if len(text.strip()) >= MIN_REPORT_CHARS:
                reports[tag] = text
            else:
                contexts[tag] = tree.writer_context(space.plans[i], pack=pack, max_words=max_words)
        if contexts:
            got = synthesize(tree.query, contexts, concurrency) or {}
            for tag in contexts:
                text = got.get(tag) or ""
                if len(text.strip()) >= MIN_REPORT_CHARS:
                    (rep_dir / f"{tag}.md").write_text(text, encoding="utf-8")
                    reports[tag] = text
                    stats["synthesized"] += 1
                else:
                    stats["writer_failed"] += 1

        pending = dict(reports)
        for attempt in range(judge_attempts):
            if not pending:
                break
            work = Path(root) / "judge" / qid / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-a{attempt}"
            got = judge(qid, tree.query, pending, work) or {}
            for tag, sc in got.items():
                if tag not in pending or not _valid(sc):
                    continue
                i = tags[tag]
                tab.add_rep(
                    keys[i],
                    space.plans[i],
                    int(space.tokens[i]),
                    {"quality": float(sc["quality"]), "kpr": float(sc["kpr"]), "report": f"{tag}.md"},
                )
                pending.pop(tag)
                stats["judged"] += 1
            tab.save()
        stats["judge_failed"] += len(pending)
        if sum(tab.n_reps(k) for k in keys) == before:
            log(f"{qid}: no plan gained a replicate in this pass; stopping (writer or judge is failing)")
            break

    tab.save()
    stats["missing"] = sum(max(0, reps - tab.n_reps(k)) for k in keys)
    return dict(stats)


def missing_work(
    tree: ResearchTree, space: PlanSpace, *, root: Path, reps: int, pack: str, max_words: int
) -> dict[str, int]:
    """Reports still to write for ``reps`` replicates, and their total context
    words (for a cost estimate before spending anything)."""
    tab = RewardTable.load(root, space.qid)
    have = [tab.n_reps(k) if tab else 0 for k in space.keys]
    need = [(i, reps - h) for i, h in enumerate(have) if h < reps]
    words = sum(n * len(tree.writer_context(space.plans[i], pack=pack, max_words=max_words).split()) for i, n in need)
    return {"plans": len(space.plans), "reports": sum(n for _, n in need), "context_words": words}


# ----------------------------------------------------------------------------- gate
@dataclass
class ScoredQuestion:
    qid: str
    space: PlanSpace
    scores: dict[str, np.ndarray]   # SCORE_KEYS -> (n_plans, n_reps) in [0, 1]

    def metric_reps(self, metric: str) -> np.ndarray:
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}, got {metric!r}")
        return np.mean([self.scores[m] for m in METRICS[metric]], axis=0)

    def metric(self, metric: str) -> np.ndarray:
        return self.metric_reps(metric).mean(axis=1)

    def reward(self, metric: str, lam: float) -> np.ndarray:
        return self.metric(metric) - lam * self.space.cost


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, np.float64)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def gate_stats(
    questions: list[ScoredQuestion], *, metric: str, lam: float, n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Is there anything for a policy to learn?

    * noise: writer + judge sd of one replicate, pooled over plans;
    * signal: sd of the true plan means within a question (noise removed);
    * headroom: per-question best plan minus the best fixed-depth rule at
      ``lam``, cross-fitted (pick on half the replicates, score on the other
      half), so judge luck does not count as headroom. The in-sample oracle is
      reported only to show how large that luck is.

    Points are metric points on the 0-100 scale.
    """
    if not questions:
        raise ValueError("no scored questions")
    n_reps = min(q.scores[SCORE_KEYS[0]].shape[1] for q in questions)
    if n_reps < 2:
        raise IncompleteTable("the gate needs >= 2 replicates per plan to separate noise from signal")
    within, between, oracle_cf, oracle_in = [], [], [], []
    fixed: dict[str, list[float]] = {rule: [] for rule in FIXED_RULES}
    for q in questions:
        m = q.metric_reps(metric)[:, :n_reps]
        cost = q.space.cost
        means = m.mean(axis=1)
        within.append(float(m.var(axis=1, ddof=1).mean()))
        if len(means) > 1:
            between.append(float(means.var(ddof=1)))
        a = m[:, 0::2].mean(axis=1) - lam * cost
        b = m[:, 1::2].mean(axis=1) - lam * cost
        oracle_cf.append(0.5 * (b[int(np.argmax(a))] + a[int(np.argmax(b))]))
        r_all = means - lam * cost
        oracle_in.append(float(r_all.max()))
        for rule in FIXED_RULES:
            fixed[rule].append(float(r_all[q.space.fixed_plan(rule)]))
    fixed_mean = {rule: float(np.mean(v)) for rule, v in fixed.items()}
    best = max(fixed_mean, key=fixed_mean.get)
    headroom = np.asarray(oracle_cf) - np.asarray(fixed[best])
    noise_var = float(np.mean(within))
    noise_sd = math.sqrt(noise_var)
    # Plan means carry noise_var / n_reps of replicate noise; remove it after
    # pooling (clipping per question would bias the signal upwards).
    signal_sd = math.sqrt(max(0.0, float(np.mean(between)) - noise_var / n_reps)) if between else 0.0
    lo, hi = bootstrap_ci(headroom, n_boot, seed)
    return {
        "n_questions": len(questions),
        "n_reps": n_reps,
        "metric": metric,
        "lam": lam,
        "noise_sd_points": 100 * noise_sd,
        "signal_sd_points": 100 * signal_sd,
        "snr": signal_sd / noise_sd if noise_sd > 0 else float("inf"),
        "fixed_reward": fixed_mean,
        "best_fixed": best,
        "oracle_crossfit_reward": float(np.mean(oracle_cf)),
        "oracle_insample_reward": float(np.mean(oracle_in)),
        "headroom_points": 100 * float(headroom.mean()),
        "headroom_ci_points": (100 * lo, 100 * hi),
    }
