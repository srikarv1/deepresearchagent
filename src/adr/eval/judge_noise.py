"""How much does a RACE score move when nothing about the research changes?

Two sources of noise sit between a fixed evidence set and its score:

* writer noise: the backbone writes a different report from the same
  ``final_context`` each time;
* judge noise: RACE (cleaning + judge model) scores the same report
  differently each time.

The oracle-depth labels compare subtrees that each get one rewrite and one RACE
call, so their noise is ``sigma_total`` (writer and judge together). This module
estimates it from finished DRB runs: every query's ``final_context`` is
rewritten ``K`` times and each report is scored once (``sigma_total``), and the
original report is scored ``J`` times (``sigma_judge``). Both are pooled
within-query standard deviations, so differences between queries do not count
as noise.

``scripts/measure_judge_noise.py`` is the command line entry point.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# One branch's depth effect is a contrast between the subtrees where that
# branch is shallow and those where it is deep, with the other branches
# balanced. For b roots and L depth levels per root there are L**(b-1)
# subtrees on each side of a two-level contrast.
_TREE_SHAPES = {"d2b4": (2, 4), "d3b4": (3, 4)}


@dataclass(frozen=True)
class NoiseQuery:
    """One finished DRB query: the benchmark prompt, its report and the
    evidence string the writer saw."""

    qid: str
    prompt: str
    report: str
    final_context: str


def pooled_within_sd(groups: Iterable[Sequence[float]]) -> tuple[float | None, int]:
    """Pooled within-group standard deviation and its degrees of freedom.

    Groups with fewer than two values carry no within-group information and are
    skipped. Returns ``(None, 0)`` when nothing is left.
    """
    ss = 0.0
    dof = 0
    for values in groups:
        xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        if len(xs) < 2:
            continue
        mean = sum(xs) / len(xs)
        ss += sum((x - mean) ** 2 for x in xs)
        dof += len(xs) - 1
    if dof == 0:
        return None, 0
    return math.sqrt(ss / dof), dof


def branch_effect_se(sigma: float, shape: str) -> float:
    """Standard error of one branch's deep-minus-shallow effect on one query
    when every subtree of ``shape`` is replayed once with per-report noise
    ``sigma``."""
    levels, roots = _TREE_SHAPES[shape]
    per_side = levels ** (roots - 1)
    return sigma * math.sqrt(2.0 / per_side)


def summarize(
    writer_scores: Mapping[str, Sequence[float]],
    judge_scores: Mapping[str, Sequence[float]],
    *,
    n_train_queries: int = 30,
) -> dict[str, Any]:
    """Noise estimates from per-query score lists.

    ``writer_scores[qid]`` holds one score per independent report written from
    the same context; ``judge_scores[qid]`` holds repeated scores of one report.
    """
    sigma_total, dof_total = pooled_within_sd(writer_scores.values())
    sigma_judge, dof_judge = pooled_within_sd(judge_scores.values())
    sigma_writer = None
    if sigma_total is not None and sigma_judge is not None:
        sigma_writer = math.sqrt(max(sigma_total**2 - sigma_judge**2, 0.0))

    out: dict[str, Any] = {
        "sigma_total": sigma_total,
        "sigma_total_dof": dof_total,
        "sigma_judge": sigma_judge,
        "sigma_judge_dof": dof_judge,
        "sigma_writer": sigma_writer,
        "n_train_queries": n_train_queries,
        "branch_effect_se": {},
        "per_query": {},
    }
    if sigma_total is not None:
        for shape in _TREE_SHAPES:
            per_query = branch_effect_se(sigma_total, shape)
            out["branch_effect_se"][shape] = {
                "per_query": per_query,
                "pooled": per_query / math.sqrt(max(n_train_queries, 1)),
            }
    for qid in sorted(set(writer_scores) | set(judge_scores), key=_qid_key):
        w = [float(x) for x in writer_scores.get(qid, [])]
        j = [float(x) for x in judge_scores.get(qid, [])]
        out["per_query"][qid] = {
            "writer_scores": w,
            "judge_scores": j,
            "writer_range": (max(w) - min(w)) if len(w) > 1 else None,
            "judge_range": (max(j) - min(j)) if len(j) > 1 else None,
        }
    return out


def _qid_key(qid: str) -> tuple[int, Any]:
    return (0, int(qid)) if str(qid).isdigit() else (1, str(qid))


# ── run directory ────────────────────────────────────────────────────────────


def load_noise_queries(run_dir: Path, query_ids: Sequence[str] | None = None) -> tuple[list[NoiseQuery], list[str]]:
    """Read each query's prompt and report from ``trajectories/<qid>.json`` and
    the writer's evidence from the fork trajectory ``gpt_researcher/<qid>.json``
    (``keep_trajectory_files: true``). Returns the loaded queries and one
    message per skipped query."""
    run_dir = Path(run_dir)
    traj_dir = run_dir / "trajectories"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"{traj_dir} does not exist; is {run_dir} an adr run directory?")
    wanted = [str(q) for q in query_ids] if query_ids else None
    paths = (
        [traj_dir / f"{q}.json" for q in wanted]
        if wanted
        else sorted(traj_dir.glob("*.json"), key=lambda p: _qid_key(p.stem))
    )

    loaded: list[NoiseQuery] = []
    skipped: list[str] = []
    for path in paths:
        qid = path.stem
        if not path.exists():
            skipped.append(f"{qid}: no {path}")
            continue
        harness = json.loads(path.read_text(encoding="utf-8"))
        if (harness.get("query") or {}).get("dataset") not in (None, "deep_research_bench"):
            skipped.append(f"{qid}: dataset is {harness['query']['dataset']}, not deep_research_bench")
            continue
        prompt = (harness.get("query") or {}).get("text") or ""
        report = (harness.get("report") or {}).get("article") or ""
        raw_path = run_dir / "gpt_researcher" / f"{qid}.json"
        if not raw_path.exists():
            skipped.append(f"{qid}: no {raw_path} (run with keep_trajectory_files: true)")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        context = raw.get("final_context") or ""
        if not prompt.strip() or not report.strip() or not context.strip():
            skipped.append(f"{qid}: empty prompt, report or final_context")
            continue
        loaded.append(NoiseQuery(qid=qid, prompt=prompt, report=report, final_context=context))
    return loaded, skipped


# ── writer ───────────────────────────────────────────────────────────────────


async def rewrite_report(researcher_cls: Any, query: NoiseQuery) -> str:
    """Write a fresh report from ``query.final_context`` the way the live run's
    final ``write_report()`` does: a top-level ``report_type="deep"``
    researcher whose context is the final evidence string.

    The deep skill is detached first, so the fork does not overwrite the
    trajectory files of the live run with this rewrite.
    """
    os.environ.pop("GR_ANSWER_FORMAT", None)
    researcher = researcher_cls(query=query.prompt, report_type="deep")
    researcher.deep_researcher = None
    researcher.context = query.final_context
    return await researcher.write_report()


# ── RACE ─────────────────────────────────────────────────────────────────────


def race_slot(
    slot: str,
    rows: Sequence[tuple[NoiseQuery, str]],
    *,
    bench_root: Path,
    env: dict[str, str],
    language: str = "en",
    workers: int = 4,
    skip_cleaning: bool = False,
    timeout_s: float | None = None,
) -> dict[str, float]:
    """Score one report per query under the RACE model name ``slot``.

    RACE keys its inputs and outputs by model name and skips ids it already
    scored, so every replicate needs its own name. Returns
    ``{qid: overall_score}`` for the rows RACE scored.
    """
    from adr.eval.deep_research_bench import _race

    raw_dir = bench_root / "data" / "test_data" / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"id": int(q.qid) if q.qid.isdigit() else q.qid, "prompt": q.prompt, "article": article},
            ensure_ascii=False,
        )
        for q, article in rows
    ]
    (raw_dir / f"{slot}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = _race(
        bench_root,
        slot,
        bench_root / "data" / "prompt_data" / "query.jsonl",
        language,
        workers,
        skip_cleaning,
        env,
        timeout_s,
        run_dir=None,
    )
    per_query = result.get("per_query") or {}
    return {
        str(qid): float(scores["overall_score"])
        for qid, scores in per_query.items()
        if "overall_score" in scores
    }
