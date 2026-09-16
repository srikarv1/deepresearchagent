"""Building Trajectory objects from reports the harness did not produce.

Useful for scoring a hand-written report, a competitor's output, or anything
else you want to push through the official judges without running an agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from adr.core.types import Query, Report, Trajectory
from adr.datasets.loader import load_queries

_URL = re.compile(r"https?://[^\s\]\)>]+")


def _citations(article: str) -> list[str]:
    return list(dict.fromkeys(_URL.findall(article)))


def trajectory_from_pair(
    *,
    question: str,
    report: str,
    query_id: str,
    dataset: str = "deep_research_bench",
    language: str = "en",
) -> Trajectory:
    """Wrap one (question, report) pair so the eval stack can score it.

    Cost metrics will be zero because no agent ran; the judge metrics are the
    meaningful output here.
    """
    return Trajectory(
        query=Query(id=str(query_id), text=question, dataset=dataset, language=language),
        report=Report(article=report, citations=_citations(report)),
        final_stats={"imported": True},
    )


def resolve_question(
    *,
    dataset: str,
    query_id: str | None = None,
    question: str | None = None,
) -> tuple[str, str]:
    """Return ``(query_id, question)``, filling in whichever side is missing.

    Matching a benchmark query matters: DeepResearch Bench pairs a report to its
    reference article by exact prompt string. Supplying a bare question is
    allowed but then only id-independent metrics will work.
    """
    if query_id is None and question is None:
        raise ValueError("Provide at least one of query_id or question")

    if query_id is not None:
        rows = load_queries(dataset, query_ids=[str(query_id)])
        if rows:
            return rows[0].id, (question or rows[0].text)
        if question is None:
            raise ValueError(
                f"Query id {query_id!r} is not in {dataset}. "
                "Pass --question too if you are scoring an off-benchmark query."
            )
        return str(query_id), question

    for row in load_queries(dataset):
        if row.text.strip() == (question or "").strip():
            return row.id, row.text
    slug = re.sub(r"[^a-z0-9]+", "-", (question or "").lower()).strip("-")[:48]
    return slug or "custom", question or ""


def trajectories_from_drb_jsonl(path: str | Path) -> list[Trajectory]:
    """Read an official DRB raw file of ``{id, prompt, article}`` rows."""
    path = Path(path)
    out: list[Trajectory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            trajectory_from_pair(
                question=row["prompt"],
                report=row.get("article") or "",
                query_id=str(row["id"]),
                dataset="deep_research_bench",
                language=row.get("language", "en"),
            )
        )
    return out


def write_trajectories(run_dir: str | Path, trajectories: list[Trajectory]) -> Path:
    """Materialize a run directory so `adr evaluate` can pick it up."""
    run_dir = Path(run_dir)
    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for traj in trajectories:
        (traj_dir / f"{traj.query.id}.json").write_text(
            traj.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if traj.report:
            reports = run_dir / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            (reports / f"{traj.query.id}.md").write_text(traj.report.article, encoding="utf-8")
    return run_dir
