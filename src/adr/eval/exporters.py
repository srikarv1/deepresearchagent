from __future__ import annotations

import json
from pathlib import Path

from adr.core.types import Trajectory


def export_deep_research_bench(trajectories: list[Trajectory], dest: Path) -> Path:
    """Official DRB raw format: one JSON object per line with id, prompt, article."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for traj in trajectories:
        article = traj.report.article if traj.report else ""
        row = {
            "id": _as_int_or_str(traj.query.id),
            "prompt": traj.query.text,
            "article": article,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest


def export_deep_research_gym(trajectories: list[Trajectory], dest_dir: Path) -> Path:
    """Official Gym report layout: <id>.q and <id>.a in one folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for traj in trajectories:
        (dest_dir / f"{traj.query.id}.q").write_text(traj.query.text.strip() + "\n", encoding="utf-8")
        article = traj.report.article if traj.report else ""
        (dest_dir / f"{traj.query.id}.a").write_text(article.strip() + "\n", encoding="utf-8")
    return dest_dir


def export_browsecomp(trajectories: list[Trajectory], dest: Path) -> Path:
    """One JSON object per line: id, question (decrypted), response (the report).

    The correct answer is deliberately absent; the judge decrypts it from the
    CSV by row index.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for traj in trajectories:
        article = traj.report.article if traj.report else ""
        row = {
            "id": traj.query.id,
            "row": traj.query.metadata.get("row"),
            "topic": traj.query.metadata.get("topic") or traj.query.topic,
            "question": traj.query.metadata.get("question") or traj.query.text,
            "response": article,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest


def _as_int_or_str(value: str) -> int | str:
    return int(value) if value.isdigit() else value
