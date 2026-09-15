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


def _as_int_or_str(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def export_browsecomp_plus(
    trajectories: list[Trajectory], dest_dir: Path, model_name: str = "agent"
) -> Path:
    """Official BrowseComp-Plus run layout: one <query_id>.json per query.

    Matches what texttron/BrowseComp-Plus scripts_evaluation/evaluate_run.py and
    evaluate_with_openai.py read (extra fields are ignored upstream):
    query_id, tool_call_counts, status, retrieved_docids, result[-1].output.

    retrieved_docids must be corpus docids from Tevatron/browsecomp-plus-corpus.
    A search backend that retrieves from that corpus records them in
    final_stats["retrieved_docids"]; live-web runs have none, so upstream's
    Recall (%) is 0 for them by construction and only Accuracy is meaningful.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for traj in trajectories:
        stats = traj.final_stats or {}
        usage = stats.get("usage") or {}
        article = traj.report.article if traj.report else ""
        docids = sorted({str(d) for d in (stats.get("retrieved_docids") or [])})
        row = {
            "query_id": traj.query.id,
            "tool_call_counts": {
                "search": int(usage.get("n_search_calls") or 0),
                "fetch": int(usage.get("n_fetch_calls") or 0),
            },
            "status": "completed" if article and not traj.error else "failed",
            "retrieved_docids": docids,
            "result": [{"type": "output_text", "output": article}],
            "metadata": {"model": model_name, "error": traj.error},
        }
        (dest_dir / f"{traj.query.id}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return dest_dir


def export_browsecomp_plus_ground_truth(trajectories: list[Trajectory], dest: Path) -> Path:
    """Decrypted {query_id, query, answer} rows for --ground_truth upstream.

    Written next to the run so the judge sees plaintext without it ever being
    committed; runs/ is gitignored.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "query_id": traj.query.id,
                "query": traj.query.text,
                "answer": traj.query.metadata.get("answer", ""),
            },
            ensure_ascii=False,
        )
        for traj in trajectories
    ]
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest
