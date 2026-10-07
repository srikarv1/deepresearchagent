"""Score many reports for one DRG query in a single batched Gym judge call.

Each report gets its own id ``<qid>__<tag>`` in one export folder, and the
query's key-point file is linked under every id, so the official quality and
KPR judges score all reports in parallel with the same rubric as Table 3.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from adr.eval.deep_research_gym import _kpr, _quality
from adr.eval.repos import find_deep_research_gym, find_key_points
from adr.runner.config import ROOT


def judge_model() -> str:
    if os.environ.get("REPLAY_JUDGE_MODEL"):
        return os.environ["REPLAY_JUDGE_MODEL"]
    cfg = yaml.safe_load((ROOT / "configs" / "eval" / "deep_research_gym.yaml").read_text()) or {}
    return str(cfg.get("judge_model") or "gpt-5-mini")


def judge_reports(qid: str, query: str, reports: dict[str, str], work_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {tag: {"quality": 0-100, "kpr": support %, "criteria": {...}}}."""
    gym = find_deep_research_gym()
    if not gym.ok:
        raise RuntimeError(gym.reason)
    kp_src = find_key_points(gym.path)
    if kp_src is None:
        raise RuntimeError("DRG key points not found")
    real_kp = kp_src / f"{qid}_aggregated.json"
    if not real_kp.exists():
        raise RuntimeError(f"no key points for query {qid}")

    work_dir = Path(work_dir)
    export = work_dir / "export"
    kp_dir = work_dir / "key_point"
    out = work_dir / "judge"
    for d in (export, kp_dir, out):
        d.mkdir(parents=True, exist_ok=True)

    ids: dict[str, str] = {}
    for tag, text in reports.items():
        if not text.strip():
            continue
        rid = f"{qid}__{tag}"
        ids[rid] = tag
        (export / f"{rid}.q").write_text(query.strip() + "\n", encoding="utf-8")
        (export / f"{rid}.a").write_text(text.strip() + "\n", encoding="utf-8")
        link = kp_dir / f"{rid}_aggregated.json"
        if not link.exists():
            os.symlink(real_kp, link)

    model = judge_model()
    q = _quality(gym.path, export, out, model, None)
    k = _kpr(gym.path, export, out, model, kp_dir, None)
    qual = (q or {}).get("per_query_normalized") or {}
    kpr = (k or {}).get("per_query") or {}
    crit = _per_query_criteria(out / f"quality_{model}.json")

    scores: dict[str, dict[str, Any]] = {}
    for rid, tag in ids.items():
        scores[tag] = {
            "quality": qual.get(rid),
            "kpr": (kpr.get(rid) or {}).get("support_rate"),
            "criteria": crit.get(rid, {}),
        }
    return scores


def _per_query_criteria(path: Path) -> dict[str, dict[str, int]]:
    import json

    from adr.eval.scoring import _rating_of

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for rid, row in data.items():
        out[rid] = {name: _rating_of(v) for name, v in ((row or {}).get("scores") or {}).items()}
    return out
