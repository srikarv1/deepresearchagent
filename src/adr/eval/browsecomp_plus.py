"""BrowseComp-Plus judging: accuracy via the official grader, recall from labelled docs.

Accuracy follows ``texttron/BrowseComp-Plus`` ``evaluate_run.py`` /
``evaluate_with_openai.py``: the simple-evals grader prompt, ``correct: yes|no``
parsing, mean accuracy. Recall is
``|retrieved_docids ∩ evidence_docs| / |evidence_docs|``, matching
``topics-qrels/qrel_evidence.txt`` (those docids are already on
``Query.metadata['evidence_docs']``, so the upstream repo is not required).

The judge LLM is built from ``configs/eval/browsecomp_plus.yaml`` through
``adr.llm.factory.build_llm``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.eval.browsecomp_plus_prompts import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE_NO_GET_DOCUMENT,
    format_query,
)
from adr.eval.exporters import export_browsecomp_plus, export_browsecomp_plus_ground_truth
from adr.llm.factory import build_llm

__all__ = [
    "GRADER_TEMPLATE",
    "QUERY_TEMPLATE_NO_GET_DOCUMENT",
    "aggregate",
    "evidence_docids",
    "format_query",
    "parse_grade",
    "recall_for",
    "run_browsecomp_plus",
]


def parse_grade(text: str) -> dict[str, Any]:
    text = text or ""
    m = (
        re.search(r"\*\*correct:\*\*\s*(yes|no)", text, re.IGNORECASE)
        or re.search(r"\*\*correct\*\*:\s*(yes|no)", text, re.IGNORECASE)
        or re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    )
    correct = bool(m and m.group(1).lower() == "yes")
    c = (
        re.search(r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        or re.search(r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        or re.search(r"confidence:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    )
    confidence = float(c.group(1)) if c else 100.0
    e = (
        re.search(r"\*\*extracted_final_answer:\*\*\s*(.*)", text, re.IGNORECASE)
        or re.search(r"\*\*extracted_final_answer\*\*:\s*(.*)", text, re.IGNORECASE)
        or re.search(r"extracted_final_answer:\s*(.*)", text, re.IGNORECASE)
    )
    extracted = e.group(1).strip() if e else None
    return {
        "correct": correct,
        "confidence": max(0.0, min(100.0, confidence)),
        "extracted_final_answer": extracted,
        "parse_ok": m is not None,
    }


def evidence_docids(traj: Trajectory) -> list[str]:
    docs = (traj.query.metadata or {}).get("evidence_docs") or []
    out: list[str] = []
    for doc in docs:
        if isinstance(doc, dict) and doc.get("docid"):
            out.append(str(doc["docid"]))
        elif isinstance(doc, str):
            out.append(doc)
    return sorted(set(out))


def recall_for(retrieved: list[str] | None, positives: list[str]) -> float | None:
    if not positives:
        return None
    hit = set(retrieved or []) & set(positives)
    return len(hit) / len(positives)


def aggregate(grades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(grades)
    if not n:
        return {
            "n": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "recall": 0.0,
            "n_recall": 0,
            "parse_failures": 0,
        }
    correct = [1.0 if g["correct"] else 0.0 for g in grades]
    recalls = [g["recall"] for g in grades if isinstance(g.get("recall"), (int, float))]
    return {
        "n": n,
        "n_correct": int(sum(correct)),
        "accuracy": round(sum(correct) / n * 100, 4),
        "n_recall": len(recalls),
        "recall": round(sum(recalls) / len(recalls) * 100, 4) if recalls else 0.0,
        "parse_failures": sum(1 for g in grades if not g["parse_ok"]),
    }


def run_browsecomp_plus(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    judge_cfg: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Export the official layout, grade every report, return Accuracy + Recall."""
    judge_cfg = dict(judge_cfg or {})
    bcp_dir = run_dir / "exports" / "browsecomp_plus"
    export_dir = export_browsecomp_plus(trajectories, bcp_dir / model_name, model_name=model_name)
    gt_path = export_browsecomp_plus_ground_truth(trajectories, bcp_dir / "ground_truth.jsonl")

    result: dict[str, Any] = {
        "bench": "browsecomp_plus",
        "official": False,
        "reason": None,
        "export": str(export_dir),
        "ground_truth": str(gt_path),
        "judge_model": judge_cfg.get("model"),
        "n": 0,
    }

    missing_answers = [
        t.query.id for t in trajectories if not (t.query.metadata or {}).get("answer")
    ]
    if missing_answers:
        result["reason"] = (
            f"{len(missing_answers)} queries have no gold answer in metadata "
            f"(e.g. {missing_answers[:3]}); cannot grade accuracy"
        )
        return result

    try:
        judge = llm or build_llm(judge_cfg)
    except Exception as exc:
        result["reason"] = f"could not build judge LLM: {exc}"
        return result

    async def _grade_all() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(int(judge_cfg.get("concurrency", 8)))

        async def one(traj: Trajectory) -> dict[str, Any]:
            question = traj.query.text
            response = traj.report.article if traj.report else ""
            gold = str((traj.query.metadata or {}).get("answer") or "")
            positives = evidence_docids(traj)
            retrieved = list((traj.final_stats or {}).get("retrieved_docids") or [])
            prompt = GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=gold
            )
            t0 = time.perf_counter()
            async with sem:
                try:
                    out = await judge.complete(
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=int(judge_cfg.get("max_output_tokens", 800)),
                    )
                    text, err = out.text, None
                except Exception as exc:
                    text, err = "", f"{type(exc).__name__}: {exc}"
            grade = parse_grade(text)
            rec = recall_for(retrieved, positives)
            grade.update(
                {
                    "id": traj.query.id,
                    "correct_answer": gold,
                    "retrieved_docids": retrieved,
                    "evidence_docids": positives,
                    "recall": rec,
                    "has_report": bool(response.strip()),
                    "judge_raw": text,
                    "judge_error": err,
                    "judge_latency_s": round(time.perf_counter() - t0, 3),
                }
            )
            return grade

        coro = asyncio.gather(*[one(t) for t in trajectories])
        if timeout_s:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return await coro

    grades = _run_sync(_grade_all())

    out_dir = run_dir / "metrics" / "browsecomp_plus"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = out_dir / f"{model_name}.jsonl"
    per_query.write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in grades) + "\n",
        encoding="utf-8",
    )

    agg = aggregate(grades)
    result.update(agg)
    result["per_query_path"] = str(per_query)
    result["official"] = agg["n"] > 0 and all(g.get("judge_error") is None for g in grades)
    if not result["official"]:
        errs = [g["judge_error"] for g in grades if g.get("judge_error")]
        result["reason"] = (
            f"{len(errs)} judge calls failed (e.g. {errs[:1]})" if errs else "no grades"
        )
    return result


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
