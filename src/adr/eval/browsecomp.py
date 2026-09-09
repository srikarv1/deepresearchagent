"""BrowseComp judging: exact-match accuracy via the simple-evals grader prompt.

For each trajectory the report is the ``[response]``; the ``[correct_answer]``
is decrypted from the CSV by row index at judge time (it is never stored on
the Query or in the run's exports). The grader LLM is built from
``configs/eval/browsecomp.yaml`` -> ``judge`` through the harness's own LLM
factory, so any OpenAI-compatible model works and its usage is recorded.

Scoring follows ``simple-evals/browsecomp_eval.py``:
  correct   = re.search(r"correct: (yes|no)") == "yes"; anything else is "no"
  accuracy  = mean(correct)
  calibration_error = mean |confidence/100 - correct| (reported for reference)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.datasets.browsecomp import answers_by_id
from adr.llm.factory import build_llm

# Verbatim from simple-evals/browsecomp_eval.py.
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

_CORRECT_RE = re.compile(r"correct:\s*(yes|no)", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence:\s*(\d+)", re.IGNORECASE)
_EXTRACTED_RE = re.compile(r"extracted_final_answer:\s*(.+)", re.IGNORECASE)


def parse_grade(text: str) -> dict[str, Any]:
    m = _CORRECT_RE.search(text or "")
    correct = bool(m and m.group(1).lower() == "yes")
    c = _CONF_RE.search(text or "")
    confidence = int(c.group(1)) if c else 100
    e = _EXTRACTED_RE.search(text or "")
    extracted = e.group(1).strip() if e else None
    return {
        "correct": correct,
        "confidence": max(0, min(100, confidence)),
        "extracted_final_answer": extracted,
        "parse_ok": m is not None,
    }


def aggregate(grades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(grades)
    if not n:
        return {"n": 0, "accuracy": 0.0, "n_correct": 0, "calibration_error": 0.0, "parse_failures": 0}
    correct = [1.0 if g["correct"] else 0.0 for g in grades]
    conf = [g["confidence"] / 100.0 for g in grades]
    return {
        "n": n,
        "n_correct": int(sum(correct)),
        "accuracy": round(sum(correct) / n * 100, 4),
        "calibration_error": round(sum(abs(c - k) for c, k in zip(conf, correct)) / n * 100, 4),
        "parse_failures": sum(1 for g in grades if not g["parse_ok"]),
    }


def run_browsecomp(
    trajectories: list[Trajectory],
    *,
    run_dir: Path,
    model_name: str,
    judge_cfg: dict[str, Any] | None = None,
    csv_path: str | Path | None = None,
    timeout_s: float | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Grade every report; write per-query rows and return the aggregate block."""
    judge_cfg = dict(judge_cfg or {})
    result: dict[str, Any] = {
        "bench": "browsecomp",
        "official": False,
        "reason": None,
        "judge_model": judge_cfg.get("model"),
        "n": 0,
    }
    ids = [t.query.id for t in trajectories]
    try:
        answers = answers_by_id(csv_path, ids)
    except FileNotFoundError as exc:
        result["reason"] = str(exc)
        return result
    missing = [i for i in ids if i not in answers]
    if missing:
        result["reason"] = f"{len(missing)} query ids have no row in the BrowseComp CSV (e.g. {missing[:3]})"
        return result

    try:
        judge = llm or build_llm(judge_cfg)
    except Exception as exc:  # missing key, unknown provider
        result["reason"] = f"could not build judge LLM: {exc}"
        return result

    async def _grade_all() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(int(judge_cfg.get("concurrency", 8)))

        async def one(traj: Trajectory) -> dict[str, Any]:
            question = traj.query.metadata.get("question") or traj.query.text
            response = traj.report.article if traj.report else ""
            prompt = GRADER_TEMPLATE.format(
                question=question, response=response, correct_answer=answers[traj.query.id]
            )
            t0 = time.perf_counter()
            async with sem:
                try:
                    out = await judge.complete(
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=int(judge_cfg.get("max_output_tokens", 800)),
                    )
                    text = out.text
                    err = None
                except Exception as exc:
                    text, err = "", f"{type(exc).__name__}: {exc}"
            grade = parse_grade(text)
            grade.update(
                {
                    "id": traj.query.id,
                    "topic": traj.query.metadata.get("topic") or traj.query.topic,
                    "correct_answer": answers[traj.query.id],
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

    out_dir = run_dir / "metrics" / "browsecomp"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = out_dir / f"{model_name}.jsonl"
    per_query.write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in grades) + "\n", encoding="utf-8"
    )

    agg = aggregate(grades)
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for g in grades:
        by_topic.setdefault(str(g.get("topic") or "unknown"), []).append(g)
    result.update(agg)
    result["per_topic"] = {k: aggregate(v) for k, v in sorted(by_topic.items())}
    result["per_query_path"] = str(per_query)
    result["official"] = agg["n"] > 0 and all(g.get("judge_error") is None for g in grades)
    if not result["official"]:
        errs = [g["judge_error"] for g in grades if g.get("judge_error")]
        result["reason"] = f"{len(errs)} judge calls failed (e.g. {errs[:1]})" if errs else "no grades"
    return result


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, whether or not a loop is already running
    in this thread (run_experiment calls us via asyncio.to_thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is running in this thread: hop to a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
