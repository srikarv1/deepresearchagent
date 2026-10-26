"""One teacher call per question, bounded retries, explicit opt-in to API calls."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable

from pydantic import Field, ValidationError

from adr.llm.factory import build_llm
from adr.llm.mock import MockLLM
from adr.runner.config import load_config
from adr.synthetic.prompt import PROMPT_VERSION, teacher_messages
from adr.synthetic.schema import (
    FEATURE_NAMES, SCHEMA_VERSION, StrictModel, SyntheticExample, TeacherTrajectory,
    derive_features, serialize_context,
)
from adr.synthetic.validate import load_heldout, validate_examples


class GenerationConfig(StrictModel):
    dataset_version: str = Field(default="v1", pattern=r"^[a-zA-Z0-9_-]+$")
    output_dir: str = "data/synthetic"
    heldout_path: str = "data/heldout/drb_baseline.json"
    questions: int = Field(default=20, ge=1, le=100)
    states_per_question: int = Field(default=5, ge=1)
    candidates_per_state: int = Field(default=4, ge=1)
    max_attempts: int = Field(default=3, ge=1, le=5)
    llm: dict = Field(default_factory=lambda: {"provider": "mock", "model": "synthetic-smoke"})
    topics: list[str] = Field(default_factory=lambda: ["municipal water reuse"] , min_length=1)


def parse_teacher(text: str) -> TeacherTrajectory:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return TeacherTrajectory.model_validate(json.loads(text))


def compile_examples(trajectory: TeacherTrajectory, *, question_id: str, config: GenerationConfig,
                     timestamp: str, provider: str, model: str) -> list[dict]:
    tid = f"{question_id}-trajectory"
    rows = []
    for index, item in enumerate(trajectory.states):
        for candidate in item.candidates:
            identity = json.dumps([tid, item.state_id, candidate.candidate_id])
            record = SyntheticExample(
                example_id="ex-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                question_id=question_id, trajectory_id=tid, state_id=item.state_id,
                candidate_id=candidate.candidate_id, original_question=trajectory.original_question,
                context_before=serialize_context(item.state), candidate_subquery=candidate.candidate_subquery,
                state=item.state, numerical_features=derive_features(trajectory.original_question, item.state),
                label=candidate.label, decision_category=candidate.decision_category, reason=candidate.reason,
                provenance={"prompt_version": PROMPT_VERSION, "dataset_version": config.dataset_version,
                            "generated_at": timestamp, "teacher_provider": provider, "teacher_model": model,
                            "state_index": index, "counterfactual_pair_id": candidate.counterfactual_pair_id},
            )
            rows.append(record.model_dump())
    return rows


def smoke_response() -> str:
    """Tiny fake trajectory for plumbing only; never presented as teacher data."""
    candidate = "What maintenance limits affect rainwater tanks at rural libraries?"
    states = []
    for index in range(2):
        states.append({"state_id": f"s{index}", "state": {
            "evidence": [] if index == 0 else [{"id": "e1", "url": "https://example.org/tanks",
                "text": "In this fictional scenario, tanks require monthly filter cleaning."}],
            "executed_subqueries": [] if index == 0 else [candidate],
            "branches": [{"id": "b1", "goal": "Establish maintenance limits", "status": "open" if index == 0 else "done"}],
            "budget": {"used_steps": index, "used_searches": index, "used_reads": index, "used_tokens": index * 1000}},
            "candidates": [{"candidate_id": "c1", "candidate_subquery": candidate,
                "label": "EXECUTE" if index == 0 else "SKIP",
                "decision_category": "unanswered_information_need" if index == 0 else "already_established",
                "reason": "Maintenance requirements are still unknown." if index == 0 else "The retained evidence establishes the requested requirement.",
                "counterfactual_pair_id": "pair1"},
                {"candidate_id": "c2", "candidate_subquery": "Which procurement options fit the library budget?",
                 "label": "SKIP" if index == 0 else "EXECUTE",
                 "decision_category": "premature" if index == 0 else "necessary_dependency",
                 "reason": "Establish maintenance feasibility first." if index == 0 else "Procurement constraints remain necessary for the recommendation."}]})
    return json.dumps({"original_question": "Compare rainwater tanks for fictional rural libraries: maintenance, procurement, and implementation tradeoffs.", "states": states})


def retry_feedback(exc: Exception) -> str:
    feedback = "Previous attempt failed local validation or request handling. Generate a fresh original question; check required counts, uniqueness, state progression, pair links and JSON schema carefully."
    # Match a known validator message, never forward raw exceptions: Pydantic
    # errors can include entire responses, and other errors can contain held-out
    # references or credentials. Only fixed, allowlisted guidance reaches the teacher.
    if isinstance(exc, ValidationError) and any(
        error["msg"] == "Value error, prior evidence removed or rewritten; change retention instead"
        for error in exc.errors(include_input=False, include_context=False, include_url=False)
    ):
        feedback += (
            " Validation failure: prior evidence removed or rewritten; change retention instead."
            " In every later state, copy ALL earlier evidence items, including pruned ones,"
            " preserving each id, url, title, and text exactly. Only retained may change"
            " on an existing item; use retained=false for pruning. Add new findings under"
            " new IDs instead of replacing or rewriting old items. Return a complete"
            " fresh trajectory under the original task and all its constraints."
        )
    prefix = "generated dataset failed local validation: "
    if type(exc) is ValueError and str(exc).startswith(prefix) and any(
        error.startswith("invalid counterfactual pair: ")
        for error in str(exc)[len(prefix):].split("; ")
    ):
        feedback += (
            " Validation failure: invalid counterfactual pair. Each non-null"
            " counterfactual_pair_id must occur exactly twice across the entire trajectory."
            " The first occurrence must be EXECUTE in an earlier state; the second"
            " must be SKIP in a later state. The two occurrences must belong to"
            " different chronological states. Use null for unrelated candidates and"
            " any additional appearances that are not those two designated records."
            " Before returning, audit every non-null pair ID for occurrence count,"
            " labels, and state indices. Return a complete fresh trajectory under"
            " the original task and all its constraints."
        )
    return feedback


async def generate_dataset(config: GenerationConfig, *, client, heldout: list[dict], timestamp: str,
                           initial_rows: list[dict] | None = None,
                           initial_attempts: list[dict] | None = None,
                           checkpoint: Callable | None = None) -> tuple[list[dict], list[dict]]:
    if not heldout:
        raise ValueError("held-out registry is required")
    rows, attempts = list(initial_rows or []), list(initial_attempts or [])
    completed = validate_completed(rows, config, heldout)
    for index in range(completed, config.questions):
        messages = teacher_messages(config.states_per_question, config.candidates_per_state,
                                    config.topics[index % len(config.topics)])
        last_error = ""
        for attempt in range(config.max_attempts):
            try:
                # JSON mode plus local Pydantic validation. Never send held-out
                # texts, IDs, or validator messages containing them to the teacher.
                response = await client.complete(messages, extra={"response_format": {"type": "json_object"}})
                attempts.append({"question_index": index, "attempt": attempt + 1,
                                 "response_sha256": hashlib.sha256(response.text.encode()).hexdigest(),
                                 "usage": response.usage.model_dump()})
                trajectory = parse_teacher(response.text)
                proposed = compile_examples(trajectory, question_id=f"{config.dataset_version}-q{index + 1:03}",
                                            config=config, timestamp=timestamp, provider=client.name, model=client.model)
                summary = validate_examples(rows + proposed, heldout, expected_states=config.states_per_question,
                                            expected_candidates=config.candidates_per_state)
                if not summary["valid"]:
                    raise ValueError("generated dataset failed local validation: " + "; ".join(summary["errors"]))
                rows.extend(proposed)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Do not echo malformed responses or held-out reference material.
                messages = teacher_messages(config.states_per_question, config.candidates_per_state,
                                            config.topics[index % len(config.topics)])
                messages.append({"role": "user", "content": retry_feedback(exc)})
        else:
            if checkpoint:
                checkpoint(rows, attempts)
            raise RuntimeError(f"Question {index + 1} failed after {config.max_attempts} attempts: {last_error}")
        # Disk failures must propagate, not retry an already accepted teacher call.
        if checkpoint:
            checkpoint(rows, attempts)
    return rows, attempts


def safe_config(config: GenerationConfig) -> dict:
    """Persist reproducibility settings, not credentials or arbitrary client extras."""
    data = config.model_dump()
    data["llm"] = {key: value for key, value in config.llm.items() if key in {
        "provider", "model", "temperature", "max_output_tokens", "api_key_env"}}
    return data


def validate_completed(rows: list[dict], config: GenerationConfig, heldout: list[dict]) -> int:
    """A checkpoint must contain only a contiguous prefix of complete questions."""
    if not rows:
        return 0
    summary = validate_examples(rows, heldout, expected_states=config.states_per_question,
                                expected_candidates=config.candidates_per_state)
    if not summary["valid"]:
        raise ValueError("Invalid checkpoint examples: " + "; ".join(summary["errors"]))
    completed = summary["questions"]
    ids = list(dict.fromkeys(row["question_id"] for row in rows))
    expected = [f"{config.dataset_version}-q{i + 1:03}" for i in range(completed)]
    if completed > config.questions or ids != expected or any(
        row["trajectory_id"] != f"{row['question_id']}-trajectory" for row in rows
    ):
        raise ValueError("Checkpoint question IDs must be a contiguous configured prefix")
    return completed


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_checkpoint(output: Path, rows: list[dict], manifest: dict) -> None:
    # One atomic source of truth: canonical files can be repaired from this if
    # interrupted between replacements. Keep downstream file schemas unchanged.
    atomic_write(output / "checkpoint.json", json.dumps(
        {"examples": rows, "manifest": manifest}, ensure_ascii=False, indent=2))
    atomic_write(output / "examples.jsonl", "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2))


def load_checkpoint(output: Path, config: GenerationConfig, heldout: list[dict],
                    heldout_sha256: str, dry_run: bool) -> tuple[list[dict], dict]:
    data = json.loads((output / "checkpoint.json").read_text())
    rows, manifest = data["examples"], data["manifest"]
    expected = {"config": safe_config(config), "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION, "feature_names": list(FEATURE_NAMES),
                "heldout_sha256": heldout_sha256, "dry_run": dry_run}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Cannot resume: checkpoint {key} differs from current settings")
    validate_completed(rows, config, heldout)
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_mlp.yaml"))
    parser.add_argument("--questions", type=int)
    parser.add_argument("--dry-run", action="store_true", help="one local fake question, two states, two candidates")
    parser.add_argument("--allow-api", action="store_true", help="explicitly permit configured non-mock provider calls")
    parser.add_argument("--resume", type=Path, help="resume an existing generation directory")
    args = parser.parse_args()
    raw = load_config(args.config)
    raw.pop("_config_path", None)
    if args.questions is not None:
        raw["questions"] = args.questions
    if args.dry_run:
        raw.update(questions=1, states_per_question=2, candidates_per_state=2)
    config = GenerationConfig.model_validate(raw)
    heldout = load_heldout(Path(config.heldout_path))
    heldout_sha256 = hashlib.sha256(Path(config.heldout_path).read_bytes()).hexdigest()
    rows, attempts = [], []
    manifest = None
    if args.resume:
        output = args.resume
        rows, manifest = load_checkpoint(output, config, heldout, heldout_sha256, args.dry_run)
        attempts = manifest["attempts"]
        if validate_completed(rows, config, heldout) == config.questions:
            save_checkpoint(output, rows, manifest)
            print(json.dumps({"output": str(output), **manifest["summary"]}, indent=2))
            return
    if args.dry_run:
        client = MockLLM(model="synthetic-smoke", replies=[smoke_response()])
    else:
        if config.llm.get("provider", "mock") not in {"mock", "none"} and not args.allow_api:
            parser.error("Non-mock generation requires --allow-api; use --dry-run for a local smoke test")
        if str(config.llm.get("provider", "mock")).lower() == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
            parser.error("OPENAI_API_KEY must be set and non-empty for OpenAI API generation; no request was sent")
        client = build_llm(config.llm)
    timestamp = manifest["generated_at"] if manifest else datetime.now(timezone.utc).isoformat()
    if manifest is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path(config.output_dir) / config.dataset_version / (("smoke-" if args.dry_run else "generation-") + tag)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION, "generated_at": timestamp,
                "config": safe_config(config), "actual_teacher": {"provider": client.name, "model": client.model},
                "dry_run": args.dry_run, "feature_names": FEATURE_NAMES, "attempts": attempts,
                "heldout_sha256": heldout_sha256}
    elif manifest["actual_teacher"] != {"provider": client.name, "model": client.model}:
        raise ValueError("Cannot resume with a different teacher")

    def checkpoint(accepted, history):
        manifest["attempts"] = list(history)
        manifest["summary"] = validate_examples(accepted, heldout,
            expected_states=config.states_per_question, expected_candidates=config.candidates_per_state)
        save_checkpoint(output, accepted, manifest)

    checkpoint(rows, attempts)
    print(f"Generation directory: {output} (completed {len(set(r['question_id'] for r in rows))}/{config.questions})",
          file=sys.stderr, flush=True)
    rows, attempts = asyncio.run(generate_dataset(config, client=client, heldout=heldout, timestamp=timestamp,
        initial_rows=rows, initial_attempts=attempts, checkpoint=checkpoint))
    summary = manifest["summary"]
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
