"""Dataset audit and separate held-out registry. No reference outputs are loaded."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import zipfile

from adr.synthetic.schema import SyntheticExample, TeacherTrajectory


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def near_duplicate(left: str, right: str) -> bool:
    a, b = normalize(left), normalize(right)
    sa, sb = set(a.split()), set(b.split())
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.85 or (
        bool(sa | sb) and len(sa & sb) / len(sa | sb) >= 0.8
    )


def load_heldout(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())["questions"]
    if not rows or any(not str(r.get("id", "")).strip() or not r.get("text", "").strip() for r in rows):
        raise ValueError("held-out registry must contain nonempty IDs and text")
    return rows


def import_heldout_zip(source: Path, destination: Path) -> None:
    """Conservatively exclude the union of query IDs; never read reports/steps.

    Keep this registry separate from training data and never send it to teacher.
    """
    rows = {}
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if "/queries/" not in name or not name.endswith(".json"):
                continue
            q = json.loads(archive.read(name))
            text = q.get("text") or q.get("prompt")
            if not text or not q.get("id"):
                raise ValueError(f"unrecognized query schema: {name}")
            key = str(q["id"])
            if key in rows and rows[key]["text"] != text:
                raise ValueError(f"conflicting held-out text for ID {key}")
            rows[key] = {"id": key, "text": text}
    if not rows:
        raise ValueError("archive contains no query records")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump({"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "policy": "union of all archive query subsets", "questions": list(rows.values())}, handle, indent=2)


def validate_examples(rows: list[dict], heldout: list[dict], *, expected_states: int | None = None,
                      expected_candidates: int | None = None) -> dict:
    errors, warnings, examples = [], [], []
    for i, row in enumerate(rows):
        try:
            examples.append(SyntheticExample.model_validate(row))
        except ValueError as exc:
            errors.append(f"row {i}: {exc}")
    seen_ids, seen_keys, seen_content = set(), set(), set()
    groups, questions, trajectories, pairs = defaultdict(dict), {}, {}, defaultdict(list)
    for ex in examples:
        key = (ex.trajectory_id, ex.state_id, ex.candidate_id)
        content = (normalize(ex.original_question), ex.context_before, normalize(ex.candidate_subquery))
        if ex.example_id in seen_ids or key in seen_keys or content in seen_content:
            errors.append(f"duplicate example: {ex.example_id}")
        seen_ids.add(ex.example_id); seen_keys.add(key); seen_content.add(content)
        if ex.question_id in questions and questions[ex.question_id] != ex.original_question:
            errors.append(f"question text changed for {ex.question_id}")
        questions[ex.question_id] = ex.original_question
        if ex.trajectory_id in trajectories and trajectories[ex.trajectory_id] != ex.question_id:
            errors.append(f"trajectory crosses question groups: {ex.trajectory_id}")
        trajectories[ex.trajectory_id] = ex.question_id
        state_key = (ex.provenance.state_index, ex.state_id)
        bucket = groups[ex.trajectory_id].setdefault(state_key, [])
        if bucket and bucket[0].state != ex.state:
            errors.append(f"inconsistent shared state: {ex.state_id}")
        bucket.append(ex)
        if ex.provenance.counterfactual_pair_id:
            pairs[(ex.trajectory_id, ex.provenance.counterfactual_pair_id)].append(ex)
    for qid, text in questions.items():
        for reference in heldout:
            if qid == str(reference["id"]) or near_duplicate(text, reference["text"]):
                errors.append(f"held-out overlap: {qid} vs reference {reference['id']}")
    question_items = list(questions.items())
    for i, (qid, text) in enumerate(question_items):
        for other_id, other_text in question_items[:i]:
            if near_duplicate(text, other_text):
                errors.append(f"duplicate question groups: {qid}, {other_id}")
    state_counts, candidate_counts = {}, {}
    for tid, states in groups.items():
        ordered = sorted(states.items())
        indexes = [key[0] for key, _ in ordered]
        if indexes != list(range(len(indexes))):
            errors.append(f"non-contiguous/repeated state indices: {tid}")
        state_counts[tid] = len(states)
        if expected_states is not None and len(states) != expected_states:
            errors.append(f"{tid}: expected {expected_states} states, got {len(states)}")
        teacher_states = []
        for (_, sid), candidates in ordered:
            candidate_counts[f"{tid}/{sid}"] = len(candidates)
            if expected_candidates is not None and len(candidates) != expected_candidates:
                errors.append(f"{tid}/{sid}: expected {expected_candidates} candidates, got {len(candidates)}")
            teacher_states.append(dict(state_id=sid, state=candidates[0].state, candidates=[
                {"candidate_id": e.candidate_id, "candidate_subquery": e.candidate_subquery,
                 "label": e.label, "decision_category": e.decision_category, "reason": e.reason}
                for e in candidates]))
        try:
            TeacherTrajectory(original_question=ordered[0][1][0].original_question, states=teacher_states)
        except ValueError as exc:
            errors.append(f"{tid}: {exc}")
    for (tid, pid), pair in pairs.items():
        pair = sorted(pair, key=lambda e: e.provenance.state_index)
        if (len(pair) != 2 or [e.label for e in pair] != ["EXECUTE", "SKIP"]
                or pair[0].provenance.state_index >= pair[-1].provenance.state_index):
            errors.append(f"invalid counterfactual pair: {tid}/{pid}")
        elif not near_duplicate(pair[0].candidate_subquery, pair[1].candidate_subquery):
            warnings.append(f"review semantic equivalence of pair: {tid}/{pid}")
        if len(pair) == 2 and pair[0].state.model_dump(exclude={"budget"}) == pair[1].state.model_dump(exclude={"budget"}):
            errors.append(f"counterfactual pair has no changed research information: {tid}/{pid}")
    counts = Counter(e.label for e in examples)
    ratio = counts["EXECUTE"] / len(examples) if examples else 0
    if not examples:
        errors.append("no valid examples")
    if not 0.35 <= ratio <= 0.65:
        warnings.append("EXECUTE fraction outside approximate balance range [0.35, 0.65]")
    if not pairs:
        warnings.append("no linked counterfactual pairs")
    warnings.append("Lexical overlap and strict fields cannot certify absence of semantic paraphrases or narrative outcome leakage; human review remains necessary.")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "examples": len(examples),
            "questions": len(questions), "trajectories": len(groups), "labels": dict(counts),
            "execute_fraction": ratio, "states_per_trajectory": state_counts,
            "candidates_per_state": candidate_counts, "counterfactual_pairs": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--import-zip", type=Path)
    args = parser.parse_args()
    if args.import_zip:
        import_heldout_zip(args.import_zip, args.heldout)
        return
    if not args.examples:
        parser.error("--examples is required unless importing a ZIP")
    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    summary = validate_examples(rows, load_heldout(args.heldout))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
