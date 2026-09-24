"""Privileged trajectory labels for training rewards and analysis.

``stats_before`` / ``compact_stats`` stay qrel-free: that is the policy
observation. Everything here is *after* the fact — gold docs, answer-string
hits, keep-recall — written to ``Trajectory.labels`` and ``labels/<id>.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from adr.core.types import Trajectory
from adr.tools.browsecomp_plus import docid_from_url

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def compact(text: str) -> str:
    return _PUNCT.sub("", normalize(text))


def answer_variants(answer: str) -> list[str]:
    raw = (answer or "").strip()
    if not raw:
        return []
    seen: list[str] = []
    for cand in (raw, normalize(raw), raw.replace("-", " "), compact(raw)):
        if cand and cand not in seen and len(cand) >= 2:
            seen.append(cand)
    return seen


def first_mention(text: str, answer: str) -> dict[str, Any] | None:
    """First occurrence of the gold answer (or a light variant) in page text."""
    if not text or not answer:
        return None
    variants = answer_variants(answer)
    lower = text.lower()
    best: tuple[int, str] | None = None
    for var in variants:
        if len(var) < 2:
            continue
        idx = lower.find(var.lower())
        if idx < 0 and " " not in var:
            idx = compact(text).find(compact(var))
            if idx >= 0:
                # compact offset is not a char index; fall back to regex on original
                idx = -1
                m = re.search(re.escape(var), text, re.IGNORECASE)
                if m:
                    idx = m.start()
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, var)
    if best is None:
        m = re.search(re.escape(answer.strip()), text, re.IGNORECASE)
        if not m:
            return None
        best = (m.start(), answer.strip())
    start, matched = best
    lo = max(0, start - 80)
    hi = min(len(text), start + len(matched) + 80)
    return {
        "char_start": start,
        "matched": matched,
        "snippet": text[lo:hi].replace("\n", " "),
        "n_mentions": len(re.findall(re.escape(matched), text, re.IGNORECASE)),
    }


def _docids_from_meta(query_meta: dict[str, Any], key: str) -> list[str]:
    out: list[str] = []
    for doc in query_meta.get(key) or []:
        if isinstance(doc, dict) and doc.get("docid"):
            out.append(str(doc["docid"]))
        elif isinstance(doc, str):
            out.append(doc)
    return sorted(set(out))


def _item_docid(item: dict[str, Any]) -> str | None:
    return docid_from_url(item.get("url")) or (
        str(item["docid"]) if item.get("docid") else None
    )


def _recall(got: set[str], labeled: list[str]) -> float | None:
    if not labeled:
        return None
    return round(len(got & set(labeled)) / len(labeled), 4)


def _event(round_id: int, item: dict[str, Any], mention: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "round": round_id,
        "item_id": item.get("id"),
        "docid": _item_docid(item),
        "url": item.get("url"),
        "subquery": item.get("subquery"),
        "mention": mention,
    }


def build_labels(
    traj: Trajectory,
    *,
    items: list[dict[str, Any]] | None = None,
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = traj.query.metadata or {}
    answer = str(meta.get("answer") or "")
    evidence_ids = _docids_from_meta(meta, "evidence_docs")
    gold_ids = _docids_from_meta(meta, "gold_docs")
    catalog = {str(it["id"]): dict(it) for it in (items or []) if it.get("id")}

    prune_steps = [s for s in traj.steps if s.action.type.value == "prune"]
    cum_retrieved: set[str] = set()
    cum_kept: set[str] = set()
    ever_kept: set[str] = set()
    ever_pruned: set[str] = set()
    first_answer = None
    first_answer_kept = None
    first_evidence = None
    first_evidence_kept = None
    first_gold = None
    first_gold_kept = None
    discarded_gold = False
    discarded_answer = False
    rounds: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    latency_cum = 0.0
    tokens_cum = 0
    searches_after_answer = 0
    rounds_after_answer = 0
    answer_found = False

    for step in prune_steps:
        extra = step.extra or {}
        round_id = int(extra.get("round_id") or step.step)
        new_ids = [str(x) for x in extra.get("new_item_ids") or []]
        kept_ids = [str(x) for x in extra.get("kept_item_ids") or []]
        pruned_ids = [str(x) for x in extra.get("pruned_item_ids") or []]
        new_docids = [str(x) for x in extra.get("new_docids") or []]
        kept_docids = [str(x) for x in extra.get("kept_docids") or []]
        pruned_docids = [str(x) for x in extra.get("pruned_docids") or []]

        for iid in new_ids:
            it = catalog.get(iid) or {"id": iid}
            did = _item_docid(it)
            if did:
                new_docids.append(did)
                cum_retrieved.add(did)
                if did in evidence_ids and first_evidence is None:
                    first_evidence = _event(round_id, it)
                if did in gold_ids and first_gold is None:
                    first_gold = _event(round_id, it)
            mention = first_mention(str(it.get("text") or ""), answer)
            if mention and first_answer is None:
                first_answer = _event(round_id, it, mention)
                answer_found = True
            if iid not in seen_items:
                seen_items.add(iid)
                item_rows.append(
                    {
                        "id": iid,
                        "docid": did,
                        "url": it.get("url"),
                        "subquery": it.get("subquery"),
                        "tree_depth": it.get("tree_depth"),
                        "retrieval_round": it.get("retrieval_round") or round_id,
                        "is_evidence": bool(did and did in evidence_ids),
                        "is_gold": bool(did and did in gold_ids),
                        "answer_mention": mention,
                        "chars": len(str(it.get("text") or "")),
                    }
                )

        for iid in kept_ids:
            it = catalog.get(iid) or {"id": iid}
            did = _item_docid(it)
            if did:
                kept_docids.append(did)
                cum_kept.add(did)
                ever_kept.add(did)
                if did in evidence_ids and first_evidence_kept is None:
                    first_evidence_kept = _event(round_id, it)
                if did in gold_ids and first_gold_kept is None:
                    first_gold_kept = _event(round_id, it)
            mention = first_mention(str(it.get("text") or ""), answer)
            if mention and first_answer_kept is None:
                first_answer_kept = _event(round_id, it, mention)
                answer_found = True

        for iid in pruned_ids:
            it = catalog.get(iid) or {"id": iid}
            did = _item_docid(it)
            if did:
                pruned_docids.append(did)
                ever_pruned.add(did)
                cum_kept.discard(did)

        new_docids = sorted(set(new_docids))
        kept_docids = sorted(set(kept_docids))
        pruned_docids = sorted(set(pruned_docids))
        latency_cum += float(step.latency_s or 0.0)
        tokens_cum += int(step.tokens.total_tokens or 0)
        search_calls = int(extra.get("search_calls") or 0)
        if answer_found and first_answer and round_id > int(first_answer["round"]):
            rounds_after_answer += 1
            searches_after_answer += search_calls

        had_answer_new = first_answer is not None and int(first_answer["round"]) == round_id
        had_gold_new = first_gold is not None and int(first_gold["round"]) == round_id
        answer_only_in_pruned = False
        if answer:
            for iid in pruned_ids:
                if first_mention(str((catalog.get(iid) or {}).get("text") or ""), answer):
                    if iid not in kept_ids:
                        answer_only_in_pruned = True
                        discarded_answer = True
        if set(pruned_docids) & set(gold_ids) and not (set(kept_docids) & set(gold_ids)):
            if not (cum_kept & set(gold_ids)):
                discarded_gold = True

        rounds.append(
            {
                "round": round_id,
                "step": step.step,
                "decision_type": extra.get("decision_type"),
                "n_new": len(new_ids),
                "n_kept": len(kept_ids),
                "n_pruned": len(pruned_ids),
                "new_docids": new_docids,
                "kept_docids": kept_docids,
                "pruned_docids": pruned_docids,
                "n_new_evidence": len(set(new_docids) & set(evidence_ids)),
                "n_new_gold": len(set(new_docids) & set(gold_ids)),
                "n_kept_evidence": len(set(kept_docids) & set(evidence_ids)),
                "n_kept_gold": len(set(kept_docids) & set(gold_ids)),
                "n_pruned_evidence": len(set(pruned_docids) & set(evidence_ids)),
                "n_pruned_gold": len(set(pruned_docids) & set(gold_ids)),
                "first_answer_this_round": had_answer_new,
                "first_gold_this_round": had_gold_new,
                "answer_only_in_pruned": answer_only_in_pruned,
                "retrieved_recall_evidence": _recall(cum_retrieved, evidence_ids),
                "retrieved_recall_gold": _recall(cum_retrieved, gold_ids),
                "keep_recall_evidence": _recall(cum_kept, evidence_ids),
                "keep_recall_gold": _recall(cum_kept, gold_ids),
                "search_queries": list(extra.get("search_queries") or []),
                "search_calls": search_calls,
                "tokens_cum": tokens_cum,
                "latency_s_cum": round(latency_cum, 4),
                "unique_retrieved": len(cum_retrieved),
            }
        )

    retrieved = set(str(x) for x in (traj.final_stats or {}).get("retrieved_docids") or [])
    cum_retrieved |= retrieved
    article = traj.report.article if traj.report else ""
    answer_in_report = bool(first_mention(article, answer)) if answer else None
    if discarded_gold and (cum_kept & set(gold_ids)):
        discarded_gold = False
    if discarded_answer and first_answer_kept:
        discarded_answer = False

    path = "never"
    if first_answer_kept or first_gold_kept:
        path = "kept"
    elif first_answer or first_gold or (retrieved & (set(gold_ids) | set(evidence_ids))):
        path = "retrieved_only"
    if answer_in_report:
        path = "written" if path == "kept" else f"{path}+written"

    summary = {
        "first_seen_answer": first_answer,
        "first_kept_answer": first_answer_kept,
        "first_seen_evidence": first_evidence,
        "first_kept_evidence": first_evidence_kept,
        "first_seen_gold": first_gold,
        "first_kept_gold": first_gold_kept,
        "answer_path": path,
        "discarded_answer": discarded_answer,
        "discarded_gold": discarded_gold,
        "rounds_before_answer": (int(first_answer["round"]) if first_answer else None),
        "rounds_after_answer": rounds_after_answer,
        "searches_after_answer": searches_after_answer,
        "retrieved_recall_evidence": _recall(cum_retrieved, evidence_ids),
        "retrieved_recall_gold": _recall(cum_retrieved, gold_ids),
        "keep_recall_evidence": _recall(cum_kept, evidence_ids),
        "keep_recall_gold": _recall(cum_kept, gold_ids),
        "n_evidence_labeled": len(evidence_ids),
        "n_gold_labeled": len(gold_ids),
        "answer_in_report": answer_in_report,
        "n_rounds": len(rounds),
        "n_items": len(item_rows),
    }
    return {
        "query_id": traj.query.id,
        "dataset": traj.query.dataset,
        "split": meta.get("split"),
        "run": dict(run or {}),
        "policy_note": (
            "Do not feed this object into the policy. s_t is step.stats_before; "
            "these fields are reward / analysis only."
        ),
        "summary": summary,
        "rounds": rounds,
        "items": item_rows,
    }


def annotate_trajectory(
    traj: Trajectory,
    *,
    items: list[dict[str, Any]] | None = None,
    run: dict[str, Any] | None = None,
) -> Trajectory:
    labels = build_labels(traj, items=items, run=run)
    traj.labels = labels
    if traj.final_stats is None:
        traj.final_stats = {}
    traj.final_stats.pop("_label_items", None)
    traj.final_stats["oracle"] = labels["summary"]
    prune = [s for s in traj.steps if s.action.type.value == "prune"]
    for step, row in zip(prune, labels["rounds"]):
        step.extra["oracle"] = {
            k: row[k]
            for k in (
                "first_answer_this_round",
                "first_gold_this_round",
                "answer_only_in_pruned",
                "n_new_evidence",
                "n_new_gold",
                "n_kept_evidence",
                "n_kept_gold",
                "n_pruned_evidence",
                "n_pruned_gold",
                "retrieved_recall_evidence",
                "retrieved_recall_gold",
                "keep_recall_evidence",
                "keep_recall_gold",
                "unique_retrieved",
            )
            if k in row
        }
    return traj


def write_labels(run_dir: Path, traj: Trajectory) -> Path | None:
    if not traj.labels:
        return None
    dest = Path(run_dir) / "labels" / f"{traj.query.id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(traj.labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest
