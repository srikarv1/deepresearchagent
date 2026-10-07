"""Robustness checks for the 5-question Bedrock test pair (learned stop vs No pruning).

A. Re-judge every report R times (Gym quality + KPR, Qwen3-32B) -> mean and spread.
B. Pairwise learned vs No pruning, both orders, R repeats, raw verdicts kept, so a
   true tie can be told apart from position bias (A,A / B,B).
C. Writer-order check on live evidence: repack the No-pruning run's evidence
   shallow-first (tree depth, then round) into the same 25K-word budget, rewrite the
   report with the agent's writer, and judge it like the others.

    set -a; source .env; source scripts/replaygym_env.sh; set +a
    REPLAY_Q_WORKERS=3 python scripts/test_robustness.py --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from adr.replay.judge import judge_reports
from adr.replay.rl import DATA, PairwiseJudge
from adr.replay.synth import apply_agent_env, synthesize_many

TEST = ["471082", "745702", "879779", "921842", "923549"]
OUT = DATA / "robustness"
MAX_WORDS = 25000


def find(arm: str, qid: str, kind: str) -> Path:
    for d in sorted(glob.glob(f"runs/*-bedrock-test-{arm}*")):
        p = Path(d) / (f"reports/{qid}.md" if kind == "report" else f"gpt_researcher/{qid}.json")
        if p.exists():
            return p
    raise FileNotFoundError(f"{arm} {qid} {kind}")


def shallow_first(traj: dict) -> str:
    items = sorted(traj["evidence"].values(), key=lambda e: (e.get("tree_depth") or 0, e.get("retrieval_round") or 0))
    out, used = [], 0
    for e in items:
        w = len((e.get("content") or "").split())
        if w and used + w <= MAX_WORDS:
            out.append(e["content"])
            used += w
    return "\n".join(out)


def pair_raw(pj: PairwiseJudge, question: str, a: str, b: str) -> tuple[float | None, float | None]:
    """Return (P(a wins | a shown first), P(a wins | a shown second)) for one repeat."""
    from openai import AsyncOpenAI

    async def run():
        client = AsyncOpenAI()
        ab = await pj._one(client, question, a, b)
        ba = await pj._one(client, question, b, a)
        return ab, (None if ba is None else 1.0 - ba)

    return asyncio.run(run())


def one_question(qid: str, repeats: int) -> tuple[str, dict]:
    qdir = OUT / qid
    qdir.mkdir(parents=True, exist_ok=True)
    traj_np = json.loads(find("nopruning", qid, "traj").read_text())
    question = traj_np["query"]
    reports = {"learned": find("learned", qid, "report").read_text(), "nopruning": find("nopruning", qid, "report").read_text()}

    # C. shallow-first rewrite of the No-pruning evidence (cached on disk)
    f = qdir / "nopruning_shallowfirst.md"
    if not (f.exists() and f.stat().st_size > 2000):
        got = asyncio.run(synthesize_many(question, {"sf": shallow_first(traj_np)}, 1))
        if got.get("sf", "").strip():
            f.write_text(got["sf"], encoding="utf-8")
    if f.exists():
        reports["nopruning_shallowfirst"] = f.read_text(encoding="utf-8")

    # A. repeated absolute judging
    memo_p = qdir / "scores.json"
    memo = json.loads(memo_p.read_text()) if memo_p.exists() else {}
    for r in range(repeats):
        need = {k: v for k, v in reports.items() if f"{k}#{r}" not in memo}
        for attempt in range(3):
            if not need:
                break
            got = judge_reports(qid, question, need, qdir / "judge" / f"r{r}a{attempt}")
            for k, sc in got.items():
                if sc.get("quality") is not None and sc.get("kpr") is not None:
                    memo[f"{k}#{r}"] = {"quality": sc["quality"], "kpr": sc["kpr"]}
                    need.pop(k, None)
        memo_p.write_text(json.dumps(memo, indent=1))

    # B. pairwise with raw verdicts
    pj = PairwiseJudge(memo_dir=qdir / "pairs_unused", concurrency=1)
    raw_p = qdir / "pairs_raw.json"
    raw = json.loads(raw_p.read_text()) if raw_p.exists() else {}
    for other in [k for k in reports if k != "learned"]:
        for r in range(repeats):
            key = f"learned|{other}#{r}"
            if key not in raw:
                first, second = pair_raw(pj, question, reports["learned"], reports[other])
                raw[key] = {"learned_first": first, "learned_second": second}
                raw_p.write_text(json.dumps(raw, indent=1))
    words = {k: len(v.split()) for k, v in reports.items()}
    return qid, {"scores": memo, "pairs": raw, "words": words}


def summarize(res: dict, repeats: int) -> dict:
    arms = sorted({k.split("#")[0] for r in res.values() for k in r["scores"]})
    summary = {}
    for arm in arms:
        q = [[res[qid]["scores"].get(f"{arm}#{r}", {}).get("quality") for r in range(repeats)] for qid in TEST]
        k = [[res[qid]["scores"].get(f"{arm}#{r}", {}).get("kpr") for r in range(repeats)] for qid in TEST]
        qm = np.array([[x for x in row if x is not None] for row in q], dtype=object)
        per_q_mean = [float(np.mean(row)) if len(row) else np.nan for row in qm]
        per_q_sd = [float(np.std(row)) if len(row) > 1 else np.nan for row in qm]
        kp = [float(np.mean([x for x in row if x is not None])) for row in k]
        summary[arm] = {"overall_mean": float(np.nanmean(per_q_mean)), "overall_judge_sd_per_q": float(np.nanmean(per_q_sd)),
                        "kpr_mean": float(np.mean(kp)), "per_q_overall": dict(zip(TEST, per_q_mean)),
                        "words_mean": float(np.mean([res[qid]["words"].get(arm, 0) for qid in TEST]))}
    pairs = {}
    for other in [a for a in arms if a != "learned"]:
        consistent_win = consistent_loss = tie = position = 0
        for qid in TEST:
            for r in range(repeats):
                v = res[qid]["pairs"].get(f"learned|{other}#{r}")
                if not v or v["learned_first"] is None or v["learned_second"] is None:
                    continue
                a, b = v["learned_first"], v["learned_second"]
                if a == 1 and b == 1:
                    consistent_win += 1
                elif a == 0 and b == 0:
                    consistent_loss += 1
                elif a == 0.5 and b == 0.5:
                    tie += 1
                else:
                    position += 1  # verdict flips with order (or mixed tie)
        pairs[other] = {"learned_wins_both_orders": consistent_win, "learned_loses_both_orders": consistent_loss,
                        "declared_tie": tie, "order_dependent": position}
    return {"arms": summary, "pairwise_vs_learned": pairs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    apply_agent_env()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("REPLAY_Q_WORKERS", "3"))) as ex:
        res = dict(ex.map(lambda q: one_question(q, a.repeats), TEST))
    s = summarize(res, a.repeats)
    (OUT / "summary.json").write_text(json.dumps({"summary": s, "raw": res}, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
