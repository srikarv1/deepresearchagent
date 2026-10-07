"""Is "deeper research doesn't help" real, or is the writer throwing the research away?

gpt-researcher builds the writer's context as [learnings..., raw contexts...] and
trims it to 25K words keeping the END of the list, so a full b4 d3 tree
(~145K words) reaches the writer as ~15% of its evidence and none of its
distilled learnings. Depth 1 (~21K words) fits whole. This replays the same
cached evidence under two alternative writer contexts:

  lf25k  learnings first, then raw contexts shallow-first, SAME 25K-word budget
         (isolates *which* evidence the writer sees)
  lf60k  same order, 60K-word budget (~80K tokens; isolates *how much*)

and scores every variant with the Gym judge (quality, KPR) plus a
position-debiased pairwise judgment against depth 1.

    set -a; source .env; source scripts/replaygym_env.sh; set +a
    REPLAY_Q_WORKERS=3 python scripts/writer_cap_experiment.py --split val
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from adr.replay.judge import judge_reports
from adr.replay.rl import DATA, PairwiseJudge, RewardService, load_split
from adr.replay.synth import synthesize_many

OUT = DATA / "writercap"


def learnings_first(tree, nodes, cap: int) -> str:
    from gpt_researcher.skills.deep_research import count_words

    ordered = sorted((n for n in tree.order if n in nodes), key=lambda n: tree.nodes[n].tree_depth)  # stable: shallow first
    learn, cites, seen = [], {}, set()
    for n in ordered:
        cites.update(tree.nodes[n].citations)
        for l in tree.nodes[n].learnings:
            if l not in seen:
                seen.add(l)
                learn.append(l)
    items = [f"{l} [Source: {cites[l]}]" if cites.get(l) else l for l in learn]
    items += [tree.nodes[n].context for n in ordered if tree.nodes[n].context]
    out, used = [], 0
    for it in items:  # take in priority order, skip what doesn't fit
        w = count_words(it)
        if used + w <= cap:
            out.append(it)
            used += w
    return "\n".join(out)


def run_question(qid, tree, svc, pj):
    qdir = OUT / qid
    qdir.mkdir(parents=True, exist_ok=True)
    plans = {"d1": tree.up_to_depth(1), "d2": tree.up_to_depth(2), "full": tree.all_nodes()}
    # default writer context (the agent's own rule): reuse the memoized eval reports/scores
    base = svc.scores(qid, tree, [frozenset(p) for p in plans.values()])
    reports = {f"{k}__default": svc.report_text(qid, frozenset(p)) for k, p in plans.items()}
    scores = {f"{k}__default": sc for k, sc in zip(plans, base)}

    variants = {"d2__lf25k": ("d2", 25000), "full__lf25k": ("full", 25000), "full__lf60k": ("full", 60000)}
    todo = {}
    for tag, (plan, cap) in variants.items():
        f = qdir / f"{tag}.md"
        if f.exists() and f.stat().st_size > 2000:
            reports[tag] = f.read_text(encoding="utf-8")
        else:
            todo[tag] = learnings_first(tree, plans[plan], cap)
    if todo:
        got = asyncio.run(synthesize_many(tree.query, todo, int(os.environ.get("REPLAY_SYNTH_CONC", "1"))))
        for tag, r in got.items():
            if r.strip():
                (qdir / f"{tag}.md").write_text(r, encoding="utf-8")
                reports[tag] = r

    memo_p = qdir / "scores.json"
    memo = json.loads(memo_p.read_text()) if memo_p.exists() else {}
    need = {t: reports[t] for t in variants if t in reports and t not in memo}
    for attempt in range(4):
        if not need:
            break
        got = judge_reports(qid, tree.query, need, qdir / "judge" / f"a{attempt}")
        for t, sc in got.items():
            if sc.get("quality") is not None and sc.get("kpr") is not None:
                memo[t] = {"quality": sc["quality"], "kpr": sc["kpr"]}
                need.pop(t, None)
    memo_p.write_text(json.dumps(memo, indent=1))
    scores.update(memo)

    # pairwise vs depth 1 (both orders, memoized per question)
    h2h = {}
    for t in reports:
        if t == "d1__default" or not reports[t].strip():
            continue
        wr = pj.win_rates(qid, tree.query, {"d1__default": reports["d1__default"], t: reports[t]})
        h2h[t] = wr.get(t)
    words = {t: len(r.split()) for t, r in reports.items()}
    return qid, {"scores": scores, "h2h_vs_d1": h2h, "report_words": words}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    trees, _ = load_split(a.split)
    svc = RewardService(lam=0.0)
    pj = PairwiseJudge(memo_dir=OUT / "pairs", concurrency=2)
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=int(os.environ.get("REPLAY_Q_WORKERS", "3"))) as ex:
        res = dict(ex.map(lambda q: run_question(q, trees[q], svc, pj), sorted(trees)))
    (OUT / f"results_{a.split}.json").write_text(json.dumps(res, indent=1))

    tags = ["d1__default", "d2__default", "full__default", "d2__lf25k", "full__lf25k", "full__lf60k"]
    print(f"\n{a.split} ({len(res)} questions)   {'quality':>8} {'KPR':>6} {'win% vs d1':>11} {'n':>3}")
    for t in tags:
        q = [r["scores"][t]["quality"] for r in res.values() if r["scores"].get(t)]
        k = [r["scores"][t]["kpr"] for r in res.values() if r["scores"].get(t)]
        h = [r["h2h_vs_d1"][t] for r in res.values() if r["h2h_vs_d1"].get(t) is not None]
        hs = f"{100 * np.mean(h):10.1f}" if h else f"{'—':>10}"
        print(f"{t:18s} {np.mean(q):8.1f} {np.mean(k):6.1f} {hs} {len(q):3d}")


if __name__ == "__main__":
    main()
