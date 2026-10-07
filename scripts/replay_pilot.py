"""Branch-replay pilot for one DRG query.

Given a run made with GR_NODE_CACHE=1 GR_BRANCH_INDEPENDENT=1, this
  1. rebuilds the report from every coalition of top-level branches (2^k),
     from depth-truncated trees, and from the full tree 3x (judge + writer noise);
  2. judges all of them, plus the live report, in one batched DRG judge call;
  3. prints payoffs, exact Shapley / Banzhaf values per branch, and replay fidelity.

    set -a; source .env; set +a
    python scripts/replay_pilot.py runs/<run_dir> <query_id>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from adr.replay.judge import judge_reports
from adr.replay.shapley import all_coalitions, banzhaf, shapley
from adr.replay.synth import apply_agent_env, synthesize_many
from adr.replay.tree import ResearchTree


def main(run_dir: str, qid: str) -> None:
    run = Path(run_dir)
    apply_agent_env()
    tree = ResearchTree.from_trajectory(run / "gpt_researcher" / f"{qid}.json")
    query = json.loads((run / "queries" / f"{qid}.json").read_text())["text"]
    players = list(tree.roots)
    label = {p: f"B{i + 1}" for i, p in enumerate(players)}
    print(f"query {qid}: {len(tree.nodes)} nodes, {len(players)} top-level branches")
    for p in players:
        print(f"  {label[p]}: {len(tree.subtree(p))} nodes, {tree.tokens(tree.subtree(p)) / 1e3:.0f}K tok | {tree.nodes[p].research_goal[:90]}")

    def nodes_of(coal: frozenset) -> set[str]:
        return {n for p in coal for n in tree.subtree(p)}

    def tag(coal: frozenset) -> str:
        return "C_" + ("".join("1" if p in coal else "0" for p in players))

    coals = all_coalitions(players)
    contexts = {tag(c): tree.context(nodes_of(c)) for c in coals}
    for d in (1, 2):
        contexts[f"D{d}"] = tree.context(tree.up_to_depth(d))
    full = tree.all_nodes()
    for r in (2, 3):
        contexts[f"FULL_r{r}"] = tree.context(full)  # C_1111 is replicate 1

    work = run / "replay" / qid
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "reports.json"
    reports = json.loads(cache.read_text()) if cache.exists() else {}
    todo = {t: c for t, c in contexts.items() if t not in reports}
    print(f"synthesizing {len(todo)} reports ({len(reports)} cached)...", flush=True)
    reports.update(asyncio.run(synthesize_many(query, todo)))
    cache.write_text(json.dumps(reports, indent=1, ensure_ascii=False))

    live = (run / "reports" / f"{qid}.md").read_text(encoding="utf-8")
    all_reports = {**reports, "LIVE": live}
    score_file = work / "scores.json"
    scores = json.loads(score_file.read_text()) if score_file.exists() else {}

    def missing() -> dict[str, str]:
        return {t: r for t, r in all_reports.items()
                if r.strip() and (scores.get(t, {}).get("kpr") is None or scores.get(t, {}).get("quality") is None)}

    # Judge only what is missing; rate-limit failures (429) leave gaps, so retry
    # those into a fresh folder and merge instead of re-judging everything.
    for attempt in range(4):
        todo_j = missing()
        if not todo_j:
            break
        print(f"judging {len(todo_j)} reports (pass {attempt + 1})", flush=True)
        got = judge_reports(qid, query, todo_j, work / f"judge_pass{attempt + 1}")
        for t, sc in got.items():
            old = scores.get(t, {})
            scores[t] = {k: (sc.get(k) if sc.get(k) is not None else old.get(k)) for k in ("quality", "kpr", "criteria")}
        score_file.write_text(json.dumps(scores, indent=1))
    if missing():
        raise SystemExit(f"still unjudged after retries: {sorted(missing())}")
    for t, r in all_reports.items():  # no research -> no report -> payoff 0
        if not r.strip():
            scores[t] = {"quality": 0.0, "kpr": 0.0, "criteria": {}}

    tok = {tag(c): tree.tokens(nodes_of(c)) for c in coals}
    tok.update({"D1": tree.tokens(tree.up_to_depth(1)), "D2": tree.tokens(tree.up_to_depth(2))})
    print(f"\n{'report':10s} {'tokens':>8s} {'quality':>8s} {'KPR':>6s}")
    for t in sorted(scores, key=lambda x: (x[0], x)):
        s = scores[t]
        print(f"{t:10s} {tok.get(t, tok[tag(frozenset(players))]) / 1e3:7.0f}K {s['quality'] or float('nan'):8.1f} {s['kpr'] if s['kpr'] is not None else float('nan'):6.1f}")

    for metric in ("kpr", "quality"):
        v = {c: float(scores[tag(c)][metric]) for c in coals}  # all present (checked above)
        phi, beta = shapley(players, v), banzhaf(players, v)
        print(f"\n{metric}: v(empty)={v[frozenset()]:.1f}  v(all)={v[frozenset(players)]:.1f}")
        for p in players:
            print(f"  {label[p]}  Shapley={phi[p]:+6.2f}  Banzhaf={beta[p]:+6.2f}  cost={tree.tokens(tree.subtree(p)) / 1e3:.0f}K")
    reps = [scores[t]["kpr"] for t in (tag(frozenset(players)), "FULL_r2", "FULL_r3") if scores.get(t, {}).get("kpr") is not None]
    print(f"\nnoise: full-tree KPR replicates {reps}; LIVE KPR {scores['LIVE']['kpr']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
