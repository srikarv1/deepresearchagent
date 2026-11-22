#!/usr/bin/env python3
"""Exact-budget replay: one research run, every budget below it.

A run made with a generous budget (``BUDGET=200000 bash scripts/sol_budget.sh``)
records every sub-query, the tokens each one cost, every evidence item with its
embedding, what the method kept round by round, and the learnings. For a smaller
total budget B this script rebuilds what the same run would have handed the
writer had research stopped at B, then writes the report under a hard cap so
that research + report prompt + report output <= B exactly:

  1. Research prefix: the plan, then sub-queries in the order they ran, as long
     as research tokens + report prompt + REPORT_OUTPUT_TOKENS still fit in B.
  2. Writer context: the fork's ``build_final_context`` (the live code) over the
     items the method had retained after those sub-queries and the learnings
     they produced, trimmed only if it does not fit.
  3. Report: the fork's writer with max_tokens = what is left of B.

Every budget point of a query reuses the same research, so differences between
budgets come from the budget, not from new search results.

    python scripts/budget_replay.py check runs/gym_sol_budget200/extractive
    python scripts/budget_replay.py plan  runs/gym_sol_budget200/extractive --budgets 100000 150000
    python scripts/budget_replay.py write runs/gym_sol_budget200/extractive --budgets 100000 150000 \\
        --out runs/gym_sol_replay            # paid: one writer call per (query, budget)

``check`` (free) rebuilds each run's own writer context from the saved pieces at
its own budget and compares it with the context the run actually used; trust
``plan``/``write`` only for runs that match. ``plan`` (free) prints the cut
points. ``write`` saves each report as a run directory
``<out>_b<K>/<arm>/<query>-s0`` that ``scripts/sol_budget_judge.sh`` and
``scripts/budget_table.py`` read like any other run (set ``--out`` so that
``<out>_b150`` matches what the judge script expects, e.g. runs/gym_sol_budget
gives runs/gym_sol_budget_b150 → use BUDGET-style roots below).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPORT_OUTPUT_TOKENS = 12_000  # the live hard cap's report_max_output_tokens at 150K
LEARNING_RE = re.compile(r"\[Source: ([^\]]*)\]\s*$")


@dataclass
class Item:
    item_id: str
    source_url: str
    content: str
    retrieval_round: int
    tree_depth: int
    embedding: Any
    subquery: str


def _dr():
    import gpt_researcher.skills.deep_research as dr
    return dr


def load(run_dir: Path) -> dict:
    qid = run_dir.name.split("-s")[0]
    raw = json.loads((run_dir / "gpt_researcher" / f"{qid}.json").read_text())
    stats = json.loads((run_dir / "trajectories" / f"{qid}.json").read_text())["final_stats"]
    z = np.load(run_dir / "gpt_researcher" / f"{qid}_emb.npz", allow_pickle=False)
    vec = {str(i): z["vectors"][k].tolist() for k, i in enumerate(z["ids"])}
    items = {
        iid: Item(iid, e["source_url"], e["content"], int(e.get("retrieval_round") or 0),
                  int(e.get("tree_depth") or 0), vec.get(iid), e.get("source_subquery") or "")
        for iid, e in raw["evidence"].items()
    }
    return dict(qid=qid, raw=raw, stats=stats, items=items, query_emb=z["query"][0].tolist(),
                subq_emb=[v.tolist() for v in z["subquestions"]])


def split_learnings(final_context: str) -> list[str]:
    """The learnings at the head of the saved writer context (before the first
    'Source: ' item), one per line, as the run passed them to the writer."""
    head = final_context.split("\nSource: ", 1)[0] if not final_context.startswith("Source: ") else ""
    return [line for line in head.split("\n") if line.strip()]


def units(stats: dict) -> tuple[int, list[int], int, list[int]]:
    """(plan tokens, tokens of each sub-query in run order, report tokens,
    sub-queries per round) from the call log; a round starts at its
    deep_search_query_gen call."""
    tok = lambda c: c.get("input_tokens", 0) + c.get("output_tokens", 0)
    plan, out, unit, report, per_round = 0, [], 0, 0, []
    for c in stats["llm_calls"]:
        tag = c.get("usage_tag")
        if tag == "report_generation":
            report += tok(c)
        elif tag == "deep_research_plan":
            plan += tok(c)
        else:
            if tag == "deep_search_query_gen":
                per_round.append(0)
            unit += tok(c)
            if tag == "deep_process_results":
                out.append(unit)
                unit = 0
                if not per_round:
                    per_round.append(0)
                per_round[-1] += 1
    return plan, out, report, [n for n in per_round if n]


def subquery_order(raw: dict, items: dict[str, Item], per_round: list[int] | None = None) -> list[set[str]]:
    """The search queries of each research step, in run order. A step (one
    deep_process_results call) runs several search queries; within a round the
    search-query groups, in the order their evidence first appears, are split
    evenly over that round's steps (``per_round`` from the call log)."""
    steps: list[set[str]] = []
    for k, r in enumerate(raw["rounds"]):
        groups: list[str] = []
        for iid in r["new_item_ids"]:
            sq = items[iid].subquery if iid in items else ""
            if sq and sq not in groups:
                groups.append(sq)
        n = (per_round[k] if per_round and k < len(per_round) else len(groups)) or 1
        size = len(groups) / n
        steps += [set(groups[int(round(j * size)):int(round((j + 1) * size))]) for j in range(n)]
    return steps


def state_after(run: dict, n_units: int) -> tuple[list[Item], list[str], list[str]]:
    """Retained items and learnings had research stopped after ``n_units`` sub-queries."""
    raw, items = run["raw"], run["items"]
    order = subquery_order(raw, items, units(run["stats"])[3])
    kept_sq = set().union(*order[:n_units]) if n_units else set()
    retained: dict[str, Item] = {}
    for r in raw["rounds"]:
        new = [i for i in r["new_item_ids"] if i in items]
        if not any(items[i].subquery in kept_sq for i in new):
            break
        keep = set(r["retained_ids"])
        retained = {i: items[i] for i in raw["evidence"] if i in keep and (i not in new or items[i].subquery in kept_sq)}
    urls = {it.source_url for it in retained.values()} | {it.source_url for it in items.values() if it.subquery in kept_sq}
    learnings = [l for l in split_learnings(raw.get("final_context") or "")
                 if not LEARNING_RE.search(l) or LEARNING_RE.search(l).group(1) in urls]
    return list(retained.values()), learnings, order


def context_for(run: dict, retained: list[Item], learnings: list[str], max_words: int) -> list[str]:
    ctx, _ = _dr().build_final_context(learnings, retained, query_embedding=run["query_emb"],
                                       subquestion_embeddings=run["subq_emb"], max_words=max_words)
    return ctx


def plan_cut(run: dict, budget: int) -> dict:
    dr = _dr()
    plan, sq_tokens, _, per_round = units(run["stats"])
    fixed = dr.REPORT_PROMPT_FIXED_TOKENS + 2 * dr._count_tokens(run["raw"]["query"])
    # Research continues while the untrimmed writer context still fits (the live
    # rule with report_reserve auto); only when even the first step does not fit
    # is the context trimmed (the live hard cap).
    chosen = None
    for n in range(1, len(sq_tokens) + 1):
        research = plan + sum(sq_tokens[:n])
        retained, learnings, order = state_after(run, n)
        ctx = context_for(run, retained, learnings, dr.MAX_CONTEXT_WORDS)
        ctx_tokens = dr._count_tokens(dr._context_text(ctx))
        if research + fixed + ctx_tokens + REPORT_OUTPUT_TOKENS > budget:
            break
        chosen = (n, research, retained, learnings, order, ctx, ctx_tokens, False)
    if chosen is None:
        research = plan + sq_tokens[0] if sq_tokens else plan
        retained, learnings, order = state_after(run, 1)
        room = budget - research - fixed - REPORT_OUTPUT_TOKENS
        if room <= 0:
            return {}
        words = dr.MAX_CONTEXT_WORDS
        ctx = context_for(run, retained, learnings, words)
        ctx_tokens = dr._count_tokens(dr._context_text(ctx))
        while ctx_tokens > room and words > 0:
            words = int(words * room / ctx_tokens * 0.97)
            ctx = context_for(run, retained, learnings, words)
            ctx_tokens = dr._count_tokens(dr._context_text(ctx))
        chosen = (1, research, retained, learnings, order, ctx, ctx_tokens, True)
    n, research, retained, learnings, order, ctx, ctx_tokens, trimmed = chosen
    return dict(n_units=n, of_units=len(sq_tokens), research_tokens=research, context=ctx,
                context_tokens=ctx_tokens, trimmed=trimmed, fixed_tokens=fixed,
                max_output=budget - research - fixed - ctx_tokens, n_retained=len(retained),
                n_learnings=len(learnings), steps_match=len(order) == len(sq_tokens))


def check(run: dict) -> dict:
    dr = _dr()
    stop = run["stats"].get("token_stop") or {}
    words = stop.get("report_context_words") if stop.get("report_context_trimmed") else dr.MAX_CONTEXT_WORDS
    retained = [run["items"][i] for i in run["raw"]["evidence"] if run["raw"]["evidence"][i].get("was_retained")]
    learnings = split_learnings(run["raw"].get("final_context") or "")
    rebuilt = dr._context_text(context_for(run, retained, learnings, int(words)))
    saved = run["raw"].get("final_context") or ""
    _, sq, _, per_round = units(run["stats"])
    steps = subquery_order(run["raw"], run["items"], per_round)
    full = state_after(run, len(sq))
    same_full = {i.item_id for i in full[0]} == {i.item_id for i in retained}
    return dict(exact=rebuilt == saved, rebuilt_chars=len(rebuilt), saved_chars=len(saved),
                units=len(sq), subqueries=len(steps), per_round=per_round,
                rounds=len(run["raw"]["rounds"]), prefix_full_ok=same_full)


async def write_one(run_dir: Path, run: dict, cut: dict, out_dir: Path) -> dict:
    from gpt_researcher import GPTResearcher
    from gpt_researcher.utils.token_tracker import TokenTracker

    dr = _dr()
    before = (TokenTracker.total_input_tokens, TokenTracker.total_output_tokens)
    researcher = GPTResearcher(query=run["raw"]["query"], report_type="deep", verbose=False)
    researcher.context = dr._context_text(cut["context"])
    limit = int(getattr(researcher.cfg, "smart_token_limit", 0) or 0)  # as the live hard cap
    researcher.cfg.smart_token_limit = max(1, min(limit, int(cut["max_output"])) if limit else int(cut["max_output"]))
    report = await researcher.write_report()
    used_in = TokenTracker.total_input_tokens - before[0]
    used_out = TokenTracker.total_output_tokens - before[1]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(run_dir, out_dir, ignore=shutil.ignore_patterns("metrics", "judge.log", "exports"))
    traj_path = out_dir / "trajectories" / f"{run['qid']}.json"
    traj = json.loads(traj_path.read_text())
    total = cut["research_tokens"] + used_in + used_out
    from adr.eval.importers import _citations
    traj["report"] = {**(traj.get("report") or {}), "article": report, "citations": _citations(report)}
    traj["final_stats"]["budget_replay"] = {k: v for k, v in cut.items() if k != "context"} | dict(
        source_run=str(run_dir), report_input_tokens=used_in, report_output_tokens=used_out, final_tokens=total,
        report_max_output_tokens=researcher.cfg.smart_token_limit)
    traj_path.write_text(json.dumps(traj, ensure_ascii=False))
    summary_path = out_dir / "metrics" / "summary.json"  # budget_table reads tokens from here
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    src = json.loads((run_dir / "metrics" / "summary.json").read_text())
    summary_path.write_text(json.dumps({k: v for k, v in src.items() if k != "official"} | dict(
        mean_tokens=float(total), total_tokens=total, n_with_report=1), indent=2))
    return dict(query=run["qid"], total=total, research=cut["research_tokens"], report_in=used_in, report_out=used_out)


def env_exports(agent_config: Path) -> str:
    """Shell exports of the writer's models and final-context settings from an agent config."""
    import shlex
    import yaml
    cfg = yaml.safe_load(agent_config.read_text())
    lines = [f"export {k}={shlex.quote(str(v))}" for k, v in (cfg.get("env") or {}).items()]
    fc = (cfg.get("orchestration") or {}).get("final_context") or {}
    lines.append(f"export GR_FINAL_CONTEXT={fc.get('mode', 'ranked')} GR_FINAL_LEARNINGS_SHARE={fc.get('learnings_share', 0.2)} "
                 f"GR_FINAL_ITEM_MAX_WORDS={fc.get('item_max_words', 1500)}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "env":
        print(env_exports(Path(sys.argv[2])))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "plan", "write"))
    ap.add_argument("arm_dir", type=Path)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100_000, 150_000])
    ap.add_argument("--queries", help="comma-separated query ids (default: every run in arm_dir)")
    ap.add_argument("--out", type=Path, help="write: output root prefix; reports go to <out><K>/<arm>/")
    args = ap.parse_args()
    runs = sorted(d for d in args.arm_dir.glob("*-s*") if (d / "gpt_researcher").is_dir())
    if args.queries:
        wanted = set(args.queries.split(","))
        runs = [d for d in runs if d.name.split("-s")[0] in wanted]
    if not runs:
        sys.exit(f"{args.arm_dir}: no runs with a gpt_researcher trajectory")
    if args.mode == "check":
        ok = 0
        for d in runs:
            r = check(load(d))
            ok_steps = r["units"] == r["subqueries"] and len(r["per_round"]) == r["rounds"] and r["prefix_full_ok"]
            ok += ok_steps and r["exact"]
            print(f"{d.name}: context {'MATCH' if r['exact'] else 'DIFFERS'} ({r['rebuilt_chars']} vs {r['saved_chars']} chars); "
                  f"steps per round {r['per_round']} over {r['rounds']} rounds; all steps = saved retained set: {r['prefix_full_ok']}")
        print(f"{ok}/{len(runs)} runs rebuild their own writer context exactly")
        return
    if args.mode == "plan":
        for d in runs:
            run = load(d)
            for b in args.budgets:
                c = plan_cut(run, b)
                print(f"{d.name} B={b // 1000}K: {c.get('n_units')}/{c.get('of_units')} sub-queries, research "
                      f"{c.get('research_tokens')}, context {c.get('context_tokens')}{' (trimmed)' if c.get('trimmed') else ''}, "
                      f"report max {c.get('max_output')}, retained {c.get('n_retained')}")
        return
    if not args.out:
        sys.exit("write needs --out")
    import asyncio
    for b in args.budgets:
        for d in runs:
            run = load(d)
            cut = plan_cut(run, b)
            if not cut:
                print(f"{d.name} B={b}: nothing fits, skipped")
                continue
            dest = Path(f"{args.out}{b // 1000}") / args.arm_dir.name / d.name
            if (dest / "trajectories" / f"{run['qid']}.json").exists():
                print(f"{d.name} B={b}: done already")
                continue
            res = asyncio.run(write_one(d, run, cut, dest))
            print(f"{d.name} B={b // 1000}K: total {res['total']} (research {res['research']}, report "
                  f"{res['report_in']}+{res['report_out']})", flush=True)


if __name__ == "__main__":
    main()
