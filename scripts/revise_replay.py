"""Research less, write better: move tokens from research to a revision pass.

From the same no-pruning research (a 200K run, scripts/sol_budget.sh), three
variants of the report, all written by the same writer model:

  r1        research stopped after the first round, report written as usual
  r1rev     the r1 report, then one revision pass against the question's sub-questions
  fullrev   the 200K run's own report, then the same revision pass
  b100rev   the existing exact-100K report (scripts/budget_replay.py), then the
            revision pass: ~130K in total, matched to the exact-150K report (~134K)
  b100polish  the same 100K report, revised for clarity only (control: no sub-question check)

r1 and r1rev share the draft, and fullrev shares the 200K report, so each
difference measures one change. The revision pass sees the question, its
sub-questions, the writer's own evidence context and the draft. It is told to
answer every sub-question from that evidence only, keep the citation format,
and return the full report.

    python scripts/revise_replay.py plan  runs/gym_sol_budget200/none          # free
    python scripts/revise_replay.py write runs/gym_sol_budget200/none --out runs/gym_sol   # paid
    # -> runs/gym_sol_r1/none, runs/gym_sol_r1rev/none, runs/gym_sol_fullrev/none

Every output is a run directory that scripts/sol_budget_judge.sh (RUN_ROOT=...)
and scripts/budget_table.py read like any other; final_stats.budget_replay
carries the token accounting (final_tokens = research + report + revision).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget_replay as br  # noqa: E402

REVISE_MAX_OUTPUT = 12_000

SYSTEM = "You are an expert research editor. You improve research reports without inventing facts or sources."

PROMPT = """Research question:
{query}

The report must answer these sub-questions:
{subqs}

Evidence the report was written from (the only sources you may use):
{context}

Draft report:
{draft}

Revise the draft.
1. For each sub-question, check whether the draft answers it with specific facts supported by cited evidence.
2. Where an answer is missing, vague or unsupported, add specific facts from the evidence above, cited in the same markdown link format the draft uses, e.g. [Source name, year](url).
3. Do not add any fact or source that is not in the evidence. Keep what is already good, the structure and every existing correct citation.
4. Keep the length similar; at most about 30% longer.

Return only the full revised report in markdown, with no comments about the changes."""

# Control: same inputs and length limit, but no sub-question check, so a gain
# over it comes from the targeting rather than from a second pass as such.
POLISH = """Research question:
{query}

Evidence the report was written from (the only sources you may use):
{context}

Draft report:
{draft}

Revise the draft for clarity, flow and organization.
1. Improve wording, transitions and structure where it helps the reader.
2. Do not add any fact or source that is not in the evidence. Keep every existing correct citation in the same markdown link format.
3. Keep the length similar; at most about 30% longer.

Return only the full revised report in markdown, with no comments about the changes."""


def round1_cut(run: dict) -> dict:
    """The writer input had research stopped after the first round."""
    dr = br._dr()
    plan, sq_tokens, _, per_round = br.units(run["stats"])
    n = per_round[0]
    retained, learnings, order = br.state_after(run, n)
    ctx = br.context_for(run, retained, learnings, dr.MAX_CONTEXT_WORDS)
    ctx_tokens = dr._count_tokens(dr._context_text(ctx))
    fixed = dr.REPORT_PROMPT_FIXED_TOKENS + 2 * dr._count_tokens(run["raw"]["query"])
    research = plan + sum(sq_tokens[:n])
    return dict(n_units=n, of_units=len(sq_tokens), research_tokens=research, context=ctx,
                context_tokens=ctx_tokens, trimmed=False, fixed_tokens=fixed,
                max_output=10 ** 9, n_retained=len(retained), n_learnings=len(learnings),
                steps_match=len(order) == len(sq_tokens), cut="round1")


async def revise(out_dir: Path, qid: str, context_text: str, prompt: str = PROMPT) -> dict:
    """Run the revision pass on the report saved in out_dir, in place."""
    from gpt_researcher.config import Config
    from gpt_researcher.utils.llm import create_chat_completion
    from gpt_researcher.utils.token_tracker import TokenTracker
    from adr.eval.importers import _citations

    traj_path = out_dir / "trajectories" / f"{qid}.json"
    traj = json.loads(traj_path.read_text())
    draft = (traj.get("report") or {}).get("article") or ""
    if not draft.strip():
        raise ValueError(f"{out_dir}: empty draft")
    subqs = traj["final_stats"].get("subquestions") or []
    q = traj.get("query") or {}
    query = q.get("text", "") if isinstance(q, dict) else str(q)
    cfg = Config()
    content = prompt.format(query=query, subqs="\n".join(f"- {s}" for s in subqs),
                            context=context_text, draft=draft)
    before = (TokenTracker.total_input_tokens, TokenTracker.total_output_tokens)
    t0 = time.perf_counter()
    revised = await create_chat_completion(
        model=cfg.smart_llm_model, llm_provider=cfg.smart_llm_provider,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
        temperature=0.25, max_tokens=REVISE_MAX_OUTPUT, llm_kwargs=cfg.llm_kwargs,  # temperature as the report writer
        usage_tag="report_revision")
    secs = time.perf_counter() - t0
    rin = TokenTracker.total_input_tokens - before[0]
    rout = TokenTracker.total_output_tokens - before[1]
    if not (revised or "").strip():
        raise ValueError(f"{out_dir}: empty revision")
    traj["report"] = {**traj["report"], "article": revised, "citations": _citations(revised), "draft_article": draft}
    rep = traj["final_stats"].setdefault("budget_replay", {})
    base = rep.get("final_tokens") or (traj["final_stats"].get("token_stop") or {}).get("final_tokens")
    rep.update(revise_input_tokens=rin, revise_output_tokens=rout, revise_s=round(secs, 2),
               draft_tokens=base, final_tokens=base + rin + rout)
    traj_path.write_text(json.dumps(traj, ensure_ascii=False))
    summary_path = out_dir / "metrics" / "summary.json"
    s = json.loads(summary_path.read_text())
    s.pop("official", None)
    s.pop("scores", None)
    s.update(mean_tokens=float(base + rin + rout), total_tokens=base + rin + rout)
    summary_path.write_text(json.dumps(s, indent=2))
    return dict(draft=base, revise_in=rin, revise_out=rout, total=base + rin + rout)


def copy_run(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("metrics", "judge.log", "exports"))
    (dest / "metrics").mkdir(exist_ok=True)
    s = json.loads((src / "metrics" / "summary.json").read_text())
    (dest / "metrics" / "summary.json").write_text(json.dumps({k: v for k, v in s.items() if k not in ("official", "scores")}, indent=2))


def done(d: Path, qid: str, revised: bool) -> bool:
    p = d / "trajectories" / f"{qid}.json"
    if not p.exists():
        return False
    rep = json.loads(p.read_text())["final_stats"].get("budget_replay") or {}
    return ("revise_input_tokens" in rep) if revised else bool(rep.get("final_tokens"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("plan", "write"))
    ap.add_argument("arm_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write: output prefix; roots <out>_r1, <out>_r1rev, <out>_fullrev")
    ap.add_argument("--queries", help="comma-separated query ids")
    ap.add_argument("--variants", nargs="+", default=["r1", "r1rev", "fullrev"],
                    help="r1 r1rev fullrev, and b100rev b100polish: revise the existing "
                         "<out>_replay100 reports (targeted / polish control)")
    args = ap.parse_args()
    runs = sorted(d for d in args.arm_dir.glob("*-s*") if (d / "gpt_researcher").is_dir())
    if args.queries:
        runs = [d for d in runs if d.name.split("-s")[0] in set(args.queries.split(","))]
    if not runs:
        sys.exit(f"{args.arm_dir}: no runs")
    dr = br._dr()
    if args.mode == "plan":
        for d in runs:
            run = br.load(d)
            c = round1_cut(run)
            full = (run["stats"].get("token_stop") or {}).get("final_tokens")
            print(f"{d.name}: round 1 = {c['n_units']}/{c['of_units']} sub-queries, research {c['research_tokens']}, "
                  f"writer context {c['context_tokens']} tokens; full run {full}")
        return
    if not args.out:
        sys.exit("write needs --out")
    arm = args.arm_dir.name
    for d in runs:
        run = br.load(d)
        qid = run["qid"]
        r1 = Path(f"{args.out}_r1") / arm / d.name
        cut = round1_cut(run)
        ctx_text = dr._context_text(cut["context"])
        if {"r1", "r1rev"} & set(args.variants) and not done(r1, qid, False):
            res = asyncio.run(br.write_one(d, run, cut, r1))
            print(f"{d.name} r1: total {res['total']} (research {res['research']}, report {res['report_in']}+{res['report_out']})", flush=True)
        if "r1rev" in args.variants:
            dest = Path(f"{args.out}_r1rev") / arm / d.name
            if not done(dest, qid, True):
                copy_run(r1, dest)
                res = asyncio.run(revise(dest, qid, ctx_text))
                print(f"{d.name} r1rev: total {res['total']} (draft {res['draft']}, revision {res['revise_in']}+{res['revise_out']})", flush=True)
        for v, prompt in (("b100rev", PROMPT), ("b100polish", POLISH)):
            if v not in args.variants:
                continue
            src100 = Path(f"{args.out}_replay100") / arm / d.name
            dest = Path(f"{args.out}_{v}") / arm / d.name
            if not done(dest, qid, True):
                cut = br.plan_cut(run, 100_000)  # the writer context the 100K report was written from
                copy_run(src100, dest)
                res = asyncio.run(revise(dest, qid, dr._context_text(cut["context"]), prompt))
                print(f"{d.name} {v}: total {res['total']} (100K report {res['draft']}, revision {res['revise_in']}+{res['revise_out']})", flush=True)
        if "fullrev" in args.variants:
            dest = Path(f"{args.out}_fullrev") / arm / d.name
            if not done(dest, qid, True):
                copy_run(d, dest)
                res = asyncio.run(revise(dest, qid, run["raw"].get("final_context") or ""))
                print(f"{d.name} fullrev: total {res['total']} (report {res['draft']}, revision {res['revise_in']}+{res['revise_out']})", flush=True)


if __name__ == "__main__":
    main()
