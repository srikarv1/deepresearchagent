#!/usr/bin/env python3
"""Measure writer and RACE judge noise on a finished DeepResearch Bench run.

For every query of the run, the final evidence string (``final_context`` of the
fork trajectory) is rewritten ``--rewrites`` times with the run's own writer,
and each report is scored once by RACE; the original report is scored
``--rescores`` times. The pooled within-query standard deviations are the
per-report noise of an oracle replay (sigma_total) and of the judge alone
(sigma_judge).

    python scripts/measure_judge_noise.py runs/<drb-run> --queries 51,57,63,70,74 \\
        --rewrites 4 --rescores 3

Reports and RACE scores are cached under ``<run>/noise/`` (``--out``), so an
interrupted measurement resumes where it stopped. ``--dry-run`` prints the
plan and an Opus 4.6 cost estimate without calling any API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from adr.agents.gpt_researcher import GPTResearcherAgent
from adr.eval.deep_research_bench import _judge_key, _merge_judge_env
from adr.eval.judge_noise import (
    NoiseQuery,
    load_noise_queries,
    race_slot,
    rewrite_report,
    summarize,
)
from adr.eval.repos import find_deep_research_bench
from adr.runner.config import ROOT, load_yaml
from adr.runner.experiment import _eval_file

_log = logging.getLogger("measure_judge_noise")

# Claude Opus 4.6 list price, USD per million tokens (input, output).
_OPUS46_PRICE = (5.0, 25.0)
_EST_REPORT_OUTPUT_TOKENS = 6000


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="finished DRB run directory (runs/<id>)")
    p.add_argument("--queries", default="", help="comma-separated query ids (default: every query in the run)")
    p.add_argument("--rewrites", type=int, default=4, help="new reports per query (default 4)")
    p.add_argument("--rescores", type=int, default=3, help="RACE scores of the original report (default 3)")
    p.add_argument(
        "--agent-config",
        type=Path,
        default=None,
        help="agent YAML whose writer to use (default: agent.config of <run>/config.yaml)",
    )
    p.add_argument("--out", type=Path, default=None, help="output directory (default: <run>/noise)")
    p.add_argument("--writer-concurrency", type=int, default=2)
    p.add_argument("--n-train-queries", type=int, default=30, help="queries the labels are pooled over")
    p.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate only")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path


def _agent_config(run_dir: Path, override: Path | None) -> dict:
    if override is not None:
        path = _resolve(override)
    else:
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        agent = run_cfg.get("agent") or {}
        if agent.get("name") != "gpt_researcher" or not agent.get("config"):
            raise SystemExit(
                f"{run_dir}/config.yaml is not a gpt_researcher run with agent.config; pass --agent-config"
            )
        path = _resolve(Path(agent["config"]))
    if not path.exists():
        raise SystemExit(f"agent config {path} not found")
    return load_yaml(path)


def _estimate(queries: list[NoiseQuery], rewrites: int, rescores: int, cfg: dict) -> str:
    n_reports = len(queries) * rewrites
    in_tok = sum(len(q.final_context) // 4 for q in queries) * rewrites
    out_tok = n_reports * _EST_REPORT_OUTPUT_TOKENS
    writer = str((cfg.get("env") or {}).get("SMART_LLM") or os.environ.get("SMART_LLM") or "?")
    lines = [
        f"queries: {len(queries)} ({', '.join(q.qid for q in queries)})",
        f"writer: {writer}",
        f"rewrites: {n_reports} reports, ~{in_tok / 1e6:.2f}M input + ~{out_tok / 1e6:.2f}M output tokens",
        f"RACE calls: {len(queries) * (rewrites + rescores)} ({rewrites + rescores} slots)",
    ]
    if "claude-opus-4-6" in writer:
        usd = in_tok / 1e6 * _OPUS46_PRICE[0] + out_tok / 1e6 * _OPUS46_PRICE[1]
        lines.append(f"writer cost: ~${usd:.2f} at Opus 4.6 list price")
    return "\n".join(lines)


async def _rewrite_all(
    queries: list[NoiseQuery], rewrites: int, reports_dir: Path, agent_cfg: dict, concurrency: int
) -> tuple[dict[tuple[str, int], str], dict]:
    reports: dict[tuple[str, int], str] = {}
    todo: list[tuple[NoiseQuery, int]] = []
    for q in queries:
        for k in range(rewrites):
            path = reports_dir / f"{q.qid}_w{k}.md"
            if path.exists() and path.read_text(encoding="utf-8").strip():
                reports[(q.qid, k)] = path.read_text(encoding="utf-8")
            else:
                todo.append((q, k))
    if not todo:
        return reports, {}

    agent = GPTResearcherAgent(agent_cfg)
    GPTResearcher, TokenTracker, _ = agent._modules()
    TokenTracker.reset()
    sem = asyncio.Semaphore(max(concurrency, 1))

    async def _one(q: NoiseQuery, k: int) -> None:
        async with sem:
            text = await rewrite_report(GPTResearcher, q)
        if not text or not text.strip():
            _log.warning("query %s rewrite %d came back empty; skipped", q.qid, k)
            return
        (reports_dir / f"{q.qid}_w{k}.md").write_text(text, encoding="utf-8")
        reports[(q.qid, k)] = text
        _log.info("query %s rewrite %d: %d chars", q.qid, k, len(text))

    results = await asyncio.gather(*(_one(q, k) for q, k in todo), return_exceptions=True)
    for (q, k), res in zip(todo, results):
        if isinstance(res, BaseException):
            _log.error("query %s rewrite %d failed: %s", q.qid, k, res)
    return reports, dict(TokenTracker.get_totals())


def _score_slot(
    slot: str,
    rows: list[tuple[NoiseQuery, str]],
    race_dir: Path,
    bench_root: Path,
    env: dict[str, str],
    eval_cfg: dict,
) -> dict[str, float]:
    cache = race_dir / f"{slot}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if all(q.qid in cached for q, _ in rows):
            return {k: float(v) for k, v in cached.items()}
    _log.info("RACE %s: %d reports", slot, len(rows))
    scores = race_slot(
        slot,
        rows,
        bench_root=bench_root,
        env=env,
        language=str(eval_cfg.get("language") or "en"),
        workers=int(eval_cfg.get("workers") or 4),
        skip_cleaning=bool(eval_cfg.get("skip_cleaning", False)),
        timeout_s=eval_cfg.get("timeout_s"),
    )
    missing = [q.qid for q, _ in rows if q.qid not in scores]
    if missing:
        _log.warning("RACE %s returned no score for %s", slot, ", ".join(missing))
    cache.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores


def _print_summary(summary: dict) -> None:
    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    print()
    print(f"sigma_total  (rewrite + judge) = {f(summary['sigma_total'])}  (dof {summary['sigma_total_dof']})")
    print(f"sigma_judge  (judge only)      = {f(summary['sigma_judge'])}  (dof {summary['sigma_judge_dof']})")
    print(f"sigma_writer (derived)         = {f(summary['sigma_writer'])}")
    n = summary["n_train_queries"]
    for shape, se in summary["branch_effect_se"].items():
        print(f"{shape}: SE of one branch's depth effect = {f(se['per_query'])} per query, "
              f"{f(se['pooled'])} pooled over {n} queries")
    print()
    print(f"{'qid':>6}  {'writer scores':<40} {'judge scores':<30}")
    for qid, row in summary["per_query"].items():
        w = " ".join(f"{x:.3f}" for x in row["writer_scores"])
        j = " ".join(f"{x:.3f}" for x in row["judge_scores"])
        print(f"{qid:>6}  {w:<40} {j:<30}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _log.setLevel(logging.INFO)
    if args.rewrites < 1 or args.rescores < 2:
        raise SystemExit("need --rewrites >= 1 and --rescores >= 2")

    run_dir = args.run_dir.resolve()
    query_ids = [q.strip() for q in args.queries.split(",") if q.strip()] or None
    queries, skipped = load_noise_queries(run_dir, query_ids)
    for msg in skipped:
        _log.warning("skipped %s", msg)
    if not queries:
        raise SystemExit("no usable queries")

    agent_cfg = _agent_config(run_dir, args.agent_config)
    print(_estimate(queries, args.rewrites, args.rescores, agent_cfg))
    if args.dry_run:
        return 0

    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    eval_cfg = _eval_file(run_cfg, "deep_research_bench")
    located = find_deep_research_bench(eval_cfg.get("third_party_dir"))
    if not located.ok:
        raise SystemExit(located.reason)
    bench_root = located.path
    judge_env: dict[str, str] = {}
    effective_judge = _merge_judge_env(judge_env, eval_cfg.get("judge"))
    key_name, backend = _judge_key(judge_env)
    if not (os.environ.get(key_name) or judge_env.get(key_name)):
        raise SystemExit(f"{key_name} is not set (LLM_BACKEND={backend}); RACE cannot run")

    out = (args.out or run_dir / "noise").resolve()
    reports_dir = out / "reports"
    race_dir = out / "race"
    reports_dir.mkdir(parents=True, exist_ok=True)
    race_dir.mkdir(parents=True, exist_ok=True)

    reports, writer_tokens = asyncio.run(
        _rewrite_all(queries, args.rewrites, reports_dir, agent_cfg, args.writer_concurrency)
    )

    # Unique per run directory: RACE caches by model name under the bench repo.
    prefix = f"noise_{run_dir.name}".replace("-", "_")
    by_qid = {q.qid: q for q in queries}
    judge_scores: dict[str, list[float]] = {q.qid: [] for q in queries}
    writer_scores: dict[str, list[float]] = {q.qid: [] for q in queries}

    for j in range(args.rescores):
        scores = _score_slot(
            f"{prefix}_j{j}", [(q, q.report) for q in queries], race_dir, bench_root, judge_env, eval_cfg
        )
        for qid, s in scores.items():
            if qid in judge_scores:
                judge_scores[qid].append(s)
    # The original report with its first judge score is one more independent
    # (rewrite, judge) draw, so it joins the writer group.
    first = json.loads((race_dir / f"{prefix}_j0.json").read_text(encoding="utf-8"))
    for qid, s in first.items():
        if qid in writer_scores:
            writer_scores[qid].append(float(s))
    for k in range(args.rewrites):
        rows = [(by_qid[qid], text) for (qid, kk), text in sorted(reports.items()) if kk == k]
        if not rows:
            continue
        scores = _score_slot(f"{prefix}_w{k}", rows, race_dir, bench_root, judge_env, eval_cfg)
        for qid, s in scores.items():
            if qid in writer_scores:
                writer_scores[qid].append(s)

    summary = summarize(writer_scores, judge_scores, n_train_queries=args.n_train_queries)
    summary.update({
        "run_dir": str(run_dir),
        "queries": [q.qid for q in queries],
        "rewrites": args.rewrites,
        "rescores": args.rescores,
        "writer": {k: v for k, v in (agent_cfg.get("env") or {}).items() if k.endswith("_LLM")},
        "writer_tokens": writer_tokens,
        "judge": effective_judge,
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    print(f"\nwritten to {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
