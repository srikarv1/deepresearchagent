"""gpt-researcher deep research mode as a harness agent.

Runs the instrumented gpt-researcher fork (WilliamOdinson/gpt-researcher, main)
in ``report_type="deep"`` and converts its trajectory into the harness
``Trajectory`` so the same exporters, judges, and ``adr compare`` apply.

gpt-researcher owns its own LLM and retrieval stack, so ``ctx.llm`` and
``ctx.search`` are not called. Cost is read from gpt-researcher's
``TokenTracker`` / ``LatencyTracker`` (API-returned usage, not tiktoken
estimates) and written into the harness ``CostMeter`` so local metrics work.
``final_stats.cost_source`` records this so it is not mistaken for
harness-metered numbers.

Config keys (configs/agents/gpt_researcher.yaml):

  repo_path        path to the fork; if null, resolved via ADR_GR_DIR,
                   third_party/gpt-researcher, or a sibling clone
  depth            DEEP_RESEARCH_DEPTH
  breadth          DEEP_RESEARCH_BREADTH
  concurrency      DEEP_RESEARCH_CONCURRENCY
  scraper          SCRAPER (bs | browser | tavily_extract | firecrawl)
  retriever        RETRIEVER (tavily | brave | ...)
  max_search_results_per_query
  env              extra env vars to set before import (dict)
  trajectory_dir   where gpt-researcher writes trajectory_*.json / _emb.npz
  keep_trajectory_files  copy gpt-researcher's files into the run dir
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from adr.agents.base import AgentContext, ResearchAgent
from adr.core.state import ResearchState
from adr.eval.repos import find_gpt_researcher
from adr.core.types import (
    ActionType,
    Evidence,
    OrchestratorAction,
    Report,
    ResearchTask,
    Subtask,
    TokenUsage,
    Trajectory,
)

_URL = re.compile(r"https?://[^\s\]\)>]+")

_ENV_MAP = {
    "depth": "DEEP_RESEARCH_DEPTH",
    "breadth": "DEEP_RESEARCH_BREADTH",
    "concurrency": "DEEP_RESEARCH_CONCURRENCY",
    "scraper": "SCRAPER",
    "retriever": "RETRIEVER",
    "max_search_results_per_query": "MAX_SEARCH_RESULTS_PER_QUERY",
    "max_scraper_workers": "MAX_SCRAPER_WORKERS",
}


class GPTResearcherAgent:
    name = "gpt_researcher"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._imported = False

    # ── import / env ──────────────────────────────────────────────
    def _prepare_import(self) -> None:
        if self._imported:
            return
        # Resolution: config repo_path -> ADR_GR_DIR -> third_party/ -> sibling
        # clone. If none resolve, fall through to whatever `gpt_researcher` is
        # importable in the env; the trajectory_logger check in run() catches
        # a plain upstream install.
        loc = find_gpt_researcher(self.config.get("repo_path"))
        if loc.ok:
            repo_path = str(loc.path)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)
        for key, env_name in _ENV_MAP.items():
            if key in self.config and self.config[key] is not None:
                os.environ[env_name] = str(self.config[key])
        for k, v in (self.config.get("env") or {}).items():
            os.environ[str(k)] = str(v)
        traj_dir = self.config.get("trajectory_dir")
        if traj_dir:
            os.environ["TRAJECTORY_OUTPUT_DIR"] = str(Path(traj_dir).expanduser().resolve())
        self._imported = True

    def _modules(self) -> tuple[Any, Any, Any]:
        self._prepare_import()
        gr = importlib.import_module("gpt_researcher")
        tok = importlib.import_module("gpt_researcher.utils.token_tracker").TokenTracker
        lat = importlib.import_module("gpt_researcher.utils.latency_tracker").LatencyTracker
        return gr.GPTResearcher, tok, lat

    # ── run ───────────────────────────────────────────────────────
    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        GPTResearcher, TokenTracker, LatencyTracker = self._modules()

        # gpt-researcher's trackers are process-global; the harness runs queries
        # under a semaphore, so reset here and read back before returning.
        TokenTracker.reset()
        LatencyTracker.reset()

        researcher = GPTResearcher(query=task.query.text, report_type="deep")
        t0 = time.perf_counter()
        await researcher.conduct_research()
        report_text = await researcher.write_report()
        wall = time.perf_counter() - t0

        deep = getattr(researcher, "deep_researcher", None)
        logger = getattr(deep, "trajectory_logger", None)
        if logger is None:
            raise RuntimeError(
                "gpt-researcher did not expose deep_researcher.trajectory_logger; "
                "is the instrumented fork on sys.path?"
            )
        gtraj = logger.trajectory

        state = self._build_state(task, gtraj, report_text)
        traj = state.trajectory()
        traj.final_stats.update(self._extra_stats(gtraj, wall))
        self._fill_meter(
            ctx,
            TokenTracker,
            LatencyTracker,
            n_fetches=len(getattr(researcher, "visited_urls", ()) or ()),
        )

        if self.config.get("keep_trajectory_files", True):
            self._copy_trajectory_files(ctx, gtraj.query_id, task.query.id)
        return traj

    # ── conversion ────────────────────────────────────────────────
    def _build_state(self, task: ResearchTask, gtraj: Any, report_text: str) -> ResearchState:
        state = ResearchState(task.query, task.budget)

        # Subtasks: one per frontier node ever seen. Frontier nodes are the
        # researchGoal strings of each round's sub-queries.
        seen_nodes: dict[str, Subtask] = {}
        for snap in gtraj.rounds:
            for fn in snap.frontier:
                if fn.node_id not in seen_nodes:
                    st = Subtask(id=fn.node_id, goal=fn.subquery, status="open")
                    seen_nodes[fn.node_id] = st
                    state.subtasks[st.id] = st

        # Evidence: bypass add_evidence() so max_evidence does not overwrite
        # gpt-researcher's own retention decisions.
        for iid, e in gtraj.evidence.items():
            ev = Evidence(
                id=iid,
                url=e.source_url,
                title="",
                snippet=e.content[:500],
                text=e.content,
                query=e.source_subquery,
                subtask_id=None,
                score=0.0,
                retained=bool(e.was_retained),
                added_step=int(e.retrieval_round),
                source_backend="gpt_researcher",
            )
            state.evidence[ev.id] = ev

        # One harness step per gpt-researcher round. Rounds are DFS-ordered
        # recursion levels; each is a PRUNE (the checkpoint's keep/prune) with
        # the layer's search + read + LLM cost attached.
        for snap in gtraj.rounds:
            rc = snap.round_cost
            kept = list(snap.decision.kept_item_ids) if snap.decision else []
            pruned = list(snap.decision.pruned_item_ids) if snap.decision else []
            for fn in snap.frontier:
                if fn.node_id in state.subtasks:
                    state.subtasks[fn.node_id].status = (
                        "done" if fn.status == "completed" else "active"
                    )
            state.record_step(
                OrchestratorAction(
                    type=ActionType.PRUNE,
                    evidence_ids=pruned,
                    rationale=f"round {snap.round_id}: EmbeddingsFilter kept {len(kept)}, pruned {len(pruned)}",
                ),
                observation=(
                    f"new={len(snap.new_item_ids)} retained_after={len(snap.retained_ids)} "
                    f"search_calls={rc.search_calls} llm_calls={rc.llm_calls}"
                ),
                tokens=TokenUsage(
                    prompt_tokens=rc.tokens_input,
                    completion_tokens=rc.tokens_output,
                    total_tokens=rc.tokens_input + rc.tokens_output,
                ),
                latency_s=float(rc.latency_seconds),
                extra={
                    "round_id": snap.round_id,
                    "decision_type": snap.decision.type if snap.decision else None,
                    "kept_item_ids": kept,
                    "pruned_item_ids": pruned,
                    "new_item_ids": list(snap.new_item_ids),
                    "n_retained_after": len(snap.retained_ids),
                    "search_calls": rc.search_calls,
                    "llm_calls": rc.llm_calls,
                    "frontier": [
                        {"node_id": fn.node_id, "status": fn.status} for fn in snap.frontier
                    ],
                },
            )

        sc = gtraj.synthesis_cost
        state.record_step(
            OrchestratorAction(
                type=ActionType.WRITE,
                report_draft=report_text,
                rationale="gpt-researcher write_report",
            ),
            observation=f"synthesis over {len(gtraj.final_context)} chars of context",
            tokens=TokenUsage(
                prompt_tokens=sc.tokens_input,
                completion_tokens=sc.tokens_output,
                total_tokens=sc.tokens_input + sc.tokens_output,
            ),
            latency_s=float(sc.latency_seconds),
            extra={"llm_calls": sc.llm_calls},
        )
        state.record_step(
            OrchestratorAction(type=ActionType.TERMINATE, rationale="deep research complete"),
            observation="stop",
        )
        # Citations are the URLs the report actually contains, matching
        # adr.eval.importers; state.citation_urls() would be every retained URL.
        state.report = Report(
            article=report_text, citations=list(dict.fromkeys(_URL.findall(report_text)))
        )
        return state

    @staticmethod
    def _extra_stats(gtraj: Any, wall: float) -> dict[str, Any]:
        return {
            "cost_source": "gpt_researcher.TokenTracker",
            "gpt_researcher_query_id": gtraj.query_id,
            "num_rounds": gtraj.num_rounds,
            "evidence_total": gtraj.evidence_total,
            "evidence_retained_final": gtraj.evidence_retained_final,
            "fraction_pruned": round(float(gtraj.fraction_pruned), 4),
            "peak_context_length": gtraj.peak_context_length,
            "total_cost_usd": round(float(gtraj.total_cost), 6),
            "synthesis_prompt_tokens": gtraj.synthesis_cost.tokens_input,
            "synthesis_completion_tokens": gtraj.synthesis_cost.tokens_output,
            "synthesis_latency_s": round(float(gtraj.synthesis_cost.latency_seconds), 4),
            "subquestions": list(gtraj.subquestions),
            "agent_wall_s": round(wall, 4),
        }

    @staticmethod
    def _fill_meter(
        ctx: AgentContext, TokenTracker: Any, LatencyTracker: Any, *, n_fetches: int = 0
    ) -> None:
        meter = (ctx.extra or {}).get("meter")
        if meter is None:
            return
        totals = TokenTracker.get_totals()
        meter.prompt_tokens = int(totals.get("input_tokens", 0))
        meter.completion_tokens = int(totals.get("output_tokens", 0))
        meter.total_tokens = meter.prompt_tokens + meter.completion_tokens
        per_type = LatencyTracker.per_type_latencies
        llm = per_type.get("llm", {})
        search = per_type.get("search", {})
        meter.n_calls = int(llm.get("count", 0))
        meter.llm_latency_s = float(llm.get("total_latency", 0.0))
        meter.n_search_calls = int(search.get("count", 0))
        meter.search_latency_s = float(search.get("total_latency", 0.0))
        # Scrapes are not in LatencyTracker; visited_urls is the set of pages fetched.
        meter.n_fetch_calls = int(n_fetches)

    def _copy_trajectory_files(
        self, ctx: AgentContext, gr_query_id: str, harness_query_id: str
    ) -> None:
        run_dir = (ctx.extra or {}).get("run_dir")
        src_dir = Path(os.environ.get("TRAJECTORY_OUTPUT_DIR", Path.cwd() / "trajectory_logs"))
        if not run_dir:
            return
        dest = Path(run_dir) / "gpt_researcher"
        dest.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", "_emb.npz"):
            src = src_dir / f"trajectory_{gr_query_id}{suffix}"
            if src.exists():
                shutil.copy2(src, dest / f"{harness_query_id}{suffix}")


def build(config: dict | None = None) -> ResearchAgent:
    return GPTResearcherAgent(config)
