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
  retriever        RETRIEVER (tavily | brave | custom | ...); ``custom`` +
                   env RETRIEVER_ENDPOINT targets ``adr serve-retriever`` for
                   BrowseComp-Plus, whose bcp:// hits are collected into
                   final_stats["retrieved_docids"]
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
from adr.eval.browsecomp_plus_prompts import QUERY_TEMPLATE_NO_GET_DOCUMENT
from adr.eval.repos import find_gpt_researcher
from adr.tools.browsecomp_plus import docids_from_urls

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

# Used only if the fork's write_report does not yet accept answer_format.
# Official openai_client.py default template; older generate_report then
# appends "Context: {context}" after this custom_prompt.
_LEGACY_BROWSECOMP_PROMPT = QUERY_TEMPLATE_NO_GET_DOCUMENT


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

    def _answer_format(self, task: ResearchTask) -> str:
        explicit = str(self.config.get("answer_format") or "").strip()
        if explicit:
            return explicit
        cfg_env = str((self.config.get("env") or {}).get("GR_ANSWER_FORMAT") or "").strip()
        if cfg_env:
            return cfg_env
        if task.query.dataset == "browsecomp_plus":
            return str(os.environ.get("GR_ANSWER_FORMAT") or "browsecomp").strip()
        return ""

    async def _write_report(self, researcher: Any, task: ResearchTask) -> str:
        """Prefer the fork's browsecomp answer_format; fall back for older checkouts."""
        fmt = self._answer_format(task)
        if fmt:
            os.environ["GR_ANSWER_FORMAT"] = fmt
        else:
            os.environ.pop("GR_ANSWER_FORMAT", None)
        try:
            if fmt:
                return await researcher.write_report(answer_format=fmt)
            return await researcher.write_report()
        except TypeError:
            if fmt:
                try:
                    return await researcher.write_report(
                        custom_prompt=_LEGACY_BROWSECOMP_PROMPT.format(
                            Question=task.query.text
                        )
                    )
                except TypeError:
                    pass
            return await researcher.write_report()

    def _modules(self) -> tuple[Any, Any, Any]:
        self._prepare_import()
        gr = importlib.import_module("gpt_researcher")
        tok = importlib.import_module("gpt_researcher.utils.token_tracker").TokenTracker
        lat = importlib.import_module("gpt_researcher.utils.latency_tracker").LatencyTracker
        return gr.GPTResearcher, tok, lat

    # ── run ───────────────────────────────────────────────────────
    async def run(self, task: ResearchTask, ctx: AgentContext) -> Trajectory:
        if (self.config.get("query_gate") or {}).get("enabled", False):
            if int(((ctx.extra or {}).get("config") or {}).get("concurrency", 1)) != 1:
                raise ValueError("Query gating requires harness concurrency=1 for global GR trackers")
        development = bool((self.config.get("development_trace") or {}).get("enabled", False))
        if development:
            if task.query.dataset != "development" or (self.config.get("query_gate") or {}).get("enabled", False):
                raise ValueError("Development trace requires a development query and disabled query gate")
            if int(((ctx.extra or {}).get("config") or {}).get("concurrency", 1)) != 1:
                raise ValueError("Development tracing requires harness concurrency=1 for global trackers")
        expansion_config = self.config.get("branch_expansion")
        if expansion_config is not None:
            from adr.orchestration.branch_budget import VERSION as expansion_version
            if expansion_config.get("policy") != expansion_version:
                raise ValueError("Unknown frozen branch-expansion policy")
            if development or self.config.get("redundancy_gate") is not None or (self.config.get("query_gate") or {}).get("enabled", False):
                raise ValueError("Branch expansion must not combine with query gating/development logging")
            if int(((ctx.extra or {}).get("config") or {}).get("concurrency", 1)) != 1:
                raise ValueError("Branch expansion requires harness concurrency=1")
        redundancy_config = self.config.get("redundancy_gate")
        if redundancy_config is not None:
            from adr.orchestration.redundancy import POLICIES
            if redundancy_config.get("policy") not in POLICIES:
                raise ValueError("Unknown frozen redundancy policy")
            if development or (self.config.get("query_gate") or {}).get("enabled", False):
                raise ValueError("Redundancy baseline cannot be combined with an MLP/development logger")
            if int(((ctx.extra or {}).get("config") or {}).get("concurrency", 1)) != 1:
                raise ValueError("Redundancy baseline requires harness concurrency=1")
        GPTResearcher, TokenTracker, LatencyTracker = self._modules()

        # gpt-researcher's trackers are process-global; the harness runs queries
        # under a semaphore, so reset here and read back before returning.
        TokenTracker.reset()
        LatencyTracker.reset()

        # Query.text stays the raw question (judge [question] + ground_truth).
        # Official openai_client wraps QUERY_TEMPLATE_NO_GET_DOCUMENT as the
        # user message; wrapping here would poison BM25, so the fork applies
        # that template only in write_report.
        researcher = GPTResearcher(query=task.query.text, report_type="deep")
        t0 = time.perf_counter()
        gate_config = self.config.get("query_gate") or {}
        if expansion_config is not None:
            from adr.orchestration.branch_budget import BranchTrace, BranchExpansion, VERSION as expansion_version
            hooks = importlib.import_module("gpt_researcher.utils.query_gate")
            expansion_hooks = importlib.import_module("gpt_researcher.utils.branch_expansion")
            if expansion_hooks.HOOK_VERSION != 1 or researcher.deep_researcher.concurrency_limit != 1:
                raise ValueError("Branch policy requires its hook and outer concurrency=1")
            if researcher.deep_researcher.depth != 3 or researcher.deep_researcher.breadth != 4:
                raise ValueError("Frozen branch policy requires depth=3 breadth=4")
            root = (ctx.extra or {}).get("run_dir")
            if not root:
                raise ValueError("Branch policy requires run_dir")
            import hashlib
            safe_id = hashlib.sha256(task.query.id.encode()).hexdigest()[:16]
            trace = BranchTrace(Path(root)/"branch_expansion"/f"{safe_id}.jsonl",
                task.query.text, ((ctx.extra or {}).get("config") or {}).get("budget", {}),
                TokenTracker, LatencyTracker, hooks.study_candidate_id,
                policy=expansion_version,evaluation=task.query.dataset != "development")
            expansion = BranchExpansion(trace)
            trace_token = hooks.active_trace.set(trace)
            expansion_token = expansion_hooks.active_expansion.set(expansion)
            status = "failed"
            try:
                await researcher.conduct_research()
                status = "research_completed"
            finally:
                expansion_hooks.active_expansion.reset(expansion_token)
                hooks.active_trace.reset(trace_token)
                trace.close(status)
                self._fill_meter(ctx, TokenTracker, LatencyTracker, n_fetches=trace.reads)
                meter = (ctx.extra or {}).get("meter")
                if meter is not None:
                    meter.n_search_calls = trace.searches
                ctx.extra["partial_research_accounting"] = trace.accounting()
        elif redundancy_config is not None:
            from adr.orchestration.live import PublicationTrace, RedundancyGate
            hooks = importlib.import_module("gpt_researcher.utils.query_gate")
            if hooks.HOOK_VERSION != 1 or researcher.deep_researcher.concurrency_limit != 1:
                raise ValueError("Redundancy gate requires v1 hooks and outer concurrency=1")
            if researcher.mcp_configs or researcher.vector_store or researcher.source_urls or any("mcpretriever" in r.__name__.lower() for r in researcher.retrievers):
                raise ValueError("Redundancy baseline supports the standard web research path only")
            root = (ctx.extra or {}).get("run_dir")
            if not root:
                raise ValueError("Redundancy baseline requires run_dir")
            import hashlib
            safe_id = hashlib.sha256(task.query.id.encode()).hexdigest()[:16]
            trace = PublicationTrace(Path(root)/"redundancy_gate"/f"{safe_id}.jsonl",
                task.query.text, ((ctx.extra or {}).get("config") or {}).get("budget", {}),
                TokenTracker, LatencyTracker, hooks.study_candidate_id,
                policy=redundancy_config["policy"], evaluation=task.query.dataset != "development")
            gate = RedundancyGate(trace, hooks, redundancy_config["policy"])
            trace_token = hooks.active_trace.set(trace)
            gate_token = hooks.active_gate.set(gate)
            status = "failed"
            try:
                await researcher.conduct_research()
                status = "research_completed"
            finally:
                hooks.active_gate.reset(gate_token)
                hooks.active_trace.reset(trace_token)
                trace.close(status)
                self._fill_meter(ctx, TokenTracker, LatencyTracker, n_fetches=trace.reads)
                meter = (ctx.extra or {}).get("meter")
                if meter is not None:
                    meter.n_search_calls = trace.searches
                ctx.extra["partial_research_accounting"] = trace.accounting()
        elif gate_config.get("enabled", False):
            # Lazy: the ordinary path imports neither Torch nor an embedding client.
            from adr.mlp.inference import FrozenMLP, DEFAULT_RUN
            from adr.mlp.query_gate import QueryGate
            hooks = importlib.import_module("gpt_researcher.utils.query_gate")
            if hooks.HOOK_VERSION != 1:
                raise RuntimeError("Incompatible GPT Researcher query-gate hooks")
            if researcher.deep_researcher.concurrency_limit != 1:
                raise ValueError("Query gating requires outer DEEP_RESEARCH_CONCURRENCY=1")
            if researcher.mcp_configs or researcher.vector_store or researcher.source_urls:
                raise ValueError("Query gating currently supports standard web research without MCP/vector-store/source-URL modes")
            if any("mcpretriever" in r.__name__.lower() for r in researcher.retrievers):
                raise ValueError("MCP query gating is unsupported")
            model = FrozenMLP(gate_config.get("run", DEFAULT_RUN))
            model.validate_endpoint()
            root = (ctx.extra or {}).get("run_dir")
            if not root:
                raise ValueError("Query gating requires a unique run_dir for audit logs")
            # Hash the identifier to avoid path traversal or secrets in filenames.
            import hashlib
            safe_id = hashlib.sha256(task.query.id.encode()).hexdigest()[:16]
            gate = QueryGate(model, researcher, task,
                             Path(root)/"query_gate"/f"{safe_id}.jsonl", TokenTracker, hooks,
                             enforce_budget=bool(((ctx.extra or {}).get("config", {}).get("budget") or {}).get("enforce", False)))
            token = hooks.active_gate.set(gate)
            try:
                await researcher.conduct_research()
            except Exception:
                self._fill_meter(ctx, TokenTracker, LatencyTracker, n_fetches=gate.reads)
                meter = (ctx.extra or {}).get("meter")
                if meter is not None:
                    meter.n_search_calls = gate.searches
                ctx.extra["partial_research_accounting"] = {
                    "research_usage": {k: TokenTracker.get_totals().get(k) for k in ("input_tokens", "output_tokens", "cost")},
                    "searches": gate.searches, "reads": gate.reads, "gate_decisions": gate.steps,
                    "scope": "partial research; excludes MLP embedding overhead",
                }
                raise
            finally:
                hooks.active_gate.reset(token)
                await gate.close()
        elif development:
            from adr.development.trace import DevelopmentTrace
            hooks = importlib.import_module("gpt_researcher.utils.development_trace")
            root = (ctx.extra or {}).get("run_dir")
            if not root:
                raise ValueError("Development trace requires run_dir")
            import hashlib
            safe_id = hashlib.sha256(task.query.id.encode()).hexdigest()[:16]
            trace = DevelopmentTrace(Path(root)/"development_trace"/f"{safe_id}.jsonl",
                                     task.query.text, ((ctx.extra or {}).get("config") or {}).get("budget", {}),
                                     TokenTracker, LatencyTracker, hooks.candidate_id,
                                     canonical_source=(lambda: researcher.deep_researcher.trajectory_logger._global_evidence)
                                     if self.config.get('development_trace', {}).get('exact_numerical_features', False) else None)
            trace.event("query_identity", query_id=task.query.id, dataset=task.query.dataset,
                        depth=researcher.deep_researcher.depth, breadth=researcher.deep_researcher.breadth,
                        concurrency=researcher.deep_researcher.concurrency_limit)
            token = hooks.active_trace.set(trace)
            status = "failed"
            try:
                await researcher.conduct_research()
                status = "research_completed"
            finally:
                hooks.active_trace.reset(token)
                trace.close(status)
                self._fill_meter(ctx, TokenTracker, LatencyTracker, n_fetches=trace.reads)
                meter = (ctx.extra or {}).get("meter")
                if meter is not None:
                    meter.n_search_calls = trace.searches
                if status == "failed":
                    ctx.extra["partial_research_accounting"] = trace.accounting()
        else:
            await researcher.conduct_research()
        report_text = await self._write_report(researcher, task)
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
        traj.final_stats["retrieved_docids"] = self._retrieved_docids(researcher, gtraj)
        self._fill_meter(
            ctx,
            TokenTracker,
            LatencyTracker,
            n_fetches=trace.reads if (development or redundancy_config is not None or expansion_config is not None) else len(getattr(researcher, "visited_urls", ()) or ()),
        )

        if expansion_config is not None:
            meter = (ctx.extra or {}).get("meter")
            if meter is not None:
                meter.n_search_calls = trace.searches
            traj.final_stats["branch_expansion"] = expansion.metrics()
            traj.final_stats["branch_expansion_audit"] = str(trace.path)
        if redundancy_config is not None:
            meter = (ctx.extra or {}).get("meter")
            if meter is not None:
                meter.n_search_calls = trace.searches
            traj.final_stats["redundancy_gate"] = gate.metrics()
            traj.final_stats["redundancy_audit"] = str(trace.path)
        if development:
            meter = (ctx.extra or {}).get("meter")
            if meter is not None:
                meter.n_search_calls = trace.searches
            traj.final_stats["development_only"] = True
            traj.final_stats["development_trace"] = str(trace.path)
        if self.config.get("keep_trajectory_files", True):
            self._copy_trajectory_files(ctx, gtraj.query_id, task.query.id)
        return traj

    # ── conversion ────────────────────────────────────────────────
    def _build_state(self, task: ResearchTask, gtraj: Any, report_text: str) -> ResearchState:
        state = ResearchState(task.query, task.budget)

        # Evidence id -> full EvidenceItem, used to look up content when
        # injecting items one round at a time.
        ev_lookup: dict[str, Any] = {}
        for iid, e in gtraj.evidence.items():
            ev_lookup[iid] = e

        # Each gpt-researcher round becomes one PRUNE step. Evidence and
        # frontier are injected *after* record_step so that stats_before
        # reflects the state prior to this round, not the end-of-run state.
        for snap in gtraj.rounds:
            rc = snap.round_cost
            kept = list(snap.decision.kept_item_ids) if snap.decision else []
            pruned = list(snap.decision.pruned_item_ids) if snap.decision else []
            kept_set = set(kept)

            # Record the step. compact_stats() runs here and captures the
            # evidence pool and frontier as they stood before this round.
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
                        {
                            "node_id": fn.node_id,
                            "status": fn.status,
                            "subquery": fn.subquery,
                            "parent_subquery": fn.parent_subquery,
                        }
                        for fn in snap.frontier
                    ],
                },
            )

            # Add this round's frontier nodes as subtasks.
            for fn in snap.frontier:
                if fn.node_id not in state.subtasks:
                    state.subtasks[fn.node_id] = Subtask(
                        id=fn.node_id,
                        goal=fn.subquery,
                        status="open",
                    )
                state.subtasks[fn.node_id].status = "done" if fn.status == "completed" else "active"

            # Add this round's new evidence. Items from earlier rounds are
            # already present; first insertion wins (preserves retrieval_round).
            for iid in snap.new_item_ids:
                if iid in state.evidence:
                    continue
                e = ev_lookup.get(iid)
                if e is None:
                    continue
                state.evidence[iid] = Evidence(
                    id=iid,
                    url=e.source_url,
                    title="",
                    snippet=e.content[:500],
                    text=e.content,
                    query=e.source_subquery,
                    subtask_id=None,
                    score=0.0,
                    retained=iid in kept_set,
                    added_step=int(e.retrieval_round),
                    source_backend="gpt_researcher",
                )

            # Apply this round's keep/prune decisions. A later round may
            # flip was_retained for an item first seen in an earlier round.
            for iid in kept:
                if iid in state.evidence:
                    state.evidence[iid].retained = True
            for iid in pruned:
                if iid in state.evidence:
                    state.evidence[iid].retained = False

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
    def _retrieved_docids(researcher: Any, gtraj: Any) -> list[str]:
        """Corpus docids the run retrieved, for BrowseComp-Plus Recall.

        Only ``bcp://<docid>`` URLs count, i.e. hits served by
        ``adr serve-retriever``; live-web runs yield []. Sources are unioned
        because each sees a different subset: the trajectory logger records
        every page that reached the embeddings filter (kept or pruned),
        ``research_sources`` is every retriever hit the deep-research rounds
        aggregated, and ``visited_urls`` covers anything scraped.
        """
        urls: list[Any] = [e.source_url for e in (getattr(gtraj, "evidence", None) or {}).values()]
        for src in getattr(researcher, "research_sources", None) or []:
            urls.append(src.get("url") if isinstance(src, dict) else src)
        urls.extend(getattr(researcher, "visited_urls", None) or ())
        return docids_from_urls(urls)

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
