"""Cost must be measured by the harness, not trusted from the agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adr.core.instrument import BudgetExceeded, CostMeter, MeteredLLM, MeteredSearch
from adr.core.types import ActionType, Budget, OrchestratorAction, Report, Trajectory
from adr.core.state import ResearchState
from adr.llm.mock import MockLLM
from adr.runner.config import load_config
from adr.runner.experiment import run_experiment
from adr.tools.search import MockSearch


class SilentAgent:
    """Spends budget and reports none of it, like a buggy or adversarial agent."""

    name = "silent"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    async def run(self, task, ctx) -> Trajectory:
        state = ResearchState(task.query, task.budget)
        state.add_subtasks([task.query.text])
        for _ in range(3):
            await ctx.search.search(task.query.text, k=2)
            await ctx.llm.complete([{"role": "user", "content": "x " * 50}])
        state.report = Report(article="# Report\n\nhttps://example.com/a\n", citations=[])
        # Deliberately records no tokens and no latency.
        state.record_step(OrchestratorAction(type=ActionType.WRITE, rationale="silent"))
        return state.trajectory()


async def test_metered_llm_counts_unreported_tokens():
    meter = CostMeter()
    budget = Budget(max_tokens=10_000)
    llm = MeteredLLM(MockLLM(replies=["a b c"]), meter, budget=budget)

    await llm.complete([{"role": "user", "content": "one two three four"}])

    assert meter.n_calls == 1
    assert meter.total_tokens == 4 + 3
    assert budget.used_tokens == 7
    assert meter.llm_latency_s >= 0.0
    assert llm.model == "mock"


async def test_metered_search_counts_calls_and_charges_budget(mock_search: MockSearch):
    meter = CostMeter()
    budget = Budget(max_searches=5, max_reads=5)
    search = MeteredSearch(mock_search, meter, budget=budget)

    hits = await search.search("chip shortage", k=2)
    await search.fetch(hits[0].url)

    assert meter.n_search_calls == 1
    assert meter.n_fetch_calls == 1
    assert meter.retrieved_chars > 0
    assert budget.used_searches == 1
    assert budget.used_reads == 1
    assert search.name == "mock"


async def test_budget_overrun_is_recorded_but_not_fatal_by_default():
    meter = CostMeter()
    budget = Budget(max_tokens=5)
    llm = MeteredLLM(MockLLM(replies=["a b c d e f"]), meter, budget=budget)

    await llm.complete([{"role": "user", "content": "one two three"}])
    await llm.complete([{"role": "user", "content": "four five six"}])

    assert meter.violations, "overrun should be recorded"
    assert any("token budget" in v for v in meter.violations)


async def test_budget_overrun_raises_when_enforced():
    meter = CostMeter()
    budget = Budget(max_tokens=1)
    llm = MeteredLLM(MockLLM(replies=["a b c"]), meter, budget=budget, enforce=True)

    await llm.complete([{"role": "user", "content": "one two"}])
    with pytest.raises(BudgetExceeded):
        await llm.complete([{"role": "user", "content": "three four"}])


async def test_search_budget_enforced(mock_search: MockSearch):
    meter = CostMeter()
    budget = Budget(max_searches=1)
    search = MeteredSearch(mock_search, meter, budget=budget, enforce=True)

    await search.search("chip shortage", k=1)
    with pytest.raises(BudgetExceeded):
        await search.search("chip shortage", k=1)


def test_runner_reports_cost_the_agent_never_logged(tmp_path: Path, monkeypatch):
    from adr.agents import registry

    monkeypatch.setitem(registry._BUILDERS, "silent", lambda cfg=None: SilentAgent(cfg))
    cfg = load_config(
        "configs/default.yaml",
        {
            "output_dir": str(tmp_path),
            "run_name": "silent",
            "concurrency": 1,
            "dataset": {"name": "deep_research_bench", "query_ids": ["51"], "limit": 1},
            "agent": {"name": "silent"},
            "llm": {"provider": "mock"},
            "search": {"backend": "mock"},
        },
    )
    manifest = run_experiment(cfg)
    summary = json.loads((manifest.run_dir / "metrics" / "summary.json").read_text())

    # The agent logged one step with no tokens; the harness still sees the truth.
    assert summary["mean_n_steps"] == 1
    assert summary["mean_n_llm_calls"] == 3
    assert summary["mean_tokens"] > 0
    assert summary["mean_wall_s"] > 0
    assert summary["total_tokens"] > 0

    per_query = json.loads((manifest.run_dir / "metrics" / "local.json").read_text())["per_query"]
    assert per_query[0]["n_searches"] == 3


def test_fixture_run_records_model_usage(tmp_path: Path):
    cfg = load_config(
        "configs/default.yaml",
        {
            "output_dir": str(tmp_path),
            "run_name": "metered",
            "concurrency": 1,
            "dataset": {"name": "deep_research_bench", "query_ids": ["51"], "limit": 1},
            "agent": {"name": "fixture"},
        },
    )
    manifest = run_experiment(cfg)
    summary = json.loads((manifest.run_dir / "metrics" / "summary.json").read_text())
    assert summary["mean_n_llm_calls"] == 1
    assert summary["mean_tokens"] > 0
    assert summary["mean_prompt_tokens"] > 0
    assert summary["mean_wall_s"] > 0
