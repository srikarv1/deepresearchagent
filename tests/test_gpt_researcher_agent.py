"""Exercise the gpt-researcher adapter without the real package.

A fake ``gpt_researcher`` module is injected into ``sys.modules`` that mimics
the instrumented fork's surface: ``GPTResearcher`` with ``conduct_research`` /
``write_report``, a ``deep_researcher.trajectory_logger.trajectory`` object,
and the process-global ``TokenTracker`` / ``LatencyTracker``.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from adr.agents.base import AgentContext
from adr.core.instrument import CostMeter
from adr.core.types import Budget, ResearchTask
from adr.eval.local_metrics import compute_local_metrics


# ── fake gpt_researcher surface ───────────────────────────────────────
@dataclass
class _RoundCost:
    tokens_input: int = 0
    tokens_output: int = 0
    latency_seconds: float = 0.0
    llm_calls: int = 0
    search_calls: int = 0


@dataclass
class _Decision:
    type: str
    kept_item_ids: list[str] = field(default_factory=list)
    pruned_item_ids: list[str] = field(default_factory=list)


@dataclass
class _Frontier:
    node_id: str
    subquery: str
    parent_subquery: str
    status: str


@dataclass
class _Snap:
    round_id: int
    new_item_ids: list[str]
    retained_ids: list[str]
    decision: _Decision
    frontier: list[_Frontier]
    round_cost: _RoundCost


@dataclass
class _Ev:
    content: str
    source_url: str
    source_subquery: str
    tree_depth: int
    retrieval_round: int
    was_retained: bool


@dataclass
class _Traj:
    query_id: str = "abc123def456"
    subquestions: list[str] = field(default_factory=lambda: ["s1", "s2"])
    num_rounds: int = 2
    rounds: list[_Snap] = field(default_factory=list)
    evidence: dict[str, _Ev] = field(default_factory=dict)
    final_context: str = "ctx " * 100
    report: str = ""
    synthesis_cost: _RoundCost = field(default_factory=_RoundCost)
    total_cost: float = 0.0
    evidence_total: int = 0
    evidence_retained_final: int = 0
    fraction_pruned: float = 0.0
    peak_context_length: int = 0


def _make_traj() -> _Traj:
    t = _Traj()
    t.evidence = {
        "e1": _Ev("page one text", "https://a.example/1", "sq1", 1, 1, True),
        "e2": _Ev("page two text", "https://b.example/2", "sq1", 1, 1, False),
        "e3": _Ev("page three text", "https://c.example/3", "sq2", 2, 2, True),
    }
    t.rounds = [
        _Snap(
            1, ["e1", "e2"], ["e1"],
            _Decision("continue", ["e1"], ["e2"]),
            [_Frontier("n1", "goal one", "root", "open"), _Frontier("n2", "goal two", "root", "open")],
            _RoundCost(1000, 100, 10.0, 3, 2),
        ),
        _Snap(
            2, ["e3"], ["e1", "e3"],
            _Decision("terminate", ["e3"], []),
            [_Frontier("n3", "goal three", "goal one", "completed")],
            _RoundCost(800, 80, 8.0, 2, 1),
        ),
    ]
    t.synthesis_cost = _RoundCost(5000, 900, 20.0, 1, 0)
    t.total_cost = 0.0421
    t.evidence_total = 3
    t.evidence_retained_final = 2
    t.fraction_pruned = 1 / 3
    t.peak_context_length = 5000
    return t


class _TokenTracker:
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    peak_input_tokens = 0

    @staticmethod
    def reset():
        _TokenTracker.total_input_tokens = 0
        _TokenTracker.total_output_tokens = 0
        _TokenTracker.total_cost = 0.0
        _TokenTracker.peak_input_tokens = 0

    @staticmethod
    def get_totals():
        return {
            "input_tokens": _TokenTracker.total_input_tokens,
            "output_tokens": _TokenTracker.total_output_tokens,
            "cost": _TokenTracker.total_cost,
            "peak_input_tokens": _TokenTracker.peak_input_tokens,
        }


class _LatencyTracker:
    per_type_latencies: dict = {}
    per_source_latencies: dict = {}

    @staticmethod
    def reset():
        _LatencyTracker.per_type_latencies.clear()
        _LatencyTracker.per_source_latencies.clear()

    @staticmethod
    def snapshot():
        return {k: v["count"] for k, v in _LatencyTracker.per_type_latencies.items()}


class _Logger:
    def __init__(self, traj: _Traj):
        self.trajectory = traj


class _Deep:
    def __init__(self, traj: _Traj):
        self.trajectory_logger = _Logger(traj)


class _FakeGPTResearcher:
    """Simulates the fork: conduct_research fills trackers, write_report adds synthesis."""

    def __init__(self, query: str, report_type: str = "research_report", **_: object):
        assert report_type == "deep"
        self.query = query
        self._traj = _make_traj()
        self.deep_researcher = _Deep(self._traj)
        self.visited_urls = {"https://a.example/1", "https://b.example/2", "https://c.example/3", "https://d.example/4"}

    async def conduct_research(self):
        for snap in self._traj.rounds:
            rc = snap.round_cost
            _TokenTracker.total_input_tokens += rc.tokens_input
            _TokenTracker.total_output_tokens += rc.tokens_output
            _LatencyTracker.per_type_latencies.setdefault("llm", {"count": 0, "total_latency": 0.0, "calls": []})
            _LatencyTracker.per_type_latencies.setdefault("search", {"count": 0, "total_latency": 0.0, "calls": []})
            _LatencyTracker.per_type_latencies["llm"]["count"] += rc.llm_calls
            _LatencyTracker.per_type_latencies["llm"]["total_latency"] += rc.latency_seconds * 0.5
            _LatencyTracker.per_type_latencies["search"]["count"] += rc.search_calls
            _LatencyTracker.per_type_latencies["search"]["total_latency"] += rc.latency_seconds * 0.5
        return self._traj.final_context

    async def write_report(self):
        sc = self._traj.synthesis_cost
        _TokenTracker.total_input_tokens += sc.tokens_input
        _TokenTracker.total_output_tokens += sc.tokens_output
        _TokenTracker.total_cost = self._traj.total_cost
        _TokenTracker.peak_input_tokens = sc.tokens_input
        _LatencyTracker.per_type_latencies["llm"]["count"] += 1
        _LatencyTracker.per_type_latencies["llm"]["total_latency"] += sc.latency_seconds
        self._traj.report = (
            "# Report\n\nSee https://a.example/1 and https://c.example/3 .\n"
        )
        return self._traj.report


@pytest.fixture
def fake_gpt_researcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pkg = types.ModuleType("gpt_researcher")
    pkg.GPTResearcher = _FakeGPTResearcher
    utils = types.ModuleType("gpt_researcher.utils")
    tok = types.ModuleType("gpt_researcher.utils.token_tracker")
    tok.TokenTracker = _TokenTracker
    lat = types.ModuleType("gpt_researcher.utils.latency_tracker")
    lat.LatencyTracker = _LatencyTracker
    monkeypatch.setitem(sys.modules, "gpt_researcher", pkg)
    monkeypatch.setitem(sys.modules, "gpt_researcher.utils", utils)
    monkeypatch.setitem(sys.modules, "gpt_researcher.utils.token_tracker", tok)
    monkeypatch.setitem(sys.modules, "gpt_researcher.utils.latency_tracker", lat)

    # Raw trajectory files the adapter copies into the run dir.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "trajectory_abc123def456.json").write_text("{}", encoding="utf-8")
    (raw / "trajectory_abc123def456_emb.npz").write_bytes(b"npz")
    monkeypatch.setenv("TRAJECTORY_OUTPUT_DIR", str(raw))
    return raw


# ── tests ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_adapter_converts_rounds_to_steps(fake_gpt_researcher: Path, gym_query, tmp_path: Path):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = GPTResearcherAgent({"trajectory_dir": str(fake_gpt_researcher)})
    task = ResearchTask(query=gym_query, budget=Budget(max_evidence=1))  # tiny cap must not clobber retention
    meter = CostMeter()
    ctx = AgentContext(llm=None, search=None, extra={"meter": meter, "run_dir": str(run_dir)})

    traj = await agent.run(task, ctx)

    # 2 rounds -> 2 PRUNE steps, then WRITE, then TERMINATE
    kinds = [s.action.type.value for s in traj.steps]
    assert kinds == ["prune", "prune", "write", "terminate"]
    assert traj.steps[0].extra["round_id"] == 1
    assert traj.steps[0].extra["pruned_item_ids"] == ["e2"]
    assert traj.steps[0].tokens.total_tokens == 1100
    assert traj.steps[1].extra["decision_type"] == "terminate"
    assert traj.steps[2].tokens.prompt_tokens == 5000

    # citations are what the report contains, not every retained URL
    assert traj.report is not None
    assert "https://a.example/1" in traj.report.article
    assert traj.report.citations == ["https://a.example/1", "https://c.example/3"]

    # retention honours gpt-researcher, not Budget.max_evidence
    stats = traj.final_stats
    assert stats["n_retained"] == 2
    assert stats["n_pruned"] == 1
    assert stats["cost_source"] == "gpt_researcher.TokenTracker"
    assert stats["total_cost_usd"] == pytest.approx(0.0421)
    assert stats["synthesis_prompt_tokens"] == 5000
    assert stats["subquestions"] == ["s1", "s2"]

    # meter populated from the fork's trackers
    assert meter.total_tokens == 1100 + 880 + 5900
    assert meter.n_calls == 3 + 2 + 1
    assert meter.n_search_calls == 3
    assert meter.n_fetch_calls == 4

    # raw files copied and renamed to the harness query id
    assert (run_dir / "gpt_researcher" / "chip.json").exists()
    assert (run_dir / "gpt_researcher" / "chip_emb.npz").exists()


@pytest.mark.asyncio
async def test_local_metrics_pick_up_adapter_output(fake_gpt_researcher: Path, gym_query, tmp_path: Path):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    agent = GPTResearcherAgent({"keep_trajectory_files": False})
    meter = CostMeter()
    ctx = AgentContext(llm=None, search=None, extra={"meter": meter})
    traj = await agent.run(ResearchTask(query=gym_query), ctx)
    # Mirror what the runner does before local metrics.
    traj.final_stats["usage"] = meter.snapshot()
    traj.final_stats["wall_s"] = 42.0

    m = compute_local_metrics([traj])
    row = m["per_query"][0]
    assert row["tokens"] == 7880
    assert row["n_llm_calls"] == 6
    assert row["n_searches"] == 3
    assert row["n_reads"] == 4
    assert row["n_retained"] == 2 and row["n_pruned"] == 1
    assert row["prune_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["n_citations"] == 2
    assert m["n_with_report"] == 1


def test_registry_exposes_gpt_researcher():
    from adr.agents.registry import available_agents, build_agent

    assert "gpt_researcher" in available_agents()
    agent = build_agent("gpt_researcher", {"depth": 3})
    assert agent.name == "gpt_researcher"
    assert agent.config["depth"] == 3


def test_missing_fork_gives_clear_error(monkeypatch: pytest.MonkeyPatch, gym_query):
    """If the real package is importable but lacks the fork's logger, say so."""
    import asyncio

    from adr.agents.gpt_researcher import GPTResearcherAgent

    class _Bare:
        def __init__(self, **_):
            self.deep_researcher = None

        async def conduct_research(self):
            return ""

        async def write_report(self):
            return ""

    pkg = types.ModuleType("gpt_researcher")
    pkg.GPTResearcher = _Bare
    utils = types.ModuleType("gpt_researcher.utils")
    tok = types.ModuleType("gpt_researcher.utils.token_tracker")
    tok.TokenTracker = _TokenTracker
    lat = types.ModuleType("gpt_researcher.utils.latency_tracker")
    lat.LatencyTracker = _LatencyTracker
    for name, mod in [
        ("gpt_researcher", pkg),
        ("gpt_researcher.utils", utils),
        ("gpt_researcher.utils.token_tracker", tok),
        ("gpt_researcher.utils.latency_tracker", lat),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    agent = GPTResearcherAgent({"keep_trajectory_files": False})
    ctx = AgentContext(llm=None, search=None, extra={})
    with pytest.raises(RuntimeError, match="trajectory_logger"):
        asyncio.run(agent.run(ResearchTask(query=gym_query), ctx))


# ── repo resolution ───────────────────────────────────────────────────
def test_find_gpt_researcher_requires_fork_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from adr.eval.repos import find_gpt_researcher

    monkeypatch.delenv("ADR_GR_DIR", raising=False)

    plain = tmp_path / "gpt-researcher"
    (plain / "gpt_researcher" / "utils").mkdir(parents=True)
    assert not find_gpt_researcher(plain).ok, "upstream clone without trajectory_logger must not resolve"

    (plain / "gpt_researcher" / "utils" / "trajectory_logger.py").write_text("", encoding="utf-8")
    loc = find_gpt_researcher(plain)
    assert loc.ok and loc.path == plain.resolve()

    monkeypatch.setenv("ADR_GR_DIR", str(plain))
    assert find_gpt_researcher().path == plain.resolve()


@pytest.mark.asyncio
async def test_agent_inserts_resolved_repo_on_sys_path(
    fake_gpt_researcher: Path, gym_query, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    fork = tmp_path / "fork"
    (fork / "gpt_researcher" / "utils").mkdir(parents=True)
    (fork / "gpt_researcher" / "utils" / "trajectory_logger.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("ADR_GR_DIR", str(fork))
    monkeypatch.setattr(sys, "path", list(sys.path))

    agent = GPTResearcherAgent({"keep_trajectory_files": False})
    ctx = AgentContext(llm=None, search=None, extra={})
    await agent.run(ResearchTask(query=gym_query), ctx)
    assert sys.path[0] == str(fork.resolve())
