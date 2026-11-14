"""Exercise the gpt-researcher adapter without the real package.

A fake ``gpt_researcher`` module is injected into ``sys.modules`` that mimics
the instrumented fork's surface: ``GPTResearcher`` with ``conduct_research`` /
``write_report``, a ``deep_researcher.trajectory_logger.trajectory`` object,
and the process-global ``TokenTracker`` / ``LatencyTracker``.
"""

from __future__ import annotations

import json
import logging
import os
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
        # A corpus hit that reached the embeddings filter (BrowseComp-Plus runs).
        "e4": _Ev("corpus page", "bcp://2882", "sq2", 2, 2, False),
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

    last_write_kwargs: dict = {}

    def __init__(self, query: str, report_type: str = "research_report", **_: object):
        assert report_type == "deep"
        self.query = query
        self._traj = _make_traj()
        self.deep_researcher = _Deep(self._traj)
        self.visited_urls = {"https://a.example/1", "https://b.example/2", "https://c.example/3", "https://d.example/4"}
        # Retriever hits aggregated by deep research; bcp:// ones come from adr serve-retriever.
        self.research_sources = [
            {"url": "bcp://74874"},
            {"url": "bcp://2882"},
            {"url": "bcp://68543"},
            {"url": "https://d.example/4"},
        ]

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

    async def write_report(self, **kwargs):
        type(self).last_write_kwargs = dict(kwargs)
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

        async def write_report(self, **_):
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


@pytest.mark.asyncio
async def test_adapter_collects_corpus_docids(fake_gpt_researcher: Path, gym_query, tmp_path: Path):
    """bcp:// URLs from the logger pool and research_sources become retrieved_docids."""
    from adr.agents.gpt_researcher import GPTResearcherAgent

    agent = GPTResearcherAgent({"trajectory_dir": str(fake_gpt_researcher), "keep_trajectory_files": False})
    task = ResearchTask(query=gym_query, budget=Budget())
    ctx = AgentContext(llm=None, search=None, extra={"meter": CostMeter(), "run_dir": str(tmp_path)})

    traj = await agent.run(task, ctx)

    assert traj.final_stats["retrieved_docids"] == ["2882", "68543", "74874"]


def test_retrieved_docids_ignores_web_urls():
    from adr.agents.gpt_researcher import GPTResearcherAgent

    class _R:
        research_sources = [{"url": "https://x.example"}, "bcp://5", {"url": None}]
        visited_urls = {"https://y.example", "bcp://5/"}

    class _T:
        evidence = {"e": types.SimpleNamespace(source_url="bcp://7")}

    assert GPTResearcherAgent._retrieved_docids(_R(), _T()) == ["5", "7"]
    assert GPTResearcherAgent._retrieved_docids(types.SimpleNamespace(), types.SimpleNamespace()) == []


@pytest.mark.asyncio
async def test_adapter_requests_browsecomp_answer_format(
    fake_gpt_researcher: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from adr.agents.gpt_researcher import GPTResearcherAgent
    from adr.core.types import Query

    monkeypatch.delenv("GR_ANSWER_FORMAT", raising=False)
    query = Query(
        id="1",
        text="An African author...",
        dataset="browsecomp_plus",
        language="en",
        metadata={"answer": "1988-96"},
    )
    agent = GPTResearcherAgent({"trajectory_dir": str(fake_gpt_researcher), "keep_trajectory_files": False})
    ctx = AgentContext(llm=None, search=None, extra={"meter": CostMeter(), "run_dir": str(tmp_path)})
    await agent.run(ResearchTask(query=query, budget=Budget()), ctx)
    assert _FakeGPTResearcher.last_write_kwargs.get("answer_format") == "browsecomp"

    gym = Query(id="chip", text="why chip", dataset="deep_research_gym", language="en")
    await agent.run(ResearchTask(query=gym, budget=Budget()), ctx)
    assert "answer_format" not in _FakeGPTResearcher.last_write_kwargs


def test_legacy_fallback_prompt_is_official_query_template():
    from adr.agents.gpt_researcher import _LEGACY_BROWSECOMP_PROMPT
    from adr.eval.browsecomp_plus_prompts import QUERY_TEMPLATE_NO_GET_DOCUMENT

    assert _LEGACY_BROWSECOMP_PROMPT == QUERY_TEMPLATE_NO_GET_DOCUMENT
    assert "You have completed research" not in _LEGACY_BROWSECOMP_PROMPT


# ── orchestration: section -> GR_* variables ──────────────────────────

from adr.agents.gpt_researcher import _ORCH_ENV_MAP as _ORCH_MAP_FOR_TESTS  # noqa: E402

_ORCH_NAMES = sorted(set(_ORCH_MAP_FOR_TESTS.values()))


def test_orchestration_env_maps_nested_keys_and_skips_null():
    from adr.agents.gpt_researcher import orchestration_env

    env = orchestration_env(
        {
            "policy": "greedy",
            "context_budget_tokens": 50000,
            "feature_tau": None,
            "greedy": {"min_gain": 0.0, "lambda": 1.0},
            "heuristic_stop": {"patience": 2, "retain": "filter", "gain_threshold": None},
            "random": {"seed": 17, "params": {"keep_ratio": 0.4, "stop": "phi"}},
            "typesafe": {"model": "jev-1.13.0", "keep_min": 0.5},
        }
    )
    assert env == {
        "GR_ORCHESTRATOR": "greedy",
        "GR_CONTEXT_BUDGET_TOKENS": "50000",
        "GR_GREEDY_MIN_GAIN": "0.0",
        "GR_GREEDY_LAMBDA": "1.0",
        "GR_STOP_PATIENCE": "2",
        "GR_STOP_RETAIN": "filter",
        "GR_ORCH_SEED": "17",
        "GR_RANDOM_PARAMS": '{"keep_ratio": 0.4, "stop": "phi"}',
        "GR_TYPESAFE_MODEL": "jev-1.13.0",
        "GR_TYPESAFE_KEEP_MIN": "0.5",
    }
    assert orchestration_env(None) == {} and orchestration_env({}) == {}


def test_orchestration_env_rejects_unknown_keys():
    from adr.agents.gpt_researcher import orchestration_env

    with pytest.raises(ValueError, match="unknown key 'greedy.lamda'"):
        orchestration_env({"greedy": {"lamda": 1.0}})
    with pytest.raises(ValueError, match="must be a mapping"):
        orchestration_env(["policy"])  # type: ignore[arg-type]


def test_shipped_agent_configs_validate_against_the_mapping():
    """Every gpt_researcher agent YAML in configs/ must flatten cleanly, so a
    typo in a checked-in config fails here rather than at run time."""
    import yaml
    from adr.agents.gpt_researcher import orchestration_env

    paths = sorted(Path("configs/agents").glob("gpt_researcher*.yaml"))
    assert paths, "no gpt_researcher agent configs found"
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = orchestration_env(cfg.get("orchestration"))
        assert env["GR_ORCHESTRATOR"] == "legacy", path
        assert env["GR_CONTEXT_BUDGET_TOKENS"] == "50000", path
        assert "GR_FEAT_TAU" not in env, f"{path}: feature_tau should stay null"


def test_prepare_import_exports_yaml_defaults_and_lets_explicit_env_win(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    for name in _ORCH_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GR_ORCHESTRATOR", "heuristic_stop")  # row selected by sweep_policies.sh
    monkeypatch.setenv("GR_STOP_PATIENCE", "")  # empty counts as unset
    monkeypatch.setenv("GR_GREEDY_MIN_GAIN", "0.1")  # a knob overridden from the shell
    monkeypatch.setattr(sys, "path", list(sys.path))

    agent = GPTResearcherAgent(
        {
            "repo_path": None,
            "orchestration": {
                "policy": "greedy",
                "context_budget_tokens": 50000,
                "greedy": {"min_gain": 0.0, "lambda": 0.5},
                "heuristic_stop": {"patience": 2},
            },
            "env": {"GR_GREEDY_LAMBDA": "0.25"},
        }
    )
    with caplog.at_level(logging.INFO, logger="adr.agents.gpt_researcher"):
        agent._prepare_import()

    assert os.environ["GR_ORCHESTRATOR"] == "heuristic_stop"  # explicit env won
    assert os.environ["GR_CONTEXT_BUDGET_TOKENS"] == "50000"  # YAML default exported
    assert os.environ["GR_STOP_PATIENCE"] == "2"  # empty env value did not block the default
    assert os.environ["GR_GREEDY_MIN_GAIN"] == "0.1"  # explicit knob override won
    assert os.environ["GR_GREEDY_LAMBDA"] == "0.25"  # the raw env block always wins
    by_msg = {r.message: r.levelno for r in caplog.records}
    assert by_msg[
        "orchestration: GR_ORCHESTRATOR=heuristic_stop from the environment overrides greedy from the agent config"
    ] == logging.INFO
    assert by_msg[
        "orchestration: GR_GREEDY_MIN_GAIN=0.1 from the environment overrides 0.0 from the agent config"
    ] == logging.WARNING
    assert agent._orchestration_effective["GR_ORCHESTRATOR"] == "heuristic_stop"
    assert agent._orchestration_effective["GR_GREEDY_LAMBDA"] == "0.25"
    assert "GR_FEAT_TAU" not in agent._orchestration_effective


# ── token stopping rule + per-call LLM log ───────────────────────────


def test_orchestration_env_maps_the_token_stopping_rule():
    from adr.agents.gpt_researcher import orchestration_env

    assert orchestration_env({"stop_total_tokens": 150000, "report_reserve_tokens": "auto"}) == {
        "GR_STOP_TOTAL_TOKENS": "150000",
        "GR_REPORT_RESERVE_TOKENS": "auto",
    }
    assert orchestration_env({"report_reserve_tokens": 45000}) == {"GR_REPORT_RESERVE_TOKENS": "45000"}
    # null leaves the fork default (no stopping rule) in place.
    assert orchestration_env({"stop_total_tokens": None, "report_reserve_tokens": None}) == {}


def test_stopping_rule_is_only_on_in_the_budget150_configs():
    """The uncapped configs feed the pilot scripts, sweep_policies.sh and the
    rollout configs; the 150k stopping rule must live only in *_budget150."""
    import yaml

    from adr.agents.gpt_researcher import orchestration_env
    from adr.runner.config import load_config

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted((root / "agents").glob("gpt_researcher*.yaml")):
        env = orchestration_env((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("orchestration"))
        if path.stem.endswith("_budget150"):
            assert env.get("GR_STOP_TOTAL_TOKENS") == "150000", path.name
            assert env.get("GR_REPORT_RESERVE_TOKENS") == "auto", path.name
        else:
            assert "GR_STOP_TOTAL_TOKENS" not in env, path.name

    for run_cfg, agent_cfg in (
        ("gym_sol_pilot_d2_budget150.yaml", "configs/agents/gpt_researcher_gym_gptsol_d2_budget150.yaml"),
        ("gpt_researcher_bench_budget150.yaml", "configs/agents/gpt_researcher_bench_budget150.yaml"),
    ):
        cfg = load_config(root / run_cfg)
        assert cfg["agent"]["config"] == agent_cfg
        assert cfg["budget"]["max_tokens"] == 150000
        assert cfg["run_name"].endswith("-budget150")


def _stop_stats(info, totals, synth):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    deep = types.SimpleNamespace(token_stop_info=info)
    tracker = types.SimpleNamespace(get_totals=lambda: totals)
    gtraj = types.SimpleNamespace(synthesis_cost=synth)
    return GPTResearcherAgent._token_stop_stats(deep, tracker, gtraj)


def test_token_stop_stats_flag_a_run_that_ended_above_the_target():
    synth = types.SimpleNamespace(tokens_input=30000, tokens_output=6000)
    info = {"stop_total_tokens": 150000, "stopped": True, "tokens_at_stop": 110000}

    over = _stop_stats(info, {"input_tokens": 140000, "output_tokens": 15000}, synth)
    assert over["final_tokens"] == 155000
    assert over["report_tokens"] == 36000
    assert over["ended_above_target"] is True
    assert over["overshoot_tokens"] == 5000
    assert over["tokens_at_stop"] == 110000  # the fork's record is kept

    under = _stop_stats(info, {"input_tokens": 130000, "output_tokens": 15000}, synth)
    assert under["ended_above_target"] is False and under["overshoot_tokens"] == 0


def test_token_stop_stats_without_a_target():
    stats = _stop_stats({}, {"input_tokens": 300000, "output_tokens": 20000}, None)
    assert stats["final_tokens"] == 320000
    assert stats["ended_above_target"] is False
    assert stats["overshoot_tokens"] is None
    assert stats["report_tokens"] is None


class _CallLog:
    records: list = []
    resets = 0

    @staticmethod
    def reset():
        # Counts only, so the test can preload the records the run returns.
        _CallLog.resets += 1

    @staticmethod
    def get_records():
        return [dict(r) for r in _CallLog.records]

    @staticmethod
    def summary():
        return {"n_calls": len(_CallLog.records), "capacity_wait_s": 2.5}


@pytest.mark.asyncio
async def test_adapter_records_token_stop_and_llm_call_log(
    fake_gpt_researcher: Path, gym_query, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from adr.agents.gpt_researcher import GPTResearcherAgent

    calllog = types.ModuleType("gpt_researcher.utils.llm_call_log")
    calllog.LLMCallLog = _CallLog
    monkeypatch.setitem(sys.modules, "gpt_researcher.utils.llm_call_log", calllog)
    _CallLog.records = [{"model": "m", "attempts": 1, "capacity_wait_s": 2.5}]
    _CallLog.resets = 0
    # The fake run is charged 7880 tokens (rounds 1100 + 880, report 5900).
    monkeypatch.setattr(
        _Deep, "token_stop_info", {"stop_total_tokens": 7000, "stopped": True}, raising=False
    )

    agent = GPTResearcherAgent({"trajectory_dir": str(fake_gpt_researcher)})
    ctx = AgentContext(llm=None, search=None, extra={"meter": CostMeter(), "run_dir": str(tmp_path)})
    traj = await agent.run(ResearchTask(query=gym_query, budget=Budget()), ctx)

    stop = traj.final_stats["token_stop"]
    assert stop["stopped"] is True
    assert stop["final_tokens"] == 7880
    assert stop["report_tokens"] == 5900
    assert stop["ended_above_target"] is True and stop["overshoot_tokens"] == 880

    assert _CallLog.resets == 1  # reset at the start of the query
    assert traj.final_stats["llm_calls"] == [{"model": "m", "attempts": 1, "capacity_wait_s": 2.5}]
    summary = traj.final_stats["llm_call_summary"]
    assert summary["n_calls"] == 1
    assert summary["wall_s_excl_capacity_wait"] == 0.0  # fake run is instant; wait clamps at 0


@pytest.mark.asyncio
async def test_adapter_without_call_log_module(fake_gpt_researcher: Path, gym_query, tmp_path: Path):
    """A fork without llm_call_log (older branches) still runs; no call log is recorded."""
    from adr.agents.gpt_researcher import GPTResearcherAgent

    agent = GPTResearcherAgent({"trajectory_dir": str(fake_gpt_researcher)})
    ctx = AgentContext(llm=None, search=None, extra={"meter": CostMeter(), "run_dir": str(tmp_path)})
    traj = await agent.run(ResearchTask(query=gym_query, budget=Budget()), ctx)
    assert "llm_calls" not in traj.final_stats
    assert traj.final_stats["token_stop"]["final_tokens"] == 7880


def test_local_metrics_average_flex_wait_and_token_stop_outcomes():
    from adr.core.types import Query, Trajectory
    from adr.eval.local_metrics import compute_local_metrics

    def traj(qid, wait, wall_excl, stopped, above, over):
        t = Trajectory(query=Query(id=qid, text="q", dataset="deep_research_gym", language="en"))
        t.final_stats = {
            "llm_call_summary": {"capacity_wait_s": wait, "wall_s_excl_capacity_wait": wall_excl},
            "token_stop": {"stop_total_tokens": 150000, "stopped": stopped,
                           "ended_above_target": above, "overshoot_tokens": over},
        }
        return t

    plain = Trajectory(query=Query(id="c", text="q", dataset="deep_research_gym", language="en"))
    m = compute_local_metrics([traj("a", 10.0, 500.0, True, True, 4000), traj("b", 0.0, 700.0, True, False, 0), plain])
    assert m["mean_capacity_wait_s"] == 5.0
    assert m["mean_wall_s_excl_capacity_wait"] == 600.0
    assert m["mean_token_stop_stopped"] == 1.0
    assert m["mean_token_stop_ended_above_target"] == 0.5
    assert m["mean_token_stop_overshoot_tokens"] == 2000.0
    assert "token_stop_stopped" not in m["per_query"][2]  # runs without the rule add nothing
