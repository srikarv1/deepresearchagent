import os
import sys
from pathlib import Path

import pytest
import yaml

from adr.agents.gpt_researcher import GPTResearcherAgent, orchestration_env
from adr.agents.registry import build_agent
from adr.datasets.loader import load_queries
from adr.runner.config import load_config


RUN_CONFIG = Path("configs/gpt_researcher_bench_gptsol_d2_learned.yaml")
AGENT_CONFIG = Path("configs/agents/gpt_researcher_bench_gptsol_d2_learned.yaml")
EXPECTED_IDS = [str(i) for i in range(61, 76)]
SOL_MODEL = "openai:gpt-6.1-sol-1gtjfzlv8i"


def _agent_config() -> dict:
    return yaml.safe_load(AGENT_CONFIG.read_text(encoding="utf-8"))


def test_drb_transfer_run_config_selects_exact_queries_and_disables_judging():
    cfg = load_config(RUN_CONFIG)

    assert cfg["dataset"] == {
        "name": "deep_research_bench",
        "language": "en",
        "limit": 15,
        "query_ids": EXPECTED_IDS,
    }
    rows = load_queries(
        cfg["dataset"]["name"],
        language=cfg["dataset"]["language"],
        limit=cfg["dataset"]["limit"],
        query_ids=cfg["dataset"]["query_ids"],
    )
    assert [row.id for row in rows] == EXPECTED_IDS
    assert cfg["concurrency"] == 1
    assert cfg["agent"] == {
        "name": "gpt_researcher",
        "config": str(AGENT_CONFIG),
    }
    assert cfg["eval"]["official_benches"] == []
    assert cfg["llm"]["provider"] == "mock"
    assert cfg["search"]["backend"] == "mock"


def test_drb_transfer_agent_config_is_sol_depth2_tavily_learned():
    cfg = _agent_config()
    loaded_agent = build_agent("gpt_researcher", AGENT_CONFIG)

    assert loaded_agent.config == cfg
    assert cfg["repo_path"] == "third_party/gpt-researcher"
    assert (cfg["depth"], cfg["breadth"], cfg["concurrency"]) == (2, 4, 1)
    assert cfg["retriever"] == "tavily"
    assert cfg["require_orchestrator_base_url"] is True
    assert cfg["allowed_orchestrator_models"] == [
        "pilot-bc-teacher",
        "pilot-dpo-teacher",
    ]
    assert orchestration_env(cfg["orchestration"])["GR_ORCHESTRATOR"] == "learned"
    assert cfg["env"]["GR_ORCHESTRATOR"] == "learned"
    for role in ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM"):
        assert cfg["env"][role] == SOL_MODEL

    # Arm selection and credentials must remain external and uncommitted.
    assert "GR_ORCH_LLM_MODEL" not in cfg["env"]
    assert "GR_ORCH_LLM_BASE_URL" not in cfg["env"]
    assert "GR_ORCH_LLM_API_KEY" not in cfg["env"]


def test_experiment_config_overrides_stale_shell_policy_and_model_roles(monkeypatch):
    cfg = _agent_config()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("GR_ORCHESTRATOR", "legacy")
    monkeypatch.setenv("STRATEGIC_LLM", "openai:wrong-model")
    monkeypatch.setenv("SMART_LLM", "openai:wrong-model")
    monkeypatch.setenv("FAST_LLM", "openai:wrong-model")
    monkeypatch.setenv("DEEP_RESEARCH_DEPTH", "99")
    monkeypatch.setenv("DEEP_RESEARCH_BREADTH", "99")
    monkeypatch.setenv("GR_ORCH_LLM_MODEL", "pilot-bc-teacher")
    monkeypatch.setenv("GR_ORCH_LLM_BASE_URL", "https://adapter.invalid/v1")

    agent = GPTResearcherAgent(cfg)
    agent._prepare_import()

    assert agent._orchestration_effective["GR_ORCHESTRATOR"] == "learned"
    for role in ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM"):
        assert os.environ[role] == SOL_MODEL
    assert os.environ["DEEP_RESEARCH_DEPTH"] == "2"
    assert os.environ["DEEP_RESEARCH_BREADTH"] == "4"


@pytest.mark.parametrize("adapter", ["pilot-bc-teacher", "pilot-dpo-teacher"])
def test_adapter_arm_remains_an_explicit_environment_selection(monkeypatch, adapter):
    cfg = _agent_config()
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("GR_ORCH_LLM_MODEL", adapter)
    monkeypatch.setenv("GR_ORCH_LLM_BASE_URL", "https://adapter.invalid/v1")

    GPTResearcherAgent(cfg)._prepare_import()

    assert os.environ["GR_ORCH_LLM_MODEL"] == adapter
    assert os.environ["GR_ORCH_LLM_BASE_URL"] == "https://adapter.invalid/v1"


@pytest.mark.parametrize(
    ("model", "base_url", "message"),
    [
        ("", "https://adapter.invalid/v1", "GR_ORCH_LLM_MODEL"),
        ("stale-adapter", "https://adapter.invalid/v1", "GR_ORCH_LLM_MODEL"),
        ("pilot-bc-teacher", "", "GR_ORCH_LLM_BASE_URL"),
    ],
)
def test_invalid_or_missing_adapter_selection_fails_before_import(
    monkeypatch, model, base_url, message
):
    cfg = _agent_config()
    monkeypatch.setattr(sys, "path", list(sys.path))
    if model:
        monkeypatch.setenv("GR_ORCH_LLM_MODEL", model)
    else:
        monkeypatch.delenv("GR_ORCH_LLM_MODEL", raising=False)
    if base_url:
        monkeypatch.setenv("GR_ORCH_LLM_BASE_URL", base_url)
    else:
        monkeypatch.delenv("GR_ORCH_LLM_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match=message):
        GPTResearcherAgent(cfg)._prepare_import()
