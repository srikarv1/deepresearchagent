"""Replayed reports come from the live agent's writer, without side effects."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from adr.replay import synth

LLM_VARS = ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM")


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for name in LLM_VARS + ("GR_ANSWER_FORMAT",):
        monkeypatch.delenv(name, raising=False)


def test_apply_agent_env_resolves_extends(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "name: gpt_researcher\nenv:\n  STRATEGIC_LLM: openai:gpt-5-mini\n"
        "  SMART_LLM: openai:gpt-5-mini\n  EMBEDDING: openai:text-embedding-3-small\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nenv:\n  SMART_LLM: anthropic:claude-opus-4-6\n")

    applied = synth.apply_agent_env(child)

    assert os.environ["SMART_LLM"] == "anthropic:claude-opus-4-6"
    assert os.environ["STRATEGIC_LLM"] == "openai:gpt-5-mini"
    assert applied["EMBEDDING"] == "openai:text-embedding-3-small"


def test_apply_agent_env_reads_the_repo_opus_config():
    applied = synth.apply_agent_env("configs/agents/gpt_researcher_bench_d2b4_opus46.yaml")
    assert all(applied[v] == "anthropic:claude-opus-4-6" for v in LLM_VARS)


def test_apply_agent_env_rejects_a_config_without_a_writer(tmp_path: Path):
    cfg = tmp_path / "no_llm.yaml"
    cfg.write_text("name: gpt_researcher\ndepth: 2\n")
    with pytest.raises(ValueError, match="default writer"):
        synth.apply_agent_env(cfg)
    assert "SMART_LLM" not in os.environ


def test_apply_agent_env_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        synth.apply_agent_env(tmp_path / "missing.yaml")


@pytest.mark.asyncio
async def test_synthesize_detaches_the_deep_skill(monkeypatch):
    calls = []

    class FakeResearcher:
        def __init__(self, query: str, report_type: str):
            self.query, self.report_type = query, report_type
            self.deep_researcher = object()

        async def write_report(self, ext_context=None):
            calls.append((self.report_type, self.deep_researcher, ext_context))
            return "report"

    monkeypatch.setitem(sys.modules, "gpt_researcher", types.SimpleNamespace(GPTResearcher=FakeResearcher))
    monkeypatch.setenv("GR_ANSWER_FORMAT", "browsecomp")

    assert await synth.synthesize("q", "ctx") == "report"
    assert calls == [("deep", None, "ctx")]
    assert "GR_ANSWER_FORMAT" not in os.environ
