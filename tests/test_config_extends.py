"""`extends:` in run and agent YAML, and the shipped Claude Opus 4.6 variants."""

from __future__ import annotations

from pathlib import Path

import pytest

from adr.agents.registry import build_agent
from adr.runner.config import load_config, load_yaml

ROOT = Path(__file__).resolve().parents[1]
OPUS = "anthropic:claude-opus-4-6"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_extends_merges_mappings_and_lets_the_child_win(tmp_path: Path):
    _write(
        tmp_path / "base" / "agent.yaml",
        "depth: 3\nrepo_path: /fork\nenv:\n  SMART_LLM: openai:gpt-5-mini\n  EMBEDDING: openai:e3\n"
        "orchestration:\n  policy: legacy\n  greedy:\n    min_gain: 0.0\n    lambda: 1.0\n",
    )
    child = _write(
        tmp_path / "variants" / "agent.yaml",
        "extends: ../base/agent.yaml\nrepo_path: null\nenv:\n  SMART_LLM: anthropic:claude-opus-4-6\n"
        "orchestration:\n  greedy:\n    lambda: 0.5\n",
    )

    cfg = load_yaml(child)

    assert cfg == {
        "depth": 3,
        "repo_path": None,
        "env": {"SMART_LLM": "anthropic:claude-opus-4-6", "EMBEDDING": "openai:e3"},
        "orchestration": {"policy": "legacy", "greedy": {"min_gain": 0.0, "lambda": 0.5}},
    }


def test_extends_chains_and_rejects_cycles(tmp_path: Path):
    _write(tmp_path / "a.yaml", "a: 1\nshared: a\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\nb: 2\nshared: b\n")
    c = _write(tmp_path / "c.yaml", "extends: b.yaml\nc: 3\n")
    assert load_yaml(c) == {"a": 1, "b": 2, "c": 3, "shared": "b"}

    loop = _write(tmp_path / "loop1.yaml", "extends: loop2.yaml\n")
    _write(tmp_path / "loop2.yaml", "extends: loop1.yaml\n")
    with pytest.raises(ValueError, match="extends cycle"):
        load_yaml(loop)


@pytest.mark.parametrize("value", ["[a.yaml]", "''", "3"])
def test_extends_must_name_a_file(tmp_path: Path, value: str):
    bad = _write(tmp_path / "bad.yaml", f"extends: {value}\n")
    with pytest.raises(ValueError, match="extends must be a file path"):
        load_yaml(bad)


def test_non_mapping_config_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml(_write(tmp_path / "list.yaml", "- a\n- b\n"))


@pytest.mark.parametrize("bench", ["bench", "gym"])
def test_opus46_agent_config_only_swaps_the_chat_models(bench: str):
    base = load_yaml(ROOT / f"configs/agents/gpt_researcher_{bench}.yaml")
    opus = load_yaml(ROOT / f"configs/agents/gpt_researcher_{bench}_opus46.yaml")

    for name in ("STRATEGIC_LLM", "SMART_LLM", "FAST_LLM"):
        assert opus["env"][name] == OPUS
    assert opus["env"]["EMBEDDING"] == base["env"]["EMBEDDING"]
    without_llms = {k: v for k, v in opus["env"].items() if not k.endswith("_LLM")}
    assert without_llms == {k: v for k, v in base["env"].items() if not k.endswith("_LLM")}
    assert {k: v for k, v in opus.items() if k != "env"} == {k: v for k, v in base.items() if k != "env"}

    agent = build_agent("gpt_researcher", ROOT / f"configs/agents/gpt_researcher_{bench}_opus46.yaml")
    assert agent.config["env"]["SMART_LLM"] == OPUS


@pytest.mark.parametrize(
    ("variant", "base", "agent_config"),
    [
        ("gpt_researcher_bench_opus46", "gpt_researcher_bench", "configs/agents/gpt_researcher_bench_opus46.yaml"),
        ("gpt_researcher_gym_opus46", "gpt_researcher_gym", "configs/agents/gpt_researcher_gym_opus46.yaml"),
        ("rollouts_drb_random_opus46", "rollouts_drb_random", "configs/agents/gpt_researcher_bench_opus46.yaml"),
        ("rollouts_gym_random_opus46", "rollouts_gym_random", "configs/agents/gpt_researcher_gym_opus46.yaml"),
    ],
)
def test_opus46_run_configs_keep_dataset_and_budget(variant: str, base: str, agent_config: str):
    cfg = load_config(ROOT / f"configs/{variant}.yaml")
    ref = load_config(ROOT / f"configs/{base}.yaml")

    assert cfg["agent"] == {"name": "gpt_researcher", "config": agent_config}
    assert (ROOT / agent_config).exists()
    for key in ("dataset", "budget", "concurrency", "llm", "search", "eval"):
        assert cfg[key] == ref[key], key
