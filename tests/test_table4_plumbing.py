"""Table 4 plumbing: per-run agent overrides reach the fork's environment, the
CLI maps --agent-env / --seed into the config, and `adr table` aggregates run
dirs by policy with mean ± std over seeds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from adr.agents.registry import build_agent
from adr.cli import parse_env_pairs
from adr.eval.table import build_table, collect_runs, to_markdown
from adr.runner.experiment import _orchestrator_of


def test_build_agent_deep_merges_overrides(monkeypatch, tmp_path: Path):
    base = tmp_path / "agent.yaml"
    base.write_text(
        yaml.safe_dump({"depth": 1, "breadth": 2, "env": {"RETRIEVER": "tavily", "GR_ORCHESTRATOR": "legacy"}})
    )
    agent = build_agent(
        "gpt_researcher", base, overrides={"depth": 2, "env": {"GR_ORCHESTRATOR": "topk", "GR_CONTEXT_BUDGET_TOKENS": "4000"}}
    )
    assert agent.config["depth"] == 2 and agent.config["breadth"] == 2
    assert agent.config["env"] == {"RETRIEVER": "tavily", "GR_ORCHESTRATOR": "topk", "GR_CONTEXT_BUDGET_TOKENS": "4000"}

    for k in ("GR_ORCHESTRATOR", "GR_CONTEXT_BUDGET_TOKENS", "DEEP_RESEARCH_DEPTH"):
        monkeypatch.delenv(k, raising=False)
    agent._prepare_import()  # sets env before the fork is imported
    assert os.environ["GR_ORCHESTRATOR"] == "topk"
    assert os.environ["GR_CONTEXT_BUDGET_TOKENS"] == "4000"
    assert os.environ["DEEP_RESEARCH_DEPTH"] == "2"


def test_parse_env_pairs():
    assert parse_env_pairs(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(Exception):
        parse_env_pairs(["NOEQUALS"])


def test_orchestrator_of_prefers_overrides():
    cfg = {"agent": {"env": {"GR_ORCHESTRATOR": "none"}, "overrides": {"env": {"GR_ORCHESTRATOR": "topk"}}}}
    assert _orchestrator_of(cfg) == "topk"
    assert _orchestrator_of({"agent": {"env": {"GR_ORCHESTRATOR": "none"}}}) == "none"
    assert _orchestrator_of({"agent": {}}) is None


def _fake_run(root: Path, name: str, policy: str, seed: int, race: float | None, tokens: float, wall: float) -> Path:
    run = root / f"20260908-1200{seed:02d}-{name}"
    (run / "metrics").mkdir(parents=True)
    cfg = {
        "run_name": name,
        "seed": seed,
        "dataset": {"name": "deep_research_bench"},
        "agent": {"name": "gpt_researcher", "overrides": {"env": {"GR_ORCHESTRATOR": policy}}},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    scores = {"race_overall_score": race, "fact_valid_rate": 70.0} if race is not None else {}
    official = {"deep_research_bench": {"official": race is not None, "race": {}, "fact": {}}}
    summary = {
        "run_id": run.name,
        "dataset": "deep_research_bench",
        "seed": seed,
        "orchestrator": policy,
        "n_queries": 2,
        "scores": scores,
        "official": official,
    }
    (run / "metrics" / "summary.json").write_text(json.dumps(summary))
    per_query = [
        {"id": "1", "tokens": tokens, "wall_s": wall, "n_llm_calls": 10, "n_searches": 4, "prune_rate": 0.25, "error": None},
        {"id": "2", "tokens": tokens, "wall_s": wall, "n_llm_calls": 12, "n_searches": 4, "prune_rate": 0.35, "error": None},
    ]
    (run / "metrics" / "local.json").write_text(json.dumps({"per_query": per_query}))
    return run


def test_table_groups_by_policy_with_mean_std(tmp_path: Path):
    _fake_run(tmp_path, "drb-none-s1", "none", 1, race=45.0, tokens=100_000, wall=300)
    _fake_run(tmp_path, "drb-none-s2", "none", 2, race=47.0, tokens=120_000, wall=320)
    _fake_run(tmp_path, "drb-topk-s1", "topk", 1, race=44.0, tokens=60_000, wall=250)
    _fake_run(tmp_path, "drb-topk-s2", "topk", 2, race=None, tokens=62_000, wall=255)  # not judged yet

    rows = collect_runs([tmp_path])
    assert len(rows) == 4
    table = build_table(rows)
    assert table["dataset"] == "deep_research_bench"
    assert [r["policy"] for r in table["rows"]] == ["none", "topk"]

    none_row = table["rows"][0]
    race_idx = table["columns"].index("RACE overall")
    tok_idx = table["columns"].index("Tokens (K)")
    assert none_row["cells"][race_idx].values == [45.0, 47.0]
    assert none_row["cells"][tok_idx].values == [100.0, 120.0]
    assert none_row["seeds"] == [1, 2]

    md = to_markdown(table)
    assert "| No pruning | 2 | 2 | 46.0 ± 1.4 |" in md
    assert "Top-k similarity" in md
    # The unjudged topk seed is flagged rather than silently averaged.
    assert any("1/2 runs not judged" in n for n in table["notes"])
    # Only one judged topk run -> no ± on its RACE cell.
    topk_row = table["rows"][1]
    assert topk_row["cells"][race_idx].fmt() == "44.0"


def test_table_refuses_mixed_datasets_without_filter(tmp_path: Path):
    _fake_run(tmp_path, "drb-none-s1", "none", 1, race=45.0, tokens=1, wall=1)
    run = _fake_run(tmp_path, "bc-none-s1", "none", 1, race=None, tokens=1, wall=1)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    cfg["dataset"]["name"] = "browsecomp"
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    summary = json.loads((run / "metrics" / "summary.json").read_text())
    summary["dataset"] = "browsecomp"
    (run / "metrics" / "summary.json").write_text(json.dumps(summary))

    rows = collect_runs([tmp_path])
    with pytest.raises(ValueError, match="several datasets"):
        build_table(rows)
    assert build_table(rows, dataset="browsecomp")["columns"][0] == "Accuracy %"
