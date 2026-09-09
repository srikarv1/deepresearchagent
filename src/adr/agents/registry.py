from __future__ import annotations

from pathlib import Path

import yaml

from adr.agents.base import ResearchAgent
from adr.agents import deep_research, fixture, gpt_researcher, pilot

_BUILDERS = {
    "fixture": fixture.build,
    "deep_research": deep_research.build,
    "gpt_researcher": gpt_researcher.build,
    "pilot": pilot.build,
}


def available_agents() -> list[str]:
    return sorted(_BUILDERS)


def build_agent(
    name: str,
    config: dict | str | Path | None = None,
    overrides: dict | None = None,
) -> ResearchAgent:
    """Build an agent from its YAML config, with optional per-run ``overrides``
    deep-merged on top (e.g. ``{"env": {"GR_ORCHESTRATOR": "topk"}}`` so one
    agent file serves every Table 4 row)."""
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown agent {name!r}. Registered: {available_agents()}")
    cfg = _load_config(config)
    if overrides:
        _deep_update(cfg, overrides)
    return _BUILDERS[key](cfg)


def _deep_update(base: dict, incoming: dict) -> None:
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _load_config(config: dict | str | Path | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Agent config must be a mapping: {path}")
    return loaded
