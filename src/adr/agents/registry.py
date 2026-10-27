from __future__ import annotations

from pathlib import Path

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


def build_agent(name: str, config: dict | str | Path | None = None) -> ResearchAgent:
    key = name.strip().lower()
    if key not in _BUILDERS:
        raise ValueError(f"Unknown agent {name!r}. Registered: {available_agents()}")
    cfg = _load_config(config)
    return _BUILDERS[key](cfg)


def _load_config(config: dict | str | Path | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    path = Path(config)
    if not path.exists():
        return {}
    # Imported here: adr.runner imports this module through experiment.py.
    from adr.runner.config import load_yaml

    return load_yaml(path)
