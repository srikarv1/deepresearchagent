"""Re-synthesize a report from a replayed context with the agent's own writer."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml

from adr.runner.config import ROOT

DEFAULT_AGENT_CONFIG = ROOT / "configs" / "agents" / "gpt_researcher_gym.yaml"
MIN_REPORT_CHARS = 2000  # real reports here are ~15-25K chars


def apply_agent_env(agent_config: str | Path | None = None) -> None:
    """Set the same model env (STRATEGIC/SMART/FAST_LLM, EMBEDDING, ...) the live
    agent used, so the replayed report comes from the same writer."""
    agent_config = agent_config or os.environ.get("REPLAY_AGENT_CONFIG") or DEFAULT_AGENT_CONFIG
    cfg = yaml.safe_load(Path(agent_config).read_text(encoding="utf-8")) or {}
    for k, v in (cfg.get("env") or {}).items():
        os.environ[str(k)] = str(v)


async def synthesize(query: str, context: str) -> str:
    from gpt_researcher import GPTResearcher

    researcher = GPTResearcher(query=query, report_type="deep")
    # write_report(ext_context=...) is the live path with the context swapped in
    # (skills/writer.py: ``context = ext_context or self.researcher.context``).
    return await researcher.write_report(ext_context=context)


async def synthesize_many(query: str, contexts: dict[str, str], concurrency: int = 6) -> dict[str, str]:
    sem = asyncio.Semaphore(concurrency)

    async def one(tag: str, ctx: str) -> tuple[str, str]:
        async with sem:
            for attempt in range(3):
                try:
                    report = await asyncio.wait_for(synthesize(query, ctx), timeout=900)
                    # The writer can swallow API errors (e.g. 429) and return an
                    # empty or error string; never let that be judged as a report.
                    if len(report.strip()) < MIN_REPORT_CHARS or report.lstrip().startswith(("Error", "Failed")):
                        raise RuntimeError(f"writer returned no report ({len(report)} chars)")
                    return tag, report
                except Exception as exc:  # retry transient API errors
                    if attempt == 2:
                        print(f"synthesis failed for {tag}: {exc}", flush=True)
                        return tag, ""
                    await asyncio.sleep(5 * (attempt + 1))
        return tag, ""

    pairs = await asyncio.gather(*(one(t, c) for t, c in contexts.items()))
    return dict(pairs)
