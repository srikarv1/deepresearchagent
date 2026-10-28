"""Re-synthesize a report from a replayed context with the agent's own writer."""

from __future__ import annotations

import asyncio
import os
import warnings
from pathlib import Path

from adr.runner.config import ROOT, load_yaml

DEFAULT_AGENT_CONFIG = ROOT / "configs" / "agents" / "gpt_researcher_gym.yaml"
MIN_REPORT_CHARS = 2000  # real reports here are ~15-25K chars


def _resolve_config(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return ROOT / p


def apply_agent_env(agent_config: str | Path | None = None) -> dict[str, str]:
    """Set the same model env (STRATEGIC/SMART/FAST_LLM, EMBEDDING, ...) the live
    agent used, so the replayed report comes from the same writer.

    The config is read with ``load_yaml``, so ``extends:`` resolves the way the
    live agent's config does (the *_opus46 configs only state their backbone
    on top of a base file). Returns the variables that were set.
    """
    agent_config = agent_config or os.environ.get("REPLAY_AGENT_CONFIG")
    if not agent_config:
        # Trees collected with another backbone would be replayed with this
        # default writer, so say which one is used.
        warnings.warn(
            f"REPLAY_AGENT_CONFIG is not set; replaying with the writer of {DEFAULT_AGENT_CONFIG.name}. "
            "Point it at the agent config the trees were collected with.",
            stacklevel=2,
        )
        agent_config = DEFAULT_AGENT_CONFIG
    path = _resolve_config(agent_config)
    if not path.exists():
        raise FileNotFoundError(f"replay agent config {path} not found (REPLAY_AGENT_CONFIG)")
    cfg = load_yaml(path)
    applied = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    if not any(k.endswith("_LLM") for k in applied):
        raise ValueError(
            f"{path} sets no STRATEGIC/SMART/FAST_LLM under env:, so replayed reports "
            "would come from the fork's default writer, not the live one"
        )
    os.environ.update(applied)
    return applied


async def synthesize(query: str, context: str) -> str:
    from gpt_researcher import GPTResearcher

    # The live write_report() of a deep run has no answer format.
    os.environ.pop("GR_ANSWER_FORMAT", None)
    researcher = GPTResearcher(query=query, report_type="deep")
    # Detach the deep skill: write_report() records the report into its
    # trajectory logger and saves it, which would leave one stray
    # trajectory_<uuid>.json in TRAJECTORY_OUTPUT_DIR per replayed plan.
    researcher.deep_researcher = None
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
