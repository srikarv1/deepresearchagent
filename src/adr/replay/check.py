"""Does replaying the whole cached tree reproduce the live writer input?

Every replayed plan's score is only comparable with the live run if
``ResearchTree.writer_context`` rebuilds exactly what the live agent handed its
writer. The full tree is the one plan where the live input is on record
(``final_context`` of the fork trajectory), so it is the acceptance check for
a GR_NODE_CACHE run: the two strings must be identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from adr.replay.tree import PACKS, ResearchTree

_SNIPPET = 120


@dataclass(frozen=True)
class ReplayCheck:
    path: str
    ok: bool
    reason: str = ""
    n_nodes: int = 0
    live_chars: int = 0
    replay_chars: int = 0
    first_diff: int | None = None
    live_snippet: str = ""
    replay_snippet: str = ""


def first_difference(a: str, b: str) -> int | None:
    """Index of the first differing character, ``None`` when equal."""
    if a == b:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def check_full_replay(path: str | Path, pack: str | None = None) -> ReplayCheck:
    """Compare the replayed writer input of the full tree with the live one.

    ``pack`` defaults to GR_FINAL_PACK (``tail``), which must be the value the
    live run used.
    """
    path = Path(path)
    pack = (pack or os.environ.get("GR_FINAL_PACK") or "tail").strip().lower()
    if pack not in PACKS:
        return ReplayCheck(str(path), False, f"GR_FINAL_PACK={pack!r} is not replayable ({', '.join(PACKS)})")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("nodes"):
        return ReplayCheck(str(path), False, "no nodes: the run did not set GR_NODE_CACHE=1")
    live = data.get("final_context") or ""
    if not live.strip():
        return ReplayCheck(str(path), False, "empty final_context: the run did not finish")
    tree = ResearchTree.from_trajectory(path)
    replay = tree.writer_context(tree.all_nodes(), pack=pack)
    diff = first_difference(live, replay)
    if diff is None:
        return ReplayCheck(str(path), True, "", len(tree.nodes), len(live), len(replay))
    lo = max(0, diff - _SNIPPET // 2)
    return ReplayCheck(
        str(path),
        False,
        "replayed writer input differs from final_context",
        len(tree.nodes),
        len(live),
        len(replay),
        diff,
        live[lo:lo + _SNIPPET],
        replay[lo:lo + _SNIPPET],
    )


def trajectory_files(paths: list[str | Path]) -> list[Path]:
    """Fork trajectories named by the arguments: files as given, and for a run
    directory every ``gpt_researcher/<qid>.json`` it holds."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            sub = p / "gpt_researcher" if (p / "gpt_researcher").is_dir() else p
            out.extend(sorted(sub.glob("*.json")))
        else:
            out.append(p)
    return out
