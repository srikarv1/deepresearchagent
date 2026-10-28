"""Research tree loaded from a GR_NODE_CACHE trajectory, replayable per sub-tree.

A deep-research run is a tree of sub-query nodes. With GR_BRANCH_INDEPENDENT=1
each node's research depends only on its parent's goal, so the report the agent
would have written had it expanded only a sub-tree S is

    Synthesize(query, context(S))

where context(S) is assembled exactly like DeepResearchSkill.run() does for the
legacy policy: every learning (with its citation) followed by every node's
context, trimmed to MAX_CONTEXT_WORDS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MAX_CONTEXT_WORDS = 25000  # gpt_researcher.skills.deep_research.MAX_CONTEXT_WORDS


@dataclass
class Node:
    node_id: str
    parent_node_id: str | None
    tree_depth: int
    query: str
    research_goal: str
    learnings: list[str]
    citations: dict[str, str]
    context: str
    cost: dict[str, float]
    children: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return int(self.cost.get("tokens_input", 0)) + int(self.cost.get("tokens_output", 0))


class ResearchTree:
    def __init__(self, query: str, nodes: list[Node]):
        self.query = query
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []  # run order (depth-first), used for context order
        for n in nodes:
            # A node id is a hash of its research goal; keep the first on collision.
            if n.node_id in self.nodes:
                continue
            self.nodes[n.node_id] = n
            self.order.append(n.node_id)
        for nid in self.order:
            p = self.nodes[nid].parent_node_id
            if p in self.nodes:
                self.nodes[p].children.append(nid)
        self.roots = [nid for nid in self.order if self.nodes[nid].parent_node_id not in self.nodes]

    # ---- loading -------------------------------------------------------------
    @classmethod
    def from_trajectory(cls, path: str | Path) -> "ResearchTree":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw = data.get("nodes") or []
        if not raw:
            raise ValueError(f"{path} has no nodes; run the agent with GR_NODE_CACHE=1")
        nodes = [
            Node(
                node_id=r["node_id"],
                parent_node_id=r.get("parent_node_id"),
                tree_depth=int(r.get("tree_depth", 1)),
                query=r.get("query", ""),
                research_goal=r.get("research_goal", ""),
                learnings=list(r.get("learnings") or []),
                citations=dict(r.get("citations") or {}),
                context=r.get("context") or "",
                cost=dict(r.get("cost") or {}),
            )
            for r in raw
        ]
        return cls(data.get("query", ""), nodes)

    # ---- sub-trees -----------------------------------------------------------
    def subtree(self, nid: str) -> list[str]:
        out, stack = [], [nid]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(reversed(self.nodes[cur].children))
        return out

    def is_valid(self, nodes: Iterable[str]) -> bool:
        """A sub-tree is realizable only if every node's parent is also in it."""
        s = set(nodes)
        return all(self.nodes[n].parent_node_id not in self.nodes or self.nodes[n].parent_node_id in s for n in s)

    def up_to_depth(self, depth: int) -> set[str]:
        return {nid for nid, n in self.nodes.items() if n.tree_depth <= depth}

    def all_nodes(self) -> set[str]:
        return set(self.order)

    # ---- replay --------------------------------------------------------------
    def context(self, nodes: Iterable[str]) -> str:
        """Single-trim approximation of the writer input (kept so existing reward
        memos stay valid). New reward tables use ``writer_context``."""
        keep = set(nodes)
        ordered = [nid for nid in self.order if nid in keep]
        items = self._learning_items(keep)
        items += [self.nodes[nid].context for nid in ordered if self.nodes[nid].context]
        return "\n".join(trim_to_word_limit(items))

    def writer_context(self, nodes: Iterable[str], pack: str = "tail", max_words: int = MAX_CONTEXT_WORDS) -> str:
        """The writer input the live agent builds when it researches exactly ``nodes``.

        Mirrors DeepResearchSkill on the legacy evidence path, which is the path
        GR_ORCHESTRATOR=replaygym runs by default: each deep_research() call
        returns its own nodes' contexts followed by every expanded node's
        sub-call, tail-trimmed to ``max_words`` (so a large sub-tree loses its
        earliest contexts before the final pack); run() then puts the learnings
        first and packs with GR_FINAL_PACK (``tail`` keeps the end, ``head``
        fills from the front).
        """
        if pack not in PACKS:
            raise ValueError(f"pack must be one of {PACKS}, got {pack!r}")
        keep = set(nodes)
        unknown = keep - self.nodes.keys()
        if unknown:
            raise KeyError(f"unknown nodes: {sorted(unknown)[:5]}")
        if not self.is_valid(keep):
            raise ValueError("nodes are not a realizable sub-tree: a node's parent is missing")
        roots = [r for r in self.roots if r in keep]
        items = self._learning_items(keep) + self._level_items(roots, keep, max_words)
        packed = pack_head(items, max_words) if pack == "head" else trim_to_word_limit(items, max_words)
        return "\n".join(packed)

    def _learning_items(self, keep: set[str]) -> list[str]:
        learnings: list[str] = []
        citations: dict[str, str] = {}
        seen: set[str] = set()
        for nid in self.order:
            if nid not in keep:
                continue
            n = self.nodes[nid]
            citations.update(n.citations)
            for learning in n.learnings:
                if learning not in seen:
                    seen.add(learning)
                    learnings.append(learning)
        return [f"{x} [Source: {citations[x]}]" if citations.get(x) else x for x in learnings]

    def _level_items(self, group: list[str], keep: set[str], max_words: int) -> list[str]:
        """What one deep_research() call over ``group`` returns as its context."""
        items = [self.nodes[n].context for n in group if self.nodes[n].context]
        for n in group:
            kids = [c for c in self.nodes[n].children if c in keep]
            if kids:
                items.extend(self._level_items(kids, keep, max_words))
        return trim_to_word_limit(items, max_words)

    def tokens(self, nodes: Iterable[str]) -> int:
        return sum(self.nodes[n].tokens for n in set(nodes))


PACKS = ("tail", "head")
# Version tag of ``writer_context``; reward tables record it so scores built
# under a different context rule are never mixed.
WRITER_CONTEXT_RULE = "live-legacy-v1"


def _fork_skill():
    """The fork's deep-research module. Replay uses the agent's own trimming and
    packing so the replayed writer input is the live one."""
    try:
        from gpt_researcher.skills import deep_research
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "replaying the writer context needs the gpt-researcher fork importable "
            "(scripts/bootstrap_third_party.sh, then pip install -e third_party/gpt-researcher)"
        ) from exc
    return deep_research


def trim_to_word_limit(items: list[str], max_words: int = MAX_CONTEXT_WORDS) -> list[str]:
    """Delegate to the agent's own trimming rule so replay and live runs match."""
    return _fork_skill().trim_context_to_word_limit(items, max_words=max_words)


def pack_head(items: list[str], max_words: int = MAX_CONTEXT_WORDS) -> list[str]:
    """GR_FINAL_PACK=head: the agent's own front-filling packer."""
    skill = _fork_skill()
    if not hasattr(skill, "pack_context_head"):
        raise RuntimeError("GR_FINAL_PACK=head needs pack_context_head from the fork's shreya/branch-replay branch")
    return skill.pack_context_head(items, max_words=max_words)
