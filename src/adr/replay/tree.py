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
        keep = set(nodes)
        ordered = [nid for nid in self.order if nid in keep]
        learnings: list[str] = []
        citations: dict[str, str] = {}
        seen: set[str] = set()
        for nid in ordered:
            n = self.nodes[nid]
            citations.update(n.citations)
            for l in n.learnings:
                if l not in seen:
                    seen.add(l)
                    learnings.append(l)
        items = [f"{l} [Source: {citations[l]}]" if citations.get(l) else l for l in learnings]
        items += [self.nodes[nid].context for nid in ordered if self.nodes[nid].context]
        return "\n".join(trim_to_word_limit(items))

    def tokens(self, nodes: Iterable[str]) -> int:
        return sum(self.nodes[n].tokens for n in set(nodes))


def trim_to_word_limit(items: list[str], max_words: int = MAX_CONTEXT_WORDS) -> list[str]:
    """Delegate to the agent's own trimming rule so replay and live runs match."""
    from gpt_researcher.skills.deep_research import trim_context_to_word_limit

    return trim_context_to_word_limit(items, max_words=max_words)
