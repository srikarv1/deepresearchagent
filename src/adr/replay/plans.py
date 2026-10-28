"""Exact plan space of a ReplayGym episode.

An episode (``rl.EpisodeState``) is a short sequence of binary "research this
branch deeper?" decisions over one cached research tree, and its outcome is the
researched sub-tree. A depth-3 tree with k top-level branches has at most 3**k
reachable sub-trees ("plans": 81 for breadth 4), so every plan, the decisions
that realize it, and the distinct decision states can be enumerated once per
question. That turns a policy's expected reward into an exact finite sum
(``grpo.py``) and makes every plan scorable ahead of training (``table.py``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from adr.replay.rl import EpisodeState, RewardService
from adr.replay.tree import ResearchTree

FIXED_RULES = ("d1", "d2", "d3")  # never expand | open children only | always expand
MAX_PLANS = 4096


@dataclass(frozen=True)
class DecisionState:
    """One distinct decision point.

    Histories that reach the same (level, group, researched) give identical
    features and prompts, so a policy that conditions only on those acts
    identically there; such histories share one state index.
    """

    level: int                  # 2 = open a branch's children, 3 = its grandchildren
    group: tuple[str, ...]      # nodes the decision would research
    researched: frozenset[str]  # nodes already researched when deciding
    features: np.ndarray        # EpisodeState.features()
    prompt: str = ""            # lm_policy.decision_prompt(), when requested


@dataclass
class PlanSpace:
    qid: str
    states: list[DecisionState]
    plans: list[frozenset[str]]                # researched sub-tree of each plan
    paths: list[tuple[tuple[int, int], ...]]   # (state index, action) per decision
    tokens: np.ndarray                         # research tokens of each plan
    full_tokens: int                           # research tokens of the whole tree

    @property
    def keys(self) -> list[str]:
        return [RewardService.key(p) for p in self.plans]

    @property
    def cost(self) -> np.ndarray:
        """Research tokens as a fraction of the full tree, in (0, 1]."""
        return self.tokens / float(max(1, self.full_tokens))

    def path_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(state index, action, mask), each (n_plans, max decisions). Padding
        points at state 0 with mask 0."""
        width = max([len(p) for p in self.paths] + [1])
        idx = np.zeros((len(self.plans), width), np.int64)
        act = np.zeros((len(self.plans), width), np.float32)
        mask = np.zeros((len(self.plans), width), np.float32)
        for i, path in enumerate(self.paths):
            for j, (s, a) in enumerate(path):
                idx[i, j], act[i, j], mask[i, j] = s, a, 1.0
        return idx, act, mask

    def plan_where(self, choose: Callable[[int], int]) -> int:
        """Index of the plan whose every decision s takes action ``choose(s)``.
        Exactly one plan matches any deterministic decision rule."""
        for i, path in enumerate(self.paths):
            if all(a == choose(s) for s, a in path):
                return i
        raise RuntimeError(f"{self.qid}: no plan matches the decision rule")  # pragma: no cover

    def fixed_plan(self, rule: str) -> int:
        if rule not in FIXED_RULES:
            raise ValueError(f"rule must be one of {FIXED_RULES}, got {rule!r}")
        if rule == "d1":
            return self.plan_where(lambda s: 0)
        if rule == "d3":
            return self.plan_where(lambda s: 1)
        return self.plan_where(lambda s: int(self.states[s].level == 2))

    def greedy_plan(self, log_odds: np.ndarray) -> int:
        """The plan a deployed policy picks: expand iff p >= 0.5, the live
        GR_REPLAYGYM_THRESHOLD default."""
        go = np.asarray(log_odds, dtype=np.float64) >= 0.0
        return self.plan_where(lambda s: int(go[s]))


def _clone(st: EpisodeState) -> EpisodeState:
    c = copy.copy(st)
    c.researched = set(st.researched)
    c.queue = list(st.queue)
    c._expand = list(st._expand)
    return c


def enumerate_plans(
    qid: str,
    tree: ResearchTree,
    emb: dict[str, np.ndarray],
    *,
    n_full: int | None = None,
    prompt_fn: Callable[[EpisodeState], str] | None = None,
    max_plans: int = MAX_PLANS,
) -> PlanSpace:
    """Every reachable plan of one question, depth first over both actions."""
    states: list[DecisionState] = []
    index: dict[tuple, int] = {}
    plans: list[frozenset[str]] = []
    paths: list[tuple[tuple[int, int], ...]] = []

    stack: list[tuple[EpisodeState, tuple[tuple[int, int], ...]]] = [
        (EpisodeState(qid, tree, emb, n_full=n_full), ())
    ]
    while stack:
        st, path = stack.pop()
        if st.done:
            plans.append(st.subtree())
            paths.append(path)
            if len(plans) > max_plans:
                raise ValueError(f"{qid}: more than {max_plans} plans; the tree is too wide to enumerate")
            continue
        level, group = st.queue[0]
        key = (level, tuple(group), frozenset(st.researched))
        i = index.get(key)
        if i is None:
            i = index[key] = len(states)
            states.append(
                DecisionState(
                    level=level,
                    group=tuple(group),
                    researched=frozenset(st.researched),
                    features=st.features(),
                    prompt=prompt_fn(st) if prompt_fn is not None else "",
                )
            )
        for a in (1, 0):
            nxt = _clone(st)
            nxt.step(a)
            stack.append((nxt, path + ((i, a),)))

    if len(set(plans)) != len(plans):  # pragma: no cover - guarded by construction
        raise RuntimeError(f"{qid}: two decision paths produced the same sub-tree")
    order = sorted(range(len(plans)), key=lambda k: paths[k])
    plans = [plans[k] for k in order]
    paths = [paths[k] for k in order]
    return PlanSpace(
        qid=qid,
        states=states,
        plans=plans,
        paths=paths,
        tokens=np.asarray([tree.tokens(p) for p in plans], dtype=np.float64),
        full_tokens=int(tree.tokens(tree.all_nodes())),
    )
