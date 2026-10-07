"""Exact Shapley and Banzhaf values of research branches from coalition payoffs."""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Hashable, Mapping, Sequence


def all_coalitions(players: Sequence[Hashable]) -> list[frozenset]:
    return [frozenset(c) for r in range(len(players) + 1) for c in combinations(players, r)]


def shapley(players: Sequence[Hashable], v: Mapping[frozenset, float]) -> dict[Hashable, float]:
    """phi_i = sum_{S ⊆ N\\{i}} |S|!(n-|S|-1)!/n! * (v(S∪{i}) - v(S))."""
    n = len(players)
    out: dict[Hashable, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for c in combinations(others, r):
                s = frozenset(c)
                total += w * (v[s | {i}] - v[s])
        out[i] = total
    return out


def banzhaf(players: Sequence[Hashable], v: Mapping[frozenset, float]) -> dict[Hashable, float]:
    """beta_i = mean over S ⊆ N\\{i} of (v(S∪{i}) - v(S)); robust to noisy v."""
    n = len(players)
    out: dict[Hashable, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        diffs = [v[frozenset(c) | {i}] - v[frozenset(c)] for r in range(n) for c in combinations(others, r)]
        out[i] = sum(diffs) / len(diffs)
    return out
