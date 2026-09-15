"""Harness-side measurement of what an agent actually spends.

Agents are free to log their own token counts, but the headline cost numbers
should not depend on them doing it correctly. These wrappers sit between the
agent and its dependencies so every model call and every retrieval is counted
whether or not the agent reports it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from adr.core.types import Budget, TokenUsage

if TYPE_CHECKING:  # avoids a cycle: tools.search imports core.state
    from adr.llm.base import LLMResponse, Message
    from adr.tools.search import SearchHit


class BudgetExceeded(RuntimeError):
    """Raised when an agent overruns its budget and enforcement is on."""


def _corpus_docid(url: str) -> str | None:
    # Local import: tools.browsecomp_plus imports tools.search, which imports core.state.
    from adr.tools.browsecomp_plus import docid_from_url

    return docid_from_url(url)


@dataclass
class CostMeter:
    """Running totals for one query."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    n_calls: int = 0
    llm_latency_s: float = 0.0
    n_search_calls: int = 0
    n_fetch_calls: int = 0
    search_latency_s: float = 0.0
    retrieved_chars: int = 0
    # Corpus docids seen in search hits (bcp:// URLs from the BrowseComp-Plus
    # backend). Not part of snapshot(); the runner writes it to
    # final_stats["retrieved_docids"] for upstream's Recall.
    retrieved_docids: set[str] = field(default_factory=set)
    violations: list[str] = field(default_factory=list)

    def add_llm(self, usage: TokenUsage, latency_s: float) -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        # Some servers omit total_tokens; fall back to the parts.
        self.total_tokens += usage.total_tokens or (
            usage.prompt_tokens + usage.completion_tokens
        )
        self.n_calls += 1
        self.llm_latency_s += latency_s

    def note(self, message: str) -> None:
        if message not in self.violations:
            self.violations.append(message)

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "n_calls": self.n_calls,
            "llm_latency_s": round(self.llm_latency_s, 4),
            "n_search_calls": self.n_search_calls,
            "n_fetch_calls": self.n_fetch_calls,
            "search_latency_s": round(self.search_latency_s, 4),
            "retrieved_chars": self.retrieved_chars,
        }


class MeteredLLM:
    """Counts tokens and latency for every completion, then charges the budget."""

    def __init__(
        self,
        inner: Any,
        meter: CostMeter,
        *,
        budget: Budget | None = None,
        enforce: bool = False,
    ) -> None:
        self._inner = inner
        self._meter = meter
        self._budget = budget
        self._enforce = enforce
        self.name = getattr(inner, "name", "llm")
        self.model = getattr(inner, "model", "unknown")

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._budget is not None and self._budget.remaining_tokens() <= 0:
            message = "token budget exhausted before an LLM call"
            self._meter.note(message)
            if self._enforce:
                raise BudgetExceeded(message)

        started = time.perf_counter()
        response = await self._inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, extra=extra
        )
        elapsed = time.perf_counter() - started

        self._meter.add_llm(response.usage, elapsed)
        if self._budget is not None:
            self._budget.charge(tokens=response.usage.total_tokens, latency_s=elapsed)
            if self._budget.remaining_tokens() <= 0:
                self._meter.note(
                    f"token budget exceeded: used {self._budget.used_tokens} "
                    f"of {self._budget.max_tokens}"
                )
        return response


class MeteredSearch:
    """Counts retrieval calls and latency, then charges the budget."""

    def __init__(
        self,
        inner: Any,
        meter: CostMeter,
        *,
        budget: Budget | None = None,
        enforce: bool = False,
    ) -> None:
        self._inner = inner
        self._meter = meter
        self._budget = budget
        self._enforce = enforce
        self.name = getattr(inner, "name", "search")

    def _check(self, kind: str, remaining: int) -> None:
        if remaining > 0:
            return
        message = f"{kind} budget exhausted"
        self._meter.note(message)
        if self._enforce:
            raise BudgetExceeded(message)

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if self._budget is not None:
            self._check("search", self._budget.remaining_searches())
        started = time.perf_counter()
        hits = await self._inner.search(query, k=k)
        elapsed = time.perf_counter() - started

        self._meter.n_search_calls += 1
        self._meter.search_latency_s += elapsed
        self._meter.retrieved_chars += sum(
            len(hit.text or hit.snippet or "") for hit in hits
        )
        for hit in hits:
            docid = _corpus_docid(hit.url)
            if docid:
                self._meter.retrieved_docids.add(docid)
        if self._budget is not None:
            self._budget.charge(searches=1, latency_s=elapsed)
        return hits

    async def fetch(self, url: str) -> str:
        if self._budget is not None:
            self._check("read", self._budget.remaining_reads())
        started = time.perf_counter()
        text = await self._inner.fetch(url)
        elapsed = time.perf_counter() - started

        self._meter.n_fetch_calls += 1
        self._meter.search_latency_s += elapsed
        self._meter.retrieved_chars += len(text or "")
        if self._budget is not None:
            self._budget.charge(reads=1, latency_s=elapsed)
        return text
