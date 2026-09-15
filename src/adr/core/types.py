from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Discrete orchestration actions in the PILOT decision problem."""

    PLAN = "plan"
    SEARCH = "search"
    READ = "read"
    PRUNE = "prune"
    ALLOCATE = "allocate"
    WRITE = "write"
    TERMINATE = "terminate"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class Budget(BaseModel):
    """Hard caps for a single research trajectory."""

    max_steps: int = 12
    max_searches: int = 6
    max_reads: int = 8
    max_tokens: int = 24_000
    max_latency_s: float = 180.0
    max_evidence: int = 24

    used_steps: int = 0
    used_searches: int = 0
    used_reads: int = 0
    used_tokens: int = 0
    used_latency_s: float = 0.0

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.used_steps)

    def remaining_searches(self) -> int:
        return max(0, self.max_searches - self.used_searches)

    def remaining_reads(self) -> int:
        return max(0, self.max_reads - self.used_reads)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def remaining_latency_s(self) -> float:
        return max(0.0, self.max_latency_s - self.used_latency_s)

    def token_frac(self) -> float:
        return 0.0 if self.max_tokens <= 0 else min(1.0, self.used_tokens / self.max_tokens)

    def step_frac(self) -> float:
        return 0.0 if self.max_steps <= 0 else min(1.0, self.used_steps / self.max_steps)

    def exhausted(self) -> bool:
        return (
            self.remaining_steps() <= 0
            or self.remaining_tokens() <= 0
            or self.remaining_latency_s() <= 0
        )

    def charge(
        self,
        *,
        steps: int = 0,
        searches: int = 0,
        reads: int = 0,
        tokens: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        self.used_steps += steps
        self.used_searches += searches
        self.used_reads += reads
        self.used_tokens += tokens
        self.used_latency_s += latency_s


class Query(BaseModel):
    id: str
    text: str
    dataset: str
    language: str = "en"
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchTask(BaseModel):
    query: Query
    budget: Budget = Field(default_factory=Budget)


class Subtask(BaseModel):
    id: str
    goal: str
    parent_id: str | None = None
    status: Literal["open", "active", "done", "dropped"] = "open"
    searches: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    priority: float = 1.0
    notes: str = ""


class Evidence(BaseModel):
    id: str
    url: str
    title: str = ""
    snippet: str = ""
    text: str | None = None
    query: str = ""
    subtask_id: str | None = None
    score: float = 0.0
    retained: bool = True
    superseded_by: str | None = None
    added_step: int = 0
    source_backend: str = "unknown"

    def body(self, max_chars: int = 2_000) -> str:
        raw = self.text or self.snippet or ""
        return raw[:max_chars]


class OrchestratorAction(BaseModel):
    type: ActionType
    subtask_id: str | None = None
    query: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    rationale: str = ""
    report_draft: str | None = None


class StepRecord(BaseModel):
    step: int
    action: OrchestratorAction
    stats_before: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    article: str
    citations: list[str] = Field(default_factory=list)
    title: str | None = None


class Trajectory(BaseModel):
    query: Query
    steps: list[StepRecord] = Field(default_factory=list)
    report: Report | None = None
    final_stats: dict[str, Any] = Field(default_factory=dict)
    # Privileged labels (qrels, first-seen answer). Never copy into stats_before.
    labels: dict[str, Any] | None = None
    error: str | None = None

    def total_tokens(self) -> int:
        return sum(step.tokens.total_tokens for step in self.steps)

    def total_latency_s(self) -> float:
        return sum(step.latency_s for step in self.steps)
