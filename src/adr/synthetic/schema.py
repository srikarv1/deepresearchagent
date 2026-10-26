"""Versioned, pre-decision-only teacher and dataset contracts.

Feature semantics are delegated to ResearchState.compact_stats/Budget. Evidence
scores are deliberately zero: the GPT Researcher adapter has no useful score.
No teacher-supplied features, context summary, or outcome fields are accepted.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adr.core.state import ResearchState
from adr.core.types import Budget, Evidence, Query, Subtask

SCHEMA_VERSION = "synthetic-v1"
FEATURE_NAMES = (
    "n_evidence", "n_retained", "n_pruned", "n_open_branches", "n_branches",
    "mean_score", "mean_chars", "query_term_coverage", "unique_urls",
    "step_frac", "token_frac", "remaining_steps", "remaining_searches",
    "remaining_reads", "remaining_tokens",
)
EXECUTE_CATEGORIES = {
    "unanswered_information_need", "weak_evidence_verification",
    "conflicting_evidence_resolution", "necessary_dependency",
}
SKIP_CATEGORIES = {
    "already_established", "near_duplicate", "irrelevant", "low_marginal_value",
    "unnecessary_confirmation", "overly_specific", "premature",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class EvidenceState(StrictModel):
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = ""
    text: str = Field(min_length=1)
    retained: bool = True


class BranchState(StrictModel):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    status: Literal["open", "active", "done", "dropped"] = "open"


class BudgetState(StrictModel):
    """Cumulative prior usage. Tokens = input + output, not context-window size.

    Steps = prior orchestration decisions; searches/reads = tool calls. These
    are explicit synthetic counters, unlike the archive's uncharged counters.
    """
    max_steps: int = Field(default=32, gt=0, strict=True)
    max_searches: int = Field(default=64, gt=0, strict=True)
    max_reads: int = Field(default=128, gt=0, strict=True)
    max_tokens: int = Field(default=50000, gt=0, strict=True)
    used_steps: int = Field(default=0, ge=0, strict=True)
    used_searches: int = Field(default=0, ge=0, strict=True)
    used_reads: int = Field(default=0, ge=0, strict=True)
    used_tokens: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def bounds(self):
        for resource in ("steps", "searches", "reads", "tokens"):
            if getattr(self, f"used_{resource}") > getattr(self, f"max_{resource}"):
                raise ValueError(f"used_{resource} exceeds maximum")
        return self


class ResearchSnapshot(StrictModel):
    evidence: list[EvidenceState] = Field(default_factory=list)
    executed_subqueries: list[str] = Field(default_factory=list)
    branches: list[BranchState] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)

    @model_validator(mode="after")
    def unique_ids(self):
        for kind in ("evidence", "branches"):
            ids = [x.id for x in getattr(self, kind)]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} IDs")
        if any(not q.strip() for q in self.executed_subqueries):
            raise ValueError("empty executed subquery")
        return self


class Candidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    candidate_subquery: str = Field(min_length=1)
    label: Literal["EXECUTE", "SKIP"]
    decision_category: str
    reason: str = Field(min_length=1)
    # Linkage is metadata, never included in context/features.
    counterfactual_pair_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def category_matches(self):
        allowed = EXECUTE_CATEGORIES if self.label == "EXECUTE" else SKIP_CATEGORIES
        if self.decision_category not in allowed:
            raise ValueError("label/category mismatch")
        return self


class TeacherState(StrictModel):
    state_id: str = Field(min_length=1)
    state: ResearchSnapshot
    candidates: list[Candidate] = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_ids(self):
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate IDs within state")
        return self


class TeacherTrajectory(StrictModel):
    original_question: str = Field(min_length=1)
    states: list[TeacherState] = Field(min_length=1)

    @model_validator(mode="after")
    def successive_states(self):
        ids = [s.state_id for s in self.states]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate state IDs")
        for old, new in zip(self.states, self.states[1:]):
            for resource in ("steps", "searches", "reads", "tokens"):
                if getattr(old.state.budget, f"max_{resource}") != getattr(new.state.budget, f"max_{resource}"):
                    raise ValueError("budget maximum changed within trajectory")
                if getattr(old.state.budget, f"used_{resource}") > getattr(new.state.budget, f"used_{resource}"):
                    raise ValueError("budget usage went backwards")
            history = old.state.executed_subqueries
            if new.state.executed_subqueries[:len(history)] != history:
                raise ValueError("executed-subquery history is not cumulative")
            new_evidence = {e.id: e for e in new.state.evidence}
            for e in old.state.evidence:
                newer = new_evidence.get(e.id)
                if newer is None or (e.url, e.title, e.text) != (newer.url, newer.title, newer.text):
                    raise ValueError("prior evidence removed or rewritten; change retention instead")
        return self


def serialize_context(state: ResearchSnapshot) -> str:
    """Stable JSON; list order is historical. Never includes candidate/supervision."""
    return json.dumps(state.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class FeatureVector(StrictModel):
    names: list[str] = Field(default_factory=lambda: list(FEATURE_NAMES))
    values: list[float]
    mean_score_status: Literal["unavailable_zero_placeholder"] = "unavailable_zero_placeholder"

    @model_validator(mode="after")
    def ordered(self):
        if tuple(self.names) != FEATURE_NAMES or len(self.values) != len(FEATURE_NAMES):
            raise ValueError("feature order/dimensionality mismatch")
        return self


def derive_features(question: str, snapshot: ResearchSnapshot) -> FeatureVector:
    state = ResearchState(Query(id="synthetic", text=question, dataset="synthetic"),
                          Budget(**snapshot.budget.model_dump()))
    state.evidence = {e.id: Evidence(**e.model_dump(), score=0.0) for e in snapshot.evidence}
    state.subtasks = {b.id: Subtask(**b.model_dump()) for b in snapshot.branches}
    stats = state.compact_stats()
    flat = {**stats, **stats["budget"]}
    return FeatureVector(values=[flat[name] for name in FEATURE_NAMES])


class Provenance(StrictModel):
    schema_version: Literal["synthetic-v1"] = SCHEMA_VERSION
    prompt_version: str
    dataset_version: str
    generated_at: str
    teacher_provider: str
    teacher_model: str
    state_index: int = Field(ge=0)
    counterfactual_pair_id: str | None = None


class SyntheticExample(StrictModel):
    example_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    context_before: str = Field(min_length=1)
    candidate_subquery: str = Field(min_length=1)
    state: ResearchSnapshot
    numerical_features: FeatureVector
    label: Literal["EXECUTE", "SKIP"]
    decision_category: str
    reason: str = Field(min_length=1)
    provenance: Provenance

    @model_validator(mode="after")
    def derived_fields(self):
        Candidate(candidate_id=self.candidate_id, candidate_subquery=self.candidate_subquery,
                  label=self.label, decision_category=self.decision_category, reason=self.reason)
        if self.context_before != serialize_context(self.state):
            raise ValueError("context_before differs from deterministic pre-decision state")
        if self.numerical_features != derive_features(self.original_question, self.state):
            raise ValueError("numerical features differ from derived pre-decision state")
        return self
