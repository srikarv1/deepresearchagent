"""Versioned v2 source-state semantics; v1 models/artifacts are not accepted implicitly."""
import re
from typing import Literal
from pydantic import Field, model_validator
from adr.synthetic.schema import StrictModel, Candidate, derive_features, FEATURE_NAMES
from adr.mlp.live_context import UnenforcedLiveSnapshot, UnenforcedLiveBudget
from adr.mlp.observation_v2 import ObservationInput, ResultCard, construct
from adr.synthetic_v2.action_space import check_retrieval

FEATURE_VERSION='mlp-features-v2-source-v1'
SCHEMA_VERSION='synthetic-v2'

class CompletedResult(ResultCard):
    branch_id: str = Field(min_length=1, description='Unique outer task ID. One card per task within a state; task must be done and its goal must match this card exactly. Persist this same card unchanged in later cumulative source states, even after observation-window eviction.')
    goal: str = Field(min_length=1, description='Exact research goal of the done outer task identified by branch_id, not a new summary heading.')
    findings: list[str] = Field(default_factory=list,max_length=3)
    followups: list[str] = Field(default_factory=list,max_length=3)

class SourceState(StrictModel):
    current_branch_id: str
    state: UnenforcedLiveSnapshot
    completed_results: list[CompletedResult] = Field(default_factory=list)

    @model_validator(mode='after')
    def lifecycle(self):
        required={f'{prefix}_{r}' for prefix in ('max','used') for r in ('steps','searches','reads','tokens')}
        if not required.issubset(self.state.budget.model_fields_set):
            raise ValueError('V2 requires explicit nominal maxima and actual usage; no inherited v1 budget defaults')
        branches={b.id:b for b in self.state.branches}
        if self.current_branch_id not in branches or branches[self.current_branch_id].status!='active':
            raise ValueError('Current research task must exist and be active')
        ids=[c.branch_id for c in self.completed_results]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate completed research result')
        for card in self.completed_results:
            if card.branch_id not in branches or branches[card.branch_id].status!='done' or card.goal!=branches[card.branch_id].goal:
                raise ValueError('Result must belong to a completed task with matching goal')
        for e in self.state.evidence:
            if not e.retained or e.title or len(e.text)>2000:
                raise ValueError('No-pruning feature pages require retained=true, empty title and <=2000 character prefix')
        if self.state.budget.used_searches<len(self.state.executed_subqueries):
            raise ValueError('Search attempts cannot be less than executed-query history')
        return self

    def observation(self):
        return ObservationInput(current_need=next(b.goal for b in self.state.branches if b.id==self.current_branch_id),
            pending_needs=[b.goal for b in self.state.branches if b.status=='open'],
            prior_results=[ResultCard(goal=c.goal,findings=c.findings,followups=c.followups) for c in self.completed_results],
            recent_queries=self.state.executed_subqueries)

    def inputs(self,question,candidate):
        context,diagnostics=construct(self.observation())
        return {'original_question':question,'context_before':context,'candidate_subquery':candidate,
                'numerical_features':derive_features(question,self.state).model_dump(),
                'observation_diagnostics':diagnostics,'feature_version':FEATURE_VERSION}

class TeacherBudget(UnenforcedLiveBudget):
    """Explicit experiment maxima, never inherited v1/context-window defaults."""
    max_steps: Literal[32] = Field(description='Nominal total-run steps; enforcement false.')
    max_searches: Literal[64]
    max_reads: Literal[128]
    max_tokens: Literal[400000] = Field(description='Cumulative research input+output tokens, NOT the separate 50K context budget. Overruns valid.')
    used_steps: int = Field(ge=0, strict=True, description='Prior candidate decisions plus published outer rounds.')
    used_searches: int = Field(ge=0, strict=True)
    used_reads: int = Field(ge=0, strict=True)
    used_tokens: int = Field(ge=0, strict=True)


class TeacherSnapshot(UnenforcedLiveSnapshot):
    budget: TeacherBudget


class TeacherSource(SourceState):
    state: TeacherSnapshot


class Publication(StrictModel):
    """Synthetic provenance only; never included in an embedding or features."""
    task_ids: list[str] = Field(min_length=1, description='Distinct completed outer tasks published together in this prior round, in planner order.')
    evidence_ids: list[str] = Field(description='New feature-page IDs first published at this same boundary, in stable order. No unpublished pages; empty allowed for no new pages.')


class Supervision(Candidate):
    supporting_finding: str | None = Field(default=None, min_length=1,
        description='Exact quote from a finding visible after local bounding; required for evidence-dependent reason categories and later counterfactual SKIPs. Never quote hidden raw pages.')

    @model_validator(mode='after')
    def retrieval_and_witness(self):
        check_retrieval(self.candidate_subquery)
        if re.search(r"\b(?:create|produce|make)\s+(?:a\s+)?comparative table|\b(?:formulate|propose)\s+(?:an?\s+)?(?:original,?\s+)?(?:specific\s+)?research question|^synthesize\s+evidence", self.candidate_subquery, re.I):
            raise ValueError('Candidate must be a retrieval query, not a synthesis/meta instruction')
        if self.decision_category in {'near_duplicate','already_established','unnecessary_confirmation',
                'weak_evidence_verification','conflicting_evidence_resolution'} and not self.supporting_finding:
            raise ValueError('Evidence-dependent category requires a visible supporting_finding')
        return self

class TeacherState(StrictModel):
    state_id: str
    source: TeacherSource
    publications_before: list[Publication] = Field(description='All outer-round publications since the previous sampled candidate batch. Empty initially. New cards/pages must appear together here, never current-batch results.')
    candidates: list[Supervision] = Field(min_length=1)

class TeacherTrajectory(StrictModel):
    original_question: str = Field(min_length=1, description='An actual specific comparative research question to answer. Do not copy the topic or instructions asking you to formulate a question.', examples=['Under which operating conditions do published studies support hybrid cooling over passive ventilation for stationary LFP containers in arid climates?'])
    states: list[TeacherState] = Field(min_length=1)

    @model_validator(mode='after')
    def cumulative(self):
        if len({s.state_id for s in self.states})!=len(self.states):
            raise ValueError('Duplicate state ID')
        if re.search(r"\b(?:formulate|propose)\s+(?:an?\s+)?(?:original,?\s+)?(?:specific\s+)?research question", self.original_question, re.I):
            raise ValueError('original_question must be the research question, not a meta instruction')
        first=self.states[0]
        if first.source.completed_results or first.source.state.evidence or first.source.state.executed_subqueries or first.publications_before:
            raise ValueError('Initial trajectory state must have no prior results, evidence, executed history or publications')
        if any(b.status not in ('open','active') for b in first.source.state.branches):
            raise ValueError('Operational task lifecycle must start with planned/open or active tasks')
        if first.source.state.budget.used_steps!=0:
            raise ValueError('Initial steps must be zero; planning calls do not count as candidate decisions')
        expected_history=[]
        expected_steps=0
        previously_active=set()
        for index,item in enumerate(self.states):
            state=item.source.state
            if index:
                previous=self.states[index-1]
                expected_history.extend(c.candidate_subquery for c in previous.candidates if c.label=='EXECUTE')
                expected_steps+=len(previous.candidates)+len(item.publications_before)
                if state.budget.used_searches-previous.source.state.budget.used_searches < sum(c.label=='EXECUTE' for c in previous.candidates):
                    raise ValueError('Search usage must account for every prior EXECUTE')
                old_cards=previous.source.completed_results
                new_cards=item.source.completed_results[len(old_cards):]
                task_ids=[t for pub in item.publications_before for t in pub.task_ids]
                page_ids=[e for pub in item.publications_before for e in pub.evidence_ids]
                old_ids={e.id for e in previous.source.state.evidence}
                new_ids=[e.id for e in state.evidence if e.id not in old_ids]
                if len(task_ids)!=len(set(task_ids)) or task_ids!=[c.branch_id for c in new_cards]:
                    raise ValueError('Publication tasks must exactly match newly published cards once in planner order')
                if len(page_ids)!=len(set(page_ids)) or page_ids!=new_ids:
                    raise ValueError('New evidence pages and result cards must publish together at a recorded boundary')
                if any(t not in previously_active for t in task_ids):
                    raise ValueError('A published task must have had an earlier sampled candidate batch')
                # Sibling reads may precede this boundary by several candidate batches.
                # Publication exposes their pages together; do not charge those reads twice.
                if state.budget.used_reads < len(state.evidence):
                    raise ValueError('Read accounting must cover cumulative published pages')
            if state.executed_subqueries!=expected_history:
                raise ValueError('Executed history must equal all prior EXECUTE candidates in order; SKIPs add nothing')
            if state.budget.used_steps!=expected_steps:
                raise ValueError('Steps must equal prior candidate decisions plus recorded published rounds')
            if sum(b.status=='active' for b in state.branches)!=1:
                raise ValueError('Synthetic trace requires one active outer task per candidate batch')
            previously_active.add(item.source.current_branch_id)
        for item in self.states:
            if len({c.candidate_id for c in item.candidates})!=len(item.candidates):
                raise ValueError('Duplicate candidate ID')
        for a,b in zip(self.states,self.states[1:]):
            old,new=a.source,b.source
            if new.completed_results[:len(old.completed_results)]!=old.completed_results:
                raise ValueError('Published result history must be cumulative and immutable')
            if new.state.executed_subqueries[:len(old.state.executed_subqueries)]!=old.state.executed_subqueries:
                raise ValueError('Executed search history must be cumulative')
            branches={b.id:b for b in new.state.branches}
            for prior in old.state.branches:
                current=branches.get(prior.id)
                if current is not None and prior.status=='open' and current.status not in ('open','active'):
                    raise ValueError('Operational task lifecycle forbids completion without an active sampled batch')
                if current is not None and prior.status=='active' and current.status=='open':
                    raise ValueError('Operational task lifecycle forbids active to open; preserve unresolved needs in followups or new tasks')
                if current is None or current.goal!=prior.goal or (prior.status in ('done','dropped') and current.status!=prior.status):
                    raise ValueError('Task identities/goals and completed statuses must persist')
            before={e.id:e for e in old.state.evidence}; after={e.id:e for e in new.state.evidence}
            if any(after.get(k)!=v for k,v in before.items()):
                raise ValueError('No-pruning page history must remain immutable')
            for r in ('steps','searches','reads','tokens'):
                if getattr(old.state.budget,'max_'+r)!=getattr(new.state.budget,'max_'+r) or getattr(old.state.budget,'used_'+r)>getattr(new.state.budget,'used_'+r):
                    raise ValueError('Nominal maxima must stay fixed; actual usage must be cumulative')
        return self
