"""Draft3 teacher event encoding; expand before applying every draft2 invariant."""
from pydantic import Field
from adr.synthetic.schema import StrictModel, BranchState
from adr.synthetic_v2.schema import CompletedResult, Supervision, TeacherTrajectory

EVENT_VERSION = 'synthetic-v2-events-v1'
MAXIMA = dict(max_steps=32, max_searches=64, max_reads=128, max_tokens=400000)


class Page(StrictModel):
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000, description='Feature-only extracted page prefix; never semantic input. Title empty and retained=true are local constants.')


class RoundPublication(StrictModel):
    results: list[CompletedResult] = Field(min_length=1, description='New cards ONLY, one per done task in planner order. Local compiler preserves all earlier cards.')
    evidence: list[Page] = Field(description='New page identities ONLY, published at this same prior boundary.')


class Usage(StrictModel):
    used_searches: int = Field(ge=0, strict=True, description='Cumulative actual attempts including planning, not just EXECUTE count.')
    used_reads: int = Field(ge=0, strict=True, description='Cumulative attempted URL reads.')
    used_tokens: int = Field(ge=0, strict=True, description='Cumulative research input+output, not teacher tokens or context-window size. Overruns allowed.')


class EventState(StrictModel):
    state_id: str = Field(min_length=1)
    current_branch_id: str = Field(min_length=1)
    branches: list[BranchState] = Field(description='Full actual task registry in stable planner order; preserve IDs/goals/status lifecycle. One active current task; open tasks supply pending needs.')
    publications_before: list[RoundPublication] = Field(description='Prior completed rounds since previous candidate batch. Empty initially. No current or unpublished sibling output.')
    usage: Usage
    candidates: list[Supervision] = Field(min_length=1)


class EventTrajectory(StrictModel):
    original_question: str = Field(min_length=1, description=TeacherTrajectory.model_fields['original_question'].description, examples=TeacherTrajectory.model_fields['original_question'].examples)
    states: list[EventState] = Field(min_length=1)


def expand(events: EventTrajectory) -> TeacherTrajectory:
    """No repair: deterministic materialization followed by unchanged strict checks."""
    cards=[];pages=[];history=[];decisions=0;rounds=0;states=[]
    for item in events.states:
        publications=[]
        for pub in item.publications_before:
            cards.extend(c.model_dump() for c in pub.results)
            pages.extend({**p.model_dump(),'title':'','retained':True} for p in pub.evidence)
            publications.append({'task_ids':[c.branch_id for c in pub.results],
                                 'evidence_ids':[p.id for p in pub.evidence]})
            rounds+=1
        states.append({'state_id':item.state_id,'publications_before':publications,
            'source':{'current_branch_id':item.current_branch_id,'completed_results':list(cards),
                      'state':{'branches':[b.model_dump() for b in item.branches],
                               'evidence':list(pages),'executed_subqueries':list(history),
                               'budget':{**MAXIMA,**item.usage.model_dump(),'used_steps':decisions+rounds}}},
            'candidates':[c.model_dump() for c in item.candidates]})
        history.extend(c.candidate_subquery for c in item.candidates if c.label=='EXECUTE')
        decisions+=len(item.candidates)
    return TeacherTrajectory.model_validate({'original_question':events.original_question,'states':states})


def encode_fixture(trajectory: TeacherTrajectory) -> EventTrajectory:
    """Offline fixture conversion, not a v1-data migration or teacher-output repair."""
    states=[]
    for item in trajectory.states:
        cards={c.branch_id:c for c in item.source.completed_results}
        pages={p.id:p for p in item.source.state.evidence}
        states.append({'state_id':item.state_id,'current_branch_id':item.source.current_branch_id,
            'branches':[b.model_dump() for b in item.source.state.branches],
            'publications_before':[{'results':[cards[t].model_dump() for t in pub.task_ids],
                'evidence':[{k:getattr(pages[e],k) for k in ('id','url','text')} for e in pub.evidence_ids]}
                for pub in item.publications_before],
            'usage':{k:getattr(item.source.state.budget,k) for k in ('used_searches','used_reads','used_tokens')},
            'candidates':[c.model_dump() for c in item.candidates]})
    return EventTrajectory.model_validate({'original_question':trajectory.original_question,'states':states})
