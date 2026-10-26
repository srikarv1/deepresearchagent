"""Author facts once; locally replay lifecycle, accounting and exact observations.

This is not a provider runner. No labels enter world replay. Each Batch freezes
all candidates before any of their execution/results become visible.
"""
from typing import Annotated, Literal
from hashlib import sha256
import json
from pydantic import BaseModel, ConfigDict, Field
from adr.mlp.branch_observation_v2 import (
    VERSION as OBSERVATION_VERSION, Task, PublishedBranch, ObservationSource, model_inputs,
)
from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot
from adr.synthetic.schema import BranchState, EvidenceState
from adr.synthetic_v2.action_space import check_retrieval, check_followups

VERSION = 'synthetic-branch-source-v1'
MAXIMA = dict(max_steps=32, max_searches=64, max_reads=128, max_tokens=400000)

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class TaskSpec(Strict):
    id: str = Field(min_length=1)
    parent_id: str | None = None
    goal: str = Field(min_length=1)

class Plan(Strict):
    kind: Literal['plan'] = 'plan'
    tasks: list[TaskSpec] = Field(min_length=1)
    planning_queries: list[str] = Field(default_factory=list)
    research_tokens: int = Field(ge=0, strict=True)

class Candidate(Strict):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)

class Batch(Strict):
    kind: Literal['batch'] = 'batch'
    task_id: str = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1)

class FeaturePage(Strict):
    """Feature-ledger excerpt, NOT an annotation source or context component."""
    id: str = Field(min_length=1)
    url: str
    text: str = Field(min_length=1, max_length=2000)

class EndBatch(Strict):
    kind: Literal['end_batch'] = 'end_batch'
    # Proposal IDs in proposal order, not independently restated query strings.
    executed_ids: list[str]
    pages: list[FeaturePage] = Field(default_factory=list)
    read_attempts: int = Field(ge=0, strict=True)
    research_tokens: int = Field(ge=0, strict=True)

class Process(Strict):
    kind: Literal['process'] = 'process'
    task_id: str = Field(min_length=1)
    findings: list[str]
    followups: list[str]
    cited_insights: dict[str, str]
    research_tokens: int = Field(ge=0, strict=True)

class Drop(Strict):
    kind: Literal['drop'] = 'drop'
    task_id: str = Field(min_length=1)

class Publish(Strict):
    kind: Literal['publish'] = 'publish'
    task_ids: list[str] = Field(min_length=1)

Event = Annotated[Plan | Batch | EndBatch | Process | Drop | Publish, Field(discriminator='kind')]

class World(Strict):
    version: Literal['synthetic-branch-source-v1'] = VERSION
    question_id: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    events: list[Event]


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compile_world(world: World, *, allow_prefix=False):
    """Return annotation packets and separate audit records, never training labels.

    Task results/pages/history become available together only on publication.
    Numerical branches include all known tasks; evidence includes the full
    published ledger plus prior completed batches of the current task.
    """
    tasks, statuses, depths, results, published = {}, {}, {}, {}, []
    history, pages, packets, audit, execution = [], {}, {}, {}, {}
    searches = reads = tokens = steps = rounds = 0
    active = batch = None
    plan_groups = {}
    for index, event in enumerate(world.events):
        if batch is not None and not isinstance(event, EndBatch):
            raise ValueError('A frozen batch must end before any other event')
        if isinstance(event, Plan):
            if active is not None:
                raise ValueError('Planning requires no active task')
            group_ids=[t.id for t in event.tasks]
            for task in event.tasks:
                plan_groups[task.id]=group_ids
                if task.id in tasks:
                    raise ValueError('Task authored more than once')
                if task.parent_id is not None and task.parent_id not in published:
                    raise ValueError('Parent must be published before child planning')
                tasks[task.id] = task
                depths[task.id] = 1 if task.parent_id is None else depths[task.parent_id] + 1
                statuses[task.id] = 'open'
            searches += len(event.planning_queries)
            tokens += event.research_tokens
        elif isinstance(event, Batch):
            if event.task_id not in tasks or statuses[event.task_id] not in ('open', 'active'):
                raise ValueError('Cannot execute completed/dropped/unknown task')
            if active is not None and active != event.task_id:
                raise ValueError('Only one active task is supported')
            active = event.task_id
            statuses[active] = 'active'
            available = set(published) | {active}
            source = ObservationSource(current_task_id=active, current_need=tasks[active].goal,
                tasks=tuple(Task(id=t.id, parent_id=t.parent_id, tree_depth=depths[t.id]) for t in tasks.values()),
                published_branches=tuple(results[t] for t in published),
                executed_queries=tuple(q for owner, q in history if owner in available))
            state = UnenforcedLiveSnapshot(
                branches=[BranchState(id=t.id, goal=t.goal, status=statuses[t.id] if statuses[t.id]!='processed' else 'done') for t in tasks.values()],
                evidence=[EvidenceState(id=p.id, url=p.url, text=p.text, retained=True) for owner,p in pages.values() if owner in available],
                executed_subqueries=list(source.executed_queries),
                budget=UnenforcedLiveBudget(**MAXIMA,used_steps=steps+rounds,used_searches=searches,used_reads=reads,used_tokens=tokens))
            for c in event.candidates:
                if c.id in packets:
                    raise ValueError('Duplicate candidate identity')
                check_retrieval(c.query)
                observation, features = model_inputs(world.original_question, c.query, source, state)
                packet = dict(original_question=world.original_question, context_before=observation.text,
                              candidate_subquery=c.query, numerical_features=features.model_dump())
                packets[c.id] = packet
                audit[c.id] = dict(source_version=VERSION, observation_version=OBSERVATION_VERSION,
                    world_digest=digest(world.model_dump()), question_id=world.question_id,
                    event_index=index, task_id=active, tree_depth=depths[active], packet_digest=digest(packet),
                    selected_task_ids=list(observation.selected_task_ids), scores=list(observation.scores),
                    tokens=observation.tokens, overflow=observation.overflow,
                    required_bypass_decision=observation.required_bypass_decision,
                    published_count=len(published), published_task_ids=list(published), history_count=len(source.executed_queries),
                    branch_count=len(tasks), budget=state.budget.model_dump())
            batch = event
        elif isinstance(event, EndBatch):
            if batch is None:
                raise ValueError('No batch to end')
            ids = [c.id for c in batch.candidates]
            if event.executed_ids != [i for i in ids if i in event.executed_ids] or len(set(event.executed_ids)) != len(event.executed_ids):
                raise ValueError('Execution IDs must be a unique proposal-order subset')
            if (event.pages or event.read_attempts) and not event.executed_ids:
                raise ValueError('Retrieval without an executed query')
            if event.read_attempts < len(event.pages):
                raise ValueError('Read accounting cannot explain new page records')
            for c in batch.candidates:
                execution[c.id] = c.id in event.executed_ids
                if execution[c.id]:
                    history.append((active, c.query))
            for p in event.pages:
                if p.id in pages:
                    raise ValueError('Feature page authored more than once')
                pages[p.id] = (active, p)
            searches += len(event.executed_ids)
            reads += event.read_attempts
            tokens += event.research_tokens
            steps += len(batch.candidates)
            batch = None
        elif isinstance(event, Process):
            if event.task_id != active or statuses[active] != 'active':
                raise ValueError('Only active task may produce a result')
            if (event.findings or event.cited_insights) and not any(owner==active for owner,_ in history):
                raise ValueError('Semantic result has no prior executed research query')
            result = PublishedBranch(task_id=active,goal=tasks[active].goal,
                findings=event.findings,followups=event.followups,cited_insights=event.cited_insights)
            check_followups([result])
            if any(not k.strip() or not v.startswith(('https://','http://')) for k,v in result.cited_insights.items()):
                raise ValueError('Cited insights need text and source URL identities')
            results[active] = result
            statuses[active] = 'processed'
            tokens += event.research_tokens
            active = None
        elif isinstance(event, Drop):
            if event.task_id not in tasks or statuses[event.task_id] not in ('open','active'):
                raise ValueError('Invalid task drop')
            statuses[event.task_id]='dropped'
            if active == event.task_id:
                active=None
        elif isinstance(event, Publish):
            if active is not None or len(set(event.task_ids)) != len(event.task_ids):
                raise ValueError('Invalid publication boundary')
            group=plan_groups.get(event.task_ids[0], [])
            expected=[t for t in group if statuses[t]!='dropped']
            if event.task_ids != expected:
                raise ValueError('Publication must commit the complete planner round')
            if event.task_ids != [t for t in tasks if t in event.task_ids]:
                raise ValueError('Publication must preserve planner order')
            for tid in event.task_ids:
                if statuses.get(tid) != 'processed':
                    raise ValueError('Only processed unpublished results may publish')
                published.append(tid)
                statuses[tid] = 'done'
            rounds += 1
    if not allow_prefix and (batch is not None or active is not None):
        raise ValueError('Incomplete source lifecycle')
    return packets, audit, execution
