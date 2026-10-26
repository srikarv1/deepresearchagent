"""Production teacher search events: derive cumulative searches, never guess them."""
from pydantic import Field
from adr.synthetic.schema import StrictModel, BranchState
from adr.synthetic_v2.events import RoundPublication
from adr.synthetic_v2.schema import Supervision
from adr.synthetic_v2.visibility import VisibleEventTrajectory, expand_visible

EVENT_VERSION = 'synthetic-v2-production-events-v1'


class Usage(StrictModel):
    planning_searches_since_previous_state: int = Field(ge=0,strict=True,
        description='Number of actual outer/planning search attempts since previous candidate batch (or before initial state). Candidate EXECUTEs are counted locally, never include them here. No unrecorded other search activity.')
    used_reads: int = Field(ge=0,strict=True,description='Actual cumulative URL read attempts, including failed reads.')
    used_tokens: int = Field(ge=0,strict=True,description='Actual cumulative research input+output tokens, not teacher tokens; nominal 400000 permits overruns.')


class State(StrictModel):
    state_id: str = Field(min_length=1)
    current_branch_id: str = Field(min_length=1)
    branches: list[BranchState]
    publications_before: list[RoundPublication]
    usage: Usage
    visible_findings: list[str] = Field(description='Exact ordered findings in current locally bounded view; never future/raw-page/evicted findings. Compiler verifies assertion.')
    candidates: list[Supervision] = Field(min_length=1)


class ProductionTrajectory(StrictModel):
    original_question: str = Field(min_length=1,description=VisibleEventTrajectory.model_fields['original_question'].description)
    states: list[State] = Field(min_length=1)


def expand_production(events):
    data=events.model_dump();searches=0;prior_executes=0
    for state in data['states']:
        usage=state['usage']
        searches+=prior_executes+usage.pop('planning_searches_since_previous_state')
        usage['used_searches']=searches
        prior_executes=sum(c['label']=='EXECUTE' for c in state['candidates'])
    # Unchanged visibility, lifecycle, history, budget and publication validators.
    return expand_visible(VisibleEventTrajectory.model_validate(data))
