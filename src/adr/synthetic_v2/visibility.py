"""Smoke4 teacher visibility assertions; local bounded constructor is authoritative."""
import json
from pydantic import Field
from adr.synthetic_v2.events import EventState, EventTrajectory, expand
from adr.synthetic_v2.schema import TeacherTrajectory


class VisibleEventState(EventState):
    visible_findings: list[str] = Field(description='Before candidates, explicitly list ALL findings currently visible in the last-four completed-card bounded observation, in card/finding order, after per-field caps and whole-view bounding. Empty initially. No raw-page, future or evicted findings. Local compiler checks exact equality.')


class VisibleEventTrajectory(EventTrajectory):
    states: list[VisibleEventState] = Field(min_length=1)


def visibility_views(trajectory: TeacherTrajectory):
    views=[]
    for state in trajectory.states:
        inputs=state.source.inputs(trajectory.original_question,'')
        context=json.loads(inputs['context_before'])
        views.append({'state_id':state.state_id,'context_before':inputs['context_before'],
            'observation':context,'visible_findings':[f for c in context['prior_results'] for f in c['findings']],
            'diagnostics':inputs['observation_diagnostics']})
    return views


def expand_visible(events: VisibleEventTrajectory):
    # Only teacher visibility assertions are removed; no semantic events repaired.
    data=events.model_dump()
    for state in data['states']:state.pop('visible_findings')
    trajectory=expand(EventTrajectory.model_validate(data))
    views=visibility_views(trajectory)
    for state,view in zip(events.states,views,strict=True):
        if state.visible_findings!=view['visible_findings']:
            raise ValueError(f'{state.state_id}: declared visible_findings differs from locally bounded view')
    return trajectory,views
