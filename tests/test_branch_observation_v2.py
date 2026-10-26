import json
import socket
from pathlib import Path
import pytest
from adr.mlp.branch_observation_v2 import (
    VERSION, EMBEDDING_CONTRACT, Task, PublishedBranch, ObservationSource, construct, model_inputs,
)
from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot
from adr.synthetic.schema import EvidenceState, BranchState, derive_features, FEATURE_NAMES
from adr.development.replay_observation_v2 import reconstruct


def source():
    return ObservationSource(current_task_id='current', current_need='Full current need',
        tasks=(Task(id='root',tree_depth=1),Task(id='parent',parent_id='root',tree_depth=2),
               Task(id='current',parent_id='parent',tree_depth=3),Task(id='other',tree_depth=1)),
        published_branches=tuple(PublishedBranch(task_id=i,goal='sleep evidence',
            findings=('Unabridged finding ü <|endoftext|>',),followups=('Find published comparisons',),
            cited_insights={'exact insight':'https://example.org/'+i}) for i in ('root','parent','other')),
        executed_queries=tuple(f'prior query {i}' for i in range(10)))


def test_exact_schema_ties_hierarchy_and_determinism():
    s=source()
    o=construct(s,'unseenword')
    expected={'current_need':s.current_need,'recent_queries':list(s.executed_queries[-6:]),
        'branches':[{'goal':b.goal,'findings':list(b.findings),'followups':list(b.followups),
            'cited_insights':b.cited_insights,'relation':r,'tree_depth':d}
            for b,r,d in zip(s.published_branches,('ancestor','parent','other'),(1,2,1))]}
    assert o.text == json.dumps(expected,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    assert o == construct(s,'unseenword')
    assert o.selected_task_ids == ('root','parent','other')
    assert o.scores == (0.,0.,0.)
    assert not o.overflow and o.required_bypass_decision is None
    assert o.version == VERSION


def test_rank_does_not_use_citations_and_is_candidate_conditioned():
    s=source().model_copy(update={'published_branches':(
        PublishedBranch(task_id='root',goal='finance credit',cited_insights={'sleep sleep':'url'}),
        PublishedBranch(task_id='parent',goal='sleep duration'),
        PublishedBranch(task_id='other',goal='finance tax'))})
    assert construct(s,'sleep duration').selected_task_ids[0]=='parent'
    assert construct(s,'finance tax').selected_task_ids[0]=='other'


def test_empty_stopwords_and_full_overflow_no_truncation():
    s=source().model_copy(update={'published_branches':()})
    assert json.loads(construct(s,'query').text)['branches']==[]
    s=s.model_copy(update={'published_branches':(PublishedBranch(task_id='root',goal='the and'),)})
    assert construct(s,'and').scores==(0.,)
    s=s.model_copy(update={'current_need':'long need ' * 5000})
    o=construct(s,'query')
    assert o.overflow and o.required_bypass_decision=='EXECUTE'
    assert json.loads(o.text)['current_need']==s.current_need


def test_numerical_features_use_full_state_and_keep_overrun():
    state=UnenforcedLiveSnapshot(evidence=[EvidenceState(id=str(i),url=f'https://example.org/{i}',text='full evidence text') for i in range(8)],
        branches=[BranchState(id=str(i),goal='goal') for i in range(9)],
        budget=UnenforcedLiveBudget(used_tokens=500001,used_steps=99,used_reads=200,used_searches=100))
    observation,features=model_inputs('question','query',source(),state)
    assert features==derive_features('question',state)
    assert tuple(features.names)==FEATURE_NAMES
    assert features.values[0]==8 and features.values[4]==9
    assert features.values[9:11]==[1,1]
    assert features.values[11:]==[0,0,0,0]
    assert len(json.loads(observation.text)['branches'])==3
    assert EMBEDDING_CONTRACT['input_dimensions']==4623
    assert 'full evidence text' not in observation.text


def test_invalid_hierarchy_and_current_result_rejected():
    with pytest.raises(ValueError):
        source().model_copy().model_validate({**source().model_dump(),'current_task_id':'missing'})
    with pytest.raises(ValueError):
        ObservationSource(current_task_id='root',current_need='need',tasks=(Task(id='root',tree_depth=1),),
            published_branches=(PublishedBranch(task_id='root',goal='future result'),))

RUNS=[
'runs/representation-study/20260926-223740-development-representation-001',
'runs/representation-study-002/20260927-003826-development-representation-002',
'runs/representation-study-003/20260927-114734-development-representation-003']

@pytest.mark.parametrize('run',RUNS)
def test_real_development_replay(run,monkeypatch):
    def no_network(*args,**kwargs):
        raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    paths=list((Path(run)/'development_trace').glob('*.jsonl'))
    if not paths:
        pytest.skip('Local development artifacts not installed')
    outputs=reconstruct(paths[0])
    assert len(outputs)==112
    assert all(o.tokens<=4096 and not o.overflow for _,o in outputs)


def test_publication_barrier_rejects_unpublished_sibling_and_history(tmp_path):
    tasks=[{'id':i,'parent_id':None,'tree_depth':1,'goal':i} for i in ('current','sibling')]
    events=[{'event':'branch_created','branch':b} for b in tasks]
    events += [{'event':'record','record_id':'result','value':{'researchGoal':'hidden','learnings':['future finding'],
        'followUpQuestions':[],'citations':{}}},
        {'event':'branch_processed','branch_id':'sibling','result_id':'result'},
        {'event':'batch_snapshot','batch_id':'batch','branch_id':'current','branches':tasks,
         'completed_branch_ids':['sibling'],'processed_result_ids':['result'],
         'evidence_ids':[],'executed_candidate_ids':[]}]
    path=tmp_path/'trace.jsonl'
    path.write_text('\n'.join(json.dumps(dict(e,sequence=i)) for i,e in enumerate(events)))
    with pytest.raises(ValueError,match='publication boundary'):
        reconstruct(path)
    events[-1].update(completed_branch_ids=[],processed_result_ids=[],executed_candidate_ids=['not-completed'])
    path.write_text('\n'.join(json.dumps(dict(e,sequence=i)) for i,e in enumerate(events)))
    with pytest.raises(ValueError,match='Non-prior'):
        reconstruct(path)


def test_same_snapshot_does_not_accumulate_candidate_results():
    s=source()
    before=s.model_dump_json()
    first=construct(s,'sleep evidence')
    construct(s,'finance question')
    assert s.model_dump_json()==before
    assert construct(s,'sleep evidence')==first
    assert len(json.loads(first.text)['recent_queries'])==6


def test_shared_source_roundtrip_top_three_and_full_insights():
    s=source()
    s=ObservationSource.model_validate({**s.model_dump(),
        'tasks':[*(t.model_dump() for t in s.tasks), {'id':'fourth','parent_id':None,'tree_depth':1}],
        'published_branches':[*(b.model_dump() for b in s.published_branches),
            {'task_id':'fourth','goal':'distinct marine biology', 'findings':['complete finding'],
             'followups':['complete followup'], 'cited_insights':{'uncertainty qualifier':'https://example.org/full'}}]})
    # Same structured snapshot from JSON (future synthetic) or objects (future live).
    from_json=ObservationSource.model_validate_json(s.model_dump_json())
    o=construct(s,'distinct marine biology')
    assert o==construct(from_json,'distinct marine biology')
    assert len(o.selected_task_ids)==3 and o.selected_task_ids[0]=='fourth'
    assert json.loads(o.text)['branches'][0]['cited_insights']=={'uncertainty qualifier':'https://example.org/full'}
