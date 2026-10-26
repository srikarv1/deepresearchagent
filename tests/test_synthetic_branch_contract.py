from copy import deepcopy
import json
import socket
import pytest
from adr.synthetic_branch_v2.source import World, compile_world, digest
from adr.synthetic_branch_v2.fixture import fixture_world, fixture_annotations, FINDING
from adr.synthetic_branch_v2.supervision import Annotation, Review, Witness, finalize, validate_annotation
from adr.synthetic.schema import FEATURE_NAMES
from adr.mlp.branch_observation_v2 import ObservationSource, Task, PublishedBranch, construct


def prepared():
    p,a,e=compile_world(fixture_world())
    annotations,reviews=fixture_annotations(p)
    return p,a,e,annotations,reviews


def test_fixture_deterministic_identity_and_full_features(monkeypatch):
    monkeypatch.setattr(socket.socket,'connect',lambda *args:pytest.fail('Network forbidden'))
    p,a,e,ann,r=prepared()
    assert (p,a,e)==compile_world(fixture_world())
    accepted,rejected=finalize(p,a,e,ann,r)
    assert [x['label'] for x in accepted]==['EXECUTE','SKIP']
    assert rejected=={'ambiguous':'ambiguous_or_rejected'}
    assert tuple(p['late']['numerical_features']['names'])==FEATURE_NAMES
    assert p['late']['numerical_features']['values'][4]==7
    assert a['late']['selected_task_ids'][0]=='humidity'
    assert len(a['late']['selected_task_ids'])==3
    view=json.loads(p['late']['context_before'])
    assert view['branches'][0]['relation']=='parent'
    assert len(view['recent_queries'])==6
    assert p['catalog-query']['context_before'].find(FINDING)==-1  # processed sibling still unpublished
    # Future live can supply the same typed source to the exact constructor.
    world=fixture_world()
    sources=[x for x in world.events if x.kind=='plan']
    tasks=[t for x in sources for t in x.tasks]
    cards=[PublishedBranch(task_id=x.task_id,goal=next(t.goal for t in tasks if t.id==x.task_id),
        findings=x.findings,followups=x.followups,cited_insights=x.cited_insights)
        for x in world.events if x.kind=='process']
    source=ObservationSource(current_task_id='followup',current_need=tasks[-1].goal,
        tasks=tuple(Task(id=t.id,parent_id=t.parent_id,tree_depth=2 if t.parent_id else 1) for t in tasks),
        published_branches=tuple(cards),executed_queries=tuple(
            x.candidates[0].query for x in world.events if x.kind=='batch' and x.task_id!='followup'))
    assert construct(source,p['late']['candidate_subquery']).text==p['late']['context_before']


@pytest.mark.parametrize('text',['future unseen finding','hidden raw page says skip','https://north-archive.invalid/survey/humidity',
    'The example access guide describes appointment booking.'])
def test_hidden_future_url_only_and_unselected_witness_rejected(text):
    p,a,e,ann,r=prepared()
    bad=ann[1].model_copy(update={'witnesses':[Witness(kind='finding',branch_index=0,text=text)]})
    with pytest.raises(ValueError,match='Witness absent'):
        validate_annotation(bad,p['late'],a['late'])


def test_packet_binding_and_semantic_review_required():
    p,a,e,ann,r=prepared()
    bad=ann[0].model_copy(update={'packet_digest':'wrong'})
    with pytest.raises(ValueError,match='bound'):
        validate_annotation(bad,p['humidity-query'],a['humidity-query'])
    single=ann[1].model_copy(update={'pair_id':None})
    rows,rejected=finalize(p,a,e,[single],{})
    assert rows==[] and rejected['late']=='needs_semantic_review'
    review=r['late'].model_copy(update={'annotation_digest':digest(single.model_dump()),'verdict':'AMBIGUOUS'})
    rows,rejected=finalize(p,a,e,[single],{'late':review})
    assert not rows and rejected['late']=='semantic_ambiguous'


@pytest.mark.parametrize('mutation',['query','earlier','witness','already_published'])
def test_counterfactual_requirements(mutation):
    p,a,e,ann,r=prepared()
    if mutation=='query':
        p['late']['candidate_subquery']+=' extra qualifier'
        a['late']['packet_digest']=digest(p['late'])
        ann[1]=ann[1].model_copy(update={'packet_digest':digest(p['late'])})
        r['late']=r['late'].model_copy(update={'annotation_digest':digest(ann[1].model_dump())})
    elif mutation=='earlier':
        a['late']['event_index']=a['humidity-query']['event_index']
    elif mutation=='already_published':
        a['humidity-query']['published_task_ids']=['humidity']
    else:
        ann[1]=ann[1].model_copy(update={'decisive_witness':None})
        r['late']=r['late'].model_copy(update={'annotation_digest':digest(ann[1].model_dump())})
    with pytest.raises(ValueError,match='Counterfactual'):
        finalize(p,a,e,ann,r)


def test_overflow_cannot_be_training_skip():
    world=fixture_world().model_dump()
    world['events'][-4]['tasks'][0]['goal']='archive need ' * 5000
    p,a,e=compile_world(World.model_validate(world))
    assert a['late']['overflow'] and a['late']['required_bypass_decision']=='EXECUTE'
    annotation=Annotation(candidate_id='late',packet_digest=digest(p['late']),label='SKIP',category='direct_answer',rationale='Unsupported overflow')
    rows,rejected=finalize(p,a,e,[annotation],{})
    assert not rows and rejected['late']=='overflow_execute_bypass'


def test_lifecycle_accounting_and_raw_exclusion():
    world=fixture_world().model_dump()
    end=next(x for x in world['events'] if x['kind']=='end_batch')
    end['pages']=[{'id':'p1','url':'https://source.invalid/','text':'hidden feature-only content'}]
    p,a,e=compile_world(World.model_validate(world))
    assert 'hidden feature-only content' not in p['late']['context_before']
    assert p['late']['numerical_features']['values'][0]==1
    assert a['late']['budget']['used_searches']==9 # 3 planning + 6 executions
    assert a['late']['budget']['used_steps']==8 # 6 decisions + 2 publications
    assert a['late']['budget']['max_tokens']==400000
    assert p['catalog-query']['numerical_features']['values'][0]==0
    end['executed_ids']=[]
    with pytest.raises(ValueError,match='Retrieval'):
        compile_world(World.model_validate(world))


def test_unpublished_parent_and_duplicate_process_rejected():
    w=fixture_world().model_dump()
    w['events'][0]['tasks'][1]['parent_id']='humidity'
    with pytest.raises(ValueError,match='Parent must be published'):
        compile_world(World.model_validate(w))
    w=fixture_world().model_dump()
    idx=next(i for i,x in enumerate(w['events']) if x['kind']=='process')
    w['events'].insert(idx+1,deepcopy(w['events'][idx]))
    with pytest.raises(ValueError,match='Only active'):
        compile_world(World.model_validate(w))


def test_realistic_28_task_three_depth_world_and_overruns():
    # Mechanical offline structural fixture, not labeled or teacher-generated data.
    events=[]; counter=0
    def round_events(parent,n,depth):
        nonlocal counter
        ids=[f't{i}' for i in range(counter,counter+n)]; counter+=n
        events.append(dict(kind='plan',tasks=[dict(id=i,parent_id=parent,goal=f'Find archive topic {i}') for i in ids],
            planning_queries=['Find published catalogue']*(5 if parent is None else 2),research_tokens=1000))
        for tid in ids:
            cs=[dict(id=tid+f'-c{i}',query=f'Find published archive {tid} detail {i}') for i in range(4)]
            events.append(dict(kind='batch',task_id=tid,candidates=cs))
            events.append(dict(kind='end_batch',executed_ids=[c['id'] for c in cs],
                pages=[dict(id=tid+f'-p{i}',url=f'https://fixture.invalid/{tid}/{i}',text='Feature ledger excerpt') for i in range(10)],
                read_attempts=15,research_tokens=15000))
            insights={f'The fictional source {tid} reports observation {i} with unresolved applicability to other collections.':f'https://fixture.invalid/{tid}/{i}' for i in range(8)}
            events.append(dict(kind='process',task_id=tid,findings=list(insights)[:3],
                followups=['Find independent replication','Find published methods'],cited_insights=insights,research_tokens=1000))
        events.append(dict(kind='publish',task_ids=ids))
        if depth<3:
            for tid in ids:
                round_events(tid,2,depth+1)
    round_events(None,4,1)
    p,a,e=compile_world(World.model_validate(dict(question_id='structural-fixture',original_question='Which archive methods have supporting published evidence?',events=events)))
    assert len(p)==112 and counter==28
    last=list(a.values())[-1]
    assert last['branch_count']==28 and last['published_count']==26 and last['history_count']==104
    assert last['budget']['used_searches']==137
    assert last['budget']['used_steps']==120
    assert last['budget']['used_tokens']>400000
    v=list(p.values())[-1]['numerical_features']['values']
    assert v[9:11]==[1,1] and v[11:]==[0,0,0,0]
    assert v[4]==28 and v[0]==260 # all published pages, not selected packets
    assert max(meta['tree_depth'] for meta in a.values())==3
    # Equal lexical relevance legitimately ties to older roots; depth is not a quota.
    assert all(meta['selected_task_ids'][:1]==['t0'] for meta in a.values() if meta['selected_task_ids'])
    assert all(audit['tokens']<=4096 for audit in a.values())


def test_partial_round_publication_rejected():
    w=fixture_world().model_dump()
    publication=next(e for e in w['events'] if e['kind']=='publish')
    publication['task_ids']=['humidity']
    with pytest.raises(ValueError,match='complete planner round'):
        compile_world(World.model_validate(w))


def test_annotation_can_precede_future_world_completion():
    full=fixture_world()
    prefix=full.model_copy(update={'events':full.events[:2]})
    p,a,e=compile_world(prefix,allow_prefix=True)
    complete,_,_=compile_world(full)
    assert p['humidity-query']==complete['humidity-query']
    assert e=={} # No invented future execution or result needed to annotate.
    with pytest.raises(ValueError,match='Incomplete'):
        compile_world(prefix)
