import asyncio
from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml
from adr.synthetic_branch_v2 import pilot, pilot_annotation as ann, pilot_ledger as ledger, smoke
from adr.synthetic_branch_v2.fixture import fixture_world, FINDING
from adr.synthetic_branch_v2.source import compile_world, World
from adr.llm.base import LLMResponse
from adr.core.types import TokenUsage

OLD=Path('runs/synthetic-branch-v2-smoke/smoke-1-annotation-only-v1')


def packet(query='Find published archive humidity guidance', branches=None):
    return dict(original_question='What is suitable archive storage?',candidate_subquery=query,
        context_before=json.dumps(dict(current_need='Find archive humidity guidance',recent_queries=[],branches=branches or [])))


def proposal(**kw):
    return dict(dict(action_eligibility='retrieval',visible_coverage='no_relevant_evidence',label='EXECUTE',
        category='unresolved_need',rationale='The requested measurement is absent.',gap=None,witnesses=[]),**kw)


def validate(p, pkt):return ann.validate(json.dumps(p),pkt)


def source(q):
    world=fixture_world().model_dump()
    world.update(question_id=q.id,original_question=q.question)
    # 8 batches -> 16 proposals; fixture already exercises 0, 2, 6 published cards.
    for e in world['events']:
        e.pop('research_tokens',None)
        if e['kind']=='batch':
            e['candidates'] += [dict(id=c['id']+'-extra',query='Find independent published evidence on '+c['query']) for c in list(e['candidates'])]
    return world


def cfg(tmp_path):
    return pilot.Config(output=tmp_path/'pilot',questions=[pilot.Question(id=f'q{i}',question=f'Original fixture question {i}?',purpose='offline fixture') for i in range(4)])


def response(value,finish='stop'):
    text=json.dumps(value)
    return LLMResponse(text=text,usage=TokenUsage(prompt_tokens=10,completion_tokens=10,total_tokens=20),
        raw={'choices':[{'finish_reason':finish,'message':{'content':text}}],'usage':{'total_tokens':20}})


class Fake:
    def __init__(self, config, fail_at=None):self.config=config;self.calls=[];self.fail_at=fail_at;self.closed=False
    async def complete(self,messages,**kwargs):
        self.calls.append(messages)
        if len(self.calls)==self.fail_at:raise RuntimeError('private provider payload')
        data=json.loads(messages[1]['content'])
        if 'world_schema' in data:
            q=next(q for q in self.config.questions if q.id==data['question_id'])
            return response(source(q))
        return response(proposal(action_eligibility='retrieval',visible_coverage='uncertain',label='AMBIGUOUS',category='ambiguous',rationale='Both interpretations are possible.'))
    async def close(self):self.closed=True


def test_witness_locations_and_initial_empty():
    pkt=packet()
    assert validate(proposal(),pkt)['label']=='EXECUTE'
    b=dict(findings=['Exact finding'],cited_insights={'Exact insight':'https://test.invalid'},followups=['Find follow-up'])
    view=json.loads(packet(branches=[b])['context_before'])
    for kind,text in [('finding','Exact finding'),('insight','Exact insight'),('followup','Find follow-up')]:
        assert ann.visible(ann.Witness(kind=kind,branch_index=0,text=text),view)
    assert not ann.visible(ann.Witness(kind='history',text='Find follow-up'),view)
    assert not ann.visible(ann.Witness(kind='finding',branch_index=0,text='exact finding'),view)
    assert not ann.visible(ann.Witness(kind='insight',branch_index=0,text='https://test.invalid'),view)
    assert not ann.visible(ann.Witness(kind='history',text=view['current_need']),view)


@pytest.mark.parametrize('query',[
    'Extract attendance definitions','Identify programmes and summarize their methods',
    'Calculate an average','Contact authors for data','Find reports and analyze the data',
    'Conduct an experiment','Draft a review','Simulate runoff'])
def test_binary_action_rejection(query):
    with pytest.raises(ValueError,match='action_space'):validate(proposal(),packet(query))
    assert validate(proposal(action_eligibility='invalid_action_space',visible_coverage='not_applicable',label='AMBIGUOUS',category='invalid_action_space'),packet(query))


@pytest.mark.parametrize('query',['Find published analyses of experiments','Find reviews summarizing methods',
    'Retrieve studies comparing attendance','Search for experiments measuring runoff',
    'Find published studies that calculate averages and summarize uncertainty'])
def test_published_work_searches_allowed(query):
    assert not ann.action_invalid(query)
    assert validate(proposal(),packet(query))


def test_specific_gap_not_completeness_and_partial_category():
    pkt=packet(branches=[dict(findings=['Some humidity guidance exists'],cited_insights={},followups=[])])
    p=proposal(visible_coverage='partial',category='insufficient_answer',witnesses=[dict(kind='finding',branch_index=0,text='Some humidity guidance exists')])
    with pytest.raises(ValueError,match='specific_gap'):validate(p,pkt)
    p['gap']=dict(dimension='measurement',requested_text='humidity',missing_information='Coverage is not comprehensive.')
    with pytest.raises(ValueError,match='generic_completeness'):validate(p,pkt)
    p['gap']['missing_information']='Recommended numerical humidity range is absent.'
    assert validate(p,pkt)
    p['gap']['requested_text']='Bangladesh'
    with pytest.raises(ValueError,match='gap_not_requested'):validate(p,pkt)
    p['category']='no_relevant_visible_evidence'
    with pytest.raises(ValueError,match='partial_coverage'):validate(p,pkt)


def test_followup_alone_cannot_support_skip():
    pkt=packet(branches=[dict(findings=[],cited_insights={},followups=['Find evidence'])])
    with pytest.raises(ValueError,match='skip_requires'):
        validate(proposal(label='SKIP',category='direct_answer',visible_coverage='sufficient',witnesses=[dict(kind='followup',branch_index=0,text='Find evidence')]),pkt)


def test_old_ten_prospective_regression_without_relabeling_or_calls():
    if not OLD.exists():pytest.skip('Local preserved smoke unavailable')
    before={p: p.read_bytes() for p in OLD.rglob('*') if p.is_file()}
    for i in range(1,11):
        stage=OLD/'stages'/f'annotation-{i:03d}'
        pkt=json.loads(smoke.read(stage/'request.json')['messages'][1]['content'])
        old=json.loads(smoke.read(stage/'response_bundle.json')['assistant_content'])
        view=json.loads(pkt['context_before'])
        if i<=4:
            assert validate(proposal(),pkt)['label']=='EXECUTE' # hypothetical schema capability, not a repaired record
            assert any(not ann.visible(ann.Witness(**w),view) for w in old['witnesses'])
        elif i in (6,7):
            with pytest.raises(ValueError,match='action_space'):validate(proposal(**old),pkt)
        elif i==9:
            p=proposal(**old);p['visible_coverage']='partial'
            with pytest.raises(ValueError,match='specific_gap'):validate(p,pkt)
            p['gap']=dict(dimension='comparison',requested_text='comparing',missing_information='Not comprehensive')
            with pytest.raises(ValueError,match='generic_completeness'):validate(p,pkt)
        elif i==10:
            w=old['witnesses'][-1]
            assert not ann.visible(ann.Witness(**w),view)
            assert ann.visible(ann.Witness(kind='followup',branch_index=0,text=w['text']),view)
    assert all(p.read_bytes()==b for p,b in before.items())


def test_ledger_derivation_and_exact_compilation(tmp_path):
    q=cfg(tmp_path).questions[0]; data=source(q); old=deepcopy(data)
    world,proxy=ledger.derive(data)
    assert data==old and proxy['provider_equivalent'] is False
    assert sum(e.research_tokens for e in world.events if hasattr(e,'research_tokens'))==proxy['total_tokens']
    assert all(o['total_tokens']==o['input_tokens']+o['output_tokens'] for o in proxy['operations'])
    assert all(o['output_tokens']>0 for o in proxy['operations'])
    # Future/unpublished sibling results absent from first query-generation input.
    first=next(o for o in proxy['operations'] if o['operation']=='generate_queries')
    assert FINDING not in first['input_text']
    assert ledger.derive(data)[0]==world
    p,a,e=compile_world(world)
    numerical=smoke.ledger(world,a)
    assert numerical['final_usage']['used_tokens']==proxy['total_tokens']
    # Tokens change, semantic observations do not.
    zero=world.model_copy(deep=True)
    for ev in zero.events:
        if hasattr(ev,'research_tokens'):ev.research_tokens=0
    p0=compile_world(zero)[0]
    assert all(p[c]['context_before']==p0[c]['context_before'] for c in p)
    assert all({name for name,x,y in zip(p[c]['numerical_features']['names'],p[c]['numerical_features']['values'],p0[c]['numerical_features']['values']) if x!=y} <= {'token_frac','remaining_tokens'} for c in p)
    bad=deepcopy(data);bad['events'][0]['research_tokens']=1
    with pytest.raises(ValueError,match='must not author'):ledger.derive(bad)
    assert all('research_tokens' not in d.get('properties',{}) for d in ledger.author_schema()['$defs'].values())


async def test_full_mocked_pilot_isolation_checkpoint_and_no_training(tmp_path):
    c=cfg(tmp_path);fake=Fake(c)
    result=await pilot.run(c,allow_api=True,client_factory=lambda:fake)
    assert result['training_rows']==0 and result['provider_calls_reserved']==68
    assert result['summary']['decisions']==64 and fake.closed
    assert not list(c.output.rglob('examples.jsonl'))
    for q in c.questions:
        packets=smoke.read(c.output/q.id/'compiled_packets.json')
        assert len(packets)==16
        ordered=compile_world(World.model_validate(smoke.read(c.output/q.id/'source_world.json')))[0]
        assert ordered==packets
        for i,p in enumerate(ordered.values(),1):
            req=smoke.read(c.output/q.id/'stages'/f'annotation-{i:03d}'/'request.json')
            payload=json.loads(req['messages'][1]['content'])
            assert set(payload)=={'original_question','context_before','candidate_subquery','supervision_schema'}
            assert all(payload[k]==p[k] for k in ('original_question','context_before','candidate_subquery'))
            assert req['sdk_retries']==0
    def forbidden():pytest.fail('Cached stage must not create client')
    again=await pilot.run(c,resume=True,allow_api=True,client_factory=forbidden)
    assert again==result


async def test_failed_calls_preserved_and_no_retry(tmp_path):
    c=cfg(tmp_path);fake=Fake(c,fail_at=3)
    await pilot.run(c,allow_api=True,client_factory=lambda:fake)
    stage=c.output/'q0/stages/annotation-002'
    assert smoke.read(stage/'outcome.json')['status']=='provider_failed'
    assert 'private provider payload' not in (stage/'outcome.json').read_text()
    count=len(fake.calls)
    await pilot.run(c,resume=True,allow_api=True,client_factory=lambda:pytest.fail('No retries'))
    assert len(fake.calls)==count


async def test_validation_failure_preserves_response_usage_and_continues(tmp_path):
    c=cfg(tmp_path);fake=Fake(c);original=fake.complete
    async def bad(messages,**kw):
        r=await original(messages,**kw)
        if len(fake.calls)==2:return response(proposal(witnesses=[dict(kind='history',text='HIDDEN')]))
        return r
    fake.complete=bad
    result=await pilot.run(c,allow_api=True,client_factory=lambda:fake)
    folder=c.output/'q0/stages/annotation-001'
    assert smoke.read(folder/'response_bundle.json')['assistant_content']
    assert smoke.read(folder/'usage.json')['provider_usage']['total_tokens']==20
    assert result['summary']['witness_failures']==1
    assert result['summary']['structural_rejections']==1
    assert result['summary']['decisions']==64


async def test_offline_preparation_and_contract_change(tmp_path):
    c=cfg(tmp_path)
    await pilot.run(c,client_factory=lambda:pytest.fail('offline call'))
    assert not list(c.output.glob('*/stages'))
    req=smoke.read(c.output/'q0/world_request_preview.json')
    payload=json.loads(req['messages'][1]['content'])
    assert payload['original_question']==c.questions[0].question
    changed=c.model_copy(deep=True);changed.questions[0].question='changed'
    with pytest.raises(ValueError,match='contract changed'):
        await pilot.run(changed,resume=True,allow_api=True,client_factory=lambda:pytest.fail('contract'))


async def test_generation_budget_failure_no_annotation(tmp_path):
    c=cfg(tmp_path);fake=Fake(c)
    async def length(messages,**kw):fake.calls.append(messages);return response('',finish='length')
    fake.complete=length
    result=await pilot.run(c,allow_api=True,client_factory=lambda:fake)
    assert len(fake.calls)==4 and result['summary']['decisions']==0
    assert all(smoke.read(c.output/q.id/'stages/world/outcome.json')['status']=='generation_failed' for q in c.questions)
    await pilot.run(c,resume=True,allow_api=True,client_factory=lambda:pytest.fail('retry'))


async def test_interrupt_recovery_only_unstarted_calls(tmp_path,monkeypatch):
    c=cfg(tmp_path);fake=Fake(c);original=smoke.stage
    async def interrupt(out,name,*args,**kwargs):
        if out.name=='q0' and name=='annotation-002':raise KeyboardInterrupt()
        return await original(out,name,*args,**kwargs)
    monkeypatch.setattr(smoke,'stage',interrupt)
    with pytest.raises(KeyboardInterrupt):await pilot.run(c,allow_api=True,client_factory=lambda:fake)
    assert len(fake.calls)==2
    monkeypatch.setattr(smoke,'stage',original)
    second=Fake(c)
    result=await pilot.run(c,resume=True,allow_api=True,client_factory=lambda:second)
    assert len(second.calls)==66 and result['summary']['decisions']==64


def test_scope_ceiling_and_question_binding(tmp_path):
    c=cfg(tmp_path);q=c.questions[0];w=source(q)
    first=next(e for e in w['events'] if e['kind']=='batch')
    first['candidates'] += [dict(id=f'max-{i}',query=f'Find published archive measurement {i}') for i in range(4)]
    assert len(compile_world(World.model_validate(pilot.validate_world(json.dumps(w),q,c)['world']))[0])==20
    first['candidates'].append(dict(id='too-many',query='Find another published measure'))
    with pytest.raises(ValueError,match='candidate count'):pilot.validate_world(json.dumps(w),q,c)
    w=source(q);w['original_question']='Wrong question'
    with pytest.raises(ValueError,match='question identity'):pilot.validate_world(json.dumps(w),q,c)


def test_hidden_information_request_exclusion():
    pkt=packet()
    pkt.update(numerical_features='PRIVATE_FEATURES',source_executed='PRIVATE_EXECUTION',
        future='PRIVATE_FUTURE',raw_evidence='PRIVATE_RAW',unselected='PRIVATE_BRANCH')
    request=ann.request(pkt)
    assert 'PRIVATE_' not in json.dumps(request)
    data=json.loads(request[1]['content'])
    assert set(data)=={'original_question','context_before','candidate_subquery','supervision_schema'}
    assert data['context_before']==pkt['context_before']
