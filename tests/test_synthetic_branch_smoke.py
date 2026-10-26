import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from adr.llm.base import LLMResponse
from adr.core.types import TokenUsage
from adr.synthetic_branch_v2.fixture import fixture_world, QUERY, FINDING
from adr.synthetic_branch_v2.source import World, compile_world
from adr.synthetic_branch_v2.smoke import Config, run, stage, StageFailure, read, ledger, azure_client
from adr.synthetic_branch_v2.smoke_prompts import annotation_request


def response(value,finish='stop'):
    text=json.dumps(value) if not isinstance(value,str) else value
    usage={'prompt_tokens':31,'completion_tokens':17,'total_tokens':48,
           'completion_tokens_details':{'reasoning_tokens':4}}
    return LLMResponse(text=text,usage=TokenUsage(prompt_tokens=31,completion_tokens=17,total_tokens=48),
        raw={'model':'gpt-5-mini','choices':[{'finish_reason':finish,'message':{'content':text}}],'usage':usage})

class Fake:
    def __init__(self, world=None, fail_call=None):
        self.world=world or fixture_world().model_dump(); self.calls=[]; self.closed=False; self.fail_call=fail_call
    async def complete(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        if len(self.calls)==self.fail_call:
            raise RuntimeError('SECRET must never be persisted')
        payload=json.loads(messages[1]['content'])
        if 'world_schema' in payload:
            return response(self.world)
        view=json.loads(payload['context_before'])
        query=payload['candidate_subquery']
        if query==QUERY and view['branches']:
            idx=next(i for i,b in enumerate(view['branches']) if FINDING in b['findings'])
            w={'kind':'finding','branch_index':idx,'text':FINDING}
            return response(dict(label='SKIP',category='direct_answer',rationale='Visible recommendation answers this query.',witnesses=[w],decisive_witness=w))
        if query=='Find further general archive storage advice':
            return response(dict(label='AMBIGUOUS',category='ambiguous',rationale='Marginal value is unclear.',witnesses=[]))
        return response(dict(label='EXECUTE',category='unresolved_need',rationale='This specific information need is not answered.',witnesses=[]))
    async def close(self): self.closed=True


def config(tmp_path): return Config(output=tmp_path/'smoke')


async def test_isolation_response_preservation_and_human_audit(tmp_path):
    world=fixture_world().model_dump()
    end=next(e for e in world['events'] if e['kind']=='end_batch')
    end['pages']=[dict(id='raw1',url='https://fixture.invalid/raw',text='HIDDEN_RAW_SENTINEL')]
    fake=Fake(world)
    cfg=config(tmp_path)
    result=await run(cfg,allow_api=True,client_factory=lambda:fake)
    assert result['status']=='awaiting_human_audit' and result['training_rows']==0
    assert len(fake.calls)==9 and fake.closed
    packets=read(cfg.output/'compiled_packets.json')
    for ((messages,kwargs),packet) in zip(fake.calls[1:],compile_world(World.model_validate(world))[0].values(),strict=True):
        assert len(messages)==2
        payload=json.loads(messages[1]['content'])
        assert set(payload)=={'original_question','context_before','candidate_subquery','supervision_schema'}
        for k in ('original_question','context_before','candidate_subquery'):
            assert payload[k]==packet[k]
        assert 'HIDDEN_RAW_SENTINEL' not in json.dumps(messages)
        assert 'numerical_features' not in payload and 'executed_ids' not in payload
        assert kwargs['extra']=={'response_format':{'type':'json_object'}}
    assert FINDING not in json.dumps(fake.calls[1][0]) # initial annotation cannot see future
    last=json.loads(fake.calls[-2][0][1]['content'])
    assert 'The example access guide describes appointment booking.' not in last['context_before']
    audit=read(cfg.output/'human_audit.json')
    assert len(audit['decisions'])==8 and len(audit['counterfactual_pairs'])==1
    assert all(r['human_review']=='pending' for r in audit['decisions'])
    assert not (cfg.output/'examples.jsonl').exists()
    assert read(cfg.output/'stages/world/response_bundle.json')['raw_provider_response']['usage']['total_tokens']==48
    assert read(cfg.output/'numerical_ledger.json')['not_provider_usage']
    # Resume a completed run must not construct a client or re-annotate anything.
    def forbidden(): pytest.fail('Client created on completed resume')
    again=await run(cfg,resume=True,allow_api=True,client_factory=forbidden)
    assert again==result


async def test_raw_and_usage_saved_before_validation_and_no_retry(tmp_path):
    fake=Fake(); out=tmp_path
    def invalid(text):
        assert read(out/'stages/x/response_bundle.json')['assistant_content']
        assert read(out/'stages/x/usage.json')['provider_usage']['total_tokens']==48
        raise ValueError('reject local fixture')
    with pytest.raises(StageFailure):
        await stage(out,'x',[{'role':'system','content':'schema'},{'role':'user','content':'{"world_schema":{}}'}],32000,lambda:fake,invalid)
    with pytest.raises(StageFailure):
        await stage(out,'x',[{'role':'system','content':'schema'},{'role':'user','content':'{"world_schema":{}}'}],32000,lambda:fake,invalid)
    assert len(fake.calls)==1


async def test_empty_length_and_provider_error_stay_checkpointed(tmp_path):
    for index,budget in enumerate((False,True)):
        fake=Fake(fail_call=1 if not budget else None)
        if budget:
            async def fail(*args,**kwargs): return response('',finish='length')
            fake.complete=fail
        cfg=Config(output=tmp_path/str(index))
        with pytest.raises(StageFailure): await run(cfg,allow_api=True,client_factory=lambda:fake)
        outcome=read(cfg.output/'stages/world/outcome.json')
        assert outcome['status']==('generation_failed' if budget else 'provider_failed')
        assert 'SECRET' not in json.dumps(outcome)
        if budget:
            assert outcome['error_type']=='GenerationBudgetFailure'
            assert (cfg.output/'stages/world/response_bundle.json').exists()
        with pytest.raises(StageFailure):
            await run(cfg,resume=True,allow_api=True,client_factory=lambda:pytest.fail('Unexpected retry'))


async def test_partial_checkpoint_and_pending_human_rows(tmp_path):
    cfg=config(tmp_path); fake=Fake(fail_call=3)
    with pytest.raises(StageFailure): await run(cfg,allow_api=True,client_factory=lambda:fake)
    audit=read(cfg.output/'human_audit.json')
    assert len(audit['decisions'])==8
    assert audit['decisions'][0]['validation_status']=='needs_semantic_review'
    assert audit['decisions'][1]['validation_status']=='stage_failed_no_retry'
    assert audit['decisions'][-1]['validation_status']=='not_requested'
    with pytest.raises(StageFailure):
        await run(cfg,resume=True,allow_api=True,client_factory=lambda:pytest.fail('Repeated call'))
    assert len(fake.calls)==3


async def test_interrupted_started_call_not_reissued_and_saved_response_recovers(tmp_path):
    cfg=config(tmp_path); fake=Fake()
    await run(cfg,allow_api=True,client_factory=lambda:fake)
    folder=cfg.output/'stages/annotation-008'
    (folder/'outcome.json').unlink() # crash after raw bundle, before validation outcome
    await run(cfg,resume=True,allow_api=True,client_factory=lambda:pytest.fail('Should recover saved response'))
    (folder/'outcome.json').unlink(); (folder/'response_bundle.json').unlink()
    with pytest.raises(StageFailure,match='uncertain'):
        await run(cfg,resume=True,allow_api=True,client_factory=lambda:pytest.fail('Unsafe retry'))


def test_request_allowlist_and_event_ledger():
    w=fixture_world(); p,a,e=compile_world(w)
    packet={**p['late'],'numerical_features':'NUMERIC_PRIVATE','hidden_world':'DO_NOT_SEND'}
    request=annotation_request(packet)
    assert 'DO_NOT_SEND' not in json.dumps(request) and 'NUMERIC_PRIVATE' not in json.dumps(request)
    totals=ledger(w,a)['final_usage']
    assert totals['used_steps']==10 # eight proposals + two round publications
    assert totals['used_searches']==9 # six executions + three planning searches
    assert totals['used_reads']==12
    assert a['late']['budget']['used_steps']==8
    assert a['late']['budget']['used_searches']==9
    assert p['late']==compile_world(World.model_validate(w.model_dump()))[0]['late']
    corrupted=json.loads(json.dumps(a)); corrupted['late']['budget']['used_searches']=100
    with pytest.raises(ValueError,match='ledger disagrees'): ledger(w,corrupted)


async def test_prepare_only_no_client_and_contract_mismatch(tmp_path):
    cfg=config(tmp_path)
    result=await run(cfg,client_factory=lambda:pytest.fail('Offline client'))
    assert result['provider_calls']==0 and not (cfg.output/'stages').exists()
    old=read(cfg.output/'contract.json'); old['max_provider_calls']=999
    (cfg.output/'contract.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='contract changed'):
        await run(cfg,resume=True,allow_api=True,client_factory=lambda:pytest.fail('Contract failure must precede call'))


def test_existing_azure_factory_no_sdk_retries(tmp_path,monkeypatch):
    from adr.llm import factory
    captured={}
    def build(cfg):
        captured.update(cfg)
        return SimpleNamespace(_client=SimpleNamespace(max_retries=2))
    monkeypatch.setattr(factory,'build_llm',build)
    monkeypatch.setenv('OPENAI_API_KEY','test-secret')
    monkeypatch.setenv('OPENAI_BASE_URL','https://azure-fixture.invalid/openai/v1/')
    client=azure_client(tmp_path)
    assert client._client.max_retries==0
    assert captured['model']=='gpt-5-mini' and captured['api_key_env']=='OPENAI_API_KEY'
    assert 'test-secret' not in (tmp_path/'endpoint_identity.json').read_text()


async def test_resume_continues_only_untouched_stages(tmp_path,monkeypatch):
    import adr.synthetic_branch_v2.smoke as smoke
    original=smoke.stage
    async def interrupted(out,name,*args,**kwargs):
        if name=='annotation-002':
            raise KeyboardInterrupt('offline simulated interruption before dispatch')
        return await original(out,name,*args,**kwargs)
    cfg=config(tmp_path); first=Fake()
    monkeypatch.setattr(smoke,'stage',interrupted)
    with pytest.raises(KeyboardInterrupt):
        await run(cfg,allow_api=True,client_factory=lambda:first)
    assert len(first.calls)==2 # world + first annotation
    monkeypatch.setattr(smoke,'stage',original)
    second=Fake()
    result=await run(cfg,resume=True,allow_api=True,client_factory=lambda:second)
    assert result['status']=='awaiting_human_audit' and len(second.calls)==7
    assert all('world_schema' not in json.loads(m[1]['content']) for m,_ in second.calls)


async def test_bad_witness_is_preserved_not_accepted(tmp_path):
    fake=Fake()
    original=fake.complete
    async def bad(messages,**kwargs):
        r=await original(messages,**kwargs)
        if len(fake.calls)==2:
            return response(dict(label='SKIP',category='direct_answer',rationale='Unsupported hidden claim',
                witnesses=[dict(kind='finding',branch_index=0,text='HIDDEN fabricated answer')]))
        return r
    fake.complete=bad
    cfg=config(tmp_path)
    with pytest.raises(StageFailure): await run(cfg,allow_api=True,client_factory=lambda:fake)
    audit=read(cfg.output/'human_audit.json')
    assert audit['decisions'][0]['proposal']['label']=='SKIP'
    assert audit['decisions'][0]['validation_status']=='stage_failed_no_retry'
    assert audit['training_rows']==0 and len(fake.calls)==2
    assert read(cfg.output/'stages/annotation-001/response_bundle.json')['raw_provider_response']
