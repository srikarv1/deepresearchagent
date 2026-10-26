import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from adr.synthetic_v2.events import EventTrajectory, encode_fixture, expand
from adr.synthetic_v2.generate import fixture, compile_trajectory
from adr.synthetic_v2.response_status import generation_failure
from adr.synthetic.validate import load_heldout
from adr.llm.openai_compat import OpenAICompatLLM


def test_expansion_preserves_all_source_and_model_inputs():
    source=fixture();events=encode_fixture(source)
    assert expand(events)==source
    heldout=load_heldout(Path('data/heldout/drb_baseline.json'))
    assert compile_trajectory(expand(events),heldout)==compile_trajectory(source,heldout)
    assert EventTrajectory.model_validate_json(Path('tests/fixtures/synthetic_v2_findings/events.json').read_text())==events
    schema=json.dumps(EventTrajectory.model_json_schema())
    for derived in ['used_steps','executed_subqueries','max_tokens','numerical_features','context_before']:
        assert derived not in schema


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d['states'][1]['publications_before'][0]['results'].append(d['states'][1]['publications_before'][0]['results'][0]),'Duplicate completed'),
    (lambda d:d['states'][1]['publications_before'][0]['results'][0].update(goal='Wrong'),'matching goal'),
    (lambda d:d['states'][1]['branches'][0].update(status='active'),'completed task'),
    (lambda d:d['states'][0]['publications_before'].append(d['states'][1]['publications_before'][0]),'completed task'),
    (lambda d:d['states'][2]['publications_before'][0]['evidence'].append(d['states'][1]['publications_before'][0]['evidence'][0]),'duplicate evidence'),
    (lambda d:d['states'][1]['usage'].update(used_searches=0),'Search attempts'),
    (lambda d:d['states'][1].update(executed_subqueries=['invented']),'Extra inputs'),
    (lambda d:d['states'][1]['usage'].update(used_steps=999),'Extra inputs'),
    (lambda d:d['states'][1]['publications_before'][0]['evidence'][0].update(retained=False),'Extra inputs'),
    (lambda d:d['states'][1]['candidates'][0].update(supporting_finding='hidden future fact'),'absent'),
])
def test_event_encoding_does_not_bypass_invariants(mutation,match):
    d=encode_fixture(fixture()).model_dump();mutation(d)
    with pytest.raises(ValueError,match=match):
        compile_trajectory(expand(EventTrajectory.model_validate(d)),load_heldout(Path('data/heldout/drb_baseline.json')))


@pytest.mark.parametrize('text,finish,completion,reasoning,kind',[
    ('','length',32000,32000,'GenerationBudgetFailure'),
    ('{"partial":','length',32000,10000,'GenerationBudgetFailure'),
    ('','stop',15,10,'EmptyAssistantContent'),
    ('{}','stop',32000,31900,'GenerationBudgetFailure'),
    ('{}','stop',50,10,None),
])
def test_generation_diagnostics(text,finish,completion,reasoning,kind):
    raw={'choices':[{'finish_reason':finish}],'usage':{'completion_tokens':completion,'completion_tokens_details':{'reasoning_tokens':reasoning}}}
    result=generation_failure(text,raw,32000)
    assert (result['error_type'] if result else None)==kind
    if result:assert result['structural_validation']=='not_run'


def test_preserved_smoke2_is_budget_failure_not_json_error():
    p=Path('runs/synthetic-v2-smoke/teacher-smoke-2-20260923')
    r=generation_failure((p/'raw_teacher.txt').read_text(),json.loads((p/'response.json').read_text()),16000)
    assert r['error_type']=='GenerationBudgetFailure' and r['reasoning_exhausted']


@pytest.mark.asyncio
async def test_client_passes_32000_unchanged_without_network(monkeypatch):
    response=NS(choices=[NS(message=NS(content='{}'))],usage=None,model_dump=lambda:{})
    create=AsyncMock(return_value=response)
    monkeypatch.setattr('adr.llm.openai_compat.AsyncOpenAI',lambda **kw:NS(chat=NS(completions=NS(create=create))))
    client=OpenAICompatLLM('gpt-5-mini',api_key='offline',default_max_tokens=32000)
    await client.complete([{'role':'user','content':'offline'}])
    args=create.call_args.kwargs
    assert args['max_completion_tokens']==32000 and args['model']=='gpt-5-mini'
    assert 'max_tokens' not in args and 'temperature' not in args


@pytest.mark.asyncio
@pytest.mark.parametrize('exhausted',[True,False])
async def test_smoke3_raw_preservation_and_no_retry(tmp_path,monkeypatch,exhausted):
    path=Path('runs/synthetic-v2-smoke/teacher-smoke-3-20260923/run_smoke.py')
    spec=importlib.util.spec_from_file_location('smoke3_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=tmp_path
    monkeypatch.setenv('OPENAI_API_KEY','offline-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL','https://offline.invalid/openai/v1/')
    monkeypatch.setattr('sys.argv',['run_smoke.py','--allow-api'])
    text='' if exhausted else encode_fixture(fixture()).model_dump_json()
    raw={'choices':[{'finish_reason':'length' if exhausted else 'stop','message':{'content':text}}],
         'usage':{'completion_tokens':32000 if exhausted else 100,'completion_tokens_details':{'reasoning_tokens':32000 if exhausted else 10}}}
    calls=[]
    class Client:
        def __init__(self):self._client=self
        def with_options(self,**kw):assert kw=={'max_retries':0};return self
        async def complete(self,*args,**kw):
            calls.append(1);return NS(text=text,raw=raw,usage=NS(model_dump=lambda:raw['usage']))
        async def close(self):pass
    def build(cfg):assert cfg['max_output_tokens']==32000;return Client()
    monkeypatch.setattr(module,'build_llm',build)
    await module.main()
    assert json.loads((tmp_path/'response.json').read_text())==raw
    assert (tmp_path/'raw_teacher.txt').read_text()==text
    result=json.loads((tmp_path/'validation.json').read_text())
    assert result['structural_validation']==('not_run' if exhausted else 'passed')
    if exhausted:assert result['error_type']=='GenerationBudgetFailure'
    with pytest.raises(SystemExit,match='already attempted'):await module.main()
    assert calls==[1]
