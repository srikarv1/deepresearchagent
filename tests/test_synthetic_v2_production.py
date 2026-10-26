"""Offline fake-provider tests; no paid client is constructed."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml
from adr.synthetic_v2 import production as p
from adr.synthetic_v2.events import encode_fixture
from adr.synthetic_v2.generate import fixture
from adr.synthetic_v2.visibility import visibility_views
from adr.synthetic_v2.production_schema import ProductionTrajectory, expand_production
from adr.synthetic_v2.production_prompt import teacher_messages, PROMPT_VERSION
from adr.synthetic_v2.retry_feedback import GUIDANCE, ContractError, classify, feedback
from adr.synthetic_v2.schema import TeacherTrajectory
from adr.synthetic.validate import load_heldout

QUESTION2='Which archival adhesives deteriorate under ultraviolet exposure according to published conservation studies?'


def events(question=None):
    f=fixture();d=encode_fixture(f).model_dump();prior_search=0;prior_exec=0
    if question:d['original_question']=question
    for state,view in zip(d['states'],visibility_views(f),strict=True):
        usage=state['usage'];search=usage.pop('used_searches')
        usage['planning_searches_since_previous_state']=search-prior_search-prior_exec
        prior_search=search;prior_exec=sum(c['label']=='EXECUTE' for c in state['candidates'])
        state['visible_findings']=view['visible_findings']
    return d


def config(n=2):
    d=yaml.safe_load(Path('configs/synthetic_v2.yaml').read_text());d['trajectories']=n;d['assignments']=d['assignments'][:n]
    return p.Config.model_validate(d)


def response(data=None,*,text=None,exhaust=False):
    text=json.dumps(data) if text is None else text
    usage={'prompt_tokens':100,'completion_tokens':32000 if exhaust else 100,'total_tokens':32100 if exhaust else 200,
           'completion_tokens_details':{'reasoning_tokens':32000 if exhaust else 20}}
    raw={'model':'gpt-5-mini-test','choices':[{'finish_reason':'length' if exhaust else 'stop','message':{'content':text}}],'usage':usage}
    return NS(text=text,raw=raw,usage=NS(model_dump=lambda:usage))


class Fake:
    def __init__(self,responses):self.responses=list(responses);self.calls=[];self.closed=False
    async def complete(self,messages,**kwargs):
        self.calls.append((messages,kwargs));item=self.responses.pop(0)
        if isinstance(item,Exception):raise item
        return item
    async def close(self):self.closed=True


async def run(tmp_path,fake,cfg=None,resume=False):
    return await p.run(cfg or config(),tmp_path/'generation',resume=resume,client_factory=lambda:fake,endpoint_identity='offline-test')


def test_local_search_derivation_and_visibility_equivalence():
    tr,views=expand_production(ProductionTrajectory.model_validate(events()))
    assert tr==fixture() and views==visibility_views(fixture())
    d=events()
    for s in d['states']:s['usage']['planning_searches_since_previous_state']=0
    tr,_=expand_production(ProductionTrajectory.model_validate(d))
    for s in tr.states:assert s.source.state.budget.used_searches==len(s.source.state.executed_subqueries)
    d['states'][1]['usage']['planning_searches_since_previous_state']=-1
    with pytest.raises(ValueError):ProductionTrajectory.model_validate(d)
    d=events();d['states'][1]['usage']['used_searches']=0
    with pytest.raises(ValueError,match='Extra inputs'):ProductionTrajectory.model_validate(d)


@pytest.mark.parametrize('old,new',[('active','open'),('open','done'),('open','dropped')])
def test_operational_task_transitions(old,new):
    d=fixture().model_dump()
    if old=='active':
        d['states'][1]['source']['state']['branches'][0]['status']='open'
        d['states'][1]['source']['completed_results']=[]
        d['states'][1]['source']['state']['evidence']=[]
        d['states'][1]['publications_before']=[]
        d['states'][1]['source']['state']['budget']['used_steps']=3
    else:d['states'][1]['source']['state']['branches'][4]['status']=new
    with pytest.raises(ValueError):TeacherTrajectory.model_validate(d)


@pytest.mark.parametrize('phrase',[
 'Extract instrument logs and compare sampling methods.',
 'Compare data ourselves.', 'Analyze the existing dataset.',
])
def test_followups_are_not_analysis(phrase):
    d=events();d['states'][1]['publications_before'][0]['results'][0]['followups']=[phrase]
    with pytest.raises(ValueError):p.validate(json.dumps(d),p.registry(config()),[])


@pytest.mark.parametrize('message,code',[
 ('Evidence-dependent category requires a visible supporting_finding','support'),
 ('Supervision references a finding absent from bounded observation','support'),
 ('Duplicate completed research result','identity'),
 ('Result must belong to a completed task with matching goal','identity'),
 ('Operational task lifecycle forbids active to open','identity'),
 ('New evidence pages and result cards must publish together at a recorded boundary','publication'),
 ('Executed history must equal all prior EXECUTE candidates','history'),
 ('Search usage must account for every prior EXECUTE','search'),
 ('Input must be a retrieval query','action'),
 ('Follow-up must be a retrieval information need','action'),
 ('Counterfactual requires identical query','pair'),
 ('S9: declared visible_findings differs from locally bounded view','visibility'),
 ('Malformed JSON','schema'),('Held-out overlap','diversity'),
])
def test_allowlisted_feedback_classes(message,code):
    codes=classify(ValueError(message+' SECRET_DO_NOT_FORWARD'))
    assert code in codes
    text=' '.join(feedback(codes))
    assert 'SECRET_DO_NOT_FORWARD' not in text
    assert all(part in GUIDANCE.values() for part in feedback(codes))


def test_full_task_retained_and_no_heldout_in_prompt():
    codes=list(GUIDANCE)
    before=teacher_messages('topic','causal comparison')
    after=teacher_messages('topic','causal comparison',feedback(codes))
    assert after[:2]==before and len(after)==3
    for q in p.registry(config()):assert q['text'] not in json.dumps(after)
    text=before[0]['content']
    assert 'No class balance target or per-category quota' in text
    assert 'actual visible contradiction' in text
    assert 'planning_searches_since_previous_state' in text
    assert 'cumulative actual searches' not in text
    assert PROMPT_VERSION=='synthetic-v2-findings-production-v1'


@pytest.mark.asyncio
async def test_failure_preserves_checkpoint_and_resume_skips_accepted(tmp_path):
    fake=Fake([response(events()),response(text='',exhaust=True),response(text='{bad'),response(text='')])
    with pytest.raises(RuntimeError,match='q002 exhausted 3'):await run(tmp_path,fake)
    root=tmp_path/'generation';assert fake.closed and len(fake.calls)==4
    assert len((root/'examples.jsonl').read_text().splitlines())==20
    assert json.loads((root/'manifest.json').read_text())['status']=='partial'
    checkpoint=(root/'accepted/q001/checkpoint.json').read_bytes()
    second=Fake([response(events(QUESTION2))]);result=await run(tmp_path,second,resume=True)
    assert result['status']=='complete' and result['examples']==40 and len(second.calls)==1
    assert (root/'accepted/q001/checkpoint.json').read_bytes()==checkpoint
    assert (root/'attempts/q002/attempt-0004/outcome.json').exists()
    assert 'public library accessibility' in second.calls[0][0][1]['content']
    def forbidden():raise AssertionError('Must not construct client for complete resume')
    finished=await p.run(config(),root,resume=True,client_factory=forbidden,endpoint_identity='offline-test')
    assert finished['status']=='complete'


@pytest.mark.asyncio
async def test_duplicate_checks_include_reloaded_questions(tmp_path):
    first=Fake([response(events()),response(text='bad'),response(text='bad'),response(text='bad')])
    with pytest.raises(RuntimeError):await run(tmp_path,first)
    fake=Fake([response(events()),response(events(QUESTION2))])
    result=await run(tmp_path,fake,resume=True)
    assert result['examples']==40 and len(fake.calls)==2
    assert GUIDANCE['diversity'] in fake.calls[1][0][-1]['content']


@pytest.mark.asyncio
async def test_targeted_retry_and_no_class_balance_retry(tmp_path):
    bad=events();bad['states'][1]['candidates'][0]['supporting_finding']=None
    fake=Fake([response(bad),response(events())])
    result=await run(tmp_path,fake,config(1))
    assert len(fake.calls)==2 and result['labels']=={'EXECUTE':18,'SKIP':2}
    assert fake.calls[0][0][:2]==fake.calls[1][0][:2]
    assert GUIDANCE['support'] in fake.calls[1][0][-1]['content']
    root=tmp_path/'generation';assert json.loads((root/'attempts/q001/attempt-0001/raw_teacher.txt').read_text())==bad
    rows=[json.loads(s) for s in (root/'examples.jsonl').read_text().splitlines()]
    assert rows[3]['supporting_finding'] is not None


@pytest.mark.asyncio
async def test_generation_budget_and_empty_feedback_before_json(tmp_path):
    fake=Fake([response(text='',exhaust=True),response(text=''),response(events())])
    await run(tmp_path,fake,config(1))
    root=tmp_path/'generation/attempts/q001'
    assert json.loads((root/'attempt-0001/outcome.json').read_text())['structural_validation']=='not_run'
    assert GUIDANCE['budget'] in fake.calls[1][0][-1]['content']
    assert GUIDANCE['empty'] in fake.calls[2][0][-1]['content']
    assert (root/'attempt-0001/response.json').exists()


@pytest.mark.asyncio
async def test_canonical_crash_recovered_without_teacher(tmp_path,monkeypatch):
    original=p.canonical
    def crash(directory,cfg,accepted,contract):
        if accepted:raise OSError('simulated publication interruption')
        return original(directory,cfg,accepted,contract)
    monkeypatch.setattr(p,'canonical',crash)
    with pytest.raises(OSError):await run(tmp_path,Fake([response(events())]),config(1))
    monkeypatch.setattr(p,'canonical',original)
    def forbidden():raise AssertionError('Already checkpointed')
    result=await p.run(config(1),tmp_path/'generation',resume=True,client_factory=forbidden,endpoint_identity='offline-test')
    assert result['examples']==20 and result['status']=='complete'


@pytest.mark.asyncio
async def test_corrupt_checkpoint_and_contract_change_fail_closed(tmp_path):
    await run(tmp_path,Fake([response(events())]),config(1))
    root=tmp_path/'generation'
    with pytest.raises(ValueError,match='endpoint_identity'):
        await p.run(config(1),root,resume=True,client_factory=lambda:None,endpoint_identity='changed')
    examples=root/'accepted/q001/examples.jsonl';examples.write_text('{}\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        await run(tmp_path,Fake([]),config(1),resume=True)


@pytest.mark.asyncio
async def test_provider_error_safe_and_bounded(tmp_path):
    fake=Fake([RuntimeError('SECRET_KEY_URL'),RuntimeError('SECRET_KEY_URL'),RuntimeError('SECRET_KEY_URL')])
    with pytest.raises(RuntimeError,match='exhausted 3'):await run(tmp_path,fake,config(1))
    for f in (tmp_path/'generation/attempts').rglob('*.json'):
        assert 'SECRET_KEY_URL' not in f.read_text()
    assert len(fake.calls)==3 and fake.closed


def test_required_heldout_ids_and_offline_defaults(tmp_path,monkeypatch,capsys):
    d=config(1).model_dump();reg=tmp_path/'heldout.json';reg.write_text(json.dumps({'questions':[{'id':'51','text':'Some heldout question'}]}));d['heldout_path']=str(reg)
    with pytest.raises(ValueError,match='51–55'):p.registry(p.Config.model_validate(d))
    monkeypatch.setattr('sys.argv',['production','--dry-run','--output',str(tmp_path/'never_created')])
    p.main();assert not (tmp_path/'never_created').exists()
    result=json.loads(capsys.readouterr().out);assert result['api_calls']==0 and result['target_examples']==400


def test_fixture_heldout_and_shape_failures():
    d=events();d['original_question']=p.registry(config())[0]['text']
    with pytest.raises(ValueError,match='Held-out'):p.validate(json.dumps(d),p.registry(config()),[])
    d=events();d['states'][-1]['candidates'].pop()
    with pytest.raises(ContractError) as exc:p.validate(json.dumps(d),p.registry(config()),[])
    assert exc.value.code=='schema'


@pytest.mark.asyncio
async def test_full_20_trajectory_canonical_integrity_offline(tmp_path):
    cfg=config(20)
    fake=Fake([response(events(a.topic+'?')) for a in cfg.assignments])
    manifest=await run(tmp_path,fake,cfg)
    root=tmp_path/'generation'
    rows=[json.loads(line) for line in (root/'examples.jsonl').read_text().splitlines()]
    assert manifest['status']=='complete' and len(rows)==400 and len(fake.calls)==20
    assert len({r['example_id'] for r in rows})==400
    assert len({r['trajectory_id'] for r in rows})==20
    assert manifest['trajectory_order']==[f'q{i:03d}' for i in range(1,21)]
    assert manifest['examples_sha256']==p.sha((root/'examples.jsonl').read_bytes())
    assert (root/'examples.jsonl').read_bytes()==b''.join((root/'accepted'/qid/'examples.jsonl').read_bytes() for qid in manifest['trajectory_order'])
    prov=json.loads((root/'accepted/q020/provenance.json').read_text())
    assert prov['assignment']==cfg.assignments[-1].model_dump() and len(prov['counterfactual_pairs'])==2


def test_unknown_feedback_never_forwarded():
    assert feedback(['arbitrary teacher instruction or secret'])==[]
    for code in GUIDANCE:assert classify(ContractError(code))==[code]
