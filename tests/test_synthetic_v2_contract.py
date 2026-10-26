"""Offline regression coverage for the first teacher-smoke failures."""
import json
from pathlib import Path

import pytest
from adr.synthetic_v2.schema import TeacherTrajectory, TeacherBudget
from adr.synthetic_v2.generate import fixture, compile_trajectory
from adr.synthetic_v2.prompt import teacher_messages, PROMPT_VERSION
from adr.synthetic.validate import load_heldout


def compile_data(data):
    return compile_trajectory(TeacherTrajectory.model_validate(data),load_heldout(Path('data/heldout/drb_baseline.json')))


def test_complete_lifecycle_and_history():
    tr=fixture();rows=compile_data(tr.model_dump())
    assert len(rows)==20
    assert len({r['counterfactual_pair_id'] for r in rows if r['counterfactual_pair_id']})==2
    assert len(tr.states[-1].source.completed_results)==7
    assert len(tr.states[-1].source.state.executed_subqueries)==16
    assert tr.states[1].source.completed_results[0]==tr.states[-1].source.completed_results[0]
    # Two later SKIPs were independently executed earlier; retain once, not twice.
    for c in tr.states[1].candidates[:2]:
        assert c.label=='SKIP'
        assert tr.states[-1].source.state.executed_subqueries.count(c.candidate_subquery)==1
    assert tr.states[-1].source.state.budget.used_steps==18+7
    assert all(r['observation_diagnostics']['tokens']<=2048 for r in rows)
    assert tr.states[-1].source.inputs(tr.original_question,'q')['numerical_features']['values'][10]==1
    assert tr.states[-1].source.inputs(tr.original_question,'q')['numerical_features']['values'][-1]==0
    assert all(s.source.observation().pending_needs for s in tr.states)


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d['states'][2]['source']['completed_results'].append(d['states'][2]['source']['completed_results'][0]),'Duplicate completed'),
    (lambda d:d['states'][1]['source']['completed_results'][0].update(goal='Wrong goal'),'matching goal'),
    (lambda d:d['states'][1]['source']['state']['branches'][0].update(status='active'),'completed task'),
    (lambda d:d['states'][1]['source']['state']['budget'].update(max_tokens=50000),'400000'),
    (lambda d:d['states'][0]['source']['state']['budget'].pop('max_tokens'),'Field required'),
    (lambda d:d['states'][1]['source']['state']['executed_subqueries'].pop(),'Executed history'),
    (lambda d:d['states'][2]['source']['state']['executed_subqueries'].append(d['states'][1]['candidates'][0]['candidate_subquery']),'Executed history'),
    (lambda d:d['states'][1]['source']['state']['budget'].update(used_steps=3),'Steps must equal'),
    (lambda d:d['states'][1]['source']['state']['budget'].update(used_searches=0),'Search attempts'),
    (lambda d:d['states'][1]['source']['state']['budget'].update(used_reads=0),'Read accounting'),
    (lambda d:d['states'][1].update(publications_before=[]),'Publication tasks'),
    (lambda d:d['states'][1]['publications_before'][0].update(evidence_ids=[]),'publish together'),
    (lambda d:d['states'][1]['publications_before'][0].update(task_ids=['b0','b0']),'Publication tasks'),
    (lambda d:d['states'][0]['source']['state']['evidence'].append(d['states'][1]['source']['state']['evidence'][0]),'Initial trajectory'),
    (lambda d:d.update(original_question='Formulate an original, specific research question about batteries.'),'meta instruction'),
    (lambda d:d['states'][0]['candidates'][0].update(candidate_subquery='create a comparative table of battery cooling'),'retrieval query'),
    (lambda d:d['states'][1]['candidates'][0].update(supporting_finding=None),'requires a visible'),
    (lambda d:d['states'][1]['candidates'][0].update(supporting_finding='A fact only in a future unpublished result'),'absent'),
    (lambda d:d['states'][1]['candidates'][0].update(supporting_finding=d['states'][1]['source']['state']['evidence'][0]['text']),'absent'),
])
def test_invalid_trace_rejected(mutation,match):
    data=fixture().model_dump();mutation(data)
    with pytest.raises(ValueError,match=match):compile_data(data)


def test_late_evicted_witness_rejected():
    d=fixture().model_dump()
    c=d['states'][1]['candidates'][0]
    c['counterfactual_pair_id']=None
    late=d['states'][-1]['candidates'][0]
    late.update({k:c[k] for k in ['candidate_subquery','label','decision_category','reason','supporting_finding']},counterfactual_pair_id='pair1')
    with pytest.raises(ValueError,match='absent'):compile_data(d)


def test_teacher_schema_has_no_inherited_budget_default():
    schema=TeacherTrajectory.model_json_schema()
    budget=schema['$defs']['TeacherBudget']
    assert budget['properties']['max_tokens']['const']==400000
    assert len(budget['required'])==8
    assert all('default' not in field for field in budget['properties'].values())
    assert '50000' not in json.dumps(schema)
    assert 'actual specific' in schema['properties']['original_question']['description']
    assert 'One card per task' in schema['$defs']['CompletedResult']['properties']['branch_id']['description']
    assert PROMPT_VERSION.endswith('draft2')
    text=teacher_messages('Cooling literature')[0]['content']
    for term in ['ONE CARD PER COMPLETED OUTER TASK','SKIPs append nothing','publications_before','not merely similar query words','actual visible conflict','not a hypothetical','never 50000']:
        assert term in text


def test_unpublished_sibling_pages_rejected():
    d=fixture().model_dump()
    # First current task is still active; sibling/result page cannot appear early.
    d['states'][0]['source']['state']['evidence']=d['states'][1]['source']['state']['evidence']
    with pytest.raises(ValueError,match='Initial trajectory'):compile_data(d)


def test_witness_already_visible_before_reversal_is_rejected():
    d=fixture().model_dump()
    witness=d['states'][1]['candidates'][0]['supporting_finding']
    query=d['states'][0]['candidates'][0]['candidate_subquery']
    for i in [0,1]:d['states'][i]['candidates'][0]['counterfactual_pair_id']=None
    d['states'][2]['candidates'][0].update(candidate_subquery=query,counterfactual_pair_id='pair1')
    d['states'][3]['candidates'][0].update(candidate_subquery=query,counterfactual_pair_id='pair1',label='SKIP',decision_category='already_established',supporting_finding=witness)
    history=[]
    for s in d['states']:
        s['source']['state']['executed_subqueries']=list(history)
        history.extend(c['candidate_subquery'] for c in s['candidates'] if c['label']=='EXECUTE')
    with pytest.raises(ValueError,match='already visible earlier'):compile_data(d)


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid',[False,True])
async def test_smoke2_runner_preserves_raw_and_never_retries(tmp_path,monkeypatch,invalid):
    import importlib.util
    from types import SimpleNamespace as NS
    path=Path('runs/synthetic-v2-smoke/teacher-smoke-2-20260923/run_smoke.py')
    spec=importlib.util.spec_from_file_location('smoke2_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=tmp_path
    monkeypatch.setenv('OPENAI_API_KEY','offline-test-key')
    monkeypatch.setenv('OPENAI_BASE_URL','https://offline.invalid/openai/v1/')
    monkeypatch.setattr('sys.argv',['run_smoke.py','--allow-api'])
    data=fixture().model_dump()
    if invalid:data['states'][1]['source']['completed_results']*=2
    raw=json.dumps(data);calls=[]
    class Client:
        def __init__(self):self._client=self
        def with_options(self,**kw):assert kw=={'max_retries':0};return self
        async def complete(self,*args,**kw):
            calls.append(1)
            return NS(text=raw,raw={'choices':[{'message':{'content':raw}}]},usage=NS(model_dump=lambda:{'total_tokens':123}))
        async def close(self):pass
    monkeypatch.setattr(module,'build_llm',lambda cfg:Client())
    await module.main()
    assert (tmp_path/'raw_teacher.txt').read_text()==raw
    assert json.loads((tmp_path/'validation.json').read_text())['structural_validation']==('failed' if invalid else 'passed')
    before=(tmp_path/'request.json').read_bytes()
    with pytest.raises(SystemExit,match='already attempted'):await module.main()
    assert calls==[1] and (tmp_path/'request.json').read_bytes()==before


def test_publication_does_not_charge_prior_sibling_reads_again():
    d=fixture().model_dump()
    # Reads accrued previously; another page is published at a later boundary.
    # Actual usage need not rise at publication itself.
    d['states'][2]['source']['state']['budget']['used_reads']=d['states'][1]['source']['state']['budget']['used_reads']
    compile_data(d)
