import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from adr.development.trace import DevelopmentTrace, sizes
from adr.datasets.loader import load_queries
from tests.test_mlp_query_gate import hooks, conductor, VENDOR


def make(tmp_path, hooks):
    return DevelopmentTrace(tmp_path/'trace.jsonl','Development question',{'max_tokens':400000,'enforce':False},
        NS(get_totals=lambda:{'input_tokens':30,'output_tokens':10}),NS(snapshot=lambda:{'llm':2}),hooks.study_candidate_id)


def rows(t):
    return [json.loads(s) for s in t.path.read_text().splitlines()]


def test_hierarchy_publication_isolation_lossless(tmp_path,hooks):
    t=make(tmp_path,hooks)
    a,b=t.plan([{'query':'a','researchGoal':'A'},{'query':'b','researchGoal':'B'}],None,3,4)
    t.activate(a)
    c=t.snapshot(a,['query'])[0]
    page={'url':'https://example.org/p','raw_content':'漢字 evidence '*5000}
    t.raw([page],a,c)
    result={'learnings':['Exact finding'],'followUpQuestions':['Exact follow-up'],'context':'full relevant text','sources':[page]}
    t.finish(a,result)
    t.activate(b);t.snapshot(b,['sibling'])
    snap=next(r for r in reversed(rows(t)) if r['event']=='batch_snapshot')
    assert snap['processed_result_ids']==snap['evidence_ids']==[]
    assert all('result_id' not in x and 'evidence_ids' not in x for x in snap['branches'])
    t.finish(b,None);t.publish([a,b],4)
    child=t.plan([{'query':'child','researchGoal':'Child'}],a,2,2)[0]
    t.activate(child);t.snapshot(child,['next'])
    snap=next(r for r in reversed(rows(t)) if r['event']=='batch_snapshot')
    assert snap['completed_branch_ids']==[a]
    assert snap['processed_result_ids']==[t.branches[a]['result_id']]
    assert next(x for x in snap['branches'] if x['id']==child)['parent_id']==a
    raw=next(r for r in rows(t) if r['event']=='record' and r['kind']=='raw_page')
    assert raw['value']==page and raw['sizes']==sizes(page)
    assert snap['raw_state_sizes']==sizes(page)
    processed=next(r for r in rows(t) if r['event']=='record' and r['kind']=='processed_result')
    assert processed['value']==result
    edge=next(r for r in rows(t) if r['event']=='record_acquired' and r['record_id']==raw['record_id'])
    assert (edge['branch_id'],edge['candidate_id'])==(a,c)
    t.close('research_completed')


@pytest.mark.asyncio
@pytest.mark.parametrize("exact_numerical", [False, True])
async def test_actual_conductor_behavior_counts_no_sibling_leakage(tmp_path,hooks,exact_numerical):
    plain=conductor(hooks)
    plain.researcher.scraper_manager.browse_urls.side_effect=lambda *args: []
    expected=await plain._get_context_by_web_search('branch query')
    t=make(tmp_path,hooks)
    if exact_numerical:
        t.canonical_source=lambda: {}
        t.budget.update(max_steps=32,max_searches=64,max_reads=128)
    b=t.plan([{'query':'branch query','researchGoal':'Need'}],None,3,4)[0];t.activate(b)
    c=conductor(hooks)
    c.researcher.scraper_manager.browse_urls.side_effect=lambda *args: []
    token=hooks.active_trace.set(t);branch=hooks.study_branch_id.set(b)
    try:actual=await c._get_context_by_web_search('branch query')
    finally:hooks.study_branch_id.reset(branch);hooks.active_trace.reset(token)
    assert actual==expected
    assert c.plan_research.await_count==plain.plan_research.await_count==1
    assert c._search_relevant_source_urls.await_args_list==plain._search_relevant_source_urls.await_args_list
    assert c.researcher.context_manager.get_similar_content_by_query.await_count==plain.researcher.context_manager.get_similar_content_by_query.await_count==3
    rr=rows(t);snap=next(r for r in rr if r['event']=='batch_snapshot')
    assert not snap['evidence_ids'] and not snap['executed_candidate_ids']
    proposed=[r for r in rr if r['event']=='candidate_proposed'];assert len(proposed)==3
    assert len({r['batch_id'] for r in proposed})==1
    assert max(r['sequence'] for r in proposed)<min(r['sequence'] for r in rr if r['event']=='candidate_dispatched')
    assert len([r for r in rr if r['event']=='record_acquired' and r['record_id'].startswith('raw_page-')])==3
    assert {r['candidate_id'] for r in rr if r['event']=='record_acquired' and r['record_id'].startswith('raw_page-')}=={r['id'] for r in proposed}
    t.close('research_completed')


def test_accounting_hooks_and_partial_close(tmp_path,hooks):
    t=make(tmp_path,hooks);token=hooks.active_trace.set(t)
    try:hooks.note_search('planning');hooks.note_reads(3)
    finally:hooks.active_trace.reset(token)
    t.close('failed');end=rows(t)[-1]
    assert end['status']=='failed'
    assert end['accounting']['search_dispatches']==1 and end['accounting']['read_attempts']==3
    assert end['accounting']['tracked_tokens']['total_tokens']==40


def test_actual_publication_hook_order_and_parent_argument():
    source=(VENDOR/'skills/deep_research.py').read_text()
    assert source.index('trace.publish(study_ids, round_id)')>source.index('self.trajectory_logger.record_round(')
    assert '_study_parent=(study_ids[' in source
    assert source.index('await self.process_research_results(')<source.index('trace.finish(bid, result)')
    assert 'results = await asyncio.gather(*[process_query(q) for q in serp_queries])' in source


def test_development_query_separate_from_benchmarks():
    qs=load_queries('development',path='data/development/representation-study.jsonl')
    assert len(qs)==1 and qs[0].dataset=='development' and qs[0].id=='dev-representation-001'
    with pytest.raises(ValueError,match='explicit local'):load_queries('development')

@pytest.mark.asyncio
async def test_actual_outer_hooks_stable_publication_and_behavior(tmp_path,hooks):
    tree=ast.parse((VENDOR/'skills/deep_research.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepResearchSkill')
    method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='deep_research')
    start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='observer_v2' for t in n.targets))
    end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='publication_results_v2' for t in n.targets))
    publication=next(n for n in method.body if isinstance(n,ast.If) and 'trace.publish' in ast.unparse(n))
    selected=[n for n in method.body[start:end+1] if not isinstance(n,ast.AsyncFunctionDef)]+[publication,ast.Return(value=ast.Name(id='results',ctx=ast.Load()))]
    fn=ast.AsyncFunctionDef(name='run',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['self','serp_queries','process_query']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=selected,decorator_list=[])
    scope={'asyncio':asyncio,'active_observation_v2':hooks.active_observation_v2,'current_branch_v2':hooks.current_branch_v2,
           'QueryGateError':hooks.QueryGateError,'active_trace':hooks.active_trace,'study_branch':hooks.study_branch_id,
           '_study_parent':None,'depth':3,'breadth':2,'round_id':1}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'offline-outer','exec'),scope)
    t=make(tmp_path,hooks);queries=[{'query':'a','researchGoal':'A'},{'query':'b','researchGoal':'B'}];calls=[]
    async def process(q):
        calls.append(q['query'])
        if hooks.active_trace.get():t.snapshot(hooks.study_branch_id.get(),['inner'])
        await asyncio.sleep(0)
        return {'learnings':[q['query']+' finding'],'followUpQuestions':['find details']}
    plain=await scope['run'](NS(concurrency_limit=1),queries,process)
    calls.clear();token=hooks.active_trace.set(t)
    try:logged=await scope['run'](NS(concurrency_limit=1),queries,process)
    finally:hooks.active_trace.reset(token)
    assert logged==plain and calls==['a','b']
    rr=rows(t);assert all(not r['processed_result_ids'] for r in rr if r['event']=='batch_snapshot')
    assert t.published==['b0001','b0002']
    assert max(r['sequence'] for r in rr if r['event']=='branch_processed')<next(r['sequence'] for r in rr if r['event']=='round_published')
    t.close('research_completed')

@pytest.mark.asyncio
async def test_candidate_failure_preserves_records_and_resets_context(tmp_path,hooks):
    t=make(tmp_path,hooks);b=t.plan([{'query':'q','researchGoal':'g'}],None,3,4)[0]
    cid=t.snapshot(b,['q'])[0]
    async def failing():
        t.raw([{'url':'u','raw_content':'preserved'}],b,cid)
        raise RuntimeError('offline failure')
    with pytest.raises(RuntimeError):await t.dispatch(failing,cid)
    assert hooks.study_candidate_id.get() is None
    assert t.queries[cid]['status']=='failed'
    t.close('failed')
    assert any(r.get('kind')=='raw_page' for r in rows(t))


def test_development_config_gate_disabled_and_no_judges():
    import yaml
    cfg=yaml.safe_load(Path('configs/representation_study.yaml').read_text())
    agent=yaml.safe_load(Path(cfg['agent']['config']).read_text())
    assert cfg['dataset']['name']=='development' and cfg['dataset']['limit']==1
    assert not cfg['eval']['official_benches'] and not agent['query_gate']['enabled']
    assert agent['development_trace']['enabled']
    assert (agent['depth'],agent['breadth'],agent['concurrency'])==(3,4,1)
    assert agent['env']['GR_ORCHESTRATOR']=='none'


@pytest.mark.asyncio
async def test_adapter_rejects_benchmark_or_gate_before_loading_provider():
    from adr.agents.gpt_researcher import GPTResearcherAgent
    from adr.core.types import ResearchTask,Query,Budget
    from unittest.mock import Mock
    for dataset,enabled in [('deep_research_bench',False),('development',True)]:
        agent=GPTResearcherAgent({'development_trace':{'enabled':True},'query_gate':{'enabled':enabled}})
        agent._modules=Mock(side_effect=AssertionError('must not initialize providers'))
        task=ResearchTask(query=Query(id='dev',text='question',dataset=dataset),budget=Budget())
        with pytest.raises(ValueError,match='Development trace'):
            await agent.run(task,NS(extra={'config':{'concurrency':1}}))
        agent._modules.assert_not_called()

@pytest.mark.asyncio
async def test_future_skip_record_has_no_dispatch(tmp_path,hooks):
    from unittest.mock import AsyncMock
    t=make(tmp_path,hooks);b=t.plan([{'query':'q','researchGoal':'g'}],None,3,4)[0]
    cid=t.snapshot(b,['q'])[0];t.decision(cid,'SKIP');operation=AsyncMock()
    assert await t.dispatch(operation,cid)==''
    operation.assert_not_awaited()
    assert not any(r['event']=='candidate_dispatched' for r in rows(t))
    assert t.accounting()['candidate_dispatches']==0
    t.close('research_completed')
