import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
import yaml
from adr.orchestration.redundancy import decide_view, decide, THRESHOLD
from adr.orchestration.live import PublicationTrace, RedundancyGate
from adr.mlp.branch_observation_v2 import construct
from tests.test_mlp_query_gate import hooks, conductor

ROOT=Path('runs/real-state-supervision/task-interpretation-regression-v1')
def packet():return json.loads((ROOT/'packets.json').read_text())['dev2-15']
def make(tmp_path,hooks,policy='conservative'):
    t=PublicationTrace(tmp_path/'audit.jsonl','Question',dict(max_tokens=400000,enforce=False),
        NS(get_totals=lambda:dict(input_tokens=10,output_tokens=5)),NS(snapshot=lambda:dict(llm=1)),
        hooks.study_candidate_id,policy=policy)
    return t,RedundancyGate(t,hooks,policy)
def seed(t):
    v=json.loads(packet()['context_before']);bs=t.plan([dict(query=b['goal'],researchGoal=b['goal']) for b in v['branches']],None,3,4)
    for bid,b in zip(bs,v['branches']):
        t.activate(bid);t.finish(bid,dict(researchGoal=b['goal'],learnings=b['findings'],followUpQuestions=b['followups'],citations=b['cited_insights']))
    t.publish(bs,1)
    current=t.plan([dict(query='current',researchGoal=v['current_need'])],bs[0],2,2)[0];t.activate(current)
    return current

def test_clear_cases_determinism_and_unknowns():
    p=packet();r=decide_view(p['context_before'],p['candidate_subquery']);assert r['decision']=='SKIP' and r==decide_view(p['context_before'],p['candidate_subquery'])
    for text in ['broken','{}','x '*20000]:assert decide_view(text,'query')['decision']=='EXECUTE'
    assert decide(None,'query')[0]['fail_open']
    assert decide_view(p['context_before'],p['candidate_subquery'],'always_execute')['decision']=='EXECUTE'
    for p in json.loads((ROOT/'packets.json').read_text()).values():assert decide_view(p['context_before'],p['candidate_subquery'])['threshold']==.22

def test_history_and_related_goals_are_not_answers():
    p=packet();v=json.loads(p['context_before']);q=p['candidate_subquery'];v['recent_queries']=[q]
    for b in v['branches']:b.update(goal=q,findings=[],followups=[q],cited_insights={})
    assert decide_view(json.dumps(v),q)['decision']=='EXECUTE'

@pytest.mark.parametrize('query',['measurement protocols heat pumps','heat pump COP distribution','heat pump backup fuel share','heat pumps in 2038'])
def test_missing_dimensions_numbers_fail_open(query):
    v=json.loads(packet()['context_before']);v['branches']=v['branches'][:1]
    v['branches'][0].update(findings=['heat pumps heat pump COP measurement backup fuel'],cited_insights={},followups=[])
    r=decide_view(json.dumps(v),query);assert r['decision']=='EXECUTE' and r['fail_open']

@pytest.mark.asyncio
@pytest.mark.parametrize('policy',['conservative','always_execute'])
async def test_actual_dispatch_no_duplicate_calls_and_snapshot_isolation(tmp_path,hooks,policy):
    t,g=make(tmp_path,hooks,policy);bid=seed(t);c=conductor(hooks);q=packet()['candidate_subquery']
    c.plan_research=AsyncMock(return_value=[q,'Find unresolved carbon costs']);c.researcher.query='another unresolved question'
    token=hooks.active_gate.set(g);trace=hooks.active_trace.set(t);branch=hooks.study_branch_id.set(bid)
    try:result=await c._get_context_by_web_search(c.researcher.query)
    finally:hooks.active_gate.reset(token);hooks.active_trace.reset(trace);hooks.study_branch_id.reset(branch)
    dispatched=[a.args[0] for a in c._search_relevant_source_urls.await_args_list]
    assert dispatched==([q,'Find unresolved carbon costs',c.researcher.query] if policy=='always_execute' else ['Find unresolved carbon costs',c.researcher.query])
    assert len(dispatched)==len(set(dispatched));assert c.plan_research.await_count==1
    assert result.count('POST RETRIEVAL RESULT')==len(dispatched)
    rows=[json.loads(x) for x in t.path.open()];ds=[r for r in rows if r['event']=='heuristic_decision']
    assert len(ds)==3 and len({r['snapshot_id'] for r in ds})==1
    assert max(r['sequence'] for r in ds)<min(r['sequence'] for r in rows if r['event']=='candidate_dispatched')
    skips={d['candidate_id'] for d in ds if d['decision']=='SKIP'}
    assert not any(r.get('candidate_id') in skips for r in rows if r['event']=='retrieval_invoked')
    assert len([r for r in rows if r['event']=='retrieval_invoked'])==len(dispatched)
    assert g.metrics()['skip_count']==(1 if policy=='conservative' else 0)
    t.close('completed')

def test_publication_parent_sibling_and_no_raw_pages(tmp_path,hooks):
    t,g=make(tmp_path,hooks);a,b=t.plan([dict(query='a',researchGoal='A'),dict(query='b',researchGoal='B')],None,3,4)
    t.finish(a,dict(researchGoal='A',learnings=['answer'],followUpQuestions=[],citations={}))
    t.snapshot(b,['candidate']);assert t.source.published_branches==()
    t.raw([{'raw_content':'FORBIDDEN BODY'}],b,None);t.finish(b,None);t.publish([a,b],1)
    child=t.plan([dict(query='c',researchGoal='C')],a,2,2)[0];t.snapshot(child,['candidate'])
    v=json.loads(construct(t.source,'candidate').text);assert v['branches'][0]['relation']=='parent'
    assert 'FORBIDDEN BODY' not in t.path.read_text();t.close('completed')

@pytest.mark.asyncio
async def test_corrupt_ledger_executes_once(tmp_path,hooks):
    t,g=make(tmp_path,hooks);b=seed(t);t.invalid='malformed_result';t.snapshot(b,['q'])
    c=NS(researcher=NS(vector_store=None),_process_sub_query=AsyncMock(return_value='ok'))
    assert await g.dispatch(c,['q'],[],[])==['ok'];assert c._process_sub_query.await_count==1;assert g.metrics()['fail_open_count']==1
    t.close('completed')

def test_config_parity_and_local_smoke_only():
    from adr.datasets.loader import load_queries
    a=yaml.safe_load(Path('configs/agents/gpt_researcher_redundancy_baseline.yaml').read_text())
    b=yaml.safe_load(Path('configs/agents/gpt_researcher_redundancy_conservative.yaml').read_text())
    assert a.pop('redundancy_gate')=={'policy':'always_execute'};assert b.pop('redundancy_gate')=={'policy':'conservative'};assert a==b
    assert (a['depth'],a['breadth'],a['concurrency'],a['scraper'],a['retriever'])==(3,4,1,'browser','tavily')
    assert a['env']['GR_ORCHESTRATOR']=='none' and a['env']['GPT_RESEARCHER_BROWSER_HEADLESS']=='true'
    cfg=yaml.safe_load(Path('configs/redundancy_smoke.yaml').read_text());qs=load_queries('development',path=cfg['dataset']['path'])
    assert len(qs)==1 and qs[0].id=='dev-redundancy-smoke-v1'
    # Intentionally do not invoke the benchmark loader or inspect its questions.
    for name in ('baseline','conservative'):
        cfg=yaml.safe_load(Path(f'configs/redundancy_{name}_evaluation.yaml').read_text())
        assert cfg['dataset']['query_ids']==['52','53','54','55'] and cfg['dataset']['limit']==4
        assert cfg['concurrency']==1 and cfg['budget']['enforce'] is False and cfg['eval']['official_benches']==[]

from tests.test_gpt_researcher_agent import fake_gpt_researcher
@pytest.mark.asyncio
@pytest.mark.parametrize('policy',['conservative','always_execute'])
async def test_adapter_mocked_report_completion_and_cleanup(fake_gpt_researcher,tmp_path,monkeypatch,hooks,policy):
    import sys
    from tests.test_gpt_researcher_agent import _FakeGPTResearcher
    from adr.agents.gpt_researcher import GPTResearcherAgent
    from adr.agents.base import AgentContext
    from adr.core.types import ResearchTask,Query
    from adr.core.instrument import CostMeter
    from adr.orchestration.metrics import summarize
    class Researcher(_FakeGPTResearcher):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self.deep_researcher.concurrency_limit=1
            self.mcp_configs=[];self.vector_store=None;self.source_urls=[];self.retrievers=[]
        async def conduct_research(self):
            trace=hooks.active_trace.get();gate=hooks.active_gate.get();bid=seed(trace)
            trace.snapshot(bid,[packet()['candidate_subquery'],'Find absent evidence'])
            async def operation(*args):hooks.note_retrieval();hooks.note_search(args[0]);hooks.note_reads(2);return 'result'
            c=NS(researcher=NS(vector_store=None),_process_sub_query=operation)
            await gate.dispatch(c,[packet()['candidate_subquery'],'Find absent evidence'],[],[])
            return await super().conduct_research()
    monkeypatch.setitem(sys.modules,'gpt_researcher.utils.query_gate',hooks)
    monkeypatch.setattr(sys.modules['gpt_researcher'],'GPTResearcher',Researcher)
    agent=GPTResearcherAgent({'redundancy_gate':{'policy':policy}});meter=CostMeter()
    task=ResearchTask(query=Query(id='offline-smoke',text='Development question',dataset='development'))
    ctx=AgentContext(None,None,{'run_dir':str(tmp_path),'meter':meter})
    trajectory=await agent.run(task,ctx)
    assert trajectory.report is not None and trajectory.final_stats['redundancy_gate']['candidate_count']==2
    assert trajectory.final_stats['redundancy_gate']['skip_count']==(policy=='conservative')
    assert meter.n_search_calls==(1 if policy=='conservative' else 2)
    assert meter.n_fetch_calls==(2 if policy=='conservative' else 4)
    assert hooks.active_trace.get() is hooks.active_gate.get() is None
    assert summarize(tmp_path)['skipped_retrieval_violations']==0

@pytest.mark.asyncio
async def test_research_exception_not_retried(tmp_path,hooks):
    t,g=make(tmp_path,hooks);b=seed(t);t.snapshot(b,['unknown query'])
    c=NS(researcher=NS(vector_store=None),_process_sub_query=AsyncMock(side_effect=RuntimeError('external failure')))
    with pytest.raises(RuntimeError,match='external failure'):await g.dispatch(c,['unknown query'],[],[])
    assert c._process_sub_query.await_count==1;t.close('failed')

@pytest.mark.asyncio
async def test_audit_write_failure_is_fail_open(tmp_path,hooks):
    t,g=make(tmp_path,hooks);bid=seed(t);t.snapshot(bid,[packet()['candidate_subquery']]);t.stream.close()
    c=NS(researcher=NS(vector_store=None),_process_sub_query=AsyncMock(return_value='ok'))
    assert await g.dispatch(c,[packet()['candidate_subquery']],[],[])==['ok']
    assert c._process_sub_query.await_count==1 and g.metrics()['skip_count']==0
    assert t.audit_errors>0;t.close('completed')

def test_observer_does_not_duplicate_raw_result_context(tmp_path,hooks):
    t,g=make(tmp_path,hooks);bid=t.plan([dict(query='q',researchGoal='goal')],None,3,4)[0]
    t.finish(bid,dict(researchGoal='goal',learnings=['finding'],followUpQuestions=[],citations={},context='RAW SECRET BODY',sources=[{'raw_content':'RAW SECRET BODY'}]))
    assert 'RAW SECRET BODY' not in t.path.read_text();t.close('completed')

def test_scoring_exception_is_fail_open(monkeypatch):
    import adr.orchestration.redundancy as m
    def broken(*args,**kwargs):raise RuntimeError('local scorer unavailable')
    monkeypatch.setattr(m,'TfidfVectorizer',broken)
    p=packet();r=m.decide_view(p['context_before'],p['candidate_subquery'])
    assert r['decision']=='EXECUTE' and r['reason']=='scoring_failed'
