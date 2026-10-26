import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from adr.orchestration.branch_budget import DepthBudget, VERSION
from tests.test_mlp_query_gate import hooks
from tests.test_gpt_researcher_agent import fake_gpt_researcher

VENDOR=Path('third_party/gpt-researcher/gpt_researcher')
def hook():
    s=importlib.util.spec_from_file_location('offline_expansion',VENDOR/'utils/branch_expansion.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def branches():return {'r':dict(tree_depth=1,parent_id=None,status='done'),'a':dict(tree_depth=2,parent_id='r',status='done'),'b':dict(tree_depth=2,parent_id='r',status='done')}
def test_rule_publication_and_idempotence():
    p=DepthBudget();b=branches()
    assert p.decide('r',3,b,list(b))['decision']=='EXPAND'
    a=p.decide('a',2,b,list(b));assert a['decision']=='EXPAND' and a==p.decide('a',2,b,list(b))
    assert p.decide('b',2,b,list(b))['decision']=='STOP'
    assert DepthBudget().decide('a',2,b,[])['fail_open']
    assert DepthBudget().decide('missing',2,b,[])['decision']=='EXPAND'

def test_disabled_and_exception_fail_open():
    h=hook();assert h.allow_expansion(None,3)
    class Broken:
        def allow(self,*a):raise ValueError()
        def failed_open(self,*a):raise ValueError()
    token=h.active_expansion.set(Broken())
    try:assert h.allow_expansion(None,2)
    finally:h.active_expansion.reset(token)

@pytest.mark.asyncio
@pytest.mark.parametrize('enabled',[False,True])
async def test_actual_recursive_boundary_preserves_work_and_blocks_descendants(enabled):
    tree=ast.parse((VENDOR/'skills/deep_research.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepResearchSkill');method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='deep_research')
    loop=next(n for n in method.body if isinstance(n,ast.For) and 'allow_expansion' in ast.unparse(n))
    assert ast.unparse(method).index('trace.publish')<ast.unparse(method).index('allow_expansion')
    # Execute the actual vendored expansion check and recursive call, without network setup.
    conditional=loop.body[1];call=next(n for n in conditional.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Await))
    conditional.body=conditional.body[:conditional.body.index(call)+1]
    args=['self','results','result_node_ids','study_results','study_ids','trace','depth','child_breadth','default_child_breadth','progress','all_learnings','all_citations','all_visited_urls','on_progress']
    fn=ast.AsyncFunctionDef(name='boundary',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in args],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[loop],decorator_list=[])
    h=hook();scope={'allow_expansion':h.allow_expansion};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'offline-boundary','exec'),scope)
    p=DepthBudget();bs={};published=[];counts={1:0,2:0,3:0}
    class Controller:
        def allow(self,b,d):return p.decide(b,d,bs,published)['decision']=='EXPAND'
    async def recurse(query='',breadth=4,depth=3,_study_parent=None,**kw):
        ids=[];results=[]
        for i in range(breadth):
            bid=f'{_study_parent}/{i}';ids.append(bid);bs[bid]=dict(tree_depth=4-depth,parent_id=_study_parent,status='done');results.append(dict(researchGoal=bid,followUpQuestions=['unresolved']))
            counts[4-depth]+=1
        published.extend(ids)
        await scope['boundary'](NS(stop_requested=False,deep_research=recurse),results,ids,results,ids,True,depth,{},2,NS(current_depth=0),[],{},set(),None)
        return {'learnings':[],'citations':{},'visited_urls':set()}
    token=h.active_expansion.set(Controller() if enabled else None)
    try:await recurse()
    finally:h.active_expansion.reset(token)
    assert counts==({1:4,2:8,3:8} if enabled else {1:4,2:8,3:16})

@pytest.mark.asyncio
async def test_adapter_completes_report(fake_gpt_researcher,tmp_path,monkeypatch,hooks):
    import sys
    from tests.test_gpt_researcher_agent import _FakeGPTResearcher
    from adr.agents.gpt_researcher import GPTResearcherAgent
    from adr.agents.base import AgentContext
    from adr.core.types import ResearchTask,Query
    h=hook()
    class Researcher(_FakeGPTResearcher):
        def __init__(self,**kw):
            super().__init__(**kw);self.deep_researcher.concurrency_limit=1;self.deep_researcher.depth=3;self.deep_researcher.breadth=4
        async def conduct_research(self):
            t=hooks.active_trace.get();assert hooks.active_gate.get() is None
            roots=t.plan([{'query':'root','researchGoal':'root'}],None,3,4)
            t.finish(roots[0],dict(researchGoal='root',learnings=['fact'],followUpQuestions=['need'],citations={}));t.publish(roots,1)
            assert h.allow_expansion(roots[0],3)
            children=t.plan([dict(query=str(i),researchGoal=str(i)) for i in range(2)],roots[0],2,2)
            for b in children:t.finish(b,dict(researchGoal=t.branches[b]['goal'],learnings=['fact'],followUpQuestions=['need'],citations={}))
            t.publish(children,2);assert h.allow_expansion(children[0],2);assert not h.allow_expansion(children[1],2)
            return await super().conduct_research()
    monkeypatch.setitem(sys.modules,'gpt_researcher.utils.query_gate',hooks);monkeypatch.setitem(sys.modules,'gpt_researcher.utils.branch_expansion',h);monkeypatch.setattr(sys.modules['gpt_researcher'],'GPTResearcher',Researcher)
    a=GPTResearcherAgent({'branch_expansion':{'policy':VERSION}})
    t=await a.run(ResearchTask(query=Query(id='offline',text='local',dataset='development')),AgentContext(None,None,{'run_dir':str(tmp_path)}))
    assert t.report and t.final_stats['branch_expansion']['expansions_stopped']==1
    assert h.active_expansion.get() is hooks.active_trace.get() is None

def test_saved_development_analysis_and_configs():
    import yaml
    c=yaml.safe_load(Path('configs/adaptive_depth_evaluation.yaml').read_text());assert c['dataset']['query_ids']==['51','52','53','54','55'];assert c['budget']['enforce'] is False
    a=yaml.safe_load(Path(c['agent']['config']).read_text());assert a['branch_expansion']['policy']==VERSION;assert a['env']['GR_ORCHESTRATOR']=='none'
    assert a['depth']==3 and a['breadth']==4 and a['concurrency']==1

def test_metrics_do_not_invent_scores_or_throughput(tmp_path):
    from adr.orchestration.branch_metrics import summarize
    (tmp_path/'metrics').mkdir();(tmp_path/'metrics/summary.json').write_text(json.dumps(dict(n_queries=5,n_with_report=5,n_errors=0,mean_tokens=460000,mean_wall_s=2412)))
    r=summarize(tmp_path);assert r['five_reports_zero_errors'];assert len(r['quality_missing'])==4
    assert r['table4_row']['tokens_K']==460 and r['table4_row']['Q_per_second'] is None
    assert r['literal_queries_per_second']==1/2412

def test_evaluation_isolated_headless_configuration():
    import yaml
    smoke=yaml.safe_load(Path('configs/adaptive_depth_smoke.yaml').read_text())
    evaluation=yaml.safe_load(Path('configs/adaptive_depth_evaluation.yaml').read_text())
    assert smoke['agent']['config']=='configs/agents/gpt_researcher_adaptive_depth.yaml'
    assert evaluation['agent']['config']=='configs/agents/gpt_researcher_adaptive_depth_headless.yaml'
    visible=yaml.safe_load(Path(smoke['agent']['config']).read_text())
    headless=yaml.safe_load(Path(evaluation['agent']['config']).read_text())
    assert visible['env']['GPT_RESEARCHER_BROWSER_HEADLESS']=='false'
    visible['env']['GPT_RESEARCHER_BROWSER_HEADLESS']='true'
    assert headless==visible
    assert evaluation['dataset']['query_ids']==['51','52','53','54','55']
    assert evaluation['dataset']['limit']==5
