from adr.mlp.embedding_limits import token_count, MAX_INPUT_TOKENS
"""Offline smoke runs the actual vendored dispatch/process/scrape method bodies.
Heavy GR dependencies and all planner, embedding and retrieval I/O are mocked.
"""
import ast
import asyncio
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from adr.core.types import Budget,Query,ResearchTask
from adr.mlp.inference import DEFAULT_RUN,FrozenMLP
from adr.mlp.query_gate import QueryGate,safe_text
from adr.synthetic.schema import ResearchSnapshot,derive_features,serialize_context

VENDOR=Path('third_party/gpt-researcher/gpt_researcher')


@pytest.fixture
def hooks():
    spec=importlib.util.spec_from_file_location('offline_query_hooks',VENDOR/'utils/query_gate.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conductor(hooks):
    source=ast.parse((VENDOR/'skills/researcher.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ResearchConductor')
    names={'_get_context_by_web_search','_process_sub_query','_scrape_data_by_urls'}
    cls.body=[n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name in names]
    scope={'active_trace':hooks.active_trace,'branch_id':hooks.study_branch_id,'candidate_id':hooks.study_candidate_id,'asyncio':asyncio,'active_observation_v2':hooks.active_observation_v2,'current_branch_v2':hooks.current_branch_v2,'active_gate':hooks.active_gate,'QueryGateError':hooks.QueryGateError,
           'note_retrieval':hooks.note_retrieval,
           '_global_scraper_cache':NS(get=lambda url:None,put=lambda *args:None)}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(VENDOR/'skills/researcher.py'),'exec'),scope)
    c=scope['ResearchConductor']()
    c.logger=logging.getLogger('offline_gate')
    c.json_handler=None
    c._mcp_results_cache=None
    c._mcp_cache_lock=asyncio.Lock()
    c._get_mcp_strategy=lambda:'disabled'
    c._tavily_mcp_redundant_with_direct=lambda *args:False
    c._combine_mcp_and_web_context=lambda m,w,q:w
    c.plan_research=AsyncMock(side_effect=lambda *args:['execute query','skip query'])
    c._search_relevant_source_urls=AsyncMock(return_value=([],[{'raw_content':'POST RETRIEVAL RESULT','url':'mock-url'}]))
    c.researcher=NS(query='branch query',retrievers=[type('MockRetriever',(),{})],verbose=False,
                    report_type='research_report',vector_store=None,
                    scraper_manager=NS(browse_urls=AsyncMock(return_value=[])),
                    context_manager=NS(get_similar_content_by_query=AsyncMock(return_value='POST RETRIEVAL RESULT')))
    return c


def bridge(tmp_path,hooks,model):
    root=NS(deep_researcher=NS(trajectory_logger=NS(_global_evidence={},trajectory=NS(subquestions=['known need'],rounds=[]))))
    task=ResearchTask(query=Query(id='smoke',text='Original question',dataset='offline'),budget=Budget(max_steps=32,max_searches=64,max_reads=128,max_tokens=50000))
    return QueryGate(model,root,task,tmp_path/'gate.jsonl',NS(get_totals=lambda:{'input_tokens':100,'output_tokens':20}),hooks)


class StubModel:
    provenance={}
    def __init__(self,probabilities):
        self.probabilities=probabilities
        self.snapshots=[]
    async def predict_batch(self,q,s,c):
        self.snapshots.append(s.model_copy(deep=True))
        return self.probabilities,serialize_context(s),derive_features(q,s).values,[0.]*15
    async def close(self): pass


@pytest.mark.asyncio
async def test_disabled_actual_dispatch_preserves_all_queries(hooks):
    c=conductor(hooks)
    result=await c._get_context_by_web_search('branch query')
    assert result.count('POST RETRIEVAL RESULT')==3
    assert [call.args[0] for call in c._search_relevant_source_urls.await_args_list]==['execute query','skip query','branch query']


@pytest.mark.asyncio
async def test_execute_skip_same_batch_snapshot_logging_and_no_leakage(tmp_path,hooks):
    c=conductor(hooks)
    model=StubModel([.8,.2,.5])
    gate=bridge(tmp_path,hooks,model)
    token=hooks.active_gate.set(gate)
    try:
        result=await c._get_context_by_web_search('branch query')
    finally:
        hooks.active_gate.reset(token)
        await gate.close()
    assert result.count('POST RETRIEVAL RESULT')==2
    assert [call.args[0] for call in c._search_relevant_source_urls.await_args_list]==['execute query','branch query']
    rows=[json.loads(l) for l in (tmp_path/'gate.jsonl').read_text().splitlines()]
    proposed=[r for r in rows if r['event']=='proposed']
    assert len(proposed)==3 and len({r['snapshot_sha256'] for r in proposed})==1
    assert all(r['raw_features']==proposed[0]['raw_features'] for r in proposed)
    skipped=next(r for r in rows if r['event']=='skipped')
    assert not skipped['retrieval_invoked'] and not skipped['retrieval_path_dispatched']
    finished=[r for r in rows if r['event']=='finished']
    assert len(finished)==2 and all(r['retrieval_invoked'] for r in finished)
    assert gate.history==['execute query','branch query']
    assert 'POST RETRIEVAL RESULT' not in (tmp_path/'gate.jsonl').read_text()
    assert model.snapshots[0].executed_subqueries==[]
    assert model.snapshots[0].budget.used_tokens==120
    assert model.snapshots[0].budget.used_steps==0
    assert max(i for i,r in enumerate(rows) if r['event']=='proposed') < min(i for i,r in enumerate(rows) if r['event']=='retrieval_invoked')


@pytest.mark.asyncio
async def test_empty_batch_no_threshold_override(tmp_path,hooks):
    c=conductor(hooks); gate=bridge(tmp_path,hooks,StubModel([.1,.2,.3]))
    token=hooks.active_gate.set(gate)
    try: assert await c._get_context_by_web_search('branch query')==[]
    finally: hooks.active_gate.reset(token); await gate.close()
    c._search_relevant_source_urls.assert_not_awaited()
    assert 'empty_batch' in (tmp_path/'gate.jsonl').read_text()


@pytest.mark.asyncio
async def test_failure_not_swallowed_and_no_provider_error_leak(tmp_path,hooks):
    c=conductor(hooks); model=StubModel([])
    model.predict_batch=AsyncMock(side_effect=RuntimeError('secret raw provider body'))
    gate=bridge(tmp_path,hooks,model); token=hooks.active_gate.set(gate)
    try:
        with pytest.raises(hooks.QueryGateError,match='Frozen query gate failed'):
            await c._get_context_by_web_search('branch query')
    finally: hooks.active_gate.reset(token); await gate.close()
    c._search_relevant_source_urls.assert_not_awaited()
    assert 'secret raw provider body' not in (tmp_path/'gate.jsonl').read_text()


@pytest.mark.asyncio
async def test_frozen_input_order_normalization_and_cache():
    calls=[]
    async def embed(texts,dimensions):
        calls.extend(texts)
        return [[float({'question':1,'candidate':3}.get(t,2))]*dimensions for t in texts]
    model=FrozenMLP(embedder=embed)
    seen=[]
    hook=model.model[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].clone()))
    snapshot=ResearchSnapshot()
    p,ctx,raw,norm=await model.predict_batch('question',snapshot,['candidate','candidate'])
    await model.predict_batch('question',snapshot,['candidate'])
    hook.remove()
    assert len(calls)==3 and len(p)==2
    X=seen[0].numpy()
    assert X.shape==(2,4623)
    np.testing.assert_array_equal(X[:,:1536],1)
    np.testing.assert_array_equal(X[:,1536:3072],2)
    np.testing.assert_array_equal(X[:,3072:4608],3)
    expected=(np.array(raw)-model.stats['mean'])/model.stats['scale']
    np.testing.assert_allclose(X[0,4608:],expected,rtol=1e-6)
    assert ctx==serialize_context(snapshot)
    assert not model.model.training and not any(p.requires_grad for p in model.model.parameters())


@pytest.mark.parametrize('change',['dimension','threshold','checkpoint','feature_order'])
def test_incompatible_artifacts_fail(tmp_path,change):
    run=tmp_path/'run';run.mkdir()
    for name in ('manifest.json','config.json','best_mlp.pt'):
        shutil.copy2(DEFAULT_RUN/name,run/name)
    manifest=json.loads((run/'manifest.json').read_text())
    if change in ('dimension','threshold'):
        cfg=json.loads((run/'config.json').read_text());cfg['input_dimensions' if change=='dimension' else 'threshold']=123
        (run/'config.json').write_text(json.dumps(cfg))
        manifest['artifacts_sha256']['config.json']=hashlib.sha256((run/'config.json').read_bytes()).hexdigest()
    elif change=='checkpoint': (run/'best_mlp.pt').write_bytes(b'incompatible')
    else:
        from adr.mlp.prepare import normalize
        with pytest.raises(ValueError,match='feature order'):
            normalize(np.zeros((1,15)),{'feature_names':list(reversed(derive_features('q',ResearchSnapshot()).names))})
        return
    (run/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incompatibility|checksum'):
        FrozenMLP(run,embedder=AsyncMock())


def test_privacy_transform(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-test-secret')
    result=safe_text('See https://private.example/path?key=abc and offline-test-secret')
    assert 'https://' not in result and 'private.example' not in result and 'offline-test-secret' not in result


@pytest.mark.asyncio
async def test_snapshot_uses_prior_committed_evidence_and_real_counters(tmp_path,hooks):
    gate=bridge(tmp_path,hooks,StubModel([]));c=conductor(hooks)
    logger=gate.researcher.deep_researcher.trajectory_logger
    logger._global_evidence['e1']=NS(content='Established evidence',source_url='https://source.example',was_retained=False,compressed=False,retrieval_round=1)
    logger.trajectory.rounds=[NS(frontier=[NS(node_id='n1',subquery='prior goal',status='completed')])]
    token=hooks.active_gate.set(gate)
    hooks.note_search('planning query');hooks.note_reads(2)
    hooks.active_gate.reset(token)
    snapshot=gate.snapshot(c)
    assert snapshot.budget.used_searches==1 and snapshot.budget.used_reads==2
    assert snapshot.budget.used_steps==1 and snapshot.executed_subqueries==['planning query']
    assert snapshot.evidence[0].retained is False
    assert next(b for b in snapshot.branches if b.id=='n1').status=='done'
    assert 'https://' not in serialize_context(snapshot)
    logger._global_evidence['e1'].content='rewritten'
    with pytest.raises(ValueError,match='body changed'): gate.snapshot(c)
    await gate.close()


@pytest.mark.asyncio
async def test_real_checkpoint_offline_smoke_through_vendor_dispatch(tmp_path,hooks):
    async def embed(texts,dimensions):
        return [[0.0]*dimensions for _ in texts]
    model=FrozenMLP(embedder=embed)
    gate=bridge(tmp_path,hooks,model);c=conductor(hooks)
    token=hooks.active_gate.set(gate)
    try: await c._get_context_by_web_search('branch query')
    finally: hooks.active_gate.reset(token);await gate.close()
    rows=[json.loads(l) for l in (tmp_path/'gate.jsonl').read_text().splitlines()]
    decisions=[r for r in rows if r['event']=='proposed']
    assert len(decisions)==3
    assert c._search_relevant_source_urls.await_count==sum(r['decision']=='EXECUTE' for r in decisions)
    assert all(len(r['raw_features'])==len(r['normalized_features'])==15 for r in decisions)

@pytest.mark.asyncio
async def test_endpoint_mismatch_fails_without_client(monkeypatch):
    monkeypatch.delenv('OPENAI_BASE_URL',raising=False)
    model=FrozenMLP()
    with pytest.raises(ValueError,match='endpoint/model identity'):
        model.validate_endpoint()
    assert model.client is None


@pytest.mark.asyncio
async def test_bad_embedding_dimensions_abort():
    model=FrozenMLP(embedder=AsyncMock(return_value=[[1.0]]))
    with pytest.raises(ValueError,match='dimensions'):
        await model.predict_batch('q',ResearchSnapshot(),['c'])


@pytest.mark.asyncio
async def test_harness_concurrency_rejected_before_import():
    from adr.agents.gpt_researcher import GPTResearcherAgent
    from adr.agents.base import AgentContext
    agent=GPTResearcherAgent({'query_gate':{'enabled':True}})
    task=ResearchTask(query=Query(id='q',text='question',dataset='offline'))
    with pytest.raises(ValueError,match='harness concurrency'):
        await agent.run(task,AgentContext(None,None,{'config':{'concurrency':2}}))
    assert agent._imported is False


@pytest.mark.asyncio
async def test_zero_query_batch_is_explicit_and_has_no_embeddings(tmp_path,hooks):
    c=conductor(hooks);c.plan_research=AsyncMock(return_value=[])
    c.researcher.report_type='subtopic_report'
    embed=AsyncMock();model=FrozenMLP(embedder=embed)
    gate=bridge(tmp_path,hooks,model);token=hooks.active_gate.set(gate)
    try: assert await c._get_context_by_web_search('branch query')==[]
    finally: hooks.active_gate.reset(token);await gate.close()
    embed.assert_not_awaited();c._search_relevant_source_urls.assert_not_awaited()
    assert 'empty_batch' in (tmp_path/'gate.jsonl').read_text()


@pytest.mark.asyncio
async def test_second_batch_has_completed_history_not_sibling_results(tmp_path,hooks):
    c=conductor(hooks);model=StubModel([.8,.2,.5]);gate=bridge(tmp_path,hooks,model)
    token=hooks.active_gate.set(gate)
    try:
        await c._get_context_by_web_search('branch query')
        await c._get_context_by_web_search('branch query')
    finally: hooks.active_gate.reset(token);await gate.close()
    assert model.snapshots[0].executed_subqueries==[]
    assert model.snapshots[1].executed_subqueries==['execute query','branch query']
    assert model.snapshots[1].budget.used_steps==3
    assert 'POST RETRIEVAL RESULT' not in serialize_context(model.snapshots[1])


def test_deep_branch_gate_error_rethrows():
    # The outer concurrent branch runner normally catches Exception. Assert the
    # dedicated error bypass stays in front of that best-effort handler.
    tree=ast.parse((VENDOR/'skills/deep_research.py').read_text())
    process=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='process_query')
    handlers=next(n for n in ast.walk(process) if isinstance(n,ast.Try)).handlers
    assert isinstance(handlers[0].type,ast.Name) and handlers[0].type.id=='QueryGateError'
    assert isinstance(handlers[0].body[0],ast.Raise)


@pytest.mark.asyncio
async def test_sensitive_candidate_fails_instead_of_changing_model_input(tmp_path,hooks):
    c=conductor(hooks);c.plan_research=AsyncMock(return_value=['https://private.example/query'])
    model=StubModel([.9,.9]);gate=bridge(tmp_path,hooks,model);token=hooks.active_gate.set(gate)
    try:
        with pytest.raises(hooks.QueryGateError):
            await c._get_context_by_web_search('branch query')
    finally: hooks.active_gate.reset(token);await gate.close()
    assert model.snapshots==[]
    c._search_relevant_source_urls.assert_not_awaited()
    assert 'private.example' not in (tmp_path/'gate.jsonl').read_text()


@pytest.mark.parametrize('enforce', [False, True])
def test_bridge_budget_enforcement_mode(tmp_path,hooks,enforce):
    gate=bridge(tmp_path,hooks,StubModel([]))
    gate.enforce_budget=enforce
    gate.searches=100
    try:
        if enforce:
            with pytest.raises(ValueError,match='used_searches exceeds maximum'):
                gate.snapshot(conductor(hooks))
        else:
            assert gate.snapshot(conductor(hooks)).budget.used_searches==100
    finally:
        gate.log.close()


@pytest.mark.asyncio
async def test_large_context_passed_to_embedder_and_logged(tmp_path,hooks):
    calls=[]
    async def embed(texts,dimensions):
        calls.extend(texts)
        assert all(token_count(t)<=MAX_INPUT_TOKENS for t in texts)
        return [[0.0]*dimensions for _ in texts]
    model=FrozenMLP(embedder=embed)
    gate=bridge(tmp_path,hooks,model)
    gate.researcher.deep_researcher.trajectory_logger._global_evidence={'e1':NS(
        compressed=False,source_url='local-source',content='prior evidence '*100000,
        was_retained=True,retrieval_round=1)}
    c=conductor(hooks)
    try:
        await gate.dispatch(c,['candidate'],[],[])
    finally:
        await gate.close()
    records=[json.loads(l) for l in (tmp_path/'gate.jsonl').read_text().splitlines()]
    bounding=next(r for r in records if r['event']=='context_bounding')
    snapshot=next(r for r in records if r['event']=='snapshot')
    assert bounding['bounded'] and bounding['original_bytes']>1000000
    assert bounding['final_bytes']==len(snapshot['context_before'].encode())
    assert token_count(snapshot['context_before'])<=MAX_INPUT_TOKENS
    assert snapshot['context_before'] in calls
    assert 'POST RETRIEVAL RESULT' not in snapshot['context_before']


@pytest.mark.asyncio
async def test_overflow_diagnostics_no_evidence_bodies(tmp_path,hooks):
    model=StubModel([])
    gate=bridge(tmp_path,hooks,model)
    gate.history=['query '*10000]
    gate.searches=22; gate.reads=70
    gate.researcher.deep_researcher.trajectory_logger._global_evidence={'e1':NS(
        compressed=False,source_url='source',content='PRIVATE EVIDENCE BODY',
        was_retained=True,retrieval_round=1)}
    try:
        with pytest.raises(hooks.QueryGateError,match='context_bounding'):
            await gate.dispatch(conductor(hooks),['candidate'],[],[])
    finally:
        await gate.close()
    assert not model.snapshots
    content=(tmp_path/'gate.jsonl').read_text()
    assert 'PRIVATE EVIDENCE BODY' not in content
    records=[json.loads(l) for l in content.splitlines()]
    d=next(r for r in records if r['event']=='context_diagnostics')
    assert d['metadata_tokens']>8192 and d['evidence_items']==1
    assert d['retained_count']==1 and d['pruned_count']==0
    assert d['history_count']==1 and d['branch_count']>=1
    assert d['budget']['used_searches']==22 and d['budget']['used_reads']==70
    assert len(d['raw_features'])==15 and len(d['candidate_ids'])==1
    assert records.index(d)<next(i for i,r in enumerate(records) if r['event']=='gate_error')
