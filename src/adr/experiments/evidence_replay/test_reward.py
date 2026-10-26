"""Deterministic synthetic vectors expose each component independently."""
import json
import socket
from dataclasses import replace

import numpy as np
import pytest

from .core import Page, Pool, Tokenizer, build_context
from .reward import CANDIDATES, RewardPool, Weights, pair_quality, total_reward
from .reward_diagnostics import correlation, ranks, summarize


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args,**kwargs):
        raise AssertionError('Offline reward tests forbid network')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket,'create_connection',denied)


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def rp():
    pages=tuple(Page(i,text,url,'s',1,1,0) for i,text,url in [
        ('root','root facts','https://one.test/a'),
        ('facet','other facts','https://two.test/b'),
        ('dup','root facts','https://three.test/c')])
    return RewardPool.from_arrays(Pool('q','test',pages,{}),[[1,0],[0,1],[1,0]],[[1,0],[0,1]])


def test_coverage_and_normalization(rp,tok):
    assert rp.score(['root'],tok)['coverage']==.5
    assert rp.score(['facet'],tok)['coverage']==.5
    assert rp.score(['root','facet'],tok)['coverage']==1
    assert rp.score(['root','dup'],tok)['coverage']==.5
    scaled=RewardPool.from_arrays(rp.pool,rp.vectors*5,rp.targets*7)
    assert scaled.score(['root','facet'],tok)['coverage']==1
    for ids in [[],['root'],['root','facet'],['root','dup']]:
        s=rp.score(ids,tok)
        assert all(0<=s[k]<=1 for k in ['coverage','diversity','redundancy','token_cost'])
        assert -1<=total_reward(s)<=1


def test_source_diversity_is_gated_and_not_semantic_pair_distance(rp,tok):
    a=rp.score(['root'],tok);b=rp.score(['root','facet'],tok)
    assert a['diversity']==0
    assert b['diversity']==pytest.approx(np.log(2)/np.log(3))
    same_host=replace(rp.pool,pages=tuple(replace(p,url='https://one.test/'+p.id) for p in rp.pool.pages))
    changed=RewardPool.from_arrays(same_host,rp.vectors,rp.targets)
    assert changed.score(['root','facet'],tok)['diversity']==0
    assert changed.score(['root','facet'],tok)['redundancy']==b['redundancy']


def test_duplicate_penalty_and_no_duplicate_coverage_gain(rp,tok):
    single=rp.score(['root'],tok);duplicates=rp.score(['root','dup'],tok)
    assert single['redundancy']==0 and duplicates['redundancy']==pytest.approx(1)
    assert duplicates['coverage']==single['coverage']
    assert total_reward(duplicates)<total_reward(single)
    assert rp.score(['root','facet'],tok)['redundancy']==0
    # Canonical text duplicates are penalized even if inconsistent vectors differ.
    altered=replace(rp.pool,pages=(rp.pool.pages[0],rp.pool.pages[1],replace(rp.pool.pages[2],text=' root  facts\n')))
    rp2=RewardPool.from_arrays(altered,[[1,0],[0,1],[0,1]],rp.targets)
    assert rp2.score(['root','dup'],tok)['redundancy']==1


def test_near_duplicate_threshold(tok):
    pool=Pool('q','t',(Page('a','a','https://a','s',1,1,0),Page('b','b','https://b','s',1,1,0)),{})
    vectors=[[1,0],[.95,np.sqrt(1-.95**2)]]
    rp=RewardPool.from_arrays(pool,vectors,[[1,0],[0,1]])
    assert rp.score(['a','b'],tok)['redundancy']==pytest.approx(.5)


def test_exact_cost_budget_boundary_and_empty(rp,tok):
    context,_,n=build_context(rp.pool,['root'],tok,50000)
    assert rp.score(['root'],tok,n)['context_tokens']==tok.count(context)
    assert rp.score(['root'],tok,n)['token_cost']==1
    valid=rp.score(['root'],tok,n)
    invalid=rp.score(['root'],tok,n-1)
    assert valid['feasible'] and not invalid['feasible']
    assert invalid['constraint']=='over_budget' and total_reward(invalid)==-1
    empty=rp.score([],tok,n)
    assert empty['coverage']==empty['diversity']==empty['redundancy']==empty['token_cost']==0
    assert total_reward(empty)==-1<total_reward(valid)
    with pytest.raises(ValueError,match='positive'):
        rp.score(['root'],tok,0)
    with pytest.raises(ValueError,match='Unknown'):
        rp.score(['unknown'],tok)


@pytest.mark.parametrize('vectors,targets',[
    ([[1,0]],[[1,0],[0,1]]),
    ([[1,0],[0,1],[np.nan,0]],[[1,0],[0,1]]),
    ([[1,0],[0,1],[0,0]],[[1,0],[0,1]]),
    ([[1,0],[0,1],[1,0]],[[1,0]]),
])
def test_missing_or_invalid_embeddings(rp,vectors,targets):
    with pytest.raises(ValueError,match='embeddings'):
        RewardPool.from_arrays(rp.pool,vectors,targets)


def test_missing_subquestion_embeddings_refused(tmp_path):
    from .test_replay import REAL
    raw=json.loads(REAL.read_bytes());p=tmp_path/'source.json';p.write_text(json.dumps(raw))
    with np.load(REAL.with_name(REAL.stem+'_emb.npz')) as z:
        np.savez(tmp_path/'source_emb.npz',**{k:z[k] for k in z.files if k!='subquestions'})
    with pytest.raises(ValueError,match='subquestion'):
        RewardPool.load(p)


def test_equal_query_cosine_different_coverage_and_redundancy(tok):
    # Equal root similarity and equal lengths, but opposite subquestion support.
    x=np.sqrt(.5)
    pool=Pool('q','t',tuple(Page(i,'facts','https://'+i+'.test','s',1,1,0) for i in ['a','b','c']),{})
    rp=RewardPool.from_arrays(pool,[[x,x],[x,-x],[x,x]],[[1,0],[0,1]])
    a=rp.score(['a'],tok);b=rp.score(['b'],tok)
    assert a['mean_query_cosine']==pytest.approx(b['mean_query_cosine'])
    assert a['coverage']>b['coverage'] and total_reward(a)>total_reward(b)
    assert rp.score(['a','c'],tok)['redundancy']==pytest.approx(1)


def test_weight_configuration_and_sensitivity(rp,tok):
    for w in CANDIDATES.values():
        assert -1<total_reward(rp.score(['root'],tok),w)<1
    with pytest.raises(ValueError,match='sum'):
        Weights(1,1,1,1)
    with pytest.raises(ValueError,match='Coverage'):
        Weights(.1,.3,.3,.3)


def test_quality_pairing_context_identity_and_finite_metrics(rp,tok):
    proxy=rp.score(['root'],tok)
    quality={'selection_id':proxy['selection_id'],'context_sha256':proxy['context_sha256'],
             'quality_version':'future-judge-v1','metrics':{'quality':.5}}
    assert pair_quality([proxy],[quality])[0]['quality']['metrics']['quality']==.5
    for changed in [quality|{'context_sha256':'wrong'},quality|{'selection_id':'wrong'},
                    quality|{'metrics':{'quality':float('nan')}}]:
        with pytest.raises(ValueError):pair_quality([proxy],[changed])
    with pytest.raises(ValueError):pair_quality([proxy,proxy],[quality])
    with pytest.raises(ValueError):pair_quality([proxy],[quality,quality])


def test_query_level_macro_averaging_and_correlations():
    def row(q,reward):
        return dict(selector='random',trajectory_id=q,reward=reward,feasible=True,
            coverage=reward,diversity=0,redundancy=0,token_cost=1,context_tokens=10,mean_query_cosine=reward)
    rows=[row('q1',0),row('q1',0),row('q2',1)]
    assert summarize(rows)[0]['reward']==.5
    assert summarize(rows)[0]['n_queries']==2
    assert correlation([row('a',v) for v in [0,.5,1]])['spearman']==pytest.approx(1)
    assert ranks([1,1,2]).tolist()==[.5,.5,2]


def test_study_artifacts_and_no_overwrite(tmp_path):
    from .reward_diagnostics import study,audit
    source=tmp_path/'source';source.mkdir()
    for query_id in ['q1','q2']:
        evidence={i:dict(item_id=i,content=text,source_url=url,source_subquery='s',tree_depth=1,
                        retrieval_round=1,source_type='page')
                  for i,text,url in [('a','facts a','https://a.test'),('b','facts b','https://b.test')]}
        (source/f'trajectory_{query_id}.json').write_text(json.dumps({
            'query':'question','query_id':query_id,'report':'completed','evidence':evidence}))
        np.savez(source/f'trajectory_{query_id}_emb.npz',ids=np.array(['a','b']),
                 vectors=np.array([[1.,0],[0,1.]]),query=np.array([[1.,0]]),subquestions=np.array([[0.,1.]]))
    output=tmp_path/'diagnostics'
    result=study(source,output,budget=100,seeds=(0,1))
    assert all(row['n_queries']==2 for row in result)
    for name in ['records.json','summary.json','weight_sensitivity.json','diagnostics.json',
                 'manifest.json','comparison.csv','component_distributions.png']:
        assert (output/name).is_file()
    assert json.loads((output/'manifest.json').read_bytes())['api_calls']==0
    assert audit(output)['n_queries']==2
    with pytest.raises(FileExistsError):audit(output)
    with pytest.raises(FileExistsError):study(source,output,budget=100,seeds=(0,1))
    rescored=tmp_path/'rescored'
    assert study(source,rescored,budget=100,seeds=(0,1),reuse_masks=output/'records.json')==result
    before=json.loads((output/'records.json').read_bytes())
    after=json.loads((rescored/'records.json').read_bytes())
    assert [r['context_sha256'] for r in before]==[r['context_sha256'] for r in after]
    with pytest.raises(ValueError,match='configuration'):
        study(source,tmp_path/'bad-seeds',budget=100,seeds=(99,),reuse_masks=output/'records.json')


def test_saved_replay_score_binds_context_and_sources_without_reading_report(tok,tmp_path,monkeypatch):
    from pathlib import Path
    from .test_replay import REAL
    from .core import load_pool,make_plan,save_plan
    from .reward import score_saved_replay
    pool=load_pool(REAL)
    plan=make_plan(pool,tok,selected_ids=[pool.pages[0].id])
    output=tmp_path/'replay';save_plan(plan,output)
    original=Path.read_bytes
    def guarded(path,*args,**kwargs):
        if path.name in {'report.md','synthesis.json'}:
            raise AssertionError('Proxy must not read downstream outcomes')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    score=score_saved_replay(output,tok)
    assert score['context_sha256']==plan['context_sha256']
    assert score['context_tokens']==plan['context_tokens']
    with (output/'context.txt').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='context mismatch'):
        score_saved_replay(output,tok)


def test_cluster_support_prevents_singleton_saturation_and_has_max_ablation(tok):
    pool=Pool('q','t',tuple(Page(i,i,'https://'+i+'.test','s',1,1,0) for i in ['a','b','c']),{})
    vectors=[[1,0],[.8,.6],[.8,-.6]];targets=[[1,0],[1,0]]
    rp=RewardPool.from_arrays(pool,vectors,targets)
    single=rp.score(['a'],tok);all_pages=rp.score(['a','b','c'],tok)
    assert single['max_coverage_ablation']==1
    assert single['coverage']==pytest.approx(1/2.6)
    assert all_pages['coverage']==1
    assert total_reward(all_pages)>total_reward(single)
    ablation=RewardPool.from_arrays(pool,vectors,targets,support_groups=1)
    assert ablation.score(['a'],tok)['coverage']==single['max_coverage_ablation']
    with pytest.raises(ValueError,match='Support groups'):
        RewardPool.from_arrays(pool,vectors,targets,support_groups=0)
