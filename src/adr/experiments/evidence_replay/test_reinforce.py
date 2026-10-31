"""Offline feasibility, gradients, deterministic checkpoints and split contracts."""
import json
import random
import socket
from dataclasses import replace

import numpy as np
import pytest
import torch

from .core import Page,Pool,Tokenizer,build_context
from .reward import RewardPool,total_reward,CANDIDATES
from .reinforce import Config,Policy,ExactAllocator,rollout,policy_loss,train,FEATURES
from .reinforce_pilot import aggregate


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a,**kw):raise AssertionError('Offline tests prohibit networking')
    monkeypatch.setattr(socket.socket,'connect',denied);monkeypatch.setattr(socket,'create_connection',denied)


@pytest.fixture
def fixture():
    pages=tuple(Page(i,text,'https://'+i+'.test','sub',1,1,float(v[0]))
        for i,text,v in [('a','CR\rCRLF\r\nLF\n facts',[1,0]),('b','other facts 日本語',[.8,.6]),('c','complement',[.8,-.6])])
    rp=RewardPool.from_arrays(Pool('question','q',pages,{}),[[1,0],[.8,.6],[.8,-.6]],[[1,0],[0,1]])
    return rp,Tokenizer()


def test_exact_allocator_matches_full_tokenizer_all_masks(fixture):
    rp,tok=fixture;allocator=ExactAllocator(rp,tok,50000)
    import itertools
    for n in range(4):
        for ids in itertools.permutations(range(3),n):
            context,_,count=build_context(rp.pool,[rp.pool.pages[i].id for i in ids],tok,50000)
            assert allocator.count(list(ids))==count==tok.count(context)
    assert len(FEATURES)==14
    assert sum(p.numel() for p in Policy().parameters())==513


def test_allocator_random_mixed_line_ending_masks():
    rng=random.Random(42);tok=Tokenizer()
    pages=tuple(Page(f'{i:012x}',''.join(rng.choice(['abc','\r','\r\n','\n','日本語','[x]',' ']) for _ in range(80)),
                     'https://example.test','s',1,rng.randrange(1,4),1) for i in range(20))
    rp=RewardPool.from_arrays(Pool('q','q',pages,{}),np.tile([1.,0],(20,1)),[[1,0],[1,0]])
    allocator=ExactAllocator(rp,tok,50000)
    for _ in range(100):
        ix=rng.sample(range(20),rng.randrange(21))
        assert allocator.count(ix)==build_context(rp.pool,[pages[i].id for i in ix],tok,50000)[2]


def test_forced_actions_excluded_from_gradients_and_empty(fixture):
    rp,tok=fixture;policy=Policy();allocator=ExactAllocator(rp,tok,1)
    e=rollout(policy,rp,allocator,generator=torch.Generator().manual_seed(0))
    assert e['forced']==3 and e['decisions']==0 and e['reward']==-1
    assert e['score']['selected_count']==e['score']['context_tokens']==0
    loss=policy_loss([e],0,.01);loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in policy.parameters())


def test_feasibility_and_policy_gradients(fixture):
    rp,tok=fixture;policy=Policy();allocator=ExactAllocator(rp,tok,50)
    e=rollout(policy,rp,allocator,generator=torch.Generator().manual_seed(42))
    assert e['score']['feasible'] and e['score']['context_tokens']<=50
    assert e['reward']==total_reward(e['score'],CANDIDATES['anti_redundancy'])
    assert e['log_prob'].requires_grad
    loss=policy_loss([e],e['reward']-.5,.005);loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.parameters())
    # The detached baseline does not acquire a gradient.
    logits=torch.tensor(0.,requires_grad=True)
    loss=policy_loss([{'reward':1.,'log_prob':logits,'entropy':logits*0}],.5,0)
    loss.backward();assert logits.grad==-.5


def test_deterministic_evaluation_and_seeded_sampling(fixture):
    rp,tok=fixture;policy=Policy();allocator=ExactAllocator(rp,tok,100)
    a=rollout(policy,rp,allocator,deterministic=True,track_grad=False)
    b=rollout(policy,rp,allocator,deterministic=True,track_grad=False)
    assert a['score']==b['score'] and not a['log_prob'].requires_grad
    a=rollout(policy,rp,allocator,generator=torch.Generator().manual_seed(3),track_grad=False)
    b=rollout(policy,rp,allocator,generator=torch.Generator().manual_seed(3),track_grad=False)
    assert a['score']==b['score']
    with torch.no_grad():
        for p in policy.parameters():p.zero_()
        policy.net[-1].bias.fill_(-100)
    assert rollout(policy,rp,allocator,deterministic=True)['reward']==-1


def test_training_reproducibility_and_exact_resume(fixture,tmp_path):
    rp,tok=fixture;allocator=ExactAllocator(rp,tok,100);config=Config(epochs=4,budget=100)
    a,history,status=train([rp],[allocator],config,7,tmp_path/'full')
    b,history2,_=train([rp],[allocator],config,7,tmp_path/'repeat')
    assert history==history2
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    train([rp],[allocator],config,7,tmp_path/'first',stop_after=2)
    c,history3,_=train([rp],[allocator],config,7,tmp_path/'resumed',resume=tmp_path/'first/epoch-002.pt')
    assert history==history3
    assert all(torch.equal(a.state_dict()[k],c.state_dict()[k]) for k in a.state_dict())
    with pytest.raises(ValueError,match='contract'):
        train([rp],[allocator],replace(config,entropy=.02),7,tmp_path/'invalid',resume=tmp_path/'first/epoch-002.pt')
    with pytest.raises(FileExistsError):train([rp],[allocator],config,7,tmp_path/'full')


def test_aggregation_weights_queries_not_masks():
    def row(q,r):
        return dict(trajectory_id=q,method='m',reward=r,coverage=r,diversity=0,redundancy=0,context_tokens=1,selected_count=1,constraint_violations=0)
    result=aggregate([row('a',0),row('a',0),row('b',1)],{'a':'train','b':'train'})
    assert result[0]['reward']==.5 and result[0]['n_queries']==2
