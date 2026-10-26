"""Gradient correctness and isolated candidate-protocol controls."""
import copy
import random
import socket

import numpy as np
import pytest
import torch
from torch import nn

from .core import Page,Pool,Tokenizer,block
from .reward import RewardPool
from .reinforce import Policy,ExactAllocator,rollout,policy_loss
from .learning_signal import sample,candidate_order,make_shortlist,order_seed,gradient_diagnostics


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a,**kw):raise AssertionError('Offline diagnostics prohibit network')
    monkeypatch.setattr(socket.socket,'connect',denied);monkeypatch.setattr(socket,'create_connection',denied)


@pytest.fixture
def fixture():
    pages=tuple(Page(i,'facts '+i,'https://'+i+'.test','s',1,1,v[0]) for i,v in zip(['a','b','c'],[[1,0],[.8,.6],[.8,-.6]]))
    rp=RewardPool.from_arrays(Pool('q','q',pages,{}),[[1,0],[.8,.6],[.8,-.6]],[[1,0],[0,1]])
    return rp,Tokenizer()


def test_cosine_protocol_reproduces_phase4_rollout_and_gradient(fixture):
    rp,tok=fixture;allocator=ExactAllocator(rp,tok,60);torch.manual_seed(4);model=Policy()
    a=rollout(model,rp,allocator,generator=torch.Generator().manual_seed(8))
    b=sample(model,rp,allocator,'cosine',action_generator=torch.Generator().manual_seed(8),ordering_seed=999)
    assert a['score']==b['score'] and a['reward']==b['reward']
    assert a['forced']==b['diagnostics']['forced_actions'] and a['decisions']==b['diagnostics']['sampled_actions']
    assert torch.equal(a['log_prob'],b['log_prob'])
    params=tuple(model.parameters())
    ga=torch.autograd.grad(policy_loss([a],.3,.005),params,retain_graph=True)
    gb=torch.autograd.grad(policy_loss([b],.3,.005),params,retain_graph=True)
    assert all(torch.equal(x,y) for x,y in zip(ga,gb))


def test_exact_expected_reinforce_gradient_with_forced_actions(fixture):
    rp,tok=fixture
    # Limit to two candidates; either one fits, but the pair does not.
    pool=Pool('q','q',rp.pool.pages[:2],{})
    rp=RewardPool.from_arrays(pool,[[1,0],[.8,.6]],[[1,0],[0,1]])
    budget=max(tok.count(block(p)) for p in pool.pages);allocator=ExactAllocator(rp,tok,budget)
    class Constant(nn.Module):
        def __init__(self):super().__init__();self.theta=nn.Parameter(torch.tensor(.2))
        def forward(self,x):return self.theta
    model=Constant();outcomes={}
    for seed in range(100):
        e=sample(model,rp,allocator,'cosine',action_generator=torch.Generator().manual_seed(seed),ordering_seed=0)
        outcomes.setdefault(tuple(e['score']['selected_ids']),e)
        if len(outcomes)==3:break
    assert len(outcomes)==3
    probs=torch.stack([e['log_prob'].exp() for e in outcomes.values()])
    assert float(probs.sum().detach())==pytest.approx(1)
    expected=sum(e['log_prob'].exp()*e['reward'] for e in outcomes.values())
    direct=torch.autograd.grad(expected,model.theta,retain_graph=True)[0]
    estimator=sum(e['log_prob'].exp().detach()*(e['reward']-.31)*e['log_prob'] for e in outcomes.values())
    reinforce=torch.autograd.grad(estimator,model.theta,retain_graph=True)[0]
    assert torch.allclose(direct,reinforce,atol=1e-6)
    # Also compare to an independently enumerated finite difference.
    rewards={key:e['reward'] for key,e in outcomes.items()}
    def exact(theta):
        p=1/(1+np.exp(-theta))
        return p*rewards[('a',)]+(1-p)*p*rewards[('b',)]+(1-p)**2*rewards[()]
    numerical=(exact(.201)-exact(.199))/.002
    assert float(direct)==pytest.approx(numerical,abs=1e-5)
    assert outcomes[('a',)]['diagnostics']['forced_actions']==1
    assert len(outcomes[('a',)]['trace'])==1


def test_random_order_is_seeded_and_independent_of_action_sampling(fixture):
    rp,tok=fixture
    assert candidate_order(rp,'random_order',17)==candidate_order(rp,'random_order',17)
    assert set(candidate_order(rp,'random_order',18))=={0,1,2}
    assert len({tuple(candidate_order(rp,'random_order',i)) for i in range(10)})>1
    assert order_seed(1,2,'q')==order_seed(1,2,'q')!=order_seed(1,3,'q')


def test_shortlist_bound_feasibility_coverage_and_sampled_decisions():
    n=40;vectors=np.zeros((n,n+1));vectors[:,0]=.5
    for i in range(n):vectors[i,i+1]=np.sqrt(.75)
    targets=np.zeros((2,n+1));targets[:,0]=1
    pages=tuple(Page(f'{i:012x}','facts '+str(i),'https://'+str(i)+'.test','s',1,1,.5) for i in range(n))
    rp=RewardPool.from_arrays(Pool('q','q',pages,{}),vectors,targets);tok=Tokenizer();allocator=ExactAllocator(rp,tok,2000)
    shortlist=make_shortlist(rp,allocator)
    assert len(shortlist)==24 and allocator.count(shortlist)<=2000
    ceiling=rp.score([pages[i].id for i in shortlist],tok,2000)
    assert ceiling['coverage']==1
    model=Policy()
    e=sample(model,rp,allocator,'shortlist',action_generator=torch.Generator().manual_seed(0),ordering_seed=0,shortlist=shortlist)
    assert e['diagnostics']['sampled_actions']==24 and e['diagnostics']['forced_actions']==0
    assert e['diagnostics']['filtered_pages']==16 and e['score']['feasible']


def test_gradient_contributions_decompose_total_loss(fixture):
    rp,tok=fixture;model=Policy();allocator=ExactAllocator(rp,tok,60)
    episodes=[sample(model,rp,allocator,'cosine',action_generator=torch.Generator().manual_seed(s),ordering_seed=s) for s in [0,1,2]]
    d=gradient_diagnostics(episodes,model,.2,.005)
    assert d['gradient_decomposition_error']<1e-6
    assert d['policy_gradient_norm']>0 and d['entropy_gradient_norm']>0
    assert 0<=d['gradient_alignment_ratio']<=1.00001
    assert d['advantage_variance']==pytest.approx(np.var([e['reward'] for e in episodes]))
    for e in episodes:
        for action in e['trace']:
            assert action['logit_score_gradient']==pytest.approx(float(action['keep'])-action['keep_probability'])


def test_diagnostic_training_reproduces_phase4_cosine(fixture,tmp_path):
    from .reinforce import Config,train
    from .learning_signal import train_one
    rp,tok=fixture;allocator=ExactAllocator(rp,tok,60);config=Config(epochs=4,budget=60)
    model,original_history,_=train([rp],[allocator],config,5,tmp_path/'phase4')
    _,diagnostic,history,status=train_one([rp],[allocator],[make_shortlist(rp,allocator)],{'q':'train'},
                                        'cosine',5,config,tmp_path/'diagnostic')
    assert status=='completed'
    assert all(torch.equal(model.state_dict()[k],diagnostic.state_dict()[k]) for k in model.state_dict())
    assert [r['reward'] for r in original_history]==[r['reward'] for r in history]
    assert max(r['gradient_decomposition_error'] for r in history)<1e-6
