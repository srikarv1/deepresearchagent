"""Bounded CPU REINFORCE pilot on frozen page pools; no model-service clients."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

from .core import Tokenizer, block, build_context, sha
from .reward import VERSION, CANDIDATES, RewardPool, total_reward

FEATURES = ('query_cosine','subquery_mean','subquery_max','page_tokens_fraction',
            'depth','round','remaining_budget','selected_count','max_selected_cosine',
            'same_host_selected','root_gap','subquery_gap','same_cluster_selected','position')
WEIGHTS = CANDIDATES['anti_redundancy']
SOURCES = {'001':'b582ce3bb9a6','002':'cfabaf63b0e2','003':'787e4b40d68d',
           '004':'4260c6f63e48','005':'1e230eef802a','006':'254955f96594',
           '007':'8f57539a4e28','008':'2e4e0919d57f'}


@dataclass(frozen=True)
class Config:
    epochs: int = 20
    budget: int = 50000
    learning_rate: float = .003
    weight_decay: float = .0001
    entropy: float = .005
    baseline_decay: float = .8
    grad_clip: float = 1.
    hidden: int = 32

    def __post_init__(self):
        if not 1<=self.epochs<=30 or self.budget<=0 or self.hidden!=32:
            raise ValueError('Pilot requires 1..30 epochs, positive budget, 32 hidden units')
        if not all(math.isfinite(x) for x in [self.learning_rate,self.weight_decay,self.entropy,self.baseline_decay,self.grad_clip]):
            raise ValueError('Nonfinite configuration')
        if self.learning_rate<=0 or self.weight_decay<0 or self.entropy<0 or not 0<=self.baseline_decay<1 or self.grad_clip<=0:
            raise ValueError('Invalid optimizer configuration')


class Policy(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.Tanh(),nn.Linear(hidden,1))
        # Near-constant initial p(KEEP)=.7 makes deterministic initialization
        # equivalent to top-k rather than accidentally all-PRUNE. No labels used.
        nn.init.normal_(self.net[-1].weight,mean=0,std=.01)
        nn.init.constant_(self.net[-1].bias,math.log(.7/.3))

    def forward(self,x):
        return self.net(x).squeeze(-1)


class ExactAllocator:
    """Cached BPE boundary counts for the existing replay page wrappers.

    For o200k_base and alphanumeric IDs, each inter-page boundary is a closed
    marker followed by two LF characters and a non-whitespace '[' header. The
    tokenizer regex never joins across that boundary into the next header. Count
    each page with its trailing separator, removing the final separator's delta.
    Final output is independently checked using core.build_context, fail-closed.
    """
    def __init__(self,rp,tokenizer,budget):
        if tokenizer.name!='o200k_base' or any(not p.id.isascii() or not p.id.isalnum() for p in rp.pool.pages):
            raise ValueError('Cached allocator requires o200k_base and ASCII alphanumeric page IDs')
        self.rp,self.tokenizer,self.budget=rp,tokenizer,budget
        self.order={i:rank for rank,i in enumerate(sorted(range(len(rp.pool.pages)),key=lambda i:(rp.pool.pages[i].round,rp.pool.pages[i].id)))}
        self.tokens=np.array([tokenizer.count(block(p)) for p in rp.pool.pages],dtype=int)
        self.separated=np.array([tokenizer.count(block(p)+'\n\n') for p in rp.pool.pages],dtype=int)

    def count(self,indices):
        if not indices:return 0
        last=max(indices,key=self.order.get)
        return int(self.separated[indices].sum()-self.separated[last]+self.tokens[last])

    def validate(self,indices):
        context,ids,tokens=build_context(self.rp.pool,[self.rp.pool.pages[i].id for i in indices],self.tokenizer,self.budget)
        if tokens!=self.count(indices):
            raise RuntimeError('Cached token count differs from independent exact allocator')
        return ids,tokens


def feature_vector(rp,allocator,i,selected,position,total):
    page=rp.pool.pages[i];n=max(1,len(selected))
    subcos=rp.vectors[i]@rp.targets[1:].T
    best=rp.affinities[selected].max(axis=0) if selected else np.zeros(rp.affinities.shape[1])
    maxsim=float(rp.pair_cosines[i,selected].max()) if selected else 0.
    return torch.tensor([
        float(rp.vectors[i]@rp.targets[0]),float(subcos.mean()),float(subcos.max()),
        min(2.,allocator.tokens[i]/allocator.budget),
        page.depth/max(1,max(p.depth for p in rp.pool.pages)),
        page.round/max(1,max(p.round for p in rp.pool.pages)),
        (allocator.budget-allocator.count(selected))/allocator.budget,min(1.,len(selected)/10),
        maxsim,sum(rp.domains[j]==rp.domains[i] for j in selected)/n,
        max(0.,rp.affinities[i,0]-best[0]),float(np.maximum(0,rp.affinities[i,1:]-best[1:]).mean()),
        float(any(rp.clusters[i]==rp.clusters[j] for j in selected)),position/max(1,total-1),
    ],dtype=torch.float32)


def rollout(policy,rp,allocator,*,generator=None,deterministic=False,track_grad=True):
    selected=[];logps=[];entropies=[];forced=0
    order=sorted(range(len(rp.pool.pages)),key=lambda i:(-rp.pool.pages[i].similarity,rp.pool.pages[i].id))
    # Sampling order is fixed descending cosine for every policy and checkpoint.
    with torch.set_grad_enabled(track_grad):
        for position,i in enumerate(order):
            if allocator.count(selected+[i])>allocator.budget:
                forced+=1
                continue  # no log probability or entropy for forced actions
            logits=policy(feature_vector(rp,allocator,i,selected,position,len(order)))
            distribution=Bernoulli(logits=logits)
            action=(logits>=0).to(torch.float32) if deterministic else torch.bernoulli(distribution.probs,generator=generator)
            if track_grad:
                logps.append(distribution.log_prob(action));entropies.append(distribution.entropy())
            if bool(action.item()):selected.append(i)
    ids,tokens=allocator.validate(selected)
    score=rp.score(ids,allocator.tokenizer,allocator.budget)
    if score['context_tokens']!=tokens or not score['feasible']:
        raise RuntimeError('Invalid rollout context budget')
    zero=next(policy.parameters()).sum()*0 if track_grad else torch.tensor(0.)
    return {'score':score,'reward':total_reward(score,WEIGHTS),'forced':forced,
            'decisions':len(order)-forced,'log_prob':torch.stack(logps).sum() if logps else zero,
            'entropy':torch.stack(entropies).mean() if entropies else zero}


def policy_loss(episodes,baseline,entropy):
    advantage=torch.tensor([e['reward'] for e in episodes],dtype=torch.float32)-float(baseline)
    return -(advantage.detach()*torch.stack([e['log_prob'] for e in episodes])).mean()-entropy*torch.stack([e['entropy'] for e in episodes]).mean()


def data_signature(pools,config):
    return hashlib.sha256(json.dumps({'sources':[(p.pool.trajectory_id,p.pool.provenance) for p in pools],
        'config':asdict(config),'features':FEATURES,'reward_version':VERSION,'weights':asdict(WEIGHTS),
        'support_parameters':[(p.support_groups,p.duplicate_threshold) for p in pools],
        'code_hashes':{name:sha(Path(__file__).with_name(name)) for name in ['reinforce.py','reward.py','core.py']}},sort_keys=True).encode()).hexdigest()


def save_checkpoint(path,policy,optimizer,epoch,baseline,generator,rng,history,signature,seed,status):
    with path.open('xb') as f:
        torch.save({'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'baseline':baseline,
                    'generator':generator.get_state(),'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                    'history':history,'signature':signature,'seed':seed,'status':status},f)


def train(pools,allocators,config,seed,output,*,resume=None,stop_after=None):
    """Only training pools enter this function. Checkpoint at epoch boundaries."""
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    policy=Policy(config.hidden);optimizer=torch.optim.AdamW(policy.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    generator=torch.Generator().manual_seed(seed);rng=random.Random(seed)
    baseline=0.;history=[];start=0;signature=data_signature(pools,config);status='completed'
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=False)
        if checkpoint['signature']!=signature or checkpoint['seed']!=seed or checkpoint['status']!='completed':
            raise ValueError('Resume contract/seed/status mismatch')
        policy.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        generator.set_state(checkpoint['generator']);rng.setstate(checkpoint['rng']);torch.set_rng_state(checkpoint['torch_rng'])
        baseline=checkpoint['baseline'];history=checkpoint['history'];start=checkpoint['epoch']
    end=min(config.epochs,stop_after if stop_after is not None else config.epochs)
    for epoch in range(start,end):
        order=list(range(len(pools)));rng.shuffle(order)
        policy.train();episodes=[rollout(policy,pools[i],allocators[i],generator=generator) for i in order]
        loss=policy_loss(episodes,baseline,config.entropy)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; training stopped')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),config.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        reward=float(np.mean([e['reward'] for e in episodes]))
        record={'epoch':epoch+1,'reward':reward,'baseline_before':baseline,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
                'entropy':float(np.mean([float(e['entropy'].detach()) for e in episodes])),
                'empty_fraction':float(np.mean([e['score']['selected_count']==0 for e in episodes])),
                'context_tokens':float(np.mean([e['score']['context_tokens'] for e in episodes])),
                'coverage':float(np.mean([e['score']['coverage'] for e in episodes])),
                'redundancy':float(np.mean([e['score']['redundancy'] for e in episodes]))}
        baseline=config.baseline_decay*baseline+(1-config.baseline_decay)*reward
        history.append(record)
        if len(history)>=3 and np.mean([h['empty_fraction'] for h in history[-3:]])>.5:status='stopped_degenerate_empty'
        save_checkpoint(output/f'epoch-{epoch+1:03d}.pt',policy,optimizer,epoch+1,baseline,generator,rng,history,signature,seed,status)
        (output/'history.json').write_text(json.dumps(history,indent=2))
        if status!='completed':break
    return policy,history,status
