"""Isolated ordering diagnostic; frozen reward, features, optimizer and episode cap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import socket
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Bernoulli

from .core import Tokenizer,sha
from .reward import VERSION,total_reward
from .reinforce import Config,Policy,WEIGHTS,FEATURES,feature_vector,policy_loss
from .reinforce_pilot import load_sources,aggregate,baseline_records

PROTOCOLS=('cosine','random_order','shortlist')
EVAL_SEEDS=(1000,1001,1002,1003,1004)


def order_seed(seed,epoch,qid):
    return int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{qid}:order-v1'.encode()).digest()[:8],'big')


def make_shortlist(rp,allocator,k=24):
    """Frozen coverage anchors then short high-affinity distinct-cluster pages.

    All shortlisted pages jointly fit B, so every subset is feasible. Anchors
    preserve best-available root/subquestion representatives where they fit.
    Selection uses only embeddings, clusters and lengths, never reward outcomes.
    """
    selected=[];clusters=set()
    def add(i):
        if i in selected or rp.clusters[i] in clusters:return False
        if len(selected)>=k or allocator.count(selected+[i])>allocator.budget:return False
        selected.append(i);clusters.add(rp.clusters[i]);return True
    for target in range(rp.affinities.shape[1]):
        ranked=sorted(range(len(rp.pool.pages)),key=lambda i:(-rp.affinities[i,target],allocator.tokens[i],rp.pool.pages[i].id))
        # If the target's best group is already present it supplies the anchor.
        if rp.clusters[ranked[0]] in clusters:continue
        for i in ranked:
            if add(i):break
    fill=sorted(range(len(rp.pool.pages)),key=lambda i:(-float(rp.affinities[i].mean()),allocator.tokens[i],rp.pool.pages[i].id))
    for i in fill:
        if allocator.tokens[i]<=allocator.budget//k:add(i)
    # Do not silently widen the token filter if fewer than k candidates fit.
    allocator.validate(selected)
    return selected


def candidate_order(rp,protocol,seed,shortlist=None):
    if protocol=='cosine':
        return sorted(range(len(rp.pool.pages)),key=lambda i:(-rp.pool.pages[i].similarity,rp.pool.pages[i].id))
    if protocol not in PROTOCOLS:raise ValueError('Unknown candidate protocol')
    order=sorted(shortlist if protocol=='shortlist' else range(len(rp.pool.pages)))
    random.Random(seed).shuffle(order)
    return order


def sample(policy,rp,allocator,protocol,*,action_generator,ordering_seed,shortlist=None,deterministic=False,track_grad=True):
    selected=[];logps=[];entropies=[];action_trace=[];forced=0
    order=candidate_order(rp,protocol,ordering_seed,shortlist)
    with torch.set_grad_enabled(track_grad):
        for position,i in enumerate(order):
            if allocator.count(selected+[i])>allocator.budget:
                forced+=1;continue
            logits=policy(feature_vector(rp,allocator,i,selected,position,len(order)))
            dist=Bernoulli(logits=logits)
            action=(logits>=0).float() if deterministic else torch.bernoulli(dist.probs,generator=action_generator)
            probability=float(dist.probs.detach());a=float(action.detach())
            lp=dist.log_prob(action);entropy=dist.entropy()
            if track_grad:logps.append(lp);entropies.append(entropy)
            action_trace.append({'page_id':rp.pool.pages[i].id,'position':position,'keep':bool(a),
                'keep_probability':probability,'entropy':float(entropy.detach()),'log_probability':float(lp.detach()),
                'logit_score_gradient':a-probability})
            if a:selected.append(i)
    ids,tokens=allocator.validate(selected);score=rp.score(ids,allocator.tokenizer,allocator.budget)
    if score['context_tokens']!=tokens or not score['feasible']:raise RuntimeError('Budget integrity failure')
    zero=next(policy.parameters()).sum()*0 if track_grad else torch.tensor(0.)
    diagnostics={'sampled_actions':len(action_trace),'forced_actions':forced,
        'filtered_pages':len(rp.pool.pages)-len(order),'candidate_count':len(order),
        'mean_keep_probability':float(np.mean([a['keep_probability'] for a in action_trace])) if action_trace else 0.,
        'mean_entropy':float(np.mean([a['entropy'] for a in action_trace])) if action_trace else 0.,
        'last_sampled_position_fraction':action_trace[-1]['position']/max(1,len(order)-1) if action_trace else 0.,
        'informative_actions':sum(.1<=a['keep_probability']<=.9 for a in action_trace)}
    return {'score':score,'reward':total_reward(score,WEIGHTS),'trace':action_trace,'diagnostics':diagnostics,
        'log_prob':torch.stack(logps).sum() if logps else zero,
        'entropy':torch.stack(entropies).mean() if entropies else zero}


def flat_grad(value,model):
    grads=torch.autograd.grad(value,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for p,g in zip(model.parameters(),grads)])


def gradient_diagnostics(episodes,model,baseline,entropy_coef):
    advantages=np.array([e['reward']-baseline for e in episodes])
    contributions=torch.stack([flat_grad(-float(a)*e['log_prob'],model)/len(episodes) for a,e in zip(advantages,episodes)])
    pg=contributions.sum(0)
    ent=flat_grad(-entropy_coef*torch.stack([e['entropy'] for e in episodes]).mean(),model)
    full=flat_grad(policy_loss(episodes,baseline,entropy_coef),model)
    residual=float(torch.linalg.vector_norm(full-pg-ent))
    if residual>1e-5:raise RuntimeError('Gradient decomposition mismatch')
    norms=torch.linalg.vector_norm(contributions,dim=1)
    return {'advantage_mean':float(advantages.mean()),'advantage_variance':float(advantages.var()),
        'reward_variance':float(np.var([e['reward'] for e in episodes])),
        'policy_gradient_norm':float(pg.norm()),'entropy_gradient_norm':float(ent.norm()),
        'gradient_decomposition_error':residual,
        'gradient_alignment_ratio':float(pg.norm()/(norms.sum()+1e-12)),
        'episode_gradient_norm_mean':float(norms.mean()),'episode_gradient_norm_max':float(norms.max())}


def eval_model(model,pools,allocators,shortlists,protocol,training_seed,epoch,trained,deterministic=False):
    model.eval();rows=[]
    for rp,allocator,shortlist in zip(pools,allocators,shortlists):
        # Same random ordering and action streams for trained and untrained.
        for seed in EVAL_SEEDS:
            e=sample(model,rp,allocator,protocol,action_generator=torch.Generator().manual_seed(seed),
                ordering_seed=order_seed(seed,0,rp.pool.trajectory_id),shortlist=shortlist,
                deterministic=deterministic,track_grad=False)
            rows.append(e['score']|e['diagnostics']|{'reward':e['reward'],'training_seed':training_seed,
                'evaluation_seed':seed,'epoch':epoch,'protocol':protocol,'trained':trained,
                'mode':'deterministic' if deterministic else 'stochastic',
                'method':('trained' if trained else 'untrained')+('_deterministic' if deterministic else '_stochastic'),
                'constraint_violations':int(not e['score']['feasible'])})
    return rows


def train_one(pools,allocators,shortlists,split,protocol,seed,config,output):
    output.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed)
    model=Policy();initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    generator=torch.Generator().manual_seed(seed);rng=random.Random(seed);baseline=0.;history=[];episode_log=[]
    training=[i for i,rp in enumerate(pools) if split[rp.pool.trajectory_id]=='train']
    with (output/'initial.pt').open('xb') as f:torch.save(initial,f)
    status='completed'
    for epoch in range(config.epochs):
        query_order=list(training);rng.shuffle(query_order);model.train();episodes=[]
        for i in query_order:
            rp=pools[i]
            episodes.append(sample(model,rp,allocators[i],protocol,action_generator=generator,
                ordering_seed=order_seed(seed,epoch,rp.pool.trajectory_id),shortlist=shortlists[i]))
        gd=gradient_diagnostics(episodes,model,baseline,config.entropy)
        loss=policy_loss(episodes,baseline,config.entropy)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
        optimizer.step()
        for e in episodes:
            advantage=e['reward']-baseline
            episode_log.append(e['score']|e['diagnostics']|{'epoch':epoch+1,'reward':e['reward'],
                'baseline':baseline,'advantage':advantage,
                'mean_abs_logit_gradient_contribution':float(np.mean([abs(advantage*a['logit_score_gradient']) for a in e['trace']])) if e['trace'] else 0.,
                'actions':e['trace']})
        mean_reward=float(np.mean([e['reward'] for e in episodes]))
        history.append({'epoch':epoch+1,'reward':mean_reward,'baseline_before':baseline,'loss':float(loss.detach()),
            'gradient_norm':float(norm),'clipped':bool(norm>config.grad_clip),**gd,
            **{k:float(np.mean([e['diagnostics'][k] for e in episodes])) for k in episodes[0]['diagnostics']},
            'empty_fraction':float(np.mean([not e['score']['selected_count'] for e in episodes]))})
        baseline=config.baseline_decay*baseline+(1-config.baseline_decay)*mean_reward
        if len(history)>=3 and np.mean([h['empty_fraction'] for h in history[-3:]])>.5:status='stopped_empty_collapse'
        if len(history)>=3 and np.mean([h['reward'] for h in history[-3:]])<0:status='stopped_reward_collapse'
        if (epoch+1)%5==0 or epoch==config.epochs-1 or status!='completed':
            with (output/f'epoch-{epoch+1:03d}.pt').open('xb') as f:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'baseline':baseline,
                    'generator':generator.get_state(),'shuffle_rng':rng.getstate(),'epoch':epoch+1,
                    'protocol':protocol,'seed':seed,'status':status},f)
        if status!='completed':break
    with (output/'history.json').open('x') as f:json.dump(history,f,indent=2)
    with (output/'episodes.json').open('x') as f:json.dump(episode_log,f,indent=2)
    return initial,model,history,status


def run(output,source_dir,baseline_path,config=Config()):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.perf_counter()
    tokenizer=Tokenizer();pools,allocators,split,families=load_sources(source_dir,tokenizer,config.budget)
    shortlists=[make_shortlist(p,a) for p,a in zip(pools,allocators)]
    baseline=baseline_records(baseline_path,pools,tokenizer,config.budget)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'config':asdict(config),'protocols':PROTOCOLS,'training_seeds':[0,1,2],'evaluation_seeds':EVAL_SEEDS,
        'weights':asdict(WEIGHTS),'reward_version':VERSION,'features':FEATURES,'families':families,
        'shortlist':{'max_pages':24,'fill_page_token_ceiling':config.budget//24,'anchor_targets':'root plus all saved subquestions',
                     'unique_duplicate_clusters':True,'joint_shortlist_fits_budget':True,'scan_order':'seeded random'},
        'matching':'same 120 training episodes / 20 updates per seed, initialization, features and optimizer; token utilization may differ',
        'code_hashes':{n:sha(Path(__file__).with_name(n)) for n in ['learning_signal.py','reinforce.py','reinforce_pilot.py','reward.py','core.py']},
        'validation_caveat':'Two queries, already inspected in proxy design; no hyperparameter/checkpoint selection.',
        'api_calls':0,'device':'cpu','threads':1}
    with (output/'config.json').open('x') as f:json.dump(manifest,f,indent=2)
    ceiling=[]
    for rp,a,ix in zip(pools,allocators,shortlists):
        score=rp.score([rp.pool.pages[i].id for i in ix],tokenizer,config.budget)
        ceiling.append(score|{'family':next(f['family'] for f in families if f['trajectory_id']==rp.pool.trajectory_id),
            'note':'all-shortlist coverage ceiling, not an optimized reward oracle'})
    with (output/'shortlist_coverage.json').open('x') as f:json.dump(ceiling,f,indent=2)
    final=[];curves=[];statuses=[]
    for protocol in PROTOCOLS:
        for seed in [0,1,2]:
            folder=output/protocol/f'seed-{seed}'
            initial,model,history,status=train_one(pools,allocators,shortlists,split,protocol,seed,config,folder)
            statuses.append({'protocol':protocol,'seed':seed,'epochs':len(history),'status':status})
            print(json.dumps(statuses[-1]),flush=True)
            cold=Policy();cold.load_state_dict(initial)
            initial_rows=eval_model(cold,pools,allocators,shortlists,protocol,seed,0,False)
            final.extend(initial_rows)
            final.extend(eval_model(cold,pools,allocators,shortlists,protocol,seed,0,False,True))
            curves.extend([r|{'epoch':0,'protocol':protocol,'training_seed':seed} for r in aggregate(initial_rows,split)])
            for checkpoint in sorted(folder.glob('epoch-*.pt')):
                state=torch.load(checkpoint,map_location='cpu',weights_only=False);epoch=state['epoch']
                model.load_state_dict(state['model'])
                rows=eval_model(model,pools,allocators,shortlists,protocol,seed,epoch,True)
                curves.extend([r|{'epoch':epoch,'protocol':protocol,'training_seed':seed} for r in aggregate(rows,split)])
                if epoch==len(history):
                    final.extend(rows);final.extend(eval_model(model,pools,allocators,shortlists,protocol,seed,epoch,True,True))
            if status!='completed':break
        if statuses[-1]['status']!='completed':break
    summary=[]
    for protocol in {r['protocol'] for r in final}:
        summary.extend([r|{'protocol':protocol} for r in aggregate([r for r in final if r['protocol']==protocol],split)])
    def save(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    save('results.json',final);save('summary.json',summary);save('curves.json',curves);save('heuristic_baselines.json',aggregate(baseline,split))
    save('status.json',{'runs':statuses,'wall_seconds_before_plotting':time.perf_counter()-started,'api_calls':0})
    flat=[{k:v for k,v in r.items() if k!='per_query'} for r in summary]
    with (output/'comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    queries=[dict(protocol=r['protocol'],split=r['split'],method=r['method'],**q) for r in summary for q in r['per_query']]
    with (output/'per_query.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(queries[0]));w.writeheader();w.writerows(queries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,7))
    for col,protocol in enumerate(PROTOCOLS):
        for ax,role in zip(axes[:,col],['train','validation']):
            for seed in [0,1,2]:
                group=sorted([r for r in curves if r['protocol']==protocol and r['split']==role and r['training_seed']==seed],key=lambda r:r['epoch'])
                if group:ax.plot([r['epoch'] for r in group],[r['reward'] for r in group],marker='o',label=f'seed {seed}')
            ax.set(title=protocol+' / '+role,xlabel='epoch',ylabel='proxy reward');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'learning_curves.png',dpi=140);plt.close(fig)
    return flat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-dir',type=Path,default=Path('runs/_gpt_researcher_real_source_v2_raw'))
    p.add_argument('--baseline-records',type=Path,default=Path('runs/evidence-replay-phase3/proxy-diagnostics-002/records.json'))
    a=p.parse_args()
    def denied(*args,**kwargs):raise RuntimeError('Offline diagnostic prohibits network')
    socket.socket.connect=denied;socket.create_connection=denied
    print(json.dumps(run(a.output,a.source_dir,a.baseline_records),indent=2))


if __name__=='__main__':main()
