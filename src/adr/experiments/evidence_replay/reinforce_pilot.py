"""Run a fixed three-seed CPU pilot, then evaluate all frozen checkpoints offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .core import Tokenizer,sha,block
from .reward import VERSION,RewardPool,total_reward
from .reinforce import Config,Policy,SOURCES,FEATURES,WEIGHTS,ExactAllocator,rollout,train


METRICS=('reward','coverage','diversity','redundancy','context_tokens','selected_count')


def evaluate(policy,pools,allocators,*,training_seed,method,epoch,eval_seeds=(1000,1001,1002,1003,1004),deterministic=False):
    policy.eval();rows=[]
    for pool,allocator in zip(pools,allocators):
        for seed in ([0] if deterministic else eval_seeds):
            e=rollout(policy,pool,allocator,generator=torch.Generator().manual_seed(seed),deterministic=deterministic,track_grad=False)
            rows.append(e['score']|{'reward':e['reward'],'forced_actions':e['forced'],'free_actions':e['decisions'],
                        'training_seed':training_seed,'evaluation_seed':seed,'method':method,'epoch':epoch,
                        'constraint_violations':int(not e['score']['feasible'])})
    return rows


def aggregate(rows,split):
    """Average evaluation masks and training seeds within query, then queries."""
    result=[]
    for role in ['train','validation']:
        for method in sorted({r['method'] for r in rows}):
            group=[r for r in rows if r['method']==method and split[r['trajectory_id']]==role]
            if not group:continue
            per_query=[]
            for q in sorted({r['trajectory_id'] for r in group}):
                qr=[r for r in group if r['trajectory_id']==q]
                per_query.append({'trajectory_id':q,**{m:float(np.mean([r[m] for r in qr])) for m in METRICS},
                    'empty_fraction':float(np.mean([r['selected_count']==0 for r in qr])),
                    'constraint_violations':sum(r['constraint_violations'] for r in qr)})
            result.append({'split':role,'method':method,'n_queries':len(per_query),'n_masks':len(group),
                **{m:float(np.mean([r[m] for r in per_query])) for m in METRICS},
                'empty_fraction':float(np.mean([r['empty_fraction'] for r in per_query])),
                'constraint_violations':sum(r['constraint_violations'] for r in per_query),
                'per_query':per_query})
    return result


def load_sources(source_dir,tokenizer,budget):
    pools=[];allocators=[];split={};families=[]
    for family,qid in SOURCES.items():
        path=source_dir/f'trajectory_{qid}.json';rp=RewardPool.load(path)
        if rp.pool.trajectory_id!=qid:raise ValueError('Unexpected source identity')
        role='train' if int(family)<=6 else 'validation'
        split[qid]=role;families.append({'family':'real-source-v2-'+family,'trajectory_id':qid,'split':role,
                'query_hash':hashlib.sha256(rp.pool.query.strip().casefold().encode()).hexdigest(),
                'provenance':rp.pool.provenance})
        allocator=ExactAllocator(rp,tokenizer,budget)
        if allocator.tokens.min()>budget:raise ValueError('No feasible page for source')
        # Independent exact-token verification before optimization begins.
        for ids in [list(range(min(3,len(rp.pool.pages)))),list(range(len(rp.pool.pages)))]:
            context='\n\n'.join(block(rp.pool.pages[i])
                               for i in sorted(ids,key=allocator.order.get))
            if tokenizer.count(context)!=allocator.count(ids):raise RuntimeError('Allocator preflight mismatch')
        pools.append(rp);allocators.append(allocator)
    if len({f['query_hash'] for f in families})!=len(families):raise ValueError('Duplicate query family across split')
    return pools,allocators,split,families


def baseline_records(path,pools,tokenizer,budget):
    cached=json.loads(path.read_bytes());rows=[]
    for pool in pools:
        existing=[r for r in cached if r['trajectory_id']==pool.pool.trajectory_id and r['selector'] in {'random','threshold','topk'}]
        if len(existing)!=7:raise ValueError('Require five random seeds plus threshold/top-k baselines')
        for prior in existing:
            if prior['budget']!=budget or prior['trajectory_sha256']!=pool.pool.provenance['trajectory_sha256'] or prior['proxy_version']!=VERSION:
                raise ValueError('Baseline source/budget/reward mismatch')
            score=pool.score(prior['selected_ids'],tokenizer,budget)
            if score['context_sha256']!=prior['context_sha256']:raise ValueError('Baseline context mismatch')
            reward=total_reward(score,WEIGHTS)
            if abs(reward-prior['rewards']['anti_redundancy'])>1e-10:raise ValueError('Proxy changed since baseline diagnostics')
            rows.append(score|{'reward':reward,'method':prior['selector'],'training_seed':None,'evaluation_seed':prior['seed'],
                               'constraint_violations':int(not score['feasible'])})
    return rows


def pilot(output,source_dir,baseline_path,config=Config(),seeds=(0,1,2)):
    if not seeds or len(seeds)!=len(set(seeds)) or len(seeds)>3:raise ValueError('Use 1..3 distinct pilot seeds')
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();torch.set_num_threads(1);tokenizer=Tokenizer()
    pools,allocators,split,families=load_sources(source_dir,tokenizer,config.budget)
    baseline=baseline_records(baseline_path,pools,tokenizer,config.budget)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'config':asdict(config),'training_seeds':list(seeds),'evaluation_seeds':[1000,1001,1002,1003,1004],
        'features':FEATURES,'weights':asdict(WEIGHTS),'proxy_version':VERSION,'families':families,
        'candidate_order':'descending root cosine, ID tie break; all pages considered',
        'primary_evaluation':'five stochastic evaluation seeds; no checkpoint selection on validation',
        'validation_caveat':'These development queries were used in proxy design; not an independent final test.',
        'baseline_path':str(baseline_path.resolve()),'baseline_sha256':sha(baseline_path),
        'torch_version':torch.__version__,'device':'cpu','threads':1,'api_calls':0,
        'code_hashes':{name:sha(Path(__file__).with_name(name)) for name in ['reinforce.py','reinforce_pilot.py','reward.py','core.py']}}
    (output/'config.json').write_text(json.dumps(manifest,indent=2))
    train_p=[p for p in pools if split[p.pool.trajectory_id]=='train']
    train_a=[a for p,a in zip(pools,allocators) if split[p.pool.trajectory_id]=='train']
    final=list(baseline);curves=[];statuses=[]
    for seed in seeds:
        root=output/f'seed-{seed}';root.mkdir()
        torch.manual_seed(seed);initial=Policy(config.hidden)
        with (root/'untrained.pt').open('xb') as f:torch.save(initial.state_dict(),f)
        initial_rows=evaluate(initial,pools,allocators,training_seed=seed,method='untrained_stochastic',epoch=0)
        final.extend(initial_rows)
        final.extend(evaluate(initial,pools,allocators,training_seed=seed,method='untrained_deterministic',epoch=0,deterministic=True))
        curves.extend([r|{'epoch':0,'training_seed':seed} for r in aggregate(initial_rows,split)])
        policy,history,status=train(train_p,train_a,config,seed,root/'training')
        statuses.append({'seed':seed,'status':status,'epochs':len(history)})
        print(json.dumps(statuses[-1]),flush=True)
        # Evaluation happens after training and cannot influence model updates or
        # choose a checkpoint. Final epoch is always the prespecified endpoint.
        checkpoints=sorted((root/'training').glob('epoch-*.pt'))
        for checkpoint in checkpoints:
            epoch=int(checkpoint.stem.split('-')[-1])
            if epoch%5!=0 and epoch!=len(history):continue
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=Policy(config.hidden);model.load_state_dict(state['model'])
            rows=evaluate(model,pools,allocators,training_seed=seed,method='reinforce_stochastic',epoch=epoch)
            curves.extend([r|{'epoch':epoch,'training_seed':seed} for r in aggregate(rows,split)])
            if epoch==len(history):
                final.extend(rows)
                final.extend(evaluate(model,pools,allocators,training_seed=seed,method='reinforce_deterministic',epoch=epoch,deterministic=True))
        if status!='completed':break  # do not launch other seeds after degeneracy
    summary=aggregate(final,split)
    def save(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    save('results.json',final);save('summary.json',summary);save('evaluation_curves.json',curves)
    save('status.json',{'seeds':statuses,'wall_seconds':time.perf_counter()-started,'api_calls':0})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,role in zip(axes,['train','validation']):
        for seed in seeds:
            group=sorted([r for r in curves if r['split']==role and r['training_seed']==seed],key=lambda r:r['epoch'])
            if group:ax.plot([r['epoch'] for r in group],[r['reward'] for r in group],marker='o',label=f'seed {seed}')
        for method in ['threshold','topk','random']:
            row=next(r for r in summary if r['split']==role and r['method']==method)
            ax.axhline(row['reward'],linestyle='--',alpha=.5,label=method)
        ax.set(title=role,xlabel='epoch',ylabel='proxy reward');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'reward_curves.png',dpi=140);plt.close(fig)
    return [{k:v for k,v in r.items() if k!='per_query'} for r in summary]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=Path('runs/_gpt_researcher_real_source_v2_raw'))
    parser.add_argument('--baseline-records',type=Path,default=Path('runs/evidence-replay-phase3/proxy-diagnostics-002/records.json'))
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--entropy',type=float,default=.005)
    parser.add_argument('--learning-rate',type=float,default=.003)
    args=parser.parse_args()
    # Deliberately block outbound connections in the entire pilot process.
    def denied(*a,**kw):raise RuntimeError('Offline REINFORCE pilot forbids network')
    socket.socket.connect=denied;socket.create_connection=denied
    print(json.dumps(pilot(args.output,args.source_dir,args.baseline_records,
                          Config(epochs=args.epochs,entropy=args.entropy,learning_rate=args.learning_rate)),indent=2))


if __name__=='__main__':main()
