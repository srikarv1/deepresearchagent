"""Read-only analysis of a completed ordering diagnostic; writes fresh audit files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .reinforce_pilot import aggregate
from .reinforce import WEIGHTS
from dataclasses import asdict


def analyze(output: Path,phase4=Path('runs/evidence-replay-phase4/cpu-pilot-001')):
    for name in ['audit.json','paired_deltas.csv','gradient_summary.csv']:
        if (output/name).exists():raise FileExistsError(output/name)
    config=json.loads((output/'config.json').read_bytes());rows=json.loads((output/'results.json').read_bytes())
    summary=json.loads((output/'summary.json').read_bytes());split={f['trajectory_id']:f['split'] for f in config['families']}
    paired=[];gradient=[];query_deltas=[];ordering=[];seed_deltas=[];deterministic_changes={}
    for protocol in config['protocols']:
        subset=[r for r in rows if r['protocol']==protocol and r['mode']=='stochastic']
        for role in ['train','validation']:
            cold=next(r for r in summary if r['protocol']==protocol and r['split']==role and r['method']=='untrained_stochastic')
            final=next(r for r in summary if r['protocol']==protocol and r['split']==role and r['method']=='trained_stochastic')
            paired.append(dict(protocol=protocol,split=role,untrained=cold['reward'],trained=final['reward'],delta=final['reward']-cold['reward'],
                untrained_tokens=cold['context_tokens'],trained_tokens=final['context_tokens']))
            for q in final['per_query']:
                initial=next(c for c in cold['per_query'] if c['trajectory_id']==q['trajectory_id'])
                query_deltas.append(dict(protocol=protocol,split=role,trajectory_id=q['trajectory_id'],
                    untrained=initial['reward'],trained=q['reward'],delta=q['reward']-initial['reward']))
        for seed in config['training_seeds']:
            sr=aggregate([r for r in subset if r['training_seed']==seed],split)
            for role in ['train','validation']:
                a=next(r for r in sr if r['split']==role and r['method']=='untrained_stochastic')
                b=next(r for r in sr if r['split']==role and r['method']=='trained_stochastic')
                seed_deltas.append(dict(protocol=protocol,seed=seed,split=role,untrained=a['reward'],trained=b['reward'],delta=b['reward']-a['reward']))
        history=[];episodes=[]
        for folder in (output/protocol).glob('seed-*'):
            history.extend(json.loads((folder/'history.json').read_bytes()))
            episodes.extend(json.loads((folder/'episodes.json').read_bytes()))
        actions=[a for e in episodes for a in e['actions']]
        probability=np.array([a['keep_probability'] for a in actions])
        rewards=np.array([e['reward'] for e in episodes])
        within=np.mean([(e['reward']-np.mean([v['reward'] for v in episodes if v['trajectory_id']==e['trajectory_id']]))**2 for e in episodes])
        record={'protocol':protocol,'episodes':len(episodes),'updates':len(history),
            **{k:float(np.mean([h[k] for h in history])) for k in ['sampled_actions','forced_actions','filtered_pages',
                'informative_actions','last_sampled_position_fraction','advantage_mean','advantage_variance','policy_gradient_norm',
                'entropy_gradient_norm','gradient_alignment_ratio','episode_gradient_norm_mean']},
            'action_probability_mean':float(probability.mean()),'action_probability_std':float(probability.std()),
            'action_probability_min':float(probability.min()),'action_probability_max':float(probability.max()),
            'action_entropy_mean':float(np.mean([a['entropy'] for a in actions])),
            'gradient_norm_mean':float(np.mean([h['gradient_norm'] for h in history])),
            'gradient_norm_max':max(h['gradient_norm'] for h in history),
            'gradient_accounting_error_max':max(h['gradient_decomposition_error'] for h in history),
            'clipped_updates':sum(h['clipped'] for h in history),'empty_episodes':sum(not e['selected_count'] for e in episodes),
            'first_epoch_advantage_mean':float(np.mean([h['advantage_mean'] for h in history if h['epoch']==1])),
            'last_epoch_advantage_mean':float(np.mean([h['advantage_mean'] for h in history if h['epoch']==20])),
            'reward_variance_pooled':float(rewards.var()),'reward_variance_after_query_centering':float(within),
            'between_query_variance_fraction':float(1-within/rewards.var()) if rewards.var()>0 else 0.}
        gradient.append(record)
        deterministic=[r for r in rows if r['protocol']==protocol and r['mode']=='deterministic']
        lookup={(r['training_seed'],r['trajectory_id'],r['evaluation_seed']):r for r in deterministic if not r['trained']}
        deterministic_changes[protocol]=sum(r['selection_id']!=lookup[(r['training_seed'],r['trajectory_id'],r['evaluation_seed'])]['selection_id'] for r in deterministic if r['trained'])
    # Cold-policy contrasts isolate protocol effects from learned parameter changes.
    for role in ['train','validation']:
        for treatment,control in [('cosine','random_order'),('shortlist','random_order'),('shortlist','cosine')]:
            a=next(r for r in summary if r['protocol']==treatment and r['split']==role and r['method']=='untrained_stochastic')
            b=next(r for r in summary if r['protocol']==control and r['split']==role and r['method']=='untrained_stochastic')
            contributions={k:(a[k]-b[k])*w*(1 if k in {'coverage','diversity'} else -1) for k,w in asdict(WEIGHTS).items() if k!='token_cost'}
            contributions['token_cost']=-(a['context_tokens']-b['context_tokens'])/config['config']['budget']*WEIGHTS.token_cost
            ordering.append(dict(split=role,treatment=treatment,control=control,untrained_reward_delta=a['reward']-b['reward'],component_contributions=contributions))
    reproduced={}
    for seed in config['training_seeds']:
        old=phase4/f'seed-{seed}/training/epoch-020.pt'
        new=output/f'cosine/seed-{seed}/epoch-020.pt'
        if old.exists() and new.exists():
            a=torch.load(old,map_location='cpu',weights_only=False)['model'];b=torch.load(new,map_location='cpu',weights_only=False)['model']
            reproduced[str(seed)]=all(torch.equal(a[k],b[k]) for k in a)
    report={'paired_macro_deltas':paired,'per_seed_deltas':seed_deltas,'per_query_deltas':query_deltas,
        'gradient_diagnostics':gradient,'cold_policy_protocol_effects':ordering,'deterministic_masks_changed':deterministic_changes,
        'phase4_cosine_parameters_exactly_reproduced':reproduced,
        'budget_violations':sum(r['constraint_violations'] for r in rows),
        'empty_evaluation_masks':sum(not r['selected_count'] for r in rows),
        'notes':['Query, not page or mask, is the analysis unit.','Two validation queries; descriptive changes are not significance claims.',
                 'Query-centered variance is a posthoc diagnostic, not an alternative training baseline.']}
    with (output/'audit.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    for name,data in [('paired_deltas.csv',paired),('gradient_summary.csv',gradient)]:
        with (output/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    r=analyze(args.output)
    print(json.dumps({k:v for k,v in r.items() if k in ['paired_macro_deltas','gradient_diagnostics',
        'deterministic_masks_changed','phase4_cosine_parameters_exactly_reproduced','budget_violations','empty_evaluation_masks']},indent=2))
