"""Small, deterministic CPU-only proxy study; query is the analysis unit."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import Tokenizer, select, sha
from .reward import VERSION, CANDIDATES, RewardPool, Weights, total_reward


def ranks(x):
    x = np.asarray(x)
    return np.array([((x < v).sum() + .5 * ((x == v).sum() - 1)) for v in x])


def correlation(rows):
    if len(rows) < 3:
        return {'n_masks': len(rows), 'pearson': None, 'spearman': None}
    x = np.array([r['mean_query_cosine'] for r in rows]);y = np.array([r['reward'] for r in rows])
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else None
    return {'n_masks': len(rows), 'pearson': corr(x,y), 'spearman': corr(ranks(x),ranks(y))}


def summarize(records):
    methods = sorted({r['selector'] for r in records})
    fields = ['coverage','diversity','redundancy','token_cost','context_tokens','reward','mean_query_cosine']
    summary = []
    for method in methods:
        group = [r for r in records if r['selector']==method]
        per_query = []
        for q in sorted({r['trajectory_id'] for r in group}):
            rows = [r for r in group if r['trajectory_id']==q]
            per_query.append({f:float(np.mean([r[f] for r in rows if r[f] is not None]))
                              if any(r[f] is not None for r in rows) else None for f in fields})
        summary.append({'selector':method, 'n_queries':len(per_query), 'n_masks':len(group),
            **{f:float(np.mean([r[f] for r in per_query if r[f] is not None]))
               if any(r[f] is not None for r in per_query) else None for f in fields},
            'feasible_fraction':float(np.mean([r['feasible'] for r in group]))})
    return summary


def study(source_dir: Path, output: Path, budget=50000, seeds=(0,1,2,3,4), threshold=.35, weights=None, reuse_masks=None):
    weights = weights or CANDIDATES
    if budget <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require a positive budget and distinct random seeds')
    if output.exists():
        raise FileExistsError(output)
    paths = sorted(source_dir.glob('trajectory_*.json'))
    if not paths:
        raise ValueError('No source trajectories')
    tokenizer = Tokenizer();records=[];source=[]
    frozen=json.loads(reuse_masks.read_bytes()) if reuse_masks else None
    for path in paths:
        rp = RewardPool.load(path)
        masks=[]
        if frozen is not None:
            old=[r for r in frozen if r['trajectory_id']==rp.pool.trajectory_id]
            expected={('random',s) for s in seeds}|{('topk',0),('threshold',0),('keep_none',None),('keep_all',None),('topk_half_mask',None)}
            if {(r['selector'],r['seed']) for r in old} != expected or len(old)!=len(expected):
                raise ValueError('Saved masks do not match selector/seed configuration')
            for r in old:
                if r['trajectory_sha256']!=rp.pool.provenance['trajectory_sha256'] or r['budget']!=budget or r['encoding']!=tokenizer.name:
                    raise ValueError('Saved mask source/budget/tokenizer mismatch')
                masks.append((r['selector'],r['seed'],r['selected_ids'],r))
        else:
            for method,seed in [('random',s) for s in seeds]+[('threshold',0),('topk',0)]:
                masks.append((method,seed,select(rp.pool,method,tokenizer,budget,seed,threshold)[1],None))
            half_ids=select(rp.pool,'topk',tokenizer,budget//2)[1]
            masks.extend((name,None,ids,None) for name,ids in [('keep_none',[]),
                         ('keep_all',[p.id for p in rp.pool.pages]),('topk_half_mask',half_ids)])
        for name,seed,ids,old in masks:
            row=rp.score(ids,tokenizer,budget)
            if old and (row['context_sha256']!=old['context_sha256'] or row['context_tokens']!=old['context_tokens']):
                raise ValueError('Reused mask context mismatch')
            records.append(row | {'selector':name,'seed':seed,'reward':total_reward(row),
                'rewards':{name:total_reward(row,w) for name,w in weights.items()}})
        source.append(rp.pool.provenance | {'trajectory_id':rp.pool.trajectory_id, 'page_count':len(rp.pool.pages)})
        print(json.dumps({'query':rp.pool.trajectory_id,'status':'scored'}),flush=True)
    summary=summarize(records)
    valid=[r for r in records if r['selector'] in {'random','threshold','topk'} and r['selected_count'] and r['feasible']]
    correlations={q:correlation([r for r in valid if r['trajectory_id']==q]) for q in sorted({r['trajectory_id'] for r in valid})}
    matched=[];similar=[]
    for q in correlations:
        rows=[r for r in valid if r['trajectory_id']==q]
        top=next((r for r in rows if r['selector']=='topk'),None)
        for row in rows:
            if top and row['selector']!='topk' and abs(row['context_tokens']-top['context_tokens'])<=.02*budget:
                matched.append({'trajectory_id':q,'selector':row['selector'],'seed':row['seed'],
                    'token_gap':row['context_tokens']-top['context_tokens'],
                    'reward_delta_vs_topk':row['reward']-top['reward']})
        for a,b in itertools.combinations(rows,2):
            if a['selection_id'] != b['selection_id'] and abs(a['mean_query_cosine']-b['mean_query_cosine'])<=.01 and abs(a['context_tokens']-b['context_tokens'])<=.02*budget:
                similar.append({'trajectory_id':q,'a':a['selection_id'],'b':b['selection_id'],
                    'cosine_gap':abs(a['mean_query_cosine']-b['mean_query_cosine']),
                    'token_gap':abs(a['context_tokens']-b['context_tokens']),
                    'reward_gap':abs(a['reward']-b['reward']),
                    'coverage_gap':abs(a['coverage']-b['coverage']),
                    'diversity_gap':abs(a['diversity']-b['diversity']),
                    'redundancy_gap':abs(a['redundancy']-b['redundancy'])})
    sensitivity={name:summarize([r|{'reward':r['rewards'][name]} for r in records]) for name in weights}
    # Rank agreement is reported per query, never as page-level evidence.
    sensitivity_ranks={}
    for name in weights:
        sensitivity_ranks[name]={q:[r['selector'] for r in sorted(
            [dict(selector=m,reward=float(np.mean([r['rewards'][name] for r in valid if r['trajectory_id']==q and r['selector']==m])))
             for m in ['random','threshold','topk'] if any(r['trajectory_id']==q and r['selector']==m for r in valid)],
            key=lambda r:-r['reward'])] for q in correlations}
    output.mkdir(parents=True,exist_ok=False)
    def write(name,obj):
        (output/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    write('records.json',records);write('summary.json',summary);write('weight_sensitivity.json',sensitivity)
    write('diagnostics.json',{'query_correlations':correlations,'similar_cosine_pairs':similar,
          'matched_token_comparisons':matched,'sensitivity_rankings':sensitivity_ranks,
          'analysis_unit':'query; random masks averaged within query before macro averages',
          'matching_tolerance_tokens':.02*budget,'similar_cosine_tolerance':.01,
          'downstream_quality_validation':'not performed; no report-quality claim'})
    write('manifest.json',{'version':'proxy-study-v1','budget':budget,'seeds':list(seeds),'threshold':threshold,
          'proxy_version':VERSION,'support_groups':3,
          'reused_masks':{'path':str(reuse_masks),'sha256':sha(reuse_masks)} if reuse_masks else None,
          'weights':{k:asdict(v) for k,v in weights.items()},'sources':source,
          'code_hashes':{n:sha(Path(__file__).with_name(n)) for n in ['reward.py','reward_diagnostics.py','core.py']},
          'api_calls':0,'training_updates':0})
    with (output/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['random','threshold','topk']
    fig,axes=plt.subplots(1,5,figsize=(16,4))
    for ax,field in zip(axes,['coverage','diversity','redundancy','token_cost','reward']):
        data=[[float(np.mean([r[field] for r in records if r['selector']==m and r['trajectory_id']==q]))
               for q in correlations] for m in methods]
        ax.boxplot(data,tick_labels=methods);ax.set_title(field);ax.tick_params(axis='x',rotation=30)
    fig.suptitle('Proxy: query-level values per selector; random seeds averaged')
    fig.tight_layout();fig.savefig(output/'component_distributions.png',dpi=130);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path('runs/_gpt_researcher_real_source_v2_raw'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=int,default=50000)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--threshold',type=float,default=.35)
    p.add_argument('--weights',type=Path,help='JSON mapping names to four weight fields; defaults to three documented candidates')
    p.add_argument('--audit-only',action='store_true',help='Add a fresh audit.json to an existing diagnostics directory without rerunning selectors')
    p.add_argument('--reuse-masks',type=Path,help='Rescore validated masks from a prior records.json; no selection rerun')
    a=p.parse_args();weights={k:Weights(**v) for k,v in json.loads(a.weights.read_text()).items()} if a.weights else None
    if a.audit_only:
        print(json.dumps(audit(a.output),indent=2))
        return
    print(json.dumps(study(a.source_dir,a.output,a.budget,a.seeds,a.threshold,weights,a.reuse_masks),indent=2))


def audit(output: Path):
    """Cheap saturation/variance audit on persisted query-level masks."""
    if (output/'audit.json').exists():
        raise FileExistsError(output/'audit.json')
    records=json.loads((output/'records.json').read_bytes())
    manifest=json.loads((output/'manifest.json').read_bytes())
    if manifest.get('proxy_version')!=VERSION:
        raise ValueError('Audit requires the current proxy version; rescore old masks first')
    weights={k:Weights(**v) for k,v in manifest['weights'].items()}
    tokenizer=Tokenizer();singletons=[];spread=[]
    for source in manifest['sources']:
        rp=RewardPool.load(Path(source['trajectory_path']))
        if any(rp.pool.provenance[k] != source[k] for k in ['trajectory_sha256','embeddings_sha256']):
            raise ValueError('Source changed since diagnostic run')
        best=sorted(rp.pool.pages,key=lambda p:(-p.similarity,p.id))[0]
        row=rp.score([best.id],tokenizer,manifest['budget'])
        top=next(r for r in records if r['trajectory_id']==rp.pool.trajectory_id and r['selector']=='topk')
        singletons.append(row|{'reward':total_reward(row),'rewards':{name:total_reward(row,w) for name,w in weights.items()},
                              'reward_delta_vs_topk':total_reward(row)-top['reward']})
        primary=[r for r in records if r['trajectory_id']==rp.pool.trajectory_id and r['selector'] in {'random','threshold','topk'} and r['feasible'] and r['selected_count']]
        for name,w in weights.items():
            sd={k:float(np.std([r[k] for r in primary]))*value for k,value in asdict(w).items()}
            total=sum(sd.values())
            spread.append({'trajectory_id':rp.pool.trajectory_id,'weights':name,'weighted_component_std':sd,
                           'std_share':{k:v/total if total else 0 for k,v in sd.items()}})
    result={'singletons':singletons,'within_query_weighted_spread':spread,
            'warning':'Weighted standard deviations are descriptive spread, not causal feature importance.',
            'singleton_beats_topk_queries':sum(r['reward_delta_vs_topk']>0 for r in singletons),
            'n_queries':len(singletons),
            'empty_sentinel_queries':sum(r['reward']==-1 for r in records if r['selector']=='keep_none'),
            'infeasible_keep_all_queries':sum(not r['feasible'] and r['reward']==-1 for r in records if r['selector']=='keep_all')}
    with (output/'audit.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    return {k:v for k,v in result.items() if k not in {'singletons','within_query_weighted_spread'}}


if __name__=='__main__':
    main()
