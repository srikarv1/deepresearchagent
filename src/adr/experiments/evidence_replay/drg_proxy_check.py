"""Does the proxy reward rank DRG reports the way the DRG judge does? Offline only.

For every judged rollout of a rollout corpus (``runs/rollouts_gym/<id>-s<seed>``)
the proxy scores the pages that run kept (``was_retained``), and is compared
with that run's Gym quality and KPR support rate. The main statistic is the
Spearman correlation across a query's rollouts, averaged over queries: after
per-query normalization the policy only learns from within-query differences.

Validation and test ids of the split are excluded, so the check cannot inform
anything later evaluated on them. Rollouts also differ in how they searched,
not only in what they kept, so this is an indicative check, not a clean one.
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .core import Tokenizer
from .reinforce import WEIGHTS
from .reward import VERSION, RewardPool, total_reward

JUDGE = {'quality': 'gym_quality', 'kpr': 'gym_average_support_rate'}
PREDICTORS = ('proxy', 'coverage', 'diversity', 'redundancy', 'token_cost', 'selected_count', 'pool_pages')
GATE = .3  # mean within-query Spearman of proxy vs quality, agreed before looking


def ranks(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind='stable')
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks over ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_rollout(run_dir, qid, tokenizer, budget):
    path = run_dir / 'gpt_researcher' / f'{qid}.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    kept = [i for i, e in raw['evidence'].items() if str(e.get('was_retained')) == 'True']
    s = RewardPool.load(path).score(kept, tokenizer, budget)
    judge = json.loads((run_dir / 'metrics' / 'summary.json').read_text(encoding='utf-8'))['scores']
    return {'run_id': run_dir.name, 'query_id': qid, 'pool_pages': len(raw['evidence']),
            'selected_count': s['selected_count'], 'context_tokens': s['context_tokens'],
            'over_budget': not s['feasible'],
            # Score kept sets that overran the cap on their components, not the -1 sentinel.
            'proxy': total_reward(s | {'feasible': True}, WEIGHTS),
            **{c: s[c] for c in ('coverage', 'diversity', 'redundancy', 'token_cost')},
            **{k: float(judge[v]) for k, v in JUDGE.items()}}


def check(output, corpus, split_path, budget=50000):
    if output.exists():
        raise FileExistsError(output)
    split = json.loads(Path(split_path).read_text(encoding='utf-8'))
    held_out = set(split.get('val', [])) | set(split.get('test', []))
    tokenizer = Tokenizer()
    rows = []
    for run_dir in sorted(Path(corpus).glob('*-s*')):
        qid = run_dir.name.rsplit('-s', 1)[0]
        if qid in held_out or not (run_dir / 'metrics' / 'summary.json').is_file():
            continue
        rows.append(score_rollout(run_dir, qid, tokenizer, budget))
    per_query = []
    for qid in sorted({r['query_id'] for r in rows}):
        group = [r for r in rows if r['query_id'] == qid]
        per_query.append({'query_id': qid, 'rollouts': len(group),
                          **{f'{p}_vs_{j}': spearman([r[p] for r in group], [r[j] for r in group])
                             for p in PREDICTORS for j in JUDGE}})

    def mean(key):
        v = [q[key] for q in per_query if q[key] is not None]
        return {'mean': float(np.mean(v)) if v else None, 'positive': sum(x > 0 for x in v), 'queries': len(v)}
    summary = {'reward_version': VERSION, 'weights': asdict(WEIGHTS), 'budget': budget,
               'rollouts': len(rows), 'queries': len(per_query), 'excluded_ids': sorted(held_out),
               'over_budget_rollouts': sum(r['over_budget'] for r in rows),
               'within_query': {f'{p}_vs_{j}': mean(f'{p}_vs_{j}') for p in PREDICTORS for j in JUDGE},
               'pooled': {f'{p}_vs_{j}': spearman([r[p] for r in rows], [r[j] for r in rows])
                          for p in PREDICTORS for j in JUDGE},
               'gate': GATE, 'api_calls': 0}
    summary['passes'] = (summary['within_query']['proxy_vs_quality']['mean'] or -1) >= GATE
    output.mkdir(parents=True)
    with (output / 'summary.json').open('x') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    for name, table in [('rollouts.csv', rows), ('per_query.csv', per_query)]:
        with (output / name).open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=Path('runs/rollouts_gym'))
    p.add_argument('--split', type=Path, default=Path('data/benchmarks/deep_research_gym/rl_small_split.json'))
    a = p.parse_args()

    def denied(*args, **kwargs):
        raise RuntimeError('Offline proxy check forbids network')
    socket.socket.connect = denied; socket.create_connection = denied
    print(json.dumps(check(a.output, a.corpus, a.split), indent=2))


if __name__ == '__main__':
    main()
