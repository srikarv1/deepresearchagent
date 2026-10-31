"""Controlled proxy-vs-judge test on DRG: same pool, different selections.

For a few training queries, several selectors pick pages from the *same*
saved pool, so selection is the only thing that differs. Each selection is
written into a report by GPT Researcher's own writer (``synthesis.py``) and
scored by the official Gym quality and KPR judges twice, which also measures
judge noise. Stages, each refusing to redo finished work:

    prepare     offline: selections, plans, proxy scores
    synthesize  one writer request per selection          (--allow-api)
    judge       quality + KPR, once per repeat            (--allow-api)
    analyze     offline: judge noise, proxy vs judge

``bottom`` (least query-similar pages) is a sanity anchor: the judge should
dislike it. Proxy-vs-judge statistics are reported with and without it.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .core import Tokenizer, make_plan, save_plan, select
from .drg import find_pool, greedy_shortlist
from .drg_proxy_check import spearman
from .learning_signal import make_shortlist
from .reinforce import WEIGHTS, ExactAllocator
from .reward import VERSION, RewardPool, total_reward

QUERIES = ('624340', '437127', '188169', '964162', '238338')
METHODS = ('topk', 'threshold', 'random_a', 'random_b', 'shortlist_all', 'shortlist_greedy', 'bottom')
ANCHOR = 'bottom'
MODEL, MAX_COMPLETION, EFFORT = 'gpt-5-mini', 16384, 'low'
JUDGE_MODEL = 'gpt-5-mini'
COMPONENTS = ('coverage', 'diversity', 'redundancy', 'token_cost')


def bottom(rp, allocator):
    """First fit over ascending query similarity: a deliberately weak selection."""
    kept = []
    for i in sorted(range(len(rp.pool.pages)), key=lambda i: (rp.pool.pages[i].similarity, rp.pool.pages[i].id)):
        if allocator.count(kept + [i]) <= allocator.budget:
            kept.append(i)
    return [rp.pool.pages[i].id for i in kept]


def selections(rp, allocator, tokenizer, budget):
    shortlist = make_shortlist(rp, allocator)
    return {'topk': select(rp.pool, 'topk', tokenizer, budget)[1],
            'threshold': select(rp.pool, 'threshold', tokenizer, budget, threshold=.35)[1],
            'random_a': select(rp.pool, 'random', tokenizer, budget, seed=1000)[1],
            'random_b': select(rp.pool, 'random', tokenizer, budget, seed=1001)[1],
            'shortlist_all': [rp.pool.pages[i].id for i in shortlist],
            'shortlist_greedy': greedy_shortlist(rp, allocator, shortlist),
            'bottom': bottom(rp, allocator)}


def entries(output):
    return json.loads((output / 'manifest.json').read_text(encoding='utf-8'))['entries']


def prepare(output, corpus, queries=QUERIES, budget=50000):
    if output.exists():
        raise FileExistsError(output)
    tokenizer = Tokenizer()
    rows, staged = [], []
    for qid in queries:
        path = find_pool(qid, [corpus])
        rp = RewardPool.load(path)
        allocator = ExactAllocator(rp, tokenizer, budget)
        for method, ids in selections(rp, allocator, tokenizer, budget).items():
            if not ids:
                raise ValueError(f'Empty {method} selection for {qid}')
            plan = make_plan(rp.pool, tokenizer, budget=budget, selected_ids=ids)
            plan['config'].update(method=method, model=MODEL, max_completion_tokens=MAX_COMPLETION,
                                  reasoning_effort=EFFORT)
            score = rp.score(ids, tokenizer, budget)
            staged.append((qid, method, plan))
            rows.append({'query_id': qid, 'method': method, 'id': f'{qid}__{method}', 'pool': str(path),
                         'pool_pages': len(rp.pool.pages), 'selected': len(ids),
                         'context_tokens': plan['context_tokens'], 'proxy': total_reward(score, WEIGHTS),
                         **{c: score[c] for c in COMPONENTS}})
    output.mkdir(parents=True)
    for qid, method, plan in staged:
        save_plan(plan, output / qid / method)
    with (output / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump({'reward_version': VERSION, 'budget': budget, 'model': MODEL, 'judge_model': JUDGE_MODEL,
                   'max_completion_tokens': MAX_COMPLETION, 'reasoning_effort': EFFORT, 'entries': rows},
                  f, indent=2)
    return rows


def archive_unanswered(folder):
    """Move aside a failed attempt that never got a response (nothing billed); keep everything else."""
    state = json.loads((folder / 'synthesis.json').read_text(encoding='utf-8'))
    if state.get('failure_reason') != 'transport_error' or state.get('usage') is not None:
        return False
    attempts = folder / 'attempts'
    attempts.mkdir(exist_ok=True)
    n = len(list(attempts.glob('synthesis-*.json'))) + 1
    for name in ('synthesis.json', 'request.json'):
        if (folder / name).exists():
            (folder / name).rename(attempts / f'{Path(name).stem}-{n}.json')
    return True


def synthesize_all(output, retry_failed=False):
    from .synthesis import synthesize
    done, failed = 0, []
    for e in entries(output):
        folder = output / e['query_id'] / e['method']
        if (folder / 'report.md').exists():
            done += 1; continue
        if (folder / 'synthesis.json').exists() and not (retry_failed and archive_unanswered(folder)):
            failed.append(e['id']); continue  # an earlier attempt failed; only unanswered ones are retried
        plan = json.loads((folder / 'plan.json').read_text(encoding='utf-8'))
        plan['context'] = (folder / 'context.txt').read_bytes().decode('utf-8')
        try:
            state = asyncio.run(synthesize(plan, folder, allow_api=True, model=MODEL,
                                           max_completion_tokens=MAX_COMPLETION, reasoning_effort=EFFORT))
            done += 1
            print(json.dumps({'id': e['id'], 'status': state['status'],
                              'latency_s': round(state['latency_seconds'], 1),
                              'usage': (state.get('usage') or {}).get('total_tokens')}), flush=True)
        except Exception as exc:
            failed.append(e['id'])
            print(json.dumps({'id': e['id'], 'status': 'failed', 'error': type(exc).__name__}), flush=True)
    return {'completed': done, 'failed': failed}


def judge(output, repeat, with_kpr=True, kpr_skip=()):
    from adr.eval.deep_research_gym import _kpr, _quality
    from adr.eval.repos import find_deep_research_gym, find_key_points
    work = output / f'judge-{repeat}'
    if work.exists():
        raise FileExistsError(work)
    gym = find_deep_research_gym(None)
    if not gym.ok:
        raise RuntimeError(gym.reason)
    key_points = find_key_points(gym.path, None)
    reports, kp = work / 'reports', work / 'key_point'
    reports.mkdir(parents=True); kp.mkdir()
    for e in entries(output):
        report = output / e['query_id'] / e['method'] / 'report.md'
        if not report.exists():
            continue
        query = json.loads((output / e['query_id'] / e['method'] / 'plan.json').read_text(encoding='utf-8'))['query']
        (reports / f"{e['id']}.q").write_text(query.strip() + '\n', encoding='utf-8')
        (reports / f"{e['id']}.a").write_text(report.read_text(encoding='utf-8').strip() + '\n', encoding='utf-8')
        if with_kpr and e['query_id'] not in kpr_skip:
            # Upstream skips a report whose key-point file is missing, before any request.
            shutil.copyfile(key_points / f"{e['query_id']}_aggregated.json", kp / f"{e['id']}_aggregated.json")
    quality = _quality(gym.path, reports, work, JUDGE_MODEL, None)
    kpr = _kpr(gym.path, reports, work, JUDGE_MODEL, kp, None) if with_kpr else {}
    scores = {rid: {'quality': (quality.get('per_query_normalized') or {}).get(rid),
                    'kpr': ((kpr.get('per_query') or {}).get(rid) or {}).get('support_rate')}
              for rid in [p.stem for p in reports.glob('*.q')]}
    with (work / 'scores.json').open('x', encoding='utf-8') as f:
        json.dump({'quality': {k: v for k, v in quality.items() if k != 'per_query_normalized'},
                   'kpr': {k: v for k, v in kpr.items() if k != 'per_query'}, 'scores': scores}, f, indent=2)
    return {'judged': sum(s['quality'] is not None for s in scores.values()), 'reports': len(scores)}


def kpr_fix(output, repeat):
    """Re-judge KPR only for reports of a repeat whose KPR request failed; quality is untouched."""
    from adr.eval.deep_research_gym import _kpr
    from adr.eval.repos import find_deep_research_gym
    base = output / f'judge-{repeat}'
    scores = json.loads((base / 'scores.json').read_text(encoding='utf-8'))['scores']
    # A report had KPR requested iff its key-point file was staged in the original repeat.
    missing = [rid for rid, s in scores.items() if s['kpr'] is None and (base / 'key_point' / f'{rid}_aggregated.json').exists()]
    if not missing:
        return {'missing': []}
    work = output / f'judge-{repeat}-kprfix-{len(list(output.glob(f"judge-{repeat}-kprfix-*"))) + 1}'
    reports, kp = work / 'reports', work / 'key_point'
    reports.mkdir(parents=True); kp.mkdir()
    for rid in missing:
        for suffix in ('.q', '.a'):
            shutil.copyfile(base / 'reports' / f'{rid}{suffix}', reports / f'{rid}{suffix}')
        shutil.copyfile(base / 'key_point' / f'{rid}_aggregated.json', kp / f'{rid}_aggregated.json')
    kpr = _kpr(find_deep_research_gym(None).path, reports, work, JUDGE_MODEL, kp, None)
    fixed = {rid: {'quality': None, 'kpr': ((kpr.get('per_query') or {}).get(rid) or {}).get('support_rate')}
             for rid in missing}
    with (work / 'scores.json').open('x', encoding='utf-8') as f:
        json.dump({'kpr': {k: v for k, v in kpr.items() if k != 'per_query'}, 'scores': fixed}, f, indent=2)
    return {'missing': missing, 'fixed': [rid for rid, s in fixed.items() if s['kpr'] is not None]}


def load_repeats(output):
    """One score table per repeat; ``judge-<n>-kprfix-*`` fill KPR values the repeat is missing."""
    runs = []
    for base in sorted(output.glob('judge-*/scores.json'), key=lambda p: p.parent.name):
        if '-kprfix-' in base.parent.name:
            continue
        run = json.loads(base.read_text(encoding='utf-8'))['scores']
        for fix in sorted(output.glob(f'{base.parent.name}-kprfix-*/scores.json')):
            for rid, s in json.loads(fix.read_text(encoding='utf-8'))['scores'].items():
                if run.get(rid, {}).get('kpr') is None and s['kpr'] is not None:
                    run[rid]['kpr'] = s['kpr']
        runs.append(run)
    return runs


def analyze(output):
    rows = entries(output)
    runs = load_repeats(output)
    if not runs:
        raise ValueError('Analysis needs at least one judge repeat')
    for r in rows:
        for metric in ('quality', 'kpr'):
            vals = [run.get(r['id'], {}).get(metric) for run in runs]
            r.update({f'{metric}_{i + 1}': v for i, v in enumerate(vals)})
            present = [v for v in vals if v is not None]  # KPR may be judged in fewer repeats
            r[metric] = float(np.mean(present)) if present else None
    scored = [r for r in rows if r['quality'] is not None]

    def within(rows_, x, y):
        v = [spearman([r[x] for r in g], [r[y] for r in g])
             for g in ([r for r in rows_ if r['query_id'] == q] for q in sorted({r['query_id'] for r in rows_}))]
        v = [c for c in v if c is not None]
        return {'mean': float(np.mean(v)) if v else None, 'positive': sum(c > 0 for c in v), 'queries': len(v)}

    def noise(m):
        pairs = [r for r in scored if r.get(f'{m}_1') is not None and r.get(f'{m}_2') is not None]
        if not pairs:
            return None
        return {'reports': len(pairs), 'within_query_spearman': within(pairs, f'{m}_1', f'{m}_2'),
                'mean_abs_difference': float(np.mean([abs(r[f'{m}_1'] - r[f'{m}_2']) for r in pairs])),
                'pooled_spearman': spearman([r[f'{m}_1'] for r in pairs], [r[f'{m}_2'] for r in pairs])}

    def has(rows_, m):
        return [r for r in rows_ if r[m] is not None]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    no_anchor = [r for r in scored if r['method'] != ANCHOR]
    result = {'reports': len(rows), 'scored': len(scored), 'judge_repeats': len(runs),
              'kpr_scored': len(has(scored, 'kpr')), 'judge_noise': {m: noise(m) for m in ('quality', 'kpr')},
              'proxy_vs_judge': {
                  f'{p}_vs_{m}': {'within_query': within(has(no_anchor, m), p, m),
                                  'within_query_with_anchor': within(has(scored, m), p, m),
                                  'pooled': spearman([r[p] for r in has(no_anchor, m)], [r[m] for r in has(no_anchor, m)])}
                  for p in ('proxy', *COMPONENTS, 'selected') for m in ('quality', 'kpr')},
              'by_method': {meth: {k: mean([r[k] for r in scored if r['method'] == meth])
                                   for k in ('proxy', 'quality', 'kpr', 'selected', 'context_tokens')}
                            for meth in METHODS if any(r['method'] == meth for r in scored)},
              'api_calls': 0}
    with (output / 'analysis.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (output / 'reports.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['prepare', 'synthesize', 'judge', 'kpr-fix', 'analyze'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path, default=Path('runs/rollouts_gym'))
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--allow-api', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='synthesize: retry attempts that got no response')
    p.add_argument('--no-kpr', action='store_true', help='judge: quality only')
    p.add_argument('--kpr-skip', nargs='*', default=[], help='judge: query ids to leave out of KPR')
    a = p.parse_args()
    if a.stage in ('synthesize', 'judge', 'kpr-fix'):
        if not a.allow_api:
            p.error(f'{a.stage} calls the API; pass --allow-api')
        from dotenv import load_dotenv
        load_dotenv('.env')
    result = {'prepare': lambda: prepare(a.output, a.corpus), 'synthesize': lambda: synthesize_all(a.output, a.retry_failed),
              'judge': lambda: judge(a.output, a.repeat, not a.no_kpr, set(a.kpr_skip)),
              'kpr-fix': lambda: kpr_fix(a.output, a.repeat), 'analyze': lambda: analyze(a.output)}[a.stage]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
