"""Three-query extension; all prior pilot inputs are read-only. API opt-in only."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import os
from pathlib import Path
import statistics
import time

from . import phase5 as base
from . import phase5_recovery as recovery
from .core import Tokenizer, load_pool, make_plan, save_plan, select, sha

SOURCES = {'51': '2bd39c9137ad', '53': '1c1b0cfed811', '55': '716465858aea'}
METHODS = ('threshold', 'shortlist', 'reinforce')
METRICS = ('comprehensiveness', 'insight', 'instruction_following', 'readability', 'overall_score')
LABEL = 'Five-query frozen-retrieval, synthesis-only evaluation; GPT-5-mini RACE judge variant. No FACT, significance claims, historical Table 4 equivalence, or established RL training benefit.'
COLD = ('runs/evidence-replay-phase4/cpu-pilot-001/seed-0/untrained.pt',
        '58c30a2aaf7aed6e8827c8f0cc8ababa17bfe1f7409000dcf9e195c2eead3073')


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def original_rows(root):
    config, manifest = recovery.verify(root / 'judge-recovery-001/config.json')
    if (config['judge_model'], config['completion_tokens'], config['clean_reasoning'], config['score_reasoning']) != ('gpt-5-mini', 16384, 'low', 'medium'):
        raise ValueError('Original judge configuration does not match extension')
    if manifest['synthesis'] != {'model': 'gpt-5-mini', 'reasoning_effort': 'low', 'max_completion_tokens': 16384}:
        raise ValueError('Original synthesis configuration mismatch')
    rows = base.summary_rows(manifest, root)
    for row in rows:
        matches = [(path, state) for path, state in recovery.result_records(root)
                   if path.name == row['alias'] + '-result.json' and state.get('status') == 'completed']
        if len(matches) != 1:
            raise ValueError('Require exactly one successful original score per report')
        _, result = matches[0]
        if result.get('model') != 'gpt-5-mini' or result.get('report_sha256') != config['reports'][row['alias']]:
            raise ValueError('Original score model/report identity mismatch')
        row['quality'] = result['score']
        row['judge_status'] = 'completed'
        stages = [state for path, state in recovery.call_records(root)
                  if recovery.legacy_identity(path, state, set(config['reports']))[0] == row['alias'] and state.get('status') == 'completed']
        if sorted(s['stage'] for s in stages) != ['clean', 'score'] or any(s['model'] != 'gpt-5-mini' for s in stages):
            raise ValueError('Original successful judge stages inconsistent')
        row['judge_usage'] = [s.get('usage') for s in stages]
    return manifest, rows


def policies():
    import torch
    from .reinforce import Policy
    models = {}
    for name, (relative, expected) in {**base.CHECKPOINTS, 'cosine_control': COLD}.items():
        path = base.ROOT / relative
        if sha(path) != expected:
            raise ValueError('Checkpoint identity changed')
        model = Policy()
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'] if name == 'reinforce' else state)
        model.eval()
        models[name] = model
    return models


def policy_masks(pool_path, tokenizer, models):
    import torch
    from .reward import RewardPool
    from .reinforce import ExactAllocator, rollout
    from .learning_signal import make_shortlist, sample, order_seed
    rp = RewardPool.load(pool_path)
    allocator = ExactAllocator(rp, tokenizer, 50000)
    shortlist = make_shortlist(rp, allocator)
    short = sample(models['shortlist'], rp, allocator, 'shortlist',
                   action_generator=torch.Generator().manual_seed(1000),
                   ordering_seed=order_seed(1000, 0, rp.pool.trajectory_id), shortlist=shortlist, track_grad=False)
    cap = short['score']['context_tokens']
    ids = {'shortlist': short['score']['selected_ids'],
           'threshold': select(rp.pool, 'threshold', tokenizer, cap, 1000, .35)[1]}
    for name in ('reinforce', 'cosine_control'):
        ids[name] = rollout(models[name], rp, ExactAllocator(rp, tokenizer, cap),
                            generator=torch.Generator().manual_seed(1000), track_grad=False)['score']['selected_ids']
    return rp.pool, cap, ids


def equality_record(ids):
    return {'trained_equals_untrained_shortlist': set(ids['reinforce']) == set(ids['shortlist']),
            'trained_equals_own_untrained_cosine_control': set(ids['reinforce']) == set(ids['cosine_control']),
            'own_untrained_cosine_control_ids': sorted(ids['cosine_control']),
            'interpretation': 'Shortlist and actor use different candidate protocols. Only the same-order cosine control isolates training; it receives no additional report.'}


def freeze(output, original):
    import torch
    if output.exists():
        raise FileExistsError(output)
    original = original.resolve()
    old, _ = original_rows(original)
    before = inventory(original)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = policies()
    tok = Tokenizer()
    entries, prepared, references, equality = [], [], {}, {}
    for qid, tid in SOURCES.items():
        pool, cap, ids = policy_masks(base.ROOT / f'runs/_gpt_researcher_adaptive_depth_raw/trajectory_{tid}.json', tok, models)
        references[qid] = base.official(pool.query, qid)
        equality[qid] = equality_record(ids)
        for method in METHODS:
            plan = make_plan(pool, tok, budget=cap, selected_ids=ids[method], seed=1000, threshold=.35)
            if not plan['selected_ids']:
                raise ValueError('Refuse empty frozen evidence selection')
            alias = 'report-' + hashlib.sha256(f'phase5-extension:1000:{qid}:{method}'.encode()).hexdigest()[:12]
            prepared.append((alias, qid, method, plan))
    # Verify control comparisons against the existing masks, without generating reports.
    for qid in ('52', '54'):
        old_entries = [e for e in old['entries'] if e['query_id'] == qid]
        plan = base.read(original / old_entries[0]['alias'] / 'plan.json')
        _, cap, ids = policy_masks(Path(plan['provenance']['trajectory_path']), tok, models)
        for e in old_entries:
            saved = base.read(original / e['alias'] / 'plan.json')
            if set(ids[e['method']]) != set(saved['selected_ids']) or cap != e['budget']:
                raise ValueError('Extension selection protocol fails to reproduce original masks')
        equality[qid] = equality_record(ids)
    if inventory(original) != before:
        raise ValueError('Original pilot changed during freeze')
    output.mkdir(parents=True, exist_ok=False)
    for alias, qid, method, plan in prepared:
        save_plan(plan, output / alias)
        entries.append({'alias': alias, 'query_id': qid, 'method': method,
                        'plan_sha256': sha(output / alias / 'plan.json'),
                        'context_sha256': plan['context_sha256'], 'context_tokens': plan['context_tokens'],
                        'budget': plan['config']['budget'], 'selected_count': len(plan['selected_ids']),
                        'selector_identity': {'threshold': 'cosine >= .35, chronological first-fit',
                          'shortlist': 'bounded24 at 50K, seed0 initial, seeded ordering/actions',
                          'reinforce': 'original Phase4 seed0 epoch20, cosine order, seeded actions'}[method]})
    base.write_new(output / 'references.json', references)
    base.write_new(output / 'original-inventory.json', before)
    code = dict(old['code_sha256'])
    for path in (Path(__file__), Path(recovery.__file__)):
        code[str(path.resolve())] = sha(path)
    manifest = {'version': 'phase5-extension-v1', 'label': LABEL, 'entries': entries,
                'sampling_seed': 1000, 'threshold': .35, 'shortlist_k': 24,
                'checkpoints': {**base.CHECKPOINTS, 'cosine_control': COLD},
                'references_sha256': sha(output / 'references.json'), 'code_sha256': code,
                'synthesis': old['synthesis'],
                'judge': {'model': 'gpt-5-mini', 'clean_reasoning': 'low', 'score_reasoning': 'medium',
                          'max_completion_tokens': 16384, 'fact': False},
                'limits': {'synthesis': 9, 'judge': 18},
                'original_path': str(original), 'original_inventory_sha256': sha(output / 'original-inventory.json'),
                'judge_endpoint': base.read(original / 'judge-recovery-001/execution-endpoint.json'),
                'mask_equality': equality, 'offline_negative_finding': base.NEGATIVE,
                'source_limitations': 'q51/q53 reviewed sequential pools; q55 recovery pool. Retrieval depth/rounds and localized source failures differ from historical Table 4. No outcome-based mask selection.',
                'budget_note': 'Shortlist sampled at 50K; other methods capped at its actual count. Actor trained at 50K; whole pages produce small count differences.'}
    base.write_new(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(sha(output / 'manifest.json'))
    return manifest


def validate(path):
    m = base.validate(path)
    if m['limits'] != {'synthesis': 9, 'judge': 18} or m['judge'] != {'model': 'gpt-5-mini', 'clean_reasoning': 'low', 'score_reasoning': 'medium', 'max_completion_tokens': 16384, 'fact': False}:
        raise ValueError('Extension call caps/judge protocol changed')
    root = path.parent
    if sha(root / 'original-inventory.json') != m['original_inventory_sha256'] or inventory(Path(m['original_path'])) != base.read(root / 'original-inventory.json'):
        raise ValueError('Original pilot changed')
    original_rows(Path(m['original_path']))
    if sorted((e['query_id'], e['method']) for e in m['entries']) != sorted((q, method) for q in SOURCES for method in METHODS):
        raise ValueError('Extension query/method coverage mismatch')
    return m


def api_preflight(m, transport=None):
    if transport is None:
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY required in executing process')
        if recovery.endpoint_identity() != m['judge_endpoint']:
            raise ValueError('Endpoint differs from completed pilot judge endpoint')


async def synthesize(path, *, allow_api=False, max_calls=9, transport=None):
    if not allow_api:
        raise PermissionError('Explicit --allow-api required')
    m = validate(path)
    if not 1 <= max_calls <= 9:
        raise ValueError('Maximum synthesis cap is 9')
    api_preflight(m, transport)
    from .synthesis import synthesize as one_report
    root = path.parent
    with (root / 'synthesis-stage.lock').open('x'):
        ledger = base.CallLedger(root / 'synthesis-calls', max_calls)
        for e in m['entries']:
            directory = root / e['alias']
            ledger.reserve(e['alias'])
            try:
                await one_report(base.load_plan(directory), directory, allow_api=True,
                                 transport=transport, **m['synthesis'])
            except Exception as exc:
                base.write_new(directory / 'failure.json', {'stage': 'synthesis', 'error_type': type(exc).__name__})
                raise RuntimeError('Extension synthesis stopped after failure; no retry') from None


def judge(path, *, allow_api=False, max_calls=18, transport=None):
    if not allow_api:
        raise PermissionError('Explicit --allow-api required')
    m = validate(path)
    if not 1 <= max_calls <= 18:
        raise ValueError('Maximum judge cap is 18')
    api_preflight(m, transport)
    root = path.parent
    reports = recovery.report_states(m, root)  # refuse incomplete/tampered synthesis batch
    with (root / 'judge-stage.lock').open('x'):
        directory = root / 'judging'
        directory.mkdir(exist_ok=False)
        client = recovery.RecoveryClient({'manifest_path': str(path.resolve()), 'reports': reports,
                    'judge_model': 'gpt-5-mini', 'retry_failed': []}, directory,
                    base.CallLedger(root / 'judge-calls', max_calls), transport=transport)
        race, refs = base.race_module(), base.read(root / 'references.json')
        for e in sorted(m['entries'], key=lambda e: e['alias']):
            try:
                result = base.score_one(race, client, e['alias'], refs[e['query_id']],
                                        (root / e['alias'] / 'report.md').read_bytes().decode())
                base.write_new(directory / (e['alias'] + '-result.json'), {'status': 'completed',
                    'model': 'gpt-5-mini', 'score': result, 'report_sha256': reports[e['alias']]})
            except Exception as exc:
                base.write_new(directory / (e['alias'] + '-result.json'), {'status': 'failed', 'score': None,
                    'model': 'gpt-5-mini', 'error_type': type(exc).__name__})
                raise RuntimeError('Extension judging stopped after failure; no retry') from None


def aggregate(rows):
    expected = {(q, method) for q in ('51', '52', '53', '54', '55') for method in METHODS}
    if len(rows) != 15 or {(r['query_id'], r['method']) for r in rows} != expected:
        raise ValueError('Require exactly one row per query/method')
    complete = [q for q in ('51', '52', '53', '54', '55') if all(r['quality'] is not None for r in rows if r['query_id'] == q)]
    result = {}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        paired = [r for r in group if r['query_id'] in complete]
        result[method] = {'completed_scores': sum(r['quality'] is not None for r in group),
            'five_query_macro': {metric: statistics.mean(r['quality'][metric] for r in group) if len(complete) == 5 else None for metric in METRICS},
            'complete_paired_query_macro': {metric: statistics.mean(r['quality'][metric] for r in paired) if paired else None for metric in METRICS},
            'mean_context_tokens': statistics.mean(r['context_tokens'] for r in group),
            'reported_synthesis_api_tokens': sum((r['synthesis_usage'] or {}).get('total_tokens', 0) for r in group),
            'unknown_synthesis_usage_reports': sum(r['synthesis_usage'] is None for r in group),
            'total_synthesis_latency_seconds': sum(r['synthesis_latency_seconds'] or 0 for r in group),
            'mean_synthesis_latency_seconds': statistics.mean(r['synthesis_latency_seconds'] for r in group if r['synthesis_latency_seconds'] is not None) if any(r['synthesis_latency_seconds'] is not None for r in group) else None}
    return result, complete


def combined(path):
    m = validate(path)
    _, rows = original_rows(Path(m['original_path']))
    new = base.summary_rows(m, path.parent)
    for row in new:
        state_path = path.parent / row['alias'] / 'synthesis.json'
        if state_path.exists():
            synthesis_state = base.read(state_path)
            if synthesis_state.get('status') == 'completed' and sha(path.parent / row['alias'] / 'report.md') != synthesis_state.get('report_sha256'):
                raise ValueError('Extension report hash mismatch')
        calls = [state for call_path, state in recovery.call_records(path.parent)
                 if state.get('alias') == row['alias']]
        row['judge_usage'] = [state.get('usage') for state in calls]
        row['judge_latency_seconds'] = sum(state.get('latency_seconds', 0) for state in calls)
        result_path = path.parent / 'judging' / (row['alias'] + '-result.json')
        if result_path.exists():
            result = base.read(result_path)
            if result.get('status') == 'completed':
                state = base.read(path.parent / row['alias'] / 'synthesis.json')
                if result.get('model') != 'gpt-5-mini' or result.get('report_sha256') != state.get('report_sha256') or sha(path.parent / row['alias'] / 'report.md') != state.get('report_sha256'):
                    raise ValueError('Extension score/report identity mismatch')
                row['quality'] = result['score']
        failure_path = path.parent / row['alias'] / 'failure.json'
        row['failure'] = base.read(failure_path) if failure_path.exists() else None
    rows = sorted(rows + new, key=lambda r: (int(r['query_id']), r['method']))
    averages, complete = aggregate(rows)
    return {'label': LABEL, 'offline_negative_finding': base.NEGATIVE, 'rows': rows,
            'aggregate': averages, 'complete_paired_queries': complete,
            'missing_results': [{'query_id': r['query_id'], 'method': r['method'], 'synthesis_status': r['synthesis_status'],
                                 'judge_status': r['judge_status']} for r in rows if r['quality'] is None],
            'mask_equality': m['mask_equality'],
            'historical_judge_failures': [{k: state.get(k) for k in ('stage', 'model', 'error_type', 'http_status', 'usage')}
                for _, state in recovery.call_records(Path(m['original_path'])) if state.get('status') != 'completed'],
            'extension_judge_failures': [{k: state.get(k) for k in ('alias', 'stage', 'model', 'error_type', 'http_status', 'usage')}
                for _, state in recovery.call_records(path.parent) if state.get('status') != 'completed'],
            'usage_note': 'Totals include reported provider usage only; absent usage is unknown, not zero billing.',
            'original_judge_attempts': len(list((Path(m['original_path']) / 'judge-calls').glob('*.json'))),
            'extension_synthesis_attempts': len(list((path.parent / 'synthesis-calls').glob('*.json'))),
            'extension_judge_attempts': len(list((path.parent / 'judge-calls').glob('*.json')))}


def summarize(path, output):
    if output.resolve().is_relative_to(Path(base.read(path)['original_path']).resolve()):
        raise ValueError('Summary must not write inside original pilot')
    if output.exists():
        raise FileExistsError(output)
    result = combined(path)
    output.mkdir(parents=True, exist_ok=False)
    base.write_new(output / 'summary.json', result)
    with (output / 'per-query.csv').open('x') as f:
        fields = ['query_id', 'method', *METRICS, 'context_tokens', 'synthesis_api_tokens', 'synthesis_latency_seconds', 'synthesis_status', 'judge_status']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result['rows']:
            writer.writerow({**{k: row[k] for k in ('query_id', 'method', 'context_tokens', 'synthesis_latency_seconds', 'synthesis_status', 'judge_status')},
                             **{k: (row['quality'] or {}).get(k) for k in METRICS},
                             'synthesis_api_tokens': (row['synthesis_usage'] or {}).get('total_tokens')})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--original', type=Path, default=Path('runs/evidence-replay-phase5/pilot-001'))
    for name in ('verify', 'synthesize', 'judge', 'summarize'):
        c = sub.add_parser(name)
        c.add_argument('--manifest', type=Path, required=True)
        if name in ('synthesize', 'judge'):
            c.add_argument('--allow-api', action='store_true')
            c.add_argument('--max-calls', type=int, default=9 if name == 'synthesize' else 18)
        if name == 'summarize':
            c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'freeze':
        result = freeze(args.output, args.original)
    elif args.command == 'verify':
        m = validate(args.manifest)
        result = {'status': 'verified', 'entries': len(m['entries']), 'original_unchanged': True}
    elif args.command == 'synthesize':
        result = asyncio.run(synthesize(args.manifest, allow_api=args.allow_api, max_calls=args.max_calls))
    elif args.command == 'judge':
        result = judge(args.manifest, allow_api=args.allow_api, max_calls=args.max_calls)
    else:
        result = summarize(args.manifest, args.output)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    main()
