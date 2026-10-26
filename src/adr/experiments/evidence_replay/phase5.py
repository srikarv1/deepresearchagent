"""Frozen-retrieval, synthesis-only two-query pilot. Network is opt-in only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

from .core import Tokenizer, build_context, load_pool, make_plan, save_plan, select, sha
from .evaluation import coverage

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT / 'third_party/deep_research_bench'
SOURCES = {'52': '37aace7baa69', '54': '540e3cd646f4'}
CHECKPOINTS = {
    'shortlist': ('runs/evidence-replay-learning-signal/diagnostic-001/shortlist/seed-0/initial.pt',
                  'c32664a15bcb57a52ef8c939613c5a9581cb5a0990f862b6a86a84c75cdd3a63'),
    'reinforce': ('runs/evidence-replay-phase4/cpu-pilot-001/seed-0/training/epoch-020.pt',
                  '2e934a9a482f693a4554c6a9a6457d650f7a0a265a4477ee0aec9e1174268052')}
LABEL = 'Frozen-retrieval, synthesis-only exploratory pilot; two query-level examples, no significance claims.'
NEGATIVE = 'Trained REINFORCE did not outperform its untrained control in offline experiments; shortlist/filtering gains are not evidence of RL learning.'


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_bytes())


def official(question, qid, bench=BENCH):
    if coverage(question, bench) != str(qid):
        raise ValueError('Official query ID mismatch')
    result = {}
    for key, relative in [('query', 'prompt_data/query.jsonl'),
                          ('reference', 'test_data/cleaned_data/reference.jsonl'),
                          ('rubric', 'criteria_data/criteria.jsonl')]:
        path = bench / 'data' / relative
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        matches = [r for r in rows if r.get('prompt') == question]
        if len(matches) != 1:
            raise ValueError('Ambiguous official reference/rubric')
        result[key] = matches[0]
    return result


def freeze(output):
    import torch
    from .reward import RewardPool
    from .reinforce import Policy, ExactAllocator, rollout
    from .learning_signal import make_shortlist, sample, order_seed
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    models = {}
    for name, (relative, expected) in CHECKPOINTS.items():
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError('Frozen checkpoint identity mismatch')
        model = Policy()
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'] if name == 'reinforce' else state)
        model.eval()
        models[name] = model
    tok = Tokenizer()
    prepared = []
    references = {}
    for qid, tid in SOURCES.items():
        rp = RewardPool.load(ROOT / f'runs/_gpt_researcher_adaptive_depth_raw/trajectory_{tid}.json')
        references[qid] = official(rp.pool.query, qid)
        allocator = ExactAllocator(rp, tok, 50000)
        shortlist = make_shortlist(rp, allocator)
        episode = sample(models['shortlist'], rp, allocator, 'shortlist',
                         action_generator=torch.Generator().manual_seed(1000),
                         ordering_seed=order_seed(1000, 0, tid), shortlist=shortlist, track_grad=False)
        cap = episode['score']['context_tokens']
        ids = {'shortlist': episode['score']['selected_ids'],
               'threshold': select(rp.pool, 'threshold', tok, cap, 1000, .35)[1],
               'reinforce': rollout(models['reinforce'], rp, ExactAllocator(rp, tok, cap),
                                    generator=torch.Generator().manual_seed(1000),
                                    track_grad=False)['score']['selected_ids']}
        for method, selected in ids.items():
            plan = make_plan(rp.pool, tok, budget=cap, selected_ids=selected, seed=1000, threshold=.35)
            if not selected:
                raise ValueError('Empty frozen mask')
            alias = 'report-' + hashlib.sha256(f'phase5:1000:{qid}:{method}'.encode()).hexdigest()[:12]
            prepared.append((alias, qid, method, plan))
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for alias, qid, method, plan in prepared:
        directory = output / alias
        save_plan(plan, directory)
        entries.append({'alias': alias, 'query_id': qid, 'method': method,
                        'plan_sha256': sha(directory / 'plan.json'),
                        'context_sha256': plan['context_sha256'],
                        'context_tokens': plan['context_tokens'], 'budget': plan['config']['budget'],
                        'selected_count': len(plan['selected_ids'])})
    write_new(output / 'references.json', references)
    code = [Path(__file__), *[Path(__file__).with_name(n + '.py') for n in
                             ('core', 'synthesis', 'reinforce', 'learning_signal', 'reward')],
            BENCH / 'deepresearch_bench_race.py', BENCH / 'utils/clean_article.py',
            BENCH / 'utils/score_calculator.py', BENCH / 'prompt/clean_prompt.py',
            BENCH / 'prompt/score_prompt_en.py', BENCH / 'utils/json_extractor.py']
    manifest = {'label': LABEL, 'offline_negative_finding': NEGATIVE,
                'sampling_seed': 1000, 'threshold': .35, 'shortlist_k': 24,
                'checkpoints': CHECKPOINTS, 'entries': entries,
                'references_sha256': sha(output / 'references.json'),
                'code_sha256': {str(p.resolve()): sha(p) for p in code},
                'synthesis': {'model': 'gpt-5-mini', 'reasoning_effort': 'low', 'max_completion_tokens': 16384},
                'judge': {'model': 'gpt-5.5', 'clean_reasoning': 'low', 'score_reasoning': 'medium',
                          'max_completion_tokens': 16384, 'fact': False},
                'limits': {'synthesis': 6, 'judge': 12},
                'budget_note': 'Shortlist sampled at 50K. Other methods capped at its actual tokens; actor trained at 50K. Whole pages cannot match exactly.'}
    write_new(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(sha(output / 'manifest.json'))
    return manifest


def validate(manifest_path):
    directory = manifest_path.parent
    if sha(manifest_path) != (directory / 'manifest.sha256').read_text():
        raise ValueError('Manifest integrity mismatch')
    m = read(manifest_path)
    if sha(directory / 'references.json') != m['references_sha256']:
        raise ValueError('Reference integrity mismatch')
    for path, digest in m['code_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen implementation changed; refuse execution')
    for relative, digest in m['checkpoints'].values():
        if sha(ROOT / relative) != digest:
            raise ValueError('Checkpoint changed')
    tok = Tokenizer()
    for e in m['entries']:
        d = directory / e['alias']
        p = read(d / 'plan.json')
        if sha(d / 'plan.json') != e['plan_sha256'] or sha(d / 'context.txt') != e['context_sha256']:
            raise ValueError('Frozen mask/context changed')
        if tok.count((d / 'context.txt').read_bytes().decode()) != e['context_tokens'] or e['context_tokens'] > e['budget']:
            raise ValueError('Frozen token accounting mismatch')
        for path_key, hash_key in [('trajectory_path', 'trajectory_sha256'), ('embeddings_path', 'embeddings_sha256')]:
            if sha(Path(p['provenance'][path_key])) != p['provenance'][hash_key]:
                raise ValueError('Source pool changed')
        pool = load_pool(Path(p['provenance']['trajectory_path']))
        context, ids, tokens = build_context(pool, p['selected_ids'], tok, e['budget'])
        if context != (d / 'context.txt').read_bytes().decode() or p['mask'] != {page.id: page.id in ids for page in pool.pages}:
            raise ValueError('Evidence mask/context mismatch')
        if official(p['query'], e['query_id']) != read(directory / 'references.json')[e['query_id']]:
            raise ValueError('Official benchmark changed')
    return m


class CallLedger:
    """Exclusive persistent reservations. Interrupted/failed calls consume a slot.

    The enclosing stage lock rejects concurrent commands; reservations survive crashes.
    """
    def __init__(self, directory, maximum):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
        self.maximum = maximum

    def reserve(self, identity):
        if (self.directory / (identity + '.json')).exists():
            raise RuntimeError('Already attempted; automatic retry forbidden')
        if len(list(self.directory.glob('*.json'))) >= self.maximum:
            raise RuntimeError('API call cap reached')
        write_new(self.directory / (identity + '.json'), {'identity': identity, 'status': 'reserved'})


def load_plan(directory):
    plan = read(directory / 'plan.json')
    plan['context'] = (directory / 'context.txt').read_bytes().decode('utf-8')
    return plan


async def run_synthesis(manifest_path, *, allow_api=False, max_calls=6):
    from .synthesis import synthesize
    if not allow_api:
        raise PermissionError('Explicit --allow-api required')
    m = validate(manifest_path)
    if not 0 < max_calls <= 6:
        raise ValueError('Synthesis cap must be between 1 and 6')
    root = manifest_path.parent
    with (root / 'synthesis-stage.lock').open('x'):
        ledger = CallLedger(root / 'synthesis-calls', max_calls)
        for e in m['entries']:
            d = root / e['alias']
            ledger.reserve(e['alias'])
            try:
                await synthesize(load_plan(d), d, allow_api=True, **m['synthesis'])
            except Exception as exc:
                write_new(d / 'failure.json', {'stage': 'synthesis', 'error_type': type(exc).__name__})
                raise RuntimeError('Synthesis failed; stopped without retry; see artifacts') from None


def race_module():
    sys.path.insert(0, str(BENCH))
    try:
        module = importlib.import_module('deepresearch_bench_race')
        if Path(module.__file__).resolve() != (BENCH / 'deepresearch_bench_race.py').resolve():
            raise ValueError('Unexpected RACE implementation')
        return module
    finally:
        sys.path.pop(0)


class JudgeClient:
    def __init__(self, ledger, directory, model='gpt-5.5', transport=None):
        self.ledger, self.directory, self.model = ledger, directory, model
        self.transport = transport

    def call(self, alias, stage, prompt):
        from .synthesis import parse_response
        identity = alias + '-' + stage
        self.ledger.reserve(identity)
        path = self.directory / (identity + '.json')
        state = {'status': 'failed', 'stage': stage, 'model': self.model, 'usage': None,
                 'response': None, 'prompt': prompt}
        started = time.perf_counter()
        client = None
        try:
            payload = dict(model=self.model, messages=[{'role': 'user', 'content': prompt}],
                           reasoning_effort='low' if stage == 'clean' else 'medium', max_completion_tokens=16384)
            if self.transport is None:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                                base_url=os.environ.get('OPENAI_BASE_URL') or None,
                                max_retries=0, timeout=600)
                response = client.chat.completions.create(**payload)
            else:
                response = self.transport(**payload)
            text, error = parse_response(response, state)
            state['text'] = text
            if error:
                raise RuntimeError('Incomplete judge response')
            state['status'] = 'completed'
            return text
        except Exception as exc:
            state['error_type'] = type(exc).__name__
            state['http_status'] = getattr(exc, 'status_code', None)
            state['request_id'] = getattr(exc, 'request_id', None)
            raise RuntimeError('Judge request failed; see saved diagnostics') from None
        finally:
            state['latency_seconds'] = time.perf_counter() - started
            write_new(path, state)
            if client is not None:
                client.close()


def validate_scores(scores, rubric):
    """Reject incomplete criteria rather than allowing upstream zero defaults."""
    for dim, criteria in rubric['criterions'].items():
        values = scores.get(dim)
        if not isinstance(values, list) or len(values) != len(criteria):
            raise ValueError('Incomplete scored criteria')
        expected = {c['criterion'].strip() for c in criteria}
        if {v.get('criterion', '').strip() for v in values} != expected:
            raise ValueError('Scored criteria mismatch')
        for v in values:
            for key in ('article_1_score', 'article_2_score'):
                value = v.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 10:
                    raise ValueError('Invalid criterion score')
    if not any(v['article_1_score'] + v['article_2_score'] > 0 for values in scores.values() for v in values):
        raise ValueError('Undefined all-zero relative score')


def score_one(race, client, alias, reference, report, *, skip_cleaning=False):
    # Explicit rejection prevents the upstream flag's silent scoring bypass.
    if skip_cleaning:
        raise ValueError('skip_cleaning is not supported; scoring must follow one cleaning request')
    cleaner = race.ArticleCleaner(None)
    cleaned = client.call(alias, 'clean', cleaner._get_clean_prompt('en').format(article=report))
    if not cleaner._is_valid_result(cleaned):
        raise ValueError('Invalid cleaned report; no recursive fallback')
    class Adapter:
        def generate(self, user_prompt, system_prompt=''):
            text = client.call(alias, 'score', user_prompt)
            validate_scores(json.loads(race.extract_json_from_markdown(text)), reference['rubric'])
            return text
    q = reference['query']['prompt']
    result = race.process_single_item({'id': alias, 'prompt': q}, {q: {'article': cleaned}},
        {q: reference['reference']}, {q: reference['rubric']}, Adapter(), threading.Lock(),
        SimpleNamespace(update=lambda n: None), max_retries=1, language='en')
    if 'error' in result or not all(math.isfinite(result.get(k, float('nan'))) for k in
                                    ('overall_score', 'comprehensiveness', 'insight', 'instruction_following', 'readability')):
        raise ValueError('Invalid RACE scoring result; no zero imputation')
    return result


def run_judge(manifest_path, *, allow_api=False, max_calls=12, skip_cleaning=False):
    if skip_cleaning:
        raise ValueError('skip_cleaning cannot skip scoring')
    if not allow_api:
        raise PermissionError('Explicit --allow-api required')
    m = validate(manifest_path)
    if not 0 < max_calls <= 12:
        raise ValueError('Judge cap must be between 1 and 12')
    root = manifest_path.parent
    with (root / 'judge-stage.lock').open('x'):
        directory = root / 'judging'
        directory.mkdir(exist_ok=False)
        client = JudgeClient(CallLedger(root / 'judge-calls', max_calls), directory, m['judge']['model'])
        race = race_module()
        references = read(root / 'references.json')
        # Opaque aliases only in judge prompts. Method/cost/proxy metadata never supplied.
        for e in sorted(m['entries'], key=lambda e: e['alias']):
            d = root / e['alias']
            state = read(d / 'synthesis.json') if (d / 'synthesis.json').exists() else {}
            if state.get('status') != 'completed':
                write_new(directory / (e['alias'] + '-result.json'), {'status': 'unavailable', 'score': None})
                continue
            if sha(d / 'report.md') != state['report_sha256']:
                raise ValueError('Report integrity mismatch')
            try:
                result = score_one(race, client, e['alias'], references[e['query_id']],
                                   (d / 'report.md').read_bytes().decode())
                write_new(directory / (e['alias'] + '-result.json'), {'status': 'completed', 'score': result,
                          'report_sha256': state['report_sha256']})
            except Exception as exc:
                write_new(directory / (e['alias'] + '-result.json'), {'status': 'failed', 'score': None,
                          'error_type': type(exc).__name__})
                raise RuntimeError('Judging failed; stopped without retry; partial results preserved') from None


def summary_rows(manifest, root):
    rows = []
    for e in manifest['entries']:
        d = root / e['alias']
        synthesis = read(d / 'synthesis.json') if (d / 'synthesis.json').exists() else {}
        result_path = root / 'judging' / (e['alias'] + '-result.json')
        judged = read(result_path) if result_path.exists() else {}
        calls = [read(p) for p in (root / 'judging').glob(e['alias'] + '-*.json') if not p.name.endswith('-result.json')]
        rows.append(e | {'synthesis_status': synthesis.get('status', 'not_run'),
            'synthesis_usage': synthesis.get('usage'), 'synthesis_latency_seconds': synthesis.get('latency_seconds'),
            'report_characters': len((d / 'report.md').read_bytes().decode()) if (d / 'report.md').exists() else None,
            'judge_usage': [c.get('usage') for c in calls],
            'judge_latency_seconds': sum(c.get('latency_seconds', 0) for c in calls),
            'judge_status': judged.get('status', 'not_run'),
            'quality': judged.get('score') if judged.get('status') == 'completed' else None})
    return rows


def summarize(manifest_path):
    m = validate(manifest_path)
    rows = summary_rows(m, manifest_path.parent)
    paired = [q for q in SOURCES if all(r['quality'] is not None for r in rows if r['query_id'] == q)]
    aggregate = {}
    for method in ('threshold', 'shortlist', 'reinforce'):
        group = [r for r in rows if r['method'] == method]
        available = [r['quality']['overall_score'] for r in group if r['quality'] is not None]
        complete = [r['quality']['overall_score'] for r in group if r['query_id'] in paired]
        aggregate[method] = {'completed_queries': len(available), 'mean_available_race': statistics.mean(available) if available else None,
                             'mean_paired_race': statistics.mean(complete) if complete else None,
                             'mean_context_tokens': statistics.mean(r['context_tokens'] for r in group)}
    result = {'label': LABEL, 'offline_negative_finding': NEGATIVE, 'rows': rows,
              'aggregate': aggregate, 'complete_paired_queries': paired,
              'api_tokens': {stage: sum((u or {}).get('total_tokens', 0) for r in rows for u in
                  ([r['synthesis_usage']] if stage == 'synthesis' else r['judge_usage']))
                  for stage in ('synthesis', 'judge')},
              'usage_note': 'Missing provider usage is unknown, not zero billed tokens; totals include only reported usage, including failed responses.',
              'limitations': 'Two queries; single reports/judge draws; FACT/citation correctness unmeasured. No claim of improved report quality from proxy reward.'}
    write_new(manifest_path.parent / ('summary-' + str(time.time_ns()) + '.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--output', type=Path, required=True)
    for name in ('synthesize', 'judge', 'summarize'):
        c = sub.add_parser(name)
        c.add_argument('--manifest', type=Path, required=True)
        if name != 'summarize':
            c.add_argument('--allow-api', action='store_true')
            c.add_argument('--max-calls', type=int, default=6 if name == 'synthesize' else 12)
        if name == 'judge':
            c.add_argument('--skip-cleaning', '--skip_cleaning', action='store_true')
    args = p.parse_args()
    if args.command == 'freeze':
        result = freeze(args.output)
    elif args.command == 'synthesize':
        result = asyncio.run(run_synthesis(args.manifest, allow_api=args.allow_api, max_calls=args.max_calls))
    elif args.command == 'judge':
        result = run_judge(args.manifest, allow_api=args.allow_api, max_calls=args.max_calls, skip_cleaning=args.skip_cleaning)
    else:
        result = summarize(args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
