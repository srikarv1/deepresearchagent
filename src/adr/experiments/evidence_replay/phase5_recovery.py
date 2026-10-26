"""Explicit, append-only recovery of the frozen Phase 5 RACE pilot.

The original driver and its hashed manifest remain untouched. No API runs in
prepare/verify/summarize. Each paid invocation requires --allow-api.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit

from . import phase5 as base
from .core import sha
from .synthesis import parse_response

VERSION = 'phase5-judge-recovery-v1'
SAFE_CODES = {'model_not_found', 'DeploymentNotFound', 'deployment_not_found',
              'ResourceNotFound', 'NotFound', 'not_found', 'invalid_api_key',
              'insufficient_quota', 'rate_limit_exceeded', 'permission_denied',
              'invalid_request_error', 'context_length_exceeded', 'unsupported_value',
              'unsupported_parameter', 'model_not_supported', 'content_filter',
              'authentication_error', 'not_found_error', 'server_error',
              'invalid_request', 'access_denied', 'Forbidden'}


def provider_error(exc):
    """Only known categorical codes/types; never persist free-text provider bodies."""
    body = getattr(exc, 'body', None)
    if not isinstance(body, dict):
        body = {}
    error = body.get('error', body)
    if not isinstance(error, dict):
        error = {}
    values = {'provider_error_code': getattr(exc, 'code', None) or error.get('code'),
              'provider_error_type': getattr(exc, 'type', None) or error.get('type')}
    return {k: (v if isinstance(v, str) and v in SAFE_CODES else
                'unrecognized_redacted' if v is not None else None) for k, v in values.items()}


def endpoint_identity():
    value = os.environ.get('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    u = urlsplit(value)
    return {'endpoint_sha256': hashlib.sha256(value.rstrip('/').encode()).hexdigest(),
            'endpoint_kind': 'azure-compatible' if 'azure' in (u.hostname or '') else
                             'openai' if u.hostname == 'api.openai.com' else 'compatible',
            'has_openai_v1_route': '/openai/v1' in u.path}


@contextmanager
def recovery_lock(root):
    # Shared across all recovery configurations for this original pilot.
    path = root / 'judge-recovery-active.lock'
    with path.open('x'):
        try:
            yield
        finally:
            path.unlink()


def result_records(root):
    paths = list((root / 'judging').glob('*-result.json'))
    for config in root.glob('judge-recovery-*/config.json'):
        paths.extend((config.parent / 'results').glob('*-result.json'))
    return [(path, base.read(path)) for path in paths]


def call_records(root):
    paths = list((root / 'judging').glob('*-clean.json')) + list((root / 'judging').glob('*-score.json'))
    for config in root.glob('judge-recovery-*/config.json'):
        paths.extend((config.parent / 'calls').glob('*.json'))
    return [(path, base.read(path)) for path in paths]


def legacy_identity(path, record, aliases):
    if record.get('alias') in aliases and record.get('stage') in ('clean', 'score'):
        return record['alias'], record['stage']
    for alias in aliases:
        for stage in ('clean', 'score'):
            if path.stem == alias + '-' + stage:
                return alias, stage
    raise ValueError('Unknown judge record identity')


def report_states(manifest, root):
    states = {}
    for e in manifest['entries']:
        d = root / e['alias']
        state = base.read(d / 'synthesis.json')
        if state.get('status') != 'completed' or sha(d / 'report.md') != state.get('report_sha256'):
            raise ValueError('Completed intact synthesis reports required')
        states[e['alias']] = state['report_sha256']
    return states


def prepare(manifest_path, output, *, judge_model, retry_failed=(), max_calls=13):
    if not judge_model or judge_model.strip() != judge_model or any(c.isspace() for c in judge_model):
        raise ValueError('Explicit nonempty judge deployment/model name required')
    if not 1 <= max_calls <= 13:
        raise ValueError('Cumulative judge cap must be at most 13')
    root = manifest_path.parent.resolve()
    output = output.resolve()
    if output.parent != root or not output.name.startswith('judge-recovery-'):
        raise ValueError('Recovery output must be pilot/judge-recovery-NAME')
    with recovery_lock(root):
        m = base.validate(manifest_path)
        reports = report_states(m, root)
        aliases = set(reports)
        records = call_records(root)
        for path, state in records:
            if state.get('status') == 'completed' and state.get('model') != judge_model:
                raise ValueError('Cannot mix judge models after any successful stage')
        # A model override is permitted only before any cleaning/scoring succeeds.
        requested = set(retry_failed)
        reservations = list((root / 'judge-calls').glob('*.json'))
        attempted = set()
        for path, state in records:
            alias, stage = legacy_identity(path, state, aliases)
            if state.get('status') != 'completed':
                attempted.add(alias + ':' + stage)
        # Reservations without metadata represent interrupted/unknown attempts.
        for path in reservations:
            record = base.read(path)
            try:
                alias, stage = legacy_identity(path, record, aliases)
            except ValueError:
                raise ValueError('Unrecognized historical reservation; manual audit required') from None
            if not any(legacy_identity(p, s, aliases) == (alias, stage) and s.get('status') == 'completed'
                       for p, s in records):
                attempted.add(alias + ':' + stage)
        if requested != attempted:
            raise ValueError('Explicit --retry-failed must exactly acknowledge failed/unknown stages: ' + ','.join(sorted(attempted)))
        needed = sum(not any(legacy_identity(p, s, aliases) == (alias, stage) and s.get('status') == 'completed'
                             for p, s in records) for alias in aliases for stage in ('clean', 'score'))
        if len(reservations) + needed > max_calls:
            raise ValueError('Insufficient cumulative call allowance for all six reports')
        # Immutable sidecar amendment, including snapshots of historical failure artifacts.
        originals = [manifest_path, root / 'manifest.sha256', root / 'references.json',
                     *reservations, *[p for p, _ in records],
                     *[p for p, _ in result_records(root)]]
        if (root / 'judge-stage.lock').exists():
            originals.append(root / 'judge-stage.lock')
        config = {'version': VERSION, 'manifest_path': str(manifest_path.resolve()),
                  'manifest_sha256': sha(manifest_path), 'judge_model': judge_model,
                  'max_cumulative_calls': max_calls, 'retry_failed': sorted(requested),
                  'historical_attempts': len(reservations), 'new_calls_needed': needed,
                  'reports': reports, 'label': base.LABEL,
                  'judge_variant': 'Explicit model override; retain official RACE prompts/scoring. Not a GPT-5.5 result.' if judge_model != m['judge']['model'] else 'Original judge model',
                  'completion_tokens': 16384, 'clean_reasoning': 'low', 'score_reasoning': 'medium',
                  'original_artifacts': {str(p.resolve()): sha(p) for p in originals},
                  'implementation_sha256': sha(Path(__file__))}
        output.mkdir(exist_ok=False)
        base.write_new(output / 'config.json', config)
        (output / 'config.sha256').write_text(sha(output / 'config.json'))
        return config


def verify(config_path):
    directory = config_path.parent
    if sha(config_path) != (directory / 'config.sha256').read_text():
        raise ValueError('Recovery configuration changed')
    c = base.read(config_path)
    if sha(Path(__file__)) != c['implementation_sha256']:
        raise ValueError('Recovery implementation changed')
    path = Path(c['manifest_path'])
    m = base.validate(path)
    if sha(path) != c['manifest_sha256'] or report_states(m, path.parent) != c['reports']:
        raise ValueError('Frozen pilot/report identity changed')
    for original, digest in c['original_artifacts'].items():
        if sha(Path(original)) != digest:
            raise ValueError('Historical artifact changed')
    return c, m


class RecoveryClient:
    def __init__(self, config, directory, ledger, *, transport=None):
        self.config, self.directory, self.ledger, self.transport = config, directory, ledger, transport
        self.directory.mkdir(exist_ok=True)
        self.root = Path(config['manifest_path']).parent
        self.aliases = set(config['reports'])

    def call(self, alias, stage, prompt):
        # Reuse successful stages, including score parsing after an interruption;
        # never repeat the successful stage's API request.
        previous = [(p, s) for p, s in call_records(self.root)
                    if legacy_identity(p, s, self.aliases) == (alias, stage)]
        complete = [s for _, s in previous if s.get('status') == 'completed']
        if complete:
            if len(complete) != 1 or complete[0].get('model') != self.config['judge_model']:
                raise ValueError('Duplicate successes or mixed judge models')
            return complete[0]['text']
        identity = self.directory.parent.name + '-' + alias + '-' + stage
        path = self.directory / (identity + '.json')
        # A reservation is permanent even after interruption; no restart retry.
        if path.exists() or (self.ledger.directory / (identity + '.json')).exists():
            raise RuntimeError('Previously attempted in this recovery; explicit new authorization required')
        if previous and alias + ':' + stage not in self.config['retry_failed']:
            raise RuntimeError('Failed stage was not explicitly authorized for recovery')
        self.ledger.reserve(identity)
        # Enrich only our new reservation, never mutate original historical files.
        reservation = self.ledger.directory / (identity + '.json')
        reservation.write_text(json.dumps({'identity': identity, 'alias': alias, 'stage': stage,
                                          'model': self.config['judge_model'], 'status': 'reserved'}))
        state = {'alias': alias, 'stage': stage, 'model': self.config['judge_model'],
                 'status': 'failed', 'usage': None, 'response': None, 'prompt': prompt}
        started = time.perf_counter()
        client = None
        try:
            payload = dict(model=self.config['judge_model'], messages=[{'role': 'user', 'content': prompt}],
                           max_completion_tokens=16384, reasoning_effort='low' if stage == 'clean' else 'medium')
            if self.transport is None:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                                base_url=os.environ.get('OPENAI_BASE_URL') or None, max_retries=0, timeout=600)
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
            state.update(provider_error(exc))
            state.update(error_type=type(exc).__name__, http_status=getattr(exc, 'status_code', None),
                         request_id=getattr(exc, 'request_id', None))
            raise RuntimeError('Judge request failed; see recovery diagnostics') from None
        finally:
            state['latency_seconds'] = time.perf_counter() - started
            base.write_new(path, state)
            if client is not None:
                client.close()


def run(config_path, *, allow_api=False, transport=None):
    if not allow_api:
        raise PermissionError('Explicit --allow-api required; prepare/verify make no requests')
    c, m = verify(config_path)
    root = Path(c['manifest_path']).parent
    with recovery_lock(root):
        # Revalidate after acquiring cross-protocol lock.
        c, m = verify(config_path)
        if transport is None and not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('OPENAI_API_KEY must be configured before recovery')
        binding = config_path.parent / 'execution-endpoint.json'
        identity = endpoint_identity()
        if binding.exists():
            if base.read(binding) != identity:
                raise ValueError('Endpoint changed within recovery protocol')
        else:
            base.write_new(binding, identity)
        aliases = set(c['reports'])
        if any(s.get('status') == 'completed' and s.get('model') != c['judge_model'] for _, s in call_records(root)):
            raise ValueError('Mixed judge models forbidden')
        directory = config_path.parent
        results = directory / 'results'
        results.mkdir(exist_ok=True)
        client = RecoveryClient(c, directory / 'calls', base.CallLedger(root / 'judge-calls', c['max_cumulative_calls']), transport=transport)
        refs = base.read(root / 'references.json')
        race = base.race_module()
        for e in sorted(m['entries'], key=lambda e: e['alias']):
            alias = e['alias']
            successes = [(p, s) for p, s in result_records(root)
                         if p.name == alias + '-result.json' and s.get('status') == 'completed']
            if successes:
                if len(successes) != 1 or successes[0][1].get('model') != c['judge_model'] or successes[0][1].get('report_sha256') != c['reports'][alias]:
                    raise ValueError('Successful result identity mismatch')
                continue
            result_path = results / (alias + '-result.json')
            if result_path.exists():
                raise RuntimeError('Failed result exists; no silent retry')
            # Detect interrupted/failed reservations before reusing any completed stage.
            for stage in ('clean', 'score'):
                reservation = root / 'judge-calls' / (directory.name + '-' + alias + '-' + stage + '.json')
                if reservation.exists():
                    metadata = directory / 'calls' / reservation.name
                    if not metadata.exists() or base.read(metadata).get('status') != 'completed':
                        raise RuntimeError('Interrupted/failed attempt; explicit recovery authorization required')
            try:
                result = base.score_one(race, client, alias, refs[e['query_id']],
                                        (root / alias / 'report.md').read_bytes().decode())
                base.write_new(result_path, {'status': 'completed', 'model': c['judge_model'],
                    'score': result, 'report_sha256': c['reports'][alias]})
            except Exception as exc:
                base.write_new(result_path, {'status': 'failed', 'model': c['judge_model'], 'score': None,
                                            'error_type': type(exc).__name__})
                raise RuntimeError('Recovery stopped after failure; no automatic retry') from None


def summarize(config_path):
    c, m = verify(config_path)
    root = Path(c['manifest_path']).parent
    rows = base.summary_rows(m, root)
    for row in rows:
        matches = [s for p, s in result_records(root) if p.name == row['alias'] + '-result.json' and s.get('status') == 'completed']
        if len(matches) > 1 or any(s.get('model') != c['judge_model'] for s in matches):
            raise ValueError('Duplicate evaluations or mixed models')
        row['quality'] = matches[0]['score'] if matches else None
        row['judge_status'] = 'completed' if matches else 'failed_or_not_run'
        row['judge_model'] = c['judge_model']
        calls = [s for p, s in call_records(root) if legacy_identity(p, s, set(c['reports']))[0] == row['alias']]
        row['judge_usage'] = [s.get('usage') for s in calls]
        row['judge_latency_seconds'] = sum(s.get('latency_seconds', 0) for s in calls)
    paired = [qid for qid in base.SOURCES if all(row['quality'] is not None for row in rows if row['query_id'] == qid)]
    aggregate = {}
    for method in ('threshold', 'shortlist', 'reinforce'):
        selected = [row for row in rows if row['method'] == method and row['query_id'] in paired]
        aggregate[method] = {'paired_queries': len(selected), 'mean_paired_race':
                             statistics.mean(row['quality']['overall_score'] for row in selected) if selected else None}
    calls = call_records(root)
    result = {'label': base.LABEL, 'offline_negative_finding': base.NEGATIVE,
              'judge_variant': c['judge_variant'], 'judge_model': c['judge_model'],
              'rows': rows, 'aggregate': aggregate, 'complete_paired_queries': paired,
              'cumulative_attempts': len(list((root / 'judge-calls').glob('*.json'))),
              'cumulative_limit': c['max_cumulative_calls'],
              'judge_reported_tokens': sum((s.get('usage') or {}).get('total_tokens', 0) for _, s in calls),
              'judge_unknown_usage_attempts': len(list((root / 'judge-calls').glob('*.json'))) - sum(s.get('usage') is not None for _, s in calls),
              'usage_note': 'Missing usage is unknown, not zero billing. All historical and recovery attempts counted.',
              'fact': False}
    base.write_new(config_path.parent / ('summary-' + str(time.time_ns()) + '.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_p = sub.add_parser('prepare')
    prepare_p.add_argument('--manifest', type=Path, required=True)
    prepare_p.add_argument('--output', type=Path, required=True)
    prepare_p.add_argument('--judge-model', required=True)
    prepare_p.add_argument('--max-calls', type=int, default=13)
    prepare_p.add_argument('--retry-failed', action='append', default=[])
    for command in ('judge', 'verify', 'summarize'):
        p = sub.add_parser(command)
        p.add_argument('--config', type=Path, required=True)
        if command == 'judge':
            p.add_argument('--allow-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.manifest, args.output, judge_model=args.judge_model,
                         max_calls=args.max_calls, retry_failed=args.retry_failed)
    elif args.command == 'judge':
        result = run(args.config, allow_api=args.allow_api)
    elif args.command == 'verify':
        c, _ = verify(args.config)
        result = {'status': 'verified', 'judge_model': c['judge_model'], 'max_cumulative_calls': c['max_cumulative_calls']}
    else:
        result = summarize(args.config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
