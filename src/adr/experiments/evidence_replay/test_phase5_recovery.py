"""Recovery tests: every model response mocked and all sockets denied."""
import json
import socket
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from . import phase5 as base
from . import phase5_recovery as r
from .test_phase5 import response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    # Lightweight control-plane fixture; existing Phase 5 tests verify real masks.
    aliases = ['report-a', 'report-b']
    entries = [{'alias': a, 'query_id': '52', 'method': m, 'context_tokens': 100}
               for a, m in zip(aliases, ('shortlist', 'reinforce'))]
    manifest = {'entries': entries, 'judge': {'model': 'gpt-5.5'}}
    base.write_new(tmp_path / 'manifest.json', manifest)
    (tmp_path / 'manifest.sha256').write_text(base.sha(tmp_path / 'manifest.json'))
    # Use a real official reference/rubric to exercise official scoring as well.
    queries = [json.loads(s) for s in (base.BENCH / 'data/prompt_data/query.jsonl').read_text().splitlines()]
    refs = base.official(next(q['prompt'] for q in queries if str(q['id']) == '52'), '52')
    base.write_new(tmp_path / 'references.json', {'52': refs})
    for alias in aliases:
        d = tmp_path / alias
        d.mkdir()
        (d / 'report.md').write_text('A blinded evidence report ' * 20)
        base.write_new(d / 'synthesis.json', {'status': 'completed', 'report_sha256': base.sha(d / 'report.md')})
    (tmp_path / 'judging').mkdir()
    (tmp_path / 'judge-calls').mkdir()
    (tmp_path / 'judge-stage.lock').touch()
    base.write_new(tmp_path / 'judge-calls/report-a-clean.json', {'identity': 'report-a-clean', 'status': 'reserved'})
    base.write_new(tmp_path / 'judging/report-a-clean.json', {'model': 'gpt-5.5', 'status': 'failed', 'stage': 'clean',
        'http_status': 404, 'error_type': 'NotFoundError', 'usage': None})
    base.write_new(tmp_path / 'judging/report-a-result.json', {'status': 'failed', 'score': None})
    monkeypatch.setattr(base, 'validate', lambda _: manifest)
    return tmp_path, manifest, refs


def config(pilot, model='gpt-5-mini', cap=13, retry=('report-a:clean',)):
    root, _, _ = pilot
    output = root / 'judge-recovery-001'
    c = r.prepare(root / 'manifest.json', output, judge_model=model, retry_failed=retry, max_calls=cap)
    return output / 'config.json', c


def test_prepare_preserves_failure_and_requires_acknowledgement(pilot):
    root, _, _ = pilot
    before = {str(p): base.sha(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='acknowledge'):
        config(pilot, retry=())
    path, c = config(pilot)
    assert c['judge_model'] == 'gpt-5-mini' and c['historical_attempts'] == 1
    assert c['max_cumulative_calls'] == 13 and c['new_calls_needed'] == 4
    assert all(base.sha(Path(p)) == digest for p, digest in before.items())
    assert r.verify(path)[0] == c
    assert not (root / 'judge-recovery-active.lock').exists()


def transport_for(refs, calls):
    scores = {dim: [{'criterion': c['criterion'], 'article_1_score': 6, 'article_2_score': 4}
                    for c in criteria] for dim, criteria in refs['rubric']['criterions'].items()}
    def transport(**payload):
        calls.append(payload)
        assert payload['model'] == 'gpt-5-mini'
        text = 'cleaned report ' * 20 if payload['reasoning_effort'] == 'low' else json.dumps(scores)
        return response(text)
    return transport


def test_recovery_all_reports_one_model_no_duplicates(pilot):
    root, _, refs = pilot
    path, _ = config(pilot)
    calls = []
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    assert len(calls) == 4
    assert len(list((root / 'judge-calls').glob('*.json'))) == 5
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    assert len(calls) == 4  # completed results are skipped, not judged again
    assert base.read(root / 'judging/report-a-result.json')['score'] is None
    result = r.summarize(path)
    assert result['judge_model'] == 'gpt-5-mini'
    assert result['cumulative_attempts'] == 5
    assert result['judge_reported_tokens'] == 52
    assert result['judge_unknown_usage_attempts'] == 1
    assert all(row['quality']['overall_score'] == pytest.approx(.6) for row in result['rows'])


def test_404_diagnostics_and_no_automatic_retry(pilot):
    root, _, _ = pilot
    path, _ = config(pilot)
    calls = []
    class NotFoundError(Exception):
        status_code = 404
        request_id = 'request-123'
        body = {'error': {'code': 'DeploymentNotFound', 'type': 'invalid_request_error', 'message': 'SECRET'}}
    def fail(**payload):
        calls.append(payload)
        raise NotFoundError('SECRET')
    with pytest.raises(RuntimeError):
        r.run(path, allow_api=True, transport=fail)
    diagnostics = list((path.parent / 'calls').glob('*.json'))
    assert len(diagnostics) == 1
    d = base.read(diagnostics[0])
    assert d['http_status'] == 404 and d['provider_error_code'] == 'DeploymentNotFound'
    assert d['provider_error_type'] == 'invalid_request_error' and d['usage'] is None
    assert 'SECRET' not in diagnostics[0].read_text()
    with pytest.raises(RuntimeError):
        r.run(path, allow_api=True, transport=fail)
    assert len(calls) == 1
    assert len(list((root / 'judge-calls').glob('*.json'))) == 2


def test_cumulative_cap_includes_historical_attempt(pilot):
    root, _, refs = pilot
    path, _ = config(pilot, cap=5)  # exactly 1 old + 4 new
    calls = []
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    ledger = base.CallLedger(root / 'judge-calls', 5)
    with pytest.raises(RuntimeError, match='cap'):
        ledger.reserve('sixth')
    assert len(calls) == 4


def test_budget_cannot_exceed_13_or_ignore_remaining_work(pilot):
    with pytest.raises(ValueError, match='13'):
        config(pilot, cap=14)
    with pytest.raises(ValueError, match='Insufficient'):
        config(pilot, cap=4)


def test_model_cannot_change_after_cleaning_succeeds(pilot):
    root, _, _ = pilot
    base.write_new(root / 'judging/report-b-clean.json', {'status': 'completed', 'model': 'gpt-5.5', 'stage': 'clean', 'text': 'old'})
    with pytest.raises(ValueError, match='mix judge models'):
        config(pilot)


def test_config_and_original_artifact_tampering_blocked(pilot):
    root, _, _ = pilot
    path, _ = config(pilot)
    (root / 'judging/report-a-clean.json').write_text('{}')
    with pytest.raises(ValueError, match='Historical'):
        r.verify(path)


def test_configuration_override_tampering_blocked(pilot):
    path, _ = config(pilot)
    d = base.read(path)
    d['judge_model'] = 'another-model'
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='configuration changed'):
        r.verify(path)


def test_missing_approval_and_endpoint_switch_refused(pilot, monkeypatch):
    _, _, refs = pilot
    path, _ = config(pilot)
    with pytest.raises(PermissionError):
        r.run(path)
    calls = []
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://different.invalid/openai/v1')
    with pytest.raises(ValueError, match='Endpoint changed'):
        r.run(path, allow_api=True, transport=transport_for(refs, calls))
    assert len(calls) == 4


def test_global_recovery_lock_and_interrupted_reservation(pilot):
    root, _, refs = pilot
    path, _ = config(pilot)
    with r.recovery_lock(root):
        with pytest.raises(FileExistsError):
            r.run(path, allow_api=True, transport=transport_for(refs, []))
    base.CallLedger(root / 'judge-calls', 13).reserve('judge-recovery-001-report-a-clean')
    with pytest.raises(RuntimeError, match='Interrupted'):
        r.run(path, allow_api=True, transport=transport_for(refs, []))


def test_unknown_provider_diagnostics_redacted():
    exc = NS(code='sk-secret-key', type='https://secret-url', body={'message': 'private'})
    assert r.provider_error(exc) == {'provider_error_code': 'unrecognized_redacted', 'provider_error_type': 'unrecognized_redacted'}


def test_reuse_completed_score_response_after_interruption(pilot):
    root, _, refs = pilot
    path, c = config(pilot)
    calls = []
    client = r.RecoveryClient(c, path.parent / 'calls', base.CallLedger(root / 'judge-calls', 13),
                             transport=transport_for(refs, calls))
    race = base.race_module()
    # Simulate interruption after successful scoring but before final result saving.
    result = base.score_one(race, client, 'report-a', refs, 'report ' * 100)
    assert result['overall_score'] == pytest.approx(.6) and len(calls) == 2
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    assert len(calls) == 4  # a reused offline; only b made two new calls
    assert base.read(path.parent / 'results/report-a-result.json')['status'] == 'completed'


def test_new_sidecar_cannot_duplicate_prior_successes(pilot):
    root, _, refs = pilot
    path, _ = config(pilot)
    calls = []
    r.run(path, allow_api=True, transport=transport_for(refs, calls))
    second = root / 'judge-recovery-002'
    c = r.prepare(root / 'manifest.json', second, judge_model='gpt-5-mini', retry_failed=['report-a:clean'], max_calls=13)
    assert c['new_calls_needed'] == 0
    r.run(second / 'config.json', allow_api=True, transport=transport_for(refs, calls))
    assert len(calls) == 4
