"""Offline extension, cap, consistency and five-query aggregation tests."""
import asyncio
import socket
from pathlib import Path

import pytest

from . import phase5 as b
from . import phase5_extension as x
from . import phase5_recovery as r
from .test_phase5 import response

ORIGINAL = b.ROOT / 'runs/evidence-replay-phase5/pilot-001'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('External networking forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    path = tmp_path_factory.mktemp('extension') / 'frozen'
    m = x.freeze(path, ORIGINAL)
    return path, m


def test_real_freeze_preserves_original_and_reference_matches(frozen):
    root, m = frozen
    assert x.validate(root / 'manifest.json')['entries'] == m['entries']
    assert len(m['entries']) == 9
    assert x.inventory(ORIGINAL) == b.read(root / 'original-inventory.json')
    refs = b.read(root / 'references.json')
    for e in m['entries']:
        plan = b.read(root / e['alias'] / 'plan.json')
        assert b.official(plan['query'], e['query_id']) == refs[e['query_id']]
        assert e['context_tokens'] <= e['budget'] <= 50000
        assert b.sha(root / e['alias'] / 'context.txt') == e['context_sha256']
        assert plan['config']['seed'] == 1000 and plan['config']['threshold'] == .35
    assert m['limits'] == {'synthesis': 9, 'judge': 18}
    assert m['judge']['model'] == 'gpt-5-mini'
    assert m['mask_equality']['52']['trained_equals_own_untrained_cosine_control']
    assert m['mask_equality']['54']['trained_equals_own_untrained_cosine_control']


def test_deterministic_freeze_and_no_overwrite(frozen, tmp_path):
    root, m = frozen
    second = x.freeze(tmp_path / 'repeat', ORIGINAL)
    assert second['entries'] == m['entries']
    assert second['mask_equality'] == m['mask_equality']
    with pytest.raises(FileExistsError):
        x.freeze(root, ORIGINAL)


def test_combined_offline_missing_results_are_null(frozen, tmp_path):
    root, _ = frozen
    result = x.summarize(root / 'manifest.json', tmp_path / 'summary')
    assert len(result['rows']) == 15
    assert result['complete_paired_queries'] == ['52', '54']
    assert len(result['missing_results']) == 9
    assert result['original_judge_attempts'] == 13
    assert len(result['historical_judge_failures']) == 1
    assert all(a['five_query_macro']['overall_score'] is None for a in result['aggregate'].values())
    assert x.inventory(ORIGINAL) == b.read(root / 'original-inventory.json')
    with pytest.raises(ValueError, match='original pilot'):
        x.summarize(root / 'manifest.json', ORIGINAL / 'forbidden-summary')


def rows():
    return [{'query_id': q, 'method': method, 'quality': {metric: int(q) / 100 for metric in x.METRICS},
             'context_tokens': 100, 'synthesis_usage': {'total_tokens': 200}, 'synthesis_latency_seconds': 2}
            for q in ('51', '52', '53', '54', '55') for method in x.METHODS]


def test_five_query_macro_and_costs():
    averages, complete = x.aggregate(rows())
    assert len(complete) == 5
    for a in averages.values():
        assert all(score == pytest.approx(.53) for score in a['five_query_macro'].values())
        assert a['mean_context_tokens'] == 100
        assert a['reported_synthesis_api_tokens'] == 1000
        assert a['total_synthesis_latency_seconds'] == 10
        assert a['mean_synthesis_latency_seconds'] == 2


def test_missing_failure_not_zero_and_duplicates_rejected():
    values = rows()
    values[0]['quality'] = None
    averages, complete = x.aggregate(values)
    assert complete == ['52', '53', '54', '55']
    assert all(a['five_query_macro']['overall_score'] is None for a in averages.values())
    assert averages['threshold']['completed_scores'] == 4
    with pytest.raises(ValueError):
        x.aggregate(values[:-1])
    with pytest.raises(ValueError):
        x.aggregate(values[:-1] + [values[0]])


@pytest.mark.parametrize('stage,cap', [('synthesize', 10), ('judge', 19)])
def test_caps_cannot_be_raised(tmp_path, monkeypatch, stage, cap):
    monkeypatch.setattr(x, 'validate', lambda _: {})
    with pytest.raises(ValueError, match='cap'):
        if stage == 'synthesize':
            asyncio.run(x.synthesize(tmp_path / 'manifest.json', allow_api=True, max_calls=cap))
        else:
            x.judge(tmp_path / 'manifest.json', allow_api=True, max_calls=cap)


def test_paid_permission_required(tmp_path):
    with pytest.raises(PermissionError):
        asyncio.run(x.synthesize(tmp_path / 'manifest.json'))
    with pytest.raises(PermissionError):
        x.judge(tmp_path / 'manifest.json')


def test_original_judge_settings_must_match(monkeypatch):
    monkeypatch.setattr(r, 'verify', lambda _: ({'judge_model': 'different', 'completion_tokens': 16384,
                                              'clean_reasoning': 'low', 'score_reasoning': 'medium'}, {}))
    with pytest.raises(ValueError, match='configuration'):
        x.original_rows(Path('unused'))


def test_synthesis_failure_stops_without_retry(tmp_path, monkeypatch):
    from . import synthesis
    monkeypatch.setattr(x, 'validate', lambda _: {'entries': [{'alias': 'a'}, {'alias': 'b'}], 'synthesis': {}})
    monkeypatch.setattr(x, 'api_preflight', lambda *a: None)
    monkeypatch.setattr(b, 'load_plan', lambda _: {})
    for alias in ('a', 'b'):
        (tmp_path / alias).mkdir()
    calls = []
    async def fail(*a, **k):
        calls.append(1)
        raise RuntimeError('failed')
    monkeypatch.setattr(synthesis, 'synthesize', fail)
    with pytest.raises(RuntimeError):
        asyncio.run(x.synthesize(tmp_path / 'manifest.json', allow_api=True))
    assert len(calls) == 1
    assert len(list((tmp_path / 'synthesis-calls').glob('*.json'))) == 1
    with pytest.raises(FileExistsError):
        asyncio.run(x.synthesize(tmp_path / 'manifest.json', allow_api=True))
    assert len(calls) == 1


def test_judge_failure_saves_usage_and_stops(tmp_path, monkeypatch):
    m = {'entries': [{'alias': 'a', 'query_id': '51'}, {'alias': 'b', 'query_id': '53'}]}
    monkeypatch.setattr(x, 'validate', lambda _: m)
    monkeypatch.setattr(x, 'api_preflight', lambda *a: None)
    reports = {}
    for alias in ('a', 'b'):
        d = tmp_path / alias
        d.mkdir()
        (d / 'report.md').write_text('report ' * 100)
        reports[alias] = b.sha(d / 'report.md')
        b.write_new(d / 'synthesis.json', {'status': 'completed', 'report_sha256': reports[alias]})
    monkeypatch.setattr(r, 'report_states', lambda *a: reports)
    b.write_new(tmp_path / 'references.json', {'51': {}, '53': {}})
    calls = []
    def transport(**payload):
        calls.append(payload)
        return response('partial', 'length')
    with pytest.raises(RuntimeError):
        x.judge(tmp_path / 'manifest.json', allow_api=True, transport=transport)
    assert len(calls) == 1 and calls[0]['model'] == 'gpt-5-mini'
    assert b.read(tmp_path / 'judging/a-result.json')['score'] is None
    assert len(list((tmp_path / 'judge-calls').glob('*.json'))) == 1
    diagnostics = next(p for p in (tmp_path / 'judging').glob('*-clean.json'))
    assert b.read(diagnostics)['usage']['total_tokens'] == 13
    with pytest.raises(FileExistsError):
        x.judge(tmp_path / 'manifest.json', allow_api=True, transport=transport)
    assert len(calls) == 1


def test_persistent_cap18_counts_prior_attempts(tmp_path):
    ledger = b.CallLedger(tmp_path / 'ledger', 18)
    for i in range(18):
        ledger.reserve(str(i))
    with pytest.raises(RuntimeError, match='cap'):
        b.CallLedger(tmp_path / 'ledger', 18).reserve('extra')


def test_endpoint_consistency():
    with pytest.raises(ValueError):
        x.api_preflight({'judge_endpoint': {'endpoint_sha256': 'different'}})
