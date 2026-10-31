"""No-network tests for frozen downstream evaluation control plane."""
import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from . import phase5 as p


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*a, **k):
        raise AssertionError('No network in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def response(text='answer', finish='stop'):
    return NS(id='r', model='mock', usage=NS(model_dump=lambda: {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}),
              choices=[NS(finish_reason=finish, message=NS(content=text, refusal=None, tool_calls=[]))])


def test_ledger_caps_and_persistent_no_retry(tmp_path):
    ledger = p.CallLedger(tmp_path / 'calls', 2)
    ledger.reserve('a')
    with pytest.raises(RuntimeError, match='Already attempted'):
        ledger.reserve('a')
    p.CallLedger(tmp_path / 'calls', 2).reserve('b')
    with pytest.raises(RuntimeError, match='cap'):
        ledger.reserve('c')


def test_official_exact_match():
    queries = [json.loads(s) for s in (p.BENCH / 'data/prompt_data/query.jsonl').read_text().splitlines()]
    for qid in ('52', '54'):
        q = next(q for q in queries if str(q['id']) == qid)
        refs = p.official(q['prompt'], qid)
        assert refs['reference']['article'] and refs['rubric']['criterions']
        with pytest.raises(ValueError):
            p.official(q['prompt'] + ' changed', qid)
        with pytest.raises(ValueError, match='ID mismatch'):
            p.official(q['prompt'], '999')


def test_real_freeze_and_integrity(tmp_path):
    root = tmp_path / 'freeze'
    m = p.freeze(root)
    assert len(m['entries']) == 6
    repeat = p.freeze(tmp_path / 'repeat')
    assert repeat['entries'] == m['entries']
    assert p.validate(root / 'manifest.json') == json.loads(json.dumps(m))
    expected = {'52': {'shortlist': 46454, 'threshold': 46284, 'reinforce': 46434},
                '54': {'shortlist': 33405, 'threshold': 33375, 'reinforce': 33374}}
    for e in m['entries']:
        assert e['context_tokens'] == expected[e['query_id']][e['method']]
        assert len(e['context_sha256']) == 64
        assert e['context_tokens'] <= e['budget']
    with pytest.raises(FileExistsError):
        p.freeze(root)
    target = root / m['entries'][0]['alias'] / 'context.txt'
    target.write_bytes(target.read_bytes() + b'leak')
    with pytest.raises(ValueError, match='changed'):
        p.validate(root / 'manifest.json')


def test_judge_partial_failure_and_usage(tmp_path):
    calls = []
    def transport(**payload):
        calls.append(payload)
        return response('partial', 'length')
    client = p.JudgeClient(p.CallLedger(tmp_path / 'ledger', 12), tmp_path, transport=transport)
    with pytest.raises(RuntimeError):
        client.call('opaque', 'clean', 'only the report')
    artifact = p.read(tmp_path / 'opaque-clean.json')
    assert artifact['usage']['total_tokens'] == 13
    assert artifact['response']['finish_reason'] == 'length'
    assert artifact['text'] == 'partial' and artifact['status'] == 'failed'
    with pytest.raises(RuntimeError, match='Already attempted'):
        client.call('opaque', 'clean', 'only the report')
    assert len(calls) == 1


def test_transport_errors_are_sanitized(tmp_path):
    def fail(**kwargs):
        raise RuntimeError('secret credential must never be saved')
    client = p.JudgeClient(p.CallLedger(tmp_path / 'ledger', 12), tmp_path, transport=fail)
    with pytest.raises(RuntimeError):
        client.call('opaque', 'score', 'prompt')
    assert 'secret' not in (tmp_path / 'opaque-score.json').read_text()


def test_skip_cleaning_rejected_and_no_api(tmp_path):
    with pytest.raises(ValueError, match='skip'):
        p.run_judge(tmp_path / 'absent', skip_cleaning=True, allow_api=True)
    with pytest.raises(PermissionError):
        p.run_judge(tmp_path / 'absent')
    with pytest.raises(PermissionError):
        asyncio.run(p.run_synthesis(tmp_path / 'absent'))


def test_official_scoring_blinding_and_no_retries():
    race = p.race_module()
    queries = [json.loads(s) for s in (p.BENCH / 'data/prompt_data/query.jsonl').read_text().splitlines()]
    ref = p.official(next(q['prompt'] for q in queries if str(q['id']) == '52'), '52')
    scores = {dim: [{'criterion': c['criterion'], 'article_1_score': 6, 'article_2_score': 4}
                    for c in criteria] for dim, criteria in ref['rubric']['criterions'].items()}
    class Client:
        def __init__(self):
            self.calls = []
        def call(self, alias, stage, prompt):
            self.calls.append((alias, stage, prompt))
            assert 'reinforce' not in prompt.lower()
            assert 'checkpoint' not in prompt.lower()
            return 'cleaned evidence report ' * 10 if stage == 'clean' else json.dumps(scores)
    client = Client()
    result = p.score_one(race, client, 'opaque', ref, 'evidence report ' * 20)
    assert result['overall_score'] == pytest.approx(.6)
    assert [c[1] for c in client.calls] == ['clean', 'score']
    class Broken(Client):
        def call(self, alias, stage, prompt):
            self.calls.append(stage)
            return 'x' * 100 if stage == 'clean' else '{}'
    broken = Broken()
    with pytest.raises(ValueError):
        p.score_one(race, broken, 'opaque', ref, 'evidence report')
    assert broken.calls == ['clean', 'score']


def test_score_validation_rejects_zero_defaults():
    rubric = {'criterions': {'insight': [{'criterion': 'a'}]}}
    for scores in ({}, {'insight': []}, {'insight': [{'criterion': 'a', 'article_1_score': 6}]}):
        with pytest.raises(ValueError):
            p.validate_scores(scores, rubric)


def test_summary_missing_is_not_zero(tmp_path, monkeypatch):
    entries = []
    for q in ('52', '54'):
        for method in ('threshold', 'shortlist', 'reinforce'):
            alias = q + method
            (tmp_path / alias).mkdir()
            entries.append({'alias': alias, 'query_id': q, 'method': method, 'context_tokens': 100})
    (tmp_path / 'judging').mkdir()
    for e in entries:
        done = e['query_id'] == '52'
        p.write_new(tmp_path / e['alias'] / 'synthesis.json', {'status': 'completed',
            'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}})
        p.write_new(tmp_path / 'judging' / (e['alias'] + '-score.json'),
            {'status': 'completed' if done else 'failed', 'usage': {'total_tokens': 17}})
        p.write_new(tmp_path / 'judging' / (e['alias'] + '-result.json'),
                    {'status': 'completed' if done else 'failed', 'score': {'overall_score': .6} if done else None})
    monkeypatch.setattr(p, 'validate', lambda path: {'entries': entries})
    result = p.summarize(tmp_path / 'manifest.json')
    assert result['api_tokens'] == {'synthesis': 78, 'judge': 102}
    assert result['complete_paired_queries'] == ['52']
    assert all(a['mean_paired_race'] == .6 and a['completed_queries'] == 1 for a in result['aggregate'].values())
    assert all(r['quality'] is None for r in result['rows'] if r['query_id'] == '54')


def test_synthesis_stops_after_failure_and_rejects_restart(tmp_path, monkeypatch):
    from . import synthesis
    manifest = {'entries': [{'alias': 'a'}, {'alias': 'b'}], 'synthesis': {}}
    monkeypatch.setattr(p, 'validate', lambda _: manifest)
    for alias in ('a', 'b'):
        (tmp_path / alias).mkdir()
    monkeypatch.setattr(p, 'load_plan', lambda _: {})
    attempts = []
    async def fail(plan, directory, **kwargs):
        attempts.append(directory.name)
        raise RuntimeError('failure')
    monkeypatch.setattr(synthesis, 'synthesize', fail)
    with pytest.raises(RuntimeError):
        asyncio.run(p.run_synthesis(tmp_path / 'manifest.json', allow_api=True))
    assert attempts == ['a']
    assert p.read(tmp_path / 'a/failure.json')['stage'] == 'synthesis'
    with pytest.raises(FileExistsError):
        asyncio.run(p.run_synthesis(tmp_path / 'manifest.json', allow_api=True))
    assert attempts == ['a']


def test_judge_pipeline_failure_stops_no_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'validate', lambda _: {'entries': [{'alias': 'a', 'query_id': '52'},
        {'alias': 'b', 'query_id': '54'}], 'judge': {'model': 'gpt-5.5'}})
    p.write_new(tmp_path / 'references.json', {'52': {}, '54': {}})
    for alias in ('a', 'b'):
        d = tmp_path / alias
        d.mkdir()
        (d / 'report.md').write_text('report')
        p.write_new(d / 'synthesis.json', {'status': 'completed', 'report_sha256': p.sha(d / 'report.md')})
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(args[2])
        raise ValueError('invalid score')
    monkeypatch.setattr(p, 'score_one', fail)
    with pytest.raises(RuntimeError):
        p.run_judge(tmp_path / 'manifest.json', allow_api=True)
    assert attempts == ['a']
    assert p.read(tmp_path / 'judging/a-result.json') == {'status': 'failed', 'score': None, 'error_type': 'ValueError'}
    assert not (tmp_path / 'judging/b-result.json').exists()


@pytest.mark.parametrize('stage,cap', [('synthesis', 7), ('judge', 13)])
def test_caps_cannot_be_raised(tmp_path, monkeypatch, stage, cap):
    monkeypatch.setattr(p, 'validate', lambda _: {})
    with pytest.raises(ValueError, match='cap'):
        if stage == 'synthesis':
            asyncio.run(p.run_synthesis(tmp_path / 'manifest.json', allow_api=True, max_calls=cap))
        else:
            p.run_judge(tmp_path / 'manifest.json', allow_api=True, max_calls=cap)
