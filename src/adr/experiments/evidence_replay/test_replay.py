"""Offline tests, including real saved evidence and the fork's actual prompt builder."""
import asyncio
import json
import socket
from pathlib import Path

import numpy as np
import pytest

from .core import Page, Pool, Tokenizer, block, build_context, load_pool, make_plan, save_plan, select
from .evaluation import coverage, evaluate
from .synthesis import synthesize

ROOT = Path(__file__).resolve().parents[4]
REAL = ROOT / 'runs/_gpt_researcher_real_source_v2_raw/trajectory_254955f96594.json'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('External networking forbidden in local tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture
def tok():
    return Tokenizer()


@pytest.fixture
def pool():
    return Pool('Question?', 't1', (
        Page('a', 'SELECTED_ONLY facts 日本語 <|endoftext|>', 'https://a.test', 'private subquery', 1, 1, .5),
        Page('b', 'UNSELECTED_SECRET ' * 10, 'https://b.test', 'hidden query', 2, 2, .9),
        Page('c', 'third source facts', 'https://c.test', 'other', 2, 3, .2),
    ), {'fixture': True})


def test_exact_budget_includes_formatting(pool, tok):
    text = block(pool.pages[0])
    n = tok.count(text)
    assert n > tok.count(pool.pages[0].text)
    assert build_context(pool, ['a'], tok, n) == (text, ['a'], n)
    with pytest.raises(ValueError, match='exceeds budget'):
        build_context(pool, ['a'], tok, n - 1)
    context, ids, tokens = select(pool, 'topk', tok, 1)
    assert (context, ids, tokens) == ('', [], 0)
    with pytest.raises(ValueError, match='nonnegative'):
        select(pool, 'random', tok, -1)


def test_determinism_and_canonical_order(pool, tok):
    for method in ['random', 'threshold', 'topk']:
        assert select(pool, method, tok, 100, 17) == select(pool, method, tok, 100, 17)
    assert build_context(pool, ['c', 'a'], tok, 1000) == build_context(pool, ['a', 'c'], tok, 1000)
    with pytest.raises(ValueError, match='Unknown selected'):
        build_context(pool, ['missing'], tok, 1000)


def test_threshold_overflow_differs_from_topk(pool, tok):
    budget = max(tok.count(block(pool.pages[0])), tok.count(block(pool.pages[1])))
    assert select(pool, 'threshold', tok, budget, threshold=.35)[1] == ['a']
    assert select(pool, 'topk', tok, budget)[1] == ['b']
    assert select(pool, 'threshold', tok, 1000, threshold=1)[1] == []


def test_no_leakage_and_empty(pool, tok):
    context, ids, n = build_context(pool, ['a'], tok, 1000)
    assert 'SELECTED_ONLY' in context and 'https://a.test' in context
    for forbidden in ['UNSELECTED_SECRET', 'b.test', 'private subquery', 'hidden query', 'third source']:
        assert forbidden not in context
    assert build_context(pool, [], tok, 0) == ('', [], 0)


def test_save_artifacts_no_overwrite(pool, tok, tmp_path):
    plan = make_plan(pool, tok, selected_ids=['a'])
    out = tmp_path / 'replay'
    save_plan(plan, out)
    saved = json.loads((out / 'plan.json').read_text())
    assert saved['mask'] == {'a': True, 'b': False, 'c': False}
    assert saved['context_tokens'] == tok.count((out / 'context.txt').read_text())
    assert 'UNSELECTED_SECRET' not in (out / 'plan.json').read_text()
    with pytest.raises(FileExistsError):
        save_plan(plan, out)


def test_loader_requires_complete_embedding_join(tmp_path):
    p = tmp_path / 't.json'
    p.write_text(json.dumps({'query': 'q', 'query_id': 't', 'report': 'done', 'evidence': {
        'a': {'item_id': 'a', 'content': 'facts', 'source_url': 'https://a',
              'source_subquery': 's', 'tree_depth': 1, 'retrieval_round': 1, 'source_type': 'page'}}}))
    np.savez(tmp_path / 't_emb.npz', ids=np.array(['wrong']), vectors=np.array([[1., 0.]]), query=np.array([[1., 0.]]))
    with pytest.raises(ValueError, match='ID mismatch'):
        load_pool(p)
    np.savez(tmp_path / 't_emb.npz', ids=np.array(['a']), vectors=np.array([[1., 0.]]), query=np.array([[1., 0.]]))
    assert load_pool(p).pages[0].similarity == 1


def test_real_completed_pool(tok, tmp_path):
    pool = load_pool(REAL)
    assert len(pool.pages) == 327
    plan = make_plan(pool, tok, budget=50000)
    assert plan['selected_ids'] and 0 < plan['context_tokens'] <= 50000
    assert plan['context_tokens'] == tok.count(plan['context'])
    assert 'final_context' not in plan and 'report' not in plan
    assert plan['provenance']['embedding_dimensions'] == 1536
    # Exercise the real context that exposed the carriage-return smoke failure.
    assert '\r' in plan['context']
    out = tmp_path / 'real-replay'
    save_plan(plan, out)
    async def transport(messages, *args):
        assert plan['context'] in messages[1]['content']
        return 'mock real-pool report', {'total_tokens': 1}
    state = asyncio.run(synthesize(plan, out, allow_api=True, transport=transport,
                                   fork_path=ROOT / 'third_party/gpt-researcher'))
    assert state['status'] == 'completed'


def test_empty_and_api_opt_in(pool, tok, tmp_path):
    empty = make_plan(pool, tok, selected_ids=[])
    out = tmp_path / 'empty'; save_plan(empty, out)
    with pytest.raises(PermissionError):
        asyncio.run(synthesize(empty, out))
    with pytest.raises(ValueError, match='Empty selection'):
        asyncio.run(synthesize(empty, out, allow_api=True))
    assert not (out / 'synthesis.json').exists()


def test_real_fork_synthesis_mock_transport(pool, tok, tmp_path):
    plan = make_plan(pool, tok, selected_ids=['a'])
    out = tmp_path / 'live-mock'; save_plan(plan, out)
    calls = []
    async def transport(messages, model, allowance):
        calls.append(messages)
        prompt = '\n'.join(m['content'] for m in messages)
        assert plan['context'] in prompt
        assert 'January 01, 1970' in prompt
        assert 'SELECTED_ONLY' in prompt
        assert all(x not in prompt for x in ['UNSELECTED_SECRET', 'private subquery', 'hidden query'])
        return 'Mock report [source](https://a.test)', {'prompt_tokens': 123, 'completion_tokens': 10, 'total_tokens': 133}
    state = asyncio.run(synthesize(plan, out, allow_api=True, transport=transport,
                                   fork_path=ROOT / 'third_party/gpt-researcher'))
    assert len(calls) == state['api_attempts'] == 1
    assert state['status'] == 'completed' and state['latency_seconds'] >= 0
    assert state['usage']['total_tokens'] == 133
    assert (out / 'request.json').exists() and (out / 'report.md').exists()
    with pytest.raises(FileExistsError):
        asyncio.run(synthesize(plan, out, allow_api=True, transport=transport))


def test_failed_transport_never_retries(pool, tok, tmp_path):
    plan = make_plan(pool, tok, selected_ids=['a']);out = tmp_path / 'failure';save_plan(plan, out)
    calls = []
    async def fail(*args):
        calls.append(True)
        raise RuntimeError('Simulated provider failure')
    with pytest.raises(RuntimeError, match='empty report'):
        asyncio.run(synthesize(plan, out, allow_api=True, transport=fail,
                               fork_path=ROOT / 'third_party/gpt-researcher'))
    assert len(calls) == 1
    assert json.loads((out / 'synthesis.json').read_text())['status'] == 'failed'
    assert not (out / 'report.md').exists()


def test_tampered_context_refused(pool, tok, tmp_path):
    plan = make_plan(pool, tok, selected_ids=['a']);out = tmp_path / 'tamper';save_plan(plan, out)
    (out / 'context.txt').write_text('leaked summary')
    with pytest.raises(ValueError, match='integrity'):
        asyncio.run(synthesize(plan, out, allow_api=True))


def test_evaluation_refuses_development_question():
    with pytest.raises(ValueError, match='prompt match'):
        coverage(load_pool(REAL).query, ROOT / 'third_party/deep_research_bench')


def test_reference_and_rubric_coverage():
    bench = ROOT / 'third_party/deep_research_bench'
    question = next(json.loads(s)['prompt'] for s in (bench / 'data/prompt_data/query.jsonl').read_text().splitlines() if json.loads(s)['id'] == 51)
    assert coverage(question, bench) == '51'


def test_no_hidden_word_trimming(tok, tmp_path):
    pool = Pool('question', 'large', (Page('long', 'word ' * 26000, 'https://a.test', 's', 1, 1, 1.),), {})
    plan = make_plan(pool, tok, selected_ids=['long'])
    out = tmp_path / 'large';save_plan(plan, out)
    async def transport(messages, *args):
        assert plan['context'] in messages[1]['content']
        assert messages[1]['content'].count('word ') == 26000
        return 'mock', {'total_tokens': 1}
    asyncio.run(synthesize(plan, out, allow_api=True, transport=transport,
                           fork_path=ROOT / 'third_party/gpt-researcher'))


def test_real_pool_context_ignores_saved_summaries(tok, tmp_path):
    raw = json.loads(REAL.read_text())
    raw['final_context'] = 'FORBIDDEN_FINAL_SUMMARY'
    raw['report'] = 'FORBIDDEN_PREVIOUS_REPORT'
    raw['learnings'] = ['FORBIDDEN_LEARNINGS']
    source = tmp_path / 'source.json';source.write_text(json.dumps(raw))
    import shutil
    shutil.copyfile(REAL.with_name(REAL.stem + '_emb.npz'), tmp_path / 'source_emb.npz')
    loaded = load_pool(source)
    plan = make_plan(loaded, tok, selected_ids=[loaded.pages[0].id])
    assert 'FORBIDDEN_' not in plan['context']


def test_missing_criteria_and_evaluation_optin(tmp_path):
    bench = tmp_path / 'bench'
    files = {'data/prompt_data/query.jsonl': {'id': 51, 'prompt': 'question'},
             'data/test_data/cleaned_data/reference.jsonl': {'prompt': 'question', 'article': 'reference'},
             'data/criteria_data/criteria.jsonl': {'prompt': 'question', 'criterions': {}}}
    for name, row in files.items():
        p = bench / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='criteria'):
        coverage('question', bench)
    (bench / 'data/criteria_data/criteria.jsonl').write_text(json.dumps({'prompt': 'question', 'criterions': {'insight': [{}]}}))
    replay = tmp_path / 'replay';replay.mkdir()
    (replay / 'plan.json').write_text(json.dumps({'query': 'question'}))
    (replay / 'report.md').write_text('report')
    from .core import sha
    (replay / 'synthesis.json').write_text(json.dumps({'status': 'completed', 'report_sha256': sha(replay / 'report.md')}))
    with pytest.raises(PermissionError):
        evaluate(replay, tmp_path / 'evaluation', bench)
    assert not (tmp_path / 'evaluation').exists()


def test_selectors_skip_oversized_pages_and_continue(tok):
    pool = Pool('q', 't', (
        Page('big', 'oversized ' * 1000, 'https://big', 's', 1, 1, .9),
        Page('small', 'fits', 'https://small', 's', 1, 2, .8),
    ), {})
    budget = tok.count(block(pool.pages[1]))
    for method in ['random', 'threshold', 'topk']:
        assert select(pool, method, tok, budget)[1] == ['small']


def test_mixed_newlines_preserved_through_synthesis(tok, tmp_path):
    text = 'bare CR\rnext\r\nCRLF then LF\nUnicode 日本語\rfinal'
    pool = Pool('newline question', 'mixed', (
        Page('mixed', text, 'https://mixed.test', 's', 1, 1, 1.),
    ), {})
    plan = make_plan(pool, tok, selected_ids=['mixed'])
    out = tmp_path / 'mixed-newlines'
    save_plan(plan, out)
    saved_bytes = (out / 'context.txt').read_bytes()
    assert saved_bytes == plan['context'].encode('utf-8')
    assert text.encode('utf-8') in saved_bytes
    assert tok.count(saved_bytes.decode('utf-8')) == plan['context_tokens']
    from .core import sha
    assert sha(out / 'context.txt') == plan['context_sha256']
    calls = []
    async def transport(messages, *args):
        calls.append(messages)
        assert plan['context'] in messages[1]['content']
        assert text in messages[1]['content']
        return 'mock mixed-newline report', {'total_tokens': 1}
    state = asyncio.run(synthesize(plan, out, allow_api=True, transport=transport,
                                   fork_path=ROOT / 'third_party/gpt-researcher'))
    assert state['status'] == 'completed' and len(calls) == 1
    # JSON escaping must also round-trip the exact context in the request audit.
    request = json.loads((out / 'request.json').read_bytes())
    assert plan['context'] in request['messages'][1]['content']


def test_newline_normalization_is_detected_as_tampering(tok, tmp_path):
    pool = Pool('q', 'mixed', (
        Page('mixed', 'CR\rCRLF\r\nLF\nend', 'https://mixed.test', 's', 1, 1, 1.),
    ), {})
    plan = make_plan(pool, tok, selected_ids=['mixed'])
    out = tmp_path / 'normalized-tamper'
    save_plan(plan, out)
    normalized = plan['context'].replace('\r\n', '\n').replace('\r', '\n')
    (out / 'context.txt').write_bytes(normalized.encode('utf-8'))
    with pytest.raises(ValueError, match='integrity'):
        asyncio.run(synthesize(plan, out, allow_api=True))
    assert not (out / 'synthesis.json').exists()


@pytest.mark.parametrize('finish,text,refusal,reason', [
    ('stop', 'A finished report', None, None),
    ('length', None, None, 'completion_budget_exhausted'),
    ('length', 'An unfinished report', None, 'completion_budget_exhausted'),
    ('stop', None, None, 'empty_report'),
    ('stop', None, 'refusal', 'refusal_or_content_filter'),
    ('content_filter', None, None, 'refusal_or_content_filter'),
    ('tool_calls', None, None, 'incomplete_or_unsupported_response'),
])
def test_sdk_response_diagnostics_and_one_request(pool, tok, tmp_path, monkeypatch,
                                                  finish, text, refusal, reason):
    from openai.types.chat import ChatCompletion
    from types import SimpleNamespace
    import openai
    from .synthesis import SynthesisFailure
    response = ChatCompletion.model_validate({
        'id': 'chatcmpl-offline', 'object': 'chat.completion', 'created': 0,
        'model': 'gpt-5-mini', 'choices': [{'index': 0, 'finish_reason': finish,
            'message': {'role': 'assistant', 'content': text, 'refusal': refusal}}],
        'usage': {'prompt_tokens': 50483, 'completion_tokens': 8192, 'total_tokens': 58675,
                  'completion_tokens_details': {'reasoning_tokens': 8192}},
    })
    calls = []
    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def create(self, **kwargs):
            calls.append(kwargs)
            return response
        async def close(self):
            pass
    monkeypatch.setattr(openai, 'AsyncOpenAI', FakeClient)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-only')
    plan = make_plan(pool, tok, selected_ids=['a'])
    out = tmp_path / 'sdk';save_plan(plan, out)
    action = synthesize(plan, out, allow_api=True, reasoning_effort='low',
                        max_completion_tokens=16384, fork_path=ROOT / 'third_party/gpt-researcher')
    if reason:
        with pytest.raises(SynthesisFailure, match=reason):
            asyncio.run(action)
    else:
        assert asyncio.run(action)['status'] == 'completed'
    assert len(calls) == 1
    assert calls[0]['reasoning_effort'] == 'low' and calls[0]['max_completion_tokens'] == 16384
    assert 'max_tokens' not in calls[0] and 'temperature' not in calls[0]
    state = json.loads((out / 'synthesis.json').read_bytes())
    assert state['response']['finish_reason'] == finish
    assert state['usage']['completion_tokens_details']['reasoning_tokens'] == 8192
    assert json.loads((out / 'response.json').read_bytes())['failure_reason'] == reason
    assert (out / 'report.md').exists() == (reason is None)
    assert (out / 'partial_report.md').exists() == (finish == 'length' and bool(text))
    assert 'offline-test-only' not in (out / 'synthesis.json').read_text()


def test_sdk_transport_error_is_preserved_without_secret_message(pool, tok, tmp_path, monkeypatch):
    import openai
    from types import SimpleNamespace
    from .synthesis import SynthesisFailure
    calls = []
    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        async def create(self, **kwargs):
            calls.append(True)
            exc = RuntimeError('sensitive provider message and credentials must not be logged')
            exc.status_code = 429
            exc.request_id = 'req-offline'
            raise exc
        async def close(self):
            pass
    monkeypatch.setattr(openai, 'AsyncOpenAI', FakeClient)
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-test-only')
    plan = make_plan(pool, tok, selected_ids=['a']);out = tmp_path / 'error';save_plan(plan, out)
    with pytest.raises(SynthesisFailure, match='transport_error'):
        asyncio.run(synthesize(plan, out, allow_api=True, fork_path=ROOT / 'third_party/gpt-researcher'))
    assert len(calls) == 1
    state = json.loads((out / 'synthesis.json').read_bytes())
    assert state['response']['http_status'] == 429 and state['response']['request_id'] == 'req-offline'
    assert state['usage'] is None
    assert 'sensitive provider message' not in (out / 'synthesis.json').read_text()


def test_response_without_choices_is_classified():
    from openai.types.chat import ChatCompletion
    from .synthesis import parse_response
    response = ChatCompletion.model_validate({'id': 'offline', 'object': 'chat.completion',
        'model': 'gpt-5-mini', 'created': 0, 'choices': []})
    state = {}
    text, error = parse_response(response, state)
    assert text == '' and error == 'incomplete_or_unsupported_response'
    assert state['response']['choice_count'] == 0
