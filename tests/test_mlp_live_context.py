from adr.mlp.embedding_limits import token_count, MAX_INPUT_TOKENS
"""Offline frozen-corpus identity and live-only compatibility checks."""
import json
from pathlib import Path

import pytest

from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot, embedding_context
from adr.synthetic.schema import BudgetState, ResearchSnapshot, EvidenceState, derive_features, serialize_context


def test_all_400_frozen_contexts_identical():
    rows = [json.loads(l) for l in Path('data/synthetic/v1/cleaned-20260923T021710222070Z-r1/examples.jsonl').read_text().splitlines()]
    assert len(rows) == 400
    for row in rows:
        state = ResearchSnapshot.model_validate(row['state'])
        text, info = embedding_context(state)
        assert not info['bounded']
        assert text.encode() == row['context_before'].encode() == serialize_context(state).encode()


@pytest.mark.parametrize('body', ['published evidence ' * 100000, '証拠🌳\\\"\n' * 100000])
def test_large_context_preserves_everything_except_body_suffixes(body):
    state = ResearchSnapshot(evidence=[EvidenceState(id=f'e{i}', url='url-sha256:'+f'{i:064x}', text=body, retained=i%2==0) for i in range(62)],
                             executed_subqueries=['previous query'], branches=[{'id':'b1','goal':'unresolved need','status':'active'}])
    before = serialize_context(state)
    text, info = embedding_context(state)
    assert (text, info) == embedding_context(state)
    assert token_count(text) <= MAX_INPUT_TOKENS and info['bounded']
    assert info['original_bytes'] == len(before.encode())
    assert info['final_bytes'] == len(text.encode())
    from adr.mlp.compact_metadata import decode_context
    actual = decode_context(text)
    for old, new in zip(state.model_dump()['evidence'], actual['evidence']):
        assert old['text'].startswith(new['text']) and new['text']
        new['text'] = old['text']
    assert actual == state.model_dump()
    assert serialize_context(state) == before


def test_metadata_overflow_fails_without_dropping_history():
    with pytest.raises(ValueError, match='metadata alone'):
        embedding_context(ResearchSnapshot(executed_subqueries=['prior ' * 10000]))


def test_live_budget_overrun_retains_actuals_and_feature_semantics():
    fields = dict(max_steps=32,max_searches=64,max_reads=128,max_tokens=400000,
                  used_steps=40,used_searches=112,used_reads=275,used_tokens=500000)
    with pytest.raises(ValueError, match='exceeds maximum'):
        BudgetState(**fields)
    state = UnenforcedLiveSnapshot(budget=UnenforcedLiveBudget(**fields))
    assert json.loads(serialize_context(state))['budget'] == fields
    features = derive_features('question', state)
    values = dict(zip(features.names, features.values))
    assert values['step_frac'] == values['token_frac'] == 1
    assert all(values['remaining_'+r] == 0 for r in ['steps','searches','reads','tokens'])
    with pytest.raises(ValueError):
        ResearchSnapshot.model_validate(state.model_dump())
    with pytest.raises(ValueError):
        UnenforcedLiveBudget(used_tokens=-1)


def test_above_old_byte_bound_passes_unchanged():
    state = ResearchSnapshot(evidence=[EvidenceState(id='e',url='source',text='evidence ' * 2000)])
    original = serialize_context(state)
    assert len(original.encode()) > 8191
    assert token_count(original) < MAX_INPUT_TOKENS
    text, info = embedding_context(state)
    assert text == original and not info['bounded']


def test_metadata_failure_emits_sizes_before_raising():
    records = []
    with pytest.raises(ValueError, match='metadata alone'):
        embedding_context(ResearchSnapshot(executed_subqueries=['prior ' * 10000]),on_diagnostics=records.append)
    assert len(records)==1
    assert records[0]['metadata_tokens'] > MAX_INPUT_TOKENS
    assert records[0]['original_bytes'] > 8191
    assert 'prior' not in json.dumps(records)


def test_tokenizer_limit_special_literals_and_offline(monkeypatch):
    import requests
    monkeypatch.setattr(requests,'get',lambda *a,**k:pytest.fail('Network forbidden'))
    from adr.mlp.embedding_limits import encoding
    encoding.cache_clear()
    assert encoding().name == 'cl100k_base'
    assert token_count('word ' * 8191) == 8192
    assert token_count('word ' * 8192) == 8193
    assert token_count('<|endoftext|>') > 1


def test_runner_preserves_partial_accounting_on_failure(tmp_path,monkeypatch):
    from adr.runner import experiment
    from adr.core.types import Query
    class FailingAgent:
        async def run(self, task, ctx):
            meter=ctx.extra['meter']
            meter.prompt_tokens=60000;meter.completion_tokens=2000;meter.total_tokens=62000
            meter.n_search_calls=22;meter.n_fetch_calls=70
            ctx.extra['partial_research_accounting']={'research_usage':{'cost':.1},'searches':22,'reads':70}
            raise ValueError('offline failure')
    monkeypatch.setattr(experiment,'build_agent',lambda *a:FailingAgent())
    monkeypatch.setattr(experiment,'load_queries',lambda *a,**k:[Query(id='offline',text='offline query',dataset='fixture')])
    manifest=experiment.run_experiment({'output_dir':str(tmp_path),'run_name':'offline-failure',
        'dataset':{'name':'fixture'},'agent':{'name':'fixture'},'llm':{'provider':'mock'},
        'search':{'backend':'mock'},'eval':{'official_benches':[]}})
    trajectory=json.loads((manifest.run_dir/'trajectories/offline.json').read_text())
    assert trajectory['report'] is None
    assert trajectory['final_stats']['usage']['total_tokens']==62000
    assert trajectory['final_stats']['partial_research_accounting']['research_usage']['cost']==.1
    metrics=json.loads((manifest.run_dir/'metrics/local.json').read_text())
    assert metrics['total_tokens']==62000 and metrics['mean_n_searches']==22 and metrics['mean_n_reads']==70
