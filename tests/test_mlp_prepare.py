import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
from openai import AsyncOpenAI
import pytest

from adr.llm.openai_compat import OpenAICompatLLM
from adr.mlp.prepare import (Config, EmbeddingCache, TEXT_FIELDS, assemble, embedding_identity,
    fill_cache, fit_normalizer, load_prepared, model_parts, normalize, prepare, split_trajectories,
    contract_differences)
from adr.synthetic.schema import FEATURE_NAMES, ResearchSnapshot, derive_features, serialize_context


def row(tid="t1", question="Stellar formation mechanisms", state=0, candidate=0):
    snapshot = ResearchSnapshot.model_validate({
        "evidence": [{"id": f"e{i}", "url": f"https://example.org/{tid}/{i}",
                      "text": f"Prior evidence {i} for {question}"} for i in range(state + 1)],
        "budget": {"used_steps": state, "used_tokens": state * 100},
    })
    label = "EXECUTE" if candidate == 1 or (candidate == 0 and state == 0) else "SKIP"
    return {"example_id": f"{tid}-s{state}-c{candidate}", "question_id": tid,
            "trajectory_id": tid, "state_id": f"s{state}", "candidate_id": f"c{candidate}",
            "original_question": question, "context_before": serialize_context(snapshot),
            "candidate_subquery": f"Retrieve study {candidate} on {question}",
            "state": snapshot.model_dump(),
            "numerical_features": derive_features(question, snapshot).model_dump(),
            "label": label, "decision_category": "unanswered_information_need" if label == "EXECUTE" else "low_marginal_value",
            "reason": "Metadata only; never embed this reason.",
            "provenance": {"schema_version": "synthetic-v1", "prompt_version": "fixture",
                           "dataset_version": "fixture", "generated_at": "2026-01-01",
                           "teacher_provider": "fixture", "teacher_model": "fixture",
                           "state_index": state, "counterfactual_pair_id": "p1" if candidate == 0 and state in [0, 4] else None}}


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://azure.invalid/openai/v1/')
    topics = ["Stellar formation mechanisms", "Aquatic microbial ecology", "Ceramic glazing history",
              "Railway signal reliability", "Vineyard frost prevention", "Library digital accessibility",
              "Volcanic magma viscosity", "Pediatric sleep patterns", "Turbine blade corrosion",
              "Agrarian land tenure", "Glacial meltwater chemistry", "Quantum photon coherence",
              "Museum textile conservation", "Ocean current modeling", "Battery electrode recycling",
              "Linguistic vowel acquisition", "Forest fungal networks", "Urban housing affordability",
              "Protein folding dynamics", "Bridge seismic isolation"]
    rows = [row(f"t{i:02d}", topic, s, c) for i, topic in enumerate(topics)
            for s in range(5) for c in range(4)]
    source = tmp_path / "source" / "examples.jsonl"
    source.parent.mkdir()
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"questions": [{"id": "external", "text": "Medieval astronomy manuscripts"}]}))
    return rows, Config(source=source, heldout=heldout, output=tmp_path / "prepared", embedding_dimensions=3)


def test_deterministic_split_no_overlap_and_pairs(corpus):
    rows, _ = corpus
    split = split_trajectories(rows, 42, 4)
    assert split == split_trajectories(list(reversed(rows)), 42, 4)
    assert len(split['train']) == 16 and len(split['validation']) == 4
    assert not set(split['train']) & set(split['validation'])
    assert set(split['train']) | set(split['validation']) == {r['trajectory_id'] for r in rows}
    for tid in split['train'] + split['validation']:
        members = [r for r in rows if r['trajectory_id'] == tid]
        assert len(members) == 20
        pair = [r for r in members if r['provenance']['counterfactual_pair_id']]
        assert [r['label'] for r in pair] == ['EXECUTE', 'SKIP']
    changed = copy.deepcopy(rows)
    for r in changed:
        r['label'] = 'SKIP'
        r['decision_category'] = 'different'
    assert split == split_trajectories(changed, 42, 4)


def test_question_cannot_cross_trajectories():
    a, b = row('t1'), row('t2')
    b['question_id'] = a['question_id']
    with pytest.raises(ValueError, match='Question crosses'):
        split_trajectories([a, b], 42, 1)


def test_projection_labels_order_and_forbidden_metadata():
    a = row()
    text, values, target = model_parts(a)
    assert text == tuple(a[k] for k in TEXT_FIELDS)
    assert len(values) == 15 and target == 1
    for field in ['reason', 'decision_category', 'example_id', 'question_id', 'trajectory_id', 'state_id', 'candidate_id']:
        a[field] = 'FORBIDDEN_SENTINEL'
    a['provenance'] = {'counterfactual_pair_id': 'FORBIDDEN_SENTINEL'}
    assert model_parts(a) == (text, values, target)
    a['label'] = 'SKIP'
    assert model_parts(a)[2] == 0
    a['numerical_features']['names'].reverse()
    with pytest.raises(ValueError, match='order'):
        model_parts(a)
    a['numerical_features']['names'] = list(FEATURE_NAMES)
    a['numerical_features']['values'] = [0] * 14
    with pytest.raises(ValueError, match='vector'):
        model_parts(a)


def test_train_only_normalization_and_constant_features():
    train = np.tile(np.arange(4)[:, None], (1, 15)).astype(float)
    train[:, 5] = 0
    stats = fit_normalizer(train)
    validation = np.full((1, 15), 100.0)
    validation[:, 5] = 0
    transformed = normalize(validation, stats)
    assert stats['mean'][0] == 1.5 and stats['n_training_rows'] == 4
    assert transformed[0, 0] == pytest.approx((100 - 1.5) / np.std(np.arange(4)))
    assert stats['scale'][5] == 1 and stats['zero_variance'][5]
    assert transformed[0, 5] == 0 and np.isfinite(transformed).all()
    assert np.allclose(normalize(train, stats).mean(axis=0), 0, atol=1e-7)


@pytest.mark.asyncio
async def test_cache_dedup_resume_and_bad_vectors(tmp_path):
    identity = {'dimensions': 3, 'model': 'fake'}
    cache = EmbeddingCache(tmp_path/'cache.sqlite', identity)
    embed = AsyncMock(side_effect=[[[1,2,3]], RuntimeError('interrupted')])
    with pytest.raises(RuntimeError):
        await fill_cache(['a', 'a', 'b'], cache, 1, embed)
    assert cache.get('a') is not None and cache.get('b') is None
    cache.close()
    cache = EmbeddingCache(tmp_path/'cache.sqlite', identity)
    resumed = AsyncMock(return_value=[[4,5,6]])
    await fill_cache(['a','b','a'], cache, 2, resumed)
    resumed.assert_awaited_once_with(['b'], dimensions=3)
    await fill_cache(['a','b'], cache, 2, resumed)
    assert resumed.await_count == 1
    for bad in [[[1]], [[float('nan'), 1, 2]], []]:
        with pytest.raises(ValueError):
            cache.put_batch(['bad'], bad)
    assert cache.get('bad') is None
    cache.close()
    with pytest.raises(ValueError, match='mismatch'):
        EmbeddingCache(tmp_path/'cache.sqlite', {'dimensions': 3, 'model': 'other'})


@pytest.mark.asyncio
async def test_offline_then_frozen_inputs_and_contract(corpus):
    rows, cfg = corpus
    def forbidden(**kwargs):
        raise AssertionError('Offline must not create a client')
    report = await prepare(cfg, client_factory=forbidden)
    assert report['status'] == 'awaiting_embeddings'
    assert report['unique_strings'] == {'original_question':20, 'context_before':100, 'candidate_subquery':80}
    assert report['missing_total'] == 200
    assert not (cfg.output/'train.npz').exists()
    with pytest.raises(ValueError, match='not ready'):
        load_prepared(cfg.output, 'train')
    split = json.loads((cfg.output/'split.json').read_text())
    training = np.array([model_parts(r)[1] for r in rows if r['trajectory_id'] in split['train']])
    assert json.loads((cfg.output/'normalization.json').read_text()) == fit_normalizer(training)
    fake = SimpleNamespace(embed=AsyncMock(side_effect=lambda texts, dimensions: _vectors(texts)), close=AsyncMock())
    report = await prepare(cfg, allow_api=True, client_factory=lambda **kw: fake)
    assert report['status'] == 'ready' and report['missing_total'] == 0
    sent = [text for call in fake.embed.call_args_list for text in call.args[0]]
    assert len(sent) == len(set(sent)) == 200
    assert set(sent) == {r[f] for r in rows for f in TEXT_FIELDS}
    X, y, metadata = load_prepared(cfg.output, 'train')
    assert X.shape == (320, 24) and y.shape == (320,)
    for idx, meta in enumerate(metadata):
        original = next(r for r in rows if r['example_id'] == meta['example_id'])
        assert y[idx] == (original['label']=='EXECUTE')
        assert np.array_equal(X[idx,:9], np.concatenate(_vectors([original[f] for f in TEXT_FIELDS])))
    before = X.copy()
    await prepare(cfg, allow_api=True, client_factory=forbidden)
    assert np.array_equal(load_prepared(cfg.output,'train')[0], before)
    with pytest.raises(ValueError, match='contract changed'):
        await prepare(cfg.model_copy(update={'seed': 43}))
    (cfg.output/'train_metadata.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        load_prepared(cfg.output, 'train')


def _vectors(texts):
    return [[len(t), sum(map(ord,t)) % 101, 7] for t in texts]


@pytest.mark.asyncio
async def test_invalid_input_rejected_before_client(corpus):
    rows, cfg = corpus
    rows[0]['numerical_features']['values'][0] = 999
    cfg.source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='validated'):
        await prepare(cfg, allow_api=True, client_factory=lambda **kw: pytest.fail('client created'))


@pytest.mark.asyncio
async def test_azure_v1_embedding_route_and_exact_payload(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://azure.invalid/openai/v1/')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-not-a-secret')
    client = OpenAICompatLLM(model='text-embedding-3-small')
    assert str(client._client.base_url) == 'https://azure.invalid/openai/v1/'
    await client.close()
    def handler(request):
        assert request.url.path == '/openai/v1/embeddings'
        payload = json.loads(request.content)
        assert payload == {'input':['exact\ntext','second'], 'model':'text-embedding-3-small',
                           'dimensions':3, 'encoding_format':'float'}
        return httpx.Response(200, json={'object':'list','model':'text-embedding-3-small',
            'data':[{'object':'embedding','index':1,'embedding':[4,5,6]},
                    {'object':'embedding','index':0,'embedding':[1,2,3]}],
            'usage':{'prompt_tokens':5,'total_tokens':5}})
    client._client = AsyncOpenAI(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await client.embed(['exact\ntext','second'], dimensions=3) == [[1,2,3],[4,5,6]]
    await client.close()
    identity = embedding_identity('text-embedding-3-small', 1536)
    assert 'azure.invalid' not in json.dumps(identity) and 'test-not-a-secret' not in json.dumps(identity)


@pytest.mark.asyncio
async def test_unbound_offline_plan_then_explicit_endpoint(corpus, monkeypatch):
    _, cfg = corpus
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    forbidden = lambda **kwargs: pytest.fail('Unexpected client construction')
    report = await prepare(cfg, client_factory=forbidden)
    assert report['status'] == 'awaiting_endpoint'
    assert 'unspecified' in report['endpoint_notice']
    assert not (cfg.output/'embeddings.sqlite3').exists()
    contract = json.loads((cfg.output/'contract.json').read_text())
    assert contract['embedding']['endpoint_sha256'] is None
    before = {p.name:p.read_bytes() for p in cfg.output.iterdir()}
    with pytest.raises(ValueError, match='OPENAI_BASE_URL must be explicitly set'):
        await prepare(cfg, allow_api=True, client_factory=forbidden)
    assert before == {p.name:p.read_bytes() for p in cfg.output.iterdir()}
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://azure.invalid/openai/v1/')
    report = await prepare(cfg, client_factory=forbidden)
    assert report['status'] == 'awaiting_embeddings'
    bound = json.loads((cfg.output/'contract.json').read_text())
    assert contract_differences(contract, bound) == ['embedding.endpoint_sha256']
    assert before['split.json'] == (cfg.output/'split.json').read_bytes()
    assert before['normalization.json'] == (cfg.output/'normalization.json').read_bytes()
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://different.invalid/openai/v1/')
    with pytest.raises(ValueError, match=r'embedding.endpoint_sha256') as error:
        await prepare(cfg, client_factory=forbidden)
    assert 'different.invalid' not in str(error.value)
    assert json.loads((cfg.output/'contract.json').read_text()) == bound


@pytest.mark.asyncio
async def test_pending_binding_does_not_allow_other_contract_changes(corpus, monkeypatch):
    _, cfg = corpus
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    await prepare(cfg)
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://azure.invalid/openai/v1/')
    with pytest.raises(ValueError) as error:
        await prepare(cfg.model_copy(update={'seed':43, 'model':'other'}))
    for field in ['embedding.endpoint_sha256', 'embedding.model', 'split.seed']:
        assert field in str(error.value)
    assert not (cfg.output/'embeddings.sqlite3').exists()


def test_mismatch_diagnostics_paths_only():
    old = {'source_sha256':'old', 'embedding':{'endpoint_sha256':'secret-a', 'dimensions':1536}}
    new = {'source_sha256':'new', 'embedding':{'endpoint_sha256':'secret-b', 'dimensions':1536}}
    assert contract_differences(old, new) == ['embedding.endpoint_sha256', 'source_sha256']


@pytest.mark.asyncio
async def test_pending_cannot_bind_over_existing_cache(corpus, monkeypatch):
    _, cfg = corpus
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    await prepare(cfg)
    (cfg.output/'embeddings.sqlite3').write_bytes(b'preexisting-cache')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://azure.invalid/openai/v1/')
    with pytest.raises(ValueError, match='embedding.endpoint_sha256'):
        await prepare(cfg)
    assert (cfg.output/'embeddings.sqlite3').read_bytes() == b'preexisting-cache'


def api_failure(status, body):
    from openai import APIStatusError
    request = httpx.Request('POST', 'https://private.invalid/openai/v1/embeddings?sig=url-secret',
                            headers={'Authorization':'Bearer header-secret', 'api-key':'header-key'})
    response = httpx.Response(status, request=request, json=body)
    return APIStatusError('RAW_UNSAFE_EXCEPTION_TEXT', response=response, body=body)


@pytest.mark.parametrize('status,code,message', [
    (404, 'DeploymentNotFound', 'The embedding deployment does not exist.'),
    (400, 'BadRequest', 'Unsupported parameter: dimensions.'),
    (401, 'Unauthorized', 'Access denied: invalid subscription key.'),
    (429, 'RateLimitReached', 'Quota exceeded; retry later.'),
])
def test_useful_api_error_allowlist(status, code, message):
    from adr.mlp.prepare import safe_embedding_error
    error = api_failure(status, {'error':{'code':code,'message':message,'param':'dimensions',
                                         'debug':'NEVER_RENDER_DEBUG'}})
    safe = safe_embedding_error(error)
    assert f'HTTP {status}' in safe and code in safe and message in safe
    assert 'param=dimensions' in safe
    for forbidden in ['RAW_UNSAFE_EXCEPTION_TEXT','NEVER_RENDER_DEBUG','header-secret','header-key','private.invalid','url-secret']:
        assert forbidden not in safe


def test_api_message_redacts_secrets_and_input(monkeypatch):
    from adr.mlp.prepare import safe_embedding_error
    monkeypatch.setenv('OPENAI_API_KEY', 'environment-secret')
    body={'message': 'Deployment missing. environment-secret header-secret header-key '
          'https://private.invalid/path?sig=url-secret api_key=unknown-key '
          'password="unknown password" Bearer unknown-bearer sk-live-unknown '
          'eyJabc.def.ghi EXACT_INPUT_TEXT', 'code':'DeploymentNotFound'}
    safe=safe_embedding_error(api_failure(404,body), ['EXACT_INPUT_TEXT'])
    assert 'Deployment missing' in safe and 'DeploymentNotFound' in safe
    for secret in ['environment-secret','header-secret','header-key','url-secret','unknown-key',
                   'unknown password','unknown-bearer','sk-live-unknown','eyJabc.def.ghi','EXACT_INPUT_TEXT']:
        assert secret not in safe
    assert '[REDACTED]' in safe


def test_unstructured_and_non_api_errors_omit_raw_messages():
    from adr.mlp.prepare import safe_embedding_error
    assert 'HTTP 502' in safe_embedding_error(api_failure(502, '<html>secret</html>'))
    assert 'secret' not in safe_embedding_error(api_failure(502, '<html>secret</html>'))
    assert safe_embedding_error(RuntimeError('password secret')) == 'RuntimeError'
    assert 'count mismatch' in safe_embedding_error(ValueError('Embedding response count mismatch'))


@pytest.mark.asyncio
async def test_safe_batch_error_keeps_committed_vectors_and_resumes(tmp_path):
    from adr.mlp.prepare import EmbeddingBatchError
    cache=EmbeddingCache(tmp_path/'cache.sqlite', {'model':'fake','dimensions':3})
    failure=api_failure(404, {'code':'DeploymentNotFound','message':'Deployment is unavailable.'})
    embed=AsyncMock(side_effect=[[[1,2,3]],failure])
    with pytest.raises(EmbeddingBatchError) as caught:
        await fill_cache(['first','second','third'],cache,1,embed)
    assert 'batch 2/3 (1 strings)' in str(caught.value)
    assert 'HTTP 404' in str(caught.value) and 'DeploymentNotFound' in str(caught.value)
    assert caught.value.__suppress_context__
    first=cache.get('first').copy()
    assert cache.get('second') is None
    cache.close()
    cache=EmbeddingCache(tmp_path/'cache.sqlite', {'model':'fake','dimensions':3})
    retry=AsyncMock(return_value=[[4,5,6],[7,8,9]])
    await fill_cache(['first','second','third'],cache,2,retry)
    retry.assert_awaited_once_with(['second','third'],dimensions=3)
    assert np.array_equal(cache.get('first'),first)
    cache.close()


@pytest.mark.asyncio
async def test_prepare_does_not_hide_batch_diagnostic(corpus):
    _, cfg=corpus
    client=SimpleNamespace(embed=AsyncMock(side_effect=api_failure(400,
        {'error':{'code':'BadRequest','message':'Unsupported parameter: dimensions.'}})), close=AsyncMock())
    with pytest.raises(RuntimeError, match='HTTP 400.*BadRequest.*Unsupported parameter'):
        await prepare(cfg,allow_api=True,client_factory=lambda **kw:client)
    client.close.assert_awaited_once()
