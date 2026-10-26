"""Offline reversible metadata and complete-depth trajectory capacity probes."""
import hashlib
import json
from pathlib import Path

import pytest

from adr.mlp.compact_metadata import decode_context, dumps, encode_metadata
from adr.mlp.embedding_limits import token_count
from adr.mlp.live_context import embedding_context, UnenforcedLiveSnapshot
from adr.synthetic.schema import ResearchSnapshot


def metadata_only(state):
    data = json.loads(dumps(state))
    for e in data['evidence']:
        e['text'] = ''
    return data


def old_encoding(state):
    data = json.loads(dumps(state))
    data['_live_context_bounding'] = 'evidence-table-prefix-v1: text values are prefixes; suffixes may be omitted'
    fields = ['id','url','title','text','retained']
    data['evidence_fields'] = fields
    data['evidence'] = [[e[f] for f in fields] for e in data['evidence']]
    return data


def growth_state(batches):
    # Four initial branches then 12 two-branch rounds: 28 branch batches at full depth 3.
    # Observed run reached 109 evidence / 52 history / 24 branches after 10 batches.
    n = round(10.9*batches)
    queries = [f'Find published Japan regional elderly population projection evidence for prefecture {i}, ages 65 and 75, through 2050.' for i in range(5*batches+2)]
    return {'budget': {'max_steps':32,'max_searches':64,'max_reads':128,'max_tokens':400000,
                      'used_steps':4*batches+batches//2,'used_searches':5*batches+2,
                      'used_reads':round(11.7*batches),'used_tokens':round(16765.6*batches)},
            'executed_subqueries':queries,
            'branches':[{'id':'active-'+hashlib.sha256(str(i).encode()).hexdigest()[:16],
                         'goal':queries[i],'status':['open','active','done','dropped'][i%4]} for i in range(2*batches+4)],
            'evidence':[{'id':hashlib.sha256(f'id{i}'.encode()).hexdigest()[:12],
                         'url':'url-sha256:'+hashlib.sha256(f'source{i}'.encode()).hexdigest(),
                         'title':'','text':'Prior published evidence. '*1000,'retained':i%7!=0} for i in range(n)]}


def test_reversible_literals_and_repeated_strings():
    data=growth_state(10)
    data['executed_subqueries'] += ['0','', '引用🌳\\"\n', '0']
    data['evidence'][0]['title']='0'
    data['evidence'][1]['title']='引用🌳'
    data['evidence'][0]['url']='literal-source'
    original=dumps(data)
    compact=encode_metadata(data)
    assert decode_context(dumps(compact))==data
    assert encode_metadata(data)==compact
    assert dumps(data)==original
    assert compact['strings']  # actual interning, not just positional encoding


@pytest.mark.parametrize('batches',[4,6,8,10,12,14,16,18,20,22,24,26,28,40])
def test_depth_three_growth(batches):
    data=growth_state(batches)
    meta=metadata_only(data)
    old=token_count(dumps(old_encoding(meta)))
    new=token_count(dumps(encode_metadata(meta)))
    assert decode_context(dumps(encode_metadata(meta)))==meta
    assert new<old
    print(f'GROWTH batches={batches} evidence={len(meta["evidence"])} history={len(meta["executed_subqueries"])} branches={len(meta["branches"])} old={old} new={new} body_capacity={8192-new}')
    state=UnenforcedLiveSnapshot.model_validate(data)
    if new>8192:
        with pytest.raises(ValueError,match='metadata alone'):
            embedding_context(state)
    else:
        text,info=embedding_context(state)
        assert token_count(text)<=8192
        decoded=decode_context(text)
        for before,after in zip(data['evidence'],decoded['evidence']):
            assert before['text'].startswith(after['text'])
        assert metadata_only(decoded)==meta


def test_saved_evidence_batches_metadata_roundtrip():
    path=Path('runs/mlp-drb-q051/20260923-064327-mlp-q051-no-pruning/query_gate/031b4af5197ec30a.jsonl')
    records=[json.loads(l) for l in path.read_text().splitlines()]
    for record in records:
        if record['event']!='snapshot' or int(record['batch_id'][1:])<5:
            continue
        meta=metadata_only(decode_context(record['context_before']))
        compact=encode_metadata(meta)
        assert decode_context(dumps(compact))==meta
        old=token_count(dumps(old_encoding(meta))); new=token_count(dumps(compact))
        assert new<old
        print(f'SAVED {record["batch_id"]} old={old} new={new} body_capacity={8192-new}')


def test_b0011_metadata_approximation():
    path=Path('runs/mlp-drb-q051/20260923-064327-mlp-q051-no-pruning/query_gate/031b4af5197ec30a.jsonl')
    records=[json.loads(l) for l in path.read_text().splitlines()]
    saved=next(r for r in records if r['event']=='snapshot' and r['batch_id']=='b0010')
    meta=metadata_only(decode_context(saved['context_before']))
    diagnostic=next(r for r in records if r['event']=='context_diagnostics' and r['batch_id']=='b0011')
    meta['budget']=diagnostic['budget']
    meta['executed_subqueries'] += [r['candidate_search_query'] for r in records if r['event']=='proposed' and r['batch_id']=='b0010']
    meta['executed_subqueries'].append('Unrecorded next planning query: Japan elderly living arrangements by prefecture')
    for i in range(31):
        meta['evidence'].append({'id':hashlib.sha256(f'missing-id{i}'.encode()).hexdigest()[:12],
            'url':'url-sha256:'+hashlib.sha256(f'missing-source{i}'.encode()).hexdigest(),
            'title':'','text':'','retained':True})
    for i in range(3):
        meta['branches'].append({'id':f'unrecorded-branch-{i}','goal':meta['executed_subqueries'][-i-1],'status':'active'})
    assert (len(meta['evidence']),len(meta['executed_subqueries']),len(meta['branches']))==(109,52,24)
    compact=encode_metadata(meta)
    assert decode_context(dumps(compact))==meta
    old=token_count(dumps(old_encoding(meta))); new=token_count(dumps(compact))
    print(f'APPROX b0011 (missing identities/goals substituted): old={old} new={new} body_capacity={8192-new}')
    assert new<old
