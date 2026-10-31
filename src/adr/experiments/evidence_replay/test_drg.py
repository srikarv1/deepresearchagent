"""DRG port: split loading, group advantages and an end-to-end offline run."""
import json
import socket

import numpy as np
import pytest
import torch

from .drg import MODES, LongConfig, advantages, diagnose, find_pool, load_drg, make_policy, run, spread
from .core import Tokenizer
from .reinforce import Config


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **kw): raise AssertionError('Offline tests prohibit networking')
    monkeypatch.setattr(socket.socket, 'connect', denied); monkeypatch.setattr(socket, 'create_connection', denied)


def write_pool(run_dir, qid, seed, n=12, dim=8):
    """A saved Gym pool in the layout of runs/<run>/gpt_researcher/<id>.json + _emb.npz."""
    rng = np.random.default_rng(seed)
    folder = run_dir / 'gpt_researcher'; folder.mkdir(parents=True, exist_ok=True)
    ids = [f'{seed:02d}{i:010x}' for i in range(n)]
    evidence = {iid: {'item_id': iid, 'content': f'page {i} of {qid} ' + 'fact ' * int(rng.integers(5, 40)),
                      'source_url': f'https://host{i % 5}.test/{i}', 'source_subquery': 's', 'tree_depth': 1 + i % 3,
                      'retrieval_round': 1 + i % 4, 'source_type': 'page', 'compressed': False}
                for i, iid in enumerate(ids)}
    (folder / f'{qid}.json').write_text(json.dumps({
        'query': f'question {qid}', 'query_id': f'traj{qid}', 'report': 'r', 'evidence': evidence,
        'rounds': [{'timestamp': 1.7e9}]}), encoding='utf-8')
    vectors = np.abs(rng.normal(size=(n, dim))) + .01
    vectors[1] = vectors[0]  # one exact near-duplicate pair
    np.savez(folder / f'{qid}_emb.npz', ids=np.array(ids), vectors=vectors,
             query=np.abs(rng.normal(size=(1, dim))) + .01, subquestions=np.abs(rng.normal(size=(3, dim))) + .01)


@pytest.fixture
def corpus(tmp_path):
    run_dir = tmp_path / 'run'
    for seed, qid in enumerate(['101', '102', '103', '201', '301']):
        write_pool(run_dir, qid, seed)
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['101', '102', '103'], 'val': ['201'], 'test': ['301']}))
    return split, run_dir


def test_advantage_modes():
    r = [.5, .7, .2, .2, .9, .1]; g = ['a', 'a', 'b', 'b', 'c', 'c']
    assert np.allclose(advantages(r, g, 'global', .4), np.array(r) - .4)
    assert np.allclose(advantages(r, g, 'global1', .4), np.array(r) - .4)
    c = advantages(r, g, 'center', 99.)
    assert np.allclose(c, [-.1, .1, 0, 0, .4, -.4])
    s = advantages(r, g, 'standardize', 99.)
    assert np.allclose(s[:2], [-1, 1], atol=1e-4) and np.allclose(s[2:4], 0) and np.allclose(s[4:], [1, -1], atol=1e-4)
    with pytest.raises(ValueError):
        advantages(r, g, 'unknown', 0)


def test_spread_separates_within_and_between_query_variation():
    within, between, zero = spread([.5, .5, .9, .9], ['a', 'a', 'b', 'b'])
    assert within == 0 and between == pytest.approx(.2) and zero == 1.


def test_split_loading_refuses_test_by_default(corpus, tmp_path):
    split, run_dir = corpus; tok = Tokenizer()
    pools, _, roles, qids, sources = load_drg(split, [run_dir], tok, 50000)
    assert sorted(qids.values()) == ['101', '102', '103', '201']
    assert sorted(roles.values()) == ['train', 'train', 'train', 'validation']
    assert len(load_drg(split, [run_dir], tok, 50000, allow_test=True)[0]) == 5
    with pytest.raises(FileNotFoundError):
        load_drg(split, [tmp_path / 'missing'], tok, 50000)


def test_rollout_corpus_layout_picks_largest_pool(tmp_path):
    corpus = tmp_path / 'rollouts'
    for seed, n in [(0, 10), (1, 14), (2, 14), (3, 9)]:
        write_pool(corpus / f'101-s{seed}', '101', seed, n=n)
    assert find_pool('101', [corpus]) == corpus / '101-s1' / 'gpt_researcher' / '101.json'
    with pytest.raises(FileNotFoundError):
        find_pool('102', [corpus])


def test_diagnose_reports_spread_per_protocol(corpus, tmp_path):
    split, run_dir = corpus
    result = diagnose(tmp_path / 'diag', split, [run_dir], group=3)
    assert {r['protocol'] for r in result['spread']} == {'shortlist', 'random_order', 'cosine'}
    assert len(result['per_query']) == 3 * 4 and (tmp_path / 'diag' / 'diagnostics.csv').exists()
    with pytest.raises(FileExistsError):
        diagnose(tmp_path / 'diag', split, [run_dir])


def test_end_to_end_offline_run(corpus, tmp_path):
    split, run_dir = corpus; out = tmp_path / 'study'
    flat = run(out, split, [run_dir], protocols=('shortlist',), modes=MODES, seeds=(0,), group=2,
               config=Config(epochs=2))
    for name in ['config.json', 'summary.json', 'curves.json', 'mask_changes.json', 'comparison.csv', 'per_query.csv']:
        assert (out / name).exists()
    episodes = {m: json.loads((out / 'shortlist' / m / 'seed-0' / 'history.json').read_text())[0]['episodes']
                for m in MODES}
    assert episodes == {'global1': 3, 'global': 6, 'center': 6, 'standardize': 6}
    methods = {(r['advantage'], r['method']) for r in flat if r['split'] == 'validation'}
    assert ('center', 'trained_deterministic') in methods and (None, 'shortlist_greedy') in methods
    changes = json.loads((out / 'mask_changes.json').read_text())
    assert len(changes) == len(MODES) * 4 and all(c['changed_pages'] >= 0 for c in changes)
    with pytest.raises(FileExistsError):
        run(out, split, [run_dir], protocols=('shortlist',), seeds=(0,), group=2, config=Config(epochs=2))


def test_spearman_handles_order_ties_and_constants():
    from .drg_proxy_check import spearman
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(.894, abs=1e-3)
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_long_config_and_neutral_start():
    assert LongConfig(epochs=200).epochs == 200
    for bad in (dict(epochs=0), dict(epochs=501), dict(epochs=50, hidden=16), dict(epochs=50, learning_rate=0)):
        with pytest.raises(ValueError):
            LongConfig(**bad)
    torch.manual_seed(0); a = make_policy(); torch.manual_seed(0); b = make_policy(.5)
    x = torch.zeros(1, a.net[0].in_features)
    assert torch.sigmoid(a(x)).item() == pytest.approx(.7, abs=.01)
    assert torch.sigmoid(b(x)).item() == pytest.approx(.5, abs=.01)
    assert torch.equal(a.net[0].weight, b.net[0].weight)  # only the starting bias moves
