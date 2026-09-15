"""Dense BrowseComp-Plus ranking without Ollama, pyserini, or the real shards."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from adr.eval.repos import find_bcp_dense_index
from adr.tools.bcp_dense import (
    DEFAULT_QUERY_PREFIX,
    DenseCorpusIndex,
    NumpyDenseIndex,
    OllamaEmbedder,
    dense_index_looks_valid,
    l2_normalize,
    load_tevatron_shards,
)
from adr.tools.bcp_server import make_server
from adr.tools.browsecomp_plus import BrowseCompPlusSearch
from adr.tools.search import build_search

np = pytest.importorskip("numpy")


class _FixedEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self.vec = vec
        self.seen: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.seen.append(text)
        return self.vec


def _unit(vec: list[float]) -> list[float]:
    return l2_normalize(vec)


def test_inner_product_ranks_nearest_normalized_vector():
    vectors = np.asarray(
        [_unit([1.0, 0.0]), _unit([0.2, 1.0]), _unit([0.0, 1.0])],
        dtype=np.float32,
    )
    index = NumpyDenseIndex(vectors, ["a", "b", "c"])
    hits = index.search([0.0, 1.0], k=2)
    assert [docid for docid, _ in hits] == ["c", "b"]
    assert hits[0][1] > hits[1][1]


def test_dim_mismatch_is_explicit():
    index = NumpyDenseIndex(np.eye(2, dtype=np.float32), ["a", "b"])
    with pytest.raises(ValueError, match="query dim 3"):
        index.search([1.0, 0.0, 0.0], k=1)


def test_tevatron_pickle_roundtrip_writes_npz(tmp_path: Path):
    reps = np.eye(3, dtype=np.float32)
    lookup = ["10", "20", "30"]
    shard = tmp_path / "corpus.shard1_of_4.pkl"
    shard.write_bytes(pickle.dumps((reps, lookup), protocol=4))
    assert dense_index_looks_valid(tmp_path)

    vectors, ids = load_tevatron_shards(tmp_path)
    assert ids == lookup
    assert vectors.shape == (3, 3)
    assert (tmp_path / "corpus.npz").is_file()

    again, ids2 = load_tevatron_shards(tmp_path)
    assert ids2 == lookup
    assert np.allclose(again, vectors)


def test_dense_corpus_returns_injected_text():
    vectors = np.asarray([_unit([1.0, 0.0]), _unit([0.0, 1.0])], dtype=np.float32)
    dense = NumpyDenseIndex(vectors, ["111", "222"])
    embedder = _FixedEmbedder([0.0, 1.0])
    index = DenseCorpusIndex(
        dense=dense, embedder=embedder, texts={"111": "alpha", "222": "beta body"}
    )
    hits = index.search("anything", k=1)
    assert hits[0]["docid"] == "222"
    assert hits[0]["text"] == "beta body"
    assert embedder.seen == ["anything"]
    assert index.document("111") == "alpha"


def test_backend_dense_http_contract():
    vectors = np.asarray([_unit([1.0, 0.0]), _unit([0.0, 1.0])], dtype=np.float32)
    index = DenseCorpusIndex(
        dense=NumpyDenseIndex(vectors, ["111", "222"]),
        embedder=_FixedEmbedder([0.0, 1.0]),
        texts={"111": "---\ntitle: A\n---\nalpha", "222": "---\ntitle: B\n---\nbeta"},
    )
    backend = BrowseCompPlusSearch(index=index, retriever="dense")
    hits = backend.search_sync("q", 1)
    assert hits[0].url == "bcp://222"
    assert hits[0].doc_id == "222"
    assert "beta" in hits[0].text

    import threading
    import urllib.parse
    import urllib.request

    srv = make_server(backend, host="127.0.0.1", port=0, default_k=1, quiet=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/search?" + urllib.parse.urlencode(
            {"query": "q"}
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.load(resp)
        assert body[0]["url"] == "bcp://222"
        assert "beta" in body[0]["raw_content"]
        with urllib.request.urlopen(
            f"http://127.0.0.1:{srv.server_address[1]}/health", timeout=5
        ) as resp:
            health = json.load(resp)
        assert health["retriever"] == "dense"
        assert health["num_docs"] == 2
        assert health["dim"] == 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_dense_factory_is_lazy(tmp_path: Path):
    backend = build_search(
        {
            "backend": "browsecomp_plus",
            "retriever": "dense",
            "dense_path": str(tmp_path / "nope"),
            "index_path": str(tmp_path / "no-lucene"),
        }
    )
    assert isinstance(backend, BrowseCompPlusSearch)
    assert backend.retriever == "dense"
    with pytest.raises(FileNotFoundError):
        backend.search_sync("anything", 1)


def test_unknown_retriever_rejected():
    with pytest.raises(ValueError, match="unknown BrowseComp-Plus retriever"):
        BrowseCompPlusSearch(retriever="hybrid")


def test_find_bcp_dense_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADR_BCP_DENSE", str(tmp_path / "missing"))
    assert not find_bcp_dense_index().ok
    good = tmp_path / "qwen"
    good.mkdir()
    (good / "corpus.shard1_of_4.pkl").write_bytes(b"x")
    assert dense_index_looks_valid(good)
    assert find_bcp_dense_index(good).ok
    monkeypatch.setenv("ADR_BCP_DENSE", str(good))
    assert find_bcp_dense_index().path == good.resolve()


def test_ollama_embedder_posts_official_prefix(monkeypatch: pytest.MonkeyPatch):
    seen: dict = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"embeddings": [[3.0, 4.0]]}).encode()

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.delenv("ADR_BCP_QUERY_PREFIX", raising=False)
    monkeypatch.setattr("adr.tools.bcp_dense.urlopen", fake_urlopen)
    vec = OllamaEmbedder(model="qwen3-embedding:0.6b", prefix=None).embed_query("hello")
    assert seen["url"].endswith("/api/embed")
    assert seen["body"]["input"] == DEFAULT_QUERY_PREFIX + "hello"
    assert vec == pytest.approx([0.6, 0.8])


def test_load_dense_requires_lucene_for_text(tmp_path: Path):
    reps = np.eye(2, dtype=np.float32)
    (tmp_path / "corpus.shard1_of_4.pkl").write_bytes(pickle.dumps((reps, ["1", "2"]), protocol=4))
    backend = BrowseCompPlusSearch(
        retriever="dense",
        dense_path=tmp_path,
        index_path=tmp_path / "no-lucene",
        embedder=_FixedEmbedder([1.0, 0.0]),
    )
    with pytest.raises(FileNotFoundError, match="document text"):
        backend.search_sync("q", 1)
