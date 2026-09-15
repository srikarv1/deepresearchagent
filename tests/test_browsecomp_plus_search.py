"""BrowseComp-Plus corpus backend and retriever server, without pyserini.

A fake ``CorpusIndex`` stands in for the Lucene index so the SearchHit mapping,
bcp:// addressing, the HTTP contract gpt-researcher's CustomRetriever relies
on, and lazy loading are all exercised in CI. A final test runs against the
real index when pyserini and the download are present.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from adr.eval.repos import find_bcp_index
from adr.tools.bcp_server import make_server
from adr.tools.browsecomp_plus import (
    BrowseCompPlusSearch,
    contents_from_raw,
    docid_from_url,
    docids_from_urls,
    index_looks_valid,
    title_from_contents,
    url_from_docid,
)
from adr.tools.search import build_search

_DOC = (
    "---\ntitle: United States of Africa\ndate: 2020-04-15\n---\n"
    + "KENYAN PROLIFIC SWAHILI AUTHOR " * 40
)


class _FakeIndex:
    num_docs = 3

    def __init__(self) -> None:
        self.docs = {
            "68543": _DOC,
            "2882": "---\n---\nKen Walibora, Kenyan author interview",
            "1": "x",
        }
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int):
        self.calls.append((query, k))
        ranked = [
            d for d in ("68543", "2882", "1")
            if query == "*" or query.lower() in self.docs[d].lower()
        ]
        return [
            {"docid": d, "score": 10.0 - i, "text": self.docs[d]} for i, d in enumerate(ranked[:k])
        ]

    def document(self, docid: str):
        return self.docs.get(docid)


def test_url_helpers():
    assert url_from_docid("68543") == "bcp://68543"
    assert docid_from_url("bcp://68543") == "68543"
    assert docid_from_url("bcp://68543/") == "68543"
    assert docid_from_url("https://example.com/bcp://1") is None
    assert docid_from_url(None) is None
    assert docid_from_url("bcp://") is None
    assert docids_from_urls(["bcp://9", "https://a", "bcp://9", None, "bcp://10"]) == ["10", "9"]


def test_document_parsing():
    assert contents_from_raw(json.dumps({"id": "1", "contents": "body"})) == "body"
    assert contents_from_raw("not json") == "not json"
    assert contents_from_raw(None) == ""
    assert title_from_contents(_DOC) == "United States of Africa"
    assert title_from_contents("---\n---\nno title") == ""
    assert title_from_contents("plain text") == ""


async def test_backend_search_and_fetch():
    index = _FakeIndex()
    backend = BrowseCompPlusSearch(index=index, snippet_chars=20)

    hits = await backend.search("kenyan", k=2)
    assert [h.url for h in hits] == ["bcp://68543", "bcp://2882"]
    assert hits[0].doc_id == "68543"
    assert hits[0].title == "United States of Africa"
    assert hits[0].text == _DOC
    assert len(hits[0].snippet) == 20
    assert hits[0].score > hits[1].score
    assert index.calls == [("kenyan", 2)]

    assert await backend.fetch("bcp://2882") == "---\n---\nKen Walibora, Kenyan author interview"
    assert await backend.fetch("bcp://missing") == ""
    assert await backend.fetch("https://example.com") == ""
    assert backend.search_sync("walibora", 5)[0].url == "bcp://2882"


def test_factory_is_lazy(tmp_path: Path):
    """Naming the backend in a run config must not require pyserini or the index."""
    backend = build_search({"backend": "browsecomp_plus", "index_path": str(tmp_path / "nope")})
    assert isinstance(backend, BrowseCompPlusSearch)
    assert backend.name == "browsecomp_plus"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        backend.search_sync("anything", 1)


def test_find_bcp_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ADR_BCP_INDEX", str(tmp_path / "missing"))
    assert not find_bcp_index().ok
    good = tmp_path / "bm25"
    good.mkdir()
    (good / "segments_3").write_bytes(b"")
    assert index_looks_valid(good)
    assert find_bcp_index(good).ok
    monkeypatch.setenv("ADR_BCP_INDEX", str(good))
    assert find_bcp_index().path == good.resolve()


@pytest.fixture
def server():
    backend = BrowseCompPlusSearch(index=_FakeIndex())
    srv = make_server(backend, host="127.0.0.1", port=0, default_k=2, max_chars=0, quiet=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.load(resp)


def test_server_matches_gpt_researcher_custom_retriever_contract(server: str):
    # CustomRetriever does GET endpoint?query=...&<RETRIEVER_ARG_*> and reads url + raw_content.
    status, body = _get(f"{server}/search?" + urllib.parse.urlencode({"query": "kenyan", "k": 1}))
    assert status == 200
    assert body == [
        {
            "url": "bcp://68543",
            "raw_content": _DOC,
            "title": "United States of Africa",
            "docid": "68543",
            "score": 10.0,
        }
    ]
    # default_k applies when the client sends no k
    _, body = _get(f"{server}/search?query=*")
    assert [row["url"] for row in body] == ["bcp://68543", "bcp://2882"]

    _, body = _get(f"{server}/doc?docid=2882")
    assert body["raw_content"] == "---\n---\nKen Walibora, Kenyan author interview"
    _, body = _get(f"{server}/doc?url=bcp://2882")
    assert body["docid"] == "2882"
    _, body = _get(f"{server}/health")
    assert body["status"] == "ok" and body["num_docs"] == 3
    assert body["retriever"] == "bm25"


def test_server_error_codes(server: str):
    cases = (
        ("/search", 400),
        ("/search?query=x&k=abc", 400),
        ("/doc?docid=nope", 404),
        ("/nope", 404),
    )
    for path, code in cases:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(server + path)
        assert exc.value.code == code, path


def test_server_max_chars_truncates():
    backend = BrowseCompPlusSearch(index=_FakeIndex())
    srv = make_server(backend, port=0, max_chars=10, quiet=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _, body = _get(f"http://127.0.0.1:{srv.server_address[1]}/search?query=kenyan&k=1")
        assert body[0]["raw_content"] == _DOC[:10]
    finally:
        srv.shutdown()
        srv.server_close()


_index = find_bcp_index()


@pytest.mark.skipif(not _index.ok, reason="BrowseComp-Plus BM25 index not downloaded")
def test_real_index_returns_evidence_docs():
    pytest.importorskip("pyserini")
    backend = BrowseCompPlusSearch(index_path=_index.path)
    hits = backend.search_sync("Ken Walibora road accident", 5)
    # Evidence docs for query 1 (topics-qrels/qrel_evidence.txt): 2882 62014 68543 74874.
    assert {"68543", "2882", "62014"} <= {h.doc_id for h in hits}
    assert all(h.url.startswith("bcp://") and h.text for h in hits)
    assert backend.fetch_sync("bcp://62014").startswith("---")


async def test_metered_search_records_corpus_docids():
    """Harness-native agents get retrieved_docids for free via the meter."""
    from adr.core.instrument import CostMeter, MeteredSearch

    meter = CostMeter()
    metered = MeteredSearch(BrowseCompPlusSearch(index=_FakeIndex()), meter)
    await metered.search("kenyan", k=5)
    await metered.search("walibora", k=5)
    assert meter.retrieved_docids == {"68543", "2882"}
    assert "retrieved_docids" not in meter.snapshot()
