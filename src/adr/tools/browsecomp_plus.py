"""BrowseComp-Plus corpus retriever: BM25 over Tevatron/browsecomp-plus-corpus.

BrowseComp-Plus is defined over a fixed corpus of ~100K documents, not the
live web. This backend searches the prebuilt Lucene index published in
Tevatron/browsecomp-plus-indexes (``bm25/``) with Pyserini's default BM25,
the retriever the reference agents use, so evidence Recall measured against
``topics-qrels/`` is comparable to the leaderboard's BM25 rows. Building a
different BM25 (rank_bm25, bm25s) would change tokenisation and parameters and
break that comparison, so this deliberately depends on Pyserini.

Hits are addressed as ``bcp://<docid>``. Everything downstream carries URLs
(evidence pools, citations, gpt-researcher's research_sources), so encoding
the docid in the URL is what lets ``retrieved_docids`` be recovered at export
time without teaching each agent about corpus ids.

Prerequisites::

    pip install -e '.[bcp]'          # pyserini; needs a Java 21 JDK on PATH
    python scripts/download_bcp_index.py   # 2.1 GB -> third_party/bcp_indexes/bm25

The index is resolved from ``index_path`` (config), then ``ADR_BCP_INDEX``,
then ``third_party/bcp_indexes/bm25``. Loading is lazy: constructing the
backend never touches the JVM, so a run config can name this backend even when
the agent under test (gpt-researcher) reaches the corpus through
``adr serve-retriever`` instead of ``ctx.search``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Protocol

from adr.tools.search import SearchHit

ROOT = Path(__file__).resolve().parents[3]

BCP_URL_PREFIX = "bcp://"
INDEX_ENV = "ADR_BCP_INDEX"
DEFAULT_INDEX_PATH = ROOT / "third_party" / "bcp_indexes" / "bm25"
INDEX_HELP = (
    "python scripts/download_bcp_index.py (or set ADR_BCP_INDEX / search.index_path)"
)

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TITLE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)


# ── docid <-> url ─────────────────────────────────────────────────


def url_from_docid(docid: str) -> str:
    return f"{BCP_URL_PREFIX}{docid}"


def docid_from_url(url: str | None) -> str | None:
    """Return the corpus docid encoded in a ``bcp://`` URL, else None."""
    if not isinstance(url, str) or not url.startswith(BCP_URL_PREFIX):
        return None
    docid = url[len(BCP_URL_PREFIX) :].strip().rstrip("/")
    return docid or None


def docids_from_urls(urls: Iterable[Any]) -> list[str]:
    """Unique corpus docids found among arbitrary URLs, sorted for stable output."""
    return sorted({d for d in (docid_from_url(u) for u in urls) if d})


# ── index access ──────────────────────────────────────────────────


class CorpusIndex(Protocol):
    """Blocking access to the corpus. ``LuceneIndex`` is the real one."""

    def search(self, query: str, k: int) -> list[dict[str, Any]]: ...

    def document(self, docid: str) -> str | None: ...


def resolve_index_path(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get(INDEX_ENV) or DEFAULT_INDEX_PATH
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def index_looks_valid(path: Path) -> bool:
    """A Lucene index directory always contains a ``segments_N`` file."""
    return path.is_dir() and any(p.name.startswith("segments_") for p in path.iterdir())


def contents_from_raw(raw: str | None) -> str:
    """Anserini stores each document as JSON ``{"id": ..., "contents": ...}``."""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(payload, dict):
        return str(payload.get("contents") or payload.get("text") or "")
    return ""


def title_from_contents(text: str) -> str:
    """Corpus documents open with a YAML-ish front matter; pull ``title:`` if present."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return ""
    title = _TITLE.search(match.group(1))
    return title.group(1).strip() if title else ""


class LuceneIndex:
    """Pyserini ``LuceneSearcher`` over the BrowseComp-Plus BM25 index.

    Importing pyserini starts a JVM in the importing thread; keep every call
    to one instance on one thread (``BrowseCompPlusSearch`` does this).
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        if not index_looks_valid(self.index_path):
            raise FileNotFoundError(
                f"BrowseComp-Plus BM25 index not found at {self.index_path}. {INDEX_HELP}"
            )
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError(
                "pyserini is not installed. pip install -e '.[bcp]' (needs a Java 21 JDK)."
            ) from exc
        self._searcher = LuceneSearcher(str(self.index_path))

    @property
    def num_docs(self) -> int:
        return int(self._searcher.num_docs)

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in self._searcher.search(query, k):
            results.append(
                {
                    "docid": str(hit.docid),
                    "score": float(hit.score),
                    "text": contents_from_raw(hit.lucene_document.get("raw")),
                }
            )
        return results

    def document(self, docid: str) -> str | None:
        doc = self._searcher.doc(docid)
        return None if doc is None else contents_from_raw(doc.raw())


# ── harness backend ───────────────────────────────────────────────


class BrowseCompPlusSearch:
    """``SearchBackend`` over the BrowseComp-Plus corpus.

    ``search`` returns full document text in ``SearchHit.text`` (the corpus is
    local, there is nothing to scrape) and a bounded snippet; ``fetch`` accepts
    ``bcp://<docid>`` and returns the full text. Non-corpus URLs fetch as "".
    """

    name = "browsecomp_plus"

    def __init__(
        self,
        *,
        index_path: str | Path | None = None,
        snippet_chars: int = 600,
        index: CorpusIndex | None = None,
    ) -> None:
        self.index_path = resolve_index_path(index_path)
        self.snippet_chars = int(snippet_chars)
        self._index: CorpusIndex | None = index
        # One worker: the JVM is started and used from a single thread, and
        # Lucene calls are serialised without a lock. BM25 lookups are ~10 ms.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bcp-lucene")

    # Blocking API (used by the HTTP server and by the async wrappers).
    def _get_index(self) -> CorpusIndex:
        if self._index is None:
            self._index = LuceneIndex(self.index_path)
        return self._index

    def _search_blocking(self, query: str, k: int) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for row in self._get_index().search(query, max(1, int(k))):
            text = row.get("text") or ""
            hits.append(
                SearchHit(
                    url=url_from_docid(row["docid"]),
                    title=title_from_contents(text),
                    snippet=text[: self.snippet_chars],
                    text=text,
                    score=float(row.get("score") or 0.0),
                    doc_id=row["docid"],
                )
            )
        return hits

    def _fetch_blocking(self, url: str) -> str:
        docid = docid_from_url(url)
        if docid is None:
            return ""
        return self._get_index().document(docid) or ""

    def search_sync(self, query: str, k: int = 5) -> list[SearchHit]:
        return self._pool.submit(self._search_blocking, query, k).result()

    def fetch_sync(self, url: str) -> str:
        return self._pool.submit(self._fetch_blocking, url).result()

    def warm_up(self) -> None:
        """Load the index now (JVM start + segment open) instead of on first query."""
        self._pool.submit(self._get_index).result()

    # SearchBackend protocol.
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._search_blocking, query, k)

    async def fetch(self, url: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._fetch_blocking, url)
