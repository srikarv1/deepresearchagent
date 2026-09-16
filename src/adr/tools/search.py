from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field

from adr.core.state import evidence_id
from adr.core.types import Evidence


class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    text: str | None = None
    score: float = 0.0
    doc_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    def to_evidence(
        self,
        *,
        query: str,
        backend: str,
        subtask_id: str | None = None,
    ) -> Evidence:
        return Evidence(
            id=evidence_id(self.url, self.title),
            url=self.url,
            title=self.title,
            snippet=self.snippet,
            text=self.text,
            query=query,
            subtask_id=subtask_id,
            score=self.score,
            source_backend=backend,
        )


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, k: int = 5) -> list[SearchHit]: ...

    async def fetch(self, url: str) -> str: ...


class MockSearch:
    """In-memory corpus for tests and dry runs."""

    name = "mock"

    def __init__(self, corpus: dict[str, list[dict]] | None = None, path: str | Path | None = None) -> None:
        if corpus is None:
            corpus_path = Path(path or Path(__file__).resolve().parents[3] / "data" / "fixtures" / "corpus.json")
            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.corpus = corpus

    def _matches(self, query: str) -> list[dict]:
        q = query.lower()
        if query in self.corpus:
            return list(self.corpus[query])
        hits: list[dict] = []
        for key, docs in self.corpus.items():
            if key.lower() in q or q in key.lower() or any(w in key.lower() for w in q.split() if len(w) > 3):
                hits.extend(docs)
        if not hits:
            for docs in self.corpus.values():
                hits.extend(docs)
                break
        return hits

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        docs = self._matches(query)[:k]
        out: list[SearchHit] = []
        for i, doc in enumerate(docs):
            out.append(
                SearchHit(
                    url=doc["url"],
                    title=doc.get("title", ""),
                    snippet=doc.get("snippet", ""),
                    text=doc.get("text"),
                    score=float(doc.get("score", 1.0 - i * 0.1)),
                )
            )
        return out

    async def fetch(self, url: str) -> str:
        for docs in self.corpus.values():
            for doc in docs:
                if doc.get("url") == url:
                    return doc.get("text") or doc.get("snippet") or ""
        return ""


class TavilySearch:
    """Live web search for DeepResearch Bench runs."""

    name = "tavily"

    def __init__(self, *, api_key: str | None = None, timeout_s: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY") or ""
        self.timeout_s = timeout_s

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is required for the tavily search backend")
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": k, "include_raw_content": True},
            )
            response.raise_for_status()
            payload = response.json()
        hits: list[SearchHit] = []
        for i, row in enumerate(payload.get("results") or []):
            hits.append(
                SearchHit(
                    url=row.get("url", ""),
                    title=row.get("title", ""),
                    snippet=row.get("content", ""),
                    text=row.get("raw_content"),
                    score=float(row.get("score") or (1.0 - i * 0.05)),
                    raw=row,
                )
            )
        return hits

    async def fetch(self, url: str) -> str:
        if not self.api_key:
            return ""
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                "https://api.tavily.com/extract",
                json={"api_key": self.api_key, "urls": [url]},
            )
            response.raise_for_status()
            payload = response.json()
        results = payload.get("results") or []
        if results:
            return str(results[0].get("raw_content") or results[0].get("content") or "")
        return ""


def _extract_hits(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "hits", "documents", "docs", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            return [row for row in value["results"] if isinstance(row, dict)]
    return []


def build_search(cfg: dict) -> SearchBackend:
    backend = str(cfg.get("backend", "mock")).lower()
    if backend == "mock":
        return MockSearch(path=cfg.get("corpus_path"))
    if backend == "tavily":
        return TavilySearch(api_key=cfg.get("api_key"))
    if backend in {"browsecomp_plus", "bcp"}:
        # Imported here so the JVM-backed module is only loaded when asked for.
        from adr.tools.browsecomp_plus import BrowseCompPlusSearch

        return BrowseCompPlusSearch(
            index_path=cfg.get("index_path"),
            snippet_chars=int(cfg.get("snippet_chars", 600)),
            retriever=str(cfg.get("retriever") or "bm25"),
            dense_path=cfg.get("dense_path"),
            embed_model=cfg.get("embed_model"),
            embed_base_url=cfg.get("embed_base_url"),
            query_prefix=cfg.get("query_prefix"),
        )
    raise ValueError(f"Unknown search backend: {backend}")
