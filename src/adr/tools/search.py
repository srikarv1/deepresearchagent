from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

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


class GymSearch:
    """DeepResearchGym retrieval sandbox over ClueWeb22 and FineWeb.

    The corpus is selected by route, not by a parameter: ``/search`` serves
    ClueWeb22 (licence-gated) and ``/fineweb/search`` serves FineWeb. See
    https://clueweb22.us/openapi.json. Authentication is the ``x-api-key``
    header.
    """

    name = "gym"

    CORPUS_ROUTES = {
        "fineweb": "/fineweb/search",
        "clueweb": "/search",
        "clueweb22": "/search",
        "clueweb22-b": "/search",
        "clueweb22-a": "/search",
    }

    def __init__(
        self,
        *,
        base_url: str | None = None,
        search_url: str | None = None,
        fetch_url: str | None = None,
        api_key: str | None = None,
        corpus: str = "fineweb",
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("DEEPRESEARCHGYM_BASE_URL") or "https://clueweb22.us"
        ).rstrip("/")
        self.corpus = (corpus or os.environ.get("DEEPRESEARCHGYM_CORPUS") or "fineweb").lower()
        if self.corpus not in self.CORPUS_ROUTES:
            raise ValueError(
                f"Unknown Gym corpus {self.corpus!r}. Choose from {sorted(self.CORPUS_ROUTES)}"
            )
        self.cw22_a = self.corpus == "clueweb22-a"
        self.search_url = (
            search_url
            or os.environ.get("DEEPRESEARCHGYM_SEARCH_URL")
            or f"{self.base_url}{self.CORPUS_ROUTES[self.corpus]}"
        )
        # The hosted API has no archival fetch route; only set this if you run one.
        self.fetch_url = fetch_url or os.environ.get("DEEPRESEARCHGYM_FETCH_URL") or ""
        self.api_key = api_key or os.environ.get("DEEPRESEARCHGYM_API_KEY") or ""
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        params: dict[str, Any] = {"query": query, "k": k, "with_distance": "true"}
        if self.cw22_a:
            params["cw22_a"] = "true"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(self.search_url, params=params, headers=self._headers())
            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"DeepResearchGym rejected the request ({response.status_code}). "
                    "Set DEEPRESEARCHGYM_API_KEY; ClueWeb22 also needs a data-use agreement."
                )
            response.raise_for_status()
            payload = response.json()
        return _gym_hits(payload, k)

    async def fetch(self, url: str) -> str:
        """Return an archived snapshot, or empty string when no fetch route exists.

        Returning empty rather than raising keeps agents that opportunistically
        read full documents working against the hosted API, which only exposes
        search. Point ``fetch_url`` at a local deployment to enable it.
        """
        if not self.fetch_url:
            return ""
        params = {"url": url}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(self.fetch_url, params=params, headers=self._headers())
            if response.status_code >= 400:
                alt = f"{self.fetch_url.rstrip('/')}/{quote(url, safe='')}"
                response = await client.get(alt, headers=self._headers())
            if response.status_code >= 400:
                return ""
            if "application/json" in response.headers.get("content-type", ""):
                payload = response.json()
                if isinstance(payload, dict):
                    return str(
                        payload.get("Clean-Text")
                        or payload.get("text")
                        or payload.get("content")
                        or payload.get("clean_text")
                        or ""
                    )
            return response.text


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


def _gym_hits(payload: Any, k: int) -> list[SearchHit]:
    """Map a Gym search payload to hits, tolerating ClueWeb and FineWeb field casing."""
    hits: list[SearchHit] = []
    for i, row in enumerate(_extract_hits(payload)[:k]):
        url = str(row.get("URL") or row.get("url") or row.get("link") or "")
        if not url:
            continue
        text = row.get("Clean-Text") or row.get("clean_text") or row.get("text") or row.get("content")
        # Lower distance means closer, so invert it into a descending score.
        distance = row.get("distance") or row.get("score")
        try:
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 1.0 - i * 0.05
        except (TypeError, ValueError):
            score = 1.0 - i * 0.05
        hits.append(
            SearchHit(
                url=url,
                title=str(row.get("title") or row.get("Title") or ""),
                snippet=str(row.get("snippet") or (text or "")[:600]),
                text=text,
                score=score,
                doc_id=row.get("ClueWeb22-ID") or row.get("docid") or row.get("id"),
                raw=row,
            )
        )
    return hits


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
    if backend in {"gym", "deepresearchgym", "clueweb", "fineweb"}:
        corpus = cfg.get("corpus") or ("clueweb22" if backend == "clueweb" else "fineweb")
        return GymSearch(
            base_url=cfg.get("base_url"),
            search_url=cfg.get("search_url"),
            fetch_url=cfg.get("fetch_url"),
            api_key=cfg.get("api_key"),
            corpus=corpus,
        )
    if backend == "tavily":
        return TavilySearch(api_key=cfg.get("api_key"))
    if backend in {"browsecomp_plus", "bcp"}:
        # Imported here so the JVM-backed module is only loaded when asked for.
        from adr.tools.browsecomp_plus import BrowseCompPlusSearch

        return BrowseCompPlusSearch(
            index_path=cfg.get("index_path"),
            snippet_chars=int(cfg.get("snippet_chars", 600)),
            searcher=str(cfg.get("searcher", "bm25")),
            model=str(cfg.get("model", "qwen3-embedding:0.6b")),
            dense_index_path=cfg.get("dense_index_path"),
            ollama_url=str(cfg.get("ollama_url", "http://localhost:11434")),
        )
    raise ValueError(f"Unknown search backend: {backend}")
