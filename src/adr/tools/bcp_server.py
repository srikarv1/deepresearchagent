"""HTTP front for the BrowseComp-Plus corpus, in gpt-researcher's ``custom`` shape.

gpt-researcher's ``RETRIEVER=custom`` does ``GET $RETRIEVER_ENDPOINT?query=...``
(plus any ``RETRIEVER_ARG_*`` env vars as extra query params, e.g.
``RETRIEVER_ARG_K=5`` -> ``k=5``) and expects a JSON list of
``{"url": ..., "raw_content": ...}``; with ``raw_content`` present it skips
scraping. Serving the corpus this way means gpt-researcher itself needs no
changes, and pyserini + the JVM (and, for dense, numpy + Ollama) live in this
process only.

Routes (all GET):
  /search?query=<q>&k=<n>   ranked hits, one object per document
  /doc?docid=<id>           full text of one document, 404 if unknown
  /health                   retriever, index path, document count

``adr serve-retriever`` is BM25. ``adr serve-retriever --retriever dense``
ranks with official Qwen3-Embedding-8B shards and encodes queries through
Ollama; Lucene still supplies ``raw_content``. Point the agent at the same
endpoint either way::

    retriever: custom
    env:
      RETRIEVER_ENDPOINT: "http://127.0.0.1:8321/search"
      RETRIEVER_ARG_K: "5"
"""

from __future__ import annotations

import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from adr.tools.browsecomp_plus import BrowseCompPlusSearch, docid_from_url, url_from_docid

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8321


def hit_payload(hit: Any, *, max_chars: int = 0) -> dict[str, Any]:
    """One search hit in the shape gpt-researcher's CustomRetriever consumes."""
    text = hit.text or ""
    if max_chars > 0:
        text = text[:max_chars]
    return {
        "url": hit.url,
        "raw_content": text,
        "title": hit.title,
        "docid": hit.doc_id,
        "score": hit.score,
    }


def make_handler(
    backend: BrowseCompPlusSearch, *, default_k: int = 5, max_chars: int = 0, quiet: bool = False
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "adr-bcp-retriever/0.1"

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            t0 = time.perf_counter()
            parts = urlsplit(self.path)
            params = parse_qs(parts.query, keep_blank_values=True)
            try:
                if parts.path == "/search":
                    status, body = self._search(params)
                elif parts.path == "/doc":
                    status, body = self._doc(params)
                elif parts.path == "/health":
                    status, body = HTTPStatus.OK, self._health()
                else:
                    status, body = HTTPStatus.NOT_FOUND, {"error": f"no route {parts.path}"}
            except Exception as exc:  # surface index errors as 500, keep serving
                status = HTTPStatus.INTERNAL_SERVER_ERROR
                body = {"error": f"{type(exc).__name__}: {exc}"}
            self._send(status, body)
            if not quiet:
                q = (params.get("query") or params.get("docid") or [""])[0]
                n = len(body) if isinstance(body, list) else ""
                print(
                    f"{status.value} {parts.path} {1000 * (time.perf_counter() - t0):.0f}ms "
                    f"n={n} {q[:80]!r}",
                    file=sys.stderr,
                    flush=True,
                )

        def _search(self, params: dict[str, list[str]]) -> tuple[HTTPStatus, Any]:
            query = (params.get("query") or [""])[0].strip()
            if not query:
                return HTTPStatus.BAD_REQUEST, {"error": "missing query"}
            try:
                k = int((params.get("k") or [default_k])[0])
            except ValueError:
                return HTTPStatus.BAD_REQUEST, {"error": "k must be an integer"}
            hits = backend.search_sync(query, max(1, min(k, 100)))
            return HTTPStatus.OK, [hit_payload(h, max_chars=max_chars) for h in hits]

        def _doc(self, params: dict[str, list[str]]) -> tuple[HTTPStatus, Any]:
            docid = (params.get("docid") or [""])[0].strip() or docid_from_url(
                (params.get("url") or [""])[0]
            )
            if not docid:
                return HTTPStatus.BAD_REQUEST, {"error": "missing docid"}
            text = backend.fetch_sync(url_from_docid(docid))
            if not text:
                return HTTPStatus.NOT_FOUND, {"error": f"unknown docid {docid}"}
            if max_chars > 0:
                text = text[:max_chars]
            payload = {"url": url_from_docid(docid), "docid": docid, "raw_content": text}
            return HTTPStatus.OK, payload

        def _health(self) -> dict[str, Any]:
            index = backend._get_index()  # noqa: SLF001 (same package)
            payload: dict[str, Any] = {
                "status": "ok",
                "retriever": getattr(backend, "retriever", "bm25"),
                "index_path": str(backend.index_path),
                "num_docs": getattr(index, "num_docs", None),
            }
            dense_path = getattr(backend, "dense_path", None)
            if dense_path is not None:
                payload["dense_path"] = str(dense_path)
                payload["dim"] = getattr(index, "dim", None)
                payload["embed_model"] = getattr(backend, "embed_model", None)
            return payload

        def _send(self, status: HTTPStatus, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # request logging is done in do_GET with timing

    return Handler


def make_server(
    backend: BrowseCompPlusSearch,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    default_k: int = 5,
    max_chars: int = 0,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    handler = make_handler(backend, default_k=default_k, max_chars=max_chars, quiet=quiet)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    *,
    index_path: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    default_k: int = 5,
    max_chars: int = 0,
    retriever: str = "bm25",
    dense_path: str | None = None,
    embed_model: str | None = None,
    embed_base_url: str | None = None,
    query_prefix: str | None = None,
) -> None:
    backend = BrowseCompPlusSearch(
        index_path=index_path,
        retriever=retriever,
        dense_path=dense_path,
        embed_model=embed_model,
        embed_base_url=embed_base_url,
        query_prefix=query_prefix,
    )
    kind = backend.retriever
    where = backend.dense_path if kind == "dense" else backend.index_path
    print(f"loading {kind} {where} ...", file=sys.stderr, flush=True)
    backend.warm_up()  # fail fast on a missing index / JDK / Ollama before binding the port
    server = make_server(
        backend, host=host, port=port, default_k=default_k, max_chars=max_chars
    )
    extra = f" embed={backend.embed_model}" if kind == "dense" else ""
    print(
        f"serving BrowseComp-Plus {kind}{extra} on http://{host}:{port}/search (k={default_k})",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
