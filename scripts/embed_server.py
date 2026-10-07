"""Local OpenAI-compatible embeddings server (bge-small), single process.

The agent calls embeddings from many threads; loading torch inside the agent
segfaults on macOS. This server owns the model and serializes requests, and
the agent reaches it through its normal OpenAI embeddings client:

    EMBEDDING=openai:bge-small
    EMBEDDING_KWARGS={"openai_api_base":"http://127.0.0.1:8323/v1","openai_api_key":"local","check_embedding_ctx_length":false}

    python scripts/embed_server.py        # POST /v1/embeddings, GET /health
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
PORT = int(os.environ.get("EMBED_PORT", "8323"))
_model = SentenceTransformer(MODEL_ID)
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._send(200, {"ok": True, "model": MODEL_ID})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/embeddings"):
            return self._send(404, {"error": "POST /v1/embeddings"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        if inputs and not isinstance(inputs[0], str):
            return self._send(400, {"error": "send strings (check_embedding_ctx_length=false)"})
        with _lock:
            vecs = _model.encode([t[:8000] for t in inputs], normalize_embeddings=True, batch_size=64)
        data = [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]
        n = sum(len(t.split()) for t in inputs)
        self._send(200, {"object": "list", "data": data, "model": MODEL_ID, "usage": {"prompt_tokens": n, "total_tokens": n}})

    def _send(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    print(f"embeddings: {MODEL_ID} on http://127.0.0.1:{PORT}/v1/embeddings", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
