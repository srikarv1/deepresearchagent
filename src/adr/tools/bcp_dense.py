"""Dense BrowseComp-Plus retrieval: official corpus vectors + a local query encoder.

The Tevatron ``qwen3-embedding-*`` pickle shards are the document side of the
BrowseComp-Plus dense rows (same corpus, same encoder family). We do **not**
re-encode 100K documents. At query time we embed the question with a matching
Qwen3-Embedding model served by Ollama (8B is the BCP paper headline; CPU
works, but query encode is slower than 0.6B) and take the top-k by inner
product. Document *text* still comes from the BM25 Lucene index already on
disk, so ``bcp://`` hits keep the same shape gpt-researcher expects.

Query encoding must use the same instruction prefix and L2-normalization as
Tevatron's example (``examples/BrowseComp-Plus``). A different Ollama tag than
the shard name (e.g. 8B queries against 0.6B shards) will not rank correctly.

This is **not** the BM25 leaderboard row. Caption Accuracy/Recall as
Qwen3-Embedding-8B (or whichever shard you loaded).
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from adr.tools.browsecomp_plus import ROOT

DENSE_ENV = "ADR_BCP_DENSE"
DEFAULT_DENSE_PATH = ROOT / "third_party" / "bcp_indexes" / "qwen3-embedding-8b"
DENSE_HELP = (
    "python scripts/download_bcp_index.py --kind dense "
    "--dense-model qwen3-embedding-8b "
    "(or set ADR_BCP_DENSE / search.dense_path)"
)

# Qwen3-Embedding instruct format used by Tevatron's BrowseComp-Plus encode.
# Trailing space after "Query:" matches `Instruct: {task}\nQuery: {query}`.
DEFAULT_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery: "
)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"


class QueryEmbedder(Protocol):
    """Blocking query encoder. Tests inject a fake; production uses Ollama."""

    def embed_query(self, text: str) -> list[float]: ...


def resolve_dense_path(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get(DENSE_ENV) or DEFAULT_DENSE_PATH
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def dense_index_looks_valid(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "corpus.npz").is_file():
        return True
    return any(path.glob("corpus*.pkl")) or any(path.glob("*.pkl"))


def l2_normalize(vec: list[float] | Any) -> list[float]:
    import math

    vals = [float(x) for x in vec]
    n = math.sqrt(sum(v * v for v in vals))
    if n <= 0:
        return vals
    return [v / n for v in vals]


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for dense BrowseComp-Plus search. "
            "pip install -e '.[bcp]' (or pip install numpy)."
        ) from exc
    return np


def _as_float_matrix(reps: Any) -> Any:
    """Tevatron shards pickle numpy arrays; some machines pickle torch tensors."""
    np = _numpy()
    if hasattr(reps, "detach"):
        reps = reps.detach().cpu().numpy()
    return np.asarray(reps, dtype=np.float32)


def load_tevatron_shards(directory: Path) -> tuple[Any, list[str]]:
    """Load ``(N, D)`` corpus matrix and aligned docids from pickle shards or npz.

    On first pickle load we write ``corpus.npz`` next to the shards so later
    starts skip pickle/torch.
    """
    np = _numpy()
    npz = directory / "corpus.npz"
    if npz.is_file():
        data = np.load(npz, allow_pickle=True)
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        ids = [str(x) for x in data["ids"].tolist()]
        return vectors, ids

    files = sorted(directory.glob("corpus*.pkl")) or sorted(directory.glob("*.pkl"))
    files = [p for p in files if p.suffix == ".pkl"]
    if not files:
        raise FileNotFoundError(f"No corpus*.pkl or corpus.npz in {directory}. {DENSE_HELP}")

    parts: list[Any] = []
    ids: list[str] = []
    for path in files:
        try:
            with path.open("rb") as fh:
                reps, lookup = pickle.load(fh)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Unpickling {path.name} needs the library that created it "
                f"({exc.name}). Install a CPU torch wheel, or convert the "
                f"shards to corpus.npz on a machine that can load them."
            ) from exc
        parts.append(_as_float_matrix(reps))
        ids.extend(str(x) for x in lookup)
    vectors = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    np.savez(npz, vectors=vectors, ids=np.array(ids, dtype=object))
    return vectors, ids


class NumpyDenseIndex:
    """Exact inner-product search over L2-normalized corpus vectors."""

    def __init__(self, vectors: Any, ids: list[str]) -> None:
        np = _numpy()
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"corpus vectors must be 2-D, got {self.vectors.shape}")
        self.ids = [str(x) for x in ids]
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError(
                f"id count {len(self.ids)} != vector rows {self.vectors.shape[0]}"
            )
        self.dim = int(self.vectors.shape[1])

    @classmethod
    def from_dir(cls, directory: str | Path) -> NumpyDenseIndex:
        path = Path(directory)
        if not dense_index_looks_valid(path):
            raise FileNotFoundError(f"BrowseComp-Plus dense index not found at {path}. {DENSE_HELP}")
        vectors, ids = load_tevatron_shards(path)
        return cls(vectors, ids)

    @property
    def num_docs(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, query_vec: list[float], k: int) -> list[tuple[str, float]]:
        np = _numpy()
        q = np.asarray(l2_normalize(query_vec), dtype=np.float32)
        if q.shape[0] != self.dim:
            raise ValueError(
                f"query dim {q.shape[0]} != corpus dim {self.dim}; "
                "the Ollama model must match the shard (0.6B vs 4B vs 8B)"
            )
        scores = self.vectors @ q
        k = max(1, min(int(k), scores.shape[0]))
        if k == scores.shape[0]:
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, kth=k - 1)[:k]
            order = part[np.argsort(-scores[part])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in order]


class OllamaEmbedder:
    """Query encoder via Ollama ``/api/embed`` (CPU-capable)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str | None = None,
        prefix: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip(
            "/"
        )
        env_prefix = os.environ.get("ADR_BCP_QUERY_PREFIX")
        self.prefix = DEFAULT_QUERY_PREFIX if env_prefix is None else env_prefix
        if prefix is not None:
            self.prefix = prefix
        self.timeout_s = timeout_s

    def embed_query(self, text: str) -> list[float]:
        prefixed = self.prefix + text
        try:
            body = self._post("/api/embed", {"model": self.model, "input": prefixed})
        except Exception as embed_exc:
            try:
                body = self._post(
                    "/api/embeddings", {"model": self.model, "prompt": prefixed}
                )
            except Exception:
                raise RuntimeError(
                    f"Ollama embed failed at {self.base_url} ({embed_exc}). "
                    f"Install Ollama, run `ollama pull {self.model}`, and keep "
                    "the daemon up (OLLAMA_HOST)."
                ) from embed_exc
        vectors = body.get("embeddings") or []
        if not vectors:
            one = body.get("embedding")
            if one:
                vectors = [one]
        if not vectors:
            data = body.get("data") or []
            if data and isinstance(data[0], dict) and data[0].get("embedding"):
                vectors = [data[0]["embedding"]]
        if not vectors:
            raise RuntimeError(f"Ollama returned no embedding: {sorted(body)[:8]}")
        return l2_normalize(vectors[0])

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))


class DenseCorpusIndex:
    """``CorpusIndex``: dense rank + Lucene (or injected) text lookup."""

    def __init__(
        self,
        *,
        dense: NumpyDenseIndex,
        embedder: QueryEmbedder,
        text_index: Any | None = None,
        texts: dict[str, str] | None = None,
    ) -> None:
        self.dense = dense
        self.embedder = embedder
        self._text_index = text_index
        self._texts = texts or {}

    @property
    def num_docs(self) -> int:
        return self.dense.num_docs

    @property
    def dim(self) -> int:
        return self.dense.dim

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        q = self.embedder.embed_query(query)
        hits: list[dict[str, Any]] = []
        for docid, score in self.dense.search(q, k):
            hits.append({"docid": docid, "score": score, "text": self.document(docid) or ""})
        return hits

    def document(self, docid: str) -> str | None:
        if docid in self._texts:
            return self._texts[docid]
        if self._text_index is None:
            return None
        return self._text_index.document(docid)
