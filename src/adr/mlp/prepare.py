"""Deterministic trajectory splits, train-only scaling and resumable frozen embeddings."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Callable
from urllib.parse import quote

import numpy as np
from openai import APIError, APIStatusError
from pydantic import BaseModel, ConfigDict, Field
import yaml

from adr.llm.openai_compat import OpenAICompatLLM
from adr.synthetic.schema import FEATURE_NAMES, SyntheticExample
from adr.synthetic.validate import load_heldout, validate_examples

TEXT_FIELDS = ("original_question", "context_before", "candidate_subquery")
LABELS = {"EXECUTE": 1, "SKIP": 0}
VERSION = "mlp-preparation-v1"
# Byte-level BPE cannot produce more tokens than UTF-8 bytes. Conservative
# preflight for text-embedding-3-small, with no tokenizer download or truncation.
MAX_TEXT_BYTES = 8191


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Path
    heldout: Path
    output: Path
    seed: int = 42
    validation_trajectories: int = Field(default=4, gt=0, lt=20)
    model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    batch_size: int = Field(default=16, gt=0, le=16)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_key(text: str) -> str:
    return digest(text.encode("utf-8"))


def write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".prepare.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another preparation process owns this output directory") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def split_trajectories(rows: list[dict], seed: int, validation_count: int) -> dict:
    """Hash rank IDs, without labels/categories or order-dependent RNG state."""
    owners, question_owners = {}, {}
    for row in rows:
        tid, qid = row["trajectory_id"], row["question_id"]
        if tid in owners and owners[tid] != qid:
            raise ValueError("Trajectory crosses questions")
        if qid in question_owners and question_owners[qid] != tid:
            raise ValueError("Question crosses trajectories")
        owners[tid], question_owners[qid] = qid, tid
    ranked = sorted(owners, key=lambda tid: (text_key(f"{seed}\0{tid}"), tid))
    if not 0 < validation_count < len(ranked):
        raise ValueError("Invalid validation trajectory count")
    return {"algorithm": "sha256(seed + NUL + trajectory_id), ascending; first N validation",
            "seed": seed, "train": sorted(ranked[validation_count:]),
            "validation": sorted(ranked[:validation_count])}


def model_parts(row: dict) -> tuple[tuple[str, ...], list[float], int]:
    """The only model-input projection. Never embed a serialized example."""
    if row["numerical_features"]["names"] != list(FEATURE_NAMES):
        raise ValueError("Numerical feature order mismatch")
    values = row["numerical_features"]["values"]
    if len(values) != 15 or not np.isfinite(values).all():
        raise ValueError("Invalid numerical vector")
    return tuple(row[field] for field in TEXT_FIELDS), values, LABELS[row["label"]]


def fit_normalizer(train: np.ndarray) -> dict:
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 15 or not len(train) or not np.isfinite(train).all():
        raise ValueError("Expected finite training array (N, 15)")
    mean, std = train.mean(axis=0), train.std(axis=0, ddof=0)
    zero = std == 0
    return {"feature_names": list(FEATURE_NAMES), "fit_split": "train", "ddof": 0,
            "n_training_rows": len(train), "mean": mean.tolist(), "std": std.tolist(),
            "scale": np.where(zero, 1.0, std).tolist(), "zero_variance": zero.tolist()}


def normalize(values: np.ndarray, stats: dict) -> np.ndarray:
    if stats["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Normalizer feature order mismatch")
    result = (np.asarray(values, dtype=np.float64) - stats["mean"]) / stats["scale"]
    if result.ndim != 2 or result.shape[1] != 15 or not np.isfinite(result).all():
        raise ValueError("Invalid normalized features")
    return result.astype(np.float32)


def embedding_identity(model: str, dimensions: int) -> dict:
    endpoint = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    return {"provider": "openai_compat", "model": model, "dimensions": dimensions,
            "endpoint_sha256": text_key(endpoint) if endpoint else None,
            "input_policy": "exact-utf8-v1"}


def contract_differences(saved: object, current: object, path: str = "") -> list[str]:
    """Report field paths only; never disclose endpoint or credential values."""
    if isinstance(saved, dict) and isinstance(current, dict):
        fields = []
        for key in sorted(saved.keys() | current.keys()):
            child = f"{path}.{key}" if path else key
            if key not in saved or key not in current:
                fields.append(child)
            else:
                fields.extend(contract_differences(saved[key], current[key], child))
        return fields
    return [path] if saved != current else []


def check_contract(directory: Path, saved: dict, current: dict) -> None:
    differences = contract_differences(saved, current)
    # An explicitly unbound offline plan may bind once, before a cache exists.
    # This is not permission to migrate any existing endpoint-bound artifact.
    pending = saved.get("embedding", {}).get("endpoint_sha256", "missing") is None
    if pending and differences == ["embedding.endpoint_sha256"]:
        manifest = directory / "manifest.json"
        if (manifest.exists() and json.loads(manifest.read_text()).get("status") == "awaiting_endpoint"
                and not any((directory / name).exists()
                            for name in ["embeddings.sqlite3", "train.npz", "validation.npz"])):
            return
    if differences:
        raise ValueError("Preparation contract changed in: " + ", ".join(differences)
                         + "; use a new output directory")


class EmbeddingCache:
    """Transactional per-batch cache; only successful, finite vectors are committed."""
    def __init__(self, path: Path, identity: dict):
        self.db = sqlite3.connect(path)
        self.identity = identity
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, text TEXT, vector BLOB)")
        encoded = json.dumps(identity, sort_keys=True)
        saved = self.db.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if saved and saved[0] != encoded:
            self.db.close()
            raise ValueError("Embedding cache model/endpoint/dimensions mismatch")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (encoded,))
        self.db.commit()

    def get(self, text: str) -> np.ndarray | None:
        row = self.db.execute("SELECT text,vector FROM vectors WHERE key=?", (text_key(text),)).fetchone()
        if row is None:
            return None
        if row[0] != text:
            raise ValueError("Embedding cache text hash mismatch")
        vector = np.frombuffer(row[1], dtype="<f4").copy()
        self.check(vector)
        return vector

    def check(self, vector: np.ndarray) -> None:
        if vector.shape != (self.identity["dimensions"],) or not np.isfinite(vector).all():
            raise ValueError("Embedding vector dimension/finite-value mismatch")

    def put_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        if len(texts) != len(vectors):
            raise ValueError("Embedding response count mismatch")
        records = []
        for text, vector in zip(texts, vectors):
            array = np.asarray(vector, dtype="<f4")
            self.check(array)
            records.append((text_key(text), text, array.tobytes()))
        with self.db:
            self.db.executemany("INSERT INTO vectors VALUES (?, ?, ?)", records)

    def close(self):
        self.db.close()


class EmbeddingBatchError(RuntimeError):
    """Safe diagnostic only; never includes a raw SDK exception/body."""


def safe_embedding_error(exc: Exception, texts: list[str] | tuple[str, ...] = ()) -> str:
    """Allowlist SDK error fields, then redact before formatting/truncation.

    Do not use str(APIError): it can include headers, URLs or complete bodies.
    Non-API exception messages are omitted unless they are our local checks.
    """
    if isinstance(exc, EmbeddingBatchError):
        return str(exc)
    secrets = list(texts)
    secrets.extend(value for name, value in os.environ.items() if value and
                   re.search(r'KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|BASE_URL|ENDPOINT', name, re.I))
    if isinstance(exc, APIError):
        # Values only used for redaction, never included as diagnostic fields.
        for name, value in exc.request.headers.items():
            if re.search(r'auth|key|token|secret|cookie', name, re.I):
                secrets.extend([value, value.removeprefix('Bearer ')])

    def clean(value: str) -> str:
        variants = {v for secret in secrets if secret for v in
                    (secret, quote(secret, safe=''), json.dumps(secret)[1:-1])}
        for secret in sorted(variants, key=len, reverse=True):
            value = value.replace(secret, '[REDACTED]')
        value = re.sub(r'https?://[^\s<>"\']+', '[URL REDACTED]', value, flags=re.I)
        value = re.sub(r'\bBearer\s+\S+', 'Bearer [REDACTED]', value, flags=re.I)
        value = re.sub(r'(?i)(["\']?(?:api[-_ ]?key|access[-_ ]?token|secret|password|authorization|sig)["\']?\s*[:=]\s*)(?:"[^"]*"|\'[^\']*\'|[^\s,;]+)',
                       r'\1[REDACTED]', value)
        value = re.sub(r'\bsk-[A-Za-z0-9_-]+|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
                       '[REDACTED]', value)
        return ' '.join(value.split())[:800]

    parts = [type(exc).__name__]
    if isinstance(exc, APIStatusError):
        parts.append(f'HTTP {exc.status_code}')
    if isinstance(exc, APIError):
        body = exc.body
        if isinstance(body, dict):
            detail = body.get('error', body)
            if isinstance(detail, dict):
                for field in ('code', 'type', 'param', 'message'):
                    value = detail.get(field)
                    if isinstance(value, str):
                        parts.append(f'{field}={clean(value)}')
        if len(parts) == (2 if isinstance(exc, APIStatusError) else 1):
            parts.append('No structured API error details available; raw response omitted')
    elif isinstance(exc, ValueError) and str(exc) in {
            'Embedding response count mismatch', 'Embedding vector dimension/finite-value mismatch',
            'Embedding response has missing or duplicate indices'}:
        parts.append(str(exc))
    return '; '.join(parts)


async def fill_cache(texts: list[str], cache: EmbeddingCache, batch_size: int,
                     embed: Callable) -> None:
    missing = [text for text in dict.fromkeys(texts) if cache.get(text) is None]
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        try:
            vectors = await embed(batch, dimensions=cache.identity["dimensions"])
            cache.put_batch(batch, vectors)
        except Exception as exc:
            total = (len(missing) + batch_size - 1) // batch_size
            raise EmbeddingBatchError(
                f"Embedding batch {start // batch_size + 1}/{total} ({len(batch)} strings) failed: "
                f"{safe_embedding_error(exc, batch)}. "
                "Successful batches remain cached. Rerun to resume."
            ) from None


def assemble(rows: list[dict], numerical: np.ndarray, cache: EmbeddingCache) -> np.ndarray:
    semantic = []
    for row in rows:
        texts, _, _ = model_parts(row)
        parts = [cache.get(text) for text in texts]
        if any(part is None for part in parts):
            raise ValueError("Missing frozen embedding; cannot build X")
        semantic.append(np.concatenate(parts))
    return np.concatenate([np.asarray(semantic, dtype=np.float32), numerical], axis=1)


def summaries(rows: list[dict]) -> dict:
    return {"examples": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "reason_categories": dict(sorted(Counter(r["decision_category"] for r in rows).items()))}


async def prepare(cfg: Config, *, allow_api: bool = False, client_factory=None) -> dict:
    source = cfg.source.resolve()
    if cfg.output.resolve() == source.parent:
        raise ValueError("Output must be separate from canonical dataset")
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    validation = validate_examples(rows, load_heldout(cfg.heldout), expected_states=5, expected_candidates=4)
    if not validation["valid"] or len(rows) != 400 or validation["trajectories"] != 20 or validation["questions"] != 20:
        raise ValueError("Expected validated 400 examples / 20 questions / 20 complete trajectories")
    if cfg.validation_trajectories != 4:
        raise ValueError("Locked pilot requires 16 train and 4 validation trajectories")
    # Stable array order even if upstream enumeration changes.
    rows.sort(key=lambda r: (r["trajectory_id"], r["provenance"]["state_index"], r["candidate_id"]))
    for row in rows:
        SyntheticExample.model_validate(row)
    split = split_trajectories(rows, cfg.seed, cfg.validation_trajectories)
    identity = embedding_identity(cfg.model, cfg.embedding_dimensions)
    if allow_api and identity["endpoint_sha256"] is None:
        raise ValueError("OPENAI_BASE_URL must be explicitly set before embedding generation; "
                         "no API calls or artifact changes were made")
    contract = {"version": VERSION, "source_sha256": digest(raw),
                "heldout_sha256": digest(cfg.heldout.read_bytes()), "split": split,
                "embedding": identity, "feature_names": list(FEATURE_NAMES),
                "text_fields": list(TEXT_FIELDS), "labels": LABELS}
    groups = {name: [r for r in rows if r["trajectory_id"] in split[name]]
              for name in ["train", "validation"]}
    pair_splits = {}
    for name, group in groups.items():
        for row in group:
            pid = row["provenance"]["counterfactual_pair_id"]
            if pid:
                key = (row["trajectory_id"], pid)
                if key in pair_splits and pair_splits[key] != name:
                    raise ValueError("Counterfactual pair crosses splits")
                pair_splits[key] = name
    stats = fit_normalizer(np.array([model_parts(r)[1] for r in groups["train"]]))
    texts = sorted({text for row in rows for text in model_parts(row)[0]}, key=text_key)
    if any(not text or len(text.encode("utf-8")) > MAX_TEXT_BYTES for text in texts):
        raise ValueError("Embedding text exceeds conservative 8191-byte input limit; no truncation allowed")
    with output_lock(cfg.output):
        contract_path = cfg.output / "contract.json"
        if contract_path.exists():
            check_contract(cfg.output, json.loads(contract_path.read_text()), contract)
        write_json(contract_path, contract)
        write_json(cfg.output / "split.json", split)
        write_json(cfg.output / "normalization.json", stats)
        write_json(cfg.output / "embedding_inputs.json", [{"key": text_key(t), "text": t} for t in texts])
        normalized = {}
        for name, group in groups.items():
            normalized[name] = normalize(np.array([model_parts(r)[1] for r in group]), stats)
            y = np.array([model_parts(r)[2] for r in group], dtype=np.int64)
            write_npz(cfg.output / f"{name}_numerical.npz", numerical=normalized[name], y=y)
            metadata = []
            for row in group:
                item = {k: row[k] for k in ["example_id", "trajectory_id", "question_id", "state_id",
                                             "candidate_id", "label", "reason", "decision_category"]}
                item.update(state_index=row["provenance"]["state_index"],
                            counterfactual_pair_id=row["provenance"]["counterfactual_pair_id"],
                            embedding_keys={f: text_key(row[f]) for f in TEXT_FIELDS})
                metadata.append(item)
            write_json(cfg.output / f"{name}_metadata.json", metadata)
        cache = (EmbeddingCache(cfg.output / "embeddings.sqlite3", identity)
                 if identity["endpoint_sha256"] is not None else None)
        try:
            def uncached(text):
                return cache is None or cache.get(text) is None

            missing = [t for t in texts if uncached(t)]
            report = {"version": VERSION, "status": "awaiting_embeddings" if cache else "awaiting_endpoint",
                      "endpoint_notice": (None if cache else
                          "OPENAI_BASE_URL is unspecified. Offline plan is unbound; no embedding cache "
                          "was created. Run again with the intended endpoint explicitly set."),
                      "split_summary": {k: summaries(v) for k, v in groups.items()},
                      "unique_strings": {f: len({r[f] for r in rows}) for f in TEXT_FIELDS},
                      "unique_total": len(texts), "cached": len(texts) - len(missing),
                      "missing_total": len(missing),
                      "missing_by_field": {f: len({r[f] for r in rows if uncached(r[f])}) for f in TEXT_FIELDS},
                      "embedding_dimensions": cfg.embedding_dimensions,
                      "input_dimensions": 3 * cfg.embedding_dimensions + 15,
                      "maximum_utf8_bytes": max(len(t.encode("utf-8")) for t in texts),
                      "source_validation": validation,
                      "leakage_contract": "Only exact three text fields and 15 train-scaled pre-decision features enter X; metadata is separate."}
            write_json(cfg.output / "manifest.json", report)
            if allow_api and missing:
                if client_factory is None:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise ValueError("OPENAI_API_KEY is required for embedding generation")
                    client_factory = OpenAICompatLLM
                client = client_factory(model=cfg.model)
                try:
                    await fill_cache(texts, cache, cfg.batch_size, client.embed)
                except EmbeddingBatchError:
                    raise
                except Exception as exc:
                    raise RuntimeError(safe_embedding_error(exc) +
                                       " Successful batches remain cached. Rerun to resume.") from None
                finally:
                    await client.close()
            missing = [t for t in texts if uncached(t)]
            report.update(cached=len(texts) - len(missing), missing_total=len(missing))
            report["missing_by_field"] = {f: len({r[f] for r in rows if uncached(r[f])}) for f in TEXT_FIELDS}
            if not missing:
                for name, group in groups.items():
                    X = assemble(group, normalized[name], cache)
                    y = np.array([model_parts(r)[2] for r in group], dtype=np.int64)
                    write_npz(cfg.output / f"{name}.npz", X=X, y=y)
                report["status"] = "ready"
            report["artifact_sha256"] = {p.name: digest(p.read_bytes()) for p in sorted(cfg.output.iterdir())
                                         if p.suffix in {".json", ".npz"} and p.name != "manifest.json"}
            write_json(cfg.output / "manifest.json", report)
            return report
        finally:
            if cache is not None:
                cache.close()


def load_prepared(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """No client or API access. Frozen arrays only; refuses incomplete artifacts."""
    if split not in {"train", "validation"}:
        raise ValueError("Unknown split")
    report = json.loads((directory / "manifest.json").read_text())
    if report["status"] != "ready":
        raise ValueError("Embeddings not generated; preparation is not ready")
    for name in [f"{split}.npz", f"{split}_metadata.json", "normalization.json", "contract.json"]:
        if digest((directory / name).read_bytes()) != report["artifact_sha256"][name]:
            raise ValueError("Prepared artifact checksum mismatch")
    with np.load(directory / f"{split}.npz", allow_pickle=False) as data:
        X, y = data["X"].copy(), data["y"].copy()
    metadata = json.loads((directory / f"{split}_metadata.json").read_text())
    if X.shape != (len(metadata), report["input_dimensions"]) or y.shape != (len(metadata),):
        raise ValueError("Prepared array shape mismatch")
    return X, y, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mlp_prepare.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Offline; never construct an API client (default)")
    mode.add_argument("--allow-api", action="store_true", help="Explicitly permit uncached embedding requests")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(override=False)
    cfg = Config.model_validate(yaml.safe_load(args.config.read_text()))
    report = asyncio.run(prepare(cfg, allow_api=args.allow_api))
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_validation", "artifact_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
