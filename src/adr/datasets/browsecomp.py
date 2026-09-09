"""BrowseComp (OpenAI, 2025) loading and decryption.

The public CSV
``https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv``
has columns ``problem, answer, problem_topic, canary``. ``problem`` and
``answer`` are XOR-encrypted with ``sha256(canary)`` and base64-encoded so
the answers stay out of web crawls; ``problem_topic`` is plaintext. Every
function here mirrors ``simple-evals/browsecomp_eval.py`` so results are
comparable to published numbers:

* decryption: ``derive_key`` / ``decrypt``
* subsampling: ``random.Random(0).sample(rows, n)``
* the query template prepended to each question

The correct answer is **never** placed on the ``Query`` (the agent receives
``task.query``); the judge re-reads the CSV by row index at scoring time.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import random
from pathlib import Path

from adr.core.types import Query

DATASET = "browsecomp"
DEFAULT_CSV = Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "browsecomp" / "browse_comp_test_set.csv"
DOWNLOAD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

# Verbatim from simple-evals/browsecomp_eval.py.
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()


def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def encrypt(plaintext: str, password: str) -> str:
    """Inverse of :func:`decrypt`; used by tests to build synthetic rows."""
    raw = plaintext.encode()
    key = derive_key(password, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()


def query_id(row_index: int) -> str:
    return f"bc_{row_index:04d}"


def row_index_of(qid: str) -> int:
    if not qid.startswith("bc_"):
        raise ValueError(f"Not a BrowseComp query id: {qid!r}")
    return int(qid[3:])


def read_rows(path: str | Path | None = None) -> list[dict]:
    """Raw (still encrypted) CSV rows with their 0-based ``row`` index."""
    source = Path(path) if path else DEFAULT_CSV
    if not source.exists():
        raise FileNotFoundError(
            f"BrowseComp CSV not found: {source}\n"
            f"Download it with:\n  curl -sSL -o {source} {DOWNLOAD_URL}"
        )
    with source.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows):
        row["row"] = i
    return rows


def correct_answer(row: dict) -> str:
    return decrypt(row["answer"], row["canary"])


def problem_text(row: dict) -> str:
    return decrypt(row["problem"], row["canary"])


def load_browsecomp(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    query_ids: list[str] | None = None,
    sample: int | None = None,
    sample_seed: int = 0,
) -> list[Query]:
    """Decrypt questions into ``Query`` objects.

    ``sample`` draws that many rows with ``random.Random(sample_seed)`` exactly
    as simple-evals does (seed 0 by default), so ``sample: 200`` is the same
    200 questions anyone else gets. ``limit`` then truncates in row order and
    ``query_ids`` filters. The decrypted answer is not placed on the Query.
    """
    rows = read_rows(path)
    if sample is not None and sample < len(rows):
        rows = random.Random(sample_seed).sample(rows, sample)
        rows.sort(key=lambda r: r["row"])
    wanted = {str(x) for x in query_ids} if query_ids else None

    out: list[Query] = []
    for row in rows:
        qid = query_id(row["row"])
        if wanted is not None and qid not in wanted:
            continue
        question = problem_text(row)
        out.append(
            Query(
                id=qid,
                text=QUERY_TEMPLATE.format(Question=question),
                dataset=DATASET,
                language="en",
                topic=(row.get("problem_topic") or None),
                metadata={
                    "row": row["row"],
                    "topic": row.get("problem_topic") or "",
                    "question": question,
                },
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def answers_by_id(path: str | Path | None, ids: list[str]) -> dict[str, str]:
    """Decrypt the correct answers for the given query ids (judge-side only)."""
    rows = read_rows(path)
    out: dict[str, str] = {}
    for qid in ids:
        idx = row_index_of(qid)
        if 0 <= idx < len(rows):
            out[qid] = correct_answer(rows[idx])
    return out
