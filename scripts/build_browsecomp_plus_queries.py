#!/usr/bin/env python
"""Rebuild data/benchmarks/browsecomp_plus/query.jsonl from Tevatron/browsecomp-plus.

Upstream ships 830 queries as ~2.8 GB of parquet because every row embeds the
full text of its evidence, gold, and hard-negative documents. The harness only
needs the query, the answer, and the relevance judgements, so this reads the
query_id / query / answer columns plus the docid and url leaves of
evidence_docs and gold_docs through parquet column projection; the document
text is never downloaded (about 20 s total over HTTP range requests).

Every field except query_id stays obfuscated exactly as upstream ships it; the
loader decrypts at load time with the fixed canary. Rows are sorted by numeric
query_id so the output is deterministic for a given revision.

    pip install "pyarrow>=15" "fsspec[http]>=2024.2"
    python scripts/build_browsecomp_plus_queries.py            # rewrite the file
    python scripts/build_browsecomp_plus_queries.py --check    # verify it matches
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "Tevatron/browsecomp-plus"
# Pin the dataset commit so the committed file is reproducible; bump deliberately.
REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"
OUTPUT = (
    Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "browsecomp_plus" / "query.jsonl"
)

# Leaf column paths as they appear in the parquet schema. Selecting these
# leaves keeps the list<struct> shape while dropping the `text` leaf.
COLUMNS = [
    "query_id",
    "query",
    "answer",
    "evidence_docs.list.element.docid",
    "evidence_docs.list.element.url",
    "gold_docs.list.element.docid",
    "gold_docs.list.element.url",
]


def _parquet_paths(repo: str, revision: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/data"
    with urllib.request.urlopen(url, timeout=60) as resp:
        entries = json.load(resp)
    paths = sorted(e["path"] for e in entries if e.get("path", "").endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"No parquet files listed at {url}")
    return paths


def build(repo: str = REPO, revision: str = REVISION) -> list[dict]:
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency hint only
        raise SystemExit(
            f"{exc}. Install the optional data extra: pip install -e '.[data]'"
        ) from exc

    fs = fsspec.filesystem("https")
    rows: list[dict] = []
    for path in _parquet_paths(repo, revision):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
        with fs.open(url, "rb", block_size=8 * 1024 * 1024) as handle:
            table = pq.ParquetFile(handle).read(columns=COLUMNS)
        rows.extend(table.to_pylist())
        print(f"{path}: {len(rows)} rows so far", file=sys.stderr)

    rows.sort(key=lambda r: int(r["query_id"]))
    return [
        {
            "query_id": str(r["query_id"]),
            "query": r["query"],
            "answer": r["answer"],
            "evidence_docs": r["evidence_docs"] or [],
            "gold_docs": r["gold_docs"] or [],
        }
        for r in rows
    ]


def render(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--revision", default=REVISION, help="Tevatron/browsecomp-plus commit or branch"
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit 1 if the existing file differs; write nothing"
    )
    args = parser.parse_args()

    text = render(build(revision=args.revision))
    if args.check:
        if not args.output.exists():
            print(f"missing: {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != text:
            print(f"stale: {args.output} differs from {REPO}@{args.revision}", file=sys.stderr)
            return 1
        print(f"ok: {args.output} matches {REPO}@{args.revision}", file=sys.stderr)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({text.count(chr(10))} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
