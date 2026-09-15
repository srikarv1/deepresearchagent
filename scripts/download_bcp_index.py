#!/usr/bin/env python
"""Download BrowseComp-Plus indexes from Tevatron/browsecomp-plus-indexes.

``bm25/`` (~2.1 GB) is the Lucene index that stores document text. The dense
``qwen3-embedding-0.6b/`` shards (~0.41 GB) are official corpus vectors for
the Qwen3-Embedding-0.6B row. Query encoding is *not* downloaded: serve those
with Ollama (``ollama pull qwen3-embedding:0.6b``).

Files that already exist with the right size are skipped, so re-running after
an interrupted download only fetches what is missing.

    python scripts/download_bcp_index.py                  # BM25 (default)
    python scripts/download_bcp_index.py --kind dense     # 0.6B shards
    python scripts/download_bcp_index.py --kind all
    python scripts/download_bcp_index.py --dest /data/bcp/bm25   # then set ADR_BCP_INDEX
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "Tevatron/browsecomp-plus-indexes"
# Pinned so every checkout searches the same index; bump deliberately.
REVISION = "b3f37f70c33829eb09d04784a54277a31871fd63"
INDEX_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "bcp_indexes"
DEFAULT_BM25 = INDEX_ROOT / "bm25"
DEFAULT_DENSE_MODEL = "qwen3-embedding-0.6b"
CHUNK = 8 * 1024 * 1024


def list_files(repo: str, revision: str, subdir: str) -> list[tuple[str, int]]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{subdir}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        entries = json.load(resp)
    files = [(e["path"], int(e.get("size") or 0)) for e in entries if e.get("type") == "file"]
    if not files:
        raise RuntimeError(f"No files listed at {url}")
    return sorted(files)


def download(url: str, dest: Path, expected_size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)
    got = tmp.stat().st_size
    if expected_size and got != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: got {got} bytes, expected {expected_size}")
    tmp.replace(dest)


def fetch_subdir(subdir: str, dest: Path, revision: str) -> None:
    files = list_files(REPO, revision, subdir)
    total = sum(size for _, size in files)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{subdir}: {len(files)} files, {total / 1e9:.2f} GB -> {dest}", file=sys.stderr)

    for path, size in files:
        target = dest / Path(path).name
        if target.exists() and target.stat().st_size == size:
            print(f"skip   {target.name}", file=sys.stderr)
            continue
        print(f"fetch  {target.name} ({size / 1e6:.1f} MB)", file=sys.stderr, flush=True)
        download(
            f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}",
            target,
            size,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kind",
        choices=("bm25", "dense", "all"),
        default="bm25",
        help="bm25 Lucene text index, dense Qwen3-Embedding shards, or both",
    )
    parser.add_argument(
        "--dense-model",
        default=DEFAULT_DENSE_MODEL,
        help="HF subdirectory under the indexes repo (must match the Ollama tag family)",
    )
    parser.add_argument("--dest", type=Path, default=None, help="Override destination directory")
    parser.add_argument("--revision", default=REVISION, help=f"{REPO} commit or branch")
    args = parser.parse_args()

    kinds = ("bm25", "dense") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        if kind == "bm25":
            dest = args.dest if args.dest is not None and args.kind != "all" else DEFAULT_BM25
            if args.kind == "all" and args.dest is not None:
                dest = Path(args.dest) / "bm25"
            fetch_subdir("bm25", dest, args.revision)
            if not any(p.name.startswith("segments_") for p in dest.iterdir()):
                print("error: no segments_N file; this is not a Lucene index", file=sys.stderr)
                return 1
            print(f"ok: {dest}", file=sys.stderr)
        else:
            dest = (
                args.dest
                if args.dest is not None and args.kind != "all"
                else INDEX_ROOT / args.dense_model
            )
            if args.kind == "all" and args.dest is not None:
                dest = Path(args.dest) / args.dense_model
            fetch_subdir(args.dense_model, dest, args.revision)
            if not any(p.suffix == ".pkl" or p.name == "corpus.npz" for p in dest.iterdir()):
                print("error: no corpus*.pkl / corpus.npz in dense dest", file=sys.stderr)
                return 1
            print(f"ok: {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
