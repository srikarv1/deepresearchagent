#!/usr/bin/env python
"""Download a BrowseComp-Plus index (Tevatron/browsecomp-plus-indexes).

Fetches the requested index at a pinned revision into
``third_party/bcp_indexes/<subdir>``. BM25 (~2.1 GB) is required for all
runs (it also stores document text used by the dense searcher). Dense shards
(``qwen3-embedding-*``, 0.4 GB for 0.6B) are optional and only needed for
``adr serve-retriever --searcher dense``.

Files that already exist with the right size are skipped, so re-running after
an interrupted download only fetches what is missing. Standard library only.

    python scripts/download_bcp_index.py
    python scripts/download_bcp_index.py --subdir qwen3-embedding-0.6b
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
SUBDIR = "bm25"
DEST_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "bcp_indexes"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subdir",
        default=SUBDIR,
        help="bm25 (default) | qwen3-embedding-0.6b | qwen3-embedding-4b | qwen3-embedding-8b",
    )
    parser.add_argument("--dest", type=Path, default=None, help=f"default: {DEST_ROOT}/<subdir>")
    parser.add_argument("--revision", default=REVISION, help=f"{REPO} commit or branch")
    args = parser.parse_args()

    dest: Path = args.dest or DEST_ROOT / args.subdir

    files = list_files(REPO, args.revision, args.subdir)
    total = sum(size for _, size in files)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {dest}", file=sys.stderr)

    for path, size in files:
        target = dest / Path(path).name
        if target.exists() and target.stat().st_size == size:
            print(f"skip   {target.name}", file=sys.stderr)
            continue
        print(f"fetch  {target.name} ({size / 1e6:.1f} MB)", file=sys.stderr, flush=True)
        download(
            f"https://huggingface.co/datasets/{REPO}/resolve/{args.revision}/{path}",
            target,
            size,
        )

    if args.subdir == "bm25":
        if not any(p.name.startswith("segments_") for p in dest.iterdir()):
            print("error: no segments_N file; this is not a Lucene index", file=sys.stderr)
            return 1
    elif not any(p.name.endswith(".pkl") for p in dest.iterdir()):
        print("error: no corpus.shard*.pkl files; this is not a dense index", file=sys.stderr)
        return 1
    print(f"ok: {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
