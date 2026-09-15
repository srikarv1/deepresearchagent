"""Pinned BrowseComp-Plus hold-out. Official leaderboard Acc is all 830; this is not that.

Train is for PILOT trajectory generation (and later BC/DPO/GRPO). Val is for
checkpoint / prompt choices. Test is the paper table: every baseline row and
PILOT must use the same test ids. Caption that Acc as hold-out, not the
830-query BrowseComp-Plus number.

The assignment is ``Random(seed).shuffle`` over numeric-sorted query ids so
counts are exact and the file can be regenerated bit-identically.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from adr.datasets.loader import BROWSECOMP_PLUS_QUERIES, DatasetName, ROOT

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEED = 17
DEFAULT_RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}
BCP_SPLITS_PATH = ROOT / "data" / "benchmarks" / "browsecomp_plus" / "splits.json"

_SPLIT_FILES = {
    DatasetName.BROWSECOMP_PLUS.value: BCP_SPLITS_PATH,
}


def _id_key(qid: str) -> tuple[int, str]:
    return (int(qid), qid) if qid.isdigit() else (10**18, qid)


def partition_ids(
    ids: Iterable[str],
    *,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """Exact-count shuffle split. Test is taken first from the shuffled list."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    ordered = sorted({str(x) for x in ids}, key=_id_key)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_test = int(round(n * float(ratios["test"])))
    n_val = int(round(n * float(ratios["val"])))
    test = sorted(ordered[:n_test], key=_id_key)
    val = sorted(ordered[n_test : n_test + n_val], key=_id_key)
    train = sorted(ordered[n_test + n_val :], key=_id_key)
    return {"train": train, "val": val, "test": test}


def build_bcp_splits(
    query_path: Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict:
    path = query_path or BROWSECOMP_PLUS_QUERIES
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ids.append(str(json.loads(line)["query_id"]))
    parts = partition_ids(ids, seed=seed, ratios=ratios or DEFAULT_RATIOS)
    payload = {
        "dataset": DatasetName.BROWSECOMP_PLUS.value,
        "n": len(ids),
        "seed": seed,
        "ratios": dict(ratios or DEFAULT_RATIOS),
        "counts": {name: len(parts[name]) for name in SPLIT_NAMES},
        "note": (
            "Hold-out for PILOT + paper table. Official BrowseComp-Plus Acc is "
            "all 830 queries; do not caption test Acc as that number."
        ),
        **parts,
    }
    assigned = [qid for name in SPLIT_NAMES for qid in parts[name]]
    if sorted(assigned, key=_id_key) != sorted(ids, key=_id_key):
        raise RuntimeError("split does not cover the query file exactly once")
    return payload


def write_bcp_splits(dest: Path | None = None, **kwargs) -> Path:
    dest = dest or BCP_SPLITS_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = build_bcp_splits(**kwargs)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _load_splits_cached.cache_clear()
    return dest


@lru_cache(maxsize=8)
def _load_splits_cached(dataset: str, loc: str) -> dict[str, tuple[str, ...]]:
    data = json.loads(Path(loc).read_text(encoding="utf-8"))
    out = {name: tuple(str(x) for x in data.get(name) or []) for name in SPLIT_NAMES}
    if not any(out.values()):
        raise ValueError(f"{loc} has no train/val/test ids")
    return out


def load_splits(dataset: str, *, path: Path | None = None) -> dict[str, list[str]]:
    loc = path or _SPLIT_FILES.get(dataset)
    if loc is None or not Path(loc).is_file():
        raise FileNotFoundError(
            f"No pinned split for {dataset!r}. BrowseComp-Plus: "
            "python scripts/build_browsecomp_plus_splits.py"
        )
    cached = _load_splits_cached(dataset, str(Path(loc).resolve()))
    return {name: list(ids) for name, ids in cached.items()}


def ids_for_split(dataset: str, split: str, *, path: Path | None = None) -> set[str]:
    name = str(split).lower().strip()
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r} (train|val|test)")
    return set(load_splits(dataset, path=path)[name])


def split_of(dataset: str, query_id: str, *, path: Path | None = None) -> str | None:
    try:
        parts = load_splits(dataset, path=path)
    except FileNotFoundError:
        return None
    qid = str(query_id)
    for name in SPLIT_NAMES:
        if qid in set(parts[name]):
            return name
    return None
