#!/usr/bin/env python3
"""Pin the DeepResearchGym train/val/test split for PILOT, mirroring the
DeepResearch Bench split: a pool of 50 queries, test = the 5 Table 2 queries
+ 5 held-out ids, val = 10, train = the remaining 30 (the randomized-rollout /
BC corpus). The pool is the first 50 queries of the sample file in file order
(what `--limit N` runs), and the extra ids are a seeded shuffle of the other
45, so the file regenerates bit-identically.

    python scripts/build_gym_splits.py
"""

from __future__ import annotations

import json

from adr.datasets.loader import load_queries
from adr.datasets.splits import DEFAULT_SEED, GYM_SPLITS_PATH, SPLIT_NAMES, _id_key, partition_ids

POOL = 50            # first ids of researchy_queries_sample_doc_click.jsonl, file order
N_PINNED_TEST = 5    # Table 2 baselines were run on the first 5
N_EXTRA_TEST = 5
N_VAL = 10


def build_gym_splits() -> dict:
    ids = [q.id for q in load_queries("deep_research_gym", limit=POOL)]
    pinned = ids[:N_PINNED_TEST]
    rest = ids[N_PINNED_TEST:]
    n = len(rest)
    parts = partition_ids(
        rest,
        seed=DEFAULT_SEED,
        ratios={"test": N_EXTRA_TEST / n, "val": N_VAL / n, "train": 1 - (N_EXTRA_TEST + N_VAL) / n},
    )
    test = sorted(pinned + parts["test"], key=_id_key)
    payload = {
        "dataset": "deep_research_gym",
        "language": "en",
        "n": len(ids),
        "pool": f"first {POOL} queries of the sample file, file order",
        "seed": DEFAULT_SEED,
        "pinned_test": pinned,
        "counts": {"train": len(parts["train"]), "val": len(parts["val"]), "test": len(test)},
        "note": "Test pins the Table 2 queries; train is the PILOT rollout/BC corpus.",
        "train": parts["train"],
        "val": parts["val"],
        "test": test,
    }
    assigned = sorted(payload["train"] + payload["val"] + payload["test"], key=_id_key)
    assert assigned == sorted(ids, key=_id_key), "split must cover the pool exactly once"
    return payload


def main() -> None:
    payload = build_gym_splits()
    GYM_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GYM_SPLITS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name in SPLIT_NAMES:
        print(f"{name:5} ({len(payload[name]):2}): {' '.join(payload[name])}")
    print(f"wrote {GYM_SPLITS_PATH}")


if __name__ == "__main__":
    main()
