#!/usr/bin/env python3
"""Pin the DeepResearch Bench (English) train/val/test split for PILOT.

Test = the 5 Table 4 queries (51-55) + 5 more held-out ids; val = 10; train =
the remaining 30, which is the randomized-rollout / BC corpus. The extra ids are
a seeded shuffle of the other 45 English queries, so the file regenerates
bit-identically.

    python scripts/build_drb_splits.py
"""

from __future__ import annotations

import json

from adr.datasets.loader import load_queries
from adr.datasets.splits import DEFAULT_SEED, DRB_SPLITS_PATH, SPLIT_NAMES, _id_key, partition_ids

PINNED_TEST = ["51", "52", "53", "54", "55"]  # Table 4 baselines were run here
N_EXTRA_TEST = 5
N_VAL = 10


def main() -> None:
    ids = [q.id for q in load_queries("deep_research_bench", language="en")]
    rest = [i for i in ids if i not in PINNED_TEST]
    n = len(rest)
    parts = partition_ids(
        rest,
        seed=DEFAULT_SEED,
        ratios={"test": N_EXTRA_TEST / n, "val": N_VAL / n, "train": 1 - (N_EXTRA_TEST + N_VAL) / n},
    )
    test = sorted(PINNED_TEST + parts["test"], key=_id_key)
    payload = {
        "dataset": "deep_research_bench",
        "language": "en",
        "n": len(ids),
        "seed": DEFAULT_SEED,
        "pinned_test": PINNED_TEST,
        "counts": {"train": len(parts["train"]), "val": len(parts["val"]), "test": len(test)},
        "note": "Test pins the Table 4 queries; train is the PILOT rollout/BC corpus.",
        "train": parts["train"],
        "val": parts["val"],
        "test": test,
    }
    assigned = sorted(payload["train"] + payload["val"] + payload["test"], key=_id_key)
    assert assigned == sorted(ids, key=_id_key), "split must cover every English id once"
    DRB_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRB_SPLITS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name in SPLIT_NAMES:
        print(f"{name:5} ({len(payload[name]):2}): {' '.join(payload[name])}")
    print(f"wrote {DRB_SPLITS_PATH}")


if __name__ == "__main__":
    main()
