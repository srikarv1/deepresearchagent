#!/usr/bin/env python
"""Write the pinned BrowseComp-Plus train/val/test ids.

    python scripts/build_browsecomp_plus_splits.py
    python scripts/build_browsecomp_plus_splits.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adr.datasets.splits import BCP_SPLITS_PATH, build_bcp_splits, write_bcp_splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if splits.json is stale")
    parser.add_argument("--dest", type=Path, default=BCP_SPLITS_PATH)
    args = parser.parse_args()
    payload = build_bcp_splits()
    if args.check:
        if not args.dest.is_file():
            print(f"missing {args.dest}", file=sys.stderr)
            return 1
        on_disk = json.loads(args.dest.read_text(encoding="utf-8"))
        if on_disk != payload:
            print(f"{args.dest} does not match seed=17 70/10/20", file=sys.stderr)
            return 1
        print(
            f"ok: {payload['counts']['train']} train / "
            f"{payload['counts']['val']} val / {payload['counts']['test']} test",
            file=sys.stderr,
        )
        return 0
    write_bcp_splits(args.dest)
    print(
        f"ok: {args.dest}  {payload['counts']['train']} train / "
        f"{payload['counts']['val']} val / {payload['counts']['test']} test",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
