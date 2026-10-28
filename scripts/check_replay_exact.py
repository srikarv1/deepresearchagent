#!/usr/bin/env python3
"""Acceptance check for branch replay: the replayed full tree must give the
live writer input byte for byte.

    python scripts/check_replay_exact.py runs/<gym-d2b4-replay-run>
    python scripts/check_replay_exact.py runs/<run>/gpt_researcher/1004217.json

Run on trajectories collected with GR_NODE_CACHE=1 and GR_BRANCH_INDEPENDENT=1
(configs/agents/*_d2b4*_replay.yaml). Exits 1 if any trajectory differs, and
prints where the two strings first part.
"""

from __future__ import annotations

import argparse
import sys

from adr.replay.check import check_full_replay, trajectory_files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories or fork trajectory JSON files")
    p.add_argument("--pack", default=None, help="GR_FINAL_PACK of the live run (default: env, else tail)")
    args = p.parse_args(argv)

    files = trajectory_files(args.paths)
    if not files:
        print("no trajectory files found", file=sys.stderr)
        return 2
    n_bad = 0
    for path in files:
        r = check_full_replay(path, pack=args.pack)
        if r.ok:
            print(f"OK    {path}  ({r.n_nodes} nodes, {r.live_chars} chars)")
            continue
        n_bad += 1
        print(f"FAIL  {path}: {r.reason}")
        if r.first_diff is not None:
            print(f"      {r.n_nodes} nodes; live {r.live_chars} chars, replay {r.replay_chars}; "
                  f"first difference at char {r.first_diff}")
            print(f"      live:   {r.live_snippet!r}")
            print(f"      replay: {r.replay_snippet!r}")
    print(f"\n{len(files) - n_bad}/{len(files)} identical")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
