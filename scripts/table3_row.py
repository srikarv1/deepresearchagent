#!/usr/bin/env python3
"""Table 3 (DeepResearchGym) rows from eval corpora written by `adr rollouts`.

Each corpus is a directory of ``<query>-s<seed>/`` runs with
``metrics/summary.json``. Columns follow the paper's Table 3 and are the
per-query means of what the harness and the official Gym judges recorded:

  Clarity Breadth Depth Support   quality.per_criterion[*].average_rating (1-10)
  KPR                             kpr.average_support_rate
  Overall                         quality.average_normalized_score
  Tokens (K)                      mean_tokens / 1000
  Latency (s)                     mean_wall_s
  Calls                           mean_n_searches
  Q per 1K t                      Overall / Tokens (K)
  Citation (--citation only)      citation.average_citation_score (0-100)

This reproduces the existing PILOT (BC) row from runs/eval_gym_table_learned
(54.7 overall, 300.2 K tokens, 535.3 s, 50 calls). With ``--baseline`` it also
prints per-query paired differences against that corpus, over the queries both
have, with a bootstrap 95% interval on the mean.

    python scripts/table3_row.py "PILOT (BC)=runs/eval_gym_table_learned" \\
        "PILOT (BC, ranked ctx)=runs/eval_gym_learned_ranked" \\
        "PILOT (BC v2q)=runs/eval_gym_v2q_ranked" --baseline "PILOT (BC)"
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

CRITERIA = ("Clarity", "Breadth", "Depth", "Support")
COLUMNS = ("Clarity", "Breadth", "Depth", "Support", "KPR", "Overall", "Tokens (K)", "Latency (s)", "Calls", "Q per 1K t")
CITATION_COLUMNS = COLUMNS[:6] + ("Citation",) + COLUMNS[6:]
DECIMALS = {"Tokens (K)": 1, "Latency (s)": 1, "Calls": 0, "Q per 1K t": 3}


def query_rows(corpus: Path, citation: bool = False) -> dict[str, dict[str, float]]:
    """Per-query Table 3 values; runs without both Gym judges (and, with
    ``citation``, without a citation score) are skipped."""
    rows: dict[str, dict[str, float]] = {}
    for summary in sorted(corpus.glob("*-s*/metrics/summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        gym = (data.get("official") or {}).get("deep_research_gym") or {}
        quality, kpr = gym.get("quality") or {}, gym.get("kpr") or {}
        cite = gym.get("citation") or {}
        if not quality.get("n") or not kpr.get("n") or (citation and not cite.get("n")):
            continue
        crit = quality.get("per_criterion") or {}
        row = {c: float(crit[c]["average_rating"]) for c in CRITERIA}
        row["KPR"] = float(kpr["average_support_rate"])
        row["Overall"] = float(quality["average_normalized_score"])
        if citation:
            row["Citation"] = float(cite["average_citation_score"])
        row["Tokens (K)"] = float(data["mean_tokens"]) / 1000.0
        row["Latency (s)"] = float(data["mean_wall_s"])
        row["Calls"] = float(data["mean_n_searches"])
        row["Q per 1K t"] = row["Overall"] / row["Tokens (K)"]
        rows[summary.parent.parent.name] = row
    return rows


def mean_row(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    cols = CITATION_COLUMNS if all("Citation" in r for r in rows.values()) else COLUMNS
    out = {c: st.mean(r[c] for r in rows.values()) for c in cols}
    # Q per 1K t of the means, as in the paper's existing rows.
    out["Q per 1K t"] = out["Overall"] / out["Tokens (K)"]
    return out


def fmt(col: str, v: float) -> str:
    return f"{v:.{DECIMALS.get(col, 1)}f}"


def bootstrap_ci(diffs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+", help='"Label=path/to/corpus"')
    ap.add_argument("--baseline", help="Label of the corpus to pair the others against")
    ap.add_argument("--citation", action="store_true",
                    help="Add a Citation column (needs the Gym citation judge in every run)")
    args = ap.parse_args()
    columns = CITATION_COLUMNS if args.citation else COLUMNS

    corpora: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.corpora:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected Label=path, got {spec!r}")
        rows = query_rows(Path(path), citation=args.citation)
        if not rows:
            raise SystemExit(f"{path}: no run with all the requested Gym judge scores")
        corpora[label] = rows

    print("| Approach | n | " + " | ".join(columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for label, rows in corpora.items():
        m = mean_row(rows)
        print(f"| {label} | {len(rows)} | " + " | ".join(fmt(c, m[c]) for c in columns) + " |")

    print("\nLaTeX (paste into Table 3):")
    for label, rows in corpora.items():
        m = mean_row(rows)
        print(f"{label} & " + " & ".join(fmt(c, m[c]) for c in columns) + r" \\")

    if not args.baseline:
        return
    if args.baseline not in corpora:
        raise SystemExit(f"--baseline {args.baseline!r} is not one of {list(corpora)}")
    base = corpora[args.baseline]
    for label, rows in corpora.items():
        if label == args.baseline:
            continue
        shared = sorted(set(base) & set(rows))
        print(f"\n{label} vs {args.baseline} (paired over {len(shared)} queries)")
        print("| Metric | mean diff | 95% CI | better / worse / tie |")
        print("|---|---|---|---|")
        paired = ("Overall", "KPR", "Support") + (("Citation",) if args.citation else ()) + ("Tokens (K)", "Latency (s)")
        for col in paired:
            diffs = [rows[q][col] - base[q][col] for q in shared]
            lo, hi = bootstrap_ci(diffs) if len(diffs) > 1 else (diffs[0], diffs[0])
            sign = -1 if col in ("Tokens (K)", "Latency (s)") else 1
            better = sum(1 for d in diffs if sign * d > 1e-9)
            worse = sum(1 for d in diffs if sign * d < -1e-9)
            print(f"| {col} | {st.mean(diffs):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {better} / {worse} / {len(diffs) - better - worse} |")


if __name__ == "__main__":
    main()
