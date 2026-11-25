#!/usr/bin/env python3
"""Split a BC-pairs corpus into train/dev by query and export the OpenAI chat
rows the training notebook (pilot_bc_qwen_lora.ipynb) consumes.

    python scripts/export_bc_oai.py data/bc/bc_pairs_gym_v1.jsonl --oai-dir data/bc/gym

Writes next to the corpus:  <stem>.train.jsonl, <stem>.dev.jsonl
                            (messages, query_id, run_id, round_id per row)
and into --oai-dir:          oai_train.jsonl, oai_dev.jsonl  ({"messages": ...})

Dev = the ``--n-dev-queries`` highest query ids (every row of those queries),
which is the rule the DeepResearch Bench corpus (bc_pairs_d3b4_v1) was split
with; rows keep the corpus order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _id_key(qid: str):
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def export(corpus: Path, oai_dir: Path, n_dev_queries: int) -> dict[str, int]:
    rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = sorted({r["query_id"] for r in rows}, key=_id_key)
    if n_dev_queries >= len(queries):
        raise ValueError(f"--n-dev-queries {n_dev_queries} leaves no training query ({len(queries)} queries)")
    dev_q = set(queries[-n_dev_queries:]) if n_dev_queries else set()
    train = [r for r in rows if r["query_id"] not in dev_q]
    dev = [r for r in rows if r["query_id"] in dev_q]

    def slim(r: dict) -> dict:
        return {k: r[k] for k in ("messages", "query_id", "run_id", "round_id")}

    stem = corpus.with_suffix("")
    oai_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        stem.with_name(stem.name + ".train.jsonl"): [slim(r) for r in train],
        stem.with_name(stem.name + ".dev.jsonl"): [slim(r) for r in dev],
        oai_dir / "oai_train.jsonl": [{"messages": r["messages"]} for r in train],
        oai_dir / "oai_dev.jsonl": [{"messages": r["messages"]} for r in dev],
    }
    for path, out_rows in outputs.items():
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    return {
        "rows": len(rows), "queries": len(queries),
        "train_rows": len(train), "train_queries": len(queries) - len(dev_q),
        "dev_rows": len(dev), "dev_queries": sorted(dev_q, key=_id_key),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path, help="bc_pairs_*.jsonl from `adr bc-pairs`")
    ap.add_argument("--oai-dir", type=Path, required=True, help="where oai_train.jsonl / oai_dev.jsonl go")
    ap.add_argument("--n-dev-queries", type=int, default=2)
    args = ap.parse_args()
    stats = export(args.corpus, args.oai_dir, args.n_dev_queries)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
