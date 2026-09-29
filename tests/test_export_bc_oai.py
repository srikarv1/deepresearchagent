"""scripts/export_bc_oai.py: train/dev split by query and the OpenAI chat export."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from adr.runner.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from export_bc_oai import export  # noqa: E402

DRB = ROOT / "data" / "bc"


def test_reproduces_the_committed_drb_files_byte_for_byte(tmp_path: Path):
    corpus = tmp_path / "bc_pairs_d3b4_v1.jsonl"
    shutil.copy(DRB / "bc_pairs_d3b4_v1.jsonl", corpus)
    stats = export(corpus, tmp_path / "oai", n_dev_queries=2)
    assert stats["dev_queries"] == ["79", "80"]
    for name in ("bc_pairs_d3b4_v1.train.jsonl", "bc_pairs_d3b4_v1.dev.jsonl"):
        assert (tmp_path / name).read_bytes() == (DRB / name).read_bytes(), name
    for name in ("oai_train.jsonl", "oai_dev.jsonl"):
        assert (tmp_path / "oai" / name).read_bytes() == (DRB / name).read_bytes(), name


def test_dev_is_the_highest_query_ids_and_rows_keep_order(tmp_path: Path):
    corpus = tmp_path / "bc_pairs_x.jsonl"
    rows = [
        {"query_id": q, "run_id": f"{q}-s0", "round_id": r, "reward": 0.5, "extra": 1,
         "messages": [{"role": "user", "content": f"{q}/{r}"}, {"role": "assistant", "content": "KEEP: ALL"}]}
        for q in ("12", "3", "101") for r in (1, 2)
    ]
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = export(corpus, tmp_path / "oai", n_dev_queries=1)
    assert stats["dev_queries"] == ["101"] and stats["train_queries"] == 2
    train = [json.loads(l) for l in (tmp_path / "bc_pairs_x.train.jsonl").read_text().splitlines()]
    assert [r["query_id"] for r in train] == ["12", "12", "3", "3"]
    assert list(train[0]) == ["messages", "query_id", "run_id", "round_id"]
    oai = [json.loads(l) for l in (tmp_path / "oai" / "oai_dev.jsonl").read_text().splitlines()]
    assert oai == [{"messages": r["messages"]} for r in rows if r["query_id"] == "101"]
    with pytest.raises(ValueError):
        export(corpus, tmp_path / "oai", n_dev_queries=3)
