from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from adr.core.types import Query

ROOT = Path(__file__).resolve().parents[3]
DRB_QUERIES = ROOT / "data" / "benchmarks" / "deep_research_bench" / "query.jsonl"
GYM_QUERIES = (
    ROOT / "data" / "benchmarks" / "deep_research_gym" / "researchy_queries_sample_doc_click.jsonl"
)


class DatasetName(str, Enum):
    DEEP_RESEARCH_BENCH = "deep_research_bench"
    DEEP_RESEARCH_GYM = "deep_research_gym"
    BROWSECOMP = "browsecomp"


def load_queries(
    dataset: str | DatasetName,
    *,
    path: str | Path | None = None,
    language: str | None = None,
    limit: int | None = None,
    query_ids: list[str] | None = None,
    sample: int | None = None,
) -> list[Query]:
    name = DatasetName(dataset)
    if name is DatasetName.BROWSECOMP:
        from adr.datasets.browsecomp import load_browsecomp

        return load_browsecomp(path, limit=limit, query_ids=query_ids, sample=sample)

    source = Path(path) if path else (DRB_QUERIES if name is DatasetName.DEEP_RESEARCH_BENCH else GYM_QUERIES)
    if not source.exists():
        raise FileNotFoundError(f"Query file not found: {source}")

    wanted = {str(x) for x in query_ids} if query_ids else None
    queries: list[Query] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query = _row_to_query(name, row)
        if language and query.language != language:
            continue
        if wanted is not None and query.id not in wanted:
            continue
        queries.append(query)
        if limit is not None and len(queries) >= limit:
            break
    return queries


def _row_to_query(dataset: DatasetName, row: dict) -> Query:
    if dataset is DatasetName.DEEP_RESEARCH_BENCH:
        return Query(
            id=str(row["id"]),
            text=row["prompt"],
            dataset=dataset.value,
            language=row.get("language", "en"),
            topic=row.get("topic"),
            metadata={"raw": row},
        )
    return Query(
        id=str(row["id"]),
        text=row["query"],
        dataset=dataset.value,
        language="en",
        metadata={"raw": row},
    )
