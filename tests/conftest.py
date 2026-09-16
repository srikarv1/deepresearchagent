from __future__ import annotations

from pathlib import Path

import pytest

from adr.core.types import Budget, Query
from adr.llm.mock import MockLLM
from adr.tools.search import MockSearch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_corpus() -> Path:
    return ROOT / "data" / "fixtures" / "corpus.json"


@pytest.fixture
def mock_search(fixture_corpus: Path) -> MockSearch:
    return MockSearch(path=fixture_corpus)


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM(replies=["ok"])


@pytest.fixture
def drb_query() -> Query:
    return Query(
        id="chip",
        text="why is there a chip shortage",
        dataset="deep_research_bench",
        language="en",
    )


@pytest.fixture
def tight_budget() -> Budget:
    return Budget(max_steps=6, max_searches=3, max_reads=3, max_tokens=4000, max_evidence=8)
