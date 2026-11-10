"""Offline regression tests for Srikar's pinned Tavily key-pool integration.

Every key and response in this module is synthetic. Network access is replaced
at the retriever's ``requests.post`` boundary, and the CSV is created under the
pytest temporary directory rather than read from the user's configured pool.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import requests


FORK_ROOT = Path("third_party/gpt-researcher").resolve()
if not (FORK_ROOT / "gpt_researcher/retrievers/tavily/key_manager.py").is_file():
    pytest.skip("pinned GPT-Researcher checkout is not available", allow_module_level=True)
sys.path.insert(0, str(FORK_ROOT))

from gpt_researcher.retrievers.tavily import key_manager as key_manager_module  # noqa: E402
from gpt_researcher.retrievers.tavily import tavily_search as tavily_module  # noqa: E402


FAKE_KEYS = [f"offline-fake-key-{index}" for index in range(1, 7)]


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"offline HTTP {self.status_code}", response=self)


def _write_fake_csv(path: Path, keys: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["api_key"])
        writer.writeheader()
        writer.writerows({"api_key": key} for key in keys)


@pytest.fixture
def fake_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    csv_path = tmp_path / "synthetic-tavily.csv"
    _write_fake_csv(csv_path, FAKE_KEYS)
    monkeypatch.setenv("TAVILY_KEYS_CSV", str(csv_path))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    key_manager_module.TavilyKeyManager._instance = None
    yield csv_path
    key_manager_module.TavilyKeyManager._instance = None


def _fake_post_sequence(monkeypatch: pytest.MonkeyPatch, responses: list[FakeResponse]):
    calls: list[dict] = []
    pending = list(responses)

    def fake_post(url, *, data, headers, timeout):
        assert pending, "unexpected extra Tavily request"
        calls.append(
            {
                "url": url,
                "data": json.loads(data),
                "headers": dict(headers),
                "timeout": timeout,
            }
        )
        return pending.pop(0)

    monkeypatch.setattr(tavily_module.requests, "post", fake_post)
    return calls


def _searcher() -> tavily_module.TavilySearch:
    return tavily_module.TavilySearch(
        "original offline query",
        topic="news",
        query_domains=["example.com"],
    )


def test_tavily_keys_csv_loads_synthetic_pool_from_environment(fake_pool):
    # The fork-local default must remain absent so TAVILY_KEYS_CSV is the source.
    default_csv = FORK_ROOT / "env/tavily.csv"
    assert not default_csv.exists()

    manager = key_manager_module.TavilyKeyManager()
    assert manager.current_key == FAKE_KEYS[0]
    assert manager.remaining == len(FAKE_KEYS)


def test_http_432_rotates_then_retries_the_identical_request(monkeypatch, fake_pool):
    calls = _fake_post_sequence(
        monkeypatch,
        [FakeResponse(432), FakeResponse(200, {"results": [{"url": "https://example.com"}]})],
    )
    searcher = _searcher()

    result = searcher._search(
        "preserve this query",
        search_depth="advanced",
        topic="news",
        days=7,
        max_results=4,
        include_domains=["example.com"],
        exclude_domains=["blocked.example"],
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        use_cache=False,
    )

    assert result["results"]
    assert [call["data"]["api_key"] for call in calls] == FAKE_KEYS[:2]
    first_without_key = {k: v for k, v in calls[0]["data"].items() if k != "api_key"}
    second_without_key = {k: v for k, v in calls[1]["data"].items() if k != "api_key"}
    assert first_without_key == second_without_key
    assert first_without_key == {
        "query": "preserve this query",
        "search_depth": "advanced",
        "topic": "news",
        "days": 7,
        "include_answer": True,
        "include_raw_content": True,
        "max_results": 4,
        "include_domains": ["example.com"],
        "exclude_domains": ["blocked.example"],
        "include_images": True,
        "use_cache": False,
    }


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_exhaustion_statuses_rotate_to_the_next_key(monkeypatch, fake_pool, status):
    calls = _fake_post_sequence(
        monkeypatch,
        [FakeResponse(status), FakeResponse(200, {"results": []})],
    )

    assert _searcher()._search("q", max_results=2) == {"results": []}
    assert [call["data"]["api_key"] for call in calls] == FAKE_KEYS[:2]


def test_multiple_consecutive_exhausted_keys_reach_first_working_key(monkeypatch, fake_pool):
    calls = _fake_post_sequence(
        monkeypatch,
        [FakeResponse(432), FakeResponse(429), FakeResponse(200, {"results": [1]})],
    )

    assert _searcher()._search("same query", max_results=6) == {"results": [1]}
    assert [call["data"]["api_key"] for call in calls] == FAKE_KEYS[:3]
    assert [call["data"]["query"] for call in calls] == ["same query"] * 3
    assert [call["data"]["max_results"] for call in calls] == [6] * 3


def test_manager_never_cycles_back_after_pool_exhaustion(fake_pool):
    manager = key_manager_module.TavilyKeyManager()

    assert manager.current_key == FAKE_KEYS[0]
    for expected in FAKE_KEYS[1:]:
        assert manager.rotate() == expected
    assert manager.rotate() is None
    assert manager.rotate() is None
    assert manager.current_key == FAKE_KEYS[-1]
    assert manager.remaining == 1


def test_all_keys_exhausted_becomes_empty_search_results(monkeypatch, fake_pool, capsys):
    calls = _fake_post_sequence(
        monkeypatch,
        [
            FakeResponse(432),
            FakeResponse(429),
            FakeResponse(401),
            FakeResponse(402),
            FakeResponse(403),
            FakeResponse(432),
        ],
    )
    searcher = _searcher()

    # Public search() catches the final HTTPError and lets research continue.
    assert searcher.search(max_results=5) == []
    assert [call["data"]["api_key"] for call in calls] == FAKE_KEYS
    assert [call["data"]["query"] for call in calls] == ["original offline query"] * len(FAKE_KEYS)
    assert "offline HTTP 432" in capsys.readouterr().out

    manager = key_manager_module.TavilyKeyManager()
    assert manager.current_key == FAKE_KEYS[-1]
    assert manager.rotate() is None
