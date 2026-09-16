from adr.core.state import ResearchState, evidence_id
from adr.core.types import ActionType, Evidence, OrchestratorAction, Query


def _ev(url: str, score: float, title: str = "t") -> Evidence:
    return Evidence(id=evidence_id(url, title), url=url, title=title, snippet="s", score=score)


def test_prune_drops_duplicate_urls_and_low_scores():
    state = ResearchState(Query(id="1", text="q", dataset="deep_research_bench"))
    state.add_evidence(
        [
            _ev("https://example.com/a", 0.9),
            _ev("https://example.com/a/", 0.4, title="dup"),
            _ev("https://example.com/b", 0.05),
            _ev("https://example.com/c", 0.7),
        ]
    )
    dropped = state.prune(drop_duplicates=True, min_score=0.1, max_keep=2)
    assert dropped
    urls = {ev.url.rstrip("/") for ev in state.retained()}
    assert urls == {"https://example.com/a", "https://example.com/c"}


def test_budget_exhaustion_stops_loop():
    from adr.core.types import Budget

    state = ResearchState(
        Query(id="1", text="q", dataset="x"),
        Budget(max_steps=2, max_tokens=10_000, max_latency_s=100),
    )
    state.record_step(OrchestratorAction(type=ActionType.SEARCH, query="q"))
    assert not state.should_stop()
    state.record_step(OrchestratorAction(type=ActionType.SEARCH, query="q"))
    assert state.should_stop()
    assert state.budget.used_searches == 2


def test_compact_stats_are_small_and_numeric():
    state = ResearchState(Query(id="1", text="chip shortage causes", dataset="x"))
    state.add_subtasks(["capacity", "autos"])
    state.add_evidence([_ev("https://example.com/chip", 0.8, title="chip shortage causes")])
    stats = state.compact_stats()
    assert stats["n_retained"] == 1
    assert stats["n_branches"] == 2
    assert "budget" in stats
    assert "branches" in stats
    assert isinstance(stats["query_term_coverage"], float)
