from adr.core.types import (
    ActionType,
    OrchestratorAction,
    Query,
    Report,
    StepRecord,
    TokenUsage,
    Trajectory,
)
from adr.eval.trajectory_labels import annotate_trajectory, first_mention


def _prune(step: int, round_id: int, *, new, kept, pruned, extra=None) -> StepRecord:
    payload = {
        "round_id": round_id,
        "new_item_ids": [x[0] for x in new],
        "kept_item_ids": [x[0] for x in kept],
        "pruned_item_ids": [x[0] for x in pruned],
        "new_docids": [x[1] for x in new if x[1]],
        "kept_docids": [x[1] for x in kept if x[1]],
        "pruned_docids": [x[1] for x in pruned if x[1]],
        "search_calls": 2,
        "decision_type": "continue",
        "search_queries": [f"q{round_id}"],
    }
    if extra:
        payload.update(extra)
    return StepRecord(
        step=step,
        action=OrchestratorAction(type=ActionType.PRUNE, evidence_ids=[x[0] for x in pruned]),
        stats_before={"n_evidence": step - 1, "n_retained": 0},
        extra=payload,
        tokens=TokenUsage(total_tokens=100),
        latency_s=1.5,
    )


def test_first_mention_finds_date_answer():
    hit = first_mention("He wrote the book between 1988-96 in Nairobi.", "1988-96")
    assert hit is not None
    assert hit["matched"] == "1988-96"
    assert "Nairobi" in hit["snippet"]


def test_oracle_tracks_first_seen_then_discarded_then_kept():
    query = Query(
        id="1",
        text="Which years?",
        dataset="browsecomp_plus",
        language="en",
        metadata={
            "answer": "1988-96",
            "split": "test",
            "evidence_docs": [{"docid": "2882"}, {"docid": "68543"}],
            "gold_docs": [{"docid": "68543"}],
        },
    )
    items = [
        {
            "id": "e_noise",
            "url": "bcp://9",
            "text": "unrelated page",
            "subquery": "noise",
            "tree_depth": 1,
            "retrieval_round": 1,
        },
        {
            "id": "e_ans",
            "url": "bcp://2882",
            "text": "The dates are 1988-96 according to the interview.",
            "subquery": "author years",
            "tree_depth": 1,
            "retrieval_round": 1,
        },
        {
            "id": "e_gold",
            "url": "bcp://68543",
            "text": "United States of Africa. He published 1988-96.",
            "subquery": "book title",
            "tree_depth": 2,
            "retrieval_round": 2,
        },
    ]
    traj = Trajectory(
        query=query,
        steps=[
            _prune(1, 1, new=[("e_noise", "9"), ("e_ans", "2882")], kept=[("e_noise", "9")], pruned=[("e_ans", "2882")]),
            _prune(2, 2, new=[("e_gold", "68543")], kept=[("e_gold", "68543")], pruned=[]),
            StepRecord(step=3, action=OrchestratorAction(type=ActionType.WRITE)),
        ],
        report=Report(article="Exact Answer: 1988-96"),
        final_stats={"retrieved_docids": ["2882", "68543", "9"]},
    )
    annotate_trajectory(traj, items=items, run={"orchestrator": "topk"})
    assert traj.labels is not None
    summary = traj.labels["summary"]
    assert summary["first_seen_answer"]["round"] == 1
    assert summary["first_seen_answer"]["docid"] == "2882"
    assert "1988-96" in summary["first_seen_answer"]["mention"]["snippet"]
    assert summary["first_kept_gold"]["round"] == 2
    assert summary["first_kept_gold"]["docid"] == "68543"
    assert summary["discarded_answer"] is False  # later kept via gold page
    assert summary["keep_recall_gold"] == 1.0
    assert summary["retrieved_recall_evidence"] == 1.0
    assert summary["answer_in_report"] is True
    assert summary["searches_after_answer"] == 2
    assert traj.steps[0].extra["oracle"]["first_answer_this_round"] is True
    assert traj.steps[0].extra["oracle"]["answer_only_in_pruned"] is True
    assert "answer" not in traj.steps[0].stats_before
    assert "gold" not in traj.steps[0].stats_before
    assert traj.labels["run"]["orchestrator"] == "topk"


def test_stats_before_stays_qrel_free_on_gym():
    query = Query(id="chip", text="chips", dataset="deep_research_gym")
    traj = Trajectory(
        query=query,
        steps=[
            _prune(1, 1, new=[("a", None)], kept=[("a", None)], pruned=[]),
        ],
        final_stats={},
    )
    annotate_trajectory(
        traj,
        items=[{"id": "a", "url": "https://x", "text": "chips everywhere", "subquery": "q"}],
    )
    assert traj.steps[0].stats_before == {"n_evidence": 0, "n_retained": 0}
    assert traj.final_stats["oracle"]["n_evidence_labeled"] == 0
