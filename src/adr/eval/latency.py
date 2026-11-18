"""Where a run's wall time goes.

Raw wall time mixes the method with the machine it ran on: web search, page
fetching and embedding calls depend on the network, and Azure Flex adds queue
time. Model calls also overlap (sub-queries run concurrently), so per-call
durations cannot simply be summed.

``latency_breakdown`` splits one run's wall time using the per-call LLM log
(``final_stats.llm_calls``) and, when the fork records them, search intervals:

  llm_active_s           time during which at least one model call was running
  orchestrator_active_s  the same for keep/allocate/stop decisions only
                         (``usage_tag == "orchestrator"``; prompted and learned)
  capacity_wait_s        Flex ``429 no_capacity`` waiting inside model calls
  search_active_s        time during which at least one search was running
                         (None for runs recorded before search intervals)
  other_s                wall time with no model call and no search running:
                         page fetching, embeddings, local processing, and
                         search when ``search_active_s`` is None

``llm_active_s`` and ``orchestrator_active_s`` depend on the model endpoints,
not on the laptop that ran the job, so they are the fair way to compare
methods that were run on different machines.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

ORCHESTRATOR_TAG = "orchestrator"


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Length of the union of ``(start, end)`` intervals."""
    total = 0.0
    cur_s: Optional[float] = None
    cur_e: Optional[float] = None
    for s, e in sorted((float(a), float(b)) for a, b in intervals if b is not None):
        if cur_e is None or s > cur_e:
            if cur_e is not None and cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None and cur_s is not None:
        total += cur_e - cur_s
    return total


def _call_interval(call: dict[str, Any]) -> Optional[tuple[float, float]]:
    start = call.get("start")
    if start is None:
        return None
    end = call.get("end")
    if end is None:
        dur = call.get("duration_s")
        if dur is None:
            return None
        end = float(start) + float(dur)
    return float(start), float(end)


def latency_breakdown(
    wall_s: float,
    llm_calls: Iterable[dict[str, Any]],
    search_intervals: Optional[Iterable[tuple[float, float]]] = None,
) -> dict[str, Any]:
    calls = list(llm_calls or [])
    llm_iv = [iv for iv in (_call_interval(c) for c in calls) if iv]
    orch_iv = [
        iv for c in calls if c.get("usage_tag") == ORCHESTRATOR_TAG
        for iv in [_call_interval(c)] if iv
    ]
    search_iv = list(search_intervals) if search_intervals else []
    llm_active = union_seconds(llm_iv)
    busy = union_seconds(llm_iv + search_iv)
    return {
        "wall_s": round(float(wall_s), 3),
        "llm_active_s": round(llm_active, 3),
        "orchestrator_active_s": round(union_seconds(orch_iv), 3),
        "n_orchestrator_calls": len(orch_iv),
        "capacity_wait_s": round(sum(float(c.get("capacity_wait_s") or 0.0) for c in calls), 3),
        "search_active_s": round(union_seconds(search_iv), 3) if search_iv else None,
        "other_s": round(max(0.0, float(wall_s) - busy), 3),
    }
