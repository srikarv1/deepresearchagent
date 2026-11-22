import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("budget_replay", Path(__file__).parents[1] / "scripts" / "budget_replay.py")
br = importlib.util.module_from_spec(spec)
sys.modules["budget_replay"] = br  # dataclasses look their module up here
spec.loader.exec_module(br)


def call(tag, tokens):
    return {"usage_tag": tag, "input_tokens": tokens, "output_tokens": 0}


def test_units_splits_the_call_log_into_steps_and_rounds():
    calls = [call("deep_research_plan", 100), call("deep_search_query_gen", 5)]
    calls += [c for _ in range(4) for c in (call("subquery_generation", 10), call("deep_process_results", 20))]
    calls += [call("deep_search_query_gen", 5), call("subquery_generation", 10), call("deep_process_results", 20)]
    calls += [call("report_generation", 300)]
    plan, steps, report, per_round = br.units({"llm_calls": calls})
    assert (plan, report, per_round) == (100, 300, [4, 1])
    assert steps == [35, 30, 30, 30, 35]  # each round's query-gen call goes to its first step


def item(iid, sq):
    return br.Item(iid, f"https://x/{iid}", "text", 1, 1, None, sq)


def test_search_queries_are_split_evenly_over_a_rounds_steps_in_order():
    items = {f"i{k}": item(f"i{k}", f"q{k // 2}") for k in range(8)}  # 4 search queries, 2 items each
    raw = {"rounds": [{"new_item_ids": [f"i{k}" for k in range(8)]}]}
    assert br.subquery_order(raw, items, [2]) == [{"q0", "q1"}, {"q2", "q3"}]


def test_learnings_are_the_lines_before_the_first_item():
    ctx = "fact one [Source: https://a]\nfact two\nSource: https://a\nContent: page\n"
    assert br.split_learnings(ctx) == ["fact one [Source: https://a]", "fact two"]
