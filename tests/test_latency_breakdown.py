import json

from adr.eval.latency import latency_breakdown, union_seconds


def test_union_counts_overlap_once():
    assert union_seconds([(0, 10), (5, 15), (20, 25)]) == 20
    assert union_seconds([]) == 0
    assert union_seconds([(3, 4), (0, 1)]) == 2


def test_parallel_searches_never_exceed_wall():
    # Four sub-query searches run at the same time, 30 s each, inside a 40 s run.
    searches = [(0, 30), (1, 31), (2, 32), (3, 33)]
    bd = latency_breakdown(40.0, [], searches)
    assert bd["search_active_s"] == 33
    assert bd["other_s"] == 7
    assert sum(e - s for s, e in searches) > 40  # the old summed figure


def test_orchestrator_time_split_from_other_model_calls():
    calls = [
        {"start": 0, "end": 10, "usage_tag": "deep_research_plan", "capacity_wait_s": 2},
        {"start": 10, "end": 13, "usage_tag": "orchestrator"},
        {"start": 12, "end": 20, "usage_tag": "report_generation"},
    ]
    bd = latency_breakdown(30.0, calls)
    assert bd["llm_active_s"] == 20
    assert bd["orchestrator_active_s"] == 3
    assert bd["n_orchestrator_calls"] == 1
    assert bd["capacity_wait_s"] == 2
    assert bd["search_active_s"] is None
    assert bd["other_s"] == 10


def test_duration_used_when_end_missing():
    bd = latency_breakdown(10.0, [{"start": 0, "duration_s": 4}])
    assert bd["llm_active_s"] == 4


def test_script_reads_existing_runs(tmp_path, capsys):
    from importlib import util
    from pathlib import Path

    spec = util.spec_from_file_location(
        "latency_breakdown_script", Path(__file__).resolve().parents[1] / "scripts" / "latency_breakdown.py"
    )
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    traj_dir = tmp_path / "bc" / "q1-s0" / "trajectories"
    traj_dir.mkdir(parents=True)
    calls = [{"start": 0, "end": 5, "usage_tag": "orchestrator"}, {"start": 5, "end": 9, "usage_tag": "x"}]
    (traj_dir / "q1.json").write_text(json.dumps({"final_stats": {"agent_wall_s": 20, "llm_calls": calls}}))
    assert mod.main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "bc" in out and "9.0" in out and "5.0" in out
