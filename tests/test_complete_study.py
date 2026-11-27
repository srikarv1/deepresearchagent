"""Resume integrity tests: all provider calls are replaced; no API usage."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location('complete_study', ROOT / 'scripts/complete_study.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def report(run, q='170566', budget=100000, arm='none'):
    write_json(run / 'trajectories' / f'{q}.json', {
        'report': {'article': 'the same completed report'}, 'error': None,
        'final_stats': {
            'research_budget': {'research_budget': budget, 'output_weight': 5, 'force_full': True,
                                'excluded_usage_tags': ['report_generation'] + (['orchestrator'] if arm in ('bc', 'dpo') else [])},
            'orchestration': {'GR_CONTEXT_BUDGET_TOKENS': '50000', 'GR_FINAL_ITEM_MAX_WORDS': '0', 'GR_ORCH_SEED': '0'},
            'report_length': {'target_words': 3000},
            'writer_context': {'writer_max_input_tokens': 250000, 'word_cap': None, 'hard_cap_trimmed': False},
            'tokens_by_model': {'gpt-6.1-sol-1gtjfzlv8i': {'input': 10, 'output': 1}},
        }})


def scores(run):
    write_json(run / 'metrics/summary.json', {'official': {'deep_research_gym': {
        'quality': {'n': 1, 'average_normalized_score': 62,
                    'per_criterion': {k: {'average_rating': 6} for k in ('Clarity', 'Breadth', 'Depth', 'Support')}},
        'kpr': {'n': 1, 'average_support_rate': 60, 'average_contradicted_rate': 10},
        'citation': {'n': 1, 'average_citation_score': 70}}}})


@pytest.fixture
def job():
    return {'bench': 'drg', 'setting': 'equal_100k', 'arm': 'none', 'q': '170566', 'budget': 100000, 'force': True}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    for rel in ('skills/deep_research.py', 'orchestration/policies.py', 'orchestration/serialize.py', 'utils/llm.py'):
        path = root / 'third_party/gpt-researcher/gpt_researcher' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# frozen source\n')
    return root


def test_query_expansion_is_nested_and_excludes_recorded_drg_training(study):
    small, large = study.select_ids('drg', 10), study.select_ids('drg', 50)
    split = study.read(ROOT / 'data/benchmarks/deep_research_gym/splits.json')
    assert large[:10] == small
    assert len(set(large)) == 50
    assert not set(large) & set(map(str, split['train'] + split['val']))
    assert len(study.select_ids('drb', 47)) == 47
    with pytest.raises(ValueError):
        study.select_ids('drb', 50)


def test_planner_counts_existing_reports_not_names(study, source, tmp_path, job):
    run = study.run_path(source / 'runs/clean_drg', job)
    report(run)
    scores(run)
    proposed = study.plan(source, tmp_path / 'out', {'drg': 10, 'drb': 10}, 250000)
    assert proposed['summary']['drg']['reusable_reports'] == 1
    assert proposed['summary']['drg']['new_reports'] == 219
    assert proposed['summary']['drg']['missing_judge_passes'] == 659
    assert proposed['summary']['drb']['new_reports'] == 220


@pytest.mark.parametrize('field,value', [('research_budget', 150000), ('output_weight', 1), ('force_full', False)])
def test_changed_research_protocol_is_rejected(study, tmp_path, job, field, value):
    run = tmp_path / 'run'
    report(run)
    path = run / 'trajectories/170566.json'
    data = study.read(path)
    data['final_stats']['research_budget'][field] = value
    write_json(path, data)
    assert not study.compatible(run, job)


def test_wrong_model_or_evidence_target_cannot_be_reused(study, tmp_path, job):
    run = tmp_path / 'run'
    report(run)
    path = run / 'trajectories/170566.json'
    data = study.read(path)
    data['final_stats']['orchestration']['GR_CONTEXT_BUDGET_TOKENS'] = '100000'
    write_json(path, data)
    assert not study.compatible(run, job)
    report(run)
    data = study.read(path)
    data['final_stats']['tokens_by_model'] = {'different-model': {}}
    write_json(path, data)
    assert not study.compatible(run, job)


def test_import_preserves_sources_and_only_links_completed_judging(study, source, tmp_path, job):
    origin = source / 'runs/clean_drg'
    run = study.run_path(origin, job)
    report(run)
    scores(run)
    second = study.run_path(origin, job, 2)
    report(second)
    before = (run / 'trajectories/170566.json').read_bytes()
    out = tmp_path / 'out'
    study.import_finished(source, out, {'jobs': [job]})
    assert study.run_path(out / 'drg', job).is_symlink()
    assert not study.run_path(out / 'drg', job, 2).is_symlink()
    assert (run / 'trajectories/170566.json').read_bytes() == before


def test_stale_pass_cannot_be_imported(study, source, tmp_path, job):
    origin = source / 'runs/clean_drg'
    report(study.run_path(origin, job))
    stale = study.run_path(origin, job, 2)
    report(stale)
    path = stale / 'trajectories/170566.json'
    data = study.read(path)
    data['report']['article'] = 'another report'
    write_json(path, data)
    with pytest.raises(ValueError, match='different report'):
        study.import_finished(source, tmp_path / 'out', {'jobs': [job]})


def test_completed_generation_never_calls_provider(study, tmp_path, job, monkeypatch):
    run = study.run_path(tmp_path, job)
    report(run)
    monkeypatch.setattr(study, 'invoke', lambda *a: pytest.fail('Completed generation was executed'))
    assert study.generate(job, tmp_path, tmp_path / 'config', {}, {}, 250000, 10)


def test_failed_generation_retries_and_clears_inherited_policy(study, tmp_path, job, monkeypatch):
    calls = []
    def invoke(cmd, env, log, timeout):
        calls.append((cmd, env))
        assert not any('private-key' in arg for arg in cmd)
        assert 'GR_ORCH_LLM_BASE_URL' not in env
        assert env['GR_RESEARCH_BUDGET'] == '100000'
        if len(calls) == 2:
            report(study.run_path(tmp_path, job))
            return 0
        return 1
    monkeypatch.setattr(study, 'invoke', invoke)
    assert study.generate(job, tmp_path, tmp_path / 'config', {'GR_ORCH_LLM_BASE_URL': 'private-key'}, {}, 250000, 10)
    assert len(calls) == 2


def test_only_missing_judge_pass_is_executed(study, tmp_path, job, monkeypatch):
    (tmp_path / 'config').write_text('{}')
    for k in (1, 3):
        run = study.run_path(tmp_path, job, k)
        report(run)
        scores(run)
    called = []
    def invoke(cmd, env, log, timeout):
        called.append(cmd)
        dest = Path(cmd[4])
        assert '_pass2' in str(dest)
        scores(dest)
        return 0
    monkeypatch.setattr(study, 'invoke', invoke)
    study.judge(job, tmp_path, tmp_path / 'config', {'JUDGE_API_KEY': 'private', 'JUDGE_BASE_URL': 'endpoint'}, None, 10)
    assert len(called) == 1
    assert all(study.score_ready(study.run_path(tmp_path, job, k), 'drg') for k in (1, 2, 3))


def test_partial_judge_retries_only_missing_metric(study, tmp_path, job, monkeypatch):
    import yaml
    config = tmp_path / 'config'
    config.write_text('{}')
    for k in (1, 2, 3):
        report(study.run_path(tmp_path, job, k))
        scores(study.run_path(tmp_path, job, k))
    path = study.run_path(tmp_path, job) / 'metrics/summary.json'
    old = study.read(path)
    old['official']['deep_research_gym']['kpr'] = {'n': 0}
    write_json(path, old)
    called = []
    def invoke(cmd, env, log, timeout):
        called.append(cmd)
        flags = yaml.safe_load(Path(cmd[-1]).read_text())['eval']['deep_research_gym']
        assert flags == {'run_quality': False, 'run_kpr': True, 'run_citation': False}
        write_json(path, {'official': {'deep_research_gym': {
            'quality': None, 'citation': None,
            'kpr': {'n': 1, 'average_support_rate': 61, 'average_contradicted_rate': 9}}}})
        return 0
    monkeypatch.setattr(study, 'invoke', invoke)
    study.judge(job, tmp_path, config, {'JUDGE_API_KEY': 'private', 'JUDGE_BASE_URL': 'endpoint'}, None, 10)
    assert len(called) == 1
    assert study.score_ready(study.run_path(tmp_path, job), 'drg')
    assert study.read(path)['official']['deep_research_gym']['quality'] == old['official']['deep_research_gym']['quality']


def test_finished_report_is_not_replaced_after_nonzero_worker_exit(study, tmp_path, job, monkeypatch):
    calls = []
    def invoke(*args):
        calls.append(1)
        report(study.run_path(tmp_path, job))
        return 17
    monkeypatch.setattr(study, 'invoke', invoke)
    assert study.generate(job, tmp_path, tmp_path / 'config', {}, {}, 250000, 10)
    assert len(calls) == 1


def test_preflight_collects_missing_resources_without_paid_calls(study, source, monkeypatch):
    def unavailable(*args, **kwargs):
        raise PermissionError('process inventory blocked')
    monkeypatch.setattr(study.subprocess, 'run', unavailable)
    with pytest.raises(ValueError) as err:
        study.preflight(source, {'summary': {'drb': {'new_reports': 1}}}, {})
    assert 'GR_ORCH_LLM_BASE_URL is missing' in str(err.value)
    assert 'Tavily' in str(err.value)
    assert 'Process inventory unavailable' in str(err.value)


def test_cancellation_stops_provider_launches(study, tmp_path, monkeypatch):
    study.STOP_REQUESTED.set()
    monkeypatch.setattr(study.subprocess, 'Popen', lambda *a, **k: pytest.fail('Launched after cancellation'))
    assert study.invoke(['unused'], {}, tmp_path / 'log', 10) == -1


def test_interrupt_signals_active_process_groups(study, monkeypatch):
    signaled = []
    study.ACTIVE_PROCESSES[123] = object()
    monkeypatch.setattr(study.os, 'killpg', lambda pid, sig: signaled.append(pid))
    with pytest.raises(KeyboardInterrupt):
        study.interrupt_run(None, None)
    assert signaled == [123]
    assert study.STOP_REQUESTED.is_set()


def test_frozen_protocol_allows_extension_but_rejects_source_drift(study, tmp_path):
    old = {'protocol': {'source': 'hash-a'}, 'queries': {'drg': ['1']}}
    study.freeze(tmp_path, old)
    study.freeze(tmp_path, {'protocol': old['protocol'], 'queries': {'drg': ['1', '2']}})
    with pytest.raises(ValueError, match='changed'):
        study.freeze(tmp_path, {'protocol': {'source': 'hash-b'}, 'queries': {'drg': ['1', '2']}})
    assert study.read(tmp_path / 'PLAN.json')['protocol'] == old['protocol']


def test_reporting_contains_every_metric_and_full_token_split(study, tmp_path):
    mod = study.load_results('drg')
    run = tmp_path / 'equal_100k/none/170566-s0'
    report(run)
    scores(run)
    target = tmp_path / 'reports'
    mod.table(tmp_path, target)
    md, tex = (target / 'RESULTS.md').read_text(), (target / 'results.tex').read_text()
    assert all(metric in md and metric in tex for metric in ('Clarity', 'Breadth', 'Depth', 'Support', 'KPR', 'Citation'))
    assert all(label in md for label in ('Research in', 'Research out', 'Orch in', 'Orch out', 'Writer in', 'Writer out'))
    assert 'every method spends the full' not in tex
    assert 'Matched allowance' in tex and 'assumed rates' in tex and '250K input guard' in tex


def test_learned_comparison_pairs_both_prompted_baselines(study, tmp_path):
    root = tmp_path / 'drg'
    for arm, quality in [('bc', 62), ('dpo', 61), ('prompted', 60), ('prompted_scores', 63)]:
        for q in ('1', '2', '3', '4'):
            for suffix in ('', '_pass2', '_pass3'):
                if arm == 'prompted_scores' and q == '4' and suffix == '_pass3':
                    continue
                run = root / f'equal_100k{suffix}' / arm / f'{q}-s0'
                report(run, q=q, arm=arm)
                scores(run)
                path = run / 'metrics/summary.json'
                data = study.read(path)
                data['official']['deep_research_gym']['quality']['average_normalized_score'] = quality
                write_json(path, data)
    study.offline_reports(tmp_path / 'source', tmp_path, {'queries': {'drg': ['1', '2', '3', '4']}})
    comparison = (root / 'LEARNED_COMPARISON.md').read_text()
    for arm, direct, scored in [('bc', '+2.00', '-1.00'), ('dpo', '+1.00', '-2.00')]:
        assert f'{arm} minus prompted, Overall: {direct} [{direct}, {direct}], n=4' in comparison
        assert f'{arm} minus prompted_scores, Overall: {scored} [{scored}, {scored}], n=3' in comparison
