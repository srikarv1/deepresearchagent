"""Plan offline; --execute fills only missing cells of the frozen clean protocol.

Completed source runs are read, never rewritten. Changes to generation settings
require another cohort, rather than relabeling or silently regenerating reports.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(os.environ.get('STUDY_SOURCE', REPO))
STOP_REQUESTED = threading.Event()
ACTIVE_PROCESSES = {}
ARMS = ('none', 'extractive', 'prompted', 'prompted_scores', 'bc', 'dpo')
STOPPERS = ('prompted', 'prompted_scores', 'bc', 'dpo')
PILOT = {'drg': '170566,313168,331361,390111,442352,546111,757743,758081,832775,883054'.split(','),
         'drb': [str(i) for i in range(61, 71)]}
DEV = {'drg': ['170195', '262266', '273800'], 'drb': ['71', '72', '73']}
DATA = {'drg': 'deep_research_gym/researchy_queries_sample_doc_click.jsonl',
        'drb': 'deep_research_bench/query.jsonl'}
CONFIG = {'drg': 'gym_sol_pilot.yaml', 'drb': 'drb_sol_clean.yaml'}
OFFICIAL = {'drg': 'deep_research_gym', 'drb': 'deep_research_bench'}


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def load_results(bench):
    spec = importlib.util.spec_from_file_location('clean_results', REPO / 'scripts/clean_results.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BENCH_KEY, mod.B = bench, mod.BENCHES[bench]
    return mod


def article(run: Path, q: str):
    t = read(run / 'trajectories' / f'{q}.json', {})
    report = t.get('report')
    text = report.get('article', '') if isinstance(report, dict) else report
    return text if isinstance(text, str) and text.strip() and not t.get('error') else None


def score_ready(run, bench):
    mod = load_results(bench)
    try:
        score = mod.B['scores'](run)
    except (KeyError, TypeError, ValueError):
        return False
    if score is None or not all(k in score and math.isfinite(score[k]) for k in mod.B['metrics']):
        return False
    if bench == 'drg':
        q = run.name.split('-s')[0]
        export = run / 'exports/deep_research_gym/gpt_researcher' / f'{q}.a'
        if export.exists() and export.read_text().strip() != (article(run, q) or '').strip():
            return False
    return True


def select_ids(bench, count):
    rows = [json.loads(line) for line in (REPO / 'data/benchmarks' / DATA[bench]).read_text().splitlines()
            if line.strip()]
    allowed = [str(r['id']) for r in rows if r.get('language', 'en') == 'en']
    excluded = set(DEV[bench])
    if bench == 'drg':
        splits = read(REPO / 'data/benchmarks/deep_research_gym/splits.json')
        excluded.update(map(str, splits['train'] + splits['val']))
    selected = PILOT[bench][:count]
    candidates = sorted(set(allowed) - excluded - set(selected), key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    selected += candidates[:max(0, count - len(selected))]
    if count < len(PILOT[bench]) or len(selected) != count or not set(selected) <= set(allowed):
        raise ValueError(f'{bench}: request 10..{len(set(allowed) - excluded)} questions; dev/training IDs cannot be reused')
    if set(selected) & excluded:
        raise ValueError(f'{bench}: evaluation IDs overlap calibration/training IDs')
    return selected


def specs(bench, ids):
    for budget in (100000, 150000, 200000):
        for arm in ARMS:
            for q in ids:
                yield {'bench': bench, 'setting': f'equal_{budget // 1000}k', 'arm': arm,
                       'q': q, 'budget': budget, 'force': True}
    for arm in STOPPERS:
        for q in ids:
            yield {'bench': bench, 'setting': 'stop_200k', 'arm': arm, 'q': q,
                   'budget': 200000, 'force': False}


def run_path(root, job, pass_no=1):
    setting = job['setting'] + (f'_pass{pass_no}' if pass_no > 1 else '')
    return root / setting / job['arm'] / f"{job['q']}-s0"


def compatible(run, job):
    """Reject reuse on a protocol mismatch; an existing filename is not proof."""
    if not article(run, job['q']):
        return False
    fs = read(run / 'trajectories' / f"{job['q']}.json", {}).get('final_stats', {})
    rb, orch, wc = (fs.get(k, {}) for k in ('research_budget', 'orchestration', 'writer_context'))
    expected = {'research_budget': job['budget'], 'output_weight': 5.0, 'force_full': job['force']}
    if any(rb.get(k) != v for k, v in expected.items()):
        return False
    if 'report_generation' not in rb.get('excluded_usage_tags', []):
        return False
    local = 'orchestrator' in rb.get('excluded_usage_tags', [])
    if local != (job['arm'] in ('bc', 'dpo')):
        return False
    if orch.get('GR_CONTEXT_BUDGET_TOKENS') != '50000' or orch.get('GR_FINAL_ITEM_MAX_WORDS') != '0':
        return False
    if fs.get('report_length', {}).get('target_words') != 3000:
        return False
    if wc.get('writer_max_input_tokens') != 250000 or wc.get('hard_cap_trimmed') or wc.get('word_cap') is not None:
        return False
    if orch.get('GR_ORCH_SEED') != '0':
        return False
    model_names = set(fs.get('tokens_by_model', {}))
    if not model_names or not all(m == 'gpt-6.1-sol-1gtjfzlv8i' or
                                  (job['arm'] in ('bc', 'dpo') and m.startswith('pilot-')) for m in model_names):
        return False
    if any(wc.get(k, 0) for k in ('words_cut_by_window', 'evidence_words_cut_by_window',
                                 'evidence_words_cut_by_item_cap', 'learnings_words_cut_by_share')):
        return False
    return True


def origin_for(source, output, job):
    for root in [output / job['bench'], source / 'runs' / f"clean_{job['bench']}"]:
        candidate = run_path(root, job)
        if article(candidate, job['q']):
            if not compatible(candidate, job):
                raise ValueError(f'Existing report has incompatible settings: {candidate}; use a separate cohort')
            return root
    return None


def plan(source, output, counts, writer_limit):
    if writer_limit != 250000:
        raise ValueError('Existing cohort used a 250000-token writer guard; changed settings require a new cohort')
    fork = source / 'third_party/gpt-researcher'
    paths = [fork / 'gpt_researcher' / f for f in
             ('skills/deep_research.py', 'orchestration/policies.py', 'orchestration/serialize.py', 'utils/llm.py')]
    paths += [REPO / 'configs' / CONFIG[b] for b in counts]
    paths += [REPO / 'configs/agents' / f for f in
              ('gpt_researcher_gym_gptsol.yaml', 'gpt_researcher_bench_sol_d3.yaml')]
    paths += [REPO / 'src/adr' / f for f in
              ('agents/gpt_researcher.py', 'runner/experiment.py', 'runner/config.py', 'datasets/loader.py')]
    paths += [REPO / 'data/benchmarks' / DATA[b] for b in counts]
    protocol = {'research_budgets': [100000, 150000, 200000], 'output_weight': 5,
                'evidence_target': 50000, 'depth': 3, 'breadth': 4, 'report_words': 3000,
                'writer_limit': writer_limit, 'judge': 'gpt-5-mini', 'judge_passes': 3,
                'generation_source_sha256': {str(p): digest(p) for p in paths}}
    ids = {b: select_ids(b, n) for b, n in counts.items()}
    jobs = [j for b in ids for j in specs(b, ids[b])]
    summary = {}
    for b in ids:
        reused, new, judges = 0, 0, 0
        for j in (j for j in jobs if j['bench'] == b):
            origin = origin_for(source, output, j)
            if origin is None:
                new += 1
                judges += 3
            else:
                reused += 1
                judges += sum(not score_ready(run_path(origin, j, k), b) for k in (1, 2, 3))
        summary[b] = {'questions': len(ids[b]), 'reusable_reports': reused,
                      'new_reports': new, 'missing_judge_passes': judges}
        calibration = output / b / 'thresholds.env'
        original = source / 'runs' / f'clean_{b}' / 'thresholds.env'
        summary[b]['calibration_reports_if_needed'] = 0 if calibration.exists() or original.exists() else 6
    return {'protocol': protocol, 'queries': ids, 'dev_queries': {b: DEV[b] for b in ids},
            'summary': summary, 'jobs': jobs,
            'notes': ['The 100K evidence follow-up is kept separate, never mixed into this 50K-target cohort.',
                      'Same research allowances; exact realized spend and stopping before depth 3 are not guaranteed.',
                      'BC/DPO superiority is an outcome to measure, not a software guarantee.',
                      'Additional DRG IDs exclude recorded train/validation splits; team-wide exposure is not certified.',
                      'Only 47 English DRB questions remain after excluding its three calibration questions.']}


@contextmanager
def lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.complete-study.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another completion runner is already using this output directory')
        yield


def freeze(output, proposed):
    old = read(output / 'PLAN.json')
    if old:
        if old['protocol'] != proposed['protocol']:
            raise ValueError('Generation settings/source changed. Choose a separate output cohort; completed runs were preserved.')
        for bench, ids in old['queries'].items():
            if not set(ids) <= set(proposed['queries'].get(bench, [])):
                raise ValueError('Cannot remove frozen evaluation IDs. Keep counts or extend them.')
    atomic_json(output / 'PLAN.json', proposed)


def import_finished(source, output, proposed):
    """Copy incomplete judge bundles; link fully judged immutable source bundles."""
    for job in proposed['jobs']:
        origin = origin_for(source, output, job)
        if origin is None or origin == output / job['bench']:
            continue
        for k in (1, 2, 3):
            src, dest = run_path(origin, job, k), run_path(output / job['bench'], job, k)
            if not src.exists() or dest.exists():
                continue
            if article(src, job['q']) != article(run_path(origin, job), job['q']):
                raise ValueError(f'Judge pass contains a different report: {src}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if score_ready(src, job['bench']):
                dest.symlink_to(src.resolve(), target_is_directory=True)
            else:
                shutil.copytree(src, dest, ignore=shutil.ignore_patterns('judge.log'))
        # Keep timings' retry filtering aligned when reports were imported.
        log = origin / job['setting'] / job['arm'] / 'logs' / f"{job['q']}-s0.log"
        dest = output / job['bench'] / job['setting'] / job['arm'] / 'logs' / log.name
        if log.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(log.resolve())


def environment(source):
    from dotenv import dotenv_values
    env = dict(os.environ)
    # Explicit session settings win over dotenv, including the GPU URL.
    env = {**{k: v for k, v in dotenv_values(source / '.env').items() if v is not None}, **env}
    if env.get('SOL_API_KEY'):
        env['OPENAI_API_KEY'] = env['SOL_API_KEY']
        env['OPENAI_BASE_URL'] = env.get('SOL_BASE_URL', '')
    elif env.get('AZURE_OPENAI_API_KEY'):
        env['OPENAI_API_KEY'] = env['AZURE_OPENAI_API_KEY']
        env['OPENAI_BASE_URL'] = env.get('SOL_BASE_URL', 'https://202610-deepres.openai.azure.com/openai/v1/')
    env['PYTHONPATH'] = str(REPO / 'src')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['ADR_GR_DIR'] = str(source / 'third_party/gpt-researcher')
    for name in ('deepresearchgym', 'deepresearch_benchmarking'):
        if (source / 'third_party' / name).exists():
            env['ADR_GYM_DIR'] = str((source / 'third_party' / name).resolve())
            break
    for name in ('GR_AZURE_SERVICE_TIER', 'LLM_KWARGS'):
        env.pop(name, None)
    return env


def preflight(source, proposed, env):
    errors = []
    needed = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'JUDGE_API_KEY', 'JUDGE_BASE_URL']
    if proposed['summary'].get('drg', {}).get('new_reports'):
        needed += ['DEEPRESEARCHGYM_API_KEY']
    if any(s['new_reports'] for s in proposed['summary'].values()):
        needed += ['GR_ORCH_LLM_BASE_URL']
    errors += [f'{key} is missing' for key in needed if not env.get(key)]
    if env.get('GR_ORCH_LLM_BASE_URL'):
        try:
            req = urllib.request.Request(env['GR_ORCH_LLM_BASE_URL'].rstrip('/') + '/models',
                  headers={'Authorization': 'Bearer ' + env.get('GR_ORCH_LLM_API_KEY', 'EMPTY')})
            with urllib.request.urlopen(req, timeout=20) as response:
                ids = {m['id'] for m in json.load(response).get('data', [])}
            if not {'pilot-bc-teacher', 'pilot-dpo-teacher'} <= ids:
                errors.append('GPU server does not serve both pilot-bc-teacher and pilot-dpo-teacher')
        except Exception as exc:
            errors.append(f'GPU server unavailable ({type(exc).__name__})')
    if 'drb' in proposed['summary']:
        if not env.get('JINA_API_KEY') and env.get('JINA_READER') != 'public':
            errors.append('DRB needs JINA_API_KEY or JINA_READER=public')
        if proposed['summary']['drb']['new_reports']:
            import csv
            pool = Path(env.get('TAVILY_KEYS_CSV') or source / 'third_party/gpt-researcher/env/tavily.csv')
            keys = [r['api_key'] for r in csv.DictReader(pool.open()) if r.get('api_key')] if pool.exists() else []
            keys = keys or ([env['TAVILY_API_KEY']] if env.get('TAVILY_API_KEY') else [])
            balances = []
            for key in keys:
                try:
                    req = urllib.request.Request('https://api.tavily.com/usage', headers={'Authorization': f'Bearer {key}'})
                    with urllib.request.urlopen(req, timeout=20) as response:
                        account = json.load(response).get('account', {})
                    balances.append(max(0, account['plan_limit'] - account['plan_usage']))
                except Exception:
                    pass
            if not balances or sum(balances) <= 0:
                errors.append('No verified positive Tavily credit balance; no search/model calls started')
        try:
            processes = subprocess.run(['ps', '-ax', '-o', 'command='], capture_output=True, text=True, timeout=10)
            if processes.returncode:
                errors.append('Process inventory unavailable: check the existing DRB watcher from a regular terminal')
            elif any('drb_autostart.sh' in line and not line.lstrip().startswith(('rg ', 'grep '))
                     for line in processes.stdout.splitlines()):
                errors.append('Existing DRB autostart watcher detected: stop it before starting this replacement runner')
        except (OSError, subprocess.TimeoutExpired):
            errors.append('Process inventory unavailable: check the existing DRB watcher from a regular terminal')
        marker = source / 'third_party/deep_research_bench/utils/api.py'
        if not marker.exists() or 'RETRY_PATCH' not in marker.read_text():
            errors.append('Patched DRB judge checkout is missing')
    if errors:
        raise ValueError('Preflight blocked:\n- ' + '\n- '.join(errors))


def configs(source, output, bench):
    import yaml
    base = yaml.safe_load((REPO / 'configs' / CONFIG[bench]).read_text())
    agent = yaml.safe_load((REPO / base['agent']['config']).read_text())
    agent['repo_path'] = str((source / 'third_party/gpt-researcher').resolve())
    agent['trajectory_dir'] = str(output / '_raw' / bench)
    config_dir = output / '_configs'
    config_dir.mkdir(exist_ok=True)
    agent_path = config_dir / f'{bench}_agent.yaml'
    agent_path.write_text(yaml.safe_dump(agent))
    base['agent']['config'] = str(agent_path)
    base['dataset']['path'] = str(REPO / 'data/benchmarks' / DATA[bench])
    path = config_dir / f'{bench}.yaml'
    path.write_text(yaml.safe_dump(base))
    return path


def invoke(command, env, log, timeout):
    if STOP_REQUESTED.is_set():
        return -1
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)
        ACTIVE_PROCESSES[proc.pid] = proc
        try:
            if STOP_REQUESTED.is_set():
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            return proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return -1
        finally:
            ACTIVE_PROCESSES.pop(proc.pid, None)


def interrupt_run(signum, frame):
    STOP_REQUESTED.set()
    for pid in list(ACTIVE_PROCESSES):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt


def generate(job, root, config, env, thresholds, writer_limit, timeout):
    if STOP_REQUESTED.is_set():
        return False
    run = run_path(root, job)
    if article(run, job['q']):
        return True
    child = {k: v for k, v in env.items() if not k.startswith('GR_')}
    child.update(GR_ORCHESTRATOR='learned' if job['arm'] in ('bc', 'dpo') else job['arm'],
                 GR_ORCH_SEED='0', GR_RESEARCH_BUDGET=str(job['budget']), GR_BUDGET_OUTPUT_WEIGHT='5',
                 GR_BUDGET_FORCE_FULL=str(int(job['force'])), GR_CONTEXT_BUDGET_TOKENS='50000',
                 GR_REPORT_WORDS='3000', GR_REPORT_REWRITE='0', GR_MAX_CONTEXT_WORDS='0',
                 GR_FINAL_ITEM_MAX_WORDS='0', GR_WRITER_MAX_INPUT_TOKENS=str(writer_limit))
    if job['arm'] in ('bc', 'dpo'):
        child.update(GR_ORCH_LLM_BASE_URL=env['GR_ORCH_LLM_BASE_URL'],
                     GR_ORCH_LLM_API_KEY=env.get('GR_ORCH_LLM_API_KEY', 'EMPTY'),
                     GR_ORCH_LLM_PROVIDER='openai', GR_ORCH_LLM_MODEL=f"pilot-{job['arm']}-teacher")
    if job['arm'] == 'prompted_scores':
        child.update(thresholds)
    arm_root = run.parent
    log = arm_root / 'logs' / f"{job['q']}-s0.log"
    cmd = [sys.executable, '-m', 'adr.cli', 'run', '-c', str(config), '--query-ids', job['q'],
           '--run-id', run.name, '--run-name', run.name, '--output-dir', str(arm_root)]
    for _ in range(3):
        if STOP_REQUESTED.is_set():
            return False
        invoke(cmd, child, log, timeout)
        # A valid completed artifact wins even if the worker failed while exiting.
        # Retrying it would silently replace a usable report.
        if article(run, job['q']):
            return True
    return False


def get_thresholds(source, output, bench, env, config, args):
    dest = output / bench / 'thresholds.env'
    dest.parent.mkdir(parents=True, exist_ok=True)
    old = source / 'runs' / f'clean_{bench}' / 'thresholds.env'
    if not dest.exists() and old.exists():
        shutil.copy2(old, dest)
    if not dest.exists():
        tune = output / bench / 'tune'
        for arm in ('prompted', 'prompted_scores'):
            for q in DEV[bench]:
                job = {'bench': bench, 'setting': 'equal_200k', 'arm': arm,
                       'q': q, 'budget': 200000, 'force': True}
                if not generate(job, tune, config, env, {}, args.writer_max_input_tokens, args.timeout):
                    raise ValueError(f'{bench}: calibration failed; main runs not started')
        mod = load_results(bench)
        if mod.tune(tune / 'equal_200k', dest):
            raise ValueError(f'{bench}: calibration incomplete')
    values = dict(line.split('=', 1) for line in dest.read_text().splitlines() if line.strip())
    if set(values) != {'GR_PROMPTED_KEEP_THRESHOLD', 'GR_PROMPTED_STOP_THRESHOLD'} or not all(
            0 <= float(v) <= 10 for v in values.values()):
        raise ValueError(f'{bench}: invalid threshold file; main runs not started')
    contract = dest.with_suffix('.sha256')
    current = digest(dest)
    if contract.exists() and contract.read_text().strip() != current:
        raise ValueError(f'{bench}: calibrated thresholds changed within the cohort')
    contract.write_text(current + '\n')
    return values


def judge(job, root, config, env, drb_worker, timeout):
    import yaml
    if STOP_REQUESTED.is_set():
        return
    run = run_path(root, job)
    if not article(run, job['q']):
        return
    child = {**env, 'OPENAI_API_KEY': env['JUDGE_API_KEY'], 'OPENAI_BASE_URL': env['JUDGE_BASE_URL'],
             'CLEAN_MODEL': 'gpt-5-mini', 'CRAWL4AI_NO_CLOUD_NOTICE': '1'}
    if job['bench'] == 'drb':
        child['ADR_DRB_DIR'] = str(drb_worker)
    for k in (1, 2, 3):
        if STOP_REQUESTED.is_set():
            return
        dest = run_path(root, job, k)
        if score_ready(dest, job['bench']):
            # Pass folders must actually contain the same report.
            if article(dest, job['q']) != article(run, job['q']):
                raise ValueError(f'Judge pass report differs: {dest}')
            continue
        if dest.is_symlink():
            raise ValueError(f'Refusing to write through imported source link: {dest}')
        if not dest.exists():
            shutil.copytree(run, dest, ignore=shutil.ignore_patterns('metrics', 'judge.log', 'exports'))
        if article(dest, job['q']) != article(run, job['q']):
            raise ValueError(f'Stale judge report: {dest}; original retained')
        for _ in range(3):
            old = read(dest / 'metrics/summary.json', {})
            block = old.get('official', {}).get(OFFICIAL[job['bench']], {})
            task_config = yaml.safe_load(config.read_text())
            settings = task_config.setdefault('eval', {}).setdefault(OFFICIAL[job['bench']], {})
            if job['bench'] == 'drg':
                quality = block.get('quality') or {}
                criteria = quality.get('per_criterion') or {}
                quality_ok = bool(quality.get('n')) and isinstance(quality.get('average_normalized_score'), (int, float)) and all(
                    isinstance((criteria.get(c) or {}).get('average_rating'), (int, float))
                    for c in ('Clarity', 'Breadth', 'Depth', 'Support'))
                complete = {'quality': quality_ok}
                for component, key in [('kpr', 'average_support_rate'), ('citation', 'average_citation_score')]:
                    value = block.get(component) or {}
                    complete[component] = bool(value.get('n')) and isinstance(value.get(key), (int, float))
                for component, ok in complete.items():
                    settings[f'run_{component}'] = not ok
                preserved = {c: block[c] for c in ('quality', 'kpr', 'citation')
                             if complete[c]}
            else:
                race = (block.get('race') or {}).get('scores') or {}
                race_ok = all(isinstance(race.get(k), (int, float)) for k in
                              ('overall_score', 'comprehensiveness', 'insight', 'instruction_following', 'readability'))
                fact_ok = bool(((block.get('fact') or {}).get('scores') or {}).get('total_citations'))
                settings.update(run_race=not race_ok, run_fact=not fact_ok)
                preserved = {c: block[c] for c, ok in [('race', race_ok), ('fact', fact_ok)] if ok}
            task_path = dest / 'judge_resume_config.yaml'
            task_path.write_text(yaml.safe_dump(task_config))
            cmd = [sys.executable, '-m', 'adr.cli', 'evaluate', str(dest), '--official', OFFICIAL[job['bench']], '-c', str(task_path)]
            invoke(cmd, child, dest / 'judge.log', timeout)
            current = read(dest / 'metrics/summary.json', {})
            target = current.setdefault('official', {}).setdefault(OFFICIAL[job['bench']], {})
            target.update(preserved)
            atomic_json(dest / 'metrics/summary.json', current)
            if score_ready(dest, job['bench']):
                break


def offline_reports(source, output, proposed):
    summary = {}
    for bench in proposed['queries']:
        mod = load_results(bench)
        root = output / bench
        mod.table(root)
        data = mod.collect(root)
        audit = []
        for (mode, budget), arms in data.items():
            for arm, queries in arms.items():
                for q, e in queries.items():
                    f = e['facts']
                    run = root / f'{mode}_{budget}k' / arm / f'{q}-s0'
                    raw = read(run / 'gpt_researcher' / f'{q}.json', {})
                    evidence = raw.get('evidence', {})
                    items = evidence.values() if isinstance(evidence, dict) else evidence
                    depths = [it.get('tree_depth', 0) for it in items if isinstance(it, dict)]
                    metas = [r.get('decision', {}).get('meta', {}) for r in raw.get('rounds', [])]
                    kp = [s.get('KPR') for s in e['scores'] if s.get('KPR') is not None]
                    audit.append({'setting': f'{mode}_{budget}k', 'arm': arm, 'query': q,
                                  'passes': len(e['scores']), 'spend': f['research_spend'], 'allowance': f['budget'],
                                  'overshoot': (f['research_spend'] - f['budget']) if f['research_spend'] is not None and f['budget'] else None,
                                  'report_words': f['report_words'], 'length_in_range': 2700 <= (f['report_words'] or 0) <= 3300,
                                  'final_context_words_cut': f['words_cut'],
                                  'max_evidence_depth': max(depths) if depths else None,
                                  'checkpoint_context_target_exceeded': sum(m.get('context_tokens_after', 0) > 50000 for m in metas),
                                  'orchestrator_parse_failures': sum(bool(m.get('parse_failure')) for m in metas),
                                  'KPR_judge_sd': st.stdev(kp) if len(kp) > 1 else None})
        atomic_json(root / 'AUDIT.json', audit)
        summary[bench] = {'reports': len(audit), 'three_pass_reports': sum(r['passes'] == 3 for r in audit),
                          'length_outliers': sum(not r['length_in_range'] for r in audit),
                          'KPR_sd_at_least_2_7': sum((r['KPR_judge_sd'] or 0) >= 2.7 for r in audit),
                          'reports_reaching_depth_3': sum((r['max_evidence_depth'] or 0) >= 3 for r in audit)}
        lines = ['# BC/DPO versus prompted', '', 'Paired differences on the same questions; intervals resample questions. '
                 'Retention/stopping associations do not establish a causal explanation.', '']
        for (mode, budget), arms in sorted(data.items()):
            for arm, comparator in (('bc', 'prompted'), ('bc', 'prompted_scores'),
                                    ('dpo', 'prompted'), ('dpo', 'prompted_scores'), ('bc', 'dpo')):
                common = sorted(set(arms.get(arm, {})) & set(arms.get(comparator, {})))
                for metric in mod.B['delta']:
                    diffs = [mod.per_report(arms[arm][q], metric) - mod.per_report(arms[comparator][q], metric)
                             for q in common if len(arms[arm][q]['scores']) == len(arms[comparator][q]['scores']) == 3]
                    if len(diffs) >= 3:
                        lo, hi = mod.boot(diffs)
                        lines.append(f'- {mode} {budget}K, {arm} minus {comparator}, {metric}: '
                                     f'{st.mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}], n={len(diffs)}.')
        if len(lines) == 4:
            lines.append('Not yet answerable: matched BC/DPO and prompted reports with three complete passes are missing.')
        (root / 'LEARNED_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    # Rebuild the follow-up separately; reuses its reports without generation/judging.
    follow = source / 'runs/clean_drg_ctx100'
    if follow.exists():
        load_results('drg').table(follow, output / 'followup_ctx100')
    atomic_json(output / 'STATUS.json', summary)
    return summary


def execute(source, output, proposed, args):
    if not any(s['new_reports'] or s['missing_judge_passes'] for s in proposed['summary'].values()):
        return []
    env = environment(source)
    preflight(source, proposed, env)
    failures = []
    for bench in proposed['queries']:
        config = configs(source, output, bench)
        thresholds = (get_thresholds(source, output, bench, env, config, args)
                      if proposed['summary'][bench]['new_reports'] else {})
        jobs = [j for j in proposed['jobs'] if j['bench'] == bench]
        def one(job):
            ok = generate(job, output / bench, config, env, thresholds, args.writer_max_input_tokens, args.timeout)
            print(f"{bench}/{job['setting']}/{job['arm']}/{job['q']}: {'report ready' if ok else 'FAILED'}", flush=True)
            return ok
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            failures += [j for j, ok in zip(jobs, pool.map(one, jobs)) if not ok]
        drb_worker = output / '_drb_judge'
        if bench == 'drb' and not drb_worker.exists():
            shutil.copytree(source / 'third_party/deep_research_bench', drb_worker,
                            ignore=shutil.ignore_patterns('.git', '__pycache__', '.adr_judge.lock'))
        # DRB staging is global within its checkout: serialize it. DRG uses independent run directories.
        with ThreadPoolExecutor(max_workers=1 if bench == 'drb' else args.judge_jobs) as pool:
            list(pool.map(lambda j: judge(j, output / bench, config, env, drb_worker, args.judge_timeout), jobs))
        offline_reports(source, output, proposed)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--output', type=Path, default=REPO / 'runs/complete_study')
    ap.add_argument('--drg-count', type=int, default=10)
    ap.add_argument('--drb-count', type=int, default=10)
    ap.add_argument('--writer-max-input-tokens', type=int, default=250000,
                    help='Must match the already completed protocol/model context guard')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--judge-jobs', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--judge-timeout', type=int, default=2400)
    ap.add_argument('--execute', action='store_true', help='Run paid generation/judging after preflight; omitted means offline only')
    ap.add_argument('--preflight', action='store_true', help='Check credentials/services without paid generation or judging')
    args = ap.parse_args(argv)
    if min(args.jobs, args.judge_jobs, args.timeout, args.judge_timeout, args.writer_max_input_tokens) <= 0:
        ap.error('Worker counts, timeouts and writer limit must be positive')
    source, output = args.source.resolve(), args.output.resolve()
    protected = [source / 'runs' / name for name in ('clean_drg', 'clean_drb', 'clean_drg_ctx100')]
    if output == source or any(output == p or p in output.parents or output in p.parents for p in protected):
        ap.error('Output must be separate from the original experiment directories')
    try:
        with lock(output):
            proposed = plan(source, output, {'drg': args.drg_count, 'drb': args.drb_count}, args.writer_max_input_tokens)
            freeze(output, proposed)
            import_finished(source, output, proposed)
            print(json.dumps(proposed['summary'], indent=2))
            if args.preflight and not args.execute:
                preflight(source, proposed, environment(source))
                atomic_json(output / 'READINESS.json', {'ready': True, 'checked_at': time.time()})
            if args.execute:
                signal.signal(signal.SIGINT, interrupt_run)
                signal.signal(signal.SIGTERM, interrupt_run)
                execute(source, output, proposed, args)
                proposed = plan(source, output, {'drg': args.drg_count, 'drb': args.drb_count}, args.writer_max_input_tokens)
                atomic_json(output / 'PLAN.json', proposed)
            offline_reports(source, output, proposed)
            pending = sum(s['new_reports'] + s['missing_judge_passes'] for s in proposed['summary'].values())
            print(f"{'Incomplete' if pending else 'Complete'}: {output / 'STATUS.json'}")
            return int(args.execute and pending > 0)
    except KeyboardInterrupt:
        print('Interrupted; completed reports were preserved. Resume with the same command.', file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        if args.preflight or args.execute:
            atomic_json(output / 'READINESS.json', {'ready': False, 'checked_at': time.time(), 'reason': str(exc)})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
