"""Opt-in official evaluation, gated on exact prompt/reference/rubric coverage."""
import json
from pathlib import Path

from .core import sha


def coverage(question, bench: Path):
    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    queries = rows(bench / 'data/prompt_data/query.jsonl')
    refs = rows(bench / 'data/test_data/cleaned_data/reference.jsonl')
    criteria = rows(bench / 'data/criteria_data/criteria.jsonl')
    matches = [q for q in queries if q.get('prompt') == question]
    if len(matches) != 1:
        raise ValueError('No unique exact DRB benchmark prompt match; evaluation refused')
    q = matches[0]
    if not any(r.get('prompt') == question and r.get('article', '').strip() for r in refs):
        raise ValueError('No DRB reference article; evaluation refused')
    # The upstream criteria file uses prompt identity; never infer coverage by ID alone.
    if not any(c.get('prompt') == question and c.get('criterions') for c in criteria):
        raise ValueError('No DRB criteria; evaluation refused')
    return str(q['id'])


def evaluate(replay: Path, output: Path, bench: Path, *, allow_api=False, fact=False):
    plan = json.loads((replay / 'plan.json').read_text())
    qid = coverage(plan['query'], bench)
    state = json.loads((replay / 'synthesis.json').read_text())
    report_path = replay / 'report.md'
    if state['status'] != 'completed' or sha(report_path) != state['report_sha256']:
        raise ValueError('Require a completed, intact synthesis report')
    if not allow_api:
        raise PermissionError('Evaluation requires explicit --allow-api approval')
    output.mkdir(parents=True, exist_ok=False)
    from adr.eval.importers import trajectory_from_pair
    from adr.eval.deep_research_bench import run_deep_research_bench
    import uuid
    result = run_deep_research_bench(
        [trajectory_from_pair(question=plan['query'], report=report_path.read_text(),
                              query_id=qid, dataset='deep_research_bench')],
        run_dir=output, model_name='evidence-replay-' + uuid.uuid4().hex,
        third_party_dir=bench, run_fact=fact)
    (output / 'evaluation.json').write_text(json.dumps({
        'coverage_query_id': qid, 'report_sha256': sha(report_path),
        'coverage_files': {str(p): sha(p) for p in [bench / 'data/prompt_data/query.jsonl',
            bench / 'data/test_data/cleaned_data/reference.jsonl', bench / 'data/criteria_data/criteria.jsonl']},
        'result': result}, indent=2), encoding='utf-8')
    return result
