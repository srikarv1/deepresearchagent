import hashlib
import json
from pathlib import Path
import pytest
import yaml
from adr.orchestration.recovery import audit, classify, equivalence, finalize, IDS

OLD=Path('runs/adaptive-depth-william-five-v1/20260930-012124-adaptive-depth-william-five-v1')

def test_equivalence_and_quality_isolation():
    assert equivalence()['equivalent']
    old=yaml.safe_load(Path('configs/adaptive_depth_evaluation.yaml').read_text())
    new=yaml.safe_load(Path('configs/adaptive_depth_recovery.yaml').read_text())
    assert {k for k in old if old[k]!=new[k]}=={'run_name','output_dir'}
    a=yaml.safe_load(Path('configs/adaptive_depth_quality.yaml').read_text())
    b=yaml.safe_load(Path('configs/adaptive_depth_recovery_quality.yaml').read_text())
    assert a['agent']!=b['agent'];a.pop('agent');b.pop('agent');assert a==b


def test_confounded_run_flags_without_modification():
    files=[p for p in OLD.rglob('*') if p.is_file()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    r=audit(OLD)
    assert len(r['per_query'])==5
    assert [x['query_id'] for x in r['per_query'] if 'zero_retained_canonical_evidence' in x['flags']]==['51','52','53']
    assert all(x['successful_reads_pages'] is None for x in r['per_query'])
    assert all(x['policy_stop']==4 and x['stop_suppression_verified'] for x in r['per_query'])
    assert r['status']=='FLAGGED_MANUAL_REVIEW_REQUIRED'
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

@pytest.mark.parametrize('text,category',[
 ('Tavily quota exhausted','quota_or_rate_limit'),('429 rate limit','quota_or_rate_limit'),
 ('401 invalid API key SECRET','authentication'),('Chrome session not created','browser_or_scraper'),
 ('Failed fetching sources. Resulting in empty response.','empty_search_response'),('provider timeout','other_provider_error')])
def test_error_categories_no_raw_echo(text,category):
    result=classify(text);assert category in result;assert 'SECRET' not in json.dumps(result)


def test_optional_saved_console_error_safe(tmp_path):
    p=tmp_path/'console.log';p.write_text('401 authentication failed SECRET-KEY')
    r=audit(OLD,p);assert r['error_signals']
    assert 'SECRET-KEY' not in json.dumps(r)


def test_missing_lifecycle_and_report_flagged(tmp_path):
    (tmp_path/'metrics').mkdir()
    (tmp_path/'metrics/summary.json').write_text(json.dumps({'n_errors':1,'n_with_report':0}))
    (tmp_path/'metrics/local.json').write_text(json.dumps({'per_query':[]}))
    r=audit(tmp_path)
    assert 'missing_or_unexpected_query_ids' in r['flags']
    assert all('missing_report' in x['flags'] for x in r['per_query'])


def test_cannot_finalize_old_outputs_or_without_review(tmp_path):
    with pytest.raises(ValueError,match='Not the frozen recovery'):finalize(OLD,tmp_path/'out',True)
    run=tmp_path/'run';run.mkdir();(run/'config.yaml').write_text(Path('configs/adaptive_depth_recovery.yaml').read_text())
    with pytest.raises(ValueError,match='Manual'):finalize(run,tmp_path/'out')
    assert not (tmp_path/'out').exists()

def test_finalization_requires_complete_fresh_judges(tmp_path,monkeypatch):
    import adr.orchestration.recovery as module
    run=tmp_path/'run';(run/'metrics').mkdir(parents=True)
    (run/'config.yaml').write_text(Path('configs/adaptive_depth_recovery.yaml').read_text())
    monkeypatch.setattr(module,'audit',lambda r:{'flags':[],'per_query':[]})
    summary={'official':{'deep_research_bench':{}}}
    (run/'metrics/summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Missing or non-recovery'):finalize(run,tmp_path/'out',True)
    assert not (tmp_path/'out').exists()
