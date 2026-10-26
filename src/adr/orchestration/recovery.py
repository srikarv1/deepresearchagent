"""Offline recovery audit/finalization. No providers, browser, or research imports.

Read attempts are URL dispatches, not successful scrapes. Canonical evidence is
reported separately; successful-page counts remain unknown without telemetry.
No raw error messages (which may contain credentials) are emitted or copied.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import yaml

AUDIT_VERSION='retrieval-integrity-v2-query-id-scope'
RECOVERY='adaptive-depth-q051-055-recovery-v1'
IDS=['51','52','53','54','55']
POLICY='hierarchical-depth-budget-one-per-root-v1'

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def equivalence():
    original=yaml.safe_load(Path('configs/adaptive_depth_evaluation.yaml').read_text())
    recovery=yaml.safe_load(Path('configs/adaptive_depth_recovery.yaml').read_text())
    for key in ('run_name','output_dir'):
        original.pop(key);recovery.pop(key)
    assert original==recovery,'Research configuration mismatch'
    assert recovery['dataset']['query_ids']==IDS and recovery['dataset']['limit']==5
    p=read('configs/adaptive_depth_recovery_provenance.json')
    for path,h in {**p['original_frozen_hashes'],**p['recovery_hashes']}.items():
        if digest(path)!=h:
            reliability=read('configs/adaptive_depth_reliability_v1.json')
            change=reliability['patched_files'].get(path,{})
            assert change.get('before_sha256')==h and change.get('after_sha256')==digest(path),f'Frozen file mismatch: {path}' 
    a=yaml.safe_load(Path(recovery['agent']['config']).read_text())
    assert a['branch_expansion']['policy']==POLICY and a['env']['GPT_RESEARCHER_BROWSER_HEADLESS']=='true'
    return {'equivalent':True,'excluded_fields':['run_name','output_dir'],'shared_agent_config':recovery['agent']['config'],'frozen_hashes_verified':True,'query_ids':IDS}

PATTERNS={
 'quota_or_rate_limit':r'\b429\b|quota.{0,50}(exceed|exhaust)|credit.{0,40}(exhaust|insufficient)|rate.?limit|usage.?limit',
 'authentication':r'\b(401|403)\b|unauthori[sz]ed|invalid.{0,20}(api.?key|credential)|authentication.{0,20}(fail|error)',
 'browser_or_scraper':r'(selenium|webdriver|chrome|scrap).{0,80}(error|fail|exception)|session not created|invalid session id',
 'empty_search_response':r'empty (search )?(response|results)|no search results|failed fetching sources',
 'other_provider_error':r'\b(error|exception|timeout|timed out|failed)\b',
}
def classify(text):return [k for k,p in PATTERNS.items() if re.search(p,text,re.I|re.S)]

def audit(run,console_log=None,expected_ids=None):
    ids=IDS if expected_ids is None else list(expected_ids)
    run=Path(run);summary=read(run/'metrics/summary.json');local=read(run/'metrics/local.json')
    byid={str(m['id']):m for m in local['per_query']};flags=[];rows=[];error_records=[]
    if set(byid)!=set(ids):flags.append('missing_or_unexpected_query_ids')
    for q in ids:
        m=byid.get(q,{})
        path=run/'branch_expansion'/f'{hashlib.sha256(q.encode()).hexdigest()[:16]}.jsonl'
        events=[json.loads(x) for x in path.open()] if path.exists() else []
        counts=Counter(e['event'] for e in events);bs={e['branch']['id']:e['branch'] for e in events if e['event']=='branch_created'}
        decisions=[e for e in events if e['event']=='branch_expansion_decision'];stops=[e for e in decisions if e.get('remaining_depth',0)>1 and e.get('decision')=='STOP']
        qflags=[]
        trajectory=run/'trajectories'/f'{q}.json';t=read(trajectory) if trajectory.exists() else {};stats=t.get('final_stats',{})
        retained=m.get('n_retained',stats.get('n_retained'))
        if m.get('n_reads',0)==0:qflags.append('zero_read_attempts_no_successful_reads_demonstrated')
        if not retained:qflags.append('zero_retained_canonical_evidence')
        report=run/'reports'/f'{q}.md'
        if not report.exists() or not report.stat().st_size:qflags.append('missing_report')
        if m.get('error') or t.get('error'):qflags.append('recorded_query_error')
        if not events or events[-1].get('status')!='research_completed':qflags.append('missing_completed_audit_session')
        published={b for e in events if e['event']=='round_published' for b in e['branch_ids']}
        if set(bs)!=published or len(bs)!=20:qflags.append('incomplete_or_unexpected_tree')
        if Counter(b['tree_depth'] for b in bs.values())!={1:4,2:8,3:8}:qflags.append('unexpected_depth_distribution')
        suppression=all(not any(b['parent_id']==e['branch_id'] for b in bs.values()) for e in stops)
        if not suppression:qflags.append('policy_stop_has_descendants')
        if len(stops)!=4 or any(e.get('depth')!=2 for e in stops):qflags.append('unexpected_policy_stops')
        if any(e.get('policy_version')!=POLICY for e in decisions):qflags.append('policy_version_mismatch')
        failopen=sum(bool(e.get('fail_open')) for e in decisions)
        if failopen:qflags.append('fail_open_requires_review')
        candidate_sets=[]
        for event in ('candidate_dispatched','retrieval_invoked','candidate_finished'):
            selected=[e for e in events if e['event']==event];candidate_ids=[e.get('candidate_id') for e in selected]
            candidate_sets.append(set(candidate_ids))
            if len(candidate_ids)!=80 or len(set(candidate_ids))!=80:qflags.append('missing_or_duplicate_'+event)
        if any(x!=candidate_sets[0] for x in candidate_sets):qflags.append('candidate_lifecycle_id_mismatch')
        if any(e.get('status')!='completed' for e in events if e['event']=='candidate_finished'):qflags.append('candidate_failure')
        reads=sum(e.get('count',0) for e in events if e['event']=='read_dispatch')
        if reads!=m.get('n_reads') or counts['search_dispatch']!=m.get('n_searches'):qflags.append('accounting_mismatch')
        for e in events:
            if 'error' in e['event'] or e.get('error'):
                error_records.append({'query_id':q,'source':str(path),'sequence':e.get('sequence'),'categories':classify(json.dumps(e)) or ['unclassified_recorded_error']})
        for value in (m.get('error'),t.get('error')):
            if value:error_records.append({'query_id':q,'source':str(trajectory),'categories':classify(str(value)) or ['unclassified_recorded_error']})
        rows.append(dict(query_id=q,searches_dispatched=counts['search_dispatch'],retrieval_invocations=counts['retrieval_invoked'],retrieval_completions=sum(e.get('status')=='completed' for e in events if e['event']=='candidate_finished'),read_attempts=reads,successful_reads_pages=None,successful_page_count_status='Not persisted by existing branch logger; never substitute canonical evidence or read attempts.',retained_canonical_evidence=retained,llm_calls=m.get('n_llm_calls'),tokens=m.get('tokens'),latency_s=m.get('wall_s'),report_exists=report.exists() and report.stat().st_size>0,errors_present=bool(m.get('error') or t.get('error')),expand=sum(e.get('decision')=='EXPAND' for e in decisions),policy_stop=len(stops),natural_terminal_stop=sum(e.get('reason')=='terminal_depth' for e in decisions),fail_open=failopen,depth_distribution=dict(Counter(b['tree_depth'] for b in bs.values())),stop_suppression_verified=suppression,flags=qflags))
    logs=list((run/'errors').glob('*'))+[p for p in run.rglob('*.log')]
    if console_log:logs.append(Path(console_log))
    for p in logs:
        if p.is_file() and p.stat().st_size:
            kinds=classify(p.read_text(errors='replace'))
            if kinds:error_records.append({'source':str(p),'categories':kinds})
    if error_records:flags.append('persisted_error_signals_require_manual_review')
    if summary.get('n_errors') or summary.get('n_with_report')!=len(ids):flags.append('incomplete_or_failed_run')
    flags.extend(f"q{r['query_id']}:{f}" for r in rows for f in r['flags'])
    return dict(audit_version=AUDIT_VERSION,run=str(run),status='FLAGGED_MANUAL_REVIEW_REQUIRED' if flags else 'NO_AUTOMATED_FLAGS_MANUAL_REVIEW_REQUIRED',flags=flags,per_query=rows,error_signals=error_records,log_sources_scanned=[str(p) for p in logs],limitations=['Successful-page counts are not separately persisted; null means unknown, not zero.','Tavily exceptions can be swallowed/printed only. Absence of saved errors cannot certify quota/auth success.','Candidate completion can return empty retrieval; inspect read/evidence counts and provider account/logs manually.'],anomaly_review={'read_attempts_by_query':{r['query_id']:r['read_attempts'] for r in rows},'retained_evidence_by_query':{r['query_id']:r['retained_canonical_evidence'] for r in rows},'rule':'Report dispersion without arbitrary minimum-count acceptance thresholds.'},judge_calls_authorized=False)

def finalize(run,output,manual_review_approved=False):
    equivalence();run=Path(run);output=Path(output)
    cfg=yaml.safe_load((run/'config.yaml').read_text())
    expected=yaml.safe_load(Path('configs/adaptive_depth_recovery.yaml').read_text())
    cfg.pop('_config_path',None)
    if cfg!=expected:raise ValueError('Not the frozen recovery run configuration')
    if not manual_review_approved:raise ValueError('Manual retrieval-integrity approval is required')
    result=audit(run)
    if result['flags']:raise ValueError('Integrity flags remain; do not finalize')
    summary=read(run/'metrics/summary.json');official=summary.get('official',{}).get('deep_research_bench',{})
    for stage in ('race','fact'):
        block=official.get(stage,{})
        if not block.get('scores') or Path(block.get('path','')).parent.name!=RECOVERY:raise ValueError('Missing or non-recovery judge results')
    race=official['race'];fact=official['fact']
    if not race.get('log',{}).get('ok') or len(fact.get('stages',[]))!=5 or not all(s.get('ok') for s in fact.get('stages',[])):raise ValueError('Judge execution failed')
    raw=[json.loads(x) for x in Path(race['raw_results']).open()]
    if len(raw)!=5 or {str(r.get('id')) for r in raw}!=set(IDS) or any(r.get('error') for r in raw):raise ValueError('RACE did not score exactly five reports')
    sources=list(run.rglob('*'))+list(Path(race['path']).parent.rglob('*'))+list(Path(fact['path']).parent.rglob('*'))
    hashes={str(p):digest(p) for p in sources if p.is_file()}
    output.mkdir(parents=True,exist_ok=False)
    scores=race['scores'];row={k:scores[k] for k in ('comprehensiveness','insight','readability')};row.update(citation_accuracy=fact['scores']['valid_rate'],tokens_K=summary['mean_tokens']/1000,latency_s=summary['mean_wall_s'],searches=summary['mean_n_searches'],reads=summary['mean_n_reads'],llm_calls=summary['mean_n_llm_calls'],Q_per_second=None)
    artifacts={'table4.json':row,'research_and_judge_metrics.json':summary,'retrieval_integrity.json':result,'source_checksums.json':hashes,'recovery_provenance.json':read('configs/adaptive_depth_recovery_provenance.json'),'manual_review.json':{'operator_attested':True,'scope':'retrieval integrity approved before judging; not an automated clean-run certification'}}
    for name,value in artifacts.items():(output/name).write_text(json.dumps(value,indent=2)+'\n')
    assert all(digest(p)==h for p,h in hashes.items())
    (output/'SHA256SUMS.json').write_text(json.dumps({p.name:digest(p) for p in output.iterdir() if p.is_file()},indent=2)+'\n')
    return {'output':str(output),'table4':row}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify-config');a=sub.add_parser('audit');a.add_argument('run',type=Path);a.add_argument('--console-log',type=Path);a.add_argument('--query-id',choices=IDS)
    a=sub.add_parser('finalize');a.add_argument('run',type=Path);a.add_argument('--output',type=Path,required=True);a.add_argument('--manual-review-approved',action='store_true')
    args=p.parse_args()
    if args.command=='verify-config':result=equivalence()
    elif args.command=='audit':result=audit(args.run,args.console_log,expected_ids=[args.query_id] if args.query_id else None)
    else:result=finalize(args.run,args.output,args.manual_review_approved)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
