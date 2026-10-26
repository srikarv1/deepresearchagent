"""Read saved run accounting and gate audit; no judges/providers invoked."""
import argparse
import json
from pathlib import Path


def summarize(run):
    run=Path(run);summary_path=run/'metrics/summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    rows=[]
    for path in sorted((run/'trajectories').glob('*.json')):
        t=json.loads(path.read_text());stats=t.get('final_stats',{})
        rows.append(dict(trajectory_file=str(path),query_id=(t.get('query') or {}).get('id'),
            report_present=bool(t.get('report')),gate=stats.get('redundancy_gate'),
            recorded_cost_usd=stats.get('total_cost_usd'),cost_source=stats.get('cost_source')))
    decisions={};retrieved=set();skipped=set()
    for path in sorted((run/'redundancy_gate').glob('*.jsonl')):
        for line in path.open():
            e=json.loads(line);key=(str(path),e.get('candidate_id'))
            if e['event'] in ('heuristic_decision','heuristic_decision_override','heuristic_finished','skipped'):decisions[key]=e
            if e['event']=='retrieval_invoked':retrieved.add(key)
            if e['event']=='skipped':skipped.add(key)
    n=len(decisions);s=sum(d['decision']=='SKIP' for d in decisions.values())
    return dict(run=str(run),research_metrics=summary,trajectories=rows,candidate_count=n,skip_count=s,
        skip_rate=s/n if n else 0,fail_open_count=sum(d['fail_open'] for d in decisions.values()),
        retrieval_invocation_count=len(retrieved),skipped_retrieval_violations=len(skipped&retrieved),
        quality_scope='Existing saved local/native/official metrics only; absent judge scores remain absent.',
        heuristic_model_embedding_search_overhead=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.run),indent=2))

if __name__=='__main__':main()
