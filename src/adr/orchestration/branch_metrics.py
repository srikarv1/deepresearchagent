"""Export saved Table 4 quality/efficiency without invoking any evaluator."""
import argparse
import json
from pathlib import Path


def summarize(run):
    run=Path(run);s=json.loads((run/'metrics/summary.json').read_text());scores=s.get('scores',{})
    official=s.get('official',{}).get('deep_research_bench',{})
    race=official.get('race',{}).get('scores',{});fact=official.get('fact',{}).get('scores',{})
    row={k:scores.get('race_'+k,race.get(k)) for k in ('comprehensiveness','insight','readability')}
    row['citation_accuracy']=scores.get('fact_valid_rate',fact.get('valid_rate'))
    row['tokens_K']=s['mean_tokens']/1000 if s.get('mean_tokens') is not None else None
    row['latency_s']=s.get('mean_wall_s');row['Q_per_second']=None
    decisions=[]
    for p in sorted((run/'branch_expansion').glob('*.jsonl')):
        decisions.extend(e for e in map(json.loads,p.read_text().splitlines()) if e['event']=='branch_expansion_decision' and e.get('remaining_depth',1)>1)
    return {'table4_row':row,'five_reports_zero_errors':s.get('n_queries')==s.get('n_with_report')==5 and s.get('n_errors')==0,
        'literal_queries_per_second':1/s['mean_wall_s'] if s.get('mean_wall_s',0)>0 else None,
        'throughput_caveat':'Table 3 Q/sec convention unresolved; no scaling assumed.',
        'expansion_decisions':len(decisions),'stopped':sum(d['decision']=='STOP' for d in decisions),
        'quality_missing':[k for k in ('comprehensiveness','insight','readability','citation_accuracy') if row[k] is None]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(summarize(a.run),indent=2))

if __name__=='__main__':main()
