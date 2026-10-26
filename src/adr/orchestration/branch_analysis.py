"""Offline development-only depth-budget replay; descendant costs are analysis ONLY."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
from adr.mlp.embedding_limits import token_count
from .branch_budget import DepthBudget, VERSION
from .calibrate import RUNS

def terms(s):return set(re.findall(r'\b[a-z][a-z0-9]+\b',s.lower()))-ENGLISH_STOP_WORDS

def jaccard(a,b):return len(a&b)/max(1,len(a|b))

def analyze(run):
    run=Path(run);paths=list((run/'development_trace').glob('*.jsonl'));assert len(paths)==1
    trace=paths[0];branches={};results={};record_kinds={};published=[];queries={};raw=set();last_publication=None;prior_account=None;rounds=[];decisions=[];policy=DepthBudget();parents_seen=set();question=''
    for line in trace.open():
        e=json.loads(line);kind=e['event']
        if kind=='session':question=e['original_question']
        elif kind=='record':
            record_kinds[e['record_id']]=e['kind']
            if e['kind']=='processed_result':results[e['record_id']]=e['value']
        elif kind=='record_acquired':
            if record_kinds.get(e['record_id'])=='raw_page':raw.add(e['record_id'])
            if record_kinds.get(e['record_id'])=='processed_result':branches[e['branch_id']]['result_id']=e['record_id']
        elif kind=='branch_created':
            b=e['branch'];parent=b['parent_id']
            if parent is not None and parent not in parents_seen:
                parents_seen.add(parent);assert last_publication is not None
                # Child planning has happened by branch_created, but the input
                # ledger below is the last publication BEFORE that planning.
                pb=branches[parent];r=results[pb['result_id']]
                decision=policy.decide(parent,4-pb['tree_depth'],branches,published)
                texts=[s for bid in published for s in results[branches[bid]['result_id']]['learnings']]
                followups=r['followUpQuestions'];overlap=[max([jaccard(terms(f),terms(s)) for s in texts]+[0.]) for f in followups]
                decision.update(source_prefix_sequence=last_publication['sequence'],first_child_created_sequence=e['sequence'],
                    signals=dict(published_results=len(published),available_raw_page_records=len(raw),
                        published_findings=len(texts),followup_count=len(followups),
                        followup_max_lexical_overlap=max(overlap,default=0),followup_mean_lexical_overlap=float(np.mean(overlap)) if overlap else 0,
                        goal_question_jaccard=jaccard(terms(r['researchGoal']),terms(question)),
                        result_tokens=token_count(json.dumps({k:r[k] for k in ('researchGoal','learnings','followUpQuestions','citations')},ensure_ascii=False,sort_keys=True))),
                    budget_state=last_publication['accounting'],research_goal=r['researchGoal'],followups=followups)
                decisions.append(decision)
            branches[b['id']]=dict(b)
        elif kind=='branch_active':branches[e['branch_id']]['status']='active'
        elif kind=='branch_processed':branches[e['branch_id']]['status']=e['status']
        elif kind=='candidate_proposed':queries[e['id']]=e
        elif kind=='round_published':
            published=list(e['available_completed'])
            for bid in published:branches[bid]['status']='done'
            a=e['accounting'];prev=prior_account or dict(search_dispatches=0,read_attempts=0,tracked_tokens={'total_tokens':0},llm_calls=0,elapsed_s=0)
            rounds.append(dict(branch_ids=e['branch_ids'],searches=a['search_dispatches']-prev['search_dispatches'],reads=a['read_attempts']-prev['read_attempts'],tokens=a['tracked_tokens']['total_tokens']-prev['tracked_tokens']['total_tokens'],llm_calls=a['llm_calls']-prev['llm_calls'],elapsed_s=a['elapsed_s']-prev['elapsed_s']))
            prior_account=a;last_publication=e
    stopped={d['branch_id'] for d in decisions if d['decision']=='STOP'}
    def ancestors(bid):
        result=set();p=branches[bid]['parent_id']
        while p:result.add(p);p=branches[p]['parent_id']
        return result
    removed={bid for bid in branches if ancestors(bid)&stopped}
    # Each removed group is a complete recursive round; no cost is allocated by
    # guesswork to partial groups. Only completed historical descendant costs.
    discarded=[r for r in rounds if set(r['branch_ids'])<=removed]
    assert all(not(set(r['branch_ids'])&removed) or set(r['branch_ids'])<=removed for r in rounds)
    metric=json.loads((run/'metrics/summary.json').read_text())
    savings={k:sum(r[k] for r in discarded) for k in ('searches','reads','tokens','llm_calls','elapsed_s')}
    for d in decisions:
        desc=sorted(b for b in branches if d['branch_id'] in ancestors(b))
        d['retrospective_descendant_ids']=desc
        d['retrospective_descendants_pruned']=len(desc) if d['decision']=='STOP' else 0
    return dict(run=str(run),question=question,trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        branches=len(branches),expansions=len(decisions),stopped_expansions=len(stopped),expansion_stop_rate=len(stopped)/len(decisions),
        removed_branches=len(removed),removed_branch_ids=sorted(removed),
        candidate_count=len(queries),candidate_searches_avoided=sum(q['branch_id'] in removed for q in queries.values()),
        depth_before=dict(Counter(b['tree_depth'] for b in branches.values())),
        depth_after=dict(Counter(b['tree_depth'] for bid,b in branches.items() if bid not in removed)),
        recorded_total_tokens=metric['total_tokens'],recorded_wall_s=metric['mean_wall_s'],
        retrospective_avoided=savings,token_savings_fraction=savings['tokens']/metric['total_tokens'],
        search_savings_fraction=savings['searches']/metric['mean_n_searches'],decisions=decisions)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=list(RUNS)
    for i in range(1,9):
        matches=list(Path(f'runs/real-source-v2/real-source-v2-{i:03}').glob('*/metrics/summary.json'));assert len(matches)==1;paths.append(str(matches[0].parents[1]))
    rows=[analyze(p) for p in paths];assert len(rows)==11
    summary=dict(version=VERSION,development_only=True,runs=rows,
        expansion_opportunities=sum(r['expansions'] for r in rows),stopped_expansions=sum(r['stopped_expansions'] for r in rows),
        candidate_searches_avoided=sum(r['candidate_searches_avoided'] for r in rows),
        retrospective_avoided={k:sum(r['retrospective_avoided'][k] for r in rows) for k in ('searches','reads','tokens','llm_calls','elapsed_s')},
        limitations=['Replay preserves observed tree/costs; live descendants and later token costs can change.','Savings include removed rounds/planning but do not predict report quality or final report cost.','Diagnostic prefixes include actual prior branches; only topology/order/cap are live decision inputs.','Follow-up lexical overlap does not establish resolution. No semantic thresholds were fitted.'])
    signals=[d['signals'] for r in rows for d in r['decisions']]
    summary['pre_expansion_signal_distributions']={k:dict(zip(('min','median','p90','max'),map(float,np.percentile([d[k] for d in signals],[0,50,90,100])))) for k in signals[0]}
    with (args.output/'analysis.json').open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False);f.write('\n')
    for r in rows:print(json.dumps({k:r[k] for k in ('run','branches','expansions','stopped_expansions','removed_branches','candidate_searches_avoided','retrospective_avoided','token_savings_fraction')}))
    print('TOTAL',json.dumps({k:summary[k] for k in ('expansion_opportunities','stopped_expansions','candidate_searches_avoided','retrospective_avoided','pre_expansion_signal_distributions')}))
if __name__=='__main__':main()
