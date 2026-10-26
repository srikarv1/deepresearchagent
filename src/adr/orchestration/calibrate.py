"""Offline replay on an explicit development allowlist. Never loads benchmarks.

No new labels are assigned. Existing semantic audit outcomes are calibration
material, NOT held-out performance estimates or training data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from adr.development.replay_observation_v2 import reconstruct
from .redundancy import decide_view, VERSION, THRESHOLD

RUNS=(
 'runs/representation-study/20260926-223740-development-representation-001',
 'runs/representation-study-002/20260927-003826-development-representation-002',
 'runs/representation-study-003/20260927-114734-development-representation-003')
ROOT=Path('runs/real-state-supervision')
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def distribution(rows):
    scores=[r['score'] for r in rows]
    return dict(n=len(rows),execute=sum(r['decision']=='EXECUTE' for r in rows),skip=sum(r['decision']=='SKIP' for r in rows),
        skip_rate=sum(r['decision']=='SKIP' for r in rows)/len(rows),
        score=dict(zip(('min','median','p90','max'),map(float,np.percentile(scores,[0,50,90,100])))),
        fail_open_reasons=dict(Counter(r['reason'] for r in rows if r['fail_open'])))

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];inputs={}
    for name in RUNS:
        path=next((Path(name)/'development_trace').glob('*.jsonl'));inputs[str(path)]=sha(path);queries={}
        for line in path.open():
            e=json.loads(line)
            if e['event']=='candidate_proposed':queries[e['id']]=e['query']
        for cid,o in reconstruct(path):
            r=decide_view(o.text,queries[cid]);r.update(run=name,candidate_id=cid,candidate=queries[cid],
                selected_branch_ids=list(o.selected_task_ids),branch_selection_scores=list(o.scores),context_digest=hashlib.sha256(o.text.encode()).hexdigest())
            rows.append(r)
    assert len(rows)==336
    packets=read(ROOT/'final-candidates-v1/packets.json');audit=read(ROOT/'final-audit-v1/per_packet.json');final=[]
    for a in audit:
        p=packets[a['packet_id']];r=decide_view(p['context_before'],p['candidate_subquery']);r.update(packet_id=a['packet_id'],audit_status=a['audit_status'],existing_label=a['label'],candidate=p['candidate_subquery']);final.append(r)
    regression=[]
    for pid,p in read(ROOT/'task-interpretation-regression-v1/packets.json').items():
        r=decide_view(p['context_before'],p['candidate_subquery']);r.update(packet_id=pid,candidate=p['candidate_subquery']);regression.append(r)
    for p in [ROOT/'final-candidates-v1/packets.json',ROOT/'final-audit-v1/per_packet.json',ROOT/'task-interpretation-regression-v1/packets.json']:inputs[str(p)]=sha(p)
    accepted=[r for r in final if r['audit_status']=='ACCEPT']
    summary=dict(version=VERSION,threshold=THRESHOLD,development_only=True,no_new_labels=True,
        per_run={name:distribution([r for r in rows if r['run']==name]) for name in RUNS},pooled=distribution(rows),
        final_audit_all=distribution(final),final_audit_accepted=distribution(accepted),
        audited_execute_false_positive_skips=[r['packet_id'] for r in accepted if r['existing_label']=='EXECUTE' and r['decision']=='SKIP'],
        sole_audited_skip=next(r for r in accepted if r['existing_label']=='SKIP'),regression=regression,
        calibration_rationale={
            'formula':'max over eligible individual findings/insights of TF-IDF cosine times candidate unigram coverage; exact duplicate statements deduplicated',
            'guards':'Required numeric literals and protocol/method/distribution/share/variance/percentile/uncertainty/comparison/cost/threshold/sensitivity terms (singular/plural) must appear in the same statement; explicit uncertainty/absence markers veto it.',
            'development_findings':'Simple relatedness falsely matched values to protocols, shares, or distributions. Dimension guard prevents those specific errors. Lower scores also match general fidelity frameworks to requests for operational instruments.',
            'threshold_interval':'A clear Illinois tax-credit answer scores 0.23485. Reviewed unresolved development cases reach 0.20048 after guards; the largest accepted-audit EXECUTE is 0.19208. 0.22 lies in that separating interval, rounded to two decimals.',
            'moderate_not_supported':'Lowering enough to catch the broad digital-divide SKIP (0.12327) also skips unresolved cases. Even the sleep-timing SKIP (0.20359) is too close to the 0.20048 operational-fidelity ambiguity to justify another threshold.',
            'limits':'Calibration includes all final-audit splits by explicit authorization. Those sources are now development material for this heuristic. Zero skips over the 336 original development decisions means useful cost reduction is NOT demonstrated.'},
        reviewed_near_threshold=[dict(run=RUNS[0],candidate_id='batch0011-c002',judgment='One site annual COP does not answer cycle variability.'),dict(run=RUNS[2],candidate_id='batch0019-c004',judgment='Framework dimensions do not supply current need operational school-specific measurement instruments/templates.')],
        inputs_sha256=inputs)
    for name,value in [('trace_decisions.json',rows),('final_audit_scores.json',final),('summary.json',summary)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')
    print(json.dumps({k:summary[k] for k in ('version','threshold','per_run','pooled','audited_execute_false_positive_skips')},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--verify-freeze',type=Path);a=p.parse_args()
    if a.verify_freeze:
        frozen=read(a.verify_freeze)
        for path,h in frozen['sha256'].items():
            if sha(path)!=h:raise ValueError('Frozen file changed: '+path)
        print(json.dumps({'freeze_verified':True,'files':len(frozen['sha256'])}))
    elif a.output:run(a.output)
    else:p.error('--output or --verify-freeze required')
if __name__=='__main__':main()
