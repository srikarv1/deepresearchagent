"""Finite offline real-source packet preparation; explicit, one-attempt annotation later."""
import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from adr.development.real_state_packets import load_run, sha
from adr.development.real_feature_parity import canonical_pages
from adr.development.task_interpretation_regression import request
from adr.mlp import branch_observation_v2 as observation, numerical_source_v2 as numeric
from adr.synthetic_branch_v2 import pilot_annotation as annotation, smoke
from types import SimpleNamespace

VERSION='real-source-v2-final-candidates-v1'
OUT=Path('runs/real-state-supervision/final-candidates-v1')
DOMAINS=[r'coral|reef|outplant|larva|genotyp',r'passkey|password|phish|webauthn|fido|authenticat|account|recovery|help.desk',r'bronze|archaeol|mediterranean|drought|dating|chronolog|radiocarbon|reservoir|speleothem|delta.r|oxcal',r'language|speech|transcri|asr|wer|cer|dialect|indigenous|speaker|acoustic',r'fib[er]|fibre|microplastic|laundr|textile|garment|wash|polyester|nylon|spinning',r'lunar|moon|regolith|lcross|volatile|psr|neutron|spectr|cryogen',r'court|hearing|judicial|litigan|legal|justice|tribunal|eviction',r'mycorrh|fung|hypha|cmn|seedling|isotope|tracer|nanosims|sip|root|soil|forest']

# Manual objective review supplements conservative lexical screening; no labels used.
DRIFT_BRANCHES={5:{'b0024','b0027'},7:{'b0009','b0010'},8:{'b0022'}}
DRIFT_TERMS={3:r'Bering Strait|Ekven|ocean circulation models',4:r'meeting corpora',6:r'boreholes',7:r'hearing aids|cochlear|criminal trial',8:r'fractured rock|Silva Cell'}

def drift(index,query,branch):
    return branch in DRIFT_BRANCHES.get(index,set()) or bool(re.search(DRIFT_TERMS.get(index,r'(?!)'),query,re.I))

def terms(q):
    return set(re.findall(r'[a-z0-9]{3,}',q.lower()))-ENGLISH_STOP_WORDS

def phase(batch):
    n=int(batch[-4:]);return 'early' if n<=9 else 'middle' if n<=18 else 'late'

def similarity(query,docs):
    if not docs:return []
    v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True)
    m=v.fit_transform(docs)
    return (m@v.transform([query]).T).toarray().ravel().tolist()

def validate_run(run):
    events=run['events'];fs={};pub=[];counts=Counter();snapshots={};branches={}
    normal_path=next((run['run']/'gpt_researcher').glob('*.json'))
    normal=smoke.read(normal_path)
    for i,e in enumerate(events,1):
        assert e['sequence']==i
        k=e['event']
        if k=='branch_created':branches[e['branch']['id']]=deepcopy(e['branch'])
        elif k=='branch_active':branches[e['branch_id']]['status']='active'
        elif k=='branch_processed':branches[e['branch_id']]['status']=e['status']
        elif k=='round_published':
            pub.append(e['legacy_round_id'])
            for b in e['branch_ids']:
                if branches[b]['status']=='processed':branches[b]['status']='done'
        elif k=='search_dispatch':counts['search']+=1
        elif k=='read_dispatch':counts['read']+=e['count']
        elif k=='candidate_proposed':counts['proposal']+=1
        elif k=='batch_snapshot':
            assert [(b['id'],b['status']) for b in e['branches']]==[(b['id'],b['status']) for b in branches.values()]
            a=e['accounting'];u=a['budget_usage']
            assert u==dict(used_steps=counts['proposal']+len(pub),used_searches=counts['search'],used_reads=counts['read'],used_tokens=a['tracked_tokens']['input_tokens']+a['tracked_tokens']['output_tokens'])
            pages=canonical_pages(normal,pub)
            snapshots[e['batch_id']]=numeric.capture(run['question'],e['branches'],{p['item_id']:SimpleNamespace(**p) for p in pages},a['nominal_budget'],u)
        elif k=='candidate_feature_snapshot':
            cid=e['candidate_id'];assert cid not in fs;fs[cid]=e
            q=run['queries'][cid];source,s=run['snapshots'][q['batch_id']]
            o=observation.construct(source,q['query'])
            numeric.persisted_vector(e,question=run['question'],candidate=q['query'],context=o.text)
            assert e['context_before']==o.text and e['observation_tokens']==o.tokens and e['observation_overflow']==o.overflow
            assert all(e[key]==value for key,value in snapshots[e['batch_id']].items())
    assert set(fs)==set(run['queries']) and len(fs)==112
    for bid in run['snapshots']:
        group=[f for f in fs.values() if f['batch_id']==bid]
        assert len(group)==4 and len({f['numerical_snapshot_digest'] for f in group})==1
        dispatch=[e['sequence'] for e in events if e['event']=='candidate_dispatched' and run['queries'][e['candidate_id']]['batch_id']==bid]
        assert len(dispatch)==4 and max(f['sequence'] for f in group)<min(dispatch)
    for kind in ['candidate_dispatched','retrieval_invoked','candidate_finished']:
        assert Counter(e['candidate_id'] for e in events if e['event']==kind)==Counter({cid:1 for cid in fs})
    assert all(e['status']=='completed' for e in events if e['event']=='candidate_finished')
    metrics=smoke.read(run['run']/'metrics/summary.json');assert metrics['n_errors']==0 and metrics['n_with_report']==1
    run['features']=fs
    sizes=[f['observation_tokens'] for f in fs.values()]
    return dict(status='valid',batches=28,candidates=112,feature_snapshots=len(fs),overflows=sum(f['observation_overflow'] for f in fs.values()),tokens=dict(median=float(np.median(sizes)),p90=float(np.percentile(sizes,90)),max=max(sizes)),trace_sha256=sha(run['path']),canonical_sha256=sha(normal_path))

def screen(run,index):
    queries=list(run['queries'].values());families=[];rows=[]
    # Connected components: normalized token Jaccard >= .65 or exact normalized text.
    parent=list(range(len(queries)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,q in enumerate(queries):
        a=terms(q['query'])
        for j in range(i):
            b=terms(queries[j]['query'])
            if a and b and len(a&b)/len(a|b)>=.65:parent[root(i)]=root(j)
    rootids={}
    for i,q in enumerate(queries):
        fam=rootids.setdefault(root(i),f"{index:03}-family-{len(rootids)+1:03}")
        text=q['query']; flags=[]
        if annotation.action_invalid(text):flags.append('action_space')
        if drift(index,text,q['branch_id']):flags.append('objective_drift_review')
        if not re.search(DOMAINS[index-1],text,re.I):flags.append('topic_lexical_review')
        if len(terms(text))<3:flags.append('underspecified')
        if run['features'][q['id']]['observation_overflow']:flags.append('overflow')
        rows.append(dict(candidate_id=q['id'],candidate=text,batch=q['batch_id'],branch=q['branch_id'],family=fam,phase=phase(q['batch_id']),flags=flags,eligible=not flags))
    return rows

def bind_packet(run,batch,query,pid):
    source,snap=run['snapshots'][batch];o=observation.construct(source,query)
    if o.overflow:raise ValueError('Selected packet overflow')
    original=next(f for f in run['features'].values() if f['batch_id']==batch)
    # Retain original numerical snapshot verbatim; only the candidate/context binding changes.
    base={k:deepcopy(original[k]) for k in ['feature_contract','feature_names','raw_vector','features','nominal_maxima','actual_usage','source_state_digest','canonical_evidence_ids','numerical_snapshot_digest']}
    bound=numeric.bind(base,batch_id=batch,candidate_id=pid,question=run['question'],candidate=query,context=o.text,observation_version=o.version)
    numeric.persisted_vector(bound,question=run['question'],candidate=query,context=o.text)
    assert bound['raw_vector']==original['raw_vector'] and bound['numerical_snapshot_digest']==original['numerical_snapshot_digest']
    return dict(original_question=run['question'],context_before=o.text,candidate_subquery=query),bound,o,original['candidate_id']

def select(run,screened,index):
    eligible=[r for r in screened if r['eligible']];selected=[];families=set();batches=set()
    def add(r):selected.append(r);families.add(r['family']);batches.add(r['batch'])
    # Four early originals, each with a later selected, newly visible finding-bearing own branch.
    opportunities=[]
    for r in eligible:
        if int(r['batch'][-4:])>8:continue
        source,_=run['snapshots'][r['batch']];old=observation.construct(source,r['candidate']).text
        for batch,(later,_) in run['snapshots'].items():
            if int(batch[-4:])<=9 or drift(index,later.current_need,later.current_task_id):continue
            o=observation.construct(later,r['candidate'])
            if o.overflow or r['branch'] not in o.selected_task_ids:continue
            b=next(b for b in later.published_branches if b.task_id==r['branch'])
            fresh=[f for f in b.findings if f not in old]
            if not fresh:continue
            scores=similarity(r['candidate'],fresh)
            score=max(scores,default=0)
            if score>0:opportunities.append((score,r,batch,fresh[scores.index(score)]))
    replay=[];usedlater=Counter()
    for score,r,batch,anchor in sorted(opportunities,key=lambda x:(-x[0],x[1]['candidate_id'],x[2])):
        if r['family'] in families or r['batch'] in batches or usedlater[batch]>=1:continue
        add(r);replay.append((r,batch,anchor,score));usedlater[batch]+=1
        if len(replay)==4:break
    if len(replay)!=4:raise ValueError('Insufficient replay opportunities')
    # 16 originals: cap one per batch and one per lexical family, with phase quotas.
    for ph,quota in [('early',6),('middle',5),('late',5)]:
        while sum(r['phase']==ph for r in selected)<quota:
            available=[r for r in eligible if r['phase']==ph and r['family'] not in families and r['batch'] not in batches]
            if not available:raise ValueError('Insufficient diverse originals')
            # Prefer qualification-bearing wording; maximize branch/lexical diversity, stable ties.
            def rank(r):
                gap=bool(re.search(r'compar|versus|effect|method|measure|uncertain|field|region|bias|condition|valid|duration',r['candidate'],re.I))
                overlap=max((len(terms(r['candidate'])&terms(s['candidate']))/max(1,len(terms(r['candidate'])|terms(s['candidate']))) for s in selected),default=0)
                return (-gap,overlap,r['candidate_id'])
            add(min(available,key=rank))
    choices=[dict(kind='original',batch=r['batch'],query=r['candidate'],family=r['family'],source_candidate=r['candidate_id'],reason='Phase/branch and lexical-family diversity; qualification wording preferred, no label assumed.') for r in selected]
    for r,batch,anchor,score in replay:
        choices.append(dict(kind='replay',batch=batch,query=r['candidate'],family=r['family'],source_candidate=r['candidate_id'],visible_anchor=anchor,lexical_answer_proxy=score,reason='Exact earlier query; newly published own-branch finding selected. Relevance does not certify resolution or SKIP.'))
    # Two explicit follow-up variants, one middle and one late. No invented qualifiers/conflicts.
    occupied=Counter(c['batch'] for c in choices)
    for ph in ['middle','late']:
        found=False
        for batch,(source,_) in run['snapshots'].items():
            if phase(batch)!=ph or occupied[batch]>=2 or drift(index,source.current_need,source.current_task_id):continue
            for b in source.published_branches:
                for f in b.followups:
                    if annotation.action_invalid(f) or not re.search(DOMAINS[index-1],f,re.I):continue
                    query=f if re.match(r'^(find|search|retrieve|locate)\b',f,re.I) else 'Find published evidence addressing: '+f
                    if any(len(terms(query)&terms(c['query']))/max(1,len(terms(query)|terms(c['query'])))>=.5 for c in choices):continue
                    o=observation.construct(source,query)
                    if o.overflow or not any(f in b['followups'] for b in json.loads(o.text)['branches']):continue
                    choices.append(dict(kind='variant',batch=batch,query=query,family=f'{index:03}-variant-{ph}',source_candidate=None,visible_anchor=f,reason='Explicit visible follow-up only; no invented condition or expected label.'))
                    occupied[batch]+=1;found=True;break
                if found:break
            if found:break
        if not found:raise ValueError('No conservative variant')
    assert len(choices)==22 and max(Counter(c['batch'] for c in choices).values())<=2
    return sorted(choices,key=lambda c:(c['batch'],c['kind'],c['query']))

def write_audit(out,rows):
    smoke.atomic(out/'human_audit.json',dict(training_accepted=0,validation_accepted=0,decisions=rows))
    lines=['# Candidate pool — no annotations or accepted training rows','', 'Replay links are opportunities, not asserted label reversals. Drift flags are heuristic, not semantic certification.','']
    for r in rows:
        lines.extend([f"## {r['packet_id']} — {r['source_split']} / {r['kind']} / {r['phase']}",r['candidate'],'',f"Family: {r['family']}; batch: {r['batch']}; tokens: {r['context_tokens']}; depth: {r['depth']}",f"Selection: {r['reason']}",f"Visible anchor: {r.get('visible_anchor')}",f"Selected goals: {json.dumps(r['selected_goals'],ensure_ascii=False)}",f"Numerical digest: {r['numerical_snapshot_digest']}; source candidate: {r['numerical_source_candidate']}",f"Context drift cautions: {r['source_topic_drift_flags']}",f"Label/category: {r.get('annotation_label')} / {r.get('category')}; validation: {r['structural_validation']}",f"Witnesses: {json.dumps(r.get('witnesses',[]),ensure_ascii=False)}",f"Rationale: {r.get('rationale')}",''])
    (out/'human_audit.md').write_text('\n'.join(lines)+'\n')

def prepare(out=OUT):
    out=Path(out)
    if out.exists():raise ValueError('Preserve existing prepared directory')
    plan=smoke.read('data/development/real-source-v2/collection_plan.json')
    packets={};rows=[];vectors={};screening={};validation={}
    for index,entry in enumerate(plan['trajectories'],1):
        paths=list(Path(entry['output_root']).glob('*/development_trace/*.jsonl'))
        if len(paths)!=1:raise ValueError('Ambiguous source run')
        run=load_run(paths[0].parents[1]);assert run['question']==entry['question']
        split='train' if index<=6 else 'validation';assert entry['source_split']==split
        validation[entry['id']]=validate_run(run)
        screened=screen(run,index);screening[entry['id']]=screened
        for n,c in enumerate(select(run,screened,index),1):
            pid=f'{index:03}-{n:03}';packet,vector,o,sourcecid=bind_packet(run,c['batch'],c['query'],pid)
            packets[pid]=packet;vectors[pid]=vector
            source,s=run['snapshots'][c['batch']]
            rows.append(dict(**{k:v for k,v in c.items() if k!='query'},packet_id=pid,candidate=c['query'],source_id=entry['id'],source_split=split,phase=phase(c['batch']),depth=next(t.tree_depth for t in source.tasks if t.id==source.current_task_id),context_tokens=o.tokens,selected_goals=[b['goal'] for b in json.loads(o.text)['branches']],selected_task_ids=list(o.selected_task_ids),cross_branch=any(b['relation']=='other' for b in json.loads(o.text)['branches']),numerical_source_candidate=sourcecid,numerical_snapshot_digest=vector['numerical_snapshot_digest'],decision_snapshot_digest=vector['decision_snapshot_digest'],source_topic_drift_flags=[b.task_id+': peripheral objective/method transfer requires human review' for b in source.published_branches if b.task_id in o.selected_task_ids and (drift(index,b.goal,b.task_id) or not re.search(DOMAINS[index-1],b.goal,re.I))],annotation_label=None,category=None,witnesses=[],structural_validation='not_called',stage=f'annotation-{len(rows)+1:03}',training_accepted=False))
    out.mkdir(parents=True)
    smoke.atomic(out/'packets.json',packets);smoke.atomic(out/'numerical_snapshots.json',vectors)
    smoke.atomic(out/'screening.json',screening);smoke.atomic(out/'source_validation.json',validation)
    smoke.atomic(out/'selection_provenance.json',rows)
    previews={pid:smoke.request(request(p),16000) for pid,p in packets.items()};smoke.atomic(out/'request_previews.json',previews)
    import adr.development.task_interpretation_regression as role
    smoke.atomic(out/'contract.json',dict(version=VERSION,observation=observation.VERSION,numerical=numeric.VERSION,supervision=annotation.VERSION,model='gpt-5-mini',maximum_calls=len(rows),attempts_per_packet=1,sdk_retries=0,files={name:sha(out/name) for name in ['packets.json','numerical_snapshots.json','selection_provenance.json','request_previews.json','source_validation.json','screening.json']},code={m.__name__:sha(m.__file__) for m in [observation,numeric,annotation,smoke,role]},runner_sha256=sha(__file__)))
    sizes=[r['context_tokens'] for r in rows]
    manifest=dict(status='prepared_offline',packets=len(rows),source_split=dict(Counter(r['source_split'] for r in rows)),composition=dict(Counter(r['kind'] for r in rows)),phases=dict(Counter(r['phase'] for r in rows)),depths=dict(Counter(r['depth'] for r in rows)),cross_branch=sum(r['cross_branch'] for r in rows),tokens=dict(min=min(sizes),median=float(np.median(sizes)),p90=float(np.percentile(sizes,90)),max=max(sizes)),screen_flags=dict(Counter(f for ss in screening.values() for r in ss for f in r['flags'])),originals_screened=sum(map(len,screening.values())),training_accepted=0,validation_accepted=0)
    smoke.atomic(out/'manifest.json',manifest);write_audit(out,rows);return manifest

async def annotate(out):
    out=Path(out);contract=smoke.read(out/'contract.json')
    if contract['runner_sha256']!=sha(__file__):raise ValueError('Runner changed')
    import importlib
    for name,h in contract['code'].items():
        if sha(importlib.import_module(name).__file__)!=h:raise ValueError('Code contract changed')
    for name,h in contract['files'].items():
        if sha(out/name)!=h:raise ValueError('Prepared artifact changed')
    packets=smoke.read(out/'packets.json');rows=smoke.read(out/'selection_provenance.json');vectors=smoke.read(out/'numerical_snapshots.json');previews=smoke.read(out/'request_previews.json')
    for pid,p in packets.items():
        assert set(p)=={'original_question','context_before','candidate_subquery'}
        numeric.persisted_vector(vectors[pid],question=p['original_question'],candidate=p['candidate_subquery'],context=p['context_before'])
        assert previews[pid]==smoke.request(request(p),16000)
    client=None
    def get_client():
        nonlocal client
        if client is None:client=smoke.azure_client(out)
        return client
    with smoke.lock(out):
        try:
            for r in rows:
                p=packets[r['packet_id']]
                try:
                    v=await smoke.stage(out,r['stage'],request(p),16000,get_client,lambda t,p=p:annotation.validate(t,p))
                    r.update(annotation_label=v['label'],category=v['category'],witnesses=v['witnesses'],rationale=v['rationale'],proposal=v,structural_validation='validated')
                except smoke.StageFailure:r['structural_validation']='failed_or_uncertain_no_retry'
                write_audit(out,rows)
        finally:
            if client is not None:await client.close()
        smoke.atomic(out/'annotation_status.json',dict(status='awaiting_human_audit',failures=sum(r['structural_validation']!='validated' for r in rows),usage=smoke.usage_summary(out),accepted_rows=0))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--allow-api',action='store_true');a=p.parse_args()
    if a.allow_api:asyncio.run(annotate(a.output))
    else:print(json.dumps(prepare(a.output)))

if __name__=='__main__':main()
