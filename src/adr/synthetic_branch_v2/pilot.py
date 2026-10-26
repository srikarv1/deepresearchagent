"""Four-question, audit-only pilot. Default mode prepares requests without calls."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Literal
import yaml
from pydantic import Field
from adr.synthetic_branch_v2 import smoke, pilot_annotation as annotation, pilot_ledger
from adr.synthetic_branch_v2.source import Strict, World, compile_world, digest
from adr.synthetic_branch_v2.smoke_prompts import WORLD_INSTRUCTIONS

VERSION='synthetic-branch-pilot-v1'

class Question(Strict):
    id: str
    question: str
    purpose: str

class Config(Strict):
    output: Path
    model: Literal['gpt-5-mini']='gpt-5-mini'
    questions: list[Question] = Field(min_length=4,max_length=4)
    target_candidates: Literal[18]=18
    min_candidates: Literal[15]=15
    max_candidates: Literal[20]=20
    world_completion_tokens: Literal[32000]=32000
    annotation_completion_tokens: Literal[16000]=16000
    max_attempts_per_stage: Literal[1]=1


def world_request(question):
    lifecycle=WORLD_INSTRUCTIONS.split('Each task is planned once')[1].split('Numerical ledger:')[0]
    prompt='''Author one fictional world for the EXACT supplied original_question and question_id.
Return JSON conforming to world_schema. All claims/sources are synthetic; use
plausible fictional titles and https://...invalid/... source URLs, not verified claims.
Target 18 candidate decisions, allowed 15–20 total, across early/middle/late.
Suggested shape: 2 root tasks ×3 candidates; publish; 3 children ×2; publish;
2 late tasks ×3. Initial observations have zero results, middle 1–3, late >3.
Do not supply labels, annotation rationales, witness fields, intended-negative
flags, or pair IDs. Actual execution events are not supervision targets.
Some specific needs may become answered; include plausible redundant searches
and genuinely unresolved needs without prescribing SKIPs or label quotas.
'''+ 'Each task is planned once'+lifecycle+'''
All candidates/follow-ups must ask to find/search/retrieve existing evidence.
Good: Find published analyses comparing metrics; Retrieve reports of experiments.
Bad: Extract definitions; Find reports and summarize them; Contact authors.
Do not perform calculations, synthesis, summaries or physical work ourselves.
Keep findings as the first up to three cited learning strings VERBATIM rather
than broader independently invented summaries. Findings may be qualified, null,
conflicting, or incomplete. Do not upgrade suggestions into established results.
end_batch inherits its owning preceding batch: NO task_id field.
cited_insights is EXACT insight text -> URL, e.g.
{"The fictional report observed an association only in rural schools.":"https://study.invalid/report"}.
Never use IDs as insight keys or concatenate claim and URL into one value.

Do NOT author research_tokens or numerical features. Local versioned accounting
renders simulated planning/query-generation/processing inputs and outputs and
counts visible tokens. Author only semantic events, planning query strings,
executed proposal IDs in proposal order, new page excerpts <=2000 characters,
and read_attempts >= pages. Searches and reads themselves are not LLM calls.
No pruning, no budget padding. Local nominal maxima 32/64/128/400000, unenforced.
End all active tasks/batches and publish all completed surviving planner rounds.
'''
    return [{'role':'system','content':prompt},{'role':'user','content':json.dumps(dict(
        question_id=question.id,original_question=question.question,world_schema=pilot_ledger.author_schema()),ensure_ascii=False,sort_keys=True)}]


def validate_world(text, q, cfg):
    world,proxy=pilot_ledger.derive(json.loads(text))
    if (world.question_id,world.original_question)!=(q.id,q.question):
        raise ValueError('Pilot question identity changed')
    packets,audit,_=compile_world(world)
    counts=[a['published_count'] for a in audit.values()]
    if not cfg.min_candidates<=len(packets)<=cfg.max_candidates:
        raise ValueError('Pilot candidate count outside 15–20')
    if not (0 in counts and any(0<n<=3 for n in counts) and any(n>3 for n in counts)):
        raise ValueError('Pilot requires early middle late publication states')
    # Retain canonical compiler's strict lifecycle rules; additionally require a
    # complete authored world rather than silently accepting unpublished tails.
    produced={e.task_id for e in world.events if e.kind=='process'}
    published={t for e in world.events if e.kind=='publish' for t in e.task_ids}
    planned={t.id for e in world.events if e.kind=='plan' for t in e.tasks}
    dropped={e.task_id for e in world.events if e.kind=='drop'}
    if produced!=published or planned!=published|dropped:
        raise ValueError('Pilot world leaves unfinished or unpublished tasks')
    return dict(world=world.model_dump(),visible_token_proxy=proxy)


def contract(cfg):
    base=smoke.contract(smoke.Config(output=cfg.output))
    return dict(version=VERSION,annotation_version=annotation.VERSION,ledger_version=pilot_ledger.VERSION,
        config=cfg.model_dump(mode='json'),dependency_contract={k:base[k] for k in ('code_sha256','runner_sha256','dependencies')},
        pilot_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(annotation.__file__),Path(pilot_ledger.__file__))},
        maximum_calls=4*(1+cfg.max_candidates),training_rows=0)


def distribution(values):
    values=sorted(values)
    if not values:return dict(count=0)
    import numpy as np
    return dict(count=len(values),min=values[0],median=float(np.median(values)),p90=float(np.percentile(values,90)),max=values[-1])


def write_audit(root, worlds):
    rows=[r for w in worlds for r in w.get('decisions',[])]
    proposed=[r for r in rows if r.get('proposal')]
    passed=[r for r in proposed if r['status']=='validated']
    pairs=[]
    # Purely descriptive repeated-query comparisons, not imposed linkage/labels.
    for w in worlds:
        seen={}
        for r in w.get('decisions',[]):
            for early in seen.get(r['candidate_query'],[]):
                pairs.append(dict(question_id=w['question_id'],earlier=early['candidate_id'],later=r['candidate_id'],
                    same_snapshot=early['event_index']==r['event_index'],
                    earlier_label=(early.get('proposal') or {}).get('label'),later_label=(r.get('proposal') or {}).get('label'),
                    newly_visible_witnesses=[v for v in (r.get('proposal') or {}).get('witnesses',[])
                        if v['text'] not in early['context_before']],human_review='pending_not_certified_counterfactual'))
            seen.setdefault(r['candidate_query'],[]).append(r)
    count=lambda key:dict(Counter(key(r) for r in proposed))
    summary=dict(training_rows=0,human_review='pending',questions=len(worlds),decisions=len(rows),
        label_distribution=count(lambda r:r['proposal']['label']),categories=count(lambda r:r['proposal']['category']),
        ambiguous_reasons=dict(Counter(r['proposal']['category'] for r in proposed if r['proposal']['label']=='AMBIGUOUS')),
        ambiguous_rate=sum(r['proposal']['label']=='AMBIGUOUS' for r in proposed)/len(proposed) if proposed else None,
        valid_retrieval_labels=dict(Counter(r['proposal']['label'] for r in passed if r['action_space_screen']=='not_flagged'
            and r['proposal']['action_eligibility']=='retrieval')),
        structural_rejections=sum(r['status']=='validation_failed' for r in rows),
        structural_rejection_rate=sum(r['status']=='validation_failed' for r in rows)/len(proposed) if proposed else None,
        witness_failures=sum(r.get('validation_code')=='witness_location_or_text' for r in rows),
        action_space_failures=sum(r.get('validation_code')=='action_space_binary_or_category' for r in rows),
        action_space_flags=sum(r['action_space_screen']=='invalid' for r in rows),
        observation_tokens=distribution([r['observation_tokens'] for r in rows]),
        phase_distribution=dict(Counter(r['phase'] for r in rows)),repeated_query_comparisons=pairs,
        per_question=[dict(question_id=w['question_id'],question=w['question'],purpose=w['purpose'],status=w['status'],
            decisions=len(w.get('decisions',[])),
            labels=dict(Counter(r['proposal']['label'] for r in w.get('decisions',[]) if r.get('proposal'))),
            phases=dict(Counter(r['phase'] for r in w.get('decisions',[]))),
            observation_tokens=distribution([r['observation_tokens'] for r in w.get('decisions',[])])) for w in worlds],
        counts_are_structural_not_semantic_accuracy=True)
    smoke.atomic(root/'human_audit.json',dict(summary=summary,worlds=worlds))
    lines=['# Four-question pilot — human audit only','',json.dumps(summary,ensure_ascii=False,indent=2),'']
    for w in worlds:
        lines += [f"## {w['question_id']}: {w['question']}",f"World status: {w['status']}",'']
        for r in w.get('decisions',[]):
            p=r.get('proposal') or {}
            lines += [f"### {r['candidate_id']} ({r['phase']}, {r['observation_tokens']} tokens)",
                r['candidate_query'],f"Label/category: {p.get('label')} / {p.get('category')}",
                f"Validation: {r['status']} {r.get('validation_code','')}; action screen: {r['action_space_screen']}",
                'Witnesses: '+json.dumps(p.get('witnesses',[]),ensure_ascii=False),
                'Gap: '+json.dumps(p.get('gap'),ensure_ascii=False),
                'Rationale: '+p.get('rationale',''),
                'Selected goals: '+json.dumps(r['selected_branch_goals'],ensure_ascii=False),
                f"Source executed: {r['source_executed']}; agreement: {r.get('ledger_agreement')}",
                'Exact context: `'+r['context_before']+'`','']
    tmp=root/'human_audit.md.tmp'; tmp.write_text('\n'.join(lines)); tmp.replace(root/'human_audit.md')
    return summary


async def run(cfg, *, resume=False, allow_api=False, client_factory=None):
    if len({q.id for q in cfg.questions})!=4 or len({q.question for q in cfg.questions})!=4:
        raise ValueError('Four independent unique question identities required')
    if any(not q.id.replace('-','').isalnum() for q in cfg.questions):
        raise ValueError('Question IDs must be safe directory components')
    root=smoke.initialize(cfg,resume); client=None
    with smoke.lock(root):
        expected=contract(cfg)
        if resume:
            if smoke.read(root/'contract.json')!=expected:raise ValueError('Pilot contract changed; use a new directory')
        else:
            smoke.atomic(root/'contract.json',expected)
            for q in cfg.questions:
                folder=root/q.id; folder.mkdir()
                smoke.atomic(folder/'world_request_preview.json',smoke.request(world_request(q),cfg.world_completion_tokens))
            smoke.atomic(root/'manifest.json',dict(status='prepared_offline',training_rows=0,maximum_calls=84,provider_calls_reserved=0))
        if not allow_api:return dict(status='prepared_offline',api_calls=0,output=str(root))
        def get_client():
            nonlocal client
            if client is None:client=client_factory() if client_factory else smoke.azure_client(root)
            return client
        worlds=[]
        try:
            for q in cfg.questions:
                folder=root/q.id
                entry=dict(question_id=q.id,question=q.question,purpose=q.purpose,status='pending',decisions=[])
                worlds.append(entry)
                try:
                    result=await smoke.stage(folder,'world',world_request(q),cfg.world_completion_tokens,get_client,
                        lambda text:validate_world(text,q,cfg))
                except smoke.StageFailure:
                    entry['status']='world_failed_no_retry'
                    print(json.dumps(dict(question_id=q.id,stage='world',status=entry['status'],automatic_retry=False)),flush=True)
                    write_audit(root,worlds); continue
                world=World.model_validate(result['world']); packets,audit,execution=compile_world(world)
                smoke.atomic(folder/'source_world.json',result['world'])
                smoke.atomic(folder/'visible_token_proxy.json',result['visible_token_proxy'])
                smoke.atomic(folder/'compiled_packets.json',packets); smoke.atomic(folder/'compilation_audit.json',audit)
                smoke.atomic(folder/'numerical_ledger.json',{**smoke.ledger(world,audit),
                    'research_tokens_origin':pilot_ledger.VERSION,'teacher_authored':'semantic events, planning queries, execution IDs, pages, read attempts; NOT token deltas'})
                entry['status']='awaiting_human_audit'
                for i,(cid,packet) in enumerate(packets.items(),1):
                    meta=audit[cid]; view=json.loads(packet['context_before']); name=f'annotation-{i:03d}'
                    row=dict(candidate_id=cid,candidate_query=packet['candidate_subquery'],event_index=meta['event_index'],
                        context_before=packet['context_before'],phase='early' if meta['published_count']==0 else 'middle' if meta['published_count']<=3 else 'late',
                        selected_branch_goals=[b['goal'] for b in view['branches']],observation_tokens=meta['tokens'],
                        action_space_screen='invalid' if annotation.action_invalid(packet['candidate_subquery']) else 'not_flagged',
                        source_executed=execution[cid],status='not_requested',proposal=None)
                    entry['decisions'].append(row)
                    if meta['overflow']:
                        row['status']='overflow_execute_bypass_excluded'; continue
                    def validate(text):
                        try:return annotation.validate(text,packet)
                        except ValueError as exc:
                            # Only deterministic codes, never arbitrary validation text.
                            allowed={'witness_location_or_text','action_space_binary_or_category','uncertain_action_requires_ambiguous',
                                'label_category','skip_requires_answer_evidence','execute_coverage','no_evidence_category_with_partial_coverage',
                                'execute_requires_specific_gap','gap_not_requested','generic_completeness_not_gap','partial_coverage_requires_evidence'}
                            smoke.atomic(folder/'stages'/name/'validation_code.json',dict(code=str(exc) if str(exc) in allowed else 'schema'))
                            raise
                    try:
                        row['proposal']=await smoke.stage(folder,name,annotation.request(packet),cfg.annotation_completion_tokens,get_client,validate)
                        row['status']='validated'
                    except smoke.StageFailure:
                        stage=folder/'stages'/name
                        row['status']=smoke.read(stage/'outcome.json')['status'] if (stage/'outcome.json').exists() else 'uncertain_provider_completion'
                        if (stage/'validation_code.json').exists():row['validation_code']=smoke.read(stage/'validation_code.json')['code']
                        if (stage/'response_bundle.json').exists():
                            try:row['proposal']=annotation.Proposal.model_validate_json(smoke.read(stage/'response_bundle.json')['assistant_content']).model_dump()
                            except ValueError:pass
                    if row['status']!='validated':
                        print(json.dumps(dict(question_id=q.id,stage=name,status=row['status'],validation_code=row.get('validation_code'),automatic_retry=False)),flush=True)
                    if row['proposal'] and row['proposal']['label'] in ('EXECUTE','SKIP'):
                        row['ledger_agreement']=execution[cid]==(row['proposal']['label']=='EXECUTE')
                    write_audit(root,worlds)
                write_audit(root,worlds)
            summary=write_audit(root,worlds)
            result=dict(status='awaiting_human_audit' if all(w['status']=='awaiting_human_audit' for w in worlds) else 'partial_failures_awaiting_audit',
                training_rows=0,maximum_calls=84,provider_calls_reserved=len(list(root.glob('*/stages/*/started.json'))),
                api_usage={q.id:smoke.usage_summary(root/q.id) for q in cfg.questions},summary=summary)
            smoke.atomic(root/'manifest.json',result)
            return result
        finally:
            if client is not None:await client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/synthetic_branch_pilot.yaml'))
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-api',action='store_true')
    args=p.parse_args();cfg=Config.model_validate(yaml.safe_load(args.config.read_text()))
    result=asyncio.run(run(cfg,resume=args.resume,allow_api=args.allow_api))
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','api_usage')}))

if __name__=='__main__':main()
