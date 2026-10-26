"""Visible-token proxy v1. Not provider usage or a calibrated research-cost model."""
from copy import deepcopy
import json
from adr.mlp.embedding_limits import token_count
from adr.synthetic_branch_v2.source import World, compile_world

VERSION='synthetic-visible-operation-tokens-v1-cl100k'


def render(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def author_schema():
    schema=deepcopy(World.model_json_schema())
    for name in ('Plan','EndBatch','Process'):
        d=schema['$defs'][name]
        del d['properties']['research_tokens']
        d['required'].remove('research_tokens')
    return schema


def derive(authored):
    data=deepcopy(authored)
    for e in data['events']:
        if 'research_tokens' in e:
            raise ValueError('Teacher must not author research_tokens')
        if e['kind'] in ('plan','end_batch','process'): e['research_tokens']=0
    # Canonical validation/lifecycle remains authoritative; no schema repair.
    world=World.model_validate(data)
    compile_world(world)
    tasks={}; results={}; published=[]; pages={}; history=[]; operations=[]; batch=None
    def context(task=None):
        available=set(published)|({task} if task else set())
        return dict(original_question=world.original_question,
            current_need=tasks[task]['goal'] if task else None,
            completed_results=[results[t] for t in published],
            prior_queries=[q for t,q in history if t in available],
            relevant_excerpts=[p for t,p in pages.values() if t in available])
    def charge(index,role,input_value,output_value):
        inp=render(dict(operation=role,inputs=input_value)); out=render(output_value)
        counts=dict(input_tokens=token_count(inp),output_tokens=token_count(out))
        total=sum(counts.values()); data['events'][index]['research_tokens']=total
        operations.append(dict(event_index=index,operation=role,input_text=inp,output_text=out,**counts,total_tokens=total))
    for i,e in enumerate(data['events']):
        kind=e['kind']
        if kind=='plan':
            charge(i,'plan',context(),dict(tasks=e['tasks'],planning_queries=e['planning_queries']))
            tasks.update({t['id']:t for t in e['tasks']})
        elif kind=='batch':
            batch=e
            # Freeze query-generation inputs before any candidate retrieval.
            batch_input=context(e['task_id'])
        elif kind=='end_batch':
            charge(i,'generate_queries',batch_input,dict(queries=[c['query'] for c in batch['candidates']]))
            owner=batch['task_id']
            history.extend((owner,c['query']) for c in batch['candidates'] if c['id'] in e['executed_ids'])
            pages.update({p['id']:(owner,p) for p in e['pages']})
            batch=None
        elif kind=='process':
            result={k:e[k] for k in ('findings','followups','cited_insights')}
            charge(i,'process_research_results',context(e['task_id']),result)
            results[e['task_id']]=dict(goal=tasks[e['task_id']]['goal'],**result)
        elif kind=='publish': published.extend(e['task_ids'])
    final=World.model_validate(data)
    compile_world(final)
    return final,dict(version=VERSION,tokenizer='cl100k_base',provider_equivalent=False,
        interpretation='Visible input/output token estimate; omits hidden reasoning, real prompts and unmodeled calls; not a provider-token lower-bound guarantee.',
        training_eligible=False,operations=operations,total_tokens=sum(o['total_tokens'] for o in operations))
