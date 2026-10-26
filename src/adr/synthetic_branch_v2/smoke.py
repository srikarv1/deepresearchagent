"""Tiny staged teacher smoke. Offline preparation by default; never accepts training rows."""
import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
import yaml
from pydantic import ValidationError
from adr.synthetic_branch_v2.source import Strict, World, compile_world, digest, MAXIMA
from adr.synthetic_branch_v2.supervision import Annotation, validate_annotation
from adr.synthetic_branch_v2.smoke_prompts import VERSION, Proposal, world_request, annotation_request
from adr.synthetic_v2.response_status import generation_failure

class Config(Strict):
    output: Path
    model: Literal['gpt-5-mini'] = 'gpt-5-mini'
    world_completion_tokens: Literal[32000] = 32000
    annotation_completion_tokens: Literal[16000] = 16000
    min_candidates: Literal[8] = 8
    max_candidates: Literal[12] = 12
    max_attempts_per_stage: Literal[1] = 1

class StageFailure(RuntimeError):
    pass


def atomic(path, value):
    path=Path(path)
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def lock(path):
    with (path/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another smoke invocation owns this directory') from None
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def contract(config):
    from adr.synthetic_branch_v2 import source, supervision, smoke_prompts
    from adr.mlp import branch_observation_v2, embedding_limits, live_context
    from adr.llm import openai_compat
    from adr.synthetic_v2 import action_space, response_status
    from adr.synthetic import schema
    from adr.core import state, types
    modules=[source,supervision,smoke_prompts,branch_observation_v2,embedding_limits,live_context,
             openai_compat,action_space,response_status,schema,state,types]
    import sklearn, tiktoken
    return {'version':VERSION,'config':config.model_dump(mode='json'),
        'code_sha256':{m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependencies':{'sklearn':sklearn.__version__,'tiktoken':tiktoken.__version__},
        'sdk_retries':0,'max_attempts_per_stage':1,'max_provider_calls':13,
        'semantic_review':'human_pending','training_rows':0}


def initialize(config, resume):
    out=config.output
    if resume:
        if not out.is_dir():
            raise ValueError('Resume directory does not exist')
    else:
        out.mkdir(parents=True,exist_ok=False)
    return out


def check_contract(out, config, resume):
    expected=contract(config)
    if resume:
        old=read(out/'contract.json')
        if old != expected:
            fields=[k for k in expected if old.get(k)!=expected[k]]
            raise ValueError('Smoke contract changed: '+', '.join(fields))
    else:
        atomic(out/'contract.json',expected)
        atomic(out/'world_request_preview.json',request(world_request(),config.world_completion_tokens))
        atomic(out/'manifest.json',{'status':'prepared_offline','provider_calls':0,'training_rows':0,
            'max_attempts_per_stage':1,'annotation_calls_receive':'only three semantic inputs plus schema/instructions'})


def request(messages, allowance):
    return {'model':'gpt-5-mini','messages':messages,'max_completion_tokens':allowance,
            'response_format':{'type':'json_object'},'sdk_retries':0}


def safe_validation_error(exc):
    info={'error_type':type(exc).__name__,'message':'Local validation failed; raw response preserved for offline investigation'}
    allowed = {
        'Smoke requires 8–12 candidates',
        'Smoke requires empty, middle and top-three-overflow publication states',
        'Witness absent from selected visible observation', 'SKIP requires explicit visible grounding',
        'Category needs visible semantic evidence', 'Label/category mismatch',
        'Counterfactual decisive finding must be visible at later SKIP',
        'Task authored more than once', 'Parent must be published before child planning',
        'Only one active task is supported', 'Cannot execute completed/dropped/unknown task',
        'A frozen batch must end before any other event', 'Duplicate candidate identity',
        'Execution IDs must be a unique proposal-order subset', 'No batch to end',
        'Retrieval without an executed query', 'Read accounting cannot explain new page records',
        'Feature page authored more than once', 'Only active task may produce a result',
        'Semantic result has no prior executed research query',
        'Cited insights need text and source URL identities', 'Invalid task drop',
        'Invalid publication boundary', 'Publication must commit the complete planner round',
        'Publication must preserve planner order', 'Only processed unpublished results may publish',
        'Incomplete source lifecycle', 'Planning requires no active task',
    }
    if str(exc) in allowed:
        info['detail']=str(exc)
    if isinstance(exc,ValidationError):
        info['fields']=[{'location':list(e['loc']),'type':e['type']}
                        for e in exc.errors(include_input=False,include_context=False,include_url=False)]
    return info


async def stage(out, name, messages, allowance, get_client, validator):
    """Exactly one attempt ever. Response bundle committed BEFORE JSON validation.

    Started-without-response is uncertain, not permission to call again. A saved
    response without outcome can be locally validated on resume, without API use.
    """
    folder=out/'stages'/name
    folder.mkdir(parents=True,exist_ok=True)
    req=request(messages,allowance)
    if (folder/'request.json').exists():
        if read(folder/'request.json')!=req:
            raise ValueError('Checkpoint request differs for stage '+name)
    else:
        atomic(folder/'request.json',req)
    if (folder/'outcome.json').exists():
        outcome=read(folder/'outcome.json')
        if outcome['status']!='validated':
            raise StageFailure(name+' is checkpointed as failed; no retry permitted')
        bundle=read(folder/'response_bundle.json')
        if digest(bundle)!=outcome['response_digest']:
            raise ValueError('Response checkpoint integrity failure')
        value=validator(bundle['assistant_content'])
        if digest(value)!=outcome['validated_digest']:
            raise ValueError('Validated checkpoint integrity failure')
        return value
    if not (folder/'response_bundle.json').exists():
        if (folder/'started.json').exists():
            raise StageFailure(name+' has uncertain provider completion; refusing another call')
        client=get_client()  # Credential/config failures do not consume a provider attempt.
        atomic(folder/'started.json',{'attempt':1,'maximum_attempts':1,'status':'in_flight'})
        print(json.dumps({'stage':name,'status':'calling','attempt':1,'maximum_attempts':1}),flush=True)
        try:
            response=await client.complete(messages,max_tokens=allowance,extra={'response_format':{'type':'json_object'}})
        except Exception as exc:
            status=getattr(exc,'status_code',None)
            atomic(folder/'outcome.json',{'status':'provider_failed','error_type':type(exc).__name__,
                'http_status':status if isinstance(status,int) else None,'arbitrary_error_text_omitted':True,
                'automatic_retry':False})
            raise StageFailure(name+' provider failure; safe checkpoint preserved') from None
        # One atomic bundle prevents a usage/content split from causing a repeat call.
        atomic(folder/'response_bundle.json',{'raw_provider_response':response.raw,
            'assistant_content':response.text,'normalized_usage':response.usage.model_dump()})
    bundle=read(folder/'response_bundle.json')
    atomic(folder/'usage.json',{'provider_usage':bundle['raw_provider_response'].get('usage'),
                              'normalized_usage':bundle['normalized_usage']})
    failure=generation_failure(bundle['assistant_content'],bundle['raw_provider_response'],allowance)
    if failure:
        atomic(folder/'outcome.json',{'status':'generation_failed',**failure,'automatic_retry':False})
        raise StageFailure(name+' generation-budget/empty-output failure')
    try:
        value=validator(bundle['assistant_content'])
    except (ValueError,KeyError,TypeError) as exc:
        atomic(folder/'outcome.json',{'status':'validation_failed',**safe_validation_error(exc),'automatic_retry':False})
        raise StageFailure(name+' local validation failure') from None
    print(json.dumps({'stage':name,'status':'response_preserved_and_locally_validated'}),flush=True)
    atomic(folder/'validated.json',value)
    atomic(folder/'outcome.json',{'status':'validated','response_digest':digest(bundle),
        'validated_digest':digest(value),'semantic_review':'not_performed','automatic_retry':False})
    return value


def validate_world(text, config):
    world=World.model_validate_json(text)
    packets,audit,_=compile_world(world)
    counts={v['published_count'] for v in audit.values()}
    if not config.min_candidates<=len(packets)<=config.max_candidates:
        raise ValueError('Smoke requires 8–12 candidates')
    if 0 not in counts or not any(0<n<=3 for n in counts) or not any(n>3 for n in counts):
        raise ValueError('Smoke requires empty, middle and top-three-overflow publication states')
    return world.model_dump()


def ledger(world, audit):
    totals=dict(used_steps=0,used_searches=0,used_reads=0,used_tokens=0)
    events=[]
    for i,e in enumerate(world.events):
        delta=dict.fromkeys(totals,0)
        if e.kind=='plan':
            delta.update(used_searches=len(e.planning_queries),used_tokens=e.research_tokens)
        elif e.kind=='batch':
            # Increment at end_batch; snapshot includes only prior proposals.
            batch_count=len(e.candidates)
        elif e.kind=='end_batch':
            delta.update(used_steps=batch_count,used_searches=len(e.executed_ids),
                         used_reads=e.read_attempts,used_tokens=e.research_tokens)
        elif e.kind=='process':
            delta['used_tokens']=e.research_tokens
        elif e.kind=='publish':
            delta['used_steps']=1
        if e.kind=='batch':
            for entry in audit.values():
                if entry['event_index']==i and any(entry['budget'][k]!=v for k,v in totals.items()):
                    raise ValueError('Numerical ledger disagrees with canonical compiler')
        totals={k:v+delta[k] for k,v in totals.items()}
        events.append({'event_index':i,'kind':e.kind,'delta':delta,'cumulative':dict(totals)})
    return {'nominal_maxima':MAXIMA,'enforce':False,'events':events,'final_usage':totals,
            'teacher_authored':'planning query strings, executed IDs, feature pages, read attempts, event research-token deltas',
            'derived':'counts/history/hierarchy/publication, cumulative usage, all 15 features',
            'not_provider_usage':True}


def pair_proposals(rows, packets, audit):
    """Local exact-query pairing only; never feed other observations to annotator."""
    used=set(); pairs=[]
    for late in rows:
        lp=late.get('proposal')
        if late['validation_status']!='needs_semantic_review' or not lp or lp['label']!='SKIP':
            continue
        witness=lp.get('decisive_witness')
        if not witness or late['candidate_id'] in used:
            continue
        lid=late['candidate_id']; text=witness['text']; index=witness['branch_index']
        if index is None: continue
        decisive_task=audit[lid]['selected_task_ids'][index]
        for early in rows:
            eid=early['candidate_id']; ep=early.get('proposal')
            if (eid in used or early['validation_status']!='needs_semantic_review' or not ep or ep['label']!='EXECUTE'
                or early['candidate_query']!=late['candidate_query']
                or audit[eid]['event_index']>=audit[lid]['event_index']
                or text in packets[eid]['context_before']
                or decisive_task in audit[eid]['published_task_ids']):
                continue
            pid=f'pair-{len(pairs)+1:03d}'
            early['counterfactual_pair_id']=late['counterfactual_pair_id']=pid
            used.update((eid,lid))
            pairs.append({'pair_id':pid,'earlier':eid,'later':lid,'decisive_witness':witness,
                          'status':'structural_candidate_pending_human_review'})
            break
    return pairs


def write_audit(out, rows, packets, audit):
    pairs=pair_proposals(rows,packets,audit)
    atomic(out/'human_audit.json',{'training_rows':0,'human_review':'pending','decisions':rows,'counterfactual_pairs':pairs})
    lines=['# Synthetic branch smoke — human review required','',
           'No decision is accepted for training. Exact contexts/features are in compiled_packets.json.', '']
    for row in rows:
        p=row.get('proposal') or {}
        lines += [f"## {row['candidate_id']}", '', f"Query: {row['candidate_query']}", '',
            f"Proposed label/category: {p.get('label','NOT RETURNED')} / {p.get('category','—')}",
            f"Status: {row['validation_status']}; context tokens: {row['observation_tokens']}",
            f"World-execution agreement: {row.get('execution_agreement')}; pair: {row.get('counterfactual_pair_id')}",
            '', 'Selected goals: '+json.dumps(row['selected_branch_goals'],ensure_ascii=False),
            '', 'Witnesses: '+json.dumps(p.get('witnesses',[]),ensure_ascii=False),
            '', 'Rationale: '+str(p.get('rationale','See preserved response/checkpoint')), '']
    target=out/'human_audit.md'; tmp=target.with_suffix('.md.tmp')
    tmp.write_text('\n'.join(lines),encoding='utf-8'); os.replace(tmp,target)


def usage_summary(out):
    records=[]
    for path in sorted((out/'stages').glob('*/response_bundle.json')):
        b=read(path); raw=b['raw_provider_response']; u=raw.get('usage') or {}
        records.append({'stage':path.parent.name,'reported_model':raw.get('model'),
            'finish_reason':(raw.get('choices') or [{}])[0].get('finish_reason'),
            'provider_usage':u,'normalized_usage':b['normalized_usage']})
    return records


async def run(config, *, resume=False, allow_api=False, client_factory=None):
    out=initialize(config,resume)
    client=None
    with lock(out):
        check_contract(out,config,resume)
        if not allow_api:
            return {'status':'prepared_offline','provider_calls':0,'output':str(out)}
        def get_client():
            nonlocal client
            if client is None:
                if client_factory is None:
                    client=azure_client(out)
                else:
                    client=client_factory()  # Mock injection for offline tests.
            return client
        try:
            source=await stage(out,'world',world_request(),config.world_completion_tokens,get_client,
                               lambda text:validate_world(text,config))
            world=World.model_validate(source)
            packets,audit,execution=compile_world(world)
            atomic(out/'source_world.json',source)
            atomic(out/'compiled_packets.json',packets)
            atomic(out/'compilation_audit.json',audit)
            atomic(out/'numerical_ledger.json',ledger(world,audit))
            rows=[]
            for cid,packet in packets.items():
                view=json.loads(packet['context_before']); meta=audit[cid]
                row={'candidate_id':cid,'candidate_query':packet['candidate_subquery'],
                    'selected_branch_goals':[b['goal'] for b in view['branches']],
                    'observation_tokens':meta['tokens'],'counterfactual_pair_id':None,
                    'world_executed':execution[cid],'execution_agreement':None,
                    'proposal':None,'human_review':'pending','validation_status':'not_requested'}
                rows.append(row)
            write_audit(out,rows,packets,audit)
            for i,(cid,packet) in enumerate(packets.items(),1):
                row=rows[i-1]; meta=audit[cid]
                if meta['overflow']:
                    row['validation_status']='overflow_execute_bypass_not_trainable'
                else:
                    def validate(text):
                        proposal=Proposal.model_validate_json(text)
                        annotation=Annotation(candidate_id=cid,packet_digest=meta['packet_digest'],**proposal.model_dump())
                        status=validate_annotation(annotation,packet,meta)
                        return {'proposal':proposal.model_dump(),'annotation':annotation.model_dump(),'status':status}
                    name=f'annotation-{i:03d}'
                    try:
                        result=await stage(out,name,annotation_request(packet),config.annotation_completion_tokens,get_client,validate)
                        row.update(proposal=result['proposal'],validation_status=result['status'])
                        if result['proposal']['label'] in ('EXECUTE','SKIP'):
                            row['execution_agreement']=execution[cid]==(result['proposal']['label']=='EXECUTE')
                    except StageFailure:
                        row['validation_status']='stage_failed_no_retry'
                        folder=out/'stages'/name
                        # Show parseable proposals even when witness/category checks fail.
                        if (folder/'response_bundle.json').exists():
                            try:
                                row['proposal']=Proposal.model_validate_json(read(folder/'response_bundle.json')['assistant_content']).model_dump()
                            except ValueError:
                                pass
                        row['stage_path']=str(folder.relative_to(out))
                        write_audit(out,rows,packets,audit)
                        raise
                write_audit(out,rows,packets,audit)
            manifest={'status':'awaiting_human_audit','worlds':1,'decisions':len(rows),
                'training_rows':0,'provider_calls_reserved':len(list((out/'stages').glob('*/started.json'))),
                'max_attempts_per_stage':1,'api_usage':usage_summary(out),
                'execution_disagreements':[r['candidate_id'] for r in rows if r['execution_agreement'] is False]}
            atomic(out/'manifest.json',manifest)
            return manifest
        except (StageFailure,ValueError) as exc:
            atomic(out/'manifest.json',{'status':'stopped_for_inspection','training_rows':0,
                'error_type':type(exc).__name__,'max_attempts_per_stage':1,
                'api_usage':usage_summary(out),'automatic_retry':False})
            raise
        finally:
            if client is not None:
                await client.close()


def azure_client(out):
    # No dotenv load or default endpoint fallback; user's terminal owns credentials.
    from adr.llm.factory import build_llm
    key=os.environ.get('OPENAI_API_KEY')
    base=os.environ.get('OPENAI_BASE_URL')
    if not key or not base:
        raise ValueError('OPENAI_API_KEY and OPENAI_BASE_URL must be set in this terminal')
    u=urlsplit(base)
    if u.scheme!='https' or not u.netloc or u.username or u.password or u.query or u.fragment or u.path.rstrip('/')!='/openai/v1':
        raise ValueError('Expected HTTPS OpenAI-compatible endpoint ending /openai/v1/')
    identity={'endpoint_sha256':hashlib.sha256(base.rstrip('/').encode()).hexdigest(),
              'api_key_env':'OPENAI_API_KEY','base_url_env':'OPENAI_BASE_URL'}
    path=out/'endpoint_identity.json'
    if path.exists() and read(path)!=identity:
        raise ValueError('Endpoint changed; existing smoke cannot resume against another endpoint')
    if not path.exists(): atomic(path,identity)
    client=build_llm({'provider':'azure_foundry','model':'gpt-5-mini','base_url':base,'api_key_env':'OPENAI_API_KEY'})
    client._client.max_retries=0
    return client


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/synthetic_branch_smoke.yaml'))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    config=Config.model_validate(yaml.safe_load(args.config.read_text()))
    try:
        result=asyncio.run(run(config,resume=args.resume,allow_api=args.allow_api))
    except (ValueError,StageFailure,FileExistsError) as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,
                          'message':'Inspect preserved stage outcomes; no automatic retries'}))
        raise SystemExit(1) from None
    print(json.dumps({k:v for k,v in result.items() if k!='api_usage'}))

if __name__=='__main__':main()
