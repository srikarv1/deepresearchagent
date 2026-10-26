"""Checkpointed synthetic-v2 production. Default is offline; paid calls require --allow-api."""
import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, model_validator
import yaml
from adr.synthetic.schema import StrictModel
from adr.synthetic.validate import load_heldout, near_duplicate
from adr.synthetic_v2.schema import FEATURE_VERSION, SCHEMA_VERSION
from adr.synthetic_v2.production_schema import ProductionTrajectory, expand_production, EVENT_VERSION
from adr.synthetic_v2.production_prompt import PROMPT_VERSION, teacher_messages
from adr.synthetic_v2.retry_feedback import ContractError, classify, feedback
from adr.synthetic_v2.generate import compile_trajectory
from adr.synthetic_v2.response_status import generation_failure
from adr.mlp.observation_v2 import VERSION
from adr.mlp.embedding_limits import token_count

COUNTS=[3,3,3,3,2,2,2,2]

class Assignment(StrictModel):
    topic: str = Field(min_length=3)
    structure: str = Field(min_length=3)

class Config(StrictModel):
    output_root: str
    heldout_path: str
    trajectories: int = Field(default=20,ge=1,le=20)
    max_attempts: Literal[3] = 3
    model: Literal['gpt-5-mini'] = 'gpt-5-mini'
    max_completion_tokens: Literal[32000] = 32000
    assignments: list[Assignment]

    @model_validator(mode='after')
    def assigned(self):
        if len(self.assignments)!=self.trajectories:
            raise ValueError('Each trajectory needs exactly one stable topic/structure assignment')
        if len({a.topic for a in self.assignments})!=len(self.assignments) or len({a.structure for a in self.assignments})!=len(self.assignments):
            raise ValueError('Topic and question-structure assignments must each be distinct')
        return self


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def sha(data):return hashlib.sha256(data).hexdigest()

def json_bytes(data):return (json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()

def jsonl(rows):return ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode()


def atomic(path,data):
    """Atomic replacement + durable content, with temporary files in same directory."""
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(name):os.unlink(name)


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


@contextmanager
def lock(directory):
    with (directory/'.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another generator is using this directory') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def registry(config):
    path=Path(config.heldout_path);heldout=load_heldout(path)
    ids={int(m.group()) for r in heldout if (m:=re.search(r'\d+$',str(r['id'])))}
    if not set(range(51,56)).issubset(ids):
        raise ValueError('Configured registry must protect DRB 51–55 as well as its broader entries')
    return heldout


def contract(config,endpoint_identity):
    package=Path(__file__).resolve().parents[1]
    sources=list((package/'synthetic_v2').glob('*.py'))
    sources += [package/p for p in ['synthetic/schema.py','synthetic/validate.py',
        'core/state.py','core/types.py','mlp/observation_v2.py','mlp/live_context.py',
        'mlp/embedding_limits.py','llm/openai_compat.py','llm/factory.py']]
    return {'config':config.model_dump(),'prompt_version':PROMPT_VERSION,'event_version':EVENT_VERSION,
        'observation_version':VERSION,'feature_version':FEATURE_VERSION,'schema_version':SCHEMA_VERSION,
        'heldout_sha256':sha(Path(config.heldout_path).read_bytes()),'endpoint_identity':endpoint_identity,
        'source_sha256':{str(p.relative_to(package)):sha(p.read_bytes()) for p in sorted(sources)},
        'dependencies':{n:importlib.metadata.version(n) for n in ['openai','pydantic','tiktoken']}}


def validate(text,heldout,previous):
    events=ProductionTrajectory.model_validate_json(text)
    trajectory,views=expand_production(events)
    rows=compile_trajectory(trajectory,heldout)
    if len(trajectory.states)!=8 or [len(s.candidates) for s in trajectory.states]!=COUNTS or len(rows)!=20:
        raise ContractError('schema')
    pairs=Counter(r['counterfactual_pair_id'] for r in rows if r['counterfactual_pair_id'])
    if len(pairs)<2 or set(pairs.values())!={2}:raise ContractError('pair')
    if len(trajectory.states[-1].source.completed_results)<=4 or len(trajectory.states[-1].source.state.executed_subqueries)<=6:
        raise ContractError('schema')
    if not any(s.source.observation().pending_needs for s in trajectory.states):raise ContractError('schema')
    if any(near_duplicate(trajectory.original_question,q) for q in previous):raise ContractError('diversity')
    if len({r['example_id'] for r in rows})!=len(rows):raise ContractError('schema')
    summary={'structural_validation':'passed','semantic_audit':'required_before_training',
        'examples':len(rows),'question':trajectory.original_question,
        'observation_tokens':[v['diagnostics']['tokens'] for v in views],
        'labels':dict(Counter(r['label'] for r in rows)),
        'categories':dict(Counter(r['decision_category'] for r in rows)),
        'counterfactual_pairs':{pid:[{'example_id':r['example_id'],'state_id':r['state_id'],
            'label':r['label'],'candidate_subquery':r['candidate_subquery'],'supporting_finding':r['supporting_finding']}
            for r in rows if r['counterfactual_pair_id']==pid] for pid in pairs},
        'warnings':['Structural checks cannot certify semantic correctness, source realism or absence of all semantic leakage.'],
        'nonpair_skip_categories':dict(Counter(r['decision_category'] for r in rows if r['label']=='SKIP' and not r['counterfactual_pair_id']))}
    return events,trajectory,views,rows,summary


def accept(directory,qid,result,provenance):
    events,trajectory,views,rows,summary=result
    parent=directory/'accepted';parent.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.pending-',dir=parent))
    artifacts={'events.json':json_bytes(events.model_dump()),'expanded_source.json':json_bytes(trajectory.model_dump()),
        'visibility_views.json':json_bytes(views),'examples.jsonl':jsonl(rows),
        'provenance.json':json_bytes({**provenance,**summary})}
    for name,data in artifacts.items():atomic(stage/name,data)
    atomic(stage/'checkpoint.json',json_bytes({'qid':qid,'sha256':{n:sha(b) for n,b in artifacts.items()}}))
    # Accepted directory rename is the commit point; it is never overwritten.
    if (parent/qid).exists():raise ValueError('Accepted trajectory already exists')
    os.rename(stage,parent/qid);sync_dir(parent)


def recover(directory,config,heldout):
    accepted=[];questions=[];gap=False
    expected={f'q{i+1:03d}' for i in range(config.trajectories)}
    parent=directory/'accepted'
    if parent.exists() and any(p.name not in expected and not p.name.startswith('.pending-') for p in parent.iterdir()):
        raise ValueError('Unexpected accepted checkpoint')
    for i in range(config.trajectories):
        qid=f'q{i+1:03d}';p=parent/qid
        if not p.exists():gap=True;continue
        if gap:raise ValueError('Accepted checkpoints are not a contiguous prefix')
        meta=json.loads((p/'checkpoint.json').read_text())
        required={'events.json','expanded_source.json','visibility_views.json','examples.jsonl','provenance.json'}
        if meta['qid']!=qid or set(meta['sha256'])!=required:
            raise ValueError('Invalid checkpoint manifest')
        for name,h in meta['sha256'].items():
            if sha((p/name).read_bytes())!=h:raise ValueError('Accepted checkpoint hash mismatch')
        result=validate((p/'events.json').read_text(),heldout,questions)
        if (p/'examples.jsonl').read_bytes()!=jsonl(result[3]) or json.loads((p/'expanded_source.json').read_text())!=result[1].model_dump() or json.loads((p/'visibility_views.json').read_text())!=result[2]:
            raise ValueError('Accepted checkpoint differs from local compilation')
        provenance=json.loads((p/'provenance.json').read_text())
        if provenance['assignment']!=config.assignments[i].model_dump() or provenance['qid']!=qid:
            raise ValueError('Checkpoint topic mapping mismatch')
        accepted.append((qid,result,provenance));questions.append(result[1].original_question)
    return accepted,questions


def canonical(directory,config,accepted,contract_data):
    rows=[r for _,result,_ in accepted for r in result[3]]
    if len({r['example_id'] for r in rows})!=len(rows):raise ValueError('Canonical example IDs duplicated')
    content=jsonl(rows);atomic(directory/'examples.jsonl',content)
    manifest={'contract_sha256':sha(json_bytes(contract_data)),'status':'complete' if len(accepted)==config.trajectories else 'partial',
        'target_trajectories':config.trajectories,'accepted_trajectories':len(accepted),'examples':len(rows),
        'examples_sha256':sha(content),'trajectory_order':[qid for qid,_,_ in accepted],
        'labels':dict(Counter(r['label'] for r in rows)),'categories':dict(Counter(r['decision_category'] for r in rows)),
        'semantic_audit':'required_before_training','checkpoint_authority':'accepted/qNNN/checkpoint.json',
        'trajectories':[{'qid':qid,'question':result[1].original_question,'assignment':prov['assignment'],
                         'checkpoint_sha256':sha((directory/'accepted'/qid/'checkpoint.json').read_bytes())}
                         for qid,result,prov in accepted]}
    atomic(directory/'manifest.json',json_bytes(manifest))
    return manifest


async def run(config,directory,*,resume=False,client_factory,endpoint_identity):
    token_count('{}')  # Fail offline before paying if the pinned tokenizer is unavailable.
    heldout=registry(config);expected=contract(config,endpoint_identity)
    if resume:
        if not directory.is_dir():raise ValueError('Resume directory does not exist')
    else:directory.mkdir(parents=True,exist_ok=False)
    client=None
    try:
        with lock(directory):
            if resume:
                actual=json.loads((directory/'contract.json').read_text())
                if actual!=expected:
                    changed=[k for k in expected if expected[k]!=actual.get(k)]
                    raise ValueError('Generation contract changed: '+', '.join(changed))
            else:atomic(directory/'contract.json',json_bytes(expected))
            accepted,questions=recover(directory,config,heldout)
            manifest=canonical(directory,config,accepted,expected)
            for index in range(len(accepted),config.trajectories):
                qid=f'q{index+1:03d}';assignment=config.assignments[index]
                attempts=directory/'attempts'/qid;attempts.mkdir(parents=True,exist_ok=True)
                prior=[int(p.name.removeprefix('attempt-')) for p in attempts.iterdir() if p.is_dir() and re.fullmatch(r'attempt-\d+',p.name)]
                number=max(prior,default=0);codes=[]
                for local_attempt in range(1,config.max_attempts+1):
                    number+=1;folder=attempts/f'attempt-{number:04d}';folder.mkdir()
                    messages=teacher_messages(assignment.topic,assignment.structure,feedback(codes))
                    atomic(folder/'request.json',json_bytes({'prompt_version':PROMPT_VERSION,'assignment':assignment.model_dump(),'messages':messages,
                        'model':config.model,'max_completion_tokens':config.max_completion_tokens,'sdk_retries':0}))
                    atomic(folder/'started.json',json_bytes({'qid':qid,'attempt':number,'invocation_attempt':local_attempt,'started_at':timestamp()}))
                    print(json.dumps({'trajectory':qid,'attempt':number,'invocation_attempt':local_attempt,'status':'generating'}),flush=True)
                    if client is None:client=client_factory()
                    try:
                        response=await client.complete(messages,max_tokens=config.max_completion_tokens,extra={'response_format':{'type':'json_object'}})
                    except Exception as exc:
                        # Never record/send arbitrary provider messages or credential-bearing request URLs.
                        status=getattr(exc,'status_code',None)
                        atomic(folder/'outcome.json',json_bytes({'status':'api_failed','exception_type':type(exc).__name__,
                            'http_status':status if isinstance(status,int) else None,'raw_error_omitted':True}))
                        codes=[]
                        if status in (401,403):raise RuntimeError('Provider authentication/authorization failed; see safe attempt outcome') from None
                        continue
                    atomic(folder/'response.json',json_bytes(response.raw))
                    atomic(folder/'raw_teacher.txt',response.text.encode())
                    atomic(folder/'usage.json',json_bytes(response.usage.model_dump()))
                    failure=generation_failure(response.text,response.raw,config.max_completion_tokens)
                    if failure:
                        codes=['budget' if failure['error_type']=='GenerationBudgetFailure' else 'empty']
                        atomic(folder/'outcome.json',json_bytes({**failure,'status':'rejected','feedback_codes':codes,
                            'automatic_regeneration':local_attempt<config.max_attempts,
                            'retry_policy':'at most 3 attempts per unfinished trajectory per invocation'}))
                        continue
                    try:result=validate(response.text,heldout,questions)
                    except ValueError as exc:
                        codes=classify(exc)
                        atomic(folder/'outcome.json',json_bytes({'status':'rejected','structural_validation':'failed',
                            'exception_type':type(exc).__name__,'feedback_codes':codes,'guidance':feedback(codes)}))
                        continue
                    provenance={'qid':qid,'assignment':assignment.model_dump(),'prompt_version':PROMPT_VERSION,
                        'event_version':EVENT_VERSION,'accepted_attempt':number,'attempt_count':number,
                        'invocation_attempt':local_attempt,'accepted_at':timestamp(),'provider':'openai_compatible',
                        'requested_model':config.model,'reported_model':response.raw.get('model'),
                        'usage':response.usage.model_dump(),'provider_usage':response.raw.get('usage'),
                        'request_sha256':sha((folder/'request.json').read_bytes()),'response_sha256':sha((folder/'response.json').read_bytes()),
                        'attempt_path':str(folder.relative_to(directory))}
                    # Filesystem failures propagate, not caught as teacher validation failures.
                    accept(directory,qid,result,provenance)
                    accepted.append((qid,result,provenance));questions.append(result[1].original_question)
                    atomic(folder/'outcome.json',json_bytes({'status':'accepted',**result[4]}))
                    manifest=canonical(directory,config,accepted,expected)
                    print(json.dumps({'trajectory':qid,'status':'checkpointed','accepted_trajectories':len(accepted),'examples':manifest['examples']}),flush=True)
                    break
                else:
                    raise RuntimeError(f'{qid} exhausted 3 attempts this invocation; accepted checkpoints preserved. Resume this directory.')
            return manifest
    finally:
        if client is not None:await client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/synthetic_v2.yaml'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--allow-api',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=Config.model_validate(yaml.safe_load(args.config.read_text()));heldout=registry(config)
    if args.output and args.resume:parser.error('Use either --output or --resume')
    if args.dry_run or not args.allow_api:
        # Offline inspection creates no production contract bound to an unspecified endpoint.
        token_count('{}')
        for a in config.assignments:teacher_messages(a.topic,a.structure)
        print(json.dumps({'offline':True,'prompt_version':PROMPT_VERSION,'trajectories':config.trajectories,
            'states_per_trajectory':8,'candidates_per_state':COUNTS,'target_examples':20*config.trajectories,
            'registry_questions':len(heldout),'api_calls':0,'artifacts_written':False}))
        return
    load_dotenv(override=False)
    endpoint=os.environ.get('OPENAI_BASE_URL','');parsed=urlparse(endpoint)
    if not os.environ.get('OPENAI_API_KEY') or parsed.scheme!='https' or parsed.path.rstrip('/')!='/openai/v1':
        raise SystemExit('Existing OPENAI_API_KEY and Azure-compatible HTTPS OPENAI_BASE_URL ending /openai/v1/ are required; no call made.')
    from adr.llm.factory import build_llm
    def factory():
        client=build_llm({'provider':'openai','model':config.model,'max_output_tokens':config.max_completion_tokens})
        client._client.max_retries=0
        return client
    directory=args.resume or args.output or Path(config.output_root)/('generation-'+timestamp())
    print('Generation directory: '+str(directory),flush=True)
    manifest=asyncio.run(run(config,directory,resume=bool(args.resume),client_factory=factory,endpoint_identity=sha(endpoint.rstrip('/').encode())))
    print(json.dumps({'status':manifest['status'],'accepted_trajectories':manifest['accepted_trajectories'],'examples':manifest['examples'],'directory':str(directory)}))

if __name__=='__main__':main()
