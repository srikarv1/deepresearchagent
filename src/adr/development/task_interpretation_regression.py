"""Four unchanged development packets; one prospective instruction insertion only."""
import argparse
import asyncio
from pathlib import Path
import json
import hashlib
from adr.synthetic_branch_v2 import pilot_annotation as annotation, smoke

VERSION = 'observation-only-task-interpretation-regression-v1'
SOURCE = Path('runs/real-state-supervision/pilot-24-v1')
OUTPUT = Path('runs/real-state-supervision/task-interpretation-regression-v1')
IDS = ('dev1-01', 'dev2-16', 'dev2-15', 'dev3-23')
CLARIFICATION = ('You are classifying a proposed search that another research agent could execute. '
    'Do not execute it yourself. Your lack of browsing tools does not make the candidate invalid. '
    'EXECUTE recommends future retrieval; SKIP means its information need is sufficiently answered by the supplied observation.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request(packet):
    messages = annotation.request(packet)
    messages[0]['content'] = CLARIFICATION + '\n\n' + messages[0]['content']
    return messages


def load_packets(source):
    all_packets = smoke.read(source/'packets.json')
    previews = smoke.read(source/'request_previews.json')
    packets = {}
    for pid in IDS:
        packet = all_packets[pid]
        if set(packet) != {'original_question','context_before','candidate_subquery'}:
            raise ValueError('Unexpected source packet fields')
        original = smoke.request(annotation.request(packet),16000)
        if original != previews[pid]:
            raise ValueError('Original supervision contract differs')
        packets[pid] = packet
    return packets


def contract(source, packets):
    from adr.llm import openai_compat
    from adr.synthetic_v2 import response_status, action_space
    from adr.synthetic_branch_v2 import supervision
    return dict(version=VERSION, source=str(source), packet_ids=list(IDS),
        source_packet_sha256=sha(source/'packets.json'),
        source_preview_sha256=sha(source/'request_previews.json'),
        code_sha256={m.__name__:sha(m.__file__) for m in
                     (annotation,smoke,openai_compat,response_status,action_space,supervision)},
        runner_sha256=sha(__file__), model='gpt-5-mini', max_completion_tokens=16000,
        sdk_retries=0, maximum_calls=4, maximum_attempts_per_packet=1,
        observation_contract='mlp-observation-v2-branches-cited-top3-v1',
        supervision_contract=annotation.VERSION, instruction_insertion=CLARIFICATION,
        development_only=True, training_rows=0)


def audit(out, packets, results):
    rows=[]
    for i,(pid,p) in enumerate(packets.items(),1):
        r=results.get(pid,{})
        rows.append(dict(packet_id=pid, candidate=p['candidate_subquery'],
            stage=f'annotation-{i:03d}', structural_status=r.get('status','not_called'),
            annotation=r.get('annotation'),human_review='pending',training_accepted=False))
    smoke.atomic(out/'human_audit.json',dict(development_only=True,training_rows=0,decisions=rows))


async def run(source=SOURCE, out=OUTPUT, *, resume=False, allow_api=False, client_factory=None):
    source,out=Path(source),Path(out)
    if out.resolve()==source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError('Output must be separate from original artifacts')
    packets=load_packets(source)
    expected=contract(source,packets)
    previews={pid:smoke.request(request(p),16000) for pid,p in packets.items()}
    if resume:
        if not out.is_dir(): raise ValueError('No prepared regression directory')
    else:
        out.mkdir(parents=True,exist_ok=False)
    with smoke.lock(out):
        if resume:
            if (smoke.read(out/'contract.json')!=expected or smoke.read(out/'packets.json')!=packets
                    or smoke.read(out/'request_previews.json')!=previews):
                raise ValueError('Prepared regression contract differs')
        else:
            smoke.atomic(out/'contract.json',expected)
            smoke.atomic(out/'packets.json',packets)
            smoke.atomic(out/'request_previews.json',previews)
            audit(out,packets,{})
        client=None
        def get_client():
            nonlocal client
            if client is None:
                client=client_factory() if client_factory else smoke.azure_client(out)
            return client
        results={}
        if allow_api:
            try:
                for i,(pid,p) in enumerate(packets.items(),1):
                    try:
                        value=await smoke.stage(out,f'annotation-{i:03d}',request(p),16000,get_client,
                            lambda text,p=p:annotation.validate(text,p))
                        results[pid]=dict(status='validated',annotation=value)
                    except smoke.StageFailure:
                        results[pid]=dict(status='failed_or_uncertain_no_retry')
                    audit(out,packets,results)
            finally:
                if client is not None: await client.close()
        failures=sum(r['status']!='validated' for r in results.values())
        manifest=dict(status=('awaiting_human_audit_with_failures' if failures else 'awaiting_human_audit')
                      if allow_api else 'prepared_offline',packet_ids=list(IDS),
                      model='gpt-5-mini',sdk_retries=0,maximum_calls=4,attempts_per_packet=1,
                      usage=smoke.usage_summary(out),training_rows=0,development_only=True)
        # A read-only/offline resume must not erase completed status/audit.
        if allow_api or not resume: smoke.atomic(out/'manifest.json',manifest)
        return smoke.read(out/'manifest.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(run(resume=args.resume,allow_api=args.allow_api))))

if __name__=='__main__': main()
