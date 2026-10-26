"""Development-only real-state packet preparation and isolated one-attempt annotation.

No world generation, feature substitution, embeddings, or training path exists here.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from adr.development.replay_observation_v2 import reconstruct
from adr.mlp import branch_observation_v2 as observation
from adr.synthetic_branch_v2 import pilot_annotation as annotation, smoke

VERSION = 'real-state-supervision-feasibility-24-v1'
FIELDS = ('original_question', 'context_before', 'candidate_subquery')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(run):
    run = Path(run)
    paths = list((run / 'development_trace').glob('*.jsonl'))
    if len(paths) != 1:
        raise ValueError('Expected one enhanced trace')
    path = paths[0]
    # Independently enforce publication, acquisition, history and hierarchy checks.
    checked = dict(reconstruct(path))
    records, queries, snapshots = {}, {}, {}
    question = None
    with path.open(encoding='utf-8') as stream:
        events = [json.loads(line) for line in stream]
    for e in events:
        kind = e['event']
        if kind == 'session':
            if e['evaluation'] or e['gate_enabled']:
                raise ValueError('Expected ungated development trace')
            question = e['original_question']
        elif kind == 'record':
            records[e['record_id']] = e
        elif kind == 'batch_snapshot':
            cards = []
            for bid, rid in zip(e['completed_branch_ids'], e['processed_result_ids'], strict=True):
                r = records[rid]['value']
                cards.append(observation.PublishedBranch(task_id=bid, goal=r['researchGoal'],
                    findings=r['learnings'], followups=r['followUpQuestions'], cited_insights=r['citations']))
            source = observation.ObservationSource(current_task_id=e['branch_id'],
                current_need=next(b['goal'] for b in e['branches'] if b['id'] == e['branch_id']),
                tasks=tuple(observation.Task(id=b['id'], parent_id=b['parent_id'], tree_depth=b['tree_depth']) for b in e['branches']),
                published_branches=tuple(cards), executed_queries=tuple(queries[c]['query'] for c in e['executed_candidate_ids']))
            snapshots[e['batch_id']] = (source, e)
        elif kind == 'candidate_proposed':
            queries[e['id']] = e
            source, snap = snapshots[e['batch_id']]
            if observation.construct(source, e['query']) != checked[e['id']]:
                raise ValueError('Frozen replay differs')
    if len(checked) != 112 or len(snapshots) != 28:
        raise ValueError('Unexpected development trace shape')
    return dict(path=path, run=run, question=question, queries=queries, snapshots=snapshots, events=events)


def prepare(config):
    runs = {k: load_run(v) for k, v in config['runs'].items()}
    packets, rows = {}, []
    selections = config['selections']
    if Counter(s['run'] for s in selections) != {r: 8 for r in runs} or len(runs) != 3:
        raise ValueError('Require exactly eight packets from each of three runs')
    for name in runs:
        if Counter(s['kind'] for s in selections if s['run'] == name) != {'original_early':2,'replay':2,'original_gap':2,'variant':2}:
            raise ValueError('Unexpected packet composition')
    for number, s in enumerate(selections, 1):
        run = runs[s['run']]
        source, snapshot = run['snapshots'][s['batch']]
        original = run['queries'].get(s.get('source_candidate'))
        if s['kind'] == 'variant':
            if original is not None:
                raise ValueError('Variant cannot pretend to be original')
            query = s['query']
        else:
            if not original:
                raise ValueError('Missing original query')
            query = original['query']
            if s['kind'].startswith('original') and original['batch_id'] != s['batch']:
                raise ValueError('Original at wrong state')
            if s['kind'] == 'replay':
                if run['snapshots'][original['batch_id']][1]['sequence'] >= snapshot['sequence']:
                    raise ValueError('Replay is not later')
                if original['branch_id'] not in snapshot['completed_branch_ids']:
                    raise ValueError('Replayed branch is not published')
                partners = [a for a in selections if a['run']==s['run'] and a['kind']=='original_early' and a['source_candidate']==s['source_candidate']]
                if len(partners) != 1:
                    raise ValueError('Missing exact-query earlier partner')
        if annotation.action_invalid(query):
            raise ValueError('Invalid retrieval candidate')
        obs = observation.construct(source, query)
        if obs.overflow or obs != observation.construct(source, query):
            raise ValueError('Overflow/non-deterministic packet')
        view = json.loads(obs.text)
        anchor = s.get('visible_anchor')
        if anchor and not any(anchor in b['findings'] or anchor in b['followups'] or anchor in b['cited_insights'] for b in view['branches']):
            raise ValueError('Selection anchor not visible after candidate-dependent selection')
        pid = f'{s["run"]}-{number:02d}'
        packet = dict(original_question=run['question'], context_before=obs.text, candidate_subquery=query)
        if packet in packets.values():
            raise ValueError('Duplicate exact packet')
        packets[pid] = packet
        ordinal = list(run['snapshots']).index(s['batch']) + 1
        rows.append(dict(packet_id=pid, candidate=query, run=s['run'], kind=s['kind'],
            batch=s['batch'], snapshot_sequence=snapshot['sequence'],
            phase='early' if ordinal<=9 else 'middle' if ordinal<=18 else 'late',
            source_candidate=s.get('source_candidate'), rationale_for_selection=s['reason'],
            visible_selection_anchor=anchor, drift_caution=s.get('drift_caution'),
            source_execution='executed_original' if s['kind'].startswith('original') else 'not_executed_at_this_state',
            selected_task_ids=list(obs.selected_task_ids), selected_branch_goals=[b['goal'] for b in view['branches']],
            completed_available=len(source.published_branches), context_tokens=obs.tokens,
            annotation_label=None, category=None, witnesses=[], rationale=None, structural_validation='not_called',
            stage=f'annotation-{number:03d}', training_accepted=False, human_review='pending'))
    return packets, rows, runs


def contract(config, runs):
    from adr.mlp import embedding_limits
    from adr.synthetic_v2 import action_space, response_status
    from adr.synthetic_branch_v2 import supervision
    from adr.llm import openai_compat
    import sklearn, tiktoken
    modules = [observation, embedding_limits, annotation, smoke, action_space, response_status, supervision, openai_compat]
    return dict(version=VERSION, config=config, observation_version=observation.VERSION,
        annotation_version=annotation.VERSION, runner_sha256=sha(__file__),
        code_sha256={m.__name__:sha(m.__file__) for m in modules},
        source_sha256={str(r['path']):sha(r['path']) for r in runs.values()},
        dependencies=dict(sklearn=sklearn.__version__, tiktoken=tiktoken.__version__),
        development_only=True, training_rows=0, validation_rows=0, maximum_provider_calls=24,
        attempts_per_candidate=1, sdk_retries=0, semantic_reviewer=None,
        q051='previously development-exposed', q052_to_q055='untouched')


def write_audit(out, rows, packets):
    smoke.atomic(out/'human_audit.json', dict(development_only=True, training_rows=0, decisions=rows))
    lines = ['# Real-state supervision feasibility — human audit', '',
        'DEVELOPMENT ONLY. No labels assigned during preparation; no examples accepted for training.',
        'Phases: chronological batches 1–9 early, 10–18 middle, 19–28 late. Replay links are contrasts, not claimed label reversals.', '']
    for r in rows:
        lines += [f'## {r["packet_id"]} — {r["kind"]}, {r["phase"]}', '',
            f'Candidate: {r["candidate"]}', '', f'Selection: {r["rationale_for_selection"]}', '',
            f'State: {r["batch"]}; tokens: {r["context_tokens"]}; completed branches: {r["completed_available"]}.',
            f'Drift caution: {r["drift_caution"] or "none identified in current task"}.',
            f'Annotation: {r["annotation_label"]}; category: {r["category"]}; validation: {r["structural_validation"]}.', '',
            'Selected goals:', *['- '+g for g in r['selected_branch_goals']], '',
            'Visible selection anchor (audit only): '+str(r['visible_selection_anchor']), '',
            'Witnesses: '+json.dumps(r['witnesses'],ensure_ascii=False),
            'Rationale: '+str(r['rationale']), '', '<details><summary>Exact frozen observation</summary>', '',
            '```json', packets[r['packet_id']]['context_before'], '```', '</details>', '']
    (out/'human_audit.md').write_text('\n'.join(lines),encoding='utf-8')


async def annotate(out, packets, rows, get_client):
    failed = 0
    for row in rows:
        p = packets[row['packet_id']]
        try:
            result = await smoke.stage(out,row['stage'],annotation.request(p),16000,get_client,
                                       lambda text,p=p:annotation.validate(text,p))
            row.update(annotation_label=result['label'],category=result['category'],witnesses=result['witnesses'],
                       rationale=result['rationale'],proposal=result,structural_validation='validated')
        except smoke.StageFailure:
            failed += 1
            row['structural_validation']='failed_or_uncertain_no_retry'
        write_audit(out,rows,packets)
    return failed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    packets,rows,runs=prepare(config)
    expected=contract(config,runs)
    out=Path(config['output'])
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir():
        raise ValueError('No prepared directory')
    with smoke.lock(out):
        if args.resume:
            if smoke.read(out/'contract.json')!=expected or smoke.read(out/'packets.json')!=packets:
                raise ValueError('Prepared contract/source/packet mismatch')
            if smoke.read(out/'request_previews.json')!={r['packet_id']:smoke.request(annotation.request(packets[r['packet_id']]),16000) for r in rows}:
                raise ValueError('Prepared requests differ')
        else:
            smoke.atomic(out/'contract.json',expected)
            smoke.atomic(out/'packets.json',packets)
            smoke.atomic(out/'request_previews.json',{r['packet_id']:smoke.request(annotation.request(packets[r['packet_id']]),16000) for r in rows})
            smoke.atomic(out/'selection_provenance.json',rows)
            write_audit(out,rows,packets)
        status='prepared_offline'
        if args.allow_api:
            client=None
            def get_client():
                nonlocal client
                if client is None: client=smoke.azure_client(out)
                return client
            async def execute():
                try:
                    return await annotate(out,packets,rows,get_client)
                finally:
                    if client is not None:
                        await client.close()
            failures=asyncio.run(execute())
            status='awaiting_human_audit' if not failures else 'awaiting_human_audit_with_failures'
        sizes=[r['context_tokens'] for r in rows]
        manifest=dict(status=status,development_only=True,training_rows=0,validation_rows=0,
            packets=24,composition=dict(Counter(r['kind'] for r in rows)),phases=dict(Counter(r['phase'] for r in rows)),
            tokens=dict(min=min(sizes),median=float(np.median(sizes)),p90=float(np.percentile(sizes,90)),max=max(sizes)),
            usage=smoke.usage_summary(out),maximum_attempts_per_candidate=1,sdk_retries=0)
        smoke.atomic(out/'manifest.json',manifest)
        print(json.dumps(manifest))

if __name__=='__main__': main()
