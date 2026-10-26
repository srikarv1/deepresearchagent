"""Read-only replay of enhanced development traces through the shared constructor.

No benchmark loader, provider, live researcher, or artifact writes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from adr.mlp.branch_observation_v2 import ObservationSource, PublishedBranch, Task, construct


def reconstruct(path):
    records, tasks, published, processed, queries, finished, snapshots = {}, {}, {}, {}, {}, {}, {}
    acquisitions = {}
    outputs = []
    for event in map(json.loads, Path(path).open()):
        kind, seq = event['event'], event['sequence']
        if kind == 'record':
            records[event['record_id']] = event
        elif kind == 'record_acquired':
            acquisitions.setdefault(event['record_id'], []).append(event)
        elif kind == 'branch_created':
            tasks[event['branch']['id']] = event['branch']
        elif kind == 'branch_processed' and event['result_id']:
            processed[event['branch_id']] = event['result_id']
        elif kind == 'round_published':
            for bid in event['available_completed']:
                if bid not in processed:
                    raise ValueError('Publication before processing')
                published.setdefault(bid, processed[bid])
        elif kind == 'candidate_finished':
            if event['status'] == 'completed':
                finished[event['candidate_id']] = seq
        elif kind == 'batch_snapshot':
            bids = event['completed_branch_ids']
            if bids != list(published) or event['processed_result_ids'] != list(published.values()):
                raise ValueError('Snapshot publication boundary mismatch')
            available = set(bids) | {event['branch_id']}
            for rid in event['evidence_ids']:
                if not any(a['sequence'] < seq and a['branch_id'] in available for a in acquisitions[rid]):
                    raise ValueError('Unpublished evidence in snapshot')
            for cid in event['executed_candidate_ids']:
                if cid not in finished or finished[cid] >= seq or queries[cid]['branch_id'] not in available:
                    raise ValueError('Non-prior/unpublished history')
            cards = []
            for bid, rid in zip(bids, event['processed_result_ids'], strict=True):
                r = records[rid]['value']
                cards.append(PublishedBranch(task_id=bid, goal=r['researchGoal'],
                    findings=r['learnings'], followups=r['followUpQuestions'], cited_insights=r['citations']))
            source = ObservationSource(current_task_id=event['branch_id'],
                current_need=tasks[event['branch_id']]['goal'],
                tasks=tuple(Task(id=b['id'], parent_id=b['parent_id'], tree_depth=b['tree_depth']) for b in event['branches']),
                published_branches=tuple(cards),
                executed_queries=tuple(queries[c]['query'] for c in event['executed_candidate_ids']))
            snapshots[event['batch_id']] = (source, event)
        elif kind == 'candidate_proposed':
            queries[event['id']] = event
            source, snapshot = snapshots[event['batch_id']]
            if snapshot['sequence'] >= seq:
                raise ValueError('Snapshot does not precede candidate')
            observation = construct(source, event['query'])
            if observation != construct(source, event['query']):
                raise ValueError('Non-deterministic reconstruction')
            view = json.loads(observation.text)
            if view['recent_queries'] != list(source.executed_queries[-6:]):
                raise ValueError('History mismatch')
            for bid, packet in zip(observation.selected_task_ids, view['branches'], strict=True):
                result = records[published[bid]]['value']
                if packet['cited_insights'] != result['citations']:
                    raise ValueError('Cited insight ownership mismatch')
                # Independent ancestry walk checks serializer hierarchy tags.
                parent = tasks[source.current_task_id]['parent_id']
                ancestors = set()
                cursor = parent
                while cursor:
                    ancestors.add(cursor)
                    cursor = tasks[cursor]['parent_id']
                expected = 'parent' if bid == parent else 'ancestor' if bid in ancestors else 'other'
                if packet['relation'] != expected or packet['tree_depth'] != tasks[bid]['tree_depth']:
                    raise ValueError('Hierarchy mismatch')
                if set(packet) != {'goal','findings','followups','cited_insights','relation','tree_depth'}:
                    raise ValueError('Unexpected embedded fields')
            outputs.append((event['id'], observation))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for run in args.runs:
        files = list((run/'development_trace').glob('*.jsonl'))
        if len(files) != 1:
            raise ValueError('Expected exactly one development trace')
        outputs = reconstruct(files[0])
        sizes = [o.tokens for _, o in outputs]
        print(json.dumps({'run': str(run), 'candidates': len(outputs),
            'median': float(np.median(sizes)), 'p90': float(np.percentile(sizes,90)),
            'max': max(sizes), 'overflow': sum(o.overflow for _,o in outputs)}))

if __name__ == '__main__':
    main()
