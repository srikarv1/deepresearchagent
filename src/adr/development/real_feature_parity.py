"""Offline source-parity audit; never replaces canonical features with raw pages.

Scope: the operational v2 source semantics implemented by LiveObservationV2,
not the retired v1 QueryGate's goal-deduplicated branch registry. No live feature
vectors were emitted in these ungated runs: this is source reconstruction, not
an empirical vector-to-vector parity claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from adr.development.real_state_packets import load_run, sha
from adr.synthetic_branch_v2.smoke import atomic
from adr.synthetic.schema import FEATURE_NAMES, BranchState, EvidenceState, derive_features
from adr.mlp.live_context import UnenforcedLiveSnapshot, UnenforcedLiveBudget
from adr.mlp.query_gate import safe_text, key


def parity_catalog():
    """Strict historical parity, distinct from computable diagnostic replay."""
    sources = {
        'n_evidence': ('Canonical global evidence registry size', 'Published legacy round new_item_ids joined to immutable normal-trajectory evidence'),
        'n_retained': ('Canonical retained evidence registry', 'Round retained_ids; verified none policy and no pruning'),
        'n_pruned': ('Canonical non-retained evidence count', 'No pruned IDs in any saved round; zero'),
        'n_open_branches': ('Open/active subtasks in the actual complete feature source', 'Enhanced planned/active operational tasks; retired v1 instead used goal-deduplicated frontier/subquestions'),
        'n_branches': ('All subtasks in the actual complete feature source', 'Enhanced stable operational task registry; not the retired v1 registry'),
        'mean_score': ('Retained Evidence.score (fixed zero by derive_features)', 'Defined constant, not a missing similarity score'),
        'mean_chars': ('Mean len(Evidence.body(2000)), integer floor, after live safe_text and stripping', 'Canonical immutable content exists; historical runtime credential substitutions not recorded'),
        'query_term_coverage': ('Mean original-question word overlap with retained title + body(400), rounded 4 decimals', 'Original question and canonical content exist; title empty; historical safe_text environment not recorded'),
        'unique_urls': ('Distinct retained evidence URL identities', 'Canonical source_url; live SHA256 URL identities preserve equality'),
        'step_frac': ('min(1, used_steps/max_steps), rounded 4 decimals', 'Prior candidate proposals + committed rounds; max_steps 32'),
        'token_frac': ('min(1, used_tokens/max_tokens), rounded 4 decimals', 'Exact snapshot input+output tracked tokens; max_tokens 400000'),
        'remaining_steps': ('max(0, max_steps-used_steps)', 'Exact pre-batch proposal/publication prefix and nominal maximum'),
        'remaining_searches': ('max(0, max_searches-used_searches)', 'All prior search_dispatch events including planning; max 64'),
        'remaining_reads': ('max(0, max_reads-used_reads)', 'Prior read_dispatch attempt counts, not successful evidence count; max 128'),
        'remaining_tokens': ('max(0, max_tokens-used_tokens)', 'Exact pre-batch tracked input+output tokens and max 400000'),
    }
    not_certified = {'n_open_branches','n_branches','mean_chars','query_term_coverage'}
    return [{
        'feature': f, 'canonical_live_source': sources[f][0], 'available_information': sources[f][1],
        'exact_historical_live_parity_certified': f not in not_certified,
        'deterministic_v2_source_replay_possible': True,
        'safe_adapter': ('Round join/event-prefix/constants without invention' if f not in not_certified else
            'Operational v2 mapping is available, but must be explicitly distinguished from retired v1 source semantics'
            if f in ('n_open_branches','n_branches') else
            'Current source transform can be replayed; cannot infer unknown historical credential-redaction substitutions'),
        'measured_live_vector_discrepancy': None,
        'discrepancy_reason': 'No gate feature vector recorded in ungated development run',
    } for f in FEATURE_NAMES]


def canonical_pages(trajectory, published_round_ids):
    """Join on published *legacy* rounds, never on a final aggregate or timestamp.

    The fork inserts canonical chunk-aggregated pages only at round aggregation;
    first insertion wins. Verify no later compression/pruning invalidates use of
    the final immutable body registry. Future items are excluded by new_item_ids.
    """
    evidence = trajectory['evidence']
    rounds = {r['round_id']:r for r in trajectory['rounds']}
    if any(e['compressed'] or e['original_word_count'] is not None for e in evidence.values()):
        raise ValueError('Historical evidence body unavailable after compression')
    if any(r['decision']['policy']!='none' or r['decision']['pruned_item_ids'] for r in rounds.values()):
        raise ValueError('Audit only supports verified no-pruning histories')
    ids = []
    for rid in published_round_ids:
        r = rounds[rid]
        for iid in r['new_item_ids']:
            if iid in ids or evidence[iid]['retrieval_round'] != rid:
                raise ValueError('Evidence first-insertion mismatch')
            ids.append(iid)
        if set(ids) != set(r['retained_ids']):
            raise ValueError('Cumulative retained registry mismatch')
    return [evidence[iid] for iid in ids]


def replay(run):
    path = next((run['run']/'gpt_researcher').glob('*.json'))
    trajectory = json.loads(path.read_text())
    records, published, rows = {}, [], []
    legacy_branches = {}
    normal_rounds = {r['round_id']: r for r in trajectory['rounds']}
    def legacy_remember(goal, status):
        # QueryGate.remember_branch merges exact safe-text goals. IDs do not
        # affect these two counts. Keep this comparison separate from v2 replay.
        legacy_branches[safe_text(goal)] = status
    counts = dict(searches=0, reads=0, proposals=0)
    for e in run['events']:
        if e['event']=='record': records[e['record_id']]=e
        elif e['event']=='search_dispatch': counts['searches']+=1
        elif e['event']=='read_dispatch': counts['reads']+=e['count']
        elif e['event']=='candidate_proposed': counts['proposals']+=1
        elif e['event']=='round_published': published.append(e['legacy_round_id'])
        elif e['event']=='batch_snapshot':
            accounting=e['accounting']; usage=accounting['budget_usage']; maxima=accounting['nominal_budget']
            expected=dict(used_steps=counts['proposals']+len(published),used_searches=counts['searches'],
                          used_reads=counts['reads'],used_tokens=accounting['tracked_tokens']['input_tokens']+accounting['tracked_tokens']['output_tokens'])
            if usage!=expected:
                raise ValueError('Trace accounting disagrees with pre-decision event prefix')
            pages=canonical_pages(trajectory,published)
            for goal in trajectory['subquestions']:
                legacy_branches.setdefault(safe_text(goal),'open')
            for rid in published:
                for node in normal_rounds[rid]['frontier']:
                    legacy_remember(node['subquery'],{'open':'open','completed':'done','pruned':'dropped'}[node['status']])
            legacy_remember(next(b['query'] for b in e['branches'] if b['id']==e['branch_id']),'active')
            state=UnenforcedLiveSnapshot(
                evidence=[EvidenceState(id=p['item_id'],url='url-sha256:'+key(p['source_url']),title='',
                    text=safe_text(p['content'])[:2000],retained=True) for p in pages],
                branches=[BranchState(id=b['id'],goal=safe_text(b['goal']),
                    status={'planned':'open','active':'active','processed':'done','done':'done','failed':'dropped'}[b['status']]) for b in e['branches']],
                budget=UnenforcedLiveBudget(**{k:maxima[k] for k in ('max_steps','max_searches','max_reads','max_tokens')},**usage))
            features=derive_features(run['question'],state)
            # Count actual raw pages separately to demonstrate why substituting
            # trace record/evidence counts would be wrong. Never used as features.
            raw_count=sum(records[rid]['kind']=='raw_page' for rid in e['evidence_ids'])
            rows.append(dict(batch=e['batch_id'],sequence=e['sequence'],published_legacy_round_ids=list(published),
                raw_page_records_available=raw_count,canonical_evidence=len(pages),
                numerical_features=features.model_dump(),usage=usage,
                independently_checked_counters=expected,operational_branch_counts=dict(total=len(e['branches']),
                    open_active=sum(b['status'] in ('planned','active') for b in e['branches'])),
                retired_v1_branch_counts=dict(total=len(legacy_branches),
                    open_active=sum(s in ('open','active') for s in legacy_branches.values()))))
    return dict(normal_trajectory=str(path),normal_sha256=sha(path),trace_sha256=sha(run['path']),
        canonical_final_evidence=len(trajectory['evidence']),snapshots=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Audit output already exists; preserve prior version')
    config=yaml.safe_load(args.config.read_text())
    result={name:replay(load_run(path)) for name,path in config['runs'].items()}
    for run in result.values():
        values=np.array([r['numerical_features']['values'] for r in run['snapshots']])
        run['ranges']={k:dict(min=float(values[:,i].min()),max=float(values[:,i].max())) for i,k in enumerate(FEATURE_NAMES)}
        run['raw_count_disagrees_at_snapshots']=sum(r['raw_page_records_available']!=r['canonical_evidence'] for r in run['snapshots'])
    atomic(args.output,dict(scope='existing operational v2 source mapping, diagnostic reconstruction only',
        feature_names=list(FEATURE_NAMES),feature_parity=parity_catalog(),runs=result,annotation_inputs_include_features=False,
        recorded_live_vectors=False,training_accepted=False,
        conclusion='Future real training trajectories need explicit pre-decision feature/source-version logging; no further development collection requested here',
        caveats=['No gate vector was logged: source replay is not direct empirical inference parity.',
                 'safe_text includes runtime credential redaction; historical environment substitutions cannot be certified from these artifacts.',
                 'Retired v1 QueryGate branch registry is goal-deduplicated and is NOT substituted for operational v2 task counts.',
                 'Frozen branch constructor accepts an external complete state; it does not yet enable a production v2 live gate.']))
    print(json.dumps({k:dict(snapshots=len(v['snapshots']),canonical_final_evidence=v['canonical_final_evidence'],raw_count_disagrees_at_snapshots=v['raw_count_disagrees_at_snapshots'],ranges=v['ranges']) for k,v in result.items()}))

if __name__=='__main__': main()
