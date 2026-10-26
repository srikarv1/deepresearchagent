"""Prospective operational-v2 numerical source. No model, retrieval or embedding calls."""
from copy import deepcopy
import hashlib
import json
import math
from adr.synthetic.schema import FEATURE_NAMES, BranchState, EvidenceState, derive_features
from adr.mlp.live_context import UnenforcedLiveBudget, UnenforcedLiveSnapshot
from adr.mlp.query_gate import safe_text, key

VERSION = 'mlp-numerical-source-v2-operational-canonical-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def capture(question, branches, canonical_evidence, maxima, usage):
    """Full committed registry, not selected branches or enhanced raw pages.

    Match existing operational v2 source text transform including 2000-character
    excerpt before schema stripping. The resulting raw vector is authoritative.
    """
    pages=[]
    for iid,e in sorted(canonical_evidence.items(),key=lambda x:(x[1].retrieval_round,x[0])):
        if e.compressed:
            raise ValueError('Compressed canonical evidence is unsupported by this source contract')
        pages.append(EvidenceState(id=safe_text(iid),url='url-sha256:'+key(e.source_url),
            title='',text=safe_text(e.content)[:2000],retained=e.was_retained))
    statuses={'planned':'open','active':'active','processed':'done','done':'done','failed':'dropped'}
    state=UnenforcedLiveSnapshot(evidence=pages,
        branches=[BranchState(id=b['id'],goal=safe_text(b['goal']),status=statuses[b['status']]) for b in branches],
        budget=UnenforcedLiveBudget(**{f'max_{k}':maxima[f'max_{k}'] for k in ('steps','searches','reads','tokens')},**usage))
    vector=derive_features(question,state)
    result=dict(feature_contract=VERSION,feature_names=list(vector.names),raw_vector=list(vector.values),
        features=dict(zip(vector.names,vector.values)),nominal_maxima={k:v for k,v in state.budget.model_dump().items() if k.startswith('max_')},
        actual_usage=dict(usage),source_state_digest=digest(state.model_dump()),
        canonical_evidence_ids=[e.id for e in state.evidence])
    result['numerical_snapshot_digest']=digest(result)
    return deepcopy(result)


def bind(snapshot, *, batch_id, candidate_id, question, candidate, context, observation_version):
    result=deepcopy(snapshot)
    result.update(batch_id=batch_id,candidate_id=candidate_id,observation_version=observation_version,
        observation_digest=digest(dict(original_question=question,context_before=context,candidate_subquery=candidate)))
    result['decision_snapshot_digest']=digest(result)
    return result


def persisted_vector(record, *, question, candidate, context):
    """Preparation/inference reader: validate stored values, NEVER reconstruct features."""
    keys = ('feature_contract','feature_names','raw_vector','features','nominal_maxima','actual_usage','source_state_digest','canonical_evidence_ids','numerical_snapshot_digest','batch_id','candidate_id','observation_version','observation_digest','decision_snapshot_digest')
    r={k:deepcopy(record[k]) for k in keys}; expected=r.pop('decision_snapshot_digest')
    if digest(r)!=expected: raise ValueError('Decision snapshot integrity mismatch')
    from adr.mlp.branch_observation_v2 import VERSION as observation_version
    if r['observation_version'] != observation_version:
        raise ValueError('Observation contract mismatch')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in r['raw_vector']):
        raise ValueError('Non-finite/non-numeric feature value')
    if r['feature_contract']!=VERSION or r['feature_names']!=list(FEATURE_NAMES):
        raise ValueError('Numerical source contract/order mismatch')
    if len(r['raw_vector'])!=15 or r['features']!=dict(zip(FEATURE_NAMES,r['raw_vector'])):
        raise ValueError('Feature mapping/vector mismatch')
    if r['observation_digest']!=digest(dict(original_question=question,context_before=context,candidate_subquery=candidate)):
        raise ValueError('Semantic observation mismatch')
    for k in ('batch_id','candidate_id','observation_version','observation_digest'):
        r.pop(k)
    numerical_digest=r.pop('numerical_snapshot_digest')
    if digest(r)!=numerical_digest: raise ValueError('Numerical snapshot integrity mismatch')
    return list(r['raw_vector'])
