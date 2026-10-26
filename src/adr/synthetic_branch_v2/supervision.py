"""Observation-only annotation validation. Exact grounding is not entailment.

Semantic reviewers must see the packet and proposed annotation, never source
world, unselected results, raw pages, or future events. URL identity is not proof.
"""
import json
from typing import Literal
from pydantic import Field
from adr.synthetic_branch_v2.source import Strict, digest

VERSION = 'synthetic-branch-supervision-v1'
SKIP = {'direct_answer','visible_duplicate','negligible_marginal_value'}
EXECUTE = {'unresolved_need','visible_uncertainty','missing_qualifier','insufficient_answer','no_relevant_visible_evidence'}

class Witness(Strict):
    kind: Literal['finding','insight','history']
    branch_index: int | None = Field(default=None,ge=0)
    text: str = Field(min_length=1)

class Annotation(Strict):
    candidate_id: str
    packet_digest: str
    label: Literal['EXECUTE','SKIP','AMBIGUOUS','REJECT']
    category: str
    rationale: str = Field(min_length=1)
    witnesses: list[Witness] = Field(default_factory=list)
    pair_id: str | None = None
    decisive_witness: Witness | None = None

class Review(Strict):
    """Human or separately audited observation-only adjudication; not self-attestation."""
    annotation_digest: str
    reviewer: str = Field(min_length=1)
    verdict: Literal['SUPPORTED','AMBIGUOUS','REJECT']
    explanation: str = Field(min_length=1)


def visible(witness, packet):
    view=json.loads(packet['context_before'])
    if witness.kind=='history':
        return witness.branch_index is None and witness.text in view['recent_queries']
    if witness.branch_index is None or witness.branch_index >= len(view['branches']):
        return False
    branch=view['branches'][witness.branch_index]
    field='findings' if witness.kind=='finding' else 'cited_insights'
    return witness.text in branch[field]  # insight KEYS, never bare URL values


def semantic_texts(packet):
    view=json.loads(packet['context_before'])
    return [t for b in view['branches'] for t in (*b['findings'],*b['cited_insights'])]


def validate_annotation(annotation, packet, audit):
    if annotation.packet_digest != digest(packet) or annotation.packet_digest != audit['packet_digest']:
        raise ValueError('Annotation is not bound to this exact observation')
    if audit['overflow']:
        return 'overflow_execute_bypass'  # exclude BOTH labels from trainable data
    if annotation.label in ('AMBIGUOUS','REJECT'):
        return 'ambiguous_or_rejected'
    if annotation.category not in (SKIP if annotation.label=='SKIP' else EXECUTE):
        raise ValueError('Label/category mismatch')
    if any(not visible(w,packet) for w in annotation.witnesses):
        raise ValueError('Witness absent from selected visible observation')
    if annotation.label=='SKIP' and not annotation.witnesses:
        raise ValueError('SKIP requires explicit visible grounding')
    if annotation.category in ('direct_answer','negligible_marginal_value','visible_uncertainty','insufficient_answer'):
        if not any(w.kind in ('finding','insight') for w in annotation.witnesses):
            raise ValueError('Category needs visible semantic evidence')
    if annotation.decisive_witness is not None:
        if annotation.label!='SKIP' or annotation.decisive_witness.kind=='history' or not visible(annotation.decisive_witness,packet):
            raise ValueError('Counterfactual decisive finding must be visible at later SKIP')
    return 'needs_semantic_review'


def finalize(packets, audit, execution, annotations, reviews):
    """All-or-error pair validation; only reviewed supported rows become canonical.

    Nontrainable decisions remain in rejected audit, not coerced labels. Historical
    execution is checked after annotation; it is never part of the reviewer packet.
    """
    accepted, rejected, groups = [], {}, {}
    seen=set()
    for a in annotations:
        if a.candidate_id in seen or a.candidate_id not in packets:
            raise ValueError('Duplicate/unknown annotation identity')
        seen.add(a.candidate_id)
        packet=packets[a.candidate_id]; meta=audit[a.candidate_id]
        status=validate_annotation(a,packet,meta)
        review=reviews.get(a.candidate_id)
        if status=='needs_semantic_review':
            if review is None:
                status='needs_semantic_review'
            elif review.annotation_digest != digest(a.model_dump()):
                raise ValueError('Semantic review does not match annotation')
            elif review.verdict!='SUPPORTED':
                status='semantic_'+review.verdict.lower()
            else:
                if execution[a.candidate_id] != (a.label=='EXECUTE'):
                    raise ValueError('World execution disagrees with reviewed label; repair source prospectively')
                status='accepted'
        if status!='accepted':
            rejected[a.candidate_id]=status
        else:
            accepted.append({**packet,'label':a.label,'metadata':{
                **meta,'candidate_id':a.candidate_id,'supervision_version':VERSION,
                'annotation':a.model_dump(),'review':review.model_dump()}})
        if a.pair_id:
            groups.setdefault(a.pair_id,[]).append(a)
    for pair in groups.values():
        if len(pair)!=2 or any(a.candidate_id in rejected for a in pair):
            raise ValueError('Counterfactual requires exactly two accepted records')
        pair.sort(key=lambda a:audit[a.candidate_id]['event_index'])
        early,late=pair
        e,l=packets[early.candidate_id],packets[late.candidate_id]
        if early.label!='EXECUTE' or late.label!='SKIP' or audit[early.candidate_id]['event_index']>=audit[late.candidate_id]['event_index']:
            raise ValueError('Counterfactual requires chronological EXECUTE then SKIP')
        if e['candidate_subquery']!=l['candidate_subquery']:
            raise ValueError('Counterfactual query must be exactly identical')
        w=late.decisive_witness
        if w is None or not visible(w,l) or w.text in e['context_before']:
            raise ValueError('Counterfactual finding must be visible later and absent earlier')
        decisive_task=audit[late.candidate_id]['selected_task_ids'][w.branch_index]
        if decisive_task in audit[early.candidate_id]['published_task_ids']:
            raise ValueError('Counterfactual needs newly published decision-changing information')
    return accepted,rejected
