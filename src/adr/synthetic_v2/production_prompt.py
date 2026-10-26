"""Production prompts retain the full task on every bounded retry."""
import json
from adr.synthetic_v2.prompt_v4 import teacher_messages as smoke_messages
from adr.synthetic_v2.production_schema import ProductionTrajectory

PROMPT_VERSION='synthetic-v2-findings-production-v1'


def teacher_messages(topic, structure, feedback=()):
    messages=smoke_messages(topic)
    system=messages[0]['content']
    system=system.replace('SCOPE: exactly 8 states','PRODUCTION TRAJECTORY: exactly 8 states')
    system=system.replace('Exactly one smoke; no retries or repair.','Exactly one trajectory in this response.')
    system=system.replace('SMOKE4 HARD ACTION CONTRACT','HARD ACTION CONTRACT')
    system=system.replace('Exactly one smoke','Exactly one trajectory')
    system=system.replace('one smoke','one trajectory')
    system=system.replace('no automatic repair or regeneration','no automatic semantic editing')
    system=system.replace('Provide only realistic cumulative actual searches (including planning), attempted reads, and research input+output tokens.',
        'Provide planning_searches_since_previous_state (actual planning search attempts only) plus cumulative attempted reads and research input+output tokens. Local search usage equals prior EXECUTEs plus cumulative planning attempts. Do not supply used_searches or count EXECUTEs as planning.')
    system+='''
PRODUCTION DETAILS: The assigned question structure is binding guidance; vary the actual wording and reasoning demands rather than recasting every topic as an optimization plan. Return an original research question, not these instructions. Do not reproduce held-out benchmarks.
Operational tasks follow open -> active -> done/dropped; an active task must never revert to open. A completed research operation can leave the information need unresolved; express that in retrieval-oriented follow-ups or NEW tasks, not reopened task IDs. Failed/empty research may yield no new evidence; never fabricate useful findings to create monotonic progress.
Follow-ups must retrieve information, not extract/compare/analyze data ourselves. GOOD: 'Retrieve published comparisons of sampling methods.' BAD: 'Extract logs and compare their sampling.'
All rationale assertions must be present in the bounded observation/original question; do not invent references, imply manufacturer baselines are cited when they are not, or overstate what the visible study establishes. conflicting_evidence_resolution requires an actual visible contradiction, never merely the possibility of differing assumptions. Evidence-dependent categories require an exact ENTIRE visible supporting_finding. If unavailable, revise the category/candidate/label honestly rather than inventing a witness.
NEGATIVE DIVERSITY: Where justified, include difficult SKIPs beyond linked repeats: an already-answered specific need, visibly redundant near-duplicate, unnecessary confirmation, low marginal value, premature query, overly-specific query before prerequisites, or plausibly related but contextually irrelevant retrieval. Do not use obvious nonsense or lexical giveaways. No class balance target or per-category quota. Do not manufacture negatives. Preserve two strong identical-query EXECUTE->SKIP pairs when feasible; this fixed production format requires two validated pairs to accept a trajectory, so design states with genuinely visible decision-changing findings rather than relabeling unsupported candidates.
Each trajectory is 8 states / 20 candidates with [3,3,3,3,2,2,2,2]. Include >4 cumulative published result cards, >6 prior EXECUTEs, and realistic pending needs. Retain all previous semantic and structural requirements. If retry guidance is supplied, regenerate this entire trajectory under the original full task; do not return patches. No raw rejected output or held-out content is supplied in feedback.'''
    messages=[{'role':'system','content':system}, {'role':'user','content':
        f'Assigned topic: {topic}\nAssigned question structure: {structure}\nSchema: '+json.dumps(ProductionTrajectory.model_json_schema(),separators=(',',':'))}]
    if feedback:
        messages.append({'role':'user','content':'Local validation failed. Correct these contract requirements while preserving the full task:\n'+'\n'.join(feedback)})
    return messages
