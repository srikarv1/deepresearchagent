"""Only these fixed strings may be sent back to a teacher. Never raw errors."""
from pydantic import ValidationError

GUIDANCE = {
 'support': 'Every evidence-dependent category needs an exact entire currently visible supporting_finding. Use a defensible different category if no visible finding supports it; never invent a witness or hypothetical conflict.',
 'identity': 'Each result belongs to one unique prior task, with exactly matching goal and status done. Persist identities/goals. Never duplicate a result ID within a state or reopen an active/done/dropped task.',
 'publication': 'Publish new result cards and their new feature pages together only after earlier task research completes. Never expose current/sibling output; preserve prior published records unchanged.',
 'history': 'Do not supply or alter executed history. It is exactly prior EXECUTE query strings in order; SKIPs add nothing. Do not invent unshown searches.',
 'search': 'Supply only nonnegative planning_searches_since_previous_state, excluding candidate executions. Local searches equal cumulative planning attempts plus prior EXECUTEs; reads must cover published pages.',
 'action': 'Every candidate/follow-up must request web retrieval of existing information. Find published experiments/analyses; do not run experiments, extract/compare/analyze data, draft, deploy, or perform work ourselves.',
 'pair': 'Each pair ID occurs exactly twice with byte-identical query text: earlier EXECUTE, later SKIP in another state. Its later exact finding must be visible then and absent earlier; unrelated records use null.',
 'schema': 'Return one complete JSON object matching the supplied schema and exact 8-state candidate counts [3,3,3,3,2,2,2,2]. Use explicit nonnegative actual usage, fixed local maxima, and valid categories. No markdown or undeclared fields.',
 'visibility': 'List visible_findings exactly as the local bounded view: only prior published last-four cards after field/token caps. Never future, evicted or raw-page findings. Keep findings concise enough for the caps.',
 'budget': 'Generation exhausted its completion allowance. Return concise complete JSON within the unchanged allowance; avoid redundant bookkeeping/prose. Preserve every semantic invariant and all required states/candidates.',
 'empty': 'The provider returned no assistant content. Return the complete required JSON object under the full original task.',
 'diversity': 'Generate a genuinely original question for the assigned topic and question structure; avoid reusing or closely paraphrasing existing benchmark or generated questions.',
}

class ContractError(ValueError):
    def __init__(self, code):
        self.code=code
        super().__init__(code)


def classify(exc):
    if isinstance(exc,ContractError):return [exc.code]
    codes=set()
    records=exc.errors(include_input=False) if isinstance(exc,ValidationError) else [{'msg':str(exc),'loc':()}]
    for record in records:
        # Classification may inspect local errors; output is restricted to keys below.
        text=(str(record.get('msg',''))+' '+'.'.join(map(str,record.get('loc',())))).lower()
        if any(x in text for x in ('held-out overlap','duplicate question')):codes.add('diversity')
        if any(x in text for x in ('supporting_finding','evidence-dependent','finding absent')):codes.add('support')
        if any(x in text for x in ('counterfactual','identical query','pair')):codes.add('pair')
        if any(x in text for x in ('completed research result','matching goal','task identities','operational task','current research task','active outer','branches','branch_id')):codes.add('identity')
        if any(x in text for x in ('publication','publish','initial trajectory','evidence history','page history','evidence.')):codes.add('publication')
        if any(x in text for x in ('executed','history')):codes.add('history')
        if any(x in text for x in ('search','read accounting','used_reads')):codes.add('search')
        if any(x in text for x in ('retrieval query','follow-up','synthesis/meta')):codes.add('action')
        if 'visible_findings' in text or 'bounded view' in text:codes.add('visibility')
    return sorted(codes) or ['schema']


def feedback(codes):
    return [GUIDANCE[code] for code in sorted(set(codes)) if code in GUIDANCE]
