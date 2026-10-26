"""Teacher instructions contain no held-out reference text or baseline outputs."""
import json

from adr.synthetic.schema import EXECUTE_CATEGORIES, SKIP_CATEGORIES, TeacherTrajectory

PROMPT_VERSION = "synthetic-teacher-v2.1"


def teacher_messages(states: int, candidates: int, topic: str) -> list[dict[str, str]]:
    instruction = f"""Generate one ORIGINAL synthetic multi-part research question about {topic}.
Return only a JSON object conforming to the provided schema, with exactly {states}
successive states and {candidates} plausible candidate subqueries per state.
Provide reasonable representation of both EXECUTE and SKIP across generated data;
do not force an exact 50/50 split within each trajectory or state. Assign labels
from the research state, not to satisfy a quota. This is MARGINAL VALUE, not relevance:
EXECUTE iff executing this candidate now is expected to add enough useful/new
information toward the original question to justify research cost. Otherwise SKIP.
EXECUTE categories: {sorted(EXECUTE_CATEGORIES)}.
SKIP categories: {sorted(SKIP_CATEGORIES)}.
Make most SKIP candidates independently plausible research actions that a competent
researcher could reasonably consider executing. Their low marginal value must come
primarily from the current state: sufficiently established information, substantial
redundancy, superseded evidence, unmet prerequisites, or an information need whose
remaining value does not justify spending the remaining budget now. Use only the
existing decision categories above; these descriptions do not add new categories.
Keep EXECUTE and SKIP wording similarly specific, professional, and plausible.
Do not signal the label through lexical giveaways such as "compile an exhaustive
bibliography of all...", "repeat the same experiment to reconfirm...", or wording
that declares the action redundant or pointless. Do not manufacture negatives by
making them intrinsically silly, irrelevant, exaggerated, or wasteful. Put the
state-dependent justification in reason, not in the candidate wording. Check that
most SKIP candidates could be worthwhile in a different plausible research state.
Generate structured PRE-DECISION state only: evidence already available, cumulative
executed subqueries, branches (including unresolved needs), cumulative prior budget.
Use fictional evidence and synthetic source URLs under example.org; do not perform
real search or retrieval. Construct coherent but imperfect research progression.
When logically appropriate, include weak, redundant, or superseded evidence and
prune items via retained=false rather than retaining everything. Evidence may only
partially answer a need, or sources may conflict and require further verification.
A branch may remain open or active despite accumulating evidence; an executed prior
subquery need not fully solve its branch or yield useful information. Use multiple
such variations when supported by the question, rather than a perfectly linear
plan in which every action succeeds. Any weakness, conflict, pruning, or unresolved
need must be semantically justified by the evidence and branch goals already
available in that state. Do not introduce arbitrary noise or force every variation.
Do not include candidate outcomes, resulting observations, future evidence, future
states, current-candidate costs/latency, summaries, numeric feature vectors, or
controller decisions inside state. Never supply numerical_features, feature values,
or context_before yourself: context_before and all 15 numerical features are
constructed deterministically by local code from the structured state. Do not put
labels/reasons in evidence or branches.
Candidates at a state are alternatives sharing the SAME state, not sequential actions.
Keep state IDs unique and candidate IDs unique within each state.
Evidence is a cumulative history, NOT a list of only currently retained items.
Once an evidence item appears, include that same item in EVERY later state's
evidence list, even after pruning. Never delete, replace, or renumber its id, and
never rewrite its original url, title, or text/body (including shortening or
summarizing it). For an existing evidence item, ONLY retained may change; the
evidence schema has no other status field. If an item becomes weak, redundant,
superseded, conflicting, or otherwise not retained, preserve it in full and set
retained=false. New states may ADD new evidence with new IDs, but must copy every
prior item unchanged except for retained. Represent corrections or conflicting
findings as NEW evidence items, not edits to old evidence. Realistic pruning means
retention changes, NEVER removal or rewriting. Before returning, check each pair of
successive states: all prior evidence IDs remain and their url/title/text match.
History and prior resource usage are cumulative, with fixed positive maxima. Token units
are cumulative input+output tokens, not context size. Default maxima: 32 decisions,
64 searches, 128 reads, 50000 tokens. Do not fabricate a relevance score.
For trajectories with at least two states, include at least one linked pair with the
same candidate EXECUTE early and SKIP later after evidence satisfies its need.
For a five-state trajectory, include at least TWO distinct linked counterfactual
pairs whenever feasible; plan multiple information needs so this is normally
possible without inventing unjustified label flips. A pair may use identical wording
(preferred) or a close semantic equivalent with the same scope and information need.
Keep the original question fixed. The later SKIP must follow acquired relevant
evidence or resolution of that need, not a change that makes the candidate itself
less plausible. Assign a distinct counterfactual_pair_id to each pair, shared by
exactly its two candidates in different states: early EXECUTE and later SKIP.
Explain the specific already-available evidence or resolved need that changes the
decision in each candidate's reason, outside the state. The early reason must not
refer to evidence acquired later. These pairs must show that the decision depends
on (question, context_before, candidate), not candidate wording alone.
Use other candidates to diversify categories. Never reproduce benchmark questions.
Schema:
{json.dumps(TeacherTrajectory.model_json_schema())}"""
    return [{"role": "system", "content": "You generate synthetic supervised research decisions. Follow the JSON contract exactly."},
            {"role": "user", "content": instruction}]
