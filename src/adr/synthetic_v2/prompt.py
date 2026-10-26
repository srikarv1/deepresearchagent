"""Teacher instructions scaffold; never called by the offline fixture command."""
import json
from adr.synthetic_v2.schema import TeacherTrajectory

PROMPT_VERSION='synthetic-v2-findings-draft2'

def teacher_messages(topic):
    return [{'role':'system','content':'''Generate original research trajectories, not held-out benchmark questions.
Return structured source states and supervision only. Never supply context embeddings, numerical features,
or final serialized context. Do not invent a resolved flag. Branch done means completed research, not solved.
Use concrete executed candidate search strings; planning searches affect used_searches but are not text history.
Use an empty initial result history, realistic pending/current research goals, and at least eight states that
cross the four-result and six-query windows. Completed cards are immutable prior research findings and
follow-ups (at most three each), available only after the outer research round and normal result processing.
No state may include findings from its current candidate or simultaneous sibling candidates.
Source evidence is separate numerical-feature input: retained=true, title empty, at most 2000 characters of
realistic extracted raw page text per source; it is not embedded in the observation. No progressive pruning.
Use fixed nominal maxima 32 steps,64 searches,128 reads,400000 cumulative input+output tokens; allow actual
overrun. Steps count candidate decisions plus published outer rounds. Searches count actual attempted
provider searches including planning. Reads count attempted URLs. Branches are actual planned outer tasks,
not inferred research needs. Preserve cumulative evidence/history/results and stable planner/result order.
The local observation uses only current need (64 tokens), last 3 pending needs (40 each), last 4 completed
cards (goal40,first3 findings64 each,first2 followups32 each), last6 executed queries32 each; cl100k_base.
JSON must fit2048 tokens: oldest queries, then pending needs, then cards are removed until it fits.
Label using ONLY this bounded observation plus original question and candidate; never justify labels
with omitted evidence or hidden numerical-source material. Include hard plausible SKIPs with state-dependent reasons.
Include linked earlier EXECUTE/later SKIP identical queries; the later decision-changing supporting_finding
must survive local bounds. Every pair ID occurs exactly twice; unrelated candidates use null.
This is a COMPLETE sampled candidate-decision trace: no unshown candidate batches or executed searches.
original_question must be an actual specific research question to answer, ending in a question mark;
never copy the topic or the instruction to formulate a research question.

ONE CARD PER COMPLETED OUTER TASK: result.branch_id equals a unique task ID in state.branches;
result.goal exactly equals that task.goal, and task.status is done. Distinct tasks use distinct IDs.
Only one occurrence of each result/task ID is allowed WITHIN each state. Across subsequent states,
persist each entire prior card unchanged, even after the last-four observation window hides it.
Never attach several findings cards to the single active root task. Put up to 3 findings in that task's
one card. A later task may investigate a follow-up without implying the earlier need was resolved.

Initial source has active/planned tasks but zero completed cards/pages/executed history and used_steps=0.
Each state's candidates belong to its current active outer task. In the NEXT state, append exactly the
previous state's EXECUTE candidate strings to executed_subqueries, in proposal order. SKIPs append nothing.
An earlier executed copy of a later SKIP query remains in cumulative history, but never append it again
because it was skipped. Account for ALL shown EXECUTEs, including empty search attempts.
Planning searches contribute only to used_searches, never this text history or used_steps.

When an outer task completes and a round publishes, mark it done and add its ONE matching card and
new evidence pages together in the next state. Record that boundary in publications_before, with task_ids
in planner order and evidence_ids listing ONLY newly published page IDs in stable order. Do not introduce
pages early or cards before processing completes. An active current task cannot publish its own results.
All publishing tasks must have had earlier sampled candidate batches. No current or sibling candidate
output enters its own pre-decision snapshot. A round can publish several previously researched tasks.
Use follow-ups as realistic new pending/current task goals, with new task IDs. Preserve cumulative full
source histories; only the LOCAL observation constructor applies windows.
used_steps = ALL prior candidate decisions (EXECUTE+SKIP) + ALL publications_before round records so far.
used_searches must increase by at least the number of prior EXECUTEs; extra planning searches are allowed.
used_reads must cover newly published pages. Maxima are explicitly 32/64/128/400000, never 50000.
Overruns remain valid. The separate GPT Researcher 50K context limit is NOT a cumulative token budget.

BOUND VIEW IS THE SOLE SEMANTIC INPUT: candidate wording, label, category and reason must follow from
original_question and the bounded current_need/pending_needs/cards/history. Do not use raw evidence pages,
full omitted cards, future findings, unpublished results or numerical features as a hidden semantic channel.
In particular do not insert a temperature/threshold/value from a raw page into a candidate before it is
visible in a published bounded finding. Source pages exist only to derive numerical features locally.

near_duplicate requires visible findings establishing meaningful redundancy, not merely similar query words.
already_established requires a visible finding materially answering that exact information need;
one brief study does not establish broad literature completeness or long-term safety.
weak_evidence_verification requires visible weak/uncertain evidence. conflicting_evidence_resolution requires
an actual visible conflict, not a hypothetical search for disagreement. Unresolved follow-ups normally argue
for EXECUTE unless another visible result resolves them. For these evidence-dependent categories and
unnecessary_confirmation supply supporting_finding as an exact quote surviving the final bounded view.
For conflicts the quote must describe the actual conflict. A matching quote alone does not prove a label.
All candidates must be plausible retrieval queries (find/retrieve/search published work), not instructions
to synthesize, create a comparative table, or formulate a research question. Use professional hard negatives.
At least two counterfactual pairs must have a decision-changing witness visible LATER but not EARLIER.
Do not force a 50/50 class balance. Teacher judgments still require human bounded-view adjudication.'''
    },{'role':'user','content':f'Topic: {topic}\nSchema: {json.dumps(TeacherTrajectory.model_json_schema())}'}]
