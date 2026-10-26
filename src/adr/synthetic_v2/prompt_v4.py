"""Retrieval-only single-call smoke; verify explicit visibility assertions locally."""
import json
from adr.synthetic_v2.prompt_v3 import teacher_messages as previous_messages
from adr.synthetic_v2.visibility import VisibleEventTrajectory

PROMPT_VERSION='synthetic-v2-findings-draft4-retrieval-visible'


def teacher_messages(topic):
    messages=previous_messages(topic)
    messages[0]['content']+='''
SMOKE4 HARD ACTION CONTRACT: Every candidate must be a web/retrieval search query that could be passed UNCHANGED to Tavily/search at that moment. The agent retrieves existing sources; it never performs new experiments, instrumentation, collection, simulation, computation, synthesis, drafting, negotiation, procurement, building or deployment. This applies even when those activities would be valuable to a human researcher. You are NOT planning laboratory or engineering work.
GOOD: "Find published instrumented LFP container experiments comparing ventilation at high ambient temperature."
BAD: "Run a six-hour instrumented container test."
GOOD: "Retrieve published analyses estimating lifecycle maintenance costs of evaporative cooling."
BAD: "Estimate lifecycle costs using our existing measurements."
GOOD: "Search published evaluations of HVAC operating policies and alarm thresholds."
BAD: "Draft an SOP integrating the findings."
GOOD: "Find independent field validation of published sensor thresholds."
BAD: "Deploy sensors and field-validate the thresholds ourselves."
GOOD: "Find published reviews comparing cooling-system energy penalties."
BAD: "Synthesize our data into a comparative table."
Other forbidden agent commands include conduct/deploy a pilot, instrument hardware, collect new measurements, simulate/model a system ourselves, calculate/analyze existing data, create a report, formulate a research question, negotiate/procure/build/implement anything. Mentioning published experiments, datasets, analyses, simulations or evaluations inside an actual retrieval query is valid.
VISIBILITY FIRST: In each state explicitly write visible_findings BEFORE candidates. List exactly the findings surviving the CURRENT local observation rule: only prior published cards, last four cards, first three findings each capped to64 tokens, and any further whole-view bounding. Initial list empty. Keep source findings short to avoid ambiguous clipping. Never copy from a future publication, evicted card, raw feature page or full-history memory. Your rationale may name ONLY facts in this visible view or original question/current/pending needs/history, not unexposed preliminary planning notes. Compiler recomputes the real view and rejects any mismatch; your declaration cannot override it.
Each supporting_finding must equal one ENTIRE currently visible finding exactly, not a substring. If no visible finding supports an evidence-dependent category, choose a genuinely appropriate non-evidence-dependent category or reconsider candidate/label; do not invent a witness.
PAIR STRINGS: Copy the earlier candidate query byte-for-byte at the later occurrence; no parenthetical edits or paraphrases. Later decision-changing finding must be currently visible and absent earlier. SKIP means this SPECIFIC information need now has low marginal value, not that one narrow study proves all literature complete. Do not label useful independent verification SKIP merely because a related query ran before.
LITERATURE FINDINGS: Attribute outcomes to retrieved publications/reports/datasets/reviews ("A published laboratory study reports...", "The retrieved review found limited field evidence..."). Never imply the research agent ran trials, measured temperatures, simulated outcomes or produced new experimental evidence. Follow-ups must also be retrieval-oriented information needs, e.g. "Find independent field validation...", not "Replicate the experiment" or "Refine our model". Retain realistic uncertainty, absent evidence, conflicting reports, unsuccessful searches and unresolved questions when appropriate; never force these for diversity. Do not convert a speculative/physically unexplained result into an established fact. Reuse the exact completed task goal; no task completion implies full resolution.
Keep all original lifecycle, category, history, budget, no-pruning, window and counterfactual rules. Exactly one smoke; no retries or repair.'''
    messages[1]['content']=f'Topic: {topic}\nSchema: {json.dumps(VisibleEventTrajectory.model_json_schema(),separators=(",",":"))}'
    return messages
