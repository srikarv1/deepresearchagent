# Four-question observation-only pilot

Status: prepared for a paid run, not executed. This is audit data only. No
`examples.jsonl`, embeddings, training acceptance, or second reviewer is produced.
The prior smoke and canonical source/compiler/observation/feature code are unchanged.

## Versions and supervision

- Runner: `synthetic-branch-pilot-v1`.
- Annotation: `observation-only-supervision-v2-marginal-value`.
- Numerical proxy: `synthetic-visible-operation-tokens-v1-cl100k`.
- Frozen observation: the existing branch/cited-insight constructor, unmodified.

Annotation schema adds `action_eligibility`, `visible_coverage`, and a structured
`gap` (dimension, exact requested phrase, concrete missing information). Evaluation
order: action eligibility, visible coverage, marginal value, exact witnesses.
Invalid extraction/synthesis/calculation/physical actions require AMBIGUOUS with
`invalid_action_space`; uncertain actions cannot receive a binary label.
The conservative local screen also rejects obvious imperative/mixed actions but
is not a comprehensive semantic classifier. Searches for published work are valid.

When selected branches contain findings/insights, EXECUTE requires a specific gap
anchored to an exact phrase in the candidate/current need. This conservative rule
also applies if the annotator considers those branches irrelevant. Partial coverage
requires a finding/insight witness. Generic incompleteness alone is not a gap.
The local generic-phrase screen is deliberately narrow: a superficially specific
but invalid rationale still requires human adjudication. No deterministic check
can certify marginal value or entailment. SKIP means sufficiently answered for
the actual objective, not exhaustive literature coverage. Neither relevance nor
query history alone establishes resolution. No label quotas or forced reversals.

Witness locations:

| kind | Exact location | index |
|---|---|---|
| finding | `branches[i].findings` element | i |
| insight | `branches[i].cited_insights` key, not URL | i |
| followup | `branches[i].followups` element | i |
| history | `recent_queries` element | null |

The original question/current need cannot masquerade as history. Empty witnesses
are explicitly valid for initial unresolved EXECUTEs. Follow-ups establish a
research need, not an answer; SKIP requires a finding/insight. All quotes are
validated even for AMBIGUOUS. Old records are never migrated or accepted.

## Questions and shape

The four exact questions are pinned in `configs/synthetic_branch_pilot.yaml`:
public-library fine removal/automatic renewal, heterogeneous four-day school-week
effects, seed-bank resilience across biological/logistical/governance branches,
and transferability of arsenic-filter evidence to rural Bangladesh.
Each world must retain its assigned exact question and ID. Target 18 decisions;
accepted world scope 15–20, with initial zero-result, middle 1–3-result, and late
>3-result snapshots. Suggested shape: 2 roots ×3 candidates, 3 children ×2,
2 late tasks ×3. Normal canonical hierarchy/publication validation remains active.
Worlds must finish/drop their tasks and publish all processed surviving results.

Question exclusion check (offline): 24 unique prior synthetic questions, 3 local
development questions, and 3 non-51–55 archived questions in the existing registry.
No exact matches; maximum normalized character SequenceMatcher similarities were
0.2732, 0.3002, 0.2560, 0.3303 respectively. This is an exact/lexical contamination
check, not a semantic proof. Registry IDs 51–55 were explicitly skipped before
accessing question text; no benchmark files/reports were inspected. The questions
were authored independently, not derived from excluded benchmark content. No
original William ZIP was found in the searched local archive locations; the
existing non-protected archive registry supplied those reference questions.

## Visible-token ledger

Teacher-facing schema is a strict projection of the existing World schema with
`research_tokens` removed from Plan/EndBatch/Process. A teacher-supplied value is
rejected, not silently ignored. All other fields/lifecycle retain the canonical
schema. No generic schema repairs are performed. Raw authored responses remain
untouched; the locally materialized source world is saved separately.

A local adapter inserts deltas, then passes the world to the unchanged compiler.
It renders compact, sorted, UTF-8 JSON for each represented synthetic operation:

1. Plan: original question and prior published context as input; new task goals
   and planning queries as output.
2. Query generation: original question, current need, available completed results,
   prior executed queries and available excerpts, frozen before the batch; candidate
   query strings as output. Charge at EndBatch, maintaining canonical pre-decision
   accounting timing.
3. Process: original question, current goal, published results/history and current
   task's acquired excerpts; findings/followups/cited-insight map as output.
4. Publish/drop: zero LLM tokens. Search/read dispatch alone has no LLM-token charge.

Each operation charges cl100k counts of the rendered input plus output. Repeated
inputs are charged each time used; future/unpublished sibling results are excluded.
All rendered operation inputs/outputs and counts are preserved for audit. This
is NOT actual GPT-5 usage: it omits hidden reasoning, real system prompts, unmodeled
calls and full retrieved content; cl100k is the offline estimate tokenizer, not a
claim about GPT-5 billing tokenization. It is not a guaranteed provider-token lower
bound. No multiplier, calibration to development traces, or budget padding is used.
Numerical rows remain ineligible for training until accounting suitability is
explicitly resolved. Actual teacher/annotation usage is saved separately.

Teacher-authored: task/query/result semantics, execution subsets, feature-page
excerpts/URLs, read-attempt counts. Derived: hierarchy/publication, branch/evidence
counts, query history, searches (planning strings + executed candidates), reads
(sum of reported attempts), steps (prior proposals + publication rounds), visible
token estimates and cumulative usage. The existing 15 feature formulas/order,
32/64/128/400000 nominal budgets, and overrun saturation are unchanged. Only
`token_frac` and `remaining_tokens` change when comparing identical worlds with
zero versus derived token deltas; no textual observation changes.

## Isolation and checkpoints

World author → canonical local compilation → one isolated annotation per candidate
→ deterministic validation → human audit. Annotation receives exactly original
question, frozen context, candidate string and supervision schema plus fixed
instructions. It receives no ledger, features, source execution, hidden rationale,
other observations, unselected branches, raw pages, or proposed negative/pair IDs.

One provider call maximum per stage for the lifetime of the directory; SDK retries
zero. Responses and usage are atomically preserved before validation. Completed
stages are hash-checked and reused. Failed stages are never called again. Started
without response means uncertain completion, not permission to retry. A saved
response without outcome can be locally validated on resume without a call.
Independent remaining annotation stages/questions continue after a failed stage;
world failure prevents all annotations for that world. Resume can execute only
previously unstarted stages. A new paid attempt requires explicit new-run planning,
not an implicit reset. Contract changes reject resume.

At target size: 4 world + 72 annotation = 76 calls. Hard maximum: 4 + 80 = 84 calls.
World completion allowance 32,000, annotation allowance 16,000, model GPT-5-mini.
These are ceilings, not token-spend estimates. No second semantic reviewer call.

## Artifacts

Root: contract, manifest, combined `human_audit.md/json`.
Per question: world request preview; preserved per-stage requests/raw response
bundles/usage/outcomes; accepted structural world; exact compiled packets and
compilation audit; visible-operation token accounting; cumulative numerical ledger.
Combined audit contains raw proposals even after structural rejection, label/category
counts, AMBIGUOUS reasons/rate, valid-structure retrieval labels (not semantic truth),
structural rejection rate, action/witness failure counts, exact observations and
sizes, phases, selected goals, source disagreement, per-question summaries and
repeated-query comparisons. Repeated strings never force a reversal or certified
counterfactual: newly visible witnesses and both labels are shown for human review.
All final training counts remain zero.

## Commands

Offline preparation (already performed by this implementation task):

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_branch_v2.pilot --config configs/synthetic_branch_pilot.yaml
```

Future paid execution from the same repo with existing `OPENAI_API_KEY` and
`OPENAI_BASE_URL` (HTTPS endpoint ending `/openai/v1/`):

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_branch_v2.pilot --config configs/synthetic_branch_pilot.yaml --resume --allow-api
```

Use the same command after interruption; it cannot retry a reserved/failed stage.
No credentials are read or needed for offline preparation.
