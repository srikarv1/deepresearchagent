# Observation and synthetic v2 scaffold

Version: `mlp-observation-v2-findings`; source schema: `synthetic-v2`;
feature-source semantics: `mlp-features-v2-source-v1`. This is an offline scaffold,
not a trained gate or an authorized real generation run.

## Canonical embedded text

`adr.mlp.observation_v2.construct` is used by synthetic compilation and the live
ledger's `inputs`. JSON uses sorted keys, UTF-8, no whitespace, and no version or
diagnostics fields inside the embedded text:

```json
{"current_need":"Compare lamp power use","pending_needs":["Check lifetime"],"prior_results":[{"findings":["Published measurements show lower LED power use."],"followups":["Find independent lifetime measurements."],"goal":"Establish power use"}],"recent_queries":["published LED power comparisons"]}
```

Source arrays are in committed chronological/planner order. Keep last 3 pending
needs, last 4 published cards, last 6 executed queries, preserving order within
these windows. Keep first 3 findings and first 2 follow-ups per card. Field caps
in cl100k_base tokens: current need 64, pending need 40, card goal 40, finding 64,
follow-up 32, query 32. Inline URLs/URL hashes become `[source]`; identifiers are
never independently serialized. Prefix token decoding discards incomplete UTF-8,
then rechecks token count. Count the final JSON including escaping. Above 2048,
drop oldest queries, then oldest pending needs, then oldest cards until it fits.
Never drop current need. Omission/truncation/reference-removal counts are separate
diagnostics. No raw page content, IDs, counts or budgets enter this text.
Question and candidate remain separate unmodified embedding inputs.

## Live lifecycle

`LiveObservationV2` is opt-in through the fork's `active_observation_v2` ContextVar.
The existing v1 adapter never activates it. A future gate must explicitly declare
this observation version; pairing the ledger with the frozen v1 gate fails.
There is no v2 checkpoint, preparation migration, or enable-by-default config.

Outer concurrency must be 1. Register actual planned outer tasks in planner
order, set active while their normal research runs, and mark done only on success
(dropped on failure). Done is operational completion, NOT resolved information.
Normal `process_research_results()` already produces findings and follow-ups;
there is no new processing call. Publish cards only after the existing outer-round
trajectory aggregation, after every sibling's research and result processing.
Thus a sibling cannot see another sibling's unpublished result. A later round can.
The inner query dispatcher freezes one deep-copied source snapshot for all proposed
queries before any query executes. Search attempts accrue immediately in accounting,
but issued search strings commit in proposal order after the batch. Planning
searches affect search counts, not executed-candidate text history. Failed/empty
search attempts still count as executed. No label/outcome of the current candidate
is present. Failed outer tasks have no result cards.

The incremental `scripts/patches/gpt_researcher_observation_v2.patch` applies AFTER
the existing query-gate patch, against the pinned fork. Both patches were checked
locally against a reconstructed clean baseline. Existing v1 hook behavior remains.

## Numerical feature source contract

The 15 names, order, formulas, saturation and zero-score placeholder are unchanged;
all derivation calls existing `derive_features`. They use the full pre-batch
committed source, not just the bounded text windows:

| Features | V2 source semantics |
|---|---|
| n_evidence, n_retained, n_pruned | Published global evidence pages; no pruning, so all retained, zero pruned. |
| n_open_branches, n_branches | Actual planned outer tasks; open/active count as open; done/dropped do not. No teacher-inferred resolved needs. |
| mean_score | Existing unavailable-zero placeholder. |
| mean_chars | Existing mean capped page-body character length (2000 per page). |
| query_term_coverage | Existing original-question lexical coverage against first 400 page characters, empty title. |
| unique_urls | Distinct source identities; hashes stay in numerical source, never context. |
| step_frac | Candidate decisions plus published rounds / nominal max_steps, saturated. |
| token_frac | Cumulative research input+output tokens / nominal max_tokens, saturated. |
| remaining_steps/searches/reads/tokens | Nominal maxima minus actual use, floored at zero. |

Searches include provider attempts including planning; reads count attempted URLs;
query history is not a numerical feature and excludes planning. Research tokens
come from the research tracker, not MLP embedding overhead. All maxima/usages must
be explicit; no inherited v1 defaults. Nominal maxima stay fixed; actual usage may
exceed them. Draft synthetic instructions use 32/64/128/400000, matching current
live total-run budget scale, not the separate 50K research context budget. The live
ledger reads actual configured maxima. No budget enforcement is introduced.

For feature calculation only, source pages retain at most the same first 2000
sanitized characters; these preserve the existing 2000-character mean and
400-character coverage calculations. Teacher-authored cards must model normal
processed research results, while feature-page prefixes must model extracted raw
material. This distinction needs review before paid generation.

## Offline synthetic path

`TeacherTrajectory -> SourceState -> shared constructor + derive_features -> rows`.
Structured teacher output cannot supply context or numerical features. Lifecycle
validation preserves cards, page history, task identities/goals, query history,
fixed maxima and cumulative actual usage. Counterfactuals require exactly two
chronological identical queries, EXECUTE then SKIP, and a later supporting finding
present in the final bounded observation. Held-out question overlap is checked.
A text witness is necessary but does not prove a label is semantically correct.

Tiny fixture (8 states, 20 examples) is under
`tests/fixtures/synthetic_v2_findings/`, explicitly marked NOT training data. It
has two linked reversals, empty initial results, seven eventual result cards,
16 prior issued queries in its final snapshot and token/search/read budget overruns. It is deliberately handcrafted, not a
conversion of v1. Reproduce into a NEW directory:

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_v2.generate --fixture --output /tmp/new-v2-fixture
```

Production generation is available via the offline-tested runner below. Structural
acceptance still requires subsequent semantic review of bounded-view labels, page/card
consistency, task timing and budget realism before training. No v2 preparation/training
path or live learned v2 checkpoint is enabled here.
Do not feed these fixtures or v2 inputs to MLP #1. V1 data/artifacts are unchanged.

## Smoke #2 contract correction (draft2)

The original failed smoke under `runs/synthetic-v2-smoke/teacher-smoke-20260923/`
is preserved, not repaired. Its duplicate task/card identities remain invalid.

Teacher-only `TeacherBudget` requires all eight budget fields explicitly. Its four
maxima are constants 32/64/128/400000; there is no inherited 50000 default. Live
`SourceState` and the 15 feature formulas are unchanged. This is a stricter
synthetic authoring contract, not a change to the deployed observation format.

Synthetic states now include `publications_before`: ordered round records with
`task_ids` and newly published `evidence_ids`. These are provenance, never text
or numeric model inputs. Every new card must belong to exactly one listed task;
every new page must appear at one of those same boundaries. Published tasks must
have an earlier sampled batch. Existing card uniqueness, done-status, matching
goal and immutable cumulative persistence checks remain intact. Only the bounded
observation drops old records; the source must never delete them.

For this complete smoke trace, every candidate batch is sampled: no unshown
candidate executions are allowed. Cumulative history must equal exactly all prior
EXECUTE query strings, preserving order and repetitions. SKIPs append nothing;
an independently earlier executed copy may persist. Steps must equal prior
candidate decisions plus recorded publication rounds. Planning searches/tokens
are allowed initially, but initial cards/pages/history/steps are empty/zero.
This intentional authoring restriction makes history and accounting checkable;
live research can have more batches, which a future sampled-data collector must
record rather than silently omit.

Evidence-dependent categories require a supporting_finding quoted from the
bounded view. Later counterfactual witnesses must not already be visible at the
earlier occurrence. This detects missing/evicted/raw-only quotes, not arbitrary
semantic leakage. Human review still must establish that findings actually imply
redundancy, uncertainty or conflict, and that candidate wording did not exploit
hidden facts. Obvious meta-question/synthesis instructions are rejected, but
regexes are not a complete action-space classifier.

Smoke #2 is prepared under `runs/synthetic-v2-smoke/teacher-smoke-2-20260923/`.
Its runner uses the shared Azure-compatible gpt-5-mini client, retains raw output
and usage before validation, disables SDK retries, and permits one call only.
The request has been prepared offline; no API call was made as part of this fix.

## Smoke #3: event authoring and completion diagnostics

`prompt_v3.py` / `events.py` add `synthetic-v2-events-v1` as a teacher-only
encoding, without replacing the validated expanded `TeacherTrajectory` contract.
Observation and feature versions remain unchanged. The teacher supplies the real
question, stable task registry/current ID, ordered NEW publication rounds (cards
and pages together), actual cumulative search/read/research-token usage, and
candidate supervision including exact visible witnesses. Cards/pages are provided
once rather than repeatedly copied through successive snapshots.

`expand()` accumulates immutable published records, derives round task/page ID
lists, all prior EXECUTE query history, prior candidate decisions + publication
rounds, and nominal 32/64/128/400000 maxima. Page title/retention are fixed empty/true
under the existing no-pruning contract. It then calls the unchanged strict
TeacherTrajectory validator; the existing compiler subsequently enforces visible
witnesses and builds bounded context and the same 15 features. Unknown teacher
bookkeeping fields are rejected, never silently ignored or repaired. Publication
metadata is not embedded. Equivalent fixture inputs produce exactly the same
expanded source, model text, numeric features, and supervision as draft2.

The new smoke-local runner uses gpt-5-mini with max_completion_tokens=32000, no
reasoning-effort override and no retries. SDK/local client accepts this integer;
the installed LiteLLM model metadata lists 128000 max output for both gpt-5-mini
and azure/gpt-5-mini. This is offline compatibility evidence, not a remote check
of a particular Azure deployment quota or service override.

Raw response/text/usage are saved first. response_status.py checks empty content,
length termination, completion exhaustion and reasoning exhaustion BEFORE parsing
JSON. Budget exhaustion reports GenerationBudgetFailure and structural validation
not_run. Empty content without budget evidence is reported distinctly rather than
asserting an unsupported cause. All these failures stop without automatic retry.

Prepared command (NOT executed during implementation):

```sh
PYTHONPATH=src .venv/bin/python runs/synthetic-v2-smoke/teacher-smoke-3-20260923/run_smoke.py --allow-api
```

Local cl100k_base content counts (excluding chat framing; not billed token counts):
draft2 system/schema/total = 1582/1823/3429; draft3 = 945/1135/2104. Fixture full
source JSON is 7612 tokens versus 3910 for equivalent event JSON. Human semantic
review remains necessary; local witness membership does not prove entailment,
realistic research results or absence of arbitrary semantic leakage.

## Smoke #4: retrieval-only actions and explicit bounded visibility

The one-call event format is preserved, with a teacher assertion `visible_findings`
on each state. `expand_visible` strips only this assertion, uses the unchanged event
expander and lifecycle validators, then compares it exactly against findings from
the authoritative shared bounded constructor. Assertions cannot change context,
features or supervision. The smoke saves locally generated `visibility_views.json`
with exact embedded context and omission diagnostics after this check succeeds.

This is a SINGLE-CALL compromise, not locally computed feedback delivered to the
teacher during generation. The teacher explicitly lists its believed current
view; local code verifies it afterward. A staged teacher workflow would be required
to supply computed states before labeling and is not introduced here. Full source
history remains present to the teacher in the one-call design, so declaration
checks do not prove freedom from semantic leakage in wording or rationales.

Draft4 says every candidate must be directly passable to Tavily/search, gives
paired good/bad examples, and distinguishes retrieving published experiments,
simulations, analyses and evaluations from performing them. Findings describe
retrieved-source results and follow-ups express retrieval needs. Hard negatives
require state-dependent marginal value, not generic dismissals or claims that one
narrow study establishes broad literature completeness. Counterfactual query strings
must match exactly; later witnesses must be visible and absent earlier.

`action_space.py` conservatively rejects obvious leading non-retrieval imperatives
in candidates and synthetic result follow-ups. Explicit find/search/retrieve
phrases and noun phrases about published experiments/analysis remain allowed.
The screen is not a semantic classifier: disguised/composite imperatives still
need review. Live result-card schema and gate are unchanged; follow-up screening
runs only during synthetic compilation. Supporting quotes now must equal an entire
visible bounded finding, rather than merely occur as a substring. Prior exact pair,
future/evicted witness, publication, task, history and budget checks remain intact.

Prepared single-call command (not executed during preparation):

```sh
PYTHONPATH=src .venv/bin/python runs/synthetic-v2-smoke/teacher-smoke-4-20260923/run_smoke.py --allow-api
```

Same gpt-5-mini, 32000 completion allowance, 8 states/20 candidates, no forced
balance, no retry. All earlier paid smoke artifacts are preserved unchanged.


## Production generation (no teacher calls during implementation)

Prompt `synthetic-v2-findings-production-v1`, teacher event schema
`synthetic-v2-production-events-v1`, unchanged observation/feature versions.
Config: `configs/synthetic_v2.yaml`. Target: 20 independently generated original
questions, exactly 8 states and [3,3,3,3,2,2,2,2] candidates each (400 examples).
Twenty stable topic/structure assignments encourage different reasoning demands,
not just different topics in the same optimization template. Existing near-duplicate
checks apply across accepted questions, including reloaded checkpoints. This is not
a guarantee against semantic/template similarity; the quality audit must check it.

The teacher now supplies actual `planning_searches_since_previous_state`, not
cumulative searches. Local cumulative usage is prior EXECUTEs + cumulative planning
attempts, eliminating candidate-search undercount. Actual read/token counters stay
teacher-authored and validated. No unrecorded candidate batches/searches are allowed.
Open tasks must become active before completion. Active tasks may remain active or
finish done/dropped; active->open is forbidden. Unresolved knowledge is represented
as follow-ups/new tasks. Analysis-oriented extract/compare/process/evaluate follow-up
commands are rejected unless phrased as retrieving published information.

The prompt asks for diverse difficult negatives when supported: established needs,
redundancy, unnecessary confirmation, low marginal value, premature/overly specific
queries, and plausibly related but irrelevant queries. No label balance or category
quota is enforced, and class imbalance never triggers retry. The fixed acceptance
shape retains two strong validated state-dependent pairs and window overflow;
unsupported forced pairs must fail rather than be semantically repaired.

Offline check (writes no production directory/contract; does not bind an endpoint):

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_v2.production --config configs/synthetic_v2.yaml --dry-run
```

Fresh generation, from the repository root using existing OPENAI_API_KEY and Azure
OPENAI_BASE_URL ending /openai/v1/ (NOT executed during implementation):

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_v2.production --config configs/synthetic_v2.yaml --output data/synthetic/v2/production-v1 --allow-api
```

Resume that exact directory:

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_v2.production --config configs/synthetic_v2.yaml --resume data/synthetic/v2/production-v1 --allow-api
```

Without --output, a fresh timestamped directory is created under configured
output_root. Fresh mode never overwrites an existing directory. Each invocation
allows at most 3 attempts for each unfinished trajectory. SDK retries are disabled;
authentication/authorization failures stop immediately. Every retry retains the
original prompt and assignment plus fixed allowlisted guidance. Raw rejected text,
held-out material, secrets, and arbitrary error messages never enter retry prompts.
Generation-budget/empty responses are detected before parsing, preserved and safely
retried within the same cap. There is no unbounded retry or automatic semantic edit.
The broader held-out registry is checked locally and must include DRB 51–55.

Structure:

```
production-v1/
  contract.json                 # config, versions/code hashes, registry hash,
                                # endpoint hash, relevant dependency versions
  manifest.json                 # partial/complete, counts, canonical hash/order
  examples.jsonl                # accepted locally compiled rows only
  attempts/q001/attempt-0001/
    request.json, started.json
    response.json, raw_teacher.txt, usage.json  # when provider returned
    outcome.json                # safe codes / validation / generation status
  accepted/q001/
    events.json, expanded_source.json, visibility_views.json
    examples.jsonl, provenance.json, checkpoint.json
```

Each accepted bundle is written and fsynced in a staging directory, then committed
by atomic directory rename BEFORE moving to the next trajectory. A per-run advisory
lock prevents simultaneous writers. Resume verifies config/code/registry/endpoint
contract and all checkpoint hashes, recompiles each accepted event record, verifies
canonical row bytes and reloads questions for duplicate checking. Completed bundles
are never regenerated or overwritten. Orphan staging directories are not accepted.
Root canonical JSONL and manifest are rebuildable from committed bundles; an
interruption between their separate atomic writes is repaired before any new call.
A completed resume does not even construct a teacher client. Contract mismatches
or corrupted accepted bundles fail closed, not by silently regenerating them.

Per-trajectory provenance records assignment, question, accepted/total attempt
number, prompt/event versions, usage/provider model, request/response hashes,
structural results, context sizes, label/category distributions, non-pair SKIP
categories, and exact counterfactual linkage/witnesses. Failed-attempt usage is
kept separately under attempts. Canonical rows contain only accepted output;
`semantic_audit: required_before_training` is deliberate: string checks cannot
prove actual conflict/entailment, scientific plausibility or absence of all semantic
leakage. Single-call teacher visibility assertions are verified afterward; this is
not a staged workflow that hides all future source events from the teacher.
