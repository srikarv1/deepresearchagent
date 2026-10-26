# Final MLP-v2 synthetic source and supervision design

Status: offline contract and hand-authored fixtures only. No provider runner,
production prompt, dataset, embedding preparation, live gating or training enabled.
Frozen observation `mlp-observation-v2-branches-cited-top3-v1` is imported directly;
its selector, serializer, 4096 bound and feature definitions are untouched.

## Source: synthetic-branch-source-v1

A world contains question_id, original_question, and ordered discriminated events.
Tasks, results and feature-page records are authored exactly once:

| Event | Author provides | Compiler derives/checks |
|---|---|---|
| plan | New task IDs/goals, optional parent IDs, actual planning query strings, token delta | Root/depth, open status, planning searches; parent already published |
| batch | Current task ID and candidate IDs/retrieval strings | Active status, identical pre-batch snapshot, hierarchy, published results, local TF-IDF, context, features |
| end_batch | Executed proposal IDs, newly acquired feature-ledger page excerpts, read attempts, token delta | Proposal-order history, actual candidate search count, steps, page ownership; no reads without execution |
| process | Active task ID, findings/follow-ups, complete insight→URL map, processing token delta | Goal copied from task; processed status; result authored once, not yet visible |
| publish | Task IDs for one complete planner round | Stable planner order, done/publication order; all surviving tasks must have processed |
| drop | Open/active task ID | Dropped status; no reopening |

No repeated cumulative history, findings, counts, budgets, features or serialized
contexts in teacher output. Parent IDs encode hierarchy; depths are local. Planning
search strings account for real planning retrieval rather than guessed cumulative
search totals. All deltas are nonnegative. One query dispatch counts once; read
attempts may exceed unique pages. Maxima are local 32/64/128/400000, unenforced.
Steps = prior candidate proposals plus completed publication rounds. Token usage
is cumulative simulated research input+output, not teacher-generation tokens.

Only published branches plus the current task's *prior completed batches* provide
available query history/evidence. Current-batch and unpublished sibling products
are unavailable. Complete known task registry supplies branch features; published
and current-prior-batch pages supply the full available feature ledger, regardless
of top-three selection. Retention is true; no progressive pruning. Empty/missing
results never mean a research need was resolved.

Feature pages retain the existing live ledger convention: a source identity and
up to 2000 characters of an extracted-page prefix, with zero score. They exist
only to calculate the unchanged 15 features. They are NOT label evidence and are
never shown to the annotator. Generating these prefixes realistically remains a
source-quality review concern: don't substitute processed findings for raw prefixes
and claim identical numeric distributions. Trace raw-page counts differ from the
canonical numerical ledger; exact historical numeric vectors are unavailable.

Compiler output separates:
- annotation packet: original_question, exact context_before, candidate_subquery,
  and existing ordered numerical_features;
- audit: source/world digest, event/task/candidate identity, selected branch IDs,
  scores, publication/history counts, budget, token count, overflow, versions;
- actual execution ledger (never shown to annotator).

`compile_world(..., allow_prefix=True)` supports annotation immediately after a
batch proposal, before any future outcomes have been authored. Complete-world
compilation requires closed batch/active-task lifecycle.

## Supervision: synthetic-branch-supervision-v1

All annotations bind the SHA-256 digest of the exact packet. Rationale, categories,
review, pairs and witnesses stay outside the three embedded semantic strings.

SKIP categories:
- direct_answer: selected findings/insights directly answer the specific query;
- visible_duplicate: visible history or results demonstrate redundancy (history
  alone does not prove a successful answer; semantic review must assess that);
- negligible_marginal_value: visible semantic evidence establishes low marginal
  value for the current need, not general literature completeness.

EXECUTE categories:
- unresolved_need;
- visible_uncertainty (a real visible conflict/qualification, not hypothetical);
- missing_qualifier (population, method, condition or context);
- insufficient_answer (relevant visible branch does not answer the precise query);
- no_relevant_visible_evidence (not merely a low numeric score).

AMBIGUOUS and REJECT are nontraining dispositions. No class balance requirement.
Absence of retrieved evidence does not prove absence in the world, nor automatically
justify EXECUTE for an irrelevant candidate. A URL alone proves no semantic claim.

Witnesses cite exact finding strings, insight-map KEYS, or recent query strings in
the compiled packet. Branch indexes refer to relevance-ranked selected packets,
not hidden source IDs. Evidence-dependent categories require semantic witnesses;
all SKIPs require visible grounding. Exact witness matching blocks future,
unselected, raw-page-only and URL-only references. A packet digest blocks stale
annotations after any observation/feature/candidate change.

**Exact grounding is not semantic entailment.** Free-text rationale can still
smuggle unsupported claims. Therefore structural success is only
`needs_semantic_review`. Training acceptance additionally requires a separate
observation-only review, bound to the complete annotation digest, with reviewer
identity, SUPPORTED verdict and explanation. AMBIGUOUS/REJECT review excludes the
row. Fixtures have explicit hand-authored review records; a teacher must not fill
its own approval field. Review cannot automatically certify a human's reasoning.

An overflow packet is excluded regardless of proposed label and records the frozen
EXECUTE bypass. No truncation or coerced trainable EXECUTE label is introduced.
Reviewed labels must agree with committed world execution before final acceptance;
otherwise reject/repair the future source, never hide a skipped query in history.

Linked pairs: exactly two accepted records, same exact query, different chronological
batch events, EXECUTE then SKIP. Later decisive witness must be a selected semantic
finding/insight, absent from earlier context and supplied by a newly published
branch. Semantic review checks equivalent paraphrases and actual resolution;
exact-string checks alone cannot establish novelty or causation. Pairs with rejected
members are errors, never silently retained as complete pairs.

## Distributional targets (three development traces only, no copied content)

All runs: 28 tasks, depths 1–3 (4/8/16), 112 candidates; gate-time known tasks
4–28 (median15), published 0–26 (median13,p90=24), available history 0–104
(median52,p90=96). Published-result windows must genuinely matter.

| Quantity | Dev1 | Dev2 | Dev3 |
|---|---|---|---|
| Findings/result | 3 | 3 | 3 |
| Follow-ups/result | 2–3 | 2–3 | 2–3 |
| Cited insights/result | 5–10, median8 | 6–10, median8 | 6–9, median8 |
| Finding tokens min/median/max | 30/53/91 | 30/46/79 | 26/50/83 |
| Insight tokens min/median/max | 30/53/91 | 30/47/79 | 26/50/83 |
| Frozen observation median/p90/max | 3250.5/3454.9/3566 | 2840.5/3101.5/3238 | 2894.5/3105.5/3126 |
| Gate-time research tokens min/median/max | 10700/210573.5/404897 | 11929/195651/374218 | 12829/198818.5/385907 |
| Read attempts min/median/max | 0/175/327 | 0/207/402 | 0/170.5/347 |

Gate-time steps 0–120 (median59.5), searches2–137 (median69.5) in all runs.
Nominal step/search/read maxima frequently saturate; synthetic examples must not
make those counts perfectly predictive of SKIP. Preserve actual overrun counters,
clamp fractions and floor remaining values through existing code. Example state
profiles should cover early (0–4 published), middle (6–16), late (18–26), with
varied task lengths, imperfect research, and qualified/conflicting findings.

Target these as ranges, not mandatory per-world quotas or fabricated findings to
fill token budgets. Source schema has no small global task/history ceiling. The
hand fixture intentionally has short text; the mechanical 28-task/112-candidate
stress test proves capacity, not production semantic realism. Neither is training
data or evidence that a teacher already matches these distributions.

## Recommended production target and strategy (not implemented)

400 examples remain adequate for pipeline debugging, but only 20 independent
worlds leave state/selector diversity fragile for a final held-out-trajectory test.
More examples alone do not solve a 1.2M-parameter model's overfitting risk.
Propose **40 original worlds: 32 training and 8 independent synthetic validation**,
about **24 reviewed decisions/world**, approximately **768 train + 192 validation
= 960 total**. Assign questions to splits before generation/annotation, fixed seed,
never optimize the split on model performance. Diverse question structures/domains,
not 40 optimization templates. Aim roughly6 early/8 middle/10 late sampled decisions
per world, relaxing this for defensibility. Background research events are not
necessarily labeled examples. Do not pad to 960 with ambiguous labels.

Seek 1–2 valid state-changing pairs per world when defensible (roughly40–80 total,
8–16 validation); no per-world requirement or percentage quota. Counterfactual
examples count within the24, not in addition. Reject inaccessible-witness pairs;
do not force TF-IDF rankings by stuffing goal keywords.

Generation should be staged/incremental, not one enormous response:
1. Original question + task plan; run held-out and cross-split contamination checks
   locally using the existing protected registry (not loaded in this task).
2. Author source events/candidate proposals without labels, a planner round at a
   time. Compile prefix through the frozen selector.
3. A separate annotation call sees ONLY the exact packets. No source world, future
   results, feature-page text or unselected branches. Request minimal exact witnesses.
4. Observation-only semantic audit approves/excludes. Commit EXECUTE IDs locally
   before asking for their result cards; SKIPs never become executed history.
   Rejected proposals have no automatic binary training label. Source/background
   research decisions can be unlabeled and recorded explicitly, not invented later.
5. Author processed findings/citations after those searches; publish only at the
   round barrier. Continue until early/middle/late coverage exists. Recompile final
   world and verify every packet digest and action/history consistency.

Predeclare max3 attempts per unfinished source/annotation unit per invocation,
not3 regenerations of an accepted world. Local deterministic errors should be
fixed in code before another paid call. Allowlisted feedback only (identity,
publication, action-space, accounting, exact visible witness, category, pair,
format/count). Annotation retries resend the same packet, never hidden facts.
Ambiguity is exclusion, not a request to manufacture a stronger label. Empty/length
exhaustion is a budget failure; preserve response/usage and stop for review rather
than blind retries at the same capacity. No automatic semantic quality regeneration
or balancing. Immediately checkpoint accepted units, resume without replaying
accepted API calls, preserve raw outputs/rejections separately. These runner
behaviors are a design for the next step, not claimed implemented here.

## Artifacts for fair baselines

Keep versioned source events, provider model/prompt/settings/usage, per-attempt
outcomes, split/question IDs, exact packets+hashes, selection audit, annotations,
reviews/rejections, pair linkage and accepted-only examples. Save a corpus manifest
with hashes, feature order, frozen observation version, dependency/tokenizer
versions and overflow counts. Metadata must never enter X.

Later use the identical trajectory split, train-only normalization, cached frozen
embeddings and eligible examples for always EXECUTE, logistic regression and MLP.
Record per-example predictions, per-question metrics, per-pair both-correct rates,
coverage/rejection/overflow and label/category counts. No synthetic test split;
external evaluation remains separate. Optional heuristic is unnecessary for the
first comparison; if added, specify exact rule/threshold from training only before
validation. TF-IDF score alone is relevance, not evidence that a query is redundant.

## Next gate before paid work

Review this source/supervision design and freeze sampling/split targets. Then add
an offline-tested staged smoke runner and two separate prompt/schema views
(world-author versus packet-only annotator), checkpoint/resume and allowlisted
feedback. Demonstrate no source leakage in annotation requests and no self-approved
semantic review. Resolve a documented numerical-ledger sampling strategy; existing
traces cannot certify exact numerical feature distributions. Prepare one tiny paid
smoke command for explicit authorization. Do not resume the failed old generator.

The offline fixture's North Archive and `.invalid` sources are explicitly fictional.
Production synthetic source identities must likewise be marked synthetic in
provenance; generated citations are not verified publications. Neither annotator
nor model should need to dereference a URL to justify a label. Do not copy any of
the three development questions, findings, citations or labels into the new corpus.

Current fixture report: 8 compiled candidate packets over 7 tasks; 2 reviewed
accepted labels (1 EXECUTE, 1 SKIP), one linked pair, one explicitly ambiguous
rejection, five unannotated background decisions. Early contexts17–20 tokens,
middle198–200, late294. A separate10287-token overflow fixture records EXECUTE
bypass and is excluded from training. This is a correctness fixture, not a claim
that tiny fixture findings match the observed production-length distributions.
