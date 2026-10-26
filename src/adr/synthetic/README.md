# Synthetic candidate-decision pilot

Scope: structured teacher generation, deterministic preprocessing of text/state
into supervised records, and local validation. No embeddings, training, policy
integration, or benchmark evaluation is implemented here.

Run commands from the repository root using the installed development environment:

```bash
.venv/bin/python -m adr.synthetic.generate --config configs/synthetic_mlp.yaml --dry-run
```

This uses an in-process fake LLM: one question, two states, two candidates per
state, four records, no API calls. Fake smoke data is marked in the manifest and
written under a `smoke-*` directory. It is not a useful training dataset.

For an actual **one-question** teacher smoke test (five states, four candidates),
explicitly opt into calls after configuring the provider/model and credentials:

```bash
.venv/bin/python -m adr.synthetic.generate --config configs/synthetic_mlp.yaml --questions 1 --allow-api
```

The pilot config targets 20 questions / 400 records. There is one request per
question and at most three attempts per question, not one request per candidate.
Provider/model/output-token settings are configurable. JSON mode is passed through
the existing LLM abstraction; local Pydantic validation is authoritative. Unsupported
provider options, malformed JSON, or invalid examples exhaust bounded retries and
fail without silently accepting bad records. API requests are never enabled by default.

## Checkpointing and resume

Start a fresh pilot with:

```bash
python -m adr.synthetic.generate --config configs/synthetic_mlp.yaml --questions 20 --allow-api
```

The generation directory is printed before teacher requests begin. After each
accepted question, its records are saved along with all earlier accepted questions.
`examples.jsonl` and `manifest.json` retain their existing formats; the manifest's
summary describes the currently saved subset, while `config.questions` remains the
target count. A partial run is not a completed pilot even if its saved subset is valid.
An additional `checkpoint.json` is atomically replaced first and is the authoritative
resume state. If interrupted while updating the canonical files, resume repairs them
from this checkpoint. Do not edit checkpoint files or run multiple writers against
the same generation directory.

Resume using the exact directory printed by the failed run:

```bash
python -m adr.synthetic.generate --config configs/synthetic_mlp.yaml --questions 20 --allow-api --resume data/synthetic/v1/generation-<timestamp>
```

Resume requires the same sanitized generation configuration, prompt/schema versions,
and held-out registry contents. Credentials remain in the environment, never in the
checkpoint. Saved examples are revalidated, including held-out and cross-question
duplicate checks. Completed questions are not regenerated; question IDs and topics
continue at the next unfinished index. That unfinished question receives the usual
maximum of three attempts for this invocation, retaining prior recorded attempt
history. A completed run resumes without constructing a teacher client or making
requests. Without `--resume`, a command always starts a new run.

Only runs created with checkpoint support can be resumed this way. Dry runs can
also be resumed by passing `--dry-run --resume <smoke-directory>`.

## Held-out isolation

Generation fails closed if the separate registry is missing/empty. Build it once:

```bash
.venv/bin/python -m adr.synthetic.validate --import-zip /Users/aadyaagarwal/Downloads/drb-baseline-5q-gpt5mini-20260922.zip --heldout data/heldout/drb_baseline.json
```

Import reads only `queries/*.json`, never trajectory/report outputs. The supplied
archive has inconsistent subsets, so the registry conservatively includes all eight
unique IDs (51–57 and 59), not just 51–55. The import refuses to overwrite a registry.
The registry is ignored and separate from `data/synthetic`; its contents never enter
teacher messages or retry feedback. It is consulted locally for ID, normalized-exact,
sequence-similarity (0.85), and token-Jaccard (0.8) overlap rejection. These checks
cannot recognize every semantic paraphrase. Human review of generated questions and
marginal-value labels is still needed; the LLM must not be shown reference questions
as demonstrations or source material.

## Record contract

Teacher output is `TeacherTrajectory`: an original question, ordered states, and
candidate alternatives per state. IDs for question/trajectory/example are assigned
locally. Each state contains existing evidence, cumulative executed subqueries,
branches, and a bounded budget. No independently written context summary or numerical
features are accepted from the teacher. All models forbid extra fields.

Each compiled `SyntheticExample` has:

- grouping IDs: `example_id`, `question_id`, `trajectory_id`, `state_id`, `candidate_id`;
- semantic inputs: `original_question`, `context_before`, `candidate_subquery`;
- structured `state` and `numerical_features` (`names`, `values`, placeholder status);
- supervision: `label`, `decision_category`, `reason`;
- separate `provenance`: teacher identity, versions, timestamp, state index, pair ID.

`context_before` is sorted-key compact JSON of **state only**. Historical list order
is preserved. Candidate, label, reason, and provenance are not serialized into it.
Candidates within a state are alternatives sharing one snapshot, not sequential
executions. Only a later state may include evidence from a previously executed
candidate. Strict fields prevent structural outcome leakage; free-text evidence can
still contain inappropriate content, so schema validation is not a semantic guarantee.

## Feature semantics (canonical order: schema.FEATURE_NAMES)

`derive_features` delegates to `ResearchState.compact_stats()` and `Budget`:

1. `n_evidence`: all evidence items, including pruned items.
2. `n_retained`: retained items.
3. `n_pruned`: non-retained items (v1 has no supersession field).
4. `n_open_branches`: open or active branches.
5. `n_branches`: all branches.
6. `mean_score`: **unavailable, fixed zero placeholder**, matching the adapter;
   teacher-supplied relevance scores are forbidden. Explicit status accompanies vector.
7. `mean_chars`: integer mean retained body length, each body capped at 2,000 chars.
8. `query_term_coverage`: mean per-retained-item lexical overlap with original-question
   terms, using existing `[a-z0-9]{3,}` tokenization and title + first 400 body chars;
   not semantic coverage or union coverage.
9. `unique_urls`: distinct exact URL strings among retained items.
10. `step_frac`: prior decisions / maximum decisions, with existing rounding/clamping.
11. `token_frac`: prior input+output tokens / maximum cumulative tokens.
12–15. `remaining_steps`, `remaining_searches`, `remaining_reads`, `remaining_tokens`:
   maxima minus prior usage. Search/read units are tool calls. All maxima are positive;
   usage must stay within bounds and increase monotonically between snapshots.

The synthetic default 50,000 is a **cumulative token budget**, not a claim about the
baseline context window. The ZIP used a 400,000 cumulative budget with enforcement
disabled. Unlike its adapter's constant search/read counters, synthetic counters are
explicitly supplied as prior usage and validated. Counters describe idealized state,
not measured live execution. All vector entries are deterministic; mean_score is the
only intentionally unavailable quantity. A later change must version these semantics.

## Validation and outputs

```bash
.venv/bin/python -m adr.synthetic.validate --examples data/synthetic/v1/<run>/examples.jsonl --heldout data/heldout/drb_baseline.json
.venv/bin/python -m pytest tests/test_synthetic.py
```

Validation checks schema/extra fields, derived context/features, IDs, duplicates,
question/trajectory consistency, successive history/evidence/budget consistency,
label/category combinations, pair direction (early EXECUTE, later SKIP), held-out
and cross-question lexical duplication. It reports class balance, states/candidates,
and pair counts; semantic equivalence of nonliteral pairs requires review. The
approximate balance band is 35–65% EXECUTE (a warning outside the band).

Generation enforces configured state/candidate counts. Output is versioned under
`data/synthetic/<version>/<run>/examples.jsonl` plus `manifest.json`. The manifest
records sanitized config, actual teacher, timestamp, schema/prompt versions, ordered
features, response hashes, token usage, held-out-registry hash, and audit summary.
No raw API response objects or secrets are persisted. Generation is stochastic;
provenance supports auditing settings, not guaranteed identical regeneration.
All generated artifacts and the held-out registry are ignored by Git. Future splits
must group by question, including all linked states/trajectories and pairs.
