# Phase 5 frozen downstream pilot

This is a **frozen-retrieval, synthesis-only exploratory evaluation**, with two
research queries as the evaluation units. It does not reproduce a full Table 4
retrieval run and does not support significance claims. Trained REINFORCE did not
outperform its untrained control offline. Shortlist filtering and its lower-cost
operating point must not be reported as RL learning.

## Frozen protocol

Official DRB questions 52 and 54 use the manually reviewed saved trajectories
`37aace7baa69` and `540e3cd646f4` from the adaptive-depth raw source directory.
The official query, cleaned reference, and rubric must match by exact prompt;
query IDs are checked independently. Snapshots and hashes are saved, and the
current official files must still match before execution.

Sampling seed is 1000. The bounded shortlist has at most 24 pages and is built at
50,000 formatted context tokens. Its seed-0 initial policy samples the frozen
natural operating-point mask. The threshold baseline uses 0.35, chronological
first-fit overflow. The original Phase 4 seed-0 epoch-20 actor samples in cosine
order. Both are capped at the shortlist's actual context count for that query.
Whole pages make exact matching impossible; the manifest records the residual.
The actor was trained at 50K and is being applied at the lower matched budget.
The diagnostic shortlist-trained checkpoint is deliberately not substituted.

Checkpoint paths and SHA256 identities, all six full masks, source provenance,
selected IDs, context bytes, exact tokenizer counts, synthesis dates, reference
snapshots, and implementation hashes are frozen before any network operation.
Changing frozen implementation files invalidates execution; do not overwrite an
old manifest or remove its hash checks. Freeze into a fresh directory if a new
protocol/version is explicitly approved.

Synthesis is fixed to GPT-5-mini, low reasoning, max 16,384 completion tokens.
The evidence-only existing GPT Researcher adapter performs no retrieval, browsing,
embedding or judge calls. Context tokens include source formatting, but exclude
prompt/question overhead; API usage is reported separately (including reasoning).

## Running after explicit paid approval

From the repository root, the already frozen pilot is
`runs/evidence-replay-phase5/pilot-001/manifest.json`. It contains no paid results.
Do not rerun freeze into this directory.

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5 synthesize \
  --manifest runs/evidence-replay-phase5/pilot-001/manifest.json \
  --max-calls 6 --allow-api

.venv/bin/python -m adr.experiments.evidence_replay.phase5 judge \
  --manifest runs/evidence-replay-phase5/pilot-001/manifest.json \
  --max-calls 12 --allow-api

.venv/bin/python -m adr.experiments.evidence_replay.phase5 summarize \
  --manifest runs/evidence-replay-phase5/pilot-001/manifest.json
```

For an approved new freeze (offline only), use a fresh output directory:

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5 freeze \
  --output runs/evidence-replay-phase5/pilot-002
```

No keys are printed or loaded from new files. Paid commands require
`OPENAI_API_KEY` in their process environment, optionally `OPENAI_BASE_URL` for an
existing compatible endpoint. The driver does not automatically source `.env`.
Both model names must be available through that endpoint. No availability probe,
model fallback, FACT, scraping or additional calls are permitted. GPT-5.5 access
has not been verified offline; if unavailable, stop and obtain approval for a
revised judge protocol rather than changing the model silently.

## Call limits and failures

Commands take exclusive, persistent stage locks. Each possible network attempt
has a persistent exclusive reservation; failed/interrupted calls consume a slot.
A stage cannot automatically restart, retry or run concurrently. Do not delete
stage locks/reservations to repeat paid calls. Stop on the first request or scoring
failure. Previously completed artifacts remain available to `summarize`.
The six-synthesis/twelve-judge maxima cannot be raised through CLI arguments.
OpenAI SDK retries are disabled. Budget cap exhaustion also stops the stage.

Judging uses the official RACE English cleaning prompt and validation, then its
`process_single_item(max_retries=1)` scoring/parsing/weighted calculation. It
bypasses the cleaner retry/chunk-recursion orchestrator deliberately: one direct
cleaning call and one scoring call per completed report. Malformed/incomplete
criterion scores are rejected before upstream permissive zero defaults. Both
`--skip-cleaning` and `--skip_cleaning` are rejected explicitly. There is no FACT
and thus no citation-correctness measurement.

Judge requests contain only the report (cleaning), or the official question,
cleaned target, reference, and rubric (scoring). Opaque report aliases and their
method mapping are local; method identities, context budgets, proxy scores and
checkpoint metadata are never passed to the judge. Content can still reveal
style or evidence choices; this is identity blinding, not a guarantee against
inference from report content. Report order is fixed by opaque aliases.

Allowlisted response metadata, finish reasons, usage, response text (including
partial text), latency and safe error classes/status/request IDs are saved.
Provider exception bodies and credentials are not saved. Crashes may leave a
reservation with unknown usage; absent usage means unknown, never zero billing.
Summary records all six rows, including missing/failed evaluations with null
quality. Available-case means include their sample count; paired means use only
queries with all three successful scores. API-token totals include reported
usage from failures as well as successes. Do not treat failed scores as zeros or
claim a winner from incomplete paired comparisons. The frozen shortlist's
natural lower-cost point is the matched comparison cap; a separate 50K threshold
quality comparison is not part of this six-report protocol.

Expected maximum is six synthesis requests and twelve judge requests. The audit
estimated approximately 243K synthesis input tokens, 30–50K synthesis output,
140–180K judge input, and 54–108K judge output, with reasoning included in output
billing. Completion ceilings are 98,304 synthesis and 196,608 judge tokens.
Allow roughly an hour; no reports or scores are cached for these exact masks.
Costs/model availability require account-specific confirmation; no dollar estimate
or downstream-quality improvement is asserted by offline validation.

## Offline validation

```sh
.venv/bin/python -m pytest src/adr/experiments/evidence_replay -q
```

Tests prohibit network connections, exercise real official references and frozen
masks, mock both transport and pipeline failures, verify blinding, criterion
coverage, hard caps, no retries, usage preservation and null-score summaries.
