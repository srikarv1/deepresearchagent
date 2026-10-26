# Explicit Phase 5 RACE recovery

The original six synthesis reports, masks, manifest, failed GPT-5.5 call,
judging directory, and legacy stage lock are preserved. The original
`phase5.py` is unchanged so its frozen implementation hashes still validate.
Recovery is an isolated driver, `phase5_recovery`, with an immutable sidecar
protocol amendment. It does not rerun synthesis or change any evidence mask.

## Endpoint/model diagnosis

The legacy pilot explicitly used `gpt-5.5`; its first cleaning request returned
HTTP 404 with no completion or usage. The original diagnostics did not preserve
the provider error code/body, so offline evidence cannot distinguish a missing
model/deployment from an incorrect route. This agent process has no configured
`OPENAI_BASE_URL` or `OPENAI_API_KEY`; the prior user terminal environment is not
available here. No network model-list or generation probe has been performed.

`configs/adaptive_depth_recovery_provenance.json` records `gpt-5-mini` for both
RACE and FACT in another existing evaluation configuration. The six successful
pilot synthesis calls also demonstrate that `gpt-5-mini` was accepted through
the prior synthesis environment. This supports it as a recovery candidate, but
is not a new access test or proof about the currently configured endpoint.
On an Azure-compatible `/openai/v1/` endpoint, the model argument must be the
actual deployment name; do not assume GPT-5.5 is deployed under that name.

The recovery candidate is explicitly **GPT-5-mini RACE**, not GPT-5.5 RACE.
Official RACE cleaning/scoring prompts and weighted score calculation are reused.
Cleaning uses low reasoning; scoring uses medium; both have a 16,384 completion
allowance. One frozen judge model is used for all six reports and both stages.
No FACT, synthesis, retraining or API access checks are performed by preparation.
The frozen-retrieval/synthesis-only label and offline negative RL finding remain.

## Commands

The following first command is offline. `--judge-model` is the explicit override
and is required when preparing a sidecar. Do not edit the original manifest or
remove old reservations/locks to recover. Use a fresh recovery directory.

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5_recovery prepare \
  --manifest runs/evidence-replay-phase5/pilot-001/manifest.json \
  --output runs/evidence-replay-phase5/pilot-001/judge-recovery-001 \
  --judge-model gpt-5-mini \
  --max-calls 13 \
  --retry-failed report-4a2ab9b989cf:clean

.venv/bin/python -m adr.experiments.evidence_replay.phase5_recovery verify \
  --config runs/evidence-replay-phase5/pilot-001/judge-recovery-001/config.json
```

If `judge-recovery-001/config.json` already exists from implementation validation,
skip preparation and use the verified configuration. No paid recovery has been
executed during implementation. A different deployment name requires a new
explicitly prepared sidecar before any stage succeeds; never silently change it.
Once any cleaning/scoring stage succeeds, a model change is refused to prevent
mixing judges. The original failed GPT-5.5 request produced no text or scores,
so it does not contaminate a consistent GPT-5-mini evaluation.

After explicit paid-execution approval, run from the terminal containing the
existing valid `OPENAI_API_KEY` and intended `OPENAI_BASE_URL` (if using a custom
endpoint). The driver does not load `.env` or create credentials, and does not
print keys or endpoint URLs. With no custom URL, the SDK uses the standard
OpenAI endpoint. A fingerprint of the endpoint is pinned on first execution;
subsequent recovery invocations must use that same endpoint.

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5_recovery judge \
  --config runs/evidence-replay-phase5/pilot-001/judge-recovery-001/config.json \
  --allow-api

.venv/bin/python -m adr.experiments.evidence_replay.phase5_recovery summarize \
  --config runs/evidence-replay-phase5/pilot-001/judge-recovery-001/config.json
```

The original `phase5 summarize` sees only legacy judging results; use the
recovery summarizer for combined historical/recovery accounting and scores.
All missing/failed quality values remain null, and paired means require all
methods to have successful scores on a query. Unknown usage is counted
separately, not treated as confirmed zero billing. The model override is labeled
in the summary; no significance or improved report-quality claim is implied.

## Attempt accounting and interruption handling

The maximum is **13 cumulative judge requests**, including the one original
404. Twelve new requests are needed (six cleaning plus six scoring), leaving no
allowance for another failed call if the complete six-report protocol is to fit.
The original manifest's limit remains 12; the sidecar explicitly records the
approved amendment to 13 without overwriting it. Preparation rejects a budget
that cannot cover the remaining stages. Runtime reservations use the same
original `judge-calls/` directory, with unique recovery identities. Counts never
reset across process restarts or sidecar directories. Failed and interrupted
reservations consume allowance. The enclosing cross-recovery lock serializes
all attempts for the original pilot; no original legacy lock is removed.

A new failure stops immediately. Reinvoking the same sidecar refuses its failed
or interrupted attempt rather than silently retrying. Successful final results
are skipped. If interrupted after a successful cleaning or scoring response,
that response is reused locally and scoring/parsing can finish without another
API call. New sidecars must explicitly acknowledge failed/unknown stage IDs
using `--retry-failed`; they also preserve completed results and enforce the
same model and cumulative cap. Unknown/unrecognized historical reservations
require manual audit rather than guessing.

New files live under `judge-recovery-001/` (configuration, call diagnostics,
results, endpoint fingerprint, timestamped summaries). New reservation files
are appended under `judge-calls/`; original files are never overwritten. The
transient `judge-recovery-active.lock` is removed on normal exit; after a hard
process kill, stop for a human lock/attempt audit before recovery. SDK and RACE
retries remain disabled. The cleaner never recursively splits a report.

New diagnostics preserve known categorical provider error codes/types, HTTP
status, request ID, usage, finish reason, response text and latency. Free-text
provider bodies/messages and unknown code/type values are redacted to avoid
persisting secrets. The original 404 cannot be retroactively enriched.

## Offline tests

```sh
.venv/bin/python -m pytest src/adr/experiments/evidence_replay -q
```

Tests deny network, mock responses, and cover explicit model overrides,
404 categorical diagnostics, preservation of original failures, historical
call accounting, hard cap 13, interrupted-stage reuse, concurrent recovery,
model/endpoint consistency, null failure scores, and prevention of duplicate
successful API calls across restarts/sidecars.
