# Phase 5 extension: q51, q53, q55

This is a **frozen-retrieval, synthesis-only evaluation with the GPT-5-mini RACE
judge variant**. It is not fully comparable with historical Table 4 retrieval
runs. No FACT or citation-accuracy evaluation is performed. No statistical
significance or RL training benefit is asserted. Offline REINFORCE training did
not outperform its untrained control consistently.

The extension driver is isolated in `phase5_extension.py`. Original Phase 5 and
recovery code are unchanged, preserving their frozen hashes. The driver reads
pilot-001, including the six completed GPT-5-mini RACE results, and never writes
inside it. Freeze snapshots every original pilot file; validation compares the
full original inventory before execution and aggregation. Changing any original
artifact, or adding an original pilot file, causes a refusal rather than silently
mixing versions. Original synthesis/judge ledgers remain entirely separate.

## Prespecified evidence sources and selectors

| DRB query | Saved trajectory | Provenance limitation |
|---|---|---|
| 51 | `2bd39c9137ad` | Reviewed sequential research pool; localized source failures possible |
| 53 | `1c1b0cfed811` | Reviewed sequential research pool; localized source failures possible |
| 55 | `716465858aea` | Recovery-run pool; not the same retrieval protocol as original historical Table 4 |

All are under `runs/_gpt_researcher_adaptive_depth_raw/`, with companion embedding
NPZ files. Official DRB prompt, reference and rubric matches are verified exactly,
including query ID. No retrieved evidence, references or masks are inferred from
report outcomes. No retrieval, embedding, Tavily, training or reward tuning occurs.

Selectors reproduce pilot-001: threshold .35 and chronological whole-page
first-fit; bounded shortlist k<=24 with seed-0 initial policy; original Phase 4
seed-0 epoch-20 actor with cosine candidate order. Sampling seed is 1000. The
shortlist is built at 50K, then sampled at its natural lower-cost point. The
other two methods are capped at its actual formatted context count. Whole-page
selection leaves small actual-token differences; no trimming is applied. The
actor was trained at 50K, which differs from some inference caps.

The nine full masks, selected IDs, contexts, SHA256 hashes, checkpoints, tokenizer
counts, selector identities, source provenance and references are frozen before
any paid execution. Untrained cosine-order control masks use the saved Phase 4
`seed-0/untrained.pt`, the same ordering, seed and cap. They are diagnostic only:
no fourth report is generated. The same protocol must exactly reproduce q52/q54
masks before freeze succeeds. Comparing trained actor against the shortlist
alone confounds training with candidate filtering/order. Equality with the
actor's own untrained control means report differences cannot establish a
training-induced evidence-selection improvement.

## Commands after paid approval

Run from the repository root. The extension manifest has already been frozen at
`runs/evidence-replay-phase5/pilot-002-extension/manifest.json`. Do not overwrite it.

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5_extension synthesize \
  --manifest runs/evidence-replay-phase5/pilot-002-extension/manifest.json \
  --max-calls 9 --allow-api

.venv/bin/python -m adr.experiments.evidence_replay.phase5_extension judge \
  --manifest runs/evidence-replay-phase5/pilot-002-extension/manifest.json \
  --max-calls 18 --allow-api

.venv/bin/python -m adr.experiments.evidence_replay.phase5_extension summarize \
  --manifest runs/evidence-replay-phase5/pilot-002-extension/manifest.json \
  --output runs/evidence-replay-phase5/pilot-002-extension/combined-final-001
```

The summary output must be fresh and outside pilot-001. It reads original and
extension results directly, generating JSON plus per-query CSV, without calling
models or modifying either input batch. An offline preview can be generated
before new reports exist; all nine pending quality values and the five-query
macros remain null. Complete-paired-query averages are explicitly distinguished
from five-query averages. Do not present the existing two-query average as a
five-query result. Unknown API usage is not assumed to be zero billing.

Offline validation command:

```sh
.venv/bin/python -m adr.experiments.evidence_replay.phase5_extension verify \
  --manifest runs/evidence-replay-phase5/pilot-002-extension/manifest.json

.venv/bin/python -m pytest src/adr/experiments/evidence_replay -q
```

## Paid execution safeguards

Synthesis uses GPT-5-mini, low reasoning and 16,384 completion allowance.
RACE cleaning uses GPT-5-mini low; scoring uses GPT-5-mini medium, each at
16,384 allowance. The official RACE cleaning prompt and scoring/calculation
are reused through the frozen original adapters. The cleaning retry/chunk
orchestrator is bypassed. Method identities, checkpoint metadata, costs and
proxy scores never enter judge prompts; opaque aliases determine judging order.

The executing terminal must contain its existing `OPENAI_API_KEY` and intended
`OPENAI_BASE_URL` if custom. The endpoint fingerprint must match the successful
pilot-001 judge endpoint; no URL/key is printed. No `.env` is implicitly loaded.
There are no model overrides, fallbacks, access probes or automatic retries.
Model access is not reverified during offline preparation.

Strict maxima are **9 synthesis requests and 18 judge requests**. Exclusive,
persistent stage locks prohibit rerunning either stage; persistent reservations
count failed/interrupted attempts. Stop on the first failure. Do not delete a
lock or reservation to retry. A later recovery would require a new explicit
protocol/approval. All nine intact synthesis reports are required before judging.
Completed/failed response metadata, partial text, usage, finish reason, latency
and safe provider diagnostics are retained. Failed quality is null, never zero.

The synthesis allowance ceiling is 147,456 completion tokens; the judging ceiling
is 294,912. These are allowances, not usage predictions. Token estimates should
use frozen context counts, actual pilot-001 report lengths and observed RACE
usage/latency; reference lengths differ by query. The saved offline cost audit
provides the measured inputs and arithmetic without making external calls.

## Combined reporting

The summary has 15 query/method rows and five RACE dimensions, including overall.
Five-query macro averages weight queries equally. Mean selected context tokens,
reported total synthesis API tokens, total/mean synthesis latency and unknown
usage counts are reported by method. Missing results, new failures and the
historical GPT-5.5 404 are explicit. Mask equality is reported for each of the
five queries, both against the shortlist and the actor's own untrained control.
Single report/judge draws, non-pristine final test questions, heterogeneous saved
retrieval pools and absent FACT constrain interpretation. No mask was chosen
based on these downstream quality scores.
