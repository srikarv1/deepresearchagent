# Evidence replay, Phase 1

Standalone experiment; no RL and no changes to existing agents or baselines.
Run from the repository root. Local planning never constructs a model client.

## Offline validation

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/adr/experiments/evidence_replay/test_replay.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m adr.experiments.evidence_replay replay \
  --trajectory runs/_gpt_researcher_real_source_v2_raw/trajectory_254955f96594.json \
  --selector topk --budget 50000 \
  --output runs/evidence-replay-phase1/offline-plan-001
```

Every output directory must be new; existing paths are refused. The selected
source has 327 page records, all joined to saved 1536-dimensional embeddings.
These embeddings are page-level vectors, not independently embedded passages.

The tokenizer uses exact `o200k_base` BPE counts, including ID/source wrappers and
separators. It reads hash-verified encoding data bundled in installed LiteLLM,
without downloading or writing a tokenizer cache. Missing local data fails closed.
The budget covers formatted evidence context, not the full chat request. Prompt,
question and chat framing consume additional input tokens. Configure the encoding
explicitly if using a model with another tokenizer.

## One live synthesis, only after API approval

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m adr.experiments.evidence_replay replay \
  --trajectory runs/_gpt_researcher_real_source_v2_raw/trajectory_254955f96594.json \
  --selector topk --budget 50000 --model gpt-5-mini \
  --max-completion-tokens 16384 --reasoning-effort low \
  --output runs/evidence-replay-phase1/smoke-api-003 --allow-api
```

Credentials come from the process environment: `OPENAI_API_KEY` and optionally
`OPENAI_BASE_URL`. This module does not automatically load `.env` or select Azure
deployments. No credentials or endpoint values are written to artifacts.

The original 8,192-token smoke returned usage with all completion tokens spent on
reasoning and no visible report. The CLI default remains 8,192 for compatibility;
the command above explicitly requests low reasoning effort and a 16,384-token
completion allowance. This reduces the observed risk but does not guarantee a
finished report. No automatic retry or output-budget increase occurs.

This invokes GPT Researcher's actual `generate_report` and deep report prompt,
with a fixed evidence-only system instruction. No GPTResearcher instance,
research controller, browser, retriever, embeddings client, images, summaries,
learnings or existing report is supplied. The fork's transport is replaced within
this standalone process by one nonstreaming OpenAI-compatible request. SDK retries
and the fork's second-request fallback are disabled. Do not embed this adapter in
a concurrent service: its temporary transport/date patches are process-global.

The source run's last round date is frozen into the report prompt. Selection,
context and prompt assembly are deterministic; remote model output is not promised
to be deterministic. No word-limit trimming is applied. A manually supplied
`--selected-ids ids.json` (JSON string list) overrides selection and is rejected
if the complete context exceeds the cap. Empty selections can be planned but
cannot invoke synthesis. Source text may itself include previously scraped page
summaries or hostile instructions; it is retained as the recorded page content,
not augmented with orchestration summaries. The system prompt treats it as data.

Context is serialized as exact UTF-8 bytes with no newline translation. Synthesis
reads it once as bytes, validates the hash and planned text, decodes UTF-8, counts
tokens on that decoded string, and passes the same string to report generation.
Embedded CR, CRLF, and LF line endings are preserved rather than normalized.

Selectors:

- `random`: canonical ID order, seeded local shuffle, feasible whole-page first fit.
- `threshold`: root-query cosine >= `--threshold` (default .35), ordered by first
  retrieval round then ID, feasible whole-page first fit. No fallback when empty.
- `topk`: descending root-query cosine, ID tie break, feasible whole-page first fit.
  Here k is the number fitting the token cap, not a separately fixed count.

All skip nonfitting pages and continue scanning; none truncates text or permits
an oversized first item. Output context uses retrieval-round/ID order for all
selectors. Equal caps can yield different actual utilization. The threshold
baseline differs from the legacy subquery-level chunk EmbeddingsFilter baseline.

Artifacts:

- `plan.json`: full boolean evidence mask, ordered selected IDs, exact context
  tokens, selection configuration, frozen date, selected source metadata, input
  file hashes, code hash and tokenizer/package versions.
- `context.txt`: exact formatted selected evidence.
- `request.json`: complete model request messages, model/output limit and text
  token count excluding chat framing; written before a synthesis request.
- `report.md`: fresh generated report, on success only.
- `synthesis.json`: completion/failure status, actual API-reported usage (including
  provider details when supplied), latency, request count, model, code hashes,
  frozen date and report hash. Interrupted runs retain the `started` reservation.
- `response.json`: allowlisted response metadata, usage and classified failure
  for received SDK responses. Includes finish reason, response/model identifiers,
  content length, refusal presence and tool-call count; no raw provider bodies,
  refusal text, credentials, or headers. Transport errors instead record exception
  type, HTTP status and request ID in `synthesis.json`; usage remains unknown if no
  response arrived. `partial_report.md` preserves nonempty incomplete output and
  is never treated as a completed `report.md`.

A synthesis failure preserves the plan and request audit and cannot be rerun into
the same directory. One successful synthesis uses one API request; at a 50K cap,
allow about 50K input tokens plus prompt/question overhead and up to 8,192 output
budget tokens (which can include reasoning tokens). No retrieval or embedding
requests occur. Actual billing comes from the provider usage record.

For the recommended command above, the completion allowance is 16,384 instead of
8,192; both visible output and internal reasoning consume that allowance. The
request uses Chat Completions `max_completion_tokens` and optionally
`reasoning_effort`, with no unsupported temperature override. Only the typed
assistant message's `content` is report text; reasoning usage is not hidden report
text that should be extracted. Responses ending in `length`, refusal/filtering,
tool calls, missing choices, or blank content fail explicitly with diagnostics.

## Optional official evaluation

The real-source development smoke above is **not eligible** for official RACE:
its question has no exact DRB prompt/reference/rubric match. The evaluation command
refuses it before constructing clients, creating output, or reporting scores.
No development benchmark scores are fabricated.

For a replay whose source question exactly matches a locally covered DRB question:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m adr.experiments.evidence_replay evaluate \
  --replay runs/evidence-replay-phase1/benchmark-smoke-api-001 \
  --output runs/evidence-replay-phase1/benchmark-eval-001 --allow-api
```

Add `--fact` to request FACT citation evaluation, including external source
scraping. Uses the existing `run_deep_research_bench` adapter and its environment
configuration (`LLM_BACKEND`, appropriate API key, `RACE_MODEL`, `FACT_MODEL`, and
`JINA_API_KEY` for FACT). It stages a uniquely named report in the benchmark repo,
so it needs write access there; it does not overwrite existing baseline reports.
Evaluation incurs multiple judge requests, upstream retries, and optional scraping;
it is not included in the single-call synthesis allowance. It is never invoked
automatically by replay. Outputs record reference/rubric hashes and the existing
adapter's complete result, including missing-key/skipped/failure details.

## Before RL

Choose question-family train/validation/test splits, reward definition, budget
comparison protocol and call limits. Custom development questions require a
separately validated reference-free reward or authored references; cached reports
provide no alternate-mask rewards. Add policy features, sampling/log probabilities,
variance-reduction baseline, optimizer and checkpointing in a later phase. This
experiment holds retrieval fixed and cannot estimate savings or quality changes
from altering research-time retrieval decisions.
