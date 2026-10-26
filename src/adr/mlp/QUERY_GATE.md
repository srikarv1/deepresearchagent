# Frozen MLP #1: inner web-query gate

This is an experimental integration, not a real-DRB evaluation. No training,
threshold tuning, training-cache writes, or live API calls are part of its tests.

## Exact hook and scope

In the current fork, `ResearchConductor._get_context_by_web_search` calls
`plan_research`, appends the branch query unless this is a subtopic report, then
normally gathers `_process_sub_query` calls. With an active task-local gate, that
same dispatch passes through `QueryGate.dispatch`. It snapshots once and completes
ALL predictions before starting ANY `_process_sub_query`. SKIP never calls that
method. EXECUTE calls it unchanged. Its `_scrape_data_by_urls` hook records actual
entry to retrieval before `_search_relevant_source_urls` executes. Cache hits can
therefore have `retrieval_invoked=true` without incurring a new provider search.

The outer `DeepResearchSkill.generate_search_queries` generates research branches;
those are NOT the gated candidates. The candidate is each exact inner planned
query (including the appended branch query). Planning-time searches in
`generate_research_plan` and `plan_research` occur before this gate, remain ungated,
and are counted as prior search attempts/history. Existing post-retrieval
EmbeddingsFilter and orchestration policy decisions are unchanged.

MCP, vector-store and provided-source-URL modes are explicitly unsupported when
this gate is enabled. Their retrieval boundaries/accounting differ. The current
integration targets the configured standard web/Tavily path. Unsupported modes
fail instead of silently bypassing the gate.

## Frozen inference

`inference.py` checks training-manifest checksums for the exact checkpoint/config
and preparation contract/scaler. It requires model dimensions 4623/256/64/1,
dropout .2, threshold .5, seed 42, embedding model `text-embedding-3-small`, 1536
embedding dimensions, correct labels, text-field order and feature order. It
loads CPU weights strictly, freezes them, uses eval mode and one Torch thread.
Endpoint identity must match the hashed endpoint of the frozen preparation.
No endpoint or credential is hard-coded or written into gate records.

Inputs are independently embedded original top-level question, serialized
pre-batch ResearchSnapshot, and individual candidate. Metadata is never appended.
The question is separate from context JSON, exactly as in training. The snapshot
uses the existing `serialize_context` and `derive_features`; normalization uses
`normalization.json` from preparation, never a new fit. X is question[1536] +
context[1536] + candidate[1536] + normalized features[15].

The existing `OpenAICompatLLM.embed` handles Azure `OPENAI_BASE_URL` (including
`/openai/v1/`) and `OPENAI_API_KEY`. No second provider implementation is added.
An instance-local bounded cache holds up to 1024 exact strings; identical strings
are deduplicated across fields/candidates and concurrent requests are serialized.
Eviction can cause a later re-embedding; training caches remain untouched. Requests retain at most 16 strings per batch. Live inputs are checked against
8192 `cl100k_base` tokens per string, using `encode_ordinary`. The locally installed
OpenAI SDK 2.54.0 (`resources/embeddings.py`, `types/embedding_create_params.py`)
documents 8192 tokens; tiktoken 0.14.0 `model.py` maps `text-embedding-3-small` to
`cl100k_base`. LangChain's 8191-token chunk default is slightly conservative, not
an 8191-byte model limit. Azure deployment-specific overrides are not verified by
these local sources. Tokenizer data must already be cached and match the official
SHA256; inference fails locally rather than downloading missing tokenizer data.
Frozen preparation artifacts and their historical byte guard are unchanged.

Contexts within the token bound retain their exact original serialization. Only
oversized contexts use the existing ordered evidence table and deterministic
body-prefix allocation; admission now counts tokens including JSON escaping.
Non-body fields are preserved. Prefix search is deterministic and size-checked,
but BPE counts are not monotone, so maximal retained content is not guaranteed.
Metadata-only token overflow remains a hard failure, before embedding or retrieval.

`state_diagnostics` precedes snapshot construction; `context_diagnostics` records
original/metadata bytes and tokens, counts, budget, features and candidate IDs
before the overflow check. Unavailable pre-construction values are null. These
new diagnostics contain no evidence bodies or arbitrary exception text. Research
failure accounting is copied to the harness meter and persisted under
`partial_research_accounting`; it excludes MLP embedding overhead and is partial,
not a completed-run cost. Search/read counts there come from the gate hooks.

## Snapshot policy and live-state differences

Harness concurrency and outer deep-research concurrency must both be 1, because
GR token accounting is process-global. Inner candidate retrieval is still
concurrent. All candidates share the same immutable pre-batch snapshot and budget,
and inference completes before the retrieval gather begins. Query history commits
in proposal order after the gather, never completion order. A second overlapping
batch fails. No sibling result or current retrieval cost enters this batch's X.

* Evidence comes only from `trajectory_logger._global_evidence`, which the fork
  commits after a completed outer research layer. It contains page-level joined
  chunks, retention flags and URLs. We intentionally exclude in-flight global
  embedding-cache records, raw planner search responses, and uncommitted sibling
  evidence. Thus history/cost may advance within a layer while evidence does not.
  Evidence ordering is by committed retrieval round then ID, so within-round
  completion order does not affect serialization. Titles are unavailable here
  and remain empty. Actual collected text is used;
  no LLM-generated state summaries or fabricated evidence are introduced.
* The live plan's follow-up questions are open information needs. Previously
  recorded frontier statuses map open→open, completed→done, pruned→dropped. The
  currently dispatched outer query is active. Exact goal duplicates are merged;
  semantic overlap is not inferred. Plan needs and outer goals are not necessarily
  a one-to-one mapping, and missing completion signals remain open/active. Merely
  executing a query does not mark a need solved. Branch counts therefore have a
  different operational granularity from teacher-authored synthetic branches.
* Previously invoked planning searches and completed inner retrieval queries form
  cumulative history. Duplicate attempts remain visible. Inner queries served
  from retrieval caches count as retrieval history, not new provider searches.
* Prior steps = completed inner EXECUTE/SKIP decisions + completed outer
  orchestration rounds. This makes explicit the live orchestration decisions;
  it differs from synthetic authors' abstract prior-step counts.
* Search usage counts actual provider search dispatch attempts (planning included,
  cache hits excluded), reads count URLs dispatched to the scraper (including
  failed attempts, excluding cached/prefetched pages). Provider-internal fetches
  and scraper-internal retries are not individually observable. Reads are per-page
  dispatches, not distinct visited URLs or one count per browse batch.
* Token usage is GR's prior API-reported LLM input+output tokens. The existing GR
  tracker does not include embedding tokens (including gate embedding overhead),
  so this is NOT complete monetary/API-cost accounting. Gate embeddings add cost.
* Maxima come explicitly from the ResearchTask budget, not fabricated live usage.
  GR historically does not enforce these harness caps. The bridge fails if prior
  usage exceeds a maximum, rather than clipping observed usage or extending caps.
  This check is at batch boundaries, not a new hard resource scheduler.
* URL/credential privacy is applied before embedding AND logging: source URLs
  become stable `url-sha256:` identifiers; URLs embedded in free text are similarly
  pseudonymized and known environment credentials/common API-key patterns are
  redacted in the state/history. Original question and current candidate remain
  exact: if either would need privacy redaction, the gate fails rather than silently
  changing the query seen by the model. This preserves URL equality/counts but differs from training's literal
  URLs and may change body lengths/lexical overlap for URL-bearing evidence. Query
  strings go unchanged to both the MLP and retrieval. This state privacy transform
  is explicit, not a claim of
  identical synthetic/live distributions.
* Evidence original bodies/IDs are preserved; only retention may change. An
  observed body rewrite/removal or compressed evidence fails. Compression policies
  therefore require a future explicit semantic design; they are not silently
  converted. Existing post-retrieval policies are not replaced by this gate.

## All 15 numerical features

These are obtained by calling the existing synthetic `derive_features` on that
snapshot, not by implementing a second feature definition. `ResearchState` rounds
coverage/fractions as it does in training.

| Order | Feature | Live snapshot mapping |
|---|---|---|
| 1 | n_evidence | All committed historical page evidence items |
| 2 | n_retained | Items currently marked retained |
| 3 | n_pruned | Items currently not retained |
| 4 | n_open_branches | Branches whose status is open or active |
| 5 | n_branches | All known plan/frontier/active goals, exact-goal deduplicated |
| 6 | mean_score | Existing unavailable-score placeholder 0, as in training |
| 7 | mean_chars | Integer mean retained evidence body length |
| 8 | query_term_coverage | Mean question-term overlap with retained title + first 400 body chars, existing regex/rounding |
| 9 | unique_urls | Distinct retained source identifiers; URL hashing preserves equality |
| 10 | step_frac | Prior steps / configured max_steps, existing capped/rounded formula |
| 11 | token_frac | Prior tracked tokens / configured max_tokens, same formula |
| 12 | remaining_steps | max_steps − prior steps |
| 13 | remaining_searches | max_searches − prior provider search attempts |
| 14 | remaining_reads | max_reads − prior per-page scraper dispatches |
| 15 | remaining_tokens | max_tokens − prior tracked LLM input+output tokens |

These mappings preserve feature computations, but do not certify synthetic/live
semantic equivalence. Distribution shift, particularly branch granularity, stale
committed evidence, and query length/style, must be monitored before evaluation.

## Switch, logs and failure behavior

`configs/agents/gpt_researcher.yaml` defaults to:

```yaml
query_gate:
  enabled: false
  run: runs/mlp/baseline-v1-20260923T044920906450Z
```

False follows the original gather path and imports/initializes no MLP or gate
embedding client. True enables it for all nested inner standard web batches using
a ContextVar; the adapter resets that scope and closes the client after research.

Logs are exclusive-created UTF-8 JSONL at
`<run_dir>/query_gate/<sha256(question_id)[:16]>.jsonl`. No overwrite/append-to-old-run
occurs. `gate_initialized` records artifact hashes/feature order. `snapshot` stores
the exact privacy-transformed context JSON, its SHA-256, original question and
budget. `proposed` records stable question/batch/candidate ID, candidate text, raw
and normalized features, probability, threshold, decision and snapshot reference.
`retrieval_invoked` is emitted at retrieval entry; `finished` records the final
invocation flags, not results. `skipped` explicitly records both invocation flags
false. Logs also include `empty_batch` and safe `gate_error` events. No raw provider
response or exception message is serialized. These privacy guarantees concern the
new gate stream; the fork's pre-existing source/citation/report logs are unchanged.

All-SKIP and zero-candidate batches return empty context deterministically, log
`empty_batch`, and never override the learned threshold. GR's existing outer
result processing may still make LLM calls for an empty context; all-SKIP does not
mean the whole run is free. Gate errors explicitly rethrow through the inner and
outer best-effort research handlers, rather than returning empty context or a
fallback decision. Audit write failures are fatal as well.

## Fork patch, offline checks and live smoke

The locally modified fork starts at commit
`024f3a4822dab2b2d7196a142059a86fcfb996f5`. Since `third_party/*` is ignored, the complete
hook patch is preserved at `scripts/patches/gpt_researcher_query_gate.patch`.
On a fresh matching fork checkout, apply it once from the repository root:

```sh
git -C third_party/gpt-researcher apply --check ../../scripts/patches/gpt_researcher_query_gate.patch
git -C third_party/gpt-researcher apply ../../scripts/patches/gpt_researcher_query_gate.patch
```

It is already applied in this workspace. Do not apply twice or overwrite a dirty
fork. The gate fails if the hook module/version is unavailable.

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mlp_query_gate.py tests/test_gpt_researcher_agent.py -q
```

The offline smoke executes actual vendored dispatch/process/scrape method bodies
with mocked planner/retriever/context-manager collaborators, and a separate case
loads the real frozen checkpoint with mock embeddings. This is not a full imported
GR run: this `.venv` currently lacks the fork's LangChain dependencies. Install the
fork's required dependencies in the intended live environment first; use an
environment that also has ADR and its `mlp` dependencies. No dependency downloads
or live requests were performed for this integration.

For a first authorized LIVE smoke, from the repository root, with existing Azure
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and the configured retriever credential (normally
`TAVILY_API_KEY`) available in the environment or `.env`:

```sh
PYTHONPATH=src .venv/bin/python -m adr.mlp.smoke --config configs/agents/gpt_researcher_mlp_smoke.yaml --allow-api
```

This uses one hard-coded nonbenchmark question about published urban shade-tree
studies, depth=1/breadth=1/outer concurrency=1, existing GPT-5-mini settings and the
fixed MLP checkpoint. It does not read held-out DRB questions. The live fork may
still generate multiple inner queries and perform ordinary planning, evidence
compression, auxiliary embeddings, and reporting. No live call is made without
`--allow-api`. The command has NOT been executed during integration. Its artifacts
are under `runs/mlp-query-gate-smoke/<UTC timestamp>/`.


## Live fallback metadata v2

Only contexts exceeding 8192 tokens use `metadata-table-prefix-v2`; original
serialization remains byte-identical for every in-limit input. Evidence and
branch field names are declared once, with records in the original order.
Constant evidence title/retention columns are declared in `evidence_defaults`.
A shared `url-sha256:` prefix is factored into `url_prefix` only when every source
has that prefix. Repeated complete metadata strings are interned in first-use
order when estimated token savings exceed the string-table entry cost. Integer
references occur only in declared string metadata fields; evidence bodies and
budget values are never interned. Budget keys/values remain unchanged.
`compact_metadata.decode_context` reverses these transformations, including the
previous v1 representation, without relying on external dictionaries. Body
suffixes remain the only intentionally omitted information, through the existing
deterministic prefix allocator. Tests compare every decoded metadata value,
record order, history multiplicity, retention flag and budget value exactly.

This is a capacity reduction, not a guarantee of complete depth-3 compatibility.
Offline measurements on saved b0005–b0010 metadata reduce 4343–6809 tokens to
3806–5885. A b0011 approximation (missing IDs/goals substituted, not recovered)
uses 7427 tokens. A plausible extrapolation of the observed all-EXECUTE run
through 28 branch batches, 305 distinct evidence sources, 142 history entries
and 60 branches uses 19853 tokens even with empty evidence bodies. It remains
unsupported and fails explicitly; no evidence/history selection or opaque
compression is used to disguise this remaining integration limitation.
