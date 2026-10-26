# Tiny staged branch-observation teacher smoke

Implementation version: `synthetic-branch-smoke-v1`. Reuses the unchanged source,
supervision and frozen observation contracts; does not implement a new representation.
No second LLM semantic reviewer and no accepted training output.

## Scope and call sequence

One original synthetic world, 8–12 candidate decisions, preferably10, covering
zero prior results, middle1–3 published results, late>3 published results. The
world prompt proposes an original textile repair/reuse research domain, qualified
fictional source claims and at least one exact repeated query. No benchmark files,
previous trajectory content or existing synthetic corpus are loaded into prompts.
This is not a production contamination validator or full-dataset generator.

1. One GPT-5-mini world-author request,32K completion allowance. Author source
   events only: no supervision, witness references, pair IDs, or label rationale.
2. Replay through `compile_world`, which directly invokes the frozen constructor.
   Save exact packets and separate audit/ledger. Scope validation precedes any
   annotation calls. A bad world stops the smoke.
3. One fresh stateless GPT-5-mini annotation request per supported observation,
   16K completion allowance each. No batch of observations in one prompt: that
   would let an earlier decision see later evidence. Maximum13 provider calls
   total (world+12 annotations); usually11 for10 candidates. SDK retries disabled.

The annotation user message is an explicit allowlist of ONLY:
`original_question`, exact `context_before` string, `candidate_subquery`, and
`supervision_schema`. A fixed system instruction explains that schema/categories.
No IDs, hashes, scores, numeric features, world records, raw page prefixes,
unselected branches, execution ledger, earlier/later packets or world-author
conversation is attached. Even numeric features are omitted from annotation in
this smoke per the narrower request; they remain saved for eventual model inputs.
The context string is neither reserialized internally nor shortened.

ID and full-packet-hash binding occurs locally after annotation. Optional pair IDs
are proposed locally only for identical chronological EXECUTE→SKIP queries with
a visible later decisive finding, absent earlier, from a newly published selected
branch. These are structural pair candidates for human review, not certified pairs.
No label balance or pair quota is enforced. No model self-approval is accepted.

The complete source world records hypothetical execution before independent
annotation. If a proposed label disagrees with that execution, the audit flags it.
It cannot later pass the unchanged finalizer as-is. This smoke does NOT rewrite
source history or outcomes to reconcile labels, and does NOT claim that it has
completed the future incremental generation/adjudication strategy.

## Numerical ledger

Teacher-authored event data:
- task/parent identities, goals, candidate and planning query strings;
- the proposal IDs actually executed in the hypothetical world;
- newly acquired feature-page identities, source URLs and <=2000-character raw
  prefixes (not annotation evidence);
- read-attempt deltas and research-token deltas, including process-result cost;
- processed findings, follow-ups and complete cited-insight maps.

Derived locally:
- depths, lifecycle, publication order and availability;
- concrete prior executed query history from IDs in proposal order;
- searches = explicit planning-query count + executed candidate count;
- steps = prior candidate proposals + completed publication rounds;
- cumulative reads/tokens = sums of event deltas, never separate cumulative claims;
- all15 features through the existing complete-state derivation; selection never
  restricts the numerical feature source;
- fixed32/64/128/400000 nominal maxima, enforce=false, unchanged saturated fractions
  and remaining-budget floors; zero score/no pruning.

Reads must account for new pages; no reads/pages without execution; results with
semantic findings require prior task execution. Page/history visibility uses the
same publication barrier as compilation. `numerical_ledger.json` independently
reconciles every candidate-batch usage snapshot with the compiler. It does not
independently invent features. Simulated research-token costs remain hypothetical
teacher-authored event deltas; they are NOT actual teacher API usage, which is
preserved separately. Small smoke ledgers need not reproduce full-depth budget
exhaustion; this is not evidence of calibrated production numeric distributions.

## Checkpoint and failure policy

**One provider attempt per stage for the lifetime of this directory. No retry
flag, automatic regeneration, or SDK retries.**

Before dispatch: persist exact request and an in-flight marker. Immediately after
return: atomically persist one response bundle containing full SDK provider
response, assistant content and usage BEFORE JSON or structural validation. Raw
responses exclude HTTP credential headers. Usage/finish-reason details are retained.
No price estimate is invented. Arbitrary provider exception text/URLs are omitted
from failure records to avoid exposing credentials.

Length/empty/reasoning exhaustion is a generation failure, not a Pydantic JSON
failure. Invalid JSON/schema/witnesses are validation failures. Stop immediately;
no additional candidate calls after a failed stage. Human audit includes pending
rows for all remaining candidates when a world was successfully compiled.

`--resume` reuses validated checkpoints, locally validates a saved response whose
validation was interrupted, and calls only untouched stages. A failed stage stays
failed. An in-flight marker without a saved response is uncertain and blocks
another call. Inspect first; do not remove/reset files to trigger a retry. A fresh
attempt would require a deliberately new output directory after review.

Code/config/dependency hashes bind checkpoints. The Azure endpoint is fingerprinted
only when the API client is first required; offline prep never binds an OpenAI
fallback endpoint. A file lock prevents simultaneous invocations. The existing
shared client handles GPT-5-mini completion parameter/temperature compatibility.

## Artifacts

- `contract.json`, `world_request_preview.json`, `endpoint_identity.json` (after
  client construction; endpoint hash only), `manifest.json`;
- `stages/world/` and `stages/annotation-NNN/`: request, started marker, raw response
  bundle, usage, local validated result, outcome;
- `source_world.json`, `compiled_packets.json`, `compilation_audit.json`;
- `numerical_ledger.json`;
- `human_audit.json` and compact `human_audit.md`: each query, exact proposed label,
  category, witnesses, rationale, selected goals, pair ID, token count, structural
  status and source-execution agreement. Full exact contexts remain in packets.

All proposed labels remain pending HUMAN semantic review. AMBIGUOUS/REJECT remain
nontraining. Overflow causes no annotation call and records EXECUTE bypass without
inventing a training label. No `examples.jsonl`, Review approval or finalizer call
is produced. Human audit must check claim entailment/qualifiers, relevance, pair
resolution and source-action disagreements; structural checks alone cannot do so.

## Commands

Offline preparation (does not instantiate an API client):

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_branch_v2.smoke --config configs/synthetic_branch_smoke.yaml
```

After offline preparation, from the repository root in the terminal that already
has `OPENAI_API_KEY` and `OPENAI_BASE_URL` exported, run the ONE smoke:

```sh
PYTHONPATH=src .venv/bin/python -m adr.synthetic_branch_v2.smoke --config configs/synthetic_branch_smoke.yaml --resume --allow-api
```

Endpoint must be HTTPS and end in `/openai/v1/`. No Tavily, scraper or embedding
credentials are used. Output: `runs/synthetic-branch-v2-smoke/smoke-1/`.
The same command can resume untouched work after an interruption, subject to the
one-attempt policy above. The paid command has not been executed by this task.
