# Frozen MLP data preparation

The `prepare` module prepares inputs only. The separate `train` module runs the
fixed offline baseline described below. Neither gates live research.
Run commands from the repository root. The `mlp` dependency extra supplies NumPy,
PyTorch, scikit-learn and Matplotlib
(`pip install -e '.[mlp]'` when needed); the current project environment already has it.

```sh
PYTHONPATH=src .venv/bin/python -m adr.mlp.prepare --config configs/mlp_prepare.yaml --dry-run
```

Default mode is offline even without `--dry-run`. It validates the full canonical
corpus and held-out registry, persists the split, fits the training-only scaler,
writes numerical arrays and separate metadata, and inventories exact embedding
strings. If `OPENAI_BASE_URL` is unset or blank, status is `awaiting_endpoint`: the
contract records a null endpoint hash and no embedding cache is created. The next
run with an explicit endpoint can bind this plan once, provided all other contract
fields match and no cache or final X arrays exist. No default OpenAI endpoint is
silently selected. API-enabled runs require an explicit endpoint before writing
artifacts or constructing a client. It never constructs an API client. It does not write dummy embeddings
or claim the model input arrays are ready.

After explicit authorization to incur embedding costs:

```sh
PYTHONPATH=src .venv/bin/python -m adr.mlp.prepare --config configs/mlp_prepare.yaml --allow-api
```

The same command resumes successful cached batches after a failure and makes no
requests for completed strings. SQLite commits each successful batch. A process
crash after a server response but before local commit may require that batch again;
there is no distributed exactly-once billing guarantee. An output lock prevents
concurrent runs against the same cache. Training only reads arrays: it never embeds.

## Existing client and Azure routing

The model agrees with `configs/agents/gpt_researcher.yaml`:
`text-embedding-3-small`, explicitly requesting 1536 dimensions. If Azure uses a
different deployment alias, set `model` in a separate preparation config before
starting that preparation. Availability of that deployment is not established by
the offline checks.

The vendored `Memory('openai', ...)` embedding wrapper already forwards
`OPENAI_BASE_URL` to LangChain's OpenAI embeddings. LangChain is not installed in
the ADR environment. This preparation layer instead adds `embed()` to ADR's existing
`OpenAICompatLLM` and uses its existing AsyncOpenAI client, credentials and endpoint
handling. It does not introduce another SDK/client stack or invoke chat completions.

`OPENAI_API_KEY` and `OPENAI_BASE_URL` are loaded from the existing environment
(or `.env`, without overriding environment values). A base URL ending in
`/openai/v1/` routes to `/openai/v1/embeddings`, verified with HTTP mock transport.
For direct OpenAI use, explicitly set `OPENAI_BASE_URL=https://api.openai.com/v1`.
No Azure legacy endpoint/deployment URL rewriting or API-version parameter is used.
No secret or endpoint value is written to artifacts; endpoint identity is hashed.
Authentication and deployment availability require the eventual real request.

Only exact UTF-8 strings are sent. There are no instructions, prefixes, joined
fields, chunking, newline changes or truncation. The conservative preflight caps
each string at 8191 UTF-8 bytes (an upper bound on byte-BPE token count), and each
batch at 16 strings. This corpus's longest string is 6269 bytes. Overlength inputs
fail before any requests rather than silently changing the locked representation.

## Reproducibility and artifacts

The split ranks trajectory IDs by SHA-256 of `str(seed) + NUL + trajectory_id`;
the first four form validation, and the other sixteen form training. Seed is 42.
The algorithm never reads labels, reasons, performance, or category distributions.
Each question must own exactly one trajectory. No synthetic test split is created.

Output: `data/mlp/cleaned-r1-seed42-azure/`:

- `contract.json`: source/held-out hashes, exact split, text/feature order,
  model/deployment, dimensions, endpoint hash and preparation version. Changed
  contracts require a new output directory (except the guarded initial binding of
  an unbound offline plan). Mismatch errors list differing field paths, never
  endpoint or credential values.
- `split.json`: the exact 16/4 trajectory IDs and algorithm.
- `normalization.json`: training-only means, population standard deviations,
  scales, zero-variance mask and feature order. Zero-variance scales become 1;
  validation is transformed with these same statistics, never refitted/clipped.
- `embedding_inputs.json`: SHA-256 keys and all exact unique strings across fields.
- `embeddings.sqlite3`: frozen vectors, keyed by exact text and bound to the
  model/dimension/endpoint identity. Empty of vectors after an endpoint-bound offline pass; absent for an unbound plan.
- `train_numerical.npz`, `validation_numerical.npz`: normalized float32 numerical
  features and int64 labels, available offline.
- `train_metadata.json`, `validation_metadata.json`: aligned analysis-only metadata
  and embedding keys, including IDs, labels, reasons/categories and pair links.
- `manifest.json`: readiness status, counts/distributions, validation and artifact hashes.
- `train.npz`, `validation.npz`: created only when all embeddings exist; contain
  float32 `X` and int64 `y`. Expected shapes are `(320, 4623)` and `(80, 4623)`.

Columns: question embedding [0:1536], context [1536:3072], candidate [3072:4608],
then the locked 15 normalized features [4608:4623]. EXECUTE=1, SKIP=0.
No metadata columns are concatenated. The deterministic `context_before` string
is used unchanged: its existing evidence and branch identifiers remain part of
the locked state serialization. Separate example/question/trajectory IDs, state
indices, pair IDs, provenance, labels and reasons are never added to text or X.
Strict validation checks local derivations and cumulative pre-decision invariants;
it cannot certify absence of every semantic/narrative leakage in free text.

Read frozen arrays later without client construction or API access:

```python
from pathlib import Path
from adr.mlp.prepare import load_prepared
X, y, metadata = load_prepared(Path('data/mlp/cleaned-r1-seed42-azure'), 'train')
```

This loader rejects incomplete preparation and checksum mismatches. Source data
and teacher/model generation settings are never changed by preparation.

Embedding failures report the failed batch number/count, exception class, HTTP
status when available, and structured API `code`, `type`, `param`, and `message`.
Diagnostics omit raw bodies, headers and exception strings. Known environment and
request credentials, endpoint URLs, common credential patterns and exact batch
input strings are redacted before display. Unstructured responses report status
and class only. The original traceback is suppressed so it cannot bypass this
sanitization. Successful batches remain committed; rerunning requests only missing
strings. The SQLite cache is authoritative for progress after an interrupted run;
the manifest may still show its pre-request count.

## Fixed offline MLP baseline

```sh
PYTHONPATH=src .venv/bin/python -m adr.mlp.train --config configs/mlp_baseline.yaml
```

This command performs a new experiment; it is not a resume command. It only reads
ready, checksum-verified prepared arrays and metadata. It never constructs an API
client or regenerates embeddings. Each invocation creates an exclusive timestamped
directory under `runs/mlp/` and leaves preparation artifacts untouched.

The fixed seed-42 CPU model is 4623 → 256 → 64 → 1, with ReLU and dropout 0.2
after each hidden layer. Training uses unweighted BCEWithLogitsLoss and AdamW
(learning rate 0.001, weight decay 0.0001), batch size 32, at most 100 epochs,
and patience 10. Strictly lower validation loss selects the best checkpoint,
which is restored before evaluation. The probability threshold stays at 0.5.
Reported train loss is measured in evaluation mode; dropout minibatch loss is
recorded separately. CPU algorithms are deterministic with one Torch thread.

Three unweighted, untuned logistic regressions use C=1, lbfgs, max_iter=2000:
all 4623 columns, candidate embedding columns 3072:4608, and normalized numerical
columns 4608:4623. No additional scaling is fitted. Majority class comes from
training labels. Metadata is used only for evaluation, never as model input.

Artifacts include `best_mlp.pt`, exact `config.json`, `history.json`,
`loss_curves.png`, `metrics.json`, `baselines.json`, linear coefficient NPZ files,
`counterfactual.json`, index-aligned `validation_predictions.npz` and separate
`validation_metadata.json`, `split.json`, and a hash/version provenance manifest.
Counterfactual probabilities within 1e-8 are treated as ties. Validation selects
the stopping epoch, so these four synthetic trajectories are not an independent
test set. No real DRB performance is implied.
