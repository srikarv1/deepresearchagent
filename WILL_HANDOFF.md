# Will handoff: DRB BC/DPO transfer runs

This package generates reports only. It does not train either adapter or run the
official RACE/FACT judges.

## Reproducibility pins

- Parent repository: `https://github.com/srikarv1/deepresearchagent.git`
- Parent branch: `aadya/drb-bc-dpo-transfer`
- Experiment/Flex pin checkpoint: `2aab7e00697592f2ec80aeaa83f5a77de76d9c85`
- GPT-Researcher fork: `https://github.com/WilliamOdinson/gpt-researcher.git`
- Fork branch: `aadya/drb-azure-flex`
- Fork commit: `f1abd1caeefc6c55146b39afbc2dda44eedd40d8`

Use the final parent handoff commit SHA sent by Aadya; it contains this guide and
the validator on top of the experiment checkpoint above. Verify both checkouts
with `git rev-parse HEAD` before running.

## Fresh clone

```bash
git clone --branch aadya/drb-bc-dpo-transfer \
  https://github.com/srikarv1/deepresearchagent.git \
  deepresearchagent-drb-transfer
cd deepresearchagent-drb-transfer

git clone --branch aadya/drb-azure-flex \
  https://github.com/WilliamOdinson/gpt-researcher.git \
  third_party/gpt-researcher
git -C third_party/gpt-researcher checkout \
  f1abd1caeefc6c55146b39afbc2dda44eedd40d8
```

## Existing clone

Do not reuse a checkout with local changes. From the dedicated transfer repo:

```bash
git status --short
git fetch origin aadya/drb-bc-dpo-transfer
git switch aadya/drb-bc-dpo-transfer
git pull --ff-only origin aadya/drb-bc-dpo-transfer

git -C third_party/gpt-researcher status --short
git -C third_party/gpt-researcher fetch origin aadya/drb-azure-flex
git -C third_party/gpt-researcher switch aadya/drb-azure-flex
git -C third_party/gpt-researcher pull --ff-only origin aadya/drb-azure-flex
test "$(git -C third_party/gpt-researcher rev-parse HEAD)" = \
  f1abd1caeefc6c55146b39afbc2dda44eedd40d8
```

Stop and ask Aadya if either status command reports changes or either pinned
commit is unavailable. Do not substitute another GPT-Researcher checkout.
If the local fork branch does not exist yet, replace its `switch` command with
`git -C third_party/gpt-researcher switch --track origin/aadya/drb-azure-flex`.

## Python environment

Python 3.11 is the tested version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e third_party/gpt-researcher
```

## Private environment

Create an ignored `.env` locally. Aadya will transfer the Tavily CSV separately;
store it outside Git. Required variable names are:

```dotenv
# Azure OpenAI-compatible endpoint used for Sol and embeddings.
OPENAI_API_KEY=<private>
OPENAI_BASE_URL=<Azure OpenAI-compatible base ending in /openai/v1>

# Absolute path is preferred. The CSV header must be: api_key
TAVILY_KEYS_CSV=<absolute path to the privately transferred CSV>

# One vLLM server exposes both adapter model names through this base.
GR_ORCH_LLM_BASE_URL=<Cloudflare tunnel URL ending in /v1>
GR_ORCH_LLM_API_KEY=EMPTY
```

The experiment config fixes Sol to `gpt-6.1-sol-1gtjfzlv8i`, embeddings to
`text-embedding-3-small-v1-651fgh`, Tavily retrieval, depth 2, breadth 4, and
queries 61–75. Do not set `LLM_KWARGS` or change these values.

Load `.env` through the shell in **each Terminal window**. This also expands any
shell variables used in paths:

```bash
set -a
source .env
set +a
```

Never commit `.env`, the CSV, logs containing credentials, or adapter files.

## Preflight checks

Confirm the private CSV is readable without displaying it:

```bash
test -r "$TAVILY_KEYS_CSV"
test "$(awk 'END {print NR-1}' "$TAVILY_KEYS_CSV")" -eq 15
```

Confirm the tunnel is reachable without making an inference or consuming model
tokens. This calls only vLLM's model-list endpoint:

```bash
curl --fail --silent --show-error --max-time 10 \
  -H "Authorization: Bearer ${GR_ORCH_LLM_API_KEY}" \
  "${GR_ORCH_LLM_BASE_URL%/}/models" |
  .venv/bin/python -c '
import json, sys
models = {row.get("id") for row in json.load(sys.stdin).get("data", [])}
needed = {"pilot-bc-teacher", "pilot-dpo-teacher"}
missing = needed - models
print("adapter model listing: PASS" if not missing else f"missing adapters: {sorted(missing)}")
raise SystemExit(bool(missing))
'
```

### Billable Azure Flex smoke test

This makes **exactly one small paid request**. Run it only after Azure access is
allowlisted. Retries are disabled; do not repeat automatically on failure.

```bash
.venv/bin/python -c '
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-6.1-sol-1gtjfzlv8i",
    service_tier="flex",
    reasoning_effort="low",
    temperature=None,
    max_completion_tokens=64,
    timeout=20,
    max_retries=0,
)
response = llm.invoke([{"role": "user", "content": "Reply OK."}])
print("Azure Flex smoke: PASS")
print("returned service_tier:", response.response_metadata.get("service_tier", "not reported"))
'
```

Stop on any Azure firewall, authentication, capacity, or tier error. Never fall
back to standard tier.

## Launch in two Terminal windows

The commands use separate output parents, run IDs, and log files. Running both
at once is supported, but they share the same Colab server, Cloudflare tunnel,
Azure resource, and Tavily key pool file. Keep the Mac on power. `caffeinate`
prevents system and display sleep while each command is alive.

Tavily rotation state is process-local: the two runs independently begin with
the first CSV key and may use the same key concurrently. The manager is safe
within each process, but it does not coordinate key allocation across windows.

In both windows:

```bash
cd /path/to/deepresearchagent-drb-transfer
source .venv/bin/activate
set -a; source .env; set +a
mkdir -p logs
set -o pipefail
```

Terminal 1 — BC:

```bash
GR_ORCH_LLM_MODEL=pilot-bc-teacher \
caffeinate -dimsu .venv/bin/adr run \
  --config configs/gpt_researcher_bench_gptsol_d2_learned.yaml \
  --run-id drb-bc-sol-d2-q61-75 \
  --output-dir runs/will-bc \
  2>&1 | tee logs/drb-bc-sol-d2-q61-75.log
```

Terminal 2 — DPO:

```bash
GR_ORCH_LLM_MODEL=pilot-dpo-teacher \
caffeinate -dimsu .venv/bin/adr run \
  --config configs/gpt_researcher_bench_gptsol_d2_learned.yaml \
  --run-id drb-dpo-sol-d2-q61-75 \
  --output-dir runs/will-dpo \
  2>&1 | tee logs/drb-dpo-sol-d2-q61-75.log
```

Do not close the terminals, stop Colab, rotate the tunnel, change networks, or
allow the Mac to lose power during the runs.

## Validate and return results

```bash
.venv/bin/python scripts/validate_drb_transfer_run.py \
  runs/will-bc/drb-bc-sol-d2-q61-75 \
  --adapter pilot-bc-teacher

.venv/bin/python scripts/validate_drb_transfer_run.py \
  runs/will-dpo/drb-dpo-sol-d2-q61-75 \
  --adapter pilot-dpo-teacher
```

Both commands must print `PASS` and exit zero. The validator checks all 15 IDs,
nonempty reports, harness errors, raw GPT-Researcher trajectories, learned
decisions, parse failures, decision errors, and adapter provenance.

The important artifacts are under each run directory:

- `config.yaml`, `manifest.json`, and `metrics/`
- `reports/` and `trajectories/`
- `gpt_researcher/` raw decision trajectories
- `exports/deep_research_bench/`
- the corresponding file under `logs/`

Tavily currently logs the final six characters of a rotated key, so do **not**
transfer raw logs. Package only the run directories—never `.env`, logs, or the
Tavily CSV:

```bash
tar -czf drb-bc-dpo-will-results.tgz \
  runs/will-bc/drb-bc-sol-d2-q61-75 \
  runs/will-dpo/drb-dpo-sol-d2-q61-75
shasum -a 256 drb-bc-dpo-will-results.tgz
```

Transfer the archive and SHA-256 to Aadya through the approved project channel.
Send only manually sanitized log excerpts if troubleshooting is necessary.
Official GPT-5-mini RACE/FACT evaluation is run separately after generation.
