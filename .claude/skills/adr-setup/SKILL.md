---
name: adr-setup
description: Set up the deepresearchagent (adr) development environment and run benchmarks. Use when the user asks about installation, environment variables, API keys, bootstrap, deploying models, running DRB or BrowseComp-Plus benchmarks, the BrowseComp-Plus retriever server, BM25 or dense Qwen3-Embedding/Ollama index, or getting the project running for the first time. Also use when the user hits errors related to missing keys, missing modules, pyserini/Java, a missing Lucene index, a refused connection to 127.0.0.1:8321, Ollama embeddings, or Tavily/OpenAI/Azure configuration.
---

# ADR Setup

## Installation

Run these steps in order.

### Step 1: Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Bootstrap third-party repos

Check if `GR_BRANCH` is already set in the environment. If it is, use it directly. If not, list available branches and let the user choose:

```bash
curl -s https://api.github.com/repos/WilliamOdinson/gpt-researcher/branches | python3 -c "import json,sys;[print(b['name']) for b in json.load(sys.stdin)]"
```

Then run:

```bash
GR_BRANCH=<chosen_branch> bash scripts/bootstrap_third_party.sh
```

### Step 3: Install packages

Run in this order:

```bash
pip install -e ".[dev]"
pip install -e third_party/gpt-researcher
pip install -r third_party/deep_research_bench/requirements.txt
pip install openai crawl4ai
pip install selenium torch llmlingua
```

The browser scraper requires Google Chrome. If not already installed, download it from https://www.google.com/chrome/.

### Step 4: BrowseComp-Plus retriever

Needs pyserini + Java 21. Dense ranking also needs numpy (pulled in by `[bcp]`), the Tevatron 8B shards, and Ollama. Caption Acc as dense-8B (BCP paper headline). Use `--dense-model qwen3-embedding-0.6b` only for the 0.6B ablation.

```bash
pip install -e ".[bcp]"
python scripts/download_bcp_index.py                 # 2.1 GB Lucene text
python scripts/download_bcp_index.py --kind dense    # 1.64 GB Qwen3-Embedding-8B shards
ollama pull qwen3-embedding:8b                       # query encoder; must match the shard
```

### Step 4: Verify

```bash
adr doctor
```

## Environment variables

Before running any benchmark, check that these are set. They may live in the shell RC file (~/.zshrc, ~/.bashrc), a .env file in the repo root, or exported in the current session. Check all three before asking the user to set them.

### Required keys

```bash
export OPENAI_API_KEY="..."              # Azure Foundry OpenAI-compatible endpoint key
export OPENAI_BASE_URL="https://<resource>.openai.azure.com/openai/v1"
export TAVILY_API_KEY="tvly-..."         # Tavily search API
export JINA_API_KEY="jina_..."           # DRB FACT judge scraping
```

### Judge and context config

```bash
export LLM_BACKEND=openai
export RACE_MODEL=gpt-5-mini
export FACT_MODEL=gpt-5-mini
export GR_CONTEXT_BUDGET_TOKENS=50000
```

### Model overrides for gpt-researcher

Defaults are in `configs/agents/gpt_researcher.yaml` under `env:`. Override via environment:

```bash
export STRATEGIC_LLM="openai:gpt-5-mini"
export SMART_LLM="openai:gpt-5-mini"
export FAST_LLM="openai:gpt-5-mini"
export EMBEDDING="openai:text-embedding-3-small"
```

### Orchestration policy (optional)

Only works if the gpt-researcher fork was bootstrapped on a branch with orchestration support.

```bash
export GR_ORCHESTRATOR="topk"            # none | topk | extractive | llmlingua | prompted | greedy | heuristic_stop
export GR_CONTEXT_BUDGET_TOKENS=50000    # retained-evidence budget per round (topk / extractive / llmlingua / greedy)

Knobs for the two coverage-based rows (all optional; defaults in parentheses):

```bash
# greedy: stop adding once the best marginal value < eps (0.0); novelty exponent (1.0, 0 disables)
export GR_GREEDY_MIN_GAIN=0.0
export GR_GREEDY_LAMBDA=1.0
# heuristic_stop: terminate once g_t = Phi(K_t) - Phi(K_{t-1}) < threshold (0.01)
# for GR_STOP_PATIENCE consecutive rounds (1), evaluated from round GR_STOP_MIN_ROUNDS (2).
# Rounds are depth-first per node, so sweep patience together with the threshold.
export GR_STOP_GAIN_THRESHOLD=0.01
export GR_STOP_MIN_ROUNDS=2
export GR_STOP_PATIENCE=1
export GR_STOP_RETAIN=filter             # filter = EmbeddingsFilter verdict (legacy retention) | all
```
```

### Pre-run checklist

Before running, ask the user:

1. Have you deployed the models specified in `STRATEGIC_LLM`, `SMART_LLM`, `FAST_LLM`, and `EMBEDDING` on your Azure Foundry account?
2. Are `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `TAVILY_API_KEY`, and `JINA_API_KEY` set?

Check programmatically:

```bash
python3 -c "
import os, sys
keys = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'TAVILY_API_KEY', 'JINA_API_KEY']
missing = [k for k in keys if not os.environ.get(k)]
if missing:
    print('missing: ' + ', '.join(missing))
    sys.exit(1)
print('all keys set')
"
```

## Running benchmarks

### Deep Research Bench

```bash
adr run --config configs/gpt_researcher_bench.yaml --limit 1
adr evaluate runs/<tab-complete> --official deep_research_bench
```

### BrowseComp-Plus

BM25 (keyword overlap; weak on obfuscated BrowseComp questions):

```bash
adr serve-retriever                           # terminal 1, leave running
adr run --config configs/gpt_researcher_browsecomp_plus.yaml --limit 5   # terminal 2
adr evaluate runs/<tab-complete> --official browsecomp_plus
```

Dense (official Qwen3-Embedding-8B vectors + Ollama query encode). Caption Acc/Recall as dense-8B, not the BM25 leaderboard or 0.6B ablation. Lucene is still required for document text.

```bash
adr serve-retriever --retriever dense         # terminal 1
adr run --config configs/gpt_researcher_browsecomp_plus_dense.yaml --limit 5
adr evaluate runs/<tab-complete> --official browsecomp_plus
```

Paper table rows are one frozen agent × `GR_ORCHESTRATOR` (needs the fork orchestration branch). Official Recall is retrieved ∩ evidence; pruning does not change it.

```bash
# terminal 1 stays on --retriever dense
for pol in none topk extractive llmlingua prompted greedy heuristic_stop; do
  GR_ORCHESTRATOR=$pol adr run \
    --config configs/gpt_researcher_browsecomp_plus_dense.yaml \
    --run-name "bcp-dense-${pol}"
  adr evaluate runs/*-bcp-dense-${pol} --official browsecomp_plus
done
```

The gpt-researcher fork must write a short `Exact Answer:` (set `GR_ANSWER_FORMAT=browsecomp` in `configs/agents/gpt_researcher_browsecomp_plus.yaml`; needs the `srikar/browsecomp-plus-short-answer` branch or a merge of it). A 2000-word research report is scored wrong even when the fact is in the text. Both that PR and the orchestration PR must be on the installed fork to fill the table.

Inspect BrowseComp-Plus queries

```bash
adr queries -d browsecomp_plus --limit 10
adr queries -d browsecomp_plus --ids 1,3,5 --show-answer
```
