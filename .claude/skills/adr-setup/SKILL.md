---
name: adr-setup
description: Set up the deepresearchagent (adr) development environment and run benchmarks. Use when the user asks about installation, environment variables, API keys, bootstrap, deploying models, running DRB or BrowseComp-Plus benchmarks, the BrowseComp-Plus retriever server (BM25 or dense), Ollama embedding, or getting the project running for the first time. Also use when the user hits errors related to missing keys, missing modules, pyserini/Java, a missing Lucene index, a refused connection to 127.0.0.1:8321, Ollama not running, faiss-cpu missing, or Tavily/OpenAI/Azure configuration.
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

**BM25 (required for all BrowseComp-Plus runs).** Needs pyserini + Java 21.

```bash
pip install -e ".[bcp]"
python scripts/download_bcp_index.py          # 2.1 GB -> third_party/bcp_indexes/bm25
```

**Dense retriever.** Uses Ollama for query encoding.

```bash
ollama pull qwen3-embedding:0.6b
python scripts/download_bcp_index.py --subdir qwen3-embedding-0.6b   # 0.4 GB shards
```

The BM25 index is still required when using dense (it stores document text).

### Step 5: Verify

```bash
adr doctor
```

For BrowseComp-Plus the `BM25 index` and `pyserini + java` rows must be green. For dense, also check `dense index` and `faiss-cpu`.

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
export GR_ORCHESTRATOR="topk"            # none | topk | extractive | llmlingua | prompted
```

### Pre-run checklist

Before running, ask the user:

1. Have you deployed the models specified in `STRATEGIC_LLM`, `SMART_LLM`, `FAST_LLM`, and `EMBEDDING` on your Azure Foundry account?
2. DRB: are `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `TAVILY_API_KEY`, and `JINA_API_KEY` set?
3. BrowseComp-Plus: are `OPENAI_API_KEY` and `OPENAI_BASE_URL` set, and is `adr serve-retriever` running? For dense, is Ollama running (`ollama list` should show `qwen3-embedding:0.6b`)?

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
curl -sf http://127.0.0.1:8321/health      # BrowseComp-Plus only
```

## Running benchmarks

### Deep Research Bench

```bash
adr run --config configs/gpt_researcher_bench.yaml --limit 1
adr evaluate runs/<tab-complete> --official deep_research_bench
```

### BrowseComp-Plus

```bash
# terminal 1, pick one:
adr serve-retriever                           # BM25
adr serve-retriever --searcher dense          # dense (Ollama + FAISS)

# terminal 2:
adr run --config configs/gpt_researcher_browsecomp_plus.yaml --limit 5
```

```bash
adr evaluate runs/<tab-complete> --official browsecomp_plus
```

The gpt-researcher fork must write a short `Exact Answer:` (set `GR_ANSWER_FORMAT=browsecomp` in `configs/agents/gpt_researcher_browsecomp_plus.yaml`; needs the `srikar/browsecomp-plus-short-answer` branch or a merge of it). A 2000-word research report is scored wrong even when the fact is in the text.

### Inspect queries

```bash
adr queries -d browsecomp_plus --limit 10
adr queries -d browsecomp_plus --ids 1,3,5 --show-answer
```
