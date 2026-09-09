# HOWTO: Populate Table 4 (DeepResearch Bench)

Table 4 in the paper reports, for each orchestration approach, the DeepResearch
Bench quality scores (Comprehensiveness, Insight, Readability, Citation
accuracy) plus Tokens (K), Latency (s), and Q per second. This document is the
exact procedure for producing those numbers with this harness, on the
`feat/gpt-researcher-agent` branch, using the instrumented gpt-researcher fork
as the fixed agent backbone.

Nothing here runs inference for you. It tells you what to run, in what order,
what comes out, and which field goes in which cell.

---

## 0. Status: what exists and what does not

Read this first so you know which steps are "run a command" and which are
"write code first".

**Exists and works**

- DRB queries (`data/benchmarks/deep_research_bench/query.jsonl`, 100 rows,
  50 `en` / 50 `zh`, byte-identical to the judge checkout).
- Export in the official `{id, prompt, article}` format
  (`src/adr/eval/exporters.py`).
- RACE + FACT invoked as subprocesses against `~/deep_research_bench`, results
  parsed into `summary.json` (`src/adr/eval/deep_research_bench.py`).
- Harness-owned token/latency metering (`src/adr/core/instrument.py`).
- The `gpt_researcher` agent, which wraps the fork's deep-research mode and
  converts its trajectory to the harness `Trajectory`
  (`src/adr/agents/gpt_researcher.py`).
- `adr run`, `adr evaluate`, `adr score`, `adr compare`, `adr dag`, `adr doctor`.

**Does not exist yet (you or I must build it before the table can be filled)**

| Item | Needed for | Where it goes |
|---|---|---|
| An orchestrator switch inside the fork's per-round checkpoint | every row except the fork default | the fork (`third_party/gpt-researcher`), see §4 |
| No-pruning policy | row 1 | fork switch value `none` |
| Top-k similarity policy | row 2 | fork switch value `topk` |
| Extractive compression policy (RECOMP-style) | row 3 | fork switch value `extractive` |
| Prompt compression policy (LLMLingua) | row 4 | fork switch value `llmlingua` |
| Prompted orchestration policy | row 5 | fork switch value `prompted` |
| PILOT (trained policy) | row 6 | **not in scope of this doc**; needs the training pipeline |
| `Q per second` / `Q per 1K tok` | last column | computed at table time (§7 snippet) |
| Multi-seed aggregation (mean ± std) | every cell | §7 snippet |
| Matched-token-budget protocol (§5.1 of the paper) | comparability of rows | §8 |

**Known wart (no code change required, but you must know it)**

The harness gates RACE/FACT on `OPENROUTER_API_KEY` (default) or
`OPENAI_API_KEY` (`LLM_BACKEND=openai`), but the checkout that `adr doctor`
resolves to (`~/deep_research_bench`, Flitternie fork) hard-codes
`GEMINI_API_KEY` in `utils/api.py` and raises if it is missing. The judge
subprocess inherits your full shell environment (`src/adr/eval/procs.py`
copies `os.environ`), so the working configuration is:

```bash
export LLM_BACKEND=openai        # satisfies the harness gate using a key you already need
export OPENAI_API_KEY=...        # gpt-researcher's LLM + embeddings
export GEMINI_API_KEY=...        # what RACE and FACT actually call
```

Judge models pinned in the checkout: RACE uses `gemini-2.5-pro-preview-06-05`,
FACT uses `gemini-2.5-flash-preview-05-20`.

---

## 1. One-time setup

```bash
cd ~/deepresearchagent
git checkout feat/gpt-researcher-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Judge repos + the instrumented fork. Symlinks existing clones instead of re-cloning.
bash scripts/bootstrap_third_party.sh
pip install -e third_party/gpt-researcher
pip install -r ~/deep_research_bench/requirements.txt     # google-genai for RACE/FACT

adr doctor
```

`adr doctor` must show all of: `DeepResearch Bench repo found`,
`gpt-researcher fork found`, `gpt_researcher importable from fork yes`.

Keys, all required for a full Table 4 run:

| Key | Used by | If missing |
|---|---|---|
| `OPENAI_API_KEY` | gpt-researcher LLM calls + `text-embedding-3-small` (the `_emb.npz` vectors are 1536-d) | agent will not run |
| `TAVILY_API_KEY` | gpt-researcher retrieval (`retriever: tavily` in the agent config) | agent will not run |
| `GEMINI_API_KEY` | RACE (Comprehensiveness / Insight / Readability) and FACT | no quality columns |
| `JINA_API_KEY` | FACT scrapes every cited URL through Jina Reader | no Citation accuracy column; RACE still runs |
| `LLM_BACKEND=openai` | harness gate, see §0 | harness refuses to launch the judge |

Put them in `.env` (already gitignored) or export them.

---

## 2. Prove the judge path works before spending on agents

RACE/FACT have never been exercised end-to-end from this harness — the test
suite stubs the judge. Do this first; it costs one judge call.

```bash
adr score --report data/fixtures/sample_report_51.md \
          --query-id 51 --dataset deep_research_bench
```

Expected: a `Local cost metrics` table, then an `Official judge scores` table
containing `race_comprehensiveness`, `race_insight`, `race_readability`,
`race_instruction_following`, `race_overall_score`, and (if `JINA_API_KEY` is
set) `fact_valid_rate`, `fact_total_citations`, `fact_total_valid_citations`.

If instead you see a yellow `deep_research_bench: ...` line, read it: it is
the `reason` string and names the missing key or repo. The subprocess logs are
in `runs/score-deep_research_bench-1q/metrics/summary.json` under
`official.deep_research_bench.race.log`.

---

## 3. Run config

Create `configs/table4/drb_base.yaml`. It mirrors the existing
`configs/gpt_researcher_gym.yaml` with the dataset swapped:

```yaml
run_name: table4-<row>-s<seed>      # you override this per run, see §5
output_dir: runs
seed: 17                            # recorded only; gpt-researcher does not consume it
concurrency: 1                      # REQUIRED: the fork's trackers are process-global

dataset:
  name: deep_research_bench
  language: en                      # 50 queries; the paper evaluates English
  limit: null
  query_ids: []

agent:
  name: gpt_researcher
  config: configs/agents/gpt_researcher.yaml

# gpt_researcher brings its own LLM + search; these stop the harness opening clients.
llm:
  provider: mock
  model: unused
search:
  backend: mock
  top_k: 5

# Recorded, not enforced: gpt-researcher does not consult harness budgets.
budget:
  max_steps: 32
  max_searches: 64
  max_reads: 128
  max_tokens: 400000
  max_latency_s: 1200.0
  max_evidence: 512
  enforce: false

eval:
  official_benches: []              # judge separately with `adr evaluate`, see §6
```

`configs/agents/gpt_researcher.yaml` controls the fork. The knobs that matter
for Table 4 and must be **identical across rows**: `depth`, `breadth`,
`concurrency`, `scraper`, `retriever`, `max_search_results_per_query`, and the
model overrides under `env:` (`STRATEGIC_LLM`, `SMART_LLM`, `FAST_LLM`). Fix
them once, commit the file, and never touch them between rows. Only the
orchestrator switch (§4) may differ.

---

## 4. Mapping the six rows onto the fork

### 4.1 What the fork does today (read this before touching anything)

The block in `gpt_researcher/skills/deep_research.py` (search for
`Orchestration checkpoint`) writes `decision = {type, kept_item_ids,
pruned_item_ids, branch_allocation}` into the raw trajectory. That has the
shape of the paper's action `a_t = (u_t, m_t, w_t)`, but **today it is a
recorder, not a decider**:

- `decision_type="continue"` is hard-coded; `finalize()` flips the last round
  to `terminate` after the fact. Nothing ever stops early.
- `branch_allocation` is never passed; recursion is fixed at
  `new_breadth = max(2, breadth // 2)` for every child.
- `all_context` is not touched. The context that reaches synthesis is whatever
  the sub-researchers already returned.

The actual retention happens upstream, inside each sub-researcher, in
`gpt_researcher/context/compression.py` (`ContextCompressor`):

1. scraped pages → 1000-char chunks →
   `EmbeddingsFilter(similarity_threshold=SIMILARITY_THRESHOLD)`, default
   **0.35, scored against the sub-query** (not the root query);
2. `pretty_print_docs(relevant_docs, top_n=10)` keeps only the **first 10
   surviving chunks** per sub-query;
3. `trim_context_to_word_limit(25000)` (recency truncation) runs at every
   recursion level and once more before synthesis.

The checkpoint then drains `_global_embedding_cache`, groups chunks by URL into
page items, and labels a page "kept" if any of its chunks survived step 1.
That is where the `kept_item_ids` in
`20260906-073953-gpt-researcher-gym` came from (4 of 69 pruned,
`fraction_pruned = 0.058`). Changing `SIMILARITY_THRESHOLD` per row would
**not** give you six rows: every row would inherit the top-10 cap and the
recency trim, and `u`/`w` would still be fixed.

### 4.2 The one structural change: make the checkpoint prescriptive

At the checkpoint, after `pages` is built, call a policy and act on its
answer. Pseudocode of the edit (one place, ~40 lines):

```python
pool = build_pool(pages, current_tree_depth, round_id)       # this round's new items
prev = self.trajectory_logger.retained_items()               # K_{t-1}, with embeddings
budget = B_tok - TokenTracker.get_totals()["total_tokens"]   # remaining, or None

u, m, w = self.orchestrator.decide(
    query_emb=self.query_emb, subq_embs=self.subq_embs,      # embed once, up front
    new_items=pool, retained_prev=prev, frontier=frontier_nodes,
    budget=budget, round_id=round_id,
)

kept_ids   = [it.id for it in pool + prev if it.id in m]
pruned_ids = [it.id for it in pool + prev if it.id not in m]
all_context = [it.context_text for it in pool if it.id in m]  # rebuilt from m, not from the filter
self.trajectory_logger.record_round(kept_item_ids=kept_ids, pruned_item_ids=pruned_ids,
                                    frontier=frontier_nodes, round_cost=round_cost,
                                    decision_type="terminate" if u else "continue",
                                    branch_allocation=w)
if u:
    return {...}                                              # u bites: no recursion
for result in results:
    child_breadth = allocate(w, result, breadth)              # w bites: replaces breadth // 2
```

Requirements that make this correct rather than approximately right:

- **Pool = `K_{t-1} ∪ new`.** The logger already stores every item with its
  unit-mean embedding; pass the retained set so policies can supersede old
  evidence, not just filter arrivals.
- **Rebuild context from `m`.** `pages[url]["chunks"]` holds every chunk for
  every page, kept or not, so the checkpoint can reconstruct the exact text for
  whatever it decides. The sub-researcher's filter then only affects the
  worker's learnings-extraction input, which is identical across rows.
- **Run the policy before the `TokenTracker.snapshot()` that closes the round**
  so its own cost (an LLM call for `prompted`, CPU seconds for `llmlingua`)
  lands in `round_cost`.
- **`allocate(w, ...)`** distributes the same total child breadth the fork
  would have spent (`Σ max(2, breadth//2)`) proportionally to `w[node_id]`,
  floor 1. Uniform `w` reproduces today's behavior exactly.
- Embed the root query and subquestions once at `run()` start (today they are
  embedded only at the end for `_emb.npz`).

Select the policy by env var so it flows through the agent config:

```yaml
# configs/agents/gpt_researcher.yaml
env:
  GR_ORCHESTRATOR: none | topk | extractive | llmlingua | prompted
  GR_TOKEN_BUDGET: "<B_tok from §8, optional>"
```

### 4.3 The six rows as policies

Every policy has the same signature,
`decide(...) -> (u: bool, m: set[item_id], w: dict[node_id, float])`. Items
carry `id, url, chunks, tokens, embedding, tree_depth, round`.

| Row | Switch | `m` (retention) | `u` | `w` | Extra cost |
|---|---|---|---|---|---|
| No pruning | `none` | all of `new ∪ prev` | never | uniform | none. Only the fork's 25K-word recency trim still cuts; it is a backbone hard limit shared by all rows — say so in the caption. |
| Top-k similarity | `topk` | `ρ_i = cos(query_emb, item.embedding)` over the whole pool, sort desc, keep while `Σ tokens ≤ B_tok` (or top-k by count without a budget) | never | uniform | none (embeddings already exist). Differs from the fork default: global ranking vs the **root** query, not a per-chunk threshold vs a sub-query. |
| Extractive compression | `extractive` | all items kept; each item's `chunks` scored by cosine to the root query and rewritten to the top chunks until budget. Bytes shrink, items don't drop. Log `compressed_chars` per item. | never | uniform | none at chunk granularity (reuses the 1000-char chunk embeddings). Sentence-level RECOMP would need a re-embed call per round; not worth it. |
| Prompt compression | `llmlingua` | all items kept; LLMLingua-2 over the concatenated kept text at `rate = B_tok / tokens_now`. Destroys item boundaries, so log `context_tokens_before/after` per round. | never | uniform | `pip install llmlingua`, ~500 MB BERT-class model, CPU. Latency is counted. |
| Prompted orchestration | `prompted` | one `create_chat_completion(..., usage_tag="orchestrator")` through the strategic model. Prompt: root query, subquestions, one line per item (`id \| ρ \| tokens \| depth \| age \| 200-char snippet`), frontier node ids, remaining budget. Ask for `KEEP: <ids>`, `ALLOC: <node>=<w>`, `DECISION: CONTINUE\|TERMINATE`. Regex-parse; on failure fall back to `none` and set `parse_failure=true` on the round. | from `DECISION` | from `ALLOC` | the LLM call. Snippets are included on purpose — the paper's baseline reasons "from the raw pool", and that expanding context is the cost being measured. |
| PILOT (full) | — | trained policy over the state defined in the paper's §4.3 | learned | learned | blank until the training pipeline exists. |

Frozen across rows (if any of these differ, the comparison is not about
orchestration): `depth`, `breadth`, `concurrency`, retriever, scraper,
`SIMILARITY_THRESHOLD` and the top-10 chunk cap inside sub-researchers, the
25K-word trim, the models, the synthesis prompt.

Keep every policy pure with respect to the trajectory logger: it must still
populate `kept_item_ids`, `pruned_item_ids`, `branch_allocation`, and `type`,
otherwise the harness adapter and the RL corpus (§9) lose those fields.

Add unit tests in the fork for each switch on a fixed synthetic pool
(deterministic embeddings) so a row can be trusted before it costs money.

---

## 5. Running the agents (no judge yet)

One run = one row × one seed × 50 queries. gpt-researcher is stochastic, so
"seed" here means "independent repeat"; the harness `seed` field is recorded
only. Use the run name to encode row and seed — §7 groups on it.

```bash
for row in none topk extractive llmlingua prompted; do
  for s in 0 1 2; do
    GR_ORCHESTRATOR=$row adr run \
      --config configs/table4/drb_base.yaml \
      --run-name "table4-${row}-s${s}"
  done
done
```

If you prefer the switch in the yaml rather than the shell, make one agent
config per row (`configs/agents/gpt_researcher_<row>.yaml`) differing only in
`env.GR_ORCHESTRATOR`, and pass `--config` accordingly.

Start small. Add `--limit 2` to the first run of each row and inspect the
outputs before committing to 50 queries.

What one run produces, under `runs/<timestamp>-table4-<row>-s<seed>/`:

```
config.yaml                          # the resolved config, keep for the paper
manifest.json                        # query ids, start/finish
queries/<id>.json
trajectories/<id>.json               # harness Trajectory (steps = one prune per round + write + terminate)
reports/<id>.md                      # the article that gets judged
gpt_researcher/<id>.json             # fork's raw trajectory (rounds, decisions, evidence, costs)
gpt_researcher/<id>_emb.npz          # ids, vectors[n,1536], query, subquestions, nodes, nodes_ids
exports/deep_research_bench/gpt_researcher.jsonl
metrics/local.json                   # per-query tokens / wall_s / retained / pruned ...
metrics/summary.json                 # means; `scores` is empty until §6
errors/<id>.txt                      # only for queries that raised
```

Cost reference from the one real run on this branch (query 879779, depth 2,
breadth 2): **151K tokens, 407 s, $0.65** agent-side, 47 tool calls, 3 rounds,
peak synthesis context 44.7K tokens. Budget roughly
50 queries × 5 rows × 3 seeds ≈ 750 trajectories ≈ **$450–500 and ~85 hours
serial** for the agents alone; `concurrency` must stay 1 per process, so
parallelize across processes/machines by splitting `query_ids`, not by raising
`concurrency`.

---

## 6. Judging (RACE + FACT)

Judge each run directory **separately, by run dir, without `--config`**:

```bash
for d in runs/*-table4-*; do
  adr evaluate "$d" --official deep_research_bench
done
```

Why this form and not `adr run --official`: the judge stages the export inside
the DRB checkout at `data/test_data/raw_data/<model_name>.jsonl` and writes
results to `results/race/<model_name>/`. With `adr run --official`,
`model_name` is the agent name, `gpt_researcher`, for every row, so each run
overwrites the previous run's judge results. `adr evaluate <run_dir>` without a
config uses the run directory name as `model_name`, which is unique.

Each `adr evaluate` rewrites `runs/<run>/metrics/summary.json` with:

```json
"scores": {
  "race_comprehensiveness": ..., "race_insight": ..., "race_readability": ...,
  "race_instruction_following": ..., "race_overall_score": ...,
  "fact_valid_rate": ..., "fact_total_citations": ..., "fact_total_valid_citations": ...
},
"official": { "deep_research_bench": { "race": {...}, "fact": {...}, "reason": null } }
```

RACE is one Gemini Pro call per report per dimension (with `--only_en`
automatically because the run's language is `en`). FACT extracts every cited
URL, scrapes each through Jina, and validates each claim: it is the slow and
expensive stage, and it needs `JINA_API_KEY`. If you must ration, run RACE on
all rows first and FACT second; both are selectable in
`configs/eval/deep_research_bench.yaml` (`run_race`, `run_fact`).

If `official.deep_research_bench.reason` is non-null, nothing was scored; the
`race.log.stderr_tail` field has the traceback.

---

## 7. Filling the table

Column mapping from `metrics/summary.json`:

| Table 4 column | Field | Notes |
|---|---|---|
| Comprehensiveness | `scores.race_comprehensiveness` | |
| Insight | `scores.race_insight` | |
| Readability | `scores.race_readability` | |
| Citation accuracy | `scores.fact_valid_rate` | fraction in [0,1]; multiply by 100 for the table |
| Tokens (K) | `mean_tokens / 1000` | from the fork's `TokenTracker`; `final_stats.cost_source` in each trajectory says so |
| Latency (s) | `mean_wall_s` | harness-measured end-to-end wall clock |
| Q per second | `scores.race_overall_score / mean_wall_s` | not stored; compute |

Aggregate across seeds with mean ± std. There is no `adr table` command yet;
this snippet does it from the run directories and groups on the
`table4-<row>-s<seed>` naming from §5:

```bash
python - <<'EOF'
import json, re, statistics as st
from pathlib import Path
rows = {}
for p in sorted(Path("runs").glob("*-table4-*/metrics/summary.json")):
    s = json.load(open(p))
    m = re.search(r"table4-(\w+)-s(\d+)$", s["run_id"])
    if not m: continue
    row = m.group(1); sc = s.get("scores") or {}
    if not sc.get("race_overall_score"): continue  # unjudged run
    rows.setdefault(row, []).append({
        "comp": sc.get("race_comprehensiveness"), "ins": sc.get("race_insight"),
        "read": sc.get("race_readability"),
        "cit": 100 * sc["fact_valid_rate"] if sc.get("fact_valid_rate") is not None else None,
        "tokK": s["mean_tokens"] / 1000, "lat": s["mean_wall_s"],
        "qps": sc["race_overall_score"] / s["mean_wall_s"] if s["mean_wall_s"] else None,
    })
def cell(vals, fmt):
    vals = [v for v in vals if v is not None]
    if not vals: return "—"
    mu = st.mean(vals); sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{mu:{fmt}} ± {sd:{fmt}}"
order = ["none", "topk", "extractive", "llmlingua", "prompted", "pilot"]
label = {"none": "No pruning", "topk": "Top-k similarity", "extractive": "Extractive compression",
         "llmlingua": "Prompt compression", "prompted": "Prompted orchestration", "pilot": "PILOT (full)"}
print("| Approach | Comprehensiveness | Insight | Readability | Citation acc. | Tokens (K) | Latency (s) | Q/s | n |")
print("|---|---|---|---|---|---|---|---|---|")
for r in order:
    rs = rows.get(r, [])
    g = lambda k, f: cell([x[k] for x in rs], f)
    print(f"| {label[r]} | {g('comp','.1f')} | {g('ins','.1f')} | {g('read','.1f')} | {g('cit','.1f')} | "
          f"{g('tokK','.1f')} | {g('lat','.1f')} | {g('qps','.3f')} | {len(rs)} |")
EOF
```

`n` is the number of seeds that were judged; it should be 3 for every row
before the table goes in the paper. Also record `mean_n_retained`,
`mean_n_pruned`, `mean_prune_rate` from the same files — they are not in
Table 4 but Table 6 and the qualitative analysis need them.

---

## 8. Matched token budget (paper §5.1)

Comparing rows at their natural operating points confounds quality with spend.
The paper fixes a token budget equal to prompted orchestration's spend and
evaluates every other row at that budget.

1. Run the `prompted` row first (all three seeds). Read `mean_tokens` from its
   summaries; average them. That is `B_tok`.
2. For `topk`, `extractive`, and `llmlingua`, the budget enters as the
   retention/compression target. Expose it to the fork as
   `GR_TOKEN_BUDGET=<B_tok>` in `env:`, and have each switch value keep/compress
   until the retained context is under `B_tok - (reserve for synthesis output)`.
3. Re-run those rows at the matched budget. Report those numbers in Table 4;
   keep the unmatched runs for the Figure 2 quality-vs-budget curve.

The harness `budget.*` fields are recorded but not enforced for
`gpt_researcher`; do not rely on them to implement the cap — the fork switch
must do it.

---

## 9. Trajectories for RL (paper §4.5)

Every run in §5 doubles as corpus generation. Nothing extra to run; here is
what each query yields and what is still missing for training.

**Per query, already produced**

- `gpt_researcher/<id>.json` — the state/action/cost sequence. Per round:
  `new_item_ids`, `retained_ids`, `decision.{type, kept_item_ids,
  pruned_item_ids, branch_allocation}`, `frontier[].{node_id, subquery,
  parent_subquery, status}`, `round_cost.{tokens_input, tokens_output,
  latency_seconds, llm_calls, search_calls}`. Plus `evidence[item_id]`
  with `content`, `source_url`, `source_subquery`, `tree_depth`,
  `retrieval_round`, `was_retained`, and `synthesis_cost`, `final_context`,
  `report`.
- `gpt_researcher/<id>_emb.npz` — `vectors[n,1536]` aligned to `ids`, plus
  `query`, `subquestions[M,1536]`, `nodes[|V|,1536]` aligned to `nodes_ids`.
  This is sufficient to compute the paper's per-item features offline with no
  model calls: ρ (cos to `query`), ν (1 − max cos to previously retained), δ
  (near-duplicate density over the pool), κ (fraction of `subquestions` above
  τ_c), ℓ (length of `content`), d (`tree_depth`), α (rounds since
  `retrieval_round`), σ (from `source_url`).
- `trajectories/<id>.json` — the harness view; one `prune` step per round
  with the same ids in `extra`, and the `write` step's cost.
- After §6, `metrics/summary.json` — `race_overall_score` is the terminal
  quality `Q(y, q)` for the reward in eq. 14, per run. Per-query RACE scores
  are in the checkout at `results/race/<model_name>/raw_results.jsonl`; copy
  them next to the trajectory or you will lose the per-query pairing.

**Not yet produced; needed before Stage 1 (behavior cloning)**

- *Randomized rollouts.* The rows in §4 are fixed policies. The corpus needs
  the fork's knobs sampled per run — keep ratio, `DEEP_RESEARCH_BREADTH`,
  `DEEP_RESEARCH_DEPTH`, stopping threshold. Add a `GR_ORCHESTRATOR=random`
  switch that samples these once per query and logs the sampled values into
  the raw trajectory so they can be recovered.
- *Counterfactual forks.* Replaying from a cached prefix with one different
  action requires a retrieval cache keyed on search query, and memoized worker
  outputs keyed on (subquery, retained-context hash). Tavily results are not
  reproducible, so cache them on first sight — this is the single largest
  piece of missing infrastructure for §4.5.
- *Parent links in the harness trajectory.* The adapter drops
  `parent_subquery`, and `compact_stats()` drops `parent_id`, so the harness
  `Trajectory` cannot reconstruct the subtask tree; the raw file can (resolve
  `parent_subquery` by prefix-matching the text after `Previous research goal:`
  against known node `subquery` strings). Use the raw file for tree features
  until the adapter is fixed.
- *Per-query reward assembly.* A script that joins raw trajectory + `_emb.npz`
  + per-query RACE score + `round_cost` into one record per orchestration step
  with `(s_t, a_t, R(τ))`. Does not exist.

Keep `keep_trajectory_files: true` in the agent config (default) or the raw
files stay in `trajectory_dir` and never enter the run directory.

---

## 10. Checklist before the numbers go in the paper

- `config.yaml` in every run dir is identical across rows except
  `env.GR_ORCHESTRATOR` (and `GR_TOKEN_BUDGET` for matched rows).
- Every row has 3 judged seeds (`n = 3` in the §7 output).
- `official.deep_research_bench.reason` is `null` in every `summary.json`.
- `n_errors` is 0 in every `summary.json`; if not, the failed queries' reports
  are empty strings and RACE scored them anyway.
- Judge model strings copied from `~/deep_research_bench/utils/api.py` into
  the paper's appendix.
- `git rev-parse HEAD` of both this repo and `third_party/gpt-researcher`
  recorded alongside the run directories.
