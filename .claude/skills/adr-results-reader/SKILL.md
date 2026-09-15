---
name: adr-results-reader
description: Read, analyze, and explain deepresearchagent (adr) harness run results. Use when the user asks about run outputs, trajectories, evidence, rounds, RACE/FACT scores, embedding analysis, or anything under runs/ or a zip from an adr run. Also use when the user mentions trajectory steps, stats_before, frontier nodes, evidence items, or npz files.
---

# ADR Results Reader

## Run directory structure

Every run directory follows this layout:

```
<timestamp>-<run_name>/
  config.yaml                       # Frozen config for this run
  manifest.json                     # Git sha, timestamp, package versions
  queries/<id>.json                  # Input query (id, text, dataset, language, topic)
  trajectories/<id>.json             # Harness-format trajectory (steps, stats_before, report)
  gpt_researcher/<id>.json           # Raw fork trajectory (rounds, evidence, subquestions)
  gpt_researcher/<id>_emb.npz       # Embedding vectors (evidence, query, subquestions, frontier nodes)
  reports/<id>.md                    # Generated markdown report
  metrics/local.json                 # Per-query computed metrics
  metrics/summary.json               # Aggregate metrics + official judge scores
  exports/deep_research_bench/       # JSONL for DRB RACE + FACT judges
  errors/                            # Stack traces for failed queries
```

## Two trajectory formats

There are two trajectory files per query. They describe the same run but at different levels.

### Harness trajectory (trajectories/<id>.json)

Top-level: `query`, `steps[]`, `report`, `final_stats`, `error`.

Each gpt-researcher round (= one DFS node = one `deep_research()` call) becomes one step:

```
step.action.type         "prune" for each round, "write" for report, "terminate" at end
step.action.evidence_ids IDs pruned this round (empty if policy keeps everything)
step.action.rationale    Human-readable summary, e.g. "round 2: EmbeddingsFilter kept 17, pruned 21"

step.stats_before        State snapshot BEFORE this round executes; only contains data from prior rounds
  .n_evidence            Evidence count from prior rounds
  .n_retained            Retained count from prior rounds
  .n_pruned              Pruned count from prior rounds
  .n_open_branches       Frontier nodes from prior rounds
  .budget                Remaining steps, tokens, searches, reads, latency
  .branches[]            Subtasks with id, goal text, status, priority

step.observation         "new=N retained_after=M search_calls=S llm_calls=L"
step.tokens              {prompt_tokens, completion_tokens, total_tokens} for this round only
step.latency_s           Wall seconds for this round only

step.extra               Raw RoundSnapshot from the fork (always reliable)
  .round_id              Preorder DFS position (1-indexed)
  .decision_type         "continue" or "terminate" (last round only)
  .kept_item_ids[]       Evidence kept by orchestration policy
  .pruned_item_ids[]     Evidence pruned
  .new_item_ids[]        Evidence first discovered this round
  .n_retained_after      Global retained count after this round
  .search_calls          Real search API calls (cache hits excluded)
  .llm_calls             LLM calls this round
  .frontier[]            This round's SERP queries with:
    .node_id             Hash of researchGoal; appears in npz nodes_ids
    .status              "open" or "completed"
    .subquery            The actual research topic text
    .parent_subquery     Parent node's query (first 200 chars)
```

### Step types

```
prune       One per deep_research() round/node; extra.round_id, kept/pruned ids
write       After all rounds, report generation; action.report_draft, tokens
terminate   Final step, marks end of run
```

### Raw fork trajectory (gpt_researcher/<id>.json)

The full `Trajectory` dataclass from `TrajectoryLogger.save()`. Contains:

```
query_id                 uuid4 hex[:12]
subquestions[]           From generate_research_plan (top-level research questions)
rounds[]                 One RoundSnapshot per deep_research() call
  .round_id, .timestamp
  .new_item_ids[], .retained_ids[]
  .decision              {type, kept_item_ids, pruned_item_ids, branch_allocation}
  .frontier[]            {node_id, subquery, parent_subquery, status}
  .round_cost            {tokens_input, tokens_output, latency_seconds, llm_calls, search_calls}
evidence{}               Keyed by item_id, each item has:
  .item_id               md5(url + ":" + sha256(content)[:16])[:12]
  .content               Deduplicated chunk texts joined
  .source_url, .source_subquery
  .tree_depth            1 = root level, increases with recursion
  .retrieval_round       First round this item appeared (first insertion wins)
  .was_retained          True if any sub-query's EmbeddingsFilter kept any chunk
  .word_count            len(content.split())
  .embedding             Always None in JSON (stored in npz)
synthesis_cost           RoundCost for write_report
total_tokens, total_cost, total_latency, peak_context_length
evidence_total, evidence_retained_final, fraction_pruned
report                   Full report text
```

## NPZ embedding file (gpt_researcher/<id>_emb.npz)

Load with `np.load(path, allow_pickle=True)`. All vectors are unit-normalized (dot = cosine).

```
ids           (N,)       <U12     Evidence item IDs, same order as vectors
vectors       (N, 1536)  float32  Evidence embeddings
query         (1, 1536)  float32  Root query embedding
subquestions  (S, 1536)  float32  Subquestion embeddings
nodes         (F, 1536)  float32  Frontier node embeddings
nodes_ids     (F,)       <U12     Frontier node IDs (match frontier[].node_id)
```

For common npz operations (ranking evidence, frontier alignment, coverage analysis, retained vs pruned separation, single-item lookup), see `scripts/npz_analysis.py`.

## Metrics (metrics/summary.json)

```
n_queries, n_errors
mean_tokens, mean_prompt_tokens, mean_completion_tokens
mean_wall_s
mean_n_retained, mean_n_pruned, mean_prune_rate
mean_n_citations, mean_article_chars
scores.race_overall_score          RACE judge overall (0-1)
scores.race_comprehensiveness      RACE sub-dimension
scores.race_insight                RACE sub-dimension
scores.race_instruction_following  RACE sub-dimension
scores.race_readability            RACE sub-dimension
scores.fact_valid_rate             FACT citation validity (0-1)
```

## Common analysis patterns

1. **Why was an item pruned?**: Look up item_id in both the fork JSON (metadata: source_url, tree_depth, was_retained) and npz (cosine to query, closest frontier node, nearest retained item). See `scripts/npz_analysis.py`.

2. **Budget utilization**: Check `stats_before.budget` on the last prune step.

3. **Frontier balance**: Use npz to compute `nodes @ subquestions.T`; shows which frontier nodes cover which subquestions. Uneven coverage means lopsided exploration.
