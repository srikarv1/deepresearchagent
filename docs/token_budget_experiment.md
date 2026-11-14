# Token-budget experiment (150K tokens per query)

Every method gets the same total token budget for one query, **150,000 tokens**,
and we compare report quality. The budget covers everything a run spends:
research and the final report, on GPT-6.1 Sol and on the orchestrator model.

Methods (arms): `none` (no pruning), `prompted` (prompted orchestration),
`extractive` (extractive compression), `bc` (PILOT BC), `dpo` (PILOT DPO).

Benchmarks: `drg` (DeepResearchGym, the 50 pilot queries) and `drb`
(DeepResearch Bench, queries 61 to 75). Both use research depth 2, breadth 4.

## How the budget works

The budget is a stopping rule inside gpt-researcher (`GR_STOP_TOTAL_TOKENS`,
fork PR #12), the same for every arm. Before each new round and each new
sub-query, the run checks:

    tokens used so far + one more step + expected cost of the report  >  150,000 ?

"One more step" is the largest sub-query plus round overhead seen so far.

If so, research stops and the report is written from the evidence gathered.
Every run still produces a report.

- The expected report cost (`report_reserve_tokens: auto`) is estimated from
  the evidence that would go to the writer. A method that prunes more leaves a
  smaller report prompt, so it can spend more of the 150K on research: an
  equal *total* budget. No arm gets a different rule. (An integer instead of
  `auto` reserves a fixed amount: an equal *research* budget.)
- It is a stopping rule, not a hard cutoff. A run can end a few thousand tokens
  over. The table reports how many runs were stopped and how many ended over.
- BC and DPO are the trained adapters as they are. Nothing is retrained or
  tuned for the budget.

Each trajectory records what happened in `final_stats.token_stop`
(`stopped`, `stopped_at`, `tokens_at_stop`, `report_reserve_at_stop`,
`final_tokens`, `report_tokens`, `ended_above_target`, `overshoot_tokens`).

## Setup

You need the harness on `main` and the gpt-researcher fork on
`aadya/drb-azure-flex` (Sol on Flex plus the stopping rule). The run script
refuses to start on a fork without them.

```bash
git switch main && git pull
git -C third_party/gpt-researcher fetch origin
git -C third_party/gpt-researcher switch aadya/drb-azure-flex
git -C third_party/gpt-researcher pull
source .venv/bin/activate
pip install -e ".[dev]" && pip install -e third_party/gpt-researcher
```

`.env` in the repository root (never commit it):

```dotenv
# Sol, for generation. Either this pair ...
SOL_API_KEY=<key of the resource that hosts gpt-6.1-sol>
SOL_BASE_URL=<its endpoint, ending in /openai/v1/>
# ... or AZURE_OPENAI_API_KEY=<key> (uses the 202610-deepres endpoint).

# Judge (gpt-5-mini), for scoring. Either this pair or OPENAI_API_KEY + OPENAI_BASE_URL.
JUDGE_API_KEY=<key of the gpt-5-mini resource>
JUDGE_BASE_URL=<its endpoint, ending in /openai/v1/>

DEEPRESEARCHGYM_API_KEY=<key>      # drg search
TAVILY_API_KEY=<key>               # drb search (or TAVILY_KEYS_CSV=<path>)
JINA_API_KEY=<key>                 # drb FACT judge

# bc and dpo only: the vLLM server that serves both adapters.
GR_ORCH_LLM_BASE_URL=<tunnel URL ending in /v1>
```

For `bc` and `dpo`, start the adapter server first
(`notebooks/wsp_qwen_orchestrator.ipynb`, serve and tunnel cells only) and put
the tunnel URL in `GR_ORCH_LLM_BASE_URL`. It must list `pilot-bc-teacher` and
`pilot-dpo-teacher`; the run script checks this before it starts.

## 1. Generate reports

One command per arm. Run arms in separate terminals if you want them in
parallel.

```bash
bash scripts/sol_budget.sh drg none
bash scripts/sol_budget.sh drg prompted
bash scripts/sol_budget.sh drg extractive
bash scripts/sol_budget.sh drg bc
bash scripts/sol_budget.sh drg dpo
```

DeepResearch Bench is the same with `drb`:

```bash
bash scripts/sol_budget.sh drb none
bash scripts/sol_budget.sh drb bc
# ... prompted, extractive, dpo
```

To run only some queries, add a comma-separated list:
`bash scripts/sol_budget.sh drg bc 546111,442352`.

- **Resuming.** Finished queries are skipped. After any interruption (laptop
  slept, tunnel died, Ctrl-C) run the same command again.
- **Output.** `runs/gym_sol_budget150/<arm>/<query>-s0/` for drg,
  `runs/drb_sol_budget150/<arm>/<query>-s0/` for drb. Every trajectory
  carries one record per LLM call (`final_stats.llm_calls`), which step 3
  needs for latency.
- **Try one query first.** `bash scripts/sol_budget.sh drg none 546111`, then
  check `final_stats.token_stop` in
  `runs/gym_sol_budget150/none/546111-s0/trajectories/546111.json`.

### Azure Flex

Sol always runs on Azure Flex here (half price); the fork forces it. Do not
switch tiers; the run script clears `LLM_KWARGS` so nothing can override it.

Flex can be slow. When Azure has no capacity it refuses the request
(`429 no_capacity`) and the run keeps retrying that call, without using up its
normal attempts, for up to an hour (`GR_LLM_CAPACITY_MAX_WAIT_S`). A request
that hangs is abandoned after 300 s (`GR_LLM_REQUEST_TIMEOUT_S`) and retried.
You will see `LLM capacity unavailable (429 no_capacity ...)` in the logs; that
is normal. This waiting is removed from latency in step 3. Azure bills tokens,
not waiting time.

## 2. Judge

Uses the judge credentials (gpt-5-mini), not Sol.

```bash
bash scripts/sol_budget_judge.sh drg            # every arm that has reports
bash scripts/sol_budget_judge.sh drb none bc    # only these arms
JOBS=3 bash scripts/sol_budget_judge.sh drg     # 3 reports at a time (default 2)
```

Already judged reports are skipped. Run it again until it says
`0 report(s) to judge`: a report that hit a rate limit (429) is retried on the
next run. On drg, Azure's content filter can block key-point recall for a few
queries (65128, 686941); those stay unjudged unless the judge deployment's
content filter is set to block High only.

## 3. Take Flex waiting out of latency

```bash
python scripts/flex_queue_adjust.py runs/gym_sol_budget150
python scripts/flex_queue_adjust.py runs/drb_sol_budget150
```

For every Flex call this removes (a) the time spent being refused and retried,
measured exactly (`capacity_wait_s` in the call log), and (b) the extra time an
accepted call waited, estimated as its duration minus what standard tier takes
for the same input and output tokens. (b) matters because Sol sends nothing
until the answer is ready, so queueing after Azure accepts a call is not
visible any other way; `llm_call_summary.wall_s_excl_capacity_wait` in the
trajectory removes only (a). Overlapping waits are counted once. It writes
`<run root>/queue_adjusted.json`.

## 4. Build the table

```bash
python scripts/budget_table.py drg
python scripts/budget_table.py drb
```

It prints one row per arm and, for every arm, the per-query paired difference
against No pruning with a bootstrap 95% interval. Extra columns:

- **Stopped**: runs the stopping rule stopped, out of runs with the rule on.
- **Over**: runs that ended above 150K (`ended_above_target`).

It warns about reports without all judge scores (left out) and about runs
starved of evidence: the first round, or at least half the rounds, found
nothing. That happens when the search service is down, and such a run looks
cheap for the wrong reason. Rerun those queries (delete the
`<query>-s0` directory, then run the same generate command), or pass
`--drop-starved`.

## Sending results

Send the whole arm directory; do not send `.env`.

```bash
tar -czf drg_budget150_<arm>.tgz runs/gym_sol_budget150/<arm>
```

To add someone else's arm to your table, unpack it so it lands at
`runs/gym_sol_budget150/<arm>/`, then rerun steps 2 to 4.

## Changing the budget

The budget is `orchestration.stop_total_tokens` (and the reserve
`orchestration.report_reserve_tokens`) in
`configs/agents/gpt_researcher_gym_gptsol_d2_budget150.yaml` (drg) and
`configs/agents/gpt_researcher_bench_budget150.yaml` (drb). The two differ only
in the retriever. Keep one run root per budget: results from different budgets
must not share a directory.
