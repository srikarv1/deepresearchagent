# HOWTO: The same approaches on BrowseComp

Goal: a table with the same rows as Table 4 (No pruning, Top-k similarity,
Extractive compression, Prompt compression, Prompted orchestration, PILOT)
evaluated on BrowseComp, with BrowseComp's own quality metric (accuracy) and
the same efficiency columns as Table 3 (Tokens (K), Latency (s), Calls,
Q per 1K tok).

Everything about the *agent side* is identical to `HOWTO_DRB.md`: same
`gpt_researcher` backbone, same fork switch `GR_ORCHESTRATOR`, same run naming,
same seeds. Read that document first; this one covers only what BrowseComp
changes. Nothing here runs inference.

---

## 0. Status

**Nothing BrowseComp-specific exists in the harness.** There is no dataset
loader, no exporter, no judge, no answer-extraction step, and no config. §2 is
the build spec; §3 onward is how to run once it is built.

The reference implementation is OpenAI's
[`simple-evals/browsecomp_eval.py`](https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py).
Every protocol detail below is taken from it verbatim so numbers are
comparable to published ones. Do not paraphrase the prompts.

---

## 1. The protocol (from `simple-evals`)

**Dataset.** 1,266 questions at
`https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv`
(1.2 MB; verified 2026-09-08). Columns: `problem`, `answer`, `problem_topic`,
`canary`. `problem` and `answer` are base64-encoded and XOR-encrypted with a
key derived from `canary`; `problem_topic` is **plaintext** and `canary` is
the same string for every row (`BENCHMARK DATA SHOULD NEVER APPEAR AS
PLAINTEXT ...`). Decrypted, questions average 103 words (max 354) and answers
2.7 words (max 39), which is why the exact-match protocol below works.

Topic distribution (use it for a stratified subset if 200 is too few):
TV shows & movies 205 · Other 197 · Science & technology 173 · Art 127 ·
History 125 · Sports 123 · Music 116 · Video games 71 · Geography 70 ·
Politics 59.

Decryption (verified against the first rows):

```python
import base64, hashlib

def derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()
```

The encryption exists to keep the answers out of web crawls. Commit the CSV
encrypted, decrypt at load/judge time, and never write decrypted **answers**
under a tracked path.

**Subsampling.** `simple-evals` draws `num_examples` with
`random.Random(0).sample(examples, num_examples)`. Use exactly that so a
"200-question subset" means the same 200 questions as anyone else's.

**Query template** sent to the system under test:

```
{Question}

Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
```

**Grader template** (LLM judge), filled with `question`, `response`,
`correct_answer`:

```
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
```

**Scoring.** `re.search(r"correct: (yes|no)", grading_response)`; anything
else counts as `no`. Accuracy = mean of `correct == yes`. The reference code
does not pin a grader model — pick one, record it in the config, and keep it
fixed across all rows and seeds.

---

## 2. Build spec

Seven pieces. File paths follow the existing layout so each one has an
obvious template next to it.

### 2.1 Dataset file

`data/benchmarks/browsecomp/browse_comp_test_set.csv` — the encrypted CSV,
downloaded once, committed as-is (the DRB and Gym query files are committed
the same way). Record the download date in `data/benchmarks/browsecomp/README.md`.

### 2.2 Loader — `src/adr/datasets/loader.py`

- Add `DatasetName.BROWSECOMP = "browsecomp"` and a `BC_QUERIES` path.
- Row → `Query(id=f"bc_{row_index:04d}", text=<decrypted problem>,
  dataset="browsecomp", language="en",
  metadata={"row": row_index, "topic": row["problem_topic"]})`. The topic
  is plaintext in the CSV and safe to carry; it lets `adr table` break
  accuracy down by topic later.
- **Do not put the decrypted answer in `metadata`.** `queries/<id>.json` is
  written to the run dir and the agent receives `task.query`; the answer must
  be unreachable from the agent. The judge re-reads the CSV by `row` at
  scoring time.
- Add a `sample: int | null` dataset option that applies
  `random.Random(0).sample(...)` before `limit` / `query_ids`, to match
  `num_examples` in the reference code.

### 2.3 Answer extraction — `src/adr/agents/gpt_researcher.py`

gpt-researcher writes a report; BrowseComp grades a response that contains an
`Exact Answer:` line. Add a config key `answer_mode: browsecomp` (default
`null`, so DRB/Gym behavior is untouched). When set, after `write_report()`:

1. Make **one** more LLM call through the fork's own LLM (so its tokens land
   in `TokenTracker` and count toward the row's cost like every other call),
   with the query template from §1 and the report as context, asking for the
   `Explanation / Exact Answer / Confidence` block.
2. Record it as an additional `write` step in the harness trajectory with
   its token/latency cost, `rationale="browsecomp answer extraction"`.
3. Keep `traj.report.article` as the full research report (so local metrics
   and `adr dag` are unchanged) and put the response in
   `traj.final_stats["answer"] = {"response": ..., "exact_answer": ...,
   "confidence": ...}`, parsed with tolerant regexes.

The extraction call is identical across rows, so it does not bias the
comparison; it does add a fixed cost that should be stated in the paper.

### 2.4 Exporter — `src/adr/eval/exporters.py`

`export_browsecomp(trajectories, dest) -> Path`: one JSON object per line,
`{"id": ..., "row": ..., "problem": ..., "response": ...}`, where `response`
is `final_stats.answer.response` (empty string if the agent failed). Written
to `runs/<run>/exports/browsecomp/<model_name>.jsonl`.

### 2.5 Judge — `src/adr/eval/browsecomp.py`

`run_browsecomp(trajectories, *, run_dir, model_name, judge_model, csv_path,
timeout_s) -> dict`, shaped like the other bench results
(`{"bench": "browsecomp", "official": bool, "reason": str | None, ...}`):

- Decrypt the correct answer for each trajectory's `row` from the CSV.
- Build the grader prompt from §1, call the judge through the existing
  OpenAI-compatible client (`src/adr/llm/openai_compat.py`), temperature 0.
- Parse `correct: (yes|no)`, `extracted_final_answer:`, `confidence:`.
- Write `runs/<run>/metrics/browsecomp/grades_<judge_model>.jsonl` with one
  record per query: `{id, row, correct, extracted_final_answer, confidence,
  reasoning, judge_raw}`. Keep `judge_raw`; disputes are settled from it.
- Aggregate: `n`, `accuracy` (0–100), `n_no_answer` (extracted answer is
  `None`), `mean_confidence`, and calibration error (reference: bucket by
  confidence, mean |accuracy − confidence| weighted by bucket size).
- Gate on `OPENAI_API_KEY` (or the judge's `api_key_env`) and return a
  `reason` string when missing, matching how the DRB and Gym judges fail.

### 2.6 Plumbing

- `src/adr/eval/scoring.py::headline_scores` — flatten `browsecomp_accuracy`,
  `browsecomp_calibration_error`, `browsecomp_n_no_answer`.
- `src/adr/runner/experiment.py` — in the export dispatch, add a
  `browsecomp` branch; in `_run_official`, `if bench in {"browsecomp", "bc"}`.
- `configs/eval/browsecomp.yaml` — `judge_model`, `csv_path`, `timeout_s`.
- `adr score --dataset browsecomp --query-id bc_0042 --report answer.txt`
  should work via `resolve_question` in `src/adr/eval/importers.py` for
  spot-checking a single response.

### 2.7 Tests — `tests/test_browsecomp.py`

- Encrypt two synthetic rows with a fake canary using the §1 functions and
  assert the loader round-trips them and never emits the answer in
  `Query.metadata`.
- Drive `run_browsecomp` against a stub judge (reuse the pattern in
  `tests/mock_judge.py`) that returns a canned `correct: yes` / `correct: no`
  and assert the aggregate is exactly 50.0.
- Assert `sample: 5` with `Random(0)` selects the same five row indices the
  reference code would.

---

## 3. Run config

`configs/table_bc/bc_base.yaml`, identical to `configs/table4/drb_base.yaml`
from `HOWTO_DRB.md` §3 except:

```yaml
run_name: tablebc-<row>-s<seed>
dataset:
  name: browsecomp
  sample: 200                 # Random(0) subset; see §5 for why not 1,266
  limit: null
  query_ids: []
agent:
  name: gpt_researcher
  config: configs/agents/gpt_researcher_bc.yaml   # = gpt_researcher.yaml + answer_mode: browsecomp
eval:
  official_benches: []        # judge with `adr evaluate`, as for DRB
```

`configs/agents/gpt_researcher_bc.yaml` must be byte-identical to the DRB
agent config except for the added `answer_mode: browsecomp` line, or the
"same approaches" claim does not hold.

---

## 4. Running

Identical loop to `HOWTO_DRB.md` §5 with the run name prefix changed:

```bash
for row in none topk extractive llmlingua prompted; do
  for s in 0 1 2; do
    GR_ORCHESTRATOR=$row adr run \
      --config configs/table_bc/bc_base.yaml \
      --run-name "tablebc-${row}-s${s}"
  done
done

for d in runs/*-tablebc-*; do
  adr evaluate "$d" --official browsecomp
done
```

Same rules: `concurrency: 1` per process, parallelize by splitting
`query_ids` across processes, judge by run dir so results do not collide,
start with `--limit 2`.

Additional artifacts per run beyond the DRB list:

```
exports/browsecomp/<model_name>.jsonl
metrics/browsecomp/grades_<judge_model>.jsonl
```

---

## 5. Filling the table

| Column | Field in `metrics/summary.json` | Notes |
|---|---|---|
| Accuracy | `scores.browsecomp_accuracy` | 0–100 |
| Tokens (K) | `mean_tokens / 1000` | includes the answer-extraction call |
| Latency (s) | `mean_wall_s` | |
| Calls | `mean_n_searches + mean_n_reads` | tool calls; cross-check against `total_tool_calls` in `gpt_researcher/<id>.json` |
| Q per 1K tok | `scores.browsecomp_accuracy / (mean_tokens / 1000)` | compute |

Reuse the aggregation snippet from `HOWTO_DRB.md` §7 with the regex changed
to `tablebc-(\w+)-s(\d+)$`, the quality key to `browsecomp_accuracy`, and the
derived column to Q per 1K tok. Report `n_no_answer` alongside accuracy: a
row that prunes too hard tends to fail by producing no answer at all, which
is a different failure than a wrong answer and the paper should say which.

**Scale.** The full set is 1,266 questions. At the cost observed on this
branch (~$0.65 and ~400 s per gpt-researcher query), 1,266 × 5 rows × 3 seeds
is roughly $12K and months of serial compute. Use `sample: 200` for the paper
table (≈$2K, ≈14 days serial, parallelizable), and `sample: 50` while
developing the fork switch. State the subset size and `Random(0)` in the
caption.

---

## 6. Caveats to decide on before spending

**Fit.** BrowseComp rewards persistent, adaptive multi-hop search for one
hard-to-find fact; OpenAI's own numbers are 1.9% for GPT-4o with browsing and
51.5% for Deep Research. gpt-researcher at `depth: 2, breadth: 2` does a fixed
tree of broad sub-queries and then synthesizes; it is not built for this and
will likely score low across **all** rows. If every row is near the floor the
table says nothing about orchestration. Mitigations, cheapest first:

1. Run `none` on `sample: 50` before anything else. If accuracy is under
   ~10%, stop and pick one of the options below rather than running the
   other rows.
2. Raise `depth` / `breadth` for BrowseComp only (documented in the caption).
   This changes the backbone relative to Table 4, so say so.
3. Switch to **BrowseComp-Plus** (Chen et al., 2025,
   [arXiv 2508.06600](https://arxiv.org/abs/2508.06600)): the same questions
   over a fixed, human-verified 100K-document corpus with a supplied retriever.
   This matches the paper's reproducibility story (it evaluates against
   DeepResearchGym's fixed corpus for the same reason), removes Tavily
   nondeterminism from the counterfactual-fork cache in `HOWTO_DRB.md` §9, and
   its published baselines span a usable range (3.9% to 70.1%). Cost: the
   fork's `retriever` must be pointed at the Plus corpus instead of Tavily,
   which is a new retriever class in the fork, not a config change.

**Retrieval nondeterminism.** With Tavily, two seeds of the same row see
different web results; part of the seed variance is the search engine, not
the agent. Note it in the appendix or use BrowseComp-Plus.

**Judge cost.** One grader call per question per run; with `sample: 200` and
15 runs that is 3,000 short calls, negligible next to the agents.

---

## 7. Trajectories for RL

Every BrowseComp run yields the same raw `gpt_researcher/<id>.json` and
`_emb.npz` as DRB (`HOWTO_DRB.md` §9), plus a terminal reward that is binary
(`correct`) rather than a rubric score. That is a cleaner training signal for
the goal-satisfaction term `G(y, q)` in eq. 14 than anything DRB provides,
and the paper's cross-benchmark transfer table (Table 11) needs trajectories
from more than one benchmark anyway. Keep `keep_trajectory_files: true`.

---

## 8. Checklist

- Prompts in `src/adr/eval/browsecomp.py` are byte-identical to
  `simple-evals/browsecomp_eval.py`.
- `sample` and judge model recorded in every run's `config.yaml`.
- No decrypted answer appears anywhere under `runs/` except
  `metrics/browsecomp/grades_*.jsonl` (which is judge output, not agent input).
- `gpt_researcher_bc.yaml` differs from the DRB agent config only by
  `answer_mode`.
- Same six rows, same three seeds, same `GR_ORCHESTRATOR` values as Table 4.
