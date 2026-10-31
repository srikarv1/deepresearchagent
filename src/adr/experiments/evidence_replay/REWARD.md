# Offline page-selection proxy, v2 (Phase 3)

This is a hypothesis about evidence utility, not a report-quality score. No report,
reference answer, judge outcome, trajectory cost, or synthesis success enters the
reward. The completed-report field is checked only to establish source completion.
No RL training is implemented.

## Definition and normalization

Given a fixed query pool, normalize each saved embedding to unit length. For each
target (root query and each saved subquestion), compute nonnegative page cosine
similarities. Near-duplicate connected components are frozen over the full pool
using cosine >= .90 or normalized exact-text identity. Coverage takes the sum of
the best affinity from each of up to three selected distinct components, divided
by the analogous best-three sum in the complete pool, separately for every target.
Missing selected support groups contribute zero. Combine half root coverage with
half the mean subquestion coverage. Components with zero best target affinity are
allowed, but a target with no positive evidence anywhere causes a load failure.
All ratios are bounded by one. This is relative topical support, not entailment.

The configurable three-group requirement is a corroboration/breadth prior, not a
measured optimum: one topical page should not saturate a multi-source research
objective. It gives diminishing returns after three independent groups per facet.
`support_groups=1` reproduces the initial max-affinity coverage ablation. That v1
scored a singleton above full-budget top-k on all eight development queries, so it
was rejected before training. Its untouched artifacts remain in diagnostics-001;
v2 rescores the identical frozen masks in diagnostics-002. The old max-only score
is also logged as `max_coverage_ablation`. Grouping is entirely embedding/text
based and does not use source-host diversity or downstream quality. Transitive
near-duplicate clustering can merge legitimately different evidence; the three-
group assumption must be validated. It is not a hard minimum page-count penalty.
Missing, zero or nonfinite vectors are rejected and never synthesized. Only the
current query's pool is used; root weight stays constant as subquestion count varies.

Diversity is coverage times source-host Shannon entropy divided by log(number of
unique hosts in the complete pool). A one-host pool receives zero diversity bonus.
This is *source balance*, an inexpensive complementary heuristic, not certified
semantic diversity. Coverage gating prevents unrelated multi-host selections from
receiving the full diversity bonus. Hosts remove a leading `www.` only.

Redundancy is the selected-page mean of each page's worst excess cosine above .90,
linearly rescaled from [.90,1] to [0,1]. The diagonal is excluded and singleton
redundancy is zero. Whitespace/case-normalized exact text duplicates count as one
regardless of potentially inconsistent cached vectors. The .90 threshold targets
near copies rather than all on-topic pages; it is a configurable constructor
parameter, not learned from report outcomes. Diversity uses source provenance and
redundancy uses semantic/text duplicates, so neither rewards the other's inverse.
A duplicated page on another host can affect both, but the default duplicate
penalty outweighs the possible source bonus.

Cost is `actual formatted_context_tokens / budget`, capped at one. The exact same
UTF-8 page/source wrappers, ordering and tokenizer as replay are used, with no
character approximation or hidden truncation. The unclipped ratio is also logged.

All four components are in [0,1]. For nonempty feasible selections:

    R = w_coverage * coverage + w_diversity * diversity
        - w_redundancy * redundancy - w_token_cost * token_cost

Weights are nonnegative and sum to one, making the formula lie within [-1,1].
Nonempty feasible masks have a higher score than the -1 constraint sentinel.
Empty and over-budget masks receive -1, with their components and constraint reason
still logged. These constraint overrides are part of the reward definition; the
empty mask does not silently receive formula-zero. KEEP-ALL is infeasible in the
real-source pools at 50K, not a free way to achieve maximum coverage. A pool that
entirely fits is not forbidden from being fully retained when it is useful.

## Candidate weights, declared before downstream quality validation

| Name | Coverage | Diversity | Redundancy | Token cost |
|---|---:|---:|---:|---:|
| coverage_first (initial default) | .70 | .10 | .15 | .05 |
| anti_redundancy | .60 | .10 | .20 | .10 |
| source_balance | .60 | .15 | .15 | .10 |

These are explicit design priors, not empirically calibrated optimums. Coverage is
the primary surrogate because report utility requires topic/facet support; the
source bonus is smaller because domain count is a weak quality signal. A duplicate
penalty at least as large as the source bonus discourages adding copies on new
hosts. A small token penalty breaks ties toward smaller contexts; a hard budget
already controls feasibility, so cost must not buy arbitrarily poor evidence.
The latter two candidates test whether the initial choices are fragile, increasing
cost and either duplicate aversion or source balance. Auxiliary weights are capped
at .25 and coverage is constrained to at least .50, avoiding a cost-only or
source-count-only objective. Components remain individually logged: normalized
bounds do not guarantee equal empirical variance or a useful training signal.

A custom JSON mapping of candidate names to the four weight fields may be passed
via `--weights`. Do not tune weights on the final report-quality evaluation set.

## Running and interpreting diagnostics

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m adr.experiments.evidence_replay.reward_diagnostics \
  --output runs/evidence-replay-phase3/proxy-diagnostics-003
```

Use a fresh output directory. This evaluates each of the eight distinct real-source
trajectories at the same 50K cap using the existing random, chronological
cosine-threshold (.35), and descending-cosine top-k selectors. Random seeds 0–4 are
fixed, not five independent queries. Each selector's random values are averaged
within query before macro averaging across queries. KEEP-NONE, KEEP-ALL, and top-k
selected under a half cap but scored under the same 50K reward budget are probes.
They are not mixed into the primary correlation analysis.

`records.json` holds all components, masks/selection identities, context hashes,
actual tokens and rewards under all candidates. `comparison.csv`/`summary.json`
hold query-macro means. `component_distributions.png` shows the distribution of
query-level selector values, with random masks averaged. `weight_sensitivity.json`
records macro scores for every candidate. `diagnostics.json` contains query-wise
Pearson/Spearman correlations with mean selected root cosine, similar-cosine mask
pairs, weight-sensitive rankings, and comparisons to top-k within 2% of the cap
(1,000 tokens). This is approximate token matching, not forced truncation. Keep
actual utilization visible; no claim of exact equality is made. Correlations use
only feasible nonempty primary selector masks, and are descriptive on seven masks
per query. They are not significance tests using pages as independent samples.

The synthetic tests deliberately hold root-query cosine fixed while changing
subquestion coverage, and separately create duplicate selections. These establish
that the formula is not algebraically identical to root cosine ranking. The real
similar-cosine diagnostic is needed to see whether this distinction occurs in the
available data. Neither diagnostic establishes improvement in actual reports.

## Pairing with future report quality

`pair_quality(proxy_records, quality_records)` joins by `selection_id`, which binds
the source trajectory hash/ID, ordered-independent mask, encoding and reward budget.
The exact context hash must also match. A quality record supplies selection_id,
context_sha256, quality_version and a mapping of finite numeric metrics. Duplicate
masks must be deduplicated before joining; multiple judge/report replicates should
first be aggregated in an auditable external quality record. Report hashes, judge
configuration, source/date and replicate metadata may be retained in that record.
No API client is created by the join function. Quality outcomes are kept outside
the proxy scorer and must never become policy inputs.

Before training, freeze a small candidate set on development queries. On held-out
query families, choose matched-cost masks covering high/low proxy scores and
similar-cosine disagreements. Synthesize identical-prompt reports for each mask,
then blind the evaluator to selector/proxy scores. Use official DRB only for exact
prompt/reference/rubric matches; custom real-source questions need a separately
specified reference-free rubric or authored references. Analyze paired quality
differences per query and proxy/quality rank consistency. Keep final evaluation
queries out of weight calibration. Report uncertainty and failure/abstention rates;
a handful of queries supports a pilot rather than a generalization claim.

## Limitations before REINFORCE

Embedding affinity measures topicality, not factual entailment, reliability,
recency, citation correctness or contradiction. The root/subquestion targets may
be semantically repetitive. Pool-best normalization can make weak pools look well
covered and max-only coverage saturated after one page in the initial ablation. The revised
three-group coverage can still saturate before important report details are covered. The source-entropy term
can favor low-value sources, subdomains or hosts represented by many cached pages.
The .90 duplicate threshold may miss paraphrases or penalize necessary corroboration.
Near-full budget selectors provide little cost variation. Cached pools reflect
historical retrieval and page aggregation, and cannot represent policy-induced
retrieval changes. Reward sensitivity and component distributions should expose
these issues; downstream report validation is required before trusting the proxy
as an RL objective. No counterfactual report rewards are assumed to exist.

For an existing replay (including the already successful smoke),
`score_saved_replay(Path(replay_dir))` computes the same proxy record, checks the
source trajectory/embedding hashes and exact context/token identity, and reads no
report or synthesis outcome. Its `selection_id` and `context_sha256` can be copied
into a future externally produced quality record before `pair_quality` joins it.

After a diagnostics run, add the inexpensive singleton/variance audit without
rerunning all selectors:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m adr.experiments.evidence_replay.reward_diagnostics \
  --output runs/evidence-replay-phase3/proxy-diagnostics-003 --audit-only
```

It refuses an existing `audit.json`, tests whether a single best-cosine page beats
the full top-k mask, and records each weighted component's within-query spread.
This reveals coverage saturation and empirical component dominance despite bounded
terms. The spread fractions are descriptive rather than causal importance scores.

The actual Phase 3 outputs are `runs/evidence-replay-phase3/proxy-diagnostics-001`
(rejected max-only v1) and `proxy-diagnostics-002` (recommended v2). They must not be
overwritten. To rescore frozen masks into a fresh directory rather than rerun
selection, add `--reuse-masks runs/evidence-replay-phase3/proxy-diagnostics-001/records.json`.
The rescore verifies trajectory hash, budget, encoding and exact context hashes.
The v2 run averages 49,956 random, 49,952 threshold and 49,975 top-k context tokens;
mean rewards with coverage-first weights are .5753, .6180 and .6105 respectively.
All 48 comparisons against top-k match within 135 tokens. Singletons score below
top-k for all eight queries in v2; both KEEP-NONE and infeasible KEEP-ALL score -1.
These eight development queries were inspected during reward design; none is an
independent final test set. Coverage accounts for about 73% of weighted within-query
spread with default weights, redundancy 20%, source diversity 7%, and cost <.1% at
matched near-full budgets. Broader mask sizes are needed to measure the cost term.

For downstream validation, the provisional recommendation is v2 with
`CANDIDATES['anti_redundancy']` (.60, .10, .20, .10), retaining coverage-first as a
fixed ablation. The recorded `reward` field/default is still coverage-first for
reproducibility; the recommended value is in `rewards.anti_redundancy` or obtained
by `total_reward(score, CANDIDATES['anti_redundancy'])`. Its within-query spread is
about 65% coverage, 28% redundancy, 7% diversity and <.1% cost on these matched-cap
masks. This reduces concentration in coverage without making the weak source
heuristic large. Singleton scores remain below full top-k on all eight queries
under all three weight sets. It is a prior for a validation pilot, not a weight
choice optimized against report quality; no such quality labels were used.
