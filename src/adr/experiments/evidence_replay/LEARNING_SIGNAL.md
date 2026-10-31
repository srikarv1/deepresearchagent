# Focused learning-signal diagnostic

This experiment changes only candidate ordering/filtering. It does not change the
Phase 4 trainer, baseline implementations, reward formula, weights, policy
architecture, feature definitions, optimizer, baseline decay or initialization.
No synthesis/evaluation adapter is imported. The command blocks outbound sockets.

## Prespecified controls

Every protocol uses training seeds 0,1,2, the existing source-family split (001–006
train, 007–008 validation), 20 epochs, six episodes per update, AdamW lr .003,
weight decay .0001, entropy coefficient .005, EMA decay .8, clipping 1, and the
fixed .60/.10/.20/.10 proxy weights. Thus each protocol receives 360 episodes and
60 optimizer updates. CPU execution is single-threaded. There is no epoch/seed
selection or further tuning after observing validation. Checkpoints are retained
at epochs 5/10/15/20; the final one is the primary comparison.

Protocols:

1. **cosine:** full pool sorted by descending root cosine, ID tie break. Sampler,
   action RNG, initialization and epoch query shuffle reproduce Phase 4. The
   instrumented sampler's score/log-probability/gradient are tested against the
   original implementation.
2. **random_order:** uniformly shuffle the complete pool using a deterministic
   per-(training seed, epoch, query) order seed. Ordering RNG is separate from
   Bernoulli action RNG. No candidate is filtered.
3. **shortlist:** at most 24 pages, all of which jointly fit 50K. First choose
   best-available root/subquestion affinity anchors where they fit, skipping
   already represented duplicate clusters. Fill with pages no larger than B/24,
   ranked by mean target affinity, with length/ID tie breaks. Keep distinct frozen
   duplicate clusters. All choices use saved embeddings/provenance/lengths only;
   no terminal rewards or validation outcomes are queried to construct the list.
   Shuffle this frozen list with the same seeded ordering scheme as random_order.
   Coverage preservation is measured in `shortlist_coverage.json`, not assumed.
   This protocol is a substantial handcrafted retrieval/filter prior. Only its
   improvement over its own untrained control can be attributed to RL.

The shortlist's joint-fit property ensures every shortlisted page is a free
Bernoulli decision, increasing the opportunity to prune rather than filling a
budget from the first few high-cosine pages. It also changes the candidate space,
page lengths and token utilization. Those are filtering effects and must not be
presented as learned improvements. All methods retain the same 50K cap, but this
is not an exact matching of actual retained tokens.

## Gradient audit

For Bernoulli logit z and sampled action a, `d log pi / dz = a - sigmoid(z)`.
Every free decision logs this quantity, its probability, entropy and log
probability. Forced decisions are never sampled and contribute neither log
probability nor entropy. Final selected contexts are reconstructed and checked by
the original exact allocator, independently of the cached action-feasibility path.

Tests enumerate all trajectories of a two-page budget-constrained problem and
compare the exact expected-reward derivative, REINFORCE expected gradient with a
constant detached baseline, and finite differences. A separate test reproduces the
original Phase 4 trajectory log probability and full gradient exactly.

Each update logs per-episode reward and baseline, advantage mean/variance,
per-episode parameter-gradient norms, reward-gradient norm, entropy-gradient norm,
clipping and the numerical error in `total_gradient = sum(episode_gradients) +
entropy_gradient`. Gradient alignment is `norm(sum(g_i)) / sum(norm(g_i))`, a
bounded descriptive cancellation measure rather than a statistical SNR estimate.
The signed score-function logit contributions are in each episode's action trace.

Entropy is in natural-log units (Bernoulli maximum log(2)=.6931). Sampled versus
forced counts are kept separate from shortlist-filtered pages. Last sampled scan
position shows how much of the ordered candidate list remained reachable before
budget saturation. A scalar EMA subtracts the same baseline from all episodes in
an update, so within-update advantage variance equals reward variance; it cannot
remove between-query difficulty variance. This fact is audited without changing
the baseline algorithm.

## Evaluation and stopping

Evaluate both trained and untrained models under their own candidate protocol
using the same five action and order streams (1000–1004). Random candidate ordering
is reproducible even for deterministic threshold-.5 inference. Primary results
are stochastic. Evaluate train and validation after optimization only; no
validation metric affects updates, filtering or checkpoint selection.

Average masks and training seeds within each query before taking macro means.
Report the trained-minus-untrained change for each protocol/seed/query; compare
untrained protocols separately to quantify ordering/filtering gains. The historical
heuristic baselines are rescored unchanged as context. Do not treat masks/pages
or repeated training seeds as independent research questions. The two validation
queries were already inspected during proxy design and are not pristine final
holdouts.

Stop remaining runs if the three-epoch empty rate exceeds 50%, rolling reward
falls below zero, or any nonfinite gradient/loss, gradient accounting mismatch or
budget integrity failure occurs. Runtime remains bounded at nine models and
1,080 training episodes. No automatic extension/restart is performed.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m adr.experiments.evidence_replay.learning_signal \
  --output runs/evidence-replay-learning-signal/diagnostic-002
```

Always use a fresh directory. `config.json` freezes sources/config/code hashes.
Each protocol/seed has checkpoints, `history.json` and `episodes.json` with action
traces. Top-level `results.json`, `summary.json`, `comparison.csv`, `per_query.csv`,
`curves.json`, `learning_curves.png`, `shortlist_coverage.json`, and `status.json`
provide audit data. `heuristic_baselines.json` contains the unchanged heuristic
controls. The actual first run is `diagnostic-001`, which must not be overwritten.

## Interpretation boundary

High action entropy with few feasible sampled actions is not broad evidence-pool
exploration. A better untrained shortlist is evidence for preprocessing, not RL.
An improvement only on training or only in one seed is insufficient to establish
generalization. Even consistent proxy gains would still require frozen-checkpoint,
matched-cost, blinded report-quality evaluation on appropriate benchmark queries
before filling report-quality cells in a Table 4 result. Historical baseline/query
and evaluator provenance must also be resolved; frozen-retrieval synthesis pruning
must not be labeled as an end-to-end retrieval-efficiency improvement.

## diagnostic-001 findings

All nine models completed their prespecified 20 epochs (1,080 total training
episodes, 180 updates) in 101.56 seconds before plotting. There were no empty
evaluation masks, budget violations or instability stops. All three cosine models
exactly reproduced Phase 4 parameters. The full offline test suite passed 58 tests.

| Protocol | Train untrained → trained | Validation untrained → trained |
|---|---:|---:|
| Cosine | .464783 → .464298 | .501668 → .499939 |
| Random order | .429065 → .426718 | .430145 → .436655 |
| Shortlist | .564279 → .564017 | .568547 → .568476 |

Mean free training decisions were 13.11 / 19.69 / 22.83 respectively; forced
exclusions were 311.72 / 305.14 / 0, with about 302 pages explicitly filtered by
the shortlist. The shortlist retained 17–24 candidates and its all-pages proxy
coverage ceiling ranged .966–.994 (mean .982). Sampled shortlist contexts used
about 34.6K training / 30.9K validation tokens versus near 50K for the full pools.

Free-action entropy stayed near .60 nats (maximum .693), KEEP probabilities stayed
between .669 and .753, and every deterministic mask was unchanged after training.
The sum-of-episode/entropy gradient check had maximum residual 4.23e-7. Mean reward
gradient norms were .161 / .148 / .210, versus entropy-gradient norms around .0017.
There were 0 / 0 / 5 clipped updates out of 60 per protocol and no nonfinite values.
Thus entropy regularization or missing/vanishing gradients is not the observed
failure. The policy remained on the KEEP side of its deterministic boundary.

Within-update advantage variance was .00439 / .00209 / .000106. About 41% / 41% /
25% of pooled reward variance was associated with query means (descriptive, across
changing policies). The zero-start EMA made first-epoch mean advantages .460 /
.429 / .565; this cold-start offset plus noisy terminal credit assignment remains a
plausible limitation, not a causally isolated explanation. The baseline and loss
were intentionally not changed in this experiment.

Random ordering improved both validation-query means (+.0063 and +.0067), but two
seeds gained +.00977 and one was unchanged, while all training-seed means fell.
It remains worse in absolute validation reward than threshold/top-k/cosine-order
controls. This small result on two previously inspected queries is not a robust
learning/generalization claim. The untrained shortlist gained .0995 training and
.0669 validation reward over untrained cosine order; REINFORCE added no mean gain.
That gain belongs to the hand-built filter and lower token cost, not learning.

Conclusion: saturation limits exploration, but increasing decisions and removing
forced saturation did not by itself produce useful learning. Do not extend the
current runs or label the shortlist improvement as REINFORCE. A future controlled
training-only baseline/initialization diagnostic could be justified if the RL
claim is essential; simply adding epochs is unsupported here. For a defensible
Table 4, preserve the negative RL result and separate the shortlist/no-RL ablation.
Freeze masks and evaluate actual report quality with matched-cost controls on an
agreed benchmark set and pinned judges before claiming quality gains. Historical
Table 4 provenance and the frozen-retrieval versus end-to-end scope still matter.

`learning_signal_audit.py` adds `audit.json`, `paired_deltas.csv` and
`gradient_summary.csv` without overwriting existing analysis files. The audit also
contains per-seed/per-query changes, cold-policy contrasts decomposed by reward
component, deterministic-mask comparisons and exact Phase 4 reproduction checks.
