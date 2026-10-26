# Phase 4: bounded CPU REINFORCE pilot

The model learns a selection distribution over a **frozen retrieved page pool**.
It does not learn retrieval or report writing. Training rewards are the existing
`page-proxy-v2-cluster-support` with fixed anti-redundancy weights
(coverage .60, source diversity .10, redundancy .20, exact token cost .10).
No reward definitions, baselines, reference questions or judge results are changed.

## Prespecified bounded experiment

- Research families 001–006 train; 007–008 validate, matching the preassigned
  real-source-v2 split. Variants of a source may not cross this boundary. The eight
  queries were previously inspected during proxy design, so validation is a check
  on policy fitting, **not an untouched test of proxy/report generalization**.
- Seeds 0, 1, 2. Single-thread deterministic CPU PyTorch. At most 20 epochs/seed,
  one rollout per training query per epoch, six episodes averaged into each update:
  120 episodes and 20 AdamW updates/seed, at most 360 episodes in the pilot.
- Shared 14 → 32 → 1 Tanh policy, 513 parameters. No page IDs, raw embeddings,
  query identities, downstream reports, rewards or judge labels enter model inputs.
- Features: root cosine, subquestion cosine mean/max, page-token/budget fraction,
  normalized depth/round, remaining budget, selected count, maximum similarity to
  selected pages, selected-source frequency, root/subquestion affinity gaps,
  whether the candidate's frozen duplicate cluster is already selected, and scan
  position. Features are deterministic bounded transformations of saved evidence
  and selection state; no validation-fitted scaler is needed.
- All pages are scanned in descending root cosine with ID tie breaks. This means
  even the untrained policy has a retrieval-ranking prior. The initial head is
  near p(KEEP)=.7; deterministic inference initially behaves like top-k. Initial
  probabilities were set without labels, to avoid accidental empty deterministic
  masks. The untrained stochastic policy is therefore the key learning control.
- Bernoulli KEEP/PRUNE with independent seeded sampling, conditional on previous
  selections and remaining budget. Infeasible candidates are forced PRUNE and do
  not enter log probability or entropy. Empty masks receive the existing -1
  reward: there is no hidden KEEP repair or fallback at inference.
- Loss = batch mean of `-(R - baseline).detach() * sum(log_probabilities)` minus
  .005 times the mean per-free-action entropy. Baseline starts at zero, is used
  before update, and moves as `.8*old + .2*batch_mean_reward`. It is a scalar
  independent of model gradients, not a value network. AdamW lr .003, weight
  decay .0001, global gradient clipping 1.0. No tuning during the pilot.
- Fail closed on token-count mismatches, over-budget contexts, nonfinite loss or
  nonfinite gradients. Stop launching seeds if >50% of training episodes are empty
  over the last three epochs. No checkpoint/epoch is selected on validation.

## Exact context budget without repeatedly tokenizing the whole pool

`ExactAllocator` caches o200k_base counts for each existing replay page wrapper,
with and without its trailing double-LF separator. The wrapper ends with a closed
marker and the next block starts with a non-whitespace `[` header. The tokenizer's
regex prevents a token from crossing that boundary into the next header. The
canonical-order context count is the sum of separated-block counts minus the last
block's separator delta. This path requires o200k_base and ASCII alphanumeric page
IDs, and does not approximate character/token ratios.

Before training, three-page and complete-pool counts are checked independently.
Every rollout is then reconstructed using the original `core.build_context` and
independently tokenized; mismatch or budget overrun stops the pilot. Tests cover all
small masks and 100 seeded masks with mixed CR/CRLF/LF and Unicode. Budget checks
occur before every sampled action. The baseline selector code is unchanged.

## Evaluation and artifacts

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m adr.experiments.evidence_replay.reinforce_pilot \
  --output runs/evidence-replay-phase4/cpu-pilot-002 \
  --epochs 20 --learning-rate 0.003 --entropy 0.005
```

The output directory must be fresh. This command blocks outbound socket
connections in the pilot process. It imports no synthesis or judge adapter.

Each seed has `untrained.pt`, epoch checkpoints, and `training/history.json`.
Checkpoints preserve model/optimizer state, epoch, moving baseline, sampling and
shuffle RNG states, global Torch RNG, history, seed, and a source/config/code
signature. `train(..., resume=checkpoint, output=fresh_path)` restores an epoch
boundary exactly; use the same Config including target epoch count. `stop_after`
can bound a partial run. A changed source, code, reward, seed or config is rejected.
Only load trusted locally created checkpoints (`weights_only=False` stores Python
RNG state). Existing checkpoints are never overwritten.

After training ends, epochs 0/5/10/15/20 are evaluated with fixed stochastic seeds
1000–1004 on both splits. The policy never sees these scores during optimization.
Primary comparison uses the final-epoch stochastic policy, averaging masks within
query, then queries. Deterministic threshold-.5 inference is separately reported.
The same five evaluation streams are used for trained/untrained policies for a
paired distribution comparison. Seed variability and query variability should not
be confused with new independent questions.

Random (seeds 0–4), chronological cosine threshold and top-k masks come from the
existing Phase 3 records. Source hash, reward version, budget and context hash are
verified and each mask is rescored with fixed weights. All methods share the strict
50K cap. Actual retained token use may differ for learned policies; equal caps are
not a claim of equal token utilization. When judging reports later, add matches in
actual retained tokens as needed. The original macro numbers over all eight
queries need not match the six/two split-specific values.

`config.json` freezes all settings, split/source hashes and code provenance.
`results.json` contains every final mask and its separate components, actual tokens,
constraint violations, training/evaluation seed, identity and context hash.
`summary.json` reports per-query and macro means for train/validation, including
untrained/stochastic/deterministic policies and all three baselines.
`evaluation_curves.json` and `reward_curves.png` record train/validation proxy curves.
`status.json` reports completed seeds, wall time and zero API calls. Training curves
in each seed history also include entropy, gradient norms, baseline and empty rate.

## Interpreting the pilot

Learning is assessed against the **untrained** policy as well as the heuristics,
without tuning seeds or reporting a best checkpoint. An improvement only on train,
a train/validation gap, falling coverage with rising reward, selection collapse,
large seed variance or worsening deterministic inference should be reported, not
fixed by selecting a different validation epoch. The proxy rewards low cost and
three independent support groups, so a learned preference for short contexts may
reflect its construction rather than sufficient evidence for a detailed report.

Next, freeze a checkpoint before report evaluation. Export its selected IDs through
the existing replay adapter, verify context hashes, and compare blind report-quality
scores with matched-cost heuristic and untrained masks on new source families.
Use `score_saved_replay`/`pair_quality` to bind downstream outcomes to masks. The
existing real-source questions require a reference-free rubric or authored
references; official DRB is allowed only where exact reference/rubric coverage exists.
No improvement in report quality follows from proxy reward improvement alone.

## Recorded cpu-pilot-001 result

All three seeds completed 20 epochs (360 training episodes, 60 AdamW updates in
total). Offline training plus checkpoint evaluation took 31.22 seconds before
plot rendering. All final masks were nonempty and feasible; there were no
nonfinite values or clipped updates. Learned parameter changes were nonzero
(L2 change .291–.402), but only 16 of 120 paired stochastic evaluation masks
changed. Deterministic output stayed identical to untrained top-k for every seed.

| Primary stochastic evaluation | Train reward | Validation reward |
|---|---:|---:|
| Untrained | .464783 | .501668 |
| Final REINFORCE | .464298 | .499939 |
| Cosine threshold | .470053 | .499455 |
| Top-k | .438930 | .504391 |
| Random | .439152 | .440611 |

Two seeds slightly decreased reward and one left evaluation masks unchanged.
There was no improvement over the untrained policy on either split, and no
sustained upward curve after epoch five. The advantage over random is already
present in the untrained cosine-ordered sampler, so it is not evidence of RL
learning. No extra epochs, hyperparameter changes or best-seed selection were run.
The training objective remained stable; lack of improvement and a saturated
initial keep decision boundary are the main issues, rather than demonstrated
reward hacking or memorization. Only two validation questions and prior proxy
inspection preclude strong generalization statements.

Artifacts are in `runs/evidence-replay-phase4/cpu-pilot-001`: full histories and
checkpoints, `reward_curves.png`, `evaluation_curves.json`, `comparison.csv`,
`per_query.csv`, `results.json`, `summary.json`, `learning_audit.json`, and source/
configuration provenance. These outputs must not be overwritten. This pilot does
not justify more report synthesis to demonstrate a learned improvement: first
approve a separately prespecified optimization diagnostic or evaluate frozen masks
as feasibility controls, without claiming better reports.
