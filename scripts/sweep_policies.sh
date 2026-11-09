#!/usr/bin/env bash
# Run each orchestration policy on one benchmark, evaluate, and collect results.
# Shared by DeepResearchGym and DeepResearch Bench; the benchmark is taken from
# --bench, or from dataset.name of the run config given with --config.
#
# Usage:
#   bash scripts/sweep_policies.sh                            # Gym, 5 queries, all policies
#   bash scripts/sweep_policies.sh --bench drb --limit 20     # DeepResearch Bench, 20 queries per policy
#   bash scripts/sweep_policies.sh --config configs/gpt_researcher_bench.yaml   # bench from the config
#   bash scripts/sweep_policies.sh --policies "none topk"     # subset of policies
#   bash scripts/sweep_policies.sh --skip-eval                # agent only, judge later
#   bash scripts/sweep_policies.sh --eval-only                # re-evaluate existing runs
#   bash scripts/sweep_policies.sh --sleep 90                 # seconds between evals (rate limit)
#   bash scripts/sweep_policies.sh --budget 30000             # override the YAML context budget for this sweep
#
# Policy knobs (budget, thresholds, TypeSafe model) come from the `orchestration:`
# section of the dataset's agent YAML (configs/agents/gpt_researcher_gym.yaml or
# gpt_researcher_bench.yaml). This script only sets GR_ORCHESTRATOR per row. A
# GR_* exported in the shell before the sweep (e.g. GR_STOP_PRUNE_GAIN=0.02 bash
# scripts/sweep_policies.sh) is passed through and overrides the YAML for every
# row it applies to.
#
# The typesafe row needs TYPESAFE_API_KEY (console.typesafe.ai/keys) and
# `pip install typesafe-sdk`; without the key that row is skipped, not fatal.
set -euo pipefail

LIMIT=5
POLICIES="none topk extractive llmlingua prompted greedy heuristic_stop typesafe"
BENCH=""            # gym | drb; empty = from the run config's dataset.name (default gym)
BASE_CONFIG=""      # empty = the benchmark's default run config
SKIP_EVAL=false
EVAL_ONLY=false
OUTPUT_DIR="results"
EVAL_SLEEP=60
CONTEXT_BUDGET="${GR_CONTEXT_BUDGET_TOKENS:-}"   # empty = the YAML value

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)       LIMIT="$2"; shift 2 ;;
    --policies)    POLICIES="$2"; shift 2 ;;
    --bench)       BENCH="$2"; shift 2 ;;
    --config)      BASE_CONFIG="$2"; shift 2 ;;
    --output)      OUTPUT_DIR="$2"; shift 2 ;;
    --sleep)       EVAL_SLEEP="$2"; shift 2 ;;
    --budget)      CONTEXT_BUDGET="$2"; shift 2 ;;
    --skip-eval)   SKIP_EVAL=true; shift ;;
    --eval-only)   EVAL_ONLY=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Benchmark ─────────────────────────────────────────────────────────────────
# Resolve --bench / --config in either direction; canonical names below.
_bench_from_config() {
  python3 - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(str((cfg.get("dataset") or {}).get("name") or ""))
PY
}
if [[ -z "$BENCH" && -n "$BASE_CONFIG" ]]; then
  BENCH="$(_bench_from_config "$BASE_CONFIG")"
fi
case "${BENCH:-gym}" in
  gym|deep_research_gym)     BENCH=gym; OFFICIAL_BENCH=deep_research_gym;   RUN_PREFIX=gym ;;
  drb|bench|deep_research_bench) BENCH=drb; OFFICIAL_BENCH=deep_research_bench; RUN_PREFIX=drb ;;
  *) echo "Unknown --bench '$BENCH' (expected gym | drb)"; exit 1 ;;
esac
if [[ -z "$BASE_CONFIG" ]]; then
  case "$BENCH" in
    gym) BASE_CONFIG="configs/gpt_researcher_gym.yaml" ;;
    drb) BASE_CONFIG="configs/gpt_researcher_bench.yaml" ;;
  esac
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SWEEP_ID="sweep_${RUN_PREFIX}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Policy sweep: ${SWEEP_ID}"
echo "  Benchmark:   ${OFFICIAL_BENCH}"
echo "  Policies:    ${POLICIES}"
echo "  Queries:     ${LIMIT} per policy"
echo "  Config:      ${BASE_CONFIG}"
echo "  Budget:      ${CONTEXT_BUDGET:-from agent YAML} tokens"
echo "  Eval:        skip=${SKIP_EVAL} eval_only=${EVAL_ONLY}"
echo "  Eval sleep:  ${EVAL_SLEEP}s between policies"
echo "═══════════════════════════════════════════════════════════"

MAP_FILE=$(mktemp)
trap 'rm -f "$MAP_FILE"' EXIT

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ── Per-policy env vars ───────────────────────────────────────────────────────
# The row is selected with GR_ORCHESTRATOR; every knob comes from the agent
# YAML. Shell overrides for these variables are snapshotted once at startup and
# re-exported for every row, so a stale value from a previous row never leaks.
# TYPESAFE_API_KEY is a credential, not a knob, so it is deliberately not here.
_POLICY_ENVS=(
  GR_ORCHESTRATOR
  GR_CONTEXT_BUDGET_TOKENS
  GR_FEAT_TAU
  GR_FEAT_TAU_C
  GR_TOPK_K
  GR_COMPRESSION_RATE
  GR_LLMLINGUA_MODEL
  GR_GREEDY_MIN_GAIN
  GR_GREEDY_LAMBDA
  GR_STOP_PRUNE_GAIN
  GR_STOP_SATISFACTION
  GR_STOP_MIN_ROUNDS
  GR_STOP_RETAIN
  GR_TYPESAFE_MODEL
  GR_TYPESAFE_KEEP_MIN
  GR_TYPESAFE_BOILERPLATE_MAX
  GR_TYPESAFE_STOP_MIN
  GR_TYPESAFE_MIN_ROUNDS
  GR_TYPESAFE_SUPPORT_K
  GR_TYPESAFE_BATCH
  GR_TYPESAFE_SNIPPET_CHARS
)

# Plain variables (no associative arrays) so this also runs under macOS bash 3.2.
for var in "${_POLICY_ENVS[@]}"; do
  declare "_OVERRIDE_${var}=${!var:-}"
done

# Returns 1 when the row cannot run in this environment (the caller skips it).
setup_policy_env() {
  local policy="$1"
  for var in "${_POLICY_ENVS[@]}"; do
    unset "$var" 2>/dev/null || true
  done
  # Re-export the shell's explicit overrides (never GR_ORCHESTRATOR itself).
  local var name value
  for var in "${_POLICY_ENVS[@]}"; do
    [[ "$var" == "GR_ORCHESTRATOR" ]] && continue
    name="_OVERRIDE_${var}"
    value="${!name:-}"
    if [[ -n "$value" ]]; then
      export "$var=$value"
    fi
  done
  if [[ -n "$CONTEXT_BUDGET" ]]; then
    export GR_CONTEXT_BUDGET_TOKENS="$CONTEXT_BUDGET"
  fi

  if [[ "$policy" == "legacy" ]]; then
    return 0
  fi
  export GR_ORCHESTRATOR="$policy"

  if [[ "$policy" == "typesafe" && -z "${TYPESAFE_API_KEY:-}" ]]; then
    return 1
  fi
  return 0
}

if [[ "$EVAL_ONLY" == "true" ]]; then
  for policy in $POLICIES; do
    latest=$(ls -dt runs/*"${RUN_PREFIX}-${policy}-"* 2>/dev/null | head -1 || true)
    if [[ -n "$latest" ]]; then
      echo "${policy} ${latest}" >> "$MAP_FILE"
      echo "FOUND    ${policy} -> ${latest}"
    else
      echo "MISSING  ${policy} (no matching run dir)"
    fi
  done
else
  for policy in $POLICIES; do
    run_name="${RUN_PREFIX}-${policy}-${LIMIT}"
    log_file="${LOG_DIR}/${SWEEP_ID}_${policy}.log"

    if ! setup_policy_env "$policy"; then
      echo "  SKIP ${policy} (TYPESAFE_API_KEY not set)"
      continue
    fi

    echo -n "  RUN  ${policy} (${LIMIT} queries) ... "
    start_ts=$(date +%s)

    adr run \
      --config "$BASE_CONFIG" \
      --limit "$LIMIT" \
      --run-name "$run_name" \
      > "$log_file" 2>&1

    elapsed=$(( $(date +%s) - start_ts ))
    run_dir=$(ls -dt runs/*"${run_name}"* 2>/dev/null | head -1)
    if [[ -z "$run_dir" ]]; then
      echo "ERROR (${elapsed}s) see ${log_file}"
      continue
    fi
    echo "${policy} ${run_dir}" >> "$MAP_FILE"
    echo "done (${elapsed}s) -> ${run_dir}"
    echo "       log: ${log_file}"
  done
fi

if [[ "$SKIP_EVAL" == "false" ]]; then
  eval_count=0
  total_policies=$(wc -l < "$MAP_FILE" | tr -d ' ')
  while IFS=' ' read -r policy run_dir; do
    eval_count=$((eval_count + 1))
    eval_log="${LOG_DIR}/${SWEEP_ID}_${policy}_eval.log"
    echo -n "  EVAL ${policy} ... "
    adr evaluate "$run_dir" --official "$OFFICIAL_BENCH" > "$eval_log" 2>&1 || true
    echo "done (log: ${eval_log})"
    if [[ $eval_count -lt $total_policies && $EVAL_SLEEP -gt 0 ]]; then
      echo "       sleeping ${EVAL_SLEEP}s (rate limit) ..."
      sleep "$EVAL_SLEEP"
    fi
  done < "$MAP_FILE"
fi

echo
echo "───────────────────────────────────────────────────────"
echo "  Aggregating results"
echo "───────────────────────────────────────────────────────"

python3 - "$OUTPUT_DIR" "$SWEEP_ID" "$MAP_FILE" "$BENCH" <<'PYEOF'
import json, sys, csv
from pathlib import Path

output_dir = Path(sys.argv[1])
sweep_id = sys.argv[2]
map_file = sys.argv[3]
bench = sys.argv[4]

# Headline columns per benchmark: (table label, key in summary["scores"]).
# Every key of summary["scores"] is written to the CSV regardless.
COLUMNS = {
    "gym": [
        ("Quality", "gym_quality"),
        ("Clarity", "gym_quality_clarity"),
        ("Depth", "gym_quality_depth"),
        ("Breadth", "gym_quality_breadth"),
        ("Support", "gym_quality_support"),
        ("KPR", "gym_average_support_rate"),
        ("Citation", "gym_citation_score"),
    ],
    "drb": [
        ("Overall", "race_overall_score"),
        ("Compreh.", "race_comprehensiveness"),
        ("Insight", "race_insight"),
        ("InstrFol", "race_instruction_following"),
        ("Readab.", "race_readability"),
        ("FACT", "fact_valid_rate"),
    ],
}[bench]

pairs = []
for line in Path(map_file).read_text().strip().splitlines():
    policy, run_dir = line.split(" ", 1)
    pairs.append((policy, run_dir))

results = {"sweep_id": sweep_id, "bench": bench, "policies": {}}
csv_rows = []


def _fmt(v, width=8):
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.4f}" if abs(v) < 10 else f"{v:>{width}.1f}"


for policy, run_dir in pairs:
    summary_path = Path(run_dir) / "metrics" / "summary.json"
    if not summary_path.exists():
        summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        print(f"  {policy}: no summary.json")
        continue

    summary = json.loads(summary_path.read_text())
    scores = summary.get("scores") or {}

    row = {
        "policy": policy,
        "n_queries": summary.get("n_queries", 0),
        "n_errors": summary.get("n_errors", 0),
        "tokens": summary.get("mean_tokens", 0),
        "prompt_tokens": summary.get("mean_prompt_tokens", 0),
        "completion_tokens": summary.get("mean_completion_tokens", 0),
        "wall_s": round(summary.get("mean_wall_s", 0), 1),
        "n_searches": summary.get("mean_n_searches", 0),
        "n_reads": summary.get("mean_n_reads", 0),
        "n_llm_calls": summary.get("mean_n_llm_calls", 0),
        "n_retained": summary.get("mean_n_retained", 0),
        "n_pruned": summary.get("mean_n_pruned", 0),
        "prune_rate": round(summary.get("mean_prune_rate", 0), 4),
        "n_citations": summary.get("mean_n_citations", 0),
        "article_chars": summary.get("mean_article_chars", 0),
        "run_dir": str(run_dir),
    }
    for key, value in scores.items():
        row[key] = value if isinstance(value, (int, float)) else None

    results["policies"][policy] = row
    csv_rows.append(row)
    headline = ", ".join(f"{label}={_fmt(row.get(key), 0).strip()}" for label, key in COLUMNS[:2])
    print(f"  {policy}: {headline}  tokens={row['tokens']:.0f}  prune={row['prune_rate']:.1%}")

json_path = output_dir / f"{sweep_id}.json"
json_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\n  JSON: {json_path}")

if csv_rows:
    fieldnames = list(dict.fromkeys(k for r in csv_rows for k in r))
    csv_path = output_dir / f"{sweep_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fieldnames} for r in csv_rows)
    print(f"  CSV:  {csv_path}")

    hdr = f"{'Policy':<14}" + "".join(f" {label:>8}" for label, _ in COLUMNS) + f" {'Tokens':>8} {'Prune%':>7} {'Wall(s)':>8} {'Errors':>6}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in csv_rows:
        print(
            f"{r['policy']:<14}"
            + "".join(" " + _fmt(r.get(key)) for _, key in COLUMNS)
            + f" {r['tokens']:>8.0f} {r['prune_rate']:>6.1%} {r['wall_s']:>8.1f} {r['n_errors']:>6}"
        )
PYEOF

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Sweep complete: ${SWEEP_ID}"
echo "═══════════════════════════════════════════════════════════"
