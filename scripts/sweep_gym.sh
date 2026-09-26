#!/usr/bin/env bash
# Run each orchestration policy on DeepResearchGym, evaluate, and collect results.
#
# Usage:
#   bash scripts/sweep_gym.sh                         # defaults: 5 queries, all policies
#   bash scripts/sweep_gym.sh --limit 20              # 20 queries per policy
#   bash scripts/sweep_gym.sh --policies "none topk"  # subset of policies
#   bash scripts/sweep_gym.sh --skip-eval             # agent only, judge later
#   bash scripts/sweep_gym.sh --eval-only             # re-evaluate existing runs
#   bash scripts/sweep_gym.sh --sleep 90              # seconds between evals (rate limit)
#   bash scripts/sweep_gym.sh --budget 30000          # override the YAML context budget for this sweep
#
# Policy knobs (budget, thresholds, TypeSafe model) come from the agent YAML's
# `orchestration:` section (configs/agents/gpt_researcher_gym.yaml). This
# script only sets GR_ORCHESTRATOR per row. A GR_* exported in the shell before
# the sweep (e.g. GR_STOP_PATIENCE=3 bash scripts/sweep_gym.sh) is passed
# through and overrides the YAML for every row it applies to.
#
# The typesafe row needs TYPESAFE_API_KEY (console.typesafe.ai/keys) and
# `pip install typesafe-sdk`; without the key that row is skipped, not fatal.
set -euo pipefail

LIMIT=5
POLICIES="none topk extractive llmlingua prompted greedy heuristic_stop typesafe"
BASE_CONFIG="configs/gpt_researcher_gym.yaml"
SKIP_EVAL=false
EVAL_ONLY=false
OUTPUT_DIR="results"
EVAL_SLEEP=60
CONTEXT_BUDGET="${GR_CONTEXT_BUDGET_TOKENS:-}"   # empty = the YAML value

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)       LIMIT="$2"; shift 2 ;;
    --policies)    POLICIES="$2"; shift 2 ;;
    --config)      BASE_CONFIG="$2"; shift 2 ;;
    --output)      OUTPUT_DIR="$2"; shift 2 ;;
    --sleep)       EVAL_SLEEP="$2"; shift 2 ;;
    --budget)      CONTEXT_BUDGET="$2"; shift 2 ;;
    --skip-eval)   SKIP_EVAL=true; shift ;;
    --eval-only)   EVAL_ONLY=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SWEEP_ID="sweep_gym_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Gym sweep: ${SWEEP_ID}"
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
  GR_STOP_GAIN_THRESHOLD
  GR_STOP_MIN_ROUNDS
  GR_STOP_PATIENCE
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
    latest=$(ls -dt runs/*gym-${policy}-* 2>/dev/null | head -1 || true)
    if [[ -n "$latest" ]]; then
      echo "${policy} ${latest}" >> "$MAP_FILE"
      echo "FOUND    ${policy} -> ${latest}"
    else
      echo "MISSING  ${policy} (no matching run dir)"
    fi
  done
else
  for policy in $POLICIES; do
    run_name="gym-${policy}-${LIMIT}"
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
    adr evaluate "$run_dir" --official deep_research_gym > "$eval_log" 2>&1 || true
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

python3 - "$OUTPUT_DIR" "$SWEEP_ID" "$MAP_FILE" <<'PYEOF'
import json, sys, csv
from pathlib import Path

output_dir = Path(sys.argv[1])
sweep_id = sys.argv[2]
map_file = sys.argv[3]

pairs = []
for line in Path(map_file).read_text().strip().splitlines():
    policy, run_dir = line.split(" ", 1)
    pairs.append((policy, run_dir))

results = {"sweep_id": sweep_id, "policies": {}}
csv_rows = []

def _fmt(v):
    return f"{v:.1f}" if v is not None and v != 0 else "n/a"

for policy, run_dir in pairs:
    summary_path = Path(run_dir) / "metrics" / "summary.json"
    if not summary_path.exists():
        summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        print(f"  {policy}: no summary.json")
        continue

    summary = json.loads(summary_path.read_text())
    scores = summary.get("scores") or {}
    official = (summary.get("official") or {}).get("deep_research_gym") or {}

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

    # Gym quality: prefer top-level scores, fall back to official block
    row["gym_quality"] = scores.get("gym_quality")
    quality = official.get("quality") or {}
    if not row["gym_quality"]:
        row["gym_quality"] = quality.get("average_normalized_score")
    for criterion in ("clarity", "depth", "balance", "breadth", "support", "insightfulness"):
        key = f"gym_quality_{criterion}"
        row[key] = scores.get(key)
        if not row[key]:
            pc = (quality.get("per_criterion") or {}).get(criterion.capitalize())
            if isinstance(pc, dict):
                row[key] = pc.get("average_rating")

    # Gym KPR (key point recall)
    kpr = official.get("kpr") or {}
    row["gym_support_rate"] = scores.get("gym_average_support_rate") or kpr.get("average_support_rate")
    row["gym_omitted_rate"] = scores.get("gym_average_omitted_rate") or kpr.get("average_omitted_rate")
    row["gym_contradicted_rate"] = scores.get("gym_average_contradicted_rate") or kpr.get("average_contradicted_rate")

    # Gym citation faithfulness (can be null)
    citation = official.get("citation") or {}
    row["gym_citation_score"] = citation.get("average_citation_score")

    results["policies"][policy] = row
    csv_rows.append(row)

    print(
        f"  {policy}: quality={_fmt(row['gym_quality'])}  "
        f"support={_fmt(row['gym_support_rate'])}  "
        f"citation={_fmt(row['gym_citation_score'])}  "
        f"tokens={row['tokens']:.0f}  prune={row['prune_rate']:.1%}"
    )

json_path = output_dir / f"{sweep_id}.json"
json_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\n  JSON: {json_path}")

if csv_rows:
    csv_path = output_dir / f"{sweep_id}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  CSV:  {csv_path}")

    hdr = (
        f"{'Policy':<14} {'Quality':>8} {'Clarity':>8} {'Depth':>8} "
        f"{'Breadth':>8} {'Support':>8} {'KPR':>8} {'Citation':>8} "
        f"{'Tokens':>8} {'Prune%':>7} {'Wall(s)':>8} {'Errors':>6}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in csv_rows:
        print(
            f"{r['policy']:<14} "
            f"{_fmt(r['gym_quality']):>8} "
            f"{_fmt(r.get('gym_quality_clarity')):>8} "
            f"{_fmt(r.get('gym_quality_depth')):>8} "
            f"{_fmt(r.get('gym_quality_breadth')):>8} "
            f"{_fmt(r.get('gym_quality_support')):>8} "
            f"{_fmt(r['gym_support_rate']):>8} "
            f"{_fmt(r['gym_citation_score']):>8} "
            f"{r['tokens']:>8.0f} "
            f"{r['prune_rate']:>6.1%} "
            f"{r['wall_s']:>8.1f} "
            f"{r['n_errors']:>6}"
        )
PYEOF

echo
echo "═══════════════════════════════════════════════════════════"
echo "  Sweep complete: ${SWEEP_ID}"
echo "═══════════════════════════════════════════════════════════"
