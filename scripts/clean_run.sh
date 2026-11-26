#!/usr/bin/env bash
# One command for the clean DeepResearchGym experiment: preflight, a 1-query
# smoke test (the first query at the top budget, reused by the full runs) with automatic checks, the full runs, three judge passes and the
# result tables. Every stage is resumable: rerun the same command after any
# interruption and finished runs / judgings are skipped.
#
#   bash scripts/clean_run.sh all        # preflight -> smoke -> check -> full -> judge x3 -> tables
#   bash scripts/clean_run.sh smoke      # preflight + smoke + check only
#   bash scripts/clean_run.sh full       # full runs only (assumes smoke passed)
#   bash scripts/clean_run.sh judge      # judge passes 1..3 of everything written
#   bash scripts/clean_run.sh table      # rebuild RESULTS.md / results.json / per_query.csv
#
# Design (research-only budget, nothing cut from the writer):
#   depth 3, breadth 4, gpt-6.1-sol for research and writing;
#   GR_RESEARCH_BUDGET = each of BUDGETS, output weighted GR_BUDGET_OUTPUT_WEIGHT (5,
#   the Flex price ratio); Qwen (BC/DPO) and the writer are not counted;
#   writer context uncapped (GR_MAX_CONTEXT_WORDS=0), fixed report length 3000 +/- 10%,
#   no length rewrite.
#   Mode "equal": GR_BUDGET_FORCE_FULL=1, every method spends the full research
#     budget (quality at equal spend), at every budget.
#   Mode "stop": stop decisions respected (what each method spends), top budget
#     only, for the methods that can stop.
#
# Settings (environment):
#   WRITER_MAX_INPUT_TOKENS  required: the writer model's context limit minus ~20K
#   BUDGETS                  default "100000 150000 200000"
#   QUERIES                  default: the 10 DRG pilot queries of the budget sweep
#   ARMS                     default "none extractive prompted prompted_scores bc dpo"
#   JOBS                     parallel arm processes (default 6)
#   ROOT                     default runs/clean_drg
#   SKIP_LEARNED=1           drop bc/dpo if the vLLM server is not reachable
#   DRYRUN=1                 print the runs it would start, spend nothing
#   CONTEXT_BUDGET           tokens of evidence a pruning method may keep (default 50000, the config's value)
# Credentials as in scripts/sol_budget.sh (Sol, DEEPRESEARCHGYM_API_KEY,
# GR_ORCH_LLM_BASE_URL for bc/dpo) and scripts/sol_budget_judge.sh (judge).
set -euo pipefail
cd "$(dirname "$0")/.."

stage=${1:-all}
ROOT=${ROOT:-runs/clean_drg}
BUDGETS=${BUDGETS:-"100000 150000 200000"}
QUERIES=${QUERIES:-"170566,313168,331361,390111,442352,546111,757743,758081,832775,883054"}
ARMS=${ARMS:-"none extractive prompted prompted_scores bc dpo"}
JOBS=${JOBS:-6}
CONFIG=configs/gym_sol_pilot.yaml   # depth 3, breadth 4, Sol, no total-token stop rule
STOPPERS="prompted prompted_scores bc dpo"
top_budget=$(echo $BUDGETS | tr ' ' '\n' | sort -n | tail -1)
fork=third_party/gpt-researcher
mkdir -p "$ROOT/logs"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$ROOT/logs/clean_run.log"; }

load_env() {
  if [ -f .env ]; then set -a; source .env; set +a; fi
  if [ -n "${SOL_API_KEY:-}" ]; then
    export OPENAI_API_KEY=$SOL_API_KEY OPENAI_BASE_URL=${SOL_BASE_URL:?SOL_BASE_URL is required with SOL_API_KEY}
  elif [ -n "${AZURE_OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
    export OPENAI_BASE_URL=${SOL_BASE_URL:-https://202610-deepres.openai.azure.com/openai/v1/}
  fi
  unset GR_AZURE_SERVICE_TIER LLM_KWARGS GR_STOP_TOTAL_TOKENS GR_STOP_HARD_CAP
}

preflight() {
  load_env
  if [ "${DRYRUN:-0}" = 1 ]; then log "DRYRUN: skipping preflight"; return; fi
  local fail=0
  : "${WRITER_MAX_INPUT_TOKENS:?set WRITER_MAX_INPUT_TOKENS (writer model context limit minus ~20K)}"
  grep -q NO_WORD_CAP "$fork/gpt_researcher/skills/deep_research.py" 2>/dev/null \
    && grep -q GR_RESEARCH_BUDGET "$fork/gpt_researcher/skills/deep_research.py" \
    || { log "FAIL fork lacks GR_RESEARCH_BUDGET / uncapped writer: git -C $fork switch aadya/drb-azure-flex && git -C $fork pull"; fail=1; }
  [ -n "${OPENAI_API_KEY:-}" ] || { log "FAIL no Sol key"; fail=1; }
  [ -n "${DEEPRESEARCHGYM_API_KEY:-}" ] || { log "FAIL no DEEPRESEARCHGYM_API_KEY"; fail=1; }
  [ -n "${JUDGE_API_KEY:-}" ] || log "WARN no JUDGE_API_KEY: the judge will use OPENAI_* from .env"
  # One tiny Sol call (a few tokens) so a bad key or endpoint fails here, not mid-run.
  .venv/bin/python - <<'PY' || { log "FAIL Sol endpoint did not answer"; fail=1; }
import os, sys
from openai import OpenAI
model = os.environ.get("SOL_MODEL", "gpt-6.1-sol-1gtjfzlv8i")
try:
    OpenAI().chat.completions.create(model=model, messages=[{"role": "user", "content": "ok"}], max_completion_tokens=16)
except Exception as e:
    print(f"Sol check: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr); sys.exit(1)
PY
  if echo " $ARMS " | grep -Eq " (bc|dpo) "; then
    local ok=1
    for m in pilot-bc-teacher pilot-dpo-teacher; do
      curl -fsS -m 20 -H "Authorization: Bearer ${GR_ORCH_LLM_API_KEY:-EMPTY}" \
        "${GR_ORCH_LLM_BASE_URL:-http://unset}/models" 2>/dev/null | grep -q "\"$m\"" || ok=0
    done
    if [ $ok = 0 ]; then
      if [ "${SKIP_LEARNED:-0}" = 1 ]; then
        ARMS=$(echo " $ARMS " | sed 's/ bc / /; s/ dpo / /' | xargs); log "WARN vLLM adapters unreachable: running without bc/dpo"
      else
        log "FAIL GR_ORCH_LLM_BASE_URL does not serve pilot-bc-teacher / pilot-dpo-teacher (SKIP_LEARNED=1 to run without them)"; fail=1
      fi
    fi
  fi
  [ $fail = 0 ] || exit 1
  log "preflight ok: arms=[$ARMS] budgets=[$BUDGETS] queries=$(echo "$QUERIES" | tr ',' '\n' | wc -l | tr -d ' ')"
}

# run_arm <out_root> <arm> <budget> <force_full 0|1> <query ids>
run_arm() {
  local out=$1 arm=$2 budget=$3 force=$4 ids=$5 policy=$2
  local extra=(--env "GR_RESEARCH_BUDGET=$budget" --env "GR_BUDGET_OUTPUT_WEIGHT=5"
    --env "GR_BUDGET_FORCE_FULL=$force" --env "GR_REPORT_WORDS=3000" --env "GR_REPORT_REWRITE=0"
    --env "GR_MAX_CONTEXT_WORDS=0" --env "GR_WRITER_MAX_INPUT_TOKENS=$WRITER_MAX_INPUT_TOKENS"
    # The agent config sets a 1,500-word per-page cut; the environment wins, so turn it off here.
    --env "GR_FINAL_ITEM_MAX_WORDS=0"
    # How much evidence a pruning method may keep (no pruning keeps everything).
    --env "GR_CONTEXT_BUDGET_TOKENS=${CONTEXT_BUDGET:-50000}")
  case $arm in
    bc|dpo) policy=learned
      extra+=(--env "GR_ORCH_LLM_BASE_URL=${GR_ORCH_LLM_BASE_URL:-}" --env "GR_ORCH_LLM_MODEL=pilot-$arm-teacher"
              --env "GR_ORCH_LLM_API_KEY=${GR_ORCH_LLM_API_KEY:-EMPTY}") ;;
  esac
  mkdir -p "$out/$arm"
  if [ "${DRYRUN:-0}" = 1 ]; then log "DRYRUN adr rollouts --out $out/$arm --policy $policy --query-ids $ids ${extra[*]}"; return; fi
  .venv/bin/adr rollouts -c "$CONFIG" --out "$out/$arm" --policy "$policy" \
    --n-seeds 1 --seed-base 0 --query-ids "$ids" "${extra[@]}" >>"$out/$arm.log" 2>&1 \
    && log "done $out/$arm" || log "ERROR $out/$arm (see $out/$arm.log); rerun to resume"
}

# All (mode, budget, arm) jobs, JOBS at a time. Methods run in parallel processes
# (gpt-researcher's trackers are per process), queries within an arm in order.
generate() {  # generate <root> <query ids> <budgets>
  local root=$1 ids=$2 budgets=$3 todo=()
  for b in $budgets; do for a in $ARMS; do todo+=("$root/equal_$((b/1000))k|$a|$b|1"); done; done
  for a in $ARMS; do echo " $STOPPERS " | grep -q " $a " && todo+=("$root/stop_$((top_budget/1000))k|$a|$top_budget|0"); done
  local caf=(); command -v caffeinate >/dev/null && caf=(caffeinate -i)
  for j in "${todo[@]}"; do
    IFS='|' read -r out arm b force <<<"$j"
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
    log "start $out/$arm (budget $b, force_full $force)"
    run_arm "$out" "$arm" "$b" "$force" "$ids" &
  done
  wait
}

judge_all() {  # judge_all <root>
  local root=$1
  for pass in 1 2 3; do
    for d in "$root"/*_*k; do
      [ -d "$d" ] || continue
      log "judge pass $pass: $d"
      JUDGE_PASS=$pass RUN_ROOT=$d JOBS=${JUDGE_JOBS:-3} bash scripts/sol_budget_judge.sh drg >>"$root/logs/judge.log" 2>&1 \
        || log "WARN judge pass $pass of $d had failures; rerun 'judge' to retry"
    done
  done
}

case $stage in
  smoke)
    preflight
    first=${QUERIES%%,*}
    generate "$ROOT" "$first" "$top_budget"
    .venv/bin/python scripts/clean_results.py check "$ROOT" --query "$first" ;;
  full)
    preflight; generate "$ROOT" "$QUERIES" "$BUDGETS" ;;
  judge)
    load_env; judge_all "$ROOT" ;;
  table)
    .venv/bin/python scripts/clean_results.py table "$ROOT" ;;
  all)
    preflight
    first=${QUERIES%%,*}
    generate "$ROOT" "$first" "$top_budget"   # the smoke runs are reused by the full stage
    .venv/bin/python scripts/clean_results.py check "$ROOT" --query "$first" || { log "smoke checks failed: fix, then rerun"; exit 1; }
    generate "$ROOT" "$QUERIES" "$BUDGETS"
    judge_all "$ROOT"
    .venv/bin/python scripts/clean_results.py table "$ROOT"
    log "ALL DONE: $ROOT/RESULTS.md" ;;
  *) echo "stage: all | smoke | full | judge | table" >&2; exit 1 ;;
esac
