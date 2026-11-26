#!/usr/bin/env bash
# One command for the clean experiment on DeepResearchGym (BENCH=drg, default)
# or DeepResearch Bench (BENCH=drb): preflight, threshold tuning for the scored
# prompted orchestrator on dev queries, a 1-query smoke test (the first query at
# the top budget, reused by the full runs) with automatic checks, the full runs,
# three judge passes and the result tables. Every stage is resumable: rerun the
# same command after any interruption and finished runs / judgings are skipped.
#
#   bash scripts/clean_run.sh all        # preflight -> tune -> smoke -> check -> full -> judge x3 -> tables
#   bash scripts/clean_run.sh tune       # prompted_scores thresholds from dev queries -> $ROOT/thresholds.env
#   bash scripts/clean_run.sh smoke      # preflight + smoke + check only
#   bash scripts/clean_run.sh full       # full runs only (assumes smoke passed)
#   bash scripts/clean_run.sh judge      # judge passes 1..3 of everything written
#   bash scripts/clean_run.sh table      # rebuild RESULTS.md / results.tex / results.json / per_query.csv
#   BENCH=drb bash scripts/clean_run.sh all
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
#   BENCH                    drg (default) | drb
#   QUERIES                  default: the 10 DRG pilot queries of the budget sweep / DRB tasks 61-70
#   DEV_QUERIES              tuning queries, never evaluated (default: 3 DRG val-split ids / DRB 71-73)
#   JUDGE_JOBS               parallel judgings (default 3; DRB uses one worker copy each)
#   JINA_READER=public       DRB FACT through the keyless Jina reader (slower) instead of JINA_API_KEY
#   ARMS                     default "none extractive prompted prompted_scores bc dpo"
#   JOBS                     parallel arm processes (default 6)
#   ROOT                     default runs/clean_<bench>
#   SKIP_LEARNED=1           drop bc/dpo if the vLLM server is not reachable
#   DRYRUN=1                 print the runs it would start, spend nothing
#   CONTEXT_BUDGET           tokens of evidence a pruning method may keep (default 50000, the config's value)
# Credentials as in scripts/sol_budget.sh (Sol; DEEPRESEARCHGYM_API_KEY for drg,
# TAVILY_API_KEY / TAVILY_KEYS_CSV for drb; GR_ORCH_LLM_BASE_URL for bc/dpo) and
# the judge (JUDGE_API_KEY + JUDGE_BASE_URL; drb FACT also JINA_API_KEY or JINA_READER=public).
set -euo pipefail
cd "$(dirname "$0")/.."

stage=${1:-all}
BENCH=${BENCH:-drg}
case $BENCH in
  drg) CONFIG=configs/gym_sol_pilot.yaml   # depth 3, breadth 4, Sol, no total-token stop rule
       QUERIES=${QUERIES:-"170566,313168,331361,390111,442352,546111,757743,758081,832775,883054"}
       DEV_QUERIES=${DEV_QUERIES:-"170195,262266,273800"} ;;   # val split, not in QUERIES
  drb) CONFIG=configs/drb_sol_clean.yaml   # depth 3, breadth 4, Sol, Tavily, no total-token stop rule
       QUERIES=${QUERIES:-"61,62,63,64,65,66,67,68,69,70"}
       DEV_QUERIES=${DEV_QUERIES:-"71,72,73"} ;;
  *) echo "BENCH must be drg or drb, got '$BENCH'" >&2; exit 1 ;;
esac
ROOT=${ROOT:-runs/clean_$BENCH}
BUDGETS=${BUDGETS:-"100000 150000 200000"}
ARMS=${ARMS:-"none extractive prompted prompted_scores bc dpo"}
JOBS=${JOBS:-6}
STOPPERS="prompted prompted_scores bc dpo"
top_budget=$(echo $BUDGETS | tr ' ' '\n' | sort -n | tail -1)
fork=third_party/gpt-researcher
mkdir -p "$ROOT/logs"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$ROOT/logs/clean_run.log"; }
# Keep the machine awake while this script runs (macOS).
command -v caffeinate >/dev/null && (caffeinate -i -w $$ >/dev/null 2>&1 &)   # subshell: not one of our jobs, so wait / jobs ignore it

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
  if [ "$BENCH" = drg ]; then
    [ -n "${DEEPRESEARCHGYM_API_KEY:-}" ] || { log "FAIL no DEEPRESEARCHGYM_API_KEY"; fail=1; }
    [ -n "${JUDGE_API_KEY:-}" ] || log "WARN no JUDGE_API_KEY: the judge will use OPENAI_* from .env"
  else
    [ -n "${JUDGE_API_KEY:-}" ] && [ -n "${JUDGE_BASE_URL:-}" ] || { log "FAIL drb judging needs JUDGE_API_KEY + JUDGE_BASE_URL"; fail=1; }
    [ -n "${JINA_API_KEY:-}" ] || [ "${JINA_READER:-}" = public ] || { log "FAIL drb FACT needs JINA_API_KEY or JINA_READER=public"; fail=1; }
    grep -q RETRY_PATCH third_party/deep_research_bench/utils/api.py 2>/dev/null \
      || { log "FAIL third_party/deep_research_bench is not patched: bash scripts/bootstrap_third_party.sh"; fail=1; }
    # One basic Tavily search per key (1 credit each): exhausted keys would write reports from no evidence.
    .venv/bin/python - "$fork" <<'PY' || { log "FAIL no Tavily key with credit"; fail=1; }
import csv, json, os, sys, urllib.request, urllib.error
from pathlib import Path
pool = Path(sys.argv[1]) / "env" / "tavily.csv"
if not pool.exists() and os.environ.get("TAVILY_KEYS_CSV"):
    pool = Path(os.environ["TAVILY_KEYS_CSV"])
keys = [r["api_key"].strip() for r in csv.DictReader(open(pool)) if (r.get("api_key") or "").strip()] if pool.exists() else []
keys = keys or [k for k in [os.environ.get("TAVILY_API_KEY", "")] if k]
ok = 0
for k in keys:
    req = urllib.request.Request("https://api.tavily.com/search", method="POST",
        data=json.dumps({"query": "preflight", "max_results": 1, "search_depth": "basic"}).encode(),
        headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"})
    try:
        ok += urllib.request.urlopen(req, timeout=30).status == 200
    except Exception as e:
        print(f"Tavily key ...{k[-4:]}: {getattr(e, 'code', type(e).__name__)}", file=sys.stderr)
print(f"Tavily: {ok}/{len(keys)} keys have credit", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
  fi
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
  # Thresholds of the scored prompted orchestrator, from the tune stage.
  if [ "$arm" = prompted_scores ] && [ -f "$ROOT/thresholds.env" ]; then
    while IFS= read -r kv; do [ -n "$kv" ] && extra+=(--env "$kv"); done < "$ROOT/thresholds.env"
  fi
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
  if [ "$BENCH" = drb ]; then
    local roots=(); for d in "$root"/*_*k; do [ -d "$d" ] && roots+=("$d"); done
    [ ${#roots[@]} -gt 0 ] || return 0
    .venv/bin/python scripts/judge_drb_parallel.py --setup -n "${JUDGE_JOBS:-3}" >>"$root/logs/judge.log" 2>&1
    log "judge drb passes 1-3: ${roots[*]}"
    .venv/bin/python scripts/judge_drb_parallel.py -n "${JUDGE_JOBS:-3}" "${roots[@]}" >>"$root/logs/judge.log" 2>&1 \
      || log "WARN drb judging had failures; rerun 'judge' to retry"
    return 0
  fi
  for pass in 1 2 3; do
    for d in "$root"/*_*k; do
      [ -d "$d" ] || continue
      log "judge pass $pass: $d"
      JUDGE_PASS=$pass RUN_ROOT=$d JOBS=${JUDGE_JOBS:-3} bash scripts/sol_budget_judge.sh drg >>"$root/logs/judge.log" 2>&1 \
        || log "WARN judge pass $pass of $d had failures; rerun 'judge' to retry"
    done
  done
}

# Thresholds for prompted_scores from dev queries (never evaluated): prompted and
# prompted_scores at the top budget with every round run (force_full), then the
# keep / stop thresholds that make prompted_scores keep and stop like prompted.
tune() {
  [ -f "$ROOT/thresholds.env" ] && { log "thresholds: $(tr '\n' ' ' < "$ROOT/thresholds.env")(delete $ROOT/thresholds.env to retune)"; return 0; }
  echo " $ARMS " | grep -q " prompted_scores " || return 0
  local t="$ROOT/tune/equal_$((top_budget/1000))k"
  for a in prompted prompted_scores; do run_arm "$t" "$a" "$top_budget" 1 "$DEV_QUERIES" & done
  wait
  [ "${DRYRUN:-0}" = 1 ] && return 0
  .venv/bin/python scripts/clean_results.py tune "$t" --out "$ROOT/thresholds.env" | tee -a "$ROOT/logs/clean_run.log"
}

case $stage in
  tune)
    preflight; tune ;;
  smoke)
    preflight
    tune
    first=${QUERIES%%,*}
    generate "$ROOT" "$first" "$top_budget"
    .venv/bin/python scripts/clean_results.py check "$ROOT" --bench "$BENCH" --query "$first" ;;
  full)
    preflight; tune; generate "$ROOT" "$QUERIES" "$BUDGETS" ;;
  judge)
    load_env; judge_all "$ROOT" ;;
  table)
    .venv/bin/python scripts/clean_results.py table "$ROOT" --bench "$BENCH" ;;
  all)
    preflight
    tune
    first=${QUERIES%%,*}
    generate "$ROOT" "$first" "$top_budget"   # the smoke runs are reused by the full stage
    .venv/bin/python scripts/clean_results.py check "$ROOT" --bench "$BENCH" --query "$first" || { log "smoke checks failed: fix, then rerun"; exit 1; }
    generate "$ROOT" "$QUERIES" "$BUDGETS"
    judge_all "$ROOT"
    .venv/bin/python scripts/clean_results.py table "$ROOT" --bench "$BENCH"
    log "ALL DONE: $ROOT/RESULTS.md" ;;
  *) echo "stage: all | tune | smoke | full | judge | table" >&2; exit 1 ;;
esac
