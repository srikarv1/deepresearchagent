#!/usr/bin/env bash
# Token-budget experiment on GPT-6.1 Sol (Azure Flex): generate reports with
# every method held to the same total token budget per query (150K).
# See docs/token_budget_experiment.md.
#
#   bash scripts/sol_budget.sh <drg|drb> <none|prompted|extractive|bc|dpo> [query ids] [adr rollouts options]
#
#   bash scripts/sol_budget.sh drg none                  # the 50 DeepResearchGym pilot queries
#   bash scripts/sol_budget.sh drb bc                    # DeepResearch Bench queries 61-75
#   bash scripts/sol_budget.sh drg dpo 546111,442352     # only these queries
#   BUDGET=100000 bash scripts/sol_budget.sh drg none    # another budget, own run root
#
# BUDGET (default 150000) is the token budget per query. It overrides
# stop_total_tokens of the config and picks the run root, so budgets never mix:
# runs/<gym|drb>_sol_budget<BUDGET/1000>/<arm>.
#
# Resumable: a finished query is skipped, so rerun the same command after any
# interruption. Reports go to runs/<gym|drb>_sol_budget150/<arm>/<query>-s0; each
# trajectory carries the per-call LLM log (final_stats.llm_calls) that
# scripts/flex_queue_adjust.py uses to take Azure Flex queueing out of latency.
#
# Needs the gpt-researcher fork on branch aadya/drb-azure-flex (Sol on Flex, the
# GR_STOP_TOTAL_TOKENS stopping rule and its hard cap, fork PRs #12 and #14).
#
# Credentials, from the environment or .env:
#   Sol:      SOL_API_KEY + SOL_BASE_URL, else AZURE_OPENAI_API_KEY (202610-deepres),
#             else OPENAI_API_KEY + OPENAI_BASE_URL as they are.
#   drg:      DEEPRESEARCHGYM_API_KEY        drb: TAVILY_API_KEY or TAVILY_KEYS_CSV
#   bc / dpo: GR_ORCH_LLM_BASE_URL (the vLLM tunnel, ending in /v1) serving
#             pilot-bc-teacher and pilot-dpo-teacher.
set -euo pipefail
cd "$(dirname "$0")/.."

bench=${1:?bench: drg | drb}
arm=${2:?arm: none | prompted | extractive | bc | dpo}
shift 2
ids=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then ids=$1; shift; fi

case $bench in
  drg) config=configs/gym_sol_pilot_d2_budget150.yaml; root=runs/gym_sol_budget150
       default_ids="546111,442352,883054,331361,390111,313168,170566,832775,758081,757743,567339,709678,303673,114491,997518,29381,273007,408274,891387,806512,446507,274220,940274,522801,872836,351339,686941,920807,736389,779551,424460,138817,579648,65128,142998,890633,862469,958655,197129,152605,1001115,791659,584169,606941,914000,766053,226603,373908,593230,131990" ;;
  drb) config=configs/gpt_researcher_bench_budget150.yaml; root=runs/drb_sol_budget150
       default_ids="61,62,63,64,65,66,67,68,69,70,71,72,73,74,75" ;;
  *) echo "bench must be drg or drb, got '$bench'" >&2; exit 1 ;;
esac
ids=${ids:-$default_ids}
budget=${BUDGET:-150000}
[[ "$budget" =~ ^[0-9]+$ ]] && [ "$budget" -ge 1000 ] || { echo "BUDGET must be a token count, got '$budget'" >&2; exit 1; }
root=${root%budget150}budget$((budget / 1000))
root=${RUN_ROOT:-$root}  # e.g. a separate root for a latency test

if [ -f .env ]; then set -a; source .env; set +a; fi

# Sol credentials for generation.
if [ -n "${SOL_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$SOL_API_KEY
  export OPENAI_BASE_URL=${SOL_BASE_URL:?SOL_BASE_URL is required with SOL_API_KEY}
elif [ -n "${AZURE_OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
  export OPENAI_BASE_URL=${SOL_BASE_URL:-https://202610-deepres.openai.azure.com/openai/v1/}
fi
: "${OPENAI_API_KEY:?no Sol key: set SOL_API_KEY, AZURE_OPENAI_API_KEY or OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?no Sol endpoint: set SOL_BASE_URL or OPENAI_BASE_URL}"

# Sol always runs on Azure Flex in this experiment; a leftover override or
# LLM_KWARGS must not move it to another tier.
unset GR_AZURE_SERVICE_TIER LLM_KWARGS

fork=third_party/gpt-researcher
grep -q GR_STOP_HARD_CAP "$fork/gpt_researcher/skills/deep_research.py" 2>/dev/null \
  && [ -f "$fork/gpt_researcher/utils/llm_call_log.py" ] || {
  echo "$fork has no GR_STOP_HARD_CAP token cap: git -C $fork switch aadya/drb-azure-flex && git -C $fork pull" >&2; exit 1; }

extra=(--env "GR_STOP_TOTAL_TOKENS=$budget")
case $arm in
  none|prompted|extractive) policy=$arm ;;
  bc|dpo)
    policy=learned
    : "${GR_ORCH_LLM_BASE_URL:?bc/dpo need GR_ORCH_LLM_BASE_URL (the vLLM tunnel, ending in /v1)}"
    model=pilot-$arm-teacher
    if ! curl -fsS -m 20 -H "Authorization: Bearer ${GR_ORCH_LLM_API_KEY:-EMPTY}" \
         "${GR_ORCH_LLM_BASE_URL%/}/models" | grep -q "\"$model\""; then
      echo "$GR_ORCH_LLM_BASE_URL does not list $model: is the adapter server up?" >&2; exit 1
    fi
    extra+=(--env "GR_ORCH_LLM_BASE_URL=$GR_ORCH_LLM_BASE_URL" --env "GR_ORCH_LLM_MODEL=$model"
            --env "GR_ORCH_LLM_API_KEY=${GR_ORCH_LLM_API_KEY:-EMPTY}") ;;
  *) echo "arm must be none, prompted, extractive, bc or dpo, got '$arm'" >&2; exit 1 ;;
esac

case $bench in
  drg) : "${DEEPRESEARCHGYM_API_KEY:?drg needs DEEPRESEARCHGYM_API_KEY}" ;;
  drb) [ -n "${TAVILY_API_KEY:-}${TAVILY_KEYS_CSV:-}" ] || { echo "drb needs TAVILY_API_KEY or TAVILY_KEYS_CSV" >&2; exit 1; } ;;
esac

# Search must work before any money is spent: with exhausted keys Tavily answers
# HTTP 432 and every report would be written from no evidence. Tests the keys
# the run will use (the fork's key pool env/tavily.csv or TAVILY_KEYS_CSV, else
# TAVILY_API_KEY) with one basic search each (1 credit); keys are never printed.
if [ "$bench" = drb ]; then
  .venv/bin/python - "$fork" <<'PY' || { echo "No Tavily key with credit (HTTP 432 = out of credits, 401 = bad key). Not starting." >&2; exit 1; }
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
    except urllib.error.HTTPError as e:
        print(f"Tavily key ...{k[-4:]}: HTTP {e.code}", file=sys.stderr)
    except Exception as e:
        print(f"Tavily key ...{k[-4:]}: {type(e).__name__}", file=sys.stderr)
print(f"Tavily: {ok}/{len(keys)} keys have credit", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
fi

keep_awake=()
command -v caffeinate >/dev/null 2>&1 && keep_awake=(caffeinate -i)

echo "token-budget run: bench=$bench arm=$arm policy=$policy budget=$budget -> $root/$arm ($(echo "$ids" | tr ',' '\n' | wc -l | tr -d ' ') queries)"
${keep_awake[@]+"${keep_awake[@]}"} .venv/bin/adr rollouts -c "$config" \
    --out "$root/$arm" --policy "$policy" \
    --n-seeds 1 --seed-base 0 --query-ids "$ids" ${extra[@]+"${extra[@]}"} "$@" &
pid=$!
started=$(mktemp); trap 'rm -f "$started"' EXIT
# Stop as soon as search is exhausted (Tavily "keys exhausted") instead of
# writing the remaining reports from no evidence. The query that was running is
# left without a trajectory, so rerunning the same command redoes it.
while kill -0 "$pid" 2>/dev/null; do
  if find "$root/$arm/logs" -name '*.log' -newer "$started" -exec grep -l "API keys exhausted" {} + 2>/dev/null | grep -q .; then
    echo "search exhausted (see $root/$arm/logs); stopping this run" >&2
    pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 3
  fi
  sleep 20
done
wait "$pid"
