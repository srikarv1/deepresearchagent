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
#
# Resumable: a finished query is skipped, so rerun the same command after any
# interruption. Reports go to runs/<gym|drb>_sol_budget150/<arm>/<query>-s0; each
# trajectory carries the per-call LLM log (final_stats.llm_calls) that
# scripts/flex_queue_adjust.py uses to take Azure Flex queueing out of latency.
#
# Needs the gpt-researcher fork on branch aadya/drb-azure-flex (Sol on Flex and
# the GR_STOP_TOTAL_TOKENS stopping rule, fork PR #12).
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
grep -q GR_STOP_TOTAL_TOKENS "$fork/gpt_researcher/skills/deep_research.py" 2>/dev/null \
  && [ -f "$fork/gpt_researcher/utils/llm_call_log.py" ] || {
  echo "$fork has no GR_STOP_TOTAL_TOKENS stopping rule: git -C $fork switch aadya/drb-azure-flex && git -C $fork pull" >&2; exit 1; }

extra=()
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

keep_awake=()
command -v caffeinate >/dev/null 2>&1 && keep_awake=(caffeinate -i)

echo "token-budget run: bench=$bench arm=$arm policy=$policy -> $root/$arm ($(echo "$ids" | tr ',' '\n' | wc -l | tr -d ' ') queries)"
exec ${keep_awake[@]+"${keep_awake[@]}"} .venv/bin/adr rollouts -c "$config" \
    --out "$root/$arm" --policy "$policy" \
    --n-seeds 1 --seed-base 0 --query-ids "$ids" ${extra[@]+"${extra[@]}"} "$@"
