#!/usr/bin/env bash
# Write the reports of smaller budgets from a larger-budget run (scripts/budget_replay.py),
# with the same writer, model and Azure Flex tier as scripts/sol_budget.sh.
#
#   BUDGET=200000 bash scripts/sol_budget.sh drg extractive            # research once, generous budget
#   bash scripts/sol_budget_replay.sh drg extractive 100000 150000     # reports at 100K and 150K
#   RUN_ROOT=runs/gym_sol_replay150 bash scripts/sol_budget_judge.sh drg extractive
#   python scripts/budget_table.py drg --root runs/gym_sol_replay150
#
# Source runs: runs/<gym|drb>_sol_budget${SOURCE_BUDGET:-200000}/<arm>. Output:
# runs/<gym|drb>_sol_replay<K>/<arm>/<query>-s0, one root per budget.
# REPLAY_QUERIES=id1,id2 limits it to those queries.
set -euo pipefail
cd "$(dirname "$0")/.."
bench=${1:?bench: drg | drb}; arm=${2:?arm}; shift 2
[ $# -gt 0 ] || { echo "give at least one budget, e.g. 100000 150000" >&2; exit 1; }
case $bench in
  drg) prefix=runs/gym_sol; agent=configs/agents/gpt_researcher_gym_gptsol_d2_budget150.yaml ;;
  drb) prefix=runs/drb_sol; agent=configs/agents/gpt_researcher_bench_budget150.yaml ;;
  *) echo "bench must be drg or drb" >&2; exit 1 ;;
esac
src=${prefix}_budget$(( ${SOURCE_BUDGET:-200000} / 1000 ))/$arm
[ -d "$src" ] || { echo "no source runs at $src" >&2; exit 1; }

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "${SOL_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$SOL_API_KEY OPENAI_BASE_URL=${SOL_BASE_URL:?SOL_BASE_URL is required with SOL_API_KEY}
elif [ -n "${AZURE_OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
  export OPENAI_BASE_URL=${SOL_BASE_URL:-https://202610-deepres.openai.azure.com/openai/v1/}
fi
: "${OPENAI_API_KEY:?no Sol key}"
unset GR_AZURE_SERVICE_TIER LLM_KWARGS
# The writer's models and the final-context settings, from the agent config.
eval "$(.venv/bin/python scripts/budget_replay.py env "$agent")"
.venv/bin/python scripts/budget_replay.py check "$src" ${REPLAY_QUERIES:+--queries "$REPLAY_QUERIES"}
exec .venv/bin/python scripts/budget_replay.py write "$src" --budgets "$@" --out "${prefix}_replay" ${REPLAY_QUERIES:+--queries "$REPLAY_QUERIES"}
