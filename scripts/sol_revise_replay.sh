#!/usr/bin/env bash
# Research less, write better (scripts/revise_replay.py) with the same writer,
# model and settings as scripts/sol_budget_replay.sh.
#
#   bash scripts/sol_revise_replay.sh drg none            # paid: writer + revision calls
#   for v in b100rev b100polish; do for p in 1 2 3; do RUN_ROOT=runs/gym_sol_$v JUDGE_PASS=$p bash scripts/sol_budget_judge.sh drg none; done; done
#   python scripts/budget_table.py drg --root runs/gym_sol_r1rev
#
# Source runs: runs/<gym|drb>_sol_budget${SOURCE_BUDGET:-200000}/<arm>.
# REPLAY_QUERIES=id1,id2 limits it to those queries; VARIANTS="r1 r1rev fullrev" for the other variants (default: b100rev b100polish).
set -euo pipefail
cd "$(dirname "$0")/.."
bench=${1:?bench: drg | drb}; arm=${2:?arm}
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
eval "$(.venv/bin/python scripts/budget_replay.py env "$agent")"
.venv/bin/python scripts/budget_replay.py check "$src" ${REPLAY_QUERIES:+--queries "$REPLAY_QUERIES"}
exec .venv/bin/python scripts/revise_replay.py write "$src" --out "$prefix" --variants ${VARIANTS:-b100rev b100polish} \
  ${REPLAY_QUERIES:+--queries "$REPLAY_QUERIES"}
