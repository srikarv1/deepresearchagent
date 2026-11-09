#!/usr/bin/env bash
# Generate DeepResearchGym pilot reports on GPT-6.1 Sol with the resumable
# rollout driver (one seed per query; finished reports are skipped on rerun).
#
#   bash scripts/gym_sol_pilot.sh <none|prompted|extractive> <comma-separated query ids> [-j N]
#
# The Qwen orchestrator arms run the learned policy against the Colab endpoint
# from notebooks/wsp_qwen_orchestrator.ipynb; ARM names their output directory:
#
#   ARM=learned-wsp bash scripts/gym_sol_pilot.sh learned <ids> \
#       --env GR_ORCH_LLM_BASE_URL=<tunnel>/v1 --env GR_ORCH_LLM_MODEL=qwen-wsp \
#       --env GR_ORCH_LLM_API_KEY=EMPTY
#
# PILOT picks the config and the run root: configs/$PILOT.yaml, runs/$PILOT/
# (default gym_sol_pilot, depth 3; gym_sol_pilot_d2 is depth 2).
#
# Generation only: OPENAI_* point at the 202610-deepres resource, so judging
# runs separately (adr evaluate -c configs/gym_sol_pilot.yaml) from the judge
# credentials.
#
# Needs the gpt-researcher fork branch srikar/sol-pilot (Sol support, ranked final
# context): GR_BRANCH=srikar/sol-pilot bash scripts/bootstrap_third_party.sh
set -euo pipefail
cd "$(dirname "$0")/.."

policy=${1:?policy: none | prompted | extractive | learned}
ids=${2:?comma-separated query ids}
shift 2
arm=${ARM:-$policy}
pilot=${PILOT:-gym_sol_pilot}
[ -f "configs/$pilot.yaml" ] || { echo "no configs/$pilot.yaml" >&2; exit 1; }

set -a; source .env; set +a
export OPENAI_API_KEY=$AZURE_OPENAI_API_KEY
export OPENAI_BASE_URL=https://202610-deepres.openai.azure.com/openai/v1/

exec .venv/bin/adr rollouts -c "configs/$pilot.yaml" \
    --out "runs/$pilot/$arm" --policy "$policy" \
    --n-seeds 1 --seed-base 0 --query-ids "$ids" "$@"
