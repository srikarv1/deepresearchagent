#!/usr/bin/env bash
# Sequential val evals: fixed-rule, then learned (needs Colab tunnel).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
# shellcheck disable=SC1091
[ -f .env ] && . ./.env
set +a
export PATH="$PWD/.venv/bin:$PATH"

echo "=== $(date) fixed-rule val ==="
adr rollouts \
  --config configs/rollouts_drb_random.yaml \
  --out runs/eval_val_fixed \
  --split val \
  --n-seeds 1 \
  --seed-base 0 \
  -j 2 \
  --policy random \
  --timeout-s 1800 \
  --evaluate deep_research_bench \
  --env 'GR_RANDOM_PARAMS={"keep_ratio":0.8,"min_rounds":2,"p_terminate":1.0,"temperature":0.02}'
echo "FIXED_EXIT=$?"

echo "=== $(date) learned val ==="
adr rollouts \
  --config configs/rollouts_drb_random.yaml \
  --out runs/eval_val_learned \
  --split val \
  --n-seeds 1 \
  --seed-base 0 \
  -j 1 \
  --policy learned \
  --timeout-s 1800 \
  --evaluate deep_research_bench \
  --env GR_ORCH_LLM_BASE_URL="${GR_ORCH_LLM_BASE_URL:-https://denied-snake-belts-benchmark.trycloudflare.com/v1}" \
  --env GR_ORCH_LLM_MODEL=pilot-bc \
  --env GR_ORCH_LLM_API_KEY=EMPTY
echo "LEARNED_EXIT=$?"
echo "=== $(date) done ==="
