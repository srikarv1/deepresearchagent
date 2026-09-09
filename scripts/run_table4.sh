#!/usr/bin/env bash
# Run every (policy, seed) cell of Table 4 for one benchmark, then judge each.
#
#   scripts/run_table4.sh drb                       # 5 policies x seeds 1 2 3, then judge
#   scripts/run_table4.sh browsecomp
#   POLICIES="none topk" SEEDS="1" scripts/run_table4.sh drb
#   BUDGET=48000 scripts/run_table4.sh drb          # GR_CONTEXT_BUDGET_TOKENS for pruning rows
#   LIMIT=5 scripts/run_table4.sh drb               # smoke: 5 questions per cell
#   JUDGE=0 scripts/run_table4.sh drb               # inference only
#
# Every cell is one `adr run`; every judge is one `adr evaluate`. The backbone
# lives in configs/table4/<bench>.yaml and is identical across cells; the only
# per-cell differences are --run-name, --seed and the GR_* env vars.
set -euo pipefail

BENCH="${1:?usage: run_table4.sh <drb|browsecomp>}"
POLICIES="${POLICIES:-none topk extractive llmlingua prompted}"
SEEDS="${SEEDS:-1 2 3}"
BUDGET="${BUDGET:-}"
LIMIT="${LIMIT:-}"
SAMPLE="${SAMPLE:-}"
JUDGE="${JUDGE:-1}"
OUT="${OUT:-runs}"

case "$BENCH" in
  drb)        CONFIG=configs/table4/drb.yaml;        OFFICIAL=deep_research_bench; PREFIX=drb ;;
  browsecomp) CONFIG=configs/table4/browsecomp.yaml; OFFICIAL=browsecomp;          PREFIX=bc ;;
  *) echo "unknown bench: $BENCH (drb|browsecomp)"; exit 2 ;;
esac

extra=()
[[ -n "$LIMIT"  ]] && extra+=(--limit "$LIMIT")
[[ -n "$SAMPLE" ]] && extra+=(--sample "$SAMPLE")

for policy in $POLICIES; do
  for seed in $SEEDS; do
    name="${PREFIX}-${policy}-s${seed}"
    env_args=(--agent-env "GR_ORCHESTRATOR=${policy}")
    # The budget applies to the pruning/compression rows only; `none` is the
    # unconstrained reference whose token mean sets the budget for the others.
    if [[ -n "$BUDGET" && "$policy" != "none" ]]; then
      env_args+=(--agent-env "GR_CONTEXT_BUDGET_TOKENS=${BUDGET}")
    fi
    echo "=== adr run  ${name}"
    before=$(ls -d "${OUT}"/*-"${name}" 2>/dev/null | sort | tail -n1 || true)
    adr run -c "$CONFIG" --run-name "$name" --seed "$seed" "${env_args[@]}" "${extra[@]}"
    run_dir=$(ls -d "${OUT}"/*-"${name}" | sort | tail -n1)
    if [[ "$run_dir" == "$before" ]]; then echo "run dir not created for ${name}"; exit 1; fi
    if [[ "$JUDGE" == "1" ]]; then
      echo "=== adr evaluate ${run_dir} --official ${OFFICIAL}"
      adr evaluate "$run_dir" --official "$OFFICIAL"
    fi
  done
done

echo
echo "=== table"
adr table "${OUT}"/*-"${PREFIX}"-*-s* --output "${OUT}/table4_${BENCH}.md"
