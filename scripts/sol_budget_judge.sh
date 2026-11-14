#!/usr/bin/env bash
# Judge the token-budget reports written by scripts/sol_budget.sh without
# regenerating them. A report that already has all its scores is skipped, so
# rerunning retries only failed or missing judging.
#
#   bash scripts/sol_budget_judge.sh <drg|drb> [arms...]      (default: every arm that has runs)
#   JOBS=3 bash scripts/sol_budget_judge.sh drg none bc       (3 reports at a time; default 2)
#
# drg: quality, key-point recall and citation on gpt-5-mini.
# drb: RACE and FACT with the judge in configs/eval/deep_research_bench.yaml
#      (needs JINA_API_KEY).
#
# Judge credentials: JUDGE_API_KEY + JUDGE_BASE_URL if set, else the OPENAI_*
# in .env. They must point at the judge resource (gpt-5-mini), not at Sol.
set -euo pipefail
cd "$(dirname "$0")/.."

bench=${1:?bench: drg | drb}
shift
case $bench in
  drg) config=configs/gym_sol_pilot_d2_budget150.yaml; root=runs/gym_sol_budget150; official=deep_research_gym ;;
  drb) config=configs/gpt_researcher_bench_budget150.yaml; root=runs/drb_sol_budget150; official=deep_research_bench ;;
  *) echo "bench must be drg or drb, got '$bench'" >&2; exit 1 ;;
esac

if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "${JUDGE_API_KEY:-}" ]; then
  export OPENAI_API_KEY=$JUDGE_API_KEY
  export OPENAI_BASE_URL=${JUDGE_BASE_URL:?JUDGE_BASE_URL is required with JUDGE_API_KEY}
fi
export CRAWL4AI_NO_CLOUD_NOTICE=1

arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
  for d in "$root"/*/; do [ -d "$d" ] && arms+=("$(basename "$d")"); done
fi

todo=$(.venv/bin/python - "$official" "$root" ${arms[@]+"${arms[@]}"} <<'PY'
import json, sys
from pathlib import Path
official, root, arms = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
for arm in arms:
    for d in sorted((root / arm).glob("*-s*")):
        qid = d.name.split("-s")[0]
        traj = d / "trajectories" / f"{qid}.json"
        if not traj.exists() or not json.loads(traj.read_text()).get("report"):
            continue
        try:
            block = json.loads((d / "metrics" / "summary.json").read_text())["official"][official]
        except Exception:
            block = {}
        if official == "deep_research_gym":
            done = all((block.get(m) or {}).get("n") for m in ("quality", "kpr", "citation"))
        else:
            done = bool(block.get("official")) and bool((block.get("race") or {}).get("scores")) \
                and bool((block.get("fact") or {}).get("scores"))
        if not done:
            print(d)
PY
)
n=$(echo "$todo" | grep -c . || true)
echo "$(date +%H:%M) $bench: $n report(s) to judge, ${JOBS:-2} at a time"
[ "$n" -eq 0 ] && exit 0
export BUDGET_OFFICIAL=$official BUDGET_CONFIG=$config
echo "$todo" | grep . | xargs -P "${JOBS:-2}" -n 1 sh -c '
  .venv/bin/adr evaluate "$1" --official "$BUDGET_OFFICIAL" -c "$BUDGET_CONFIG" > "$1/judge.log" 2>&1 \
    && echo "$(date +%H:%M) judged $1" || echo "$(date +%H:%M) FAILED $1 (see $1/judge.log)"' _ || true
echo "$(date +%H:%M) done; run this again to retry anything still missing, then scripts/budget_table.py $bench"
