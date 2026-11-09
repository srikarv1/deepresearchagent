#!/usr/bin/env bash
# Judge the saved gym_sol_pilot reports (quality, key-point recall, citation on
# gpt-5-mini) without regenerating them. A run dir that already has all three
# scores is skipped, so rerunning retries only failed or missing judging.
#
#   bash scripts/gym_sol_pilot_judge.sh [none] [prompted]     (default: both)
#   bash scripts/gym_sol_pilot_judge.sh learned-base learned-wsp   (the Qwen orchestrator arms)
#   PILOT=gym_sol_pilot_d2 bash scripts/gym_sol_pilot_judge.sh ...  (the depth-2 pilot)
#
# Uses the OPENAI_* in .env (the resource with gpt-5-mini), not the Sol endpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a
export CRAWL4AI_NO_CLOUD_NOTICE=1

pilot=${PILOT:-gym_sol_pilot}
policies=("$@")
[ ${#policies[@]} -eq 0 ] && policies=(none prompted)

for policy in "${policies[@]}"; do
  for d in runs/"$pilot"/"$policy"/*-s0; do
    [ -d "$d" ] || continue
    state=$(.venv/bin/python - "$d" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); qid = d.name.split("-s")[0]
traj = d / "trajectories" / f"{qid}.json"
if not traj.exists() or not json.loads(traj.read_text()).get("report"):
    print("no-report"); sys.exit()
try:
    g = json.loads((d / "metrics" / "summary.json").read_text())["official"]["deep_research_gym"]
except Exception:
    print("todo"); sys.exit()
ok = g.get("judge_model") == "gpt-5-mini" and all(
    (g.get(m) or {}).get("n") for m in ("quality", "kpr", "citation"))
print("done" if ok else "todo")
EOF
)
    case $state in
      done) echo "skip  $d (already judged)";;
      no-report) echo "skip  $d (no finished report yet)";;
      *) echo "judge $d"
         .venv/bin/adr evaluate "$d" --official deep_research_gym -c "configs/$pilot.yaml" \
           > "$d/judge.log" 2>&1 && echo "      ok" || echo "      FAILED, see $d/judge.log";;
    esac
  done
done
