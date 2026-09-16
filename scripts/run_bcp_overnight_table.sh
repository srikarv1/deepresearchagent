#!/usr/bin/env bash
# Overnight PILOT table: frozen gpt-researcher (depth=3, breadth=2) × librarian × GR_ORCHESTRATOR.
#
# Full grid (10 rows):
#   BM25                    × none|topk|extractive|llmlingua|prompted
#   dense Qwen3-Embedding-8B × none|topk|extractive|llmlingua|prompted
#
# Slice: BrowseComp-Plus *test* hold-out, first LIMIT queries (default 12).
# Not the official 830 Acc. Judge is gpt-5-mini (gpt-4.1 unavailable).
# Trajectories for every row are zipped to runs/overnight-table/bcp-overnight-trajectories.zip.
#
# Usage:
#   set -a; source .env; set +a
#   bash scripts/run_bcp_overnight_table.sh
#   # or: BCP_TABLE_LIMIT=8 bash scripts/run_bcp_overnight_table.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT="${BCP_TABLE_LIMIT:-12}"
POLICIES="${BCP_TABLE_POLICIES:-none topk extractive llmlingua prompted}"
# Both librarians, always. Override with BCP_TABLE_LIBRARIANS="bm25" only for a
# debug slice — the meeting table needs BM25 and dense-8B.
LIBRARIANS="${BCP_TABLE_LIBRARIANS:-bm25 dense}"
OUT_DIR="$ROOT/runs/overnight-table"
STATUS="$OUT_DIR/STATUS.md"
TABLE="$OUT_DIR/TABLE.md"
LOG="$OUT_DIR/overnight.log"
RETRIEVER_LOG="$OUT_DIR/retriever.log"
PORT="${BCP_RETRIEVER_PORT:-8321}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export JAVA_HOME
export PATH="$JAVA_HOME/bin:$PATH"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

latest_run() {
  local prefix="$1"
  ls -1dt "$ROOT"/runs/*-"$prefix" 2>/dev/null | head -1 || true
}

source_env() {
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
}

wait_for_key() {
  source_env
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    log "OPENAI_API_KEY is set (len=${#OPENAI_API_KEY})"
    return 0
  fi
  log "Waiting for OPENAI_API_KEY in $ROOT/.env or the process environment..."
  local n=0
  while [[ -z "${OPENAI_API_KEY:-}" ]]; do
    sleep 20
    source_env
    n=$((n + 1))
    if (( n % 15 == 0 )); then
      log "still waiting for OPENAI_API_KEY (${n} checks)"
    fi
  done
  log "OPENAI_API_KEY arrived (len=${#OPENAI_API_KEY})"
}

need_venv() {
  if [[ ! -x "$ROOT/.venv/bin/adr" ]]; then
    log "FATAL: $ROOT/.venv/bin/adr missing. Finish bootstrap first."
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
}

stop_retriever() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  fi
  pkill -f "adr serve-retriever" >/dev/null 2>&1 || true
  sleep 2
}

start_retriever() {
  local kind="$1"
  stop_retriever
  log "starting retriever kind=$kind on :$PORT"
  nohup adr serve-retriever --retriever "$kind" --port "$PORT" \
    >"$RETRIEVER_LOG" 2>&1 &
  local i
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log "retriever healthy: $(curl -sf "http://127.0.0.1:${PORT}/health" | head -c 200)"
      return 0
    fi
    sleep 4
  done
  log "FATAL: retriever did not become healthy. last log:"
  tail -40 "$RETRIEVER_LOG" || true
  exit 1
}

write_status() {
  local line="$1"
  {
    echo "# Overnight BCP table"
    echo
    echo "- updated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- limit: $LIMIT  split: test  depth=3 breadth=2"
    echo "- librarians: $LIBRARIANS"
    echo "- dense: Qwen3-Embedding-8B   judge: gpt-5-mini (unofficial Acc)"
    echo "- policies: $POLICIES"
    echo "- last: $line"
    echo
    echo "## runs"
    ls -1dt "$ROOT"/runs/*-bcp-overnight-* 2>/dev/null | head -20 || echo "(none yet)"
  } >"$STATUS"
}

evaluate_run() {
  local run_dir="$1"
  local cfg="$2"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    log "skip evaluate: missing $run_dir"
    return 0
  fi
  log "evaluate $run_dir"
  adr evaluate "$run_dir" --official browsecomp_plus -c "$cfg" || log "evaluate failed for $run_dir"
}

summarize() {
  python "$ROOT/scripts/summarize_bcp_table.py" --out "$TABLE" --zip "$OUT_DIR/bcp-overnight-trajectories.zip" \
    || log "summarize failed"
}

run_done() {
  local name="$1"
  local run_dir
  run_dir="$(latest_run "$name")"
  if [[ -z "$run_dir" ]]; then
    return 1
  fi
  python - "$run_dir" "$LIMIT" <<'PY'
import json, sys
from pathlib import Path
run, limit = Path(sys.argv[1]), int(sys.argv[2])
summary = run / "metrics" / "summary.json"
traj = run / "trajectories"
n_traj = len(list(traj.glob("*.json"))) if traj.is_dir() else 0
n = 0
if summary.exists():
    n = int(json.loads(summary.read_text()).get("n_queries") or 0)
sys.exit(0 if max(n, n_traj) >= limit else 1)
PY
}

run_one() {
  local name="$1"
  local cfg="$2"
  local policy="$3"
  if run_done "$name"; then
    log "skip $name (already have >= $LIMIT trajectories)"
    summarize
    return 0
  fi
  export GR_ORCHESTRATOR="$policy"
  write_status "running $name policy=$policy"
  log "adr run --run-name $name --limit $LIMIT GR_ORCHESTRATOR=$policy"
  adr run --config "$cfg" --limit "$LIMIT" --split test --run-name "$name" \
    || log "adr run failed for $name (continuing)"
  local run_dir
  run_dir="$(latest_run "$name")"
  evaluate_run "$run_dir" "$cfg"
  summarize
}

need_venv
source_env
wait_for_key
source_env

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://ssvadlam.openai.azure.com/openai/v1}"
export LLM_BACKEND="${LLM_BACKEND:-openai}"
export GR_ANSWER_FORMAT="${GR_ANSWER_FORMAT:-browsecomp}"
export GR_CONTEXT_BUDGET_TOKENS="${GR_CONTEXT_BUDGET_TOKENS:-50000}"
# pyserini imports openai; key must be present before serve-retriever.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  log "FATAL: OPENAI_API_KEY empty after wait"
  exit 1
fi

write_status "starting ${LIBRARIANS} × ${POLICIES}"

for lib in $LIBRARIANS; do
  case "$lib" in
    bm25)
      cfg="$ROOT/configs/gpt_researcher_browsecomp_plus_bm25_overnight.yaml"
      ;;
    dense)
      cfg="$ROOT/configs/gpt_researcher_browsecomp_plus_dense_overnight.yaml"
      ;;
    *)
      log "FATAL: unknown librarian $lib (want bm25 or dense)"
      exit 1
      ;;
  esac
  start_retriever "$lib"
  for pol in $POLICIES; do
    run_one "bcp-overnight-${lib}-${pol}" "$cfg" "$pol"
  done
done

write_status "all rows finished"
summarize
log "done. table -> $TABLE"
log "trajectories zip -> $OUT_DIR/bcp-overnight-trajectories.zip"
cat "$TABLE" || true
