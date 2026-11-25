#!/usr/bin/env bash
# One-shot VM bootstrap for the overnight BCP table. Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/runs/overnight-table"
LOG="$ROOT/runs/overnight-table/bootstrap.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== bootstrap $(date -u) ==="

if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip curl
fi

if [[ ! -x "$ROOT/.venv/bin/pip" ]]; then
  rm -rf "$ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip wheel
# CPU torch first so later extras (llmlingua / gpt-researcher) do not pull CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,bcp]"
pip install openai llmlingua
if [[ -d "$ROOT/third_party/gpt-researcher" ]]; then
  pip install -e "$ROOT/third_party/gpt-researcher"
fi
python -c "import adr, pyserini, numpy, torch, llmlingua, gpt_researcher; print('imports ok')"
echo PIP_DONE

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sudo sh
fi
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
if ! pgrep -x ollama >/dev/null 2>&1 && ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi
ollama pull qwen3-embedding:8b
echo OLLAMA_DONE
ollama list

python - <<'PY'
try:
    from llmlingua import PromptCompressor
    print("loading llmlingua-2 model...")
    PromptCompressor("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)
    print("llmlingua ready")
except Exception as e:
    print("llmlingua warmup skipped:", type(e).__name__, e)
PY

echo BOOTSTRAP_EXIT:0
