#!/usr/bin/env bash
# No arguments: offline plan and reports. --execute: fill missing paid work.
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$script_dir/.." && pwd)
source_repo=${STUDY_SOURCE:-$checkout}
if [ ! -x "$source_repo/.venv/bin/python" ] && [ -z "${STUDY_SOURCE:-}" ]; then
  origin=$(git -C "$checkout" config --get remote.origin.url || true)
  if [ -d "$origin" ] && [ -x "$origin/.venv/bin/python" ]; then
    source_repo=$origin
  fi
fi
runtime_python=${STUDY_PYTHON:-$source_repo/.venv/bin/python}
if [ ! -x "$runtime_python" ]; then
  echo "Set STUDY_PYTHON to the project Python interpreter with its dependencies installed." >&2
  exit 1
fi
exec "$runtime_python" "$script_dir/complete_study.py" --source "$source_repo" "$@"
