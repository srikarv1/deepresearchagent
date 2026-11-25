#!/usr/bin/env bash
# Idempotent bootstrap for the deepResearchAgent (adr) harness.
# Runs after the repository is checked out. Safe to run repeatedly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The stdlib venv module needs the distro venv package to bootstrap pip.
if ! dpkg -s python3.12-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

# Project virtualenv. Recreated only when missing so reruns are cheap.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip -q
# Core harness + dev/test dependencies (pytest, numpy).
pip install -e ".[dev]"

# Official judge checkouts (DeepResearch Bench, DeepResearchGym, gpt-researcher
# fork). The script pulls existing clones and clones missing ones, so it is
# idempotent across reruns.
bash scripts/bootstrap_third_party.sh

# Provide a .env so `adr doctor` reads defaults; real keys stay in secrets.
if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "adr environment ready. Activate with: source .venv/bin/activate"
