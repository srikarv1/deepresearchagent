#!/usr/bin/env bash
# Pack what Stage B (Colab A100) needs into one zip: both repos' code (current
# working tree, no .git/.venv/.env), the Bedrock replay trees, judged-reward
# memo, pairwise preferences, embeddings, and the DRG judge + key points.
# Keys are NOT included: add them in Colab's Secrets panel.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-replaygym_colab_bundle.zip}"
rm -f "$OUT"
zip -qr "$OUT" \
  deepresearchagent/src deepresearchagent/scripts deepresearchagent/configs deepresearchagent/data \
  deepresearchagent/pyproject.toml deepresearchagent/tests/test_replay.py \
  deepresearchagent/runs/replaygym/bedrock \
  deepresearchagent/third_party/deepresearchgym \
  gpt-researcher/gpt_researcher gpt-researcher/pyproject.toml gpt-researcher/requirements.txt \
  -x "*/__pycache__/*" "*.pyc" "*/.git/*" "*/judge/*/export/*"
for d in deepresearchagent/runs/*replaygym-bedrock-*; do
  q=${d##*-}
  zip -qr "$OUT" "$d/gpt_researcher/$q.json" "$d/queries/$q.json" "$d/metrics/summary.json" "$d/reports/$q.md"
done
ls -lh "$OUT"
