#!/usr/bin/env bash
# Make the official judge repos reachable under third_party/.
#
# Existing clones elsewhere on disk are symlinked rather than re-cloned, so a
# checkout you already have (including a fork) is used as-is. Override the
# lookup with ADR_DRB_DIR / ADR_GYM_DIR, or set DRB_REPO_URL / GYM_REPO_URL to
# clone from a different remote.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
PARENT="$(dirname "$ROOT")"
mkdir -p "$DEST"

DRB_REPO_URL="${DRB_REPO_URL:-https://github.com/Ayanami0730/deep_research_bench.git}"
GYM_REPO_URL="${GYM_REPO_URL:-https://github.com/cxcscmu/deepresearch_benchmarking.git}"
GR_REPO_URL="${GR_REPO_URL:-https://github.com/WilliamOdinson/gpt-researcher.git}"
GR_BRANCH="${GR_BRANCH:-main}"

# link_or_clone <dest-dir> <marker-file> <repo-url> [--branch <branch>] <candidate-dir>...
link_or_clone() {
  local dest="$1"; shift
  local marker="$1"; shift
  local url="$1"; shift

  local branch=""
  if [[ "${1:-}" == "--branch" ]]; then
    branch="$2"; shift 2
  fi

  if [[ -e "$dest/$marker" ]]; then
    if [[ -d "$dest/.git" ]]; then
      echo "PULLING  $dest"
      git -C "$dest" pull --ff-only --quiet
    else
      echo "OK       $dest (symlink, skipping pull)"
    fi
    return
  fi

  for candidate in "$@"; do
    if [[ -n "$candidate" && -e "$candidate/$marker" ]]; then
      rm -rf "$dest"
      ln -s "$(cd "$candidate" && pwd)" "$dest"
      echo "LINKED   $dest -> $candidate"
      return
    fi
  done

  echo "CLONING  $url -> $dest${branch:+ (branch: $branch)}"
  if [[ -n "$branch" ]]; then
    git clone --depth 1 --branch "$branch" "$url" "$dest"
  else
    git clone --depth 1 "$url" "$dest"
  fi
}

link_or_clone "$DEST/deep_research_bench" "deepresearch_bench_race.py" "$DRB_REPO_URL" \
  "${ADR_DRB_DIR:-}" "$PARENT/deep_research_bench"

link_or_clone "$DEST/deepresearchgym" "eval_quality_async.py" "$GYM_REPO_URL" \
  "${ADR_GYM_DIR:-}" "$PARENT/deepresearchgym" "$PARENT/deepresearch_benchmarking"

# Patch Gym judges: (1) temperature=0 guard for gpt-5-mini,
# (2) exponential backoff retry on 429 rate limit errors.
_patch_gym_judges() {
  local gym="$DEST/deepresearchgym"
  local patched=0

  # --- temperature guard ---
  local t_pattern='temperature=0'
  local t_replacement='**({"temperature": 0} if model not in ("gpt-5-mini",) else {})'
  for f in \
    "$gym/eval_quality_async.py" \
    "$gym/eval_kpr_async.py" \
    "$gym/eval_citation_async.py" \
    "$gym/eval_citation_recall_async.py" \
    "$gym/eval_citation_clueweb_async.py" \
    "$gym/key_point/aggregate.py" \
    "$gym/key_point/key_point_extract.py"; do
    [[ -f "$f" ]] || continue
    if grep -q "$t_pattern" "$f" 2>/dev/null; then
      sed -i.bak "s|$t_pattern|$t_replacement|g" "$f" && rm -f "$f.bak"
      patched=$((patched + 1))
    fi
  done

  # --- retry wrapper injection ---
  # Adds _retry_parse() with exponential backoff (max 5 attempts, 2/4/8/16/32s)
  # and replaces bare client.beta.chat.completions.parse calls with it.
  local retry_marker='# _retry_parse injected'
  for f in \
    "$gym/eval_quality_async.py" \
    "$gym/eval_kpr_async.py" \
    "$gym/eval_citation_async.py" \
    "$gym/eval_citation_recall_async.py" \
    "$gym/eval_citation_clueweb_async.py"; do
    [[ -f "$f" ]] || continue
    if grep -q "$retry_marker" "$f" 2>/dev/null; then
      continue  # already patched
    fi
    # Insert retry helper after the client = AsyncOpenAI(...) line
    sed -i.bak '/^client = AsyncOpenAI/a\
\
'"$retry_marker"'\
import random as _rand\
async def _retry_parse(**kwargs):\
    \"\"\"Retry client.beta.chat.completions.parse with exponential backoff.\"\"\"\
    for _attempt in range(5):\
        try:\
            return await client.beta.chat.completions.parse(**kwargs)\
        except Exception as _e:\
            if "429" in str(_e) or "rate_limit" in str(_e):\
                _wait = (2 ** (_attempt + 1)) + _rand.random()\
                print(f"Rate limited, retrying in {_wait:.1f}s (attempt {_attempt + 1}/5)")\
                await asyncio.sleep(_wait)\
            else:\
                raise\
    return await client.beta.chat.completions.parse(**kwargs)
' "$f" && rm -f "$f.bak"
    # Replace calls outside of _retry_parse. The function body uses
    # _orig_parse to avoid infinite recursion.
    sed -i.bak 's|await client\.beta\.chat\.completions\.parse(|await _retry_parse(|g' "$f" && rm -f "$f.bak"
    # _retry_parse itself must call the original client method, not itself.
    sed -i.bak 's|return await _retry_parse(\*\*kwargs)|return await client.beta.chat.completions.parse(**kwargs)|g' "$f" && rm -f "$f.bak"
    patched=$((patched + 1))
  done

  if [[ $patched -gt 0 ]]; then
    echo "PATCHED  ${patched} Gym judge files (temperature guard + retry backoff)"
  fi
}
_patch_gym_judges

# Agent under test, not a judge. Marker is the fork's trajectory logger so a
# plain upstream checkout is not mistaken for the instrumented one.
link_or_clone "$DEST/gpt-researcher" "gpt_researcher/utils/trajectory_logger.py" "$GR_REPO_URL" \
  --branch "$GR_BRANCH" \
  "${ADR_GR_DIR:-}" "$PARENT/gpt-researcher" "$PARENT/gpt_researcher"

echo
echo "Judge dependencies (install into the same env that runs adr):"
echo "  pip install -r $DEST/deep_research_bench/requirements.txt   # google-genai for RACE/FACT"
echo "  pip install openai crawl4ai                                 # Gym judges (+ citation)"
echo "  pip install -e $DEST/gpt-researcher                          # gpt_researcher agent"
echo
echo "BrowseComp-Plus corpus retriever (optional, 2.1 GB index + pyserini/Java 21):"
echo "  pip install -e '.[bcp]' && python $ROOT/scripts/download_bcp_index.py"
echo
echo "Verify with: adr doctor"
