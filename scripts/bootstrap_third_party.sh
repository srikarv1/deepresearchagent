#!/usr/bin/env bash
# Make the official judge repos reachable under third_party/.
#
# Existing clones elsewhere on disk are symlinked rather than re-cloned, so a
# checkout you already have (including a fork) is used as-is. Override the
# lookup with ADR_DRB_DIR, or set DRB_REPO_URL to clone from a different remote.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
PARENT="$(dirname "$ROOT")"
mkdir -p "$DEST"

DRB_REPO_URL="${DRB_REPO_URL:-https://github.com/Ayanami0730/deep_research_bench.git}"
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


# Agent under test, not a judge. Marker is the fork's trajectory logger so a
# plain upstream checkout is not mistaken for the instrumented one.
link_or_clone "$DEST/gpt-researcher" "gpt_researcher/utils/trajectory_logger.py" "$GR_REPO_URL" \
  --branch "$GR_BRANCH" \
  "${ADR_GR_DIR:-}" "$PARENT/gpt-researcher" "$PARENT/gpt_researcher"

echo
echo "Judge dependencies (install into the same env that runs adr):"
echo "  pip install -r $DEST/deep_research_bench/requirements.txt   # google-genai for RACE/FACT"
echo "  pip install -e $DEST/gpt-researcher                          # gpt_researcher agent"
echo
echo "BrowseComp-Plus corpus retriever (optional, 2.1 GB index + pyserini/Java 21):"
echo "  pip install -e '.[bcp]' && python $ROOT/scripts/download_bcp_index.py"
echo
echo "Verify with: adr doctor"
