"""Locating the official benchmark repositories on disk.

The judges live in upstream repos that we do not vendor. A checkout can sit in
``third_party/`` (via ``scripts/bootstrap_third_party.sh``), somewhere else
entirely, or under a different directory name depending on which fork was
cloned. Resolution order per repo:

1. explicit ``third_party_dir`` from config
2. the ``ADR_DRB_DIR`` / ``ADR_GYM_DIR`` environment variable
3. ``third_party/<known name>`` inside this repo
4. ``<parent of this repo>/<known name>`` -- the common sibling-clone layout

A candidate only counts if it contains the marker files we actually invoke, so
a wrong-but-existing directory is reported as missing rather than failing later
inside a subprocess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DRB_MARKERS = ("deepresearch_bench_race.py",)
GYM_MARKERS = ("eval_quality_async.py", "eval_kpr_async.py")

DRB_NAMES = ("deep_research_bench",)
GYM_NAMES = ("deepresearchgym", "deepresearch_benchmarking")

# The agent under test, not a judge. The marker is the instrumented fork's
# trajectory logger; a plain upstream clone is reported as missing.
GR_MARKERS = ("gpt_researcher/utils/trajectory_logger.py",)
GR_NAMES = ("gpt-researcher", "gpt_researcher")


@dataclass(frozen=True)
class RepoLocation:
    path: Path | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None


def _has_markers(path: Path, markers: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / marker).exists() for marker in markers)


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def _resolve(
    explicit: str | Path | None,
    env_var: str,
    names: tuple[str, ...],
    markers: tuple[str, ...],
    label: str,
) -> RepoLocation:
    needed = "/".join(markers)

    # A path someone set on purpose is never silently replaced by a different
    # checkout -- that would score a run against a repo they did not choose.
    for source, value in (("config third_party_dir", explicit), (env_var, os.environ.get(env_var))):
        if not value:
            continue
        resolved = _absolute(value)
        if _has_markers(resolved, markers):
            return RepoLocation(resolved.resolve())
        return RepoLocation(
            None,
            f"{label} checkout not found at {resolved} (from {source}); "
            f"expected {needed} inside it.",
        )

    searched: list[Path] = []
    for parent in (ROOT / "third_party", ROOT.parent):
        for name in names:
            candidate = parent / name
            searched.append(candidate)
            if _has_markers(candidate, markers):
                return RepoLocation(candidate.resolve())

    tried = ", ".join(str(c) for c in searched)
    return RepoLocation(
        None,
        f"{label} checkout not found (need {needed}). "
        f"Set {env_var} or run scripts/bootstrap_third_party.sh. Tried: {tried}",
    )


def find_deep_research_bench(explicit: str | Path | None = None) -> RepoLocation:
    return _resolve(explicit, "ADR_DRB_DIR", DRB_NAMES, DRB_MARKERS, "DeepResearch Bench")


def find_deep_research_gym(explicit: str | Path | None = None) -> RepoLocation:
    return _resolve(explicit, "ADR_GYM_DIR", GYM_NAMES, GYM_MARKERS, "DeepResearchGym")


def find_gpt_researcher(explicit: str | Path | None = None) -> RepoLocation:
    """Locate the instrumented gpt-researcher fork (WilliamOdinson/gpt-researcher)."""
    return _resolve(explicit, "ADR_GR_DIR", GR_NAMES, GR_MARKERS, "gpt-researcher fork")


def find_key_points(gym_root: Path, explicit: str | Path | None = None) -> Path | None:
    """Locate the aggregated key-point directory used for key-point recall."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(gym_root / "key_point")
    candidates.append(gym_root / "deepresearch_benchmarking" / "key_point")
    for candidate in candidates:
        resolved = _absolute(candidate)
        if resolved.is_dir() and any(resolved.glob("*_aggregated.json")):
            return resolved.resolve()
    return None
