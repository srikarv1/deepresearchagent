"""Load the fork's orchestration modules without importing ``gpt_researcher``.

``gpt_researcher/orchestration/serialize.py`` and ``features.py`` are
self-contained (stdlib + numpy), but importing them as package members pulls
in the whole ``gpt_researcher`` package and its LLM/scraper dependencies. The
BC builder only needs the two files, so they are loaded straight from disk.
Keeping the *definitions* in the fork guarantees the offline prompt is the one
the online policy (and later the learned policy) uses.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from adr.eval.repos import find_gpt_researcher

_MODULES = ("serialize", "features")


class ForkModuleMissing(RuntimeError):
    pass


@lru_cache(maxsize=None)
def load_fork_module(name: str, repo_path: str | None = None) -> ModuleType:
    if name not in _MODULES:
        raise ValueError(f"unknown fork module {name!r}; known: {_MODULES}")
    loc = find_gpt_researcher(repo_path)
    if not loc.ok:
        raise ForkModuleMissing(loc.reason)
    path = Path(loc.path) / "gpt_researcher" / "orchestration" / f"{name}.py"
    if not path.exists():
        raise ForkModuleMissing(
            f"{path} not found; the gpt-researcher checkout predates the RandomizedPolicy "
            "branch (needs gpt_researcher/orchestration/serialize.py)"
        )
    mod_name = f"_adr_fork_orchestration_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ForkModuleMissing(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def fork_available(repo_path: str | None = None) -> bool:
    try:
        load_fork_module("serialize", repo_path)
        return True
    except (ForkModuleMissing, OSError):
        return False
