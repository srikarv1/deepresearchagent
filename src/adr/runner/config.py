from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

EXTENDS_KEY = "extends"
_MAX_EXTENDS_DEPTH = 8


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, resolving ``extends: <base.yaml>`` first.

    ``extends`` is relative to the file that names it. The file's own keys
    override the base: nested mappings merge key by key, anything else
    (including an explicit ``null``) replaces the base value. Run and agent
    configs both load through here, so a variant such as a different backbone
    only states what changes.
    """
    return _load_yaml(Path(path), ())


def _load_yaml(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in (*chain, resolved))
        raise ValueError(f"extends cycle: {cycle}")
    if len(chain) >= _MAX_EXTENDS_DEPTH:
        raise ValueError(f"extends chain deeper than {_MAX_EXTENDS_DEPTH} files at {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    base_ref = data.pop(EXTENDS_KEY, None)
    if base_ref is None:
        return data
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f"{resolved}: {EXTENDS_KEY} must be a file path, got {base_ref!r}")
    base = _load_yaml(resolved.parent / base_ref.strip(), (*chain, resolved))
    return _merge(base, data)


def _merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    merged = load_yaml(cfg_path)
    if overrides:
        _deep_update(merged, overrides)
    merged["_config_path"] = str(cfg_path)
    return merged


def _deep_update(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
