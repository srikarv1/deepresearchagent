"""Running the upstream judge scripts as subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TAIL = 2_000


def run_script(
    args: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Invoke ``python -u <args>`` and capture a summary of the outcome."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": None, "timeout": True, "ok": False}
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout_tail": completed.stdout[-TAIL:],
        "stderr_tail": completed.stderr[-TAIL:],
    }
