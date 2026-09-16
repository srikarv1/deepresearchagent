"""Search backend dispatch and import-order contract."""

from __future__ import annotations

import subprocess
import sys

import pytest

from adr.tools.search import MockSearch, build_search


def test_import_order_does_not_break():
    """tools.search imports core.state, so core must not import tools at runtime."""
    for first in ("adr.tools.search", "adr.core"):
        subprocess.run(
            [sys.executable, "-c", f"import {first}; import adr.core, adr.tools.search"],
            check=True,
            capture_output=True,
        )


def test_build_search_dispatch():
    assert isinstance(build_search({"backend": "mock"}), MockSearch)
    with pytest.raises(ValueError, match="Unknown search backend"):
        build_search({"backend": "nope"})
