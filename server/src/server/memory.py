"""How much memory the system can hand out right now, so TripoSR never starts (or keeps
running) when it would push the machine into swap.

On Apple Silicon the GPU shares system RAM, so TripoSR's ~2 GB on MPS comes out of the same
pool as every other app. Starting it on a full machine makes macOS swap until everything,
the browser included, freezes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

GB = 1024 ** 3


def _macos() -> int | None:
    """free + inactive + speculative + purgeable pages: what macOS can give out without
    compressing or swapping anything (roughly Activity Monitor's 'available')."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = re.search(r"page size of (\d+) bytes", out)
    if not page:
        return None
    pages = 0
    for key in ("free", "inactive", "speculative", "purgeable"):
        m = re.search(rf"^Pages {key}:\s+(\d+)\.", out, re.M)
        if not m:
            return None
        pages += int(m.group(1))
    return pages * int(page.group(1))


def _linux() -> int | None:
    try:
        m = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
    except OSError:
        return None
    return int(m.group(1)) * 1024 if m else None


def available_memory() -> int | None:
    """Bytes available without swapping, or None if this platform can't tell us (then the
    checks that use it are skipped rather than blocking)."""
    if sys.platform == "darwin":
        return _macos()
    if sys.platform.startswith("linux"):
        return _linux()
    return None


__all__ = ["available_memory", "GB"]
