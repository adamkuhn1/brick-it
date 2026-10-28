import subprocess
import sys

import pytest

from server import memory

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     9239.
Pages active:                                 102548.
Pages inactive:                               100272.
Pages speculative:                              1475.
Pages throttled:                                   0.
Pages wired down:                             139555.
Pages purgeable:                                 133.
"""


def test_macos_counts_free_inactive_speculative_purgeable(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, VM_STAT, ""))
    assert memory._macos() == (9239 + 100272 + 1475 + 133) * 16384  # ~1.8 GB


def test_macos_unreadable_output_means_unknown(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "nope", ""))
    assert memory._macos() is None


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="no reader on this platform")
def test_reads_this_machine():
    free = memory.available_memory()
    assert free is not None and 0 < free < 1 << 50
