"""Run a heavy function in a child process that can be stopped.

A Python thread can't be killed, so a TripoSR run stuck in swap could only end by killing
the whole server. In a child process it can be stopped on a timeout, on a cancel request, or
when system memory runs low, and killing the child hands all of its memory straight back to
the OS. The child also exits after every run, so the model doesn't stay resident between photos.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from typing import Callable

from .memory import available_memory


class Stopped(RuntimeError):
    """The run was stopped on purpose. `reason` is "cancelled", "timeout" or "memory"."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ChildFailed(RuntimeError):
    """The function raised, or the child died (e.g. killed by the OS)."""


def _entry(conn, fn, args) -> None:
    try:
        result = (True, fn(*args))
    except BaseException as e:  # noqa: BLE001 - report anything back to the parent
        result = (False, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")
    try:
        conn.send(result)
    finally:
        conn.close()


def run_isolated(fn: Callable, args: tuple = (), *, timeout: float,
                 cancel: threading.Event | None = None, min_free: int | None = None,
                 free_memory: Callable[[], int | None] = available_memory,
                 poll: float = 1.0):
    """fn(*args) in a fresh 'spawn' child; returns its (picklable) result.

    Raises Stopped if `cancel` is set, `timeout` seconds pass, or free memory drops under
    `min_free` bytes; ChildFailed if fn raises or the child dies. The child is always gone
    when this returns. `fn` must be importable at module level (spawn pickles it by name).
    """
    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_entry, args=(send, fn, args), daemon=True)
    proc.start()
    send.close()
    deadline = time.monotonic() + timeout
    try:
        while True:
            if recv.poll(poll):
                try:
                    ok, value = recv.recv()
                except EOFError:
                    raise ChildFailed(f"the worker process exited (code {proc.exitcode})") from None
                if ok:
                    return value
                raise ChildFailed(value.splitlines()[0])
            if not proc.is_alive():
                raise ChildFailed(f"the worker process exited (code {proc.exitcode})")
            if cancel is not None and cancel.is_set():
                raise Stopped("cancelled")
            if time.monotonic() > deadline:
                raise Stopped("timeout")
            if min_free is not None:
                free = free_memory()
                if free is not None and free < min_free:
                    raise Stopped("memory")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(10)
        recv.close()


__all__ = ["run_isolated", "Stopped", "ChildFailed"]
