"""run_isolated: results come back, and every way of stopping really ends the child."""

import os
import threading
import time

import pytest

from server.isolated import ChildFailed, Stopped, run_isolated


# Child entry points must be module level: 'spawn' imports them by name.
def double(x):
    return x * 2


def boom():
    raise ValueError("bad photo")


def die():
    os._exit(3)


def write_pid_then_sleep(path):
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    time.sleep(60)


def _gone(pid_file) -> bool:
    pid = int(open(pid_file).read())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_returns_result():
    assert run_isolated(double, (21,), timeout=30) == 42


def test_exception_in_child_is_reported():
    with pytest.raises(ChildFailed, match="ValueError: bad photo"):
        run_isolated(boom, timeout=30)


def test_child_that_dies_is_reported():
    with pytest.raises(ChildFailed, match="code 3"):
        run_isolated(die, timeout=30)


def _wait_for(path, timeout=30.0):
    end = time.monotonic() + timeout
    while not (os.path.exists(path) and open(path).read()):
        assert time.monotonic() < end, "child never started"
        time.sleep(0.05)


@pytest.mark.parametrize("reason", ["timeout", "cancelled", "memory"])
def test_stopping_kills_the_child(tmp_path, reason):
    pid_file = str(tmp_path / "pid")
    cancel = threading.Event()
    started = threading.Event()
    free = {"bytes": 8 << 30}

    def trigger():  # wait for the child to be up, then pull the chosen lever
        _wait_for(pid_file)
        started.set()
        if reason == "cancelled":
            cancel.set()
        elif reason == "memory":
            free["bytes"] = 100 << 20

    threading.Thread(target=trigger, daemon=True).start()
    t = time.monotonic()
    with pytest.raises(Stopped) as e:
        run_isolated(write_pid_then_sleep, (pid_file,), poll=0.05, cancel=cancel,
                     timeout=3 if reason == "timeout" else 30,
                     min_free=1 << 30, free_memory=lambda: free["bytes"])
    assert e.value.reason == reason
    assert started.is_set() and time.monotonic() - t < 20
    assert _gone(pid_file)
