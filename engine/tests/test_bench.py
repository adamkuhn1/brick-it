"""scripts/bench.py copies engine.build's stage order to time it. Keep the two in step."""

import importlib.util
from pathlib import Path

import pytest
import trimesh

from engine import build
from meshes import car

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts" / "bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


@pytest.mark.parametrize("name", ["car", "chair"])
def test_timed_build_matches_build(name):
    if name == "car":
        mesh, up = car(), "z"
    else:
        mesh, up = trimesh.load(ROOT / "engine/tests/fixtures/chair.glb", force="mesh"), "y"
    expected = build(mesh, 8, up=up)
    got, timings = bench.timed_build(mesh, 8, up=up)
    assert got.bricks == expected.bricks
    assert got.steps == expected.steps
    assert got.warnings == expected.warnings
    assert set(timings) == set(bench.STAGES)
    assert all(v >= 0 for v in timings.values())


def test_main_prints_table(capsys):
    assert bench.main(["--sizes", "8", "--repeats", "1"]) == 0
    out = capsys.readouterr().out
    assert "| chair.glb | 8 |" in out
    assert "| car (synthetic) | 8 |" in out
