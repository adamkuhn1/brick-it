"""Time each brick-engine stage at several sizes and print a Markdown table.

    uv run python scripts/bench.py                    # 8, 16, 32 studs, median of 3 runs
    uv run python scripts/bench.py --sizes 8 16 32 48 --repeats 5

`timed_build` runs the same stages in the same order as `engine.build`, with a timer around
each one. engine/tests/test_bench.py checks the two stay in step.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import trimesh

from engine import MAX_BRICKS_WARNING
from engine.colors import colorize
from engine.export import to_bricklink_xml, to_csv, to_ldraw
from engine.hollow import hollow
from engine.layout import layout
from engine.slopes import add_slopes, make_room
from engine.steps import plan_steps
from engine.structure import components, repair
from engine.types import Build
from engine.voxelize import voxelize

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "engine" / "tests"
STAGES = ["voxelize", "colorize", "hollow", "layout", "repair", "slopes", "steps", "export"]


def timed_build(mesh: trimesh.Trimesh, size_studs: int, up: str = "y",
                color_ref=None) -> tuple[Build, dict[str, float]]:
    """engine.build with a timer per stage. Returns the build and seconds per stage."""
    t: dict[str, float] = dict.fromkeys(STAGES, 0.0)
    clock = time.perf_counter

    s = clock(); vox = voxelize(mesh, size_studs, up=up); t["voxelize"] = clock() - s
    s = clock(); colored = colorize(vox, color_ref); t["colorize"] = clock() - s
    s = clock(); grid = hollow(colored, size_studs); t["hollow"] = clock() - s
    s = clock(); bricks = layout(grid); t["layout"] = clock() - s
    s = clock(); bricks, warnings = repair(bricks, grid); t["repair"] = clock() - s
    s = clock()
    if vox.tilt is not None and len(components(bricks, grid.shape)) == 1:
        supports = [w for w in warnings if w.startswith("Added")]
        bricks, warnings = repair(make_room(bricks, grid, vox.tilt, vox.tilt_down), grid)
        warnings = supports + warnings
        bricks = add_slopes(bricks, vox.tilt, grid.shape, vox.tilt_down)
    t["slopes"] = clock() - s
    s = clock(); steps = plan_steps(bricks); t["steps"] = clock() - s
    if len(bricks) > MAX_BRICKS_WARNING:
        warnings.append(f"{len(bricks)} bricks — that's a big build. Try a smaller size.")

    X, Y, Z = grid.shape
    b = Build(id="bench", size_studs=size_studs, dims=(X, Y, Z), bricks=bricks, steps=steps,
              warnings=warnings)
    s = clock()
    parts = b.parts()
    to_ldraw(b.bricks, b.steps)
    to_bricklink_xml(parts)
    to_csv(parts)
    t["export"] = clock() - s
    return b, t


def load_meshes() -> dict[str, tuple[trimesh.Trimesh, str]]:
    sys.path.insert(0, str(TESTS))
    from meshes import car  # synthetic test mesh, Z-up

    chair = trimesh.load(TESTS / "fixtures" / "chair.glb", force="mesh")
    return {"chair.glb": (chair, "y"), "car (synthetic)": (car(), "z")}


def machine() -> str:
    """CPU name, e.g. "Apple M2" (falls back to platform.processor())."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown CPU"


def fmt(seconds: float) -> str:
    ms = seconds * 1000
    return f"{ms:.0f}" if ms >= 10 else f"{ms:.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--repeats", type=int, default=3, help="runs per size; the median is shown")
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

    print(f"{machine()} · {platform.system()} {platform.machine()} · "
          f"Python {platform.python_version()}")
    print(f"Median of {args.repeats} run(s). Times in ms.\n")
    print("| Mesh | Studs | " + " | ".join(STAGES) + " | **Total** | Bricks |")
    print("|---|--:|" + "--:|" * len(STAGES) + "--:|--:|")
    for name, (mesh, up) in load_meshes().items():
        for size in args.sizes:
            runs = []
            for _ in range(args.repeats):
                b, t = timed_build(mesh, size, up=up)
                runs.append(t)
            med = {k: statistics.median(r[k] for r in runs) for k in STAGES}
            total = statistics.median(sum(r.values()) for r in runs)
            cells = " | ".join(fmt(med[k]) for k in STAGES)
            print(f"| {name} | {size} | {cells} | **{fmt(total)}** | {len(b.bricks)} |",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
