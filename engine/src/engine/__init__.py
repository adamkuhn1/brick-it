"""Brick engine: mesh -> LEGO build. Pure logic, no web or model imports."""

from __future__ import annotations

import hashlib

import numpy as np
import trimesh

from .colors import colorize
from .hollow import hollow
from .layout import layout
from .slopes import add_slopes, make_room
from .steps import plan_steps
from .structure import components, repair, weak_joints
from .types import Brick, Build, Step
from .voxelize import voxelize

MAX_BRICKS_WARNING = 2000


def build(
    mesh: trimesh.Trimesh,
    size_studs: int,
    up: str = "y",
    build_id: str | None = None,
    color_ref: np.ndarray | None = None,
) -> Build:
    """Run the full pipeline.

    up: the mesh's up axis ("y" for glTF/TripoSR, "z" for CAD).
    color_ref: optional (N, 3) uint8 foreground pixels of the source photo; mesh colours are
        matched to them (image-to-3D models return darker colours than the photo).
    """
    vox = voxelize(mesh, size_studs, up=up)
    grid = hollow(colorize(vox, color_ref), size_studs)
    bricks, warnings = repair(layout(grid), grid)
    if vox.tilt is not None and len(components(bricks, grid.shape)) == 1:
        # Make room for slopes on sloped edges, then repair again: the re-tiling can leave a
        # single-stud joint nearby. Supports were all added by the first pass, and the
        # second pass skips weak joints the first one already failed to fix.
        supports = [w for w in warnings if w.startswith("Added")]
        known = weak_joints(bricks, grid.shape)
        bricks, warnings = repair(make_room(bricks, grid, vox.tilt, vox.tilt_down), grid, known)
        warnings = supports + warnings
        bricks = add_slopes(bricks, vox.tilt, grid.shape, vox.tilt_down)
    steps = plan_steps(bricks)
    if len(bricks) > MAX_BRICKS_WARNING:
        warnings.append(f"{len(bricks)} bricks — that's a big build. Try a smaller size.")
    if build_id is None:
        h = hashlib.sha256(mesh.vertices.tobytes())
        h.update(f"{size_studs}:{up}".encode())
        build_id = h.hexdigest()[:16]
    X, Y, Z = grid.shape
    return Build(id=build_id, size_studs=size_studs, dims=(X, Y, Z), bricks=bricks, steps=steps,
                 warnings=warnings)


__all__ = ["build", "Build", "Brick", "Step"]
