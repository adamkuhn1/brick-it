"""Stage 5b: smooth staircases with 45-degree slope bricks where the object leans.

A sloped surface (a roof, a car bonnet, the shoulder of a sphere) comes out of layout as a
staircase. Real models smooth it with slope bricks. This stage uses two, with the same
footprints as the 1x2 and 2x2 bricks: Slope 45 2x1 (3040) and Slope 45 2x2 (3039).

A brick becomes a slope facing direction d when:
  - it is a visible 1x2 or 2x2, 2 studs deep along d,
  - the mesh surface in its front row (the row on the d side) leans toward d,
  - its front row is exposed: nothing beside it on the d side and nothing on top of it.
A slope has studs only on its back row. The front row had nothing on top, so the swap
changes neither connectivity nor step planning.

Layout prefers long bricks, so most sloped edges start out under a 2x8 or 1x6. `make_room`
first re-tiles each sloped edge cell so a 1x2 or 2x2 lands there (displaced bricks are
re-laid by the normal layout). That can leave a single-stud joint somewhere nearby, so the
caller runs the repair stage again before `add_slopes`.
"""

from __future__ import annotations

import numpy as np

from .structure import _State, id_map
from .types import Brick

SLOPE_PARTS = {"3004": "3040", "3003": "3039"}  # brick -> Slope 45 with the same footprint
DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
TILT_MIN = 0.4  # how strongly the front row must lean toward d (see Voxels.tilt)


def front_row(b: Brick, d: str) -> list[tuple[int, int]]:
    """Cells on the low (d) side of a 2-deep footprint."""
    dx, dy = DIRECTIONS[d]
    if dx:
        x = b.x + b.size_x - 1 if dx > 0 else b.x
        return [(x, b.y + j) for j in range(b.size_y)]
    y = b.y + b.size_y - 1 if dy > 0 else b.y
    return [(b.x + i, y) for i in range(b.size_x)]


def _depth(b: Brick, d: str) -> int:
    return b.size_x if DIRECTIONS[d][0] else b.size_y


def _leaning_cells(ids: np.ndarray, tilt: np.ndarray):
    """(x, y, z, d) for exposed edge cells whose surface leans toward d."""
    X, Y, _ = ids.shape
    occ = ids >= 0
    covered = np.zeros_like(occ)
    covered[:, :, :-1] = occ[:, :, 1:]
    for d, (dx, dy) in DIRECTIONS.items():
        beside = np.zeros_like(occ)  # beside[x, y] = occ[x + dx, y + dy]
        beside[max(-dx, 0):X + min(-dx, 0), max(-dy, 0):Y + min(-dy, 0)] = \
            occ[max(dx, 0):X + min(dx, 0), max(dy, 0):Y + min(dy, 0)]
        lean = tilt[..., 0] * dx + tilt[..., 1] * dy
        for x, y, z in np.argwhere(occ & ~covered & ~beside & (lean >= TILT_MIN)):
            yield int(x), int(y), int(z), d


def _candidates(x: int, y: int, z: int, d: str):
    """2x2 then 1x2 footprints with (x, y) in the front row; the back row is one step behind."""
    dx, dy = DIRECTIONS[d]
    x0, y0 = min(x, x - dx), min(y, y - dy)
    for across in (2, 1):
        for off in range(across):
            yield (z, x0, y - off, 2, across) if dx else (z, x - off, y0, across, 2)


def _studs_held(state: _State, z: int, x: int, y: int, sx: int, sy: int) -> int:
    """Studs a brick with this footprint shares with the courses above and below."""
    if x < 0 or y < 0:
        return 0
    n = 0
    for zz in (z - 1, z + 1):
        if 0 <= zz < state.ids.shape[2]:
            n += int((state.ids[x:x + sx, y:y + sy, zz] >= 0).sum())
    return n


def make_room(bricks: list[Brick], grid: np.ndarray, tilt: np.ndarray) -> list[Brick]:
    """Re-tile sloped edges so a 1x2 or 2x2 sits on each. Run repair afterwards."""
    state = _State(bricks, grid)
    for x, y, z, d in list(_leaning_cells(state.ids, tilt)):
        here = state.bricks[state.ids[x, y, z]]
        if here.hidden or (here.part in SLOPE_PARTS and _depth(here, d) == 2
                           and (x, y) in front_row(here, d)):
            continue
        for rect in _candidates(x, y, z, d):
            if _studs_held(state, *rect) < 2:
                continue
            new = state.force(*rect)
            if new is None:
                continue
            # Keep it only if every brick it produced clips on by 2+ studs.
            if all(_studs_held(state, b.layer, b.x, b.y, b.size_x, b.size_y) >= 2
                   for b in (state.bricks[i] for i in new)):
                break
            state.undo()
    out = sorted(state.list(), key=lambda b: (b.layer, b.x, b.y))
    for i, b in enumerate(out):
        b.id = i
    return out


def add_slopes(bricks: list[Brick], tilt: np.ndarray, shape) -> list[Brick]:
    """Swap qualifying 1x2 / 2x2 bricks for slopes, in place."""
    ids = id_map(bricks, shape)
    X, Y, Z = shape

    def occupied(x: int, y: int, z: int) -> bool:
        return 0 <= x < X and 0 <= y < Y and 0 <= z < Z and ids[x, y, z] >= 0

    for b in bricks:
        if b.hidden or b.part not in SLOPE_PARTS:
            continue
        z = b.layer
        best = None
        for d, (dx, dy) in DIRECTIONS.items():
            if _depth(b, d) != 2:
                continue
            front = front_row(b, d)
            if any(occupied(x, y, z + 1) or occupied(x + dx, y + dy, z) for x, y in front):
                continue
            score = float(np.mean([tilt[x, y, z] @ (dx, dy) for x, y in front]))
            if score >= TILT_MIN and (best is None or score > best[0]):
                best = (score, d)
        if best:
            b.part = SLOPE_PARTS[b.part]
            b.slope = best[1]
    return bricks


__all__ = ["make_room", "add_slopes", "front_row", "SLOPE_PARTS", "DIRECTIONS"]
