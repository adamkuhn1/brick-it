"""Stage 4: tile each layer with real bricks.

Greedy per layer: walk cells in scan order; at the first uncovered cell, place the best
brick anchored there. Candidates are scored by area (bigger first), how many bricks below
they bridge (connectivity), seams lining up with the layer below (penalized), and the
layer's preferred orientation (alternates per layer so walls interlock).
"""

from __future__ import annotations

import numpy as np

from .colors import DEFAULT_COLOR
from .types import DONT_CARE, EMPTY, PLATES_PER_BRICK, Brick

# (short side, long side) -> LDraw/BrickLink part number
PARTS: dict[tuple[int, int], str] = {
    (1, 1): "3005", (1, 2): "3004", (1, 3): "3622", (1, 4): "3010", (1, 6): "3009",
    (1, 8): "3008", (2, 2): "3003", (2, 3): "3002", (2, 4): "3001", (2, 6): "2456",
    (2, 8): "3007",
}
SIZES = sorted(PARTS, key=lambda s: (-s[0] * s[1], -s[1]))


def _candidates(prefer_x: bool):
    """(size_x, size_y, w, l, rot) in a stable order."""
    out = []
    for w, l in SIZES:
        along_x = (l, w, w, l, 90)
        along_y = (w, l, w, l, 0)
        if w == l:
            out.append(along_y)
        else:
            out.extend([along_x, along_y] if prefer_x else [along_y, along_x])
    return out


def _aligned_sides(below: np.ndarray, x0: int, y0: int, sx: int, sy: int) -> int:
    """Number of the brick's 4 edges that coincide with a seam in the layer below."""
    X, Y = below.shape
    count = 0
    # Left / right edges (constant x)
    for xi, xo in ((x0, x0 - 1), (x0 + sx - 1, x0 + sx)):
        if 0 <= xo < X:
            a, b = below[xi, y0:y0 + sy], below[xo, y0:y0 + sy]
            if np.all((a >= 0) & (b >= 0) & (a != b)):
                count += 1
    for yi, yo in ((y0, y0 - 1), (y0 + sy - 1, y0 + sy)):
        if 0 <= yo < Y:
            a, b = below[x0:x0 + sx, yi], below[x0:x0 + sx, yo]
            if np.all((a >= 0) & (b >= 0) & (a != b)):
                count += 1
    return count


def layout_layer(
    layer: np.ndarray, below_ids: np.ndarray | None, z: int, start_id: int
) -> tuple[list[Brick], np.ndarray]:
    """Tile one layer. Returns bricks and an id map (-1 where empty)."""
    X, Y = layer.shape
    ids = np.full((X, Y), -1, dtype=np.int32)
    occupied = layer != EMPTY
    prefer_x = z % 2 == 0
    cands = _candidates(prefer_x)
    bricks: list[Brick] = []

    order = [(x, y) for x in range(X) for y in range(Y)] if prefer_x else [
        (x, y) for y in range(Y) for x in range(X)
    ]
    for x, y in order:
        if not occupied[x, y] or ids[x, y] >= 0:
            continue
        best = None
        for sx, sy, w, l, rot in cands:
            if x + sx > X or y + sy > Y:
                continue
            region = layer[x:x + sx, y:y + sy]
            if np.any(region == EMPTY) or np.any(ids[x:x + sx, y:y + sy] >= 0):
                continue
            colors = np.unique(region[region >= 0])
            if len(colors) > 1:
                continue
            score = sx * sy * 10.0
            if below_ids is not None:
                under = below_ids[x:x + sx, y:y + sy]
                score += 4 * len(np.unique(under[under >= 0]))
                score -= 3 * _aligned_sides(below_ids, x, y, sx, sy)
            if (sx >= sy) == prefer_x:
                score += 1
            if best is None or score > best[0]:
                color = int(colors[0]) if len(colors) else DEFAULT_COLOR
                best = (score, sx, sy, w, l, rot, color, len(colors) == 0)
        assert best is not None  # 1x1 always fits an occupied, uncovered cell
        _, sx, sy, w, l, rot, color, hidden = best
        b = Brick(
            id=start_id + len(bricks), part=PARTS[(w, l)], w=w, l=l, x=x, y=y,
            z=z * PLATES_PER_BRICK, rot=rot, color=color, hidden=hidden,
        )
        ids[x:x + sx, y:y + sy] = b.id
        bricks.append(b)
    return bricks, ids


def layout(grid: np.ndarray) -> list[Brick]:
    bricks: list[Brick] = []
    below = None
    for z in range(grid.shape[2]):
        layer_bricks, ids = layout_layer(grid[:, :, z], below, z, len(bricks))
        bricks.extend(layer_bricks)
        below = ids
    return bricks


def id_maps(bricks: list[Brick], shape: tuple[int, int, int]) -> np.ndarray:
    """[X, Y, Z] brick id per cell, -1 where empty."""
    m = np.full(shape, -1, dtype=np.int32)
    for b in bricks:
        m[b.x:b.x + b.size_x, b.y:b.y + b.size_y, b.layer] = b.id
    return m


__all__ = ["layout", "layout_layer", "id_maps", "PARTS", "DONT_CARE"]
