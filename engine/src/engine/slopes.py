"""Stage 5b: smooth staircases with 45-degree slope bricks where the object leans.

A sloped surface (a roof, a car bonnet, the shoulder of a sphere) comes out of layout as a
staircase. Real models smooth it with slope bricks. This stage uses three, all with the
footprint of a 1x2 or 2x2 brick: Slope 45 2x1 (3040), Slope 45 2x2 (3039), and, where the
surface leans two ways at once (a pyramid's corner, the shoulder of a sphere seen at 45
degrees), Slope 45 2x2 Double Convex (3045).

A brick becomes a slope facing direction d when:
  - it is a visible 1x2 or 2x2, 2 studs deep along d,
  - the mesh surface in its front row (the row on the d side) leans toward d,
  - its front row is exposed: nothing beside it on the d side and nothing on top of it.
A visible 2x2 becomes a corner slope facing two perpendicular directions (e.g. "+x+y") when
both of its front rows lean out, nothing sits on its three front cells, and its outer corner
cell is open on both sides. It keeps one stud, on the back corner. A corner slope wins over a
straight one: where both fit, the surface is turning a corner.
A slope has studs only on its back cells. The stud-less cells had nothing on top, so the swap
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
CORNER_PART = "3045"  # Slope 45 2x2 Double Convex, replaces a 2x2 brick
DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
# Corner facings: the two axis directions a double-convex slope leans toward.
CORNERS = {f"{a}{b}": (a, b) for a in ("+x", "-x") for b in ("+y", "-y")}
TILT_MIN = 0.4  # how strongly the front row must lean toward d (see Voxels.tilt)


def axes(slope: str) -> tuple[str, ...]:
    """The axis directions a slope faces: one for a straight slope, two for a corner."""
    return CORNERS.get(slope, (slope,))


def facing(slope: str) -> tuple[int, int]:
    """Horizontal vector (dx, dy) a slope's low side points along (diagonal for corners)."""
    return tuple(int(sum(DIRECTIONS[a][i] for a in axes(slope))) for i in (0, 1))


def front_row(b: Brick, d: str) -> list[tuple[int, int]]:
    """Cells on the low (d) side of a 2-deep footprint."""
    dx, dy = DIRECTIONS[d]
    if dx:
        x = b.x + b.size_x - 1 if dx > 0 else b.x
        return [(x, b.y + j) for j in range(b.size_y)]
    y = b.y + b.size_y - 1 if dy > 0 else b.y
    return [(b.x + i, y) for i in range(b.size_x)]


def studless(b: Brick, slope: str) -> list[tuple[int, int]]:
    """Cells of a slope brick without a stud: its front row(s)."""
    return sorted({c for a in axes(slope) for c in front_row(b, a)})


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


def _corner_rect(x: int, y: int, z: int, corner: str):
    """The 2x2 footprint with (x, y) on its outer corner, the one both slopes lean toward."""
    (dx, _), (_, dy) = (DIRECTIONS[a] for a in CORNERS[corner])
    return z, min(x, x - dx), min(y, y - dy), 2, 2


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


def _rect(b: Brick) -> tuple[int, int, int, int, int]:
    return b.layer, b.x, b.y, b.size_x, b.size_y


def _try_place(state: _State, rects, keep=frozenset()) -> bool:
    """Force the first footprint whose bricks all clip on by 2+ studs, without displacing
    any brick whose footprint is in `keep`."""
    for rect in rects:
        z, x, y, sx, sy = rect
        if _studs_held(state, *rect) < 2 or any(
                _rect(state.bricks[i]) in keep
                for i in np.unique(state.ids[max(x, 0):x + sx, max(y, 0):y + sy, z]) if i >= 0):
            continue
        new = state.force(*rect)
        if new is None:
            continue
        if all(_studs_held(state, b.layer, b.x, b.y, b.size_x, b.size_y) >= 2
               for b in (state.bricks[i] for i in new)):
            return True
        state.undo()
    return False


def make_room(bricks: list[Brick], grid: np.ndarray, tilt: np.ndarray) -> list[Brick]:
    """Re-tile sloped edges so a 1x2 or 2x2 sits on each. Run repair afterwards."""
    state = _State(bricks, grid)
    leaning: dict[tuple[int, int, int], set[str]] = {}
    for x, y, z, d in _leaning_cells(state.ids, tilt):
        leaning.setdefault((x, y, z), set()).add(d)

    # Cells leaning two perpendicular ways first: give each a 2x2 with the cell on its outer
    # corner, so add_slopes can turn it into a corner slope. Corner 2x2s don't displace each
    # other; the straight pass below may still re-tile one where a 1x2 or 2x2 fits better.
    corners: set[tuple[int, int, int, int, int]] = set()
    for (x, y, z), ds in leaning.items():
        for corner, pair in CORNERS.items():
            if not set(pair) <= ds:
                continue
            rect = _corner_rect(x, y, z, corner)
            here = state.bricks[state.ids[x, y, z]]
            if here.hidden:
                break
            if _rect(here) == rect or _try_place(state, [rect], corners):
                corners.add(rect)
            break

    for (x, y, z), ds in leaning.items():
        for d in sorted(ds):
            here = state.bricks[state.ids[x, y, z]]
            if here.hidden or (here.part in SLOPE_PARTS and _depth(here, d) == 2
                               and (x, y) in front_row(here, d)):
                continue
            _try_place(state, _candidates(x, y, z, d))
    out = sorted(state.list(), key=lambda b: (b.layer, b.x, b.y))
    for i, b in enumerate(out):
        b.id = i
    return out


def add_slopes(bricks: list[Brick], tilt: np.ndarray, shape) -> list[Brick]:
    """Swap qualifying 1x2 / 2x2 bricks for straight or corner slopes, in place."""
    ids = id_map(bricks, shape)
    X, Y, Z = shape

    def occupied(x: int, y: int, z: int) -> bool:
        return 0 <= x < X and 0 <= y < Y and 0 <= z < Z and ids[x, y, z] >= 0

    for b in bricks:
        if b.hidden or b.part not in SLOPE_PARTS:
            continue
        z = b.layer
        # Lean score for each direction whose front row is exposed and 2 deep behind it.
        scores: dict[str, float] = {}
        for d, (dx, dy) in DIRECTIONS.items():
            if _depth(b, d) != 2:
                continue
            front = front_row(b, d)
            if any(occupied(x, y, z + 1) or occupied(x + dx, y + dy, z) for x, y in front):
                continue
            score = float(np.mean([tilt[x, y, z] @ (dx, dy) for x, y in front]))
            if score >= TILT_MIN:
                scores[d] = score
        # A 2x2 corner slope: nothing on its three stud-less cells, the outer corner cell open
        # on both sides, and both front rows leaning out. The other front cells may have a
        # neighbour beside them: on a diagonal staircase the next step sits there.
        corners = []
        for k, pair in CORNERS.items() if b.part == "3003" else ():
            (cx, cy), = set(front_row(b, pair[0])) & set(front_row(b, pair[1]))
            if any(occupied(x, y, z + 1) for x, y in studless(b, k)) or any(
                    occupied(cx + DIRECTIONS[a][0], cy + DIRECTIONS[a][1], z) for a in pair):
                continue
            lean = min(float(np.mean([tilt[x, y, z] @ DIRECTIONS[a] for x, y in front_row(b, a)]))
                       for a in pair)
            if lean >= TILT_MIN:
                corners.append((lean, k))
        if corners:
            b.part, b.slope = CORNER_PART, max(corners)[1]
        elif scores:
            b.part, b.slope = SLOPE_PARTS[b.part], max(scores, key=scores.get)
    return bricks


__all__ = ["make_room", "add_slopes", "front_row", "studless", "axes", "facing",
           "SLOPE_PARTS", "CORNER_PART", "DIRECTIONS", "CORNERS"]
