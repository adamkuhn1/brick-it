"""Stage 5: make the build hold together.

Bricks connect when they sit in adjacent layers and share at least one stud. We fix:
  (a) disconnected components, by re-tiling a layer so one brick spans the joint between
      two components; if no joint exists, by adding a support column under the floating part;
  (b) single-stud bridges (cut edges whose removal splits off more than one brick), by
      re-tiling the joint with a larger brick that grips more studs.
Whatever cannot be fixed within the iteration cap becomes a warning. Center of mass outside
the base footprint is a warning only.
"""

from __future__ import annotations

import numpy as np

from .colors import DEFAULT_COLOR
from .layout import PARTS, layout_layer
from .types import DONT_CARE, EMPTY, PLATES_PER_BRICK, Brick

MAX_ITER = 300
MAX_DROP = 2


# ---------- graph ----------

def id_map(bricks: list[Brick], shape) -> np.ndarray:
    m = np.full(shape, -1, dtype=np.int32)
    for b in bricks:
        m[b.x:b.x + b.size_x, b.y:b.y + b.size_y, b.layer] = b.id
    return m


def adjacency(bricks: list[Brick], shape) -> dict[int, dict[int, int]]:
    """brick id -> {neighbour id: overlapping studs}"""
    ids = id_map(bricks, shape)
    adj: dict[int, dict[int, int]] = {b.id: {} for b in bricks}
    for z in range(shape[2] - 1):
        lo, hi = ids[:, :, z].ravel(), ids[:, :, z + 1].ravel()
        both = (lo >= 0) & (hi >= 0)
        if not both.any():
            continue
        pairs, counts = np.unique(np.stack([lo[both], hi[both]], 1), axis=0, return_counts=True)
        for (a, b), c in zip(pairs.tolist(), counts.tolist()):
            adj[a][b] = c
            adj[b][a] = c
    return adj


def components(bricks: list[Brick], shape=None, adj=None) -> list[set[int]]:
    if adj is None:
        adj = adjacency(bricks, shape or _shape(bricks))
    seen: set[int] = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        comp, stack = set(), [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.add(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def weak_bridges(adj: dict[int, dict[int, int]]) -> list[tuple[int, int]]:
    """Cut edges with 1-stud overlap whose removal splits off more than one brick."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    size: dict[int, int] = {}
    result = []
    counter = 0
    for root in adj:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        size[root] = 1
        comp_size = 1
        cuts = []  # (parent, child, subtree size of child)
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            node, parent, it = stack[-1]
            for nxt in it:
                if nxt == parent:
                    continue
                if nxt in index:
                    low[node] = min(low[node], index[nxt])
                else:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    size[nxt] = 1
                    comp_size += 1
                    stack.append((nxt, node, iter(adj[nxt])))
                    break
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[node])
                    size[p] += size[node]
                    if low[node] > index[p]:
                        cuts.append((p, node, size[node]))
        for a, b, s in cuts:
            if adj[a][b] == 1 and min(s, comp_size - s) > 1:
                result.append((a, b))
    return result


def _shape(bricks: list[Brick]) -> tuple[int, int, int]:
    X = max(b.x + b.size_x for b in bricks)
    Y = max(b.y + b.size_y for b in bricks)
    Z = max(b.layer for b in bricks) + 1
    return X, Y, Z


# ---------- local re-tiling ----------

class _State:
    def __init__(self, bricks: list[Brick], grid: np.ndarray):
        self.grid = grid.copy()
        self.bricks: dict[int, Brick] = {b.id: b for b in bricks}
        self.next_id = max(self.bricks, default=-1) + 1
        self.ids = id_map(bricks, grid.shape)

    def list(self) -> list[Brick]:
        return list(self.bricks.values())

    def force(self, z: int, x: int, y: int, sx: int, sy: int) -> list[int] | None:
        """Place a brick at (x, y, layer z), re-tiling whatever it displaces. Returns undo info."""
        X, Y, _ = self.grid.shape
        if x < 0 or y < 0 or x + sx > X or y + sy > Y:
            return None
        key = (min(sx, sy), max(sx, sy))
        if key not in PARTS:
            return None
        region = self.grid[x:x + sx, y:y + sy, z]
        if np.any(region == EMPTY):
            return None
        colors = np.unique(region[region >= 0])
        if len(colors) > 1:
            return None
        layer_ids = self.ids[:, :, z]
        affected = set(np.unique(layer_ids[x:x + sx, y:y + sy]).tolist()) - {-1}
        old = [self.bricks[i] for i in affected]
        # Already exactly this brick?
        if len(old) == 1 and (old[0].x, old[0].y, old[0].size_x, old[0].size_y) == (x, y, sx, sy):
            return None

        freed = np.zeros(layer_ids.shape, bool)
        for b in old:
            freed[b.x:b.x + b.size_x, b.y:b.y + b.size_y] = True
            del self.bricks[b.id]
        freed[x:x + sx, y:y + sy] = False

        w, l = key
        rot = 90 if sx > sy else 0
        forced = Brick(
            id=self.next_id, part=PARTS[key], w=w, l=l, x=x, y=y, z=z * PLATES_PER_BRICK,
            rot=rot, color=int(colors[0]) if len(colors) else DEFAULT_COLOR,
            hidden=len(colors) == 0,
        )
        self.next_id += 1
        masked = np.where(freed, self.grid[:, :, z], EMPTY)
        below = self.ids[:, :, z - 1] if z > 0 else None
        refill, _ = layout_layer(masked, below, z, self.next_id)
        self.next_id += len(refill)

        new = [forced, *refill]
        for b in old:
            layer_ids[b.x:b.x + b.size_x, b.y:b.y + b.size_y] = -1
        for b in new:
            self.bricks[b.id] = b
            layer_ids[b.x:b.x + b.size_x, b.y:b.y + b.size_y] = b.id
        self._undo = (z, old, new)
        return [b.id for b in new]

    def undo(self) -> None:
        z, old, new = self._undo
        layer_ids = self.ids[:, :, z]
        for b in new:
            del self.bricks[b.id]
            layer_ids[b.x:b.x + b.size_x, b.y:b.y + b.size_y] = -1
        for b in old:
            self.bricks[b.id] = b
            layer_ids[b.x:b.x + b.size_x, b.y:b.y + b.size_y] = b.id

    def add_support(self, x: int, y: int, z_from: int) -> int:
        """1x1 support column downward from layer z_from-1 until it lands on something."""
        added = 0
        z = z_from - 1
        while z >= 0 and self.ids[x, y, z] < 0:
            b = Brick(id=self.next_id, part=PARTS[(1, 1)], w=1, l=1, x=x, y=y,
                      z=z * PLATES_PER_BRICK, rot=0, color=DEFAULT_COLOR, hidden=False)
            self.next_id += 1
            self.bricks[b.id] = b
            self.ids[x, y, z] = b.id
            self.grid[x, y, z] = DONT_CARE
            added += 1
            z -= 1
        return added


def _joint_rects(state: _State, a_cells, b_cells):
    """Brick rectangles in an adjacent layer that would span a horizontal joint between comps."""
    a_set = a_cells
    X, Y, Z = state.grid.shape
    for (x, y, z) in a_set:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny, z) not in b_cells:
                continue
            for dz in (0, 1, -1):  # 0: merge across the joint within the same course
                zz = z + dz
                if not 0 <= zz < Z:
                    continue
                x0, y0 = min(x, nx), min(y, ny)
                horiz = dx != 0
                for length in (2, 3, 4):
                    for off in range(length - 1):
                        if horiz:
                            yield zz, x0 - off, y0, length, 1
                        else:
                            yield zz, x0, y0 - off, 1, length
                for length in (2, 3, 4):  # 2-wide spans
                    for off in range(length - 1):
                        for side in (0, -1):
                            if horiz:
                                yield zz, x0 - off, y0 + side, length, 2
                            else:
                                yield zz, x0 + side, y0 - off, 2, length


def _cells(state: _State, comp: set[int]) -> set[tuple[int, int, int]]:
    out = set()
    for i in comp:
        b = state.bricks[i]
        for cx, cy in b.cells():
            out.add((cx, cy, b.layer))
    return out


def _connect(state: _State, shape) -> tuple[bool, int]:
    """Try to reduce the number of components by one. Returns (changed, supports_added)."""
    comps = components(state.list(), shape)
    if len(comps) <= 1:
        return False, 0
    main_cells = _cells(state, comps[0])
    for comp in reversed(comps[1:]):  # smallest first
        cells = _cells(state, comp)
        for rect in _joint_rects(state, cells, main_cells):
            if state.force(*rect) is None:
                continue
            if len(components(state.list(), shape)) < len(comps):
                return True, 0
            state.undo()
    # No joint fix worked: drop a support column under the lowest brick of the smallest
    # component that is floating.
    for comp in reversed(comps[1:]):
        lowest = min((state.bricks[i] for i in comp), key=lambda b: (b.layer, b.id))
        if lowest.layer == 0:
            continue
        for cx, cy in lowest.cells():
            if state.ids[cx, cy, lowest.layer - 1] < 0:
                added = state.add_support(cx, cy, lowest.layer)
                if added:
                    return True, added
    return False, 0


def _strengthen(state: _State, shape, tried: set) -> bool:
    adj = adjacency(state.list(), shape)
    before = len(weak_bridges(adj))
    if before == 0:
        return False
    for a, b in weak_bridges(adj):
        if (a, b) in tried:
            continue
        tried.add((a, b))
        ba, bb = state.bricks[a], state.bricks[b]
        upper, lower = (ba, bb) if ba.layer > bb.layer else (bb, ba)
        shared = set(upper.cells()) & set(lower.cells())
        (sx_, sy_), = shared
        for target in (upper, lower):
            z = target.layer
            for sx, sy in ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (1, 3), (3, 1), (1, 4), (4, 1)):
                for x0 in range(sx_ - sx + 1, sx_ + 1):
                    for y0 in range(sy_ - sy + 1, sy_ + 1):
                        if state.force(z, x0, y0, sx, sy) is None:
                            continue
                        new_adj = adjacency(state.list(), shape)
                        if len(components(state.list(), shape, new_adj)) == 1 and len(
                            weak_bridges(new_adj)
                        ) < before:
                            return True
                        state.undo()
    return False


def com_warning(bricks: list[Brick]) -> str | None:
    if not bricks:
        return None
    pts = np.array([[cx + 0.5, cy + 0.5] for b in bricks for cx, cy in b.cells()])
    com = pts.mean(axis=0)
    base = np.array([[cx + 0.5, cy + 0.5] for b in bricks if b.layer == 0 for cx, cy in b.cells()])
    if len(base) == 0:
        return "Model has no bricks on the ground."
    lo, hi = base.min(axis=0) - 0.5, base.max(axis=0) + 0.5
    inside = bool(np.all(com >= lo) and np.all(com <= hi))
    if inside and len(np.unique(base, axis=0)) >= 3:
        from scipy.spatial import Delaunay, QhullError

        corners = np.concatenate([base + d for d in ([-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5])])
        try:
            inside = Delaunay(corners).find_simplex(com) >= 0
        except QhullError:
            pass
    return None if inside else "Center of mass is outside the base; model may tip."


def repair(bricks: list[Brick], grid: np.ndarray) -> tuple[list[Brick], list[str]]:
    shape = grid.shape
    state = _State(bricks, grid)
    supports = 0
    tried: set = set()
    for _ in range(MAX_ITER):
        changed, added = _connect(state, shape)
        supports += added
        if changed:
            continue
        if not _strengthen(state, shape, tried):
            break

    # Leftover specks of 1-2 bricks that couldn't be joined are image-to-3D noise; a real set
    # never ships a loose 1x1. Bigger leftovers stay and get a warning.
    dropped = 0
    comps = components(state.list(), shape)
    for comp in comps[1:]:
        if len(comp) <= MAX_DROP:
            for i in comp:
                del state.bricks[i]
            dropped += len(comp)

    out = sorted(state.list(), key=lambda b: (b.layer, b.x, b.y))
    for i, b in enumerate(out):
        b.id = i
    warnings = []
    if supports:
        warnings.append(f"Added {supports} support brick{'s' if supports > 1 else ''} under floating parts.")
    adj = adjacency(out, shape)
    n = len(components(out, shape, adj))
    if n > 1:
        warnings.append(f"Build has {n} separate pieces that don't connect.")
    weak = len(weak_bridges(adj))
    if weak:
        warnings.append(f"{weak} weak joint{'s' if weak > 1 else ''} held by a single stud.")
    w = com_warning(out)
    if w:
        warnings.append(w)
    return out, warnings


__all__ = ["repair", "components", "adjacency", "weak_bridges", "com_warning", "id_map"]
