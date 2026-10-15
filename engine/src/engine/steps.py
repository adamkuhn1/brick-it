"""Stage 6: order bricks into instruction steps, the way a real set is built.

Invariant: after every step the model in your hands is one connected piece. Every brick
added must clip onto a brick that is already placed (above or below it), or onto a brick
added earlier in the same step. Nothing is ever left loose on the table.

Order:
  1. Start from the *foundation*: the lowest layer whose largest contiguous area is at
     least half of the model's biggest layer. For a hamburger that is the bottom bun; for a
     car on wheels it is the chassis, not four separate tyre patches.
  2. Grow upward from there, lowest layer first, so the foundation and the course above it
     interlock before anything higher goes on.
  3. Then attach what hangs underneath (wheels, legs), working downward from the foundation.
Each step is a compact cluster of up to 6 bricks spanning at most two neighbouring courses
(hidden filler: up to 10).
"""

from __future__ import annotations

from .structure import _shape, adjacency, components
from .types import Brick, Step

MAX_GROUP, MAX_HIDDEN_GROUP = 6, 10
FOUNDATION_SHARE = 0.5


def _touching(a: Brick, b: Brick) -> bool:
    """Same-layer bricks that share an edge (not just a corner)."""
    ox = min(a.x + a.size_x, b.x + b.size_x) - max(a.x, b.x)
    oy = min(a.y + a.size_y, b.y + b.size_y) - max(a.y, b.y)
    return (ox == 0 and oy > 0) or (oy == 0 and ox > 0) or (ox > 0 and oy > 0)


def _center(b: Brick) -> tuple[float, float]:
    return b.x + b.size_x / 2, b.y + b.size_y / 2


def _regions(bricks: list[Brick]) -> list[list[Brick]]:
    """Groups of edge-touching bricks within one layer."""
    left = list(bricks)
    out = []
    while left:
        region, stack = [], [left.pop()]
        while stack:
            b = stack.pop()
            region.append(b)
            near = [o for o in left if _touching(b, o)]
            for o in near:
                left.remove(o)
            stack.extend(near)
        out.append(region)
    return out


def foundation(bricks: list[Brick]) -> tuple[int, Brick]:
    """(foundation layer, seed brick at the centre of its largest region)."""
    by_layer: dict[int, list[Brick]] = {}
    for b in bricks:
        by_layer.setdefault(b.layer, []).append(b)
    best: dict[int, list[Brick]] = {
        z: max(_regions(bs), key=lambda r: sum(b.w * b.l for b in r)) for z, bs in by_layer.items()
    }
    area = {z: sum(b.w * b.l for b in r) for z, r in best.items()}
    biggest = max(area.values())
    layer = min(z for z, a in area.items() if a >= FOUNDATION_SHARE * biggest)
    region = best[layer]
    cx = sum(_center(b)[0] * b.w * b.l for b in region) / area[layer]
    cy = sum(_center(b)[1] * b.w * b.l for b in region) / area[layer]
    seed = min(region, key=lambda b: ((_center(b)[0] - cx) ** 2 + (_center(b)[1] - cy) ** 2, b.id))
    return layer, seed


def _order_component(comp: list[Brick], adj: dict[int, dict[int, int]]) -> list[list[int]]:
    by_id = {b.id: b for b in comp}
    base, seed = foundation(comp)

    def priority(b: Brick) -> tuple:
        # Upward from the foundation first (lowest layer first), then the underside top-down.
        return (0, b.layer) if b.layer >= base else (1, -b.layer)

    placed: set[int] = set()
    frontier: set[int] = {seed.id}
    groups: list[list[int]] = []
    while frontier:
        best_key = min(priority(by_id[i]) for i in frontier)
        start = min(
            (i for i in frontier if priority(by_id[i]) == best_key),
            key=lambda i: (_dist2(by_id[i], groups[-1] if groups else None, by_id), i),
        )
        layer = by_id[start].layer
        hidden = by_id[start].hidden
        limit = MAX_HIDDEN_GROUP if hidden else MAX_GROUP
        group = [start]
        frontier.discard(start)
        placed.add(start)
        frontier |= {n for n in adj[start] if n not in placed}
        # Keep the step compact: nearest bricks within ~6 studs, same visibility, in this
        # course or the neighbouring one it locks into (above when building up, below when
        # attaching the underside). Same-course bricks win ties.
        other = layer + 1 if layer >= base else layer - 1
        while len(group) < limit:
            sx, sy = _center(by_id[start])
            cands = [
                i for i in frontier
                if by_id[i].layer in (layer, other) and by_id[i].hidden == hidden
                and _dist2(by_id[i], group, by_id) <= 36
            ]
            if not cands:
                break
            nxt = min(cands, key=lambda i: (
                by_id[i].layer != layer,
                (_center(by_id[i])[0] - sx) ** 2 + (_center(by_id[i])[1] - sy) ** 2, i))
            group.append(nxt)
            frontier.discard(nxt)
            placed.add(nxt)
            frontier |= {n for n in adj[nxt] if n not in placed}
        groups.append(group)  # attach order: each brick clips onto earlier ones
    return _merge_small(groups, by_id)


def _merge_small(groups: list[list[int]], by_id: dict[int, Brick]) -> list[list[int]]:
    """Fold 1-2 brick steps into a neighbouring step when the result still fits one step."""
    out: list[list[int]] = []
    for g in groups:
        if out:
            prev = out[-1]
            merged = prev + g
            hidden = {by_id[i].hidden for i in merged}
            limit = MAX_HIDDEN_GROUP if hidden == {True} else MAX_GROUP
            courses = {by_id[i].layer for i in merged}
            small = len(g) <= 2 or len(prev) <= 2
            if small and len(merged) <= limit and max(courses) - min(courses) <= 1:
                out[-1] = merged
                continue
        out.append(list(g))
    return out


def _dist2(b: Brick, group: list[int] | None, by_id: dict[int, Brick]) -> float:
    if not group:
        return 0.0
    bx, by = _center(b)
    return min((_center(by_id[g])[0] - bx) ** 2 + (_center(by_id[g])[1] - by) ** 2 for g in group)


def plan_steps(bricks: list[Brick]) -> list[Step]:
    if not bricks:
        return []
    adj = adjacency(bricks, _shape(bricks))
    by_id = {b.id: b for b in bricks}
    groups: list[list[int]] = []
    # Normally one component. Leftover pieces the repair stage couldn't join (already warned
    # about) come last, each built as its own small sub-assembly.
    for comp in components(bricks, adj=adj):
        groups.extend(_order_component([by_id[i] for i in comp], adj))
    return [Step(n=i + 1, brick_ids=g) for i, g in enumerate(groups)]


__all__ = ["plan_steps", "foundation"]
