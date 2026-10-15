"""Stage 6: order bricks into instruction steps.

Bottom to top. A brick can be placed once it would attach to something already built:
the ground (layer 0), a placed brick below it, or a placed brick above it (for bricks that
hang under an overhang). Within a layer we flood outward from a seed so each step is a
compact cluster of 3-6 bricks. Steps of only hidden bricks are batched up to 10.
"""

from __future__ import annotations

from .structure import adjacency, _shape
from .types import Brick, Step

MIN_GROUP, MAX_GROUP, MAX_HIDDEN_GROUP = 3, 6, 10


def _touching(a: Brick, b: Brick) -> bool:
    """Same-layer bricks that share an edge."""
    return not (
        a.x + a.size_x < b.x or b.x + b.size_x < a.x or a.y + a.size_y < b.y or b.y + b.size_y < a.y
    ) and not (
        (a.x + a.size_x == b.x or b.x + b.size_x == a.x)
        and (a.y + a.size_y == b.y or b.y + b.size_y == a.y)
    )


def _center(b: Brick) -> tuple[float, float]:
    return b.x + b.size_x / 2, b.y + b.size_y / 2


def plan_steps(bricks: list[Brick]) -> list[Step]:
    if not bricks:
        return []
    adj = adjacency(bricks, _shape(bricks))
    by_id = {b.id: b for b in bricks}
    placed: set[int] = set()
    pending = set(by_id)
    steps: list[Step] = []

    def placeable(i: int) -> bool:
        return by_id[i].layer == 0 or any(n in placed for n in adj[i])

    while pending:
        ready = [i for i in pending if placeable(i)]
        if not ready:  # disconnected leftovers (already warned about): place them anyway
            ready = list(pending)
        layer = min(by_id[i].layer for i in ready)
        ready = [i for i in ready if by_id[i].layer == layer]
        ready_set = set(ready)

        seed = min(ready, key=lambda i: (by_id[i].hidden is False, by_id[i].x, by_id[i].y))
        hidden_only = by_id[seed].hidden
        limit = MAX_HIDDEN_GROUP if hidden_only else MAX_GROUP
        pool = {i for i in ready_set if by_id[i].hidden == hidden_only}

        group = [seed]
        frontier = [seed]
        while frontier and len(group) < limit:
            cur = frontier.pop(0)
            for j in sorted(pool - set(group), key=lambda j: (by_id[j].x, by_id[j].y)):
                if len(group) >= limit:
                    break
                if _touching(by_id[cur], by_id[j]):
                    group.append(j)
                    frontier.append(j)
        # Top up tiny groups with the nearest remaining ready bricks in this layer.
        if len(group) < MIN_GROUP:
            cx, cy = _center(by_id[seed])
            rest = sorted(
                pool - set(group),
                key=lambda j: (_center(by_id[j])[0] - cx) ** 2 + (_center(by_id[j])[1] - cy) ** 2,
            )
            group.extend(rest[: MIN_GROUP - len(group)])

        steps.append(Step(n=len(steps) + 1, brick_ids=sorted(group)))
        placed.update(group)
        pending.difference_update(group)
    return steps
