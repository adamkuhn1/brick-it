"""Core data types shared by every engine stage. Build JSON is produced from these."""

from __future__ import annotations

from dataclasses import dataclass, field

# Grid cell values for color grids (int16). Non-negative values are palette indices.
EMPTY = -1
DONT_CARE = -2

PLATES_PER_BRICK = 3


@dataclass(frozen=True)
class Color:
    name: str
    hex: str
    bricklink: int
    ldraw: int

    def to_json(self) -> dict:
        return {"bricklink": self.bricklink, "ldraw": self.ldraw, "name": self.name, "hex": self.hex}


@dataclass
class Brick:
    id: int
    part: str
    w: int  # footprint along x after rotation is applied via `rot`
    l: int
    x: int  # studs
    y: int  # studs
    z: int  # plate units
    rot: int  # 0: long side along y; 90: long side along x
    color: int  # palette index
    hidden: bool = False
    # None for a plain brick. For a 45-degree slope brick, the engine direction its low side
    # faces: "+x", "-x", "+y" or "-y" (the footprint is 2 studs deep along that axis). For a
    # 2x2 corner slope (3045), two directions: "+x+y", "+x-y", "-x+y" or "-x-y"; its one stud
    # is on the opposite corner.
    slope: str | None = None
    # True for an inverted slope (3665, 3660): `slope` is still the side its low edge faces,
    # but the slope is on the underside and the whole top has studs.
    slope_inverted: bool = False

    @property
    def layer(self) -> int:
        return self.z // PLATES_PER_BRICK

    @property
    def size_x(self) -> int:
        return self.l if self.rot == 90 else self.w

    @property
    def size_y(self) -> int:
        return self.w if self.rot == 90 else self.l

    def cells(self) -> list[tuple[int, int]]:
        return [(self.x + i, self.y + j) for i in range(self.size_x) for j in range(self.size_y)]


@dataclass
class Step:
    n: int
    brick_ids: list[int]


@dataclass
class Build:
    id: str
    size_studs: int
    dims: tuple[int, int, int]  # x studs, y studs, z bricks
    bricks: list[Brick]
    steps: list[Step]
    warnings: list[str] = field(default_factory=list)

    def parts(self) -> list[dict]:
        from .colors import PALETTE

        counts: dict[tuple[str, int], int] = {}
        for b in self.bricks:
            key = (b.part, PALETTE[b.color].bricklink)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"part": p, "color_bricklink": c, "qty": q}
            for (p, c), q in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    def to_json(self) -> dict:
        from .colors import PALETTE
        from .export import estimate_cost

        return {
            "id": self.id,
            "size_studs": self.size_studs,
            "dims": {"x": self.dims[0], "y": self.dims[1], "z": self.dims[2]},
            "bricks": [
                {
                    "id": b.id, "part": b.part, "w": b.w, "l": b.l, "x": b.x, "y": b.y, "z": b.z,
                    "rot": b.rot, "color": PALETTE[b.color].to_json(), "hidden": b.hidden,
                    "slope": b.slope, "slope_inverted": b.slope_inverted,
                }
                for b in self.bricks
            ],
            "steps": [{"n": s.n, "brick_ids": s.brick_ids} for s in self.steps],
            "parts": self.parts(),
            "stats": {
                "brick_count": len(self.bricks),
                "step_count": len(self.steps),
                "est_cost_usd": estimate_cost(self.bricks),
            },
            "warnings": self.warnings,
        }
