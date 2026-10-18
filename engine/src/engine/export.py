"""Stage 7: LDraw model, BrickLink Wanted List, parts CSV, rough cost."""

from __future__ import annotations

import csv
import io
from xml.sax.saxutils import escape

from .colors import PALETTE
from .types import Brick

LDU_STUD = 20
LDU_BRICK = 24

PART_NAMES = {
    "3005": "Brick 1 x 1", "3004": "Brick 1 x 2", "3622": "Brick 1 x 3", "3010": "Brick 1 x 4",
    "3009": "Brick 1 x 6", "3008": "Brick 1 x 8", "3003": "Brick 2 x 2", "3002": "Brick 2 x 3",
    "3001": "Brick 2 x 4", "2456": "Brick 2 x 6", "3007": "Brick 2 x 8",
    "3040": "Slope 45 2 x 1", "3039": "Slope 45 2 x 2", "3045": "Slope 45 2 x 2 Double Convex",
    "3665": "Slope 45 2 x 1 Inverted", "3660": "Slope 45 2 x 2 Inverted",
}

# Rough average used-market prices (USD) per piece. Labelled as an estimate in the UI.
PRICE_USD = {
    "3005": 0.06, "3004": 0.07, "3622": 0.09, "3010": 0.10, "3009": 0.15, "3008": 0.20,
    "3003": 0.09, "3002": 0.12, "3001": 0.14, "2456": 0.25, "3007": 0.30,
    "3040": 0.07, "3039": 0.10, "3045": 0.12, "3665": 0.08, "3660": 0.12,
}

# Rotations about the vertical (LDraw Y) axis, as LDraw's row-major 3x3.
_R0, _R90 = "1 0 0 0 1 0 0 0 1", "0 0 -1 0 1 0 1 0 0"  # _R90 turns part -Z to model +X
_R180, _R270 = "-1 0 0 0 1 0 0 0 -1", "0 0 1 0 1 0 -1 0 0"

# Engine y maps to LDraw Z. Straight slopes (3040, 3039) face -Z with their origin over the
# back stud row: the footprint runs z = -30..10, so its centre is at (0, -10) from the origin.
# The inverted slopes 3665 (2x1) and 3660 (2x2) are laid out the same way in their .dat
# files (3665a/3660a): the underside rises from the back row toward -Z, to a lip at the top.
# The corner slope 3045 faces +X and -Z (engine "+x-y") with its one stud at the origin: the
# footprint runs x = -10..30, z = -30..10, so its centre is at (10, -10).
# Rotation that turns each part toward each engine facing:
SLOPE_ROT = {
    "-y": _R0, "+y": _R180, "+x": _R90, "-x": _R270,
    "+x-y": _R0, "-x+y": _R180, "+x+y": _R90, "-x-y": _R270,
}
SLOPE_CENTRE = {"3040": (0, -10), "3039": (0, -10), "3665": (0, -10), "3660": (0, -10),
                "3045": (10, -10)}  # (x, z) in part space


def estimate_cost(bricks: list[Brick]) -> float:
    return round(sum(PRICE_USD[b.part] for b in bricks), 2)


def to_ldraw(bricks: list[Brick], steps=None, name: str = "lego-build") -> str:
    """LDraw .ldr. LDraw parts are long along X; -Y is up; origin at the top centre of a brick."""
    lines = [f"0 {name}", f"0 Name: {name}.ldr", "0 Author: lego (photo-to-LEGO)", ""]
    by_id = {b.id: b for b in bricks}
    groups = [s.brick_ids for s in steps] if steps else [[b.id for b in bricks]]
    for ids in groups:
        for i in ids:
            b = by_id[i]
            cx = (b.x + b.size_x / 2) * LDU_STUD
            cz = (b.y + b.size_y / 2) * LDU_STUD
            top = -(b.layer + 1) * LDU_BRICK
            m = "1 0 0 0 1 0 0 0 1" if b.rot == 90 or b.w == b.l else "0 0 1 0 1 0 -1 0 0"
            if b.slope:
                m = SLOPE_ROT[b.slope]
                r = [float(v) for v in m.split()]
                px, pz = SLOPE_CENTRE[b.part]  # the origin sits this far from the centre
                cx -= r[0] * px + r[2] * pz
                cz -= r[6] * px + r[8] * pz
            lines.append(f"1 {PALETTE[b.color].ldraw} {_n(cx)} {_n(top)} {_n(cz)} {m} {b.part}.dat")
        if steps:
            lines.append("0 STEP")
    return "\n".join(lines) + "\n"


def _n(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def to_bricklink_xml(parts: list[dict]) -> str:
    items = "".join(
        "<ITEM><ITEMTYPE>P</ITEMTYPE>"
        f"<ITEMID>{escape(p['part'])}</ITEMID><COLOR>{p['color_bricklink']}</COLOR>"
        f"<MINQTY>{p['qty']}</MINQTY></ITEM>\n"
        for p in parts
    )
    return f"<INVENTORY>\n{items}</INVENTORY>\n"


def to_csv(parts: list[dict]) -> str:
    by_bl = {c.bricklink: c for c in PALETTE}
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["part", "description", "color", "bricklink_color_id", "ldraw_color_id", "qty",
                "est_unit_price_usd"])
    for p in parts:
        c = by_bl[p["color_bricklink"]]
        w.writerow([p["part"], PART_NAMES[p["part"]], c.name, c.bricklink, c.ldraw, p["qty"],
                    PRICE_USD[p["part"]]])
    return buf.getvalue()
