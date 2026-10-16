import re

import numpy as np

from engine import build
from engine.export import to_ldraw
from engine.slopes import DIRECTIONS, SLOPE_PARTS, front_row
from engine.structure import id_map
from engine.voxelize import voxelize
from meshes import car, cube, pyramid, sphere


def slopes(out):
    return [b for b in out.bricks if b.slope]


def test_tilt_marks_leaning_surfaces_only():
    assert np.abs(voxelize(cube(), 10, up="z").tilt).max() < 1e-6  # box edges are not slopes
    assert np.abs(voxelize(pyramid(), 16, up="z").tilt).max() > 0.5


def test_pyramid_gets_outward_slopes():
    out = build(pyramid(), 16, up="z")
    found = slopes(out)
    assert len(found) >= 8, f"only {len(found)} slopes"
    cx, cy = out.dims[0] / 2, out.dims[1] / 2
    for b in found:
        dx, dy = DIRECTIONS[b.slope]
        bx, by = b.x + b.size_x / 2 - cx, b.y + b.size_y / 2 - cy
        assert bx * dx + by * dy >= 0, f"slope {b.id} faces inward"  # 0: the apex


def test_box_shapes_get_no_slopes():
    assert not slopes(build(cube(), 10, up="z"))


def test_slope_geometry_is_buildable():
    for mesh, size in ((pyramid(), 16), (sphere(), 16), (car(), 32)):
        out = build(mesh, size, up="z")
        ids = id_map(out.bricks, out.dims)
        for b in slopes(out):
            assert b.part in SLOPE_PARTS.values() and not b.hidden
            dx, dy = DIRECTIONS[b.slope]
            assert (b.size_x if dx else b.size_y) == 2
            for x, y in front_row(b, b.slope):
                # nothing sits on the stud-less front row
                assert b.layer + 1 >= out.dims[2] or ids[x, y, b.layer + 1] < 0


def test_ldraw_places_slopes_over_their_footprint():
    out = build(pyramid(), 16, up="z")
    lines = {l.split()[-1] for l in to_ldraw(out.bricks).splitlines() if l.startswith("1 ")}
    assert {"3040.dat", "3039.dat"} & lines
    for b in slopes(out):
        line = next(l for l in to_ldraw([b]).splitlines() if l.startswith("1 "))
        v = list(map(float, line.split()[2:14]))
        pos, m = np.array(v[:3]), np.array(v[3:]).reshape(3, 3)
        # part footprint centre (0, -10) in LDraw part space -> brick centre in the model
        centre = pos + m @ np.array([0, 0, -10])
        assert np.allclose(centre[[0, 2]], [(b.x + b.size_x / 2) * 20, (b.y + b.size_y / 2) * 20])
        assert re.fullmatch(r"\d{4}\.dat", line.split()[-1])
