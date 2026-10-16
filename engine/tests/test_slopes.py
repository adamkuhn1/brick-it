import re

import numpy as np

from engine import build
from engine.export import to_ldraw
from engine.slopes import (CORNER_PART, DIRECTIONS, SLOPE_PARTS, _leaning_cells, axes, facing,
                           front_row, studless)
from engine.structure import id_map
from engine.voxelize import voxelize
from meshes import car, cube, pyramid, sphere


def slopes(out):
    return [b for b in out.bricks if b.slope]


def corners(out):
    return [b for b in out.bricks if b.part == CORNER_PART]


def assert_outward(out):
    cx, cy = out.dims[0] / 2, out.dims[1] / 2
    for b in slopes(out):
        bx, by = b.x + b.size_x / 2 - cx, b.y + b.size_y / 2 - cy
        for a in axes(b.slope):  # a corner must face outward along both of its directions
            dx, dy = DIRECTIONS[a]
            assert bx * dx + by * dy >= 0, f"slope {b.id} ({b.slope}) faces inward"  # 0: apex


def test_tilt_marks_leaning_surfaces_only():
    assert np.abs(voxelize(cube(), 10, up="z").tilt).max() < 1e-6  # box edges are not slopes
    assert np.abs(voxelize(pyramid(), 16, up="z").tilt).max() > 0.5


def test_pyramid_gets_outward_slopes():
    out = build(pyramid(), 16, up="z")
    found = slopes(out)
    assert len(found) >= 8, f"only {len(found)} slopes"
    assert_outward(out)


def test_pyramid_corners_get_corner_slopes_facing_outward():
    out = build(pyramid(), 16, up="z")
    found = corners(out)
    assert len(found) >= 8, f"only {len(found)} corner slopes"
    assert {b.slope for b in found} == {"+x+y", "+x-y", "-x+y", "-x-y"}  # all four corners
    cx, cy = out.dims[0] / 2, out.dims[1] / 2
    for b in found:
        assert (b.size_x, b.size_y) == (2, 2) and len(axes(b.slope)) == 2
        bx, by = b.x + b.size_x / 2 - cx, b.y + b.size_y / 2 - cy
        fx, fy = facing(b.slope)
        at_apex = (bx, by) == (0, 0)  # the top 2x2 sits on the centre, any facing is outward
        assert at_apex or (bx * fx > 0 and by * fy > 0), f"corner {b.id} ({b.slope}) at {bx, by}"


def test_sphere_gets_outward_corner_slopes():
    out = build(sphere(), 16, up="z")
    assert len(corners(out)) >= 4
    assert_outward(out)


def test_corner_slopes_cover_more_sloped_edges():
    """Corners used to stay stepped; the pyramid's diagonal edges now mostly get slopes."""
    out = build(pyramid(), 16, up="z")
    tilt = voxelize(pyramid(), 16, up="z").tilt
    cells = {(x, y, z) for x, y, z, _ in _leaning_cells(id_map(out.bricks, out.dims), tilt)}
    covered = {(x, y, b.layer) for b in slopes(out) for x, y in studless(b, b.slope)}
    assert len(cells & covered) >= 0.85 * len(cells)


def test_box_shapes_get_no_slopes():
    assert not slopes(build(cube(), 10, up="z"))


def test_slope_geometry_is_buildable():
    for mesh, size in ((pyramid(), 16), (sphere(), 16), (car(), 32)):
        out = build(mesh, size, up="z")
        ids = id_map(out.bricks, out.dims)
        for b in slopes(out):
            assert b.part in (*SLOPE_PARTS.values(), CORNER_PART) and not b.hidden
            assert (b.part == CORNER_PART) == (len(axes(b.slope)) == 2)
            for a in axes(b.slope):
                dx, dy = DIRECTIONS[a]
                assert (b.size_x if dx else b.size_y) == 2
            for x, y in studless(b, b.slope):
                # nothing sits on the stud-less cells
                assert b.layer + 1 >= out.dims[2] or ids[x, y, b.layer + 1] < 0


def test_ldraw_places_slopes_over_their_footprint():
    out = build(pyramid(), 16, up="z")
    lines = {l.split()[-1] for l in to_ldraw(out.bricks).splitlines() if l.startswith("1 ")}
    assert {"3040.dat", "3039.dat"} & lines and "3045.dat" in lines
    for b in slopes(out):
        pos, m = _placement(b)
        # Footprint centre in LDraw part space (from the official .dat files): 3040/3039 have
        # their origin over the back row, 3045 over its one stud. It must land on the brick.
        local = np.array([10, 0, -10] if b.part == CORNER_PART else [0, 0, -10])
        centre = pos + m @ local
        assert np.allclose(centre[[0, 2]], [(b.x + b.size_x / 2) * 20, (b.y + b.size_y / 2) * 20])


def _placement(b):
    line = next(l for l in to_ldraw([b]).splitlines() if l.startswith("1 "))
    assert re.fullmatch(r"\d{4}\.dat", line.split()[-1])
    v = list(map(float, line.split()[2:14]))
    m = np.array(v[3:]).reshape(3, 3)
    assert np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1)  # a rotation
    return np.array(v[:3]), m


def test_ldraw_turns_slopes_to_face_their_direction():
    """3040/3039 slope down toward part -Z; 3045 toward part +X and -Z (official .dat files).
    After rotation those must point along the engine facing (engine y = LDraw Z)."""
    out = build(pyramid(), 16, up="z")
    for b in slopes(out):
        _, m = _placement(b)
        downhill = [(1, 0, 0), (0, 0, -1)] if b.part == CORNER_PART else [(0, 0, -1)]
        got = {tuple(int(round(v)) for v in (m @ np.array(d))[[0, 2]]) for d in downhill}
        assert got == {DIRECTIONS[a] for a in axes(b.slope)}, (b.part, b.slope, got)
