import numpy as np

from engine.colors import colorize
from engine.hollow import hollow
from engine.layout import layout
from engine.structure import adjacency, com_warning, components, repair, weak_bridges
from engine.types import Brick
from engine.voxelize import voxelize
from meshes import arch, cube, l_shape, overhanging_t, sphere


def _built(mesh, size):
    g = hollow(colorize(voxelize(mesh, size, up="z")), size)
    return repair(layout(g), g), g


def test_bridge_detection():
    # Tower of three 1x2s, each overlapping the next by 1 stud; plus one more on top.
    b = lambda i, x, z: Brick(i, "3004", 1, 2, x, 0, z * 3, 90, 0)
    bricks = [b(0, 0, 0), b(1, 1, 1), b(2, 2, 2), b(3, 2, 3)]
    adj = adjacency(bricks, (4, 1, 4))
    assert len(components(bricks, adj=adj)) == 1
    # 0-1 splits off 1 brick (ignored); 1-2 splits 2|2 (weak); 2-3 overlap 2 (not weak)
    assert weak_bridges(adj) == [(1, 2)] or weak_bridges(adj) == [(2, 1)]


def test_shapes_single_component_no_floating():
    for mesh, size in [(cube(), 12), (sphere(), 16), (l_shape(), 16), (arch(), 16), (overhanging_t(), 16)]:
        (bricks, warnings), g = _built(mesh, size)
        assert len(components(bricks, g.shape)) == 1, (mesh, warnings)
        # ids are sequential
        assert [b.id for b in bricks] == list(range(len(bricks)))


def test_overhanging_t_warns_or_balances():
    (bricks, warnings), _ = _built(overhanging_t(), 16)
    assert com_warning(bricks) is None  # symmetric T is balanced


def test_com_warning_for_lopsided():
    b = lambda i, x, z: Brick(i, "3008", 1, 8, x, 0, z * 3, 90, 0)
    base = Brick(0, "3005", 1, 1, 0, 0, 0, 0, 0)
    arm = b(1, 0, 1)
    assert com_warning([base, arm]) is not None


def test_floating_part_gets_connected():
    g = hollow(colorize(voxelize(cube(), 8, up="z")), 8)
    g = g.copy()
    # carve a gap so the top two layers float
    from engine.types import EMPTY
    g[:, :, 3] = EMPTY
    bricks, warnings = repair(layout(g), g)
    assert len(components(bricks, g.shape)) == 1
    assert any("support" in w for w in warnings)
