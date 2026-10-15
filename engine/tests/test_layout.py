import numpy as np

from engine.colors import colorize
from engine.hollow import hollow
from engine.layout import PARTS, id_maps, layout
from engine.types import DONT_CARE, EMPTY
from engine.voxelize import voxelize
from meshes import cube, sphere


def _grid(mesh, size):
    return hollow(colorize(voxelize(mesh, size, up="z")), size)


def seam_stagger_ratio(bricks, shape):
    """Fraction of vertical brick boundaries that do NOT continue straight into the layer below."""
    ids = id_maps(bricks, shape)
    total = staggered = 0
    X, Y, Z = shape
    for z in range(1, Z):
        for axis in (0, 1):
            a = ids[:, :, z]
            b = np.roll(a, -1, axis=axis)
            seam = (a >= 0) & (b >= 0) & (a != b)
            if axis == 0:
                seam[-1, :] = False
            else:
                seam[:, -1] = False
            ab = ids[:, :, z - 1]
            bb = np.roll(ab, -1, axis=axis)
            below_seam = (ab >= 0) & (bb >= 0) & (ab != bb)
            total += seam.sum()
            staggered += (seam & ~below_seam).sum()
    return staggered / max(total, 1)


def test_every_cell_covered_exactly_once():
    g = _grid(sphere(), 16)
    bricks = layout(g)
    count = np.zeros(g.shape, int)
    for b in bricks:
        count[b.x:b.x + b.size_x, b.y:b.y + b.size_y, b.layer] += 1
    assert np.array_equal(count, (g != EMPTY).astype(int))


def test_vocabulary_and_single_color():
    g = _grid(sphere(), 16)
    for b in layout(g):
        assert PARTS[(b.w, b.l)] == b.part
        assert b.rot in (0, 90) and b.z % 3 == 0
        cells = g[b.x:b.x + b.size_x, b.y:b.y + b.size_y, b.layer]
        visible = np.unique(cells[cells >= 0])
        assert len(visible) <= 1
        if len(visible):
            assert b.color == visible[0] and not b.hidden
        else:
            assert (cells == DONT_CARE).all() and b.hidden


def test_solid_block_seams_are_staggered():
    g = _grid(cube(), 10)
    bricks = layout(g)
    assert seam_stagger_ratio(bricks, g.shape) > 0.5
    # Uses big bricks, not a pile of 1x1s
    assert len(bricks) < (g != EMPTY).sum() / 4
