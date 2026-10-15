import numpy as np
import trimesh

from engine.colors import colorize
from engine.hollow import hollow
from engine.types import EMPTY
from engine.voxelize import voxelize
from meshes import cube


def test_small_build_stays_solid():
    g = colorize(voxelize(cube(), 12, up="z"))
    assert np.array_equal(hollow(g, 12), g)


def test_large_cube_is_hollowed_but_shell_intact():
    g = colorize(voxelize(cube(), 20, up="z"))
    h = hollow(g, 20)
    assert (h != EMPTY).sum() < 0.7 * (g != EMPTY).sum()
    visible = g >= 0
    assert np.array_equal(h[visible], g[visible])


def test_support_columns_hold_up_roof():
    g = colorize(voxelize(cube(), 24, up="z"))
    h = hollow(g, 24)
    X, Y, Z = h.shape
    occ = h != EMPTY
    # Every hollow cell is within 2 studs (chessboard) of something solid: max span 4.
    from scipy import ndimage
    dist = ndimage.distance_transform_cdt(~occ[:, :, Z // 2], metric="chessboard")
    assert dist.max() <= 2


def test_tiny_mesh_survives_hollow():
    flat = trimesh.creation.box(extents=[1, 1, 0.01])
    g = colorize(voxelize(flat, 8, up="z"))
    assert (hollow(g, 8) != EMPTY).sum() >= 1
