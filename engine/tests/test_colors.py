import numpy as np

from engine.colors import PALETTE, colorize, nearest_color, smooth, visible_mask
from engine.types import DONT_CARE, EMPTY
from engine.voxelize import voxelize
from meshes import cube


def name(i):
    return PALETTE[int(i)].name


def test_nearest_color_basic():
    assert name(nearest_color([201, 26, 9])) == "Red"
    assert name(nearest_color([255, 255, 255])) == "White"
    assert name(nearest_color([0, 0, 0])) == "Black"
    assert name(nearest_color([250, 210, 50])) == "Yellow"


def test_visible_mask_cube():
    occ = np.ones((5, 5, 5), bool)
    vis = visible_mask(occ)
    assert vis.sum() == 125 - 27


def test_smooth_removes_speck():
    g = np.full((3, 3, 3), 4, np.int16)
    g[1, 1, 1] = 0
    assert smooth(g)[1, 1, 1] == 4


def test_colorize_red_cube_shell_red_interior_dont_care():
    g = colorize(voxelize(cube(), 10, up="z"))
    vis = g >= 0
    assert all(name(c) == "Red" for c in np.unique(g[vis]))
    assert (g == DONT_CARE).any()
    assert not (g == EMPTY).all()
