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


def test_shadowed_blue_stays_blue():
    # same paint, lit vs in shadow
    lit, shade = nearest_color([90, 150, 200]), nearest_color([40, 75, 105])
    assert "Blue" in name(lit) or "Azure" in name(lit)
    assert "Blue" in name(shade) or "Azure" in name(shade)


def test_match_to_reference_restores_brightness():
    from engine.colors import match_to_reference, rgb_to_lab
    rng = np.random.default_rng(0)
    ref = np.clip(rng.normal([110, 160, 200], 15, (500, 3)), 0, 255).astype(np.uint8)
    dark = (ref * 0.45).astype(np.uint8)
    out = match_to_reference(dark, ref)
    assert abs(rgb_to_lab(out)[:, 0].mean() - rgb_to_lab(ref)[:, 0].mean()) < 3


def test_rare_colors_fold_into_common_ones():
    import trimesh
    from meshes import colored
    m = trimesh.creation.box(extents=[1, 1, 1])
    for _ in range(4):
        m = m.subdivide()
    m = colored(m, (200, 20, 20))
    vc = m.visual.vertex_colors.copy()
    vc[0, :3] = (20, 200, 20)  # one green vertex: a speck
    m.visual.vertex_colors = vc
    g = colorize(voxelize(m, 10, up="z"))
    assert set(name(c) for c in np.unique(g[g >= 0])) == {"Red"}
