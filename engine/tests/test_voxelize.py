import numpy as np
import trimesh

from engine.voxelize import voxelize
from meshes import cube


def test_cube_fills_grid_with_brick_aspect():
    v = voxelize(cube(), 10, up="z")
    x, y, z = v.occ.shape
    assert (x, y) == (10, 10)
    # 10 studs = 80mm wide; 80mm tall / 9.6mm per brick ≈ 8 bricks
    assert z in (8, 9)
    assert v.occ.mean() > 0.95
    assert v.has_color
    # sampled color close to the red we painted
    r, g, b = v.rgb[v.surface].mean(axis=0)
    assert r > 150 and g < 60


def test_y_up_tall_box_stands_upright():
    tall = trimesh.creation.box(extents=[1, 3, 1])  # tall along Y (glTF up)
    v = voxelize(tall, 8, up="y")
    assert v.occ.shape[2] > v.occ.shape[0]


def test_mesh_without_colors_gets_default_gray():
    m = trimesh.creation.box(extents=[1, 1, 1])
    m.visual = trimesh.visual.ColorVisuals(m)  # no colors set
    v = voxelize(m, 8, up="z")
    assert not v.has_color
    assert v.occ.any()
    assert np.all(v.rgb[v.occ] == v.rgb[v.occ][0])


def test_tiny_flat_mesh_produces_at_least_one_voxel():
    flat = trimesh.creation.box(extents=[1, 1, 0.01])
    v = voxelize(flat, 8, up="z")
    assert v.occ.sum() >= 1
    assert v.occ[:, :, 0].any()
