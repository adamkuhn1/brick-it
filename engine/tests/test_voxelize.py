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


def test_rotated_box_is_squared_up_to_the_grid():
    box = trimesh.creation.box(extents=[3, 1, 1])
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(35), [0, 0, 1]))
    v = voxelize(box, 24, up="z")
    X, Y, _ = v.occ.shape
    assert X == 24 and Y in (8, 9)  # long side along X, ~1/3 as deep
    assert v.occ[:, :, 0].mean() > 0.9  # a solid rectangle, not a diagonal staircase


def test_small_floating_blobs_are_dropped():
    body = trimesh.creation.box(extents=[2, 2, 1])
    body.apply_translation([0, 0, 0.5])
    blob = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    blob.apply_translation([0, 0, 1.6])  # floating above the body
    v = voxelize(trimesh.util.concatenate([body, blob]), 16, up="z")
    from scipy import ndimage
    assert ndimage.label(v.occ, structure=np.ones((3, 3, 3)))[1] == 1
