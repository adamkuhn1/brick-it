"""Synthetic test meshes (Z-up unless noted)."""

import numpy as np
import trimesh


def colored(mesh: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    mesh.visual.vertex_colors = np.tile([*rgb, 255], (len(mesh.vertices), 1)).astype(np.uint8)
    return mesh


def cube(size=1.0, rgb=(200, 20, 20)):
    return colored(trimesh.creation.box(extents=[size, size, size]), rgb)


def sphere(rgb=(30, 90, 200)):
    return colored(trimesh.creation.icosphere(subdivisions=3, radius=1.0), rgb)


def l_shape():
    a = trimesh.creation.box(extents=[2, 1, 1])
    b = trimesh.creation.box(extents=[1, 1, 2])
    b.apply_translation([-0.5, 0, 1.5])
    a.apply_translation([0, 0, 0.5])
    return colored(trimesh.util.concatenate([a, b]), (240, 200, 0))


def arch():
    left = trimesh.creation.box(extents=[0.5, 1, 2])
    left.apply_translation([-1.25, 0, 1])
    right = trimesh.creation.box(extents=[0.5, 1, 2])
    right.apply_translation([1.25, 0, 1])
    top = trimesh.creation.box(extents=[3, 1, 0.5])
    top.apply_translation([0, 0, 2.25])
    return colored(trimesh.util.concatenate([left, right, top]), (120, 120, 120))


def overhanging_t():
    stem = trimesh.creation.box(extents=[0.5, 0.5, 2])
    stem.apply_translation([0, 0, 1])
    bar = trimesh.creation.box(extents=[3, 0.5, 0.5])
    bar.apply_translation([0, 0, 2.25])
    return colored(trimesh.util.concatenate([stem, bar]), (20, 150, 60))


def car(yaw_degrees=35):
    """Low car body on four wheels, turned like a 3/4-view photo reconstruction (Z-up)."""
    body = colored(trimesh.creation.box(extents=[4.2, 1.8, 0.7]), (110, 165, 205))
    body.apply_translation([0, 0, 0.75])
    cabin = colored(trimesh.creation.box(extents=[2.0, 1.5, 0.55]), (25, 30, 35))
    cabin.apply_translation([-0.2, 0, 1.37])
    parts = [body, cabin]
    for x in (-1.4, 1.4):
        for y in (-0.8, 0.8):
            w = trimesh.creation.cylinder(radius=0.4, height=0.3, sections=24)
            w.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            w.apply_translation([x, y, 0.4])
            parts.append(colored(w, (190, 160, 90)))
    m = trimesh.util.concatenate(parts)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw_degrees), [0, 0, 1]))
    return m


def pyramid(rgb=(200, 120, 40)):
    """Square pyramid with 45-degree faces (each face leans toward +-x or +-y)."""
    pts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1.0]])
    return colored(trimesh.convex.convex_hull(pts), rgb)
