"""Stage 1: mesh -> occupancy grid at 1 stud x 1 brick resolution, with per-voxel RGB.

A LEGO brick is 8 mm wide per stud and 9.6 mm tall, so a voxel is 1.2x taller than wide.
We scale the mesh so its largest horizontal extent equals `size_studs` stud widths, then
squash Z by 1.2 so every grid unit is one stud (x, y) or one brick (z).

Occupancy comes from dense surface sampling (robust to non-watertight meshes) followed by
3D hole filling. Surface samples also carry the color, interpolated from vertex colors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage

BRICK_ASPECT = 9.6 / 8.0
DEFAULT_RGB = (160, 165, 169)  # light bluish gray


@dataclass
class Voxels:
    occ: np.ndarray  # bool [X, Y, Z]
    rgb: np.ndarray  # uint8 [X, Y, Z, 3]; meaningful where surface[...] is True
    surface: np.ndarray  # bool [X, Y, Z], voxels touched by the mesh surface
    has_color: bool


def _to_z_up(mesh: trimesh.Trimesh, up: str) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if up == "y":
        # (x, y, z) -> (x, -z, y): glTF/TripoSR Y-up becomes Z-up.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up != "z":
        raise ValueError(f"up must be 'y' or 'z', got {up!r}")
    return mesh


def _vertex_colors(mesh: trimesh.Trimesh) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (vertex_rgb, face_rgb); at most one is not None."""
    visual = mesh.visual
    if visual is None:
        return None, None
    if visual.kind == "texture":
        try:
            visual = visual.to_color()
        except Exception:
            return None, None
    if visual.kind == "vertex":
        return np.asarray(visual.vertex_colors)[:, :3].astype(np.float32), None
    if visual.kind == "face":
        return None, np.asarray(visual.face_colors)[:, :3].astype(np.float32)
    return None, None


def voxelize(mesh: trimesh.Trimesh, size_studs: int, up: str = "y", seed: int = 0) -> Voxels:
    if size_studs < 1:
        raise ValueError("size_studs must be >= 1")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    vcol, fcol = _vertex_colors(mesh)
    mesh = _to_z_up(mesh, up)

    lo, hi = mesh.bounds
    extent = hi - lo
    horizontal = max(extent[0], extent[1])
    if horizontal <= 0:
        horizontal = max(extent.max(), 1e-9)
    s = size_studs / horizontal
    scale = np.array([s, s, s / BRICK_ASPECT])

    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    dims = np.maximum(np.ceil(extent * scale - 1e-6).astype(int), 1)
    dims[:2] = np.minimum(dims[:2], size_studs)
    half = dims[:2] / 2.0

    def to_grid(p: np.ndarray) -> np.ndarray:
        g = (p - center) * scale
        g[:, :2] += half
        idx = np.floor(g).astype(int)
        return np.clip(idx, 0, dims - 1)

    # Dense surface samples in grid space: ~40 samples per square stud.
    area = mesh.area * s * s
    n = int(np.clip(area * 40, 20_000, 400_000))
    pts, face_idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    pts = np.vstack([pts, mesh.vertices])
    face_idx = np.concatenate([face_idx, _vertex_face(mesh)])

    idx = to_grid(pts)
    occ = np.zeros(dims, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    surface = occ.copy()

    filled = ndimage.binary_fill_holes(occ)
    occ = filled if filled is not None else occ

    # Colors
    has_color = vcol is not None or fcol is not None
    rgb = np.zeros((*dims, 3), dtype=np.uint8)
    rgb[...] = DEFAULT_RGB
    if has_color:
        if vcol is not None:
            tri = mesh.triangles[face_idx]
            bary = trimesh.triangles.points_to_barycentric(tri, pts)
            bary = np.nan_to_num(np.clip(bary, 0, 1))
            bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-9)
            colors = (vcol[mesh.faces[face_idx]] * bary[:, :, None]).sum(axis=1)
        else:
            colors = fcol[face_idx]
        flat = np.ravel_multi_index(idx.T, dims)
        total = np.zeros((occ.size, 3))
        count = np.zeros(occ.size)
        np.add.at(total, flat, colors)
        np.add.at(count, flat, 1)
        hit = count > 0
        avg = (total[hit] / count[hit, None]).round().astype(np.uint8)
        rgb.reshape(-1, 3)[hit] = avg

    return Voxels(occ=occ, rgb=rgb, surface=surface, has_color=has_color)


def _vertex_face(mesh: trimesh.Trimesh) -> np.ndarray:
    """One adjacent face per vertex (used to color the raw vertex samples)."""
    out = np.zeros(len(mesh.vertices), dtype=int)
    out[mesh.faces[:, 0]] = np.arange(len(mesh.faces))
    out[mesh.faces[:, 1]] = np.arange(len(mesh.faces))
    out[mesh.faces[:, 2]] = np.arange(len(mesh.faces))
    return out
