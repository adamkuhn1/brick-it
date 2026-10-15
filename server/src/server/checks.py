"""Reject bad inputs early with messages a person can act on."""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

MIN_SIDE = 256
BLANK_STD = 6.0
BLUR_VAR = 25.0


class InputRejected(ValueError):
    pass


def check_image(img: Image.Image) -> None:
    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise InputRejected(
            f"Photo is too small ({w}×{h}). Use one at least {MIN_SIDE} px on each side."
        )
    gray = img.convert("L")
    gray.thumbnail((512, 512))
    a = np.asarray(gray, dtype=np.float32)
    if a.std() < BLANK_STD:
        raise InputRejected("Photo looks blank. Make sure the object is in frame and lit.")
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    if lap.var() < BLUR_VAR:
        raise InputRejected("Photo looks blurry. Hold steady and tap to focus on the object.")


def check_mesh(mesh: trimesh.Trimesh) -> None:
    if mesh is None or len(mesh.faces) == 0 or len(mesh.vertices) < 4:
        raise InputRejected(
            "Couldn't find an object in that photo. Try one object on a plain background."
        )
    ext = np.sort(mesh.extents)
    if ext[-1] <= 0 or ext[0] < 0.1 * ext[-1]:
        raise InputRejected(
            "The 3D shape came out nearly flat. Try another photo from a slightly higher, "
            "angled view so depth is visible."
        )
