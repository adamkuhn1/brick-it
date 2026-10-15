"""Stage 2: map voxel RGB to a curated palette of common solid LEGO colors (CIELAB ΔE).

Output grid values: palette index for visible voxels, DONT_CARE for occupied voxels hidden
inside the model, EMPTY elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .types import DONT_CARE, EMPTY, Color
from .voxelize import Voxels

PALETTE: list[Color] = [
    Color("White", "#FFFFFF", 1, 15),
    Color("Black", "#05131D", 11, 0),
    Color("Light Bluish Gray", "#A0A5A9", 86, 71),
    Color("Dark Bluish Gray", "#6C6E68", 85, 72),
    Color("Red", "#C91A09", 5, 4),
    Color("Dark Red", "#720E0F", 59, 320),
    Color("Orange", "#FE8A18", 4, 25),
    Color("Bright Light Orange", "#F8BB3D", 110, 191),
    Color("Yellow", "#F2CD37", 3, 14),
    Color("Lime", "#BBE90B", 34, 27),
    Color("Green", "#237841", 6, 2),
    Color("Dark Green", "#184632", 80, 288),
    Color("Sand Green", "#A0BCAC", 48, 378),
    Color("Blue", "#0055BF", 7, 1),
    Color("Medium Blue", "#5A93DB", 42, 73),
    Color("Dark Azure", "#078BC9", 153, 321),
    Color("Dark Blue", "#0A3463", 63, 272),
    Color("Tan", "#E4CD9E", 2, 19),
    Color("Dark Tan", "#958A73", 69, 28),
    Color("Nougat", "#D09168", 28, 92),
    Color("Medium Nougat", "#AA7D55", 150, 84),
    Color("Reddish Brown", "#582A12", 88, 70),
    Color("Bright Pink", "#E4ADC8", 104, 29),
    Color("Magenta", "#923978", 71, 26),
]

DEFAULT_COLOR = next(i for i, c in enumerate(PALETTE) if c.name == "Light Bluish Gray")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 [..., 3] -> CIELAB (D65) float [..., 3]."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


PALETTE_LAB = rgb_to_lab(np.array([_hex_to_rgb(c.hex) for c in PALETTE]))


def nearest_color(rgb) -> np.ndarray:
    """Palette index of the closest color (ΔE76) for each RGB in [..., 3]."""
    lab = rgb_to_lab(rgb)
    d = ((lab[..., None, :] - PALETTE_LAB) ** 2).sum(axis=-1)
    return d.argmin(axis=-1)


def visible_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one empty face-neighbour (outside the grid counts as empty)."""
    padded = np.pad(occ, 1, constant_values=False)
    interior = padded.copy()
    for axis in range(3):
        for shift in (1, -1):
            interior &= np.roll(padded, shift, axis=axis)
    return occ & ~interior[1:-1, 1:-1, 1:-1]


def smooth(grid: np.ndarray) -> np.ndarray:
    """Recolor isolated single-voxel specks to the most common colour among their neighbours."""
    out = grid.copy()
    X, Y, Z = grid.shape
    for x, y, z in np.argwhere(grid >= 0):
        c = grid[x, y, z]
        neigh = grid[max(x - 1, 0):x + 2, max(y - 1, 0):y + 2, max(z - 1, 0):z + 2].ravel()
        neigh = neigh[neigh >= 0]
        counts = np.bincount(neigh, minlength=len(PALETTE))
        counts[c] -= 1  # exclude self
        if counts[c] == 0 and counts.sum() >= 3:
            out[x, y, z] = counts.argmax()
    return out


def colorize(vox: Voxels) -> np.ndarray:
    grid = np.full(vox.occ.shape, EMPTY, dtype=np.int16)
    grid[vox.occ] = DONT_CARE
    visible = visible_mask(vox.occ)
    if not vox.has_color:
        grid[visible] = DEFAULT_COLOR
        return grid

    sampled = vox.surface & vox.occ
    if not sampled.any():
        grid[visible] = DEFAULT_COLOR
        return grid
    # Visible voxels the surface sampling missed borrow the nearest sampled voxel's RGB.
    _, idx = ndimage.distance_transform_edt(~sampled, return_indices=True)
    src = tuple(i[visible] for i in idx)
    grid[visible] = nearest_color(vox.rgb[src])
    return smooth(grid)
