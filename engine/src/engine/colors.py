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


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (D65) [..., 3] -> sRGB uint8."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    d = 6 / 29
    f = lambda t: np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))  # noqa: E731
    xyz = np.stack([f(fx) * 0.95047, f(fy), f(fz) * 1.08883], axis=-1)
    m = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]]
    )
    c = np.clip(xyz @ m.T, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return (c * 255).round().astype(np.uint8)


PALETTE_LAB = rgb_to_lab(np.array([_hex_to_rgb(c.hex) for c in PALETTE]))

# Shading on a photo-derived mesh mostly changes lightness, not hue. Weighting L down keeps a
# shadowed blue panel blue instead of flipping it to a darker palette colour.
L_WEIGHT = 0.6
MIN_COLOR_SHARE = 0.03  # colours on less of the surface than this fold into a common one


def nearest_color(rgb, allowed: np.ndarray | None = None) -> np.ndarray:
    """Palette index of the closest color (L-weighted ΔE) for each RGB in [..., 3]."""
    lab = rgb_to_lab(rgb)
    w = np.array([L_WEIGHT, 1.0, 1.0])
    d = (((lab[..., None, :] - PALETTE_LAB) * w) ** 2).sum(axis=-1)
    if allowed is not None:
        d[..., ~allowed] = np.inf
    return d.argmin(axis=-1)


def match_to_reference(rgb: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Quantile-match RGB samples to the photo's foreground colours, per Lab channel.

    Image-to-3D models return darker, flatter colours than the photo they came from.
    Matching the distributions restores the photo's brightness and saturation without
    needing to know which mesh point maps to which pixel.
    """
    if len(rgb) == 0 or len(ref) < 16:
        return rgb
    src, dst = rgb_to_lab(rgb), rgb_to_lab(ref)
    q = np.linspace(0, 100, 101)
    out = np.empty_like(src)
    for ch in range(3):
        out[:, ch] = np.interp(src[:, ch], np.percentile(src[:, ch], q), np.percentile(dst[:, ch], q))
    return lab_to_rgb(out)


def visible_mask(occ: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one empty face-neighbour (outside the grid counts as empty)."""
    padded = np.pad(occ, 1, constant_values=False)
    interior = padded.copy()
    for axis in range(3):
        for shift in (1, -1):
            interior &= np.roll(padded, shift, axis=axis)
    return occ & ~interior[1:-1, 1:-1, 1:-1]


def smooth(grid: np.ndarray, passes: int = 2) -> np.ndarray:
    """Majority filter over visible voxels: a voxel takes the most common colour among its
    26 neighbours when its own colour is in a clear minority there (removes specks and
    ragged one-voxel fringes, keeps real edges between colour regions)."""
    out = grid.copy()
    for _ in range(passes):
        src = out.copy()
        for x, y, z in np.argwhere(src >= 0):
            c = src[x, y, z]
            neigh = src[max(x - 1, 0):x + 2, max(y - 1, 0):y + 2, max(z - 1, 0):z + 2].ravel()
            neigh = neigh[neigh >= 0]
            counts = np.bincount(neigh, minlength=len(PALETTE))
            counts[c] -= 1  # exclude self
            top = counts.argmax()
            if top != c and counts[top] >= 3 and counts[c] * 2 < counts[top]:
                out[x, y, z] = top
    return out


def colorize(vox: Voxels, ref: np.ndarray | None = None) -> np.ndarray:
    """`ref`: optional (N, 3) uint8 foreground pixels from the source photo."""
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
    rgb = vox.rgb[src]
    if ref is not None:
        rgb = match_to_reference(rgb, ref)
    first = nearest_color(rgb)
    share = np.bincount(first, minlength=len(PALETTE)) / len(first)
    allowed = share >= MIN_COLOR_SHARE
    if not allowed.any():
        allowed[share.argmax()] = True
    grid[visible] = nearest_color(rgb, allowed)
    return smooth(grid)
