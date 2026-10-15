"""Stage 3: hollow the model to save bricks, keeping a 2-voxel-thick shell plus support columns.

Interior voxels are removed except 2x2 support columns on a 6-stud lattice, so no
unsupported span inside the model is wider than 4 studs. Small builds stay solid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .types import EMPTY

SOLID_MAX_STUDS = 12
SHELL_THICKNESS = 2  # visible layer + one behind it
COLUMN_PITCH = 6
COLUMN_WIDTH = 2


def hollow(grid: np.ndarray, size_studs: int) -> np.ndarray:
    if size_studs <= SOLID_MAX_STUDS:
        return grid.copy()
    occ = grid != EMPTY
    padded = np.pad(occ, 1, constant_values=False)
    depth = ndimage.distance_transform_cdt(padded, metric="taxicab")[1:-1, 1:-1, 1:-1]
    interior = occ & (depth > SHELL_THICKNESS)

    X, Y, _ = grid.shape
    xs = (np.arange(X) % COLUMN_PITCH) < COLUMN_WIDTH
    ys = (np.arange(Y) % COLUMN_PITCH) < COLUMN_WIDTH
    lattice = (xs[:, None] & ys[None, :])[:, :, None]

    out = grid.copy()
    out[interior & ~lattice] = EMPTY
    return out
