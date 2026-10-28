"""TripoSR wrapper. Heavy imports are lazy so the server runs without the model installed.

Each run happens in a child process (see isolated.py): the server can stop it when it takes
too long, is cancelled, or the machine runs low on memory, and the ~2 GB model is released
when the child exits instead of staying resident between photos.
"""

from __future__ import annotations

import gc
import io
import sys
import threading
import types
from pathlib import Path

import numpy as np
from PIL import Image

from .isolated import run_isolated
from .memory import GB

ROOT = Path(__file__).resolve().parents[3]
TRIPOSR_DIR = ROOT / "vendor" / "TripoSR"

# Tuned for an 8 GB M2: low marching-cubes resolution and small render chunks.
MC_RESOLUTION = 160
CHUNK_SIZE = 4096

# Memory budget (see memory.py). TripoSR peaks around 2 GB; refuse to start below
# MIN_FREE_TO_START and stop a run if the system drops below MIN_FREE_RUNNING, well
# before macOS starts swapping hard enough to freeze.
MIN_FREE_TO_START = int(2.5 * GB)
MIN_FREE_RUNNING = int(0.4 * GB)
MESH_TIMEOUT = 300  # s; a healthy run is ~30 s, a cold first run a minute or two
FOREGROUND_TIMEOUT = 120


class ModelUnavailable(RuntimeError):
    pass


def _install_torchmcubes_shim() -> None:
    """TripoSR imports `torchmcubes`, which needs a C++/CUDA build. PyMCubes is a drop-in."""
    if "torchmcubes" in sys.modules:
        return
    import mcubes
    import torch

    def marching_cubes(vol, thresh):
        v, f = mcubes.marching_cubes(vol.detach().cpu().numpy(), thresh)
        v = v[:, [2, 1, 0]]  # torchmcubes returns (z, y, x)-ordered coordinates
        return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


class _InProcess:
    """The model itself, loaded in whichever process calls it (the child, in practice)."""

    def __init__(self) -> None:
        self._model = None
        self._rembg = None
        self._device = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> tuple[bool, str]:
        if not TRIPOSR_DIR.exists():
            return False, "TripoSR is not installed (run ./scripts/setup.sh)."
        try:
            import mcubes  # noqa: F401
            import rembg  # noqa: F401
            import torch  # noqa: F401
        except ImportError as e:
            return False, f"Model dependencies missing ({e.name}). Run ./scripts/setup.sh."
        return True, ""

    def _load(self) -> None:
        if self._model is not None:
            return
        ok, why = self.available()
        if not ok:
            raise ModelUnavailable(why)
        import rembg
        import torch

        _install_torchmcubes_shim()
        if str(TRIPOSR_DIR) not in sys.path:
            sys.path.insert(0, str(TRIPOSR_DIR))
        from tsr.system import TSR

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = TSR.from_pretrained(
            "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(CHUNK_SIZE)
        model.to(self._device)
        model.eval()
        self._model = model
        if self._rembg is None:
            self._rembg = rembg.new_session("u2net")

    def foreground(self, image: Image.Image) -> np.ndarray:
        session = None
        if not has_alpha(image) and self.available()[0]:
            with self._lock:
                if self._rembg is None:
                    import rembg

                    self._rembg = rembg.new_session("u2net")
                session = self._rembg
        return foreground_pixels(image, session)

    def image_to_mesh(self, image: Image.Image):
        """Returns a Y-up trimesh with vertex colors."""
        import torch
        import trimesh

        with self._lock:
            self._load()
            from tsr.utils import remove_background, resize_foreground

            # Keep the photo's own transparency if it has one; TripoSR then skips rembg.
            src = image.convert("RGBA") if has_alpha(image) else image.convert("RGB")
            img = remove_background(src, self._rembg)
            img = resize_foreground(img, 0.85)
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            img = Image.fromarray((arr * 255.0).astype(np.uint8))

            with torch.no_grad():
                codes = self._model([img], device=self._device)
                mesh = self._model.extract_mesh(codes, True, resolution=MC_RESOLUTION)[0]
            del codes
            gc.collect()
            if self._device == "mps":
                torch.mps.empty_cache()

        # TripoSR output is Z-up; store everything Y-up (glTF convention).
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        return mesh


# Child-process entry points: module level so 'spawn' can import them by name.
_child_model: _InProcess | None = None


def _child() -> _InProcess:
    global _child_model
    if _child_model is None:
        _child_model = _InProcess()
    return _child_model


def _child_image_to_mesh(image: Image.Image) -> bytes:
    return _child().image_to_mesh(image).export(file_type="glb")


def _child_foreground(image: Image.Image) -> np.ndarray:
    return _child().foreground(image)


class TripoSR:
    """Runs each request in a fresh child process (one at a time; MeshStore queues them)."""

    available = staticmethod(_InProcess.available)

    def foreground(self, image: Image.Image) -> np.ndarray:
        if has_alpha(image) or not self.available()[0]:
            return foreground_pixels(image)  # no model needed
        return run_isolated(_child_foreground, (image,), timeout=FOREGROUND_TIMEOUT,
                            min_free=MIN_FREE_RUNNING)

    def image_to_mesh(self, image: Image.Image, cancel: threading.Event | None = None):
        """Returns a Y-up trimesh with vertex colors. Raises isolated.Stopped when cancelled,
        out of time, or low on memory."""
        import trimesh

        glb = run_isolated(_child_image_to_mesh, (image,), timeout=MESH_TIMEOUT, cancel=cancel,
                           min_free=MIN_FREE_RUNNING)
        return trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")


def has_alpha(image: Image.Image) -> bool:
    if image.mode not in ("RGBA", "LA", "P", "PA"):
        return False
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    return bool((alpha < 250).mean() > 0.01)


def foreground_pixels(image: Image.Image, rembg_session=None, n: int = 4000) -> np.ndarray:
    """Up to n RGB samples of the photo's subject, for colour matching. Uses the photo's own
    alpha if present, else rembg; falls back to the centre of the frame."""
    if has_alpha(image):
        rgba = np.asarray(image.convert("RGBA"))
    elif rembg_session is not None:
        import rembg

        small = image.convert("RGB")
        small.thumbnail((512, 512))
        rgba = np.asarray(rembg.remove(small, session=rembg_session))
    else:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        rgba = np.dstack([rgb, np.zeros((h, w), np.uint8)])
        rgba[h // 4: 3 * h // 4, w // 4: 3 * w // 4, 3] = 255
    px = rgba[rgba[..., 3] > 200][:, :3]
    if len(px) > n:
        px = px[np.random.default_rng(0).choice(len(px), n, replace=False)]
    return px.astype(np.uint8)


_instance: TripoSR | None = None


def get_model() -> TripoSR:
    global _instance
    if _instance is None:
        _instance = TripoSR()
    return _instance
