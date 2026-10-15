"""TripoSR wrapper. Heavy imports are lazy so the server runs without the model installed."""

from __future__ import annotations

import gc
import sys
import threading
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
TRIPOSR_DIR = ROOT / "vendor" / "TripoSR"

# Tuned for an 8 GB M2: low marching-cubes resolution and small render chunks.
MC_RESOLUTION = 160
CHUNK_SIZE = 4096


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


class TripoSR:
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
