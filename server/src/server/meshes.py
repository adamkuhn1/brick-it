"""Mesh cache + background job runner. Meshes are stored Y-up as .glb, keyed by input sha256."""

from __future__ import annotations

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps

from .checks import InputRejected, check_image, check_mesh
from .model import ModelUnavailable, get_model

MESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Job:
    status: str  # queued | running | done | error
    error: str | None = None


class MeshStore:
    def __init__(self, root: Path, model_factory=get_model):
        self.dir = root / "meshes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)  # one inference at a time (8 GB budget)
        self.model_factory = model_factory

    def path(self, mesh_id: str) -> Path:
        return self.dir / f"{mesh_id}.glb"

    def status(self, mesh_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(mesh_id)
        if job is None and self.path(mesh_id).exists():
            return Job("done")
        return job

    def load(self, mesh_id: str) -> trimesh.Trimesh:
        return trimesh.load(self.path(mesh_id), force="mesh")

    def ref_path(self, mesh_id: str) -> Path:
        return self.dir / f"{mesh_id}.colors.npy"

    def color_ref(self, mesh_id: str) -> np.ndarray | None:
        p = self.ref_path(mesh_id)
        return np.load(p) if p.exists() else None

    def _save_ref(self, mesh_id: str, img: Image.Image) -> None:
        if self.ref_path(mesh_id).exists():
            return
        try:
            px = self.model_factory().foreground(img)
        except Exception:
            return  # colour matching is an improvement, never a blocker
        if len(px) >= 16:
            np.save(self.ref_path(mesh_id), px)

    # ----- submissions -----

    def submit(self, data: bytes, filename: str) -> str:
        ext = Path(filename or "").suffix.lower()
        mesh_id = hashlib.sha256(data).hexdigest()
        existing = self.status(mesh_id)
        if existing and existing.status != "error":
            if ext not in MESH_EXTS and not self.ref_path(mesh_id).exists():
                try:
                    self._save_ref(mesh_id, self._open_image(data))
                except InputRejected:
                    pass
            return mesh_id
        if ext in MESH_EXTS:
            self._import_mesh(mesh_id, data, ext)
        elif ext in IMAGE_EXTS or not ext:
            self._queue_image(mesh_id, data)
        else:
            raise InputRejected(f"Unsupported file type '{ext}'. Use a photo (.jpg/.png) or .glb/.obj.")
        return mesh_id

    def _import_mesh(self, mesh_id: str, data: bytes, ext: str) -> None:
        try:
            mesh = trimesh.load(io.BytesIO(data), file_type=ext[1:], force="mesh")
        except Exception as e:
            raise InputRejected(f"Couldn't read that 3D file: {e}") from e
        check_mesh(mesh)
        mesh.export(self.path(mesh_id))
        with self.lock:
            self.jobs[mesh_id] = Job("done")

    @staticmethod
    def _open_image(data: bytes) -> Image.Image:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
            img.load()
            return img
        except Exception as e:
            raise InputRejected("That file isn't an image we can read. Use a .jpg or .png.") from e

    def _queue_image(self, mesh_id: str, data: bytes) -> None:
        img = self._open_image(data)
        check_image(img)
        ok, why = self.model_factory().available()
        if not ok:
            raise ModelUnavailable(why + " You can still upload a .glb or .obj model.")
        with self.lock:
            self.jobs[mesh_id] = Job("queued")
        self.pool.submit(self._run, mesh_id, img)

    def _run(self, mesh_id: str, img: Image.Image) -> None:
        with self.lock:
            self.jobs[mesh_id] = Job("running")
        try:
            mesh = self.model_factory().image_to_mesh(img)
            check_mesh(mesh)
            self._save_ref(mesh_id, img)
            mesh.export(self.path(mesh_id))
            job = Job("done")
        except InputRejected as e:
            job = Job("error", str(e))
        except Exception as e:  # model crash, OOM, etc.
            job = Job("error", f"3D generation failed ({type(e).__name__}: {e}). Try another photo.")
        with self.lock:
            self.jobs[mesh_id] = job
