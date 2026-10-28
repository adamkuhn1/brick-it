"""HTTP API. See spec §3 for the data flow."""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import engine
from engine.export import to_bricklink_xml, to_csv, to_ldraw

from .checks import InputRejected
from .memory import available_memory
from .meshes import MeshStore
from .model import ModelUnavailable, get_model

ROOT = Path(__file__).resolve().parents[3]
MAX_UPLOAD = 30 * 1024 * 1024
MIN_SIZE, MAX_SIZE = 8, 48
ID_RE = re.compile(r"^[0-9a-f]{64}$")
BUILD_RE = re.compile(r"^([0-9a-f]{64})-(\d+)$")


class BuildRequest(BaseModel):
    mesh_id: str
    size_studs: int = Field(16, ge=MIN_SIZE, le=MAX_SIZE)


def create_app(cache_dir: Path | None = None, model_factory=get_model,
               free_memory=available_memory) -> FastAPI:
    cache = cache_dir or Path(os.environ.get("LEGO_CACHE", ROOT / ".cache"))
    store = MeshStore(cache, model_factory, free_memory)
    builds: OrderedDict[str, engine.Build] = OrderedDict()
    builds_lock = threading.Lock()
    app = FastAPI(title="lego")

    def _mesh_id(mesh_id: str) -> str:
        if not ID_RE.match(mesh_id):
            raise HTTPException(404, "Unknown mesh.")
        return mesh_id

    def _get_build(build_id: str) -> engine.Build:
        with builds_lock:
            if build_id in builds:
                builds.move_to_end(build_id)
                return builds[build_id]
        m = BUILD_RE.match(build_id)
        if not m:
            raise HTTPException(404, "Unknown build.")
        mesh_id, size = m.group(1), int(m.group(2))
        if not (MIN_SIZE <= size <= MAX_SIZE):
            raise HTTPException(404, "Unknown build.")
        job = store.status(mesh_id)
        if job is None or job.status != "done":
            raise HTTPException(404, "Mesh not ready.")
        b = engine.build(
            store.load(mesh_id), size, up="y", build_id=build_id, color_ref=store.color_ref(mesh_id)
        )
        with builds_lock:
            builds[build_id] = b
            while len(builds) > 32:
                builds.popitem(last=False)
        return b

    @app.get("/api/health")
    def health():
        ok, why = model_factory().available()
        return {"ok": True, "model_available": ok, "model_reason": why}

    @app.post("/api/meshes")
    async def create_mesh(file: UploadFile):
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "File is too large (max 30 MB).")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            mesh_id = store.submit(data, file.filename or "")
        except InputRejected as e:
            raise HTTPException(422, str(e))
        except ModelUnavailable as e:
            raise HTTPException(503, str(e))
        job = store.status(mesh_id)
        return {"mesh_id": mesh_id, "status": job.status, "error": job.error}

    @app.post("/api/meshes/{mesh_id}/cancel")
    def cancel_mesh(mesh_id: str):
        job = store.cancel(_mesh_id(mesh_id))
        if job is None:
            raise HTTPException(404, "Unknown mesh.")
        return {"mesh_id": mesh_id, "status": job.status, "error": job.error}

    @app.get("/api/meshes/{mesh_id}")
    def mesh_status(mesh_id: str):
        job = store.status(_mesh_id(mesh_id))
        if job is None:
            raise HTTPException(404, "Unknown mesh.")
        return {"mesh_id": mesh_id, "status": job.status, "error": job.error}

    @app.get("/api/meshes/{mesh_id}/file")
    def mesh_file(mesh_id: str):
        p = store.path(_mesh_id(mesh_id))
        if not p.exists():
            raise HTTPException(404, "Unknown mesh.")
        return FileResponse(p, media_type="model/gltf-binary")

    @app.post("/api/builds")
    def create_build(req: BuildRequest):
        _mesh_id(req.mesh_id)
        job = store.status(req.mesh_id)
        if job is None:
            raise HTTPException(404, "Unknown mesh.")
        if job.status != "done":
            raise HTTPException(409, "Mesh is not ready yet.")
        return _get_build(f"{req.mesh_id}-{req.size_studs}").to_json()

    @app.get("/api/builds/{build_id}/export/{fmt}")
    def export(build_id: str, fmt: str):
        b = _get_build(build_id)
        name = f"lego-{build_id[:8]}-{b.size_studs}"
        if fmt == "ldraw":
            body, mime, ext = to_ldraw(b.bricks, b.steps, name), "text/plain", "ldr"
        elif fmt == "bricklink":
            body, mime, ext = to_bricklink_xml(b.parts()), "application/xml", "xml"
        elif fmt == "csv":
            body, mime, ext = to_csv(b.parts()), "text/csv", "csv"
        else:
            raise HTTPException(404, "Unknown format. Use ldraw, bricklink or csv.")
        return Response(
            body, media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="{name}.{ext}"'},
        )

    dist = ROOT / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


app = create_app()
