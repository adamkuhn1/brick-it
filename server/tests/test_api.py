import io
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from server.app import create_app
from server.memory import GB

FIXTURE = Path(__file__).parents[2] / "engine" / "tests" / "fixtures" / "chair.glb"


class FakeModel:
    def __init__(self, ok=True, flat=False, hang=False):
        self.ok, self.flat, self.hang = ok, flat, hang

    def available(self):
        return (True, "") if self.ok else (False, "TripoSR is not installed.")

    def foreground(self, img):
        from server.model import foreground_pixels
        return foreground_pixels(img)

    def image_to_mesh(self, img, cancel=None):
        if self.hang:  # like a run stuck in swap: only a cancel ends it
            from server.isolated import Stopped
            assert cancel is not None and cancel.wait(10), "never cancelled"
            raise Stopped("cancelled")
        if self.flat:
            return trimesh.creation.box(extents=[1, 1, 0.01])
        return trimesh.creation.icosphere(subdivisions=2)


def client(tmp_path, free=8 * GB, **kw):
    free_memory = free if callable(free) else lambda: free
    return TestClient(create_app(tmp_path, model_factory=lambda: FakeModel(**kw),
                                 free_memory=free_memory))


def photo(size=512, noise=True):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (size, size, 3), dtype=np.uint8) if noise else np.full((size, size, 3), 128, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


def wait_done(c, mesh_id):
    for _ in range(100):
        s = c.get(f"/api/meshes/{mesh_id}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.05)
    raise AssertionError("timeout")


def test_glb_upload_build_and_exports(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/meshes", files={"file": ("chair.glb", FIXTURE.read_bytes())})
    assert r.status_code == 200, r.text
    mesh_id = r.json()["mesh_id"]
    assert r.json()["status"] == "done"

    r = c.post("/api/builds", json={"mesh_id": mesh_id, "size_studs": 12})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["stats"]["brick_count"] > 0 and b["dims"]["x"] <= 12

    ldr = c.get(f"/api/builds/{b['id']}/export/ldraw")
    assert ldr.status_code == 200 and ldr.text.count("\n1 ") == len(b["bricks"])
    xml = c.get(f"/api/builds/{b['id']}/export/bricklink")
    assert ET.fromstring(xml.text).tag == "INVENTORY"
    csv = c.get(f"/api/builds/{b['id']}/export/csv")
    assert csv.text.startswith("part,")
    assert "attachment" in csv.headers["content-disposition"]

    # Same file -> same id (cache hit). Fresh app -> export still works from disk cache.
    again = c.post("/api/meshes", files={"file": ("chair.glb", FIXTURE.read_bytes())}).json()
    assert again["mesh_id"] == mesh_id
    c2 = client(tmp_path)
    assert c2.get(f"/api/builds/{b['id']}/export/csv").status_code == 200


def test_build_is_deterministic(tmp_path):
    c = client(tmp_path)
    mesh_id = c.post("/api/meshes", files={"file": ("chair.glb", FIXTURE.read_bytes())}).json()["mesh_id"]
    a = c.post("/api/builds", json={"mesh_id": mesh_id, "size_studs": 10}).json()
    c2 = client(tmp_path)
    b = c2.post("/api/builds", json={"mesh_id": mesh_id, "size_studs": 10}).json()
    assert a == b


def test_photo_runs_through_model(tmp_path):
    c = client(tmp_path)
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    assert r.status_code == 200, r.text
    s = wait_done(c, r.json()["mesh_id"])
    assert s["status"] == "done"


def test_flat_mesh_from_model_reports_error(tmp_path):
    c = client(tmp_path, flat=True)
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    s = wait_done(c, r.json()["mesh_id"])
    assert s["status"] == "error" and "flat" in s["error"]


@pytest.mark.parametrize("data,msg", [(photo(100), "too small"), (photo(noise=False), "blank")])
def test_bad_photos_rejected(tmp_path, data, msg):
    r = client(tmp_path).post("/api/meshes", files={"file": ("p.png", data)})
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_blurry_photo_rejected(tmp_path):
    x = np.linspace(0, 255, 512)
    a = np.tile(x, (512, 1)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.stack([a] * 3, -1)).save(buf, "PNG")
    r = client(tmp_path).post("/api/meshes", files={"file": ("p.png", buf.getvalue())})
    assert r.status_code == 422 and "blurry" in r.json()["detail"]


def test_model_unavailable_still_accepts_glb(tmp_path):
    c = client(tmp_path, ok=False)
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    assert r.status_code == 503 and ".glb" in r.json()["detail"]
    assert c.get("/api/health").json()["model_available"] is False
    r = c.post("/api/meshes", files={"file": ("chair.glb", FIXTURE.read_bytes())})
    assert r.status_code == 200


def test_bad_requests(tmp_path):
    c = client(tmp_path)
    assert c.post("/api/meshes", files={"file": ("x.txt", b"hello")}).status_code == 422
    assert c.post("/api/meshes", files={"file": ("x.glb", b"garbage")}).status_code == 422
    assert c.post("/api/builds", json={"mesh_id": "0" * 64, "size_studs": 16}).status_code == 404
    assert c.post("/api/builds", json={"mesh_id": "0" * 64, "size_studs": 48}).status_code == 404
    assert c.post("/api/builds", json={"mesh_id": "0" * 64, "size_studs": 49}).status_code == 422
    assert c.post("/api/builds", json={"mesh_id": "0" * 64, "size_studs": 99}).status_code == 422
    assert c.get("/api/meshes/../../etc").status_code == 404
    assert c.get("/api/builds/nope/export/csv").status_code == 404


@pytest.mark.slow
def test_real_triposr(tmp_path):
    from server.model import TRIPOSR_DIR

    c = TestClient(create_app(tmp_path))
    if not c.get("/api/health").json()["model_available"]:
        pytest.skip("TripoSR not installed")
    data = (TRIPOSR_DIR / "examples" / "chair.png").read_bytes()
    r = c.post("/api/meshes", files={"file": ("chair.png", data)})
    if r.status_code == 503 and "low on memory" in r.text:
        pytest.skip(r.json()["detail"])
    mesh_id = r.json()["mesh_id"]
    for _ in range(600):
        s = c.get(f"/api/meshes/{mesh_id}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert s["status"] == "done", s


def test_photo_colours_saved_for_matching(tmp_path):
    c = client(tmp_path)
    mesh_id = c.post("/api/meshes", files={"file": ("p.png", photo())}).json()["mesh_id"]
    wait_done(c, mesh_id)
    assert (tmp_path / "meshes" / f"{mesh_id}.colors.npy").exists()
    assert c.post("/api/builds", json={"mesh_id": mesh_id, "size_studs": 8}).status_code == 200


def test_transparent_png_keeps_its_alpha():
    from server.model import foreground_pixels, has_alpha
    a = np.zeros((300, 300, 4), np.uint8)
    a[100:200, 100:200] = (90, 150, 200, 255)  # blue subject; transparent black elsewhere
    img = Image.fromarray(a, "RGBA")
    assert has_alpha(img)
    px = foreground_pixels(img)
    assert (px.mean(axis=0) > [80, 140, 190]).all()  # no black background leaking in


def test_low_memory_refuses_photo_but_not_glb(tmp_path):
    c = client(tmp_path, free=int(1.8 * GB))
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    assert r.status_code == 503
    assert "low on memory (1.8 GB free" in r.json()["detail"]
    r = c.post("/api/meshes", files={"file": ("chair.glb", FIXTURE.read_bytes())})
    assert r.status_code == 200 and r.json()["status"] == "done"


def test_memory_checked_again_when_run_starts(tmp_path):
    readings = iter([8 * GB, int(1.0 * GB)])  # fine at upload, low by the time it runs
    c = client(tmp_path, free=lambda: next(readings, int(1.0 * GB)))
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    assert r.status_code == 200
    s = wait_done(c, r.json()["mesh_id"])
    assert s["status"] == "error" and "low on memory" in s["error"]


def test_unknown_free_memory_does_not_block(tmp_path):
    c = client(tmp_path, free=None)
    r = c.post("/api/meshes", files={"file": ("p.png", photo())})
    assert wait_done(c, r.json()["mesh_id"])["status"] == "done"


def test_cancel_running_photo(tmp_path):
    c = client(tmp_path, hang=True)
    mesh_id = c.post("/api/meshes", files={"file": ("p.png", photo())}).json()["mesh_id"]
    for _ in range(100):
        if c.get(f"/api/meshes/{mesh_id}").json()["status"] == "running":
            break
        time.sleep(0.02)
    r = c.post(f"/api/meshes/{mesh_id}/cancel")
    assert r.status_code == 200 and r.json() == {"mesh_id": mesh_id, "status": "error",
                                                "error": "Cancelled."}
    time.sleep(0.2)  # the run notices and exits; its result must not overwrite the cancel
    assert c.get(f"/api/meshes/{mesh_id}").json()["error"] == "Cancelled."
    # The same photo can be tried again after a cancel.
    again = c.post("/api/meshes", files={"file": ("p.png", photo())}).json()
    assert again["status"] in ("queued", "running")
    c.post(f"/api/meshes/{mesh_id}/cancel")


def test_cancel_unknown_mesh(tmp_path):
    assert client(tmp_path).post(f"/api/meshes/{'0' * 64}/cancel").status_code == 404
    assert client(tmp_path).post("/api/meshes/nope/cancel").status_code == 404


def test_cancel_while_queued_then_resubmit_runs_once(tmp_path):
    from server.meshes import MeshStore

    calls = []

    class Counting(FakeModel):
        def image_to_mesh(self, img, cancel=None):
            calls.append(img.size)
            return super().image_to_mesh(img, cancel)

    store = MeshStore(tmp_path, lambda: Counting(), lambda: 8 * GB)
    gate = __import__("threading").Event()
    store.pool.submit(gate.wait)  # keep the single worker busy so the photo stays queued
    mesh_id = store.submit(photo(), "p.png")
    assert store.status(mesh_id).status == "queued"
    store.cancel(mesh_id)
    assert store.submit(photo(), "p.png") == mesh_id  # resubmit while the old run is queued
    gate.set()
    store.pool.shutdown(wait=True)
    assert store.status(mesh_id).status == "done"
    assert len(calls) == 1
