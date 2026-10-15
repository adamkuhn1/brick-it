import csv
import io
import json
import re
import xml.etree.ElementTree as ET

import pytest
import trimesh

from engine import build
from engine.export import to_bricklink_xml, to_csv, to_ldraw
from engine.layout import PARTS
from engine.structure import adjacency, components
from meshes import arch, cube, l_shape, overhanging_t, sphere

SHAPES = {"cube": cube, "sphere": sphere, "l": l_shape, "arch": arch, "t": overhanging_t}


@pytest.fixture(scope="module", params=list(SHAPES))
def built(request):
    return build(SHAPES[request.param](), 16, up="z")


def test_single_component(built):
    assert len(components(built.bricks, (*built.dims[:2], built.dims[2]))) == 1, built.warnings


def test_every_step_buildable(built):
    adj = adjacency(built.bricks, built.dims)
    by_id = {b.id: b for b in built.bricks}
    placed = set()
    seen = []
    for s in built.steps:
        assert 1 <= len(s.brick_ids) <= 10
        for i in s.brick_ids:
            assert by_id[i].layer == 0 or any(n in placed for n in adj[i]), f"step {s.n} brick {i}"
        placed.update(s.brick_ids)
        seen.extend(s.brick_ids)
    assert sorted(seen) == [b.id for b in built.bricks]


def test_vocabulary(built):
    for b in built.bricks:
        assert PARTS[(b.w, b.l)] == b.part


def test_json_matches_contract(built):
    j = json.loads(json.dumps(built.to_json()))
    assert set(j) == {"id", "size_studs", "dims", "bricks", "steps", "parts", "stats", "warnings"}
    b = j["bricks"][0]
    assert set(b) == {"id", "part", "w", "l", "x", "y", "z", "rot", "color", "hidden"}
    assert set(b["color"]) == {"bricklink", "ldraw", "name", "hex"}
    assert j["stats"]["brick_count"] == len(j["bricks"])
    assert sum(p["qty"] for p in j["parts"]) == len(j["bricks"])
    assert j["stats"]["est_cost_usd"] > 0


def test_exports_parse(built):
    ldr = to_ldraw(built.bricks, built.steps)
    type1 = [l for l in ldr.splitlines() if l.startswith("1 ")]
    assert len(type1) == len(built.bricks)
    pat = re.compile(r"^1 \d+( -?\d+(\.\d+)?){12} \d+\.dat$")
    assert all(pat.match(l) for l in type1)
    parts = built.parts()
    root = ET.fromstring(to_bricklink_xml(parts))
    assert root.tag == "INVENTORY" and len(root) == len(parts)
    rows = list(csv.DictReader(io.StringIO(to_csv(parts))))
    assert sum(int(r["qty"]) for r in rows) == len(built.bricks)


def test_deterministic():
    a = build(sphere(), 12, up="z").to_json()
    b = build(sphere(), 12, up="z").to_json()
    assert a == b


def test_plain_mesh_no_colors_and_tiny_size():
    m = trimesh.creation.box(extents=[1, 1, 0.02])
    m.visual = trimesh.visual.ColorVisuals(m)
    out = build(m, 8, up="z")
    assert len(out.bricks) >= 1
    assert all(b.color == out.bricks[0].color for b in out.bricks)


def test_size_range_runs():
    for size in (8, 32):
        out = build(sphere(), size, up="z")
        assert out.dims[0] == size and out.steps
