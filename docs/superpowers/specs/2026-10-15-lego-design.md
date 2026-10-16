# Lego — Photo to LEGO Build (Design Spec)

**Date:** 2026-10-15
**Status:** Draft, awaiting review
**Delivery:** MVP built in one continuous pass (all milestones below, in order).

## 1. Goal

A user uploads or takes a photo of an object. The app generates:

1. a 3D LEGO version of the object built from real, purchasable bricks,
2. a rotatable 3D model of that build, and
3. step-by-step instructions that look like a real LEGO instruction booklet.

Runs entirely on the user's machine (clone repo, run setup, open in browser). No cloud AI APIs. The only ML is a free, local image-to-3D model (TripoSR) plus its bundled local background remover. Everything after the mesh is hand-written algorithms.

**Audience/success:** a working, demo-able local app plus a strong technical write-up. Works well on single, clearly photographed objects; not required to handle every photo.

## 2. Non-goals (MVP)

- Plates or other non-brick parts beyond the two 45° slopes (3040, 3039). The grid is designed to allow plates later.
- 3D "build mode" that animates steps inside the viewer.
- Desktop app packaging, browser-only model inference.
- Buy button / cart integration (MVP only exports the BrickLink Wanted List file).
- Automatic rebalancing of builds that would tip over (MVP warns only).
- Multi-photo capture, accounts, cloud hosting.

## 3. Architecture

```
lego/
├── engine/   Python package — the brick engine. Pure logic; no web or model imports.
├── server/   Python (FastAPI) — runs TripoSR, caches meshes, calls the engine.
└── web/      TypeScript (Vite + React + three.js) — all UI.
```

- Python managed with `uv` (one workspace, `engine` and `server` as packages). Web uses `npm`.
- One setup command (`./scripts/setup.sh`) installs Python deps, downloads model weights, installs web deps. One run command (`./scripts/dev.sh`) starts server + web.

### Data flow

1. `POST /api/meshes` (photo) → background removal → TripoSR → mesh (vertex colors) saved to `.cache/meshes/<sha256-of-image>.glb`. Returns `mesh_id`. Same photo = cache hit.
2. `POST /api/builds` `{mesh_id, size_studs}` → engine runs on cached mesh → returns **Build JSON**. Slider changes only re-run this call.
3. `GET /api/builds/<id>/export/{ldraw|bricklink|csv}` → file download.
4. Dev/fallback: `POST /api/meshes` also accepts a `.glb`/`.obj` upload, skipping the model. Used for tests and if TripoSR is unavailable.

### Build JSON (the single contract between Python and TypeScript)

```json
{
  "id": "…",
  "size_studs": 16,
  "dims": { "x": 16, "y": 12, "z": 14 },
  "bricks": [
    { "id": 0, "part": "3001", "w": 2, "l": 4, "x": 0, "y": 0, "z": 0,
      "rot": 0, "color": { "bricklink": 5, "ldraw": 4, "name": "Red", "hex": "#C91A09" },
      "hidden": false, "slope": null }
  ],
  "steps": [ { "n": 1, "brick_ids": [0, 1, 2] } ],
  "parts": [ { "part": "3001", "color_bricklink": 5, "qty": 12 } ],
  "stats": { "brick_count": 214, "step_count": 41, "est_cost_usd": 21.40 },
  "warnings": ["Center of mass is outside the base; model may tip."]
}
```

Coordinates: `x`,`y` in studs, `z` in **plate units** (1 brick = 3 plates; MVP always uses multiples of 3). `rot` is 0 or 90. `slope` is `null` for a plain brick; for a Slope 45 part (3040 2×1, 3039 2×2) it is the engine direction its low side faces (`"+x"`, `"-x"`, `"+y"`, `"-y"`), and the footprint is 2 studs deep along that axis. Only the back row has studs.

## 4. Engine pipeline

Each stage is its own module with a pure function interface, tested in isolation.

1. **Voxelize** (`voxelize.py`): scale mesh so its largest horizontal extent = `size_studs`; voxel = 1 stud × 1 brick (accounts for the 8 mm : 9.6 mm stud-to-height ratio). Fill interior with trimesh. Output: 3D boolean occupancy grid + per-surface-voxel RGB sampled from mesh vertex colors. Mesh is re-oriented so its lowest point sits on z=0.
2. **Colorize** (`colors.py`): map each surface voxel's RGB to the nearest color in a curated palette of ~20 common solid LEGO colors, using CIELAB ΔE. Palette table stores name, hex, BrickLink ID, LDraw ID. Light smoothing pass removes isolated single-voxel color specks.
3. **Hollow** (`hollow.py`): keep voxels within 1 voxel of the exterior (shell). Remove interior voxels except support columns: 2×2 columns placed under shell voxels that sit over a void wider than 4 studs. Builds ≤ 12 studs stay solid. Interior voxels are marked "color: don't care."
4. **Layout** (`layout.py`): per layer, greedily tile with the largest fitting brick from the vocabulary `{1×1,1×2,1×3,1×4,1×6,1×8,2×2,2×3,2×4,2×6,2×8}` (parts 3005, 3004, 3622, 3010, 3009, 3008, 3003, 3002, 3001, 2456, 3007). A brick covers cells of a single color; "don't care" cells merge with anything and the brick takes the neighbouring visible color or a default (light bluish gray). Layers alternate preferred orientation; candidate placements are scored to penalize seams that align with seams in the layer below.
5. **Validate & repair** (`structure.py`): build a connectivity graph (bricks connected if vertically adjacent with ≥1 overlapping stud). Fix (a) disconnected components and (b) single-stud bridges (cut edges whose removal splits off >1 brick) by re-tiling the local region with bricks that span the joint, falling back to adding hidden support bricks. Repeat until connected or an iteration cap; remaining problems become `warnings`. Center of mass vs. base footprint check → warning only.
6. **Plan steps** (`steps.py`): order bottom layer → top; every brick must rest on already-placed bricks or the ground. Within a layer, flood outward from a seed brick, grouping 3–6 adjacent bricks per step. Steps containing only hidden bricks are batched up to ~10 bricks.
7. **Export** (`export.py`): LDraw `.ldr` (one type-1 line per brick using LDraw color IDs and LDU coordinates), BrickLink Wanted List XML, parts list CSV. Cost estimate uses a static per-part average price table (labelled "rough estimate").

Top-level entry: `engine.build(mesh, size_studs) -> Build`.

## 5. Model (server)

- **TripoSR** (MIT license) with its bundled `rembg` background removal, running on PyTorch MPS on Apple Silicon, CPU fallback.
- Memory tuned for an 8 GB M2: reduced marching-cubes resolution and chunk size. Low resolution is fine; LEGO grids are ≤ 48 studs.
- Server checks input before inference: rejects tiny (<256 px), near-blank, or very blurry (low Laplacian variance) images with a clear message. After inference, rejects meshes that are empty or nearly flat (thinnest bounding box axis <10% of largest) with a "try another photo" hint.
- Inference runs in a worker thread; client polls `GET /api/meshes/<id>` for status (`queued | running | done | error`).

## 6. Web app

Single page, three states:

1. **Capture:** drag-and-drop or file picker, plus webcam capture (`getUserMedia`). Short tip line: one object, plain background, angled view.
2. **Generating:** progress state while polling mesh status.
3. **Result:**
   - 3D viewer (react-three-fiber): orbit controls, bricks rendered as boxes with studs, instanced for performance.
   - Size slider (8–48 studs, default 16) showing brick count, step count, est. cost; debounced re-build call.
   - Tabs: **Model** | **Instructions**.
   - Download buttons: LDraw, BrickLink XML, parts CSV.

### Instruction booklet

- Each step rendered offscreen by three.js from one fixed isometric camera, flat (toon) shading + black outline, on LEGO's light blue-gray page color. Model shown cumulatively; that step's bricks are simply present.
- Step layout: large step number; top-left parts callout with each new part rendered in 3D and a "2x" count.
- Pages: cover (finished model render), 2–4 steps per page, parts inventory at the back.
- Pages sized to A4/Letter with print CSS, so "Print → Save as PDF" gives a booklet.

## 7. Error handling summary

| Situation | Behaviour |
|---|---|
| Bad photo (tiny/blank/blurry) | Rejected before inference, message shown on Capture screen |
| Model fails / flat mesh | Error state with photo tips; user can retry |
| Model not installed / unavailable | Capture screen still accepts `.glb/.obj` upload |
| Unrepairable structure | Build returned with `warnings`, shown above viewer |
| > 2,000 bricks | Warning beside slider |

## 8. Testing

- **Engine (pytest):** synthetic meshes — cube, sphere, L-shape, arch, overhanging T. Assertions: single connected component, no floating bricks, seam-stagger ratio above threshold, every step buildable from prior steps, each brick in vocabulary, exports parse (LDraw line format, XML well-formed).
- **Server:** end-to-end test with a stored sample `.glb` (model-skipping path) through `/api/builds` and exports; one optional slow test running TripoSR on a sample photo.
- **Web:** typecheck + a smoke test that viewer and booklet render from a fixture Build JSON.

## 9. Milestones (single pass)

0. **Model check:** TripoSR produces a mesh on the M2 within 8 GB. If it can't, continue with the `.glb` upload path and revisit model choice.
1. **Engine core:** voxelize → colorize → hollow → layout → validate/repair, tests passing.
2. **Steps & exports.**
3. **Server API** (meshes, builds, exports, caching, input checks).
4. **Web:** capture, generating, result viewer + slider + downloads.
5. **Booklet:** LEGO-style instruction pages + print-to-PDF.
6. **README** with setup, screenshots, and a short technical write-up of the engine.

## 10. Future work

Plates (z already in plate units) → slopes; 3D build mode reusing `steps`; buy button opening BrickLink with the Wanted List; desktop packaging (ONNX export of TripoSR as a prerequisite spike); automatic stability optimization (Luo et al., "Legolization", 2015); single-photo depth model in browser if WebGPU matures.
