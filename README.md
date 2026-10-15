# Brick it — photo to LEGO

Take a photo of an object and get:

1. a LEGO version of it built only from real, purchasable bricks,
2. a 3D model of the build you can rotate, and
3. a step-by-step instruction booklet in the style of a real LEGO booklet (print it or save it as a PDF).

It also exports a **BrickLink Wanted List** (for ordering parts), an **LDraw** `.ldr` model, and a parts list in CSV.

Everything runs on your machine. There are no cloud APIs. The only machine learning is [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (MIT), which turns the image into 3D, plus `rembg` for background removal. Every step after the mesh is ordinary algorithmic code.

| Model | Instructions |
|---|---|
| ![3D model view](docs/images/model.jpg) | ![Instruction booklet](docs/images/booklet.jpg) |

## Quick start

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), Node 20+, and about 4 GB of free disk space for the model.

```bash
./scripts/setup.sh        # Python deps, TripoSR + weights, web deps
./scripts/dev.sh          # API on :8000, web app on :5173
```

Open http://127.0.0.1:5173.

To skip the model install, run `./scripts/setup.sh --no-model`. You can still drop a `.glb`/`.obj` file in place of a photo.

**Photo tips:** use one object on a plain background, shot from slightly above at an angle. Very small, blank, or blurry photos are rejected before the model runs.

On an 8 GB M2 MacBook Air, TripoSR takes about 30 s per photo on the GPU (MPS) and peaks at roughly 2 GB of memory. The brick engine takes under 1 s even at 32 studs. Results are cached by image hash, so moving the size slider only reruns the engine.

## Tests

```bash
uv run pytest                 # engine + server (fast)
uv run pytest -m slow         # also runs real TripoSR on a sample photo
cd web && npm test && npm run typecheck
```

## Layout

```
engine/   pure-Python brick engine (numpy, trimesh, scipy). No web or model imports.
server/   FastAPI: TripoSR wrapper, mesh cache, input checks, build + export endpoints.
web/      Vite + React + three.js: capture, 3D viewer, instruction booklet.
docs/     design spec, implementation plan, screenshots.
```

Python and TypeScript share one contract, the **Build JSON** described in `docs/superpowers/specs/2026-10-15-lego-design.md` §3.

## How the engine works

`engine.build(mesh, size_studs)` runs seven stages. Each stage is a pure function in its own module with its own tests.

**1. Voxelize** (`voxelize.py`). A LEGO brick is 8 mm per stud and 9.6 mm tall, so each voxel is one stud wide and one brick (1.2 studs) tall. The mesh is rotated to Z-up and scaled so its widest horizontal extent equals the requested stud count, with Z squashed by 1.2. TripoSR meshes are not always watertight, so occupancy comes from dense surface sampling (about 40 samples per square stud) followed by 3D hole filling, not from inside/outside ray tests. Each surface sample carries a color interpolated from the vertex colors with barycentric weights, and those colors are averaged per voxel.

**2. Colorize** (`colors.py`). Voxel colors are matched to a palette of 24 common solid LEGO colors by ΔE in CIELAB, which follows perceived difference much better than RGB distance. Only visible voxels (those with an empty face neighbour) get a color. Interior voxels are marked *don't care*. Visible voxels the sampling missed take the color of the nearest sampled voxel, found with a distance transform. A smoothing pass then recolors isolated single-voxel specks.

**3. Hollow** (`hollow.py`). Above 12 studs, everything deeper than a 2-voxel shell is removed, except 2×2 support columns on a 6-stud lattice. That keeps every unsupported span inside the model at 4 studs or less. At 32 studs this saves roughly half the bricks.

**4. Layout** (`layout.py`). Each layer is tiled greedily with the 11-brick vocabulary (1×1 to 2×8). At the first uncovered cell in scan order, every brick that fits there is scored:
- area (bigger is better),
- the number of different bricks it would bridge in the layer below (good for connectivity),
- a penalty for each edge that lines up with a seam below (the "running bond" rule real walls follow),
- a small bonus for the layer's preferred direction, which alternates per layer.

A brick may only cover cells of one color. *Don't care* cells join any brick, and a brick made only of them becomes a hidden filler brick in light bluish gray.

**5. Validate and repair** (`structure.py`). Bricks become nodes in a graph, with edges between bricks in adjacent layers weighted by the number of shared studs. Two problems get fixed:
- **Disconnected components.** For each pair of horizontally touching cells in different components, the engine tries placing a brick in the layer above or below that spans the joint. The displaced bricks around it are re-tiled with the same greedy layout. A change is kept only if the component count goes down. If no joint exists (a truly floating part), a 1×1 support column is dropped until it lands on something.
- **Weak joints.** Tarjan's bridge-finding algorithm finds cut edges with a 1-stud overlap whose removal would split off more than one brick. The engine tries larger bricks over the joint and keeps one only if the build stays connected and the number of weak joints drops.

Anything left unfixed becomes a warning, along with a center-of-mass check against the convex hull of the base.

**6. Plan steps** (`steps.py`). Steps go bottom-up. A brick is placeable once it would attach to something already built: the ground, a placed brick below it, or a placed brick above it (for parts hanging under an overhang). Within a layer, each step floods outward from a seed brick to collect a compact group of 3–6 touching bricks. Steps made only of hidden filler are batched up to 10 bricks.

**7. Export** (`export.py`). LDraw output uses LDraw color IDs and LDU coordinates (20 LDU per stud, 24 per brick, −Y up, with parts rotated about Y when their long side runs along engine-y), plus a `0 STEP` after each step so LDraw viewers replay the instructions. The BrickLink Wanted List XML and the CSV come from the same parts tally. The cost is a rough estimate from a static table of average used-part prices.

### The instruction booklet

Booklet images are rendered in the browser with three.js on an offscreen canvas: an orthographic camera at a fixed booklet-style angle, toon shading with a 3-step ramp, and black outlines built from edge geometry. The whole model is a set of instanced meshes sorted by placement order, so drawing step *n* means showing the first *k* instances and a prefix of the outline buffer, with no scene rebuild. Pages are A4-sized HTML with print CSS, so **Print → Save as PDF** produces the booklet.

### Known limits

- One photo gives one viewpoint. TripoSR has to invent the back of the object, and thin parts such as chair legs often come out fragmented, which the repair stage then patches with support bricks and warnings.
- Bricks only: no plates or slopes yet. The grid's Z axis is already in plate units so plates can be added later.
- The stability check only warns; it does not rebalance the build.

## Notes on the model setup

- TripoSR imports `torchmcubes`, which needs a C++/CUDA build. The server injects a tiny shim backed by `PyMCubes`, so no compiler is needed.
- Recent `rembg` releases default to a 1 GB background-removal model with a non-commercial license. The server pins the original `u2net` model (MIT) instead.
- `transformers` is pinned below 4.50 for TripoSR's ViT tokenizer.
