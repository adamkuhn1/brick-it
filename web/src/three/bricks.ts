import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Brick, Build, SlopeDir } from "../types";
import { sizeX, sizeY } from "../types";

// World units: 1 stud pitch = 1. A brick is 9.6/8 = 1.2 tall; a plate (z unit) is 0.4.
export const BRICK_H = 1.2;
export const PLATE_H = 0.4;
const GAP = 0.025; // visible seam between neighbouring bricks
const STUD_R = 0.3;
const STUD_H = 0.21;

// Slope 45 parts: the back row is full height with studs; the front row slopes down to a
// low lip at the front, as in the real parts (LDraw 3040/3039: a 4 LDU lip on a 24 LDU brick).
const SLOPE_LIP = BRICK_H * (4 / 24);
// Turn a wedge built facing world +X toward each engine direction (engine y = world -Z).
const SLOPE_YAW: Record<SlopeDir, number> = { "+x": 0, "-x": Math.PI, "+y": Math.PI / 2, "-y": -Math.PI / 2 };

const faceCache = new Map<string, THREE.BufferGeometry>();
const edgeCache = new Map<string, Float32Array>();

const key = (sx: number, sy: number, slope: SlopeDir | null = null) => `${sx}x${sy}${slope ?? ""}`;

/** Stud centres (x, z) on top of an sx × sy footprint, centred on the box. */
function studs(sx: number, sy: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < sx; i++) for (let j = 0; j < sy; j++) out.push([i - sx / 2 + 0.5, j - sy / 2 + 0.5]);
  return out;
}

/**
 * Body and stud positions for a part, centred on its footprint. A slope is a wedge 2 deep
 * and `across` wide, built facing +X with studs on the back row, then turned to face `slope`.
 */
function shape(sx: number, sy: number, slope: SlopeDir | null) {
  if (!slope) return { body: new THREE.BoxGeometry(sx - GAP, BRICK_H - GAP, sy - GAP), studs: studs(sx, sy), yaw: 0 };
  const across = slope === "+x" || slope === "-x" ? sy : sx;
  const d = 1 - GAP / 2, h = (BRICK_H - GAP) / 2;
  const profile = new THREE.Shape([
    new THREE.Vector2(-d, -h), new THREE.Vector2(d, -h), new THREE.Vector2(d, -h + SLOPE_LIP),
    new THREE.Vector2(0, h), new THREE.Vector2(-d, h),
  ]);
  const body = new THREE.ExtrudeGeometry(profile, { depth: across - GAP, bevelEnabled: false });
  body.translate(0, 0, -(across - GAP) / 2);
  const back = studs(1, across).map(([, z]): [number, number] => [-0.5, z]);
  return { body, studs: back, yaw: SLOPE_YAW[slope] };
}

/** Box (or slope wedge) plus studs for an sx × sy part, centred on its footprint. */
export function brickGeometry(sx: number, sy: number, slope: SlopeDir | null = null): THREE.BufferGeometry {
  const k = key(sx, sy, slope);
  let g = faceCache.get(k);
  if (!g) {
    const { body, studs: at, yaw } = shape(sx, sy, slope);
    const parts: THREE.BufferGeometry[] = [body];
    for (const [x, z] of at) {
      const s = new THREE.CylinderGeometry(STUD_R, STUD_R, STUD_H, 20);
      s.translate(x, BRICK_H / 2 + STUD_H / 2 - GAP / 2, z);
      parts.push(s);
    }
    g = mergeGeometries(parts.map((p) => p.toNonIndexed()))!;
    g.rotateY(yaw);
    g.computeVertexNormals();
    faceCache.set(k, g);
  }
  return g;
}

/** Line-segment vertices (body outline + stud outlines) for an sx × sy part. */
export function brickEdges(sx: number, sy: number, slope: SlopeDir | null = null): Float32Array {
  const k = key(sx, sy, slope);
  let e = edgeCache.get(k);
  if (!e) {
    const { body, studs: at, yaw } = shape(sx, sy, slope);
    const lines: THREE.BufferGeometry[] = [new THREE.EdgesGeometry(body)];
    for (const [x, z] of at) {
      const s = new THREE.EdgesGeometry(new THREE.CylinderGeometry(STUD_R, STUD_R, STUD_H, 16), 30);
      s.translate(x, BRICK_H / 2 + STUD_H / 2 - GAP / 2, z);
      lines.push(s);
    }
    const merged = mergeGeometries(lines)!;
    merged.rotateY(yaw);
    e = new Float32Array(merged.attributes.position.array as Float32Array);
    edgeCache.set(k, e);
  }
  return e;
}

/**
 * Centre of a brick in world space. Engine axes are x, y (studs) and z (up, plates); world
 * is Y-up, so engine y maps to world -Z to keep the model un-mirrored.
 */
export function brickCenter(b: Brick, dims: Build["dims"]): THREE.Vector3 {
  return new THREE.Vector3(
    b.x + sizeX(b) / 2 - dims.x / 2,
    b.z * PLATE_H + BRICK_H / 2,
    -(b.y + sizeY(b) / 2 - dims.y / 2),
  );
}

export interface BrickScene {
  group: THREE.Group;
  /** Show only the first n bricks of `order`. */
  setVisible(n: number): void;
  bounds: THREE.Box3;
  dispose(): void;
}

/** Instanced scene with bricks laid out in `order` (ids), so prefixes can be shown cheaply. */
export function buildBrickScene(
  build: Build,
  order: number[],
  material: THREE.Material,
  withEdges = false,
): BrickScene {
  const byId = new Map(build.bricks.map((b) => [b.id, b]));
  const group = new THREE.Group();
  const bySize = new Map<string, { sx: number; sy: number; slope: SlopeDir | null; bricks: Brick[]; rank: number[] }>();
  order.forEach((id, rank) => {
    const b = byId.get(id)!;
    const sx = sizeX(b);
    const sy = sizeY(b);
    const slope = b.slope ?? null;
    const k = key(sx, sy, slope);
    if (!bySize.has(k)) bySize.set(k, { sx, sy, slope, bricks: [], rank: [] });
    const entry = bySize.get(k)!;
    entry.bricks.push(b);
    entry.rank.push(rank);
  });

  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  const meshes: { mesh: THREE.InstancedMesh; rank: number[] }[] = [];
  for (const { sx, sy, slope, bricks, rank } of bySize.values()) {
    const mesh = new THREE.InstancedMesh(brickGeometry(sx, sy, slope), material, bricks.length);
    bricks.forEach((b, i) => {
      m.makeTranslation(brickCenter(b, build.dims));
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, c.set(b.color.hex));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    group.add(mesh);
    meshes.push({ mesh, rank });
  }

  let lines: THREE.LineSegments | null = null;
  const edgeOffsets: number[] = [0];
  if (withEdges) {
    const chunks: Float32Array[] = [];
    let total = 0;
    for (const id of order) {
      const b = byId.get(id)!;
      const base = brickEdges(sizeX(b), sizeY(b), b.slope ?? null);
      const p = brickCenter(b, build.dims);
      const out = new Float32Array(base.length);
      for (let i = 0; i < base.length; i += 3) {
        out[i] = base[i] + p.x;
        out[i + 1] = base[i + 1] + p.y;
        out[i + 2] = base[i + 2] + p.z;
      }
      chunks.push(out);
      total += out.length / 3;
      edgeOffsets.push(total);
    }
    const all = new Float32Array(total * 3);
    let o = 0;
    for (const ch of chunks) {
      all.set(ch, o);
      o += ch.length;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(all, 3));
    lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x14232e }));
    group.add(lines);
  }

  const bounds = new THREE.Box3();
  for (const b of build.bricks) {
    const p = brickCenter(b, build.dims);
    bounds.expandByPoint(new THREE.Vector3(p.x - sizeX(b) / 2, p.y - BRICK_H / 2, p.z - sizeY(b) / 2));
    bounds.expandByPoint(new THREE.Vector3(p.x + sizeX(b) / 2, p.y + BRICK_H / 2 + STUD_H, p.z + sizeY(b) / 2));
  }

  return {
    group,
    bounds,
    setVisible(n: number) {
      for (const { mesh, rank } of meshes) {
        // ranks are ascending within each size bucket
        let count = 0;
        while (count < rank.length && rank[count] < n) count++;
        mesh.count = count;
      }
      if (lines) lines.geometry.setDrawRange(0, edgeOffsets[Math.min(n, order.length)]);
    },
    dispose() {
      for (const { mesh } of meshes) mesh.dispose();
      lines?.geometry.dispose();
      (lines?.material as THREE.Material | undefined)?.dispose();
    },
  };
}

/** Placement order: bricks in step order. */
export function stepOrder(build: Build): number[] {
  return build.steps.flatMap((s) => s.brick_ids);
}
