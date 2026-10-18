import * as THREE from "three";
import type { Build } from "../types";
import { BRICK_H, brickCenter, buildBrickScene, stepOrder, type BrickScene } from "./bricks";
import { CORNER_PART, sizeX, sizeY } from "../types";

/** Renders instruction-booklet images: flat toon shading, black outlines, one fixed iso camera. */
export interface BookletRenderer {
  cover(): Promise<string>;
  step(n: number): Promise<string>; // 1-based step number, cumulative model
  part(part: string, hex: string, w: number, l: number, slope?: boolean, inverted?: boolean): Promise<string>;
  dispose(): void;
}

const VIEW_DIR = new THREE.Vector3(-1, 1.05, 1.25).normalize(); // from front-left, above

function toonMaterial(): THREE.MeshToonMaterial {
  const data = new Uint8Array([90, 170, 255]);
  const ramp = new THREE.DataTexture(data, 3, 1, THREE.RedFormat);
  ramp.minFilter = ramp.magFilter = THREE.NearestFilter;
  ramp.needsUpdate = true;
  return new THREE.MeshToonMaterial({ color: 0xffffff, gradientMap: ramp });
}

function lights(scene: THREE.Scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(-2, 4, 3);
  scene.add(sun);
}

/** Orthographic camera along VIEW_DIR that frames `box` with a small margin. */
function fitCamera(box: THREE.Box3, aspect: number, margin = 1.08): THREE.OrthographicCamera {
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length();
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, radius * 4 + 10);
  cam.position.copy(center).addScaledVector(VIEW_DIR, radius * 2 + 5);
  cam.lookAt(center);
  cam.updateMatrixWorld();
  const inv = cam.matrixWorldInverse;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    const p = new THREE.Vector3(
      i & 1 ? box.max.x : box.min.x,
      i & 2 ? box.max.y : box.min.y,
      i & 4 ? box.max.z : box.min.z,
    ).applyMatrix4(inv);
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  let w = (maxX - minX) * margin;
  let h = (maxY - minY) * margin;
  if (w / h > aspect) h = w / aspect;
  else w = h * aspect;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  cam.left = cx - w / 2; cam.right = cx + w / 2;
  cam.top = cy + h / 2; cam.bottom = cy - h / 2;
  cam.updateProjectionMatrix();
  return cam;
}

function toUrl(canvas: HTMLCanvasElement): Promise<string> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(URL.createObjectURL(b)) : reject(new Error("render failed"))), "image/png"),
  );
}

export function createBookletRenderer(build: Build, width = 900, height = 680): BookletRenderer {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  const material = toonMaterial();

  const order = stepOrder(build);
  const cumulative: number[] = [];
  let acc = 0;
  for (const s of build.steps) cumulative.push((acc += s.brick_ids.length));

  const scene = new THREE.Scene();
  lights(scene);
  const model: BrickScene = buildBrickScene(build, order, material, true);
  scene.add(model.group);
  const camera = fitCamera(model.bounds, width / height);

  // Like a real booklet, frame the part being built rather than the finished model, but
  // never zoom past MIN_FRAME of the full model so the scale doesn't jump around.
  const MIN_FRAME = 0.45;
  const byId = new Map(build.bricks.map((b) => [b.id, b]));
  const fullSize = model.bounds.getSize(new THREE.Vector3());
  const stepCameras: THREE.OrthographicCamera[] = [];
  const sofar = new THREE.Box3();
  let k = 0;
  for (const s of build.steps) {
    for (; k < cumulative[s.n - 1]; k++) {
      const b = byId.get(order[k])!;
      const p = brickCenter(b, build.dims);
      sofar.expandByPoint(new THREE.Vector3(p.x - sizeX(b) / 2, p.y - BRICK_H / 2, p.z - sizeY(b) / 2));
      sofar.expandByPoint(new THREE.Vector3(p.x + sizeX(b) / 2, p.y + BRICK_H / 2 + 0.2, p.z + sizeY(b) / 2));
    }
    const size = sofar.getSize(new THREE.Vector3()).max(fullSize.clone().multiplyScalar(MIN_FRAME));
    const frame = new THREE.Box3().setFromCenterAndSize(sofar.getCenter(new THREE.Vector3()), size);
    stepCameras.push(fitCamera(frame, width / height, 1.15));
  }

  const partScene = new THREE.Scene();
  lights(partScene);

  async function draw(s: THREE.Scene, cam: THREE.Camera, w: number, h: number) {
    renderer.setSize(w, h, false);
    renderer.render(s, cam);
    return toUrl(renderer.domElement);
  }

  return {
    async cover() {
      model.setVisible(order.length);
      return draw(scene, camera, width, height);
    },
    async step(n: number) {
      model.setVisible(cumulative[n - 1]);
      return draw(scene, stepCameras[n - 1], width, height);
    },
    async part(part, hex, w, l, slope = false, inverted = false) {
      const fake: Build = {
        id: "part", size_studs: l, dims: { x: l, y: w, z: 1 },
        bricks: [{ id: 0, part, w, l, x: 0, y: 0, z: 0, rot: 90, color: { hex, name: "", bricklink: 0, ldraw: 0 }, hidden: false,
          // Laid long side along x, so a slope (2 deep) faces -x: toward the booklet camera,
          // which looks from -x and -y (engine axes). A corner slope faces it on both sides.
          // An inverted slope shows its profile on the -y side.
          slope: !slope ? null : part === CORNER_PART ? "-x-y" : "-x", slope_inverted: slope && inverted }],
        steps: [{ n: 1, brick_ids: [0] }], parts: [], warnings: [],
        stats: { brick_count: 1, step_count: 1, est_cost_usd: 0 },
      };
      const one = buildBrickScene(fake, [0], material, true);
      one.setVisible(1);
      partScene.add(one.group);
      // Size the frame by the largest brick (1x8) so parts keep a consistent scale.
      const ref = new THREE.Box3(new THREE.Vector3(-4, 0, -1), new THREE.Vector3(4, 1.5, 1));
      const box = one.bounds.clone().union(new THREE.Box3().setFromCenterAndSize(
        one.bounds.getCenter(new THREE.Vector3()), ref.getSize(new THREE.Vector3()).multiplyScalar(0.45)));
      const url = await draw(partScene, fitCamera(box, 1.4, 1.15), 280, 200);
      partScene.remove(one.group);
      one.dispose();
      return url;
    },
    dispose() {
      model.dispose();
      material.dispose();
      renderer.dispose();
    },
  };
}
