import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { inventory, paginate, stepParts } from "../booklet";
import Booklet from "../components/Booklet";
import Capture, { SAMPLE_URL } from "../components/Capture";
import Generating, { SLOW_AFTER_S } from "../components/Generating";
import StepScrubber from "../components/StepScrubber";
import { BRICK_H, brickEdges, brickGeometry, buildBrickScene, stepAt, stepOrder } from "../three/bricks";
import * as THREE from "three";
import type { Build } from "../types";
import fixture from "./fixture-build.json";

const build = fixture as unknown as Build;

describe("fixture build", () => {
  it("has every brick in exactly one step", () => {
    const ids = stepOrder(build).sort((a, b) => a - b);
    expect(ids).toEqual(build.bricks.map((b) => b.id));
  });

  it("paginates 2-4 steps per page and tallies parts", () => {
    const pages = paginate(build.steps);
    expect(pages.flat().length).toBe(build.steps.length);
    pages.slice(0, -1).forEach((p) => expect(p.length).toBe(4));
    const total = inventory(build).reduce((n, p) => n + p.qty, 0);
    expect(total).toBe(build.stats.brick_count);
    expect(stepParts(build, build.steps[0]).reduce((n, p) => n + p.qty, 0)).toBe(build.steps[0].brick_ids.length);
  });
});

describe("viewer scene", () => {
  it("instances every brick and can show prefixes for steps", () => {
    const s = buildBrickScene(build, stepOrder(build), new THREE.MeshBasicMaterial(), true);
    const meshes = s.group.children.filter((c): c is THREE.InstancedMesh => c instanceof THREE.InstancedMesh);
    s.setVisible(build.bricks.length);
    expect(meshes.reduce((n, m) => n + m.count, 0)).toBe(build.bricks.length);
    const first = build.steps[0].brick_ids.length;
    s.setVisible(first);
    expect(meshes.reduce((n, m) => n + m.count, 0)).toBe(first);
    expect(s.bounds.isEmpty()).toBe(false);
    // ground is at y = 0 and model is centred
    expect(s.bounds.min.y).toBeCloseTo(0, 5);
  });
});

describe("step scrubber", () => {
  const total = build.bricks.length;
  const first = build.steps[0].brick_ids.length;

  it("maps a brick count to its instruction step", () => {
    expect(stepAt(build, 0)).toBe(0);
    expect(stepAt(build, 1)).toBe(1);
    expect(stepAt(build, first)).toBe(1);
    expect(stepAt(build, first + 1)).toBe(2);
    expect(stepAt(build, total)).toBe(build.steps.length);
  });

  it("labels progress and moves one brick at a time within bounds", () => {
    const onChange = vi.fn();
    const { rerender } = render(<StepScrubber build={build} value={first + 1} onChange={onChange} />);
    expect(screen.getByText(`Step 2 of ${build.steps.length} · brick ${first + 1} of ${total}`)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Previous brick" }));
    expect(onChange).toHaveBeenLastCalledWith(first);
    fireEvent.click(screen.getByRole("button", { name: "Next brick" }));
    expect(onChange).toHaveBeenLastCalledWith(first + 2);
    fireEvent.change(screen.getByRole("slider", { name: "Build progress" }), { target: { value: "3" } });
    expect(onChange).toHaveBeenLastCalledWith(3);

    rerender(<StepScrubber build={build} value={total} onChange={onChange} />);
    expect(screen.getByText(`Finished model · ${total} bricks`)).toBeTruthy();
    expect((screen.getByRole("button", { name: "Next brick" }) as HTMLButtonElement).disabled).toBe(true);
    rerender(<StepScrubber build={build} value={0} onChange={onChange} />);
    expect(screen.getByText("No bricks placed yet")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Previous brick" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("slope bricks", () => {
  const top = (g: THREE.BufferGeometry) => {
    g.computeBoundingBox();
    return g.boundingBox!;
  };
  // Highest body point (excluding studs) along world x, sampled at both ends of the footprint.
  const heightAt = (g: THREE.BufferGeometry, x: number) => {
    const pos = g.attributes.position;
    let h = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getX(i) - x) < 0.02 && pos.getY(i) <= BRICK_H / 2) h = Math.max(h, pos.getY(i));
    }
    return h;
  };

  it("draws a wedge that is low on the side it faces, same footprint as the brick", () => {
    const brick = top(brickGeometry(2, 2));
    const slope = brickGeometry(2, 2, "+x");
    const box = top(slope);
    expect(box.min.x).toBeCloseTo(brick.min.x, 3);
    expect(box.max.z).toBeCloseTo(brick.max.z, 3);
    expect(heightAt(slope, box.max.x)).toBeLessThan(heightAt(slope, box.min.x) - 0.5);
    const flipped = brickGeometry(2, 2, "-x");
    expect(heightAt(flipped, top(flipped).min.x)).toBeLessThan(heightAt(flipped, top(flipped).max.x) - 0.5);
  });

  it("has studs only on the back row", () => {
    const studTops = (g: THREE.BufferGeometry) => {
      const pos = g.attributes.position;
      const xs = new Set<number>();
      for (let i = 0; i < pos.count; i++) if (pos.getY(i) > BRICK_H / 2 + 0.1) xs.add(Math.round(pos.getX(i)));
      return xs;
    };
    expect([...studTops(brickGeometry(2, 1, "+x"))].every((x) => x <= 0)).toBe(true);
    expect([...studTops(brickGeometry(2, 1))].some((x) => x > 0)).toBe(true);
  });

  // Highest body point (excluding studs) near world (x, z).
  const heightNear = (g: THREE.BufferGeometry, x: number, z: number) => {
    const pos = g.attributes.position;
    let h = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getX(i) - x) < 0.05 && Math.abs(pos.getZ(i) - z) < 0.05 && pos.getY(i) <= BRICK_H / 2) h = Math.max(h, pos.getY(i));
    }
    return h;
  };
  const studCells = (g: THREE.BufferGeometry) => {
    const pos = g.attributes.position;
    const cells = new Set<string>();
    for (let i = 0; i < pos.count; i++) if (pos.getY(i) > BRICK_H / 2 + 0.1) cells.add(`${Math.sign(pos.getX(i))},${Math.sign(pos.getZ(i))}`);
    return cells;
  };

  it("draws a corner slope low on both sides it faces, with one stud on the back corner", () => {
    const brick = top(brickGeometry(2, 2));
    // Engine y is world -Z, so "+x+y" faces world +X and -Z; its back corner is at -X, +Z.
    for (const [dir, fx, fz] of [["+x+y", 1, -1], ["+x-y", 1, 1], ["-x+y", -1, -1], ["-x-y", -1, 1]] as const) {
      const g = brickGeometry(2, 2, dir);
      const box = top(g);
      expect(box.min.x).toBeCloseTo(brick.min.x, 3);
      expect(box.max.z).toBeCloseTo(brick.max.z, 3);
      const d = 1 - 0.0125;
      const back = heightNear(g, -fx * d, -fz * d);
      expect(back).toBeCloseTo(heightNear(brickGeometry(2, 2), -fx * d, -fz * d), 3); // full height at the back
      expect(heightNear(g, fx * d, fz * d)).toBeLessThan(back - 0.5); // front corner
      expect(heightNear(g, fx * d, -fz * d)).toBeLessThan(back - 0.5); // low along the x side
      expect(heightNear(g, -fx * d, fz * d)).toBeLessThan(back - 0.5); // and along the z side
      expect([...studCells(g)]).toEqual([`${-fx},${-fz}`]);
    }
  });

  it("builds a closed corner wedge with outward-facing triangles", () => {
    // Signed volume (sum of tetrahedra to the origin) is only right for a closed mesh wound
    // outward. A double-convex corner is the intersection of two straight wedges.
    const volume = (g: THREE.BufferGeometry) => {
      const pos = g.attributes.position;
      let v = 0;
      for (let i = 0; i < pos.count; i += 3) {
        const [a, b, c] = [0, 1, 2].map((k) => new THREE.Vector3().fromBufferAttribute(pos, i + k));
        v += a.dot(b.cross(c)) / 6;
      }
      return v;
    };
    const box = volume(brickGeometry(2, 2)), straight = volume(brickGeometry(2, 2, "+x"));
    for (const dir of ["+x+y", "+x-y", "-x+y", "-x-y"] as const) {
      const corner = volume(brickGeometry(2, 2, dir));
      expect(corner).toBeGreaterThan(box / 2);
      expect(corner).toBeLessThan(straight - 0.2);
    }
    expect(straight).toBeLessThan(box);
  });

  it("instances slopes separately from bricks of the same size", () => {
    const b = build.bricks[0];
    const withSlope: Build = { ...build, bricks: [{ ...b, slope: null }, { ...b, id: 1, x: b.x + 3, part: "3040", slope: "+y" }],
      steps: [{ n: 1, brick_ids: [0, 1] }] };
    const s = buildBrickScene(withSlope, [0, 1], new THREE.MeshBasicMaterial(), true);
    expect(s.group.children.filter((c) => c instanceof THREE.InstancedMesh).length).toBe(2);
  });

  // Lowest body point along world x: the underside of the part there.
  const bottomAt = (g: THREE.BufferGeometry, x: number) => {
    const pos = g.attributes.position;
    let lo = Infinity;
    for (let i = 0; i < pos.count; i++) if (Math.abs(pos.getX(i) - x) < 0.02) lo = Math.min(lo, pos.getY(i));
    return lo;
  };

  it("draws an inverted slope as a flipped wedge: flat top, underside rising on the side it faces", () => {
    const brick = top(brickGeometry(2, 2));
    for (const [dir, front] of [["+x", 1], ["-x", -1]] as const) {
      const g = brickGeometry(2, 2, dir, true);
      const box = top(g);
      expect(box.min.x).toBeCloseTo(brick.min.x, 3);
      expect(box.max.x).toBeCloseTo(brick.max.x, 3);
      expect(box.max.y).toBeCloseTo(brick.max.y, 3); // studs on top, full height
      const edge = front * (1 - 0.0125);
      expect(heightAt(g, edge)).toBeCloseTo(heightAt(g, -edge), 3); // flat top front to back
      expect(bottomAt(g, -edge)).toBeCloseTo(box.min.y, 3); // back row reaches the bottom
      expect(bottomAt(g, edge)).toBeGreaterThan(box.min.y + 0.5); // front is undercut
    }
  });

  it("puts studs on the whole top of an inverted slope", () => {
    const xs = (g: THREE.BufferGeometry) => {
      const pos = g.attributes.position;
      const out = new Set<number>();
      for (let i = 0; i < pos.count; i++) if (pos.getY(i) > BRICK_H / 2 + 0.1) out.add(Math.sign(pos.getX(i)));
      return out;
    };
    expect(xs(brickGeometry(2, 1, "+x", true))).toEqual(new Set([1, -1]));
    expect(xs(brickGeometry(2, 1, "+x"))).toEqual(new Set([-1]));
  });

  it("instances inverted slopes separately from slopes facing the same way", () => {
    const b = build.bricks[0];
    const bricks = [{ ...b, part: "3040", slope: "+y" as const }, { ...b, id: 1, x: b.x + 3, part: "3665", slope: "+y" as const, slope_inverted: true }];
    const s = buildBrickScene({ ...build, bricks, steps: [{ n: 1, brick_ids: [0, 1] }] }, [0, 1], new THREE.MeshBasicMaterial(), true);
    expect(s.group.children.filter((c) => c instanceof THREE.InstancedMesh).length).toBe(2);
    expect(brickEdges(1, 2, "+y", true)).not.toEqual(brickEdges(1, 2, "+y"));
  });

  it("instances corner slopes separately and outlines them", () => {
    const b = { ...build.bricks[0], w: 2, l: 2 };
    const bricks = [{ ...b, part: "3003", slope: null }, { ...b, id: 1, x: b.x + 3, part: "3039", slope: "+x" as const },
      { ...b, id: 2, x: b.x + 6, part: "3045", slope: "-x+y" as const }];
    const s = buildBrickScene({ ...build, bricks, steps: [{ n: 1, brick_ids: [0, 1, 2] }] }, [0, 1, 2], new THREE.MeshBasicMaterial(), true);
    expect(s.group.children.filter((c) => c instanceof THREE.InstancedMesh).length).toBe(3);
    expect(brickEdges(2, 2, "-x+y").length).toBeGreaterThan(0);
  });
});

describe("try a sample", () => {
  it("uploads the bundled sample mesh without a photo", async () => {
    const calls: { url: string; body?: unknown }[] = [];
    const ok = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body });
      if (url === "/api/health") return ok({ model_available: false, model_reason: "no weights." });
      if (url === SAMPLE_URL) return new Response(new Blob([new Uint8Array([1, 2, 3])]));
      return ok({ mesh_id: "a".repeat(64), status: "done", error: null });
    }));
    const onUploaded = vi.fn();
    render(<Capture onUploaded={onUploaded} />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Try a sample" }));
    });
    const upload = calls.find((c) => c.url === "/api/meshes");
    expect(upload).toBeTruthy();
    expect(((upload!.body as FormData).get("file") as File).name).toBe("chair.glb");
    expect(onUploaded).toHaveBeenCalledWith({ mesh_id: "a".repeat(64), status: "done", error: null });
    vi.unstubAllGlobals();
  });
});

describe("booklet", () => {
  it("renders cover, step pages and inventory from a fixture build", async () => {
    const fake = {
      cover: vi.fn(async () => "blob:cover"),
      step: vi.fn(async (n: number) => `blob:step-${n}`),
      part: vi.fn(async (p: string) => `blob:part-${p}`),
      dispose: vi.fn(),
    };
    const create = () => fake;
    globalThis.URL.revokeObjectURL = vi.fn();
    await act(async () => {
      render(<Booklet build={build} createRenderer={create} />);
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });
    expect(fake.step).toHaveBeenCalledTimes(build.steps.length);
    expect(screen.getByAltText("Finished model")).toBeTruthy();
    expect(screen.getByAltText(`Step ${build.steps.length}`)).toBeTruthy();
    expect(screen.getByText("All the bricks you need")).toBeTruthy();
    const btn = screen.getByRole("button", { name: "Print or save as PDF" }) as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });
});

describe("generating", () => {
  it("says when a run is unusually slow and cancels it on the server", async () => {
    vi.useFakeTimers();
    const id = "b".repeat(64);
    const calls: { url: string; method?: string }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method });
      const status = url.endsWith("/cancel") ? "error" : "running";
      return new Response(JSON.stringify({ mesh_id: id, status, error: null }), { status: 200 });
    }));
    const onCancel = vi.fn();
    render(<Generating meshId={id} onDone={vi.fn()} onError={vi.fn()} onCancel={onCancel} />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(screen.queryByText(/taking much longer than usual/)).toBeNull();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SLOW_AFTER_S * 1000);
    });
    expect(screen.getByText(/taking much longer than usual/)).toBeTruthy();
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    });
    expect(calls).toContainEqual({ url: `/api/meshes/${id}/cancel`, method: "POST" });
    expect(onCancel).toHaveBeenCalledTimes(1);
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });
});
