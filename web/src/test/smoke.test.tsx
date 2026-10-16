import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { inventory, paginate, stepParts } from "../booklet";
import Booklet from "../components/Booklet";
import { BRICK_H, brickGeometry, buildBrickScene, stepOrder } from "../three/bricks";
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

  it("instances slopes separately from bricks of the same size", () => {
    const b = build.bricks[0];
    const withSlope: Build = { ...build, bricks: [{ ...b, slope: null }, { ...b, id: 1, x: b.x + 3, part: "3040", slope: "+y" }],
      steps: [{ n: 1, brick_ids: [0, 1] }] };
    const s = buildBrickScene(withSlope, [0, 1], new THREE.MeshBasicMaterial(), true);
    expect(s.group.children.filter((c) => c instanceof THREE.InstancedMesh).length).toBe(2);
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
