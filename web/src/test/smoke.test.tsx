import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { inventory, paginate, stepParts } from "../booklet";
import Booklet from "../components/Booklet";
import { buildBrickScene, stepOrder } from "../three/bricks";
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
