// Mirrors the Build JSON contract in docs/superpowers/specs/2026-10-15-lego-design.md §3.
export interface BrickColor {
  bricklink: number;
  ldraw: number;
  name: string;
  hex: string;
}

export interface Brick {
  id: number;
  part: string;
  w: number;
  l: number;
  x: number;
  y: number;
  z: number; // plate units
  rot: 0 | 90;
  color: BrickColor;
  hidden: boolean;
}

export interface Step {
  n: number;
  brick_ids: number[];
}

export interface PartCount {
  part: string;
  color_bricklink: number;
  qty: number;
}

export interface Build {
  id: string;
  size_studs: number;
  dims: { x: number; y: number; z: number };
  bricks: Brick[];
  steps: Step[];
  parts: PartCount[];
  stats: { brick_count: number; step_count: number; est_cost_usd: number };
  warnings: string[];
}

export type MeshStatus = "queued" | "running" | "done" | "error";

export const sizeX = (b: Brick) => (b.rot === 90 ? b.l : b.w);
export const sizeY = (b: Brick) => (b.rot === 90 ? b.w : b.l);
