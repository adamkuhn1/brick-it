// Mirrors the Build JSON contract in docs/superpowers/specs/2026-10-15-lego-design.md §3.
export interface BrickColor {
  bricklink: number;
  ldraw: number;
  name: string;
  hex: string;
}

/** Direction (engine axes) the low side of a straight 45° slope brick faces. */
export type StraightDir = "+x" | "-x" | "+y" | "-y";
/** The two directions a 2×2 corner slope (3045, double convex) slopes down toward. */
export type CornerDir = "+x+y" | "+x-y" | "-x+y" | "-x-y";
export type SlopeDir = StraightDir | CornerDir;

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
  /**
   * null for a plain brick. For Slope 45 parts (3040, 3039) the direction their low side
   * faces; the footprint is 2 studs deep along it. For the corner slope 3045, two directions;
   * its one stud is on the opposite corner.
   */
  slope: SlopeDir | null;
  /**
   * True for an inverted slope (3665 2×1, 3660 2×2): the slope is on the underside, `slope`
   * is the side its low edge faces, and the whole top has studs.
   */
  slope_inverted: boolean;
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

/** Slope 45 2×2 Double Convex: the corner slope. */
export const CORNER_PART = "3045";

export const sizeX = (b: Brick) => (b.rot === 90 ? b.l : b.w);
export const sizeY = (b: Brick) => (b.rot === 90 ? b.w : b.l);
