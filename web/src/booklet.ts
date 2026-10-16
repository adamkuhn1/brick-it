import type { Brick, Build, Step } from "./types";

export const STEPS_PER_PAGE = 4;

export interface PartLine {
  key: string; // part + color, unique per image
  part: string;
  w: number;
  l: number;
  hex: string;
  colorName: string;
  qty: number;
  slope: boolean;
}

const partKey = (b: Pick<Brick, "part"> & { color: { hex: string } }) => `${b.part}-${b.color.hex}`;

function tally(bricks: Brick[]): PartLine[] {
  const map = new Map<string, PartLine>();
  for (const b of bricks) {
    const k = partKey(b);
    const line = map.get(k);
    if (line) line.qty++;
    else map.set(k, { key: k, part: b.part, w: b.w, l: b.l, hex: b.color.hex, colorName: b.color.name, qty: 1, slope: !!b.slope });
  }
  return [...map.values()].sort((a, b) => b.w * b.l - a.w * a.l || b.qty - a.qty);
}

export function stepParts(build: Build, step: Step): PartLine[] {
  const byId = new Map(build.bricks.map((b) => [b.id, b]));
  return tally(step.brick_ids.map((id) => byId.get(id)!));
}

export function inventory(build: Build): PartLine[] {
  return tally(build.bricks);
}

/** "Red 2×2 slope" — what a part line shows in alt text. */
export const partLabel = (p: PartLine) => `${p.colorName} ${p.w}×${p.l} ${p.slope ? "slope" : "brick"}`;

export function paginate<T>(items: T[], perPage = STEPS_PER_PAGE): T[][] {
  const pages: T[][] = [];
  for (let i = 0; i < items.length; i += perPage) pages.push(items.slice(i, i + perPage));
  return pages;
}
