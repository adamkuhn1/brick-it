import { stepAt } from "../three/bricks";
import type { Build } from "../types";

interface Props {
  build: Build;
  /** Number of bricks placed, 0..build.bricks.length. */
  value: number;
  onChange: (k: number) => void;
}

/** Slider that scrubs through the build one brick at a time, in booklet order. */
export default function StepScrubber({ build, value, onChange }: Props) {
  const total = build.bricks.length;
  const set = (k: number) => onChange(Math.max(0, Math.min(total, k)));
  const label =
    value >= total
      ? `Finished model · ${total.toLocaleString()} bricks`
      : value <= 0
        ? "No bricks placed yet"
        : `Step ${stepAt(build, value)} of ${build.steps.length} · brick ${value.toLocaleString()} of ${total.toLocaleString()}`;

  return (
    <div className="scrubber">
      <output className="scrubber-label" htmlFor="scrubber-range">
        {label}
      </output>
      <div className="scrubber-row">
        <button type="button" className="scrubber-btn" onClick={() => set(value - 1)} disabled={value <= 0} aria-label="Previous brick">
          ‹
        </button>
        <input
          id="scrubber-range"
          type="range"
          min={0}
          max={total}
          step={1}
          value={value}
          aria-label="Build progress"
          aria-valuetext={label}
          onChange={(e) => set(Number(e.target.value))}
        />
        <button type="button" className="scrubber-btn" onClick={() => set(value + 1)} disabled={value >= total} aria-label="Next brick">
          ›
        </button>
      </div>
    </div>
  );
}
