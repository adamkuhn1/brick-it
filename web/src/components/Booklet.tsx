import { useEffect, useMemo, useState } from "react";
import { inventory, paginate, partLabel, stepParts, type PartLine } from "../booklet";
import { createBookletRenderer, type BookletRenderer } from "../three/stepRenderer";
import type { Build } from "../types";

interface Props {
  build: Build;
  createRenderer?: (build: Build) => BookletRenderer;
}

type Images = Record<string, string>;

const INVENTORY_PER_PAGE = 24;

/** Renders all booklet images up front (progressively) so Print gets complete pages. */
function useBookletImages(build: Build, create: (b: Build) => BookletRenderer) {
  const [images, setImages] = useState<Images>({});
  const [done, setDone] = useState(0);
  const parts = useMemo(() => inventory(build), [build]);
  const total = 1 + build.steps.length + parts.length;

  useEffect(() => {
    let cancelled = false;
    const urls: string[] = [];
    const r = create(build);
    setImages({});
    setDone(0);
    (async () => {
      const jobs: [string, () => Promise<string>][] = [
        ["cover", () => r.cover()],
        ...parts.map((p): [string, () => Promise<string>] => [p.key, () => r.part(p.part, p.hex, p.w, p.l, p.slope, p.inverted)]),
        ...build.steps.map((s): [string, () => Promise<string>] => [`step-${s.n}`, () => r.step(s.n)]),
      ];
      let batch: Images = {};
      for (let i = 0; i < jobs.length; i++) {
        if (cancelled) return;
        const [k, job] = jobs[i];
        const url = await job();
        urls.push(url);
        batch[k] = url;
        if (i % 8 === 7 || i === jobs.length - 1) {
          const flush = batch;
          batch = {};
          setImages((prev) => ({ ...prev, ...flush }));
          setDone(i + 1);
          await new Promise((res) => setTimeout(res, 0)); // let the page paint
        }
      }
    })().catch((e) => console.error("booklet render failed", e));
    return () => {
      cancelled = true;
      r.dispose();
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [build, create, parts]);

  return { images, done, total, parts };
}

function Img({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  return src ? <img src={src} alt={alt} className={className} /> : <div className={`${className ?? ""} pending`} aria-label={alt} />;
}

function Callout({ lines, images }: { lines: PartLine[]; images: Images }) {
  return (
    <div className="callout">
      {lines.map((p) => (
        <figure key={p.key}>
          <Img src={images[p.key]} alt={partLabel(p)} />
          <figcaption>{p.qty}x</figcaption>
        </figure>
      ))}
    </div>
  );
}

export default function Booklet({ build, createRenderer = createBookletRenderer }: Props) {
  const { images, done, total, parts } = useBookletImages(build, createRenderer);
  const pages = useMemo(() => paginate(build.steps), [build]);
  const invPages = useMemo(() => paginate(parts, INVENTORY_PER_PAGE), [parts]);
  const ready = done >= total;

  return (
    <div className="booklet">
      <div className="booklet-bar no-print">
        <p>
          {ready
            ? `${pages.length + invPages.length + 1} pages, ${build.steps.length} steps.`
            : `Drawing pages… ${done} of ${total}`}
        </p>
        <button className="btn" onClick={() => window.print()} disabled={!ready}>
          Print or save as PDF
        </button>
      </div>

      <section className="page cover">
        <div className="cover-badge">
          <span>{build.stats.brick_count}</span> bricks
        </div>
        <Img src={images.cover} alt="Finished model" className="cover-img" />
        <footer>
          <strong>Your build</strong>
          <span>{build.size_studs} studs wide, {build.dims.z} bricks tall</span>
        </footer>
      </section>

      {pages.map((steps, i) => (
        <section className="page steps" key={i} data-count={steps.length}>
          {steps.map((s) => (
            <article className="step" key={s.n}>
              <div className="step-head">
                <span className="step-n">{s.n}</span>
                <Callout lines={stepParts(build, s)} images={images} />
              </div>
              <Img src={images[`step-${s.n}`]} alt={`Step ${s.n}`} className="step-img" />
            </article>
          ))}
          <span className="page-n">{i + 2}</span>
        </section>
      ))}

      {invPages.map((lines, i) => (
        <section className="page inventory" key={`inv-${i}`}>
          {i === 0 && <h2>All the bricks you need</h2>}
          <div className="inv-grid">
            {lines.map((p) => (
              <figure key={p.key}>
                <Img src={images[p.key]} alt={partLabel(p)} />
                <figcaption>
                  <strong>{p.qty}x</strong>
                  <span>{p.part}</span>
                  <span className="muted">{p.colorName}</span>
                </figcaption>
              </figure>
            ))}
          </div>
          <span className="page-n">{pages.length + i + 2}</span>
        </section>
      ))}
    </div>
  );
}
