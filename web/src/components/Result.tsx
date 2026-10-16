import { lazy, Suspense, useEffect, useState } from "react";
import { createBuild, exportUrl } from "../api";
import type { Build } from "../types";

const BrickViewer = lazy(() => import("./BrickViewer"));
const Booklet = lazy(() => import("./Booklet"));

const MIN = 8;
const MAX = 48;
const BIG_BUILD = 2000;

interface Props {
  meshId: string;
  onRestart: () => void;
}

export default function Result({ meshId, onRestart }: Props) {
  const [size, setSize] = useState(16);
  const [build, setBuild] = useState<Build | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"model" | "instructions">("model");

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    const t = setTimeout(() => {
      createBuild(meshId, size, ctrl.signal)
        .then((b) => {
          setBuild(b);
          setError(null);
        })
        .catch((e) => {
          if ((e as Error).name !== "AbortError") setError((e as Error).message);
        })
        .finally(() => {
          if (!ctrl.signal.aborted) setLoading(false);
        });
    }, 300);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [meshId, size]);

  const fmtCost = (n: number) => n.toLocaleString(undefined, { style: "currency", currency: "USD" });

  return (
    <main className="result">
      <div className="toolbar no-print">
        <nav className="segmented" role="tablist">
          <button role="tab" aria-selected={tab === "model"} onClick={() => setTab("model")}>
            Model
          </button>
          <button role="tab" aria-selected={tab === "instructions"} onClick={() => setTab("instructions")} disabled={!build}>
            Instructions
          </button>
        </nav>
        <button className="btn ghost" onClick={onRestart}>
          ← New photo
        </button>
      </div>

      {build && build.warnings.length > 0 && (
        <ul className="warnings no-print" role="status">
          {build.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
      {error && (
        <p className="error no-print" role="alert">
          {error}
        </p>
      )}

      {tab === "model" ? (
        <div className="workbench">
          <div className={`viewer${loading ? " busy" : ""}`}>
            {build ? (
              <Suspense fallback={<div className="viewer-empty">Loading viewer…</div>}>
                <BrickViewer build={build} />
              </Suspense>
            ) : (
              <div className="viewer-empty">{error ? "No build yet." : "Placing bricks…"}</div>
            )}
            {loading && build && <span className="rebuilding">Rebuilding…</span>}
          </div>

          <aside className="panel">
            <section>
              <label className="size">
                <span className="size-head">
                  <span>Size</span>
                  <output>{size} studs wide</output>
                </span>
                <input
                  type="range"
                  min={MIN}
                  max={MAX}
                  step={1}
                  value={size}
                  onChange={(e) => setSize(Number(e.target.value))}
                />
                <span className="range-ends muted small">
                  <span>{MIN}</span>
                  <span>{MAX}</span>
                </span>
              </label>
            </section>

            <section>
              {build && (
                <dl className="stats">
                  <div>
                    <dt>Bricks</dt>
                    <dd>{build.stats.brick_count.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt>Steps</dt>
                    <dd>{build.stats.step_count}</dd>
                  </div>
                  <div>
                    <dt>Parts cost</dt>
                    <dd>{fmtCost(build.stats.est_cost_usd)}</dd>
                  </div>
                </dl>
              )}
              {build && build.stats.brick_count > BIG_BUILD && (
                <p className="note small">That's a lot of bricks. A smaller size will be quicker to build.</p>
              )}
              <p className="muted small">Cost is a rough estimate from average used-part prices.</p>
            </section>

            {build && (
              <section className="downloads">
                <h2>Download</h2>
                <a className="btn primary" href={exportUrl(build.id, "bricklink")} download>
                  BrickLink wanted list
                </a>
                <a className="btn download" href={exportUrl(build.id, "ldraw")} download>
                  LDraw model (.ldr)
                </a>
                <a className="btn download" href={exportUrl(build.id, "csv")} download>
                  Parts list (.csv)
                </a>
              </section>
            )}
          </aside>
        </div>
      ) : (
        build && (
          <Suspense fallback={<p className="muted">Loading booklet…</p>}>
            <Booklet build={build} />
          </Suspense>
        )
      )}
    </main>
  );
}
