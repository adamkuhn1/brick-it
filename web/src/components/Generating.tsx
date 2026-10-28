import { useEffect, useState } from "react";
import { cancelMesh, meshStatus } from "../api";
import type { MeshStatus } from "../types";

interface Props {
  meshId: string;
  onDone: () => void;
  onError: (message: string) => void;
  onCancel: () => void;
}

/** After this long, say it's unusually slow: a normal run is 30-90 s. */
export const SLOW_AFTER_S = 90;

const LABEL: Record<MeshStatus, string> = {
  queued: "Waiting for the model to start",
  running: "Turning your photo into a 3D shape",
  done: "Done",
  error: "Something went wrong",
};

export default function Generating({ meshId, onDone, onError, onCancel }: Props) {
  const [status, setStatus] = useState<MeshStatus>("queued");
  const [seconds, setSeconds] = useState(0);
  const [cancelling, setCancelling] = useState(false);

  const cancel = async () => {
    setCancelling(true);
    try {
      await cancelMesh(meshId);
    } catch {
      /* already finished or gone: leaving this screen is still right */
    }
    onCancel();
  };

  useEffect(() => {
    let stop = false;
    const started = Date.now();
    const tick = setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 1000);
    (async () => {
      while (!stop) {
        try {
          const s = await meshStatus(meshId);
          setStatus(s.status);
          if (s.status === "done") return onDone();
          if (s.status === "error") return onError(s.error ?? "3D generation failed. Try another photo.");
        } catch (e) {
          return onError((e as Error).message);
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    })();
    return () => {
      stop = true;
      clearInterval(tick);
    };
  }, [meshId, onDone, onError]);

  return (
    <main className="generating" aria-live="polite">
      <div className="stack" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      <h1>{LABEL[status]}…</h1>
      <p className="muted">
        This usually takes 30–90 seconds on a laptop. The first run also downloads the model.
      </p>
      <p className="timer">{seconds}s</p>
      {seconds >= SLOW_AFTER_S && (
        <p className="note small" role="status">
          This is taking much longer than usual. Your computer may be short on memory; cancelling frees it right away.
        </p>
      )}
      <button className="btn" onClick={cancel} disabled={cancelling}>
        {cancelling ? "Cancelling…" : "Cancel"}
      </button>
    </main>
  );
}
