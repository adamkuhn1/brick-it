import { useEffect, useState } from "react";
import { meshStatus } from "../api";
import type { MeshStatus } from "../types";

interface Props {
  meshId: string;
  onDone: () => void;
  onError: (message: string) => void;
}

const LABEL: Record<MeshStatus, string> = {
  queued: "Waiting for the model to start",
  running: "Turning your photo into a 3D shape",
  done: "Done",
  error: "Something went wrong",
};

export default function Generating({ meshId, onDone, onError }: Props) {
  const [status, setStatus] = useState<MeshStatus>("queued");
  const [seconds, setSeconds] = useState(0);

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
    </main>
  );
}
