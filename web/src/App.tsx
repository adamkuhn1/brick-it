import { useCallback, useEffect, useState } from "react";
import Capture from "./components/Capture";
import Generating from "./components/Generating";
import Result from "./components/Result";

type View =
  | { name: "capture"; error?: string }
  | { name: "generating"; meshId: string }
  | { name: "result"; meshId: string };

const ID = /^[0-9a-f]{64}$/;

function fromHash(): View {
  const id = new URLSearchParams(location.hash.slice(1)).get("mesh");
  return id && ID.test(id) ? { name: "generating", meshId: id } : { name: "capture" };
}

export default function App() {
  const [view, setView] = useState<View>(fromHash);

  // Keep the mesh id in the URL so a reload resumes where you were.
  useEffect(() => {
    const hash = view.name === "capture" ? "" : `#mesh=${view.meshId}`;
    if (location.hash !== hash) history.replaceState(null, "", hash || location.pathname);
  }, [view]);

  const meshId = view.name === "capture" ? null : view.meshId;
  const onDone = useCallback(() => meshId && setView({ name: "result", meshId }), [meshId]);
  const onError = useCallback((error: string) => setView({ name: "capture", error }), []);
  const onCancel = useCallback(() => setView({ name: "capture" }), []);

  return (
    <div className="app">
      <header className="masthead no-print">
        <a href="/" className="brand" onClick={(e) => { e.preventDefault(); setView({ name: "capture" }); }}>
          <span className="brand-brick" aria-hidden="true" />
          Brick it
        </a>
      </header>
      {view.name === "capture" && (
        <Capture
          key={view.error ?? "fresh"}
          initialError={view.error}
          onUploaded={(r) =>
            setView(r.status === "done" ? { name: "result", meshId: r.mesh_id } : { name: "generating", meshId: r.mesh_id })
          }
        />
      )}
      {view.name === "generating" && <Generating meshId={view.meshId} onDone={onDone} onError={onError} onCancel={onCancel} />}
      {view.name === "result" && <Result meshId={view.meshId} onRestart={() => setView({ name: "capture" })} />}
    </div>
  );
}
