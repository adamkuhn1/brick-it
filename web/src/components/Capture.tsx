import { useEffect, useRef, useState } from "react";
import { health, uploadFile, type MeshResponse } from "../api";

const ACCEPT = "image/jpeg,image/png,image/webp,.glb,.gltf,.obj,.ply,.stl";

interface Props {
  onUploaded: (res: MeshResponse) => void;
  initialError?: string | null;
}

function Webcam({ onShot, onClose }: { onShot: (f: File) => void; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } } })
      .then((s) => {
        stream = s;
        if (video.current) video.current.srcObject = s;
      })
      .catch(() => setErr("Camera access was blocked. Allow it in your browser, or upload a photo instead."));
    if (!navigator.mediaDevices) setErr("This browser can't use the camera here. Upload a photo instead.");
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  const shoot = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    c.toBlob((b) => b && onShot(new File([b], "camera.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
  };

  return (
    <div className="webcam">
      {err ? <p className="error">{err}</p> : <video ref={video} autoPlay playsInline muted />}
      <div className="row">
        <button className="btn primary" onClick={shoot} disabled={!!err}>
          Take photo
        </button>
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}

export default function Capture({ onUploaded, initialError = null }: Props) {
  const [error, setError] = useState<string | null>(initialError);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [camera, setCamera] = useState(false);
  const [model, setModel] = useState<{ ok: boolean; reason: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    health()
      .then((h) => setModel({ ok: h.model_available, reason: h.model_reason }))
      .catch(() => setError("Can't reach the local server. Start it with ./scripts/dev.sh."));
  }, []);

  async function send(file: File) {
    setBusy(true);
    setError(null);
    setCamera(false);
    try {
      onUploaded(await uploadFile(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="capture">
      <header className="intro">
        <span className="eyebrow">Photo to LEGO</span>
        <h1>Turn a photo into a LEGO build</h1>
        <p>
          Real bricks you can buy, a model you can spin around, and a step-by-step instruction booklet.
          Everything runs on this computer.
        </p>
      </header>

      <ol className="howto">
        <li className="current">
          <span className="num">1</span>
          <h2>Add a photo</h2>
          <p>One object, plain background, shot from a slight angle above.</p>
        </li>
        <li>
          <span className="num">2</span>
          <h2>Pick a size</h2>
          <p>8 to 48 studs wide.</p>
        </li>
        <li>
          <span className="num">3</span>
          <h2>Build it</h2>
          <p>Print the booklet and order the parts.</p>
        </li>
      </ol>

      {camera ? (
        <Webcam onShot={send} onClose={() => setCamera(false)} />
      ) : (
        <div
          className={`drop${dragging ? " over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) send(f);
          }}
        >
          <span className="drop-icon" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="5" width="18" height="14" rx="2" />
              <circle cx="9" cy="10" r="1.6" />
              <path d="m21 16-5-5-8 8" />
            </svg>
          </span>
          <p className="drop-title">{busy ? "Uploading…" : "Drop a photo here"}</p>
          <div className="row">
            <button className="btn primary" onClick={() => input.current?.click()} disabled={busy}>
              Choose photo
            </button>
            <button className="btn" onClick={() => setCamera(true)} disabled={busy || model?.ok === false}>
              Use camera
            </button>
          </div>
          <p className="muted small">JPG or PNG. You can also drop a .glb or .obj 3D model.</p>
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) send(f);
            }}
          />
        </div>
      )}

      {model && !model.ok && (
        <p className="note">
          Photo-to-3D isn't set up on this machine ({model.reason}) You can still upload a .glb or .obj model.
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </main>
  );
}
