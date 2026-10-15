import type { Build, MeshStatus } from "./types";

export interface MeshResponse {
  mesh_id: string;
  status: MeshStatus;
  error: string | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function uploadFile(file: File): Promise<MeshResponse> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch("/api/meshes", { method: "POST", body: form }));
}

export async function meshStatus(meshId: string): Promise<MeshResponse> {
  return json(await fetch(`/api/meshes/${meshId}`));
}

export async function createBuild(meshId: string, sizeStuds: number, signal?: AbortSignal): Promise<Build> {
  return json(
    await fetch("/api/builds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mesh_id: meshId, size_studs: sizeStuds }),
      signal,
    }),
  );
}

export async function health(): Promise<{ model_available: boolean; model_reason: string }> {
  return json(await fetch("/api/health"));
}

export const exportUrl = (buildId: string, fmt: "ldraw" | "bricklink" | "csv") =>
  `/api/builds/${buildId}/export/${fmt}`;
