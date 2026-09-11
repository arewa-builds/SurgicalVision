import type { AnalysisResult } from "./types";

export type DemoProfile = "suturing" | "knot_tying" | "needle_passing";

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function startDemo(profile: DemoProfile): Promise<AnalysisResult> {
  return parse(
    await fetch("/api/analyses/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile }),
    })
  );
}

export async function uploadVideo(file: File): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("file", file);
  return parse(await fetch("/api/analyses", { method: "POST", body }));
}

export async function getAnalysis(id: string): Promise<AnalysisResult> {
  return parse(await fetch(`/api/analyses/${id}`));
}

export function overlayUrl(id: string): string {
  return `/api/analyses/${id}/overlay`;
}
