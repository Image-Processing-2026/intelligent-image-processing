import { useQuery } from "@tanstack/react-query";

import { apiFetch } from "./client";
import type {
  MemoryStats,
  Rating,
  Sample,
  SessionDetail,
  SessionSummary,
  SystemInfo,
} from "./types";

export const SESSIONS_KEY = ["workspace", "sessions"] as const;
export const sessionKey = (id: string) => ["workspace", "session", id] as const;

const sessionPath = (id: string) => `/workspace/sessions/${encodeURIComponent(id)}`;

export function useSessions() {
  return useQuery({
    queryKey: SESSIONS_KEY,
    queryFn: () => apiFetch<SessionSummary[]>("/workspace/sessions"),
  });
}

export function useSession(id: string | undefined) {
  return useQuery({
    queryKey: sessionKey(id ?? ""),
    queryFn: () => apiFetch<SessionDetail>(sessionPath(id ?? "")),
    enabled: Boolean(id),
  });
}

export function useSamples() {
  return useQuery({
    queryKey: ["workspace", "samples"],
    queryFn: () => apiFetch<Sample[]>("/workspace/samples"),
    staleTime: Infinity,
  });
}

export interface NewSessionInput {
  file?: File | null;
  sample?: string | null;
  groundTruth?: File | null;
  maxIterations: number;
}

export function createSession(input: NewSessionInput): Promise<SessionDetail> {
  const form = new FormData();
  if (input.file) form.append("file", input.file);
  if (input.sample) form.append("sample", input.sample);
  if (input.groundTruth) form.append("ground_truth", input.groundTruth);
  form.append("max_iterations", String(input.maxIterations));
  return apiFetch<SessionDetail>("/workspace/sessions", { method: "POST", body: form });
}

export function renameSession(id: string, title: string): Promise<SessionSummary> {
  return apiFetch<SessionSummary>(sessionPath(id), { method: "PATCH", json: { title } });
}

export function deleteSession(id: string): Promise<void> {
  return apiFetch<void>(sessionPath(id), { method: "DELETE" });
}

export function answerQuestions(
  id: string,
  answers: Record<string, string>,
  notes: string,
): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`${sessionPath(id)}/answers`, {
    method: "POST",
    json: { answers, notes },
  });
}

export function addAnalysisNote(id: string, message: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`${sessionPath(id)}/analysis-notes`, {
    method: "POST",
    json: { message },
  });
}

export function deleteAnalysisEdit(id: string, index: number): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`${sessionPath(id)}/analysis-edits/${index}`, {
    method: "DELETE",
  });
}

export function clarifyAgain(id: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`${sessionPath(id)}/clarify`, { method: "POST" });
}

export function renderVariant(id: string, variantId: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(
    `${sessionPath(id)}/variants/${encodeURIComponent(variantId)}/render`,
    { method: "POST" },
  );
}

export function refineTreatment(
  id: string,
  feedback: string,
  treatmentId?: string | null,
): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`${sessionPath(id)}/refine`, {
    method: "POST",
    json: treatmentId ? { feedback, treatment_id: treatmentId } : { feedback },
  });
}

export function activateTreatment(id: string, treatmentId: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(
    `${sessionPath(id)}/treatments/${encodeURIComponent(treatmentId)}/activate`,
    { method: "POST" },
  );
}

export interface RatingInput {
  run_id: string;
  treatment_id: string | null;
  rating: "good" | "bad";
  note: string;
}

export function saveRating(id: string, input: RatingInput): Promise<Rating> {
  return apiFetch<Rating>(`${sessionPath(id)}/ratings`, { method: "PUT", json: input });
}

/** URL tải file JSON của phiên (cookie cùng origin nên dùng trực tiếp trong thẻ <a>). */
export function exportUrl(id: string): string {
  return `/api/v1${sessionPath(id)}/export`;
}

export const SYSTEM_KEY = ["workspace", "system"] as const;

export function useSystemInfo() {
  return useQuery({
    queryKey: SYSTEM_KEY,
    queryFn: () => apiFetch<SystemInfo>("/workspace/system"),
    staleTime: 5 * 60 * 1000,
  });
}

export function useMemoryStats() {
  return useQuery({
    queryKey: ["workspace", "memory"],
    queryFn: () => apiFetch<MemoryStats>("/workspace/memory"),
  });
}
