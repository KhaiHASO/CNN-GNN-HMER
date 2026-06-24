import axios from "axios";
import type { BBox, ExpressionBox, ExpressionStatus, PageItem, Project } from "./types";

export const api = axios.create({ baseURL: "" });

export const assetUrl = (url?: string) => (url ? url : "");

export async function ensureProject(): Promise<Project> {
  const { data } = await api.get<Project[]>("/api/projects");
  return data[0];
}

export async function uploadPages(projectId: string, files: FileList): Promise<PageItem[]> {
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const { data } = await api.post<{ pages: PageItem[] }>(`/api/projects/${projectId}/pages/upload`, body);
  return data.pages;
}

export async function scanPage(pageId: string): Promise<ExpressionBox[]> {
  const { data } = await api.post<{ expressions: ExpressionBox[] }>(`/api/pages/${pageId}/scan`, {
    preset: "white_paper",
    detectMode: "classical_cv",
    saveLayers: true
  });
  return data.expressions;
}

export async function getPage(pageId: string): Promise<PageItem> {
  const { data } = await api.get<PageItem>(`/api/pages/${pageId}`);
  return data;
}

export async function deletePage(pageId: string): Promise<void> {
  await api.delete(`/api/pages/${pageId}`);
}

export async function clearPageExpressions(pageId: string): Promise<void> {
  await api.delete(`/api/pages/${pageId}/expressions`);
}

export async function createExpression(pageId: string, bbox: BBox): Promise<ExpressionBox> {
  const { data } = await api.post<ExpressionBox>(`/api/pages/${pageId}/expressions`, { bbox, status: "edited" });
  return data;
}

export async function patchExpression(id: string, payload: { bbox?: BBox; status?: ExpressionStatus }): Promise<ExpressionBox> {
  const { data } = await api.patch<ExpressionBox>(`/api/expressions/${id}`, payload);
  return data;
}

export async function deleteExpression(id: string): Promise<void> {
  await api.delete(`/api/expressions/${id}`);
}

export async function mergeExpressions(pageId: string, expressionIds: string[]) {
  const { data } = await api.post("/api/expressions/merge", { pageId, expressionIds });
  return data as { mergedExpression: ExpressionBox; removedExpressionIds: string[] };
}

export async function splitExpression(id: string, mode: "horizontal" | "vertical", position = 0.5) {
  const { data } = await api.post(`/api/expressions/${id}/split`, { mode, position });
  return data as { createdExpressions: ExpressionBox[]; removedExpressionId: string };
}

export async function reorderExpressions(pageId: string, orderedExpressionIds: string[]) {
  const { data } = await api.post<ExpressionBox[]>(`/api/pages/${pageId}/expressions/reorder`, { orderedExpressionIds });
  return data;
}

export async function exportProject(projectId: string) {
  const { data } = await api.post<{ downloadUrl: string }>(`/api/projects/${projectId}/export`, {
    includeStatuses: ["accepted"],
    includeCrops: true,
    includeOverlays: true,
    format: "jsonl"
  });
  return data.downloadUrl;
}

export async function normalizeExpression(expressionId: string): Promise<ExpressionBox> {
  const { data } = await api.post<{ expression: ExpressionBox }>(`/api/expressions/${expressionId}/normalize`);
  return data.expression;
}

export async function normalizePage(pageId: string): Promise<ExpressionBox[]> {
  const { data } = await api.post<{ normalized: ExpressionBox[] }>(`/api/pages/${pageId}/normalize-all`, {
    only_status: ["accepted", "need_review", "edited", "auto_detected"],
    skip_noise: true,
    profile: "m4_crohme_like"
  });
  return data.normalized;
}

export async function recognizeExpression(expressionId: string): Promise<ExpressionBox> {
  const { data } = await api.post<{ expression: ExpressionBox }>(`/api/expressions/${expressionId}/recognize`);
  return data.expression;
}

export async function recognizeAccepted(pageId: string): Promise<ExpressionBox[]> {
  const { data } = await api.post<{ recognized: ExpressionBox[] }>(`/api/pages/${pageId}/recognize-accepted`);
  return data.recognized;
}

export async function updateLatex(expressionId: string, latexClean: string): Promise<ExpressionBox> {
  const { data } = await api.patch<ExpressionBox>(`/api/expressions/${expressionId}/latex`, {
    latex_clean: latexClean,
    manual_override: true
  });
  return data;
}

export async function exportCrohmeM4(projectId: string, mode = "accepted_recognized") {
  const { data } = await api.post<{ downloadUrl: string }>("/api/export/crohme-m4", { projectId, mode });
  return data.downloadUrl;
}
