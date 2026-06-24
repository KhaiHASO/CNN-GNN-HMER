export type PageStatus = "unscanned" | "scanning" | "scanned" | "need_review" | "completed" | "error";
export type ExpressionStatus = "auto_detected" | "need_review" | "accepted" | "rejected" | "edited" | "exported" | "noise" | "fragment";
export type Tool = "select" | "pan" | "draw" | "split" | "merge";
export type LayerKey = "original" | "cleaned" | "binary" | "components" | "expressions" | "normalized" | "m4_ready";
export type CandidateType = "single_expression" | "multiline_block" | "fragment" | "noise" | "uncertain";
export type LatexStatus = "not_run" | "running" | "ok" | "syntax_error" | "model_error" | "manual";

export type BBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type ExpressionQuality = {
  foregroundRatio?: number;
  aspectRatio?: number;
  touchBorder?: boolean;
  maybeMultipleExpressions?: boolean;
  maybeNoise?: boolean;
  tooSmall?: boolean;
  tooLarge?: boolean;
  componentCount?: number;
  isMultiline?: boolean;
  isFragment?: boolean;
  emptyAfterNormalize?: boolean;
  warnings: string[];
};

export type ExpressionBox = {
  id: string;
  pageId: string;
  bbox: BBox;
  order: number;
  status: ExpressionStatus;
  quality: ExpressionQuality;
  cropPreviewUrl?: string;
  cleanedPreviewUrl?: string;
  binaryPreviewUrl?: string;
  normalizedPreviewUrl?: string;
  normalizedUrl?: string;
  componentsPreviewUrl?: string;
  candidateType: CandidateType;
  warnings: string[];
  latexRaw?: string;
  latexClean?: string;
  latexStatus: LatexStatus;
  latexConfidence?: number;
  latexRenderSvgUrl?: string;
  latexRenderPngUrl?: string;
  recognitionHistory: { [key: string]: unknown }[];
  createdBy: "auto" | "manual";
  history: { at: string; action: string; by: "auto" | "user"; payload?: Record<string, unknown> }[];
  createdAt: string;
  updatedAt: string;
};

export type PageItem = {
  id: string;
  fileName: string;
  imageUrl: string;
  thumbnailUrl?: string;
  width: number;
  height: number;
  status: PageStatus;
  expressions: ExpressionBox[];
  layers: Record<string, string>;
  createdAt: string;
  updatedAt: string;
};

export type Project = {
  id: string;
  name: string;
  pages: PageItem[];
  createdAt: string;
  updatedAt: string;
};
