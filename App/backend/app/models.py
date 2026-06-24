from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ExpressionQuality(BaseModel):
    foregroundRatio: Optional[float] = None
    aspectRatio: Optional[float] = None
    touchBorder: Optional[bool] = None
    maybeMultipleExpressions: Optional[bool] = None
    maybeNoise: Optional[bool] = None
    tooSmall: Optional[bool] = None
    tooLarge: Optional[bool] = None
    componentCount: Optional[int] = None
    isMultiline: Optional[bool] = None
    isFragment: Optional[bool] = None
    emptyAfterNormalize: Optional[bool] = None
    warnings: List[str] = Field(default_factory=list)


ExpressionStatus = Literal[
    "auto_detected",
    "need_review",
    "accepted",
    "rejected",
    "edited",
    "exported",
    "noise",
    "fragment",
]

CandidateType = Literal[
    "single_expression",
    "multiline_block",
    "fragment",
    "noise",
    "uncertain",
]

LatexStatus = Literal[
    "not_run",
    "running",
    "ok",
    "syntax_error",
    "model_error",
    "manual",
]

PageStatus = Literal[
    "unscanned",
    "scanning",
    "scanned",
    "need_review",
    "completed",
    "error",
]


class ExpressionBox(BaseModel):
    id: str
    pageId: str
    bbox: BBox
    order: int
    status: ExpressionStatus
    quality: ExpressionQuality = Field(default_factory=ExpressionQuality)
    cropPreviewUrl: Optional[str] = None
    cleanedPreviewUrl: Optional[str] = None
    binaryPreviewUrl: Optional[str] = None
    normalizedPreviewUrl: Optional[str] = None
    normalizedUrl: Optional[str] = None
    componentsPreviewUrl: Optional[str] = None
    candidateType: CandidateType = "uncertain"
    warnings: List[str] = Field(default_factory=list)
    latexRaw: Optional[str] = None
    latexClean: Optional[str] = None
    latexStatus: LatexStatus = "not_run"
    latexConfidence: Optional[float] = None
    latexRenderSvgUrl: Optional[str] = None
    latexRenderPngUrl: Optional[str] = None
    recognitionHistory: List[Dict[str, Any]] = Field(default_factory=list)
    createdBy: Literal["auto", "manual"] = "auto"
    history: List[Dict[str, Any]] = Field(default_factory=list)
    createdAt: str = Field(default_factory=now_iso)
    updatedAt: str = Field(default_factory=now_iso)


class PageItem(BaseModel):
    id: str
    fileName: str
    imageUrl: str
    thumbnailUrl: Optional[str] = None
    width: int
    height: int
    status: PageStatus = "unscanned"
    expressions: List[ExpressionBox] = Field(default_factory=list)
    layers: Dict[str, str] = Field(default_factory=dict)
    createdAt: str = Field(default_factory=now_iso)
    updatedAt: str = Field(default_factory=now_iso)


class Project(BaseModel):
    id: str
    name: str
    pages: List[PageItem] = Field(default_factory=list)
    createdAt: str = Field(default_factory=now_iso)
    updatedAt: str = Field(default_factory=now_iso)


class CreateProjectRequest(BaseModel):
    name: str = "CROHME-like A4 Dataset"


class ScanRequest(BaseModel):
    preset: str = "white_paper"
    detectMode: str = "classical_cv"
    saveLayers: bool = True


class ExpressionCreateRequest(BaseModel):
    bbox: BBox
    status: ExpressionStatus = "edited"


class ExpressionPatchRequest(BaseModel):
    bbox: Optional[BBox] = None
    status: Optional[ExpressionStatus] = None


class MergeRequest(BaseModel):
    pageId: str
    expressionIds: List[str]


class SplitRequest(BaseModel):
    mode: Literal["horizontal", "vertical"]
    position: float = 0.5


class ReorderRequest(BaseModel):
    orderedExpressionIds: List[str]


class ExportRequest(BaseModel):
    includeStatuses: List[ExpressionStatus] = Field(default_factory=lambda: ["accepted"])
    includeCrops: bool = True
    includeOverlays: bool = True
    format: Literal["json", "jsonl"] = "jsonl"


class NormalizeAllRequest(BaseModel):
    only_status: List[ExpressionStatus] = Field(default_factory=lambda: ["accepted", "need_review", "edited", "auto_detected"])
    skip_noise: bool = True
    profile: str = "m4_crohme_like"


class LatexPatchRequest(BaseModel):
    latex_clean: str
    manual_override: bool = True


class CrohmeM4ExportRequest(BaseModel):
    projectId: Optional[str] = None
    mode: Literal["all", "accepted_only", "accepted_recognized", "m4_ready_only"] = "accepted_recognized"
