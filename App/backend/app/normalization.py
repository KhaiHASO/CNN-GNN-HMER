from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import BBox, ExpressionBox, ExpressionQuality, PageItem, now_iso
from .services import static_to_path
from .storage import DATA_DIR, clamp_bbox


NORMALIZATION_PROFILE = {
    "name": "m4_crohme_like",
    "background_value": 0,
    "foreground_value": 255,
    "padding_px": 16,
    "min_padding_px": 8,
    "target_height": 128,
    "max_width": 1024,
    "component_min_area": 12,
    "save_debug_steps": True,
}


@dataclass
class NormalizationResult:
    expression: ExpressionBox
    debug_path: Path


def expression_dir(expr: ExpressionBox) -> Path:
    path = DATA_DIR / "expressions" / expr.id
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_page_expression(page: PageItem, expr: ExpressionBox) -> NormalizationResult:
    image = cv2.imread(str(static_to_path(page.imageUrl)))
    if image is None:
        raise ValueError("Không đọc được ảnh gốc của page")
    return normalize_expression_crop(image, page, expr)


def normalize_expression_crop(image_bgr: np.ndarray, page: PageItem, expr: ExpressionBox) -> NormalizationResult:
    out_dir = expression_dir(expr)
    bbox = clamp_bbox(BBox(x=expr.bbox.x - 4, y=expr.bbox.y - 4, width=expr.bbox.width + 8, height=expr.bbox.height + 8), page.width, page.height)
    x, y, w, h = map(int, [bbox.x, bbox.y, bbox.width, bbox.height])
    original = image_bgr[y : y + h, x : x + w]
    cv2.imwrite(str(out_dir / "original_crop.png"), original)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    cleaned = cv2.divide(gray, background, scale=255)
    cleaned = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cleaned)
    cleaned = cv2.medianBlur(cleaned, 3)
    cv2.imwrite(str(out_dir / "cleaned_crop.png"), cleaned)

    binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 14)
    if np.mean(binary == 0) > 0.5:
        binary = 255 - binary
    binary = _remove_noise_black_on_white(binary, NORMALIZATION_PROFILE["component_min_area"])
    cv2.imwrite(str(out_dir / "binary_black_on_white.png"), binary)

    content = _crop_to_foreground(binary)
    warnings: list[str] = []
    if content is None:
        content = np.full((NORMALIZATION_PROFILE["target_height"], NORMALIZATION_PROFILE["target_height"]), 255, dtype=np.uint8)
        warnings.append("EMPTY_AFTER_NORMALIZE")
    content = _pad_white(content, NORMALIZATION_PROFILE["padding_px"])
    resized, width_too_large = _resize_to_height(content, NORMALIZATION_PROFILE["target_height"], NORMALIZATION_PROFILE["max_width"])
    if width_too_large:
        warnings.append("TOO_WIDE")
    normalized = 255 - resized
    normalized = np.where(normalized > 127, 255, 0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "normalized_crohme.png"), normalized)

    comp_debug, component_count = _components_debug(binary)
    cv2.imwrite(str(out_dir / "components_debug.png"), comp_debug)

    foreground_ratio = float(np.count_nonzero(normalized)) / max(1, normalized.size)
    touch = bool(
        np.any(normalized[0, :] > 0)
        or np.any(normalized[-1, :] > 0)
        or np.any(normalized[:, 0] > 0)
        or np.any(normalized[:, -1] > 0)
    )
    if touch:
        warnings.append("TOUCH_BORDER")
    if foreground_ratio < 0.005:
        warnings.append("LOW_FOREGROUND_RATIO")
    if foreground_ratio > 0.35:
        warnings.append("HIGH_FOREGROUND_RATIO")
    if normalized.shape[1] < 24:
        warnings.append("TOO_SMALL")
    is_multiline = _looks_multiline(255 - resized)
    if is_multiline:
        warnings.append("MULTILINE_BLOCK")

    quality = ExpressionQuality(
        foregroundRatio=round(foreground_ratio, 4),
        aspectRatio=round(normalized.shape[1] / max(1, normalized.shape[0]), 3),
        touchBorder=touch,
        maybeMultipleExpressions=is_multiline,
        maybeNoise=foreground_ratio < 0.005,
        tooSmall=normalized.shape[1] < 24,
        tooLarge=normalized.shape[1] >= NORMALIZATION_PROFILE["max_width"],
        componentCount=component_count,
        isMultiline=is_multiline,
        isFragment=expr.candidateType == "fragment",
        emptyAfterNormalize="EMPTY_AFTER_NORMALIZE" in warnings,
        warnings=sorted(set([*expr.quality.warnings, *expr.warnings, *warnings])),
    )

    debug = {
        "profile": NORMALIZATION_PROFILE,
        "expression_id": expr.id,
        "bbox": expr.bbox.model_dump(),
        "normalized_width": int(normalized.shape[1]),
        "normalized_height": int(normalized.shape[0]),
        "quality": quality.model_dump(),
        "warnings": quality.warnings,
        "created_at": now_iso(),
    }
    debug_path = out_dir / "normalization_debug.json"
    debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    base = f"/static/expressions/{expr.id}"
    expr.cropPreviewUrl = f"{base}/original_crop.png"
    expr.cleanedPreviewUrl = f"{base}/cleaned_crop.png"
    expr.binaryPreviewUrl = f"{base}/binary_black_on_white.png"
    expr.normalizedPreviewUrl = f"{base}/normalized_crohme.png"
    expr.normalizedUrl = expr.normalizedPreviewUrl
    expr.componentsPreviewUrl = f"{base}/components_debug.png"
    expr.quality = quality
    expr.warnings = quality.warnings
    if expr.candidateType == "uncertain" and not is_multiline and not quality.maybeNoise:
        expr.candidateType = "single_expression"
    if is_multiline and expr.candidateType != "noise":
        expr.candidateType = "multiline_block"
        if expr.status not in ("rejected", "noise", "fragment"):
            expr.status = "need_review"
    expr.history.append({"at": now_iso(), "action": "normalize_m4_ready", "by": "auto", "payload": {"warnings": quality.warnings}})
    expr.updatedAt = now_iso()
    return NormalizationResult(expression=expr, debug_path=debug_path)


def _remove_noise_black_on_white(binary: np.ndarray, min_area: int) -> np.ndarray:
    foreground = 255 - binary
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    keep = np.zeros_like(foreground)
    for idx, stat in enumerate(stats[1:], start=1):
        if stat[cv2.CC_STAT_AREA] >= min_area:
            keep[labels == idx] = 255
    return 255 - keep


def _crop_to_foreground(binary: np.ndarray) -> np.ndarray | None:
    foreground = np.where(binary < 128, 255, 0).astype(np.uint8)
    coords = cv2.findNonZero(foreground)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return binary[y : y + h, x : x + w]


def _pad_white(image: np.ndarray, pad: int) -> np.ndarray:
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _resize_to_height(image: np.ndarray, target_height: int, max_width: int) -> tuple[np.ndarray, bool]:
    scale = target_height / max(1, image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    width_too_large = width > max_width
    if width_too_large:
        width = max_width
    resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
    return np.where(resized > 127, 255, 0).astype(np.uint8), width_too_large


def _components_debug(binary: np.ndarray) -> tuple[np.ndarray, int]:
    foreground = 255 - binary
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    count = 0
    for x, y, w, h, area in stats[1:]:
        if area < 8:
            continue
        count += 1
        cv2.rectangle(debug, (x, y), (x + w, y + h), (37, 99, 235), 1)
    return debug, count


def _looks_multiline(binary_black_on_white: np.ndarray) -> bool:
    foreground = binary_black_on_white < 128
    projection = np.count_nonzero(foreground, axis=1).astype(np.float32)
    if projection.max(initial=0) <= 0:
        return False
    projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 9), 0).ravel()
    active = projection > max(2.0, projection.max() * 0.13)
    runs = 0
    in_run = False
    for value in active:
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return runs >= 2 and binary_black_on_white.shape[0] > 72

