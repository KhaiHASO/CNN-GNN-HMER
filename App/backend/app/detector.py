from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import BBox, ExpressionQuality
from .storage import clamp_bbox


@dataclass
class ExpressionCandidate:
    bbox: BBox
    quality: ExpressionQuality
    candidate_type: str
    status: str
    warnings: list[str]


def preprocess_page_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 31)
    corrected = cv2.divide(gray, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cleaned = clahe.apply(corrected)
    cleaned = cv2.medianBlur(cleaned, 3)
    binary_bow = cv2.adaptiveThreshold(
        cleaned,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        15,
    )
    if np.mean(binary_bow == 0) > 0.5:
        binary_bow = 255 - binary_bow
    binary_inv = 255 - binary_bow
    return gray, cleaned, binary_inv


def connected_component_boxes(binary_inv: np.ndarray) -> list[BBox]:
    height, width = binary_inv.shape[:2]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary_inv, 8)
    boxes: list[BBox] = []
    for x, y, w, h, area in stats[1:]:
        if area < 8 or w < 2 or h < 2:
            continue
        if area > width * height * 0.18:
            continue
        aspect = w / max(1, h)
        if aspect > 45 and h < 25:
            continue
        if aspect < 0.04 and w < 25:
            continue
        if (x <= 3 or y <= 3 or x + w >= width - 3 or y + h >= height - 3) and area < 80:
            continue
        boxes.append(BBox(x=float(x), y=float(y), width=float(w), height=float(h)))
    return boxes


def detect_expression_candidates(page_id: str, image_bgr: np.ndarray) -> tuple[list[ExpressionCandidate], np.ndarray, np.ndarray, list[BBox]]:
    _, cleaned, binary_inv = preprocess_page_for_detection(image_bgr)
    comp_boxes = connected_component_boxes(binary_inv)
    expr_boxes = _group_boxes(comp_boxes, binary_inv.shape)
    candidates = [_classify_candidate(box, binary_inv, comp_boxes) for box in expr_boxes]
    return candidates, cleaned, binary_inv, comp_boxes


def _group_boxes(boxes: list[BBox], shape: tuple[int, int]) -> list[BBox]:
    if not boxes:
        return []
    median_h = max(8.0, float(np.median([b.height for b in boxes])))
    rows: list[list[BBox]] = []
    for box in sorted(boxes, key=lambda b: (b.y + b.height / 2, b.x)):
        center = box.y + box.height / 2
        target = None
        for row in rows:
            row_box = _union(row)
            row_center = row_box.y + row_box.height / 2
            if abs(center - row_center) <= median_h * 1.45 or _overlap_y(box, row_box) > 0.25:
                target = row
                break
        if target is None:
            rows.append([box])
        else:
            target.append(box)

    chunks: list[BBox] = []
    x_gap = max(160.0, median_h * 16.0)
    for row in rows:
        current: list[BBox] = []
        previous: BBox | None = None
        for box in sorted(row, key=lambda b: b.x):
            if previous and box.x - (previous.x + previous.width) > x_gap and current:
                chunks.append(_union(current))
                current = []
            current.append(box)
            previous = box
        if current:
            chunks.append(_union(current))

    merged: list[BBox] = []
    for box in sorted(chunks, key=lambda b: (b.y, b.x)):
        if merged and _should_merge_vertical(merged[-1], box, median_h):
            merged[-1] = _union([merged[-1], box])
        else:
            merged.append(box)

    h, w = shape
    padded = []
    for box in merged:
        pad = max(8, int(median_h * 0.65))
        padded.append(clamp_bbox(BBox(x=box.x - pad, y=box.y - pad, width=box.width + pad * 2, height=box.height + pad * 2), w, h))
    row_bucket = max(24.0, float(np.median([b.height for b in padded])) * 1.8)
    return sorted(padded, key=lambda b: (round((b.y + b.height / 2) / row_bucket), b.x))


def _classify_candidate(box: BBox, binary_inv: np.ndarray, comp_boxes: list[BBox]) -> ExpressionCandidate:
    h_img, w_img = binary_inv.shape[:2]
    x, y, w, h = map(int, [box.x, box.y, box.width, box.height])
    crop = binary_inv[y : y + h, x : x + w]
    foreground_ratio = float(np.count_nonzero(crop)) / max(1, crop.size)
    aspect = box.width / max(1.0, box.height)
    touch = box.x <= 3 or box.y <= 3 or box.x + box.width >= w_img - 3 or box.y + box.height >= h_img - 3
    component_count = sum(_inside(c, box) for c in comp_boxes)
    warnings: list[str] = []

    if box.width * box.height < 0.0001 * w_img * h_img or box.width < 12 or box.height < 12:
        warnings.append("TOO_SMALL")
    if foreground_ratio < 0.005:
        warnings.extend(["LOW_FOREGROUND_RATIO", "POSSIBLE_NOISE"])
    if foreground_ratio > 0.35:
        warnings.append("HIGH_FOREGROUND_RATIO")
    if touch:
        warnings.append("TOUCH_BORDER")
    if aspect > 35 and box.height < 25:
        warnings.append("POSSIBLE_RULE_LINE")
    if _is_multiline(crop, box):
        warnings.append("MULTILINE_BLOCK_SUGGEST_SPLIT_H")

    candidate_type = "single_expression"
    status = "auto_detected"
    if "POSSIBLE_RULE_LINE" in warnings or "POSSIBLE_NOISE" in warnings:
        candidate_type = "noise"
        status = "noise"
    elif component_count <= 1 and (box.width < 40 or box.height < 30):
        candidate_type = "fragment"
        status = "fragment"
        warnings.append("FRAGMENT")
    elif "MULTILINE_BLOCK_SUGGEST_SPLIT_H" in warnings:
        candidate_type = "multiline_block"
        status = "need_review"
    elif warnings:
        candidate_type = "uncertain"
        status = "need_review"

    quality = ExpressionQuality(
        foregroundRatio=round(foreground_ratio, 4),
        aspectRatio=round(aspect, 3),
        touchBorder=touch,
        maybeMultipleExpressions=candidate_type == "multiline_block",
        maybeNoise=candidate_type == "noise",
        tooSmall="TOO_SMALL" in warnings,
        tooLarge=box.width * box.height > w_img * h_img * 0.5,
        componentCount=component_count,
        isMultiline=candidate_type == "multiline_block",
        isFragment=candidate_type == "fragment",
        warnings=sorted(set(warnings)),
    )
    return ExpressionCandidate(box, quality, candidate_type, status, quality.warnings)


def _is_multiline(crop: np.ndarray, box: BBox) -> bool:
    if box.height < 70:
        return False
    projection = np.count_nonzero(crop, axis=1).astype(np.float32)
    if projection.max(initial=0) <= 0:
        return False
    projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 11), 0).ravel()
    active = projection > max(2.0, projection.max() * 0.12)
    runs = []
    start = None
    for idx, value in enumerate(active):
        if value and start is None:
            start = idx
        if not value and start is not None:
            if idx - start > 6:
                runs.append((start, idx))
            start = None
    if start is not None and len(active) - start > 6:
        runs.append((start, len(active)))
    return len(runs) >= 2 and box.height > 80


def _union(boxes: list[BBox]) -> BBox:
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x + b.width for b in boxes)
    y2 = max(b.y + b.height for b in boxes)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _inside(inner: BBox, outer: BBox) -> bool:
    cx = inner.x + inner.width / 2
    cy = inner.y + inner.height / 2
    return outer.x <= cx <= outer.x + outer.width and outer.y <= cy <= outer.y + outer.height


def _overlap_y(a: BBox, b: BBox) -> float:
    top = max(a.y, b.y)
    bottom = min(a.y + a.height, b.y + b.height)
    return max(0.0, bottom - top) / max(1.0, min(a.height, b.height))


def _should_merge_vertical(a: BBox, b: BBox, median_h: float) -> bool:
    gap = b.y - (a.y + a.height)
    overlap = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    return gap <= max(14.0, median_h * 1.15) and overlap / max(1.0, min(a.width, b.width)) > 0.25
