from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .detector import detect_expression_candidates
from .models import BBox, ExpressionBox, ExpressionQuality, PageItem, Project, now_iso
from .storage import CROPS_DIR, DATA_DIR, EXPORTS_DIR, PROCESSED_DIR, UPLOADS_DIR, clamp_bbox


def static_to_path(url: str) -> Path:
    rel = url.replace("/static/", "")
    return UPLOADS_DIR.parent / rel


def save_crop(page: PageItem, expr: ExpressionBox, binary: np.ndarray | None = None) -> ExpressionBox:
    image_path = static_to_path(page.imageUrl)
    image = Image.open(image_path).convert("RGB")
    bbox = clamp_bbox(expr.bbox, page.width, page.height)
    crop = image.crop((int(bbox.x), int(bbox.y), int(bbox.x + bbox.width), int(bbox.y + bbox.height)))
    crop_name = f"{page.id}_{expr.id}.png"
    crop.save(CROPS_DIR / crop_name)
    expr.cropPreviewUrl = f"/static/crops/{crop_name}"

    if binary is not None:
        x, y, w, h = map(int, [bbox.x, bbox.y, bbox.width, bbox.height])
        bin_crop = 255 - binary[y : y + h, x : x + w]
        bin_name = f"{page.id}_{expr.id}_binary.png"
        cv2.imwrite(str(CROPS_DIR / bin_name), bin_crop)
        expr.binaryPreviewUrl = f"/static/crops/{bin_name}"
    expr.updatedAt = now_iso()
    return expr


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )


def components(binary: np.ndarray) -> list[BBox]:
    height, width = binary.shape[:2]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    items: list[BBox] = []
    for x, y, w, h, area in stats[1:]:
        if area < 8 or w < 2 or h < 2:
            continue
        if area > width * height * 0.2:
            continue
        items.append(BBox(x=float(x), y=float(y), width=float(w), height=float(h)))
    return items


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    items = list(boxes)
    x1 = min(b.x for b in items)
    y1 = min(b.y for b in items)
    x2 = max(b.x + b.width for b in items)
    y2 = max(b.y + b.height for b in items)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def group_expressions(items: list[BBox], shape: tuple[int, int]) -> list[BBox]:
    if not items:
        return []
    median_h = float(np.median([b.height for b in items]))
    y_threshold = max(12.0, median_h * 1.4)
    x_gap_threshold = max(48.0, median_h * 5.0)
    padded: list[BBox] = []

    rows: list[list[BBox]] = []
    for box in sorted(items, key=lambda b: (b.y + b.height / 2, b.x)):
        center = box.y + box.height / 2
        target = None
        for row in rows:
            row_box = union_bbox(row)
            row_center = row_box.y + row_box.height / 2
            if abs(center - row_center) <= y_threshold or _overlap_y(box, row_box) > 0.25:
                target = row
                break
        if target is None:
            rows.append([box])
        else:
            target.append(box)

    for row in rows:
        row = sorted(row, key=lambda b: b.x)
        chunk: list[BBox] = []
        prev: BBox | None = None
        for box in row:
            if prev and box.x - (prev.x + prev.width) > x_gap_threshold and chunk:
                padded.append(union_bbox(chunk))
                chunk = []
            chunk.append(box)
            prev = box
        if chunk:
            padded.append(union_bbox(chunk))

    merged: list[BBox] = []
    for box in sorted(padded, key=lambda b: (b.y, b.x)):
        if merged and _should_vertical_merge(merged[-1], box, median_h):
            merged[-1] = union_bbox([merged[-1], box])
        else:
            merged.append(box)

    height, width = shape
    result = []
    for box in merged:
        pad = max(8, int(median_h * 0.6))
        result.append(
            clamp_bbox(
                BBox(x=box.x - pad, y=box.y - pad, width=box.width + pad * 2, height=box.height + pad * 2),
                width,
                height,
            )
        )
    return sort_reading_order(result)


def _overlap_y(a: BBox, b: BBox) -> float:
    top = max(a.y, b.y)
    bottom = min(a.y + a.height, b.y + b.height)
    return max(0.0, bottom - top) / max(1.0, min(a.height, b.height))


def _should_vertical_merge(a: BBox, b: BBox, median_h: float) -> bool:
    gap = b.y - (a.y + a.height)
    overlap_x = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    overlap_ratio = overlap_x / max(1.0, min(a.width, b.width))
    return gap <= max(18.0, median_h * 1.4) and overlap_ratio > 0.2


def sort_reading_order(boxes: list[BBox]) -> list[BBox]:
    if not boxes:
        return []
    median_h = max(20.0, float(np.median([b.height for b in boxes])))
    return sorted(boxes, key=lambda b: (round((b.y + b.height / 2) / (median_h * 1.8)), b.x))


def quality_for(binary: np.ndarray, bbox: BBox, page_shape: tuple[int, int]) -> ExpressionQuality:
    height, width = page_shape
    x, y, w, h = map(int, [bbox.x, bbox.y, bbox.width, bbox.height])
    crop = binary[y : y + h, x : x + w]
    foreground = float(np.count_nonzero(crop)) / max(1, crop.size)
    aspect = bbox.width / max(1.0, bbox.height)
    warnings: list[str] = []
    touch = bbox.x <= 2 or bbox.y <= 2 or bbox.x + bbox.width >= width - 2 or bbox.y + bbox.height >= height - 2
    if foreground < 0.005:
        warnings.extend(["too_sparse", "maybe_noise"])
    if foreground > 0.35:
        warnings.append("too_dense")
    if touch:
        warnings.append("touching_border")
    if bbox.height < 24 or bbox.width < 24:
        warnings.append("too_small")
    if aspect > 20:
        warnings.append("too_wide_maybe_multiple")
    if bbox.height / max(1.0, bbox.width) > 3:
        warnings.append("too_tall")
    maybe_multi = aspect > 12 or bbox.height > height * 0.18
    if maybe_multi:
        warnings.append("maybe_multiple_expressions")
    return ExpressionQuality(
        foregroundRatio=round(foreground, 4),
        aspectRatio=round(aspect, 3),
        touchBorder=touch,
        maybeMultipleExpressions=maybe_multi,
        maybeNoise="maybe_noise" in warnings,
        tooSmall="too_small" in warnings,
        tooLarge=bbox.width * bbox.height > width * height * 0.5,
        warnings=sorted(set(warnings)),
    )


def scan_page(page: PageItem) -> PageItem:
    image_path = static_to_path(page.imageUrl)
    image = cv2.imread(str(image_path))
    if image is None:
        page.status = "error"
        return page

    candidates, cleaned_gray, binary, comp_boxes = detect_expression_candidates(page.id, image)
    cleaned = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2BGR)

    processed_prefix = page.id
    cv2.imwrite(str(PROCESSED_DIR / f"{processed_prefix}_binary.png"), 255 - binary)
    cv2.imwrite(str(PROCESSED_DIR / f"{processed_prefix}_cleaned.png"), cleaned)
    components_img = image.copy()
    for box in comp_boxes:
        cv2.rectangle(
            components_img,
            (int(box.x), int(box.y)),
            (int(box.x + box.width), int(box.y + box.height)),
            (37, 99, 235),
            1,
        )
    cv2.imwrite(str(PROCESSED_DIR / f"{processed_prefix}_components.png"), components_img)
    page.layers.update(
        {
            "original": page.imageUrl,
            "cleaned": f"/static/processed/{processed_prefix}_cleaned.png",
            "binary": f"/static/processed/{processed_prefix}_binary.png",
            "components": f"/static/processed/{processed_prefix}_components.png",
        }
    )

    expressions: list[ExpressionBox] = []
    for idx, candidate in enumerate(candidates, start=1):
        quality = candidate.quality
        expr = ExpressionBox(
            id=f"{page.id}_expr_{idx:03d}",
            pageId=page.id,
            bbox=candidate.bbox,
            order=idx,
            status=candidate.status,
            quality=quality,
            candidateType=candidate.candidate_type,
            warnings=candidate.warnings,
            createdBy="auto",
            history=[{"at": now_iso(), "action": "auto_scan", "by": "auto"}],
        )
        expressions.append(save_crop(page, expr, binary))

    page.expressions = expressions
    page.status = "need_review" if any(e.status == "need_review" for e in expressions) else "scanned"
    page.updatedAt = now_iso()
    return page


def refresh_expression(page: PageItem, expr: ExpressionBox) -> ExpressionBox:
    image = cv2.imread(str(static_to_path(page.imageUrl)))
    binary = binarize(image) if image is not None else None
    if binary is not None:
        expr.quality = quality_for(binary, clamp_bbox(expr.bbox, page.width, page.height), binary.shape)
        expr.warnings = expr.quality.warnings
        if expr.candidateType == "uncertain" and not expr.quality.warnings:
            expr.candidateType = "single_expression"
    return save_crop(page, expr, binary)


def export_project(project: Project, statuses: list[str], include_crops: bool) -> str:
    export_root = EXPORTS_DIR / f"export_{project.id}"
    if export_root.exists():
        shutil.rmtree(export_root)
    images_dir = export_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project_id": project.id,
        "project_name": project.name,
        "created_at": now_iso(),
        "pages": [],
    }
    jsonl: list[str] = []
    for page in project.pages:
        page_meta = {
            "page_id": page.id,
            "source_image": page.fileName,
            "width": page.width,
            "height": page.height,
            "expressions": [],
        }
        for expr in sorted(page.expressions, key=lambda e: e.order):
            if expr.status not in statuses:
                continue
            crop_file = f"images/{page.id}_{expr.id}.png"
            if include_crops and expr.cropPreviewUrl:
                src = static_to_path(expr.cropPreviewUrl)
                if src.exists():
                    shutil.copyfile(src, export_root / crop_file)
            row = {
                "id": f"{page.id}_{expr.id}",
                "source_image": page.fileName,
                "order": expr.order,
                "bbox": [expr.bbox.x, expr.bbox.y, expr.bbox.width, expr.bbox.height],
                "status": expr.status,
                "warnings": expr.quality.warnings,
                "crop_file": crop_file,
            }
            page_meta["expressions"].append(row)
            jsonl.append(json.dumps(row, ensure_ascii=False))
        metadata["pages"].append(page_meta)

    (export_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (export_root / "metadata.jsonl").write_text("\n".join(jsonl), encoding="utf-8")
    (export_root / "project.json").write_text(project.model_dump_json(indent=2), encoding="utf-8")

    zip_path = EXPORTS_DIR / f"{project.id}_export.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in export_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(export_root))
    return f"/static/exports/{zip_path.name}"


def export_crohme_m4_dataset(project: Project, mode: str = "accepted_recognized") -> str:
    export_root = EXPORTS_DIR / f"export_crohme_m4_{project.id}"
    if export_root.exists():
        shutil.rmtree(export_root)
    for folder in ["pages", "crops_original", "crops_crohme_like", "latex", "render", "debug"]:
        (export_root / folder).mkdir(parents=True, exist_ok=True)

    metadata = {"project_id": project.id, "project_name": project.name, "created_at": now_iso(), "pages": []}
    jsonl: list[str] = []

    for page in project.pages:
        page_source = static_to_path(page.imageUrl)
        page_ext = page_source.suffix or ".png"
        page_file = f"pages/{page.id}_original{page_ext}"
        if page_source.exists():
            shutil.copyfile(page_source, export_root / page_file)
        page_meta = {"page_id": page.id, "source_image": page_file, "expressions": []}

        for expr in sorted(page.expressions, key=lambda e: e.order):
            if not _include_expr_for_m4_export(expr, mode):
                continue
            original_file = f"crops_original/{expr.id}.png"
            normalized_file = f"crops_crohme_like/{expr.id}.png"
            if expr.cropPreviewUrl and static_to_path(expr.cropPreviewUrl).exists():
                shutil.copyfile(static_to_path(expr.cropPreviewUrl), export_root / original_file)
            if expr.normalizedUrl and static_to_path(expr.normalizedUrl).exists():
                shutil.copyfile(static_to_path(expr.normalizedUrl), export_root / normalized_file)
            if expr.latexClean:
                (export_root / "latex" / f"{expr.id}.txt").write_text(expr.latexClean, encoding="utf-8")

            debug_src = DATA_DIR / "expressions" / expr.id / "normalization_debug.json"
            if debug_src.exists():
                shutil.copyfile(debug_src, export_root / "debug" / f"{expr.id}_normalization.json")
            comp_src = DATA_DIR / "expressions" / expr.id / "components_debug.png"
            if comp_src.exists():
                shutil.copyfile(comp_src, export_root / "debug" / f"{expr.id}_components.png")

            row = {
                "id": expr.id,
                "page_id": page.id,
                "source_image": page_file,
                "bbox": {"x": expr.bbox.x, "y": expr.bbox.y, "w": expr.bbox.width, "h": expr.bbox.height},
                "status": expr.status,
                "candidate_type": expr.candidateType,
                "normalized_image": normalized_file if expr.normalizedUrl else None,
                "latex_raw": expr.latexRaw,
                "latex_clean": expr.latexClean,
                "latex_status": expr.latexStatus,
                "quality": expr.quality.model_dump(),
                "warnings": sorted(set([*expr.warnings, *expr.quality.warnings])),
            }
            page_meta["expressions"].append(row)
            jsonl.append(json.dumps(row, ensure_ascii=False))
        metadata["pages"].append(page_meta)

    (export_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (export_root / "metadata.jsonl").write_text("\n".join(jsonl), encoding="utf-8")
    (export_root / "project.json").write_text(project.model_dump_json(indent=2), encoding="utf-8")

    zip_path = EXPORTS_DIR / f"export_crohme_m4_dataset_{project.id}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in export_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(export_root))
    return f"/static/exports/{zip_path.name}"


def _include_expr_for_m4_export(expr: ExpressionBox, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "accepted_only":
        return expr.status == "accepted"
    if mode == "m4_ready_only":
        return bool(expr.normalizedUrl) and expr.candidateType == "single_expression"
    return expr.status == "accepted" and expr.latexStatus in ("ok", "manual")
