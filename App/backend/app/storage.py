from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from .models import BBox, ExpressionBox, PageItem, Project, now_iso


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
CROPS_DIR = DATA_DIR / "crops"
EXPORTS_DIR = DATA_DIR / "exports"
EXPRESSIONS_DIR = DATA_DIR / "expressions"


def ensure_dirs() -> None:
    for path in [PROJECTS_DIR, UPLOADS_DIR, PROCESSED_DIR, CROPS_DIR, EXPORTS_DIR, EXPRESSIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def project_path(project_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}.json"


def list_projects() -> list[Project]:
    ensure_dirs()
    return [Project.model_validate_json(p.read_text()) for p in sorted(PROJECTS_DIR.glob("*.json"))]


def save_project(project: Project) -> Project:
    ensure_dirs()
    project.updatedAt = now_iso()
    project_path(project.id).write_text(project.model_dump_json(indent=2), encoding="utf-8")
    return project


def get_project(project_id: str) -> Project:
    path = project_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy project")
    return Project.model_validate_json(path.read_text(encoding="utf-8"))


def delete_project(project_id: str) -> None:
    path = project_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy project")
    path.unlink()


def find_page(page_id: str) -> tuple[Project, PageItem]:
    for project in list_projects():
        for page in project.pages:
            if page.id == page_id:
                return project, page
    raise HTTPException(status_code=404, detail="Không tìm thấy trang")


def find_expression(expression_id: str) -> tuple[Project, PageItem, ExpressionBox]:
    for project in list_projects():
        for page in project.pages:
            for expr in page.expressions:
                if expr.id == expression_id:
                    return project, page, expr
    raise HTTPException(status_code=404, detail="Không tìm thấy biểu thức")


def replace_page(project: Project, page: PageItem) -> Project:
    for index, item in enumerate(project.pages):
        if item.id == page.id:
            page.updatedAt = now_iso()
            project.pages[index] = page
            return save_project(project)
    raise HTTPException(status_code=404, detail="Không tìm thấy trang")


def static_to_data_path(url: str | None) -> Path | None:
    if not url or not url.startswith("/static/"):
        return None
    return DATA_DIR / url.replace("/static/", "", 1)


def remove_expression_files(expr: ExpressionBox) -> None:
    for url in [
        expr.cropPreviewUrl,
        expr.cleanedPreviewUrl,
        expr.binaryPreviewUrl,
        expr.normalizedPreviewUrl,
        expr.normalizedUrl,
        expr.componentsPreviewUrl,
        expr.latexRenderSvgUrl,
        expr.latexRenderPngUrl,
    ]:
        path = static_to_data_path(url)
        if path and path.exists() and path.is_file():
            path.unlink()
    expr_dir = EXPRESSIONS_DIR / expr.id
    if expr_dir.exists():
        shutil.rmtree(expr_dir)


def remove_page_files(page: PageItem) -> None:
    for expr in page.expressions:
        remove_expression_files(expr)
    for url in [page.imageUrl, page.thumbnailUrl, *page.layers.values()]:
        path = static_to_data_path(url)
        if path and path.exists() and path.is_file():
            path.unlink()
    for folder in [CROPS_DIR, PROCESSED_DIR]:
        for path in folder.glob(f"{page.id}*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def remove_page(project: Project, page_id: str) -> Project:
    target = None
    for page in project.pages:
        if page.id == page_id:
            target = page
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy trang")
    remove_page_files(target)
    project.pages = [page for page in project.pages if page.id != page_id]
    return save_project(project)


def next_id(prefix: str, existing: Iterable[str]) -> str:
    numbers = []
    for value in existing:
        suffix = value.replace(prefix + "_", "")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return f"{prefix}_{(max(numbers) + 1) if numbers else 1:03d}"


async def save_upload(project: Project, upload: UploadFile) -> PageItem:
    suffix = Path(upload.filename or "page.jpg").suffix.lower() or ".jpg"
    page_id = next_id("page", [p.id for p in project.pages])
    file_name = f"{page_id}{suffix}"
    file_path = UPLOADS_DIR / file_name
    with file_path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)

    with Image.open(file_path) as image:
        width, height = image.size
        image.thumbnail((360, 480))
        thumb_name = f"{page_id}_thumb.jpg"
        image.convert("RGB").save(UPLOADS_DIR / thumb_name, quality=88)

    return PageItem(
        id=page_id,
        fileName=upload.filename or file_name,
        imageUrl=f"/static/uploads/{file_name}",
        thumbnailUrl=f"/static/uploads/{thumb_name}",
        width=width,
        height=height,
        status="unscanned",
        layers={"original": f"/static/uploads/{file_name}"},
    )


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x = max(0.0, min(float(width), bbox.x))
    y = max(0.0, min(float(height), bbox.y))
    w = max(1.0, min(float(width) - x, bbox.width))
    h = max(1.0, min(float(height) - y, bbox.height))
    return BBox(x=x, y=y, width=w, height=h)
