from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import (
    CrohmeM4ExportRequest,
    CreateProjectRequest,
    ExportRequest,
    ExpressionBox,
    ExpressionCreateRequest,
    ExpressionPatchRequest,
    LatexPatchRequest,
    MergeRequest,
    NormalizeAllRequest,
    Project,
    ReorderRequest,
    ScanRequest,
    SplitRequest,
    now_iso,
)
from .latex_tools import clean_latex, validate_latex_basic
from .normalization import normalize_page_expression
from .recognition import run_recognition
from .services import export_crohme_m4_dataset, export_project, refresh_expression, scan_page, union_bbox
from .storage import (
    DATA_DIR,
    clamp_bbox,
    delete_project,
    ensure_dirs,
    find_expression,
    find_page,
    get_project,
    list_projects,
    next_id,
    remove_expression_files,
    remove_page,
    replace_page,
    save_project,
    save_upload,
)


ensure_dirs()
app = FastAPI(title="Expression Page Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")


@app.get("/api/health")
def health():
    return {"ok": True, "message": "Expression Page Explorer API sẵn sàng"}


@app.post("/api/projects")
def create_project(req: CreateProjectRequest):
    project = Project(id=f"project_{uuid4().hex[:8]}", name=req.name)
    return save_project(project)


@app.get("/api/projects")
def get_projects():
    projects = list_projects()
    if not projects:
        projects.append(save_project(Project(id="project_default", name="CROHME-like A4 Dataset")))
    return projects


@app.get("/api/projects/{project_id}")
def read_project(project_id: str):
    return get_project(project_id)


@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, req: CreateProjectRequest):
    project = get_project(project_id)
    project.name = req.name
    return save_project(project)


@app.delete("/api/projects/{project_id}")
def remove_project(project_id: str):
    delete_project(project_id)
    return {"ok": True}


@app.post("/api/projects/{project_id}/pages/upload")
async def upload_pages(project_id: str, files: list[UploadFile] = File(...)):
    project = get_project(project_id)
    pages = []
    for upload in files:
        page = await save_upload(project, upload)
        project.pages.append(page)
        pages.append(page)
    save_project(project)
    return {"pages": pages}


@app.get("/api/projects/{project_id}/pages")
def project_pages(project_id: str):
    return get_project(project_id).pages


@app.get("/api/pages/{page_id}")
def read_page(page_id: str):
    _, page = find_page(page_id)
    return page


@app.delete("/api/pages/{page_id}")
def delete_page(page_id: str):
    project, _ = find_page(page_id)
    remove_page(project, page_id)
    return {"ok": True}


@app.post("/api/pages/{page_id}/scan")
def scan(page_id: str, _: ScanRequest = ScanRequest()):
    project, page = find_page(page_id)
    page.status = "scanning"
    replace_page(project, page)
    page = scan_page(page)
    replace_page(project, page)
    return {"pageId": page.id, "status": page.status, "layers": page.layers, "expressions": page.expressions}


@app.delete("/api/pages/{page_id}/expressions")
def clear_page_expressions(page_id: str):
    project, page = find_page(page_id)
    for expr in page.expressions:
        remove_expression_files(expr)
    page.expressions = []
    page.status = "unscanned"
    page.updatedAt = now_iso()
    replace_page(project, page)
    return {"ok": True, "pageId": page.id}


@app.post("/api/pages/{page_id}/expressions")
def create_expression(page_id: str, req: ExpressionCreateRequest):
    project, page = find_page(page_id)
    expr_id = next_id(f"{page.id}_expr", [e.id for e in page.expressions])
    expr = ExpressionBox(
        id=expr_id,
        pageId=page.id,
        bbox=clamp_bbox(req.bbox, page.width, page.height),
        order=len(page.expressions) + 1,
        status=req.status,
        candidateType="single_expression",
        createdBy="manual",
        history=[{"at": now_iso(), "action": "create_manual", "by": "user"}],
    )
    expr = refresh_expression(page, expr)
    page.expressions.append(expr)
    page.status = "need_review"
    replace_page(project, page)
    return expr


@app.patch("/api/expressions/{expression_id}")
def patch_expression(expression_id: str, req: ExpressionPatchRequest):
    project, page, expr = find_expression(expression_id)
    if req.bbox is not None:
        expr.bbox = clamp_bbox(req.bbox, page.width, page.height)
        expr.history.append({"at": now_iso(), "action": "update_bbox", "by": "user", "payload": expr.bbox.model_dump()})
    if req.status is not None:
        expr.status = req.status
        expr.history.append({"at": now_iso(), "action": f"set_status_{req.status}", "by": "user"})
    expr.updatedAt = now_iso()
    expr = refresh_expression(page, expr)
    for index, item in enumerate(page.expressions):
        if item.id == expression_id:
            page.expressions[index] = expr
    replace_page(project, page)
    return expr


def _save_expression(project: Project, page, updated: ExpressionBox) -> ExpressionBox:
    for index, item in enumerate(page.expressions):
        if item.id == updated.id:
            page.expressions[index] = updated
            break
    replace_page(project, page)
    return updated


@app.delete("/api/expressions/{expression_id}")
def delete_expression(expression_id: str):
    project, page, _ = find_expression(expression_id)
    removed = [e for e in page.expressions if e.id == expression_id]
    for expr in removed:
        remove_expression_files(expr)
    page.expressions = [e for e in page.expressions if e.id != expression_id]
    for idx, expr in enumerate(sorted(page.expressions, key=lambda e: e.order), start=1):
        expr.order = idx
    replace_page(project, page)
    return {"ok": True}


@app.post("/api/expressions/{expression_id}/accept")
def accept_expression(expression_id: str):
    return patch_expression(expression_id, ExpressionPatchRequest(status="accepted"))


@app.post("/api/expressions/{expression_id}/reject")
def reject_expression(expression_id: str):
    return patch_expression(expression_id, ExpressionPatchRequest(status="rejected"))


@app.post("/api/expressions/merge")
def merge_expressions(req: MergeRequest):
    project, page = find_page(req.pageId)
    selected = [e for e in page.expressions if e.id in req.expressionIds]
    if len(selected) < 2:
        raise HTTPException(status_code=400, detail="Cần chọn ít nhất 2 bbox để gộp")
    merged_id = next_id(f"{page.id}_expr", [e.id for e in page.expressions])
    merged = ExpressionBox(
        id=merged_id,
        pageId=page.id,
        bbox=union_bbox([e.bbox for e in selected]),
        order=min(e.order for e in selected),
        status="edited",
        candidateType="single_expression",
        createdBy="manual",
        history=[{"at": now_iso(), "action": "merge", "by": "user", "payload": {"from": req.expressionIds}}],
    )
    merged = refresh_expression(page, merged)
    page.expressions = [e for e in page.expressions if e.id not in req.expressionIds] + [merged]
    page.expressions.sort(key=lambda e: (e.order, e.bbox.y, e.bbox.x))
    for idx, expr in enumerate(page.expressions, start=1):
        expr.order = idx
    replace_page(project, page)
    return {"mergedExpression": merged, "removedExpressionIds": req.expressionIds}


@app.post("/api/expressions/{expression_id}/split")
def split_expression(expression_id: str, req: SplitRequest):
    project, page, expr = find_expression(expression_id)
    pos = min(0.9, max(0.1, req.position))
    b = expr.bbox
    if req.mode == "horizontal":
        first = b.model_copy(update={"height": b.height * pos})
        second = b.model_copy(update={"y": b.y + b.height * pos, "height": b.height * (1 - pos)})
    else:
        first = b.model_copy(update={"width": b.width * pos})
        second = b.model_copy(update={"x": b.x + b.width * pos, "width": b.width * (1 - pos)})
    created = []
    base_order = expr.order
    for idx, bbox in enumerate([first, second], start=1):
        new_expr = ExpressionBox(
            id=next_id(f"{page.id}_expr", [e.id for e in page.expressions] + [e.id for e in created]),
            pageId=page.id,
            bbox=clamp_bbox(bbox, page.width, page.height),
            order=base_order + idx - 1,
            status="edited",
            candidateType="single_expression",
            createdBy="manual",
            history=[{"at": now_iso(), "action": f"split_{req.mode}", "by": "user", "payload": {"from": expression_id}}],
        )
        created.append(refresh_expression(page, new_expr))
    page.expressions = [e for e in page.expressions if e.id != expression_id] + created
    page.expressions.sort(key=lambda e: (e.order, e.bbox.y, e.bbox.x))
    for idx, item in enumerate(page.expressions, start=1):
        item.order = idx
    replace_page(project, page)
    return {"createdExpressions": created, "removedExpressionId": expression_id}


@app.post("/api/pages/{page_id}/expressions/reorder")
def reorder(page_id: str, req: ReorderRequest):
    project, page = find_page(page_id)
    index = {expr_id: idx + 1 for idx, expr_id in enumerate(req.orderedExpressionIds)}
    for expr in page.expressions:
        expr.order = index.get(expr.id, expr.order)
    page.expressions.sort(key=lambda e: e.order)
    replace_page(project, page)
    return page.expressions


@app.post("/api/projects/{project_id}/export")
def export(project_id: str, req: ExportRequest):
    project = get_project(project_id)
    return {"downloadUrl": export_project(project, req.includeStatuses, req.includeCrops)}


@app.post("/api/expressions/{expression_id}/normalize")
def normalize_expression(expression_id: str):
    project, page, expr = find_expression(expression_id)
    result = normalize_page_expression(page, expr)
    _save_expression(project, page, result.expression)
    return {
        "expression": result.expression,
        "normalized_url": result.expression.normalizedUrl,
        "quality": result.expression.quality,
        "warnings": result.expression.warnings,
    }


@app.post("/api/pages/{page_id}/normalize-all")
def normalize_all(page_id: str, req: NormalizeAllRequest = NormalizeAllRequest()):
    project, page = find_page(page_id)
    normalized = []
    for index, expr in enumerate(page.expressions):
        if req.skip_noise and expr.candidateType in ("noise", "fragment"):
            continue
        if expr.status not in req.only_status:
            continue
        result = normalize_page_expression(page, expr)
        page.expressions[index] = result.expression
        normalized.append(result.expression)
    replace_page(project, page)
    return {"pageId": page.id, "normalized": normalized}


@app.post("/api/expressions/{expression_id}/recognize")
def recognize_expression(expression_id: str):
    project, page, expr = find_expression(expression_id)
    if not expr.normalizedUrl:
        raise HTTPException(status_code=400, detail="Chưa có ảnh M4-ready. Hãy bấm Normalize Preview trước khi Run M4.")
    if expr.quality.emptyAfterNormalize:
        raise HTTPException(status_code=400, detail="Ảnh M4-ready rỗng hoặc lỗi normalize. Hãy chỉnh bbox rồi normalize lại.")
    expr = run_recognition(expr)
    _save_expression(project, page, expr)
    return {"expression": expr}


@app.post("/api/pages/{page_id}/recognize-accepted")
def recognize_accepted(page_id: str):
    project, page = find_page(page_id)
    recognized = []
    for index, expr in enumerate(page.expressions):
        if expr.status != "accepted":
            continue
        if not expr.normalizedUrl:
            continue
        if expr.quality.emptyAfterNormalize:
            continue
        expr = run_recognition(expr)
        page.expressions[index] = expr
        if expr.latexStatus in ("ok", "syntax_error", "model_error"):
            recognized.append(expr)
    replace_page(project, page)
    return {"pageId": page.id, "recognized": recognized}


@app.patch("/api/expressions/{expression_id}/latex")
def update_latex(expression_id: str, req: LatexPatchRequest):
    project, page, expr = find_expression(expression_id)
    cleaned = clean_latex(req.latex_clean)
    validation = validate_latex_basic(cleaned)
    expr.latexClean = cleaned
    if expr.latexRaw is None:
        expr.latexRaw = cleaned
    expr.latexStatus = "manual" if req.manual_override else ("ok" if validation.ok else "syntax_error")
    expr.recognitionHistory.append(
        {
            "run_id": f"manual_{now_iso()}",
            "model_name": "manual",
            "input_image_path": expr.normalizedUrl,
            "latex_raw": expr.latexRaw,
            "latex_clean": cleaned,
            "status": expr.latexStatus,
            "error": validation.error,
            "created_at": now_iso(),
        }
    )
    expr.updatedAt = now_iso()
    return _save_expression(project, page, expr)


@app.post("/api/export/crohme-m4")
def export_crohme_m4(req: CrohmeM4ExportRequest):
    project = get_project(req.projectId) if req.projectId else list_projects()[0]
    return {"downloadUrl": export_crohme_m4_dataset(project, req.mode)}
