# Expression Page Explorer — FE React + BE Python Specification

> **Mục tiêu:** Xây dựng giao diện và backend cho **bài toán 1**: từ ảnh chụp một trang giấy/A4 chứa nhiều biểu thức toán học, hệ thống tự động phát hiện các vùng biểu thức, hiển thị trực quan để người dùng kiểm tra/chỉnh sửa bbox, sau đó xuất danh sách crop/metadata để đưa sang bước chuẩn hóa CROHME-like và nhận dạng M4 ở giai đoạn sau.

---

## 0. Tóm tắt cho Agent

Hãy xây dựng một ứng dụng web gồm:

- **Frontend:** React + TypeScript + Vite + TailwindCSS + React-Konva.
- **Backend:** Python + FastAPI + OpenCV + Pydantic.
- **Mục tiêu MVP:** Upload ảnh trang A4, backend auto scan và trả về danh sách bbox biểu thức, frontend hiển thị ảnh với overlay bbox, cho phép người dùng vẽ/sửa/xóa/merge/split bbox, gán trạng thái Accepted/Need Review/Rejected, xem crop preview, và export metadata JSON.

Tên sản phẩm nội bộ:

```text
Expression Page Explorer
```

Tên màn hình chính:

```text
Page Splitter Workspace
```

Không cần nhận dạng LaTeX ở phase này. M4 chỉ nhận một biểu thức nên phase này chỉ tạo **Expression Queue** gồm các crop biểu thức đơn.

---

## 1. Bối cảnh bài toán

Mô hình M4/HMER hiện tại chỉ nhận dạng được **một biểu thức toán học trong một ảnh**. Tuy nhiên ảnh đầu vào thực tế có thể là:

- Một biểu thức đơn.
- Một phần bài làm.
- Một trang A4 chứa nhiều biểu thức.
- Ảnh chụp nghiêng, tối, có bóng, có giấy kẻ dòng/ô ly.

Vì vậy cần xây dựng một module đứng trước M4:

```text
Input Page Image
    ↓
Page-to-Expression Pipeline
    ↓
Expression Queue
    ↓
CROHME-like Normalizer / M4 later
```

Trong scope hiện tại, chỉ làm phần:

```text
Ảnh A4 → phát hiện vùng biểu thức → sửa bbox thủ công → xuất metadata/crop
```

---

## 2. Scope của Phase 1

### 2.1. In scope

Ứng dụng phải hỗ trợ:

1. Upload một hoặc nhiều ảnh.
2. Hiển thị danh sách page ở sidebar.
3. Auto scan một page để phát hiện bbox biểu thức.
4. Hiển thị ảnh page với overlay bbox.
5. Zoom, pan, fit page.
6. Chọn bbox.
7. Kéo bbox, resize bbox.
8. Vẽ bbox mới.
9. Xóa/reject bbox.
10. Accept bbox.
11. Merge nhiều bbox thành một bbox.
12. Split bbox theo ngang/dọc.
13. Hiển thị crop preview của bbox đang chọn.
14. Hiển thị warning/quality thông tin sơ bộ.
15. Quản lý Expression Queue ở bottom panel.
16. Kéo thả đổi thứ tự biểu thức trong Expression Queue.
17. Export metadata JSON/JSONL.
18. Export crop ảnh nếu backend đã hỗ trợ.
19. Lưu project state trong backend/local file.

### 2.2. Out of scope trong Phase 1

Chưa cần:

- Nhận dạng LaTeX bằng M4.
- Train detector YOLO/Detectron.
- Tài khoản người dùng.
- Cloud storage.
- OCR chữ thường.
- Hoàn thiện CROHME-like normalizer chuyên sâu.
- InkML thật của CROHME online.

---

## 3. Kiến trúc tổng thể

```text
Frontend React
  ├── Page Manager
  ├── Page Viewer Canvas
  ├── Annotation Editor
  ├── Expression Queue
  ├── Inspector Panel
  └── Export Center

Backend FastAPI
  ├── Image Upload Service
  ├── Page Processing Service
  ├── Expression Detection Service
  ├── Crop Preview Service
  ├── Annotation State Service
  └── Export Service

Storage
  ├── uploads/
  ├── processed/
  ├── crops/
  ├── overlays/
  └── projects/project.json
```

---

## 4. Công nghệ đề xuất

### 4.1. Frontend

```text
React + TypeScript
Vite
TailwindCSS
React-Konva
Zustand
TanStack Query
Axios
Dnd Kit hoặc React Beautiful DnD
Lucide React icons
```

Vai trò:

- **React:** xây giao diện.
- **React-Konva:** vẽ ảnh, bbox, zoom/pan, drag/resize.
- **TailwindCSS:** style nhanh, sạch.
- **Zustand:** quản lý state page/expression đang chọn.
- **TanStack Query:** gọi API backend và cache.
- **Dnd Kit:** kéo thả đổi thứ tự Expression Queue.

### 4.2. Backend

```text
Python 3.10+
FastAPI
Uvicorn
OpenCV
NumPy
Pillow
Pydantic
python-multipart
```

Tuỳ chọn:

```text
scikit-image        # threshold Sauvola/Niblack nếu cần
Shapely             # xử lý bbox nếu cần
```

---

## 5. UI/UX Design

## 5.1. Layout chính

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ Header                                                                       │
│ Expression Page Explorer | Upload | Auto Scan | Review | Export | Settings   │
├──────────────────────┬──────────────────────────────────────┬────────────────┤
│ Page List            │ Main Page Viewer                     │ Inspector      │
│                      │                                      │                │
│ page_001 ✅           │ Ảnh A4 + overlay bbox                 │ Selected bbox   │
│ page_002 ⚠️           │ Zoom / Pan / Draw / Edit              │ Crop preview    │
│ page_003 ⏳           │                                      │ Quality info    │
│                      │                                      │ Actions         │
├──────────────────────┴──────────────────────────────────────┴────────────────┤
│ Expression Queue                                                             │
│ expr_001 ✅ | expr_002 ⚠️ | expr_003 ✅ | expr_004 ❌                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2. Header

Header cần có:

```text
Expression Page Explorer
Project: [project name]
[Upload Images]
[Auto Scan]
[Review Warnings]
[Save]
[Export]
[Settings]
```

Bên phải hiển thị thống kê:

```text
Pages: 24 | Expressions: 137 | Accepted: 112 | Need Review: 18 | Rejected: 7
```

---

## 5.3. Sidebar trái — Page List

Mỗi page card:

```text
┌─────────────────────┐
│ page_001.jpg         │
│ 5 expressions        │
│ 4 accepted | 1 warn  │
│ [thumbnail]          │
└─────────────────────┘
```

Trạng thái page:

```ts
type PageStatus =
  | "unscanned"
  | "scanning"
  | "scanned"
  | "need_review"
  | "completed"
  | "error";
```

Màu trạng thái:

```text
Green: completed
Yellow: need_review
Blue: scanning/scanned
Gray: unscanned
Red: error
```

Bộ lọc:

- All pages
- Unscanned
- Need review
- Completed
- Error

---

## 5.4. Main Page Viewer

Đây là vùng chính dùng React-Konva.

### Toolbar

```text
[Select]
[Pan]
[Draw Box]
[Split]
[Merge]
[Reject/Delete]
[Zoom In]
[Zoom Out]
[Fit Page]
[Original]
[Cleaned]
[Binary]
[Components]
[Expressions]
```

### Layer cần hỗ trợ

```text
Original        Ảnh gốc
Rectified       Ảnh đã sửa nghiêng/cắt trang nếu có
Cleaned         Ảnh khử nền
Binary          Ảnh đen trắng
Components      Connected components
Expressions     Bbox biểu thức
Warnings        Vùng cảnh báo
```

### Overlay bbox

Mỗi bbox có label:

```text
expr_001
expr_002
expr_003
```

Màu bbox:

```text
Accepted:      #22C55E
Need Review:   #FACC15
Rejected:      #94A3B8
Error:         #EF4444
Selected:      #38BDF8
Manual Edited: #A855F7
```

### Tương tác bbox

Cần hỗ trợ:

- Click để chọn.
- Drag để di chuyển.
- Drag corner để resize.
- Double click để mở crop preview lớn.
- Shift + click để multi-select.
- Delete để reject/delete.
- Right click mở context menu.

Context menu:

```text
Accept
Reject
Duplicate
Split Horizontal
Split Vertical
Merge Selected
Mark as Noise
Re-run Detection Inside Box
```

---

## 5.5. Inspector Panel

Khi chọn một expression, panel phải hiện:

```text
Selected: expr_004
Status: Need Review

Preview:
- Crop original
- Crop binary
- Crop padded preview nếu có

Quality:
- Width: 1240 px
- Height: 210 px
- Aspect ratio: 5.9
- Foreground ratio: 7.4%
- Touching border: No
- Maybe multiple expressions: Maybe

Warnings:
- maybe_multiple_expressions
- touching_border

Actions:
[Accept]
[Reject]
[Split]
[Merge]
[Re-detect]
[Normalize Preview]
```

Inspector tabs:

1. Crop
2. Quality
3. Components
4. History

---

## 5.6. Expression Queue

Bottom panel hiển thị các expression đã detect.

Card:

```text
┌──────────┐
│ expr_001 │
│ ✅ Ready │
│ [crop]   │
└──────────┘
```

Mỗi card có:

- Thumbnail crop.
- Expression ID.
- Status.
- Warning icon.
- Reading order.

Cho phép drag/drop đổi thứ tự đọc.

Thứ tự đọc mặc định:

```text
top-to-bottom, left-to-right
```

---

## 5.7. Review Warnings Mode

Khi bấm `Review Warnings`, app chỉ hiển thị expression có vấn đề.

Flow:

```text
expr_004 warning
→ user xem crop
→ Accept / Split / Merge / Reject
→ Next warning
```

Hotkeys:

```text
A = Accept
R = Reject
S = Split
M = Merge
N = Next
B = Back
```

---

## 6. Data Model Frontend

### 6.1. PageItem

```ts
export type PageStatus =
  | "unscanned"
  | "scanning"
  | "scanned"
  | "need_review"
  | "completed"
  | "error";

export type PageItem = {
  id: string;
  fileName: string;
  imageUrl: string;
  thumbnailUrl?: string;
  width: number;
  height: number;
  status: PageStatus;
  expressionIds: string[];
  createdAt: string;
  updatedAt: string;
};
```

### 6.2. ExpressionBox

```ts
export type ExpressionStatus =
  | "auto_detected"
  | "need_review"
  | "accepted"
  | "rejected"
  | "edited"
  | "exported";

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
  binaryPreviewUrl?: string;
  normalizedPreviewUrl?: string;
  createdBy: "auto" | "manual";
  history: ExpressionHistoryItem[];
  createdAt: string;
  updatedAt: string;
};

export type ExpressionHistoryItem = {
  at: string;
  action: string;
  by: "auto" | "user";
  payload?: Record<string, unknown>;
};
```

---

## 7. Backend Data Model / Pydantic

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

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
    warnings: List[str] = []

ExpressionStatus = Literal[
    "auto_detected",
    "need_review",
    "accepted",
    "rejected",
    "edited",
    "exported",
]

class ExpressionBox(BaseModel):
    id: str
    pageId: str
    bbox: BBox
    order: int
    status: ExpressionStatus
    quality: ExpressionQuality
    cropPreviewUrl: Optional[str] = None
    binaryPreviewUrl: Optional[str] = None
    normalizedPreviewUrl: Optional[str] = None
    createdBy: Literal["auto", "manual"]
    history: List[Dict[str, Any]] = []
    createdAt: datetime
    updatedAt: datetime

PageStatus = Literal[
    "unscanned",
    "scanning",
    "scanned",
    "need_review",
    "completed",
    "error",
]

class PageItem(BaseModel):
    id: str
    fileName: str
    imageUrl: str
    thumbnailUrl: Optional[str] = None
    width: int
    height: int
    status: PageStatus
    expressions: List[ExpressionBox] = []
    createdAt: datetime
    updatedAt: datetime
```

---

## 8. API Contract

Base URL:

```text
http://localhost:8000/api
```

### 8.1. Project

```http
POST /api/projects
GET  /api/projects
GET  /api/projects/{project_id}
PATCH /api/projects/{project_id}
DELETE /api/projects/{project_id}
```

Create project request:

```json
{
  "name": "CROHME-like A4 Dataset"
}
```

---

### 8.2. Upload images

```http
POST /api/projects/{project_id}/pages/upload
```

Form-data:

```text
files: image files
```

Response:

```json
{
  "pages": [
    {
      "id": "page_001",
      "fileName": "a4_001.jpg",
      "imageUrl": "/static/uploads/page_001.jpg",
      "thumbnailUrl": "/static/uploads/page_001_thumb.jpg",
      "width": 3024,
      "height": 4032,
      "status": "unscanned",
      "expressions": []
    }
  ]
}
```

---

### 8.3. Get pages

```http
GET /api/projects/{project_id}/pages
GET /api/pages/{page_id}
```

---

### 8.4. Scan page

```http
POST /api/pages/{page_id}/scan
```

Request:

```json
{
  "preset": "white_paper",
  "detectMode": "classical_cv",
  "saveLayers": true
}
```

Response:

```json
{
  "pageId": "page_001",
  "status": "need_review",
  "layers": {
    "original": "/static/uploads/page_001.jpg",
    "cleaned": "/static/processed/page_001_cleaned.png",
    "binary": "/static/processed/page_001_binary.png",
    "components": "/static/processed/page_001_components.png"
  },
  "expressions": [
    {
      "id": "expr_001",
      "pageId": "page_001",
      "bbox": { "x": 120, "y": 340, "width": 980, "height": 160 },
      "order": 1,
      "status": "auto_detected",
      "quality": {
        "foregroundRatio": 0.074,
        "aspectRatio": 6.125,
        "touchBorder": false,
        "maybeMultipleExpressions": false,
        "maybeNoise": false,
        "warnings": []
      },
      "cropPreviewUrl": "/static/crops/expr_001.png",
      "createdBy": "auto"
    }
  ]
}
```

---

### 8.5. Layers

```http
GET /api/pages/{page_id}/layers/original
GET /api/pages/{page_id}/layers/cleaned
GET /api/pages/{page_id}/layers/binary
GET /api/pages/{page_id}/layers/components
```

Có thể trả file ảnh hoặc URL.

---

### 8.6. Expression CRUD

```http
POST   /api/pages/{page_id}/expressions
PATCH  /api/expressions/{expression_id}
DELETE /api/expressions/{expression_id}
```

Create manual expression:

```json
{
  "bbox": { "x": 200, "y": 450, "width": 800, "height": 160 },
  "status": "edited"
}
```

Patch expression:

```json
{
  "bbox": { "x": 210, "y": 455, "width": 820, "height": 170 },
  "status": "edited"
}
```

---

### 8.7. Accept / Reject

```http
POST /api/expressions/{expression_id}/accept
POST /api/expressions/{expression_id}/reject
```

---

### 8.8. Merge expressions

```http
POST /api/expressions/merge
```

Request:

```json
{
  "pageId": "page_001",
  "expressionIds": ["expr_003", "expr_004"]
}
```

Response:

```json
{
  "mergedExpression": {
    "id": "expr_003_merged",
    "bbox": { "x": 100, "y": 300, "width": 1000, "height": 250 },
    "status": "edited"
  },
  "removedExpressionIds": ["expr_003", "expr_004"]
}
```

---

### 8.9. Split expression

```http
POST /api/expressions/{expression_id}/split
```

Request horizontal split:

```json
{
  "mode": "horizontal",
  "position": 0.52
}
```

Request vertical split:

```json
{
  "mode": "vertical",
  "position": 0.48
}
```

Response:

```json
{
  "createdExpressions": [
    { "id": "expr_007a", "bbox": { "x": 100, "y": 300, "width": 1000, "height": 120 } },
    { "id": "expr_007b", "bbox": { "x": 100, "y": 430, "width": 1000, "height": 130 } }
  ],
  "removedExpressionId": "expr_007"
}
```

---

### 8.10. Crop preview

```http
GET /api/expressions/{expression_id}/crop
GET /api/expressions/{expression_id}/binary-crop
GET /api/expressions/{expression_id}/preview
```

---

### 8.11. Reorder expressions

```http
POST /api/pages/{page_id}/expressions/reorder
```

Request:

```json
{
  "orderedExpressionIds": ["expr_001", "expr_002", "expr_003"]
}
```

---

### 8.12. Export

```http
POST /api/projects/{project_id}/export
```

Request:

```json
{
  "includeStatuses": ["accepted"],
  "includeCrops": true,
  "includeOverlays": true,
  "format": "jsonl"
}
```

Response:

```json
{
  "downloadUrl": "/static/exports/project_001_export.zip"
}
```

---

## 9. Backend Processing Pipeline

File/module đề xuất:

```text
backend/
  app/
    main.py
    config.py
    models.py
    storage.py
    routers/
      projects.py
      pages.py
      expressions.py
      export.py
    services/
      image_io.py
      page_rectifier.py
      background_cleaner.py
      binarizer.py
      component_analyzer.py
      expression_detector.py
      crop_service.py
      quality_checker.py
      export_service.py
```

### 9.1. Pipeline scan page

```python
def scan_page(page_image_path: str, preset: str):
    image = load_image(page_image_path)

    # Phase 1 MVP: có thể chưa cần perspective correction sâu
    rectified = rectify_page_if_possible(image)
    cleaned = clean_background(rectified, preset=preset)
    binary = binarize_image(cleaned, preset=preset)

    components = find_connected_components(binary)
    components = filter_noise_components(components)

    expression_boxes = group_components_into_expressions(components, binary.shape)
    expression_boxes = sort_reading_order(expression_boxes)

    results = []
    for idx, box in enumerate(expression_boxes):
        crop = crop_with_padding(binary, box)
        quality = check_expression_quality(crop, box, binary.shape)
        status = "need_review" if quality.warnings else "auto_detected"
        preview_url = save_crop_preview(crop)
        results.append(make_expression(idx, box, quality, status, preview_url))

    save_layers(rectified, cleaned, binary, components)
    return results
```

---

## 10. Thuật toán MVP cho Expression Detection

### 10.1. Binarization

MVP:

```python
# grayscale
# Gaussian blur nhẹ
# adaptive threshold
# invert nếu cần để foreground là màu trắng hoặc đen nhất quán
```

OpenCV pseudo:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35,
    15,
)
```

Lưu ý: nội bộ có thể dùng `THRESH_BINARY_INV` để foreground là trắng trên nền đen. Khi xuất preview thì invert lại nếu muốn nền trắng chữ đen.

---

### 10.2. Connected Components

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
```

Filter component:

```python
area >= min_area
height >= min_height
width >= min_width
not too large compared to page
```

Config:

```yaml
component:
  min_area: 8
  min_width: 2
  min_height: 2
  max_area_ratio: 0.2
```

---

### 10.3. Group components into expressions

MVP heuristic:

1. Tính bbox của từng component.
2. Tính median component height.
3. Gom các component gần nhau theo trục y thành text/math line.
4. Merge các component trong cùng line nếu khoảng cách ngang không quá lớn.
5. Giữ chung các component nhỏ phía trên/dưới nếu gần bbox chính để không làm mất mũ/chỉ số.
6. Sau khi có line groups, merge các line gần nhau nếu có khả năng là fraction/matrix/system.

Pseudo:

```python
def group_components_into_expressions(components, image_shape):
    median_h = median([c.h for c in components])
    y_threshold = median_h * 1.2
    x_gap_threshold = median_h * 4.0
    vertical_merge_threshold = median_h * 1.5

    line_groups = group_by_vertical_overlap(components, y_threshold)
    expr_groups = []

    for line in line_groups:
        chunks = split_line_by_large_x_gap(line, x_gap_threshold)
        expr_groups.extend(chunks)

    expr_groups = merge_math_structures(expr_groups, vertical_merge_threshold)
    bboxes = [union_bbox(group) for group in expr_groups]
    return bboxes
```

---

### 10.4. Reading order

```python
def sort_reading_order(boxes):
    # sort top-to-bottom first
    # if y centers are close, sort left-to-right
    return sorted(boxes, key=lambda b: (round(b.y / row_bucket), b.x))
```

---

## 11. Quality Checker

Quality checker gắn warning cho từng bbox/crop.

### Rule đề xuất

```text
foreground_ratio < 0.005       → maybe_noise / too_sparse
foreground_ratio > 0.35        → too_dense / bad_threshold
touches crop border            → touching_border
height < 24 px                 → too_small
width / height > 20            → too_wide_maybe_multiple
height / width > 3             → too_tall
large vertical whitespace      → maybe_multiple_expressions
many small detached components → maybe_split_symbol
```

Pydantic warning codes:

```text
maybe_noise
too_sparse
too_dense
touching_border
too_small
too_large
too_wide_maybe_multiple
maybe_multiple_expressions
maybe_fraction_split
maybe_superscript_detached
not_math_like
```

Status rule:

```python
if warnings:
    status = "need_review"
else:
    status = "auto_detected"
```

Người dùng có thể đổi sang `accepted` thủ công.

---

## 12. Frontend Folder Structure

```text
frontend/
  src/
    app/
      App.tsx
      router.tsx
    api/
      client.ts
      projectApi.ts
      pageApi.ts
      expressionApi.ts
      exportApi.ts
    components/
      layout/
        AppHeader.tsx
        MainLayout.tsx
      pages/
        PageList.tsx
        PageCard.tsx
      viewer/
        PageViewer.tsx
        KonvaStage.tsx
        ImageLayer.tsx
        BBoxLayer.tsx
        BBoxRect.tsx
        ViewerToolbar.tsx
        LayerSwitcher.tsx
      inspector/
        InspectorPanel.tsx
        CropTab.tsx
        QualityTab.tsx
        ComponentsTab.tsx
        HistoryTab.tsx
      queue/
        ExpressionQueue.tsx
        ExpressionQueueCard.tsx
      dialogs/
        UploadDialog.tsx
        ExportDialog.tsx
        SplitDialog.tsx
        SettingsDialog.tsx
    store/
      projectStore.ts
      viewerStore.ts
      annotationStore.ts
    types/
      page.ts
      expression.ts
      project.ts
    utils/
      geometry.ts
      bbox.ts
      readingOrder.ts
      hotkeys.ts
    styles/
      globals.css
```

---

## 13. Backend Folder Structure

```text
backend/
  app/
    main.py
    config.py
    models.py
    storage.py
    routers/
      projects.py
      pages.py
      expressions.py
      export.py
    services/
      image_io.py
      page_rectifier.py
      background_cleaner.py
      binarizer.py
      component_analyzer.py
      expression_detector.py
      crop_service.py
      quality_checker.py
      export_service.py
    data/
      projects/
      uploads/
      processed/
      crops/
      exports/
  requirements.txt
  README.md
```

---

## 14. Frontend State Store

Zustand store đề xuất:

```ts
type AnnotationStore = {
  currentProjectId?: string;
  currentPageId?: string;
  selectedExpressionIds: string[];
  activeTool: "select" | "pan" | "draw" | "split" | "merge";
  activeLayer: "original" | "cleaned" | "binary" | "components" | "expressions";
  pages: Record<string, PageItem>;
  expressions: Record<string, ExpressionBox>;

  setCurrentPage: (pageId: string) => void;
  selectExpression: (id: string, multi?: boolean) => void;
  updateExpressionBBox: (id: string, bbox: BBox) => void;
  setExpressionStatus: (id: string, status: ExpressionStatus) => void;
  reorderExpressions: (pageId: string, orderedIds: string[]) => void;
};
```

---

## 15. Canvas / Konva Requirements

### 15.1. Must-have

- Render image page.
- Render bbox overlay.
- Support zoom/pan.
- Support fit-to-screen.
- Support bbox drag.
- Support bbox resize using Transformer.
- Support drawing new bbox with mouse drag.
- Support multi-select.
- Support keyboard delete.

### 15.2. Coordinate system

Important:

- BBox must be stored in **original image coordinates**, not screen coordinates.
- Konva stage can scale/translate for zoom/pan.
- Need utility functions:

```ts
screenToImagePoint(point, stageScale, stagePosition)
imageToScreenPoint(point, stageScale, stagePosition)
normalizeBBox(bbox)
clampBBoxToImage(bbox, imageWidth, imageHeight)
unionBBoxes(bboxes)
splitBBoxHorizontal(bbox, ratio)
splitBBoxVertical(bbox, ratio)
```

---

## 16. Export Format

### 16.1. metadata.json

```json
{
  "project_id": "project_001",
  "project_name": "CROHME-like A4 Dataset",
  "created_at": "2026-06-24T00:00:00Z",
  "pages": [
    {
      "page_id": "page_001",
      "source_image": "page_001.jpg",
      "width": 3024,
      "height": 4032,
      "expressions": [
        {
          "id": "expr_001",
          "order": 1,
          "bbox": [120, 340, 980, 160],
          "status": "accepted",
          "warnings": [],
          "crop_file": "images/page_001_expr_001.png"
        }
      ]
    }
  ]
}
```

### 16.2. metadata.jsonl

Mỗi dòng:

```json
{"id":"page_001_expr_001","source_image":"page_001.jpg","order":1,"bbox":[120,340,980,160],"status":"accepted","warnings":[],"crop_file":"images/page_001_expr_001.png"}
```

### 16.3. Folder export

```text
export_project_001/
  images/
    page_001_expr_001.png
    page_001_expr_002.png
  overlays/
    page_001_overlay.jpg
  metadata.json
  metadata.jsonl
  project.json
```

---

## 17. Hotkeys

```text
V       Select
B       Draw Box
Space   Pan
A       Accept selected
R       Reject selected
S       Split selected
M       Merge selected
Delete  Delete/Reject selected
Ctrl+Z  Undo
Ctrl+Y  Redo
+       Zoom in
-       Zoom out
F       Fit page
N       Next warning
P       Previous warning
1       Original layer
2       Cleaned layer
3       Binary layer
4       Components layer
5       Expressions layer
```

---

## 18. UI Style Guide

### Light mode recommended

```text
Background: #F8FAFC
Panel: #FFFFFF
Border: #E2E8F0
Text primary: #0F172A
Text secondary: #64748B
Primary: #2563EB
Success: #16A34A
Warning: #F59E0B
Error: #DC2626
Selected: #0284C7
Edited: #9333EA
```

### Dark mode optional

```text
Background: #0F172A
Panel: #1E293B
Text primary: #F8FAFC
Text secondary: #94A3B8
```

---

## 19. MVP Implementation Plan

### Milestone 1 — Backend skeleton

- Create FastAPI app.
- Create project/page/expression models.
- Implement upload image.
- Serve static uploaded files.
- Store project JSON on disk.

Acceptance:

- Can upload image.
- Can list pages.
- Can view image URL.

---

### Milestone 2 — Frontend skeleton

- Create React Vite app.
- Create layout: Header + Page List + Viewer + Inspector + Queue.
- Implement upload dialog.
- Load page image into viewer.

Acceptance:

- User can upload and select a page.
- Image appears in central viewer.

---

### Milestone 3 — Auto scan MVP

Backend:

- Implement grayscale.
- Implement adaptive threshold.
- Implement connected components.
- Implement simple grouping into expression bbox.
- Return bbox list.
- Save crop previews.

Frontend:

- Auto Scan button.
- Render bbox overlay.
- Show expression cards in queue.

Acceptance:

- Click Auto Scan returns bbox.
- Bbox visible on image.
- Crop preview appears in inspector.

---

### Milestone 4 — Annotation editor

- Drag bbox.
- Resize bbox.
- Draw new bbox.
- Delete/reject bbox.
- Accept bbox.
- Patch backend state.

Acceptance:

- User can manually fix bbox.
- State persists after refresh.

---

### Milestone 5 — Merge/Split

- Multi-select bbox.
- Merge selected bbox.
- Split selected bbox horizontally/vertically.
- Recompute crop preview.

Acceptance:

- User can fix common segmentation errors.

---

### Milestone 6 — Export

- Export accepted bbox metadata.
- Export crops if supported.
- Download ZIP.

Acceptance:

- Export folder contains `metadata.json`, `metadata.jsonl`, and crop images.

---

## 20. Definition of Done for Phase 1

Phase 1 hoàn thành khi:

1. Upload được ảnh A4.
2. Auto detect được vùng biểu thức sơ bộ.
3. Hiển thị bbox trực quan trên ảnh.
4. Người dùng có thể sửa bbox bằng kéo thả.
5. Người dùng có thể vẽ bbox mới.
6. Người dùng có thể reject/accept bbox.
7. Người dùng có thể merge/split bbox.
8. Có Expression Queue gồm các crop biểu thức.
9. Có crop preview.
10. Có warning/quality cơ bản.
11. Có export metadata.
12. Code FE/BE chạy local ổn định.

---

## 21. Common Edge Cases cần xử lý

### 21.1. Ảnh chỉ có một biểu thức

Auto scan nên trả về một bbox lớn quanh toàn bộ biểu thức.

### 21.2. Ảnh A4 có nhiều dòng bài giải

Mỗi dòng độc lập nên là một expression riêng, trừ khi có cấu trúc toán nhiều dòng như hệ phương trình/ma trận.

### 21.3. Phân số bị tách tử/mẫu

Quality checker hoặc suggestion nên cảnh báo `maybe_fraction_split`. Người dùng có thể merge.

### 21.4. Mũ/chỉ số bị tách bbox riêng

Nên có suggestion merge nếu component nhỏ nằm gần phía trên/phía dưới bbox chính.

### 21.5. Hai biểu thức bị gộp thành một

Quality checker cảnh báo `maybe_multiple_expressions`. Người dùng dùng Split.

### 21.6. Nhiễu/chấm/bụi thành bbox

Quality checker cảnh báo `maybe_noise` hoặc `too_small`. Người dùng reject.

### 21.7. Ảnh nghiêng/tối

Phase 1 chỉ cần xử lý cơ bản. Nâng cấp page rectification ở phase sau.

---

## 22. Future Phase — M4 Integration

Sau khi Phase 1 ổn, thêm:

```text
Expression Queue
  ↓
CROHME-like Normalizer
  ↓
M4 Single Expression Recognizer
  ↓
LaTeX Output
```

M4 không nhận ảnh A4. M4 chỉ nhận từng crop trong Expression Queue.

API tương lai:

```http
POST /api/expressions/{expression_id}/normalize
POST /api/expressions/{expression_id}/recognize
POST /api/pages/{page_id}/recognize-ready-expressions
```

Expression sau này có thêm field:

```ts
type ExpressionRecognition = {
  normalizedImageUrl?: string;
  latex?: string;
  confidence?: number;
  latexValid?: boolean;
};
```

---

## 23. Agent Coding Instructions

Khi implement, ưu tiên:

1. Code chạy được local trước.
2. UI rõ ràng, thao tác bbox mượt.
3. API đơn giản, dễ debug.
4. Không over-engineer database; Phase 1 có thể lưu JSON file.
5. Bbox phải dùng coordinate ảnh gốc.
6. Mọi chỉnh sửa bbox phải persist.
7. Crop preview phải update sau khi sửa bbox.
8. Export phải dựa trên trạng thái accepted.

Không làm trong Phase 1:

- Không tích hợp M4.
- Không train detector.
- Không làm auth.
- Không làm deployment production.
- Không cố tạo InkML CROHME thật.

---

## 24. Local Run Proposal

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Default URLs:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
API docs: http://localhost:8000/docs
```

---

## 25. Minimal Backend requirements.txt

```txt
fastapi
uvicorn[standard]
python-multipart
opencv-python
numpy
pillow
pydantic
```

Optional:

```txt
scikit-image
```

---

## 26. Minimal Frontend package dependencies

```json
{
  "dependencies": {
    "@tanstack/react-query": "latest",
    "axios": "latest",
    "@dnd-kit/core": "latest",
    "@dnd-kit/sortable": "latest",
    "konva": "latest",
    "react-konva": "latest",
    "lucide-react": "latest",
    "zustand": "latest"
  },
  "devDependencies": {
    "typescript": "latest",
    "vite": "latest",
    "tailwindcss": "latest"
  }
}
```

---

## 27. Final Product Description

> Expression Page Explorer là giao diện trực quan hỗ trợ phân tích ảnh chụp trang giấy chứa nhiều biểu thức toán học. Hệ thống tự động đề xuất các vùng biểu thức, hiển thị bbox trên ảnh gốc, cho phép người dùng hiệu chỉnh bằng thao tác kéo thả, merge, split, accept/reject và quản lý danh sách Expression Queue. Các biểu thức đã xác nhận sẽ được xuất thành metadata và crop ảnh để phục vụ bước chuẩn hóa CROHME-like và nhận dạng biểu thức đơn bằng M4 ở giai đoạn tiếp theo.

