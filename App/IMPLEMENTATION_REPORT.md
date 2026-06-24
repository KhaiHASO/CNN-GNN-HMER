# Báo cáo triển khai Expression Page Explorer

## Tổng quan

Đã xây dựng ứng dụng **Expression Page Explorer** trong thư mục `App` theo đặc tả `expression_page_explorer_spec.md`.

Mục tiêu phase 1 đã được triển khai ở mức MVP: upload ảnh trang giấy, tự động phát hiện vùng biểu thức bằng OpenCV, hiển thị bbox trên ảnh, chỉnh sửa bbox thủ công, quản lý Expression Queue, accept/reject, split/merge, reorder và export metadata/crop.

## Công nghệ sử dụng

### Frontend

- React + TypeScript + Vite.
- Bootstrap cho giao diện nền.
- CSS custom để hoàn thiện layout workspace, canvas, sidebar, inspector và queue.
- React-Konva/Konva để render ảnh và bbox trên canvas.
- Axios để gọi API.
- Lucide React cho icon.

Lưu ý: đã bỏ Tailwind theo yêu cầu, frontend hiện dùng Bootstrap.

### Backend

- Python FastAPI.
- OpenCV + NumPy cho xử lý ảnh.
- Pillow cho đọc ảnh, thumbnail và crop.
- Pydantic cho data model.
- Lưu state bằng JSON file local, chưa dùng database.

## Cấu trúc chính

```text
App/
  backend/
    app/
      main.py
      models.py
      services.py
      storage.py
    requirements.txt
    README.md
  frontend/
    src/
      main.tsx
      api.ts
      geometry.ts
      types.ts
      styles.css
    package.json
    vite.config.ts
  expression_page_explorer_spec.md
  IMPLEMENTATION_REPORT.md
  .gitignore
```

## Chức năng đã triển khai

### Backend

- Tạo và đọc project mặc định.
- Upload một hoặc nhiều ảnh.
- Lưu ảnh upload và thumbnail.
- Auto scan page:
  - grayscale;
  - Gaussian blur;
  - adaptive threshold;
  - connected components;
  - gom component thành bbox biểu thức;
  - sắp xếp reading order;
  - tạo crop preview và binary crop;
  - tính quality/warnings cơ bản.
- API CRUD expression:
  - tạo bbox thủ công;
  - update bbox/status;
  - xóa expression;
  - accept/reject;
  - split ngang/dọc;
  - merge nhiều bbox;
  - reorder queue.
- Export ZIP gồm:
  - `metadata.json`;
  - `metadata.jsonl`;
  - `project.json`;
  - crop ảnh nếu có.
- Serve static files từ `backend/data`.

### Frontend

- Header với upload, auto scan, review warnings, export, settings và thống kê.
- Sidebar danh sách page có thumbnail, số expression và trạng thái.
- Viewer chính dùng Konva:
  - hiển thị ảnh;
  - overlay bbox;
  - chọn bbox;
  - multi-select bằng Shift;
  - kéo bbox;
  - resize bbox;
  - vẽ bbox mới;
  - zoom bằng wheel;
  - fit page;
  - đổi layer original/cleaned/binary/components.
- Toolbar:
  - Select;
  - Pan;
  - Draw Box;
  - Split H;
  - Split V;
  - Merge;
  - Reject;
  - Delete;
  - Accept.
- Inspector:
  - crop preview;
  - status;
  - quality info;
  - warnings;
  - actions;
  - history.
- Expression Queue:
  - crop thumbnail;
  - expression id;
  - status tiếng Việt;
  - warning count;
  - drag/drop đổi thứ tự.
- Hotkeys cơ bản:
  - `V`: select;
  - `B`: draw box;
  - `Space`: pan;
  - `A`: accept;
  - `R`: reject;
  - `S`: split ngang;
  - `M`: merge;
  - `Delete`: reject;
  - `1-4`: đổi layer.

## Sửa lỗi trong quá trình rà soát

- Đổi ID expression từ dạng `expr_001` sang dạng `page_001_expr_001`.
- Lý do: API expression là global (`/api/expressions/{id}`), nếu nhiều page cùng có `expr_001` sẽ dễ update nhầm expression.
- Đã thêm `.gitignore` để bỏ qua:
  - `frontend/node_modules`;
  - `frontend/dist`;
  - `frontend/tsconfig.tsbuildinfo`;
  - `backend/data`;
  - `backend/.venv`;
  - cache Python.

## Kiểm thử đã chạy

### Build và kiểm tra cú pháp

```bash
cd App/frontend
npm run build
```

Kết quả: pass.

```bash
cd App/backend
python -m py_compile app/main.py app/models.py app/storage.py app/services.py
```

Kết quả: pass.

### Playwright E2E

Đã dùng Playwright headless để kiểm tra luồng chính:

- mở app;
- upload ảnh demo `DataDemo/anh01.jpg`;
- auto scan;
- kiểm tra bbox được tạo;
- accept expression;
- reject expression;
- kéo bbox và kiểm tra bbox mới persist;
- draw box tạo expression manual;
- split bbox;
- merge bbox;
- drag/drop reorder queue;
- export metadata/crop;
- đổi layer cleaned/binary/components;
- kiểm tra không có failed request hoặc console error.

Kết quả: pass.

## Cách chạy local

### Backend

```bash
cd App/backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### Frontend

```bash
cd App/frontend
npm install
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

## Trạng thái hiện tại

- App chạy được local.
- Backend và frontend đã được kiểm thử qua build và Playwright E2E.
- Dữ liệu test đã được dọn sau khi kiểm thử.
- `backend/data` là dữ liệu runtime, không nên commit.

## Giới hạn hiện tại

- Detector vẫn là heuristic OpenCV MVP, chưa phải model detector chuyên sâu.
- Chưa có auth/user/cloud storage.
- Chưa có database thật.
- Chưa tối ưu code splitting frontend, build hiện có warning bundle lớn do Bootstrap/Konva.

## Cập nhật phase CROHME-like + M4

Đã triển khai tiếp theo file `CROHME_M4_NEXT_PHASE_SPEC.md`, không viết lại app từ đầu.

### Backend mới

- Thêm `detector.py`:
  - preprocess page để khử nền/khử bóng nhẹ;
  - connected components;
  - lọc component nhỏ, đường kẻ, artifact biên;
  - gom component thành candidate;
  - phân loại `single_expression`, `multiline_block`, `fragment`, `noise`, `uncertain`.
- Thêm `normalization.py`:
  - tạo `original_crop.png`;
  - tạo `cleaned_crop.png`;
  - tạo `binary_black_on_white.png`;
  - tạo `normalized_crohme.png`;
  - ảnh normalized là PNG grayscale/binary nền đen chữ trắng, target height 128;
  - lưu `components_debug.png` và `normalization_debug.json`.
- Thêm `recognition.py`:
  - adapter M4 `local_http`;
  - mặc định `Run M4` gửi ảnh `normalized_crohme.png` tới M4 HTTP server thật;
  - mock recognizer chỉ dùng khi đặt rõ `M4_BACKEND=mock`;
  - không chạy M4 cho `noise`, `fragment`, `multiline_block`, `uncertain`, `rejected`.
- Thêm `latex_tools.py`:
  - clean LaTeX nhẹ;
  - validate ngoặc/basic LaTeX.
- Thêm `tools/mock_m4_server.py` để chạy mock M4 HTTP server nếu cần.

### API mới

- `POST /api/expressions/{id}/normalize`
- `POST /api/pages/{id}/normalize-all`
- `POST /api/expressions/{id}/recognize`
- `POST /api/pages/{id}/recognize-accepted`
- `PATCH /api/expressions/{id}/latex`
- `POST /api/export/crohme-m4`

### Frontend mới

- Thêm nút header:
  - `Normalize All`;
  - `Run M4 Accepted`;
  - `Export M4 Dataset`.
- Inspector hiển thị 4 preview:
  - Original crop;
  - Cleaned crop;
  - Binary crop;
  - M4-ready crop.
- Inspector có panel Recognition:
  - Normalize Preview;
  - Run M4;
  - Latex Raw;
  - Latex Clean/Edit;
  - Save LaTeX;
  - Copy LaTeX;
  - Copy Markdown;
  - render visual bằng KaTeX.
- Queue hiển thị thêm badge candidate type.
- Thêm layer label `normalized` và `m4_ready`.

### Export CROHME/M4 dataset

ZIP mới có cấu trúc:

```text
metadata.json
metadata.jsonl
project.json
pages/
crops_original/
crops_crohme_like/
latex/
render/
debug/
```

### Kiểm thử phase mới

- `python -m py_compile ...`: pass.
- `npm run build`: pass.
- Playwright E2E phase M4: pass.
- Kiểm tra normalized image:
  - shape example: `128 x 86`;
  - pixel min/max: `0/255`;
  - unique pixel sample: `[0, 255]`;
  - foreground ratio example: `0.065`.
- Kiểm tra ZIP export:
  - có `metadata.jsonl`;
  - có `crops_crohme_like`;
  - có `latex`;
  - có `debug`.

### Cấu hình M4 thật

Mặc định backend dùng `local_http` và gọi M4 HTTP server thật. Cấu hình:

```bash
M4_BACKEND=local_http
M4_API_URL=http://127.0.0.1:7860/recognize
M4_TIMEOUT_SECONDS=120
M4_IMAGE_FIELD=image
```

Chỉ bật mock khi test/dev không có M4 thật:

```bash
M4_BACKEND=mock
```

Contract request:

```text
POST ${M4_API_URL}
multipart/form-data
image = normalized_crohme.png
```

Response hỗ trợ:

```json
{"latex":"...","confidence":0.91}
```

hoặc:

```json
{"pred":"...","score":0.91}
```
