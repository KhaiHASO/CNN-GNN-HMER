# Đặc tả triển khai tiếp: CROHME-like Normalization + M4 LaTeX Recognition

> Dùng file này làm prompt/spec cho agent tiếp tục phát triển trên code hiện có của **Expression Page Explorer**.
>
> Trạng thái hiện tại: app đã có React + TypeScript + Vite, Bootstrap, React-Konva ở frontend; FastAPI + OpenCV + NumPy + Pillow ở backend; đã upload ảnh, auto scan bbox, chỉnh sửa bbox, split/merge, accept/reject, queue và export metadata/crop. Không viết lại app từ đầu. Chỉ mở rộng đúng pipeline: **detect page/expression → chuẩn hóa CROHME-like → đưa vào M4 → trả LaTeX → render visual đẹp → export dataset**.

---

## 1. Mục tiêu phase mới

Xây dựng module hoàn chỉnh để biến ảnh chụp trang giấy lộn xộn thành các crop biểu thức toán học đơn lẻ theo chuẩn **M4-ready / CROHME-like bitmap**:

```text
Ảnh trang A4 / ảnh chụp lộn xộn
        ↓
Detect vùng biểu thức / dòng biểu thức
        ↓
Review bbox bằng giao diện hiện có
        ↓
Normalize từng biểu thức thành ảnh nền đen chữ trắng
        ↓
Gửi từng ảnh đã normalize vào model M4
        ↓
Nhận LaTeX raw
        ↓
Clean/validate LaTeX
        ↓
Render visual bằng KaTeX/MathJax
        ↓
Export ảnh normalized + LaTeX + metadata
```

M4 hiện chỉ nhận **một biểu thức/lần**, vì vậy mọi block nhiều dòng hoặc nhiều biểu thức phải được split trước khi gọi M4.

---

## 2. Nguyên tắc quan trọng

### 2.1. Không viết lại app

Code hiện có đã chạy được MVP. Agent phải giữ nguyên kiến trúc chính:

```text
App/
  backend/
    app/
      main.py
      models.py
      services.py
      storage.py
  frontend/
    src/
      main.tsx
      api.ts
      geometry.ts
      types.ts
      styles.css
```

Chỉ bổ sung module mới và cập nhật API/UI tương ứng.

### 2.2. Giữ Bootstrap, không quay lại Tailwind

Frontend hiện dùng Bootstrap + CSS custom. Không thêm Tailwind.

### 2.3. CROHME-like ở đây nghĩa là M4-ready bitmap

Không tuyên bố sinh lại CROHME gốc hoàn toàn nếu không có InkML/stroke ground truth. Ảnh chụp chỉ có raster bitmap nên mục tiêu thực tế là:

```text
CROHME-like bitmap dataset phục vụ M4
```

Chuẩn đầu ra bắt buộc:

```text
background = black, pixel value 0
foreground = white, pixel value 255
format     = PNG grayscale hoặc binary
content    = một biểu thức duy nhất
crop       = sát nội dung nhưng có padding đều
noise      = đã loại bỏ tối đa
skew       = đã deskew nhẹ nếu cần
border     = không dính mép giấy, đường kẻ, bóng, viền crop
```

---

## 3. Chuẩn ảnh normalized cho M4

Tạo một profile cấu hình tên `m4_crohme_like`.

### 3.1. Thông số mặc định

```python
NORMALIZATION_PROFILE = {
    "name": "m4_crohme_like",
    "output_mode": "white_on_black",
    "background_value": 0,
    "foreground_value": 255,
    "binarize": True,
    "remove_shadows": True,
    "deskew": True,
    "crop_to_content": True,
    "padding_px": 16,
    "min_padding_px": 8,
    "target_height": 128,
    "max_width": 1024,
    "preserve_aspect_ratio": True,
    "center_on_canvas": True,
    "thin_noise_min_area": 8,
    "component_min_area": 12,
    "save_debug_steps": True
}
```

### 3.2. Output cần lưu cho mỗi expression

Với mỗi expression `page_001_expr_002`, backend cần tạo:

```text
backend/data/projects/default/expressions/page_001_expr_002/
  original_crop.png              # crop từ ảnh gốc theo bbox
  cleaned_crop.png               # đã khử nền, khử bóng, tăng tương phản
  binary_black_on_white.png      # binary trung gian: nền trắng chữ đen
  normalized_crohme.png          # ảnh cuối cho M4: nền đen chữ trắng
  components_debug.png           # optional, hiển thị connected components
  normalization_debug.json       # thông số xử lý, warnings, quality
```

`normalized_crohme.png` là ảnh duy nhất dùng để gọi M4.

---

## 4. Pipeline tiền xử lý full page

Tạo file mới:

```text
App/backend/app/normalization.py
App/backend/app/detector.py
```

### 4.1. Page preprocessing

Hàm đề xuất:

```python
def preprocess_page_for_detection(image_bgr: np.ndarray) -> PagePreprocessResult:
    """
    Input: ảnh upload gốc.
    Output: gray, cleaned, binary, debug layers.
    """
```

Các bước:

```text
1. Load ảnh BGR.
2. Convert grayscale.
3. Resize nếu ảnh quá lớn để detect nhanh, nhưng lưu scale ratio.
4. Khử bóng/nền giấy:
   - background = morphology close/open kernel lớn hoặc Gaussian blur lớn;
   - corrected = gray / background * 255 hoặc cv2.divide(gray, background, scale=255).
5. Tăng tương phản bằng CLAHE.
6. Blur nhẹ bằng median hoặc Gaussian nhỏ.
7. Adaptive threshold hoặc Otsu.
8. Đảm bảo dạng black ink on white background ở bước detect.
9. Morphological close theo chiều ngang để gom ký tự trong cùng dòng.
10. Remove border/page edge artifact.
```

### 4.2. Page deskew

Deskew chỉ áp dụng nếu góc nghiêng nhỏ.

```text
- Tìm pixel foreground.
- Dùng minAreaRect hoặc Hough line để ước lượng góc.
- Nếu abs(angle) <= 7 độ thì rotate lại.
- Nếu góc quá lớn hoặc không chắc thì không deskew, chỉ warning.
```

Lưu ý: Không rotate quá mạnh làm méo biểu thức.

---

## 5. Detect expression trên một trang

Detector hiện là heuristic OpenCV MVP. Nâng cấp theo hướng ổn định hơn, không cần model deep learning ở phase này.

### 5.1. Connected components cơ bản

Hàm đề xuất:

```python
def detect_expression_candidates(page_id: str, image_bgr: np.ndarray) -> list[ExpressionCandidate]:
    pass
```

Pipeline:

```text
1. Lấy binary black ink on white background.
2. connectedComponentsWithStats.
3. Lọc component rác:
   - area quá nhỏ;
   - width/height quá nhỏ;
   - nằm sát mép ảnh;
   - aspect ratio dị thường;
   - foreground ratio quá thấp;
   - component giống đường kẻ ngang/dọc dài.
4. Gom component thành line bằng y-center clustering.
5. Trong mỗi line, gom các component gần nhau theo khoảng cách ngang.
6. Tạo bbox candidate.
7. Merge bbox gần nhau nếu cùng dòng và khoảng cách nhỏ.
8. Split block nhiều dòng nếu projection theo y có valley rõ.
9. Sắp xếp theo reading order: từ trên xuống, trái sang phải.
10. Gán type: single_expression / multiline_block / fragment / noise / uncertain.
```

### 5.2. Rule lọc noise bắt buộc

Những candidate sau không đưa vào queue chính như expression bình thường, mà auto đánh dấu `noise` hoặc `fragment`:

```text
- bbox area < 0.0001 * page_area
- width < 12 px hoặc height < 12 px
- foreground_ratio < 0.005
- aspect_ratio > 35 và height < 25 px: nghi đường kẻ ngang
- aspect_ratio < 0.05 và width < 25 px: nghi đường kẻ dọc
- bbox chạm mép ảnh trong margin 3 px và area nhỏ: nghi artifact biên
- số connected components <= 1 và area nhỏ: fragment
```

### 5.3. Detect multi-line block

Một bbox bị nghi là nhiều dòng khi:

```text
- horizontal projection có từ 2 cụm foreground lớn trở lên;
- chiều cao bbox lớn hơn median line height * 1.8;
- khoảng trống ngang giữa hai cụm y đủ lớn;
- có nhiều baseline khác nhau.
```

Khi phát hiện multi-line:

```text
status = "need_review"
warnings += ["MULTILINE_BLOCK_SUGGEST_SPLIT_H"]
candidate_type = "multiline_block"
```

Không được auto accept multi-line để đưa vào M4.

---

## 6. Normalize một expression crop

Tạo hàm chính:

```python
def normalize_expression_crop(
    image_bgr: np.ndarray,
    bbox: BBox,
    profile: NormalizationProfile
) -> NormalizationResult:
    pass
```

### 6.1. Bước xử lý chi tiết

```text
1. Crop bbox từ ảnh gốc, mở rộng nhẹ 2-4 px để tránh cắt mất nét.
2. Convert grayscale.
3. Khử bóng/nền bằng cv2.divide hoặc background subtraction.
4. Tăng tương phản bằng CLAHE.
5. Binarize:
   - ưu tiên adaptive threshold cho ảnh chụp không đều sáng;
   - fallback Otsu nếu adaptive tạo nhiều noise.
6. Chuẩn hóa orientation:
   - detect foreground là chữ đen trên nền trắng;
   - nếu ảnh bị đảo thì invert lại về black ink on white.
7. Remove noise bằng connected components:
   - bỏ component quá nhỏ;
   - giữ component có khả năng thuộc biểu thức.
8. Crop sát content theo bounding rect của foreground.
9. Thêm padding đều.
10. Resize theo target_height, preserve aspect ratio.
11. Nếu width > max_width thì scale xuống hoặc warning WIDTH_TOO_LARGE.
12. Tạo canvas đen.
13. Paste foreground trắng vào canvas.
14. Save `normalized_crohme.png`.
```

### 6.2. Quy tắc invert cuối

Ở bước cuối:

```python
# binary_black_on_white: background=255, ink=0
normalized = 255 - binary_black_on_white
# normalized: background=0, ink=255
```

Đảm bảo kiểm tra:

```python
assert normalized.dtype == np.uint8
assert normalized.min() in [0, 255]
assert normalized.max() in [0, 255]
```

Nếu có antialiasing do resize, cần threshold lại về 0/255 hoặc lưu thêm bản gray riêng.

### 6.3. Quality metrics

Mỗi lần normalize cần trả về:

```json
{
  "width": 640,
  "height": 128,
  "aspect_ratio": 5.0,
  "foreground_ratio": 0.071,
  "component_count": 43,
  "touch_border": false,
  "is_multiline": false,
  "is_fragment": false,
  "warnings": []
}
```

Warnings cần có:

```text
TOUCH_BORDER
LOW_FOREGROUND_RATIO
HIGH_FOREGROUND_RATIO
TOO_SMALL
TOO_WIDE
TOO_TALL
MULTILINE_BLOCK
POSSIBLE_NOISE
EMPTY_AFTER_NORMALIZE
SKEW_UNCERTAIN
```

---

## 7. Backend data model cần bổ sung

Cập nhật `models.py`.

### 7.1. Expression model

Bổ sung các field:

```python
class Expression(BaseModel):
    id: str
    page_id: str
    bbox: BBox
    status: Literal["need_review", "accepted", "rejected", "noise", "fragment"]
    candidate_type: Literal[
        "single_expression",
        "multiline_block",
        "fragment",
        "noise",
        "uncertain"
    ] = "uncertain"

    crop_url: Optional[str] = None
    cleaned_url: Optional[str] = None
    binary_url: Optional[str] = None
    normalized_url: Optional[str] = None
    components_url: Optional[str] = None

    quality: Optional[ExpressionQuality] = None
    warnings: list[str] = []

    latex_raw: Optional[str] = None
    latex_clean: Optional[str] = None
    latex_status: Literal[
        "not_run",
        "running",
        "ok",
        "syntax_error",
        "model_error"
    ] = "not_run"
    latex_confidence: Optional[float] = None
    latex_render_svg_url: Optional[str] = None
    latex_render_png_url: Optional[str] = None
    recognition_history: list[RecognitionRun] = []
```

### 7.2. Normalization result

```python
class NormalizationResult(BaseModel):
    expression_id: str
    original_crop_path: str
    cleaned_crop_path: str
    binary_path: str
    normalized_path: str
    debug_json_path: str
    quality: ExpressionQuality
    warnings: list[str]
```

### 7.3. Recognition result

```python
class RecognitionRun(BaseModel):
    run_id: str
    model_name: str = "M4"
    input_image_path: str
    latex_raw: str
    latex_clean: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    error: Optional[str] = None
    created_at: str
    elapsed_ms: Optional[int] = None
```

---

## 8. API cần thêm

Cập nhật `main.py` và `services.py`.

### 8.1. Normalize API

```text
POST /api/expressions/{expression_id}/normalize
```

Chức năng:

```text
- Lấy page image gốc.
- Lấy bbox expression.
- Chạy normalize_expression_crop.
- Lưu ảnh normalized và debug.
- Update expression.normalized_url, quality, warnings.
- Trả expression mới.
```

Response:

```json
{
  "expression": {...},
  "normalized_url": "/data/.../normalized_crohme.png",
  "quality": {...},
  "warnings": []
}
```

### 8.2. Normalize toàn bộ accepted/need_review

```text
POST /api/pages/{page_id}/normalize-all
```

Body:

```json
{
  "only_status": ["accepted", "need_review"],
  "skip_noise": true,
  "profile": "m4_crohme_like"
}
```

### 8.3. Run M4 cho một expression

```text
POST /api/expressions/{expression_id}/recognize
```

Chức năng:

```text
1. Nếu expression chưa có normalized_url thì tự normalize trước.
2. Nếu expression là multiline_block/noise/fragment thì không gọi M4, trả warning.
3. Gửi normalized_crohme.png vào M4.
4. Nhận latex_raw.
5. Clean/validate LaTeX.
6. Render visual.
7. Update expression.
```

### 8.4. Run M4 hàng loạt

```text
POST /api/pages/{page_id}/recognize-accepted
```

Chỉ chạy với expression:

```text
status = accepted
candidate_type = single_expression
normalized_url != null hoặc normalize được thành công
```

### 8.5. Update LaTeX thủ công

```text
PATCH /api/expressions/{expression_id}/latex
```

Body:

```json
{
  "latex_clean": "\\frac{x^2}{1+x^2}",
  "manual_override": true
}
```

Dùng khi M4 nhận sai, người dùng sửa lại.

---

## 9. Tích hợp M4

Tạo file:

```text
App/backend/app/recognition.py
```

### 9.1. Không hard-code M4

M4 có thể chạy theo nhiều kiểu. Thiết kế adapter để sau này đổi dễ.

```python
class M4Recognizer:
    def recognize(self, image_path: str) -> RecognitionResult:
        raise NotImplementedError
```

Cấu hình bằng `.env`:

```text
M4_BACKEND=local_http
M4_API_URL=http://127.0.0.1:7860/recognize
M4_TIMEOUT_SECONDS=120
M4_IMAGE_FIELD=image
```

Hỗ trợ 3 mode:

```text
local_http   : gọi API M4 đang chạy riêng qua HTTP
subprocess   : gọi script Python inference của M4 bằng subprocess
python_module: import trực tiếp model M4 nếu repo nằm chung máy
```

Phase này ưu tiên `local_http` vì dễ tích hợp qua ngrok/local port.

### 9.2. Contract với M4 HTTP server

Request:

```text
POST ${M4_API_URL}
Content-Type: multipart/form-data
field: image = normalized_crohme.png
```

M4 response nên hỗ trợ một trong hai dạng:

```json
{
  "latex": "(a+b)^n=\\sum_{k=0}^n C_n^k a^{n-k}b^k",
  "confidence": 0.91
}
```

hoặc:

```json
{
  "pred": "...",
  "score": 0.91
}
```

Adapter cần normalize response về format nội bộ.

### 9.3. Không gọi M4 với ảnh sai chuẩn

Trước khi gọi M4 phải check:

```text
- normalized file tồn tại;
- foreground_ratio không quá thấp;
- không phải noise/fragment;
- không phải multiline_block;
- width/height hợp lệ;
- không empty sau normalize.
```

Nếu fail, trả lỗi rõ ràng cho frontend.

---

## 10. LaTeX clean, validate và render

Tạo file:

```text
App/backend/app/latex_tools.py
```

### 10.1. Clean LaTeX mức an toàn

Không được tự ý sửa quá mạnh làm sai ý nghĩa toán. Chỉ clean nhẹ:

```text
- strip whitespace thừa;
- bỏ wrapper rác như \begin{matrix} ... \end{matrix} nếu chỉ có một dòng;
- chuẩn hóa khoảng trắng;
- thay một số token lỗi rõ ràng nếu không làm đổi nghĩa;
- giữ latex_raw để đối chiếu.
```

Ví dụ:

```python
def clean_latex(latex_raw: str) -> str:
    pass
```

### 10.2. Validate LaTeX

Backend có thể validate nhẹ bằng regex/cân bằng ngoặc. Frontend dùng KaTeX/MathJax để render và báo lỗi trực quan.

```python
def validate_latex_basic(latex: str) -> LatexValidationResult:
    """
    Check bracket balance, empty command, invalid begin/end obvious cases.
    Không cần chứng minh đúng toán học.
    """
```

### 10.3. Render visual đẹp

Frontend nên dùng KaTeX trước vì nhẹ và đẹp.

Cài thêm:

```bash
cd App/frontend
npm install katex react-katex
```

Import CSS:

```ts
import 'katex/dist/katex.min.css';
```

Tạo component:

```text
src/components/LatexPreview.tsx
```

Yêu cầu UI:

```text
- hiển thị latex_raw;
- hiển thị latex_clean;
- cho sửa latex_clean bằng textarea;
- render visual bằng KaTeX;
- nếu render lỗi thì hiện error message;
- có nút Save LaTeX;
- có nút Copy LaTeX;
- có nút Copy Markdown;
```

---

## 11. Frontend cần nâng cấp

### 11.1. Thêm layer mới

Hiện đã có:

```text
original / cleaned / binary / components
```

Bổ sung:

```text
normalized
m4_ready
```

Trong thực tế `normalized` và `m4_ready` có thể cùng ảnh `normalized_crohme.png`, nhưng UI đặt tên rõ để người dùng hiểu đây là ảnh cuối đưa vào M4.

### 11.2. Inspector mới

Inspector khi chọn expression phải có 4 preview:

```text
Original crop
Cleaned crop
Binary crop
M4-ready crop: black background, white foreground
```

Bên dưới thêm panel:

```text
Recognition
- Button: Normalize Preview
- Button: Run M4
- Button: Save LaTeX
- Button: Copy LaTeX
- Latex Raw
- Latex Clean/Edit
- Visual Render
- Recognition status
- Confidence
- Runtime
```

### 11.3. Badge trạng thái candidate

Mỗi bbox/queue item cần badge rõ:

```text
Single Expression
Multi-line: cần split
Fragment
Noise
Uncertain
```

Màu gợi ý:

```text
single_expression: xanh
multiline_block : tím/cam
fragment        : xám
noise           : đỏ nhạt
uncertain       : vàng
```

### 11.4. Warning UX

Nếu chọn bbox multi-line, inspector phải hiện cảnh báo nổi bật:

```text
Block này có vẻ chứa nhiều dòng/biểu thức. M4 chỉ nhận một biểu thức/lần. Hãy dùng Split ngang trước khi Run M4.
```

Nút `Run M4` bị disable nếu:

```text
candidate_type != single_expression
status in [rejected, noise, fragment]
quality.empty_after_normalize = true
```

### 11.5. Queue filter

Thêm filter ở Expression Queue:

```text
All
Accepted
Need Review
Warnings
Single
Multi-line
Noise/Fragment
Recognized
LaTeX Error
```

---

## 12. Export dataset

Nâng cấp export ZIP.

### 12.1. Cấu trúc export mới

```text
export_crohme_m4_dataset.zip
  project.json
  metadata.json
  metadata.jsonl

  pages/
    page_001_original.png
    page_001_cleaned.png
    page_001_binary.png

  crops_original/
    page_001_expr_001.png
    page_001_expr_002.png

  crops_crohme_like/
    page_001_expr_001.png
    page_001_expr_002.png

  latex/
    page_001_expr_001.txt
    page_001_expr_002.txt

  render/
    page_001_expr_001.svg
    page_001_expr_001.png

  debug/
    page_001_expr_001_normalization.json
    page_001_expr_001_components.png
```

### 12.2. metadata.jsonl mỗi dòng

```json
{
  "id": "page_001_expr_002",
  "page_id": "page_001",
  "source_image": "pages/page_001_original.png",
  "bbox": {"x": 310, "y": 218, "w": 700, "h": 110},
  "status": "accepted",
  "candidate_type": "single_expression",
  "normalized_image": "crops_crohme_like/page_001_expr_002.png",
  "latex_raw": "...",
  "latex_clean": "...",
  "latex_status": "ok",
  "quality": {
    "width": 640,
    "height": 128,
    "foreground_ratio": 0.071,
    "touch_border": false,
    "is_multiline": false
  },
  "warnings": []
}
```

### 12.3. Export mode

Thêm lựa chọn:

```text
Export all
Export accepted only
Export accepted + recognized
Export M4-ready dataset only
```

Mặc định nên là:

```text
accepted + recognized
```

---

## 13. Backend task checklist cho agent

### 13.1. File mới

Tạo các file:

```text
App/backend/app/normalization.py
App/backend/app/detector.py
App/backend/app/recognition.py
App/backend/app/latex_tools.py
```

### 13.2. Cập nhật file cũ

Cập nhật:

```text
App/backend/app/models.py
App/backend/app/services.py
App/backend/app/storage.py
App/backend/app/main.py
App/backend/requirements.txt
```

Thêm requirements nếu cần:

```text
python-multipart
requests
scikit-image     # optional, nếu dùng threshold_sauvola hoặc morphology nâng cao
```

Không thêm package nặng nếu OpenCV làm được.

### 13.3. Endpoint bắt buộc

```text
POST /api/expressions/{id}/normalize
POST /api/pages/{id}/normalize-all
POST /api/expressions/{id}/recognize
POST /api/pages/{id}/recognize-accepted
PATCH /api/expressions/{id}/latex
POST /api/export/crohme-m4
```

---

## 14. Frontend task checklist cho agent

### 14.1. File/component mới

Tạo hoặc tách component:

```text
App/frontend/src/components/NormalizedPreview.tsx
App/frontend/src/components/RecognitionPanel.tsx
App/frontend/src/components/LatexPreview.tsx
App/frontend/src/components/QualityBadge.tsx
App/frontend/src/components/QueueFilter.tsx
```

Nếu app hiện đang để nhiều thứ trong `main.tsx`, có thể refactor vừa đủ, không cần đại phẫu.

### 14.2. API client

Cập nhật `api.ts`:

```ts
normalizeExpression(expressionId: string)
normalizePage(pageId: string, options?: NormalizeAllOptions)
recognizeExpression(expressionId: string)
recognizeAccepted(pageId: string)
updateLatex(expressionId: string, latexClean: string)
exportCrohmeM4(options: ExportOptions)
```

### 14.3. UI button mới

Header:

```text
Normalize All
Run M4 Accepted
Export M4 Dataset
```

Inspector:

```text
Normalize Preview
Run M4
Save LaTeX
Copy LaTeX
```

Layer toolbar:

```text
original | cleaned | binary | normalized | components
```

---

## 15. Logic trạng thái expression

### 15.1. Status nghiệp vụ

```text
need_review : cần người dùng duyệt
accepted    : bbox được chấp nhận là một expression hợp lệ
rejected    : người dùng loại bỏ
noise       : hệ thống tự nhận là rác
fragment    : mảnh nhỏ, có thể merge hoặc reject
```

### 15.2. Candidate type

```text
single_expression : có thể normalize + gọi M4
multiline_block   : phải split trước
fragment          : mảnh nhỏ, không gọi M4
noise             : rác, không gọi M4
uncertain         : cần review
```

### 15.3. LaTeX status

```text
not_run      : chưa chạy M4
running      : đang chạy
ok           : nhận dạng và render được
syntax_error : M4 có output nhưng LaTeX render lỗi
model_error  : lỗi khi gọi M4
manual       : người dùng đã sửa thủ công
```

---

## 16. Thuật toán split gợi ý

### 16.1. Split ngang tự động

Khi user bấm `Split H` trên một bbox multi-line:

```text
1. Lấy binary crop.
2. Tính horizontal projection: số pixel foreground theo từng hàng.
3. Smooth projection.
4. Tìm valley dài giữa các cụm foreground.
5. Nếu tìm được valley hợp lệ, split theo valley.
6. Nếu không tìm được, split chính giữa như hiện tại.
7. Sau split, normalize lại từng bbox con.
```

### 16.2. Split dọc tự động

Khi user bấm `Split V`:

```text
1. Tính vertical projection.
2. Tìm khoảng trắng dọc đủ rộng.
3. Split theo valley gần vị trí người dùng chọn hoặc valley lớn nhất.
4. Nếu không có valley, split chính giữa.
```

---

## 17. Kiểm thử bắt buộc

### 17.1. Backend unit test tối thiểu

Tạo test hoặc script kiểm tra:

```text
- normalize crop tạo ảnh nền đen chữ trắng;
- normalized không empty;
- foreground_ratio nằm trong khoảng hợp lý;
- bbox noise bị đánh dấu noise/fragment;
- multiline block sinh warning MULTILINE_BLOCK;
- không gọi M4 nếu candidate_type không phải single_expression;
- update latex lưu được;
- export zip có đúng file normalized + latex + metadata.
```

### 17.2. Frontend E2E tối thiểu

Dùng Playwright như phase trước:

```text
1. Upload ảnh demo.
2. Auto scan.
3. Chọn expression.
4. Bấm Normalize Preview.
5. Kiểm tra M4-ready preview hiển thị nền đen chữ trắng.
6. Accept expression.
7. Bấm Run M4 với mock M4 API.
8. Kiểm tra LaTeX hiện trong panel.
9. Kiểm tra render visual hiện đẹp.
10. Sửa LaTeX thủ công và Save.
11. Export M4 dataset.
12. Kiểm tra zip có normalized_crohme.png và latex txt.
```

### 17.3. Mock M4 server để test

Tạo script:

```text
App/backend/tools/mock_m4_server.py
```

Response cố định:

```json
{
  "latex": "(a+b)^n=\\sum_{k=0}^{n} C_n^k a^{n-k}b^k",
  "confidence": 0.99
}
```

Dùng để test UI trước khi M4 thật chạy.

---

## 18. Tiêu chí hoàn thành

Phase này chỉ được xem là xong khi đạt đủ:

```text
[ ] Upload một trang A4 có nhiều biểu thức.
[ ] Auto scan tách được candidate hợp lý, ít rác hơn MVP.
[ ] Rác nhỏ/đường biên không còn lẫn nhiều vào queue chính.
[ ] Multi-line block có warning rõ và không gọi M4 trực tiếp.
[ ] Chọn một expression bất kỳ và bấm Normalize Preview được.
[ ] Preview cuối là nền đen chữ trắng đúng chuẩn M4-ready.
[ ] Bấm Run M4 nhận được LaTeX.
[ ] LaTeX render đẹp bằng KaTeX/MathJax.
[ ] Người dùng sửa LaTeX được nếu model sai.
[ ] Export được dataset gồm ảnh normalized + latex + metadata.
[ ] Build frontend pass.
[ ] Backend py_compile pass.
[ ] Playwright E2E pass với mock M4.
```

---

## 19. Lưu ý kỹ thuật quan trọng

### 19.1. Đừng optimize nhận dạng trước khi normalize ổn

M4 có mạnh đến đâu cũng fail nếu crop đưa vào còn:

```text
- dính nhiều dòng;
- dính mép giấy;
- nền xám/bóng;
- chữ quá mảnh;
- quá nhiều noise;
- crop quá sát mất mũ/chỉ số;
- chưa đúng nền đen chữ trắng.
```

Ưu tiên thứ tự:

```text
Normalize đúng → Split đúng → M4 → LaTeX render
```

### 19.2. Luôn lưu ảnh debug

Mỗi expression nên có debug để nhìn lại pipeline sai ở đâu:

```text
original_crop → cleaned_crop → binary → normalized_crohme → M4 output
```

Không chỉ trả ảnh cuối.

### 19.3. Không auto sửa LaTeX quá thông minh

LaTeX cleaner chỉ làm cho output render được trong trường hợp lỗi nhẹ. Không được tự biến đổi công thức mạnh đến mức sai toán.

Luôn giữ:

```text
latex_raw
latex_clean
manual_override
```

### 19.4. M4 chỉ nhận single expression

Mọi chức năng batch recognition phải bỏ qua:

```text
multiline_block
noise
fragment
rejected
uncertain chưa accept
```

---

## 20. Prompt ngắn đưa thẳng cho coding agent

```text
Bạn hãy tiếp tục phát triển app Expression Page Explorer hiện có, không viết lại từ đầu. App hiện có React + TypeScript + Vite + Bootstrap + React-Konva ở frontend và FastAPI + OpenCV + Pillow ở backend. MVP đã có upload ảnh, auto scan bbox, chỉnh bbox, queue, accept/reject, split/merge, export metadata/crop.

Nhiệm vụ phase mới: xây dựng pipeline CROHME-like/M4-ready. Với mỗi bbox expression, backend phải normalize thành ảnh PNG nền đen chữ trắng, crop sát nội dung, padding đều, khử bóng/nền, deskew nhẹ, remove noise, resize preserve aspect ratio. File cuối tên normalized_crohme.png và đây là ảnh duy nhất đưa vào M4.

Bổ sung detector để giảm rác: lọc component nhỏ, đường biên, đường kẻ, foreground ratio thấp; phân loại candidate_type gồm single_expression, multiline_block, fragment, noise, uncertain. Nếu multi-line thì warning MULTILINE_BLOCK_SUGGEST_SPLIT_H và không cho Run M4.

Tích hợp M4 qua adapter recognition.py, ưu tiên local_http với env M4_API_URL. Thêm API normalize, normalize-all, recognize, recognize-accepted, update-latex, export-crohme-m4. Sau M4, lưu latex_raw, latex_clean, confidence, latex_status. Frontend dùng KaTeX hoặc MathJax để render LaTeX đẹp, có panel cho sửa và lưu LaTeX thủ công.

Cập nhật UI: thêm M4-ready/normalized preview trong Inspector, nút Normalize Preview, Run M4, Save LaTeX, Copy LaTeX, Export M4 Dataset; thêm layer normalized; thêm badge candidate_type và warning rõ. Export ZIP phải có crops_crohme_like, latex, render, metadata.jsonl.

Yêu cầu test: build frontend pass, backend py_compile pass, Playwright E2E pass với mock M4 server. Không dùng Tailwind. Không gọi M4 cho noise/fragment/multiline/rejected.
```
