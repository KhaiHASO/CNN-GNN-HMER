Đúng ý ông thì logic chi tiết phải hiểu như vầy:

**CROHME-like normalizer không nhận dạng toán. Nó chỉ làm nhiệm vụ biến ảnh lộn xộn thành nhiều ảnh biểu thức riêng, sạch, đúng format mà model M4 quen khi train CROHME.**

Công thức tổng:

```text
Ảnh chụp lộn xộn
→ làm sạch toàn trang
→ tìm toàn bộ nét viết
→ gom nét thành từng biểu thức
→ crop từng biểu thức
→ chuẩn hóa giống CROHME
→ gửi từng ảnh vào M4
→ gom kết quả lại
```

---

# 1. Chuẩn CROHME-like cần đạt là gì?

Với model M4 của ông, “giống CROHME” nên hiểu là ảnh cuối cùng phải có đặc điểm:

```text
- Chỉ chứa 1 biểu thức toán duy nhất.
- Không còn nền giấy, bóng, chữ mờ phía sau.
- Nền trắng, nét đen hoặc nền đen, nét trắng tùy dataloader.
- Crop sát biểu thức nhưng không cụt nét.
- Có padding đều quanh biểu thức.
- Resize theo chiều cao cố định, ví dụ H = 128.
- Giữ nguyên tỉ lệ, không kéo méo ngang/dọc.
- Width được pad về bội số 16 hoặc 32.
```

Ví dụ ảnh chuẩn cuối:

```text
(a+b)^2 = a^2 + 2ab + b^2
```

phải thành một ảnh riêng kiểu:

```text
crohme_white_h128.png
```

hoặc:

```text
crohme_black_h128.png
```

---

# 2. Logic chương trình chia thành 2 tầng

## Tầng A — Page Scanner

Tầng này xử lý **nguyên tấm ảnh lộn xộn**.

Nhiệm vụ:

```text
- Làm sạch ảnh toàn trang.
- Tìm vùng có nét viết.
- Gom các nét thành từng cụm biểu thức.
- Tạo danh sách bbox cần quét.
```

## Tầng B — CROHME Normalizer

Tầng này xử lý **từng vùng biểu thức đã crop**.

Nhiệm vụ:

```text
- Làm sạch lại ROI.
- Binarize lại.
- Crop sát foreground.
- Padding.
- Resize H=128.
- Xuất ảnh đúng format cho M4.
```

M4 chỉ nằm sau cùng:

```text
crohme_like_image → M4 → latex
```

---

# 3. Luồng chi tiết toàn bộ chương trình

```text
Input image
  ↓
[1] Load image
  ↓
[2] Resize ảnh về kích thước xử lý
  ↓
[3] Sửa nghiêng / sửa méo trang nếu cần
  ↓
[4] Khử nền giấy, bóng sáng, texture
  ↓
[5] Binarize toàn trang
  ↓
[6] Tách connected components
  ↓
[7] Xóa nhiễu
  ↓
[8] Gom components thành vùng biểu thức
  ↓
[9] Sắp xếp vùng theo thứ tự đọc
  ↓
[10] Với từng vùng:
      crop → clean → binarize → crop sát → padding → resize → M4
  ↓
[11] Gom kết quả + xuất overlay + manifest.json
```

---

# 4. Bước 1 — Load ảnh

Input có thể là ảnh chụp điện thoại:

```python
image = cv2.imread(input_path)
```

Sau đó lấy kích thước:

```python
H, W = image.shape[:2]
```

Nếu ảnh quá lớn thì resize để xử lý nhanh hơn:

```python
max_side = 2200

scale = max_side / max(H, W) if max(H, W) > max_side else 1.0
image_work = resize(image, scale)
```

Nhưng phải lưu `scale`, vì bbox sau này cần map ngược về ảnh gốc.

---

# 5. Bước 2 — Sửa nghiêng trang

Ảnh chụp thường bị nghiêng nhẹ. Nếu không sửa, bbox và crop sẽ lệch.

Có 2 kiểu nghiêng:

```text
- Nghiêng toàn trang: tờ giấy bị xoay.
- Nghiêng dòng chữ: tay viết lệch.
```

Logic:

```python
gray = cv2.cvtColor(image_work, cv2.COLOR_BGR2GRAY)

# lấy foreground tạm
tmp_bin = quick_threshold(gray)

# tìm tọa độ các pixel nét viết
coords = np.column_stack(np.where(tmp_bin > 0))

# dùng minAreaRect để ước lượng góc
angle = cv2.minAreaRect(coords)[-1]

# nếu góc hợp lý thì xoay lại
if abs(angle) < 8:
    image_deskew = rotate(image_work, angle)
else:
    image_deskew = image_work
```

Không nên sửa quá mạnh. Nếu estimate góc lớn bất thường thì bỏ qua.

---

# 6. Bước 3 — Khử nền giấy

Đây là bước cực quan trọng để ảnh đời thực giống CROHME.

Ảnh giấy thường có:

```text
- nền xám
- bóng tối
- ánh sáng lệch
- chữ in mặt sau
- texture giấy
```

Logic khử nền:

```python
gray = cv2.cvtColor(image_deskew, cv2.COLOR_BGR2GRAY)

# ước lượng nền bằng blur lớn
background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)

# chia ảnh cho nền để cân bằng sáng
normalized = gray / (background + 1e-6) * 255
normalized = np.clip(normalized, 0, 255).astype(np.uint8)
```

Sau đó tăng tương phản nhẹ:

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clean_gray = clahe.apply(normalized)
```

Kết quả mong muốn:

```text
trước: giấy xám, bóng, chữ mờ
sau: nền gần trắng, nét bút nổi rõ
```

---

# 7. Bước 4 — Binarize toàn trang

Không nên chỉ dùng một threshold. Nên thử nhiều cách rồi chọn bản tốt nhất.

Các candidate:

```python
bin_otsu = otsu_threshold(clean_gray)
bin_adaptive = adaptive_threshold(clean_gray)
bin_sauvola = sauvola_threshold(clean_gray)
```

Sau đó chấm điểm từng bản.

Tiêu chí chọn threshold:

```text
- foreground ratio không quá thấp, không quá cao
- số connected components hợp lý
- nét không bị đứt vụn quá nhiều
- không có mảng đen lớn do bóng
- không mất dấu nhỏ
```

Ví dụ scoring:

```python
def score_binary(binary):
    fg_ratio = count_foreground(binary) / binary.size
    components = connected_components(binary)

    score = 0

    if 0.003 <= fg_ratio <= 0.25:
        score += 2
    else:
        score -= 3

    if component_count_reasonable(components):
        score += 2

    if not has_large_black_blob(components):
        score += 2

    if not too_fragmented(components):
        score += 1

    return score
```

Chọn bản tốt nhất:

```python
page_binary = max(
    [bin_otsu, bin_adaptive, bin_sauvola],
    key=score_binary
)
```

---

# 8. Bước 5 — Connected components

Sau khi có ảnh nhị phân, tìm tất cả cụm nét nhỏ.

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(page_binary)
```

Mỗi component có:

```text
x, y, w, h, area, center_x, center_y
```

Ví dụ:

```json
{
  "id": 17,
  "bbox": [512, 230, 19, 31],
  "area": 122,
  "center": [521.5, 245.5]
}
```

Lúc này một biểu thức như:

```text
(a+b)^2 = a^2 + 2ab + b^2
```

sẽ bị tách thành nhiều component nhỏ:

```text
(, a, +, b, ), 2, =, a, 2, +, 2, a, b, +, b, 2
```

Nhiệm vụ sau đó là gom chúng lại.

---

# 9. Bước 6 — Xóa nhiễu

Nhiễu thường là:

```text
- chấm bụi trên giấy
- vệt sáng tối
- mép giấy
- chữ mặt sau mờ
- đốm camera
```

Không được xóa theo ngưỡng cứng kiểu:

```python
if area < 20: remove
```

vì dấu chấm, dấu phẩy, số mũ có thể rất nhỏ.

Nên làm theo tương đối:

```python
median_area = median([c.area for c in components])
median_h = median([c.h for c in components])

for c in components:
    if c.area < 0.02 * median_area and c.is_far_from_all_components():
        remove(c)
    else:
        keep(c)
```

Rule giữ dấu nhỏ:

```text
Nếu component nhỏ nhưng nằm gần cụm nét lớn → giữ.
Nếu component nhỏ nằm phía trên bên phải ký tự → có thể là số mũ → giữ.
Nếu component nhỏ nằm phía dưới bên phải ký tự → có thể là chỉ số → giữ.
Nếu component nhỏ cô lập ở góc giấy → bỏ.
```

---

# 10. Bước 7 — Ước lượng kích thước nét/ký hiệu

Trước khi gom biểu thức, chương trình phải biết “khoảng cách gần” là bao nhiêu.

Không dùng pixel cứng. Phải dựa vào kích thước chữ trong ảnh.

Tính:

```python
symbol_h = median([c.h for c in valid_components])
symbol_w = median([c.w for c in valid_components])
stroke_area = median([c.area for c in valid_components])
```

Ví dụ:

```text
symbol_h = 34 px
symbol_w = 18 px
```

Từ đó suy ra:

```python
near_x = 1.8 * symbol_h
near_y = 0.8 * symbol_h
superscript_distance = 1.2 * symbol_h
```

Tức là nếu ảnh lớn thì khoảng cách gom lớn, ảnh nhỏ thì khoảng cách gom nhỏ.

---

# 11. Bước 8 — Gom component thành candidate biểu thức

Đây là lõi của Page Scanner.

Nên dùng 3 cách cùng lúc:

```text
1. Dilation-based grouping
2. Graph-based grouping
3. Projection-based splitting
```

---

## 11.1. Dilation-based grouping

Ý tưởng:

```text
Các ký hiệu trong cùng một biểu thức nằm gần nhau.
Ta làm dày/nối nét theo chiều ngang để chúng dính lại thành một vùng.
```

Code logic:

```python
kernel_w = int(1.2 * symbol_h)
kernel_h = int(0.35 * symbol_h)

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (kernel_w, kernel_h)
)

dilated = cv2.dilate(page_binary, kernel, iterations=1)

contours = cv2.findContours(dilated)
boxes = [cv2.boundingRect(cnt) for cnt in contours]
```

Kết quả:

```text
component riêng lẻ → bbox vùng biểu thức
```

Ưu điểm:

```text
- nhanh
- dễ làm
- tốt với công thức viết ngang
```

Nhược điểm:

```text
- nếu nhiều dòng gần nhau có thể merge nhầm
- nếu công thức cách xa dấu bằng có thể tách nhầm
```

---

## 11.2. Graph-based grouping

Đây là cách thông minh hơn.

Mỗi component là một node:

```text
node = một ký hiệu hoặc một phần ký hiệu
```

Nối cạnh giữa hai node nếu chúng có quan hệ không gian.

Hai component `a` và `b` được nối nếu:

```python
dx = horizontal_gap(a, b)
dy = vertical_gap(a, b)
y_overlap = overlap_y_ratio(a, b)

if dx < 1.8 * symbol_h and y_overlap > 0.2:
    connect(a, b)
```

Thêm rule số mũ:

```python
if b.is_small() and b.is_upper_right_of(a) and distance(a, b) < 1.3 * symbol_h:
    connect(a, b)
```

Thêm rule chỉ số dưới:

```python
if b.is_small() and b.is_lower_right_of(a) and distance(a, b) < 1.3 * symbol_h:
    connect(a, b)
```

Thêm rule phân số:

```python
if has_horizontal_line_between(a, b):
    connect(a, b)
```

Sau đó tìm connected subgraph:

```python
clusters = graph_connected_components(graph)
```

Mỗi cluster là một candidate biểu thức.

Ưu điểm:

```text
- ít tách sai số mũ/chỉ số
- giữ được phân số
- tốt hơn dilation nếu biểu thức có layout 2D
```

---

## 11.3. Projection-based splitting

Sau khi có vùng lớn, cần tách nhiều dòng.

Ví dụ ảnh có:

```text
(a+b)^2 = ...
(a-b)^2 = ...
a^2-b^2 = ...
```

Dùng projection theo trục Y:

```python
row_sum = np.sum(page_binary, axis=1)
```

Những vùng `row_sum` gần 0 là khoảng trắng giữa các dòng.

Logic:

```python
blank_rows = find_long_blank_segments(row_sum)

if blank_gap_height > 0.7 * symbol_h:
    split_region_by_y_gap()
```

Nhưng phải tránh tách sai phân số. Nếu trong vùng có đường ngang dài và phần tử trên/dưới gần nhau thì giữ chung.

---

# 12. Bước 9 — Merge và lọc candidate boxes

Ba phương pháp trên sẽ tạo ra nhiều bbox trùng nhau.

Ví dụ:

```text
dilation tạo box A
graph tạo box B
projection tạo box C
```

Cần merge lại.

Logic:

```python
all_boxes = boxes_dilation + boxes_graph + boxes_projection

boxes = remove_tiny_boxes(all_boxes)
boxes = merge_overlapping_boxes(boxes, iou_threshold=0.3)
boxes = merge_near_boxes_same_baseline(boxes)
boxes = sort_reading_order(boxes)
```

Rule merge:

```text
Nếu 2 bbox overlap nhiều → merge.
Nếu 2 bbox gần nhau, cùng baseline → merge.
Nếu bbox nhỏ nằm trên/phải bbox lớn → merge vì có thể là mũ.
Nếu bbox nhỏ nằm dưới/phải bbox lớn → merge vì có thể là chỉ số.
```

Rule bỏ:

```text
bbox quá nhỏ → có thể là noise
bbox quá cao/dài sát mép → có thể là vệt giấy
bbox chỉ có 1 component cô lập → bỏ hoặc đưa vào review
```

---

# 13. Bước 10 — Sắp xếp thứ tự quét

Vì ảnh có nhiều biểu thức, cần quét theo thứ tự đọc.

Không sort đơn giản bằng `y` rồi `x`, vì dòng chữ có thể hơi lệch.

Logic tốt hơn:

```python
def sort_reading_order(boxes):
    # gom box thành các dòng theo center_y
    lines = group_by_y_center(boxes, tolerance=0.8 * symbol_h)

    # sort dòng từ trên xuống
    lines = sorted(lines, key=lambda line: line.mean_y)

    # trong mỗi dòng sort từ trái sang phải
    for line in lines:
        line.boxes = sorted(line.boxes, key=lambda box: box.x1)

    return flatten(lines)
```

Kết quả:

```text
expr_001: dòng 1 bên trái
expr_002: dòng 2 bên trái
expr_003: dòng 3 bên trái
...
```

---

# 14. Bước 11 — Crop từng candidate

Với mỗi bbox:

```python
x1, y1, x2, y2 = box
```

Không crop đúng bbox ngay. Phải mở rộng một chút:

```python
pad_x = int(0.10 * (x2 - x1))
pad_y = int(0.25 * (y2 - y1))

x1 = max(0, x1 - pad_x)
y1 = max(0, y1 - pad_y)
x2 = min(W, x2 + pad_x)
y2 = min(H, y2 + pad_y)
```

Vì toán có:

```text
- số mũ nằm cao
- chỉ số nằm thấp
- dấu căn kéo dài
- dấu chấm nhỏ
- nét cuối có thể sát mép
```

---

# 15. Bước 12 — Làm sạch riêng từng ROI

Dù toàn trang đã clean, mỗi ROI vẫn cần clean lại.

```python
roi = crop(page_rectified, box)

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi_clean = clean_background(roi_gray)
roi_binary = binarize_auto(roi_clean)
roi_binary = remove_noise(roi_binary)
```

Lý do: threshold tốt cho toàn trang chưa chắc tốt cho từng vùng nhỏ.

---

# 16. Bước 13 — Crop sát foreground trong ROI

Sau khi binarize ROI, tìm bbox thật của nét:

```python
ys, xs = np.where(roi_binary == foreground_value)

fg_x1 = xs.min()
fg_y1 = ys.min()
fg_x2 = xs.max()
fg_y2 = ys.max()
```

Crop sát:

```python
expr = roi_binary[fg_y1:fg_y2+1, fg_x1:fg_x2+1]
```

Nếu không tìm thấy foreground:

```text
status = empty_roi
bỏ qua hoặc đưa vào review
```

---

# 17. Bước 14 — Padding CROHME-like

Sau khi crop sát, thêm padding.

Không padding cứng. Padding theo kích thước biểu thức:

```python
h, w = expr.shape

pad_top = max(8, int(0.18 * h))
pad_bottom = max(8, int(0.18 * h))
pad_left = max(8, int(0.08 * w))
pad_right = max(8, int(0.08 * w))
```

Tạo canvas trắng:

```python
canvas = np.ones(
    (h + pad_top + pad_bottom, w + pad_left + pad_right),
    dtype=np.uint8
) * 255

canvas[pad_top:pad_top+h, pad_left:pad_left+w] = expr
```

Tác dụng:

```text
- không cụt mũ/chỉ số
- model không bị thấy nét dính sát biên
- giống ảnh biểu thức isolated hơn
```

---

# 18. Bước 15 — Resize về chuẩn M4

Ví dụ ông chọn `target_height = 128`.

Giữ aspect ratio:

```python
target_h = 128
scale = target_h / canvas_h
target_w = int(canvas_w * scale)

resized = cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_AREA)
```

Sau đó pad width về bội số 16 hoặc 32:

```python
width_multiple = 16
new_w = ceil(target_w / width_multiple) * width_multiple

final = white_canvas(target_h, new_w)
final[:, :target_w] = resized
```

Không được resize ép:

```python
cv2.resize(expr, (512, 128))
```

nếu làm vậy sẽ méo biểu thức.

---

# 19. Bước 16 — Xuất hai phiên bản trắng/đen

Tùy model M4 train kiểu nào.

Nền trắng, nét đen:

```python
crohme_white = final
```

Nền đen, nét trắng:

```python
crohme_black = 255 - final
```

Lưu:

```text
expr_001/crohme_white_h128.png
expr_001/crohme_black_h128.png
```

---

# 20. Bước 17 — Quality Check trước khi gửi M4

Trước khi gửi vào M4, kiểm tra ảnh chuẩn hóa.

Tính:

```python
fg_ratio = foreground_pixels / total_pixels
touch_border = check_foreground_touch_border(final)
aspect_ratio = final_width / final_height
component_count = count_components(final)
```

Rule:

```text
fg_ratio quá thấp:
  → gần như ảnh trắng, bỏ

fg_ratio quá cao:
  → threshold lỗi hoặc dính mảng đen

touch_border = true:
  → crop có thể cụt, thử expand bbox

aspect_ratio quá lớn:
  → có thể merge nhiều biểu thức

aspect_ratio quá nhỏ:
  → có thể chỉ là ký hiệu rời

component_count quá cao:
  → có thể nhiễu hoặc cả đoạn chữ dài
```

Ví dụ:

```python
if fg_ratio < 0.003:
    status = "empty_or_too_light"

elif fg_ratio > 0.35:
    status = "too_dark_or_bad_threshold"

elif touch_border:
    status = "need_expand_bbox"

else:
    status = "ok"
```

Chỉ gửi M4 khi:

```text
status == ok
```

hoặc nếu muốn mạnh hơn thì gửi nhưng đánh dấu `low_quality`.

---

# 21. Bước 18 — Gửi vào M4

Mỗi biểu thức một request:

```python
for expr in normalized_expressions:
    result = m4.predict(expr.crohme_white_h128)
```

Output:

```json
{
  "expr_id": "expr_001",
  "bbox": [271, 238, 1218, 409],
  "image": "expr_001/crohme_white_h128.png",
  "latex": "(a+b)^2=a^2+2ab+b^2",
  "confidence": 0.94
}
```

---

# 22. Bước 19 — Feedback loop từ M4

Đây là phần giúp chương trình ngon hơn.

Nếu M4 trả confidence thấp hoặc LaTeX lỗi, không bỏ ngay. Thử lại.

## Trường hợp 1: crop cụt

Dấu hiệu:

```text
foreground chạm biên
confidence thấp
```

Xử lý:

```text
expand bbox thêm 15%
normalize lại
gửi lại M4
```

## Trường hợp 2: crop dư nhiều biểu thức

Dấu hiệu:

```text
ROI quá cao
có nhiều baseline
latex rối
```

Xử lý:

```text
split theo dòng
normalize từng dòng
gửi lại M4
```

## Trường hợp 3: bị tách đôi biểu thức

Ví dụ detect ra:

```text
(a+b)^2 =
```

và:

```text
a^2 + 2ab + b^2
```

Xử lý:

```text
nếu hai bbox gần nhau, cùng baseline
→ merge lại
→ normalize lại
→ gửi lại M4
```

## Trường hợp 4: threshold làm mất dấu

Xử lý:

```text
thử lại Otsu
thử lại Adaptive
thử lại Sauvola
chọn output có confidence M4 cao nhất
```

---

# 23. Logic chọn kết quả tốt nhất

Mỗi candidate có thể sinh nhiều phiên bản:

```text
version_1: crop gốc + Otsu
version_2: crop gốc + Adaptive
version_3: crop rộng hơn + Otsu
version_4: crop rộng hơn + Sauvola
version_5: merge với neighbor
```

Mỗi version gửi M4 và chấm điểm:

```python
final_score = (
    0.60 * m4_confidence
    + 0.20 * crop_quality
    + 0.10 * latex_validity
    + 0.10 * image_quality
)
```

Chọn version có `final_score` cao nhất.

---

# 24. Pseudo-code hoàn chỉnh

```python
def page_to_m4(image_path, config):
    # =========================
    # 1. LOAD
    # =========================
    image = load_image(image_path)
    image_work, scale = resize_for_processing(image, max_side=2200)

    # =========================
    # 2. PAGE PREPROCESS
    # =========================
    page = correct_perspective_if_possible(image_work)
    page = deskew_if_needed(page)

    gray = to_grayscale(page)
    clean_gray = remove_background(gray)
    clean_gray = enhance_contrast(clean_gray)

    # =========================
    # 3. PAGE BINARIZATION
    # =========================
    binary_candidates = [
        otsu(clean_gray),
        adaptive_gaussian(clean_gray),
        sauvola(clean_gray)
    ]

    page_binary = choose_best_binary(binary_candidates)

    # =========================
    # 4. COMPONENT EXTRACTION
    # =========================
    components = find_connected_components(page_binary)
    components = filter_noise_components(components)

    symbol_stats = estimate_symbol_statistics(components)

    # =========================
    # 5. REGION PROPOSAL
    # =========================
    boxes_dilation = detect_regions_by_dilation(
        page_binary,
        symbol_stats
    )

    boxes_graph = detect_regions_by_component_graph(
        components,
        symbol_stats
    )

    boxes_projection = split_regions_by_projection(
        page_binary,
        boxes_dilation
    )

    candidate_boxes = merge_and_filter_boxes(
        boxes_dilation,
        boxes_graph,
        boxes_projection,
        symbol_stats
    )

    candidate_boxes = sort_reading_order(candidate_boxes)

    # =========================
    # 6. SCAN QUEUE
    # =========================
    results = []

    for idx, box in enumerate(candidate_boxes):
        expr_id = f"expr_{idx+1:04d}"

        versions = []

        # crop thường
        versions += normalize_candidate_variants(
            page,
            box,
            config
        )

        # nếu crop nghi ngờ cụt thì crop rộng hơn
        expanded_box = expand_box(box, ratio=0.15)
        versions += normalize_candidate_variants(
            page,
            expanded_box,
            config
        )

        best = None

        for version in versions:
            qc = quality_check(version.image)

            if qc.is_too_bad:
                continue

            m4_result = call_m4(version.image)

            score = compute_final_score(
                m4_confidence=m4_result.confidence,
                quality=qc,
                latex=m4_result.latex
            )

            candidate_result = {
                "expr_id": expr_id,
                "bbox": version.bbox,
                "normalized_image": version.image_path,
                "latex": m4_result.latex,
                "confidence": m4_result.confidence,
                "score": score,
                "quality": qc.to_dict()
            }

            if best is None or score > best["score"]:
                best = candidate_result

        # fallback nếu chưa tốt
        if best is None or best["score"] < config.min_accept_score:
            retry_results = retry_split_merge_threshold(
                page=page,
                box=box,
                neighbor_boxes=candidate_boxes,
                config=config
            )
            best = choose_best([best] + retry_results)

        results.append(best)

    # =========================
    # 7. OUTPUT
    # =========================
    overlay = draw_boxes_and_latex(page, results)
    manifest = save_manifest(results)

    return manifest, overlay
```

---

# 25. Cấu trúc thư mục output

```text
outputs/job_001/
  page_original.jpg
  page_deskew.png
  page_clean_gray.png
  page_binary.png
  page_components.png
  page_candidates.png
  page_overlay.png

  expressions/
    expr_0001/
      roi_original.png
      roi_clean_gray.png
      roi_binary.png
      crohme_white_h128.png
      crohme_black_h128.png
      quality.json
      m4_result.json

    expr_0002/
      roi_original.png
      roi_clean_gray.png
      roi_binary.png
      crohme_white_h128.png
      crohme_black_h128.png
      quality.json
      m4_result.json

  manifest.json
```

---

# 26. Manifest chuẩn

```json
{
  "job_id": "job_001",
  "source_image": "input/page.jpg",
  "page_size": [1920, 1080],
  "expression_count": 2,
  "expressions": [
    {
      "expr_id": "expr_0001",
      "bbox_original": [271, 238, 1218, 409],
      "bbox_processed": [250, 220, 1260, 430],
      "normalized_image": "expressions/expr_0001/crohme_white_h128.png",
      "latex": "(a+b)^2=a^2+2ab+b^2",
      "m4_confidence": 0.94,
      "quality": {
        "foreground_ratio": 0.047,
        "touch_border": false,
        "aspect_ratio": 5.2,
        "component_count": 17,
        "status": "ok"
      }
    }
  ]
}
```

---

# 27. Cấu hình nên dùng trước

```yaml
page:
  max_side: 2200
  deskew: true
  perspective: false

background:
  method: "divide_blur"
  gaussian_sigma: 35
  clahe: true

binarization:
  mode: "auto"
  methods:
    - otsu
    - adaptive_gaussian
    - sauvola

components:
  remove_noise: true
  preserve_small_marks: true

scanner:
  methods:
    - dilation
    - component_graph
    - projection

  dilation:
    kernel_w_ratio: 1.2
    kernel_h_ratio: 0.35
    multi_scale: true

  graph:
    max_x_gap_ratio: 1.8
    max_y_gap_ratio: 0.8
    preserve_superscript: true
    preserve_subscript: true
    preserve_fraction: true

normalizer:
  target_height: 128
  width_multiple: 16
  preserve_aspect_ratio: true
  pad_top_ratio: 0.18
  pad_bottom_ratio: 0.18
  pad_left_ratio: 0.08
  pad_right_ratio: 0.08
  min_padding: 8
  save_white: true
  save_black: true

quality:
  min_foreground_ratio: 0.003
  max_foreground_ratio: 0.35
  retry_if_touch_border: true

m4:
  endpoint: "http://localhost:8000/predict"
  variant: "white"
  min_confidence: 0.75
```

---

# 28. Logic quan trọng nhất cần nhớ

Chương trình nên chạy theo kiểu:

```text
Không cố nhận dạng nguyên trang.
Không ép M4 xử lý ảnh lộn xộn.
Không chỉ threshold rồi đưa thẳng model.

Mà phải:
1. Trang lộn xộn → nhiều vùng biểu thức.
2. Mỗi vùng biểu thức → ảnh CROHME-like.
3. Mỗi ảnh CROHME-like → M4.
4. M4 confidence thấp → scanner tự thử lại crop/split/merge/threshold.
```

Nói gọn: **chuẩn hóa kiểu CROHME là đưa mọi ảnh về cùng một phân phối dữ liệu mà M4 đã quen: isolated expression, clean binary, tight crop, safe padding, fixed height, preserved aspect ratio.**
