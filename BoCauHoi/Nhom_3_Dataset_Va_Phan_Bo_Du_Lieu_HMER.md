# NHÓM 3 — DATASET VÀ PHÂN BỐ DỮ LIỆU

> **Mục tiêu nhóm:** Nắm chắc nguồn dữ liệu, cách chia tập, số lượng mẫu, dạng dữ liệu mà mô hình thật sự sử dụng, quy trình tiền xử lý, nguồn tạo vocabulary, nguy cơ rò rỉ dữ liệu và bằng chứng cần có cho mọi nhận định về phân bố dữ liệu.

---

## 0. Bốn kết luận phải chốt trước khi trả lời hội đồng

1. **Repo thực nghiệm chính dùng CROHME.** Việc có file cấu hình HME100K không đồng nghĩa luận văn đã huấn luyện và có kết quả trên HME100K.
2. **Mô hình nhận ảnh raster một kênh**, đọc từ `images.pkl`; model không nhận trực tiếp chuỗi stroke online khi train hoặc inference.
3. **Repo hiện không có validation set độc lập.** Trong `HMEDatamodule.setup()`, `val_dataset` và `test_dataset` cùng lấy từ biến `test_folder`. Cấu hình hiện đặt `test_folder: 2014`, trong khi checkpoint được chọn theo `val_ExpRate`. Vì thế CROHME 2014 đang tham gia model selection.
4. Các thống kê như phân bố độ dài, tần suất cấu trúc, số mẫu tích phân có cận, token hiếm và trùng lặp **chưa được repo xuất thành báo cáo chính thức**. Không được tự ước lượng hoặc nói “dataset không có” trước khi chạy audit.

---

## 1. Bảng dữ liệu có thể chốt ở thời điểm hiện tại

| Tập | Vai trò benchmark thông thường | Vai trò trong repo hiện tại | Số mẫu chuẩn thường được báo cáo | Ghi chú |
|---|---|---|---:|---|
| `train` | Huấn luyện | Huấn luyện | 8.836 | Cần đếm lại số hiệu dụng sau bộ lọc |
| `2014` | Test CROHME 2014 | Validation mặc định và cũng được báo cáo test | 986 | Không còn hoàn toàn untouched nếu dùng chọn checkpoint |
| `2016` | Test CROHME 2016 | Đánh giá sau huấn luyện | 1.147 | Độc lập hơn nếu không dùng để tune |
| `2019` | Test CROHME 2019 | Đánh giá sau huấn luyện | 1.199 | Độc lập hơn nếu không dùng để tune |

### Phân biệt số mẫu thô và số mẫu hiệu dụng

Code `data_iterator()` chỉ loại mẫu khi `is_train=True` nếu:

- độ dài nhãn lớn hơn 200 token;
- diện tích ảnh lớn hơn `max_size`, mặc định 320.000 pixel.

Do đó:

$$ N_{\text{train hiệu dụng}} \leq N_{\text{train thô}} $$

Con số 8.836 là quy mô train chuẩn thường được báo cáo. Muốn nói chính xác bao nhiêu mẫu thật sự đi qua pipeline train của repo, phải chạy script audit ở cuối tài liệu.

---

# Câu 1 — Repo hiện tại sử dụng chính xác những bộ dữ liệu nào cho train, validation và test?

## 1.1. Bản trả lời nhanh 45–60 giây

> **Repo hiện sử dụng gói CROHME tại `data/crohme`. Thư mục `train` được dùng huấn luyện. Tuy nhiên code không có validation set riêng: cả `val_dataset` và `test_dataset` đều được tạo từ biến `test_folder`. Trong cấu hình hiện tại, `test_folder` là `2014`, nên CROHME 2014 được dùng để chọn checkpoint theo `val_ExpRate`.**
>
> **Sau huấn luyện, các script thay `test_folder` để đánh giá trên CROHME 2014, 2016 và 2019. Vì vậy em phải nói rõ 2014 là model-selection set trong protocol hiện tại, còn 2016 và 2019 là các tập đánh giá độc lập hơn nếu chúng không được dùng để điều chỉnh mô hình.**

## 1.2. Bằng chứng trong config và loader

Trong `config/crohme.yaml`:

```yaml
data:
  folder: data/crohme
  test_folder: 2014
```

Trong `HMEDatamodule.setup()`:

```python
self.train_dataset = HMEDataset(
    build_dataset(self.folder, "train", ...)
)

self.val_dataset = HMEDataset(
    build_dataset(self.folder, self.test_folder, ...)
)

self.test_dataset = HMEDataset(
    build_dataset(self.folder, self.test_folder, ...)
)
```

Vì vậy:

- train luôn lấy `train`;
- validation lấy `test_folder`;
- test cũng lấy `test_folder`.

## 1.3. Checkpoint được chọn bằng tập nào?

Cấu hình checkpoint có dạng:

```yaml
monitor: val_ExpRate
mode: max
save_top_k: 1
```

Với:

```yaml
test_folder: 2014
```

checkpoint tốt nhất được chọn theo ExpRate trên CROHME 2014.

## 1.4. Đây có phải data leakage không?

Cần phân biệt hai cấp độ.

### Không phải leakage trực tiếp vào gradient

- Nhãn 2014 không nằm trong training loss.
- Model không backpropagate trên 2014.

### Nhưng có model-selection contamination

- Kết quả 2014 quyết định epoch/checkpoint nào được giữ.
- Nếu hyperparameter hoặc kiến trúc được điều chỉnh sau khi xem 2014, tập này còn tham gia development ở cấp độ con người.
- Sau đó cùng score 2014 được báo cáo như test.

Câu trả lời khoa học:

> **CROHME 2014 không tham gia cập nhật gradient, nhưng tham gia model selection. Do đó trong protocol hiện tại, nó đóng vai trò validation và không nên đồng thời được xem là test hoàn toàn untouched.**

## 1.5. HME100K có được dùng không?

Repo có file cấu hình HME100K, nhưng sự tồn tại của config không chứng minh:

- đã train;
- có checkpoint;
- có log;
- có output đánh giá;
- có bảng kết quả.

Cách nói an toàn:

> **Phạm vi kết quả hiện có của luận văn là CROHME. HME100K mới là khả năng cấu hình hoặc hướng mở rộng, trừ khi có đầy đủ run, checkpoint và output thực nghiệm.**

## 1.6. Cách sửa protocol tốt nhất

### Phương án chuẩn

```text
Official CROHME train
├── train_internal
└── validation_internal

Official benchmark tests
├── CROHME 2014
├── CROHME 2016
└── CROHME 2019
```

Quy trình:

1. Tách validation cố định từ train, ví dụ 5–10%.
2. Nếu có writer ID, ưu tiên writer-disjoint split.
3. Chọn checkpoint trên validation nội bộ.
4. Chốt mọi hyperparameter.
5. Chạy 2014, 2016 và 2019 một lần cuối.

### Nếu không kịp retrain

Ghi rõ:

> **Các run hiện tại dùng CROHME 2014 làm validation theo cấu hình pipeline; do đó kết quả 2014 chịu model-selection bias và cần được diễn giải thận trọng.**

## 1.7. Không nên nói

- “Repo có train, validation và ba test set hoàn toàn tách biệt.”
- “CROHME 2014 chỉ được dùng test.”
- “Không backpropagate trên test thì chắc chắn không có contamination.”
- “Có file HME100K nghĩa là luận văn đã thực nghiệm HME100K.”
- “2014, 2016 và 2019 đều độc lập như nhau.”

---

# Câu 2 — Mỗi tập có bao nhiêu mẫu và số liệu này được lấy từ file hoặc script nào?

## 2.1. Bản trả lời nhanh 45–60 giây

> **Quy mô benchmark thường được dùng là 8.836 mẫu train, 986 mẫu CROHME 2014, 1.147 mẫu CROHME 2016 và 1.199 mẫu CROHME 2019. Trong repo, số mẫu không được hard-code; `extract_data()` đọc từng dòng của `caption.txt`, lấy image ID và token, rồi ghép với ảnh trong `images.pkl`.**
>
> **Riêng train, code loại nhãn dài hơn 200 token hoặc ảnh có diện tích trên 320.000 pixel. Vì vậy báo cáo phải có hai con số: số mẫu thô trong `caption.txt` và số mẫu hiệu dụng sau bộ lọc.**

## 2.2. Các con số chuẩn

| Tập | Số mẫu chuẩn |
|---|---:|
| Train | 8.836 |
| CROHME 2014 | 986 |
| CROHME 2016 | 1.147 |
| CROHME 2019 | 1.199 |

Tổng số expression nếu chỉ cộng một lần từng tập:

$$ 8.836 + 986 + 1.147 + 1.199 = 12.168 $$

Không được gọi 12.168 là số mẫu huấn luyện, vì ba test set không được gộp vào train.

## 2.3. Repo đếm sample như thế nào?

`extract_data()` có logic:

```python
with open(split_dir / "images.pkl", "rb") as f:
    images = pickle.load(f)

with open(split_dir / "caption.txt", "r") as f:
    captions = f.readlines()

for line in captions:
    parts = line.strip().split()
    image_id = parts[0]
    formula_tokens = parts[1:]
    image = images[image_id]
    data.append((image_id, image, formula_tokens))
```

Một dòng hợp lệ trong `caption.txt` tương ứng một expression:

```text
image_id token_1 token_2 ... token_T
```

## 2.4. Nguồn sự thật nên tạo ra

Nên lưu artifact:

```text
dataset_statistics.json
```

Ví dụ:

```json
{
  "train": {
    "raw_caption_count": 8836,
    "effective_after_filter": 8800,
    "filtered_too_long": 12,
    "filtered_too_large": 24
  },
  "2014": {
    "caption_count": 986
  },
  "2016": {
    "caption_count": 1147
  },
  "2019": {
    "caption_count": 1199
  }
}
```

Các số trên chỉ là cấu trúc minh họa; phải thay bằng output thật của script.

## 2.5. Vì sao `len(train_dataset)` có thể không phải số expression?

Repo gom nhiều expression thành batch nội bộ trước khi đưa vào `HMEDataset`.

Sau đó `DataLoader` dùng `collate_fn` với giả định mỗi phần tử DataLoader đã là một batch.

Vì vậy:

```python
len(train_dataset)
```

có thể phản ánh số batch đã gom, không phải số expression.

Cách đúng:

- đếm dòng caption;
- hoặc cộng số expression bên trong từng batch;
- hoặc chạy audit trước và sau filter.

## 2.6. Bộ lọc train

Mặc định:

```python
maxlen = 200
max_size = 320000
```

Mẫu train bị loại khi:

```python
len(label_tokens) > 200
```

hoặc:

```python
height * width > 320000
```

Val/test không bị hai bộ lọc này vì điều kiện có `is_train`.

## 2.7. Câu trả lời khi bị hỏi “986 lấy từ đâu?”

> **Em lấy từ số dòng hợp lệ của `2014/caption.txt` trong đúng gói dữ liệu đã dùng, đồng thời đối chiếu với quy mô benchmark CROHME 2014. Em không lấy từ số batch vì batching được tạo động theo kích thước ảnh.**

## 2.8. Không nên nói

- “Số mẫu được khai báo trong YAML.”
- “Số batch chính là số expression.”
- “Train chắc chắn đủ 8.836 mẫu sau lọc.”
- “Test cũng bị lọc giống train.”
- “Tổng 12.168 đều được dùng huấn luyện.”

---

# Câu 3 — Dữ liệu gốc là stroke online hay ảnh raster offline; repo sử dụng dạng nào?

## 3.1. Bản trả lời nhanh 30–45 giây

> **CROHME gốc là dữ liệu online handwriting, thường lưu quỹ đạo nét bút trong InkML. Tuy nhiên repo hiện tại không đưa stroke sequence vào model. Dữ liệu đã được chuyển thành ảnh raster lưu trong `images.pkl`, còn nhãn token nằm trong `caption.txt`.**
>
> **Đầu vào model có dạng `[B, 1, H, W]`, nên bài toán mà luận văn thực hiện là offline image recognition, dù ảnh có nguồn gốc từ dữ liệu online.**

## 3.2. Phân biệt ba cấp độ

### Nguồn thu thập

Dữ liệu online có thể biểu diễn:

$$ S = \{(x_t,y_t,\text{pen-state}_t)\}_{t=1}^{N} $$

Thông tin có thể gồm:

- tọa độ;
- thứ tự điểm;
- pen-up/pen-down;
- thời gian;
- stroke grouping.

### Dạng lưu trong gói repo

```text
images.pkl
caption.txt
dictionary.txt
```

`images.pkl` chứa ảnh dưới dạng mảng NumPy.

### Dạng model nhận

Sau `ToTensor()` và padding:

$$ X \in \mathbb{R}^{B \times 1 \times H \times W} $$

Đây là offline raster input.

## 3.3. Thông tin nào bị mất khi raster hóa?

Model không trực tiếp biết:

- stroke order;
- hướng vẽ;
- thời điểm nhấc bút;
- tốc độ;
- áp lực;
- stroke nào được viết trước;
- stroke nào thuộc cùng một thao tác.

Model chỉ nhìn hình ảnh cuối cùng.

## 3.4. Điểm mạnh của offline input

Có thể áp dụng cho:

- ảnh scan;
- ảnh chụp;
- canvas xuất PNG;
- ảnh crop từ tài liệu;
- dữ liệu không có stroke metadata.

## 3.5. Điểm yếu

- mất tín hiệu thời gian;
- khó tách ký hiệu dính;
- khó giải thích nét giao nhau;
- nhạy với rasterization;
- có domain gap với ảnh camera.

## 3.6. Câu bẫy: “Người dùng vẽ trên canvas thì là online đúng không?”

> **Không nhất thiết. Nếu canvas chỉ xuất bitmap cuối cùng rồi gửi vào model thì model vẫn là offline. Chỉ khi model nhận chuỗi stroke theo thời gian mới gọi là online recognition.**

## 3.7. Không nên nói

- “CROHME là dataset offline nguyên gốc.”
- “Repo dùng cả stroke lẫn ảnh.”
- “Canvas đồng nghĩa online.”
- “Raster giữ nguyên toàn bộ thông tin InkML.”

---

# Câu 4 — Quy trình chuyển từ dữ liệu gốc sang ảnh đầu vào của mô hình được thực hiện như thế nào?

## 4.1. Bản trả lời nhanh 60 giây

> **Cần tách hai giai đoạn. Giai đoạn tạo gói dữ liệu là render InkML/stroke thành ảnh raster, chuẩn hóa nhãn thành chuỗi token và đóng gói thành `images.pkl`, `caption.txt`, `dictionary.txt`. Repo hiện chứa gói đã chuẩn bị nhưng không có đầy đủ script chứng minh thông số raster hóa gốc, nên em không được tự bịa DPI, độ dày nét hoặc margin.**
>
> **Giai đoạn model-side thì code xác định rõ: đọc mảng ảnh, tùy chọn scale augmentation, resize giữ tỷ lệ để chiều cao nằm trong 16–256 và chiều rộng 16–1024, chuyển tensor một kênh, gom batch theo diện tích, pad bằng 0 đến kích thước lớn nhất và tạo mask cho vùng padding. Cấu hình hiện tại tắt scale augmentation.**

## 4.2. Giai đoạn A — Từ InkML sang gói dữ liệu

Pipeline khái quát:

```text
InkML strokes
      ↓
Render stroke lên canvas
      ↓
Crop vùng nét
      ↓
Tạo ảnh grayscale hoặc binary
      ↓
Chuẩn hóa nhãn LaTeX thành token
      ↓
images.pkl + caption.txt + dictionary.txt
```

Nhưng repo hiện không đủ bằng chứng để khẳng định:

- độ phân giải render;
- độ dày stroke;
- margin crop;
- nền đen hay nền trắng ở bước tạo gói;
- quy tắc chuyển annotation gốc sang caption;
- quy tắc canonicalization.

Cách nói đúng:

> **Repo chứng minh trực tiếp bước loading và preprocessing sau khi đã có ảnh; bước render InkML sang ảnh thuộc dữ liệu đã chuẩn bị và cần truy tài liệu hoặc script nguồn nếu muốn mô tả chi tiết.**

## 4.3. Giai đoạn B — Đọc ảnh và nhãn

`extract_data()`:

1. mở `images.pkl`;
2. đọc `caption.txt`;
3. lấy image ID;
4. lấy danh sách token;
5. tra ảnh theo ID;
6. tạo tuple `(image_id, image, formula_tokens)`.

## 4.4. Giai đoạn C — Scale augmentation

Trong dataset:

```python
if is_train and scale_aug:
    transforms.append(ScaleAugmentation(0.7, 1.4))
```

Nhưng config hiện tại:

```yaml
scale_aug: false
```

Do đó không được viết rằng run hiện tại dùng:

- random rotation;
- Gaussian noise;
- horizontal flip;
- contrast enhancement;
- random crop

nếu code và log không chứng minh.

## 4.5. Giai đoạn D — Resize giữ tỷ lệ

Giới hạn:

```python
H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024
```

Nếu ảnh quá lớn:

$$ s = \min\left(\frac{256}{H},\frac{1024}{W}\right) $$

Nếu ảnh quá nhỏ:

$$ s = \max\left(\frac{16}{H},\frac{16}{W}\right) $$

Ảnh được scale đồng nhất theo hai trục, nên giữ tỷ lệ hình học.

## 4.6. Giai đoạn E — Chuyển tensor

```python
ToTensor()
```

Tensor một ảnh có dạng:

```text
[1, H, W]
```

## 4.7. Giai đoạn F — Dynamic batching

Dữ liệu được sort theo diện tích:

```python
height * width
```

Một batch được đóng khi:

```python
largest_image_area * number_of_samples > max_size
```

hoặc đạt giới hạn batch size.

Mục đích:

- giảm padding;
- tránh OOM;
- gom ảnh kích thước gần nhau.

## 4.8. Giai đoạn G — Padding và mask

Batch:

```python
x = zeros(B, 1, H_max, W_max)
x_mask = ones(B, H_max, W_max)
```

Vùng ảnh thật:

```python
x[..., :H_i, :W_i] = image_i
x_mask[..., :H_i, :W_i] = 0
```

Ý nghĩa:

- vùng padding có pixel 0;
- mask `True/1` là vùng padding;
- mask được truyền vào encoder/attention.

## 4.9. Những gì phải kiểm tra với demo

Ảnh demo phải khớp train về:

- polarity;
- mức xám;
- crop;
- margin;
- tỷ lệ;
- độ dày nét;
- kích thước;
- số kênh;
- scale;
- cách padding.

Nếu train dùng nền đen nét trắng nhưng app đưa nền trắng nét đen, model có thể suy giảm mạnh.

## 4.10. Không nên nói

- “Tất cả ảnh được resize về 128 pixel.”
- “Repo dùng Gaussian noise và horizontal flip.”
- “Ảnh được normalize mean/std” nếu code không có.
- “Em tự render InkML bằng script trong repo” nếu chưa tìm thấy script.
- “Val/test cũng bị loại ảnh quá lớn.”
- “Có mask nên padding chắc chắn không ảnh hưởng.”

---

# Câu 5 — CROHME 2014, 2016 và 2019 khác nhau ở những điểm nào liên quan đến phân bố dữ liệu?

## 5.1. Bản trả lời nhanh 60 giây

> **Ba tập là các test set từ những kỳ CROHME khác nhau, có quy mô 986, 1.147 và 1.199 expression. Vì được thu thập ở các đợt khác nhau, chúng có thể khác về người viết, phong cách nét, độ dài, tần suất token và loại cấu trúc.**
>
> **Tuy nhiên chỉ từ tên năm hoặc ExpRate em không được khẳng định tập nào có nhiều phân số, căn hay tích phân hơn. Những khác biệt đó phải được đo trực tiếp từ `caption.txt` và ảnh bằng thống kê độ dài, token, cấu trúc, kích thước và overlap.**

## 5.2. Khác biệt biết chắc

| Thuộc tính | 2014 | 2016 | 2019 |
|---|---:|---:|---:|
| Số expression | 986 | 1.147 | 1.199 |
| Kỳ competition | CROHME 2014 | CROHME 2016 | CROHME 2019 |
| Đợt thu thập | Khác nhau | Khác nhau | Khác nhau |
| Vai trò trong repo | Validation mặc định + test report | Test report | Test report |

## 5.3. Khác biệt cần đo

- mean, median, P90, P95 độ dài;
- tỷ lệ chuỗi dài hơn 50, 100, 150 token;
- tần suất `\frac`;
- tần suất `\sqrt`;
- tần suất `^`, `_`;
- tần suất `\int`, `\sum`, `\lim`;
- độ sâu ngoặc;
- cấu trúc lồng;
- token hiếm;
- kích thước ảnh;
- aspect ratio;
- writer diversity;
- độ dày hoặc độ nghiêng nét.

## 5.4. Mô hình hóa distribution shift

Có thể tồn tại:

$$ P_{2014}(X,Y) \neq P_{2016}(X,Y) \neq P_{2019}(X,Y) $$

Trong đó:

- $X$ là ảnh;
- $Y$ là chuỗi token.

## 5.5. Số mẫu nhiều hơn không đồng nghĩa khó hơn

2019 có nhiều mẫu hơn 2014, nhưng độ khó còn phụ thuộc:

- thành phần cấu trúc;
- độ dài;
- writer style;
- rare token;
- domain match với train.

Sample size lớn chủ yếu làm estimate ổn định hơn.

## 5.6. Cách nói khoa học

> **Các test set khác nhau về kỳ thu thập và quy mô. Chênh lệch kết quả cho thấy model nhạy với phân bố, nhưng nguyên nhân cụ thể phải được kiểm chứng bằng thống kê dữ liệu.**

## 5.7. Không nên nói

- “2019 khó nhất vì mới nhất.”
- “2016 dễ nhất vì một model cao hơn.”
- “2014 có ít cấu trúc phức tạp hơn” khi chưa đếm.
- “Số mẫu nhiều hơn làm ExpRate thấp.”
- “Ba tập cùng phân bố hoàn toàn.”

---

# Câu 6 — Vì sao cần đánh giá trên nhiều tập test thay vì chỉ dùng một tập?

## 6.1. Bản trả lời nhanh 45–60 giây

> **Một test set chỉ phản ánh hiệu quả trên một phân bố. Đánh giá 2014, 2016 và 2019 giúp kiểm tra thay đổi kiến trúc có ổn định qua nhiều đợt thu thập hay chỉ phù hợp một tập. Điều này quan trọng vì M3 chỉ nhỉnh hơn baseline trên 2016 nhưng không trên 2014 và 2019.**
>
> **Nhiều test set cũng giảm nguy cơ cherry-picking. Tuy nhiên 2014 đã được dùng chọn checkpoint trong protocol hiện tại, nên vai trò của nó phải được ghi rõ và không nên xem cả ba tập độc lập như nhau.**

## 6.2. Kiểm tra độ ổn định

Với model $m$, xét:

$$ \Delta_m^{2014},\quad \Delta_m^{2016},\quad \Delta_m^{2019} $$

Nếu cải tiến chỉ dương ở một tập nhưng âm ở hai tập:

- chưa có cải thiện ổn định;
- có khả năng dataset-specific effect.

## 6.3. Giảm cherry-picking

Nếu chỉ báo cáo năm model tốt nhất:

- kết luận bị thiên lệch;
- không biết có tổng quát hay không;
- khó so sánh với paper khác.

## 6.4. Hỗ trợ error analysis

Nếu cùng loại lỗi xuất hiện ở cả ba tập:

- khả năng cao là hạn chế mô hình.

Nếu lỗi chỉ tăng ở một năm:

- có thể là distribution shift.

## 6.5. Nhiều test set vẫn chưa đủ nếu protocol không sạch

Cần thêm:

- validation nội bộ;
- test untouched;
- nhiều seed;
- confidence interval;
- cùng decoding;
- cùng preprocessing.

## 6.6. Không nên nói

- “Ba test set đảm bảo tổng quát ngoài đời.”
- “Chỉ cần lấy trung bình là đủ.”
- “Ba năm là ba lần train độc lập” nếu cùng checkpoint.
- “Có nhiều test nên không cần validation.”

---

# Câu 7 — Có nguy cơ trùng người viết, trùng biểu thức hoặc rò rỉ dữ liệu giữa train và test không?

## 7.1. Bản trả lời nhanh 60–90 giây

> **Có bốn loại nguy cơ cần tách. Một là CROHME 2014 được dùng chọn checkpoint rồi báo cáo lại, đây là model-selection contamination. Hai là trùng filename hoặc trùng ảnh giữa train và test, đây là leakage nghiêm trọng và cần kiểm tra bằng hash. Ba là cùng chuỗi LaTeX có thể xuất hiện ở train và test; điều này chưa chắc là leakage nếu ảnh do người viết khác tạo ra, nhưng phải báo cáo. Bốn là trùng người viết, cần writer metadata mới kết luận được.**
>
> **Repo chưa có audit chính thức, nên em không được nói “không trùng” chỉ vì các thư mục khác nhau. Em phải kiểm tra filename, SHA-256 ảnh, normalized LaTeX và writer ID nếu có.**

## 7.2. Bốn loại overlap

### A. Filename overlap

$$ F_{\text{train}} \cap F_{\text{test}} $$

Nếu khác rỗng, cần điều tra.

### B. Image hash overlap

Tạo hash:

$$ H(I)=\operatorname{SHA256} (\text{dtype},\text{shape},I.\text{bytes}) $$

Hash giống nhau giữa train và test là dấu hiệu duplicate image.

### C. Exact normalized LaTeX overlap

$$ Y_{\text{train}} \cap Y_{\text{test}} $$

Cùng LaTeX có thể hợp lệ nếu:

- khác ảnh;
- khác người viết;
- cùng công thức toán học.

Tuy nhiên cần báo cáo:

- tỷ lệ test label đã thấy trong train;
- tỷ lệ label mới;
- accuracy trên seen-label và unseen-label.

### D. Writer overlap

Đây là style overlap. Muốn kiểm tra cần:

- writer ID;
- InkML metadata;
- mapping chính thức.

Không nên suy writer ID chỉ từ filename nếu chưa biết quy ước tên.

## 7.3. Model-selection contamination

Do:

```yaml
test_folder: 2014
monitor: val_ExpRate
```

nên 2014 ảnh hưởng:

- checkpoint;
- epoch;
- có thể cả quyết định kiến trúc.

Nếu sau khi xem 2016/2019 ta tiếp tục thay model, hai tập đó cũng có thể trở thành development set ở cấp độ nghiên cứu.

## 7.4. Trùng LaTeX không nhất thiết là leakage

Ví dụ cùng ground truth:

```latex
x + 1
```

nhưng hai ảnh do hai người viết khác nhau.

HMER cần nhận dạng nhiều phong cách cho cùng ngôn ngữ toán học. Vì vậy cần phân biệt:

- duplicate image;
- duplicate label.

## 7.5. Audit nên chạy

```text
Audit 1: duplicate filenames
Audit 2: duplicate image hashes
Audit 3: exact normalized LaTeX overlap
Audit 4: token n-gram overlap
Audit 5: writer overlap nếu có metadata
Audit 6: validation/test role
```

## 7.6. Nếu phát hiện duplicate image

- loại khỏi train hoặc test theo protocol;
- tạo lại split;
- retrain;
- báo cáo lại;
- lưu audit artifact.

## 7.7. Khi chưa có writer metadata

> **Em chưa có bằng chứng đủ để khẳng định writer-disjoint. Từ gói hiện tại em có thể audit image hash và label overlap; writer overlap cần metadata từ nguồn CROHME.**

## 7.8. Không nên nói

- “Khác folder là chắc chắn không trùng.”
- “Cùng LaTeX nghĩa là leakage.”
- “Không backpropagate trên test nên không có contamination.”
- “Tên file chắc chắn là writer ID.”
- “Benchmark công khai thì không cần audit.”

---

# Câu 8 — Từ điển token có bao nhiêu phần tử và được xây dựng từ tập nào?

## 8.1. Bản trả lời nhanh 45–60 giây

> **Cấu hình model dùng `vocab_size: 113`, gồm 110 token trong `dictionary.txt` cộng ba token đặc biệt `<pad>`, `<sos>` và `<eos>`. `CROHMEVocab.init()` đọc từng dòng của `data/crohme/dictionary.txt` rồi gán chỉ số.**
>
> **Tuy nhiên repo không kèm script tạo dictionary, nên từ code hiện tại em chưa chứng minh được nó được xây chỉ từ train hay từ toàn benchmark. Đây là điểm phải audit. Ngoài ra code không có `<unk>`; token ngoài từ điển gây lỗi tra cứu thay vì được ánh xạ sang unknown.**

## 8.2. Ba token đặc biệt

| Token | Index | Vai trò |
|---|---:|---|
| `<pad>` | 0 | Padding chuỗi |
| `<sos>` | 1 | Bắt đầu chuỗi |
| `<eos>` | 2 | Kết thúc chuỗi |

Tổng vocabulary:

$$ 110 + 3 = 113 $$

## 8.3. Vocabulary không hoàn toàn bằng số lớp ký hiệu vật lý

Dictionary decoder có thể gồm:

- chữ số;
- chữ cái;
- Greek;
- toán tử;
- macro `\frac`, `\sqrt`;
- `{`, `}`;
- `^`, `_`;
- token điều khiển khác.

Do đó cần phân biệt:

- symbol classes;
- LaTeX decoder tokens.

## 8.4. Nguồn dictionary chưa được chứng minh

Code chỉ cho biết:

```python
vocab.init(root / "dictionary.txt")
```

Nó không chứng minh dictionary được tạo bằng:

```text
unique_tokens(train_only)
```

hay:

```text
unique_tokens(train + tests)
```

Cần audit:

```python
train_tokens = set(...)
test_tokens = set(...)
dictionary_tokens = set(...)
```

## 8.5. Rủi ro vocabulary leakage

Nếu dictionary được xây từ test:

- model biết trước token universe;
- nhưng không biết chuỗi hoặc tần suất cụ thể;
- mức leakage nhẹ hơn dùng nhãn test để train;
- vẫn phải khai báo protocol.

Nếu benchmark có vocabulary cố định chính thức, việc dùng vocabulary chung có thể hợp lệ, nhưng phải dẫn đúng protocol.

## 8.6. Không có `<unk>`

`words2indices()` tra trực tiếp:

```python
self.word2idx[word]
```

Nếu token không tồn tại sẽ gây `KeyError`.

Hệ quả:

- caption train/test phải nằm hoàn toàn trong dictionary;
- model không thể sinh token mới ngoài 113 class.

## 8.7. Câu hỏi phụ: “Vì sao 113 không bằng số symbol class?”

> **Vì vocabulary của decoder gồm cả token cấu trúc và ba token đặc biệt; nó không nhất thiết bằng số loại ký hiệu được nhìn thấy trên ảnh.**

## 8.8. Không nên nói

- “Có 113 symbol classes” nếu chưa phân loại.
- “Dictionary chắc chắn được xây từ train.”
- “Có `<unk>`.”
- “Model sinh được ký hiệu mới ngoài vocabulary.”
- “Token hiếm bị loại tự động.”

---

# Câu 9 — Phân bố độ dài chuỗi LaTeX trong train, validation và từng test set ra sao?

## 9.1. Bản trả lời nhanh trung thực

> **Repo hiện chưa có bảng phân bố độ dài được lưu, nên em chưa được phép đọc một số trung bình từ trí nhớ. Mỗi nhãn đã được token hóa bằng khoảng trắng trong `caption.txt`; độ dài là số token sau image ID. Code loại mẫu train dài hơn 200 token, còn decoder giới hạn `max_len: 150`.**
>
> **Em cần báo cáo count, mean, median, standard deviation, P90, P95, max và tỷ lệ mẫu dài hơn 50, 100, 150 token cho train, 2014, 2016 và 2019. Đặc biệt phải đếm test label dài hơn 150 vì model có nguy cơ không thể sinh hết chuỗi.**

## 9.2. Điều biết chắc từ code

Training filter:

```python
if is_train and len(label_tokens) > 200:
    ignore
```

Inference:

```yaml
max_len: 150
```

Có mismatch:

$$ L_{\text{train max accepted}}=200 $$

nhưng:

$$ L_{\text{decode max}}=150 $$

Nếu ground truth dài hơn 150 token, decoder có thể bị cắt trước khi hoàn tất.

## 9.3. Bảng bắt buộc

| Tập | N | Mean | Median | Std | P90 | P95 | Max | >50 | >100 | >150 | >200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train raw |  |  |  |  |  |  |  |  |  |  |  |
| Train effective |  |  |  |  |  |  |  |  |  |  |  |
| 2014 | 986 |  |  |  |  |  |  |  |  |  |  |
| 2016 | 1.147 |  |  |  |  |  |  |  |  |  |  |
| 2019 | 1.199 |  |  |  |  |  |  |  |  |  |  |

## 9.4. Vì sao mean chưa đủ?

Hai tập có thể cùng mean nhưng tail khác nhau.

Mẫu dài thường:

- nhiều cấu trúc lồng;
- nhiều ngoặc;
- tăng exposure bias;
- khó Exact Match;
- tốn bộ nhớ hơn.

Cần histogram và percentile.

## 9.5. Accuracy theo độ dài

Chia bin:

```text
1–10
11–20
21–30
31–50
51–75
76–100
>100
```

Tính:

$$ \operatorname{ExpRate}_b = \frac{\#\text{mẫu đúng trong bin }b} {\#\text{mẫu trong bin }b} $$

## 9.6. Câu trả lời trước khi có số liệu

> **Em có thể nói rõ giới hạn code nhưng chưa khẳng định phân bố cụ thể. Bảng độ dài là một artifact bắt buộc trước khi chốt luận văn.**

## 9.7. Không nên nói

- “Chuỗi trung bình khoảng 20 token” nếu chưa đo.
- “Không có mẫu dài hơn 150.”
- “Max train là 150.”
- “Ba test set có độ dài giống nhau.”
- “Biểu thức dài chắc chắn là nguyên nhân chính.”

---

# Câu 10 — Những loại cấu trúc nào xuất hiện nhiều và những loại nào hiếm trong tập huấn luyện?

## 10.1. Bản trả lời nhanh trung thực

> **Repo chưa có bảng tần suất cấu trúc train, nên em chưa thể nói loại nào nhiều hoặc hiếm bằng cảm giác. Có thể thống kê proxy từ token như `\frac`, `\sqrt`, `^`, `_`, `\int`, `\sum`, `\lim`, nhưng token count chưa phản ánh đầy đủ cấu trúc lồng hoặc quan hệ ngữ cảnh.**
>
> **Em cần báo cáo số expression chứa từng cấu trúc, tổng số lần xuất hiện và accuracy riêng trên từng nhóm. Chỉ sau đó mới được gọi một cấu trúc là frequent hoặc long-tail.**

## 10.2. Danh mục nên thống kê

### Cấu trúc cơ bản

- số và biến;
- cộng, trừ, nhân, chia;
- dấu bằng, bất đẳng thức;
- ngoặc.

### Cấu trúc 2D

- superscript `^`;
- subscript `_`;
- fraction `\frac`;
- root `\sqrt`;
- integral `\int`;
- summation `\sum`;
- limit `\lim`;
- product `\prod`;
- matrix/environment nếu có.

### Cấu trúc kết hợp

- fraction nested;
- root nested;
- fraction trong root;
- exponent trong fraction;
- integral có cả hai cận;
- sum có cả hai cận;
- nhiều chỉ số liên tiếp.

## 10.3. Ba loại count

### Số expression chứa cấu trúc

$$ N_c = \sum_i \mathbf{1}[c \in Y_i] $$

### Tổng số occurrence

$$ O_c = \sum_i \operatorname{count}(c,Y_i) $$

### Độ phức tạp có điều kiện

Ví dụ:

- fraction lồng;
- root lồng;
- số cấu trúc con tối đa.

Không được dùng occurrence count thay cho expression count.

## 10.4. Hạn chế của token proxy

`_` có thể là:

- subscript;
- cận dưới tích phân;
- cận dưới tổng.

`^` có thể là:

- số mũ;
- cận trên.

Vì vậy cần parser theo ngữ cảnh.

## 10.5. Định nghĩa long-tail

Có thể đặt ngưỡng trước:

- frequent: trên 10% expression;
- medium: 1–10%;
- rare: 0,1–1%;
- very rare: dưới 0,1%.

Ngưỡng phải được ghi trước khi nhìn kết quả.

## 10.6. Bảng cần tạo

| Cấu trúc | Train expressions | Tỷ lệ train | 2014 | 2016 | 2019 | ExpRate M1 | ExpRate M4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `\frac` |  |  |  |  |  |  |  |
| `\sqrt` |  |  |  |  |  |  |  |
| Superscript |  |  |  |  |  |  |  |
| Subscript |  |  |  |  |  |  |  |
| `\int` |  |  |  |  |  |  |  |
| `\sum` |  |  |  |  |  |  |  |
| `\lim` |  |  |  |  |  |  |  |

## 10.7. Không nên nói

- “Dataset có nhiều phân số.”
- “Tích phân có cận rất ít.”
- “Căn là khó nhất.”
- “M4 tốt hơn ở cấu trúc phức tạp.”

Mọi câu cần số liệu.

---

# Câu 11 — Dataset có bao nhiêu mẫu chứa tích phân, tích phân có cận dưới, cận trên và có cả hai cận?

## 11.1. Bản trả lời nhanh trung thực

> **Repo chưa có thống kê này, nên câu trả lời đúng hiện tại là em chưa được phép kết luận “dataset không có”. Em phải đếm trực tiếp trong từng `caption.txt`: số expression chứa `\int`, không cận, chỉ cận dưới, chỉ cận trên, có cả hai cận và có nhiều tích phân.**
>
> **Việc đếm phải dựa vào `_` và `^` gắn ngay sau token `\int`, không được tìm `^` ở bất kỳ vị trí nào trong chuỗi vì đó có thể là số mũ của biến. Sau đó em cần tính accuracy riêng cho subset tích phân trên 2014, 2016 và 2019.**

## 11.2. Các nhóm cần tách

| Nhóm | Điều kiện |
|---|---|
| Integral total | Có token `\int` |
| No bounds | Không có `_` hoặc `^` gắn ngay sau `\int` |
| Lower only | Có `_` nhưng không có `^` gắn với `\int` |
| Upper only | Có `^` nhưng không có `_` gắn với `\int` |
| Both bounds | Có cả `_` và `^` gắn với `\int` |
| Multiple integrals | Có từ hai `\int` trở lên |

## 11.3. Không được đếm ngây thơ

Ví dụ:

```latex
\int x ^ { 2 } d x
```

Có `^` trong chuỗi, nhưng đó là số mũ của $x$, không phải cận trên của tích phân.

Sai:

```python
contains_int and contains_caret
```

Đúng:

- bắt đầu tại token `\int`;
- chỉ đọc các modifier `_` và `^` ngay sau đó;
- skip group `{...}`;
- dừng khi gặp phần thân tích phân.

## 11.4. Parser cần hỗ trợ hai thứ tự

```latex
\int _ { 0 } ^ { 1 }
```

và:

```latex
\int ^ { 1 } _ { 0 }
```

## 11.5. Bảng bắt buộc

| Tập | Tổng mẫu | Có `\int` | Không cận | Chỉ dưới | Chỉ trên | Cả hai | Nhiều tích phân |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train raw | 8.836 |  |  |  |  |  |  |
| Train effective |  |  |  |  |  |  |  |
| 2014 | 986 |  |  |  |  |  |  |
| 2016 | 1.147 |  |  |  |  |  |  |
| 2019 | 1.199 |  |  |  |  |  |  |

## 11.6. Accuracy riêng cho subset

$$ \operatorname{ExpRate}_{\text{int,both}} = \frac{\#\text{tích phân hai cận dự đoán đúng}} {\#\text{tích phân hai cận}} $$

Nên bổ sung:

- recall `\int`;
- recall `_`;
- recall `^`;
- Mean Edit Distance subset;
- lỗi ngoặc;
- lỗi truncate.

## 11.7. Câu trả lời demo đúng

> **Em chưa quy lỗi demo cho việc dataset không có. Em cần kiểm tra tần suất cấu trúc, độ rõ của cận sau resize, domain gap và token nào bị mất trong output.**

## 11.8. Không nên nói

- “Dataset không có tích phân có cận.”
- “Có `^` trong chuỗi nghĩa là tích phân có cận trên.”
- “Nhận đúng `\int` là nhận đúng toàn bộ tích phân.”
- “Cả hai cận rất hiếm” khi chưa đếm.

---

# Câu 12 — Các token hiếm hoặc ngoài từ điển được xử lý như thế nào?

## 12.1. Bản trả lời nhanh 45–60 giây

> **Code hiện không có `<unk>`. `words2indices()` tra trực tiếp từng token trong dictionary; token ngoài từ điển gây `KeyError`. Vì vậy train, validation và test phải được chuẩn hóa sao cho toàn bộ token nằm trong dictionary.**
>
> **Token hiếm nhưng có trong dictionary vẫn được học bằng cross-entropy bình thường. Repo chưa có class weighting, focal loss hoặc oversampling theo token, nên token hiếm có thể nhận ít tín hiệu huấn luyện hơn và cần được thống kê riêng.**

## 12.2. OOV trong train/test

Code tương đương:

```python
indices = [word2idx[token] for token in tokens]
```

Không có:

```python
word2idx.get(token, unk_idx)
```

Do đó:

- OOV gây lỗi;
- dictionary coverage phải đạt 100%.

## 12.3. OOV ở inference

Input là ảnh, nên không có OOV input token.

Nhưng decoder chỉ có:

$$ P(y_t \mid \cdot),\quad y_t \in V $$

Nếu ảnh chứa ký hiệu ngoài $V$:

- model buộc chọn token gần nhất;
- bỏ token;
- hoặc sinh chuỗi sai;
- không thể tạo class mới.

## 12.4. Token hiếm

Cross-entropy tiêu chuẩn:

$$ \mathcal{L} = -\sum_t \log P(y_t) $$

không tự cân bằng theo tần suất token.

Token hiếm có thể:

- recall thấp;
- dễ nhầm với token phổ biến;
- làm ExpRate giảm mạnh ở nhóm long-tail.

## 12.5. Audit token

Với token $v$:

$$ f_{\text{train}}(v) = \sum_i \operatorname{count}(v,Y_i) $$

Cần báo cáo:

- 20 token phổ biến nhất;
- 20 token hiếm nhất;
- token có dưới 5, 10, 20 occurrence;
- token có trong test nhưng không train;
- token có test/train ratio bất thường.

## 12.6. Hướng xử lý có thể thử

- synthetic augmentation;
- oversampling expression chứa token hiếm;
- weighted loss;
- curriculum;
- external data hợp lệ;
- constrained decoding.

Không được nói đã dùng nếu repo chưa có.

## 12.7. Không nên nói

- “Token hiếm đã bị loại.”
- “Có `<unk>`.”
- “Vocabulary bao phủ mọi toán học.”
- “Cross-entropy tự cân bằng token.”
- “Model có thể tự thêm token mới.”

---

# Câu 13 — Có thể kết luận một test set khó hơn test set khác chỉ từ ExpRate thấp hơn hay không? Vì sao?

## 13.1. Bản trả lời nhanh 45–60 giây

> **Không. ExpRate thấp hơn chỉ cho biết một model cụ thể đạt kết quả thấp hơn trên tập đó. Nó chưa chứng minh bản thân dataset khó hơn, vì score còn phụ thuộc model, checkpoint, decoding, mức phù hợp với train, độ dài, cấu trúc, token hiếm, writer style và sai số thống kê.**
>
> **Muốn gọi một tập khó hơn, cần nhiều model cùng suy giảm nhất quán và có phân tích dữ liệu: độ dài, cấu trúc, rare token, kích thước ảnh, matched subsets và confidence interval.**

## 13.2. ExpRate là hàm của cả model và dataset

$$ \operatorname{ExpRate} = f(\text{model},\text{checkpoint},\text{decoding},\text{dataset}) $$

Nó không phải thuộc tính chỉ của dataset.

## 13.3. Các nguyên nhân làm score khác nhau

- sequence length;
- nesting depth;
- token rarity;
- writer style;
- image scale;
- aspect ratio;
- label normalization;
- vocabulary coverage;
- beam-search behavior;
- checkpoint selection;
- random seed.

## 13.4. Điều kiện mạnh hơn để nói “khó hơn”

### Nhiều model cùng giảm

Nếu các kiến trúc độc lập đều giảm trên tập B.

### Confidence interval

Dùng bootstrap:

1. lấy mẫu test có hoàn lại;
2. tính ExpRate;
3. lặp nhiều lần;
4. lấy khoảng tin cậy 95%.

### Matched subsets

So sánh các tập sau khi khớp:

- độ dài;
- cấu trúc;
- token;
- kích thước.

### Nhiều metric cùng xấu đi

- ExpRate;
- $\leq 1$;
- $\leq 2$;
- Mean Edit Distance;
- syntax validity.

## 13.5. Cách nói đúng

Sai:

> “CROHME 2019 khó hơn vì M1 thấp hơn.”

Đúng:

> **M1 có ExpRate thấp hơn trên 2019. Đây là quan sát model–dataset; để kết luận độ khó cần phân tích phân bố và xu hướng trên nhiều model.**

## 13.6. Không nên nói

- “ExpRate thấp đồng nghĩa dataset khó.”
- “Năm mới hơn chắc chắn khó hơn.”
- “Nhiều mẫu hơn là khó hơn.”
- “Một model đủ để xếp hạng độ khó.”

---

# Câu 14 — Nếu demo ngoài đời thất bại nhưng test set tốt, đó là lỗi mô hình hay lỗi phân bố dữ liệu? Cần kiểm chứng thế nào?

## 14.1. Bản trả lời nhanh 60–90 giây

> **Có thể là cả hai. Test set đo hiệu quả in-domain trên ảnh đã được render và chuẩn hóa gần CROHME, trong khi demo có thể khác về nền, nét, crop, độ dày, tỷ lệ, thiết bị và phong cách viết. Đó là domain shift. Nhưng nếu model cũng sai trên các mẫu CROHME có cùng cấu trúc với demo, thì model hoặc data coverage còn yếu.**
>
> **Để tách nguyên nhân, em cần lưu ảnh ở từng bước preprocessing, đưa một ảnh CROHME qua đúng pipeline demo, tạo subset in-domain cùng cấu trúc, so sánh token-by-token và thử controlled perturbations. Không được đổ toàn bộ cho người dùng viết xấu hoặc cho dataset.**

## 14.2. Hai miền dữ liệu

Benchmark:

$$ (X,Y)\sim P_{\text{CROHME}} $$

Demo:

$$ (X,Y)\sim P_{\text{demo}} $$

Nếu:

$$ P_{\text{demo}}(X) \neq P_{\text{CROHME}}(X) $$

thì có covariate shift.

Nếu cấu trúc cũng khác:

$$ P_{\text{demo}}(Y) \neq P_{\text{CROHME}}(Y) $$

thì có label hoặc structure shift.

## 14.3. Nguồn domain shift

### Ảnh

- nền trắng hoặc đen;
- antialiasing;
- RGB sang grayscale;
- compression;
- blur;
- noise;
- shadow;
- đường kẻ;
- crop;
- margin.

### Nét

- chuột so với bút;
- nét dày hoặc mảnh;
- nét đứt;
- ký hiệu dính;
- độ nghiêng.

### Bố cục

- cận quá nhỏ;
- khoảng cách khác train;
- biểu thức dài;
- nhiều dòng;
- bố trí không chuẩn.

### Nhãn

- macro ngoài dictionary;
- cách LaTeX khác protocol;
- cấu trúc hiếm.

## 14.4. Quy trình chẩn đoán bảy bước

### Bước 1 — Lưu ảnh trung gian

```text
raw_input.png
cropped.png
deskewed.png
binarized.png
resized.png
normalized_model_input.png
```

### Bước 2 — Kiểm tra polarity

So sánh:

- mean pixel;
- foreground ratio;
- histogram;
- nền và nét.

### Bước 3 — Kiểm tra pipeline consistency

Lấy một ảnh từ `images.pkl`:

1. xuất PNG;
2. chạy trực tiếp vào model;
3. chạy qua app preprocessing;
4. so sánh output.

Nếu direct path đúng nhưng app path sai, lỗi nằm ở integration/preprocessing.

### Bước 4 — Tạo structural subset

Nếu demo là tích phân hai cận:

- lấy tất cả test expression có hai cận;
- tính accuracy riêng;
- xem năng lực in-domain.

### Bước 5 — Controlled perturbation

Từ ảnh CROHME đúng:

- tăng độ dày;
- giảm độ dày;
- nghiêng;
- thêm margin;
- đảo polarity;
- blur;
- thu nhỏ cận.

Đo điểm bắt đầu suy giảm.

### Bước 6 — Token-level diagnosis

Ground truth:

```latex
\int _ { 0 } ^ { 1 } x d x
```

Phân loại:

- mất `\int`;
- mất `_`;
- mất `^`;
- sai nội dung cận;
- sai body;
- sai ngoặc;
- truncate.

### Bước 7 — Confidence và beam analysis

- beam scores;
- top-k alternatives;
- token entropy;
- attention map;
- nearest train examples.

## 14.5. Ma trận kết luận

| In-domain subset | Demo | Kết luận chính |
|---|---|---|
| Tốt | Kém | Domain shift hoặc preprocessing là nghi phạm lớn |
| Kém | Kém | Model hoặc data coverage còn yếu |
| Direct path tốt, app path kém | Kém | Integration/preprocessing lỗi |
| Kém riêng một cấu trúc | Kém đúng cấu trúc đó | Long-tail hoặc structural limitation |
| Tốt hơn sau domain augmentation | Tốt hơn | Domain adaptation có tác dụng |

## 14.6. Cách nói khoa học

> **Demo failure là bằng chứng hệ thống chưa tổng quát tốt ngoài benchmark, nhưng chưa đủ để xác định nguyên nhân. Em tách data coverage, preprocessing consistency và model capacity bằng controlled evaluation.**

## 14.7. Không nên nói

- “Người dùng viết xấu.”
- “Dataset không có.”
- “Test tốt nên model không có lỗi.”
- “Demo sai chắc chắn do frontend.”
- “Chỉ cần augmentation là giải quyết.”
- “Một ảnh demo đủ để kết luận domain shift.”

---

# 15. Script audit dataset bắt buộc phải chạy

Lưu script sau thành:

```text
tools/audit_crohme_dataset.py
```

Chạy:

```bash
python tools/audit_crohme_dataset.py \
  --root data/crohme \
  --output dataset_audit
```

```python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "2014", "2016", "2019")
MAX_TRAIN_TOKENS = 200
MAX_TRAIN_AREA = 320_000


def read_dictionary(root: Path) -> list[str]:
    path = root / "dictionary.txt"
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def read_captions(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / split / "caption.txt"
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            rows.append(
                {
                    "line_no": line_no,
                    "image_id": parts[0],
                    "tokens": parts[1:],
                    "latex": " ".join(parts[1:]),
                }
            )

    return rows


def read_images(root: Path, split: str) -> dict[str, np.ndarray]:
    path = root / split / "images.pkl"
    with path.open("rb") as file:
        images = pickle.load(file)

    if not isinstance(images, dict):
        raise TypeError(
            f"{path} phải chứa dict image_id -> numpy array, "
            f"nhưng nhận {type(images)!r}"
        )

    return images


def summarize_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "max": None,
            "gt_50": 0,
            "gt_100": 0,
            "gt_150": 0,
            "gt_200": 0,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(len(values)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
        "gt_50": int((array > 50).sum()),
        "gt_100": int((array > 100).sum()),
        "gt_150": int((array > 150).sum()),
        "gt_200": int((array > 200).sum()),
    }


def image_hash(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def skip_group(tokens: list[str], index: int) -> int:
    """Bỏ qua một group LaTeX bắt đầu tại index."""

    if index >= len(tokens):
        return index

    if tokens[index] != "{":
        return index + 1

    depth = 0
    cursor = index

    while cursor < len(tokens):
        if tokens[cursor] == "{":
            depth += 1
        elif tokens[cursor] == "}":
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1

    return cursor


def classify_integral_at(
    tokens: list[str],
    integral_index: int,
) -> tuple[bool, bool]:
    """Trả về has_lower và has_upper cho một token \\int."""

    cursor = integral_index + 1
    has_lower = False
    has_upper = False

    while cursor < len(tokens):
        token = tokens[cursor]

        if token == "_":
            has_lower = True
            cursor = skip_group(tokens, cursor + 1)
            continue

        if token == "^":
            has_upper = True
            cursor = skip_group(tokens, cursor + 1)
            continue

        break

    return has_lower, has_upper


def classify_integral_expression(tokens: list[str]) -> dict[str, Any]:
    positions = [
        index
        for index, token in enumerate(tokens)
        if token == r"\int"
    ]

    result: dict[str, Any] = {
        "contains_integral": bool(positions),
        "integral_occurrences": len(positions),
        "no_bounds": False,
        "lower_only": False,
        "upper_only": False,
        "both_bounds": False,
        "multiple_integrals": len(positions) >= 2,
    }

    for position in positions:
        lower, upper = classify_integral_at(tokens, position)

        if lower and upper:
            result["both_bounds"] = True
        elif lower:
            result["lower_only"] = True
        elif upper:
            result["upper_only"] = True
        else:
            result["no_bounds"] = True

    return result


def expression_structures(tokens: list[str]) -> set[str]:
    token_set = set(tokens)
    structures: set[str] = set()

    mapping = {
        r"\frac": "fraction",
        r"\sqrt": "sqrt",
        r"\int": "integral",
        r"\sum": "sum",
        r"\prod": "product",
        r"\lim": "limit",
        "^": "superscript_or_upper_bound",
        "_": "subscript_or_lower_bound",
    }

    for token, name in mapping.items():
        if token in token_set:
            structures.add(name)

    if tokens.count(r"\frac") >= 2:
        structures.add("multiple_or_nested_fraction")

    if tokens.count(r"\sqrt") >= 2:
        structures.add("multiple_or_nested_sqrt")

    if r"\frac" in token_set and r"\sqrt" in token_set:
        structures.add("fraction_and_sqrt")

    integral = classify_integral_expression(tokens)

    if integral["no_bounds"]:
        structures.add("integral_no_bounds")
    if integral["lower_only"]:
        structures.add("integral_lower_only")
    if integral["upper_only"]:
        structures.add("integral_upper_only")
    if integral["both_bounds"]:
        structures.add("integral_both_bounds")

    return structures


def audit_split(
    root: Path,
    split: str,
    dictionary: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_captions(root, split)
    images = read_images(root, split)

    missing_images: list[str] = []
    extra_images = set(images)

    lengths: list[int] = []
    heights: list[int] = []
    widths: list[int] = []
    areas: list[int] = []

    token_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    integral_counts: Counter[str] = Counter()
    oov_counts: Counter[str] = Counter()

    image_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    latex_to_ids: dict[str, list[str]] = defaultdict(list)

    effective_rows: list[dict[str, Any]] = []

    filtered_too_long = 0
    filtered_too_large = 0

    for row in rows:
        image_id = row["image_id"]
        tokens = row["tokens"]

        if image_id not in images:
            missing_images.append(image_id)
            continue

        extra_images.discard(image_id)

        image = np.asarray(images[image_id])

        if image.ndim < 2:
            raise ValueError(
                f"Ảnh {split}/{image_id} có shape không hợp lệ: "
                f"{image.shape}"
            )

        height = int(image.shape[0])
        width = int(image.shape[1])
        area = height * width

        lengths.append(len(tokens))
        heights.append(height)
        widths.append(width)
        areas.append(area)

        token_counts.update(tokens)

        for token in tokens:
            if token not in dictionary:
                oov_counts[token] += 1

        structures = expression_structures(tokens)
        structure_counts.update(structures)

        integral = classify_integral_expression(tokens)
        for key, value in integral.items():
            if key == "integral_occurrences":
                integral_counts[key] += int(value)
            elif value:
                integral_counts[key] += 1

        digest = image_hash(image)
        image_hash_to_ids[digest].append(image_id)
        latex_to_ids[row["latex"]].append(image_id)

        keep = True
        reasons: list[str] = []

        if split == "train":
            if len(tokens) > MAX_TRAIN_TOKENS:
                keep = False
                filtered_too_long += 1
                reasons.append("too_long")

            if area > MAX_TRAIN_AREA:
                keep = False
                filtered_too_large += 1
                reasons.append("too_large")

        enriched = dict(row)
        enriched.update(
            {
                "height": height,
                "width": width,
                "area": area,
                "image_hash": digest,
                "keep_for_training": keep,
                "filter_reason": reasons,
            }
        )

        if keep:
            effective_rows.append(enriched)

    stats = {
        "split": split,
        "raw_caption_count": len(rows),
        "images_count": len(images),
        "missing_image_count": len(missing_images),
        "missing_images_first_20": missing_images[:20],
        "extra_image_count": len(extra_images),
        "extra_images_first_20": sorted(extra_images)[:20],
        "lengths": summarize_numbers(lengths),
        "image_height": summarize_numbers(heights),
        "image_width": summarize_numbers(widths),
        "image_area": summarize_numbers(areas),
        "effective_count_after_repo_train_filters": len(effective_rows),
        "filtered_too_long": filtered_too_long,
        "filtered_too_large": filtered_too_large,
        "token_occurrences_total": sum(token_counts.values()),
        "unique_tokens": len(token_counts),
        "oov_token_types": len(oov_counts),
        "oov_token_occurrences": sum(oov_counts.values()),
        "oov_counts": dict(oov_counts.most_common()),
        "top_30_tokens": token_counts.most_common(30),
        "rare_tokens_le_5": sorted(
            (token, count)
            for token, count in token_counts.items()
            if count <= 5
        ),
        "structure_expression_counts": dict(
            structure_counts.most_common()
        ),
        "integral_counts": dict(integral_counts),
        "duplicate_image_hash_groups_within_split": sum(
            len(ids) > 1
            for ids in image_hash_to_ids.values()
        ),
        "duplicate_latex_groups_within_split": sum(
            len(ids) > 1
            for ids in latex_to_ids.values()
        ),
    }

    return stats, effective_rows


def pairwise_overlap(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for left_index, left_name in enumerate(SPLITS):
        for right_name in SPLITS[left_index + 1:]:
            left_rows = split_rows[left_name]
            right_rows = split_rows[right_name]

            left_ids = {row["image_id"] for row in left_rows}
            right_ids = {row["image_id"] for row in right_rows}

            left_hashes = {row["image_hash"] for row in left_rows}
            right_hashes = {row["image_hash"] for row in right_rows}

            left_latex = {row["latex"] for row in left_rows}
            right_latex = {row["latex"] for row in right_rows}

            pair_name = f"{left_name}__{right_name}"

            seen_count = sum(
                row["latex"] in left_latex
                for row in right_rows
            )

            result[pair_name] = {
                "filename_overlap": len(left_ids & right_ids),
                "image_hash_overlap": len(
                    left_hashes & right_hashes
                ),
                "exact_latex_overlap": len(
                    left_latex & right_latex
                ),
                "right_expressions_with_latex_seen_in_left":
                    seen_count,
                "right_seen_latex_rate": (
                    seen_count / len(right_rows)
                    if right_rows
                    else math.nan
                ),
            }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root: Path = args.root
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    dictionary_list = read_dictionary(root)
    dictionary = set(dictionary_list)

    report: dict[str, Any] = {
        "dictionary_file_tokens": len(dictionary_list),
        "dictionary_unique_tokens": len(dictionary),
        "model_vocab_size_with_pad_sos_eos":
            len(dictionary) + 3,
        "special_tokens": ["<pad>", "<sos>", "<eos>"],
        "splits": {},
    }

    split_rows: dict[str, list[dict[str, Any]]] = {}

    for split in SPLITS:
        stats, effective_rows = audit_split(
            root,
            split,
            dictionary,
        )
        report["splits"][split] = stats
        split_rows[split] = effective_rows

    report["pairwise_overlap"] = pairwise_overlap(split_rows)

    json_path = output / "dataset_audit.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    lines = [
        "# CROHME Dataset Audit",
        "",
        f"- Dictionary tokens: {len(dictionary)}",
        (
            "- Model vocabulary including special tokens: "
            f"{len(dictionary) + 3}"
        ),
        "",
        "## Split sizes",
        "",
        "| Split | Raw | Effective | Mean length | "
        "P95 | Max | >150 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for split in SPLITS:
        stats = report["splits"][split]
        lengths = stats["lengths"]

        lines.append(
            f"| {split} | "
            f"{stats['raw_caption_count']} | "
            f"{stats['effective_count_after_repo_train_filters']} | "
            f"{lengths['mean']:.2f} | "
            f"{lengths['p95']:.2f} | "
            f"{lengths['max']} | "
            f"{lengths['gt_150']} |"
        )

    lines.extend(
        [
            "",
            "## Integral distribution",
            "",
            "| Split | Integral expressions | No bounds | "
            "Lower only | Upper only | Both bounds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for split in SPLITS:
        counts = report["splits"][split]["integral_counts"]

        lines.append(
            f"| {split} | "
            f"{counts.get('contains_integral', 0)} | "
            f"{counts.get('no_bounds', 0)} | "
            f"{counts.get('lower_only', 0)} | "
            f"{counts.get('upper_only', 0)} | "
            f"{counts.get('both_bounds', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise overlap",
            "",
            "| Pair | Filename | Image hash | Exact LaTeX | "
            "Right labels seen in left |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for pair, values in report["pairwise_overlap"].items():
        lines.append(
            f"| {pair} | "
            f"{values['filename_overlap']} | "
            f"{values['image_hash_overlap']} | "
            f"{values['exact_latex_overlap']} | "
            f"{values['right_expressions_with_latex_seen_in_left']} |"
        )

    markdown_path = output / "dataset_audit.md"
    markdown_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Đã ghi: {json_path}")
    print(f"Đã ghi: {markdown_path}")


if __name__ == "__main__":
    main()
```

---

# 16. Script tính accuracy riêng cho tích phân hai cận

Repo có thể lưu prediction dạng:

```text
pred_2014.json
pred_2016.json
pred_2019.json
```

Mỗi sample thường gồm:

```json
{
  "sample_id": {
    "pred": "...",
    "gt": "...",
    "dist": 0
  }
}
```

Lưu script:

```text
tools/evaluate_integral_bounds.py
```

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def skip_group(tokens: list[str], index: int) -> int:
    if index >= len(tokens):
        return index

    if tokens[index] != "{":
        return index + 1

    depth = 0

    while index < len(tokens):
        if tokens[index] == "{":
            depth += 1
        elif tokens[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1

    return index


def has_integral_with_both_bounds(
    tokens: list[str],
) -> bool:
    for index, token in enumerate(tokens):
        if token != r"\int":
            continue

        cursor = index + 1
        lower = False
        upper = False

        while cursor < len(tokens):
            current = tokens[cursor]

            if current == "_":
                lower = True
                cursor = skip_group(tokens, cursor + 1)
                continue

            if current == "^":
                upper = True
                cursor = skip_group(tokens, cursor + 1)
                continue

            break

        if lower and upper:
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prediction_json",
        type=Path,
    )
    args = parser.parse_args()

    with args.prediction_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        data: dict[str, dict[str, Any]] = json.load(file)

    subset: list[dict[str, Any]] = []

    for sample_id, item in data.items():
        gt_tokens = item["gt"].split()

        if not has_integral_with_both_bounds(gt_tokens):
            continue

        subset.append(
            {
                "id": sample_id,
                "pred": item["pred"],
                "gt": item["gt"],
                "dist": int(item["dist"]),
                "exact": item["pred"] == item["gt"],
            }
        )

    total = len(subset)
    exact = sum(row["exact"] for row in subset)

    mean_distance = (
        sum(row["dist"] for row in subset) / total
        if total
        else float("nan")
    )

    print(f"Integral both-bounds samples: {total}")
    print(f"Exact: {exact}")

    if total:
        print(f"ExpRate: {exact / total:.4%}")
    else:
        print("ExpRate: N/A")

    print(f"Mean edit distance: {mean_distance:.4f}")

    worst = sorted(
        subset,
        key=lambda row: row["dist"],
        reverse=True,
    )[:20]

    print("\nTop errors:")

    for row in worst:
        print("-" * 80)
        print("ID  :", row["id"])
        print("GT  :", row["gt"])
        print("Pred:", row["pred"])
        print("Dist:", row["dist"])


if __name__ == "__main__":
    main()
```

---

# 17. Bảng trả lời học thuộc

| Câu | Ý chính phải nói |
|---|---|
| 1 | Train lấy `train`; val và test cùng lấy `test_folder`; mặc định là 2014 |
| 2 | 8.836 / 986 / 1.147 / 1.199; phải phân biệt raw count và effective train count |
| 3 | Nguồn gốc online InkML nhưng model dùng offline raster `[B,1,H,W]` |
| 4 | Đọc `images.pkl`, resize giữ tỷ lệ, ToTensor, dynamic batch, zero padding và mask |
| 5 | Ba kỳ khác nhau về quy mô và đợt thu thập; phân bố chi tiết phải đo |
| 6 | Nhiều test giúp kiểm tra tính ổn định và tránh cherry-picking |
| 7 | Audit filename, image hash, LaTeX và writer; 2014 có model-selection contamination |
| 8 | 110 dictionary token + 3 special = 113; chưa chứng minh dictionary train-only; không có UNK |
| 9 | Chưa có stats; train filter 200 nhưng decode max 150 |
| 10 | Không nói cấu trúc nào hiếm trước khi đếm |
| 11 | Không nói dataset không có tích phân có cận; phải parse ngay sau `\int` |
| 12 | OOV gây KeyError; rare token không được reweight |
| 13 | ExpRate thấp là thuộc tính model–dataset, chưa đủ xếp hạng độ khó |
| 14 | Demo failure có thể do domain shift, preprocessing và model; cần controlled diagnosis |

---

# 18. Các câu tuyệt đối không nên nói

1. “CROHME 2014 chỉ là test set.”
2. “Repo có validation riêng.”
3. “Train chắc chắn có đúng 8.836 mẫu sau lọc.”
4. “Dataset không có tích phân có cận.”
5. “2019 khó nhất vì ExpRate thấp.”
6. “CROHME là offline dataset nguyên gốc.”
7. “Repo sử dụng stroke order.”
8. “Dictionary chắc chắn tạo từ train.”
9. “Có token `<unk>`.”
10. “Token hiếm đã được xử lý cân bằng.”
11. “Khác thư mục thì chắc chắn không trùng.”
12. “Demo sai chỉ vì người dùng viết xấu.”
13. “Random horizontal flip được dùng.”
14. “Ảnh đều resize về một kích thước cố định.”
15. “Ba test set đều untouched như nhau.”

---

# 19. Checklist bắt buộc trước khi đưa Nhóm 3 vào luận văn

## Dataset identity

- [ ] Lưu checksum SHA-256 của `CROHME.zip`.
- [ ] Ghi nguồn tải và ngày tải.
- [ ] Ghi cấu trúc thư mục sau giải nén.
- [ ] Ghi commit code dùng để train.

## Split

- [ ] Chạy count từng `caption.txt`.
- [ ] Đếm train sau filter.
- [ ] Chốt validation protocol.
- [ ] Ghi rõ 2014 dùng model selection hay không.
- [ ] Không tune tiếp trên 2016/2019 sau khi xem kết quả.

## Vocabulary

- [ ] Xuất toàn bộ 110 dictionary token.
- [ ] Xác định nguồn xây dictionary.
- [ ] Audit OOV từng split.
- [ ] Phân loại symbol token và structural token.

## Distribution

- [ ] Bảng length statistics.
- [ ] Histogram token length.
- [ ] Histogram image size và aspect ratio.
- [ ] Token frequency.
- [ ] Structure frequency.
- [ ] Integral-bound counts.
- [ ] Seen-label và unseen-label rate.

## Leakage

- [ ] Filename overlap.
- [ ] Image hash overlap.
- [ ] Exact normalized LaTeX overlap.
- [ ] Writer overlap nếu có metadata.
- [ ] Ghi rõ model-selection contamination.

## Demo

- [ ] Lưu ảnh trước và sau preprocessing.
- [ ] Kiểm tra polarity.
- [ ] Test ảnh CROHME qua app path.
- [ ] Test subset tích phân có cận.
- [ ] Token-level error analysis.

---

# 20. Nguồn đối chiếu chính

## Trong repo

- `chuyende_tamer_temp/1-cnn-gnn/config/crohme.yaml`
- `chuyende_tamer_temp/1-cnn-gnn/tamer/datamodule/datamodule.py`
- `chuyende_tamer_temp/1-cnn-gnn/tamer/datamodule/dataset.py`
- `chuyende_tamer_temp/1-cnn-gnn/tamer/datamodule/transforms.py`
- `chuyende_tamer_temp/1-cnn-gnn/tamer/datamodule/vocab.py`
- `chuyende_tamer_temp/KetQua/*/config/crohme.yaml`
- `chuyende_tamer_temp/KetQua/*/evaluation_results/pred_*.json`

## Nguồn benchmark cần trích trong luận văn

- Báo cáo CROHME 2014.
- Báo cáo ICFHR CROHME 2016.
- Báo cáo CROHME 2019.
- Paper hoặc repo TAMER gốc để đối chiếu protocol.
- Tài liệu mô tả định dạng InkML và annotation CROHME.

---

# 21. Đoạn trả lời tổng hợp khoảng hai phút

> **Repo hiện dùng gói CROHME đã được chuyển từ dữ liệu stroke online sang ảnh raster. Model đọc `images.pkl`, còn nhãn token nằm trong `caption.txt`; vì đầu vào cuối là tensor một kênh `[B,1,H,W]`, đây là offline HMER. Quy mô chuẩn là 8.836 mẫu train, còn 2014, 2016 và 2019 có 986, 1.147 và 1.199 mẫu. Tuy nhiên train có bộ lọc nhãn dài hơn 200 token và ảnh lớn hơn 320.000 pixel, nên em phải báo cáo thêm số mẫu hiệu dụng.**
>
> **Một điểm quan trọng là code không có validation riêng: `val_dataset` và `test_dataset` cùng lấy từ `test_folder`. Config hiện đặt `test_folder: 2014`, trong khi checkpoint được chọn theo `val_ExpRate`; do đó 2014 đang là model-selection set và không còn là test hoàn toàn untouched. Em cần sửa protocol bằng validation nội bộ hoặc ít nhất ghi rõ hạn chế này.**
>
> **Vocabulary có 113 class đầu ra, gồm 110 token trong `dictionary.txt` và ba token `<pad>`, `<sos>`, `<eos>`. Code không có `<unk>`, nên mọi caption phải nằm trong dictionary. Repo chưa có báo cáo phân bố độ dài, cấu trúc, token hiếm và tích phân có cận; vì vậy em không được nói dataset không có cấu trúc đó. Em sẽ chạy audit trên caption và image để đếm chính xác, kiểm tra OOV, duplicate image, exact-LaTeX overlap và accuracy theo subset.**
>
> **Cuối cùng, ExpRate thấp hơn trên một năm không đủ chứng minh test set đó khó hơn. Cần phân tích độ dài, cấu trúc, rare token, writer style và confidence interval. Nếu demo ngoài đời kém nhưng benchmark tốt, em sẽ tách domain shift, preprocessing và giới hạn mô hình bằng controlled tests, thay vì đổ lỗi cho người viết hoặc dataset.**

---

# 22. Câu kết thúc Nhóm 3

> **Đối với dataset, câu trả lời có điểm không phải là câu nghe hợp lý nhất, mà là câu có thể truy ngược đến đúng file dữ liệu, đúng script đếm và đúng protocol chia tập. Bất kỳ nhận định nào về “nhiều”, “ít”, “khó” hoặc “không có” đều phải đi kèm số liệu.**
