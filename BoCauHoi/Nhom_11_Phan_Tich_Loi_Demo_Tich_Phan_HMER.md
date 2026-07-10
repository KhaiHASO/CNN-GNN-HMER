# NHÓM 11 — PHÂN TÍCH LỖI, DEMO, NÉT VIẾT QUẰN VÀ TÍCH PHÂN CÓ CẬN

> **Mục tiêu nhóm:** Chuẩn bị cho các câu hỏi thực chiến về demo thất bại, chữ viết xấu, lỗi encoder/decoder và dữ liệu ngoài phân bố.

---

## 0. Nguyên tắc forensic khi demo sai

Không trả lời bằng một nguyên nhân duy nhất. Luôn đi theo chuỗi:

```text
Ground truth/data coverage
→ raw image
→ preprocessing
→ tensor model input
→ DenseNet feature
→ GAT feature
→ decoder beam
→ output token
```

Một lỗi cuối cùng có thể đến từ nhiều tầng. Câu trả lời có điểm là câu chỉ ra **cách phân biệt** các nguyên nhân.

---

# Câu 1 — Vì sao ảnh trong test set nhận dạng tốt nhưng ảnh người dùng tự vẽ trên demo có thể nhận dạng kém?

## 1.1. Bản trả lời nhanh

> **Test set và demo có thể khác phân bố. CROHME đã được render, crop và chuẩn hóa theo một protocol; ảnh demo có thể khác về nền, polarity, antialiasing, độ dày nét, margin, tỷ lệ, thiết bị và phong cách viết. Ngoài ra pipeline app có thể không giống dataloader.**
>
> **Do đó demo kém có thể là domain shift, preprocessing mismatch hoặc giới hạn model. Muốn tách phải đưa một ảnh CROHME qua đúng API demo và so tensor/output với đường dataloader.**

## 1.2. Các dạng domain shift

### Appearance shift

- nét chuột versus bút cảm ứng;
- anti-aliased versus binary;
- nền trắng/nét đen versus nền đen/nét trắng;
- độ dày;
- blur;
- compression;
- shadow;
- line noise.

### Geometry shift

- crop sát;
- margin lớn;
- nghiêng;
- scale;
- cận quá nhỏ;
- nhiều dòng;
- khoảng cách không giống train.

### Content shift

- token hiếm;
- macro ngoài vocabulary;
- cấu trúc hiếm;
- biểu thức dài.

### Integration mismatch

- RGB thay vì 1 channel;
- range 0–255 thay vì 0–1;
- resize khác;
- pad khác;
- invert sai;
- mask sai.

## 1.3. Phép kiểm chứng parity

1. Lấy ảnh từ `images.pkl`.
2. Forward trực tiếp qua dataloader.
3. Export PNG.
4. Upload PNG qua app.
5. Lưu tensor ngay trước model.
6. So shape, min/max/mean và pixel difference.
7. So output beam.

Nếu cùng ảnh nhưng app path kém:

> lỗi ở preprocessing/integration.

Nếu app path vẫn đúng với CROHME nhưng ảnh tự vẽ sai:

> domain gap hoặc content gap.

## 1.4. Không nên nói

- “Người dùng viết xấu.”
- “Dataset tốt nên model không lỗi.”
- “Demo chỉ khác giao diện.”
- “Một ảnh sai là đủ kết luận domain shift.”

---

# Câu 2 — Tích phân thường nhận đúng nhưng tích phân có cận trên/dưới nhận sai có thể do những nguyên nhân nào?

## 2.1. Bản trả lời nhanh

> **Có ít nhất năm nhóm nguyên nhân: cấu trúc hai cận hiếm trong train; cận nhỏ bị resize/downsample làm yếu; bố cục demo khác CROHME; decoder thiên về chuỗi phổ biến `\int ... dx` và bỏ `_`/`^`; hoặc max length/beam ranking làm mất token.**
>
> **Phải xác định model mất dấu tích phân, token cận, nội dung cận hay ngoặc trước khi kết luận.**

## 2.2. Phân rã bài toán

Ground truth:

```latex
\int _ { 0 } ^ { 1 } f ( x ) d x
```

Các thành phần:

1. token `\int`;
2. relation lower `_`;
3. group lower `{ 0 }`;
4. relation upper `^`;
5. group upper `{ 1 }`;
6. body;
7. differential.

Mỗi phần có failure mode riêng.

### Data imbalance

Cần đếm:

- integral no bounds;
- lower only;
- upper only;
- both bounds;
- multiple integrals.

### Visual resolution

Cận chiếm ít pixel, dễ mất qua:

- threshold;
- resize;
- DenseNet downsampling;
- padding/margin.

### Structural ambiguity

`^` có thể là exponent của body, không phải upper bound. Decoder phải gắn đúng token vào `\int`.

### Language-model bias

Nếu `\int x dx` phổ biến hơn, decoder có thể ưu tiên chuỗi ngắn quen thuộc.

### Search/length bias

EOS sớm hoặc beam score có thể bỏ cận.

## 2.3. Không nên nói

- “Dataset không có.”
- “Model nhận được `\int` nên encoder tốt.”
- “Chỉ cần thêm vài ảnh.”
- “Lỗi chắc chắn ở GAT.”

---

# Câu 3 — Làm sao xác định mô hình mất ký hiệu tích phân, mất dấu _/^ hay mất nội dung cận?

## 3.1. Bản trả lời nhanh

> **So sánh ground truth và prediction bằng token-level alignment, rồi phân loại lỗi theo slot: integral token, lower marker, upper marker, lower content, upper content, braces và body. Không chỉ nhìn chuỗi bằng mắt.**
>
> **Sau đó kiểm tra top-k beam để biết candidate đúng có tồn tại nhưng bị xếp hạng thấp hay model hoàn toàn không sinh ra.**

## 3.2. Taxonomy lỗi

| Mã lỗi | Mô tả |
|---|---|
| INT_MISS | Thiếu `\int` |
| LOW_MARK | Thiếu/nhầm `_` |
| UP_MARK | Thiếu/nhầm `^` |
| LOW_CONTENT | Sai nội dung cận dưới |
| UP_CONTENT | Sai nội dung cận trên |
| BRACE | Sai `{`/`}` |
| BODY | Sai hàm dưới dấu tích phân |
| DIFF | Sai `d x` |
| TRUNC | Dừng sớm/max length |
| EXTRA | Sinh token thừa |

### Alignment

Dùng Levenshtein traceback, không chỉ distance scalar.

### Beam diagnosis

- Ground truth hoặc canonical equivalent có trong beam nhưng không top-1 → ranking/calibration.
- Không có trong beam → representation/probability/search space.
- Beam nào cũng mất `^` → systematic bias.

### Token probability

Ở bước cần sinh `_`/`^`, lưu:

- top-10 token;
- probability;
- entropy;
- cross-attention map.

## 3.3. Không nên nói

- “Edit distance 2 nghĩa là mất hai cận.”
- “Mất `_` chắc chắn do decoder.”
- “Top-1 sai nhưng beam có đúng thì model đã đúng” — output cuối vẫn sai.

---

# Câu 4 — Làm sao kiểm tra cận trên và cận dưới còn nhìn rõ sau preprocessing?

## 4.1. Bản trả lời nhanh

> **Lưu ảnh ở mọi bước, crop vùng quanh dấu tích phân và đo số foreground pixel, bounding box, contrast và kích thước sau resize. So với ảnh train cùng cấu trúc. Sau đó xem feature activation tại vùng tương ứng.**
>
> **Không chỉ zoom ảnh đã render trên màn hình, vì interpolation của trình xem có thể làm cận trông rõ hơn tensor thật.**

## 4.2. Pipeline lưu ảnh

```text
raw
crop
grayscale
threshold
invert
resize
pad
tensor
DenseNet stage 1/2/3 feature
```

### Chỉ số ảnh

- foreground pixel count;
- local contrast;
- cận height/width;
- khoảng cách với integral;
- scale factor;
- signal-to-background ratio.

### Controlled test

Tạo nhiều phiên bản cùng biểu thức:

- cận 100%, 75%, 50%, 25% size;
- nét dày/mảnh;
- margin khác;
- threshold khác.

Vẽ curve:

$$ \text{Recall}_{^/_} \text{ theo kích thước cận} $$

### Feature-level test

- activation norm vùng cận;
- saliency;
- occlusion: che cận và xem probability thay đổi;
- compare with no-bound expression.

## 4.3. Không nên nói

- “Mắt nhìn thấy nên model phải thấy.”
- “Resize giữ tỷ lệ nên cận không mất.”
- “Feature map vẫn có cận” nếu chưa visualize.

---

# Câu 5 — Nét viết quá cong, nghiêng, dày, mảnh hoặc đứt đoạn ảnh hưởng đến feature extractor thế nào?

## 5.1. Bản trả lời nhanh

> **DenseNet học pattern từ phân bố train. Nét quá khác có thể làm edge/curve response thay đổi, khiến ký hiệu giống class khác hoặc mất stroke. Nét dày làm ký hiệu dính; nét mảnh/đứt có thể biến mất sau resize; nghiêng làm relative geometry thay đổi.**
>
> **Mức robustness phải được đo bằng controlled perturbation, không chỉ nói model “nhạy với chữ xấu”.**

## 5.2. Tác động theo loại

### Quá dày

- lấp khoảng trống;
- dính `=` thành block;
- dính cận vào `\int`;
- fraction bar nhập với chữ.

### Quá mảnh

- interpolation làm nhạt;
- threshold xóa;
- downsampling mất nét.

### Đứt đoạn

- một glyph thành nhiều component;
- `=` thành hai dash rời bất thường;
- căn thiếu hook.

### Nghiêng/cong

- thay stroke orientation;
- baseline lệch;
- superscript/subscript khó phân biệt.

### Augmentation phù hợp

- stroke dilation/erosion nhỏ;
- affine nhẹ;
- elastic nhẹ;
- broken-stroke simulation;
- antialias variants.

Cần giữ nhãn.

## 5.3. Phép đo robustness

Với perturbation strength $\epsilon$:

$$ R(\epsilon) = \operatorname{ExpRate}(T_\epsilon(X)) $$

Báo cáo degradation curve, không chỉ vài screenshot.

---

# Câu 6 — Hai ký hiệu viết dính hoặc chồng lên nhau có thể gây lỗi ở đâu trong pipeline?

## 6.1. Bản trả lời nhanh

> **Lỗi có thể xuất hiện từ ảnh: hai glyph thành một blob; DenseNet tạo feature trộn; grid node không có ranh giới symbol; GAT lan truyền feature đã trộn; decoder có thể bỏ một token hoặc sinh một token thay thế. Không có detector nên hệ thống không có bước tách ký hiệu tường minh để sửa.**

## 6.2. Các tầng lỗi

1. **Preprocessing:** morphology nối thêm nét.
2. **CNN:** receptive field không tách được pattern.
3. **Graph:** local mixing tăng sự nhập nhằng.
4. **Decoder:** language prior chọn chuỗi phổ biến.
5. **Evaluation:** insertion/deletion/substitution.

### Phân biệt encoder và decoder

- Nếu occlusion/tách ảnh làm prediction đúng → visual issue.
- Nếu top-k feature classifier/probe phân biệt được hai symbol nhưng decoder vẫn bỏ → decoder issue.
- Nếu beam có candidate đủ hai token → ranking issue.

### Hướng xử lý

- data augmentation ký hiệu dính;
- higher-resolution branch;
- auxiliary symbol count;
- stroke-aware input;
- detector/segmentation;
- coverage mechanism;
- CTC/monotonic auxiliary loss.

## 6.3. Không nên nói

- “GAT tự tách ký hiệu.”
- “Dính nét chỉ là lỗi người viết.”
- “Không có detector nên không thể nhận được” — end-to-end vẫn có thể học, chỉ không tách tường minh.

---

# Câu 7 — Có được trả lời “dataset không có tích phân có cận” khi chưa thống kê hay không?

## 7.1. Bản trả lời nhanh

> **Không. Phải nói “em chưa có thống kê xác nhận”. Sau đó đếm số mẫu `\int` không cận, một cận và hai cận trong train/test bằng parser token đúng ngữ cảnh.**
>
> **Ngay cả khi ít, phải nói số cụ thể và tỷ lệ, không nói tuyệt đối “không có”.**

## 7.2. Câu thay thế an toàn

> **Khả năng đầu tiên là cấu trúc hai cận ít hoặc phân bố khác demo; em đang kiểm chứng bằng thống kê. Các khả năng khác là resize, domain shift và decoder bias.**

### Vì sao câu “không có” nguy hiểm?

Hội đồng sẽ hỏi:

- script đâu;
- count bao nhiêu;
- `\int_^` xử lý thế nào;
- test có không;
- tại sao vocabulary có token.

Nếu không trả lời được sẽ mất uy tín.

---

# Câu 8 — Có được đổ lỗi cho người dùng viết xấu khi preprocessing hoặc giao diện vẽ chưa tương thích với train data không?

## 8.1. Bản trả lời nhanh

> **Không. “Viết xấu” là mô tả chủ quan. Hệ thống phải xác định cụ thể nét khác train ở độ dày, nghiêng, scale, spacing hay polarity. Nếu app tạo ảnh khác pipeline train thì đó là lỗi tích hợp hoặc giới hạn robustness của hệ thống, không thể đổ hoàn toàn cho người dùng.**

## 8.2. Cách nói đúng

> **Mẫu demo nằm ngoài miền phân bố mà model đã được huấn luyện, đặc biệt ở thuộc tính X; hệ thống hiện chưa robust với thay đổi đó.**

### Trách nhiệm hệ thống

- hướng dẫn canvas;
- preview normalized image;
- cảnh báo crop;
- thu thập feedback;
- confidence/OOD;
- fallback cho người dùng sửa.

### User study nếu cần

- nhiều người viết;
- nhiều thiết bị;
- standardized prompts;
- success rate;
- không chọn vài mẫu thuận lợi.

## 8.3. Không nên nói

- “Người dùng phải viết đẹp.”
- “Ảnh sai do khách hàng.”
- “Model đúng vì benchmark tốt.”
- “Out-of-distribution không thuộc trách nhiệm hệ thống.”

---

# Câu 9 — Làm sao phân biệt lỗi thị giác của encoder với lỗi ngôn ngữ/cấu trúc của decoder?

## 9.1. Bản trả lời nhanh

> **Không thể biết chỉ từ output cuối. Cần probe và intervention. Nếu feature vùng ký hiệu không phân biệt được class hoặc prediction không đổi khi che vùng đó, nghi encoder. Nếu feature/alignment có thông tin nhưng decoder chọn token phổ biến hoặc beam chứa đáp án đúng ở hạng thấp, nghi decoder/ranking.**

## 9.2. Các thí nghiệm

### Visual probe

- crop feature vùng ký hiệu;
- train linear classifier symbol/token;
- nearest-neighbor retrieval;
- compare feature same symbol/different style.

### Decoder oracle

Cung cấp ground-truth prefix và xem token tiếp theo:

- nếu vẫn sai → visual memory/decoder local issue;
- nếu đúng với GT prefix nhưng free-run sai → exposure bias.

### Beam oracle

Ground truth có trong top-k?

### Attention/occlusion

Che vùng cận:

- probability `_`/`^` có giảm không.

### Synthetic clean input

Cùng LaTeX render font sạch:

- nếu nhận đúng clean nhưng sai handwriting → visual/domain.
- nếu vẫn sai → decoder/label structure.

### Auxiliary symbol count

Nếu count head thấy đủ ký hiệu nhưng sequence thiếu, decoder có thể bỏ token.

## 9.3. Không nên nói

- “Sai ký hiệu là encoder, sai ngoặc là decoder” như quy luật tuyệt đối.
- “Attention nhìn đúng vùng nên encoder đúng.”
- “Beam có đáp án đúng nghĩa là decoder không lỗi.”

---

# Câu 10 — Nếu ground truth có cấu trúc đúng nhưng output thiếu ngoặc, đó là lỗi cú pháp hay lỗi nhận dạng cấu trúc?

## 10.1. Bản trả lời nhanh

> **Có thể là cả hai cách phân loại ở hai tầng. Thiếu ngoặc làm chuỗi LaTeX có thể không hợp lệ nên là syntax error; đồng thời nó cho thấy model không đóng đúng phạm vi cấu trúc nên là structure-generation error.**
>
> **Trong error taxonomy nên gắn nhiều nhãn thay vì ép chỉ một nguyên nhân.**

## 10.2. Ví dụ

GT:

```latex
\sqrt { x + 1 }
```

Pred:

```latex
\sqrt { x + 1
```

- token deletion: `}`;
- syntax invalid;
- root scope chưa đóng;
- exact match sai.

### Multi-label error taxonomy

```json
{
  "token_error": "deletion",
  "syntax_error": true,
  "structure_error": "unclosed_root",
  "visual_symbol_error": false
}
```

### Cần parser

Chỉ nhìn braces chưa đủ để phân loại macro scope đầy đủ.

## 10.3. Không nên nói

- “Chỉ là lỗi ngoặc nhỏ.”
- “Syntax error không liên quan structure.”
- “Mọi thiếu ngoặc do decoder.”

---

# Câu 11 — Cần xây dựng bộ test chuyên biệt nào để đánh giá tích phân có cận một cách thuyết phục?

## 11.1. Bản trả lời nhanh

> **Bộ test phải chia theo không cận, cận dưới, cận trên, cả hai cận; thay đổi kích thước cận, độ dài body, kiểu nét, spacing và domain. Phải có cả mẫu in-domain lấy từ CROHME và mẫu controlled/synthetic ngoài domain, kèm ground truth token chuẩn.**
>
> **Metric không chỉ ExpRate tổng mà còn recall `\int`, `_`, `^`, accuracy nội dung cận, syntax validity và MED.**

## 11.2. Ma trận test

### Theo cấu trúc

- `\int f(x)dx`;
- `\int_{a} f(x)dx`;
- `\int^{b} f(x)dx`;
- `\int_{a}^{b} f(x)dx`;
- multiple integrals;
- nested expression in bounds.

### Theo hình ảnh

- cận lớn/vừa/nhỏ;
- thick/thin;
- straight/slanted integral;
- close/far spacing;
- short/long body;
- mouse/stylus/camera.

### Theo content

- digit bounds;
- variable bounds;
- expressions in bounds;
- infinity;
- negative bound;
- fraction bound.

### Số lượng

Tối thiểu nên đủ mỗi cell để báo cáo CI; nếu ít, gọi exploratory set, không benchmark kết luận mạnh.

## 11.3. Metric subset

- integral symbol recall;
- lower-marker recall;
- upper-marker recall;
- both-bound exact;
- bound-content exact;
- full-expression exact;
- MED;
- syntax error;
- latency.

### Paired perturbation

Cùng công thức, chỉ thay kích thước cận. Đây là bằng chứng causal mạnh hơn ảnh ngẫu nhiên.

---

# Câu 12 — Nếu thêm dữ liệu tích phân có cận mà kết quả chung giảm, em sẽ phân tích hiện tượng này thế nào?

## 12.1. Bản trả lời nhanh

> **Không kết luận dữ liệu thêm là xấu ngay. Cần kiểm tra chất lượng nhãn, tỷ lệ trộn, domain mismatch, class imbalance ngược, catastrophic interference và training schedule. Dữ liệu mới có thể cải thiện subset tích phân nhưng làm distribution tổng thay đổi hoặc gây overfit synthetic style.**
>
> **Phải báo cáo metric chung và metric subset trước/sau, cùng nhiều seed.**

## 12.2. Các nguyên nhân

### Label noise

- braces sai;
- bound association sai;
- render/caption lệch.

### Domain mismatch

Synthetic quá sạch hoặc font-style khác handwriting.

### Sampling imbalance

Oversample tích phân làm model giảm exposure với cấu trúc phổ biến.

### Optimization

Dataset lớn hơn nhưng giữ nguyên số epoch:

- số update per sample thay đổi;
- scheduler không phù hợp;
- model chưa hội tụ.

### Catastrophic interference

Feature/language prior dịch quá mạnh sang pattern tích phân.

### Vocabulary/sequence length

Bound expression làm sequence dài hơn, tăng loss distribution.

## 12.3. Phân tích cần làm

| Metric | Trước | Sau |
|---|---:|---:|
| Overall ExpRate | | |
| Integral both-bound ExpRate | | |
| Non-integral ExpRate | | |
| MED | | |
| Token `_` recall | | |
| Token `^` recall | | |

Thử ratio:

- 5%;
- 10%;
- 20%;
- balanced batch.

Kiểm tra real-only versus synthetic-only subset.

## 12.4. Không nên nói

- “Thêm dữ liệu luôn phải tăng.”
- “Kết quả giảm nên xóa dữ liệu.”
- “Do overfitting” khi chưa xem curves.
- “Subset tăng là đủ” nếu overall application giảm.

---

# Phụ lục A — Bộ hồ sơ forensic cho mỗi demo lỗi

Mỗi lỗi nên lưu một thư mục:

```text
case_001/
├── raw.png
├── cropped.png
├── normalized.png
├── tensor_stats.json
├── gt.txt
├── prediction.txt
├── beams.json
├── token_alignment.json
├── attention/
└── diagnosis.md
```

Nội dung `diagnosis.md`:

- hiện tượng;
- tầng đầu tiên xuất hiện lỗi;
- bằng chứng;
- giả thuyết;
- thí nghiệm kiểm tra;
- kết luận tạm thời.

# Phụ lục B — Bản trả lời tổng hợp khoảng hai phút

> **Demo có thể kém test vì domain shift hoặc preprocessing mismatch. Với tích phân có cận, em tách lỗi thành nhận `\int`, marker `_`/`^`, nội dung cận, braces, body và truncate. Em không nói dataset không có nếu chưa đếm.**
>
> **Quy trình chẩn đoán là lưu ảnh ở mọi bước, kiểm tra cận còn pixel hay không, so đường app với dataloader, chạy subset CROHME cùng cấu trúc, xem token alignment và top-k beam. Nếu ảnh sau preprocessing đã mất cận thì lỗi trực tiếp ở preprocessing; nếu cận còn rõ nhưng feature không phản ứng thì nghi encoder; nếu beam có đáp án đúng nhưng top-1 sai thì nghi ranking/decoder.**
>
> **Nét quằn, dày, mảnh hoặc dính phải được chuyển thành thuộc tính đo được và kiểm tra bằng controlled perturbation. Bộ test tích phân phải phân tầng cấu trúc, kích thước cận và domain. Nếu thêm dữ liệu mà overall giảm, em phân tích label quality, sampling ratio, domain mismatch và interference thay vì kết luận đơn giản.**
