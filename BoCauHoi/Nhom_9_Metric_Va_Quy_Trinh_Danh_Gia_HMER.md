# NHÓM 9 — METRIC VÀ QUY TRÌNH ĐÁNH GIÁ

> **Mục tiêu nhóm:** Trả lời chính xác “đo cái gì”, “đo trên đơn vị nào”, “cao hay thấp thì tốt” và “metric được tính trong code ra sao”.

---

## 0. Bảng nhớ nhanh

| Metric | Đơn vị | Tốt hơn khi | Ý nghĩa |
|---|---|---|---|
| ExpRate / Exact Match | Biểu thức | Cao | Tỷ lệ toàn chuỗi đúng hoàn toàn |
| ≤1 error | Biểu thức | Cao | Tỷ lệ có token edit distance không quá 1 |
| ≤2 errors | Biểu thức | Cao | Tỷ lệ có token edit distance không quá 2 |
| Mean Edit Distance | Phép chỉnh sửa token / biểu thức | Thấp | Mức sai trung bình |
| Cross-entropy loss | Token/probability | Thấp trong cùng protocol | Mục tiêu tối ưu, không phải accuracy |
| Symbol Accuracy | Token/ký hiệu | Cao | Độ đúng cục bộ, cần định nghĩa alignment |
| Syntax Error Rate | Chuỗi | Thấp | Tỷ lệ output không hợp lệ theo parser/rule |
| Detection F1 | Bounding box | Cao | Chỉ phù hợp nếu có detector và GT box |

---

# Câu 1 — ExpRate/Exact Match được định nghĩa chính xác như thế nào?

## 1.1. Bản trả lời nhanh

> **ExpRate là tỷ lệ biểu thức có chuỗi token dự đoán trùng hoàn toàn với chuỗi token ground truth theo protocol đánh giá. Với $N$ mẫu:**
>
> $$
>
> \operatorname{ExpRate}
> =
> \frac{1}{N}
> \sum_{i=1}^{N}
> \mathbf{1}[\hat Y_i=Y_i]
>
$$
>
> **Cao hơn là tốt hơn. Chỉ cần sai, thiếu hoặc thừa một token thì mẫu đó có Exact Match bằng 0.**

## 1.2. Giải thích chi tiết

Đơn vị quan sát là **toàn biểu thức**, không phải ký hiệu.

Ví dụ ground truth:

```latex
\frac { x } { y }
```

Dự đoán thiếu một `}`:

```latex
\frac { x } { y
```

Mẫu bị tính sai hoàn toàn trong ExpRate.

### ExpRate không đo gì?

- không đo tương đương đại số;
- không phản ánh số lỗi trong một mẫu sai;
- không cho biết lỗi ở encoder hay decoder;
- không cho biết syntax hợp lệ;
- không đo confidence.

### Vì sao đây là metric chính?

Đối với HMER, một sai khác nhỏ có thể đổi nghĩa:

```latex
x ^ { 2 }
```

khác:

```latex
x _ { 2 }
```

Exact Match buộc model đúng cả ký hiệu và cấu trúc tuần tự hóa.

## 1.3. Không nên nói

- “ExpRate là accuracy từng token.”
- “Sai một token vẫn được tính một phần.”
- “ExpRate đo biểu thức tương đương toán học.”
- “ExpRate cao chứng minh model hiểu toán.”

---

# Câu 2 — Một mẫu sai một token có được tính là đúng một phần trong ExpRate hay không?

## 2.1. Bản trả lời nhanh

> **Không. Trong ExpRate, một mẫu chỉ có giá trị 1 hoặc 0. Sai một token, thiếu một token hoặc thừa một token đều làm mẫu bằng 0. Mức gần đúng chỉ được phản ánh ở ≤1 error, ≤2 errors hoặc edit distance.**

## 2.2. Minh họa

Ground truth:

```text
x ^ { 2 }
```

Prediction:

```text
x _ { 2 }
```

Edit distance token bằng 1, nhưng:

$$ \mathbf{1}[\hat Y=Y]=0 $$

Trong tổng metric:

- ExpRate: mẫu sai;
- ≤1: mẫu đúng điều kiện;
- ≤2: mẫu đúng điều kiện;
- edit distance: 1.

### Vì sao cần nhiều metric?

Nếu chỉ ExpRate, hai prediction sai 1 token và sai 20 token đều bằng 0. Edit distance bổ sung mức độ nghiêm trọng.

---

# Câu 3 — Edit distance trong repo được tính theo token hay theo ký tự?

## 3.1. Bản trả lời nhanh

> **Trong pipeline hiện tại, prediction và ground truth được chuyển thành danh sách token rồi truyền vào `editdistance.eval`, nên khoảng cách được tính theo token, không phải từng ký tự của chuỗi LaTeX đã join.**
>
> **Một macro như `\frac` được tính là một token nếu vocabulary xem nó là một token.**

## 3.2. Định nghĩa Levenshtein

Khoảng cách nhỏ nhất gồm:

- insertion;
- deletion;
- substitution.

Gọi:

$$ d(\hat Y,Y) $$

là số phép chỉnh sửa token nhỏ nhất để biến prediction thành ground truth.

Ví dụ:

```text
GT   : \frac { x } { y }
Pred : \sqrt { x } { y }
```

Nếu `\frac` và `\sqrt` là token đơn, substitution count là 1.

### Tại sao phải xác nhận kiểu dữ liệu?

Nếu code join thành string trước khi gọi edit distance, macro có thể bị tính từng ký tự. Vì vậy phải chỉ ra biến thực tế là list token trong evaluation.

### Hạn chế

Token edit distance xem mọi token có cost bằng nhau:

- sai `{` cost 1;
- sai `\int` cost 1;
- sai một biến cost 1.

Nó không phản ánh mức độ semantic khác nhau.

## 3.3. Không nên nói

- “Edit distance là số ký tự sai.”
- “Một macro luôn tương đương nhiều lỗi.”
- “Edit distance đo tree structure.”
- “Distance 1 nghĩa là sai một ký hiệu vật lý” — có thể là token cấu trúc.

---

# Câu 4 — Mean Edit Distance càng cao hay càng thấp càng tốt, và vì sao?

## 4.1. Bản trả lời nhanh

> **Càng thấp càng tốt. Mean Edit Distance là trung bình số phép chèn, xóa hoặc thay token để biến output thành ground truth:**
>
> $$
>
> \operatorname{MED}
> =
> \frac{1}{N}
> \sum_{i=1}^{N}
> d(\hat Y_i,Y_i)
>
$$
>
> **MED bằng 0 nghĩa là tất cả biểu thức đúng hoàn toàn.**

## 4.2. Cách diễn giải đúng

M4 có MED 2,06 và M1 có 2,10:

- trung bình M4 cần ít hơn 0,04 phép chỉnh sửa token/mẫu;
- không có nghĩa mỗi mẫu M4 tốt hơn;
- không có nghĩa M4 có nhiều exact matches hơn;
- không chứng minh lỗi cấu trúc cụ thể giảm.

### MED bị ảnh hưởng bởi outlier

Một số mẫu cực dài và sai nhiều có thể kéo mean lên.

Nên báo cáo thêm:

- median;
- P90/P95;
- histogram;
- normalized edit distance:

$$ d_{\text{norm}} = \frac{d(\hat Y,Y)} {\max(|\hat Y|,|Y|)} $$

Nếu benchmark chính dùng raw MED, normalized chỉ là metric phụ.

## 4.3. Không nên nói

- “MED cao là tốt.”
- “M4 chính xác nhất vì MED thấp.”
- “Chênh 0,14 chắc chắn có ý nghĩa thống kê.”
- “MED trực tiếp đo cấu trúc.”

---

# Câu 5 — Các chỉ số ≤1 error và ≤2 errors được tính như thế nào?

## 5.1. Bản trả lời nhanh

> **Chúng là tỷ lệ biểu thức có token edit distance không vượt ngưỡng 1 hoặc 2:**
>
> $$
>
> R_{\le k}
> =
> \frac{1}{N}
> \sum_i
> \mathbf{1}[d(\hat Y_i,Y_i)\le k]
>
$$
>
> **Càng cao càng tốt. ExpRate chính là trường hợp khoảng cách bằng 0.**

## 5.2. Quan hệ giữa các metric

Luôn có:

$$ R_{0} \le R_{\le1} \le R_{\le2} $$

Trong đó:

$$ R_0=\operatorname{ExpRate} $$

### Ý nghĩa

- khoảng cách 0: đúng hoàn toàn;
- khoảng cách 1: chỉ cần một chỉnh sửa;
- khoảng cách 2: chỉ cần tối đa hai chỉnh sửa.

### Không phải “accuracy 1 lỗi”

Nó không nói token nào sai hoặc lỗi có làm đổi nghĩa nặng hay nhẹ.

### Từ bảng có thể suy count gần đúng

Ví dụ M4 Avg:

- ExpRate 48,98%;
- ≤1 67,43%;
- ≤2 76,61%.

Tỷ lệ distance đúng 1 gần:

$$ 67,43-48,98=18,45\% $$

Tỷ lệ distance đúng 2 gần:

$$ 76,61-67,43=9,18\% $$

Đây là phép suy từ tỷ lệ trung bình, cần cẩn thận vì average theo dataset không đồng nhất với gộp sample.

## 5.3. Không nên nói

- “≤1 là Symbol Accuracy.”
- “≤2 nghĩa là sai hai ký tự.”
- “Mọi lỗi ≤2 đều chấp nhận được trong ứng dụng.”

---

# Câu 6 — Kết quả được đánh giá sau khi normalize LaTeX hay trên chuỗi thô?

## 6.1. Bản trả lời nhanh

> **Code hiện cho thấy prediction và ground truth được đổi từ index về token rồi so trực tiếp. Chưa thấy một bước canonicalize LaTeX semantic được gọi trước ExpRate/edit distance. Vì vậy cách nói an toàn là đánh giá trên token sequence theo định dạng đã chuẩn hóa sẵn trong dataset.**
>
> **Muốn khẳng định các biểu diễn tương đương được gộp, phải chỉ ra hàm normalizer cụ thể.**

## 6.2. Ba lớp cần phân biệt

1. Caption đã được chuẩn hóa khi tạo dataset.
2. Tokenizer đọc caption.
3. Evaluation-time normalization.

Repo chứng minh rõ lớp 2; lớp 1 cần script nguồn; lớp 3 chưa thấy.

### Hệ quả

Nếu prediction:

```latex
x ^ 2
```

và GT:

```latex
x ^ { 2 }
```

thì có thể bị xem khác nếu token sequence khác.

### Protocol cần viết

- có bỏ SOS/EOS/PAD trước so không;
- whitespace;
- braces;
- alias macro;
- invalid tokens;
- EOS early;
- padding.

## 6.3. Không nên nói

- “Đánh giá trên LaTeX render.”
- “Tất cả biểu diễn tương đương được normalize.”
- “Chuỗi thô ký tự” nếu thực tế là token list.

---

# Câu 7 — Symbol Accuracy khác ExpRate ở đâu và khi nào hai chỉ số có thể mâu thuẫn?

## 7.1. Bản trả lời nhanh

> **Symbol Accuracy đo mức đúng ở cấp token/ký hiệu sau alignment, còn ExpRate đo toàn biểu thức. Symbol Accuracy có thể cao nhưng ExpRate thấp vì mỗi biểu thức chỉ cần sai một token là bị Exact Match tính sai.**
>
> **Repo hiện không dùng Symbol Accuracy như metric chính; nếu báo cáo phải định nghĩa alignment và denominator chính xác.**

## 7.2. Ví dụ

100 biểu thức, mỗi biểu thức 20 token. Model sai đúng 1 token trong mỗi biểu thức:

- token accuracy xấp xỉ 95%;
- ExpRate = 0%.

### Định nghĩa có thể dùng

Từ edit distance:

$$ \operatorname{TokenAccuracy} = 1- \frac{\sum_i d(\hat Y_i,Y_i)} {\sum_i |Y_i|} $$

Nhưng giá trị có thể âm nếu prediction rất dài, nên cần clamp hoặc định nghĩa khác.

Hoặc sau Levenshtein alignment:

$$ \frac{\text{correct aligned tokens}} {\text{GT tokens}} $$

Phải công bố công thức.

### Symbol Accuracy không đồng nghĩa visual symbol recognition

Token gồm `{`, `}`, `\frac`, nên “symbol” có thể là LaTeX token, không phải glyph detector class.

## 7.3. Không nên nói

- “Repo đạt Symbol Accuracy 80,3%” nếu script/log không có.
- “Symbol Accuracy cao nghĩa là cấu trúc đúng.”
- “Mỗi token là một symbol trên ảnh.”

---

# Câu 8 — Loss khác metric đánh giá như thế nào; loss thấp có đồng nghĩa ExpRate cao không?

## 8.1. Bản trả lời nhanh

> **Loss là hàm khả vi tối ưu xác suất token, còn ExpRate là quyết định rời rạc toàn chuỗi sau decoding. Loss thấp thường là tín hiệu tốt trong cùng protocol nhưng không bảo đảm ExpRate cao.**
>
> **Model có thể giảm loss bằng cách tăng confidence ở token đã đúng mà không sửa token sai quyết định Exact Match.**

## 8.2. Cross-entropy

$$ \mathcal{L}_{CE} = -\sum_t\log p(y_t^{GT}\mid y_{ \lt t}^{GT},X) $$

Repo còn có structure loss phụ.

### Metric sau decoding

$$ \hat Y = \operatorname{BeamSearch}(p_\theta) $$

sau đó so sequence.

### Nguyên nhân không đồng biến hoàn toàn

- teacher forcing versus autoregressive inference;
- beam ranking;
- calibration;
- length bias;
- auxiliary loss;
- class imbalance;
- exact-match discontinuity.

### Chỉ so loss khi nào?

- cùng loss definition;
- cùng vocabulary;
- cùng data;
- cùng masking;
- cùng reduction;
- cùng sequence length treatment.

Không so raw loss giữa hai codebase khác nhau rồi xếp hạng accuracy.

## 8.3. Không nên nói

- “Loss thấp nhất là model tốt nhất.”
- “Loss 0,2 tương đương 80% accuracy.”
- “Train loss giảm chứng minh tổng quát.”

---

# Câu 9 — Syntax Error Rate phải được định nghĩa bằng parser, quy tắc ngoặc hay bộ kiểm tra nào?

## 9.1. Bản trả lời nhanh

> **Syntax Error Rate chỉ có ý nghĩa khi định nghĩa một validator cụ thể. Có thể dùng parser/renderer LaTeX trong môi trường sandbox hoặc một grammar checker token-level. Chỉ kiểm tra cân bằng ngoặc là chưa đủ để gọi toàn bộ chuỗi hợp lệ.**
>
> **Repo hiện không thấy Syntax Error Rate là metric chính, nên không được báo cáo con số nếu chưa có script tái lập.**

## 9.2. Công thức

$$ \operatorname{SER} = \frac{\#\text{output bị validator từ chối}} {N} $$

Càng thấp càng tốt.

### Các cấp kiểm tra

1. token hợp lệ;
2. braces cân bằng;
3. macro đúng số argument;
4. parser chấp nhận;
5. renderer compile thành công.

Một chuỗi có braces cân bằng vẫn có thể sai:

```latex
\frac { x }
```

### Validator cần đảm bảo

- timeout;
- sandbox;
- không cho lệnh nguy hiểm;
- version cố định;
- cùng preamble;
- log reason.

### Tách syntax và semantic

Chuỗi hợp lệ:

```latex
x _ { 2 }
```

nhưng GT là:

```latex
x ^ { 2 }
```

Syntax đúng, recognition sai.

## 9.3. Không nên nói

- “Thiếu ngoặc là mọi loại syntax error.”
- “Render được thì nhận dạng đúng.”
- “Syntax Error Rate 14,6%” nếu không truy được script.

---

# Câu 10 — F1-score phù hợp với module nào và có phù hợp với recognizer không có detector hay không?

## 10.1. Bản trả lời nhanh

> **F1 phù hợp cho detection, segmentation hoặc classification mất cân bằng khi có TP, FP, FN được định nghĩa rõ. Core recognizer hiện không có symbol detector hoặc bounding box output, nên detection F1 không phải metric chính phù hợp.**
>
> **Có thể dùng token-level F1 cho một số token, nhưng phải định nghĩa task riêng và không thay ExpRate.**

## 10.2. Công thức

$$ P=\frac{TP}{TP+FP} $$

$$ R=\frac{TP}{TP+FN} $$

$$ F1=\frac{2PR}{P+R} $$

### Với detector cần

- ground-truth box;
- predicted box;
- IoU threshold;
- class matching;
- NMS;
- micro/macro averaging.

Repo không có các đầu ra này.

### Token F1

Có thể đo per-token presence, nhưng bỏ thứ tự và structure nếu chỉ dùng bag-of-token. Không phù hợp thay thế sequence metric.

## 10.3. Không nên nói

- “F1 của GAT là ...”
- “F1 detection áp dụng vì feature map có node.”
- “F1 cao nghĩa là LaTeX đúng.”

---

# Câu 11 — Kết quả dùng greedy decoding hay beam search; beam size bao nhiêu?

## 11.1. Bản trả lời nhanh

> **Evaluation hiện dùng bidirectional beam search, không phải greedy. Config đặt beam size 10, max length 150, alpha 1, temperature 1 và `early_stopping: false`.**
>
> **Vì decoding ảnh hưởng score, mọi model phải dùng cùng beam config.**

## 11.2. Greedy versus beam

Greedy:

$$ \hat y_t=\arg\max p(y_t\mid\hat y_{ \lt t},X) $$

Beam giữ $K$ hypothesis có score cao nhất.

### Beam không bảo đảm tối ưu toàn cục

- search vẫn hữu hạn;
- length normalization ảnh hưởng;
- EOS bias;
- candidate pruning.

### Bidirectional TAMER

Model có thể sinh theo hai hướng và kết hợp/rerank tùy implementation. Phải mô tả đúng code, không chỉ nói “beam search thường”.

### Ablation nên có

- greedy;
- beam 5;
- beam 10;
- beam 20;
- latency;
- oracle beam accuracy.

## 11.3. Không nên nói

- “Beam search luôn tốt hơn.”
- “Beam 10 nghĩa là accuracy tăng 10 lần.”
- “Training dùng beam.”
- “Mọi paper dùng cùng beam.”

---

# Câu 12 — x^2 và x^{2} nên được xem là cùng một dự đoán hay hai dự đoán khác nhau?

## 12.1. Bản trả lời nhanh

> **Theo semantic/rendering chúng có thể tương đương, nhưng theo Exact Match token-level chúng chỉ giống nếu protocol normalizer canonicalize về cùng dạng. Với repo hiện tại chưa thấy evaluation normalizer, nên nếu sequence khác thì bị tính khác.**
>
> **Luận văn phải báo cáo metric benchmark theo protocol và có thể thêm normalized metric phụ.**

## 12.2. Nguyên tắc

Không tự sửa prediction theo GT.

Normalizer phải:

- deterministic;
- áp dụng cả hai phía;
- công khai;
- không thay đổi semantic;
- có unit test.

### Không được canonicalize đại số

Không biến:

```latex
x+1
```

và:

```latex
1+x
```

thành giống nhau trong recognition benchmark, dù có thể tương đương trong đại số giao hoán.

---

# Câu 13 — Có nên báo cáo kết quả tốt nhất của một lần chạy hay trung bình nhiều seed?

## 13.1. Bản trả lời nhanh

> **Nên báo cáo mean ± standard deviation của nhiều seed, ít nhất cho M1, M3, M4 và M5. Một best run có thể là dao động may mắn và làm chênh lệch nhỏ như 0,09 điểm phần trăm trở nên không đáng tin.**
>
> **Có thể ghi best run để tái lập checkpoint, nhưng kết luận chính nên dựa trên phân bố nhiều run.**

## 13.2. Báo cáo đề xuất

$$ \bar x\pm s $$

kèm:

- số seed;
- seed list;
- best/worst;
- confidence interval;
- test protocol.

### Khi tài nguyên hạn chế

- chạy 3 seed model chính;
- run ngắn pilot cho phần còn lại;
- thừa nhận single-run limitation.

### Không dùng test để chọn seed tốt nhất

Nếu train nhiều seed rồi chỉ báo cáo seed cao nhất trên test, đó là test selection bias.

## 13.3. Không nên nói

- “Một seed là kết quả chính thức.”
- “Deterministic nên không cần nhiều seed.”
- “Best checkpoint của best seed là estimate không thiên lệch.”

---

# Câu 14 — Một mô hình có edit distance tốt nhất nhưng ExpRate thấp hơn thì nên kết luận mô hình nào tốt hơn?

## 14.1. Bản trả lời nhanh

> **Không có câu trả lời tuyệt đối; phụ thuộc mục tiêu. Nếu ưu tiên công thức hoàn toàn đúng, M1 tốt hơn theo Avg ExpRate. Nếu ưu tiên giảm số token cần sửa, M4 tốt hơn theo MED.**
>
> **Cách kết luận đúng là có trade-off, không gọi một model tốt nhất chung.**

## 14.2. Theo ứng dụng

### Tự động nhập công thức không hậu kiểm

Ưu tiên Exact Match.

### Công cụ gợi ý có người sửa

MED thấp có giá trị vì giảm effort chỉnh sửa.

### Hệ thống chấm bài

Một lỗi nhỏ vẫn có thể đổi nghĩa; cần exact/semantic check.

### Cách chọn model

Định nghĩa utility:

$$ U = w_0R_0 + w_1R_{\le1} - w_dd - w_t\text{latency} $$

Trọng số tùy ứng dụng.

### Kết luận hiện tại

- M1: best exact average.
- M4: best average error severity.
- Chưa có significance test và latency comparison.

## 14.3. Không nên nói

- “M4 tốt nhất.”
- “M1 tốt nhất mọi mặt.”
- “MED quan trọng hơn Exact Match.”
- “Hai metric mâu thuẫn nên bỏ một.”

---

# Phụ lục A — Bản trả lời tổng hợp khoảng hai phút

> **Metric chính là ExpRate: tỷ lệ toàn bộ chuỗi token trùng ground truth. Sai một token thì mẫu bằng 0. Để đo gần đúng, repo dùng token-level Levenshtein distance; ≤1 và ≤2 là tỷ lệ mẫu có distance tối đa 1 hoặc 2, còn Mean Edit Distance càng thấp càng tốt.**
>
> **Code hiện đánh giá trên token sequence theo định dạng dataset, chưa thấy semantic LaTeX normalizer. Vì vậy hai chuỗi render tương đương vẫn có thể bị tính khác. Loss là cross-entropy/structure loss khả vi, không đồng nghĩa ExpRate. Symbol Accuracy, Syntax Error Rate và F1 chỉ được dùng nếu có công thức và script rõ; core repo không có detector nên detection F1 không phù hợp.**
>
> **Evaluation dùng bidirectional beam search với beam size 10 và max length 150. Kết quả nên báo cáo nhiều seed. Với M1 và M4, kết luận đúng là trade-off: M1 có Exact Match trung bình cao nhất, M4 có MED thấp nhất. Không gọi một model tốt nhất nếu không nêu metric và mục tiêu ứng dụng.**

# Phụ lục B — Checklist metric

- [ ] Xác nhận list token trước `editdistance.eval`.
- [ ] Xác nhận cách loại SOS/EOS/PAD.
- [ ] Tài liệu normalization.
- [ ] Cùng beam config.
- [ ] Count distance 0/1/2/>2.
- [ ] Histogram và median distance.
- [ ] Syntax validator nếu báo cáo SER.
- [ ] Nhiều seed và confidence interval.
- [ ] Không dùng Symbol Accuracy/F1 không có script.
