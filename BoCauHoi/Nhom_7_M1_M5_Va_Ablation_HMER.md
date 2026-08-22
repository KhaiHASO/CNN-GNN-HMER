# NHÓM 7 — M1–M5 VÀ CÁC THÍ NGHIỆM ABLATION

> **Mục tiêu nhóm:** Giải thích được từng phiên bản, giả thuyết thay đổi, kết quả và kết luận khoa học mà không phóng đại.

---

## 0. Bảng kết quả gốc cần học thuộc

### 0.1. Kết quả theo từng tập kiểm thử

| Mô hình | CROHME 2014 ExpRate | ≤1 | ≤2 | MED | CROHME 2016 ExpRate | ≤1 | ≤2 | MED | CROHME 2019 ExpRate | ≤1 | ≤2 | MED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | **51,12** | **69,98** | **77,69** | 1,99 | 50,65 | **67,92** | **76,02** | 2,17 | **48,54** | **68,14** | 77,23 | 2,14 |
| M2 | 49,39 | 66,53 | 75,25 | 2,22 | 47,43 | 64,95 | 74,72 | 2,31 | 46,71 | 66,89 | 75,90 | 2,11 |
| M3 | 48,88 | 66,73 | 75,36 | 2,19 | **50,74** | 66,96 | 75,85 | 2,19 | 47,87 | 67,81 | **77,98** | **2,02** |
| M4 | 49,90 | 67,44 | 77,18 | **1,98** | 49,17 | 67,13 | 75,76 | **2,13** | 47,87 | 67,72 | 76,90 | 2,08 |
| M5 | 46,65 | 63,99 | 73,43 | 2,48 | 45,68 | 63,03 | 73,23 | 2,53 | 37,70 | 58,97 | 70,14 | 2,93 |

### 0.2. Trung bình ba tập

| Mô hình | Avg ExpRate | Avg ≤1 | Avg ≤2 | Mean Edit Distance |
|---|---:|---:|---:|---:|
| M1 | **50,10** | **68,68** | **76,98** | 2,10 |
| M2 | 47,84 | 66,12 | 75,29 | 2,21 |
| M3 | 49,17 | 67,17 | 76,40 | 2,14 |
| M4 | 48,98 | 67,43 | 76,61 | **2,06** |
| M5 | 43,35 | 62,00 | 72,27 | 2,65 |

### 0.3. Bốn kết luận phải giữ nguyên

1. **M1 có ExpRate trung bình cao nhất.**
2. **M4 có Mean Edit Distance trung bình thấp nhất.**
3. **M3 chỉ nhỉnh hơn M1 về ExpRate trên CROHME 2016; không thắng M1 trên toàn bộ benchmark.**
4. **M5 thấp rõ rệt trên tất cả các metric tổng hợp.**

---

# Câu 1 — M1 là baseline gì và mục đích tồn tại của M1 trong nghiên cứu là gì?

## 1.1. Bản trả lời nhanh

> **M1 là baseline DenseNet–Transformer không có GAT. DenseNet trích xuất feature map, positional encoding được đưa vào encoder memory, rồi Transformer Decoder sinh LaTeX. Mục đích của M1 là tạo nhóm đối chứng để đo tác động thực sự của GAT và các thiết kế vị trí ở M2–M5.**
>
> **Nếu không có M1, em không thể biết thay đổi kết quả đến từ GAT hay chỉ do backbone, decoder, dữ liệu hoặc cách đánh giá.**

## 1.2. Giải thích chi tiết

M1 trả lời câu hỏi:

> Với cùng dữ liệu và decoder, mô hình không có graph refinement đạt kết quả bao nhiêu?

Pipeline khái quát:

```text
Image
→ DenseNet
→ projected feature map
→ 2D positional encoding
→ Transformer Decoder
→ LaTeX
```

M1 phải được giữ ổn định về:

- dataset;
- vocabulary;
- preprocessing;
- optimizer;
- số epoch;
- checkpoint criterion;
- decoding;
- beam size;
- seed;
- metric.

M1 không phải một mô hình “yếu để làm nền”. Kết quả cho thấy M1 là baseline rất mạnh và vẫn có ExpRate trung bình tốt nhất. Điều này làm nghiên cứu đáng tin hơn vì các biến thể GAT phải vượt qua một đối chứng thực sự cạnh tranh.

## 1.3. Câu hỏi truy tiếp

**“Nếu M1 vẫn cao nhất, tại sao không dừng ở M1?”**

> M1 phù hợp mục tiêu tối đa hóa Exact Match hiện tại. Tuy nhiên luận văn còn nghiên cứu cách message passing và positional information ảnh hưởng representation. M4 giảm Mean Edit Distance, M2–M3 cho thấy thứ tự PE quan trọng, và M5 cung cấp negative result về scale-up. Giá trị nghiên cứu không chỉ là chọn số cao nhất.

## 1.4. Không nên nói

- “M1 không hiểu cấu trúc.”
- “M1 chỉ là CNN đơn giản.”
- “M1 kém hơn GAT.”
- “Baseline tồn tại chỉ để làm M4 đẹp hơn.”

---

# Câu 2 — M2 thay đổi module nào so với M1 và giả thuyết ban đầu là gì?

## 2.1. Bản trả lời nhanh

> **M2 chèn GAT vào encoder sau khi feature đã được cộng positional encoding, tức PE đi vào message passing. Cấu hình được mô tả là GAT 2 lớp, 8 head. Giả thuyết ban đầu là GAT sẽ tổng hợp ngữ cảnh lân cận và hỗ trợ cấu trúc 2D tốt hơn baseline.**
>
> **Kết quả ngược kỳ vọng: Avg ExpRate giảm từ 50,10% xuống 47,84%, còn MED gần như không cải thiện. Điều này bác bỏ cách tích hợp GAT ngây thơ trong thiết lập đã thử.**

## 2.2. Giải thích chi tiết

M2 có thể mô tả:

$$ H_{\text{M2}} = \operatorname{GAT}(F+PE) $$

Sau đó decoder nhận memory đã qua GAT.

Giả thuyết ban đầu:

- feature lân cận bổ sung ngữ cảnh;
- GAT học neighbor importance;
- biểu thức 2D được biểu diễn tốt hơn.

Quan sát:

$$ \Delta\text{ExpRate}_{M2-M1} = 47,84-50,10 = -2,26 $$

điểm phần trăm trung bình.

Không nên giải thích bằng một nguyên nhân chắc chắn. Các khả năng gồm:

- PE tuyệt đối bị trộn trong message passing;
- GAT làm quá mượt feature;
- optimizer chưa phù hợp;
- nhiều layer/head hơn mức cần thiết;
- dropout;
- parameter count và training budget;
- biến thể code khác giữa các run.

## 2.3. Kết luận khoa học

> **Việc chỉ thêm GAT không bảo đảm cải thiện. Vị trí chèn và cách mã hóa vị trí là biến thiết kế quan trọng.**

---

# Câu 3 — M3 khác M2 chính xác ở vị trí đưa positional encoding như thế nào?

## 3.1. Bản trả lời nhanh

> **M2 cộng positional encoding trước GAT, còn M3 cho GAT xử lý visual feature trước rồi mới cộng PE trước decoder. Các phần còn lại được kỳ vọng giữ giống nhau.**
>
> **M3 tăng Avg ExpRate từ 47,84% lên 49,17% và giảm MED từ 2,21 xuống 2,14 so với M2. Vì vậy trong thiết lập này, PE sau GAT hiệu quả hơn PE trước GAT.**

## 3.2. Công thức so sánh

M2:

$$ Z_{M2} = \operatorname{GAT}(F+PE) $$

M3:

$$ Z_{M3} = \operatorname{GAT}(F)+PE $$

Trực giác:

- GAT ở M3 tập trung tổng hợp visual feature;
- PE tuyệt đối được giữ như tín hiệu tọa độ rõ ràng cho decoder;
- absolute position không bị neighbor mixing.

## 3.3. Điều kiện để ablation M2–M3 sạch

Phải xác nhận từ code/config từng run rằng các yếu tố khác giống nhau:

- adjacency;
- GAT layer/head;
- d_model;
- dropout;
- optimizer;
- seed;
- epoch;
- checkpoint;
- beam search.

Nếu không có commit riêng, không được nói M2–M3 chỉ khác đúng một dòng dựa trên README.

## 3.4. Kết luận đúng mức

Được nói:

> **PE sau GAT tốt hơn PE trước GAT trong cấu hình đã thử.**

Chưa được nói:

- mọi GNN đều phải đặt PE sau;
- đã chứng minh PE bị “xóa”;
- M3 tốt hơn M1 nói chung.

---

# Câu 4 — M4 bổ sung những thành phần nào liên quan đến tọa độ và relative position?

## 4.1. Bản trả lời nhanh

> **M4 dùng graph lưới 8 hướng có self-loop, thêm relative directional bias 9 trạng thái vào attention logits của từng head, giữ absolute 2D PE sau GAT, và dùng cấu hình gọn 1 GAT layer, 4 head.**
>
> **Chín trạng thái tương ứng với $\Delta x,\Delta y\in\{-1,0,1\}$: tám hướng lân cận và vị trí self.**

## 4.2. Công thức relative state

Với node $i,j$:

$$ r_{ij} = 3(\Delta y+1)+(\Delta x+1) $$

nên:

$$ r_{ij}\in\{0,\ldots,8\} $$

Mỗi head $k$ có bias:

$$ b_{k,r_{ij}} $$

Attention logit:

$$ e_{ij}^{(k)} = e_{ij,\text{content}}^{(k)} + b_{k,r_{ij}} $$

M4 không concatenate tọa độ tuyệt đối vào node. Nó tách:

- relative direction trong GAT;
- absolute PE sau GAT.

## 4.3. Kết quả

M4:

- Avg ExpRate 48,98%;
- Avg MED 2,06, tốt nhất;
- không vượt M1 về Exact Match.

Cách diễn giải:

> M4 tạo chuỗi gần ground truth hơn trung bình, nhưng chưa làm nhiều mẫu đạt khoảng cách 0 hơn M1.

## 4.4. Hạn chế ablation

M4 thay nhiều biến so với M3:

- connectivity;
- relative bias;
- layer;
- head.

Không thể quy toàn bộ chênh lệch cho relative bias.

---

# Câu 5 — M5 tăng độ sâu hoặc độ rộng ở đâu so với M4?

## 5.1. Bản trả lời nhanh

> **M5 giữ hướng coordinate-aware nhưng tăng từ 1 GAT layer, 4 head ở M4 lên 2 GAT layer, 8 head. Với `d_model=256`, head dimension giảm từ 64 xuống 32 khi số head tăng.**
>
> **Kết quả không cải thiện mà giảm mạnh: Avg ExpRate 43,35% và MED 2,65. Điều này cho thấy tăng capacity/message-passing depth không mặc định tốt hơn.**

## 5.2. Những gì thay đổi

| Thuộc tính | M4 | M5 |
|---|---:|---:|
| GAT layers | 1 | 2 |
| Heads | 4 | 8 |
| d_model | 256 | 256 |
| Head dimension | 64 | 32 |
| Relative bias | Có | Có |
| PE sau GAT | Có | Có |

Cần xác nhận dropout và mọi config khác từ run.

## 5.3. Các giả thuyết giải thích

- over-smoothing;
- overfitting;
- optimization khó;
- attention dropout qua hai tầng;
- head dimension quá nhỏ;
- relative bias qua nhiều layer;
- seed variance;
- checkpoint/training budget chưa tối ưu.

Đây là **giả thuyết**, không phải kết luận đã chứng minh.

---

# Câu 6 — Bảng cấu hình M1–M5 cần những cột nào để hội đồng nhìn là hiểu ngay?

## 6.1. Bản trả lời nhanh

> **Bảng phải cho thấy mỗi phiên bản khác ở đâu và những yếu tố nào được giữ nguyên. Tối thiểu cần: backbone, loại node, connectivity, GAT on/off, PE trước/sau, relative bias, số layer/head, d_model, dropout, parameter count, optimizer, epoch, seed, checkpoint criterion và decoding.**

## 6.2. Mẫu bảng đề xuất

| Thuộc tính | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| Backbone | DenseNet | DenseNet | DenseNet | DenseNet | DenseNet |
| Node | — | Feature cell | Feature cell | Feature cell | Feature cell |
| Graph | — | Cần xác nhận | Cần xác nhận | 8-neighbor+self | 8-neighbor+self |
| GAT | Không | Có | Có | Có | Có |
| Absolute PE | Trước decoder | Trước GAT | Sau GAT | Sau GAT | Sau GAT |
| Relative bias | Không | Không | Không | 9-state | 9-state |
| GAT layers | 0 | 2 | 2 | 1 | 2 |
| Heads | 0 | 8 | 8 | 4 | 8 |
| d_model | 256 | 256 | 256 | 256 | 256 |
| Dropout |  |  |  |  |  |
| Params |  |  |  |  |  |
| Seed |  |  |  |  |  |
| Best epoch |  |  |  |  |  |

Các ô trống phải điền từ log, không đoán.

## 6.3. Vì sao cần parameter count?

Nếu M4/M5 có nhiều tham số hơn M1, hội đồng có thể hỏi cải thiện do graph hay capacity. Parameter count giúp kiểm soát lập luận.

---

# Câu 7 — Vì sao phải giữ nguyên dataset, preprocessing và quy trình đánh giá khi so sánh M1–M5?

## 7.1. Bản trả lời nhanh

> **Ablation nhằm đo tác động của một thay đổi. Nếu đồng thời đổi dữ liệu, preprocessing, checkpoint criterion hoặc beam size, chênh lệch metric bị confound và không thể quy cho GAT/PE.**
>
> **Do đó mọi model phải dùng cùng split, vocabulary, preprocessing, seed policy, train budget và evaluation script.**

## 7.2. Nguyên tắc kiểm soát biến

Muốn ước lượng tác động của biến $A$:

$$ \Delta_A = \operatorname{Metric}(A=1) - \operatorname{Metric}(A=0) $$

các biến khác phải giữ nguyên hoặc được randomize có hệ thống.

### Những yếu tố dễ gây confound

- data version;
- validation/test role;
- augmentation;
- image resolution;
- optimizer;
- learning rate;
- max epoch;
- seed;
- beam size;
- normalization;
- vocabulary;
- checkpoint selection.

### Repo hiện có rủi ro

- CROHME 2014 dùng validation;
- code main có thể không đúng code từng run;
- M4 đổi nhiều biến;
- mới có một seed.

Phải ghi rõ để không phóng đại ablation.

---

# Câu 8 — Kết luận chính của toàn bộ chuỗi ablation M1–M5 là gì?

## 8.1. Bản trả lời nhanh

> **Thứ nhất, thêm GAT theo cách ngây thơ không cải thiện: M2 thấp hơn M1. Thứ hai, vị trí PE quan trọng: M3 tốt hơn M2 khi chuyển PE ra sau GAT. Thứ ba, cấu hình coordinate-aware gọn M4 đạt Mean Edit Distance tốt nhất nhưng không thắng Exact Match. Thứ tư, scale-up lên M5 làm giảm mạnh.**
>
> **Vì vậy kết luận không phải “GAT tốt hơn Transformer”, mà là hiệu quả GAT phụ thuộc mạnh vào thiết kế vị trí, relative bias và độ sâu.**

## 8.2. Kết luận theo RQ

- **RQ1:** GAT có tự động giúp không? → Không.
- **RQ2:** PE order có ảnh hưởng không? → Có, PE sau tốt hơn PE trước trong run hiện có.
- **RQ3:** Coordinate-aware compact design có ích không? → Có lợi cho MED, chưa có lợi cho Avg ExpRate.
- **RQ4:** GAT sâu/rộng hơn có tốt hơn không? → Không trong cấu hình đã thử.

## 8.3. Giá trị của negative result

M2 và M5 chỉ ra:

- tích hợp sai có thể làm baseline giảm;
- capacity lớn hơn không đảm bảo tổng quát;
- cần thiết kế và kiểm chứng, không ghép module theo trực giác.

---

# Câu 9 — Vì sao đưa positional encoding trước GAT có thể làm giảm hiệu quả?

## 9.1. Bản trả lời nhanh

> **Khi PE được cộng trước GAT, message passing trộn cả visual feature lẫn tọa độ tuyệt đối giữa các node lân cận. Điều này có thể làm tín hiệu vị trí của từng node kém phân biệt hơn hoặc khiến attention tối ưu khó hơn.**
>
> **Kết quả M2–M3 phù hợp với giả thuyết này, nhưng chưa đo trực tiếp nên phải gọi là “có thể” chứ không khẳng định đã chứng minh position blurring.**

## 9.2. Minh họa

Giả sử:

$$ h_i=f_i+p_i $$

GAT tạo:

$$ h_i' = \sum_j\alpha_{ij}W(f_j+p_j) $$

Thành phần position sau aggregation:

$$ \sum_j\alpha_{ij}Wp_j $$

không còn chỉ là $p_i$.

### Cách kiểm chứng

- linear probe dự đoán $(x,y)$ từ embedding trước/sau GAT;
- cosine similarity PE;
- attention alignment;
- kiểm tra decoder localization;
- so nhiều seed.

---

# Câu 10 — Vì sao đưa positional encoding sau GAT có thể phục hồi kết quả?

## 10.1. Bản trả lời nhanh

> **GAT trước PE cho phép message passing xử lý visual feature thuần. Sau đó mỗi node được gắn lại absolute position riêng trước decoder, nên decoder nhận cả feature đã làm giàu cục bộ và tọa độ không bị neighbor mixing.**
>
> **M3 phục hồi phần lớn khoảng giảm của M2, nhưng vẫn chưa vượt M1 trung bình, nên PE order là một phần của lời giải chứ chưa đủ bảo đảm cải thiện.**

## 10.2. Công thức

$$ \tilde h_i = h_i+\operatorname{GAT}(h)_i $$

$$ z_i = \operatorname{LN}(\tilde h_i+PE_i) $$

Residual giữ feature gốc; PE gắn sau giúp decoder phân biệt node.

### Hạn chế

- PE sau không sửa được feature đã mất do downsampling;
- GAT vẫn có thể over-smooth visual component;
- decoder vẫn phải học structure;
- kết quả còn phụ thuộc optimizer và data.

---

# Câu 11 — Relative position bias trong M4 được định nghĩa theo bao nhiêu trạng thái hoặc khoảng cách?

## 11.1. Bản trả lời nhanh

> **M4 dùng 9 trạng thái dựa trên $\Delta x,\Delta y\in\{-1,0,1\}$: tám hướng lân cận và self. Đây là direction category, không phải khoảng cách nhiều mức. Mỗi attention head có một bảng bias riêng cho 9 trạng thái.**

## 11.2. Bản đồ trạng thái

Có thể trình bày:

```text
0 1 2
3 4 5
6 7 8
```

Tùy quy ước trục trong code, index giữa là self; các index còn lại là các hướng.

Công thức:

$$ r=3(\Delta y+1)+(\Delta x+1) $$

### Điều không được nói

- 9 quan hệ toán học;
- 9 khoảng cách;
- có relation “superscript” riêng.

---

# Câu 12 — Vì sao M4 có thể giảm edit distance nhưng không tăng exact match?

## 12.1. Bản trả lời nhanh

> **Exact Match chỉ nhận giá trị 1 khi toàn bộ chuỗi đúng. Nếu M4 giảm một mẫu từ 4 lỗi xuống 1 lỗi thì edit distance cải thiện, nhưng mẫu vẫn bị ExpRate tính 0.**
>
> **Do đó M4 có thể làm nhiều prediction “gần đúng hơn” mà chưa đủ chuyển chúng thành chuỗi hoàn toàn đúng.**

## 12.2. Ví dụ

Ground truth:

```latex
\frac { x + 1 } { y }
```

M1 sai 4 token; M4 chỉ thiếu một `}`.

- ExpRate cả hai mẫu: 0.
- Edit distance M4 thấp hơn.

### Cần xem phân bố

Nếu M4 tốt thật theo mức độ lỗi, histogram có thể cho thấy:

- ít mẫu distance lớn;
- nhiều mẫu dồn về 1–2;
- nhưng số distance 0 chưa tăng.

Nên báo cáo count:

| Distance | M1 | M4 |
|---:|---:|---:|
| 0 | | |
| 1 | | |
| 2 | | |
| 3–5 | | |
| >5 | | |

---

# Câu 13 — Vì sao mô hình sâu hơn như M5 không nhất thiết tốt hơn?

## 13.1. Bản trả lời nhanh

> **Nhiều layer tăng receptive field graph và capacity nhưng cũng tăng khó tối ưu, nguy cơ over-smoothing, overfitting và tích lũy dropout. Với `d_model` cố định, tăng head làm mỗi head hẹp hơn. Dataset nhỏ và graph local có thể không cần mô hình lớn hơn.**
>
> **Kết quả M5 chỉ cho thấy cấu hình 2L8H hiện tại kém hơn M4; chưa chứng minh nguyên nhân duy nhất là over-smoothing.**

## 13.2. Các cơ chế có thể xảy ra

### Over-smoothing

Embedding các node trở nên giống nhau:

$$ \operatorname{cos}(h_i,h_j)\uparrow $$

### Overfitting

Train loss giảm nhưng validation metric kém.

### Optimization

- gradient khó;
- learning rate chưa phù hợp;
- scheduler không giảm trong 100 epoch nếu milestone là 300/350.

### Head dimension

- M4: 64/head;
- M5: 32/head.

### Dropout

Attention dropout ở nhiều layer làm luồng message bất ổn hơn.

### Cách kiểm chứng

- train/val curves;
- node similarity theo layer;
- 1L8H và 2L4H;
- nhiều seed;
- parameter-matched baseline.

---

# Câu 14 — Có thể tách ảnh hưởng của số layer, số head, d_model và dropout bằng thí nghiệm nào?

## 14.1. Bản trả lời nhanh

> **Dùng factorial hoặc one-factor-at-a-time ablation. Giữ ba yếu tố cố định và chỉ đổi một yếu tố, chạy nhiều seed. Tối thiểu cần 1L4H, 1L8H, 2L4H, 2L8H với cùng d_model/dropout; sau đó đổi d_model và dropout riêng.**

## 14.2. Ma trận tối thiểu

| Run | Layers | Heads | d_model | Dropout |
|---|---:|---:|---:|---:|
| A | 1 | 4 | 256 | 0,2 |
| B | 1 | 8 | 256 | 0,2 |
| C | 2 | 4 | 256 | 0,2 |
| D | 2 | 8 | 256 | 0,2 |
| E | 1 | 4 | 128 | 0,2 |
| F | 1 | 4 | 512 | 0,2 |
| G | 1 | 4 | 256 | 0,0 |
| H | 1 | 4 | 256 | 0,4 |

Mỗi run ít nhất 3 seed.

### Phân tích tương tác

Factorial design cho biết layer × head có interaction hay không. Chỉ so M4 với M5 không tách được.

---

# Câu 15 — Nếu M3 chỉ tốt hơn M1 trên một test set nhưng thấp hơn ở hai test set khác, có được gọi M3 là tốt hơn không?

## 15.1. Bản trả lời nhanh

> **Không được gọi M3 tốt hơn một cách tổng quát. Chỉ được nói M3 đạt ExpRate cao hơn M1 trên CROHME 2016 với chênh lệch 0,09 điểm phần trăm, nhưng thấp hơn trên 2014, 2019 và thấp hơn trung bình.**
>
> **Với một run duy nhất, chênh lệch 0,09 còn có thể nằm trong dao động seed.**

## 15.2. Cách viết đúng

> **M3 phục hồi đáng kể so với M2 và đạt kết quả tương đương baseline; lợi ích chưa ổn định xuyên tập kiểm thử.**

Không viết:

> **M3 vượt baseline.**

### Cần nhiều seed

Nếu mean ± std chồng nhau, không thể gọi khác biệt có ý nghĩa.

---

# Câu 16 — Nếu nhiều thay đổi được đưa vào M4 cùng lúc, làm sao biết thành phần nào thật sự tạo ra cải thiện?

## 16.1. Bản trả lời nhanh

> **Không thể biết chắc từ bảng hiện tại. M4 đồng thời đổi connectivity, relative bias, số layer và số head nên có confounding. Muốn tách phải xây ablation tuần tự giữ nguyên các yếu tố còn lại.**
>
> **Đây là hạn chế thực nghiệm phải thừa nhận, không nên quy Mean Edit Distance tốt hơn riêng cho relative bias.**

## 16.2. Chuỗi ablation đề xuất

| Bước | Graph | Bias | Layers | Heads | Mục đích |
|---|---|---|---:|---:|---|
| A | 4-neighbor | Không | 2 | 8 | M3 control |
| B | 8-neighbor | Không | 2 | 8 | Connectivity |
| C | 8-neighbor | Có | 2 | 8 | Relative bias |
| D | 8-neighbor | Có | 1 | 8 | Depth |
| E | 8-neighbor | Có | 1 | 4 | Heads |

### Kết luận hiện tại phải dùng

> **Gói thiết kế M4 đạt MED tốt nhất. Thành phần đóng góp chính xác cần thí nghiệm tách biến.**

---

# Phụ lục A — Bản trả lời tổng hợp khoảng hai phút

> **M1 là baseline DenseNet–Transformer không GAT và hiện có Avg ExpRate tốt nhất 50,10%. M2 chèn GAT 2 lớp 8 head sau khi feature đã có PE; kết quả giảm còn 47,84%, cho thấy naive insertion không hiệu quả. M3 chuyển PE ra sau GAT, phục hồi lên 49,17% và giảm MED còn 2,14, nên thứ tự PE có ảnh hưởng rõ.**
>
> **M4 dùng 8-neighbor graph, relative directional bias 9 trạng thái, PE sau GAT và cấu hình gọn 1 lớp 4 head. Nó không vượt baseline về Exact Match nhưng có MED tốt nhất 2,06, nghĩa là prediction gần ground truth hơn trung bình. M5 tăng lên 2 lớp 8 head nhưng giảm mạnh còn 43,35% ExpRate, cho thấy tăng capacity không tự động tốt hơn.**
>
> **Kết luận của chuỗi ablation không phải GAT tốt hơn Transformer. Kết luận là hiệu quả của GAT phụ thuộc vào cách đưa vị trí và độ sâu; PE sau GAT tốt hơn PE trước trong thiết lập đã thử; cấu hình M4 có trade-off tốt về error severity; còn M5 là negative result. M4 thay nhiều biến cùng lúc nên cần ablation tách connectivity, bias, layer và head trước khi quy nguyên nhân.**

# Phụ lục B — Checklist bằng chứng

- [ ] Config và commit riêng cho M1–M5.
- [ ] Parameter count.
- [ ] Best epoch và checkpoint hash.
- [ ] Cùng data/preprocessing/beam.
- [ ] Ít nhất 3 seed.
- [ ] Histogram edit distance.
- [ ] Ablation tách 4/8-neighbor.
- [ ] Bias on/off.
- [ ] 1L/2L và 4H/8H riêng.
- [ ] Node similarity để kiểm tra over-smoothing.
- [ ] Linear probe vị trí để kiểm tra PE blurring.
