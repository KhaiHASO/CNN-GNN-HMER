# NHÓM 1 — BÀI TOÁN, MỤC TIÊU VÀ PHẠM VI NGHIÊN CỨU

> **Mục tiêu của nhóm:** Làm rõ luận văn giải quyết bài toán gì, đầu vào–đầu ra là gì, đóng góp nằm ở đâu, nghiên cứu đến mức nào và những điều nào không được tuyên bố vượt quá bằng chứng thực nghiệm.

---

## 0. Quy ước sử dụng tài liệu này

Tài liệu này được viết theo **repo luận văn hiện tại** `KhaiHASO/CNN-GNN-HMER`, không lấy kiến trúc Symbol Layout Graph, bounding box, YOLO hoặc các con số trong cuốn chuyên đề cũ làm mô tả mặc định cho hệ thống mới.

Mỗi câu được chuẩn bị theo bốn tầng:

1. **Bản trả lời nhanh:** dùng khi hội đồng chỉ cần câu trả lời 30–45 giây.
2. **Bản trả lời đầy đủ:** dùng khi thầy cô hỏi sâu hoặc hỏi tiếp “vì sao?”.
3. **Câu không nên nói:** tránh tuyên bố sai hoặc quá mạnh.
4. **Bằng chứng phải chỉ ra được:** file code, config, log hoặc bảng kết quả tương ứng.

---

# Câu 1 — Bài toán nghiên cứu chính xác của luận văn là gì, đầu vào và đầu ra của hệ thống là gì?

## 1.1. Bản trả lời nhanh 30–45 giây

> **Luận văn giải quyết bài toán nhận dạng biểu thức toán học viết tay từ ảnh, thuộc nhóm offline Handwritten Mathematical Expression Recognition. Đầu vào là ảnh raster một biểu thức toán học viết tay độc lập; trong code, ảnh được biểu diễn dưới dạng tensor một kênh. Đầu ra là chuỗi token LaTeX mô tả cả ký hiệu và cấu trúc của biểu thức.**
>
> **Pipeline hiện tại là: ảnh → DenseNet trích xuất feature map → chuyển các ô đặc trưng thành grid graph → GAT làm giàu ngữ cảnh không gian cục bộ → bổ sung thông tin vị trí → Transformer Decoder sinh chuỗi LaTeX bằng beam search. Mô hình không phát hiện từng ký hiệu bằng bounding box và đầu ra chính không phải graph, mà là chuỗi LaTeX.**

## 1.2. Bản trả lời đầy đủ

### a. Phát biểu bài toán bằng lời

Bài toán của luận văn là xây dựng và đánh giá một mô hình học sâu ánh xạ trực tiếp:

$$ \text{Ảnh biểu thức toán học viết tay} \longrightarrow \text{Chuỗi LaTeX} $$

Ảnh đầu vào chứa **một biểu thức toán học độc lập**, ví dụ:

- $x^2 + y^2$
- $\frac{a+b}{c}$
- $\sqrt{x+1}$
- $\int f(x)\,dx$

Mục tiêu không chỉ là nhìn đúng từng ký hiệu riêng lẻ, mà còn phải sinh đúng cách mã hóa cấu trúc:

- `x ^ { 2 }` khác với `x 2`;
- `\frac { a } { b }` khác với `a / b` theo chuẩn nhãn;
- `x _ { i }` khác với `x ^ { i }`;
- nội dung nằm trong căn, tử số hoặc mẫu số phải được đóng mở ngoặc đúng.

### b. Phát biểu bài toán bằng ký hiệu

Gọi ảnh đầu vào là:

$$ I \in \mathbb{R}^{1 \times H \times W} $$

Trong đó:

- $1$ là số kênh ảnh;
- $H$ là chiều cao;
- $W$ là chiều rộng.

Với batch, tensor trong code có dạng:

$$ I \in \mathbb{R}^{B \times 1 \times H \times W} $$

Mô hình cần dự đoán chuỗi token:

$$ Y = (y_1, y_2, \ldots, y_T) $$

trong đó mỗi $y_t$ là một token thuộc từ điển LaTeX của mô hình.

Về xác suất, mô hình học:

$$ P(Y\mid I) = \prod_{t=1}^{T} P(y_t \mid y_{ \lt t}, I) $$

Nghĩa là tại bước $t$, decoder dự đoán token tiếp theo dựa trên:

- ảnh đã được encoder mã hóa;
- các token đã sinh trước đó.

### c. Đầu vào chính xác là gì?

Đầu vào mà **model thực sự nhìn thấy** là ảnh raster, không phải dữ liệu stroke theo thời gian.

Theo code hiện tại:

- ảnh là tensor một kênh;
- ảnh đi kèm `mask` để phân biệt vùng dữ liệu thật và vùng padding;
- model xử lý cả batch ảnh có kích thước sau khi collate/padding.

Điều này khác với dữ liệu online nguyên thủy có thể chứa:

- tọa độ từng điểm bút;
- thứ tự nét;
- thời gian;
- trạng thái nhấc bút và đặt bút.

Các thông tin stroke đó không được đưa trực tiếp vào model hiện tại.

### d. Đầu ra chính xác là gì?

Đầu ra là **chuỗi token LaTeX**, sau đó được ghép thành chuỗi để:

- hiển thị;
- render bằng KaTeX/MathJax;
- so sánh với ground truth;
- tính ExpRate và edit distance.

Đầu ra không phải:

- ảnh được làm sạch;
- danh sách bounding box;
- nhãn từng ký hiệu riêng lẻ;
- Symbol Layout Graph;
- cây cú pháp được xuất trực tiếp;
- giá trị số của biểu thức.

### e. Pipeline kỹ thuật đúng với repo

Pipeline nên được nói theo đúng thứ tự sau:

```text
Ảnh raster một biểu thức
        ↓
DenseNet/CNN Encoder
        ↓
Feature map H' × W' × D
        ↓
Mỗi ô trên feature map trở thành một node
        ↓
Xây dựng grid graph cục bộ
        ↓
GAT truyền và tổng hợp thông tin giữa các node lân cận
        ↓
Bổ sung positional information
        ↓
Transformer Decoder
        ↓
Beam search
        ↓
Chuỗi token LaTeX
```

Điểm đặc biệt của hướng nghiên cứu là **GAT được chèn giữa DenseNet và Transformer Decoder**. Mục tiêu là nghiên cứu xem cách message passing và cách đặt thông tin vị trí ảnh hưởng thế nào đến khả năng nhận dạng.

### f. Mô hình có học cấu trúc hay không?

Có, nhưng cần nói chính xác.

Trong code huấn luyện hiện tại:

- có loss sinh chuỗi LaTeX;
- có thêm một thành phần `struct_loss` phụ trợ.

Tuy nhiên:

- nhiệm vụ đầu ra chính vẫn là sinh chuỗi LaTeX;
- metric chính vẫn đánh giá chuỗi;
- mô hình không xuất một Symbol Layout Graph hoàn chỉnh để người dùng sử dụng;
- không được mô tả hệ thống như một symbol detector nếu repo không có detector đó.

### g. Một câu chốt có chất lượng bảo vệ

> **Vì vậy, đây là bài toán image-to-markup end-to-end: đầu vào là ảnh raster một biểu thức và đầu ra là chuỗi LaTeX. GNN trong luận văn đóng vai trò tăng cường encoder ở mức feature-grid, chứ không thay đổi bản chất đầu ra thành bài toán phân loại node hoặc phát hiện ký hiệu.**

## 1.3. Các câu thầy có thể hỏi tiếp

### “Node của em là ký hiệu hay là pixel?”

Câu trả lời:

> **Node không phải ký hiệu đã được phân đoạn. Mỗi node là vector đặc trưng ứng với một vị trí trên feature map của DenseNet. Vì vậy cách gọi chính xác là feature-grid node hoặc feature-cell node.**

### “Graph của em có phải Symbol Layout Graph không?”

Câu trả lời:

> **Không theo nghĩa tường minh của symbol graph. Graph hiện tại là grid graph trên feature map. Nó học quan hệ cục bộ giữa các vùng đặc trưng, không sử dụng bounding box và nhãn cạnh above/below/subscript được gán thủ công.**

### “Mô hình nhận dạng ký hiệu hay nhận dạng cả biểu thức?”

Câu trả lời:

> **Mục tiêu cuối là nhận dạng toàn biểu thức. Nhận biết ký hiệu chỉ là một năng lực trung gian; kết quả chỉ được xem là đúng hoàn toàn khi chuỗi LaTeX và cấu trúc khớp ground truth.**

## 1.4. Không nên nói

- “Đầu vào là graph ký hiệu.”
- “Mỗi node là một ký hiệu toán học.”
- “Mô hình phát hiện bounding box từng ký hiệu.”
- “GAT thay thế hoàn toàn Transformer.”
- “Đầu ra là cây cú pháp.”
- “Mô hình hiểu giá trị toán học của biểu thức.”
- “Mô hình chứng minh biểu thức đúng hay sai về mặt toán học.”

## 1.5. Bằng chứng trong repo

- `README.md`: mô tả hai phiên bản DenseNet + Transformer và DenseNet + GAT + Transformer.
- `README.md`: pipeline `Image -> DenseNet -> Feature Map -> Flatten -> GAT Layers -> Reshape -> Feature Map -> Positional Encoding -> Transformer Decoder`.
- `chuyende_tamer_temp/1-cnn-gnn/tamer/lit_tamer.py`: đầu vào có dạng `[B, 1, H, W]`, output là phân bố token `[2B, L, vocab_size]`.
- `chuyende_tamer_temp/1-cnn-gnn/tamer/lit_tamer.py`: validation/test dùng beam search và chuyển chỉ số token về từ LaTeX.
- `chuyende_tamer_temp/1-cnn-gnn/config/crohme.yaml`: `beam_size: 10`, `max_len: 150`, `vocab_size: 113`.

---

# Câu 2 — Vì sao nhận dạng biểu thức toán học viết tay khó hơn OCR văn bản thông thường?

## 2.1. Bản trả lời nhanh 30–45 giây

> **OCR văn bản thông thường chủ yếu nhận dạng ký tự theo một thứ tự đọc tương đối ổn định, còn biểu thức toán học vừa có biến thiên nét viết, vừa có cấu trúc không gian hai chiều và phân cấp. Cùng một ký hiệu nhưng đặt ở bên phải, phía trên, phía dưới, trong tử số, mẫu số hoặc trong căn sẽ tạo nghĩa khác nhau.**
>
> **Do đó HMER phải giải quyết đồng thời hai bài toán: nhận biết ký hiệu và suy ra quan hệ cấu trúc để tuần tự hóa thành LaTeX. Chỉ cần sai một dấu ngoặc, một chỉ số hoặc một token thì cả biểu thức có thể bị tính sai theo Exact Match.**

## 2.2. Bản trả lời đầy đủ

### a. OCR văn bản có trật tự đọc ổn định hơn

Trong một dòng văn bản thông thường, phần lớn ký tự được đọc:

$$ \text{trái} \rightarrow \text{phải} $$

hoặc theo một quy ước dòng tương đối ổn định.

Trong biểu thức toán học, thứ tự hình học không đồng nhất với thứ tự LaTeX.

Ví dụ ảnh hiển thị:

$$ \frac{x+1}{y} $$

Nhưng chuỗi cần sinh là:

```latex
\frac { x + 1 } { y }
```

Decoder phải biết:

1. ký hiệu nào là `\frac`;
2. vùng nào thuộc tử số;
3. vùng nào thuộc mẫu số;
4. vị trí mở và đóng ngoặc;
5. thứ tự tuần tự hóa toàn bộ cấu trúc.

### b. Quan hệ không gian hai chiều quyết định ngữ nghĩa

Các quan hệ quan trọng gồm:

- bên phải;
- phía trên;
- phía dưới;
- chỉ số trên;
- chỉ số dưới;
- nằm trong căn;
- thuộc tử số;
- thuộc mẫu số;
- nằm trong ngoặc;
- cấu trúc lồng nhau.

Ví dụ:

$$ x^2,\qquad x_2,\qquad x2 $$

có thể chứa cùng hai hình dạng cơ bản `x` và `2`, nhưng ý nghĩa khác do vị trí tương đối.

### c. Cấu trúc có tính phân cấp và lồng nhau

Biểu thức có thể chứa:

$$ \frac{1}{\sqrt{x^2+1}} $$

Mô hình phải xử lý nhiều tầng:

1. phân số;
2. căn ở mẫu số;
3. số mũ nằm trong căn;
4. dấu cộng trong biểu thức con.

Một lỗi ở một tầng có thể làm chuỗi đóng mở sai toàn bộ.

### d. Ký hiệu toán học dễ nhầm về hình dạng

Một số cặp dễ nhầm:

- `1`, `l`, `|`;
- `0`, `O`;
- `x`, `\times`;
- `-`, thanh phân số;
- `c`, `(`;
- `v`, `\nu`;
- `\sum` viết xấu với ký hiệu khác;
- dấu chấm thập phân với nhiễu;
- dấu phẩy với nét nhỏ.

Trong toán học, một nét rất nhỏ có thể làm thay đổi hoàn toàn token.

### e. Kích thước ký hiệu không đồng đều

Trong cùng một ảnh:

- ký hiệu chính có thể lớn;
- chỉ số trên/dưới rất nhỏ;
- dấu chấm, dấu phẩy hoặc nét ngang mảnh;
- căn hoặc thanh phân số kéo dài qua nhiều ký hiệu.

Khi resize ảnh:

- các ký hiệu nhỏ dễ mất;
- nét mảnh dễ mờ;
- khoảng cách tương đối có thể bị nén;
- cận tích phân dễ bị nhập với vùng xung quanh.

### f. Nét viết tay có biến thiên rất lớn

Cùng một ký hiệu được viết khác nhau theo:

- người viết;
- độ nghiêng;
- độ dày;
- tốc độ;
- nét liền hoặc đứt;
- ký hiệu bị dính;
- khoảng cách không đều;
- cấu trúc lệch chuẩn.

Model phải khái quát từ mẫu train sang phong cách chưa gặp.

### g. Nhãn đầu ra có tính cú pháp

LaTeX có token điều khiển và ngoặc nhóm:

```latex
\frac { a } { b }
x ^ { 2 }
\sqrt { x + 1 }
```

Mô hình có thể nhìn đúng ký hiệu nhưng vẫn sinh sai:

- thiếu `{`;
- thừa `}`;
- đảo tử số và mẫu số;
- đặt số mũ sai;
- đóng căn sai vị trí.

### h. Metric rất khắt khe

Với ExpRate/Exact Match:

- đúng toàn bộ chuỗi → mẫu đúng;
- sai một token → mẫu sai;
- thiếu một ngoặc → mẫu sai;
- sai chỉ số trên thành chỉ số dưới → mẫu sai.

Do đó một mô hình có phần lớn token đúng vẫn có thể có ExpRate không cao.

### i. Ví dụ minh họa

Ground truth:

```latex
\int _ { 0 } ^ { 1 } x ^ { 2 } d x
```

Các dự đoán sau đều bị Exact Match tính sai:

```latex
\int x ^ { 2 } d x
```

```latex
\int _ { 0 } x ^ { 2 } d x
```

```latex
\int _ { 0 } ^ { 1 } x _ { 2 } d x
```

```latex
\int _ { 0 } ^ { 1 } x ^ { 2 d x
```

Mỗi trường hợp sai theo một cơ chế khác:

- mất cả hai cận;
- mất cận trên;
- sai số mũ thành chỉ số dưới;
- sai cú pháp ngoặc.

## 2.3. Cách trả lời có chiều sâu

> **Điểm khó nhất không nằm riêng ở nhận dạng hình dạng, mà ở sự kết hợp giữa thị giác và ngôn ngữ cấu trúc. Model phải suy ra cấu trúc 2D từ ảnh rồi chuyển cấu trúc đó thành chuỗi 1D đúng cú pháp. Vì vậy HMER là bài toán image-to-structured-sequence, không chỉ là character classification.**

## 2.4. Không nên nói

- “OCR văn bản không có cấu trúc.”
- “HMER khó chỉ vì chữ viết xấu.”
- “Chỉ cần nhận đúng tất cả ký hiệu thì biểu thức chắc chắn đúng.”
- “GNN tự động giải quyết hoàn toàn cấu trúc 2D.”
- “Mọi biểu thức toán học đều là graph ký hiệu trong code hiện tại.”

OCR hiện đại cũng có thể xử lý layout, bảng và văn bản nhiều cột. Cách nói an toàn là:

> **Trong phạm vi so sánh với OCR dòng văn bản thông thường, HMER có thêm cấu trúc 2D phân cấp và yêu cầu tuần tự hóa cú pháp.**

## 2.5. Bằng chứng cần chuẩn bị

- 3–5 ảnh cùng ký hiệu nhưng nét viết khác nhau.
- 3 cặp biểu thức có cùng ký hiệu nhưng bố trí khác nhau.
- 3 ví dụ model nhìn đúng ký hiệu nhưng sai cấu trúc LaTeX.
- Histogram lỗi theo loại: ký hiệu, chỉ số, phân số, căn, ngoặc.
- Ví dụ Exact Match sai nhưng edit distance chỉ bằng 1.

---

# Câu 3 — Luận văn tập trung vào nhận dạng ký hiệu, phân tích cấu trúc hay sinh chuỗi LaTeX end-to-end?

## 3.1. Bản trả lời nhanh 30–45 giây

> **Luận văn tập trung vào sinh chuỗi LaTeX end-to-end từ ảnh. Nhận dạng ký hiệu và biểu diễn cấu trúc là hai năng lực trung gian mà encoder–GAT–decoder phải học để hoàn thành nhiệm vụ cuối.**
>
> **Hệ thống không tách thành một symbol detector độc lập rồi mới ghép cây. DenseNet trích xuất đặc trưng thị giác, GAT lan truyền ngữ cảnh trên grid feature, và Transformer sinh chuỗi. Vì vậy cách mô tả đúng nhất là offline image-to-LaTeX end-to-end có tăng cường cấu trúc ở encoder.**

## 3.2. Phân biệt ba bài toán

### a. Nhận dạng ký hiệu riêng lẻ

Đầu vào:

- một ảnh crop của một ký hiệu.

Đầu ra:

- một nhãn như `x`, `2`, `+`, `\alpha`.

Metric:

- classification accuracy;
- precision, recall, F1 theo lớp.

Đây **không phải** nhiệm vụ đầu ra chính của repo hiện tại.

### b. Phân tích cấu trúc tường minh

Đầu vào:

- tập ký hiệu hoặc bounding box.

Đầu ra:

- quan hệ `right`, `above`, `below`, `superscript`, `subscript`;
- cây hoặc graph bố cục.

Metric:

- edge accuracy;
- relation F1;
- tree edit distance;
- structural accuracy.

Repo hiện tại không xuất trực tiếp graph ký hiệu có nhãn quan hệ để đánh giá như một module độc lập.

### c. Sinh chuỗi LaTeX end-to-end

Đầu vào:

- toàn bộ ảnh một biểu thức.

Đầu ra:

- toàn bộ chuỗi LaTeX.

Metric:

- ExpRate;
- ≤1 error;
- ≤2 errors;
- mean edit distance.

Đây là bài toán chính của luận văn.

## 3.3. “End-to-end” ở đây có nghĩa gì?

Trong phạm vi mô hình:

- ảnh được đưa vào encoder;
- đặc trưng được xử lý;
- decoder sinh token;
- loss được lan truyền ngược qua các module;
- không cần người dùng cung cấp bounding box hoặc graph ký hiệu khi inference.

Tuy nhiên, không nên hiểu “end-to-end” là:

- không có bất kỳ preprocessing nào;
- không có tokenizer;
- không có padding;
- không có beam search;
- không có hậu xử lý hiển thị.

“End-to-end” ở đây nhấn mạnh rằng model học ánh xạ trực tiếp từ ảnh biểu thức sang chuỗi token, thay vì pipeline cổ điển bắt buộc phân đoạn từng ký hiệu bằng module tách biệt.

## 3.4. Vai trò của từng thành phần

### DenseNet/CNN

Học:

- nét;
- góc;
- đường cong;
- hình dạng ký hiệu;
- texture và đặc trưng cục bộ.

### Grid graph + GAT

Học cách:

- trao đổi thông tin giữa các vùng lân cận;
- làm giàu feature bằng ngữ cảnh;
- cân trọng số vùng lân cận;
- trong M4, xét thêm bias vị trí tương đối.

### Positional encoding

Giúp decoder phân biệt:

- node ở vị trí nào;
- thứ tự không gian;
- vị trí tuyệt đối sau khi feature đã được xử lý.

### Transformer Decoder

Sinh chuỗi token theo điều kiện:

- ảnh đã mã hóa;
- token trước đó;
- cú pháp đã học từ dữ liệu.

## 3.5. `struct_loss` có làm bài toán thành structure prediction không?

Không hoàn toàn.

Trong code, `struct_loss` là một mục tiêu phụ trợ giúp huấn luyện representation. Nhưng để gọi luận văn là bài toán “phân tích cấu trúc tường minh”, cần có:

- định nghĩa đầu ra cấu trúc rõ ràng;
- annotation cấu trúc tương ứng;
- module xuất cấu trúc;
- metric đánh giá cấu trúc độc lập;
- bảng kết quả cấu trúc.

Hiện tại, sản phẩm đánh giá chính vẫn là chuỗi LaTeX. Vì vậy câu nói an toàn là:

> **Mô hình có sử dụng tín hiệu cấu trúc phụ trợ, nhưng nhiệm vụ chính và đầu ra cuối vẫn là image-to-LaTeX end-to-end.**

## 3.6. Câu trả lời khi thầy ép chọn một trong ba

> **Nếu buộc chọn một, em chọn sinh chuỗi LaTeX end-to-end. Nhận dạng ký hiệu và phân tích cấu trúc là các bài toán con ẩn bên trong quá trình học, không phải hai module đầu ra độc lập trong hệ thống hiện tại.**

## 3.7. Không nên nói

- “Luận văn giải quyết đồng thời ba bài toán với ba đầu ra.”
- “Em có module symbol detection” nếu code không có.
- “Em đánh giá structural accuracy” nếu chưa có metric tương ứng.
- “GAT dự đoán trực tiếp cạnh superscript/subscript” nếu graph hiện tại chỉ là grid graph.
- “End-to-end nghĩa là không có preprocessing.”

## 3.8. Bằng chứng trong repo

- `README.md`: DenseNet + GAT + Transformer.
- `lit_tamer.py`: model nhận ảnh và token mục tiêu, trả phân bố token.
- `lit_tamer.py`: loss chính là cross-entropy chuỗi và có thêm `struct_loss`.
- `lit_tamer.py`: test trả `preds` và `gts` dưới dạng token LaTeX.
- `README.md`: kết quả chính là ExpRate và edit distance, không phải detection F1.

---

# Câu 4 — Mục tiêu tổng quát và các mục tiêu cụ thể của luận văn là gì?

## 4.1. Bản trả lời nhanh 45–60 giây

> **Mục tiêu tổng quát là nghiên cứu và đánh giá cách tích hợp Graph Attention Network vào encoder DenseNet–Transformer cho bài toán nhận dạng ảnh biểu thức toán học viết tay sang LaTeX, đặc biệt tập trung vào cách xây dựng graph cục bộ và cách đưa thông tin vị trí vào mô hình.**
>
> **Các mục tiêu cụ thể gồm: xây dựng baseline M1; thiết kế các biến thể M2–M5; kiểm chứng ảnh hưởng của positional encoding trước và sau GAT; đánh giá relative position bias và số lớp/head; so sánh trên CROHME 2014, 2016, 2019 bằng ExpRate và edit distance; cuối cùng phân tích lỗi, điểm mạnh, điểm yếu và giới hạn của mô hình.**

## 4.2. Mục tiêu tổng quát nên viết trong luận văn

> **Nghiên cứu, xây dựng và đánh giá một kiến trúc nhận dạng biểu thức toán học viết tay dạng offline, trong đó đặc trưng ảnh từ DenseNet được mô hình hóa dưới dạng grid graph và xử lý bằng Graph Attention Network trước khi Transformer Decoder sinh chuỗi LaTeX; qua đó khảo sát tác động của cơ chế truyền tin đồ thị, thông tin vị trí tuyệt đối, vị trí tương đối và độ sâu GAT đến chất lượng nhận dạng.**

Từ khóa quan trọng:

- **nghiên cứu**;
- **xây dựng**;
- **đánh giá**;
- **khảo sát tác động**;
- **offline image-to-LaTeX**;
- **grid graph**;
- **GAT**;
- **positional information**;
- **ablation**.

Không nên đặt mục tiêu tổng quát là:

> “Chứng minh GNN tốt hơn Transformer.”

Bởi vì kết quả hiện tại không cho thấy GNN thắng baseline trên mọi metric và mọi tập.

## 4.3. Các mục tiêu cụ thể

### Mục tiêu 1 — Xây dựng đường cơ sở đáng tin cậy

- Xây dựng hoặc tái lập M1: DenseNet + Transformer Decoder.
- Dùng cùng dataset, preprocessing, tokenizer và evaluation.
- Tạo mốc để đo tác động thực sự của GAT.

Ý nghĩa:

> Không có baseline kiểm soát, không thể biết cải tiến đến từ GAT hay từ thay đổi khác.

### Mục tiêu 2 — Tích hợp GAT vào feature map

- Chuyển feature map thành các node;
- xây graph lân cận;
- áp dụng multi-head graph attention;
- đưa feature đã làm giàu vào decoder.

Câu hỏi nghiên cứu:

> GAT có giúp biểu diễn ngữ cảnh không gian tốt hơn so với feature map CNN thuần hay không?

### Mục tiêu 3 — Kiểm chứng vị trí đặt positional encoding

So sánh:

- **M2:** positional encoding trước GAT;
- **M3:** positional encoding sau GAT.

Câu hỏi nghiên cứu:

> Thông tin vị trí tuyệt đối có nên tham gia message passing hay nên được bổ sung sau khi GAT hoàn tất tổng hợp ngữ cảnh?

Kết quả hiện tại cho thấy:

- M2 thấp hơn baseline;
- M3 phục hồi đáng kể;
- điều này ủng hộ giả thuyết rằng thứ tự GAT và PE quan trọng.

Nhưng cần dùng từ:

- “cho thấy”;
- “gợi ý”;
- “phù hợp với giả thuyết”.

Không nên nói “đã chứng minh tuyệt đối cơ chế nhòe vị trí” nếu chưa đo trực tiếp embedding hoặc alignment.

### Mục tiêu 4 — Bổ sung inductive bias vị trí tương đối

M4 đưa relative position bias vào graph attention.

Câu hỏi nghiên cứu:

> Khi hai node lân cận có cùng độ tương đồng thị giác nhưng nằm ở các hướng khác nhau, việc cung cấp thông tin hướng tương đối có giúp GAT xử lý lỗi cục bộ tốt hơn không?

Kết quả:

- M4 không có ExpRate trung bình cao hơn M1;
- M4 có Mean Edit Distance tốt nhất;
- cho thấy đầu ra trung bình gần ground truth hơn.

### Mục tiêu 5 — Đánh giá ảnh hưởng của quy mô GAT

So sánh:

- M4: 1 layer, 4 head;
- M5: 2 layer, 8 head.

Câu hỏi:

> Tăng độ sâu và số head có luôn cải thiện không?

Kết quả:

- M5 giảm rõ rệt;
- negative result này chỉ ra rằng tăng năng lực mô hình không tự động dẫn đến tổng quát hóa tốt hơn.

### Mục tiêu 6 — Đánh giá đa tập kiểm thử và đa metric

Đánh giá trên:

- CROHME 2014;
- CROHME 2016;
- CROHME 2019.

Metric:

- ExpRate;
- ExpRate ≤1;
- ExpRate ≤2;
- Mean Edit Distance.

Mục đích:

- không kết luận từ một test set;
- phân biệt exact match với mức độ gần đúng;
- đánh giá độ ổn định của thay đổi kiến trúc.

### Mục tiêu 7 — Phân tích lỗi và giới hạn

Phân tích:

- lỗi token;
- lỗi chỉ số trên/dưới;
- lỗi phân số, căn, tích phân;
- ảnh ngoài phân bố train;
- tác động preprocessing;
- giới hạn của grid graph;
- giới hạn của một biểu thức mỗi lần.

## 4.4. Chuyển mục tiêu thành các câu hỏi nghiên cứu

Có thể trình bày thành bốn Research Questions:

### RQ1

> Việc chèn GAT vào giữa DenseNet và Transformer Decoder ảnh hưởng thế nào đến ExpRate và edit distance?

### RQ2

> Vị trí đặt positional encoding trước hoặc sau GAT ảnh hưởng thế nào đến khả năng nhận dạng?

### RQ3

> Relative position bias theo hướng lân cận có cải thiện mức độ gần đúng của dự đoán không?

### RQ4

> Tăng số lớp và số head của Coordinate-Aware GAT có tạo ra cải thiện ổn định không?

Cách trình bày này rất mạnh vì toàn bộ M1–M5 trở thành một chuỗi thí nghiệm có logic, thay vì năm model chạy rời rạc.

## 4.5. Mối liên hệ M1–M5 với mục tiêu

| Mô hình | Vai trò nghiên cứu | Câu hỏi được kiểm chứng |
|---|---|---|
| M1 | Baseline | Không dùng GAT thì kết quả thế nào? |
| M2 | Naive GAT, PE trước | Đưa PE vào message passing có hiệu quả không? |
| M3 | GAT, PE sau | Tách GAT khỏi PE tuyệt đối có phục hồi kết quả không? |
| M4 | Coordinate-Aware GAT | Relative spatial bias có giảm mức độ nghiêm trọng của lỗi không? |
| M5 | Scale-up | GAT sâu và nhiều head hơn có tốt hơn không? |

## 4.6. Mục tiêu không phải là gì?

Luận văn hiện tại không nên tuyên bố mục tiêu là:

- đạt SOTA;
- vượt TAMER trên mọi benchmark;
- nhận dạng mọi loại công thức;
- hiểu ngữ nghĩa toán học;
- phát hiện biểu thức trên toàn trang;
- nhận dạng đồng thời văn bản và toán học;
- xây dựng Symbol Layout Graph từ bounding box;
- giải bài toán tương đương LaTeX về mặt ngữ nghĩa;
- xử lý dữ liệu stroke online.

## 4.7. Câu chốt khi bị hỏi “mục tiêu có đạt không?”

> **Mục tiêu nghiên cứu không được định nghĩa là bắt buộc vượt baseline, mà là thiết kế và kiểm chứng có kiểm soát các cách tích hợp GAT và positional information. Kết quả đạt được cho thấy M3 phục hồi suy giảm của M2, M4 đạt edit distance tốt nhất, còn M5 cung cấp một negative result quan trọng về việc tăng độ sâu. Vì vậy mục tiêu khảo sát và rút ra quy luật thiết kế đã đạt, còn mục tiêu tăng Exact Match ổn định trên mọi tập thì chưa đạt và được ghi nhận là hạn chế.**

## 4.8. Không nên nói

- “Mục tiêu là chứng minh mô hình em tốt nhất.”
- “M4 chính xác nhất” nếu xét ExpRate trung bình.
- “M3 vượt baseline” mà không nói chỉ vượt trên CROHME 2016.
- “M5 thất bại vì over-smoothing” như một kết luận chắc chắn khi chưa đo trực tiếp.
- “GAT hiểu cấu trúc tốt hơn” mà không nói metric hoặc bằng chứng nào.

---

# Câu 5 — Phạm vi dữ liệu, loại biểu thức và điều kiện thực nghiệm được giới hạn như thế nào?

## 5.1. Bản trả lời nhanh 45–60 giây

> **Phạm vi thực nghiệm hiện tại tập trung vào CROHME, với các tập kiểm thử 2014, 2016 và 2019. Mặc dù dữ liệu CROHME gốc là online stroke, mô hình chỉ sử dụng ảnh raster một kênh của từng biểu thức, nên đây là offline recognition.**
>
> **Mỗi lần model nhận một biểu thức độc lập, không trực tiếp xử lý cả trang giấy, nhiều dòng, văn bản lẫn công thức hoặc dữ liệu stroke theo thời gian. Đầu ra bị giới hạn bởi từ điển và độ dài chuỗi của cấu hình. Các thí nghiệm M1–M5 được so sánh bằng cùng nhóm metric; cấu hình mặc định trong repo dùng một GPU, FP16, tối đa 100 epoch, batch train 8, nhưng khi báo cáo chính thức em phải dùng đúng thông số trong log của từng run nếu có override.**

## 5.2. Phạm vi dữ liệu

### a. Dataset chính

Các kết quả M1–M5 trong repo được tổng hợp trên:

- CROHME 2014;
- CROHME 2016;
- CROHME 2019.

Repo có file config cho HME100K, nhưng điều đó không tự động có nghĩa luận văn đã huấn luyện và báo cáo kết quả HME100K.

Cách nói an toàn:

> **Repo có khả năng cấu hình cho HME100K, nhưng phạm vi thực nghiệm đã có kết quả và được sử dụng trong luận văn hiện tại là CROHME.**

Không được nói:

> “Em đã đánh giá trên cả CROHME và HME100K”

nếu chưa có log, checkpoint và bảng kết quả HME100K.

### b. Dạng dữ liệu

CROHME gốc được thu thập theo dạng online handwriting, nhưng model hiện tại dùng:

- ảnh raster;
- grayscale/một kênh;
- mask;
- nhãn token LaTeX.

Vì vậy phạm vi triển khai là:

> **Offline HMER trên ảnh raster được tạo hoặc chuẩn hóa từ dữ liệu biểu thức viết tay.**

### c. Đơn vị mẫu

Một mẫu là:

- một ảnh;
- chứa một biểu thức;
- có một chuỗi LaTeX ground truth.

M4 hiện được đặc tả chỉ nhận một biểu thức mỗi lần. Nếu đầu vào là cả trang:

1. cần phát hiện vùng biểu thức;
2. crop;
3. split block nhiều dòng;
4. normalize;
5. mới gọi recognizer.

Do đó full-page detection là một pipeline ứng dụng bổ sung, không nên trộn với core recognition model.

## 5.3. Phạm vi loại biểu thức

Model học các cấu trúc xuất hiện trong từ điển và phân bố CROHME, ví dụ:

- phép toán cơ bản;
- biến và số;
- phân số;
- căn thức;
- số mũ;
- chỉ số dưới;
- tổng;
- tích phân;
- ngoặc;
- một số hàm toán học.

Nhưng không nên tuyên bố bao phủ đầy đủ:

- mọi macro LaTeX;
- mọi ký hiệu toán học;
- biểu thức hóa học;
- vật lý chuyên ngành;
- ma trận rất lớn;
- hệ phương trình nhiều dòng;
- chứng minh toán học dài;
- văn bản tiếng Việt lẫn công thức;
- ký hiệu ngoài vocabulary.

Khả năng với một loại cấu trúc phụ thuộc vào:

- số mẫu train;
- kích thước ký hiệu;
- độ dài chuỗi;
- preprocessing;
- mức độ giống với CROHME;
- việc token có trong vocabulary hay không.

## 5.4. Phạm vi đầu vào ảnh

Ảnh phù hợp nhất với model là ảnh:

- một biểu thức;
- nền và nét gần với phân bố train;
- đã crop sát;
- có padding;
- ít nhiễu;
- không bị nghiêng lớn;
- không bị cắt nét;
- kích thước hợp lý;
- ký hiệu nhỏ vẫn đủ rõ.

Ảnh ngoài phạm vi hoặc có rủi ro cao:

- ảnh nguyên trang;
- ảnh có bóng, đường kẻ, viền giấy;
- nhiều biểu thức chung một crop;
- công thức nhiều dòng;
- ảnh quá dài;
- ảnh mờ;
- nét cực mảnh hoặc cực dày;
- cận tích phân rất nhỏ;
- ký hiệu dính;
- background khác mạnh.

## 5.5. Phạm vi đầu ra

Theo config hiện tại:

- `vocab_size: 113`;
- `max_len: 150`;
- `beam_size: 10`.

Ý nghĩa:

- model chỉ sinh token thuộc vocabulary;
- chuỗi quá dài có thể bị cắt hoặc dừng trước khi hoàn tất;
- ký hiệu ngoài từ điển không thể được sinh đúng theo token chưa tồn tại;
- beam search tìm trong một số giả thuyết hữu hạn, không đảm bảo nghiệm tối ưu toàn cục.

Cần nói rõ:

> **Các con số này là cấu hình hiện được commit. Nếu các run M1–M5 được chạy bằng tham số override khác, báo cáo phải lấy thông số từ log của run, không lấy YAML làm bằng chứng duy nhất.**

## 5.6. Điều kiện huấn luyện mặc định trong repo

File `crohme.yaml` hiện thể hiện:

- seed: 7;
- deterministic: true;
- 1 GPU;
- precision 16;
- max epochs: 100;
- validation mỗi 2 epoch;
- checkpoint theo `val_ExpRate`;
- train batch size: 8;
- eval batch size: 2;
- `d_model: 256`;
- GAT 2 layer, 8 head trong cấu hình mặc định;
- dropout decoder: 0.3;
- GAT dropout: 0.2;
- beam size: 10;
- max length: 150;
- optimizer thực tế trong code: Adadelta;
- scheduler: MultiStepLR.

Nhưng đây là **cấu hình mặc định**, không nhất thiết phản ánh chính xác mọi run M1–M5. Có thể khi chạy Kaggle đã override:

- số GPU;
- epoch;
- config GAT;
- test folder;
- logger;
- checkpoint.

Do đó bảng thực nghiệm chính thức cần thêm:

| Thành phần | Nguồn sự thật |
|---|---|
| Kiến trúc | commit/code của từng M |
| Hyperparameter | config được lưu cùng run |
| Epoch thực tế | log/checkpoint |
| GPU | log môi trường |
| Dataset | command và data folder |
| Metric | script evaluation |
| Kết quả | file output của run |

## 5.7. Những việc ngoài phạm vi nghiên cứu cốt lõi

- OCR toàn trang;
- page layout analysis;
- detector biểu thức;
- detector ký hiệu;
- nhận dạng stroke online;
- semantic equivalence của hai chuỗi LaTeX;
- kiểm tra biểu thức đúng toán học;
- giải phương trình;
- chấm điểm lời giải;
- dịch công thức thành ngôn ngữ tự nhiên;
- đảm bảo thời gian thực trên thiết bị yếu;
- SOTA trên HME100K;
- hỗ trợ mọi macro LaTeX.

## 5.8. Câu trả lời khi thầy hỏi “demo ảnh trang giấy có thuộc phạm vi không?”

> **Core model không nhận trực tiếp cả trang. Ứng dụng có thể bổ sung bước phát hiện vùng biểu thức, crop và normalize, nhưng phần đó là tầng tiền xử lý ứng dụng. Phạm vi nhận dạng của M4 vẫn là một biểu thức đã được tách và chuẩn hóa trong mỗi lần suy luận.**

## 5.9. Câu trả lời khi thầy hỏi “tại sao tích phân có cận không nhận được?”

Không trả lời:

> “Dataset không có.”

Trả lời:

> **Trường hợp đó nằm trong từ vựng toán học nhưng khả năng tổng quát còn hạn chế. Em cần kiểm tra bốn yếu tố: tần suất cấu trúc tích phân có đủ cả hai cận trong train; chất lượng cận sau resize; độ lệch phân bố giữa nét demo và CROHME; và token nào bị mất trong output. Chỉ sau khi thống kê mới kết luận nguyên nhân chính.**

## 5.10. Không nên nói

- “Model nhận mọi loại biểu thức.”
- “CROHME là dataset offline nguyên gốc.”
- “Ứng dụng nhận nguyên trang nên model cũng nhận nguyên trang.”
- “Có file config HME100K nghĩa là đã thực nghiệm HME100K.”
- “Cấu hình hiện tại chắc chắn giống tất cả run đã chạy.”
- “Chỉ cần tăng GPU là kết quả sẽ tăng.”
- “Max length 150 không ảnh hưởng vì biểu thức thường ngắn.”

---

# Câu 6 — Vì sao chọn đầu ra LaTeX thay vì MathML, cây cú pháp hoặc ảnh đã chuẩn hóa?

## 6.1. Bản trả lời nhanh 45–60 giây

> **LaTeX được chọn vì nhãn của benchmark và pipeline hiện tại đã ở dạng token LaTeX; nó tương đối gọn, dễ token hóa, phù hợp với decoder sinh chuỗi, dễ render và thuận tiện cho các ứng dụng soạn thảo hoặc số hóa tài liệu. Nó cũng cho phép đánh giá trực tiếp bằng Exact Match và edit distance.**
>
> **MathML dài và nhiều thẻ hơn; cây cú pháp cần annotation cấu trúc và thêm bước chuyển sang định dạng người dùng; còn ảnh chuẩn hóa chỉ cải thiện hình ảnh chứ không tạo biểu diễn máy có thể chỉnh sửa. Tuy nhiên LaTeX không phải biểu diễn ngữ nghĩa duy nhất, nên luận văn phải chuẩn hóa nhãn và thừa nhận hai chuỗi khác nhau có thể render tương đương.**

## 6.2. Vì sao LaTeX phù hợp với bài toán hiện tại?

### a. Phù hợp với nhãn dữ liệu và benchmark

Ground truth của pipeline hiện tại được xử lý thành các token LaTeX.

Khi model sinh đúng cùng chuẩn token:

- dễ so sánh;
- dễ tính ExpRate;
- dễ tính edit distance;
- dễ đối chiếu với nghiên cứu trước.

Nếu chuyển sang định dạng khác, cần:

- bộ chuyển đổi;
- quy tắc chuẩn hóa mới;
- metric mới;
- kiểm tra lỗi chuyển đổi;
- có thể không còn so sánh công bằng với benchmark.

### b. Phù hợp với kiến trúc decoder tuần tự

Transformer Decoder được thiết kế để sinh chuỗi:

$$ y_1, y_2, \ldots, y_T $$

LaTeX là dạng chuỗi token tự nhiên cho decoder.

Ví dụ:

```latex
\frac { x + 1 } { y }
```

có thể được token hóa thành:

```text
\frac | { | x | + | 1 | } | { | y | }
```

### c. Gọn hơn MathML

Ví dụ LaTeX:

```latex
\frac{x+1}{y}
```

MathML tương ứng thường cần nhiều thẻ:

```xml
<mfrac>
  <mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow>
  <mi>y</mi>
</mfrac>
```

MathML có lợi thế về cấu trúc tường minh, nhưng:

- chuỗi dài hơn;
- vocabulary thẻ lớn hơn;
- khó giải mã hơn;
- dễ sai đóng mở tag;
- tăng độ dài sequence;
- dữ liệu huấn luyện hiện tại không trực tiếp tối ưu theo MathML.

### d. Dễ sử dụng ở tầng ứng dụng

LaTeX có thể:

- render bằng KaTeX hoặc MathJax;
- chèn vào tài liệu;
- chỉnh sửa bằng tay;
- lưu vào cơ sở dữ liệu;
- chuyển đổi sang PDF, HTML hoặc MathML;
- phục vụ số hóa đề thi và bài giảng.

### e. Dễ đọc và kiểm tra lỗi hơn

Đối với người nghiên cứu:

- dễ xem output;
- dễ so sánh token;
- dễ tìm lỗi ngoặc;
- dễ tạo error analysis;
- dễ sửa nhãn thủ công.

## 6.3. Vì sao không chọn MathML làm đầu ra chính?

MathML không phải lựa chọn sai. Nó có ưu điểm:

- cấu trúc cây tường minh;
- phù hợp web;
- có thể biểu diễn presentation hoặc content.

Nhưng đối với luận văn hiện tại:

1. dataset và code dùng LaTeX;
2. decoder đã được thiết kế cho vocabulary LaTeX;
3. benchmark phổ biến báo cáo theo chuỗi LaTeX/token;
4. MathML làm sequence dài hơn;
5. chuyển nhãn sang MathML có thể đưa thêm lỗi chuyển đổi.

Câu nói có chiều sâu:

> **LaTeX được chọn vì tính tương thích thực nghiệm và tính thực dụng, không phải vì LaTeX luôn tốt hơn MathML về biểu diễn ngữ nghĩa.**

## 6.4. Vì sao không chọn cây cú pháp làm đầu ra chính?

Cây cú pháp có ưu điểm:

- thể hiện cấu trúc 2D/quan hệ phân cấp;
- dễ kiểm tra tử, mẫu, số mũ;
- có thể giảm một số lỗi cú pháp.

Nhưng cần:

- ground truth tree chính xác;
- tokenizer/tree vocabulary;
- tree decoder;
- tree evaluation;
- bước chuyển tree sang LaTeX;
- xử lý nhiều cây tương đương.

Trong repo hiện tại, đóng góp nằm ở encoder GAT và positional design, còn output benchmark là LaTeX. Chuyển sang tree output sẽ làm thay đổi bài toán và khó tách ảnh hưởng của GAT khỏi decoder mới.

Cách nói tốt:

> **Cây cú pháp có thể là hướng phát triển hoặc mục tiêu phụ trợ, nhưng không phải định dạng đầu ra chính của thí nghiệm hiện tại.**

## 6.5. Vì sao không chọn ảnh đã chuẩn hóa làm đầu ra?

Ảnh chuẩn hóa chỉ cho:

- ảnh rõ hơn;
- nền sạch hơn;
- kích thước chuẩn hơn.

Nhưng không cung cấp trực tiếp:

- token;
- cấu trúc có thể chỉnh sửa;
- chuỗi để tìm kiếm;
- mã nguồn công thức;
- biểu diễn để chèn vào tài liệu.

Nếu đầu ra là ảnh, đó gần với bài toán:

- image enhancement;
- image-to-image translation;
- denoising;
- rendering.

Nó không giải quyết mục tiêu số hóa biểu thức thành markup máy có thể xử lý.

## 6.6. Bảng so sánh

| Đầu ra | Ưu điểm | Hạn chế trong luận văn hiện tại |
|---|---|---|
| LaTeX | Gọn, dễ token hóa, phổ biến, tương thích benchmark | Không duy nhất về cú pháp; cần normalize |
| MathML | Cấu trúc tường minh, phù hợp web | Dài, nhiều thẻ, không khớp pipeline hiện tại |
| Cây cú pháp | Phản ánh phân cấp rõ | Cần annotation, decoder và metric riêng |
| Ảnh chuẩn hóa | Dễ hiển thị, cải thiện chất lượng ảnh | Không phải markup, khó chỉnh sửa và tìm kiếm |
| Giá trị toán học/AST ngữ nghĩa | Phù hợp tính toán ký hiệu | Khó hơn nhiều, không tương ứng trực tiếp nhãn hiện có |

## 6.7. Hạn chế của LaTeX phải thừa nhận

### a. Không có biểu diễn duy nhất

Ví dụ:

```latex
x^2
```

và:

```latex
x^{2}
```

có thể render tương đương nhưng chuỗi khác nhau.

### b. Presentation không đồng nghĩa semantic

LaTeX chủ yếu mô tả trình bày. Hai công thức có thể:

- tương đương toán học;
- nhưng khác chuỗi.

Ví dụ:

$$ x+1 \quad \text{và} \quad 1+x $$

có thể tương đương đại số trong nhiều ngữ cảnh, nhưng recognition benchmark không được phép tự xem là cùng ground truth.

### c. Exact Match phụ thuộc normalization

Nếu không chuẩn hóa:

- macro tương đương có thể bị tính sai;
- khoảng trắng hoặc ngoặc có thể gây khác biệt;
- kết quả giữa nghiên cứu khó so sánh.

## 6.8. Câu trả lời khi thầy hỏi “LaTeX có biểu diễn được cấu trúc 2D không?”

> **LaTeX là chuỗi 1D nhưng các token điều khiển và ngoặc nhóm mã hóa cấu trúc 2D. Ví dụ `\frac{a}{b}` tuần tự hóa quan hệ tử–mẫu, còn `x^{2}` tuần tự hóa quan hệ số mũ. Vì vậy decoder phải học phép chuyển từ bố cục 2D trong ảnh sang cấu trúc tuần tự của LaTeX.**

## 6.9. Không nên nói

- “LaTeX là chuẩn ngữ nghĩa duy nhất.”
- “Hai chuỗi LaTeX khác nhau chắc chắn biểu diễn hai công thức khác nhau.”
- “MathML kém hơn LaTeX.”
- “Cây cú pháp không cần thiết.”
- “Ảnh chuẩn hóa không có giá trị.”
- “Exact Match trên chuỗi thô luôn phản ánh chính xác ngữ nghĩa toán học.”

---

# Câu 7 — Luận văn giải quyết bài toán online handwriting hay offline image recognition, và vì sao?

## 7.1. Bản trả lời nhanh 30–45 giây

> **Luận văn hiện tại giải quyết offline image recognition. CROHME gốc được thu thập ở dạng online stroke, nhưng pipeline của em dùng ảnh raster một kênh làm đầu vào; model không nhận thứ tự nét, thời gian hay trạng thái nhấc bút.**
>
> **Em chọn hướng offline vì phù hợp với DenseNet, GAT trên feature map và ứng dụng nhận ảnh upload hoặc ảnh crop từ trang giấy. Đổi lại, mô hình mất thông tin động của nét viết, nên đây cũng là một giới hạn so với các phương pháp khai thác stroke sequence.**

## 7.2. Phân biệt online và offline

### Online handwriting recognition

Đầu vào thường gồm chuỗi điểm:

$$ S = \{(x_t, y_t, p_t, t)\}_{t=1}^{N} $$

Trong đó có thể có:

- tọa độ;
- thời gian;
- áp lực bút;
- pen-down;
- pen-up;
- thứ tự các stroke.

Ưu điểm:

- biết thứ tự viết;
- biết nét nào thuộc cùng stroke;
- dễ phân biệt một số ký hiệu dính nhau;
- có thêm tín hiệu động.

Hạn chế:

- cần thiết bị thu thập stroke;
- không áp dụng trực tiếp cho ảnh scan hoặc ảnh chụp;
- pipeline khác với CNN xử lý ảnh.

### Offline handwriting recognition

Đầu vào chỉ là raster:

$$ I \in \mathbb{R}^{H\times W} $$

Model chỉ nhìn thấy:

- hình dạng cuối cùng;
- cường độ pixel;
- vị trí không gian.

Không biết:

- nét nào viết trước;
- người viết nhấc bút ở đâu;
- hướng di chuyển bút;
- tốc độ;
- thời gian.

## 7.3. Repo hiện tại thuộc loại nào?

Dấu hiệu xác định:

1. `forward` nhận ảnh `[B, 1, H, W]`;
2. dùng DenseNet/CNN;
3. feature map được biến thành grid graph;
4. demo và app chuẩn hóa ảnh PNG;
5. không có input stroke sequence trong inference.

Do đó:

> **Dù nguồn dữ liệu ban đầu có thể là online, bài toán được mô hình hóa và thực nghiệm trong luận văn là offline HMER.**

Đây là phân biệt rất quan trọng:

- **nguồn thu thập dữ liệu** có thể online;
- **dạng dữ liệu model sử dụng** là offline raster.

## 7.4. Vì sao chọn offline?

### a. Phù hợp kiến trúc

DenseNet làm việc trực tiếp với ảnh. GAT của luận văn được đặt trên feature map của CNN, nên đầu vào tự nhiên là raster.

### b. Phù hợp ứng dụng

Offline model có thể nhận:

- ảnh scan;
- ảnh chụp;
- ảnh upload;
- crop từ trang giấy;
- ảnh người dùng vẽ rồi raster hóa.

Ngay cả khi người dùng vẽ trực tiếp trên canvas, nếu frontend gửi ảnh PNG vào model thì đối với model đó vẫn là offline recognition.

### c. Dễ thống nhất pipeline

Tất cả nguồn đầu vào được chuyển về:

- grayscale;
- crop;
- resize;
- padding;
- tensor ảnh.

Nhờ đó không cần xây hai encoder riêng cho ảnh và stroke.

### d. Phù hợp mục tiêu nghiên cứu GAT trên feature map

Đóng góp của luận văn là:

- graph hóa feature map;
- GAT;
- positional encoding;
- relative bias.

Các thành phần này được thiết kế trên không gian ảnh, không phải graph stroke.

## 7.5. Đánh đổi khi bỏ thông tin online

Model mất:

- stroke order;
- pen trajectory;
- pen-up/pen-down;
- temporal segmentation;
- hướng viết.

Ví dụ hai ký hiệu có raster gần giống nhau có thể được phân biệt tốt hơn nếu biết thứ tự nét.

Vì vậy không nên nói offline luôn tốt hơn. Cách nói khoa học:

> **Offline mở rộng khả năng áp dụng cho ảnh thực tế, nhưng phải giải bài toán khó hơn theo nghĩa không có tín hiệu thời gian của nét viết.**

## 7.6. Vẽ trên web có biến thành online recognition không?

Không nhất thiết.

- Nếu app lưu và gửi từng stroke theo thứ tự → online.
- Nếu app chỉ xuất canvas thành PNG rồi model nhận PNG → offline.

Câu trả lời:

> **Giao diện có thể cho người dùng viết trực tiếp, nhưng cách phân loại phụ thuộc dữ liệu truyền vào model. Hệ thống hiện tại nhận bitmap cuối cùng nên vẫn là offline HMER.**

## 7.7. Hướng phát triển kết hợp

Có thể xây multimodal/hybrid model:

- nhánh ảnh raster;
- nhánh stroke sequence;
- fusion;
- decoder LaTeX.

Nhưng cần:

- dữ liệu stroke đầy đủ;
- đồng bộ ảnh–stroke;
- kiến trúc mới;
- ablation mới;
- không còn là phạm vi hiện tại.

## 7.8. Không nên nói

- “CROHME là online nên model của em là online.”
- “Người dùng vẽ trên canvas nên đây là online.”
- “Offline giữ nguyên toàn bộ thông tin stroke.”
- “Online và offline chỉ khác định dạng file.”
- “Ảnh raster luôn thực tế hơn stroke.”
- “Model có thể suy ra chính xác thứ tự nét từ ảnh.”

## 7.9. Bằng chứng trong repo

- `lit_tamer.py`: input `[B, 1, H, W]`.
- `README.md`: `Image -> DenseNet -> Feature Map`.
- `App/CROHME_M4_NEXT_PHASE_SPEC.md`: đầu ra chuẩn hóa là ảnh PNG grayscale/binary; M4 nhận ảnh một biểu thức mỗi lần.
- App spec: ảnh trang phải được crop và normalize thành `normalized_crohme.png` trước khi gọi M4.

---

# Câu 8 — Trong trường hợp kết quả chưa vượt baseline trên mọi tập kiểm thử, luận văn còn giá trị khoa học ở điểm nào?

## 8.1. Bản trả lời nhanh 45–60 giây

> **Giá trị khoa học không chỉ nằm ở việc tạo một con số cao hơn baseline, mà còn ở việc kiểm chứng có kiểm soát một giả thuyết kiến trúc và rút ra quy luật thiết kế. Chuỗi M1–M5 cho thấy positional encoding đặt trước GAT làm kết quả giảm, chuyển PE ra sau GAT giúp phục hồi, Coordinate-Aware GAT đạt Mean Edit Distance tốt nhất, còn tăng từ 1 lớp 4 head lên 2 lớp 8 head làm kết quả giảm mạnh.**
>
> **Vì vậy luận văn cung cấp cả kết quả dương và kết quả âm: GAT chưa tăng Exact Match ổn định, nhưng cách đặt vị trí và độ sâu ảnh hưởng rõ rệt; M4 giúp dự đoán gần ground truth hơn. Em sẽ trình bày đây là nghiên cứu ablation và design insight, không tuyên bố GAT vượt baseline hoặc đạt SOTA.**

## 8.2. Trước hết phải đọc kết quả trung thực

Kết quả trung bình trong repo:

| Mô hình | ExpRate | ≤1 | ≤2 | Mean Edit Distance |
|---|---:|---:|---:|---:|
| M1 — Baseline | **50.10%** | **68.68%** | **76.98%** | 2.10 |
| M2 — PE trước GAT | 47.84% | 66.12% | 75.29% | 2.21 |
| M3 — PE sau GAT | 49.17% | 67.17% | 76.40% | 2.14 |
| M4 — Coord-Aware 1L 4H | 48.98% | 67.43% | 76.61% | **2.06** |
| M5 — Coord-Aware 2L 8H | 43.35% | 62.00% | 72.27% | 2.65 |

Kết luận chính xác:

- M1 có ExpRate trung bình cao nhất.
- M4 có Mean Edit Distance trung bình thấp nhất.
- M3 vượt M1 một lượng nhỏ trên CROHME 2016, nhưng không vượt ở 2014 và 2019.
- M5 giảm trên tất cả các chỉ số tổng hợp.
- Không có bằng chứng để nói GAT thắng baseline trên mọi phương diện.
- Có bằng chứng để nói thiết kế GAT và positional information ảnh hưởng đáng kể.

## 8.3. Giá trị khoa học thứ nhất — Ablation có kiểm soát

Nếu M1–M5 chỉ khác nhau ở các yếu tố được xác định rõ, luận văn trả lời được:

- GAT có tác động gì?
- PE đặt trước hay sau khác nhau thế nào?
- Relative bias có tác động gì?
- Tăng depth/head có hiệu quả không?

Một nghiên cứu tốt không bắt buộc mọi biến thể đều tăng điểm. Điều quan trọng là:

1. giả thuyết rõ;
2. thiết kế thí nghiệm công bằng;
3. metric đúng;
4. phân tích không bịa;
5. kết luận phù hợp dữ liệu.

## 8.4. Giá trị khoa học thứ hai — Negative result

M2 và M5 là các negative result có giá trị.

### M2

Quan sát:

- PE trước GAT thấp hơn baseline và thấp hơn M3.

Giả thuyết giải thích:

- message passing trộn feature của các node lân cận;
- thành phần vị trí tuyệt đối có thể bị làm mượt;
- decoder nhận alignment kém rõ hơn.

Nhưng cần nói:

> **Kết quả phù hợp với giả thuyết position blurring, chưa phải chứng minh trực tiếp.**

Muốn chứng minh mạnh hơn cần:

- đo cosine similarity của positional component trước/sau GAT;
- visualize attention alignment;
- so sánh độ phân biệt tọa độ;
- chạy nhiều seed.

### M5

Quan sát:

- tăng layer/head làm kết quả giảm mạnh.

Các giả thuyết:

- overfitting;
- over-smoothing;
- optimization khó hơn;
- dropout nhiều hơn;
- propagation trên graph thưa;
- relative bias bị biến đổi qua nhiều lớp.

Không được chọn một nguyên nhân làm kết luận duy nhất khi chưa tách thí nghiệm.

Giá trị:

> **Cho thấy không thể tăng quy mô GAT một cách máy móc trên feature-grid nhỏ và dataset hạn chế.**

## 8.5. Giá trị khoa học thứ ba — Phân biệt Exact Match và mức độ gần đúng

M4:

- ExpRate thấp hơn M1;
- Mean Edit Distance tốt hơn M1.

Điều này cho thấy:

- nhiều dự đoán của M4 có thể ít lỗi token hơn;
- nhưng chưa đạt khoảng cách 0 nên Exact Match vẫn không tăng.

Ví dụ:

Ground truth:

```latex
\frac { x + 1 } { y }
```

M1 có thể dự đoán sai 4 token.

M4 có thể chỉ thiếu một dấu `}`.

Với ExpRate:

- cả hai đều bằng 0 cho mẫu đó.

Với edit distance:

- M4 được ghi nhận là gần đúng hơn.

Vì vậy M4 có giá trị như một thiết kế giảm mức độ nghiêm trọng của lỗi, dù chưa tăng số mẫu hoàn toàn chính xác.

## 8.6. Giá trị khoa học thứ tư — Quy luật thiết kế

Chuỗi thí nghiệm gợi ý ba quy luật:

### Quy luật 1

> **Không nên cho positional encoding tuyệt đối tham gia message passing một cách ngây thơ.**

### Quy luật 2

> **Relative spatial bias có thể giúp đầu ra gần ground truth hơn ngay cả khi Exact Match chưa tăng.**

### Quy luật 3

> **GAT sâu hơn không mặc định tốt hơn trên grid graph cục bộ và dataset nhỏ.**

Đây là design insight mà người khác có thể sử dụng khi thiết kế CNN–GNN hybrid.

## 8.7. Giá trị khoa học thứ năm — Tính minh bạch và tái lập

Repo có:

- source code;
- config;
- cấu trúc M1–M5;
- run ID;
- kết quả từng test set;
- evaluation;
- error JSON/predictions.

Nếu được hoàn thiện tốt, nghiên cứu cho phép người khác:

- kiểm tra;
- tái chạy;
- bác bỏ hoặc xác nhận;
- xây biến thể mới.

Một kết quả âm nhưng tái lập được có giá trị hơn một kết quả cao không truy xuất được.

## 8.8. Cách định vị đóng góp để không bị phản biện

### Không định vị như sau

> “Em đề xuất GAT và GAT tốt hơn baseline.”

### Nên định vị

> **Em nghiên cứu cách tích hợp GAT và thông tin vị trí vào feature-grid của mô hình HMER. Kết quả cho thấy hiệu quả phụ thuộc mạnh vào thứ tự PE, relative bias và độ sâu. M4 chưa tăng Exact Match trung bình nhưng đạt edit distance tốt nhất, còn M5 cho thấy giới hạn của scale-up. Đóng góp chính là thiết kế, ablation và các kết luận thực nghiệm, không phải SOTA.**

## 8.9. Khi nào nghiên cứu sẽ bị xem là yếu?

Nếu chỉ có:

- một lần chạy;
- không có baseline công bằng;
- không cùng preprocessing;
- không cùng decoder;
- không có log;
- không biết checkpoint;
- không biết metric;
- giải thích nguyên nhân hoàn toàn bằng suy đoán;
- chọn metric có lợi và bỏ metric bất lợi.

Do đó để giá trị khoa học vững hơn, cần bổ sung:

1. ít nhất 3 seed cho M1, M3, M4;
2. mean ± standard deviation;
3. parameter count;
4. inference time;
5. histogram edit distance;
6. error analysis theo cấu trúc;
7. visualization attention/feature;
8. thống kê độ dài;
9. test tích phân có cận;
10. ablation từng thành phần của M4.

## 8.10. Các câu thầy có thể hỏi tiếp

### “Không thắng baseline thì sao gọi là mô hình đề xuất?”

> **Mô hình đề xuất là thiết kế được nghiên cứu, không đồng nghĩa với SOTA. Em trình bày rõ M4 là cấu hình đề xuất về coordinate-aware bias và tính gọn, nhưng không gọi nó là mô hình chính xác nhất theo ExpRate.**

### “Có phải em chọn edit distance vì ExpRate không thắng không?”

> **Không. Edit distance đã là metric bổ sung hợp lý vì Exact Match không phản ánh độ lớn của lỗi. Tuy nhiên em vẫn báo cáo đầy đủ ExpRate và thừa nhận baseline cao hơn. Giá trị của M4 là trade-off, không phải thay thế metric có lợi.**

### “M4 có bảo toàn cấu trúc tốt hơn không?”

Câu trả lời an toàn:

> **Mean Edit Distance thấp hơn cho thấy chuỗi token gần ground truth hơn trung bình. Để khẳng định riêng khả năng bảo toàn cấu trúc, em cần thêm metric theo loại cấu trúc hoặc parse tree; edit distance một mình chưa đủ chứng minh mọi lỗi cấu trúc đã giảm.**

### “PE blurring có thật không?”

> **Ablation M2–M3 ủng hộ giả thuyết đó, nhưng chưa đo trực tiếp. Em xem đây là lời giải thích cơ chế hợp lý cần được củng cố bằng phân tích embedding và attention alignment.**

## 8.11. Không nên nói

- “M4 tốt nhất” mà không nêu metric.
- “Edit distance quan trọng hơn ExpRate trong mọi trường hợp.”
- “M2 chứng minh chắc chắn PE bị nhòe.”
- “M5 thất bại chắc chắn do over-smoothing.”
- “Chỉ cần thêm GPU là GAT sẽ thắng baseline.”
- “Kết quả âm cũng là đóng góp” nhưng không có phân tích hoặc bài học cụ thể.
- “Mô hình em tốt hơn về cấu trúc” khi chưa có metric cấu trúc trực tiếp.
- “Baseline không quan trọng vì không có GNN.”

## 8.12. Bản trả lời đầy đủ khoảng 2 phút

> **Kết quả hiện tại không cho phép em nói GAT vượt baseline trên mọi tập và mọi metric. M1 vẫn có ExpRate trung bình cao nhất là 50,10%, còn M4 đạt 48,98%. Tuy nhiên M4 có Mean Edit Distance thấp nhất là 2,06 so với 2,10 của baseline, nghĩa là đầu ra của M4 trung bình gần ground truth hơn dù nhiều mẫu vẫn còn một hoặc vài lỗi nên chưa được tính Exact Match.**
>
> **Giá trị khoa học của luận văn nằm ở chuỗi ablation. M2 cho thấy đặt PE trước GAT làm kết quả giảm; M3 chuyển PE ra sau và phục hồi; M4 thêm relative position bias và đạt edit distance tốt nhất với kiến trúc gọn; M5 tăng lên hai lớp tám head nhưng giảm mạnh. Từ đó nghiên cứu rút ra rằng hiệu quả của GAT phụ thuộc vào cách xử lý vị trí và độ sâu, chứ không phải cứ thêm GAT hoặc tăng tham số là tốt hơn.**
>
> **Em sẽ trình bày các cơ chế như position blurring, over-smoothing hoặc dropout trên graph thưa dưới dạng giả thuyết phù hợp với kết quả, không khẳng định là nguyên nhân duy nhất nếu chưa có phép đo trực tiếp. Vì vậy đóng góp là thiết kế, thực nghiệm kiểm soát, kết quả âm và hướng dẫn thiết kế; còn việc chưa tăng Exact Match ổn định là hạn chế cần được thừa nhận.**

---

# 9. Bảng tóm tắt tám câu để học thuộc

| Câu | Ý chính phải nói |
|---|---|
| 1 | Offline image-to-LaTeX; ảnh một biểu thức → DenseNet → feature-grid graph → GAT → PE → Transformer → LaTeX |
| 2 | Khó vì vừa nhận ký hiệu, vừa hiểu bố cục 2D phân cấp, vừa sinh chuỗi đúng cú pháp |
| 3 | Nhiệm vụ chính là end-to-end LaTeX generation; ký hiệu và cấu trúc là năng lực trung gian |
| 4 | Nghiên cứu cách tích hợp GAT + positional information qua M1–M5, không đặt mục tiêu bắt buộc thắng baseline |
| 5 | CROHME, raster một biểu thức, test 2014/2016/2019; không phải full-page hoặc stroke model |
| 6 | LaTeX phù hợp nhãn, benchmark, decoder và ứng dụng; nhưng cần normalize và không phải semantic canonical |
| 7 | Model là offline vì nhận bitmap; nguồn CROHME online không làm model trở thành online |
| 8 | Giá trị ở ablation, design insight, best edit distance và negative result; không tuyên bố SOTA |

---

# 10. Một đoạn mở đầu hoàn chỉnh cho phần bảo vệ

> **Luận văn của em giải quyết bài toán nhận dạng biểu thức toán học viết tay ở dạng ảnh offline. Đầu vào là ảnh raster một biểu thức độc lập, đầu ra là chuỗi LaTeX. Khác với cuốn chuyên đề trước đây mô tả hướng symbol-level graph, hệ thống hiện tại không phát hiện từng ký hiệu bằng bounding box. DenseNet tạo feature map, mỗi ô đặc trưng được xem là một node trong grid graph, GAT làm giàu ngữ cảnh không gian cục bộ, sau đó positional information được bổ sung và Transformer Decoder sinh chuỗi bằng beam search.**
>
> **Trọng tâm nghiên cứu không phải chứng minh GNN luôn tốt hơn, mà là khảo sát cách tích hợp GAT và thông tin vị trí. Em xây chuỗi M1–M5 để kiểm chứng PE trước/sau GAT, relative position bias và độ sâu mô hình trên CROHME 2014, 2016, 2019. Kết quả cho thấy baseline vẫn có Exact Match trung bình cao nhất, nhưng M4 đạt Mean Edit Distance tốt nhất; đồng thời M2 và M5 cung cấp các negative result về vị trí PE và việc scale-up GAT. Vì vậy đóng góp chính là thiết kế và phân tích thực nghiệm có kiểm soát, kèm giới hạn được trình bày rõ ràng.**

---

# 11. Danh sách bằng chứng cần hoàn thiện trước khi dùng câu trả lời này tại hội đồng

## Bắt buộc

- [ ] Xác nhận số lượng chính xác train/validation/test từ data loader hoặc script thống kê.
- [ ] Xác nhận run M1–M5 sử dụng seed nào.
- [ ] Xác nhận số GPU và số epoch thực tế từng run.
- [ ] Xác nhận config có bị override khi chạy Kaggle hay không.
- [ ] Tạo bảng M1–M5 từ code, không chỉ dựa vào README.
- [ ] Xác nhận graph 4 hướng ở M2/M3 và 8 hướng ở M4/M5.
- [ ] Xác nhận vocabulary 113 áp dụng cho tất cả run.
- [ ] Xác nhận cách normalize LaTeX trước khi tính ExpRate.
- [ ] Xác nhận edit distance tính trên token hay ký tự.
- [ ] Xác nhận test dùng beam size 10 cho tất cả mô hình.
- [ ] Lưu command line đánh giá 2014/2016/2019.
- [ ] Có file kết quả gốc cho từng bảng số liệu.

## Nên có để nhắm 9–9,5

- [ ] Mean ± standard deviation trên nhiều seed.
- [ ] Parameter count của M1–M5.
- [ ] Thời gian train và inference.
- [ ] Histogram edit distance.
- [ ] Error analysis theo độ dài.
- [ ] Error analysis theo phân số, căn, mũ, chỉ số, tích phân.
- [ ] Thống kê riêng tích phân có cận.
- [ ] Hình ảnh trước/sau preprocessing.
- [ ] Visualization attention hoặc node feature.
- [ ] Ablation riêng relative bias, số layer và số head.

---

# 12. Nguồn đối chiếu trong repo

1. `README.md`
   - Kiến trúc baseline và CNN–GNN.
   - Pipeline DenseNet → feature map → GAT → positional encoding → Transformer.
   - Mô tả M1–M5.
   - Bảng kết quả CROHME 2014, 2016, 2019.

2. `chuyende_tamer_temp/1-cnn-gnn/config/crohme.yaml`
   - Seed, GPU, FP16, epoch.
   - Kích thước mô hình.
   - Vocabulary.
   - Cấu hình GAT.
   - Beam search, max length.
   - Batch size và data folder.

3. `chuyende_tamer_temp/1-cnn-gnn/tamer/lit_tamer.py`
   - Shape ảnh đầu vào.
   - Loss chuỗi và `struct_loss`.
   - Validation/test bằng beam search.
   - Chuyển token prediction và ground truth.
   - Edit distance trong output analysis.
   - Optimizer Adadelta và scheduler.

4. `App/CROHME_M4_NEXT_PHASE_SPEC.md`
   - M4 nhận một biểu thức mỗi lần.
   - Ảnh được chuẩn hóa thành bitmap grayscale/binary.
   - Full-page cần detection, crop, split và normalization trước recognizer.

---

# 13. Cảnh báo cuối cùng khi trình bày

Ba ranh giới phải giữ tuyệt đối:

1. **Không gọi feature-grid graph là Symbol Layout Graph.**
2. **Không gọi M4 là mô hình có ExpRate tốt nhất.**
3. **Không gọi các giả thuyết như PE blurring hoặc over-smoothing là nguyên nhân đã được chứng minh tuyệt đối khi chưa có phép đo trực tiếp.**

Câu nói khoa học an toàn nhất là:

> **Kết quả thực nghiệm quan sát được là X; cơ chế Y là giả thuyết phù hợp với quan sát; để xác nhận Y cần thí nghiệm Z.**
