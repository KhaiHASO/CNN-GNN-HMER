# NHÓM 2 — KHOẢNG TRỐNG NGHIÊN CỨU VÀ ĐÓNG GÓP MỚI

> **Mục tiêu nhóm:** Chứng minh đề tài không chỉ là ghép các module có sẵn, mà xuất phát từ một vấn đề nghiên cứu cụ thể, có giả thuyết rõ ràng, có chuỗi thí nghiệm kiểm chứng và có kết luận phù hợp với bằng chứng.

---

## 0. Nguyên tắc trả lời của Nhóm 2

Nhóm này là phần dễ bị hội đồng hỏi khó nhất, bởi chỉ cần nói quá mạnh một câu như:

> “Em là người đầu tiên dùng GNN cho HMER.”

hoặc:

> “M4 chứng minh GAT tốt hơn Transformer.”

thì hội đồng có thể bác ngay bằng các công trình graph-based HMER đã tồn tại hoặc bằng chính bảng kết quả M1–M5.

Vì vậy, toàn bộ phần trả lời cần giữ ba lớp phân biệt:

### Lớp 1 — Điều đã tồn tại trong cộng đồng nghiên cứu

- CNN, Transformer, GNN và GAT đều không phải thuật toán do luận văn phát minh.
- Graph-based HMER đã có các hướng stroke graph, symbol graph, graph-to-graph, node–edge classification và link prediction.
- Positional encoding và relative position bias cũng đã là các khái niệm phổ biến trong Transformer, graph transformer và vision attention.

### Lớp 2 — Điều luận văn thật sự thiết kế

- Chuyển feature map của DenseNet thành một **feature-grid graph**.
- Chèn GAT vào giữa CNN encoder và Transformer decoder.
- Khảo sát vị trí đặt positional encoding trước hoặc sau message passing.
- Bổ sung relative directional bias gồm 9 trạng thái trên lưới 8 hướng.
- Khảo sát ảnh hưởng của số lớp và số head bằng chuỗi M1–M5.
- Phân tích đồng thời Exact Match và Mean Edit Distance.

### Lớp 3 — Điều luận văn mới chỉ quan sát hoặc giả thuyết

- “PE blurring” là lời giải thích phù hợp với chênh lệch M2–M3, nhưng chưa phải cơ chế đã được đo trực tiếp.
- “Over-smoothing”, “đứt luồng do dropout” hoặc “méo relative bias” là các giả thuyết giải thích M5, chưa được phép xem là nguyên nhân duy nhất.
- Mean Edit Distance thấp hơn không tự động chứng minh mọi cấu trúc toán học được bảo toàn tốt hơn.
- M4 chưa vượt baseline M1 về ExpRate trung bình.

Một mẫu câu khoa học nên dùng xuyên suốt là:

> **Kết quả quan sát được là X. Cơ chế Y là giả thuyết phù hợp với kết quả đó. Để khẳng định Y, cần bổ sung thí nghiệm Z.**

---

# Câu 1 — Khoảng trống nghiên cứu mà luận văn muốn giải quyết là gì?

## 1.1. Bản trả lời nhanh 45–60 giây

> **Khoảng trống của luận văn không phải là “chưa ai dùng GNN cho HMER”, vì đã có các phương pháp graph-based ở mức stroke hoặc symbol. Khoảng trống cụ thể hơn là cách tích hợp message passing đồ thị vào feature map của một hệ image-to-LaTeX mà vẫn bảo toàn thông tin vị trí cần cho decoder.**
>
> **Luận văn tập trung vào ba câu hỏi chưa được giải quyết trong chính pipeline này: feature-grid GAT có giúp gì so với DenseNet–Transformer baseline; positional encoding nên đặt trước hay sau GAT; và relative directional bias cùng độ sâu GAT ảnh hưởng thế nào đến Exact Match và edit distance. Chuỗi M1–M5 được xây để kiểm chứng ba câu hỏi đó.**

## 1.2. Trước hết: không được phát biểu khoảng trống quá rộng

Một phát biểu yếu và dễ bị bác:

> “Các mô hình hiện tại chỉ xử lý chuỗi nên không hiểu cấu trúc 2D; chưa có ai dùng graph cho HMER.”

Phát biểu này không chính xác vì:

1. Các mô hình CNN–attention và Transformer có thể học quan hệ 2D từ feature map.
2. TAMER đã bổ sung nhiệm vụ dự đoán cấu trúc cây để cải thiện tính hợp lệ của LaTeX.
3. Các công trình graph-based HMER đã mô hình hóa stroke hoặc symbol thành node và quan hệ không gian thành edge.
4. Các phương pháp syntax-aware, tree-aware và graph-based đều đã cố gắng đưa inductive bias cấu trúc vào HMER.

Do đó, khoảng trống không nên được mô tả là sự vắng mặt hoàn toàn của cấu trúc hoặc GNN.

## 1.3. Khoảng trống hợp lý ở ba tầng

### Tầng A — Khoảng trống biểu diễn

Baseline DenseNet–Transformer tạo feature map:

$$ F \in \mathbb{R}^{H' \times W' \times D} $$

Các vùng trên feature map chứa đặc trưng thị giác cục bộ. CNN đã có receptive field, nhưng phép tổng hợp chủ yếu do kernel tích chập cố định theo vị trí tương đối.

Câu hỏi còn mở trong pipeline của luận văn là:

> Có thể xem mỗi ô đặc trưng là một node và cho phép node học trọng số tổng hợp khác nhau đối với các láng giềng hay không?

Đây là động cơ của feature-grid GAT:

$$ h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j $$

Trong đó:

- $h_i$ là feature tại node $i$;
- $\mathcal{N}(i)$ là các node lân cận trên lưới;
- $\alpha_{ij}$ là trọng số học được;
- $W$ là phép biến đổi tuyến tính.

Điểm nghiên cứu không phải là công thức GAT mới, mà là **vai trò của GAT trong encoder HMER cụ thể này**.

### Tầng B — Khoảng trống về vị trí

HMER rất nhạy với vị trí:

- phía trên hay phía dưới;
- chỉ số trên hay chỉ số dưới;
- tử số hay mẫu số;
- nằm trong hay ngoài căn.

Nếu positional encoding tuyệt đối được cộng vào feature trước message passing, GAT sẽ tổng hợp đồng thời:

- thành phần thị giác;
- thành phần vị trí.

Điều đó đặt ra câu hỏi:

> Positional encoding có nên tham gia vào quá trình message passing hay chỉ nên được bổ sung sau khi các feature thị giác đã trao đổi ngữ cảnh?

M2 và M3 được xây để trả lời trực tiếp câu hỏi này:

- M2: PE trước GAT;
- M3: PE sau GAT.

Đây là khoảng trống thiết kế quan trọng hơn câu nói chung chung “GNN có tốt hay không”.

### Tầng C — Khoảng trống về hướng tương đối và quy mô GAT

GAT tiêu chuẩn học trọng số từ feature của hai node, nhưng nếu không cung cấp vị trí tương đối, hai láng giềng có feature gần giống có thể khó phân biệt về hướng.

Trong biểu thức toán học, hướng mang ý nghĩa:

- trên;
- dưới;
- trái;
- phải;
- bốn đường chéo;
- chính node đó.

M4 bổ sung một bias học được theo 9 trạng thái:

$$ r_{ij} \in \{0,1,\ldots,8\} $$

và:

$$ e_{ij}^{(k)} = e_{ij,\text{content}}^{(k)} + b_{k,r_{ij}} $$

Trong đó:

- $k$ là attention head;
- $b_{k,r_{ij}}$ là bias vị trí tương đối học được cho từng head.

Sau đó M5 kiểm tra câu hỏi:

> Nếu 1 lớp, 4 head có hiệu quả nhất định, tăng lên 2 lớp, 8 head có cải thiện hay không?

Kết quả M5 cho thấy “nhiều hơn” không mặc định “tốt hơn”.

## 1.4. Phát biểu khoảng trống nên dùng trong luận văn

### Phiên bản ngắn

> **Luận văn tập trung vào khoảng trống thiết kế khi tích hợp GAT trên feature-grid của mô hình image-to-LaTeX: cách xây dựng quan hệ lân cận, thứ tự giữa message passing và positional encoding, cơ chế mã hóa hướng tương đối, và độ sâu thích hợp của GAT chưa được kiểm chứng có hệ thống trong pipeline DenseNet–Transformer đang sử dụng.**

### Phiên bản đầy đủ

> **Mặc dù các mô hình CNN–Transformer đạt hiệu quả cao trong HMER và các phương pháp graph-based đã được nghiên cứu ở mức stroke hoặc symbol, việc sử dụng graph attention trực tiếp trên các ô của feature map trong một pipeline sinh LaTeX đặt ra vấn đề riêng: message passing cần làm giàu ngữ cảnh cục bộ mà không làm suy giảm độ phân biệt vị trí tuyệt đối cần cho decoder. Bên cạnh đó, GAT dựa trên nội dung có thể thiếu thông tin hướng tương đối, trong khi tăng độ sâu có thể gây khó tối ưu trên graph lưới. Luận văn xây dựng chuỗi M1–M5 để khảo sát có kiểm soát các yếu tố này.**

## 1.5. Những điều luận văn không được gọi là khoảng trống

- “Chưa ai dùng CNN cho HMER.”
- “Transformer không xử lý được ảnh 2D.”
- “Chưa ai dùng GNN/GAT trong nhận dạng biểu thức toán học.”
- “Chưa ai dùng relative position bias.”
- “Chưa có mô hình nào học cấu trúc cây.”
- “Các mô hình trước chỉ nhận ký hiệu mà không nhận cấu trúc.”
- “GNN là giải pháp duy nhất cho HMER.”

## 1.6. Bằng chứng phải chuẩn bị

- Bảng related work chia thành:
  - sequence-based;
  - syntax/tree-aware;
  - stroke/symbol graph-based;
  - feature-grid graph-based.
- Một sơ đồ chỉ rõ khoảng trống của luận văn nằm **giữa DenseNet feature map và Transformer Decoder**.
- Bảng M1–M5 gắn với từng câu hỏi nghiên cứu.
- Code chứng minh:
  - feature-grid node;
  - adjacency 8 hướng;
  - self-loop;
  - residual GAT;
  - PE sau GAT;
  - 9-state relative bias.
- Một câu tuyên bố novelty đã được thu hẹp, tránh “first-ever”.

## 1.7. Câu hỏi phụ có thể gặp

### “Graph-based HMER đã có rồi, đề tài em còn khoảng trống gì?”

> **Các công trình đó thường xây graph ở mức stroke hoặc symbol và giải bài toán node–edge classification hoặc graph structure recognition. Công trình của em nghiên cứu GAT ở mức feature-grid trong một recognizer image-to-LaTeX, đồng thời tập trung vào thứ tự PE–GAT và bias hướng tương đối. Vì vậy đối tượng graph, vị trí tích hợp và câu hỏi thực nghiệm khác nhau.**

### “Khoảng trống này có đủ lớn cho luận văn không?”

> **Đối với luận văn thạc sĩ, khoảng trống không cần là phát minh một họ thuật toán hoàn toàn mới. Nó cần là một câu hỏi nghiên cứu có ý nghĩa, có thiết kế kỹ thuật, baseline công bằng, thí nghiệm kiểm soát và kết luận tái lập được. Chuỗi M1–M5 đáp ứng cấu trúc đó, nhưng em cần bổ sung thêm ablation tách từng yếu tố của M4 để kết luận mạnh hơn.**

---

# Câu 2 — Vì sao các mô hình CNN–Transformer hoặc sequence-to-sequence vẫn có hạn chế với cấu trúc 2D?

## 2.1. Bản trả lời nhanh 45–60 giây

> **CNN–Transformer không phải là không xử lý được cấu trúc 2D; thực tế chúng là baseline mạnh. Hạn chế nằm ở inductive bias. CNN tổng hợp lân cận bằng kernel cố định, còn decoder phải tuần tự hóa bố cục 2D thành chuỗi LaTeX. Khi gặp số mũ, chỉ số, phân số hoặc cấu trúc lồng nhau, một sai lệch attention nhỏ có thể sinh sai ngoặc hoặc sai quan hệ.**
>
> **Self-attention có thể học quan hệ toàn cục, nhưng nó phụ thuộc mạnh vào positional information và dữ liệu để tự học ý nghĩa của hướng. Luận văn thử bổ sung một graph lân cận và relative directional bias để đưa giả định không gian cục bộ vào encoder, chứ không phủ nhận năng lực của CNN–Transformer.**

## 2.2. Cần trả lời công bằng với baseline

Không nên nói:

> “CNN chỉ nhìn cục bộ nên không hiểu cấu trúc.”

DenseNet sâu có receptive field lớn. Transformer Decoder có cross-attention tới toàn bộ feature map. Các mô hình loại này đã đạt kết quả mạnh trên CROHME.

Phát biểu chính xác hơn:

> **CNN–Transformer có thể học cấu trúc 2D, nhưng phải học phần lớn quan hệ đó gián tiếp từ dữ liệu và positional encoding. Luận văn nghiên cứu liệu một inductive bias đồ thị cục bộ có giúp representation rõ hơn hay không.**

## 2.3. Hạn chế của CNN trong ngữ cảnh này

### a. Kernel chia sẻ cố định theo vị trí tương đối

Một phép tích chập có dạng:

$$ F'_{i,j} = \sum_{\Delta i,\Delta j} K_{\Delta i,\Delta j} F_{i+\Delta i,j+\Delta j} $$

Trọng số kernel $K_{\Delta i,\Delta j}$:

- được chia sẻ trên toàn ảnh;
- không thay đổi theo nội dung của từng cặp node trong một mẫu.

Trong khi đó, GAT cho phép:

$$ \alpha_{pq} = f(h_p,h_q) $$

nghĩa là cùng hai vị trí lân cận nhưng trọng số có thể khác tùy nội dung.

Ví dụ:

- một nét nhỏ phía trên có thể là số mũ;
- một nét nhỏ tương tự ở chỗ khác có thể chỉ là nhiễu.

GAT có khả năng học trọng số thích nghi theo feature, dù khả năng đó không đảm bảo tự động tạo ra hiểu biết toán học.

### b. Receptive field lý thuyết khác receptive field hiệu dụng

CNN sâu có receptive field lớn về lý thuyết, nhưng đóng góp thực tế của vùng xa có thể không đồng đều.

Với cấu trúc như:

$$ \sqrt{\frac{x^2+1}{y_1+y_2}} $$

decoder cần representation có thông tin:

- ký hiệu nào nằm trong căn;
- phân số nằm trong căn;
- mũ và chỉ số nằm trong nhánh con nào.

Feature cục bộ mạnh chưa chắc tạo alignment tốt cho mọi cấu trúc lồng nhau.

### c. Downsampling có thể làm yếu ký hiệu nhỏ

DenseNet và transition layer làm giảm kích thước không gian. Điều này có lợi cho chi phí tính toán, nhưng các chi tiết nhỏ như:

- dấu chấm;
- số mũ;
- chỉ số dưới;
- cận tích phân;
- dấu âm ngắn

có thể bị suy giảm.

GAT không tự khôi phục thông tin đã mất, nhưng có thể giúp feature còn lại trao đổi ngữ cảnh với vùng lân cận.

## 2.4. Hạn chế của sequence-to-sequence decoder

### a. Chuyển cấu trúc 2D thành chuỗi 1D

Ảnh biểu diễn quan hệ đồng thời, nhưng decoder sinh tuần tự:

$$ P(Y\mid I) = \prod_{t=1}^{T} P(y_t\mid y_{ \lt t},I) $$

Một cấu trúc như:

```latex
\frac { a ^ { 2 } } { b _ { i } }
```

đòi hỏi decoder ghi nhớ:

- đang ở tử số hay mẫu số;
- đã mở bao nhiêu ngoặc;
- nhánh nào đã hoàn tất;
- token hiện tại thuộc cấu trúc nào.

### b. Lỗi tích lũy trong autoregressive decoding

Khi inference, token dự đoán trước trở thành điều kiện cho token sau.

Nếu model sinh sai:

```latex
\frac { a
```

thành:

```latex
\sqrt { a
```

thì toàn bộ phần tiếp theo có thể đi theo cấu trúc sai.

### c. Exact Match trừng phạt mạnh lỗi cấu trúc nhỏ

Dự đoán:

```latex
x ^ { 2 }
```

so với:

```latex
x _ { 2 }
```

chỉ khác một token nhưng khác ý nghĩa.

Dự đoán thiếu một dấu `}` cũng bị ExpRate tính sai toàn mẫu.

## 2.5. Hạn chế của self-attention nếu không có positional bias phù hợp

Self-attention cơ bản dựa vào nội dung:

$$ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^\top}{\sqrt{d}} \right)V $$

Nếu không có positional information, attention không tự biết:

- node nào ở trên;
- node nào ở dưới;
- hai node cách nhau bao xa;
- node nào là đường chéo.

Do đó positional encoding hoặc relative bias là cần thiết.

Tuy nhiên, khi positional encoding tuyệt đối đi qua message passing, nó có thể bị trộn với feature láng giềng. Đây là câu hỏi M2–M3 kiểm tra.

## 2.6. Vì sao graph có thể hữu ích?

Graph đưa ra ba inductive bias:

### a. Topology bias

Chỉ cho node tương tác với láng giềng được xác định.

Trong M4:

- trên;
- dưới;
- trái;
- phải;
- bốn đường chéo;
- self-loop.

### b. Adaptive aggregation

Các láng giềng có trọng số khác nhau qua $\alpha_{ij}$.

### c. Directional bias

M4 cộng bias theo hướng tương đối vào attention logits.

Nhưng cần lưu ý kỹ:

> **Graph của luận văn là feature-grid graph, không phải graph quan hệ toán học tường minh. Nó không trực tiếp biết cạnh nào là superscript hoặc denominator.**

## 2.7. Một điểm kỹ thuật cần nói trung thực

Mặc dù graph topology là thưa, implementation hiện tại:

- tạo adjacency dạng $N \times N$;
- tính attention score dạng $N \times N$;
- sau đó mask các cặp không có cạnh.

Vì vậy không nên nói:

> “GAT hiện tại chắc chắn tiết kiệm hơn global self-attention nhờ graph thưa.”

Về mặt logic, số cạnh hợp lệ là thưa. Nhưng về bộ nhớ và phép tính của code hiện tại, tensor attention vẫn có dạng toàn cặp, gần chi phí bậc hai theo số node:

$$ O(N^2) $$

Đây là một hạn chế kỹ thuật có thể tối ưu bằng sparse edge list hoặc thư viện graph chuyên dụng.

## 2.8. Câu kết luận chuẩn

> **Vì vậy, luận văn không xuất phát từ giả định CNN–Transformer không thể học 2D. Khoảng hạn chế là mô hình phải tự học nhiều quan hệ hình học từ dữ liệu, trong khi HMER rất nhạy với hướng và phân cấp. GAT được dùng như một inductive bias cục bộ thích nghi, còn M2–M5 kiểm tra cách đưa vị trí vào inductive bias đó.**

## 2.9. Không nên nói

- “CNN chỉ nhìn một pixel.”
- “Transformer không có khả năng xử lý ảnh.”
- “Sequence model chắc chắn sai cấu trúc phức tạp.”
- “GAT nhìn được toàn graph trong một lớp.”
- “Graph hiện tại biểu diễn trực tiếp tử số, mẫu số, số mũ.”
- “GAT có chi phí tuyến tính trong implementation hiện tại.”
- “Self-attention không dùng thông tin vị trí.”

---

# Câu 3 — Đóng góp mới cụ thể của luận văn nằm ở kiến trúc, cách xây dựng graph, thông tin vị trí hay quy trình đánh giá?

## 3.1. Bản trả lời nhanh 60 giây

> **Đóng góp không nằm ở một yếu tố duy nhất mà ở một gói thiết kế có thứ bậc. Đóng góp kiến trúc là chèn GAT có residual vào giữa DenseNet và Transformer trên feature-grid. Đóng góp biểu diễn graph là lưới 8 hướng có self-loop và mask padding. Đóng góp vị trí là tách absolute PE ra sau GAT, đồng thời thêm relative directional bias 9 trạng thái vào từng attention head.**
>
> **Đóng góp thực nghiệm là chuỗi M1–M5 kiểm chứng từng quyết định trên ba tập CROHME và nhiều metric. Tuy nhiên, em không gọi việc dùng GAT hoặc relative bias nói chung là mới trên thế giới. Tính mới an toàn nằm ở cách kết hợp, câu hỏi ablation và bằng chứng trong pipeline cụ thể này.**

## 3.2. Phân cấp đóng góp

### Đóng góp A — Kiến trúc lai ở mức feature-grid

Pipeline:

```text
Image
→ DenseNet
→ projected feature map
→ flatten thành grid nodes
→ GAT
→ residual connection
→ LayerNorm
→ 2D positional encoding
→ Transformer Decoder
→ LaTeX
```

Trong code hiện tại:

```python
feature_flat = feature_flat + self.gat(feature_flat, adj, h, w)
```

Đây là residual connection:

$$ H_{\text{out}} = H_{\text{in}} + \operatorname{GAT}(H_{\text{in}},A) $$

Ý nghĩa:

- giữ lại feature gốc;
- cho GAT học phần hiệu chỉnh;
- giảm nguy cơ representation bị thay thế hoàn toàn bởi message passing.

Đây là đóng góp kiến trúc của hệ thống, không phải phát minh residual hoặc GAT.

### Đóng góp B — Xây dựng feature-grid graph

Với feature map kích thước:

$$ H' \times W' \times D $$

số node là:

$$ N = H'W' $$

Mỗi node tương ứng với một ô feature, không phải ký hiệu.

Adjacency hiện tại gồm:

- ngang;
- dọc;
- hai đường chéo;
- cạnh hai chiều;
- self-loop;
- loại bỏ kết nối tới padding.

Điểm đáng trình bày:

- graph được tạo động theo shape feature;
- padding mask được phản ánh vào adjacency;
- self-loop ngăn node mất hoàn toàn thông tin và tránh hàng softmax rỗng.

### Đóng góp C — Thứ tự absolute positional encoding

M2 và M3 thay đổi thứ tự:

#### M2

```text
Visual feature
→ Absolute PE
→ GAT
→ Decoder
```

#### M3/M4/M5

```text
Visual feature
→ GAT
→ Absolute PE
→ Decoder
```

Giả thuyết:

- GAT nên tổng hợp feature thị giác thuần;
- PE tuyệt đối được thêm sau như tọa độ rõ ràng cho decoder.

Kết quả:

- M2 trung bình 47,84% ExpRate;
- M3 trung bình 49,17%;
- tăng 1,33 điểm phần trăm;
- M3 tiệm cận M1 50,10%;
- M3 cao hơn M1 trên CROHME 2016 một lượng nhỏ.

Cách kết luận an toàn:

> **Thứ tự PE–GAT ảnh hưởng rõ ràng đến kết quả; PE sau GAT tốt hơn PE trước GAT trong thiết lập đã thử.**

Không nên kết luận:

> “Đã chứng minh về toán học PE bị triệt tiêu.”

### Đóng góp D — Relative directional bias 9 trạng thái

M4 mã hóa:

$$ \Delta x,\Delta y \in \{-1,0,1\} $$

tạo 9 trạng thái:

$$ r = 3(\Delta y+1)+(\Delta x+1) $$

Mỗi head có bảng bias học được:

$$ B \in \mathbb{R}^{K \times 9} $$

Attention logit:

$$ e_{ij}^{(k)} = e_{ij,\text{content}}^{(k)} + B_{k,r_{ij}} $$

Điểm đáng chú ý:

- bias khác nhau theo head;
- hướng được thêm trực tiếp vào logits;
- không cần nối tọa độ vào node feature;
- self-relation cũng có bias riêng.

Tuy nhiên:

- relative bias là khái niệm đã có trong attention;
- 9-state direction encoding cũng không nên tuyên bố first-ever nếu chưa có literature review toàn diện;
- tính mới nên gắn với **feature-grid GAT cho HMER và chuỗi kiểm chứng cụ thể**.

### Đóng góp E — Kết quả âm khi scale-up

M5 tăng từ:

- 1 layer, 4 head;

lên:

- 2 layer, 8 head.

Kết quả trung bình:

- M4 ExpRate 48,98%;
- M5 ExpRate 43,35%;
- M4 Mean Edit Distance 2,06;
- M5 Mean Edit Distance 2,65.

Giá trị:

- bác bỏ giả định “nhiều lớp/head hơn luôn tốt hơn”;
- cho thấy cấu hình gọn có thể phù hợp hơn;
- tạo hướng nghiên cứu về depth, dropout và graph propagation.

### Đóng góp F — Quy trình đánh giá đa tập, đa metric

Đánh giá trên:

- CROHME 2014;
- CROHME 2016;
- CROHME 2019.

Metric:

- ExpRate;
- $\leq 1$ error;
- $\leq 2$ errors;
- Mean Edit Distance.

Ý nghĩa:

- không chọn một test set có lợi;
- phân biệt đúng tuyệt đối với gần đúng;
- phát hiện M4 có edit distance tốt dù Exact Match chưa thắng.

## 3.3. Đóng góp nào mạnh nhất?

Xếp theo mức thuyết phục hiện tại:

### Mạnh nhất

1. Thiết kế thí nghiệm PE trước/sau GAT.
2. Relative directional bias trong feature-grid GAT.
3. Chuỗi M1–M5 và negative result M5.
4. M4 đạt Mean Edit Distance tốt nhất với cấu hình gọn.

### Trung bình

- 8-connected adjacency;
- residual connection;
- mask padding;
- tích hợp app và preprocessing.

Những yếu tố này có giá trị kỹ thuật nhưng bản thân không đủ để tuyên bố novelty lớn.

### Không nên xem là đóng góp mới

- dùng DenseNet;
- dùng Transformer;
- dùng GAT cơ bản;
- dùng beam search;
- dùng CROHME;
- dùng ExpRate;
- dùng edit distance.

## 3.4. Hạn chế của gói đóng góp hiện tại

M4 khác M3 ở nhiều yếu tố cùng lúc:

- adjacency có thể thay từ 4 hướng sang 8 hướng;
- thêm relative bias;
- giảm layer từ 2 xuống 1;
- giảm head từ 8 xuống 4.

Do đó từ M3 → M4, chưa thể khẳng định cải thiện edit distance đến riêng từ relative bias.

Cần ablation bổ sung:

| Thí nghiệm | Connectivity | Relative bias | Layer | Head |
|---|---:|---:|---:|---:|
| A | 4 | Không | 2 | 8 |
| B | 8 | Không | 2 | 8 |
| C | 8 | Có | 2 | 8 |
| D | 8 | Có | 1 | 8 |
| E | 8 | Có | 1 | 4 |

Nhờ vậy mới tách được:

- ảnh hưởng connectivity;
- ảnh hưởng relative bias;
- ảnh hưởng depth;
- ảnh hưởng head count.

## 3.5. Câu trả lời khi hội đồng hỏi “rốt cuộc mới ở đâu?”

> **Nếu tách rõ, em không nhận GAT là thuật toán mới. Phần mới của luận văn là thiết kế tích hợp GAT trên feature-grid của DenseNet cho image-to-LaTeX, phát hiện thực nghiệm về thứ tự PE trước/sau message passing, cơ chế bias hướng 9 trạng thái trong attention và chuỗi ablation về độ sâu. Đây là novelty ở mức kiến trúc ứng dụng và kết quả thực nghiệm, không phải một họ GNN mới hoàn toàn.**

## 3.6. Không nên nói

- “Em phát minh GAT.”
- “Graph 8 hướng là hoàn toàn mới.”
- “Relative position bias chưa từng xuất hiện.”
- “M4 chứng minh bảo toàn cấu trúc tốt nhất” nếu chỉ dựa edit distance.
- “M4 là tốt nhất mọi mặt.”
- “M3–M4 chỉ khác relative bias.”
- “M1–M5 là ablation hoàn hảo” khi M4 thay nhiều biến cùng lúc.

---

# Câu 4 — Điểm khác biệt giữa công trình hiện tại và cuốn chuyên đề cũ là gì?

## 4.1. Bản trả lời nhanh 45–60 giây

> **Cuốn chuyên đề cũ là định hướng symbol-level graph: phát hiện từng ký hiệu bằng bounding box, mỗi ký hiệu là node và cạnh mang nhãn như above, below, right hoặc subscript. Công trình hiện tại không triển khai pipeline đó.**
>
> **Repo mới dùng DenseNet tạo feature map, mỗi ô feature là một node, graph nối các ô lân cận và GAT làm giàu feature trước Transformer Decoder. Vì vậy không có symbol detector, không có bounding box node và không có edge label ngữ nghĩa tường minh. Luận văn mới cũng thay trọng tâm từ “xây Symbol Layout Graph” sang “nghiên cứu vị trí PE, relative bias và độ sâu GAT qua M1–M5”.**

## 4.2. Bảng đối chiếu

| Thành phần | Chuyên đề cũ | Công trình hiện tại |
|---|---|---|
| Vai trò tài liệu | Định hướng ban đầu | Nguồn sự thật của luận văn |
| Đơn vị node | Một ký hiệu | Một ô trên feature map |
| Cách có node | Symbol detection/bounding box | Flatten feature map |
| Cạnh | Quan hệ ngữ nghĩa như above/below/right/subscript | Láng giềng hình học trên grid |
| Edge label | Có ý tưởng nhãn quan hệ | Không có edge class ngữ nghĩa |
| Detector | YOLO/Faster R-CNN hoặc module phát hiện | Không có symbol detector trong core recognizer |
| Graph | Symbol Layout Graph | Feature-grid graph |
| Encoder thị giác | Detector + local symbol features | DenseNet + projection |
| GNN | Học quan hệ giữa ký hiệu | Làm giàu ngữ cảnh giữa feature cells |
| Decoder | Transformer sinh LaTeX | Transformer sinh LaTeX |
| Thí nghiệm | Baseline/GCN/GAT theo mô tả cũ | M1–M5 theo PE, relative bias, depth/head |
| Metric chính | Có mô tả thêm Symbol Acc, Syntax Error, F1 | Repo tập trung ExpRate và edit distance |
| Bằng chứng | Báo cáo định hướng, có chỗ chưa khớp code | Code, run ID, checkpoint, output |

## 4.3. Sự khác nhau về bản chất khoa học

### Chuyên đề cũ hỏi

> Có thể mô hình hóa biểu thức thành graph ký hiệu và dùng GNN để học quan hệ không gian hay không?

### Luận văn hiện tại hỏi

> Khi feature map của CNN được xem là graph lưới, GAT nên được tích hợp và cung cấp thông tin vị trí như thế nào để hỗ trợ Transformer sinh LaTeX?

Hai câu hỏi có liên quan nhưng không giống nhau.

## 4.4. Vì sao thay đổi hướng là hợp lý?

### a. Symbol-level graph cần nhiều module phụ

Muốn triển khai đúng chuyên đề cũ cần:

- symbol segmentation;
- symbol detection;
- bounding box;
- symbol classification;
- spatial relation labels;
- relation classification;
- graph construction;
- xử lý lỗi tích lũy;
- metric detection và relation.

Đó là một pipeline lớn và phụ thuộc annotation.

### b. Feature-grid graph giữ được end-to-end recognition

Repo hiện tại:

- không cần bounding box tại inference;
- không cần nhãn edge ngữ nghĩa;
- cho loss LaTeX lan truyền về CNN và GAT;
- dễ so sánh với baseline DenseNet–Transformer.

### c. Phù hợp tài nguyên và thời gian luận văn

Feature-grid graph tận dụng trực tiếp:

- backbone có sẵn;
- decoder có sẵn;
- dataset LaTeX có sẵn.

Nhờ vậy trọng tâm được thu hẹp vào câu hỏi GAT và positional information.

## 4.5. Điều gì từ chuyên đề cũ vẫn còn giá trị?

Không phải bỏ toàn bộ tài liệu cũ.

Có thể giữ:

- tổng quan HMER;
- khó khăn của cấu trúc 2D;
- lý thuyết CNN, GNN, GAT;
- động cơ dùng graph;
- tổng quan CROHME;
- hướng phát triển symbol-level graph trong tương lai.

Nhưng phải cập nhật hoặc loại bỏ:

- sơ đồ YOLO → symbol graph nếu không triển khai;
- phát biểu “mỗi node là ký hiệu”;
- metric detector F1;
- structure loss nếu không đúng code;
- con số không truy được;
- cấu hình GPU/epoch không đúng run mới;
- kết quả cũ không thuộc M1–M5.

## 4.6. Cách trình bày sự chuyển hướng trước hội đồng

> **Chuyên đề là giai đoạn hình thành ý tưởng graph-based HMER. Khi triển khai luận văn, em đánh giá symbol-level pipeline đòi hỏi detector và annotation quan hệ riêng, làm khó việc huấn luyện end-to-end và kiểm soát sai số. Vì vậy em thu hẹp thành feature-grid graph: giữ DenseNet và Transformer, chèn GAT vào giữa, sau đó thiết kế M1–M5 để nghiên cứu riêng vai trò message passing và vị trí. Đây là sự cụ thể hóa và điều chỉnh phạm vi, không phải lấy nguyên chuyên đề làm kết quả luận văn.**

## 4.7. Nếu hội đồng hỏi “vậy chuyên đề cũ có sai không?”

> **Chuyên đề cũ là đề xuất hướng và có một số mô tả chưa phải implementation cuối cùng. Trong luận văn, em sẽ phân biệt rõ phần định hướng với phần đã triển khai, đồng thời chỉ báo cáo những gì truy được tới code, log và script đánh giá.**

Không nên nói:

- “Chuyên đề cũ chỉ viết cho có.”
- “Em bỏ hoàn toàn chuyên đề.”
- “Hai kiến trúc giống nhau vì đều dùng GNN.”
- “Node là ký hiệu hoặc feature cell đều như nhau.”

## 4.8. Bằng chứng cần đưa vào slide

Một slide “Từ chuyên đề đến luận văn”:

```text
Ý tưởng ban đầu:
Image → Symbol Detection → Symbol Graph → GNN → LaTeX

Implementation luận văn:
Image → DenseNet → Feature-Grid Graph → GAT → PE → Transformer → LaTeX
```

Dưới slide ghi:

- bỏ detector;
- đổi node;
- đổi edge;
- giữ mục tiêu image-to-LaTeX;
- thêm M1–M5 ablation.

---

# Câu 5 — Vì sao lựa chọn GNN/GAT thay vì chỉ tăng độ sâu CNN hoặc dùng thêm self-attention?

## 5.1. Bản trả lời nhanh 60 giây

> **Em chọn GAT vì muốn kiểm tra một inductive bias khác, không đơn thuần tăng capacity. CNN sâu hơn vẫn tổng hợp bằng kernel cố định, còn GAT học trọng số khác nhau cho từng láng giềng theo nội dung. So với global self-attention, graph lân cận cho phép giới hạn tương tác theo topology và dễ đưa bias hướng tương đối vào từng cạnh.**
>
> **Tuy nhiên em không cho rằng GAT luôn vượt CNN hoặc self-attention. Kết quả M1–M5 cho thấy baseline vẫn mạnh và GAT rất nhạy với PE, depth và dropout. Vì vậy việc chọn GAT là một giả thuyết nghiên cứu về adaptive local message passing, không phải kết luận trước rằng GAT tốt nhất.**

## 5.2. Ba hướng thay đổi khác nhau

### Hướng 1 — Tăng độ sâu CNN

Tác dụng chính:

- tăng receptive field;
- tăng số phép biến đổi phi tuyến;
- tăng capacity.

Nhưng vẫn dựa trên:

- kernel chia sẻ;
- pattern cố định theo offset;
- không có attention weight riêng cho từng cặp node trong từng ảnh.

### Hướng 2 — Thêm self-attention

Tác dụng:

- cho phép tương tác toàn cục;
- học quan hệ theo nội dung;
- phù hợp feature sequence.

Nhưng:

- cần positional encoding tốt;
- dễ có chi phí bậc hai;
- không mặc định có topology cục bộ;
- trên dataset nhỏ có thể học quan hệ không cần thiết;
- khó tách riêng ảnh hưởng của neighborhood structure.

### Hướng 3 — GAT trên grid graph

Tác dụng:

- dùng topology cục bộ được xác định;
- học trọng số theo nội dung;
- thêm edge/directional bias thuận tiện;
- kiểm soát số hop bằng số layer;
- phù hợp mục tiêu nghiên cứu “local relational refinement”.

## 5.3. So sánh toán học trực giác

### CNN

$$ h_i' = \sum_{\delta \in \mathcal{K}} w_\delta h_{i+\delta} $$

Trọng số $w_\delta$ phụ thuộc offset, nhưng được dùng chung cho mọi mẫu.

### GAT

$$ h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j $$

Trong đó $\alpha_{ij}$ phụ thuộc nội dung node.

### Coordinate-Aware GAT

$$ \alpha_{ij}^{(k)} = \operatorname{softmax}_j \left[ \operatorname{LeakyReLU} \left( a^\top[Wh_i \Vert Wh_j] + b_{k,r_{ij}} \right) \right] $$

Nó kết hợp:

- tương đồng nội dung;
- topology;
- hướng tương đối;
- attention head.

## 5.4. Vì sao không chỉ tăng CNN?

### a. Không trả lời đúng câu hỏi nghiên cứu

Luận văn muốn kiểm tra message passing và position handling. Tăng CNN chỉ kiểm tra capacity/receptive field.

### b. Khó giải thích quan hệ node–neighbor

GAT cho phép quan sát attention weights, ít nhất về mặt kỹ thuật có thể phân tích node nào được ưu tiên.

### c. Thêm depth có thể tăng tham số nhưng không thêm inductive bias mong muốn

Mục tiêu không phải chỉ tăng điểm, mà kiểm tra một cơ chế quan hệ.

Tuy vậy, để kết luận công bằng, nên có thêm baseline:

- deeper CNN với parameter count tương đương;
- một local self-attention block;
- một convolution 3×3 tương đương tham số.

Nếu không có các baseline này, câu trả lời chỉ là lý do lựa chọn, chưa phải bằng chứng GAT ưu việt hơn.

## 5.5. Vì sao không dùng global self-attention?

### a. Mục tiêu ưu tiên cục bộ

Feature-grid graph chỉ nối các vùng gần nhau. Một layer GAT là một bước refinement cục bộ.

### b. Directional bias rõ ràng

9-state bias gắn trực tiếp vào cặp node có quan hệ hình học.

### c. Dễ kiểm soát receptive field theo hop

- 1 layer: 1-hop;
- 2 layer: tối đa 2-hop về mặt truyền thông tin.

Điều này hỗ trợ ablation depth.

### d. Nhưng cần trung thực về implementation

Code hiện tại vẫn xây score $N \times N$ rồi mask, nên chưa tận dụng sparse computation.

Vì vậy lợi thế “graph thưa tiết kiệm tính toán” mới là lợi thế tiềm năng về thiết kế, chưa phải lợi thế đã đạt trong implementation.

## 5.6. Vì sao GAT thay vì GCN?

GCN thường tổng hợp với trọng số chuẩn hóa theo adjacency:

$$ H' = \sigma(\hat{D}^{-\frac12}\hat{A}\hat{D}^{-\frac12}HW) $$

Các node láng giềng được trộn theo quy tắc tương đối cố định.

GAT cho phép:

- trọng số theo nội dung;
- khác nhau theo head;
- chèn relative bias;
- thích nghi với ký hiệu và nét viết.

Nhưng GAT cũng:

- nhiều tham số hơn;
- dễ bất ổn;
- nhạy dropout;
- có nguy cơ attention không thực sự biểu diễn “tầm quan trọng” theo nghĩa giải thích;
- có chi phí cao hơn.

## 5.7. Câu trả lời cân bằng

> **GAT được chọn vì nó là công cụ phù hợp để kiểm tra adaptive local aggregation và relative directional bias. Em không chọn vì tin rằng GAT mặc định tốt hơn. Chính M2 và M5 cho thấy nếu tích hợp không đúng, GAT có thể làm kết quả giảm.**

## 5.8. Những baseline nên bổ sung

Để hội đồng khó bác “chỉ do thêm tham số”, nên có:

1. M1 + convolution block có parameter count gần M4.
2. M1 + local self-attention không relative bias.
3. M1 + local self-attention có relative bias.
4. M1 + GCN.
5. M1 + GAT không bias.
6. M1 + Coordinate-Aware GAT.

Nếu tài nguyên không đủ, ít nhất báo cáo parameter count và thừa nhận chưa so sánh đầy đủ.

## 5.9. Không nên nói

- “CNN không có khả năng học quan hệ.”
- “Self-attention không dùng topology.”
- “GAT luôn ít tham số hơn.”
- “GAT hiện tại nhanh hơn self-attention.”
- “Attention weight luôn giải thích được quyết định.”
- “M4 tốt vì GAT, không liên quan giảm layer/head.”
- “M5 thấp chứng minh CNN tốt hơn GNN.”

---

# Câu 6 — Đóng góp nào là đóng góp phương pháp, đóng góp thực nghiệm và đóng góp kỹ thuật triển khai?

## 6.1. Bản trả lời nhanh 60 giây

> **Đóng góp phương pháp gồm feature-grid GAT, residual integration, PE sau message passing và relative directional bias 9 trạng thái. Đóng góp thực nghiệm là chuỗi M1–M5 trên ba tập CROHME với Exact Match, ≤1, ≤2 và Mean Edit Distance, qua đó thu được cả kết quả dương lẫn âm.**
>
> **Đóng góp kỹ thuật triển khai gồm code cấu hình hóa GAT, mask padding, beam-search evaluation, lưu run/checkpoint và pipeline ứng dụng chuẩn hóa ảnh trước M4. Ba loại đóng góp phải tách rõ: app tốt không tự tạo novelty phương pháp, còn một ý tưởng phương pháp chỉ có giá trị khi có thực nghiệm kiểm chứng.**

## 6.2. Đóng góp phương pháp

### P1 — Feature-grid graph representation

- node là feature cell;
- graph tạo từ kích thước feature;
- không cần symbol annotation.

### P2 — GAT refinement có residual

$$ H' = \operatorname{LayerNorm} \left( H + \operatorname{GAT}(H,A) \right) $$

### P3 — Absolute PE sau GAT

$$ Z = \operatorname{LayerNorm} \left( H' + PE_{2D} \right) $$

### P4 — Relative directional bias

$$ e_{ij}^{(k)} \leftarrow e_{ij}^{(k)} + b_{k,r_{ij}} $$

### P5 — Cấu hình gọn M4

- 1 GAT layer;
- 4 heads;
- 8-connected grid;
- relative bias;
- PE sau GAT.

Lưu ý:

- M4 có thể được gọi là “cấu hình đề xuất chính”;
- không nên gọi “mô hình tốt nhất tuyệt đối” vì M1 có ExpRate cao hơn.

## 6.3. Đóng góp thực nghiệm

### E1 — Baseline công bằng

M1 giúp xác định:

> Nếu không GAT, hệ thống đạt bao nhiêu?

### E2 — So sánh thứ tự PE

M2 vs M3:

- cùng ý tưởng GAT 2L, 8H;
- khác vị trí PE;
- kết quả M3 cao hơn M2.

Đây là cặp ablation sạch nhất nếu mọi cấu hình khác thật sự giống nhau.

### E3 — Coordinate-aware design

M4 kiểm tra:

- relative bias;
- adjacency 8 hướng;
- model gọn.

### E4 — Scale-up negative result

M5 kiểm tra:

- tăng layer/head;
- kết quả giảm.

### E5 — Đánh giá nhiều tập

- 2014;
- 2016;
- 2019.

### E6 — Đánh giá nhiều mức lỗi

- exact match;
- gần đúng tối đa 1;
- gần đúng tối đa 2;
- edit distance trung bình.

### E7 — Phát hiện trade-off

- M1 tốt hơn về Exact Match trung bình;
- M4 tốt hơn về Mean Edit Distance trung bình.

## 6.4. Đóng góp kỹ thuật triển khai

### T1 — Module hóa cấu hình

Các tham số GAT:

- bật/tắt;
- số layer;
- số head;
- hidden dimension;
- dropout.

### T2 — Dynamic adjacency

Graph được xây theo:

- $H'$;
- $W'$;
- mask batch;
- padding.

### T3 — Self-loop và padding mask

- đảm bảo mỗi node có kết nối;
- tránh softmax toàn $-\infty$;
- không truyền thông tin giữa vùng padding và vùng hợp lệ.

### T4 — Relative bias theo từng head

Bảng bias có shape:

$$ [K,9] $$

### T5 — Quy trình đánh giá tái lập

- script train;
- script eval;
- run ID;
- thư mục kết quả;
- predictions và edit-distance analysis.

### T6 — Pipeline ứng dụng M4-ready

Ứng dụng chuẩn hóa:

- crop;
- khử nền;
- deskew;
- binarize;
- padding;
- resize giữ tỷ lệ;
- nền đen, nét trắng;
- chỉ đưa một biểu thức mỗi lần vào M4.

Đây là đóng góp engineering, không nên trộn với novelty thuật toán.

## 6.5. Bảng phân loại

| Hạng mục | Loại đóng góp | Bằng chứng |
|---|---|---|
| Grid graph trên feature map | Phương pháp | `encoder.py` |
| 8 hướng + self-loop | Phương pháp/kỹ thuật | `_build_grid_adjacency` |
| PE sau GAT | Phương pháp | `encoder.py`, M2–M3 |
| 9-state relative bias | Phương pháp | `gat.py` |
| Residual GAT | Phương pháp/kỹ thuật | `encoder.py` |
| M1–M5 | Thực nghiệm | README, run folders |
| 3 test sets | Thực nghiệm | bảng kết quả |
| ExpRate + edit distance | Thực nghiệm | eval output |
| App normalization | Kỹ thuật triển khai | app spec |
| API/adapters | Kỹ thuật triển khai | app spec |

## 6.6. Cách viết phần “Đóng góp của luận văn”

> **Thứ nhất, luận văn xây dựng một encoder lai DenseNet–GAT, trong đó feature map được biểu diễn thành lưới graph 8 hướng và được cập nhật bằng multi-head graph attention có residual.**
>
> **Thứ hai, luận văn khảo sát và đề xuất thứ tự xử lý GAT trước, positional encoding tuyệt đối sau, đồng thời bổ sung relative directional bias 9 trạng thái vào attention logits.**
>
> **Thứ ba, luận văn thực hiện chuỗi ablation M1–M5 trên CROHME 2014, 2016 và 2019, qua đó phân tích trade-off giữa Exact Match và edit distance, cũng như giới hạn khi tăng độ sâu GAT.**
>
> **Thứ tư, luận văn hiện thực hóa pipeline đánh giá và ứng dụng chuẩn hóa ảnh phù hợp với đầu vào M4, hỗ trợ kiểm thử và tái lập.**

## 6.7. Không nên trộn ba loại đóng góp

Ví dụ sai:

> “Em có một app đẹp nên mô hình có tính mới.”

App là engineering.

Ví dụ sai:

> “Em có relative bias nên chắc chắn hiệu quả.”

Cần thí nghiệm.

Ví dụ sai:

> “M4 edit distance thấp nên thuật toán mới đã được chứng minh.”

M4 thay nhiều yếu tố; cần ablation tách biến.

---

# Câu 7 — Ablation M1–M5 giúp kiểm chứng giả thuyết nghiên cứu nào?

## 7.1. Bản trả lời nhanh 60–90 giây

> **M1 là control baseline. M2 kiểm tra việc thêm GAT theo cách ngây thơ khi PE đã nằm trong feature. M3 giữ GAT nhưng chuyển PE ra sau để kiểm tra giả thuyết rằng message passing không nên trộn positional encoding tuyệt đối. M4 kiểm tra việc thêm bias hướng tương đối và dùng cấu hình GAT gọn. M5 kiểm tra giả thuyết scale-up: tăng layer và head có giúp hay không.**
>
> **Kết quả cho thấy M2 giảm so với M1; M3 phục hồi so với M2; M4 có Mean Edit Distance tốt nhất nhưng Exact Match trung bình vẫn dưới M1; M5 giảm mạnh. Vì vậy ba kết luận an toàn là thứ tự PE quan trọng, relative-aware cấu hình gọn có lợi cho mức độ gần đúng, và tăng depth/head không tự động cải thiện.**

## 7.2. Bảng giả thuyết M1–M5

| Model | Thay đổi chính | Giả thuyết | Kết quả quan sát | Kết luận an toàn |
|---|---|---|---|---|
| M1 | Không GAT | Control | Avg ExpRate 50,10%; MED 2,20 | Baseline rất mạnh |
| M2 | GAT 2L 8H, PE trước | GAT làm giàu feature | Avg ExpRate 47,84%; MED 2,21 | Naive integration gây giảm |
| M3 | GAT 2L 8H, PE sau | Giữ absolute position rõ hơn | Avg ExpRate 49,17%; MED 2,14 | PE sau tốt hơn PE trước |
| M4 | 8-neighbor + relative bias, 1L 4H | Direction bias và model gọn giúp | Avg ExpRate 48,98%; MED 2,06 | Gần GT hơn trung bình, chưa thắng exact |
| M5 | 2L 8H relative-aware | Tăng capacity sẽ tốt hơn | Avg ExpRate 43,35%; MED 2,65 | Scale-up thất bại |

## 7.3. Giả thuyết H0 — Baseline có cần GAT không?

M1 định nghĩa đường cơ sở:

$$ \text{Image} \to \text{DenseNet} \to PE \to \text{Transformer} $$

Mọi mô hình sau phải so với M1 trong cùng điều kiện:

- dataset;
- preprocessing;
- vocabulary;
- decoder;
- training budget;
- beam search;
- seed hoặc nhiều seed.

Nếu không giữ các điều kiện này, M1 không còn là control đáng tin cậy.

## 7.4. Giả thuyết H1 — Thêm GAT theo cách trực tiếp có cải thiện không?

M2:

$$ H_{\text{M2}} = \operatorname{GAT}(H + PE) $$

Kỳ vọng ban đầu:

- GAT tổng hợp ngữ cảnh;
- tăng khả năng nhận cấu trúc.

Quan sát:

- ExpRate giảm 2,26 điểm phần trăm trung bình so với M1;
- Mean Edit Distance gần như không cải thiện.

Kết luận:

> **Chỉ thêm GAT không bảo đảm cải thiện; vị trí tích hợp có thể làm representation kém hơn.**

## 7.5. Giả thuyết H2 — PE sau GAT tốt hơn PE trước GAT

M3:

$$ H_{\text{M3}} = \operatorname{GAT}(H) + PE $$

Quan sát:

- M3 cao hơn M2 1,33 điểm phần trăm ExpRate trung bình;
- edit distance giảm từ 2,21 xuống 2,14;
- M3 tiệm cận M1;
- trên CROHME 2016, M3 50,74% so với M1 50,65%.

Kết luận an toàn:

> **Trong thiết lập 2 layer, 8 head đã thử, PE sau GAT hiệu quả hơn PE trước GAT.**

Chưa đủ để khẳng định:

- mọi GNN đều phải đặt PE sau;
- PE thật sự “biến mất”;
- M3 tốt hơn baseline nói chung.

## 7.6. Giả thuyết H3 — Relative direction và mô hình gọn giúp giảm lỗi

M4 thay đổi nhiều yếu tố:

- 8-neighbor;
- 9-state bias;
- 1 layer;
- 4 head.

Quan sát:

- ExpRate trung bình 48,98%;
- MED 2,06, tốt nhất;
- 2014 MED 1,98;
- 2016 MED 2,13;
- 2019 MED 2,08.

Kết luận an toàn:

> **Gói thiết kế M4 làm đầu ra gần ground truth hơn trung bình, nhưng chưa tăng Exact Match trung bình so với M1.**

Chưa được phép kết luận:

> “Relative bias một mình làm MED giảm.”

Vì M4 thay nhiều biến.

## 7.7. Giả thuyết H4 — Scale-up GAT có tăng hiệu quả không?

M5 tăng:

- số layer;
- số head.

Quan sát:

- giảm trên mọi tập;
- giảm exact match;
- edit distance tăng.

Kết luận:

> **Trong cấu hình và training budget đã thử, tăng GAT từ 1L4H lên 2L8H làm tổng quát hóa kém hơn.**

Các giả thuyết nguyên nhân:

- overfitting;
- over-smoothing;
- dropout qua nhiều tầng;
- optimization khó;
- parameter count tăng;
- relative bias bị biến đổi qua nhiều layer;
- seed variance.

Cần thí nghiệm để tách.

## 7.8. M1–M5 chưa kiểm chứng được gì?

- Không chứng minh GAT tốt hơn CNN sâu tương đương tham số.
- Không chứng minh graph 8 hướng tốt hơn 4 hướng.
- Không chứng minh relative bias tốt hơn không bias khi giữ nguyên layer/head.
- Không chứng minh 4 head tốt hơn 8 head khi giữ nguyên layer.
- Không chứng minh 1 layer tốt hơn 2 layer khi giữ nguyên head.
- Không chứng minh “PE blurring” bằng measurement.
- Không chứng minh “over-smoothing” bằng node similarity.
- Không chứng minh cải thiện riêng trên phân số, căn hoặc tích phân.

## 7.9. Ablation bổ sung nên làm

### A — Tách connectivity

- 4-neighbor, no bias;
- 8-neighbor, no bias.

### B — Tách relative bias

- 8-neighbor, no bias;
- 8-neighbor, bias.

### C — Tách depth

- 1L4H;
- 2L4H.

### D — Tách head

- 1L4H;
- 1L8H.

### E — Đo PE blurring

- cosine similarity của PE trước và sau GAT;
- linear probe dự đoán tọa độ node;
- visualization decoder cross-attention.

### F — Đo over-smoothing

Với layer $l$:

$$ S^{(l)} = \frac{1}{N(N-1)} \sum_{i\ne j} \cos \left( h_i^{(l)},h_j^{(l)} \right) $$

Nếu $S^{(l)}$ tăng mạnh khi layer tăng, representation có xu hướng giống nhau hơn.

### G — Nhiều seed

Báo cáo:

$$ \bar{x}\pm s $$

thay vì chỉ một run.

## 7.10. Một câu kết luận tốt

> **M1–M5 không phải năm model để chọn model có số cao nhất; chúng là một chuỗi kiểm chứng. M2 bác bỏ naive insertion, M3 xác nhận lợi ích tương đối của PE sau GAT, M4 cho thấy một cấu hình coordinate-aware gọn giảm edit distance, và M5 bác bỏ giả định scale-up đơn giản.**

---

# Câu 8 — Nếu hội đồng cho rằng đây chỉ là tích hợp các thành phần đã có, em sẽ chứng minh tính mới như thế nào?

## 8.1. Bản trả lời nhanh 60–90 giây

> **Em đồng ý rằng DenseNet, GAT, positional encoding và Transformer đều là thành phần đã có; em không nhận mình phát minh các thành phần đó. Tính mới của luận văn nằm ở bài toán thiết kế và bằng chứng thực nghiệm: graph hóa feature map, chèn GAT có residual vào encoder, khảo sát có kiểm soát PE trước/sau message passing, bổ sung bias hướng 9 trạng thái và kiểm tra giới hạn khi tăng độ sâu.**
>
> **Để chứng minh đây không phải ghép module tùy ý, em phải chỉ ra mỗi thay đổi gắn với một giả thuyết, một control và một kết quả M1–M5. Đồng thời em sẽ định vị novelty ở mức kiến trúc ứng dụng và design insight, không tuyên bố thuật toán nền tảng hoàn toàn mới. Nếu hội đồng yêu cầu novelty cấp bài báo mạnh hơn, em cần bổ sung literature matrix, ablation tách biến và nhiều seed.**

## 8.2. Trước hết: thừa nhận phần đúng trong phản biện

Hội đồng nói “ghép module có sẵn” không hoàn toàn sai.

Các thành phần:

- DenseNet;
- GAT;
- residual;
- LayerNorm;
- positional encoding;
- relative bias;
- Transformer Decoder;
- beam search

đều có nền tảng từ công trình trước.

Không nên phản ứng:

> “Không, tất cả đều là em tự tạo.”

Câu trả lời tốt bắt đầu bằng:

> **Đúng là các khối cơ sở đã có. Đóng góp của em không nằm ở việc phát minh lại các khối đó, mà ở cách đặt bài toán, cách kết hợp, giả thuyết thiết kế và bằng chứng thực nghiệm.**

## 8.3. Phân biệt “ghép module” và “nghiên cứu tích hợp”

### Ghép module tùy ý

- thêm GAT vì thấy phổ biến;
- không có giả thuyết;
- không có baseline;
- thay nhiều cấu hình nhưng không kiểm soát;
- chỉ báo cáo model tốt nhất;
- kết quả thấp thì đổ GPU;
- không phân tích nguyên nhân.

### Nghiên cứu tích hợp có khoa học

- xác định failure mode;
- đặt giả thuyết;
- xây control;
- thay đổi có chủ đích;
- đánh giá đa tập;
- báo cáo cả kết quả âm;
- phân biệt quan sát và suy luận;
- lưu code, config, run và output;
- đề xuất thí nghiệm xác minh tiếp.

M1–M5 có thể được trình bày theo hướng thứ hai.

## 8.4. Bốn trụ cột chứng minh tính mới

### Trụ cột 1 — Một vấn đề thiết kế cụ thể

Không nói:

> “Em ghép GAT vào TAMER.”

Mà nói:

> **Em nghiên cứu xung đột giữa message passing và positional information khi GAT hoạt động trên feature-grid của HMER.**

Đây là một vấn đề cụ thể, có thể kiểm chứng.

### Trụ cột 2 — Một cơ chế được thiết kế cho vấn đề đó

- GAT trước PE;
- residual;
- relative directional bias;
- 8-connected graph.

Mỗi thành phần có lý do.

### Trụ cột 3 — Một chuỗi ablation

- M1 control;
- M2 failure;
- M3 correction;
- M4 directional bias;
- M5 scale-up.

### Trụ cột 4 — Một tập kết luận không hiển nhiên

- GAT naive làm giảm kết quả;
- thứ tự PE quan trọng;
- model gọn có MED tốt nhất;
- scale-up làm giảm mạnh;
- Exact Match và edit distance có thể xung đột.

Nếu chỉ “ghép module”, không nhất thiết tạo được các kết luận này.

## 8.5. Tính mới ở ba cấp độ

### Cấp độ 1 — Novelty thuật toán nền tảng

Ví dụ:

- một công thức GAT mới;
- một theorem mới;
- một training objective mới.

Luận văn hiện tại chưa nên tuyên bố mạnh ở cấp này.

### Cấp độ 2 — Novelty kiến trúc ứng dụng

- cách đặt GAT;
- cách graph hóa feature map;
- phối hợp absolute và relative position;
- residual integration.

Đây là cấp độ phù hợp nhất.

### Cấp độ 3 — Novelty thực nghiệm/design insight

- PE order;
- scale-up limitation;
- trade-off metric;
- negative results.

Đây là phần có thể bảo vệ tốt nếu thí nghiệm sạch.

## 8.6. Cần tránh tuyên bố “first” khi chưa rà soát đầy đủ

Đã có:

- graph-to-graph HMER;
- syntax-aware HMER;
- tree-aware Transformer;
- stroke-level EGAT;
- symbol-level graph/link prediction;
- relative position trong nhiều mô hình attention.

Vì vậy không nên viết:

> “Đây là công trình đầu tiên sử dụng GAT cho HMER.”

Cách viết an toàn:

> **Theo phạm vi tài liệu đã khảo sát, luận văn tập trung vào một cấu hình ít được phân tích có hệ thống: GAT trên feature-grid kết hợp nghiên cứu thứ tự absolute PE và directional relative bias trong pipeline DenseNet–Transformer.**

Trước khi nộp luận văn, vẫn cần literature review chính thức để xác nhận câu này.

## 8.7. Bằng chứng cụ thể để phản biện “chỉ ghép”

### Bằng chứng code

- `encoder.py`:
  - dynamic adjacency;
  - 8-connectivity;
  - self-loop;
  - padding mask;
  - residual;
  - PE sau GAT.
- `gat.py`:
  - per-head 9-state relative bias;
  - bias cộng vào attention logits;
  - multi-layer configuration.

### Bằng chứng thực nghiệm

- M2 thấp hơn M1;
- M3 cao hơn M2;
- M4 MED thấp nhất;
- M5 giảm mạnh;
- kết quả trên 3 test set.

### Bằng chứng phương pháp luận

- mỗi model trả lời một RQ;
- báo cáo cả metric bất lợi;
- không che baseline tốt hơn;
- phân tích giới hạn.

## 8.8. Phản biện mà hội đồng có thể đưa tiếp

### “PE sau GAT là đổi hai dòng code, sao gọi là đóng góp?”

> **Độ dài code không quyết định giá trị nghiên cứu. Giá trị nằm ở việc xác định đúng failure mode, thiết kế control M2–M3 và chứng minh ảnh hưởng lặp lại trên nhiều tập. Tuy nhiên để tuyên bố mạnh, em cần nhiều seed và phân tích representation.**

### “Relative bias đã có từ lâu.”

> **Đúng. Em không nhận relative bias nói chung là mới. Đóng góp là cách mã hóa 9 hướng theo từng head trên feature-grid HMER, tích hợp với PE sau GAT và đánh giá trong chuỗi ablation.**

### “M4 còn thấp hơn baseline, vậy novelty có tác dụng gì?”

> **M4 chưa tạo cải thiện Exact Match trung bình, nhưng đạt Mean Edit Distance tốt nhất và cho thấy một trade-off. Kết quả này chứng minh tác động của thiết kế, không chứng minh ưu thế tuyệt đối. Em xem đây là design insight và nền tảng cho thí nghiệm tiếp theo, đồng thời thừa nhận hạn chế.**

### “M4 thay quá nhiều thứ, sao biết relative bias có tác dụng?”

> **Với bảng hiện tại em chưa thể tách riêng. Đây là hạn chế của ablation và em cần bổ sung các cấu hình giữ nguyên layer/head/connectivity rồi chỉ bật tắt bias.**

Câu trả lời này không làm mất điểm. Ngược lại, nó thể hiện hiểu thiết kế thí nghiệm.

## 8.9. Bộ tiêu chí để tính mới đủ thuyết phục

Trước hội đồng, cần có:

- [ ] Related-work matrix tối thiểu 10–15 công trình.
- [ ] Phân biệt stroke graph, symbol graph và feature-grid graph.
- [ ] Một novelty statement chỉ 2–3 câu, không phóng đại.
- [ ] Bảng M1–M5 với biến thay đổi rõ ràng.
- [ ] Parameter count từng model.
- [ ] Cùng training budget.
- [ ] Nhiều seed cho model chính.
- [ ] Ablation tách relative bias.
- [ ] Phân tích PE representation hoặc attention alignment.
- [ ] Error analysis theo cấu trúc.
- [ ] Code/tag hoặc commit riêng cho từng M.
- [ ] Kết quả gốc truy được từ run ID.

## 8.10. Bản trả lời đầy đủ khoảng 2 phút

> **Em thừa nhận các khối DenseNet, GAT, positional encoding và Transformer đều đã có. Vì vậy em không định vị đóng góp là phát minh một thuật toán nền tảng mới. Điểm nghiên cứu của em là một vấn đề tích hợp cụ thể: khi GAT message passing trên feature-grid của HMER, thông tin vị trí tuyệt đối nên đi qua GAT hay được bổ sung sau; GAT có cần bias hướng tương đối hay không; và tăng độ sâu có giúp hay làm representation kém hơn.**
>
> **Em biến các câu hỏi đó thành M1–M5. M1 là baseline; M2 cho PE trước GAT và giảm; M3 chuyển PE sau GAT và phục hồi; M4 thêm bias 9 hướng với cấu hình 1 lớp 4 head, đạt Mean Edit Distance tốt nhất; M5 tăng 2 lớp 8 head nhưng giảm mạnh. Như vậy đây không phải việc ghép module rồi chỉ chọn kết quả tốt, mà là một chuỗi kiểm chứng có cả kết quả dương và âm.**
>
> **Tuy nhiên em cũng giới hạn tuyên bố: M4 chưa vượt baseline về ExpRate trung bình; relative bias không phải khái niệm hoàn toàn mới; và M4 đang thay nhiều biến cùng lúc. Do đó tính mới phù hợp nhất là novelty ở mức kiến trúc ứng dụng và design insight. Để nâng lên mức công bố mạnh hơn, em sẽ bổ sung related-work matrix, ablation tách biến và nhiều seed.**

---

# 9. Bảng tóm tắt tám câu để học thuộc

| Câu | Nội dung cốt lõi |
|---|---|
| 1 | Khoảng trống hẹp: GAT trên feature-grid, thứ tự PE, relative direction và depth |
| 2 | CNN–Transformer mạnh nhưng inductive bias 2D chưa tường minh; không nói chúng “không hiểu 2D” |
| 3 | Đóng góp gồm kiến trúc, graph 8 hướng, PE sau GAT, bias 9 trạng thái và đánh giá M1–M5 |
| 4 | Chuyên đề cũ là symbol graph; repo mới là feature-grid graph |
| 5 | Chọn GAT để kiểm tra adaptive local message passing, không phải vì GAT mặc định tốt nhất |
| 6 | Tách phương pháp, thực nghiệm và engineering |
| 7 | M1 control; M2 naive; M3 PE correction; M4 coordinate-aware; M5 scale-up negative result |
| 8 | Không phủ nhận module có sẵn; novelty ở cách kết hợp, giả thuyết và bằng chứng |

---

# 10. Novelty statement đề xuất cho luận văn

## Bản thận trọng

> **Luận văn nghiên cứu cách tích hợp Graph Attention Network vào feature-grid của một hệ thống DenseNet–Transformer cho nhận dạng biểu thức toán học viết tay. Trọng tâm là khảo sát thứ tự giữa message passing và positional encoding, đồng thời bổ sung relative directional bias trên graph lân cận 8 hướng. Chuỗi M1–M5 cung cấp bằng chứng thực nghiệm về lợi ích của PE sau GAT, trade-off giữa Exact Match và edit distance, và giới hạn khi tăng độ sâu GAT.**

## Bản dùng khi thuyết trình

> **Điểm mới của em không phải là dùng GAT nói chung, mà là nghiên cứu GAT nên được đặt và cung cấp vị trí như thế nào trong encoder HMER.**

## Bản không nên dùng

> **Đây là mô hình GNN đầu tiên cho HMER và vượt trội hoàn toàn so với Transformer.**

---

# 11. Bốn câu hỏi nghiên cứu đề xuất

### RQ1

> **Việc chèn GAT vào feature-grid của DenseNet ảnh hưởng thế nào đến chất lượng sinh LaTeX so với baseline không GAT?**

### RQ2

> **Đặt absolute positional encoding trước hay sau GAT ảnh hưởng thế nào đến Exact Match và edit distance?**

### RQ3

> **Relative directional bias 9 trạng thái trên graph 8 hướng có giúp giảm mức độ nghiêm trọng của lỗi hay không?**

### RQ4

> **Tăng số layer và head của Coordinate-Aware GAT có tạo ra cải thiện ổn định hay làm suy giảm khả năng tổng quát hóa?**

---

# 12. Sơ đồ lập luận toàn nhóm

```text
HMER cần nhận dạng hình dạng + quan hệ 2D + sinh chuỗi đúng cú pháp
                            ↓
CNN–Transformer là baseline mạnh nhưng phải học nhiều quan hệ gián tiếp
                            ↓
Đề xuất thử adaptive local message passing trên feature-grid
                            ↓
M1: Không GAT — control
                            ↓
M2: PE trước GAT — naive integration giảm
                            ↓
M3: GAT trước, PE sau — phục hồi
                            ↓
M4: Relative directional bias + cấu hình gọn — MED tốt nhất
                            ↓
M5: Scale-up — giảm mạnh
                            ↓
Kết luận:
- Thứ tự PE quan trọng
- Bias hướng có tiềm năng giảm mức độ lỗi
- GAT sâu hơn không mặc định tốt hơn
- Chưa vượt baseline về ExpRate trung bình
```

---

# 13. Các điểm có thể bị hội đồng bắt lỗi trong repo và cách xử lý

## 13.1. README gọi M4 là “đề xuất tốt nhất”

Nên sửa cách diễn đạt thành:

> **M4 là cấu hình đề xuất chính theo tiêu chí trade-off giữa kiến trúc gọn và Mean Edit Distance, không phải mô hình có ExpRate cao nhất.**

## 13.2. README nói Mean Edit Distance “chính xác nhất” cho bảo toàn cấu trúc

Cách nói này quá mạnh.

Nên sửa:

> **Mean Edit Distance bổ sung thông tin về mức độ gần đúng của chuỗi. Nó không thay thế ExpRate và không phải metric cấu trúc trực tiếp.**

Muốn nói bảo toàn cấu trúc cần:

- tree edit distance;
- relation accuracy;
- parse validity;
- accuracy theo loại cấu trúc.

## 13.3. README mô tả M2/M3 graph 4 hướng nhưng code main hiện là 8 hướng

Cần:

- lưu code/tag riêng từng model;
- chỉ rõ commit hoặc notebook;
- không lấy code main hiện tại làm bằng chứng duy nhất cho M2/M3.

## 13.4. M4 thay nhiều biến

Cần thừa nhận confounding và bổ sung ablation.

## 13.5. Kết quả chỉ một run

Cần nhiều seed hoặc ghi rõ đây là hạn chế.

## 13.6. Giả thuyết nguyên nhân được viết như sự thật

Các câu như:

- “PE bị nhòe”;
- “dropout làm đứt graph”;
- “relative bias bị méo”;
- “overfitting”;

phải được gắn nhãn:

> **giả thuyết giải thích**.

---

# 14. Checklist trước khi bảo vệ Nhóm 2

## Bắt buộc

- [ ] Viết related-work matrix.
- [ ] Không tuyên bố first-ever.
- [ ] Chốt novelty statement thận trọng.
- [ ] Tách symbol graph và feature-grid graph.
- [ ] Gắn M1–M5 với RQ1–RQ4.
- [ ] Sửa câu “M4 tốt nhất” thành đúng metric.
- [ ] Sửa câu “MED đo cấu trúc tốt nhất”.
- [ ] Xác nhận cấu hình từng run từ log.
- [ ] Lưu code/commit riêng cho M1–M5.
- [ ] Kiểm tra M2/M3 thật sự chỉ khác PE order.
- [ ] Thừa nhận M4 có nhiều biến thay đổi.
- [ ] Không gọi PE blurring là đã chứng minh trực tiếp.

## Nên có để nhắm 9–9,5

- [ ] Ba seed cho M1, M3, M4.
- [ ] Mean ± standard deviation.
- [ ] Parameter count.
- [ ] FLOPs hoặc inference time.
- [ ] Ablation 4-neighbor vs 8-neighbor.
- [ ] Ablation bias on/off.
- [ ] Ablation 1L vs 2L giữ nguyên head.
- [ ] Ablation 4H vs 8H giữ nguyên layer.
- [ ] Linear probe tọa độ hoặc visualization attention.
- [ ] Node similarity để kiểm tra over-smoothing.
- [ ] Error analysis theo cấu trúc.
- [ ] Kiểm định riêng tích phân có cận.

---

# 15. Nguồn đối chiếu

## Repo luận văn

- `README.md`
  - Pipeline DenseNet–GAT–Transformer.
  - M1–M5 và run ID.
  - Bảng kết quả CROHME 2014, 2016, 2019.

- `chuyende_tamer_temp/1-cnn-gnn/tamer/model/encoder.py`
  - Feature-grid graph.
  - Adjacency 8 hướng.
  - Self-loop.
  - Padding mask.
  - Residual GAT.
  - PE sau GAT.

- `chuyende_tamer_temp/1-cnn-gnn/tamer/model/gat.py`
  - Multi-head GAT.
  - Relative position bias 9 trạng thái.
  - Attention score, masking và dropout.

- `App/CROHME_M4_NEXT_PHASE_SPEC.md`
  - Pipeline M4-ready.
  - Một biểu thức mỗi lần.
  - Chuẩn hóa ảnh và tích hợp ứng dụng.

## Tài liệu nghiên cứu cần đặt trong related work

- **Graph Attention Networks** — cơ sở của GAT.
- **TAMER: Tree-Aware Transformer for Handwritten Mathematical Expression Recognition** — cấu trúc cây phụ trợ cho sequence decoder.
- **Syntax-Aware Network for Handwritten Mathematical Expression Recognition** — đưa cú pháp vào encoder–decoder.
- **Local and Global Graph Modeling with Edge-weighted Graph Attention Network for HMER** — graph ở mức stroke, node–edge classification.
- **Link Prediction Graph Neural Networks for Structure Recognition of HME** — graph ở mức symbol và quan hệ.
- Các công trình graph-to-graph HMER và graph structure recognition liên quan.

---

# 16. Câu kết thúc Nhóm 2

> **Luận văn không có giá trị vì đã ghép được nhiều module, mà có giá trị khi mỗi module được đưa vào để trả lời một câu hỏi nghiên cứu. GAT là công cụ; PE order, directional bias và depth là các biến nghiên cứu; M1–M5 là bằng chứng; còn kết luận phải trung thực với cả điểm tăng lẫn điểm giảm.**
