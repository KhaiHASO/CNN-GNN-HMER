# NHÓM 6 — GRAPH, GNN, GAT VÀ THUẬT TOÁN XÂY DỰNG ĐỒ THỊ

> **Mục tiêu nhóm:** Hiểu chính xác graph là gì, được tạo ra thế nào, message passing hoạt động ra sao và giới hạn của thiết kế hiện tại.

## Định nghĩa graph chính xác

Graph của repo là **8-connected feature-grid graph**:

- node: feature cell sau DenseNet;
- edge: neighborhood hình học;
- node count: $N=H_fW_f$;
- semantic relation: không được gán tường minh;
- implementation attention: dense $N\times N$ rồi mask.

---

# Câu 1 — Graph trong repo được xây dựng từ feature map theo thuật toán cụ thể nào?

## 1.1. Bản trả lời nhanh

> **Sau DenseNet và projection, feature có shape `[B,H,W,D]`. Code flatten thành $N=H\times W$ node theo row-major. Nó tạo ma trận kề cho một grid 8-connected: phải, xuống, chéo phải xuống, chéo trái xuống, rồi thêm cạnh ngược để có đủ tám hướng; cuối cùng thêm self-loop. Adjacency được nhân bản cho batch và xóa các cạnh đi/đến node padding theo mask.**

## 1.2. Giải thích chi tiết

### Thuật toán

1. Tạo index:

$$ i=0,\ldots,N-1 $$

2. Xác định cạnh phải khi $x \lt W-1$:

$$ (i,i+1) $$

3. Cạnh xuống khi $y \lt H-1$:

$$ (i,i+W) $$

4. Chéo phải xuống:

$$ (i,i+W+1) $$

5. Chéo trái xuống:

$$ (i,i+W-1) $$

6. Thêm reverse edges.
7. Thêm $(i,i)$.
8. Expand thành `[B,N,N]`.
9. Mask row/column của padding.
10. Tái thêm self-loop để tránh NaN.

### Vì sao chỉ tạo bốn loại rồi reverse?

Đó là cách tránh thêm trùng. Sau reverse, node nội bộ có tối đa tám neighbor cộng self.

### Hạn chế

- ma trận dense $N\times N$;
- topology cố định theo grid;
- không dựa nội dung;
- không có long-range edge;
- padding self-loop vẫn tồn tại.

## 1.4. Không nên nói

- “Graph được học tự động.”
- “Edge nối theo semantic relation.”
- “Graph fully connected.”
- “Adjacency thưa về cấp phát bộ nhớ.”

## 1.5. Bằng chứng cần chỉ ra

- `_build_grid_adjacency()` trong `encoder.py`.

---

# Câu 2 — Mỗi node chứa những thành phần đặc trưng nào?

## 2.1. Bản trả lời nhanh

> **Node ban đầu chứa vector visual feature 256 chiều từ DenseNet sau convolution $1\times1$. Trong M3–M5, absolute positional encoding được thêm sau GAT, nên đầu vào GAT là visual feature thuần. Trong M4/M5, hướng tương đối không nối vào node feature mà đi vào attention logits qua bảng bias 9 trạng thái.**

## 2.2. Giải thích chi tiết

Node $h_i$ có nguồn từ receptive field ảnh. Nó mã hóa ngầm:

- nét;
- góc;
- texture;
- shape;
- local context.

Nó không chứa tường minh:

- symbol class;
- bounding box;
- width/height ký hiệu;
- relation label;
- raw $(x,y)$ concatenated, theo code M4 hiện tại.

### Absolute position

Sau GAT:

$$ z_i=\operatorname{LN}(h_i'+PE(x_i,y_i)) $$

### Relative position

Trong attention:

$$ e_{ij}^{(k)} = e_{ij,\text{content}}^{(k)} + b_{k,r_{ij}} $$

Vì vậy phải phân biệt:

- node feature;
- node absolute position;
- edge relative direction.

## 2.4. Không nên nói

- “Node chứa class symbol.”
- “Node chứa bounding box.”
- “Tọa độ được concatenate vào feature” nếu code không làm.

## 2.5. Bằng chứng cần chỉ ra

- `feature_proj`, `view` và `pos_enc_2d` trong `encoder.py`.
- `rel_bias` trong `gat.py`.

---

# Câu 3 — Hai node được nối cạnh khi thỏa điều kiện gì?

## 3.1. Bản trả lời nhanh

> **Hai node được nối nếu tọa độ grid của chúng lệch tối đa một ô theo mỗi trục và không đồng thời bằng 0, tức là hàng xóm 8 hướng; self-loop nối node với chính nó. Cạnh chỉ hợp lệ khi node không phải padding, ngoại trừ self-loop được tái thêm để giữ softmax ổn định.**

## 3.2. Giải thích chi tiết

Với tọa độ $(x_i,y_i)$ và $(x_j,y_j)$:

$$ |\Delta x|\le1,\quad|\Delta y|\le1 $$

và với neighbor:

$$ (\Delta x,\Delta y) e(0,0) $$

Self-loop là trường hợp $(0,0)$.

### Không có điều kiện về feature similarity

Adjacency không hỏi:

- cosine similarity;
- cùng symbol;
- khoảng cách Euclidean ảnh gốc;
- confidence.

Attention weight mới dùng feature để quyết định mức đóng góp.

### Padding

Nếu $M_i=1$ hoặc $M_j=1$, cạnh thường bị xóa. Self-loop được thêm lại cho tất cả node.

## 3.4. Không nên nói

- “Edge được tạo nếu attention cao.”
- “Edge chỉ nối cùng ký hiệu.”
- “Padding node hoàn toàn không đi qua GAT” — nó vẫn có self-loop nội bộ.

## 3.5. Bằng chứng cần chỉ ra

- Adjacency code và mask.

---

# Câu 4 — Vì sao chọn lưới 8 láng giềng thay vì 4 láng giềng, k-NN hoặc fully connected graph?

## 4.1. Bản trả lời nhanh

> **8-neighbor bổ sung tương tác chéo, phù hợp các quan hệ xiên như số mũ và chỉ số, trong khi vẫn giữ topology cục bộ. So với 4-neighbor, đường truyền chéo ngắn hơn; so với fully connected, nó áp đặt local inductive bias và tránh mọi node tương tác logic với nhau; so với k-NN, grid không cần tính khoảng cách và ổn định theo shape.**
>
> **Tuy nhiên repo chưa có ablation giữ nguyên mọi thứ rồi chỉ đổi 4/8/k-NN, nên đây là lý do thiết kế chứ chưa phải bằng chứng 8-neighbor tối ưu.**

## 4.2. Giải thích chi tiết

### 4-neighbor

Ưu:

- ít edge;
- đơn giản;
- directional rõ.

Nhược:

- node chéo cần hai hop;
- có thể kém với superscript/subscript xiên.

### 8-neighbor

Ưu:

- diagonal one-hop;
- mỗi node nội bộ có neighborhood vuông $3\times3$;
- tương thích 9-state bias.

Nhược:

- thêm edge;
- diagonal có thể trộn vùng không liên quan;
- vẫn rất local.

### k-NN

Trên grid đều, k-NN gần giống neighborhood theo khoảng cách. Nếu dùng feature-space k-NN:

- graph phụ thuộc nội dung;
- chi phí xây graph;
- có nguy cơ nối sai sớm.

### Fully connected

Ưu:

- global interaction một layer.

Nhược:

- mất local bias;
- $O(N^2)$ edge logic;
- dễ học shortcut;
- trùng với self-attention.

### Cần thí nghiệm

| Graph | Same GAT | Same params | Same seed |
|---|---|---|---|
| 4-neighbor | ✓ | ✓ | ✓ |
| 8-neighbor | ✓ | ✓ | ✓ |
| radius-2 | ✓ | ✓ | ✓ |
| k-NN | ✓ | ✓ | ✓ |

## 4.4. Không nên nói

- “8-neighbor chắc chắn tốt nhất.”
- “Fully connected không thể học.”
- “k-NN luôn chậm.”

## 4.5. Bằng chứng cần chỉ ra

- M4 code.
- Ablation connectivity cần bổ sung.

---

# Câu 5 — Graph là có hướng hay vô hướng; có self-loop hay không?

## 5.1. Bản trả lời nhanh

> **Ma trận kề được xây đối xứng: mỗi cạnh phải/xuống/chéo đều được thêm cạnh ngược, nên về topology nó là graph vô hướng được biểu diễn bằng hai directed entries. Graph có self-loop cho mọi node.**
>
> **Attention $\alpha_{ij}$ và $\alpha_{ji}$ vẫn có thể khác vì softmax theo neighborhood của từng node, nên message passing có tính định hướng dù adjacency đối xứng.**

## 5.2. Giải thích chi tiết

Adjacency:

$$ A_{ij}=A_{ji}=1 $$

Nhưng attention:

$$ \alpha_{ij} = \frac{\exp e_{ij}} {\sum_{k\in\mathcal{N}(i)}\exp e_{ik}} $$

và:

$$ \alpha_{ji} = \frac{\exp e_{ji}} {\sum_{k\in\mathcal{N}(j)}\exp e_{jk}} $$

không bắt buộc bằng nhau.

### Vai trò self-loop

- giữ thông tin của node;
- tránh neighborhood rỗng;
- tránh softmax toàn $-\infty$;
- relative state trung tâm có index riêng.

Residual bên ngoài GAT còn thêm một đường giữ feature khác.

## 5.4. Không nên nói

- “Vô hướng nên attention hai chiều bằng nhau.”
- “Không cần self-loop vì có residual.”
- “Self-loop là cạnh semantic.”

## 5.5. Bằng chứng cần chỉ ra

- Các assignment đối xứng và diagonal trong adjacency.

---

# Câu 6 — Một lớp GAT cập nhật node embedding theo công thức và trực giác nào?

## 6.1. Bản trả lời nhanh

> **Đầu tiên node được chiếu tuyến tính và chia thành nhiều head. Với mỗi cặp node kề, model tính attention logit từ feature của node nguồn và node đích, cộng relative direction bias, qua LeakyReLU, mask cạnh không tồn tại và softmax trên các láng giềng. Output là tổng có trọng số của value feature từ các neighbor.**

## 6.2. Giải thích chi tiết

Với head $k$:

$$ z_i^{(k)}=W^{(k)}h_i $$

Logit nội dung:

$$ e_{ij,\text{content}}^{(k)} = a_1^\top z_i^{(k)} + a_2^\top z_j^{(k)} $$

Thêm bias:

$$ e_{ij}^{(k)} = \operatorname{LeakyReLU} \left( e_{ij,\text{content}}^{(k)} + b_{k,r_{ij}} \right) $$

Mask và chuẩn hóa:

$$ \alpha_{ij}^{(k)} = \frac{\exp e_{ij}^{(k)}} {\sum_{m\in\mathcal{N}(i)}\exp e_{im}^{(k)}} $$

Update:

$$ h_i'= \operatorname{Concat}_{k=1}^{K} \sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{(k)}z_j^{(k)} $$

Encoder dùng residual:

$$ \tilde h_i=h_i+\operatorname{GAT}(h)_i $$

### Trực giác

Mỗi node hỏi:

> Trong vùng $3\times3$ quanh mình, feature nào cần được ưu tiên để làm rõ representation hiện tại?

Không nên diễn giải $\alpha$ trực tiếp là “xác suất node j là số mũ”.

## 6.4. Không nên nói

- “GAT phân loại edge.”
- “Attention weight là xác suất quan hệ toán học.”
- “Một layer nhìn toàn graph.”

## 6.5. Bằng chứng cần chỉ ra

- `GATLayer.forward`.
- `_compute_attention_scores`.
- residual trong encoder.

---

# Câu 7 — Attention coefficient trong GAT biểu diễn điều gì và được chuẩn hóa trên tập láng giềng ra sao?

## 7.1. Bản trả lời nhanh

> **$\alpha_{ij}$ là trọng số tương đối mà node $i$ dành cho feature của node $j$ trong một head và một lần forward. Nó được softmax theo chiều node đích sau khi các cặp không có cạnh bị đặt $-\infty$, nên tổng trọng số trên neighborhood của $i$ bằng 1 trước dropout.**
>
> **Nó không phải xác suất ground-truth của một quan hệ và không nên được xem là lời giải thích tuyệt đối.**

## 7.2. Giải thích chi tiết

Trước dropout:

$$ \sum_{j\in\mathcal{N}(i)} \alpha_{ij}^{(k)} =1 $$

Sau dropout attention, tổng có thể không còn đúng 1 trong một sample train, tùy scaling của dropout.

### Phụ thuộc

Coefficient phụ thuộc:

- feature hiện tại;
- head;
- relative direction bias;
- neighborhood;
- model parameters;
- train/eval mode.

### Boundary effect

Node ở biên có ít neighbor hơn, nên phân phối softmax trên tập nhỏ hơn. Padding cũng thay đổi neighborhood theo sample.

### Interpretability caution

Attention visualization hữu ích để tạo giả thuyết, nhưng attention weight không tự chứng minh causal importance. Có thể kiểm tra bằng:

- edge ablation;
- attention rollout;
- gradient attribution;
- perturb neighbor feature.

## 7.4. Không nên nói

- “Attention cao nghĩa là node quan trọng thật.”
- “Tổng attention sau dropout luôn bằng 1.”
- “Mọi head học cùng quan hệ.”

## 7.5. Bằng chứng cần chỉ ra

- Softmax dimension `-1`.
- Attention dropout.
- Perturbation experiment.

---

# Câu 8 — Số head trong GAT ảnh hưởng thế nào đến năng lực biểu diễn và chi phí tính toán?

## 8.1. Bản trả lời nhanh

> **Nhiều head cho phép học nhiều kiểu tổng hợp hoặc ưu tiên hướng khác nhau. Nhưng với `out_features` cố định, tăng head làm `head_dim=out_features/heads` nhỏ hơn; đồng thời attention tensor và bias có thêm chiều head, tăng bộ nhớ/chi phí. Nhiều head cũng không bảo đảm tốt hơn, như M5 8 head thấp hơn M4 4 head khi đồng thời tăng depth.**

## 8.2. Giải thích chi tiết

Với $D=256$:

- 4 head → 64 chiều/head;
- 8 head → 32 chiều/head.

Trade-off:

- nhiều subspaces;
- mỗi subspace hẹp hơn;
- attention matrix tăng theo $K$;
- optimization phức tạp hơn.

Bộ nhớ score:

$$ O(BKN^2) $$

### Không thể tách head effect từ M4–M5

M4 và M5 đồng thời đổi:

- layer;
- head;
- có thể training dynamics.

Muốn kết luận head:

- 1L4H vs 1L8H;
- cùng seed, dropout, d_model;
- parameter count báo cáo.

## 8.4. Không nên nói

- “Nhiều head luôn tốt.”
- “Head tương ứng cố định với trên/dưới/trái/phải.”
- “M5 chứng minh 8 head xấu” khi depth cũng đổi.

## 8.5. Bằng chứng cần chỉ ra

- `head_dim`.
- Ablation giữ nguyên layer.

---

# Câu 9 — Vì sao dùng GAT thay vì GCN, GraphSAGE hoặc graph transformer?

## 9.1. Bản trả lời nhanh

> **GAT phù hợp giả thuyết vì nó học trọng số khác nhau cho từng neighbor và cho phép cộng relative directional bias trực tiếp vào attention logits. GCN dùng aggregation chuẩn hóa tương đối cố định; GraphSAGE tập trung hàm aggregate/sample neighbor; graph transformer mạnh hơn nhưng thường phức tạp và dễ mất local bias.**
>
> **Đây là rationale, chưa phải bằng chứng GAT tốt hơn nếu chưa có baseline cùng budget.**

## 9.2. Giải thích chi tiết

### GCN

$$ H'=\sigma(\hat D^{-1/2}\hat A\hat D^{-1/2}HW) $$

Neighbor weights chủ yếu do degree normalization.

### GraphSAGE

Có thể mean/max/LSTM aggregate, hữu ích graph lớn và inductive sampling. Grid hiện nhỏ và cố định nên sampling không phải nhu cầu chính.

### GAT

- adaptive neighbor weights;
- multi-head;
- relative bias tự nhiên;
- phù hợp 8-neighbor.

### Graph transformer

- global hoặc edge-aware attention;
- capacity lớn;
- có thể thay decoder/encoder rộng hơn;
- cần dữ liệu và compute;
- khó cô lập đóng góp.

### Baseline cần có nếu muốn khẳng định

- GCN cùng layers/d_model;
- local self-attention;
- GAT no bias;
- coordinate-aware GAT.

## 9.4. Không nên nói

- “GCN không có trọng số.”
- “GraphSAGE chỉ dùng social network.”
- “Graph transformer luôn overfit.”

## 9.5. Bằng chứng cần chỉ ra

- Công thức và implementation.
- So sánh thực nghiệm nếu bổ sung.

---

# Câu 10 — Độ phức tạp của graph thay đổi thế nào theo chiều cao và chiều rộng feature map?

## 10.1. Bản trả lời nhanh

> **Số node $N=HW$. Số edge logic của grid 8 hướng cộng self tăng tuyến tính, xấp xỉ $9N$ entries có hướng ở vùng nội bộ. Nhưng implementation dùng adjacency và attention dense $N\times N$, nên bộ nhớ và tính toán attention tăng bậc hai theo $HW$.**

## 10.2. Giải thích chi tiết

### Edge logic

Số undirected edges:

- ngang: $H(W-1)$;
- dọc: $(H-1)W$;
- chéo: $2(H-1)(W-1)$;
- self: $HW$.

Nếu biểu diễn hai hướng, số entry neighbor xấp xỉ:

$$ 2[H(W-1)+(H-1)W+2(H-1)(W-1)]+HW $$

Tăng $O(HW)$.

### Dense implementation

Adjacency:

$$ A\in\mathbb{R}^{N\times N} $$

Attention per head:

$$ E\in\mathbb{R}^{N\times N} $$

Tăng:

$$ O((HW)^2) $$

Nếu tăng gấp đôi H và W, $N$ tăng 4 lần, $N^2$ tăng 16 lần.

### Hướng tối ưu

- edge index;
- scatter softmax;
- sparse attention;
- neighborhood tensor `[B,N,9,D]`;
- window gather;
- cache relative index.

## 10.4. Không nên nói

- “Complexity là $O(N)$.”
- “8-neighbor tự động làm implementation sparse.”
- “Tăng resolution gấp đôi chỉ tăng chi phí gấp đôi.”

## 10.5. Bằng chứng cần chỉ ra

- Shapes trong `gat.py`.
- Memory profile theo H,W.

---

# Câu 11 — Graph grid có mô hình hóa trực tiếp quan hệ superscript, subscript, above và below hay không?

## 11.1. Bản trả lời nhanh

> **Không. Nó chỉ mã hóa hướng tương đối một ô: trên, dưới, trái, phải, chéo và self. Không có edge label “superscript”, “subscript”, “numerator” hay “denominator”. Những quan hệ toán học chỉ có thể được học ngầm qua feature, nhiều hop và decoder.**

## 11.2. Giải thích chi tiết

Relative state mô tả geometry:

$$ (\Delta x,\Delta y)\in\{-1,0,1\}^2 $$

Geometry không đồng nhất semantics.

Ví dụ node ở phía trên có thể thuộc:

- số mũ;
- tử số;
- cận trên;
- ký hiệu khác ở dòng trên;
- nhiễu.

Muốn semantic graph cần:

- symbol instances;
- relation labels;
- relation classifier;
- graph supervision.

### Cách nói an toàn

> **M4 đưa directional inductive bias, không đưa semantic relation supervision.**

## 11.4. Không nên nói

- “Bias trên chính là superscript.”
- “9 trạng thái là 9 quan hệ toán học.”
- “Graph hiểu tử/mẫu tường minh.”

## 11.5. Bằng chứng cần chỉ ra

- `rel_indices` chỉ dựa dx,dy.

---

# Câu 12 — Thông tin tọa độ tuyệt đối và vị trí tương đối được đưa vào GAT như thế nào?

## 12.1. Bản trả lời nhanh

> **Trong M3–M5, absolute 2D positional encoding không đi vào GAT; nó được cộng sau GAT. Relative position được đưa vào bên trong GAT bằng bảng bias học được `[num_heads,9]`, tra theo $\Delta x,\Delta y$ đã clamp về $-1,0,1$ và cộng vào attention logits trước LeakyReLU/softmax.**

## 12.2. Giải thích chi tiết

### Absolute position

$$ PE_i=PE(x_i,y_i) $$

Sau residual GAT:

$$ z_i=\operatorname{LN}(\tilde h_i+PE_i) $$

Mục tiêu thiết kế: giữ tọa độ tuyệt đối rõ cho decoder.

### Relative direction

Index:

$$ r_{ij}=3(\Delta y+1)+(\Delta x+1) $$

nằm 0–8.

Bias:

$$ b_{k,r_{ij}} $$

khác theo head.

### Điểm cần chú ý

Code tính relative index cho mọi cặp node rồi clamp, nhưng adjacency mask chỉ giữ neighbor. Vì vậy các cặp xa không thực sự attention, dù chúng cũng nhận một index đã clamp trước khi mask.

### Không có distance magnitude

Một neighbor grid luôn lệch 1 hoặc 0. Bias không mã hóa khoảng cách 2,3,... vì graph chỉ local.

## 12.4. Không nên nói

- “Absolute coordinate concatenate vào node.”
- “Relative bias mã hóa khoảng cách xa.”
- “PE trước GAT trong M4.”

## 12.5. Bằng chứng cần chỉ ra

- `encoder.py`: PE sau GAT.
- `gat.py`: `rel_bias`, `rel_indices`.

---

# Câu 13 — Nếu graph chỉ nối các ô lân cận, làm sao thông tin giữa hai vùng xa nhau có thể tương tác?

## 13.1. Bản trả lời nhanh

> **Qua nhiều cơ chế: một lớp GAT truyền một hop, nhiều lớp truyền nhiều hop; DenseNet trước đó đã có receptive field lớn; và Transformer Decoder cross-attend toàn bộ feature map nên có thể kết hợp vùng xa. Vì vậy graph không phải kênh global duy nhất.**
>
> **Tuy nhiên M4 chỉ một GAT layer nên phần GAT trực tiếp chỉ local; không được nói một layer GAT truyền toàn graph.**

## 13.2. Giải thích chi tiết

Sau $L$ layer, receptive field graph tối đa khoảng $L$ hop:

$$ h_i^{(L)} \text{ phụ thuộc } \mathcal{N}^{L}(i) $$

Nhưng:

- CNN feature mỗi node đã bao phủ vùng ảnh rộng;
- decoder có global memory access;
- autoregressive context liên kết các token.

### Vì sao M4 1 layer vẫn có thể hoạt động?

GAT chỉ làm local refinement, không phải thay toàn bộ global reasoning.

### Hướng nếu cần long-range

- dilated edges;
- hierarchical graph;
- pooling/coarsening;
- global virtual node;
- sparse long-range links;
- alternating local GAT và global attention.

## 13.4. Không nên nói

- “Một layer GAT nhìn toàn ảnh.”
- “Hai vùng xa không bao giờ tương tác.”
- “Phải tăng nhiều GAT layer mới có global context.”

## 13.5. Bằng chứng cần chỉ ra

- Layer count từng model.
- Decoder cross-attention.
- Receptive field analysis.

---

# Câu 14 — GAT có thật sự 'hiểu cấu trúc toán học' hay chỉ học tương quan cục bộ trên feature map?

## 14.1. Bản trả lời nhanh

> **Bằng chứng hiện tại chỉ cho phép nói GAT học representation và tương quan không gian cục bộ có ích cho nhiệm vụ sinh LaTeX. Không có supervision edge semantic hoặc probe chứng minh nó hiểu khái niệm tử số, mẫu số hay số mũ.**
>
> **“Hiểu cấu trúc” nên được dùng thận trọng và phải gắn với metric/analysis cụ thể.**

## 14.2. Giải thích chi tiết

### Ba mức tuyên bố

1. **Representation level:** GAT làm thay đổi feature theo neighborhood — đã có code.
2. **Task level:** một số metric thay đổi — có bảng thực nghiệm.
3. **Semantic understanding:** node/edge mã hóa quan hệ toán học — chưa được chứng minh.

### Cần gì để chứng minh mạnh hơn?

- probe dự đoán relation từ embedding;
- attention alignment với symbol relations;
- accuracy riêng cấu trúc;
- intervention: xóa edge hướng trên và xem superscript giảm;
- tree/graph annotation;
- counterfactual examples cùng glyph khác bố cục.

### Cách nói trước hội đồng

> **Em không dùng “hiểu” theo nghĩa nhận thức. Em nói GAT cung cấp inductive bias và học biểu diễn không gian hỗ trợ decoder.**

## 14.4. Không nên nói

- “GAT hiểu toán như con người.”
- “Attention map là bằng chứng hiểu semantics.”
- “MED thấp chứng minh hiểu cấu trúc.”

## 14.5. Bằng chứng cần chỉ ra

- Ablation.
- Probe/intervention nếu bổ sung.

---

# Phụ lục — Bảng so sánh graph

| Thiết kế | Node | Edge | Global? | Semantic edge? |
|---|---|---|---|---|
| 4-grid | feature cell | 4 hướng | Không trong 1 layer | Không |
| 8-grid M4 | feature cell | 8 hướng + self | Không trong 1 layer | Không |
| k-NN feature | feature cell | gần trong feature space | Tùy k | Không |
| Fully connected | feature cell | mọi cặp | Có | Không |
| Symbol Layout Graph | symbol | quan hệ symbol | Tùy | Có thể có |

# Nguồn đối chiếu

- `tamer/model/encoder.py`
- `tamer/model/gat.py`
- `tamer/model/pos_enc.py`
- `tamer/model/decoder.py`
