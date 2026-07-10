# NHÓM 5 — KIẾN TRÚC TỔNG THỂ CỦA HỆ THỐNG

> **Mục tiêu nhóm:** Có thể mô tả đúng pipeline trong repo từ tensor ảnh đến chuỗi LaTeX, không lẫn với symbol graph của chuyên đề cũ.

## Câu chuyện kiến trúc phải thống nhất

Tên pipeline nên dùng:

> **DenseNet visual encoder → feature-grid graph → Coordinate-Aware GAT → 2D positional encoding → Transformer decoder → LaTeX.**

Không dùng sơ đồ detector/symbol graph của chuyên đề cũ để mô tả code hiện tại.

---

# Câu 1 — Hãy mô tả toàn bộ pipeline hiện tại từ ảnh đầu vào đến chuỗi LaTeX đầu ra.

## 1.1. Bản trả lời nhanh

> **Ảnh raster một kênh và mask được đưa vào DenseNet. DenseNet downsample và trích xuất feature map, sau đó một convolution $1\times1$ chiếu số kênh về `d_model=256`. Feature map được đổi từ `[B,D,H,W]` sang `[B,H,W,D]`, flatten thành $N=H\times W$ node, xây graph lưới và chạy GAT có residual. Sau GAT, model thêm positional encoding 2D, rồi Transformer Decoder cross-attend vào memory ảnh và sinh token LaTeX.**
>
> **Khi train, decoder nhận target prefix và tối ưu cross-entropy cộng structure loss phụ trợ. Khi đánh giá, model dùng bidirectional beam search với beam size 10 và max length 150.**

## 1.2. Giải thích chi tiết

### Pipeline shape-aware

```text
Input image: [B,1,H0,W0]
Input mask : [B,H0,W0]
        ↓
DenseNet-B
        ↓
Visual feature: [B,C,Hf,Wf]
Feature mask  : [B,Hf,Wf]
        ↓
1×1 projection
        ↓
[B,256,Hf,Wf]
        ↓ rearrange
[B,Hf,Wf,256]
        ↓ flatten
[B,N,256], N=Hf×Wf
        ↓
8-connected adjacency + self-loop + padding mask
        ↓
GAT
        ↓
Residual: H + GAT(H)
        ↓
LayerNorm
        ↓
2D absolute positional encoding
        ↓
LayerNorm
        ↓
Transformer Decoder memory
        ↓
Autoregressive token distribution
        ↓
Beam search
        ↓
LaTeX token sequence
```

### Training path

Model tạo target hai chiều trong TAMER, nhân đôi feature và mask để phục vụ bidirectional target. Loss chính:

$$ \mathcal{L}_{seq} = -\sum_t \log p(y_t \mid y_{ \lt t},X) $$

Ngoài ra decoder tạo `sim` và code tính:

$$ \mathcal{L} = \mathcal{L}_{seq} + \mathcal{L}_{struct} $$

Không nên gọi `struct_loss` là edge classification của graph nếu code không làm vậy.

### Inference path

Encoder chỉ chạy một lần. Decoder beam search giữ nhiều hypothesis, dùng các tham số:

- beam size 10;
- max length 150;
- alpha 1.0;
- temperature 1.0;
- `early_stopping: false` trong config.

### Sản phẩm cuối

Chuỗi index được đổi về token bằng vocabulary rồi join để lưu/hiển thị. Model không xuất bounding box hoặc graph ký hiệu.

## 1.4. Không nên nói

- “Ảnh → detector → symbol graph.”
- “GAT thay Transformer Decoder.”
- “Đầu ra là cây cú pháp.”
- “Mỗi node là một ký hiệu.”

## 1.5. Bằng chứng cần chỉ ra

- `encoder.py`: DenseNet, projection, GAT, residual, PE.
- `tamer.py`: encoder–decoder và beam search.
- `decoder.py`: embedding, causal mask, cross-attention, projection.
- `lit_tamer.py`: loss và evaluation.

---

# Câu 2 — CNN/DenseNet trong repo có nhiệm vụ gì và đầu ra có kích thước như thế nào?

## 2.1. Bản trả lời nhanh

> **DenseNet biến ảnh pixel thành feature map thị giác có độ phân giải thấp hơn và số kênh giàu hơn. Nó học nét, góc, đường cong, hình dạng và ngữ cảnh cục bộ. Sau DenseNet, convolution $1\times1$ chiếu số kênh về `d_model=256`; output encoder trước GAT có dạng `[B,Hf,Wf,256]`.**
>
> **$H_f,W_f$ phụ thuộc kích thước ảnh và các bước stride/pooling, không phải một hằng số. Theo cấu trúc hiện tại, spatial size giảm qua convolution stride 2, max pooling và hai transition pooling.**

## 2.2. Giải thích chi tiết

DenseNet dùng dense connectivity:

$$ x_l=H_l([x_0,x_1,\ldots,x_{l-1}]) $$

Mỗi block nối feature cũ với feature mới theo channel, giúp:

- tái sử dụng đặc trưng;
- cải thiện gradient flow;
- giữ cả low-level và high-level cues.

Config:

- growth rate 24;
- `num_layers=16` cho mỗi dense block theo implementation;
- convolution đầu vào 1 channel;
- projection cuối về 256.

### Kích thước không gian

Spatial size giảm bởi:

1. conv stride 2;
2. max pool 2;
3. transition 1 average pool 2;
4. transition 2 average pool 2.

Xấp xỉ tổng downsampling khoảng 16 lần mỗi chiều, nhưng `ceil_mode=True` khiến công thức cần dùng ceil tại từng bước.

Nếu input $H_0 \times W_0$, có thể biểu diễn gần đúng:

$$ H_f \approx \left\lceil\frac{H_0}{16}\right\rceil, \quad W_f \approx \left\lceil\frac{W_0}{16}\right\rceil $$

Cần dùng một forward hook để ghi shape thật.

### Ý nghĩa của projection 1×1

DenseNet output channel không nhất thiết bằng `d_model`. Convolution $1\times1$:

- không thay đổi spatial size;
- trộn channel;
- đưa feature về cùng chiều với GAT và decoder.

## 2.4. Không nên nói

- “DenseNet phát hiện từng symbol.”
- “Output luôn có kích thước cố định.”
- “Mỗi feature cell tương ứng đúng một pixel.”
- “Downsampling không ảnh hưởng ký hiệu nhỏ.”

## 2.5. Bằng chứng cần chỉ ra

- `encoder.py`: `DenseNet`, `_Transition`, `feature_proj`.
- Log shape cho 5 kích thước ảnh đại diện.

---

# Câu 3 — Node trong graph hiện tại là ký hiệu, bounding box, pixel hay một ô trên feature map?

## 3.1. Bản trả lời nhanh

> **Node là một vector đặc trưng tại một ô của feature map sau DenseNet và projection. Nó không phải ký hiệu đã được phân đoạn, không phải bounding box và cũng không phải pixel ảnh gốc.**
>
> **Mỗi node có chiều 256 và receptive field bao phủ một vùng ảnh gốc. Một ký hiệu có thể trải trên nhiều node; một node cũng có thể chứa nét của nhiều ký hiệu gần nhau.**

## 3.2. Giải thích chi tiết

Feature map:

$$ F \in \mathbb{R}^{B\times H_f\times W_f\times D} $$

Flatten row-major:

$$ H \in \mathbb{R}^{B\times N\times D}, \quad N=H_fW_f $$

Node $i$ được ánh xạ từ tọa độ:

$$ y=\left\lfloor\frac{i}{W_f}\right\rfloor, \quad x=i\bmod W_f $$

Node feature chứa representation CNN của vùng receptive field. Nó không có:

- class symbol tường minh;
- bounding box;
- confidence detection;
- edge label superscript;
- ID ký hiệu.

### Hệ quả

Không thể trực tiếp hỏi:

> “Node này là chữ x hay số 2?”

trừ khi thêm probe hoặc visualization.

Graph hiện tại là latent feature graph. Nó học representation hỗ trợ decoder, không cung cấp giải thích symbol-level.

## 3.4. Không nên nói

- “Node là pixel.”
- “Node là ký hiệu.”
- “Một node luôn tương ứng một ký hiệu.”
- “Graph có ground-truth node label.”

## 3.5. Bằng chứng cần chỉ ra

- `feature.view(b, h*w, d)` trong `encoder.py`.
- Không có symbol detector/bounding box trong pipeline.

---

# Câu 4 — Decoder nhận đầu vào gì từ encoder và sinh token theo cơ chế nào?

## 4.1. Bản trả lời nhanh

> **Decoder nhận memory ảnh dạng `[B,Hf,Wf,D]` cùng mask. Nó flatten memory thành chuỗi $H_fW_f$ vector, embedding các target token, thêm word positional encoding và dùng causal self-attention cùng cross-attention tới memory ảnh. Cuối mỗi bước, linear projection tạo phân bố trên 113 token.**
>
> **Khi train, decoder dùng prefix ground truth; khi inference, beam search tự dùng token đã sinh.**

## 4.2. Giải thích chi tiết

### Memory ảnh

```python
src = rearrange(src, "b h w d -> (h w) b d")
src_mask = rearrange(src_mask, "b h w -> b (h w)")
```

### Target branch

- token embedding;
- LayerNorm;
- word positional encoding;
- causal mask;
- target padding mask.

Causal mask đảm bảo bước $t$ không nhìn token tương lai.

### Xác suất token

$$ p(y_t \mid y_{ \lt t},X) = \operatorname{softmax}(W_o z_t+b_o) $$

với vocabulary 113.

### Cross-attention

Query đến từ trạng thái decoder; key/value đến từ memory ảnh. Decoder học alignment giữa token đang sinh và vùng feature.

### Coverage/refinement

Config bật `cross_coverage` và `self_coverage`. Đây là phần TAMER hỗ trợ tinh chỉnh attention, không nên bỏ qua khi mô tả baseline.

### Train–test gap

Train dùng ground-truth prefix, inference dùng prediction prefix. Sai token sớm có thể lan sang các token sau; đây là exposure bias của autoregressive decoding.

## 4.4. Không nên nói

- “Decoder đọc graph trực tiếp dưới dạng edge list.”
- “Decoder sinh cả chuỗi một lần.”
- “Beam search được dùng trong training loss.”
- “Cross-attention tự động đảm bảo cú pháp.”

## 4.5. Bằng chứng cần chỉ ra

- `decoder.py`.
- `tamer.py`.
- `lit_tamer.py`.

---

# Câu 5 — Mô hình hiện tại có thật sự phát hiện và phân đoạn từng ký hiệu hay không?

## 5.1. Bản trả lời nhanh

> **Không. Core recognizer hiện tại không có object detector, bounding box prediction, NMS hay loss detection. Nó mã hóa toàn ảnh thành feature map và sinh LaTeX end-to-end.**
>
> **Vì vậy không được báo cáo detection precision/recall/F1 hoặc nói model xây symbol graph, trừ khi bổ sung một detector độc lập và đánh giá riêng.**

## 5.2. Giải thích chi tiết

Dấu hiệu của detector thường gồm:

- box regression;
- class logits theo box;
- anchors/queries;
- IoU;
- NMS;
- ground-truth bounding boxes;
- detection loss.

Repo core không có chuỗi đó.

DenseNet không “phát hiện symbol” theo nghĩa object detection. Nó có thể học feature phản ứng với ký hiệu, nhưng không xuất vị trí tường minh.

### End-to-end không phân đoạn

Ưu điểm:

- tránh lỗi phân đoạn tích lũy;
- không cần box annotation;
- pipeline gọn.

Nhược điểm:

- khó giải thích lỗi symbol;
- không có node symbol rõ;
- khó dùng edge relation tường minh;
- khó tách symbol recognition và structure parsing.

## 5.4. Không nên nói

- “DenseNet tự tạo bounding box.”
- “Node graph chính là symbol detector output.”
- “F1 detection của model là ...” nếu chưa có.

## 5.5. Bằng chứng cần chỉ ra

- Danh sách module và loss trong repo.
- Không có annotation bounding box trong dataloader.

---

# Câu 6 — Vì sao dùng DenseNet hoặc backbone hiện tại thay vì ResNet, ViT hay CNN đơn giản hơn?

## 6.1. Bản trả lời nhanh

> **DenseNet là backbone gốc mạnh và đã được dùng rộng trong HMER vì tái sử dụng feature, giữ gradient tốt và tạo feature map dày đặc cho ký hiệu nhỏ. Giữ DenseNet còn giúp cô lập tác động của GAT khi so M1–M5.**
>
> **Điều này không chứng minh DenseNet tối ưu hơn ResNet hoặc ViT. Muốn khẳng định, cần backbone ablation với cùng decoder, budget và preprocessing.**

## 6.2. Giải thích chi tiết

### Lý do phương pháp luận

Mục tiêu của luận văn là GAT/position. Nếu đồng thời đổi backbone:

- khó biết cải thiện đến từ đâu;
- mất baseline công bằng;
- tăng không gian hyperparameter.

### Lý do kỹ thuật

Dense connectivity:

- feature reuse;
- gradient flow;
- low-level stroke cues vẫn truyền sâu;
- phù hợp data tương đối nhỏ.

### So với ResNet

ResNet dùng additive skip; có thể:

- hiệu quả hơn;
- dễ dùng pretrained weights;
- có feature hierarchy khác.

Chưa có thí nghiệm nên không xếp hạng.

### So với ViT

ViT:

- global self-attention;
- cần patch/token positional design;
- thường cần dữ liệu/pretraining lớn;
- có thể trùng vai trò với GAT.

### So với CNN đơn giản

CNN nhỏ:

- nhanh hơn;
- ít overfit;
- nhưng có thể thiếu capacity.

### Thí nghiệm công bằng

Giữ:

- output resolution;
- d_model;
- parameter count gần nhau;
- decoder;
- train budget;
- seed;
- preprocessing.

## 6.4. Không nên nói

- “DenseNet tốt nhất.”
- “ResNet làm mất feature.”
- “ViT không phù hợp HMER.”
- “Dùng DenseNet vì paper dùng” mà không nói control rationale.

## 6.5. Bằng chứng cần chỉ ra

- Baseline M1.
- Parameter count và shape.
- Backbone ablation nếu có.

---

# Câu 7 — Feature map được flatten hoặc sắp xếp thành chuỗi/graph theo thứ tự nào?

## 7.1. Bản trả lời nhanh

> **Code rearrange feature thành `[B,H,W,D]` rồi dùng `view(B,H*W,D)`, nên node được sắp theo row-major: đi từ trái sang phải trong một hàng, rồi xuống hàng tiếp theo. Node index $i=yW+x$.**
>
> **Sau GAT, feature được reshape lại đúng $H\times W$ trước khi thêm 2D positional encoding và đưa cho decoder.**

## 7.2. Giải thích chi tiết

Ánh xạ:

$$ i=yW+x $$

Ngược lại:

$$ y=\lfloor i/W\rfloor, \quad x=i\bmod W $$

Adjacency dùng chính mapping này để tìm:

- phải: $i+1$;
- xuống: $i+W$;
- chéo phải dưới: $i+W+1$;
- chéo trái dưới: $i+W-1$.

Sau đó thêm cạnh ngược để graph đối xứng.

### Tại sao thứ tự flatten quan trọng?

- relative position index phụ thuộc mapping;
- decoder flatten memory cũng theo row-major;
- mask phải cùng thứ tự;
- sai reshape sẽ nối nhầm node.

### Cần unit test

Với grid $2\times3$, in node ID:

```text
0 1 2
3 4 5
```

Kiểm tra adjacency và relative direction bằng tay.

## 7.4. Không nên nói

- “Flatten tùy ý.”
- “Thứ tự không quan trọng vì có PE.”
- “Graph tự biết tọa độ mà không cần H,W.”

## 7.5. Bằng chứng cần chỉ ra

- `rearrange` và `view` trong `encoder.py`.
- `_build_grid_adjacency`.
- `decoder.py` flatten memory.

---

# Câu 8 — Thông tin padding và mask được truyền qua encoder và decoder như thế nào?

## 8.1. Bản trả lời nhanh

> **Mask ban đầu đánh dấu vùng padding của ảnh. DenseNet downsample mask song song với các bước stride/pooling. Encoder dùng mask đã giảm kích thước để xóa cạnh tới node padding trong adjacency. Decoder flatten mask thành `memory_key_padding_mask`, nên cross-attention không xem vùng padding như memory hợp lệ. Target có mask riêng dựa trên PAD token.**
>
> **Self-loop được tái áp dụng cả cho node padding để tránh một hàng attention toàn $-\infty$ gây NaN, nhưng decoder vẫn mask các node đó ở memory.**

## 8.2. Giải thích chi tiết

### Image mask

Quy ước code:

- 0: valid;
- 1: padding.

Mask được lấy mẫu cách 2 sau conv/pool để theo spatial feature.

### Graph mask

Adjacency ban đầu giống nhau cho mọi sample cùng shape. Sau đó:

- xóa hàng của node padding;
- xóa cột của node padding;
- tái thêm diagonal self-loop.

Mục tiêu:

- valid node không nhận message từ padding;
- tránh softmax trên hàng không có cạnh.

### Decoder memory mask

```python
memory_key_padding_mask = src_mask
```

Cross-attention bỏ memory padding.

### Target mask

```python
tgt_pad_mask = tgt == PAD_IDX
```

Causal mask và padding mask làm hai việc khác nhau:

- causal: không nhìn tương lai;
- padding: không nhìn token giả.

### Rủi ro cần kiểm tra

- downsample mask bằng slicing có khớp chính xác `ceil_mode` không;
- self-loop padding có tạo feature không mong muốn trước decoder mask không;
- BatchNorm/CNN vẫn xử lý pixel pad trước mask.

## 8.4. Không nên nói

- “Mask được dùng từ pixel đầu tiên để CNN bỏ padding hoàn toàn.”
- “Self-loop padding là lỗi chắc chắn.”
- “Causal mask và padding mask là một.”

## 8.5. Bằng chứng cần chỉ ra

- `datamodule.py`.
- `DenseNet.forward`.
- `_build_grid_adjacency`.
- `decoder.py`.

---

# Câu 9 — Điểm nghẽn tính toán và bộ nhớ lớn nhất của pipeline nằm ở module nào?

## 9.1. Bản trả lời nhanh

> **Ứng viên lớn nhất là attention có tensor bậc hai theo số node: GAT tạo score `[B,heads,N,N]`, và Transformer decoder cross-attend giữa token và $N$ memory positions. Dù graph logic chỉ có tối đa 8 láng giềng, implementation hiện vẫn tính ma trận $N\times N$ rồi mask, nên chưa tận dụng sparse computation.**
>
> **Beam search còn nhân chi phí decoder theo beam size 10. Muốn kết luận định lượng phải profile peak VRAM và thời gian từng module.**

## 9.2. Giải thích chi tiết

### GAT complexity

$N=H_fW_f$.

Attention score:

$$ O(BKN^2) $$

bộ nhớ gần:

$$ B\times K\times N\times N $$

Dù số edge hợp lệ $E\approx9N$, code dense vẫn cấp phát toàn cặp.

### Decoder

Self-attention target:

$$ O(L^2) $$

Cross-attention:

$$ O(LN) $$

Beam search gần tăng theo beam size, dù có cache hay batching.

### CNN

DenseNet có nhiều convolution và activation, đặc biệt ở high-resolution stage. Khi train, activation phải giữ cho backward.

### Cách profile

- `torch.profiler`;
- `torch.cuda.max_memory_allocated`;
- NVTX range;
- batch theo aspect ratio;
- benchmark input ngắn/dài.

Báo cáo:

| Module | ms/sample | Peak VRAM | % time |
|---|---:|---:|---:|
| Preprocess | | | |
| DenseNet | | | |
| Graph build | | | |
| GAT | | | |
| Decoder greedy | | | |
| Decoder beam 10 | | | |

### Hướng tối ưu

- sparse GAT edge list;
- local-window tensor;
- flash attention nếu phù hợp;
- cache relative index;
- dynamic beam;
- quantization;
- lower-resolution có kiểm soát.

## 9.4. Không nên nói

- “Graph thưa nên GAT là $O(N)$ trong code hiện tại.”
- “DenseNet chắc chắn là bottleneck.”
- “Beam search không ảnh hưởng latency.”

## 9.5. Bằng chứng cần chỉ ra

- Shape `e=[B,K,N,N]` trong `gat.py`.
- Profile thực tế trên GPU dùng triển khai.

---

# Câu 10 — Nếu không có symbol detector, có được gọi graph hiện tại là Symbol Layout Graph hay không?

## 10.1. Bản trả lời nhanh

> **Không nên. Symbol Layout Graph hàm ý node là ký hiệu và edge là quan hệ bố cục giữa ký hiệu. Graph hiện tại có node là feature cell và edge là lân cận grid. Tên chính xác là feature-grid graph, spatial feature graph hoặc 8-connected graph trên feature map.**
>
> **Gọi sai sẽ khiến hội đồng hỏi bounding box, node label và edge relation mà code không có.**

## 10.2. Giải thích chi tiết

### Điều kiện để gọi Symbol Layout Graph

Thông thường cần:

- symbol instances;
- bounding box hoặc stroke grouping;
- node identity;
- relation label như right/sup/sub/above/below/inside;
- graph construction theo geometry của symbol.

Graph hiện tại chỉ có:

- tọa độ grid;
- feature vector;
- 8-neighbor;
- self-loop;
- relative direction 9 trạng thái.

Nó có thể hỗ trợ học layout một cách ẩn, nhưng không phải layout graph tường minh.

### Tên gọi nên thống nhất

Trong luận văn và slide:

> **Feature-Grid Graph trên feature map DenseNet**

Lần đầu định nghĩa:

> Mỗi vị trí không gian trên feature map là một node; các node kề theo 8 hướng được nối cạnh.

### Symbol-level graph là hướng phát triển

Có thể nói:

> Trong tương lai, detector/segmenter sẽ tạo symbol nodes và relation classifier tạo semantic edges.

## 10.4. Không nên nói

- “Symbol graph nhưng node là feature.”
- “Edge above/below” khi chỉ có dy/dx.
- “GAT phân loại quan hệ symbol.”

## 10.5. Bằng chứng cần chỉ ra

- `encoder.py`, `gat.py`.
- So sánh sơ đồ chuyên đề cũ và repo mới.

---

# Phụ lục — Shape table nên có trong slide

| Bước | Shape |
|---|---|
| Ảnh | `[B,1,H0,W0]` |
| Mask ảnh | `[B,H0,W0]` |
| DenseNet output | `[B,C,Hf,Wf]` |
| Projection | `[B,256,Hf,Wf]` |
| Grid feature | `[B,Hf,Wf,256]` |
| Node feature | `[B,N,256]` |
| Adjacency | `[B,N,N]` |
| Decoder memory | `[Hf*Wf,B,256]` |
| Token logits | `[B,L,113]` |

# Nguồn đối chiếu

- `tamer/model/encoder.py`
- `tamer/model/gat.py`
- `tamer/model/decoder.py`
- `tamer/model/tamer.py`
- `tamer/lit_tamer.py`
- `config/crohme.yaml`
