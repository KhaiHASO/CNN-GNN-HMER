# NHÓM 12 — HẠN CHẾ, TÍNH TÁI LẬP, ỨNG DỤNG VÀ HƯỚNG PHÁT TRIỂN

> **Mục tiêu nhóm:** Thể hiện sự trung thực khoa học, khả năng tái lập và kế hoạch phát triển có căn cứ.

---

# Câu 1 — Ba hạn chế kỹ thuật quan trọng nhất của mô hình hiện tại là gì?

## 1.1. Bản trả lời nhanh

> **Thứ nhất, graph hiện là feature-grid graph cục bộ, không phải symbol graph có quan hệ toán học tường minh; vì vậy GAT chỉ học tương quan không gian ẩn. Thứ hai, GAT implementation dùng attention dense $N\times N$ dù topology chỉ 8-neighbor, nên chi phí bộ nhớ tăng bậc hai theo số node. Thứ ba, protocol dữ liệu và đánh giá còn hạn chế: CROHME 2014 đang được dùng chọn checkpoint, mới có ít seed và chưa có error analysis đầy đủ theo cấu trúc/domain.**
>
> **Ngoài ra demo còn domain gap với dữ liệu CROHME và M4 chưa vượt baseline về Exact Match trung bình.**

## 1.2. Hạn chế 1 — Biểu diễn graph chưa tường minh semantics

Node là feature cell, edge là neighbor direction.

Không có:

- symbol instance;
- bounding box;
- edge superscript/subscript;
- relation supervision;
- graph output để kiểm tra.

Hệ quả:

- khó giải thích;
- khó chứng minh “hiểu cấu trúc”;
- long-range relation phụ thuộc CNN/decoder;
- một symbol trải nhiều node.

### Cách nói

> **Mô hình đưa spatial inductive bias cục bộ chứ chưa xây Symbol Layout Graph.**

## 1.3. Hạn chế 2 — Hiệu quả tính toán

Với:

$$ N=H_fW_f $$

code tạo score:

$$ [B,K,N,N] $$

nên:

$$ O(KN^2) $$

trong khi edge logic chỉ:

$$ E\approx9N $$

Hệ quả:

- tăng resolution rất tốn VRAM;
- batch nhỏ;
- latency cao;
- khó scale biểu thức dài.

### Hướng sửa

- sparse edge index;
- gather 9-neighbor;
- scatter softmax;
- hierarchical graph;
- local window kernel.

## 1.4. Hạn chế 3 — Protocol và bằng chứng thực nghiệm

- validation/test chưa tách sạch;
- một seed;
- M4 thay nhiều biến;
- chưa significance test;
- chưa thống kê dataset;
- chưa benchmark latency;
- chưa external-domain test;
- normalization chưa được đặc tả rõ.

Hạn chế này quan trọng ngang kiến trúc vì nó ảnh hưởng độ tin cậy kết luận.

## 1.5. Những hạn chế phụ

- output vocabulary cố định 113;
- không có UNK;
- max decode length 150;
- ảnh nhỏ có thể mất qua downsampling;
- demo preprocessing chưa chắc parity;
- no grammar-constrained decoding;
- không có confidence/OOD;
- M1 vẫn cao hơn M4 về Avg ExpRate.

## 1.6. Không nên nói

- “Hạn chế chính chỉ là GPU.”
- “Mô hình không có hạn chế về dữ liệu.”
- “M4 tốt hơn nhưng chưa train đủ.”
- “Graph hiện đã biểu diễn đầy đủ cấu trúc toán.”

---

# Câu 2 — Người khác cần những file, seed, checkpoint và lệnh nào để tái lập kết quả?

## 2.1. Bản trả lời nhanh

> **Một run tái lập cần commit hash, environment lock, checksum dữ liệu, config sau mọi override, seed, lệnh train, checkpoint hash, lệnh evaluate, vocabulary và output prediction. Chỉ chia sẻ file `.ckpt` hoặc README là chưa đủ.**
>
> **Mỗi M1–M5 nên có manifest riêng và một lệnh chạy end-to-end.**

## 2.2. Bộ artifact tối thiểu

```text
repro/
├── README_REPRO.md
├── environment.yml
├── requirements-lock.txt
├── data_manifest.json
├── vocabulary.txt
├── configs/
│   ├── m1.yaml
│   ├── m2.yaml
│   ├── m3.yaml
│   ├── m4.yaml
│   └── m5.yaml
├── checkpoints/
│   └── SHA256SUMS
├── scripts/
│   ├── train.sh
│   ├── eval_2014.sh
│   ├── eval_2016.sh
│   └── eval_2019.sh
├── outputs/
│   ├── predictions.json
│   └── metrics.json
└── run_manifest.json
```

## 2.3. Run manifest cần có

```json
{
  "git_commit": "...",
  "data_sha256": "...",
  "seed": 7,
  "gpu": "NVIDIA T4",
  "cuda": "...",
  "pytorch": "...",
  "lightning": "...",
  "config": "configs/m4.yaml",
  "train_command": "...",
  "best_epoch": 0,
  "checkpoint_sha256": "...",
  "beam_size": 10,
  "max_len": 150
}
```

## 2.4. Seed và determinism

Cần ghi:

- Python seed;
- NumPy;
- torch CPU/GPU;
- dataloader workers;
- deterministic algorithms;
- cuDNN flags.

Nên có nhiều seed và danh sách cố định:

```text
7, 17, 27
```

## 2.5. Dữ liệu

Do dataset có thể không được phép phân phối lại, cung cấp:

- nguồn tải;
- script chuẩn bị;
- checksum;
- count;
- audit statistics;
- expected directory tree.

## 2.6. Lệnh mẫu

```bash
python train.py --config configs/m4.yaml --seed 7
python eval.py --config configs/m4.yaml \
  --checkpoint checkpoints/m4_seed7.ckpt \
  --test-folder 2014
```

Lệnh thực tế phải khớp entry point repo.

## 2.7. Không nên nói

- “Clone repo là tái lập được.”
- “Checkpoint đủ rồi.”
- “Seed 7 đảm bảo giống 100%.”
- “Dữ liệu CROHME nào cũng giống nhau.”

---

# Câu 3 — Ứng dụng thực tế nào phù hợp với mức độ chính xác hiện tại và ứng dụng nào chưa phù hợp?

## 3.1. Bản trả lời nhanh

> **Với khoảng một nửa biểu thức đúng hoàn toàn, hệ thống phù hợp hơn với trợ lý nhập công thức có preview và người dùng sửa, công cụ nghiên cứu hoặc tiền xử lý bán tự động. Nó chưa phù hợp cho chấm điểm tự động hoàn toàn, nhập dữ liệu y tế/kỹ thuật không hậu kiểm, hoặc số hóa hàng loạt yêu cầu độ chính xác gần tuyệt đối.**
>
> **M4 MED thấp có thể hữu ích cho human-in-the-loop, nhưng vẫn cần confidence và giao diện sửa.**

## 3.2. Ứng dụng phù hợp

### Human-in-the-loop editor

- dự đoán LaTeX;
- render preview;
- highlight token confidence thấp;
- người dùng sửa;
- lưu correction để cải thiện model.

### Trợ lý giảng viên/nghiên cứu

- nhập công thức nhanh;
- số hóa bài tập đơn giản;
- tạo draft.

### Công cụ benchmark/nghiên cứu

- thử graph architecture;
- error analysis;
- education demo có disclaimer.

### Retrieval hỗ trợ

Dùng prediction làm candidate search, không làm ground truth cuối.

## 3.3. Ứng dụng chưa phù hợp

- chấm bài tự động hoàn toàn;
- quyết định điểm số không review;
- nhập công thức an toàn-critical;
- xử lý nguyên trang phức tạp;
- mọi ký hiệu/macro ngoài vocabulary;
- camera in-the-wild không preprocessing;
- real-time mobile nếu chưa benchmark.

## 3.4. Tiêu chí triển khai

Không chỉ accuracy:

- latency;
- uptime;
- confidence;
- OOD detection;
- correction UX;
- privacy;
- audit log;
- failure fallback.

### Cách định vị trung thực

> **Prototype hỗ trợ người dùng, chưa phải hệ thống tự động đáng tin cậy hoàn toàn.**

---

# Câu 4 — Hướng phát triển nào giải quyết trực tiếp hạn chế hiện tại thay vì chỉ nói chung chung “thêm dữ liệu, thêm GPU”?

## 4.1. Bản trả lời nhanh

> **Mỗi hướng phải gắn một hạn chế và metric. Với graph dense, chuyển sang sparse 9-neighbor attention và đo VRAM/latency. Với lỗi vị trí, tách ablation relative bias và dùng multi-scale/high-resolution feature. Với syntax, thêm grammar-constrained decoding. Với domain gap, xây parity pipeline và targeted augmentation. Với protocol, tạo validation nội bộ và nhiều seed.**
>
> **Đó là kế hoạch có giả thuyết, không phải khẩu hiệu thêm dữ liệu/GPU.**

## 4.2. Bảng hạn chế → giải pháp → phép đo

| Hạn chế | Hướng trực tiếp | Metric |
|---|---|---|
| Dense $N^2$ GAT | Sparse neighbor gather | VRAM, latency, ExpRate |
| Ký hiệu nhỏ mất | High-res branch/multi-scale | bound recall |
| Graph thiếu semantics | Symbol/region proposal + relation edge | relation F1, ExpRate |
| Syntax sai | Grammar-constrained beam | SER, ExpRate, latency |
| Domain gap | Unified preprocessing + targeted augmentation | demo set ExpRate |
| One seed | Multi-seed protocol | mean ± std |
| 2014 dùng validation | Internal validation split | untouched test score |
| M4 confounded | Factorial ablation | causal attribution |

## 4.3. Hướng ưu tiên cao

1. Sửa protocol.
2. Audit dataset và integral subset.
3. Ablation tách M4.
4. Sparse GAT.
5. Demo parity.
6. Grammar constraints.
7. External-domain evaluation.

## 4.4. Không nên nói

- “Thêm nhiều layer.”
- “Train lâu hơn.”
- “GPU mạnh hơn.”
- “Thêm dữ liệu” mà không nói loại dữ liệu và failure mode.

---

# Câu 5 — Nếu chuyển từ grid graph sang symbol-level graph, cần bổ sung những module và nhãn nào?

## 5.1. Bản trả lời nhanh

> **Cần một module tạo symbol instance — từ stroke grouping, segmentation hoặc detector — để có bounding box và symbol feature. Sau đó cần graph construction hoặc relation classifier cho các cạnh như right, superscript, subscript, above, below, inside. Dataset phải có node/edge annotation hoặc cách sinh pseudo-label đáng tin cậy.**
>
> **Đây là thay đổi lớn, không chỉ đổi adjacency.**

## 5.2. Pipeline symbol-level

```text
Image/strokes
→ symbol proposal/detection
→ ROI feature
→ symbol classification
→ relation candidate generation
→ relation classification
→ symbol layout graph
→ graph encoder
→ tree/LaTeX decoder
```

## 5.3. Nhãn cần có

### Node

- symbol class;
- bounding box;
- stroke IDs;
- confidence;
- optional role.

### Edge

- right;
- superscript;
- subscript;
- above;
- below;
- inside;
- numerator;
- denominator;
- root content;
- next relation;
- no-relation.

Phải thống nhất ontology với CROHME relation grammar.

## 5.4. Module cần bổ sung

- detector/segmenter;
- NMS hoặc set prediction;
- ROIAlign;
- relation candidate pruning;
- edge classifier;
- graph batching;
- graph-to-sequence/tree decoder;
- loss node/box/edge;
- matching giữa prediction và GT.

## 5.5. Metric

- box AP/F1;
- symbol classification;
- edge relation F1;
- graph edit distance;
- expression ExpRate;
- end-to-end latency.

## 5.6. Rủi ro

- error propagation;
- ký hiệu chạm/dính;
- dấu phân số/căn khó box;
- annotation tốn kém;
- symbol graph không còn fully end-to-end nếu heuristic.

## 5.7. Không nên nói

- “Chỉ cần lấy mỗi grid node làm symbol.”
- “CROHME caption đủ làm edge label.”
- “Symbol graph chắc chắn tốt hơn.”
- “Detector có thể dùng F1 chung chung” không nêu IoU/protocol.

---

# Câu 6 — Nếu sử dụng grammar-constrained decoding, metric nào có thể tăng và rủi ro nào có thể xuất hiện?

## 6.1. Bản trả lời nhanh

> **Grammar constraints có thể giảm Syntax Error Rate, giảm ngoặc thiếu/thừa và có thể tăng ExpRate nếu nhiều lỗi hiện tại là chuỗi không hợp lệ. Nó không sửa được ký hiệu nhìn sai và có thể làm giảm accuracy nếu grammar không bao phủ annotation hoặc loại bỏ hypothesis đúng.**
>
> **Chi phí beam và độ phức tạp triển khai cũng tăng.**

## 6.2. Các dạng constraint

### Hard constraint

Mask token không hợp lệ:

$$ p'(y_t)=0 $$

cho transition vi phạm grammar.

### Soft constraint

Cộng penalty hoặc rerank score:

$$ S(Y) = \log P_\theta(Y\mid X) - \lambda C_{\text{grammar}}(Y) $$

### Incremental parser

Theo dõi:

- brace stack;
- macro arity;
- allowed token state;
- EOS legality.

## 6.3. Metric có thể cải thiện

- Syntax Error Rate ↓;
- bracket match ↑;
- ExpRate có thể ↑;
- MED có thể ↓;
- valid render rate ↑.

### Rủi ro

- grammar mismatch;
- canonical form hạn chế;
- valid nhưng sai semantic;
- search chậm;
- hard constraint loại đúng;
- error bị chuyển sang token hợp lệ nhưng sai.

### Thí nghiệm

So:

- unconstrained;
- brace-only;
- full grammar;
- soft rerank.

Báo cáo accuracy, syntax, latency và failure cases.

## 6.4. Không nên nói

- “Grammar đảm bảo recognition đúng.”
- “Syntax hợp lệ nghĩa là biểu thức đúng.”
- “Constraint luôn tăng ExpRate.”
- “Parser không ảnh hưởng beam latency.”

---

# Câu 7 — Làm sao đánh giá thời gian suy luận, bộ nhớ và khả năng triển khai thời gian thực?

## 7.1. Bản trả lời nhanh

> **Phải benchmark end-to-end và từng module trên phần cứng xác định, có warm-up và đồng bộ CUDA. Báo cáo median, P95 latency, throughput, peak VRAM theo kích thước ảnh và beam size. Không chỉ đo một ảnh thuận lợi.**
>
> **Khả năng real-time phải gắn SLA cụ thể, ví dụ dưới 200 ms/biểu thức trên GPU mục tiêu.**

## 7.2. Protocol benchmark

### Phần cứng và phần mềm

- GPU/CPU;
- CUDA;
- PyTorch;
- precision;
- batch;
- power mode.

### Warm-up

Chạy 20–100 iterations trước đo.

### Đồng bộ

```python
torch.cuda.synchronize()
start = time.perf_counter()
...
torch.cuda.synchronize()
```

### Kích thước test

- ảnh ngắn;
- median;
- P95 width;
- max allowed;
- integral/matrix;
- batch 1 và batch nhiều.

### Các phần cần đo

```text
decode/upload
preprocess
DenseNet
graph build
GAT
decoder beam
postprocess/render
total
```

### Metric

- mean/median/P95/P99;
- samples/s;
- peak allocated/reserved VRAM;
- CPU RAM;
- model size;
- startup time.

## 7.3. Real-time không chỉ latency model

Ứng dụng còn:

- network;
- queue;
- image decode;
- rendering;
- concurrency;
- cold start.

### Stress test

- concurrent users;
- long expressions;
- timeout;
- OOM recovery.

## 7.4. So sánh beam

Beam 10 có thể tốn hơn greedy. Nên vẽ:

$$ \text{Accuracy versus latency} $$

cho beam 1/5/10/20.

## 7.5. Không nên nói

- “Inference gần như tức thì.”
- “Chạy được trên T4 nên real-time.”
- “Đo một lần bằng `time.time()` là đủ.”
- “Graph thưa nên nhanh” khi code dense.

---

# Câu 8 — Nếu có thêm ba tháng nghiên cứu, thí nghiệm nào nên ưu tiên để tăng sức thuyết phục của luận văn?

## 8.1. Bản trả lời nhanh

> **Ưu tiên không phải xây thêm nhiều model. Tháng đầu sửa protocol và audit dữ liệu; tháng hai chạy ablation tách M4 cùng nhiều seed; tháng ba làm error analysis tích phân/demo và benchmark sparse/latency. Những việc này trực tiếp tăng độ tin cậy của đóng góp hiện tại.**

## 8.2. Kế hoạch 12 tuần

### Tuần 1–2 — Reproducibility và dữ liệu

- validation nội bộ;
- freeze test protocol;
- dataset checksum;
- length/token/structure audit;
- integral-bound count;
- OOV/overlap audit.

### Tuần 3–4 — Baseline sạch

- retrain M1;
- verify M3/M4;
- 3 seed;
- save manifests;
- confidence intervals.

### Tuần 5–7 — Factorial ablation

- 4 vs 8 neighbor;
- bias off/on;
- 1L vs 2L;
- 4H vs 8H;
- parameter count;
- same training budget.

### Tuần 8 — Mechanism probes

- coordinate linear probe;
- node similarity;
- attention/occlusion;
- oracle beam.

### Tuần 9–10 — Error analysis

- length;
- structure;
- rare tokens;
- integral bounds;
- demo parity;
- controlled perturbation.

### Tuần 11 — Efficiency

- profile;
- sparse 9-neighbor prototype;
- beam latency trade-off.

### Tuần 12 — Consolidation

- final tables;
- figures;
- scripts;
- README reproduction;
- defense Q&A.

## 8.3. Thứ tự ưu tiên theo giá trị khoa học

1. Protocol sạch.
2. Nhiều seed.
3. Ablation tách biến.
4. Dataset/error analysis.
5. Mechanism probes.
6. Efficiency.
7. App polish.

App đẹp không bù được thí nghiệm thiếu kiểm soát.

## 8.4. Deliverables cuối ba tháng

- bảng mean ± std;
- M1–M5 configs;
- dataset audit;
- integral benchmark;
- sparse GAT profile;
- normalization spec;
- reproducibility package;
- updated thesis narrative.

## 8.5. Không nên nói

- “Em sẽ train thêm 1.000 epoch.”
- “Em sẽ thêm mọi dataset.”
- “Em sẽ đổi toàn bộ sang model mới.”
- “Em sẽ làm app trước rồi mới sửa evaluation.”

---

# Phụ lục A — Bản trả lời tổng hợp khoảng hai phút

> **Ba hạn chế lớn nhất là graph mới ở mức feature-grid, GAT dense có chi phí $N^2$, và protocol/bằng chứng thực nghiệm chưa đủ sạch do validation 2014, ít seed và M4 thay nhiều biến. Vì vậy hướng phát triển phải trực tiếp: sparse local attention cho chi phí, symbol/relation graph hoặc probe cho semantics, internal validation và multi-seed cho độ tin cậy, high-resolution/domain adaptation cho lỗi demo.**
>
> **Để tái lập, người khác cần commit, environment, checksum dữ liệu, config sau override, seed, lệnh train/eval, checkpoint hash và prediction files. Với độ chính xác hiện tại, hệ thống phù hợp trợ lý nhập công thức có người hậu kiểm hơn là chấm điểm tự động hoàn toàn.**
>
> **Grammar-constrained decoding có thể giảm syntax error nhưng không sửa visual recognition và có rủi ro grammar mismatch. Real-time phải được benchmark bằng median/P95 latency, throughput và peak VRAM trên phần cứng mục tiêu. Nếu có thêm ba tháng, em ưu tiên sửa protocol, chạy nhiều seed và ablation tách M4, rồi mới làm error analysis tích phân và tối ưu sparse GAT.**

# Phụ lục B — Checklist tái lập và triển khai

- [ ] Commit/tag từng model.
- [ ] Lock dependencies.
- [ ] Data checksum.
- [ ] Config sau override.
- [ ] Seed list.
- [ ] Checkpoint SHA-256.
- [ ] Prediction và metric JSON.
- [ ] Internal validation protocol.
- [ ] Multi-seed results.
- [ ] Latency/VRAM benchmark.
- [ ] Demo parity tests.
- [ ] Failure fallback và confidence.
- [ ] Application disclaimer.
