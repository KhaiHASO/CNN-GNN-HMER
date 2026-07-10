# NHÓM 8 — HUẤN LUYỆN VÀ HYPERPARAMETER

> **Mục tiêu nhóm:** Nắm rõ cấu hình train thực tế, lý do chọn siêu tham số và cách nhận diện underfitting, overfitting, bất ổn tối ưu.

---

## 0. Cấu hình mặc định đang thể hiện trong repo

| Thành phần | Giá trị |
|---|---|
| Seed | 7 |
| Deterministic | `true` |
| GPU | 1 |
| Precision | 16-bit mixed precision |
| Max epochs | 100 |
| Validation frequency | Mỗi 2 epoch |
| Checkpoint monitor | `val_ExpRate` |
| Train batch size | 8 |
| Eval batch size | 2 |
| Optimizer thực tế | Adadelta |
| Learning rate | 1,0 |
| Epsilon | $10^{-6}$ |
| Weight decay | $10^{-4}$ |
| Scheduler | MultiStepLR |
| Milestones | 300, 350 |
| Gamma | 0,1 |
| Gradient accumulation | Không thấy cấu hình, mặc định 1 nếu không override |
| Early stopping | Không thấy callback active |
| d_model | 256 |
| Decoder layers | 3 |
| Decoder heads | 8 |
| Decoder dropout | 0,3 |
| GAT dropout | 0,2 trong config chính |
| Beam size | 10 |
| Max decode length | 150 |

> **Cảnh báo:** Đây là cấu hình đang commit. Báo cáo M1–M5 phải lấy config và log lưu cùng từng run, vì notebook/Kaggle có thể override.

---

# Câu 1 — Optimizer, learning rate, scheduler và weight decay thực tế trong repo là gì?

## 1.1. Bản trả lời nhanh

> **Code `configure_optimizers()` hiện dùng Adadelta với learning rate lấy từ config, mặc định 1,0; epsilon $10^{-6}$ và weight decay $10^{-4}$. Scheduler là MultiStepLR, gamma 0,1 tại milestone 300 và 350. AdamW có xuất hiện trong code nhưng đang bị comment, nên không phải optimizer thực tế của run theo code hiện tại.**
>
> **Với max epoch 100, milestone 300/350 không bao giờ được chạm nếu không có override; do đó learning rate thực tế có thể giữ nguyên suốt run. Đây là điểm cần nêu và kiểm tra log.**

## 1.2. Công thức cập nhật và vai trò

Adadelta điều chỉnh bước cập nhật theo lịch sử bình phương gradient và update. Trực giác:

- không yêu cầu thiết kế decay schedule phức tạp ngay từ đầu;
- từng tham số có effective step size thích nghi;
- từng được dùng trong nhiều hệ HMER encoder–decoder.

Weight decay thêm regularization:

$$ \theta \leftarrow \theta-\eta\lambda\theta $$

với:

$$ \lambda=10^{-4} $$

### Scheduler

MultiStepLR:

$$ \eta_t = \eta_0\gamma^{m(t)} $$

trong đó $m(t)$ là số milestone đã vượt.

Nhưng nếu:

$$ t_{\max}=100 \lt 300 $$

thì:

$$ m(t)=0 $$

và scheduler không giảm LR.

## 1.3. Điều phải xác minh

- Log learning rate từng epoch.
- Run M1–M5 có max epoch >350 hay không.
- Notebook có override milestones không.
- Best checkpoint ở epoch nào.
- Có resume training làm epoch global vượt 300 không.

## 1.4. Không nên nói

- “Optimizer là AdamW.”
- “Scheduler giảm LR khi validation không cải thiện.”
- “Learning rate đã giảm ở epoch 30/35.”
- “Adadelta chắc chắn tốt nhất.”

---

# Câu 2 — Batch size train, validation và gradient accumulation được thiết lập ra sao?

## 2.1. Bản trả lời nhanh

> **Config hiện đặt train batch size 8 và eval batch size 2. Tuy nhiên dataloader còn gom batch động theo kích thước ảnh để kiểm soát tổng diện tích, nên số expression thực tế trong một batch có thể phụ thuộc shape và giới hạn bộ nhớ. Repo không thấy cấu hình gradient accumulation, nên mặc định là 1 nếu Lightning không được override ở lệnh chạy.**

## 2.2. Batch size danh nghĩa và batch động

Trong HMER, ảnh có width rất khác nhau. Nếu chỉ dùng batch count cố định:

- batch ảnh dài dễ OOM;
- batch ảnh ngắn lãng phí VRAM ít hơn.

Repo sort/gom theo diện tích và giới hạn:

$$ A_{\max}\times B $$

nên “batch size 8” là giới hạn tối đa hoặc tham số mục tiêu, không nhất thiết mọi step đều có đúng 8 expression.

### Effective batch size

Nếu có gradient accumulation $G$:

$$ B_{\text{effective}} = B_{\text{per GPU}}\times N_{\text{GPU}}\times G $$

Với config hiện thấy:

$$ 8\times1\times1=8 $$

nhưng phải kiểm tra run logs.

### Validation batch 2

Beam search và memory ảnh làm evaluation tốn VRAM, nên batch nhỏ hơn train có thể hợp lý.

## 2.3. Câu hỏi truy tiếp

**“Batch nhỏ có làm kết quả kém không?”**

> Có thể làm gradient nhiễu hơn, nhưng không thể quy toàn bộ kết quả thấp cho batch size. Adadelta, BatchNorm behavior, gradient variance và data ordering đều liên quan. Cần thử accumulation hoặc batch khác trong controlled experiment.

## 2.4. Không nên nói

- “Mọi batch đúng 8 mẫu.”
- “Không có accumulation chắc chắn.”
- “Batch nhỏ là nguyên nhân chính.”
- “Hai GPU thì batch tự động gấp đôi” nếu DDP/config chưa xác nhận.

---

# Câu 3 — Checkpoint tốt nhất được chọn theo metric nào và vì sao?

## 3.1. Bản trả lời nhanh

> **Checkpoint được monitor theo `val_ExpRate`, mode `max`, giữ top 1. Lý do hợp lý là ExpRate là metric mục tiêu cuối cùng: toàn chuỗi LaTeX phải đúng.**
>
> **Nhưng repo hiện dùng `test_folder: 2014` cho cả validation và test, nên chọn checkpoint theo CROHME 2014 làm score 2014 không còn hoàn toàn độc lập. Đây là model-selection contamination cần sửa hoặc khai báo.**

## 3.2. Vì sao không chọn train loss?

Loss:

- differentiable;
- dùng để tối ưu;
- không đồng nhất với Exact Match.

Một model loss thấp hơn có thể phân bố xác suất tốt hơn nhưng beam output vẫn không tăng exact sequence.

Checkpoint theo ExpRate phù hợp mục tiêu application, nhưng metric có thể dao động mạnh. Có thể cân nhắc:

- val ExpRate chính;
- val loss phụ;
- Mean Edit Distance phụ;
- patience/smoothing nếu nhiều seed.

## 3.3. Rủi ro selection bias

Nếu mỗi model chọn checkpoint tốt nhất trên cùng test benchmark, rồi so score đó:

- score bị optimistic;
- đặc biệt khi đánh giá nhiều epoch;
- càng nhiều trial, selection bias càng cao.

Protocol tốt:

- validation nội bộ;
- test một lần sau chốt.

## 3.4. Không nên nói

- “Checkpoint chọn theo validation loss.”
- “2014 hoàn toàn untouched.”
- “Top checkpoint cao nhất chắc chắn tổng quát nhất.”

---

# Câu 4 — Số epoch tối đa, early stopping và tần suất đánh giá được cấu hình thế nào?

## 4.1. Bản trả lời nhanh

> **Config hiện đặt tối đa 100 epoch và chạy validation mỗi 2 epoch. Repo không thấy EarlyStopping callback active; có các dòng liên quan patience/ReduceLROnPlateau nhưng không phải scheduler đang chạy. Vì vậy model thường train tới max epoch, đồng thời giữ checkpoint có `val_ExpRate` cao nhất.**
>
> **Nếu run thực tế có 150 epoch hoặc early stopping khác, phải lấy từ log của run, không lấy từ chuyên đề cũ hoặc trí nhớ.**

## 4.2. Phân biệt ba cơ chế

### Max epochs

Giới hạn cứng số epoch.

### Checkpointing

Lưu model tốt nhất nhưng không dừng train.

### Early stopping

Dừng khi metric không cải thiện trong patience.

Repo hiện có checkpointing, chưa thấy early stopping active.

### Validation every 2 epochs

Nếu max 100:

- tối đa khoảng 50 lần validation;
- best checkpoint nằm ở một trong các mốc đó;
- metric giữa hai mốc không được đo.

## 4.3. Có phải 100 epoch là ít?

Không thể kết luận chỉ từ số epoch. Cần xem:

- train loss còn giảm không;
- val ExpRate plateau chưa;
- LR có thay đổi không;
- overfit gap;
- gradient norm;
- best epoch.

100 epoch với LR không decay có thể khác hoàn toàn 100 epoch với schedule tốt.

## 4.4. Không nên nói

- “Có early stopping 20 epoch.”
- “Train 100 epoch vì model đã hội tụ.”
- “Nhiều epoch luôn tốt hơn.”
- “150 epoch chắc chắn là run thực tế” nếu log không có.

---

# Câu 5 — Vì sao chọn optimizer hiện tại thay vì AdamW hoặc SGD?

## 5.1. Bản trả lời nhanh

> **Lý do thực tế là Adadelta kế thừa từ pipeline HMER/TAMER và cho phép tối ưu ổn định với learning rate danh nghĩa lớn mà không cần momentum schedule phức tạp. Tuy nhiên repo chưa có optimizer ablation, nên em không được nói Adadelta tốt hơn AdamW hoặc SGD.**
>
> **Cách trả lời khoa học là: em giữ optimizer của baseline để cô lập tác động kiến trúc; optimizer comparison là hướng bổ sung.**

## 5.2. So sánh khái quát

### Adadelta

- adaptive per-parameter;
- ít phụ thuộc absolute gradient scale;
- lịch sử dùng trong HMER.

### AdamW

- momentum bậc nhất và bậc hai;
- decoupled weight decay;
- phổ biến với Transformer;
- cần tune LR/warmup.

### SGD

- đơn giản;
- có thể tổng quát tốt;
- thường cần schedule và momentum kỹ;
- khó hơn với transformer hybrid nếu không tune.

### Thiết kế so sánh

Cùng:

- seed;
- epochs;
- batch;
- schedule budget;
- weight decay search;
- best validation criterion.

Không so một AdamW chưa tune với Adadelta đã tune.

## 5.3. Không nên nói

- “AdamW không phù hợp GAT.”
- “SGD không train được Transformer.”
- “Adadelta không cần learning rate.”
- “Giữ optimizer cũ là bằng chứng tối ưu.”

---

# Câu 6 — Mixed precision/FP16 được dùng như thế nào và có kiểm soát overflow hay không?

## 6.1. Bản trả lời nhanh

> **Trainer đặt precision 16, nên Lightning/PyTorch chạy automatic mixed precision: nhiều phép tính dùng FP16, còn một số state và phép nhạy cảm được giữ FP32 tùy backend. Gradient scaling thường được framework dùng để giảm underflow và phát hiện overflow.**
>
> **Tuy nhiên muốn khẳng định không có overflow, em phải kiểm tra log scaler, skipped steps, NaN/Inf và gradient. Chỉ bật FP16 không chứng minh tính ổn định.**

## 6.2. Lợi ích

- giảm VRAM;
- tăng throughput trên T4;
- cho batch/resolution lớn hơn.

### Rủi ro

- underflow gradient;
- overflow attention logits;
- NaN softmax;
- loss scaling dao động;
- giảm chính xác số học.

### Điểm nhạy cảm

- softmax;
- attention score;
- LayerNorm;
- loss;
- GAT dense logits có `-inf`.

Framework thường xử lý nhiều phần, nhưng cần profile.

### Bằng chứng nên có

- không NaN/Inf trong loss;
- số skipped optimizer steps;
- dynamic scale curve;
- so một run FP32 ngắn;
- reproducibility.

## 6.3. Không nên nói

- “FP16 làm chính xác giảm chắc chắn.”
- “FP16 giống FP32 hoàn toàn.”
- “Có GradScaler nên không thể overflow.”
- “SOTA đều dùng FP32” nếu chưa kiểm chứng từng paper.

---

# Câu 7 — Teacher forcing, label smoothing hoặc dropout có được sử dụng không?

## 7.1. Bản trả lời nhanh

> **Teacher forcing được dùng theo nghĩa decoder train nhận chuỗi target đã dịch phải/trái làm prefix ground truth. Repo không thấy label smoothing trong cross-entropy hiện tại. Dropout có ở DenseNet, decoder và GAT; config chính lần lượt khoảng 0,2, 0,3 và 0,2.**
>
> **Beam search chỉ dùng evaluation, không thay teacher forcing trong training.**

## 7.2. Teacher forcing

Training tối ưu:

$$ p(y_t\mid y_{ \lt t}^{GT},X) $$

Inference dùng:

$$ p(y_t\mid \hat y_{ \lt t},X) $$

Khoảng cách này gây exposure bias.

### Label smoothing

Nếu dùng:

$$ q(y)= (1-\epsilon)\mathbf{1}[y=y^*] + \frac{\epsilon}{V} $$

Nhưng code hiện chưa thấy `label_smoothing`.

### Dropout

- DenseNet dropout trong dense layer;
- Transformer dropout;
- attention/GAT dropout.

Dropout quá cao có thể:

- regularize;
- nhưng làm message passing mất ổn định.

## 7.3. Không nên nói

- “Không có teacher forcing.”
- “Có label smoothing 0,1.”
- “Dropout chỉ ở decoder.”
- “Teacher forcing là lỗi của model.”

---

# Câu 8 — Random seed được cố định ở đâu và kết quả có lặp lại ổn định không?

## 8.1. Bản trả lời nhanh

> **Config đặt seed 7 và deterministic true; Lightning có thể seed Python, NumPy và PyTorch khi gọi đúng utility. Tuy nhiên một seed duy nhất không chứng minh kết quả ổn định. CUDA, dataloader workers, AMP và một số kernel vẫn có thể gây sai khác nếu chưa cấu hình đầy đủ.**
>
> **Muốn kết luận lặp lại, phải chạy ít nhất 3–5 seed và báo cáo mean ± standard deviation.**

## 8.2. Các nguồn nondeterminism

- cuDNN kernels;
- multi-worker dataloader;
- operation atomic;
- AMP;
- graph construction cache;
- shuffle;
- checkpoint resume;
- package versions.

### Artifact tái lập

- seed;
- deterministic flags;
- CUDA/cuDNN version;
- PyTorch/Lightning version;
- GPU;
- commit hash;
- data checksum.

### Báo cáo

$$ \bar{x} = \frac{1}{K}\sum_{k=1}^{K}x_k $$

$$ s = \sqrt{ \frac{1}{K-1} \sum_k(x_k-\bar{x})^2 } $$

## 8.3. Không nên nói

- “Có seed nên kết quả chắc chắn giống.”
- “Một run là đủ.”
- “Chênh 0,09% chắc chắn có ý nghĩa.”
- “Deterministic true xử lý mọi nondeterminism.”

---

# Câu 9 — Có thể giải thích mọi kết quả thấp bằng GPU yếu hoặc số epoch ít hay không?

## 9.1. Bản trả lời nhanh

> **Không. GPU chủ yếu ảnh hưởng thời gian, batch và resolution; nó không tự quyết định accuracy nếu protocol tương đương. Số epoch ít chỉ là nguyên nhân khi curve cho thấy model chưa hội tụ. Kết quả thấp còn có thể do kiến trúc, data, optimizer, LR schedule, preprocessing, vocabulary, decoding hoặc seed.**
>
> **Đổ mọi thứ cho phần cứng là giải thích không có bằng chứng.**

## 9.2. Khi nào GPU gián tiếp ảnh hưởng?

- buộc batch nhỏ;
- buộc resolution thấp;
- giới hạn beam;
- giảm số trial;
- dùng FP16;
- không chạy nhiều seed.

Nhưng mỗi tác động phải đo.

### Khi nào “train chưa đủ epoch” hợp lý?

- train và val metric còn tăng;
- loss chưa plateau;
- best epoch sát max epoch;
- LR schedule chưa kích hoạt;
- train thêm tạo cải thiện lặp lại.

### Khi nào không hợp lý?

- val metric đã giảm;
- gap train–val tăng;
- loss giảm nhưng exact không tăng;
- model overfit.

## 9.3. Không nên nói

- “Do T4 yếu.”
- “Thêm GPU sẽ tăng ExpRate.”
- “Train 400 epoch chắc chắn đạt paper.”
- “FP16 là nguyên nhân chính” khi chưa có FP32 control.

---

# Câu 10 — Train loss tiếp tục giảm nhưng ExpRate không tăng có thể do những nguyên nhân nào?

## 10.1. Bản trả lời nhanh

> **Cross-entropy là metric mềm theo xác suất từng token, còn ExpRate là quyết định cứng toàn chuỗi. Loss có thể giảm vì model tự tin hơn ở các token đã đúng hoặc sửa một phần lỗi, nhưng nhiều chuỗi vẫn còn ít nhất một token sai nên Exact Match không đổi.**
>
> **Ngoài ra có thể do overfitting, exposure bias, beam search, class imbalance, calibration hoặc loss phụ trợ không đồng hướng với ExpRate.**

## 10.2. Các cơ chế

### Metric mismatch

Một mẫu từ 5 lỗi xuống 1 lỗi:

- loss giảm;
- edit distance giảm;
- ExpRate vẫn 0.

### Confidence without correction

Model tăng xác suất token đúng ở vị trí dễ, nhưng vị trí khó vẫn sai.

### Overfitting

- train loss giảm;
- val loss/ExpRate không tăng hoặc giảm.

### Sequence-level mismatch

Teacher-forced loss không tối ưu trực tiếp beam Exact Match.

### Auxiliary loss trade-off

Structure loss có thể giảm nhưng không chuyển thành sequence exact.

### Decoding bottleneck

- beam size;
- length normalization;
- EOS bias;
- max length.

### Cách chẩn đoán

- train/val loss riêng;
- token accuracy;
- histogram edit distance;
- sequence length bins;
- calibration;
- oracle beam accuracy;
- loss component curves.

## 10.3. Oracle beam test

Nếu ground truth xuất hiện trong top-k beam nhưng top-1 sai:

- encoder/decoder có candidate đúng;
- ranking/decoding là vấn đề.

Nếu không xuất hiện:

- representation/model probability còn yếu.

## 10.4. Không nên nói

- “Loss giảm nên model chắc chắn tốt hơn.”
- “ExpRate không tăng vì metric quá khắt khe” rồi bỏ qua.
- “Chỉ cần train tiếp.”
- “Overfitting” khi chưa xem validation curve.

---

# Phụ lục A — Bản trả lời tổng hợp khoảng hai phút

> **Cấu hình repo hiện dùng Adadelta, learning rate 1,0, epsilon $10^{-6}$, weight decay $10^{-4}$ và MultiStepLR milestone 300/350. Max epoch lại là 100 nên nếu không override, scheduler thực tế không giảm LR. Train batch danh nghĩa 8, eval 2, không thấy gradient accumulation và không thấy EarlyStopping callback active; validation chạy mỗi 2 epoch và top checkpoint được chọn theo `val_ExpRate`.**
>
> **Trainer dùng FP16 mixed precision. Framework thường dùng gradient scaling nhưng em phải kiểm tra log overflow/NaN trước khi khẳng định ổn định. Training dùng teacher forcing; chưa thấy label smoothing; dropout tồn tại ở DenseNet, decoder và GAT. Seed 7 và deterministic true giúp tái lập nhưng một seed không đủ chứng minh ổn định.**
>
> **Em không giải thích kết quả thấp bằng GPU yếu hoặc ít epoch nếu chưa có curve. GPU chỉ ảnh hưởng gián tiếp qua batch, resolution và số trial. Train loss giảm nhưng ExpRate không tăng có thể do mismatch giữa token-level loss và sequence exact, overfitting, exposure bias hoặc decoding. Do đó em cần log learning rate, gradient, loss components, nhiều seed và oracle beam analysis.**

# Phụ lục B — Checklist log bắt buộc

- [ ] Config cuối sau override.
- [ ] Commit hash.
- [ ] Data checksum.
- [ ] GPU, CUDA, PyTorch, Lightning.
- [ ] Epoch/step.
- [ ] Learning rate theo epoch.
- [ ] Train loss, structure loss, val loss.
- [ ] Val ExpRate, ≤1, ≤2, MED.
- [ ] Gradient norm và NaN/Inf.
- [ ] AMP scale/skipped steps nếu truy được.
- [ ] Best checkpoint hash.
- [ ] Seed và nhiều run.
