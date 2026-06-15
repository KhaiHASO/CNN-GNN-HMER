# Báo cáo So sánh và Phân tích Hiệu năng: CNN-Transformer Baseline vs. CNN-GNN

Báo cáo này tổng hợp kết quả đánh giá (Evaluation) chính thức của mô hình **CNN-GNN** so với **CNN-Transformer Baseline** trên cả 3 tập dữ liệu kiểm thử chuẩn **CROHME 2014, 2016, và 2019**. 

---

## 📊 1. Bảng So sánh Định lượng (Quantitative Comparison)

Dưới đây là bảng đối chiếu chi tiết các chỉ số đo lường giữa hai mô hình:
* **Baseline:** Checkpoint `epoch=95-step=72095-val_ExpRate=0.5091.ckpt` (Run ID: `8ivyzmlm`)
* **CNN-GNN:** Checkpoint `epoch=77-step=58577-val_ExpRate=0.4939.ckpt` (Run ID: `8b964c54...`)

| Tập dữ liệu (Dataset) | Mô hình | ExpRate (Chính xác tuyệt đối) | ExpRate $\le$ 1 (Sai lệch $\le$ 1 ký tự) | ExpRate $\le$ 2 (Sai lệch $\le$ 2 ký tự) | Mean Edit Distance (Khoảng cách chỉnh sửa TB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CROHME 2014** | Baseline | **51.12%** | **69.98%** | **77.69%** | **1.99** |
| | CNN-GNN | 49.39% | 66.53% | 75.25% | 2.22 |
| | *Chênh lệch* | *-1.73%* | *-3.45%* | *-2.44%* | *+0.23 (tệ hơn)* |
| **CROHME 2016** | Baseline | **50.65%** | **67.92%** | **76.02%** | **2.21** |
| | CNN-GNN | 47.43% | 64.95% | 74.72% | 2.45 |
| | *Chênh lệch* | *-3.22%* | *-2.97%* | *-1.30%* | *+0.24 (tệ hơn)* |
| **CROHME 2019** | Baseline | **48.54%** | **68.14%** | **77.23%** | **2.40** |
| | CNN-GNN | 46.71% | 66.89% | 75.90% | 2.62 |
| | *Chênh lệch* | *-1.83%* | *-1.25%* | *-1.33%* | *+0.22 (tệ hơn)* |
| **Trung bình (Average)** | Baseline | **50.10%** | **68.68%** | **76.98%** | **2.20** |
| | CNN-GNN | **47.84%** | **66.12%** | **75.29%** | **2.43** |
| | *Chênh lệch* | *-2.26%* | *-2.56%* | *-1.69%* | *+0.23 (tệ hơn)* |

> [!IMPORTANT]
> Mô hình CNN-GNN cho hiệu năng **thấp hơn trung bình 2.26% về tỷ lệ nhận diện tuyệt đối (ExpRate)** và có khoảng cách chỉnh sửa trung bình (Mean Edit Distance) cao hơn so với Baseline trên cả 3 tập dữ liệu kiểm thử.

---

## 🔍 2. Phân tích Nguyên nhân Kỹ thuật (Technical Root Causes)

Tại sao việc tích hợp Graph Attention Network (GAT) trên lưới pixel lại làm giảm nhẹ hiệu năng thay vì cải thiện? Có 3 lý do cốt lõi từ mặt kiến trúc:

### 2.1. Hiện tượng làm mờ thông tin vị trí (Position Encoding Blurring)
Trong file [encoder.py](file:///home/khai/Desktop/github/CNN-GNN-HMER/chuyende_tamer_temp/1-cnn-gnn/tamer/model/encoder.py#L275-L291), luồng tính toán hiện tại là:
1. Trích xuất đặc trưng pixel bằng DenseNet.
2. Cộng thêm **2D Positional Encoding** (`pos_enc_2d`) để đánh dấu tọa độ tuyệt đối từng pixel.
3. Thực hiện truyền tin (message passing) qua các lớp **GAT** trên lưới đồ thị kề 8-hướng.
4. Đưa đặc trưng sau GAT vào Transformer Decoder.

* **Vấn đề:** Khi GAT thực hiện lấy tổng trọng số Attention của các node lân cận, nó sẽ **vô tình làm mịn/làm mờ (blend/average)** các vector Positional Encoding của các pixel cạnh nhau. Đối với nhận diện cấu trúc toán học viết tay (HMER), tọa độ chính xác của các ký hiệu (để xác định xem nó là số mũ, chỉ số dưới, hay nằm trong căn) cực kỳ nhạy cảm. Việc làm mờ vị trí này khiến bộ giải mã Transformer dễ bị lệch Attention (Alignment Shift), dẫn đến dự đoán sai lệch cấu trúc hoặc ký tự.

### 2.2. GAT thiếu nhận thức về hướng (Direction-Agnostic Attention)
* Phép tính attention trong GAT tiêu chuẩn chỉ phụ thuộc vào sự tương đồng đặc trưng giữa các node:
  $$e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j])$$
* Tuy đồ thị được kết nối 8-hướng (trên, dưới, trái, phải, chéo), GAT treats các láng giềng như một **tập hợp không có thứ tự (bag of neighbors)** và không có khái niệm về mặt tọa độ định hướng. 
* Trong khi đó, các phép toán Convolution 2D thông thường luôn có bộ lọc định hướng riêng (học các trọng số khác nhau cho pixel phía trên vs. phía dưới). Việc mất đi tính định hướng này khiến GAT khó học được các mối quan hệ hình học phân tầng (như tử số nằm *trên* mẫu số, số mũ nằm *trên bên phải*).

### 2.3. Hiện tượng quá khớp nhanh hơn (Earlier Overfitting)
* **Số lượng tham số lớn:** GAT với cấu hình 8-heads và 2 layers tăng thêm lượng tham số đáng kể trên phần encoder. 
* Với tập dữ liệu huấn luyện tương đối nhỏ (CROHME chỉ có 8,834 ảnh), mô hình CNN-GNN học thuộc lòng (memorize) nhiễu nhanh hơn. Điều này thể hiện qua việc loss huấn luyện của CNN-GNN giảm rất sâu (Train Loss = `0.1600`) nhưng Validation Loss lại cao (`0.5059` so với baseline `0.4378`), và mô hình đạt đỉnh sớm ở epoch 77 rồi suy giảm dần tới epoch 99.

---

## 🔬 3. Phân tích Lỗi Định tính (Qualitative Error Cases)

So sánh trực tiếp các file lỗi `errors_2014.json` của hai mô hình cho thấy các đặc trưng lỗi rõ rệt:

### 3.1. Lỗi cấu trúc phân tầng phức tạp (GNN thất bại, Baseline thành công)
GNN có xu hướng phá vỡ cấu trúc của các biểu thức lồng nhau do đặc trưng bị làm mịn:
* **Ví dụ 1 (Phân số lồng nhau):** 
  * **GT:** `\frac { d y } { d x } = \frac { 1 } { \frac { d x } { d y } }`
  * **GNN dự đoán:** `\frac { d y } { d x } = \frac { 1 } { d x } \frac { d x } { d y }`
  * *Nhận xét:* GNN hoàn toàn không nhận diện được phân số lồng ở mẫu số mà duỗi thẳng nó ra thành tích của hai phân số đứng cạnh nhau.
* **Ví dụ 2 (Số mũ phân số):**
  * **GT:** `y ^ { \frac { 1 } { b } } \leq x ^ { \frac { 1 } { b } }`
  * **GNN dự đoán:** `y ^ { \frac { 1 } { 5 } } \leq x ^ { \frac { 1 } { 5 } }`
  * *Nhận xét:* GNN nhầm lẫn ký tự nhỏ `b` thành `5` khi nó nằm sâu trong cấu trúc mũ của phân số.

### 3.2. Lỗi nhầm lẫn ký tự mảnh/nhỏ do bị làm mịn đặc trưng (Feature Smoothing)
Do cơ chế lan truyền cục bộ của GNN làm mịn thông tin của các node lân cận, các nét vẽ mảnh hoặc ký tự nhỏ dễ bị đồng hóa với môi trường xung quanh:
* **GT:** `\lim \limits _ { z \rightarrow z _ { 0 } } f ( z ) = k` $\rightarrow$ **GNN PRED:** `\lim \limits _ { z \rightarrow \infty } f ( z ) = t` (Nhầm `z_0` thành `\infty`)
* **GT:** `y \in B` $\rightarrow$ **GNN PRED:** `y E B` (Nhầm dấu thuộc `\in` thành chữ `E`)
* **GT:** `z _ { 1 } z _ { 2 }` $\rightarrow$ **GNN PRED:** `z , z _ { 2 }` (Nhầm chỉ số dưới `_1` thành dấu phẩy `, `)

---

## 🚀 4. Phương án Cải tiến Đề xuất (Improvement Roadmap)

Để nâng hiệu năng của CNN-GNN vượt qua Baseline, chúng ta cần thực hiện các điều chỉnh cấu trúc sau (xếp theo thứ tự ưu tiên):

### 4.1. Thay đổi thứ tự bổ sung Positional Encoding (Ưu tiên số 1 - Dễ làm nhất)
* **Ý tưởng:** Đưa Positional Encoding ra **sau** khối GAT thay vì trước GAT.
* **Mã nguồn điều chỉnh gợi ý (trong `Encoder.forward`):**
  ```python
  # 1. Trích xuất đặc trưng DenseNet và chiếu sang d_model
  feature, mask = self.model(img, img_mask)
  feature = self.feature_proj(feature)
  feature = rearrange(feature, "b d h w -> b h w d")

  # 2. Áp dụng GAT trên đặc trưng hình ảnh thuần túy (không bị lẫn PE)
  if self.use_gat:
      b, h, w, d = feature.shape
      feature_flat = feature.view(b, h * w, d)
      adj = self._build_grid_adjacency(mask)
      feature_flat = feature_flat + self.gat(feature_flat, adj)
      feature = feature_flat.view(b, h, w, d)
      feature = self.norm(feature)

  # 3. Cộng Positional Encoding tọa độ sắc nét VÀO CUỐI CÙNG
  feature = self.pos_enc_2d(feature, mask)
  feature = self.norm(feature)
  ```
* **Lợi ích:** Đảm bảo thông tin tọa độ tuyệt đối gửi tới Transformer Decoder luôn sắc nét 100%, không bị làm mịn bởi các bước truyền tin đồ thị.

### 4.2. Thiết lập GAT nhận biết khoảng cách/hướng (Coordinate-Aware GAT)
* **Ý tưởng:** Tích hợp vector khoảng cách tương đối $r_{ij} = (\Delta x, \Delta y)$ giữa node $i$ và node $j$ trên lưới vào hàm Attention.
* **Công thức mới:** 
  $$e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j] + W_R r_{ij})$$
  Trong đó $r_{ij}$ là vector định hướng được mã hóa qua hàm sin/cos (giống positional encoding tương đối).
* **Lợi ích:** Giúp GAT phân biệt rõ ràng 8 hướng xung quanh (trên khác dưới, trái khác phải), tăng khả năng học cấu trúc 2D hình học.

### 4.3. Giảm độ phức tạp và Tăng tính điều chuẩn (Regularization)
* Giảm số layer GAT xuống còn **1 layer** (tránh hiện tượng over-smoothing khi truyền tin quá sâu).
* Giảm số head của GAT từ **8 xuống 4** để giảm số lượng tham số, tránh quá khớp sớm.
* Tăng `gat_dropout` từ **0.1 lên 0.2 hoặc 0.3** để ép mô hình học các đặc trưng tổng quát hóa tốt hơn.
* Sử dụng cơ chế Early Stopping dựa trên `val_ExpRate` để lưu checkpoint tối ưu nhất một cách tự động, tránh hiện tượng giảm chất lượng ở các epoch cuối cùng.
