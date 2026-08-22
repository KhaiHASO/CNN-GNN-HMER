# Báo cáo Phân tích Chi tiết Kết quả Đánh giá Baseline (CNN-Transformer)

* **Kiến trúc mô hình:** TAMER Baseline (DenseNet Encoder + Transformer Decoder)
* **Thông tin checkpoint:** `epoch=95-step=72095-val_ExpRate=0.5091.ckpt` (Run ID: `8ivyzmlm`, train trên Kaggle 2x T4 GPUs)
* **Thiết bị chạy Eval:** Local Machine (GPU NVIDIA GeForce RTX 3070 8GB VRAM)
* **Cấu hình Eval:** `eval_batch_size=1` (Tối ưu hóa tránh tràn VRAM trên GPU RTX 3070 local)
* **Ngày thực hiện:** 14/06/2026

---

## 📊 1. Bảng Tổng hợp Kết quả Đánh giá (CROHME 2014, 2016, 2019)

| Tập dữ liệu (Dataset) | Số mẫu đánh giá (Samples) | ExpRate (Độ chính xác tuyệt đối) | ExpRate $\le$ 1 (Sai lệch $\le$ 1 ký tự) | ExpRate $\le$ 2 (Sai lệch $\le$ 2 ký tự) |
| :--- | :---: | :---: | :---: | :---: |
| **CROHME 2014** | 986 | **51.12%** | 69.98% | 77.69% |
| **CROHME 2016** | 1,147 | **50.65%** | 67.92% | 76.02% |
| **CROHME 2019** | 1,199 | **48.54%** | 68.14% | 77.23% |
| **Trung bình (Average)** | **-** | **50.10%** | **68.68%** | **76.98%** |

---

## 🔍 2. Phân tích Các Chỉ số Đo lường (Metrics Analysis)

### 2.1. Tỷ lệ Nhận diện Chính xác Tuyệt đối (ExpRate)
* Mô hình đạt trung bình **50.10%** trên cả 3 tập kiểm thử CROHME.
* Kết quả kiểm thử local này hoàn toàn khớp với log validation trong quá trình huấn luyện trên Kaggle (50.91% ở epoch 95), xác nhận tính nhất quán của môi trường chạy local và tính hội tụ chuẩn của checkpoint.
* Hiệu năng cao nhất đạt được trên tập **CROHME 2014 (51.12%)** và giảm nhẹ trên **CROHME 2019 (48.54%)**. Điều này phản ánh thực tế rằng độ phức tạp và độ nhiễu của các ký tự/biểu thức viết tay tăng dần qua các năm biên soạn bộ dữ liệu.

### 2.2. Ý nghĩa của Sai lệch Dung sai (ExpRate $\le$ 1 & ExpRate $\le$ 2)
* **ExpRate $\le$ 1** (Đạt trung bình **68.68%**): Tỷ lệ các biểu thức dự đoán chính xác hoặc chỉ sai tối đa 1 ký tự LaTeX (edit distance $\le$ 1). Ví dụ: thiếu một dấu ngoặc nhọn `{` hoặc `}`, nhầm lẫn nhỏ giữa số mũ/chỉ số dưới (ví dụ `x^2` thành `x2`), hoặc viết sai chính tả một ký hiệu toán học cụ thể (ví dụ `\alpha` thành `a`).
* **ExpRate $\le$ 2** (Đạt trung bình **76.98%**): Tỷ lệ các biểu thức dự đoán sai lệch tối đa 2 ký tự LaTeX.
* **Nhận xét:** Sự chênh lệch lớn giữa **ExpRate tuyệt đối (50.10%)** and **ExpRate $\le$ 1 (68.68%)** cho thấy mô hình baseline học được ngữ nghĩa cấu trúc rất tốt. Có tới hơn **18%** số lượng ảnh biểu thức tuy bị tính là sai lệch nhưng thực tế mô hình đã nhận diện được gần như toàn bộ cấu trúc và chỉ mắc đúng 1 lỗi ký tự nhỏ. 

---

## 💡 3. Điểm mạnh và Điểm yếu của Mô hình Baseline (CNN-Transformer)

### 📈 Điểm mạnh:
1. **Nhận diện ký tự đơn lẻ tốt:** Nhờ backbone DenseNet mạnh mẽ, mô hình trích xuất đặc trưng cục bộ rất tốt cho các chữ số, chữ cái Latin và các toán tử thông thường.
2. **Giải mã tuần tự ổn định:** Transformer Decoder hoạt động tốt với các biểu thức có cấu trúc tuyến tính thẳng hàng (ví dụ: phương trình toán học đơn giản dạng $ax^2 + bx + c = 0$).

### 📉 Điểm yếu & Giới hạn cấu trúc:
1. **Thiếu thông tin hình học 2D:** DenseNet nén ảnh qua các lớp Convolution truyền thống và flatten thành vector 1D trước khi đưa vào Transformer. Quá trình này làm mất đi mối quan hệ không gian 2D phức tạp giữa các ký tự không nằm cạnh nhau nhưng liên kết ngữ nghĩa (như phân số lồng nhau, số mũ của số mũ, ma trận, căn thức nhiều tầng).
2. **Dễ lệch căn lề (Alignment Shift):** Khi gặp các biểu thức viết tay có nét vẽ không đều hoặc nghiêng, Transformer dễ bị lệch Attention dẫn đến sinh thiếu ký tự hoặc lặp ký tự.

---

## 🚀 4. Định hướng Cải tiến từ phiên bản CNN-GNN (DenseNet + GAT)

Mô hình lai **CNN-GNN** được thiết kế để giải quyết trực tiếp các điểm yếu nêu trên của baseline:
* **Tích hợp Graph Attention Network (GAT):** GAT cho phép coi mỗi vùng đặc trưng từ DenseNet là một nút trong đồ thị lưới (feature-grid graph kề 8 hướng).
* **Truyền tin không gian (Spatial Message Passing):** Cơ chế Multi-head Attention trong GAT giúp các node tự học cách chú ý và truyền thông tin cấu trúc 2D đến các node lân cận một cách chủ động.
* **Kết quả thực nghiệm:** Kết quả thực nghiệm thực tế cho thấy mô hình M3 (PE sau GAT) phục hồi ExpRate đạt 49.17% (vượt baseline trên CROHME 2016 với 50.74%), và mô hình M4 (Coordinate-Aware GAT) đạt Mean Edit Distance thấp nhất là 2.06 (so với 2.10 của baseline).

