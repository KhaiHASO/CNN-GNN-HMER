# Báo cáo Phân tích Dự án CNN-Transformer Baseline (HMER)

## 1. Giới thiệu dự án baseline
Dự án **CNN-Transformer Baseline** (Handwritten Mathematical Expression Recognition) đại diện cho mô hình cơ sở trích xuất đặc trưng hình ảnh bằng **CNN** và giải mã chuỗi ký tự LaTeX trực tiếp bằng mạng **Transformer** tiêu chuẩn (ở `chuyende_tamer_temp/0-cnn-transformer-baseline`). 
Phiên bản này được dùng làm hệ quy chiếu (baseline) để so sánh hiệu năng với mô hình cải tiến kết hợp mạng Attention đồ thị (CNN-GNN).

---

## 2. Thông tin mã nguồn (Metadata)
*   **Repository nguồn:** [KhaiHASO/CNN-GNN-HMER](https://github.com/KhaiHASO/CNN-GNN-HMER) (Private Repository)
*   **Commit Hash:** `a0768eb3c8528cbe6e1226514b26729fd7c979d5`
*   **Tệp chạy chính (Entrypoint):** `chuyende_tamer_temp/0-cnn-transformer-baseline/train.py`
*   **Môi trường chạy:** Conda environment `tamer` (Python 3.7.12) trên Kaggle GPU (2 Tesla T4)

---

## 3. Tổng quan về các Lượt chạy (Runs) trên W&B
Dự án chứa **4 runs** trên hệ thống Weights & Biases (W&B) của người dùng `khaihaso`.

| Tên Run | ID Run | Trạng thái | Thời gian chạy | Kết quả / Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| `baseline-crohme-2gpu-t4` | `8ivyzmlm` | Finished | 27,095 giây (~7.5 giờ) | Lượt huấn luyện chính hoàn thành **100 epochs**, đạt `val_ExpRate` cao nhất là **50.91%** |
| `baseline-crohme-2gpu-t4` | `o2er7nve` | Finished | 14 giây | Tiến trình khởi tạo trống / không ghi nhận log |
| `baseline-crohme-2gpu-t4` | `dqr2zb7q` | Finished | 15 giây | Tiến trình khởi tạo trống / không ghi nhận log |
| `baseline-crohme-2gpu-t4` | `js0odsbn` | Finished | 18 giây | Tiến trình khởi tạo trống / không ghi nhận log |

---

## 4. So sánh hiệu năng & Độ ổn định: CNN-Transformer Baseline vs. CNN-GNN

### 4.1. Độ chính xác nhận diện (Accuracy / ExpRate)
*   **CNN-Transformer Baseline (Run `8ivyzmlm`):** Đạt tỷ lệ nhận diện biểu thức chính xác hoàn toàn (Validation Expression Rate) tốt nhất là **50.91%** ở Epoch 95.
*   **CNN-GNN (Run `1nzxiodq`):** Đạt `val_ExpRate` tốt nhất là **24.09%** ở Epoch 19 trước khi bị crash.
*   *Nhận xét:* Bản Baseline cho kết quả nhận diện chính xác cao gấp hơn **2 lần** so với phiên bản CNN-GNN tại thời điểm crash. 

### 4.2. Độ ổn định bộ nhớ GPU (VRAM Stability)
*   **CNN-Transformer Baseline:** Sử dụng cơ chế Self-Attention và Cross-Attention chuẩn của Transformer. Cơ chế này hoạt động ổn định trên phân bổ dữ liệu chuỗi tuần tự, VRAM duy trì đều đặn ở mức cho phép suốt 100 epoch huấn luyện mà không bị tràn bộ nhớ hay crash.
*   **CNN-GNN:** Sử dụng mạng GAT (`tamer/model/gat.py`) với phép toán nhân bản dữ liệu đồ thị phức tạp `.repeat()`. Lượng VRAM tiêu thụ tăng theo hàm **bình phương** của số node đặc trưng trích xuất từ ảnh ($O(n^2)$). Khi gặp các batch chứa ảnh dài/phức tạp, GPU bị tràn bộ nhớ đột ngột và tiến trình bị hệ thống gửi tín hiệu `SIGKILL` buộc dừng lập tức (crashed).

---

## 5. Đề xuất phát triển tiếp theo
*   **Khắc phục lỗi GAT để huấn luyện CNN-GNN lâu hơn:** Nhằm đưa mô hình GNN đi hết 100 epoch như baseline, bắt buộc phải loại bỏ phép toán `.repeat` trong file `tamer/model/gat.py` để chuyển sang Broadcasting, hoặc giảm kích thước batch size xuống còn 8 hoặc 16.
*   **Kết hợp Hybrid:** Nghiên cứu lý do tại sao baseline lại có điểm số chính xác cao vượt trội hơn hẳn. Có thể do cơ chế giải mã Transformer (giải mã chuỗi tốt hơn cấu trúc Graph Decoder của mô hình GNN hiện tại). Hướng đi tiềm năng là sử dụng CNN-GNN để trích xuất đặc trưng không gian tốt hơn nhưng vẫn giữ bộ giải mã Transformer tiêu chuẩn của bản baseline để sinh LaTeX.

---

## 6. Kết quả tốt nhất ghi nhận của Baseline
*   **Train Loss (Struct):** 0.0129
*   **Val Loss (Total):** 0.4851
*   **Val Loss (Struct):** 0.0210
*   **Validation Expression Rate (Accuracy):** 50.91%

> [!TIP]
> Do dự án baseline chạy ổn định trọn vẹn 100 epochs, các đồ thị loss và độ chính xác của nó phản ánh đúng quá trình hội tụ mượt mà của mô hình. Đây là một baseline rất mạnh để so sánh.
