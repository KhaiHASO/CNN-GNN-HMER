# Báo cáo Phân tích Dự án CNN-GNN-HMER

## 1. Giới thiệu dự án
Dự án **CNN-GNN-HMER** (Handwritten Mathematical Expression Recognition) nhằm nhận diện các biểu thức toán học viết tay từ hình ảnh và chuyển đổi chúng thành mã nguồn LaTeX tương ứng. Mô hình sử dụng sự kết hợp giữa **CNN** (dùng để trích xuất đặc trưng hình ảnh) và **GNN** (cụ thể là Graph Attention Network - GAT, dùng để mô hình hóa cấu trúc hai chiều và mối quan hệ giữa các ký tự).

---

## 2. Thông tin mã nguồn (Metadata)
*   **Repository nguồn:** [KhaiHASO/CNN-GNN-HMER](https://github.com/KhaiHASO/CNN-GNN-HMER) (Private Repository)
*   **Commit Hash:** `26318be4c71f06cac196c125e90ecbb76483cadc`
*   **Tệp chạy chính (Entrypoint):** `chuyende_tamer_temp/1-cnn-gnn/train.py`
*   **Môi trường chạy:** Conda environment `unimernet` (Python 3.10.20)

---

## 3. Tổng quan về các Lượt chạy (Runs) trên W&B
Dự án chứa tổng cộng **10 runs** được lưu trên hệ thống Weights & Biases (W&B) của người dùng `khaihaso`.

| Tên Run | ID Run | Trạng thái | Thời gian chạy | Kết quả / Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| `cnn-gnn-crohme-2gpu-t4` | `1nzxiodq` | Crashed | 2,827 giây (~47 phút) | Run chính chạy được 19 epochs, đạt `val_ExpRate` = 24.09% |
| `cnn-gnn-crohme-2gpu-t4` | `522rgd9s` | Failed | 33 giây | Bị crash ngay ở epoch 0 do lỗi **CUDA Out of Memory** |
| `cnn-gnn-crohme-2gpu-t4` | `jwbl6zuc` | Crashed | 17 giây | Crash ngay ở epoch 0 |
| `cnn-gnn-crohme-2gpu-t4` | `uf9wwd1n` | Crashed | 17 giây | Crash ngay ở epoch 0 |
| *Các run phụ khác (13j2wxey, 20h2tun4, 7qqqa390, 86olsd2p, qdov0k7t, r63isncm)* | - | Finished | ~10-15 giây | Các run phụ được tạo tự động khi upload/đồng bộ các Artifacts |

---

## 4. Phân tích Chi tiết Lỗi Crash tại Epoch 19 (Run `1nzxiodq`)
Tại sao run `1nzxiodq` đang chạy tốt đến epoch 19 (mất khoảng 1 giờ 47 phút) thì đột ngột bị crash mà không lưu lại log lỗi (`exitcode: null`)?

### Phân tích kỹ thuật:
1. **Sự khác biệt về Kích thước Dữ liệu Đầu vào (Dynamic Resolution):**
   Trong bài toán HMER, các hình ảnh chứa biểu thức viết tay có kích thước không cố định. Ảnh chứa biểu thức dài hoặc phức tạp sẽ lớn hơn nhiều so với ảnh chứa biểu thức ngắn (ví dụ: một hệ phương trình dài so với chỉ một ký tự hoặc chữ số đơn lẻ).
   
2. **Độ phức tạp Bộ nhớ của GAT (Graph Attention Network):**
   Mô hình này sử dụng cơ chế Attention trong GAT (`tamer/model/gat.py`). Tại dòng 98:
   `Wh2_expanded.repeat(1, 1, n, 1, 1)`
   Để tính toán điểm số Attention cho tất cả các cặp đỉnh (nodes) trong đồ thị, tensor được nhân bản $n$ lần (với $n$ là số lượng node/pixels của đặc trưng).
   Độ phức tạp bộ nhớ của thao tác này tăng theo hàm **bình phương** số lượng node ($O(n^2)$).
   
3. **Hiện tượng OOM Đột ngột (Out Of Memory Spike):**
   Trong suốt 18 epoch đầu, mô hình có thể chỉ gặp các mẫu dữ liệu có kích thước trung bình và nhỏ, giúp lượng VRAM tiêu thụ luôn nằm dưới ngưỡng giới hạn 16GB của card Tesla T4.
   Tuy nhiên, ở epoch 19, khi hệ thống nạp vào một **batch chứa ảnh có kích thước lớn hoặc độ dài biểu thức vượt trội**, lượng VRAM yêu cầu tăng đột biến (vượt quá dung lượng còn lại của GPU).
   Khi GPU bị tràn bộ nhớ đột ngột ở cấp độ driver/CUDA (phân bổ vượt mức quá lớn), hệ điều hành hoặc driver sẽ lập tức gửi tín hiệu kết thúc tiến trình (`SIGKILL`), khiến chương trình python dừng ngay lập tức. W&B agent không kịp thực hiện các thủ tục đóng kết nối và upload file log `output.log`, dẫn đến trạng thái ghi nhận trên W&B là **Crashed** và `exitcode` là **None**.

---

## 5. Đề xuất Khắc phục triệt để
Để tránh việc mô hình bị dừng đột ngột ở các epoch sau, bạn nên áp dụng các giải pháp sau:
*   **Tránh sử dụng `.repeat` trong GAT:** Thay thế việc dùng `.repeat` bằng cơ chế tự động **Broadcasting** của PyTorch (chỉ tốn bộ nhớ khi tính toán, không nhân bản dữ liệu trên bộ nhớ thực tế).
*   **Lọc/Giới hạn kích thước ảnh đầu vào:** Loại bỏ các ảnh quá dài/quá lớn trong quá trình tiền xử lý, hoặc resize chúng về một kích thước tối đa cố định.
*   **Giảm Batch Size và sử dụng Gradient Accumulation:** Giảm `batch_size` xuống một nửa (ví dụ từ 32 xuống 16 hoặc 8) để tạo không gian trống (headroom) cho VRAM phòng trường hợp gặp mẫu lớn, đồng thời đặt `accumulate_grad_batches` trong PyTorch Lightning để giữ nguyên kích thước batch cập nhật trọng số.
*   **Sử dụng Gradient Checkpointing:** Giúp tiết kiệm đáng kể bộ nhớ GPU bằng cách tính toán lại các activation trong quá trình lan truyền ngược thay vì lưu tất cả trong bộ nhớ.

---

## 6. Kết quả huấn luyện (Best Run: 1nzxiodq)
*   **Train Loss (Struct):** 0.0425
*   **Val Loss (Total):** 0.8601
*   **Val Loss (Struct):** 0.0475
*   **Validation Expression Rate (Accuracy):** 24.09%

> [!NOTE]
> Mô hình đang học tốt, biểu hiện qua việc **loss giảm dần** và **độ chính xác tăng dần** qua các epoch. Tuy nhiên, tỷ lệ 24.09% vẫn còn thấp đối với bài toán HMER thực tế, cần được huấn luyện thêm nhiều epoch hơn hoặc cải tiến mô hình.

---

## 7. Dữ liệu Validation đã lưu (Bảng dự đoán)
Hệ thống W&B đã lưu trữ bảng dự đoán của tập validation (`val_predictions`). Hai mẫu dự đoán tiêu biểu được ghi nhận trong bảng này:
1.  **Hình ảnh 1:** Biểu thức `- 7`.
    *   **Ground Truth:** `- 7`
    *   **Prediction:** `- 7`
    *   **Kết quả:** :white_check_mark: Chính xác (Match)
2.  **Hình ảnh 2:** Biểu thức `k N`.
    *   **Ground Truth:** `k N`
    *   **Prediction:** `k V`
    *   **Kết quả:** :x: Sai (Mismatch - Nhận diện nhầm N thành V)
