# Milestone 02: Thực nghiệm Tiền xử lý Ảnh (Preprocessing Ablation)

## 1. Mục tiêu và Phương pháp
Milestone này thực hiện so sánh hiệu năng của mô hình UniMERNet trên tập 50 ảnh thực nghiệm nhanh dưới 3 cấu hình tiền xử lý ảnh:
* **P0**: Ảnh gốc (Baseline).
* **P1**: Chuyển xám, cắt biên thừa và thay đổi kích thước giữ nguyên tỉ lệ (chiều cao 192px).
* **P2**: Thực hiện P1 kết hợp nhị phân hóa Otsu và lọc nhiễu Median.

---

## 2. Kết quả thực nghiệm chính

| Cấu hình | Khớp tuyệt đối (Exact Match) | Tỉ lệ (%) | Render thành công | Tỉ lệ (%) | Thời gian TB/ảnh (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P0 (Ảnh gốc)** | 24 | 48.0% | 41 | 82.0% | 4,058 ms |
| **P1 (Gray+Crop+Resize)** | 8 | 16.0% | 35 | 70.0% | 5,297 ms |
| **P2 (Threshold+Denoise)** | 1 | 2.0% | 28 | 56.0% | 7,970 ms |

---

## 3. Khám phá quan trọng (Key Findings)
* **Tiền xử lý ảnh làm giảm hiệu năng mô hình**: Các thuật toán biến đổi hình ảnh truyền thống phá vỡ các đặc trưng nét vẽ (stroke features) và làm mất dải biên mượt (anti-aliasing) mà mô hình học sâu UniMERNet dựa vào để nhận diện cấu trúc.
* **Tăng lỗi cú pháp lồng nhau**: Số lượng lỗi cấu trúc liên quan đến số mũ, chỉ số dưới và căn thức tăng mạnh khi ảnh bị nhị phân hóa và co giãn (P2 có tới 15 lỗi số mũ so với 0 lỗi ở P0).
* **Ảnh hưởng đến thời gian suy luận**: Chất lượng đặc trưng ảnh suy giảm khiến decoder của mô hình mất nhiều thời gian hơn để tìm điểm dừng chuỗi sinh mã LaTeX, dẫn đến thời gian xử lý trung bình tăng gần gấp đôi.

---

## 4. Định hướng cho luận văn (Thesis Relevance)
Kết quả thực nghiệm này cung cấp cơ sở lập luận khoa học vững chắc cho Chương 3 và Chương 4 của luận văn:
1. **Lập luận thiết kế**: Luận văn sẽ giữ nguyên ảnh gốc đầu vào cho mô hình nhận diện (Backbone).
2. **Hướng nghiên cứu tiếp theo**: Đề xuất giải pháp sửa lỗi ở tầng đầu ra (Output-level validation) thay vì can thiệp tầng hình ảnh. Module tiếp theo sẽ tập trung xây dựng **Rule-based post-processing** và **LaTeX syntax validator (Graph/GNN-based)** để phát hiện và tự động sửa các lỗi cú pháp mất cân bằng dấu ngoặc hoặc sai cấu trúc phân số/căn thức.
