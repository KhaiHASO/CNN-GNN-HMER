# Nhật ký Thử nghiệm Tiền xử lý Ảnh (Preprocessing Ablation Log)

Ngày thử nghiệm: 13/06/2026  
Mô hình sử dụng: UniMERNet Base  
Bộ dữ liệu thử nghiệm: 50 ảnh thực nghiệm nhanh từ Task 1 (quick_test_50)  

## 1. Phương pháp Tiền xử lý (Preprocessing Configurations)

Chúng tôi thử nghiệm 3 cấu hình tiền xử lý ảnh khác nhau để đánh giá tầm ảnh hưởng của các bước biến đổi ảnh đến khả năng nhận diện của UniMERNet:

| Cấu hình | Ý nghĩa kỹ thuật | Các bước thực hiện |
| :--- | :--- | :--- |
| **P0** (Original) | Ảnh gốc | Giữ nguyên tập ảnh baseline từ tập test gốc, không qua xử lý bổ sung. |
| **P1** (Gray+Crop+Resize) | Ảnh chuẩn hóa hình học | Chuyển ảnh sang thang xám (Grayscale) -> Tự động cắt bỏ các viền trắng/đen thừa (Crop margin) -> Thay đổi kích thước giữ nguyên tỉ lệ (Resize) với chiều cao cố định 192px và chiều rộng tối đa 672px. |
| **P2** (Threshold+Denoise) | Ảnh nhị phân hóa sạch | Thực hiện các bước của P1 -> Nhị phân hóa Otsu để tạo nét chữ đen nền trắng sắc cạnh -> Làm mịn khử nhiễu bằng bộ lọc Median Blur (kích thước nhân 3x3). |

---

## 2. Kết quả Tổng hợp Thực nghiệm (Ablation Summary)

Bảng so sánh kết quả thực nghiệm giữa 3 cấu hình trên tập 50 ảnh:

| Cấu hình | Số ảnh | Khớp tuyệt đối (Exact Match) ↑ | Tỉ lệ khớp (%) | Render thành công ↑ | Tỉ lệ render (%) | Thời gian TB/ảnh (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0 (Original)** | 50 | 24 | 48.0% | 41 | 82.0% | 4,058 ms |
| **P1 (Gray+Crop+Resize)** | 50 | 8 | 16.0% | 35 | 70.0% | 5,297 ms |
| **P2 (Threshold+Denoise)** | 50 | 1 | 2.0% | 28 | 56.0% | 7,970 ms |

---

## 3. Nhận xét ban đầu

1. **Hiệu năng giảm mạnh khi tiền xử lý**: Việc áp dụng P1 làm giảm tỉ lệ khớp chính xác từ 48% xuống còn 16%, và P2 làm giảm nghiêm trọng xuống chỉ còn 2%.
2. **Thời gian suy luận tăng lên**: Thời gian xử lý trung bình mỗi ảnh tăng từ 4.05 giây (P0) lên 7.97 giây (P2). Điều này do chất lượng ảnh xấu đi khiến mô hình sinh ra chuỗi mã LaTeX dài dòng hoặc bị lặp cú pháp lỗi, kéo dài thời gian giải mã của mBART decoder.
3. **Kết luận sơ bộ**: Mô hình học sâu UniMERNet cực kỳ nhạy cảm với sự thay đổi về độ dày nét chữ (stroke width), tỷ lệ nén ảnh và tính mượt mà của biên chữ (anti-aliasing). Các thuật toán nhị phân hóa và resize làm mất đi các đặc trưng này, gây suy giảm nghiêm trọng độ chính xác.
