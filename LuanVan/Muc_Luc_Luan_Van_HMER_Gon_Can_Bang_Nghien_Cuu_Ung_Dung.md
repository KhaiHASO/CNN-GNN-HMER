# MỤC LỤC ĐỀ XUẤT LUẬN VĂN THẠC SĨ

> **Định hướng:** Cân bằng nghiên cứu và ứng dụng  
> **Đề tài:** Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay  
> **Nguyên tắc:** Mục lục chính chỉ giữ các luận điểm cần thiết để tạo một câu chuyện liền mạch. Các cấu hình chi tiết, bảng thống kê, log, mã nguồn, trường hợp lỗi và câu trả lời phòng thủ được đặt ở phụ lục hoặc hồ sơ minh chứng riêng.  
> **Cấp mục:** Sử dụng chủ yếu đến cấp 3. Cấp 4 chỉ bổ sung khi một mục thật sự cần tách thành nhiều thí nghiệm hoặc quy trình độc lập.

---

# PHẦN MỞ ĐẦU

## Lời cam đoan

## Lời cảm ơn

## Tóm tắt luận văn

## Abstract

## Danh mục từ viết tắt

## Danh mục bảng

## Danh mục hình

## Mục lục

---

# CHƯƠNG 1. TỔNG QUAN VỀ ĐỀ TÀI

## 1.1. Bối cảnh và lý do chọn đề tài

### 1.1.1. Nhu cầu nhận dạng biểu thức toán học viết tay

### 1.1.2. Những khó khăn của bài toán HMER

### 1.1.3. Động cơ kết hợp CNN, GNN/GAT và Transformer

## 1.2. Phát biểu bài toán

### 1.2.1. Đầu vào và đầu ra của hệ thống

### 1.2.2. Bài toán image-to-LaTeX

### 1.2.3. Các yêu cầu chính đối với hệ thống

## 1.3. Mục tiêu của luận văn

### 1.3.1. Mục tiêu tổng quát

### 1.3.2. Mục tiêu về mô hình

### 1.3.3. Mục tiêu về thực nghiệm và ứng dụng

## 1.4. Đối tượng và phạm vi thực hiện

### 1.4.1. Đối tượng nghiên cứu

### 1.4.2. Phạm vi dữ liệu và biểu thức

### 1.4.3. Phạm vi chức năng của hệ thống

## 1.5. Phương pháp thực hiện

### 1.5.1. Khảo sát và kế thừa kiến trúc nền

### 1.5.2. Thiết kế các biến thể M1–M5

### 1.5.3. Huấn luyện, đánh giá và phân tích lỗi

### 1.5.4. Xây dựng ứng dụng minh họa

## 1.6. Đóng góp của luận văn

### 1.6.1. Đóng góp về thiết kế mô hình

### 1.6.2. Đóng góp về thực nghiệm

### 1.6.3. Đóng góp về hiện thực hệ thống

## 1.7. Cấu trúc luận văn

## 1.8. Kết luận chương

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN

## 2.1. Tổng quan bài toán nhận dạng biểu thức toán học viết tay

### 2.1.1. Nhận dạng trực tuyến và ngoại tuyến

### 2.1.2. Nhận dạng ký hiệu và nhận dạng toàn biểu thức

### 2.1.3. Phân tích cấu trúc hai chiều và sinh LaTeX

## 2.2. Mạng nơ-ron tích chập và DenseNet

### 2.2.1. Trích xuất đặc trưng ảnh

### 2.2.2. Dense connectivity và khả năng tái sử dụng đặc trưng

### 2.2.3. Vai trò của DenseNet trong HMER

## 2.3. Transformer cho bài toán image-to-sequence

### 2.3.1. Self-attention và cross-attention

### 2.3.2. Positional encoding

### 2.3.3. Autoregressive decoding và beam search

## 2.4. Graph Neural Network và Graph Attention Network

### 2.4.1. Graph, node, edge và message passing

### 2.4.2. Graph attention và multi-head attention

### 2.4.3. Relative position và directional bias

### 2.4.4. Giới hạn về độ sâu và chi phí tính toán

## 2.5. Các hướng tiếp cận liên quan trong HMER

### 2.5.1. Phương pháp dựa trên phân đoạn ký hiệu

### 2.5.2. Phương pháp image-to-sequence

### 2.5.3. Phương pháp syntax-aware, tree-aware và graph-based

## 2.6. Nhận xét và định vị giải pháp của luận văn

### 2.6.1. Khoảng trống về cách tích hợp GAT vào feature-grid

### 2.6.2. Vấn đề thứ tự positional encoding và message passing

### 2.6.3. Định hướng cân bằng độ chính xác và khả năng triển khai

## 2.7. Kết luận chương

---

# CHƯƠNG 3. DỮ LIỆU, TIỀN XỬ LÝ VÀ PHÂN TÍCH YÊU CẦU

## 3.1. Bộ dữ liệu sử dụng

### 3.1.1. Tổng quan CROHME

### 3.1.2. Các tập train, validation và test

### 3.1.3. Đặc điểm dữ liệu raster và nhãn LaTeX

## 3.2. Thống kê và kiểm tra dữ liệu

### 3.2.1. Số lượng mẫu và phân bố độ dài

### 3.2.2. Phân bố token và cấu trúc

### 3.2.3. Kiểm tra trùng lặp, OOV và chất lượng dữ liệu

## 3.3. Tiền xử lý ảnh

### 3.3.1. Resize giữ tỷ lệ và giới hạn kích thước

### 3.3.2. Chuyển tensor, dynamic batching, padding và mask

### 3.3.3. Sự khác biệt giữa pipeline benchmark và demo

## 3.4. Xử lý nhãn LaTeX

### 3.4.1. Định dạng caption và quy tắc token hóa

### 3.4.2. Từ điển và các token đặc biệt

### 3.4.3. Vấn đề chuẩn hóa biểu diễn LaTeX

## 3.5. Phân tích yêu cầu hệ thống

### 3.5.1. Yêu cầu chức năng

### 3.5.2. Yêu cầu phi chức năng

### 3.5.3. Giới hạn đầu vào, đầu ra và phạm vi sử dụng

## 3.6. Các trường hợp khó cần quan tâm

### 3.6.1. Biểu thức dài và cấu trúc lồng nhau

### 3.6.2. Ký hiệu nhỏ, chỉ số và cận tích phân

### 3.6.3. Dữ liệu ngoài phân bố

## 3.7. Kết luận chương

---

# CHƯƠNG 4. PHƯƠNG PHÁP ĐỀ XUẤT VÀ HIỆN THỰC HỆ THỐNG

## 4.1. Kiến trúc tổng thể

### 4.1.1. Luồng xử lý từ ảnh đến chuỗi LaTeX

### 4.1.2. Các thành phần chính của hệ thống

### 4.1.3. Kích thước tensor tại các giai đoạn chính

## 4.2. Bộ mã hóa ảnh DenseNet

### 4.2.1. Trích xuất feature map

### 4.2.2. Downsampling và chiếu về không gian `d_model`

## 4.3. Xây dựng feature-grid graph

### 4.3.1. Định nghĩa node và thứ tự flatten

### 4.3.2. Graph tám láng giềng và self-loop

### 4.3.3. Xử lý node padding

## 4.4. Graph Attention Network có thông tin vị trí

### 4.4.1. Cơ chế cập nhật node bằng graph attention

### 4.4.2. Relative directional bias chín trạng thái

### 4.4.3. Residual connection và LayerNorm

### 4.4.4. Vị trí của absolute positional encoding

## 4.5. Transformer Decoder và cơ chế sinh LaTeX

### 4.5.1. Token embedding và causal self-attention

### 4.5.2. Cross-attention với feature ảnh

### 4.5.3. Teacher forcing, hàm mất mát và beam search

## 4.6. Các phiên bản phát triển M1–M5

### 4.6.1. M1 — DenseNet–Transformer baseline

### 4.6.2. M2 và M3 — Khảo sát vị trí positional encoding

### 4.6.3. M4 và M5 — Coordinate-Aware GAT và ảnh hưởng của quy mô

## 4.7. Hiện thực hệ thống

### 4.7.1. Tổ chức module dữ liệu, mô hình và huấn luyện

### 4.7.2. Quy trình suy luận

### 4.7.3. Ứng dụng demo và hiển thị kết quả

## 4.8. Độ phức tạp và giới hạn kiến trúc

### 4.8.1. Chi phí của DenseNet, GAT và Transformer Decoder

### 4.8.2. Giới hạn của graph cục bộ và attention dense

### 4.8.3. Khác biệt với symbol-level graph

## 4.9. Kết luận chương

---

# CHƯƠNG 5. THỰC NGHIỆM, ĐÁNH GIÁ VÀ THẢO LUẬN

## 5.1. Thiết lập thực nghiệm

### 5.1.1. Môi trường phần cứng và phần mềm

### 5.1.2. Cấu hình huấn luyện và suy luận

### 5.1.3. Tiêu chí chọn checkpoint

## 5.2. Các metric đánh giá

### 5.2.1. ExpRate

### 5.2.2. Tỷ lệ không quá một và hai lỗi

### 5.2.3. Mean Edit Distance

### 5.2.4. Chỉ số hiệu năng hệ thống

## 5.3. Kết quả của các mô hình M1–M5

### 5.3.1. Kết quả trên CROHME 2014, 2016 và 2019

### 5.3.2. Kết quả trung bình

### 5.3.3. So sánh tổng quan giữa các phiên bản

## 5.4. Phân tích tác động của các thay đổi kiến trúc

### 5.4.1. Ảnh hưởng của việc bổ sung GAT

### 5.4.2. Ảnh hưởng của vị trí positional encoding

### 5.4.3. Ảnh hưởng của relative directional bias

### 5.4.4. Ảnh hưởng của số lớp và số head

## 5.5. Phân tích trade-off giữa các metric

### 5.5.1. Exact Match và mức độ gần đúng

### 5.5.2. Ý nghĩa của Mean Edit Distance đối với ứng dụng

### 5.5.3. Lựa chọn mô hình theo mục tiêu sử dụng

## 5.6. Phân tích lỗi

### 5.6.1. Lỗi ký hiệu và token

### 5.6.2. Lỗi cấu trúc và cú pháp LaTeX

### 5.6.3. Lỗi ký hiệu nhỏ và tích phân có cận

### 5.6.4. Lỗi trên dữ liệu ngoài phân bố

## 5.7. Đánh giá ứng dụng demo

### 5.7.1. Kết quả trên ảnh benchmark qua pipeline demo

### 5.7.2. Kết quả trên ảnh người dùng

### 5.7.3. Ảnh hưởng của preprocessing và domain shift

## 5.8. Đánh giá hiệu năng và khả năng triển khai

### 5.8.1. Thời gian xử lý và bộ nhớ

### 5.8.2. Ảnh hưởng của beam size

### 5.8.3. Phạm vi ứng dụng phù hợp

## 5.9. Thảo luận và hạn chế

### 5.9.1. Những kết quả được hỗ trợ bởi thực nghiệm

### 5.9.2. Những kết quả chưa đạt kỳ vọng

### 5.9.3. Hạn chế về dữ liệu, protocol và ablation

### 5.9.4. Hạn chế về kiến trúc và triển khai

## 5.10. Kết luận chương

---

# CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## 6.1. Tóm tắt nội dung đã thực hiện

## 6.2. Kết quả và đóng góp chính

### 6.2.1. Kết quả về mô hình

### 6.2.2. Kết quả về thực nghiệm

### 6.2.3. Kết quả về hiện thực hệ thống

## 6.3. Đánh giá mức độ hoàn thành mục tiêu

## 6.4. Hạn chế của luận văn

### 6.4.1. Hạn chế về dữ liệu và quy trình đánh giá

### 6.4.2. Hạn chế về kiến trúc và hiệu năng

### 6.4.3. Hạn chế về khả năng tổng quát ngoài benchmark

## 6.5. Hướng phát triển

### 6.5.1. Sparse graph attention và tối ưu hiệu năng

### 6.5.2. Multi-scale hoặc high-resolution encoder

### 6.5.3. Symbol-level graph và grammar-constrained decoding

### 6.5.4. Mở rộng dữ liệu và đánh giá ngoài CROHME

### 6.5.5. Hoàn thiện ứng dụng thực tế

## 6.6. Kết luận chung

---

# TÀI LIỆU THAM KHẢO

---

# PHỤ LỤC A. CẤU HÌNH MÔ HÌNH VÀ HUẤN LUYỆN

## A.1. Bảng cấu hình M1–M5

## A.2. Cấu hình huấn luyện và suy luận

## A.3. Môi trường phần cứng và phần mềm

## A.4. Lệnh chạy và thông tin checkpoint

---

# PHỤ LỤC B. THỐNG KÊ VÀ KIỂM TRA DỮ LIỆU

## B.1. Số lượng mẫu và phân bố độ dài

## B.2. Phân bố token và cấu trúc

## B.3. Thống kê tích phân có cận

## B.4. Kiểm tra duplicate, overlap và OOV

---

# PHỤ LỤC C. KẾT QUẢ VÀ PHÂN TÍCH BỔ SUNG

## C.1. Bảng kết quả chi tiết

## C.2. Phân bố edit distance

## C.3. Kết quả theo độ dài và loại cấu trúc

## C.4. Các trường hợp lỗi tiêu biểu

## C.5. Kết quả top-k beam

---

# PHỤ LỤC D. TÀI LIỆU TÁI LẬP VÀ TRIỂN KHAI

## D.1. Cấu trúc mã nguồn

## D.2. Checksum dữ liệu và checkpoint

## D.3. Run manifest

## D.4. Prediction và metric files

## D.5. Kiểm thử pipeline demo

---

# GHI CHÚ SỬ DỤNG

- Mục lục chính thức dừng ở các luận điểm cần thiết; không đưa toàn bộ câu hỏi phòng thủ vào luận văn.
- Các số liệu chi tiết, script audit, log, cấu hình và trường hợp lỗi được đặt ở phụ lục hoặc hồ sơ minh chứng riêng.
- Mỗi mục cấp 3 nên có đủ nội dung để tạo thành ít nhất một phần hoàn chỉnh; không tạo mục chỉ để chứa một đoạn ngắn.
- Khi chưa có số liệu cho một mục như nhiều seed, benchmark thời gian hoặc thống kê cấu trúc, chuyển nội dung đó sang phần hạn chế hoặc hướng phát triển.
- Cấp 4 chỉ bổ sung trong quá trình viết khi một mục có nhiều quy trình hoặc thí nghiệm độc lập thật sự.
