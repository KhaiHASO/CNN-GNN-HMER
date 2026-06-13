# Phân tích Lỗi Hệ thống: CNN-GNN vs UniMERNet (Error Analysis Report)

Tài liệu này so sánh, phân tích các loại lỗi đặc trưng giữa mô hình lai **CNN-GNN/GAT** và mô hình end-to-end **UniMERNet** dựa trên kết quả thực nghiệm từ Task 1 đến Task 4.

---

## 1. Các lỗi điển hình của UniMERNet (Mô hình dựa trên Chuỗi - Sequence-based)

Qua thực nghiệm trên 50 ảnh mẫu viết tay ở Task 1 và Task 2, UniMERNet gặp các lỗi điển hình liên quan đến mất cân đối cú pháp hoặc nhầm lẫn ký tự cục bộ:

1. **Lỗi mất cân đối cú pháp (Syntax / Bracket Mismatch)**:
   * *Ví dụ*: Ở ảnh `hwe_0000012.png`, UniMERNet sinh ra chuỗi chứa thừa hoặc thiếu dấu ngoặc nhọn `{` hoặc ngoặc vuông `]`, khiến LaTeX không thể render (`render_error` chiếm 18% tổng số lỗi ở cấu hình P0).
   * *Nguyên nhân*: Cơ chế giải mã tự hồi quy (Autoregressive Decoder) của Transformer sinh ra các token tuần tự và dễ bị trôi ngữ cảnh khi gặp công thức quá dài hoặc ảnh bị mờ.
2. **Lỗi phá hủy đặc trưng do tiền xử lý (Preprocessing Degradation)**:
   * *Hiện tượng*: Khi áp dụng các bước tiền xử lý như binarization (P2) hay resize thô bạo (P1), tỷ lệ nhận dạng chính xác của UniMERNet giảm mạnh từ **48% xuống còn 2%**.
   * *Nguyên nhân*: Các mô hình Deep Transformer rất nhạy cảm với sự thay đổi phân phối pixel đầu vào. Việc nhị phân hóa làm mất đi đặc trưng nét vẽ mịn (fine-grained stroke features) cần thiết để nhận diện ký hiệu.

---

## 2. Các lỗi điển hình của CNN-GNN/GAT (Mô hình dựa trên Đồ thị - Graph-based)

Mặc dù đạt kết quả khả quan **52.27% ExpRate** trên CROHME, mô hình lai CNN-GNN/GAT có những lỗi đặc thù liên quan đến chất lượng đồ thị bố cục (Layout Graph):

1. **Lỗi phát hiện ký hiệu sai (Symbol Detection Miss / False Alarm)**:
   * *Hiện tượng*: Tầng CNN/YOLO bỏ sót các ký hiệu nhỏ như dấu chấm phụ, dấu phẩy, hoặc các nét gạch ngang phân số quá mảnh.
   * *Hệ quả*: Một ký hiệu bị bỏ sót sẽ làm mất hoàn toàn nút tương ứng trên đồ thị, dẫn đến việc giải mã LaTeX bị thiếu thành phần hoặc sai lệch cấu trúc nghiêm trọng.
2. **Lỗi phân loại sai mối quan hệ không gian (Spatial Relation Classification Error)**:
   * *Hiện tượng*: Nhầm lẫn giữa quan hệ đứng ngang hàng (Horizontal) và quan hệ số mũ (Superscript) hoặc chỉ số dưới (Subscript).
   * *Ví dụ*: Biểu thức $x_i$ bị nhận nhầm thành $xi$ hoặc $x^i$ do nét viết tay của người dùng bị lệch hoặc có độ cao không đồng đều.
   * *Hệ quả*: GNN nhận thông tin từ các cạnh sai lệch sẽ truyền thông điệp sai (wrong message passing), khiến Decoder sinh ra cấu trúc LaTeX không đúng thực tế.

---

## 3. Ưu thế của việc mô hình hóa bằng Đồ thị (Graph Modeling Benefits)

Mô hình hóa dưới dạng Đồ thị Bố cục Ký hiệu (Symbol Layout Graph) mang lại các điểm ưu việt cốt lõi mà các mô hình giải mã chuỗi phẳng như UniMERNet không có:

* **Bảo toàn cấu trúc 2 chiều**: Cấu trúc đồ thị phản ánh trực tiếp quan hệ không gian vật lý (trên, dưới, trong, ngoài). Điều này giúp khống chế cú pháp LaTeX cực kỳ tốt, ngăn chặn các lỗi sinh token ngẫu nhiên gây mất cân đối ngoặc.
* **Cơ chế truyền tin tường minh (Explicit Message Passing)**: Mạng chú ý đồ thị GAT cho phép các nút ký hiệu tương tác trực tiếp với các nút lân cận dựa trên khoảng cách địa lý 2D, thay vì phải phụ thuộc vào khoảng cách token trong chuỗi 1D.
* **Hỗ trợ hậu xử lý dựa trên luật (Rule-based Post-processing)**: Đồ thị AST xây dựng từ Task 3 cho phép kiểm tra trực quan các lỗi cấu trúc (như nút `frac` bắt buộc phải có hai nhánh con là tử số và mẫu số). Nếu thiếu, mô hình có thể tự động sửa đổi cấu trúc đồ thị trước khi xuất chuỗi LaTeX.

---

## 4. Hạn chế hiện tại của mô hình CNN-GNN đề xuất

* **Phụ thuộc vào chất lượng Object Detection (CNN/YOLO)**: Nếu tầng phát hiện ký hiệu ban đầu hoạt động kém, toàn bộ luồng xử lý phía sau (Graph Construction -> GNN -> Decoder) sẽ bị ảnh hưởng theo hiệu ứng domino.
* **Độ phức tạp tính toán**: Việc xây dựng đồ thị động từ tọa độ hộp bao và tính toán truyền tin trên đồ thị (Message Passing) tốn thêm chi phí CPU/GPU so với các mô hình end-to-end chạy trực tiếp một lượt qua mạng Transformer phẳng.
* **Độ nhạy với chữ viết tay cẩu thả**: Khi các nét viết quá dính nhau hoặc bố cục viết tay quá lệch chuẩn, việc phân tách hộp bao ký hiệu gặp khó khăn, dẫn đến đồ thị được dựng không chính xác.
