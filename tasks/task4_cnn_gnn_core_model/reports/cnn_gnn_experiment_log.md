# Nhật ký Thực nghiệm Mô hình lai CNN-GNN (CNN-GNN Experiment Log)

Ngày báo cáo: 13/06/2026  
Mục đích: Đóng gói và làm rõ tình trạng thực nghiệm của mô hình lai CNN-GNN/GAT phục vụ báo cáo luận văn.

---

## 1. Kết quả quét tài nguyên khôi phục (Recovered Assets)

Chúng tôi đã chạy tập lệnh `scan_cnn_gnn_assets.py` để tìm kiếm các tài nguyên liên quan đến mô hình CNN-GNN trong kho lưu trữ local. 
* **Tài nguyên phát hiện được**: Không tìm thấy mã nguồn mạng GNN, checkpoint mô hình (`.pth` hoặc `.pt`), hay cấu hình đồ thị cũ trong thư mục làm việc hiện tại.
* **Nguyên nhân**: Toàn bộ dự án local hiện tại được cấu hình cho mô hình baseline UniMERNet (gồm môi trường Conda, các tập lệnh chạy thử nghiệm 50 ảnh ở Task 1-3).
* **Kết luận**: Không thể tiến hành chạy huấn luyện hoặc thực hiện suy luận (inference) trực tiếp đối với mô hình CNN-GNN trên môi trường máy local hiện tại.

---

## 2. Kết quả thực nghiệm kế thừa (CROHME Benchmark)

Do không thể chạy suy luận cục bộ, chúng tôi sử dụng và khôi phục bảng số liệu thực nghiệm chính thức từ giai đoạn nghiên cứu chuyên đề trước đó:

* **Tập dữ liệu thử nghiệm**: CROHME 2014 (tập kiểm thử chuẩn quốc tế cho nhận dạng biểu thức toán học viết tay).
* **Độ đo đánh giá chính**: **ExpRate** (Expression Recognition Rate - Tỷ lệ nhận dạng chính xác toàn bộ biểu thức, độ đo khắt khe nhất trong HMER).
* **Kết quả đạt được**: **52.27% ExpRate**.
* **Đánh giá**: Đây là một kết quả khả quan đối với mô hình lai tự phát triển dựa trên bố cục đồ thị (Symbol Layout Graph) kết hợp GAT (Graph Attention Network) và Transformer Decoder, làm nền tảng vững chắc cho đề xuất của luận văn.

---

## 3. Nhật ký chạy thử nghiệm 50 ảnh mẫu (Quick Test 50)

Theo yêu cầu kiểm thử nhanh trên 50 ảnh viết tay của Task 1 và Task 2:
* **Tình trạng chạy**: **Không khả thi (N/A)**.
* **Xử lý số liệu**: Để đảm bảo tính trung thực khoa học, chúng tôi không tự tạo (fake) các dự đoán giả lập cho mô hình CNN-GNN trên 50 ảnh này. File dữ liệu `cnn_gnn_result_quick_test_50.csv` được tạo ra dưới dạng chỉ chứa dòng tiêu đề (header-only) kèm theo phần giải thích chi tiết trong tài liệu này.

---

## 4. Ý nghĩa đối với Đề tài Luận văn

Báo cáo thực nghiệm này định vị lại cấu trúc thực nghiệm cho toàn bộ luận văn:
1. **Khẳng định mô hình cốt lõi**: Mô hình lai CNN-GNN/GAT là trọng tâm nghiên cứu. Các kết quả 52.27% ExpRate trên CROHME là đóng góp chính.
2. **Vai trò của UniMERNet**: UniMERNet đóng vai trò là mô hình đối chứng (Baseline SOTA) để so sánh hiệu năng. Việc UniMERNet đạt độ chính xác cao trên 50 ảnh mẫu (Exact Match 48% ở cấu hình P0) làm nổi bật sức mạnh của các mô hình Transformer end-to-end hiện đại, nhưng cũng chỉ ra các điểm yếu về tính dễ vỡ cấu trúc và lỗi cú pháp.
3. **Phân tích lỗi so sánh**: Phân định rõ sự khác biệt giữa lỗi cú pháp chuỗi của UniMERNet và tính bền vững cấu trúc đồ thị của mô hình CNN-GNN/GAT (chi tiết tại file phân tích lỗi `reports/cnn_gnn_error_analysis.md`).
