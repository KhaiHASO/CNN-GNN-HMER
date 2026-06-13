# Cột mốc 04: Đóng gói và Định vị lại Mô hình lai CNN-GNN (Milestone 4 Report)

Mốc hoàn thành: Task 4 - Định vị Mô hình lai CNN-GNN làm trọng tâm nghiên cứu Luận văn (Tham chiếu GETD - Pattern Recognition 2024).

---

## 1. Mục tiêu và Sự cần thiết của Task 4

### 1.1. Mục tiêu
* Khôi phục, tổ chức và lập hồ sơ đầy đủ cho mô hình lai **CNN-GNN/GAT** gốc dựa trên bài báo tham chiếu chính **GETD (Tang et al., 2024)**.
* Định vị lại cấu trúc luận văn, chuyển mô hình **UniMERNet** (từ Task 1 và Task 2) về đúng vai trò là mô hình đối chứng (Baseline), và **LaTeX Graph Validator** (ở Task 3) làm module bổ trợ hậu xử lý.
* Trả lời rõ ràng 6 câu hỏi cốt lõi của hội đồng chấm luận văn về vị trí và vai trò của CNN, GNN, đồ thị, cơ chế sinh LaTeX, kết quả cũ và vai trò của UniMERNet.

### 1.2. Sự cần thiết
Trong quá trình phát triển nhanh các thử nghiệm thực tế ở Task 1-3, hướng đi của luận văn có nguy cơ bị lệch trọng tâm sang việc nghiên cứu và cải tiến UniMERNet (một mô hình end-to-end Transformer của bên thứ ba). Do tiêu đề chính thức của luận văn là **“Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay”**, việc đóng gói và khẳng định lại vai trò cốt lõi của mô hình lai CNN-GNN/GAT (với nền tảng lý thuyết từ bài báo **GETD 2024** và **GRN**) là bước đi sống còn để bảo vệ sự nhất quán khoa học trước hội đồng.

---

## 2. Kết quả đối chiếu & Câu trả lời cho Hội đồng

Dưới đây là 6 câu trả lời cốt lõi được chuẩn bị kỹ lưỡng cho hội đồng dựa trên thiết kế tham chiếu GETD:

| Câu hỏi | Câu trả lời chính thức | Tài liệu chi tiết |
| :--- | :--- | :--- |
| **1. CNN nằm ở đâu?** | Sử dụng mạng CNN (YOLOv5) để quét ảnh phát hiện vị trí các hộp bao ký hiệu và trích xuất đặc trưng visual ban đầu cho từng nút. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **2. GNN nằm ở đâu?** | GNN đóng vai trò làm **Graph Encoder**, truyền thông điệp (Message Passing) qua mạng chú ý đồ thị GAT để làm giàu thông tin không gian cho nút/cạnh. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **3. Graph là gì?** | **Line-of-Sight (LOS) Symbol Graph**: Đồ thị liên kết ký hiệu, trong đó nút là hộp bao ký hiệu, cạnh nối được thiết lập theo thuật toán tầm nhìn Line-of-Sight. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **4. Sinh LaTeX thế nào?** | Đầu ra đặc trưng nút của Graph Encoder được đưa vào mạng **Transformer Decoder** sử dụng cơ chế Cross-attention để giải mã chuỗi mã LaTeX tuần tự. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **5. Kết quả mô hình?** | Đạt **52.27% ExpRate** trên tập dữ liệu chuẩn quốc tế CROHME (kế thừa từ nghiên cứu chuyên đề, khớp với xu thế mô hình lai đồ thị). | [cnn_gnn_result_crohme.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/cnn_gnn_result_crohme.csv) |
| **6. UniMERNet đóng vai trò gì?** | Là mô hình đối chứng SOTA (Baseline) hiện đại dựa trên kiến trúc Transformer phẳng (Image-to-Sequence), không phải mô hình đề xuất. | [model_role_in_thesis.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/model_role_in_thesis.md) |

---

## 3. Các tài liệu và tệp tin đã tạo

Thư mục `tasks/task4_cnn_gnn_core_model/` đã được đóng gói đầy đủ cấu trúc:
1. **Tài liệu hướng dẫn & Định vị**:
   * [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md): Đặc tả kiến trúc đề xuất tham chiếu GETD.
   * [model_role_in_thesis.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/model_role_in_thesis.md): Làm rõ vai trò mô hình lai CNN-GNN và các baseline.
   * [recovered_assets.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/recovered_assets.md): Báo cáo quét tài nguyên local.
2. **Số liệu thực nghiệm**:
   * [cnn_gnn_result_crohme.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/cnn_gnn_result_crohme.csv): Lưu trữ mốc kết quả 52.27% ExpRate kế thừa.
   * [cnn_gnn_result_quick_test_50.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/cnn_gnn_result_quick_test_50.csv): Khởi tạo tệp tin chỉ chứa tiêu đề để phản ánh trung thực việc thiếu checkpoint chạy cục bộ.
   * [comparison_cnn_gnn_vs_unimernet.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/comparison_cnn_gnn_vs_unimernet.csv): Bảng so sánh 5 cấu hình A0, A1, A2, A3, A4.
3. **Báo cáo phân tích**:
   * [cnn_gnn_experiment_log.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/reports/cnn_gnn_experiment_log.md): Nhật ký chạy thực nghiệm khôi phục.
   * [cnn_gnn_error_analysis.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/reports/cnn_gnn_error_analysis.md): Phân tích so sánh lỗi hệ thống giữa các phương pháp chuỗi và đồ thị.
4. **Kịch bản tự động hóa**:
   * [scan_cnn_gnn_assets.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/scan_cnn_gnn_assets.py): Quét tài nguyên hệ thống.
   * [run_cnn_gnn_inference.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/run_cnn_gnn_inference.py): Kịch bản kiểm thử suy luận.
   * [evaluate_cnn_gnn.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/evaluate_cnn_gnn.py): Kịch bản đánh giá mô hình.
   * [compare_with_unimernet.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/compare_with_unimernet.py): Tổng hợp số liệu so sánh.

---

## 4. Các bước tiếp theo (Next Steps)

1. **Trích dẫn khoa học chính thức**: Đưa bài báo của **Tang et al. (2024) - GETD** vào danh mục tài liệu tham khảo chính của luận văn.
2. **Khai thác đồ thị Line-of-Sight**: Đưa lý thuyết xây dựng đồ thị tầm nhìn (Line-of-Sight) vào phần thiết kế Chương 3 của luận văn.
3. **So sánh luận điểm khoa học**: Sử dụng lập luận từ bài báo **GRN** và **Graph-to-Graph** để phản biện với hội đồng về tại sao mô hình lai CNN-GNN lại có triển vọng giải thích được (interpretability) và hạn chế lỗi cú pháp tốt hơn các baseline Transformer end-to-end thông thường.
