# Cột mốc 04: Đóng gói và Định vị lại Mô hình lai CNN-GNN (Milestone 4 Report)

Mốc hoàn thành: Task 4 - Định vị Mô hình lai CNN-GNN làm trọng tâm nghiên cứu Luận văn.

---

## 1. Mục tiêu và Sự cần thiết của Task 4

### 1.1. Mục tiêu
* Khôi phục, tổ chức và lập hồ sơ đầy đủ cho mô hình lai **CNN-GNN/GAT** gốc.
* Định vị lại cấu trúc luận văn, chuyển mô hình **UniMERNet** (từ Task 1 và Task 2) về đúng vai trò là mô hình đối chứng (Baseline), và **LaTeX Graph Validator** (ở Task 3) làm module bổ trợ hậu xử lý.
* Trả lời rõ ràng 6 câu hỏi cốt lõi của hội đồng chấm luận văn về vị trí và vai trò của CNN, GNN, đồ thị, cơ chế sinh LaTeX, kết quả cũ và vai trò của UniMERNet.

### 1.2. Sự cần thiết
Trong quá trình phát triển nhanh các thử nghiệm thực tế ở Task 1-3, hướng đi của luận văn có nguy cơ bị lệch trọng tâm sang việc nghiên cứu và cải tiến UniMERNet (một mô hình end-to-end Transformer của bên thứ ba). Do tiêu đề chính thức của luận văn là **“Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay”**, việc đóng gói và khẳng định lại vai trò cốt lõi của mô hình lai CNN-GNN/GAT là bước đi sống còn để bảo vệ sự nhất quán khoa học trước hội đồng.

---

## 2. Kết quả đối chiếu & Câu trả lời cho Hội đồng

Dưới đây là 6 câu trả lời cốt lõi được chuẩn bị kỹ lưỡng cho hội đồng:

| Câu hỏi | Câu trả lời chính thức | Tài liệu chi tiết |
| :--- | :--- | :--- |
| **1. CNN nằm ở đâu?** | Trong module trích xuất đặc trưng visual và phát hiện ký hiệu (Object Detection bằng YOLO/CNN backbone). | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **2. GNN nằm ở đâu?** | Trong module học mối quan hệ cấu trúc không gian 2D giữa các ký hiệu thông qua mạng chú ý đồ thị GAT. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **3. Graph là gì?** | Symbol Layout Graph (Đồ thị bố cục ký hiệu). Các nút đại diện cho hộp bao ký hiệu và cạnh đại diện cho quan hệ hình học 2D. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **4. Sinh LaTeX thế nào?** | Đầu ra đặc trưng nút từ GNN/GAT được đưa vào mạng Transformer Decoder để sinh ra chuỗi mã LaTeX tuần tự. | [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) |
| **5. Kết quả cũ bao nhiêu?** | Đạt **52.27% ExpRate** trên tập dữ liệu chuẩn quốc tế CROHME (kế thừa từ nghiên cứu chuyên đề). | [cnn_gnn_result_crohme.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/cnn_gnn_result_crohme.csv) |
| **6. UniMERNet đóng vai trò gì?** | Là mô hình đối chứng SOTA (Baseline) hiện đại, không phải mô hình đề xuất chính của luận văn. | [model_role_in_thesis.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/model_role_in_thesis.md) |

---

## 3. Các tài liệu và tệp tin đã tạo

Thư mục `tasks/task4_cnn_gnn_core_model/` đã được đóng gói đầy đủ cấu trúc:
1. **Tài liệu hướng dẫn & Định vị**:
   * [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md): Đặc tả kiến trúc đề xuất.
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
   * [run_cnn_gnn_inference.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/run_cnn_gnn_inference.py): Kịch bản kiểm thử suy luận (có cảnh báo thiếu checkpoint).
   * [evaluate_cnn_gnn.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/evaluate_cnn_gnn.py): Kịch bản đánh giá mô hình.
   * [compare_with_unimernet.py](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/scripts/compare_with_unimernet.py): Tổng hợp số liệu so sánh.

---

## 4. Các bước tiếp theo (Next Steps)

1. **Chuẩn bị slide và thuyết trình luận văn**: Sử dụng trực tiếp cấu trúc đồ thị bố cục ký hiệu (Symbol Layout Graph) và sơ đồ kiến trúc tại [architecture_cnn_gnn.md](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/docs/architecture_cnn_gnn.md) để giải thích phần lý thuyết đề xuất (Chương 3).
2. **Khai thác dữ liệu thực nghiệm**: Đưa bảng so sánh tại [comparison_cnn_gnn_vs_unimernet.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task4_cnn_gnn_core_model/experiments/comparison_cnn_gnn_vs_unimernet.csv) vào Chương 4 làm minh chứng khoa học.
3. **Phát triển ứng dụng Demo (nếu cần)**: Khi phát triển ứng dụng giao diện (Streamlit hoặc Web App), thiết lập luồng xử lý: Ảnh đầu vào -> Dự đoán LaTeX bằng UniMERNet -> Chuyển chuỗi LaTeX thành đồ thị kiểm chứng cấu trúc bằng LaTeX Graph Validator (Task 3) -> Hiển thị kết quả kiểm chứng lên giao diện. Luồng này giúp tận dụng tính chính xác của UniMERNet đồng thời thể hiện được ý tưởng đồ thị (Graph) trong đề tài.
