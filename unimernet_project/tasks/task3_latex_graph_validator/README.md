# Task 3: LaTeX Syntax Validator and Graph Construction

Thư mục này chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và báo cáo phân tích cho **Task 3: LaTeX Syntax Validator + Graph Construction**.

## 📂 Cấu trúc thư mục
* **`scripts/`**:
  * `latex_tokenizer.py`: Phân tích chuỗi LaTeX thành các tokens toán học.
  * `latex_validator.py`: Kiểm tra và phân loại 6 nhóm lỗi cấu trúc cú pháp LaTeX.
  * `latex_graph_builder.py`: Dựng đồ thị có hướng (DiGraph) từ cây cú pháp AST bằng NetworkX.
  * `visualize_graph.py`: Trực quan hóa và xuất đồ thị thành ảnh PNG.
  * `run_graph_validation.py`: Chạy batch kiểm chứng trên tập 50 ảnh mẫu P0.
* **`experiments/`**:
  * `validation_result_p0.csv`: Chi tiết kết quả kiểm chứng từng ảnh.
  * `graph_statistics.csv`: Thống kê số lượng đỉnh, cạnh, độ sâu trung bình của đồ thị.
  * `error_summary.csv`: Thống kê tần suất các loại lỗi cú pháp toán học.
* **`reports/`**:
  * `graph_validator_log.md`: Nhật ký thiết kế đồ thị và ý nghĩa với luận văn CNN-GNN.
  * `graph_error_analysis.md`: Báo cáo phân tích sâu về các lỗi cấu trúc.
  * `figures/`: Chứa các ảnh trực quan hóa đồ thị mẫu.
* **`docs/`**:
  * `milestone_03_latex_graph_validator.md`: Báo cáo Milestone 3.

## 🚀 Cách chạy thử nghiệm
Kích hoạt môi trường conda của dự án và chạy các lệnh sau:

1. **Khởi chạy kiểm chứng batch 50 ảnh**:
   ```bash
   conda run -n unimernet python tasks/task3_latex_graph_validator/scripts/run_graph_validation.py
   ```
2. **Khởi chạy trực quan hóa đồ thị mẫu**:
   ```bash
   conda run -n unimernet python tasks/task3_latex_graph_validator/scripts/visualize_graph.py
   ```
