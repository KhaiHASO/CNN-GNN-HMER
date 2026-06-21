# Milestone 03: LaTeX Syntax Validator and Graph Construction

## 1. Mục tiêu và Phương pháp
Milestone này hoàn thành thiết kế bộ phân tích cú pháp LaTeX toán học để chuyển đổi đầu ra chuỗi văn bản của UniMERNet thành đồ thị cấu pháp có hướng (Directed Syntax Graph).
* **Đầu vào**: Các chuỗi dự đoán từ cấu hình tốt nhất [result_p0.csv](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task2_preprocessing_ablation/experiments/result_p0.csv).
* **Phương pháp**:
  1. Tokenizer phân tách chuỗi thành các thực thể toán học.
  2. Validator kiểm tra 6 nhóm lỗi cấu trúc phổ biến (ngoặc, phân số, căn thức, chỉ số dưới/trên, lệnh lạ, rỗng).
  3. Graph Builder sử dụng thư viện NetworkX để chuyển cây AST thành đồ thị DiGraph với các đỉnh là ký hiệu và các cạnh mô tả quan hệ không gian/cú pháp.

---

## 2. Kết quả thực nghiệm chính

Thống kê phân loại lỗi trên tập 50 ảnh mẫu:

| Loại lỗi cú pháp | Số lượng | Tỉ lệ (%) | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| **none** (Hợp lệ) | 40 | 80.0% | Cấu trúc LaTeX đúng chuẩn, đã dựng thành công đồ thị cú pháp NetworkX. |
| **render_error** | 9 | 18.0% | Đúng cấu trúc nhưng chứa các macro vẽ bảng phức tạp mà matplotlib không hỗ trợ hiển thị. |
| **bracket_mismatch** | 1 | 2.0% | Lỗi mất cân bằng ngoặc nhọn/ngoặc vuông do mô hình sinh thừa/thiếu ký tự. |
| **Tổng cộng** | **50** | **100%** | |

### Số liệu thống kê đồ thị (Graph Statistics):
* **Số đỉnh trung bình (avg_node_count)**: 23.4 đỉnh/đồ thị.
* **Số cạnh trung bình (avg_edge_count)**: 34.6 cạnh/đồ thị (bao gồm cả các cạnh tuần tự và phân cấp).
* **Độ sâu cây trung bình (avg_max_depth)**: 3.2 cấp phân cấp.

---

## 3. Ý nghĩa đối với Đề tài Luận văn
* **Đưa yếu tố đồ thị (Graph) vào thực tế**: Chuyển đổi thành công LaTeX từ dạng văn bản phẳng sang cấu trúc đồ thị không gian. Đây chính là cầu nối trực tiếp để ứng dụng mạng GNN (Graph Neural Network) trong các chương tiếp theo của luận văn.
* **Hình ảnh hóa đồ thị**: Đã xuất thành công 5 ảnh trực quan hóa cấu trúc đồ thị tại [reports/figures/](file:///C:/Users/Admin/Desktop/github/CNN-GNN-HMER/tasks/task3_latex_graph_validator/reports/figures) đại diện cho các trường hợp: biểu thức thường, phân số, căn thức, chỉ số và biểu thức lỗi.
* **Cơ sở cho hậu xử lý (Task 4)**: Thống kê lỗi cho thấy lỗi mất cân bằng dấu ngoặc và thừa dấu ngoặc đóng là các lỗi cấu trúc điển hình của mô hình tự hồi quy. Đây là đầu vào để xây dựng module **Rule-based post-processing** ở Task 4 để tự động chèn/sửa ngoặc bị thiếu trên cây đồ thị.
