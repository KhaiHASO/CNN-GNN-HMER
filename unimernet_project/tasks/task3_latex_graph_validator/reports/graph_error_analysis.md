# Phân tích Lỗi Cú pháp và Đồ thị (Graph Error Analysis)

Bài viết này phân tích sâu về các loại lỗi cú pháp LaTeX mà mô hình UniMERNet sinh ra trên tập 50 ảnh mẫu, đồng thời chỉ rõ vai trò của bộ kiểm chứng đồ thị trong việc phát hiện và định hướng sửa lỗi.

---

## 1. Phân tích các loại lỗi phát hiện trong thực nghiệm

Qua thống kê thực nghiệm trên 50 ảnh thuộc cấu hình tốt nhất P0, bộ kiểm chứng phát hiện:

| Loại lỗi | Số lượng | Mô tả hành vi của mô hình |
| :--- | :---: | :--- |
| **none** (Hợp lệ) | 40 | Mã LaTeX sinh ra có cấu trúc hoàn chỉnh, đúng cú pháp và xây dựng được đồ thị cây AST. |
| **render_error** | 9 | Cú pháp toán học hợp lệ nhưng chứa các lệnh phức tạp (ví dụ: `\begin{array}`) mà bộ thư viện render của matplotlib không hỗ trợ hiển thị trực tiếp. |
| **bracket_mismatch** | 1 | Có lỗi mất cân bằng dấu đóng/mở ngoặc đơn, ngoặc nhọn hoặc ngoặc vuông. |
| **unknown_command** | 0 | (Đã được làm sạch sau khi tối ưu hóa bộ từ điển Whitelist). |

### 1.1. Ví dụ lỗi cấu trúc điển hình (`bracket_mismatch`)
Lỗi xảy ra ở ảnh viết tay mẫu **`hwe_0000012.png`**:
* **Nhãn gốc (GT)**: `\{ a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } \}` (Mô tả một tập hợp trong cặp dấu ngoặc nhọn).
* **UniMERNet dự đoán**: `\sum a _ { 1 } a _ { 2 , a _ { 3 } , a _ { 4 } } ]`
* **Kết quả kiểm chứng**: `Unmatched closing ]` (Phát hiện dấu đóng ngoặc vuông thừa ở cuối chuỗi).
* **Nguyên nhân**: Mô hình sinh tự hồi quy (Autoregressive) bị mất dấu ngữ cảnh mở ngoặc hoặc nhầm lẫn hình ảnh nét vẽ ngoặc nhọn viết tay thành một ký tự khác, dẫn đến việc đóng ngoặc bừa bãi ở cuối chuỗi.

---

## 2. Khả năng và Giới hạn của Bộ kiểm chứng Đồ thị cú pháp (Rule-based)

### 2.1. Khả năng phát hiện (What it can do)
* **Lỗi cú pháp cứng (Hard syntax errors)**: Bất kỳ lỗi nào làm gãy cây cú pháp AST như thiếu ngoặc nhọn `}`, thiếu tham số của phân số `\frac{A}`, hoặc toán tử số mũ `^` đứng chơ vơ đều bị chặn đứng ở bước dựng đồ thị.
* **Lỗi cấu trúc rỗng**: Phát hiện các dấu ngoặc rỗng `{}` thường sinh ra do mô hình dự đoán nhầm.
* **Định hướng khôi phục**: Nhờ cấu trúc đồ thị, ta có thể biết chính xác vị trí nào thiếu dấu đóng ngoặc để tự động chèn thêm dấu đóng ngoặc tương ứng (Rule-based correction ở Task 4).

### 2.2. Giới hạn (What it CANNOT do)
* **Lỗi nhận diện sai ký hiệu (Symbol Misrecognition)**: 
  * Ví dụ: Ảnh chụp phép toán $1 + 1$ nhưng UniMERNet nhận diện thành $1 - 1$. 
  * Về mặt cú pháp, chuỗi `1 - 1` hoàn toàn đúng và dựng được đồ thị hoàn chỉnh. Bộ kiểm chứng rule-based **không thể biết** mô hình nhận diện đúng hay sai so với ảnh gốc vì nó không hề nhìn vào bức ảnh.
* **Sai ngữ nghĩa toán học**: Bộ kiểm chứng chỉ kiểm tra tính hợp lệ về cấu trúc viết LaTeX chứ không kiểm tra tính đúng đắn của logic toán (ví dụ chia cho 0).

---

## 3. Giải pháp khắc phục bằng Graph Neural Network (GNN) và Hậu xử lý

Để khắc phục giới hạn của bộ kiểm chứng rule-based, hướng nghiên cứu của luận văn đề xuất:
1. **Kết hợp đặc trưng hình ảnh (Visual-Syntax Consistency Check)**:
   * GNN sẽ nhận các đặc trưng vùng ảnh (từ CNN) gán vào các nút tương ứng trên đồ thị.
   * GNN học cách phân loại xem mối quan hệ cạnh (ví dụ `superscript` giữa nút $x$ và nút $2$) có thực sự khớp với vị trí địa lý của chúng trong ảnh viết tay hay không. Nếu ảnh không có cấu trúc số mũ mà đồ thị lại có, GNN sẽ phát hiện ra sự mâu thuẫn này để điều chỉnh.
2. **Hậu xử lý sửa lỗi tự động (Rule-based Post-processing)**:
   * Đối với lỗi mất cân bằng ngoặc nhọn (`bracket_mismatch`), viết thuật toán quét chuỗi đếm số lượng mở `{` và đóng `}` để tự động chèn thêm dấu đóng ngoặc vào cuối biểu thức hoặc trước toán tử quan trọng. Việc này sẽ được triển khai ngay tại **Task 4**.
