# Nhật ký Thiết kế và Biểu diễn Đồ thị Công thức (Graph Validator Log)

Ngày: 13/06/2026  
Mục tiêu: Chuyển đổi mã LaTeX nhận diện được từ UniMERNet thành đồ thị cú pháp dạng cây để phục vụ kiểm chứng cấu trúc và làm nền tảng cho mạng GNN (Graph Neural Network).

---

## 1. Tại sao biểu thức Toán học nên biểu diễn dưới dạng Đồ thị (Graph)?

Biểu thức toán học không phải là một chuỗi văn bản tuyến tính thông thường. Nó có cấu trúc không gian 2 chiều phân cấp sâu sắc (2D hierarchical structure):
* **Phân số ($\frac{A}{B}$)**: Tử số nằm trên, mẫu số nằm dưới, được phân tách bởi gạch ngang.
* **Căn thức ($\sqrt{A}$)**: Toàn bộ biểu thức con nằm dưới dấu căn.
* **Số mũ ($x^2$) và Chỉ số dưới ($y_i$)**: Các biểu thức con nằm ở góc trên bên phải hoặc dưới bên phải của ký hiệu cơ sở.

Nếu chỉ xử lý dưới dạng chuỗi (String), ta khó lòng nắm bắt được mối quan hệ ngữ nghĩa không gian này. Biểu diễn biểu thức dưới dạng đồ thị có hướng (Directed Graph - DiGraph) giúp chuyển hóa trực tiếp cấu trúc không gian thành cấu trúc liên kết mạng (Network topology), rất phù hợp để làm đầu vào cho mô hình lai **CNN-GNN** (CNN trích xuất đặc trưng hình ảnh cục bộ, GNN tối ưu hóa và kiểm chứng mối liên hệ cú pháp giữa các thực thể ký hiệu).

---

## 2. Cách chuyển đổi Token thành Node và Cấu trúc quan hệ thành Edge

Chúng tôi đã thiết kế bộ dịch cú pháp (Parser AST) và xây dựng đồ thị NetworkX theo nguyên tắc:

### 2.1. Định nghĩa Node (Đỉnh)
Mỗi thành phần trong biểu thức được ánh xạ thành một Node với các thuộc tính:
* `label`: Nội dung ký hiệu (ví dụ: `x`, `+`, `\frac`, `\sqrt`).
* `type`: Kiểu của token (ví dụ: `variable`, `operator`, `frac`, `sqrt`, `group`, `sup`, `sub`).
* `position`: Vị trí ký tự trong chuỗi gốc để tiện đối chiếu ngược.
* `depth`: Độ sâu phân cấp của nút đó trong cây cú pháp (ví dụ: các biến trong tử số phân số sẽ có độ sâu lớn hơn ký hiệu phân số).

### 2.2. Định nghĩa Edge (Cạnh có hướng)
Các cạnh mô tả mối quan hệ ngữ nghĩa giữa các nút:
* `frac_numerator`: Nối từ nút `frac` đến đỉnh gốc của tử số.
* `frac_denominator`: Nối từ nút `frac` đến đỉnh gốc của mẫu số.
* `sqrt_body`: Nối từ nút `sqrt` đến phần thân bên trong căn thức.
* `superscript`: Nối từ nút cơ sở (hoặc nút `sup`) đến số mũ.
* `subscript`: Nối từ nút cơ sở (hoặc nút `sub`) đến chỉ số dưới.
* `inside_group`: Nối từ các nhóm bao bọc `{}` hoặc `()` đến nội dung bên trong.
* `sequential`: Cạnh nối ngang giữa các thành phần liền kề nhau cùng một cấp độ (ví dụ: $x \to + \to 1$).

---

## 3. Liên hệ với Đề tài Luận văn "Hybrid CNN-GNN HMER"

Trong kiến trúc lai CNN-GNN dành cho nhận dạng công thức toán học viết tay:
1. **CNN (Convolutional Neural Network)**: Quét qua ảnh đầu vào để trích xuất đặc trưng visual của các nét vẽ ký hiệu (ví dụ: vùng chứa ký hiệu $x$, vùng chứa nét gạch ngang của phân số).
2. **GNN (Graph Neural Network)**: Nhận đầu vào là các đặc trưng visual từ CNN kết hợp với cấu trúc đồ thị quan hệ không gian được dựng ở đây. GNN thực hiện truyền tin (message passing) giữa các nút để tinh chỉnh nhãn của từng ký hiệu.
   * Ví dụ: Một nét vẽ trông giống dấu trừ `_` nếu có liên kết hướng lên trên dạng `subscript` với một biến $x$, GNN sẽ nhận diện chính xác đó là chỉ số dưới chứ không phải toán tử trừ thông thường.
3. **Graph-based Validator (Bộ kiểm chứng)**: Đồ thị cú pháp giúp phát hiện nhanh các lỗi không thể render hoặc mất cân đối cấu trúc. Đây là cầu nối để đưa ra luật hậu xử lý (Rule-based post-processing) sửa lỗi trực tiếp trên đồ thị trước khi xuất ra mã LaTeX hoàn chỉnh.
