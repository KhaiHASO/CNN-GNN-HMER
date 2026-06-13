# Kiến trúc Mô hình lai CNN-GNN (CNN-GNN Core Architecture)

Tài liệu này trình bày chi tiết về kiến trúc mô hình chính thức của đề tài: **"Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay"**.

---

## 1. Sơ đồ khối tổng quan (Pipeline Diagram)

Kiến trúc của mô hình lai CNN-GNN được thiết kế theo luồng xử lý từ hình ảnh nét vẽ viết tay đến mã LaTeX ngữ nghĩa:

```text
+-----------------------+      +-----------------------+
|  Ảnh biểu thức viết   | ---> |       CNN / YOLO      | (Trích xuất đặc trưng visual
|          tay          |      |   (Symbol Detection)  |  và phát hiện hộp bao ký hiệu)
+-----------------------+      +-----------------------+
                                           |
                                           v
+-----------------------+      +-----------------------+
|   GNN / GAT Encoder   | <--- |  Symbol Layout Graph  | (Xây dựng đồ thị liên kết không gian:
|   (Message Passing)   |      |  (Graph Construction) |  Node = Ký hiệu, Edge = Quan hệ)
+-----------------------+      +-----------------------+
            |
            v
+-----------------------+      +-----------------------+
|  Transformer Decoder  | ---> |   Mã LaTeX đầu ra     |
|   (Sequence Gen)      |      |  (LaTeX Expression)   |
+-----------------------+      +-----------------------+
```

---

## 2. Chi tiết các thành phần trong kiến trúc

Mô hình lai kết hợp sức mạnh trích xuất đặc trưng ảnh của mạng nơ-ron tích chập (CNN) và khả năng học mối quan hệ cấu trúc phi tuyến tính của mạng nơ-ron đồ thị (GNN):

### 2.1. Tầng trích xuất đặc trưng và phát hiện ký hiệu (CNN / YOLO)
* **Đầu vào**: Ảnh nhị phân hoặc xám của biểu thức toán học viết tay.
* **Vai trò**:
  * Sử dụng mạng xương sườn (Backbone) CNN (ví dụ: ResNet hoặc YOLOv8/YOLOv9) để quét qua ảnh.
  * Tự động phát hiện các hộp bao (bounding box) của từng ký hiệu toán học riêng lẻ (như chữ số, biến số, toán tử, dấu phân số, dấu căn).
  * Trích xuất vector đặc trưng visual (visual feature vector) của từng ký hiệu nằm trong hộp bao.

### 2.2. Xây dựng đồ thị bố cục ký hiệu (Symbol Layout Graph Construction)
* **Vai trò**:
  * Mỗi ký hiệu được phát hiện sẽ được ánh xạ thành một **Nút (Node)** trong đồ thị. Đặc trưng ban đầu của nút chính là vector đặc trưng visual trích xuất từ CNN.
  * Thiết lập các **Cạnh (Edge)** dựa trên khoảng cách và quan hệ hình học không gian 2D giữa các hộp bao:
    * Cạnh nối ngang (Horizontal): Ký hiệu đứng cạnh nhau tuyến tính ($x$ nối với $+$).
    * Cạnh số mũ (Superscript): Ký hiệu nằm lệch trên bên phải ($x$ nối với $2$).
    * Cạnh chỉ số dưới (Subscript): Ký hiệu nằm lệch dưới bên phải ($y$ nối với $i$).
    * Cạnh phân số (Fraction): Quan hệ giữa dấu gạch ngang phân số và tử số (nút nằm trên), mẫu số (nút nằm dưới).
    * Cạnh căn thức (Sqrt): Quan hệ giữa dấu căn và các ký hiệu nằm lọt bên trong.

### 2.3. Tầng mã hóa mạng đồ thị (GNN / GAT Encoder)
* **Vai trò**:
  * Sử dụng mạng lan truyền thông tin đồ thị (GNN), cụ thể là mạng chú ý đồ thị **GAT (Graph Attention Network)**.
  * GAT thực hiện cơ chế **Message Passing** (truyền thông điệp) giữa các nút lân cận. Cơ chế chú ý (attention head) giúp mô hình tự động gán trọng số liên kết quan trọng giữa các ký hiệu (ví dụ: số mũ $2$ phải chú ý mạnh đến cơ số $x$).
  * Đầu ra của GAT là các vector đặc trưng của nút đã được làm giàu thông tin ngữ cảnh cấu trúc toán học 2D.

### 2.4. Tầng giải mã chuỗi (Transformer Decoder)
* **Vai trò**:
  * Nhận các vector đặc trưng nút từ GNN.
  * Sử dụng cơ chế chú ý chéo (Cross-attention) để dịch chuyển thông tin đồ thị thành chuỗi tuần tự mã LaTeX.
  * Áp dụng thuật toán tìm kiếm Beam Search để sinh ra mã LaTeX tối ưu nhất, đảm bảo cân bằng cú pháp.
