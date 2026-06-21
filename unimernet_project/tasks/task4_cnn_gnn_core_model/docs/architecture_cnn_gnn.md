# Kiến trúc Mô hình lai CNN-GNN (CNN-GNN Core Architecture)

Tài liệu này trình bày chi tiết về kiến trúc mô hình chính thức của đề tài: **"Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay"**, dựa trên kiến trúc tham chiếu từ nghiên cứu SOTA của bài báo **GETD (Graph Encoder and Transformer Decoder)** đăng trên tạp chí *Pattern Recognition (2024)*.

---

## 1. Sơ đồ khối tổng quan (Pipeline Diagram)

Kiến trúc của mô hình lai CNN-GNN tham chiếu trực tiếp từ mô hình **GETD (Tang et al., 2024)**, đi theo luồng xử lý từ hình ảnh nét vẽ viết tay đến mã LaTeX ngữ nghĩa:

```text
+-----------------------+      +-----------------------+
|  Ảnh biểu thức viết   | ---> |        YOLOv5         | (CNN-based Backbone phát hiện
|      tay (Offline)    |      |   (Symbol Detector)   |  hộp bao và trích đặc trưng)
+-----------------------+      +-----------------------+
                                           |
                                           v
+-----------------------+      +-----------------------+
|  Graph Encoder (GNN)  | <--- |  Symbol Layout Graph  | (Xây dựng đồ thị ký hiệu dựa
|   (Message Passing)   |      |  (Line-of-Sight - LOS)|  trên thuật toán Line-of-Sight)
+-----------------------+      +-----------------------+
            |
            v
+-----------------------+      +-----------------------+
|  Transformer Decoder  | ---> |   Mã LaTeX đầu ra     |
|   (Sequence Gen)      |      |  (LaTeX Expression)   |
+-----------------------+      +-----------------------+
```

---

## 2. Chi tiết các thành phần trong kiến trúc tham chiếu (GETD)

Mô hình lai kết hợp khả năng phát hiện đối tượng cục bộ mạnh mẽ của CNN (YOLOv5), cấu trúc đồ thị bố cục ký hiệu không gian 2D, và cơ chế giải mã tự hồi quy của Transformer:

### 2.1. Phát hiện ký hiệu bằng CNN (YOLOv5 Symbol Detector)
* **Đầu vào**: Ảnh biểu thức toán học viết tay (Offline).
* **Cơ chế**:
  * Sử dụng mạng YOLOv5 làm Backbone tích chập (CNN) để phát hiện và định vị các hộp bao (bounding box) cho từng ký hiệu toán học riêng biệt (như chữ số, chữ cái, toán tử, phân số).
  * Mỗi hộp bao ký hiệu sau khi phát hiện sẽ được trích xuất vector đặc trưng thị giác (visual feature representation) tích hợp thông tin vị trí không gian $(x, y, w, h)$.

### 2.2. Xây dựng đồ thị bố cục ký hiệu bằng Line-of-Sight (LOS Graph Construction)
* **Cơ chế**:
  * Ánh xạ các ký hiệu được phát hiện thành các **Nút (Nodes)** trong đồ thị.
  * Sử dụng thuật toán **Line-of-Sight (LOS)** để xác định liên kết **Cạnh (Edges)**. Theo thuật toán LOS, hai ký hiệu sẽ có cạnh nối nếu chúng có thể "nhìn thấy" nhau theo các hướng hình học 2D mà không bị che khuất bởi các hộp bao ký hiệu khác.
  * Việc này loại bỏ các cạnh dư thừa và giữ lại đúng mối quan hệ không gian thực tế (trên-dưới của phân số, số mũ, chỉ số dưới, hoặc quan hệ tuần tự ngang).

### 2.3. Mã hóa Đồ thị bằng GNN (Graph Encoder)
* **Cơ chế**:
  * Sử dụng mạng nơ-ron đồ thị (GNN), điển hình là **Graph Attention Network (GAT)** hoặc các biến thể học đặc trưng cạnh (Edge GNN).
  * Thực hiện cơ chế truyền thông điệp (Message Passing) qua các nút lân cận trên đồ thị để tích hợp thông tin ngữ cảnh không gian 2D.
  * Vector đặc trưng của mỗi ký hiệu sau bước này không chỉ chứa thông tin nét vẽ vật lý (từ CNN) mà còn chứa thông tin phân cấp toán học của cấu trúc xung quanh nó.

### 2.4. Giải mã chuỗi bằng Transformer Decoder
* **Cơ chế**:
  * Nhận các đặc trưng nút đồ thị đã được làm giàu cấu trúc từ Graph Encoder.
  * Transformer Decoder sử dụng cơ chế chú ý chéo (Cross-attention) để giải mã tự hồi quy (autoregressive) sang chuỗi ký tự LaTeX tương ứng.
  * Đảm bảo tính cân bằng cú pháp toán học thông qua việc học phân phối chuỗi toàn cục.

---

## 3. Các Nghiên cứu Liên quan và Cơ sở Lý thuyết (Related Works)

Để củng cố nền tảng lý thuyết cho mô hình đề xuất trong luận văn, chúng tôi sử dụng các tài liệu khoa học sau làm minh chứng và tài liệu tham chiếu bổ trợ:

### 3.1. Nghiên cứu chính: GETD (Tang et al., 2024)
* **Tên bài báo**: *Offline Handwritten Mathematical Expression Recognition with Graph Encoder and Transformer Decoder* (Pattern Recognition, 2024).
* **Đóng góp**: Đề xuất trực tiếp luồng xử lý YOLOv5 $\rightarrow$ LOS Symbol Graph $\rightarrow$ GNN Encoder $\rightarrow$ Transformer Decoder. Đây là bài báo khớp 90% với định hướng thiết kế và đặt tên đề tài của luận văn này, làm tài liệu tham chiếu chính thức để thiết kế hệ thống.

### 3.2. Nghiên cứu nền tảng: GRN (Graph Reasoning Network)
* **Tên bài báo**: *Offline Handwritten Mathematical Expression Recognition via Graph Reasoning Network* (ResearchGate).
* **Đóng góp**: Đưa ra lập luận lý thuyết cực kỳ vững chắc về sự cần thiết của GNN trong HMER: HMER có bố cục 2D phức tạp mà các mô hình phẳng Image-to-Sequence truyền thống dễ bị mất cấu trúc. GRN xây dựng đồ thị ký hiệu và dùng suy luận đồ thị để dịch sang cây cú pháp (Symbol Layout Tree). Luận văn sẽ trích dẫn bài này để bảo vệ lập luận về mặt lý thuyết đồ thị tại Chương 1 và Chương 3.

### 3.3. Nghiên cứu mở rộng: Graph-to-Graph (G2G)
* **Tên bài báo**: *Graph-to-Graph: Towards Accurate and Interpretable Online Handwritten Mathematical Expression Recognition* (AAAI).
* **Đóng góp**: Định nghĩa bài toán HMER dưới dạng Graph-to-Graph, dùng GNN ở cả Encoder và Decoder để đảm bảo tính giải thích được (interpretability). Mặc dù nghiên cứu này nhắm vào dữ liệu Online (nét viết stroke), nó hỗ trợ mạnh mẽ lý thuyết về biểu diễn biểu thức dưới dạng đồ thị có hướng.

### 3.4. Code tham khảo cấu trúc GAT/EGAT: math_online_egat
* **Repo tham khảo**: `math_online_egat` (PyTorch + DGL).
* **Mô tả**: Triển khai Edge Graph Attention Network (EGAT) để dự đoán liên kết và chuẩn hóa cấu trúc đồ thị. Đây là nguồn mã nguồn mở hữu ích để tham khảo cách xây dựng đồ thị biểu thức toán học và thiết lập truyền thông điệp, tuy nhiên cần lưu ý repo này thiết kế cho dữ liệu Online (stroke-level) bằng BLSTM và CYK parser chứ không phải ảnh offline.
