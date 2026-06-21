# Vai trò của các Mô hình trong Luận văn (Model Roles in Thesis)

Tài liệu này làm rõ vị trí, vai trò của từng mô hình và thực nghiệm trong luận văn để đảm bảo tính nhất quán của đề tài nghiên cứu chính, có đối chiếu với các công trình khoa học quốc tế.

---

## 1. Định vị Đề tài Nghiên cứu

Đề tài luận văn chính thức là: **"Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay"**.  
Do đó, mọi phần trình bày lý thuyết, thiết kế kiến trúc đề xuất và đóng góp khoa học chính phải xoay quanh mô hình lai **CNN-GNN/GAT**.

Để cụ thể hóa thiết kế hệ thống, luận văn lựa chọn nghiên cứu của **Tang et al. (2024) - GETD (Graph Encoder and Transformer Decoder)** công bố trên tạp chí *Pattern Recognition* làm **kiến trúc tham chiếu cốt lõi**. Luồng xử lý chính bao gồm:
$$\text{YOLOv5 (CNN)} \rightarrow \text{Line-of-Sight Graph} \rightarrow \text{GNN Encoder} \rightarrow \text{Transformer Decoder} \rightarrow \text{LaTeX}$$

---

## 2. Phân định vai trò các mô hình trong thực nghiệm

Để tránh sự nhầm lẫn giữa mô hình đề xuất và mô hình đối chứng, cấu trúc luận văn quy định vai trò rõ ràng như sau:

### 2.1. Mô hình chính đề xuất: CNN-GNN/GAT Core Pipeline (Tham chiếu GETD)
* **Vị trí**: Nằm ở trung tâm của Chương 3 (Thiết kế hệ thống) và Chương 4 (Thực nghiệm & Đánh giá).
* **Đóng góp**: Là kiến trúc lai kết hợp mạng tích chập (CNN) phát hiện ký hiệu, cấu trúc đồ thị Line-of-Sight, GNN/GAT mã hóa đặc trưng không gian và Transformer giải mã chuỗi.
* **Số liệu chính thức**: Kế thừa kết quả chuyên đề đạt **52.27% ExpRate trên tập dữ liệu CROHME 2014**.
* **Ý nghĩa khoa học**: Minh chứng cho việc mô hình hóa quan hệ không gian 2D rõ ràng bằng đồ thị (theo lập luận của bài báo **GRN - Graph Reasoning Network**) giúp nhận dạng cấu trúc toán học bền vững hơn so với việc ép phẳng ảnh thành chuỗi.

### 2.2. Mô hình đối chứng (Baseline): UniMERNet
* **Vị trí**: Nằm ở phần Baseline đối chứng trong Chương 4.
* **Vai trò**: Đại diện cho SOTA (State-of-the-art) của mô hình end-to-end dựa hoàn toàn trên mạng Transformer thị giác phẳng hiện nay.
* **Mục đích**: Lấy số liệu đối chứng thực tế để chỉ ra ưu/nhược điểm của mô hình end-to-end so với mô hình lai dựa trên đồ thị bố cục ký hiệu (Symbol Layout Graph).

---

## 3. Vai trò của các thực nghiệm phụ (Task 1 - Task 3)

Các Task đã thực hiện trước đó đóng vai trò là các mảnh ghép thực nghiệm bổ trợ để làm giàu nội dung luận văn:

* **Task 1 (UniMERNet Baseline)**: Thiết lập mốc so sánh (baseline). Đo lường hiệu năng của một mô hình SOTA thực tế trên tập 50 ảnh mẫu viết tay tự chọn.
* **Task 2 (Preprocessing Ablation)**: Nghiên cứu ảnh hưởng của tiền xử lý ảnh. Kết quả chỉ ra tiền xử lý mức pixel (như binarization) làm giảm hiệu năng mô hình học sâu. Điều này củng cố luận điểm của luận văn là **nên giữ nguyên ảnh gốc đầu vào để CNN tự trích đặc trưng và tập trung giải quyết phân cấp cấu trúc ở tầng đồ thị/hậu xử lý**.
* **Task 3 (LaTeX Syntax Validator)**: Module kiểm chứng cú pháp đầu ra dựa trên đồ thị có hướng (AST-like Graph). Đây là một đóng góp bổ trợ ứng dụng thực tế, lấy cảm hứng từ lý thuyết biểu diễn đồ thị biểu thức trong nghiên cứu **Graph-to-Graph (AAAI)** và cấu trúc `math_online_egat`. Nó giúp lọc và phát hiện các lỗi mất cân đối ngoặc (như lỗi `bracket_mismatch` thừa ngoặc đóng `]` ở ảnh `hwe_0000012.png`), định hướng cho các giải pháp hậu xử lý.
