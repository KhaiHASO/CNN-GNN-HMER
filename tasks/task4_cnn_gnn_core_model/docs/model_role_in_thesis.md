# Vai trò của các Mô hình trong Luận văn (Model Roles in Thesis)

Tài liệu này làm rõ vị trí, vai trò của từng mô hình và thực nghiệm trong luận văn để đảm bảo tính nhất quán của đề tài nghiên cứu chính.

---

## 1. Định vị Đề tài Nghiên cứu

Đề tài luận văn chính thức là: **"Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay"**.  
Do đó, mọi phần trình bày lý thuyết, thiết kế kiến trúc đề xuất và đóng góp khoa học chính phải xoay quanh mô hình lai **CNN-GNN/GAT**.

---

## 2. Phân định vai trò các mô hình

Để tránh sự nhầm lẫn giữa mô hình đề xuất và mô hình đối chứng, cấu trúc luận văn quy định vai trò rõ ràng như sau:

### 2.1. Mô hình chính đề xuất: CNN-GNN/GAT Core Pipeline
* **Vị trí**: Nằm ở trung tâm của Chương 3 (Thiết kế hệ thống) và Chương 4 (Thực nghiệm & Đánh giá).
* **Đóng góp**: Là kiến trúc lai do tác giả nghiên cứu, kế thừa và phát triển từ giai đoạn chuyên đề (đã đạt kết quả **52.27% ExpRate trên tập dữ liệu CROHME**).
* **Ý nghĩa**: Chứng minh tính hiệu quả của mạng đồ thị (GNN) trong việc học cấu trúc không gian toán học 2D so với các mô hình dịch chuỗi phẳng truyền thống.

### 2.2. Mô hình đối chứng (Baseline): UniMERNet
* **Vị trí**: Nằm ở phần Baseline đối chứng trong Chương 4.
* **Vai trò**: Đại diện cho SOTA (State-of-the-art) của mô hình end-to-end dựa hoàn toàn trên Transformer hiện nay. 
* **Mục đích**: Chạy thực nghiệm UniMERNet để lấy số liệu đối chứng, chỉ ra những ưu điểm và nhược điểm của mô hình end-to-end so với mô hình lai dựa trên đồ thị bố cục ký hiệu (Symbol Layout Graph).

---

## 3. Vai trò của các thực nghiệm phụ (Task 1 - Task 3)

Các Task đã thực hiện trước đó đóng vai trò là các mảnh ghép thực nghiệm bổ trợ để làm giàu nội dung luận văn:

* **Task 1 (UniMERNet Baseline)**: Thiết lập mốc so sánh (baseline). Đo lường hiệu năng của một mô hình SOTA thực tế trên tập 50 ảnh mẫu viết tay tự chọn.
* **Task 2 (Preprocessing Ablation)**: Nghiên cứu ảnh hưởng của tiền xử lý ảnh. Kết quả chỉ ra tiền xử lý mức pixel (như binarization) làm giảm hiệu năng mô hình học sâu. Điều này củng cố luận điểm của luận văn là **nên giữ nguyên ảnh gốc đầu vào và tập trung xử lý ở tầng đồ thị/hậu xử lý**.
* **Task 3 (LaTeX Syntax Validator)**: Module kiểm chứng cú pháp đầu ra. Đây là một đóng góp bổ trợ ứng dụng thực tế. Nó giúp lọc và phát hiện các lỗi mất cân đối ngoặc (như lỗi `bracket_mismatch` thừa ngoặc đóng `]` ở ảnh `hwe_0000012.png`), định hướng cho các giải pháp hậu xử lý.
