# Phân tích Lỗi Tiền xử lý (Preprocessing Error Analysis)

Báo cáo này phân tích chi tiết nguyên nhân tại sao các phương pháp tiền xử lý ảnh truyền thống (như cắt biên, thay đổi kích thước và nhị phân hóa) lại làm suy giảm nghiêm trọng độ chính xác của UniMERNet.

## 1. Phân loại Lỗi chi tiết theo cấu hình

Số lượng lỗi phát hiện trong từng nhóm trên tập 50 ảnh:

| Cấu hình | Lỗi Ngoặc (Bracket) | Lỗi Phân Số (Fraction) | Lỗi Số Mũ/Chỉ Số (Sup/Sub) | Lỗi Căn Thức (Sqrt) | Lỗi Ký Hiệu (Symbol) | Lỗi Khác |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0 (Ảnh gốc)** | 13 | 0 | 0 | 0 | 7 | 6 |
| **P1 (Chuẩn hóa)** | 22 | 4 | 3 | 3 | 6 | 4 |
| **P2 (Nhị phân)** | 12 | 6 | 15 | 9 | 6 | 1 |

---

## 2. Phân tích nguyên nhân suy giảm hiệu năng

### 2.1. Sự biến dạng nét vẽ và tỉ lệ (Stroke & Aspect Ratio Distortion) - Ảnh hưởng của P1
* **Hiện tượng**: Khi áp dụng P1 (cắt biên rồi resize chiều cao về 192px), các công thức quá ngắn hoặc quá dài sẽ bị co giãn không tự nhiên.
* **Nguyên nhân**: UniMERNet sử dụng backbone Swin Transformer được tiền huấn luyện trên các ảnh công thức có kích thước font chữ và độ dày nét vẽ tương đối nhất quán. Việc thay đổi kích thước làm biến đổi độ dày nét chữ (strokes) khiến bộ trích xuất đặc trưng nhận diện sai các ký hiệu nhỏ (như số mũ, dấu chấm, dấu phẩy).
* **Minh chứng**: Nhiều công thức ở P1 bắt đầu xuất hiện lỗi nhận diện cấu trúc phân số (`fraction_structure`) và số mũ (`superscript_subscript`) mà ở ảnh gốc P0 không hề bị.

### 2.2. Mất thông tin vùng biên (Loss of Anti-Aliasing) - Ảnh hưởng của P2
* **Hiện tượng**: Nhị phân hóa Otsu trong P2 đưa các giá trị pixel về tuyệt đối 0 hoặc 255. 
* **Nguyên nhân**: Các mô hình mạng nơ-ron tích chập (CNN) hoặc Transformer trích xuất đặc trưng biên dựa trên các dải gradient xám mượt mà (anti-aliasing). Nhị phân hóa làm cho viền chữ bị răng cưa (jaggy) và mất đi độ sâu của nét vẽ.
* **Hậu quả**: Mô hình mất khả năng nhận diện các cấu trúc lồng nhau phức tạp. Số lỗi số mũ/chỉ số dưới nhảy vọt từ 0 (ở P0) lên **15 lỗi** (ở P2). Số lỗi căn thức tăng từ 0 lên **9 lỗi**.

### 2.3. Bộ lọc nhiễu "nuốt" mất ký tự (Filter Over-smoothing) - Ảnh hưởng của P2
* **Hiện tượng**: Bộ lọc Median Blur (khử nhiễu trung vị) trong P2 làm mượt các chấm nhiễu.
* **Nguyên nhân**: Bộ lọc này không phân biệt được đâu là nhiễu hạt và đâu là các ký tự toán học siêu nhỏ như dấu chấm nhân (`\cdot`), dấu chấm của chữ $i$ hoặc $j$, dấu phẩy, hoặc các chỉ số mũ cực nhỏ.
* **Hậu quả**: Các nét vẽ thanh mảnh hoặc dấu chấm quan trọng bị xóa nhòa, dẫn đến việc mô hình dịch sai hoàn toàn ký hiệu toán học hoặc bỏ sót cấu trúc.

---

## 3. Ý nghĩa đối với Đề tài Luận văn

Thử nghiệm này mang lại một đóng góp thực nghiệm quan trọng cho Chương 4 của luận văn:
1. **Phủ định quan điểm cũ**: Đối với các hệ thống MER (nhận diện công thức) hiện đại dựa trên mô hình Sequence-to-Sequence (như UniMERNet), việc áp dụng các bước tiền xử lý ảnh cổ điển (nhị phân hóa, lọc nhiễu) không những không giúp ích mà còn gây hại nghiêm trọng cho hệ thống.
2. **Định hướng nghiên cứu**: Thay vì tập trung vào tiền xử lý ảnh ở mức pixel (Image-level processing), luận văn sẽ tập trung tối ưu ở mức hậu xử lý cú pháp (Post-processing & Graph-based Validation) để sửa lỗi đầu ra LaTeX của mô hình, đảm bảo tính nguyên bản của đặc trưng hình ảnh đầu vào.
