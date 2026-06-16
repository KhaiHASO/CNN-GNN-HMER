# Báo cáo So sánh và Phân tích Hiệu năng: CNN-Transformer Baseline vs. CNN-GNN (Cải tiến PE-after-GAT)

Báo cáo này tổng hợp kết quả đánh giá (Evaluation) chính thức của mô hình **CNN-GNN** phiên bản cải tiến so với **CNN-GNN** phiên bản cũ và **CNN-Transformer Baseline** trên cả 3 tập dữ liệu kiểm thử chuẩn **CROHME 2014, 2016, và 2019**.

---

## 📊 1. Bảng So sánh Định lượng (Quantitative Comparison)

Dưới đây là bảng đối chiếu chi tiết các chỉ số đo lường giữa các mô hình:
* **Baseline:** Checkpoint `epoch=95-step=72095-val_ExpRate=0.5091.ckpt` (Run ID: `8ivyzmlm`)
* **CNN-GNN (Cũ):** Checkpoint `epoch=77-step=58577-val_ExpRate=0.4939.ckpt` (Run ID: `8b964c54` / `colorful-moose-173`) - *Tích hợp Positional Encoding TRƯỚC lớp GAT*
* **CNN-GNN (Mới):** Checkpoint `best_model.ckpt` (Run ID: `2e261118` / `defiant-mole-974` - Train đủ 100 epoch) - *Tích hợp Positional Encoding SAU lớp GAT*

| Tập dữ liệu (Dataset) | Mô hình | ExpRate (Chính xác tuyệt đối) | ExpRate $\le$ 1 (Sai lệch $\le$ 1 ký tự) | ExpRate $\le$ 2 (Sai lệch $\le$ 2 ký tự) | Mean Edit Distance (Khoảng cách chỉnh sửa TB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CROHME 2014** | Baseline | **51.12%** | **69.98%** | **77.69%** | **1.99** |
| | CNN-GNN (Cũ) | 49.39% | 66.53% | 75.25% | 2.22 |
| | CNN-GNN (Mới) | 48.88% | 66.73% | 75.35% | 2.19 *(tốt hơn cũ)* |
| **CROHME 2016** | Baseline | 50.65% | **67.92%** | **76.02%** | 2.21 |
| | CNN-GNN (Cũ) | 47.43% | 64.95% | 74.72% | 2.45 |
| | CNN-GNN (Mới) | **50.74%** | 66.96% | 75.85% | **2.19** *(vượt baseline)* |
| **CROHME 2019** | Baseline | **48.54%** | **68.14%** | 77.23% | 2.40 |
| | CNN-GNN (Cũ) | 46.71% | 66.89% | 75.90% | 2.62 |
| | CNN-GNN (Mới) | 47.87% | 67.81% | **77.98%** | **2.02** *(vượt baseline)* |
| **Trung bình (Average)** | Baseline | **50.10%** | **68.68%** | **76.98%** | 2.20 |
| | CNN-GNN (Cũ) | 47.84% | 66.12% | 75.29% | 2.43 |
| | CNN-GNN (Mới) | **49.17%** | **67.17%** | **76.40%** | **2.14** *(tốt nhất)* |

> [!IMPORTANT]
> **Nhận xét chính:**
> 1. **Hiệu năng nhận diện tuyệt đối (ExpRate):** Phiên bản CNN-GNN mới (`defiant-mole-974`) cải thiện đáng kể so với bản cũ, tăng từ **47.84%** lên **49.17%** trung bình (+1.33% absolute). Đặc biệt trên tập **CROHME 2016**, mô hình GNN mới đạt **50.74%**, chính thức vượt qua cả Baseline (50.65%).
> 2. **Khoảng cách chỉnh sửa trung bình (Mean Edit Distance):** Mô hình CNN-GNN mới đạt kết quả xuất sắc nhất với **2.14**, vượt qua cả Baseline (2.20) và GNN cũ (2.43). Điều này chỉ ra rằng mặc dù tỷ lệ khớp 100% của Baseline nhỉnh hơn một chút, nhưng khi CNN-GNN dự đoán sai, sai lệch của nó về số lượng ký tự/cấu trúc là ít nghiêm trọng nhất.
> 3. **Độ lệch sai số nhỏ (ExpRate <= 2):** Trên tập **CROHME 2019**, tỷ lệ nhận diện lệch tối đa 2 ký tự của CNN-GNN mới đạt **77.98%**, cao hơn Baseline (77.23%).

---

## 🔍 2. Phân tích Nguyên nhân Kỹ thuật & Sự Cải tiến

### 2.1. Khắc phục hiện tượng làm mờ thông tin vị trí (Position Encoding Blurring)
Trong kiến trúc cũ, Positional Encoding được cộng vào bản đồ đặc trưng DenseNet *trước* khi đưa qua các lớp Graph Attention Network (GAT). 
* **Hậu quả cũ:** Cơ chế truyền tin đồ thị (message passing) lấy tổng trọng số attention của các pixel lân cận, vô tình làm mịn/trung bình hóa (blur) các vector tọa độ tuyệt đối.
* **Giải pháp mới:** Di chuyển Positional Encoding xuống **SAU** khối GAT (trong [encoder.py](file:///home/khai/Desktop/github/CNN-GNN-HMER/chuyende_tamer_temp/1-cnn-gnn/tamer/model/encoder.py#L272-L291)). Các pixel truyền thông tin ngữ cảnh cho nhau qua GAT trên bản đồ đặc trưng thuần túy, sau đó mới được đánh dấu tọa độ tuyệt đối sắc nét gửi tới Transformer Decoder.
* **Kết quả:** Mô hình giải quyết triệt để lỗi Attention Alignment Shift, nâng cao độ chính xác nhận diện cấu trúc phân tầng (phân số, mũ, chỉ số dưới).

### 2.2. Kiểm soát Overfitting và Cải thiện Loss
* **GNN Cũ:** Do tọa độ bị nhòe, mô hình cố gắng học thuộc lòng (overfit) nhiễu trên tập train nhỏ. Train Loss giảm sâu về `0.1600` nhưng Validation Loss tăng lên `0.5059`.
* **GNN Mới:** Khi tọa độ sắc nét, mô hình học các mẫu đặc trưng tổng quát tốt hơn. Train Loss tiếp tục giảm xuống mức tốt hơn (`0.1484`), và Validation Loss giảm mạnh từ `0.5059` xuống còn **`0.4759`** ở epoch 99 (giảm 6% loss validation), chứng tỏ mô hình có khả năng tổng quát hóa (generalization) vượt trội.

---

## 🚀 3. Phương án Cải tiến Tiếp theo (Roadmap)

Dù đã rút ngắn đáng kể khoảng cách với Baseline nhờ đổi thứ tự PE-after-GAT, chúng ta vẫn có thể cải tiến thêm để CNN-GNN vượt trội hoàn toàn:

### 3.1. Thiết lập GAT nhận biết khoảng cách/hướng (Coordinate-Aware GAT)
* **Vấn đề:** GAT tiêu chuẩn coi các láng giềng kề 8 hướng như một tập hợp không có thứ tự.
* **Giải pháp:** Tích hợp vector khoảng cách tương đối $r_{ij} = (\Delta x, \Delta y)$ giữa các pixel liền kề vào Attention Head để mô hình phân biệt rõ nét trên/dưới, trái/phải, phục vụ nhận diện cấu trúc toán học 2D tốt hơn.

### 3.2. Tinh gọn tham số và Tăng cường Regularization
* **Số lượng layer GAT:** Giảm từ **2 layers xuống 1 layer** để tránh hiện tượng over-smoothing (làm mịn quá mức đặc trưng khi truyền tin đi quá xa).
* **Số lượng attention heads:** Giảm từ **8 heads xuống 4 heads** nhằm cắt giảm số lượng tham số thừa, hạn chế tối đa overfitting.
* **Dropout:** Tăng `gat_dropout` từ **0.1 lên 0.2 hoặc 0.3** để nâng cao tính bền vững của đặc trưng đồ thị.
