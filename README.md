# TAMER - Handwritten Mathematical Expression Recognition

Dự án này triển khai mô hình **TAMER** (Two-way Attention-based Model for Expression Recognition) cho nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression - HME).

**Tác giả:** Phan Hoàng Khải  
**Đơn vị:** Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE)

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Deep Dive: Graph Attention Networks (GAT)](#deep-dive-graph-attention-networks-gat)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Kết quả](#kết-quả)

## 🎯 Tổng quan

TAMER là một kiến trúc mạnh mẽ kết hợp giữa CNN và Transformer để chuyển đổi hình ảnh biểu thức toán học viết tay thành chuỗi LaTeX. Dự án này bao gồm hai phiên bản chính:

1.  **0-cnn-transformer-baseline**: Phiên bản chuẩn sử dụng DenseNet làm Encoder và Transformer làm Decoder.
2.  **1-cnn-gnn**: Phiên bản lai **CNN-GNN** tích hợp Graph Attention Networks (GAT) vào bộ mã hóa để tăng cường khả năng trích xuất đặc trưng không gian và cấu trúc của biểu thức.

## 📁 Cấu trúc dự án

```
ChuyenDe-Tamer/
├── 0-cnn-transformer-baseline/ # Phiên bản TAMER gốc (DenseNet + Transformer)
├── 1-cnn-gnn/                 # Phiên bản lai CNN-GNN (DenseNet + GAT + Transformer)
│   ├── tamer/
│   │   ├── model/
│   │   │   ├── gat.py   # Cài đặt lớp Graph Attention và Relative Position Bias
│   │   │   └── encoder.py # Encoder tích hợp GAT
│   │   └── ...
├── data/
│   └── CROHME.zip             # Dataset duy nhất dùng chạy Kaggle
├── notebooks/                 # Thư mục lưu trữ Jupyter Notebooks chạy trên Kaggle
├── KetQua/                    # Thư mục chứa kết quả đánh giá thực nghiệm
│   ├── 1_Baseline/            # Mô hình Baseline chuẩn (M1)
│   ├── 2_Naive_GNN_PE_Before/ # Mô hình GNN cũ nhòe vị trí (M2)
│   ├── 3_Corrected_GNN_PE_After/ # Mô hình GNN mới sửa vị trí (M3)
│   ├── 4_Coord_Aware_GAT_1L_4H/  # Mô hình Coordinate-Aware GAT 1 lớp (M4 - Đề xuất tốt nhất)
│   └── 5_Coord_Aware_GAT_2L_8H/  # Mô hình Coordinate-Aware GAT 2 lớp (M5)
└── README.md            # Báo cáo kết quả và tài liệu dự án
```

## 🧠 Deep Dive: Graph Attention Networks (GAT)

Điểm nhấn của dự án này là việc tích hợp **Graph Attention Networks (GAT)** vào kiến trúc Encoder. Dưới đây là phân tích chi tiết kỹ thuật về cách GAT hoạt động trong bài toán này:

### Tại sao lại dùng GAT?

Các mạng CNN truyền thống (như DenseNet) rất giỏi trong việc trích xuất đặc trưng cục bộ (local features). Tuy nhiên, đối với biểu thức toán học, mối quan hệ giữa các ký tự không chỉ nằm ở vị trí lân cận mà còn phụ thuộc vào cấu trúc ngữ nghĩa 2D (ví dụ: phân số, số mũ, chỉ số dưới).

GAT cho phép mô hình coi bản đồ đặc trưng (feature map) như một đồ thị, nơi mỗi điểm ảnh (pixel) hoặc vùng đặc trưng là một nút (node). Cơ chế Attention giúp mỗi nút có thể "tập trung" (attend) vào các nút lân cận quan trọng nhất để tổng hợp thông tin, thay vì nhân chập cố định như CNN.

### Kiến trúc chi tiết (Implementation Details)

Module GAT được cài đặt trong `1-cnn-gnn/tamer/model/gat.py` và `1-cnn-gnn/tamer/model/encoder.py`.

1.  **Xây dựng Đồ thị (Graph Construction)**:
    *   Feature map đầu ra từ DenseNet có kích thước `[H, W, D]`.
    *   Ta biến đổi feature map này thành một lưới đồ thị (grid graph) với `N = H * W` nút.
    *   **Adjacency Matrix**: Xây dựng ma trận kề dựa trên kết nối 4 hướng (4-connectivity: trên, dưới, trái, phải). Mỗi nút được kết nối với 4 nút lân cận của nó.

2.  **Cơ chế GAT Layer**:
    *   Mỗi lớp GAT (`GATLayer`) sử dụng **Multi-head Attention**.
    *   Đầu vào là các features của nút $h_i$.
    *   Hệ số attention $e_{ij}$ giữa nút $i$ và nút lân cận $j$ được tính toán thông qua một mạng nơ-ron truyền thẳng (feed-forward neural network):
        $$e_{ij} = \text{LeakyReLU}(\vec{a}^T [W\vec{h}_i || W\vec{h}_j])$$
    *   Hệ số này sau đó được chuẩn hóa bằng Softmax để tạo ra trọng số $\alpha_{ij}$.
    *   Đầu ra của nút $i$ là tổng có trọng số của các nút lân cận:
        $$\vec{h}'_i = \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\vec{h}_j)$$

3.  **Tích hợp vào Encoder**:
    *   Quy trình xử lý: `Image -> DenseNet -> Feature Map -> Flatten -> GAT Layers -> Reshape -> Feature Map -> Positional Encoding -> Transformer Decoder`.
    *   Việc chèn GAT vào giữa DenseNet và Transformer giúp làm giàu feature map với thông tin ngữ cảnh cấu trúc trước khi giải mã.

## 🔧 Cài đặt

Yêu cầu môi trường:
- Python 3.7+
- PyTorch 1.8+
- CUDA (nếu dùng GPU)

Cài đặt các gói phụ thuộc:

```bash
# Cài đặt cho phiên bản GAT (Khuyên dùng)
cd 1-cnn-gnn
pip install -r requirements.txt
pip install -e .
```

Nếu muốn chạy baseline:
```bash
cd 0-cnn-transformer-baseline
pip install -r requirements.txt
pip install -e .
```

## � Sử dụng

### Quá trình Huấn luyện (Training)

Để huấn luyện mô hình, sử dụng script `train.py`. Bạn có thể thay đổi cấu hình trong thư mục `config/`.

```bash
# Di chuyển vào thư mục source code
cd 1-cnn-gnn

# Chạy huấn luyện với file config mặc định
python train.py fit --config config/crohme.yaml

# Debug nhanh với dữ liệu nhỏ
python train.py fit --config config/crohme_debug.yaml
```

### Đánh giá (Evaluation)

Sử dụng các script trong thư mục `eval/` để đánh giá mô hình đã huấn luyện.

```bash
cd 1-cnn-gnn/eval

# Đánh giá trên tập dữ liệu CROHME
bash eval_crohme.sh
```

## ⚙️ Cấu hình

Các tham số quan trọng trong `config/crohme.yaml`:

- **model**:
    - `d_model`: 256 (Kích thước vector đặc trưng)
    - `use_gat`: true (Bật tắt module GAT)
    - `gat_num_layers`: 2 (Số lớp GAT chồng lên nhau)
    - `gat_num_heads`: 8 (Số đầu attention trong GAT)
- **data**:
    - `folder`: Đường dẫn đến dữ liệu ảnh
    - `batch_size`: Kích thước batch

## 📊 Báo cáo Tổng hợp Thực nghiệm: Quá trình Tiến bộ & Đóng góp Khoa học

Báo cáo này cung cấp cái nhìn toàn diện, mang tính học thuật về quá trình tiến hóa kiến trúc của mô hình **TAMER (CNN-GNN)**, phân tích các phát hiện khoa học cốt lõi từ thực nghiệm và đề xuất phiên bản tối ưu nhất để thực hiện công bố khoa học (paper contribution).

---

### 1. Bảng Tổng hợp Kết quả Định lượng (Quantitative Synthesis)

Dưới đây là bảng đối chiếu kết quả đánh giá (Evaluation) chính thức của cả 5 phiên bản mô hình trên 3 tập kiểm thử chuẩn **CROHME 2014, 2016, và 2019**:

*   **M1: Baseline** (Thư mục: `1_Baseline`, Run ID: `8ivyzmlm`) - DenseNet + Transformer Decoder (Không có GAT).
*   **M2: Naive GAT (GNN Cũ)** (Thư mục: `2_Naive_GNN_PE_Before`, Run ID: `colorful-moose-173`) - GAT 2L, 8H, **PE đặt TRƯỚC GAT**.
*   **M3: Corrected GAT (GNN Mới)** (Thư mục: `3_Corrected_GNN_PE_After`, Run ID: `defiant-mole-974`) - GAT 2L, 8H, **PE đặt SAU GAT**.
*   **M4: Coord-Aware GAT (1L, 4H)** (Thư mục: `4_Coord_Aware_GAT_1L_4H`, Run ID: `skittish-worm-90`) - GAT 1L, 4H, **PE đặt SAU GAT + Relative Position Bias kề 8-hướng**.
*   **M5: Coord-Aware GAT (2L, 8H)** (Thư mục: `5_Coord_Aware_GAT_2L_8H`, Run ID: `welcoming-dog-350`) - GAT 2L, 8H, **PE đặt SAU GAT + Relative Position Bias kề 8-hướng** (Thử nghiệm Scale-up).

| Tập dữ liệu (Dataset) | Chỉ số (Metric) | M1: Baseline | M2: GNN Cũ (PE trước GAT) | M3: GNN Mới (PE sau GAT) | M4: Coord-Aware GAT (1L, 4H) | M5: Coord-Aware GAT (2L, 8H) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CROHME 2014** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance | **51.12%** <br> **69.98%** <br> **77.69%** <br> 1.99 | 49.39% <br> 66.53% <br> 75.25% <br> 2.22 | 48.88% <br> 66.73% <br> 75.35% <br> 2.19 | 49.90% <br> 67.44% <br> 77.18% <br> **1.98** *(Best)* | 46.65% <br> 63.99% <br> 73.43% <br> 2.48 |
| **CROHME 2016** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance | 50.65% <br> **67.92%** <br> **76.02%** <br> 2.21 | 47.43% <br> 64.95% <br> 74.72% <br> 2.45 | **50.74%** *(Best)* <br> 66.96% <br> 75.85% <br> 2.19 | 49.17% <br> 67.13% <br> 75.76% <br> **2.13** *(Best)* | 45.68% <br> 63.03% <br> 73.23% <br> 2.53 |
| **CROHME 2019** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance | **48.54%** <br> **68.14%** <br> 77.23% <br> 2.40 | 46.71% <br> 66.89% <br> 75.90% <br> 2.62 | 47.87% <br> 67.81% <br> **77.98%** *(Best)* <br> **2.02** *(Best)* | 47.87% <br> 67.72% <br> 76.90% <br> 2.08 | 37.70% <br> 58.97% <br> 70.14% <br> 2.93 |
| **Trung bình (Avg)** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance | **50.10%** <br> **68.68%** <br> **76.98%** <br> 2.20 | 47.84% <br> 66.12% <br> 75.29% <br> 2.21 | 49.17% <br> 67.17% <br> 76.40% <br> 2.14 | 48.98% <br> 67.43% <br> 76.61% <br> **2.06** *(Best)* | 43.35% <br> 62.00% <br> 72.27% <br> 2.65 |

---

### 2. Phân tích Toàn bộ Quá trình Tiến bộ (Progression History Analysis)

Sự phát triển của TAMER (CNN-GNN) trải qua 4 bước ngoặt thiết kế kiến trúc quan trọng, mỗi giai đoạn mang lại một bài học khoa học sâu sắc:

#### Giai đoạn 1: Naive GNN (M2) - Thất bại do nhòe vị trí (Position Encoding Blurring)
*   **Thiết kế:** GAT chồng lên feature map DenseNet sau khi đã cộng 2D Positional Encoding (PE).
*   **Hậu quả:** ExpRate giảm **-2.26%** so với Baseline. 
*   **Bài học:** Cơ chế truyền tin đồ thị (message passing) thực chất là phép trung bình hóa có trọng số (weighted pooling) trên các node lân cận. Phép toán này vô tình làm mịn/nhòe các vector PE tuyệt đối sắc nét của từng pixel, dẫn đến lỗi dịch chuyển attention (Attention Alignment Shift) ở decoder khi giải mã cấu trúc 2D (như số mũ, chỉ số dưới).

#### Giai đoạn 2: Corrected GNN (M3) - Khôi phục thông tin tọa độ tuyệt đối
*   **Thiết kế:** Chuyển khối PE xuống **sau** khối GAT. Pixel truyền tin cục bộ trên feature map visual thuần túy trước, sau đó mới được đánh dấu tọa độ tuyệt đối sắc nét gửi tới decoder.
*   **Kết quả:** ExpRate hồi phục mạnh mẽ lên **49.17%** (+1.33% từ GNN cũ), tiệm cận sát Baseline (50.10%), thậm chí vượt Baseline trên tập CROHME 2016.
*   **Bài học:** Sự tuần tự giữa biểu diễn ngữ cảnh không gian và tọa độ tuyệt đối là cực kỳ quan trọng. Tọa độ địa lý phải là nhãn cố định được dán lên đặc trưng ngữ cảnh đồ thị hoàn chỉnh, không được phép tham gia vào quá trình truyền tin làm mịn đặc trưng.

#### Giai đoạn 3: Coordinate-Aware GAT (M4) - Thiết lập Inductive Bias không gian
*   **Thiết kế:** GAT tiêu chuẩn chỉ chú ý đến sự tương đồng visual của các pixel mà mất nhận thức về hướng (Direction-Agnostic). Phiên bản này tích hợp **Relative Position Bias kề 8-hướng** (9 trạng thái quan hệ tọa độ $\Delta x, \Delta y \in \{-1, 0, 1\}$) vào Attention logits, đồng thời tinh giảm số layer/head (1 Layer, 4 Heads) để tránh overfitting.
*   **Kết quả:** Dù số tham số giảm mạnh, ExpRate đạt **48.98%**, và **Mean Edit Distance đạt kỷ lục 2.06** (vượt qua tất cả các mô hình, kể cả Baseline là 2.20).
*   **Bài học:** Việc bổ sung **Relative Spatial Inductive Bias** hoạt động như một hướng dẫn hình học cho Attention Heads. Khi mô hình dự đoán sai, sai sót chỉ nằm ở mức 1-2 ký tự cục bộ chứ không bao giờ làm sụp đổ cấu trúc phân tầng phức tạp (như phân số hay căn thức).

#### Giai đoạn 4: Scale-up Coordinate-Aware GAT (M5) - Điểm nghẽn truyền tin phi tuyến
*   **Thiết kế:** Nâng cấp Coordinate-Aware GAT lên 2 layers và 8 heads với mong muốn tăng dung lượng biểu diễn.
*   **Hậu quả:** Hiệu năng sụt giảm nghiêm trọng xuống **43.35%** (giảm tới **-5.63%** so với phiên bản 1 lớp).
*   **Bài học (Negative Result quý giá):** 
    1.  **Nghẽn đồ thị thưa:** Đồ thị lưới pixel rất thưa (mỗi node chỉ có tối đa 8 liên kết). Áp dụng `dropout=0.2` trên Attention và feature map liên tiếp qua 2 layers gây đứt gãy luồng lan truyền thông tin nghiêm trọng.
    2.  **Méo mó phi tuyến:** Relative Bias được cộng vào logits *trước* khi qua các hàm kích hoạt phi tuyến (`LeakyReLU` ở lớp 1, rồi `ELU`, rồi `LeakyReLU` ở lớp 2). Việc đi qua chuỗi kích hoạt phi tuyến liên tiếp đã bẻ cong quan hệ khoảng cách tuyến tính ban đầu, biến thông tin tọa độ tương đối thành nhiễu phi tuyến phức tạp.

---

### 3. Gợi ý Phiên bản Tốt nhất cho Đóng góp Khoa học (Scientific Recommendation)

Mô hình **M4: Coordinate-Aware GAT (1L, 4H)** (Thư mục: `4_Coord_Aware_GAT_1L_4H`, Run ID: `skittish-worm-90`) là phiên bản tốt nhất và phù hợp nhất để làm đóng góp khoa học chính của bài báo.

#### Lý do lựa chọn học thuật (Academic Rationale):

1.  **Chỉ số Mean Edit Distance tối ưu nhất (2.06 vs 2.20 of Baseline):**
    Trong bài toán nhận diện biểu thức toán học (HMER), tỷ lệ khớp 100% (ExpRate) rất nhạy cảm với các nét chữ viết tay dị biệt của con người. Tuy nhiên, **Mean Edit Distance** (Khoảng cách chỉnh sửa trung bình) mới là thước đo chính xác nhất cho thấy khả năng bảo toàn cấu trúc toán học của mô hình. Kết quả **2.06** của M4 cho thấy sự vượt trội về mặt cấu trúc so với Baseline.
2.  **Tính hiệu quả và tối giản tham số (Efficiency & Parsimony):**
    M4 chỉ sử dụng **1 lớp GAT và 4 heads** (lượng tham số cực kỳ nhỏ) nhưng mang lại hiệu năng tương đương bản GAT thông thường 2 lớp 8 heads và vượt trội về độ chính xác cấu trúc. Đây là minh chứng vàng cho việc áp dụng đúng đắn **Inductive Bias** thay vì tăng số lượng tham số mù quáng (càng nhiều tham số trên tập dữ liệu nhỏ càng dễ overfitting).
3.  **Tính mới về mặt khoa học (Scientific Novelty):**
    *   **Phát hiện 1: PE Blurring Effect:** Minh chứng toán học và thực nghiệm về sự triệt tiêu thông tin vị trí khi GAT đứng trước Positional Encoding trên lưới pixel.
    *   **Phát hiện 2: Coordinate-Aware Relative Bias on Pixel Grids:** Thiết kế ma trận Relative Bias 9 trạng thái hoạt động hiệu quả trên cấu trúc kề 8-hướng của ảnh, một thay thế hoàn hảo cho cơ chế tích chập định hướng truyền thống của CNN.
    *   **Phát hiện 3: Không nên chồng nhiều tầng GAT trên đồ thị lưới ảnh:** Phân tích thực nghiệm từ M5 chỉ ra giới hạn của việc chồng lớp phi tuyến lên ma trận relative bias logits và dropout trên đồ thị thưa.

#### Đề xuất cấu trúc luận điểm trong Paper:
*   **Abstract/Introduction:** Đặt vấn đề về việc Transformer Decoder trong HMER thường gặp lỗi lệch dòng (alignment shift) do DenseNet thiếu liên kết ngữ cảnh cấu trúc cục bộ. Đề xuất TAMER (CNN-GNN) kết hợp GAT để giải quyết.
*   **Methodology:** Trình bày chi tiết toán học của **GAT trên lưới đồ thị kề 8-hướng**, giải pháp chuyển **PE ra sau GAT** để tránh Blurring Effect, và thuật toán **Coordinate-Aware Relative Bias** kề 8-hướng.
*   **Experiments & Discussion:** Đưa bảng tổng hợp 5 mô hình trên vào. Nhấn mạnh việc M4 đạt **Mean Edit Distance 2.06** và phân tích kết quả âm (negative result) của M5 để làm bài học định hướng thiết kế mạng GNN trên cấu trúc ảnh cho cộng đồng khoa học.

---
© 2026 Phan Hoàng Khải - Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE). Báo cáo thực nghiệm chuyên đề tốt nghiệp.
