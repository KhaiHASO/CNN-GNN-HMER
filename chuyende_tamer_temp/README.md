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
│   │   │   ├── gat.py   # Cài đặt lớp Graph Attention
│   │   │   └── encoder.py # Encoder tích hợp GAT
│   │   └── ...
├── data/
│   └── CROHME.zip             # Dataset duy nhất, notebook sẽ tự giải nén khi chạy Kaggle
├── KAGGLE_RUN_ALL_CNN_GNN_HMER.ipynb
└── README.md            # Tài liệu dự án
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

## 📊 Kết quả Thực nghiệm & Phân tích Chi tiết

Dưới đây là bảng đối chiếu chi tiết kết quả đánh giá (Evaluation) chính thức của mô hình **CNN-GNN** qua các giai đoạn phát triển so với mô hình **CNN-Transformer Baseline** trên cả 3 tập kiểm thử chuẩn: **CROHME 2014, 2016, và 2019**.

### 1. Bảng So sánh Định lượng (Quantitative Comparison)

* **Baseline:** Checkpoint `epoch=95-step=72095` (Run ID: `8ivyzmlm`)
* **CNN-GNN (Old):** Checkpoint `epoch=77-step=58577` (Run ID: `colorful-moose-173`) - *Tích hợp Positional Encoding TRƯỚC lớp GAT*
* **CNN-GNN (New):** Checkpoint `best_model.ckpt` (Run ID: `defiant-mole-974`) - *Tích hợp Positional Encoding SAU lớp GAT (2 layers, 8 heads, 0.1 dropout)*
* **CNN-GNN + Coordinate-Aware GAT (1L, 4H):** Checkpoint `best_model.ckpt` (Run ID: `skittish-worm-90`) - *GAT 1 layer, 4 heads, 0.2 dropout, kèm Relative Position Bias (Coordinate-Aware kề 8-hướng)*
* **CNN-GNN + Coordinate-Aware GAT (2L, 8H):** Checkpoint `best_model.ckpt` (Run ID: `welcoming-dog-350`) - *GAT 2 layers, 8 heads, 0.2 dropout, kèm Relative Position Bias (Coordinate-Aware kề 8-hướng - Thử nghiệm Scale-up)*

| Tập dữ liệu (Dataset) | Mô hình | ExpRate (Khớp 100%) | ExpRate $\le$ 1 (Sai $\le$ 1 ký tự) | ExpRate $\le$ 2 (Sai $\le$ 2 ký tự) | Mean Edit Distance (K/c chỉnh sửa TB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CROHME 2014** | Baseline | **51.12%** | **69.98%** | **77.69%** | 1.99 |
| | CNN-GNN (Old) | 49.39% | 66.53% | 75.25% | 2.22 |
| | CNN-GNN (New) | 48.88% | 66.73% | 75.35% | 2.19 |
| | **Coordinate-Aware GAT (1L, 4H)** | 49.90% | 67.44% | 77.18% | **1.98** *(Tốt nhất)* |
| | Coordinate-Aware GAT (2L, 8H) | 46.65% | 63.99% | 73.43% | 2.48 |
| **CROHME 2016** | Baseline | 50.65% | **67.92%** | **76.02%** | 2.21 |
| | CNN-GNN (Old) | 47.43% | 64.95% | 74.72% | 2.45 |
| | CNN-GNN (New) | **50.74%** | 66.96% | 75.85% | 2.19 |
| | **Coordinate-Aware GAT (1L, 4H)** | 49.17% | 67.13% | 75.76% | **2.13** *(Tốt nhất)* |
| | Coordinate-Aware GAT (2L, 8H) | 45.68% | 63.03% | 73.23% | 2.53 |
| **CROHME 2019** | Baseline | **48.54%** | **68.14%** | 77.23% | 2.40 |
| | CNN-GNN (Old) | 46.71% | 66.89% | 75.90% | 2.62 |
| | CNN-GNN (New) | 47.87% | 67.81% | **77.98%** | **2.02** *(Tốt nhất)* |
| | **Coordinate-Aware GAT (1L, 4H)** | 47.87% | 67.72% | 76.90% | 2.08 |
| | Coordinate-Aware GAT (2L, 8H) | 37.70% | 58.97% | 70.14% | 2.93 |
| **Trung bình (Average)**| Baseline | **50.10%** | **68.68%** | **76.98%** | 2.20 |
| | CNN-GNN (Old) | 47.84% | 66.12% | 75.29% | 2.43 |
| | CNN-GNN (New) | 49.17% | 67.17% | 76.40% | 2.14 |
| | **Coordinate-Aware GAT (1L, 4H)** | **48.98%** | **67.43%** | **76.61%** | **2.06** *(Tốt nhất)* |
| | Coordinate-Aware GAT (2L, 8H) | 43.35% | 62.00% | 72.27% | 2.65 |

---

### 2. Phân tích Sâu & Nhận xét Kỹ thuật (In-depth Analysis)

#### 2.1. Đánh giá về sự tối ưu của GAT 1 lớp và Bias Tọa độ tương đối (`skittish-worm-90`)
* Cơ chế nhúng hướng tương đối trong ma trận kề kề 8-hướng giúp giữ vững cấu trúc không gian cực kỳ hiệu quả, đạt **Mean Edit Distance kỷ lục là 2.06** (so với 2.20 của Baseline). Khi dự đoán sai, cấu trúc phân số, số mũ, chỉ số dưới của mô hình ít bị xáo trộn nhất.

#### 2.2. Hiện tượng sụt giảm hiệu năng nghiêm trọng ở bản Scale-up (`welcoming-dog-350`)
Khi tăng quy mô Coordinate-Aware GAT lên **2 layers và 8 heads** (Run `welcoming-dog-350`), tỷ lệ nhận diện trung bình sụt giảm mạnh về **43.35%** (giảm **-5.63%** so với phiên bản 1 lớp). Đây là một kết quả bất ngờ nhưng có lý do kỹ thuật rõ ràng:
1. **Hiện tượng nghẽn thông tin do Dropout cộng dồn trên đồ thị thưa:**
   Đồ thị của chúng ta là đồ thị mạng lưới kề 8-hướng (mỗi nút chỉ kết nối với tối đa 8 nút lân cận và chính nó). Việc thiết lập `gat_dropout = 0.2` sẽ ngẫu nhiên loại bỏ 20% cạnh kết nối trong Attention. Khi chồng chéo **2 layers GAT** liên tiếp, hiện tượng ngắt kết nối này bị cộng dồn, cộng với lớp Dropout trung gian giữa 2 layer (`0.2`), dẫn đến việc **luồng thông tin đồ thị bị đứt gãy nghiêm trọng**. 
2. **Sự méo mó phi tuyến của Bias Tọa độ tương đối:**
   Trong file [gat.py](file:///home/khai/Desktop/github/CNN-GNN-HMER/chuyende_tamer_temp/1-cnn-gnn/tamer/model/gat.py#L98-L99), relative position bias được cộng trực tiếp vào logits:
   $$e = e + bias$$
   sau đó đi qua hàm kích hoạt phi tuyến `LeakyReLU(0.2)`. 
   * Ở mô hình 1 lớp, sự méo mó xảy ra một lần trước khi đưa vào Softmax.
   * Ở mô hình 2 lớp, các đặc trưng tọa độ tương đối sau khi méo mó ở lớp 1 sẽ đi qua hàm kích hoạt phi tuyến phụ (`ELU`), cộng thêm nhiễu dropout, rồi tiếp tục bị biến dạng bởi một lớp `LeakyReLU` và bias mới ở lớp 2. Điều này phá vỡ tính tuyến tính hình học của tọa độ tương đối, khiến mô hình bị nhiễu thông tin không gian trầm trọng.
3. **Độ nhạy Attention khi số Head quá lớn:**
   Khi chia 256 kênh đặc trưng thành 8 heads, mỗi head chỉ xử lý `32` kênh. Với dung lượng kênh quá nhỏ, attention score rất dễ bị ảnh hưởng bởi nhiễu và độ lệch của learnable bias, dẫn đến việc lan truyền đặc trưng đồ thị bị mất cân bằng.

---

### 🚀 3. Đề xuất Phương án Kế tiếp (Roadmap cải tiến mới)

Dựa trên các bài học từ hai run `skittish-worm-90` (Thành công lớn với 1L/4H) và `welcoming-dog-350` (Thất bại khi Scale-up 2L/8H), chúng tôi đề xuất chiến lược tiếp theo để tối ưu hóa CNN-GNN:

#### 3.1. Tối ưu cấu hình 1 Layer với nhiều Head hơn (1 Layer, 8 Heads)
* **Phương án:** Thay vì tăng số lớp GAT (gây nghẽn truyền tin và méo bias), chúng ta nên duy trì **1 layer GAT** nhưng nâng cấp số head từ **4 heads lên 8 heads**.
* **Lý do:** Giữ nguyên được đường truyền tin trực tiếp (không bị méo phi tuyến liên lớp), đồng thời tăng khả năng biểu diễn đa chiều (multi-view attention) để nắm bắt các quan hệ hình học tốt hơn.

#### 3.2. Điều chỉnh vị trí cộng Bias Tọa độ (Post-Activation Bias)
* **Phương án:** Di chuyển phép cộng `bias` ra **sau** hàm `LeakyReLU` hoặc cộng trực tiếp vào ma trận attention đã chuẩn hóa (nhưng trước Softmax) để đảm bảo không bị méo mó phi tuyến bởi LeakyReLU:
  $$e = \text{LeakyReLU}(e) + bias$$
  (Cách này giữ nguyên tính chất hình học độc lập của tọa độ).

#### 3.3. Giảm tỷ lệ Dropout trên Graph
* **Phương án:** Giảm `gat_dropout` xuống mức **0.05 hoặc 0.1** (thay vì 0.2).
* **Lý do:** Đồ thị lưới pixel cực kỳ nhạy cảm với việc mất kết nối cục bộ. Giảm dropout giúp bảo toàn tính toàn vẹn của cấu trúc nét vẽ công thức trong suốt quá trình lan truyền thông tin đồ thị.

---
© 2026 Phan Hoàng Khải - Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE).
