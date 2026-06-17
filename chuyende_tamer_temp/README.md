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
* **CNN-GNN + Coordinate-Aware GAT:** Checkpoint `best_model.ckpt` (Run ID: `skittish-worm-90`) - *GAT 1 layer, 4 heads, 0.2 dropout, kèm Relative Position Bias (Coordinate-Aware kề 8-hướng)*

| Tập dữ liệu (Dataset) | Mô hình | ExpRate (Khớp 100%) | ExpRate $\le$ 1 (Sai $\le$ 1 ký tự) | ExpRate $\le$ 2 (Sai $\le$ 2 ký tự) | Mean Edit Distance (K/c chỉnh sửa TB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CROHME 2014** | Baseline | **51.12%** | **69.98%** | **77.69%** | 1.99 |
| | CNN-GNN (Old) | 49.39% | 66.53% | 75.25% | 2.22 |
| | CNN-GNN (New) | 48.88% | 66.73% | 75.35% | 2.19 |
| | **Coordinate-Aware GAT** | 49.90% | 67.44% | 77.18% | **1.98** *(Tốt nhất)* |
| **CROHME 2016** | Baseline | 50.65% | **67.92%** | **76.02%** | 2.21 |
| | CNN-GNN (Old) | 47.43% | 64.95% | 74.72% | 2.45 |
| | CNN-GNN (New) | **50.74%** | 66.96% | 75.85% | 2.19 |
| | **Coordinate-Aware GAT** | 49.17% | 67.13% | 75.76% | **2.13** *(Tốt nhất)* |
| **CROHME 2019** | Baseline | **48.54%** | **68.14%** | 77.23% | 2.40 |
| | CNN-GNN (Old) | 46.71% | 66.89% | 75.90% | 2.62 |
| | CNN-GNN (New) | 47.87% | 67.81% | **77.98%** | **2.02** *(Tốt nhất)* |
| | **Coordinate-Aware GAT** | 47.87% | 67.72% | 76.90% | 2.08 |
| **Trung bình (Average)**| Baseline | **50.10%** | **68.68%** | **76.98%** | 2.20 |
| | CNN-GNN (Old) | 47.84% | 66.12% | 75.29% | 2.43 |
| | CNN-GNN (New) | 49.17% | 67.17% | 76.40% | 2.14 |
| | **Coordinate-Aware GAT** | 48.98% | 67.43% | 76.61% | **2.06** *(Tốt nhất)* |

---

### 2. Phân tích Sâu & Nhận xét Kỹ thuật (In-depth Analysis)

> [!IMPORTANT]
> **Điểm sáng lớn nhất: Khoảng cách chỉnh sửa (Mean Edit Distance) đạt kỷ lục mới**
> Mô hình sử dụng **Coordinate-Aware GAT (`skittish-worm-90`)** đạt khoảng cách chỉnh sửa trung bình thấp nhất: **2.06** (giảm 6.3% so với Baseline và 3.7% so với GNN mới phiên bản thường).
> Điều này minh chứng rằng cơ chế nhúng hướng tương đối trong ma trận kề (8 hướng kề) giúp bộ giải mã Transformer giữ vững cấu trúc 2D của công thức toán (không bị lệch dòng, đảo lộn số mũ/chỉ số dưới hoặc phân số) tốt hơn hẳn. Khi dự đoán sai, mô hình chỉ bị sai lệch rất nhỏ về ký tự chứ không phá vỡ cấu trúc biểu thức.

1. **Trade-off giữa Capacity (Dung lượng mô hình) và Mối quan hệ Không gian:**
   * **CNN-GNN (New - `defiant-mole-974`)** dùng 2 lớp GAT và 8 heads đạt ExpRate trung bình **49.17%** (nhờ dung lượng tham số lớn hơn).
   * **Coordinate-Aware GAT (`skittish-worm-90`)** chỉ dùng 1 lớp GAT và 4 heads đạt ExpRate trung bình **48.98%** (thấp hơn 0.19% so với bản New, nhưng Mean Edit Distance lại tối ưu hơn nhiều: **2.06** so với **2.14**).
   * **Kết luận:** Việc tinh giản mô hình giúp giảm overfitting rõ rệt (Validation Loss ổn định ở mức `0.512`), đồng thời việc nhúng Relative Position Bias bù đắp cực tốt cho việc thiếu hụt số lớp GAT bằng cách cung cấp thông tin không gian trực quan hơn.

2. **Dung sai lỗi (ExpRate $\le$ 1 và $\le$ 2):**
   * Tỷ lệ sai lệch dưới 1 ký tự (ExpRate $\le$ 1) của mô hình Coordinate-Aware đạt **67.43%**, cải thiện so với bản New thường (67.17%). Điều này chứng tỏ mô hình có xu hướng đoán "gần đúng hoàn toàn" cao hơn.

---

### 🚀 3. Đề xuất Phương án Kế tiếp (Roadmap)

Để vượt qua baseline **50.10% ExpRate** trung bình một cách toàn diện (cả về tỷ lệ khớp 100% lẫn Edit Distance), chúng tôi đề xuất các phương án tối ưu hóa như sau:

#### 3.1. Phục hồi dung lượng GAT (Scale-up Coordinate-Aware GAT)
* **Phương án:** Giữ nguyên cơ chế **Coordinate-Aware Relative Position Bias** nhưng nâng cấp GAT từ **1 layer / 4 heads** lên **2 layers / 8 heads** (hoặc tăng kích thước `gat_hidden_dim` / `d_model`).
* **Mục tiêu:** Tận dụng tối đa khả năng biểu diễn cấu trúc không gian của Coordinate-Aware GAT đồng thời khôi phục lại sức mạnh học biểu diễn sâu (deep representation capacity) của mô hình 2 lớp, giúp đẩy ExpRate trung bình vượt qua mốc 50.10%.

#### 3.2. Huấn luyện chính thức (Kaggle Production Run)
* **Phương án:** Chạy huấn luyện chính thức với cấu hình lai tối ưu này trên Kaggle trong **100-120 epochs** kèm theo **Learning Rate Cosine Annealing scheduler**. Chạy thử nghiệm RTX 3070 hiện tại đã cho thấy xu hướng hội tụ rất tốt và ổn định.

#### 3.3. Tăng cường dữ liệu hình ảnh (Data Augmentation)
* **Phương án:** Bổ sung các kỹ thuật tăng cường như **Elastic Deformation** (biến dạng đàn hồi để mô phỏng nét viết tay tự nhiên), **Random Scaling**, và **Perspective Transform** để mô hình GNN học được các quan hệ không gian bền vững hơn trước các biến thể viết tay khác nhau.

---
© 2025 Phan Hoàng Khải - Đại học Sư phạm Kỹ thuật TPHCM (HCMUTE).
