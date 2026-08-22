# TAMER - Handwritten Mathematical Expression Recognition

Dự án này triển khai mô hình **TAMER** (Two-way Attention-based Model for Expression Recognition) cho nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression - HMER) theo hướng nghiên cứu mô hình lai **CNN-GNN**.

* **Đề tài:** Nghiên cứu mô hình lai CNN-GNN trong nhận dạng biểu thức toán học viết tay
* **Tác giả:** Phan Hoàng Khải (MSHV: 2531308)  
* **Người hướng dẫn:** TS. Bùi Mạnh Quân  
* **Đơn vị:** Trường Đại học Công nghệ Kỹ thuật TP.HCM

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Deep Dive: Graph Attention Networks (GAT)](#deep-dive-graph-attention-networks-gat)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cấu hình](#cấu-hình)
- [Báo cáo Tổng hợp Thực nghiệm](#báo-cáo-tổng-hợp-thực-nghiệm-quá-trình-tiến-bộ--đóng-góp-khoa-học)

## 🎯 Tổng quan

TAMER là một kiến trúc end-to-end kết hợp giữa CNN và Transformer để chuyển đổi hình ảnh biểu thức toán học viết tay thành chuỗi LaTeX. Dự án này bao gồm hai nhánh nghiên cứu chính:

1. **0-cnn-transformer-baseline**: Phiên bản chuẩn sử dụng DenseNet làm Encoder và Transformer làm Decoder (không dùng GAT).
2. **1-cnn-gnn**: Phiên bản lai **CNN-GNN** tích hợp Graph Attention Networks (GAT) trên lưới đặc trưng ảnh (feature-grid graph) để tăng cường mô hình hóa quan hệ không gian 2D trước khi giải mã.

## 📁 Cấu trúc dự án

```text
CNN-GNN-HMER/
├── 0-cnn-transformer-baseline/ # Phiên bản baseline chuẩn (DenseNet + Transformer)
├── 1-cnn-gnn/                 # Phiên bản lai CNN-GNN (DenseNet + GAT + Transformer)
│   ├── tamer/
│   │   ├── model/
│   │   │   ├── gat.py         # Cài đặt Graph Attention và Relative Position Bias
│   │   │   └── encoder.py     # Encoder tích hợp GAT trên feature map
│   │   └── ...
├── data/
│   └── CROHME.zip             # Dataset CROHME dùng trong thực nghiệm
├── notebooks/                 # Thư mục lưu trữ Jupyter Notebooks chạy trên Kaggle (2x Tesla T4)
├── KetQua/                    # Thư mục chứa kết quả đánh giá thực nghiệm đóng băng
│   ├── 1_Baseline/            # Mô hình M1: Baseline chuẩn
│   ├── 2_Naive_GNN_PE_Before/ # Mô hình M2: GNN đặt PE trước GAT
│   ├── 3_Corrected_GNN_PE_After/ # Mô hình M3: GNN chuẩn của luận văn (PE sau GAT)
│   ├── 4_Coord_Aware_GAT_1L_4H/  # Mô hình M4: Coordinate-Aware GAT 1L4H
│   └── 5_Coord_Aware_GAT_2L_8H/  # Mô hình M5: Coordinate-Aware GAT 2L8H (Negative result)
├── App/                       # Ứng dụng Expression Page Explorer (FastAPI + React Konva)
├── BoCauHoi/                  # Bộ 140 câu hỏi - đáp chuẩn bị bảo vệ luận văn
├── LuanVan/                   # Tài liệu luận văn tốt nghiệp
└── README.md                  # Tài liệu tổng hợp dự án
```

## 🧠 Deep Dive: Graph Attention Networks (GAT)

Điểm nhấn của dự án là việc tích hợp **Graph Attention Networks (GAT)** trực tiếp trên feature map của Encoder:

### Tại sao lại dùng GAT?

Các mạng CNN truyền thống (như DenseNet) trích xuất tốt đặc trưng cục bộ (local visual features). Tuy nhiên, đối với biểu thức toán học, mối quan hệ giữa các ký hiệu chứa cấu trúc không gian 2D đa hướng (phân số, số mũ, chỉ số dưới, căn thức, tích phân).

GAT cho phép coi bản đồ đặc trưng (feature map $H' \times W'$) như một đồ thị lưới (feature-grid graph), nơi mỗi ô đặc trưng là một nút (node). Cơ chế Attention giúp các nút lân cận truyền tin và tổng hợp ngữ cảnh không gian có trọng số.

### Kiến trúc chi tiết (Implementation Details)

Module GAT được cài đặt trong `1-cnn-gnn/tamer/model/gat.py` và `1-cnn-gnn/tamer/model/encoder.py`:

1. **Xây dựng Đồ thị (Graph Construction)**:
   * Feature map đầu ra từ DenseNet có kích thước `[B, D, H', W']` được chiếu về chiều $D = 256$.
   * Biến đổi feature map thành đồ thị lưới với $N = H' \times W'$ nút.
   * **Adjacency Matrix**: Ma trận kề kết nối theo lưới 8 hướng (ngang, dọc, 2 đường chéo và self-loop).
2. **Cơ chế GAT Layer**:
   * Mỗi lớp GAT sử dụng **Multi-head Attention**.
   * Hệ số chú ý $e_{ij}$ giữa nút $i$ và lân cận $j$ được tính:
     $$e_{ij} = \text{LeakyReLU}(\vec{a}^T [W\vec{h}_i \,||\, W\vec{h}_j] + b_{ij})$$
     *(trong đó $b_{ij}$ là Relative Position Bias 9 trạng thái ở biến thể M4/M5)*.
   * Chuẩn hóa bằng Softmax trên tập lân cận $\mathcal{N}_i$ thành $\alpha_{ij}$.
   * Cập nhật đặc trưng nút:
     $$\vec{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W\vec{h}_j\right)$$
3. **Tích hợp vào Encoder**:
   * Quy trình xử lý: `Image -> DenseNet -> Feature Map -> Feature Grid Graph -> GAT Layers -> Positional Encoding -> Transformer Decoder -> LaTeX Sequence`.
   * Thứ tự đặt Positional Encoding (PE) **sau** GAT là mấu chốt để tránh hiện tượng làm nhòe tọa độ (PE blurring).

## 🔧 Cài đặt

Yêu cầu môi trường:
- Python 3.8+
- PyTorch 1.8+
- CUDA (nếu dùng GPU)

```bash
# Cài đặt cho phiên bản GAT
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

## 🚀 Sử dụng

### Quá trình Huấn luyện (Training)

```bash
cd 1-cnn-gnn
python train.py fit --config config/crohme.yaml
```

### Đánh giá (Evaluation)

```bash
cd 1-cnn-gnn/eval
bash eval_crohme.sh
```

## 📊 Báo cáo Tổng hợp Thực nghiệm: Quá trình Tiến bộ & Đóng góp Khoa học

### 1. Bảng Tổng hợp Kết quả Định lượng Đóng băng (Frozen Results)

Dưới đây là bảng đối chiếu kết quả đánh giá chính thức của 5 phiên bản mô hình trên 3 tập kiểm thử chuẩn **CROHME 2014, 2016, và 2019** (huấn luyện 100 epochs trên 2x NVIDIA Tesla T4):

* **M1: Baseline** (`1_Baseline`, W&B `8ivyzmlm`) — DenseNet + Transformer Decoder (Không có GAT).
* **M2: Naive GAT** (`2_Naive_GNN_PE_Before`, MLflow `8b964c54a2d94b8ca0e667db6ceba820`) — GAT 2L, 8H, **PE đặt TRƯỚC GAT**.
* **M3: Corrected GAT** (`3_Corrected_GNN_PE_After`, MLflow `2e2611189af24ca5955fd73ceaa57d9c`, run `defiant-mole-974`) — GAT 2L, 8H, **PE đặt SAU GAT** (*Mô hình CNN-GAT chính của luận văn*).
* **M4: Coord-Aware GAT (1L, 4H)** (`4_Coord_Aware_GAT_1L_4H`, MLflow `c861d5eb87304ca0933fa8c603c9dac9`) — GAT 1L, 4H, PE sau GAT + Relative Position Bias kề 8 hướng (*Thử nghiệm mở rộng*).
* **M5: Coord-Aware GAT (2L, 8H)** (`5_Coord_Aware_GAT_2L_8H`, MLflow `e44ee12972f8447482b89d0a5f1acbf2`) — GAT 2L, 8H, PE sau GAT + Relative Position Bias kề 8 hướng (*Scale-up Negative Result*).

| Tập dữ liệu (Dataset) | Chỉ số (Metric) | M1: Baseline | M2: Naive GAT (PE trước GAT) | M3: Corrected GAT (PE sau GAT) | M4: Coord-Aware (1L, 4H) | M5: Coord-Aware (2L, 8H) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CROHME 2014** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance (MED) | **51.12%** <br> **69.98%** <br> **77.69%** <br> 1.99 | 49.39% <br> 66.53% <br> 75.25% <br> 2.22 | 48.88% <br> 66.73% <br> 75.36% <br> 2.19 | 49.90% <br> 67.44% <br> 77.18% <br> **1.98** | 46.65% <br> 63.99% <br> 73.43% <br> 2.48 |
| **CROHME 2016** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance (MED) | 50.65% <br> **67.92%** <br> **76.02%** <br> 2.17 | 47.43% <br> 64.95% <br> 74.72% <br> 2.31 | **50.74%** <br> 66.96% <br> 75.85% <br> 2.19 | 49.17% <br> 67.13% <br> 75.76% <br> **2.13** | 45.68% <br> 63.03% <br> 73.23% <br> 2.53 |
| **CROHME 2019** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance (MED) | **48.54%** <br> **68.14%** <br> 77.23% <br> 2.14 | 46.71% <br> 66.89% <br> 75.90% <br> 2.11 | 47.87% <br> 67.81% <br> **77.98%** <br> **2.02** | 47.87% <br> 67.72% <br> 76.90% <br> 2.08 | 37.70% <br> 58.97% <br> 70.14% <br> 2.93 |
| **Trung bình (Macro Avg)** | ExpRate (Khớp 100%) <br> ExpRate $\le$ 1 <br> ExpRate $\le$ 2 <br> Mean Edit Distance (MED) | **50.10%** <br> **68.68%** <br> **76.98%** <br> 2.10 | 47.84% <br> 66.12% <br> 75.29% <br> 2.21 | **49.17%** <br> 67.17% <br> 76.40% <br> 2.14 | 48.98% <br> 67.43% <br> 76.61% <br> **2.06** | 43.35% <br> 62.00% <br> 72.27% <br> 2.65 |

---

### 2. Phân tích Các Phát hiện Khoa học Cốt lõi

1. **Baseline M1 là đối chứng mạnh nhất về ExpRate (50.10%):**
   Mô hình DenseNet + Transformer Decoder không dùng GAT đạt tỷ lệ khớp chính xác 100% cao nhất trên trung bình 3 tập kiểm thử. Nghiên cứu giữ nguyên M1 làm đối chứng minh bạch, không loại bỏ kết quả bất lợi.
2. **Ảnh hưởng của Vị trí Positional Encoding (M2 vs M3):**
   * *M2 (PE trước GAT):* Message passing trên vector đã chứa PE làm mịn và nhòe thông tin tọa độ tuyệt đối, khiến ExpRate trung bình sụt giảm xuống 47.84% (-2.26% so với Baseline).
   * *M3 (PE sau GAT):* Để GAT truyền tin trên đặc trưng visual thuần túy rồi mới cộng PE trước khi đưa vào Decoder giúp phục hồi ExpRate lên **49.17%** (+1.33% so với M2), vượt Baseline trên CROHME 2016 (50.74%) và đạt ExpRate $\le 2$ cao nhất trên CROHME 2019 (77.98%).
3. **Khảo sát Coordinate-Aware Relative Bias (M4):**
   * M4 bổ sung 9 quan hệ relative position bias trên lưới 8 hướng, đồng thời rút gọn cấu hình xuống 1 lớp / 4 heads. M4 đạt Mean Edit Distance trung bình thấp nhất (**2.06** so với **2.10** của Baseline), cho thấy khả năng bảo toàn cấu trúc tốt hơn khi gặp dự đoán sai.
4. **Giới hạn Scale-up trên Đồ thị Lưới Thưa (M5 - Negative Result):**
   * M5 nâng lên 2 lớp / 8 heads làm hiệu năng giảm mạnh xuống **43.35%**. Việc xếp chồng nhiều lớp phi tuyến (LeakyReLU/ELU) lên relative bias logits và dropout trên đồ thị lưới thưa gây đứt gãy luồng thông tin và làm biến dạng quan hệ khoảng cách tương đối.

---
© 2026 Phan Hoàng Khải — Trường Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE). Đề án Thạc sĩ Khoa học Máy tính.

