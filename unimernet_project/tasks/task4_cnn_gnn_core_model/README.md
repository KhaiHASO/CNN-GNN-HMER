# Task 4: Đóng gói Mô hình CNN-GNN Chính thức (Official CNN-GNN Core Model Packaging)

Thư mục này thực hiện **Task 4: Đóng gói lại mô hình CNN-GNN chính thức của đề tài** nhằm đưa mô hình lai CNN-GNN/GAT trở lại trung tâm luận văn, dựa trên nghiên cứu chính thức **GETD (Tang et al., 2024)** đăng trên tạp chí *Pattern Recognition*.

---

## 1. Cấu trúc Thư mục (Folder Structure)

```text
tasks/task4_cnn_gnn_core_model/
├── docs/
│   ├── architecture_cnn_gnn.md       # Giải thích chi tiết kiến trúc lai đề xuất (YOLOv5 + Line-of-Sight + GAT + Decoder)
│   ├── model_role_in_thesis.md       # Phân định rõ vai trò CNN-GNN (chính) vs UniMERNet (baseline đối chứng)
│   ├── recovered_assets.md           # Báo cáo kết quả quét tài nguyên mô hình trên máy local
│   └── milestone_04_cnn_gnn_core.md  # Cột mốc hoàn thành Task 4 và câu trả lời phục vụ Hội đồng
├── experiments/
│   ├── cnn_gnn_result_crohme.csv            # Số liệu 52.27% ExpRate trên CROHME (kế thừa từ chuyên đề)
│   ├── cnn_gnn_result_quick_test_50.csv     # File rỗng/header-only do không chạy được cục bộ
│   └── comparison_cnn_gnn_vs_unimernet.csv  # Bảng so sánh 5 cấu hình A0, A1, A2, A3, A4
├── reports/
│   ├── cnn_gnn_experiment_log.md     # Nhật ký chạy thực nghiệm khôi phục
│   ├── cnn_gnn_error_analysis.md     # Phân tích so sánh lỗi hệ thống (Chuỗi phẳng vs Đồ thị bố cục)
│   └── figures/                      # Thư mục chứa hình ảnh minh họa (nếu có)
├── scripts/
│   ├── scan_cnn_gnn_assets.py        # Kịch bản quét tìm tài nguyên CNN-GNN trên máy local
│   ├── run_cnn_gnn_inference.py      # Kịch bản kiểm thử suy luận (chứa cảnh báo thiếu checkpoint)
│   ├── evaluate_cnn_gnn.py           # Kịch bản đánh giá mô hình trên CROHME
│   └── compare_with_unimernet.py     # Kịch bản tổng hợp bảng so sánh kết quả
└── README.md                         # Hướng dẫn này
```

---

## 2. Tài liệu Tham khảo Chính (Core Reference & Literature)

| Hạng mục | Tên công trình / Tài liệu tham khảo | Ý nghĩa đối với Luận văn |
| :--- | :--- | :--- |
| **Kiến trúc chính** | **GETD**: *Offline HMER with Graph Encoder and Transformer Decoder* (PR 2024) | Thiết lập luồng xử lý: YOLOv5 $\rightarrow$ Line-of-Sight Graph $\rightarrow$ Graph Encoder $\rightarrow$ Transformer Decoder. |
| **Nền tảng lý thuyết** | **GRN**: *Offline HMER via Graph Reasoning Network* | Cung cấp luận điểm bảo vệ sự cần thiết của GNN trong việc học cấu trúc 2D phức tạp của biểu thức toán học. |
| **Lý thuyết đồ thị** | **Graph-to-Graph (G2G)**: *Towards Accurate and Interpretable Online HMER* (AAAI) | Hỗ trợ lập luận về khả năng giải thích được (interpretability) của biểu diễn đồ thị. |
| **Mã nguồn bổ trợ** | **math_online_egat** (GitHub) | Tham khảo triển khai thực tế của mạng chú ý đồ thị GAT/EGAT trên biểu thức toán học. |

---

## 3. Hướng dẫn sử dụng các Kịch bản (Scripts Usage)

### 3.1. Quét tài nguyên hệ thống
```bash
python tasks/task4_cnn_gnn_core_model/scripts/scan_cnn_gnn_assets.py
```

### 3.2. Thử nghiệm chạy suy luận (Inference Test)
```bash
python tasks/task4_cnn_gnn_core_model/scripts/run_cnn_gnn_inference.py
```

### 3.3. Đánh giá mô hình trên CROHME
```bash
python tasks/task4_cnn_gnn_core_model/scripts/evaluate_cnn_gnn.py
```

### 3.4. Tổng hợp bảng số liệu so sánh
```bash
python tasks/task4_cnn_gnn_core_model/scripts/compare_with_unimernet.py
```
