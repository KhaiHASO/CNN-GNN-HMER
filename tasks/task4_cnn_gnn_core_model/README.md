# Task 4: Đóng gói Mô hình CNN-GNN Chính thức (Official CNN-GNN Core Model Packaging)

Thư mục này thực hiện **Task 4: Đóng gói lại mô hình CNN-GNN chính thức của đề tài** nhằm đưa mô hình lai CNN-GNN/GAT trở lại trung tâm luận văn, đồng thời định vị UniMERNet là mô hình baseline đối chứng.

---

## 1. Cấu trúc Thư mục (Folder Structure)

```text
tasks/task4_cnn_gnn_core_model/
├── docs/
│   ├── architecture_cnn_gnn.md       # Giải thích chi tiết kiến trúc lai đề xuất (CNN/YOLO + GAT + Transformer Decoder)
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

## 2. Hướng dẫn sử dụng các Kịch bản (Scripts Usage)

### 2.1. Quét tài nguyên hệ thống
Để kiểm tra lại các tài nguyên liên quan đến CNN-GNN trong repo:
```bash
python tasks/task4_cnn_gnn_core_model/scripts/scan_cnn_gnn_assets.py
```

### 2.2. Thử nghiệm chạy suy luận (Inference Test)
Kịch bản chạy thử nghiệm trên máy local (sẽ đưa ra cảnh báo thiếu checkpoint):
```bash
python tasks/task4_cnn_gnn_core_model/scripts/run_cnn_gnn_inference.py
```

### 2.3. Đánh giá mô hình trên CROHME
Hiển thị kết quả đánh giá CROHME kế thừa từ chuyên đề nghiên cứu:
```bash
python tasks/task4_cnn_gnn_core_model/scripts/evaluate_cnn_gnn.py
```

### 2.4. Tổng hợp bảng số liệu so sánh
Tự động kết hợp dữ liệu từ Task 2 và Task 3 với kết quả CNN-GNN để ghi ra tệp `comparison_cnn_gnn_vs_unimernet.csv`:
```bash
python tasks/task4_cnn_gnn_core_model/scripts/compare_with_unimernet.py
```

---

## 3. 6 Câu hỏi Cốt lõi của Hội đồng (Core Q&A for Defense)

| Câu hỏi | Vị trí và Ý nghĩa của Câu trả lời |
| :--- | :--- |
| **CNN nằm ở đâu?** | Ở tầng trích xuất đặc trưng visual ban đầu và phát hiện các hộp bao ký hiệu (YOLO/ResNet). |
| **GNN nằm ở đâu?** | Nằm ở bộ mã hóa (GNN/GAT Encoder) để thực hiện truyền thông điệp giữa các nút lân cận trên đồ thị bố cục. |
| **Graph là gì?** | Đồ thị bố cục ký hiệu (Symbol Layout Graph): Ký hiệu là nút (Node), quan hệ hình học 2D là cạnh (Edge). |
| **Sinh LaTeX ra sao?** | Giải mã thông tin đồ thị từ GNN bằng cơ chế Cross-attention của Transformer Decoder để sinh ra chuỗi LaTeX. |
| **Kết quả mô hình cũ?** | Đạt **52.27% ExpRate** trên tập kiểm thử CROHME chuẩn quốc tế. |
| **Vai trò UniMERNet?** | Chỉ làm mô hình baseline đối chứng (SOTA hiện tại), không phải mô hình đề xuất. |
