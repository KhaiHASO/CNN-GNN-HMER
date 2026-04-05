# GNN Sandbox

Phase 1 da hoan thanh. Cau truc hien tai da tach sang tung phase de chuyen sang Phase 2: GNN Design & Edge Generation.

Thư mục này là môi trường tách rời khỏi pipeline `baseline`, dùng để kiểm tra nhanh các giả thuyết CNN + GNN cho HMER trước khi đụng vào training stack thật.

## Phase 1 tập trung vào gì

- Tách phần nghiên cứu thuật toán ra khỏi môi trường train/inference chính.
- Xác nhận shape dữ liệu từ dataset giả lập sang encoder.
- Hiểu cách LaTeX được parse thành cây quan hệ không gian.
- Thử luồng `feature map -> graph -> message passing/GNN -> feature map`.

## Phase 1 không làm gì

- Không train mô hình đầy đủ.
- Không phụ thuộc vào `baseline` runtime.
- Không tối ưu hiệu năng, config, checkpoint, hoặc pipeline dữ liệu thật.

## Cấu trúc thư mục

- `phase_1_baseline_analysis/`: dataset sandbox, parser cay khong gian, va cac test phan tich co ban.
- `phase_2_gnn_design/`: encoder, graph utilities, va cac script thiet ke GNN/edge generation.
- `phase_3_hybrid_integration/`: cho buoc dua code tro lai pipeline chinh.
- `run_phase1.py`: entrypoint chay lai toan bo 3 kich ban cua Phase 1 tren cau truc moi.
- `PHASE1.md`: scope, output kỳ vọng, và tiêu chí hoàn tất của Phase 1.
- `requirements-phase1.txt`: dependency tối thiểu để replay lại Phase 1.
- `log_utils.py`: helper logging dùng chung.
- `logs/`: nơi ghi log moi nhat cho tung script.

## Cách chạy

Đứng tại thư mục:

```bash
cd /home/khai/Desktop/github/CNN-GNN-HMER/gnn_sandbox
```

Chạy toàn bộ Phase 1:

```bash
pip install -r requirements-phase1.txt
python run_phase1.py
```

Chạy từng kịch bản riêng:

```bash
python -m phase_1_baseline_analysis.test_data
python -m phase_1_baseline_analysis.test_tree
python -m phase_2_gnn_design.test_gnn_graph
```

Chạy từng mô-đun độc lập:

```bash
python -m phase_1_baseline_analysis.sandbox_dataset
python -m phase_2_gnn_design.sandbox_encoder
python -m phase_1_baseline_analysis.sandbox_latex2gtd
```

## Output kỳ vọng

- `phase_1_baseline_analysis.test_data`: encoder nhận batch phuc tap co padding/mask va tra feature map da encode.
- `phase_1_baseline_analysis.test_tree`: log cay khong gian cho bo cong thuc cuc doan trong sandbox dataset.
- `phase_2_gnn_design.test_gnn_graph`: tạo `edge_index` lưới `8x8` và reshape output về `(2, 256, 8, 8)`.

## Log

Mỗi script ghi log riêng trong `logs/` ở chế độ ghi đè. Có thể xem nhanh bằng:

```bash
ls logs
tail -n 30 logs/test_gnn_graph.log
```

## Phase 2 Focus

- Viet cac ham tao canh trong `phase_2_gnn_design/graph_utils.py`.
- Thu nghiem Local Grid, KNN, va cac bien the edge generation khac.
- Giu ro rang bien doi shape tensor qua tung buoc trong graph pipeline.