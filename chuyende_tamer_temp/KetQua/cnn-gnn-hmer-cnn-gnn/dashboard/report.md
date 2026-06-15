# Báo cáo phân tích CNN-GNN HMER

## 1. Tổng quan

Dashboard này tổng hợp kết quả chạy mô hình **CNN-GNN HMER** tại thư mục `chuyende_tamer_temp/1-cnn-gnn`. Đây là phiên bản cải tiến so với CNN-Transformer baseline bằng cách bật nhánh Graph Attention Network qua tham số `use_gat=True`.

Nguồn dữ liệu được lấy từ bộ metadata MLflow/Dagshub trong thư mục:

- `metadata/run.json`
- `metadata/artifacts.json`
- `README_DAGSHUB_DOWNLOAD.json`
- `artifacts/checkpoints/epoch=77-step=58577-val_ExpRate=0.4939.ckpt`

## 2. Metadata run

| Trường | Giá trị |
| :--- | :--- |
| Run name | `colorful-moose-173` |
| Run ID | `8b964c54a2d94b8ca0e667db6ceba820` |
| Trạng thái | `FINISHED` |
| User | `root` |
| MLflow user tag | `khaihaso` |
| Source | `/kaggle/working/CNN-GNN-HMER/chuyende_tamer_temp/1-cnn-gnn/train.py` |
| Commit | `c367169c3b5f6e7bbde717481ff4868c64361e35` |
| Artifact URI | `mlflow-artifacts:/0fad7ceceed4469bbbdafa27397817f4/8b964c54a2d94b8ca0e667db6ceba820/artifacts` |

Thời lượng chạy theo metadata:

- Start time: `1781437670343`
- End time: `1781467087676`
- Runtime xấp xỉ: **29,417 giây**, tương đương **8.17 giờ**

## 3. Cấu hình mô hình

| Nhóm | Tham số | Giá trị |
| :--- | :--- | :--- |
| Backbone | `growth_rate` | `24` |
| Backbone | `num_layers` | `16` |
| Decoder | `num_decoder_layers` | `3` |
| Decoder | `d_model` | `256` |
| Decoder | `nhead` | `8` |
| Decoder | `dim_feedforward` | `1024` |
| GNN | `use_gat` | `True` |
| GNN | `gat_num_layers` | `2` |
| GNN | `gat_num_heads` | `8` |
| GNN | `gat_dropout` | `0.1` |
| Training | `learning_rate` | `1.0` |
| Training | `dropout` | `0.3` |
| Training | `early_stopping` | `False` |
| Inference | `beam_size` | `10` |
| Inference | `max_len` | `150` |
| Objective | `self_coverage` | `True` |
| Objective | `cross_coverage` | `True` |

## 4. Kết quả chính

| Chỉ số | Giá trị |
| :--- | ---: |
| Epoch cuối | `99` |
| Global step cuối | `75099` |
| Train loss cuối | `0.1600` |
| Train struct loss cuối | `0.0154` |
| Validation loss cuối | `0.5059` |
| Validation struct loss cuối | `0.0274` |
| Validation ExpRate cuối | **46.15%** |
| Checkpoint tốt nhất ghi nhận | **49.39%** tại epoch `77` |

Điểm quan trọng là kết quả cuối của run thấp hơn checkpoint tốt nhất. Điều này cho thấy mô hình đã đạt đỉnh trước khi kết thúc huấn luyện, sau đó có dấu hiệu dao động hoặc suy giảm nhẹ trên validation.

## 5. Diễn biến checkpoint

Các checkpoint được lưu khi `val_ExpRate` cải thiện hoặc đạt mốc quan trọng:

| Epoch | Step | Val ExpRate |
| ---: | ---: | ---: |
| 1 | 1501 | 0.00% |
| 7 | 6007 | 0.81% |
| 9 | 7509 | 7.09% |
| 11 | 9011 | 11.23% |
| 13 | 10513 | 11.74% |
| 15 | 12015 | 19.64% |
| 17 | 13517 | 23.18% |
| 19 | 15019 | 28.54% |
| 23 | 18023 | 33.10% |
| 27 | 21027 | 36.23% |
| 31 | 24031 | 36.94% |
| 37 | 28537 | 40.08% |
| 41 | 31541 | 40.18% |
| 47 | 36047 | 42.61% |
| 53 | 40553 | 45.14% |
| 63 | 48063 | 47.17% |
| 77 | 58577 | **49.39%** |

Quỹ đạo học tăng ổn định từ epoch 1 đến 77. Sau epoch 77, metadata cuối epoch 99 chỉ còn `val_ExpRate=46.15%`, tức giảm **3.24 điểm phần trăm** so với checkpoint tốt nhất.

## 6. So sánh với baseline

| Mô hình | Run | Epoch cuối | Val ExpRate cuối | Checkpoint tốt nhất |
| :--- | :--- | ---: | ---: | ---: |
| CNN-Transformer baseline | `8ivyzmlm` | 99 | 50.30% | 50.91% |
| CNN-GNN | `8b964c54...` | 99 | 46.15% | 49.39% |

Chênh lệch chính:

- Theo checkpoint tốt nhất, CNN-GNN thấp hơn baseline **1.52 điểm phần trăm**.
- Theo metric cuối run, CNN-GNN thấp hơn baseline **4.15 điểm phần trăm**.
- CNN-GNN đã chạy đủ 100 epoch và ổn định hơn so với lần ghi nhận cũ bị crash ở epoch 19, nhưng vẫn chưa vượt baseline.

## 7. Nhận xét kỹ thuật

CNN-GNN cho thấy tác dụng tích cực của nhánh GAT khi nhìn vào tốc độ cải thiện checkpoint: từ 28.54% tại epoch 19 lên 49.39% tại epoch 77. Điều này cho thấy mô hình không còn bị chặn sớm và có khả năng học biểu diễn không gian tốt hơn khi được huấn luyện đủ lâu.

Tuy nhiên, khoảng cách với baseline chưa được đóng hoàn toàn. Có ba dấu hiệu cần chú ý:

- Validation loss cuối `0.5059` cao hơn baseline cuối `0.4378`.
- Checkpoint tốt nhất xuất hiện ở epoch 77, không phải epoch cuối.
- ExpRate cuối giảm xuống 46.15%, cho thấy cần dùng checkpoint tốt nhất để đánh giá, không nên lấy epoch cuối làm mô hình đại diện.

## 8. Khuyến nghị

1. Dùng checkpoint `epoch=77-step=58577-val_ExpRate=0.4939.ckpt` cho đánh giá chính thức thay vì epoch cuối.
2. Chạy lại evaluation chi tiết trên CROHME 2014/2016/2019 để có bảng lỗi giống baseline.
3. Bật early stopping hoặc lưu best checkpoint theo `val_ExpRate` để tránh chọn mô hình sau khi hiệu năng giảm.
4. So sánh lỗi theo nhóm biểu thức dài, ký hiệu chồng/lồng và cấu trúc phân số để xác định nhánh GAT giúp hoặc làm hại ở loại mẫu nào.
5. Thử ablation `use_gat=False`, `gat_num_layers=1`, `gat_num_heads=4/8` để tách ảnh hưởng của GAT khỏi decoder Transformer.

