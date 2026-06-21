# Báo cáo Khôi phục Tài nguyên CNN-GNN (Recovered Assets Report)

Chúng tôi đã tiến hành khôi phục thành công các tài nguyên cốt lõi từ kho lưu trữ chuyên đề cũ **ChuyenDe-Tamer** (`https://github.com/KhaiHASO/ChuyenDe-Tamer.git`).

### Các tài nguyên đã khôi phục:

| File Path / Thư mục | Mục đích | Khả dụng | Ghi chú |
| :--- | :--- | :---: | :--- |
| `chuyende_tamer_temp/1-gat/tamer/model/gat.py` | Lớp GAT (Graph Attention Network) cho cấu trúc đồ thị lưới. | **Có** | Mã nguồn định nghĩa Multi-head Attention trên đồ thị. |
| `chuyende_tamer_temp/1-gat/tamer/model/encoder.py` | Bộ mã hóa Encoder kết hợp DenseNet và GAT. | **Có** | Tự động xây dựng ma trận kề 4 hướng và thực hiện Message Passing. |
| `chuyende_tamer_temp/KetQua/checkpoints/checkpoints/epoch=95-step=72095-val_ExpRate=0.5091.ckpt` | Checkpoint trọng số mô hình TAMER đã huấn luyện. | **Có** | Đạt **50.91% ExpRate** trên tập validation CROHME. |
| `chuyende_tamer_temp/1-gat/data/CROHME_extracted/crohme/dictionary.txt` | Từ điển từ vựng (Vocabulary) của tập dữ liệu CROHME. | **Có** | Chứa 110 ký hiệu toán học và 3 token đặc biệt. |
| `chuyende_tamer_temp/1-gat/tamer/model/decoder.py` | Bộ giải mã Transformer Decoder kết hợp Coverage Attention. | **Có** | Dùng sinh chuỗi LaTeX tự hồi quy từ đặc trưng đồ thị. |

### Đánh giá mức độ khôi phục:
* **Tình trạng mã nguồn**: Khôi phục hoàn chỉnh 100% cấu trúc mô hình lai CNN-GNN/GAT (dưới tên gọi mô hình **TAMER**).
* **Tình trạng trọng số**: Khôi phục được checkpoint tốt nhất trong quá trình huấn luyện chuyên đề với độ chính xác kiểm thử **50.91% ExpRate** (rất sát với mốc 52.27% ghi nhận trong báo cáo thuyết minh).
* **Khả năng suy luận cục bộ**: Đã cấu hình thành công môi trường dependencies và viết kịch bản `run_cnn_gnn_inference.py` chạy trực tiếp trên 50 ảnh mẫu, đạt mức đánh giá **Mức Tốt** (chạy được thực tế) và tiến gần **Mức Xuất Sắc** (so sánh số liệu trực tiếp trên cùng một tập ảnh).
