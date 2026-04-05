# Phase 1

## Tên pha

Sandbox Decoupling & Pure Algorithm Prototyping

## Mục tiêu

- Dựng môi trường nghiên cứu độc lập, ít phụ thuộc.
- Kiểm tra đúng đắn về shape, graph topology, và logic parse cây.
- Tạo nền để tích hợp dần vào `baseline` ở các pha sau.

## Deliverables

- Dataset giả lập chạy độc lập.
- Encoder standalone chạy được với tensor giả.
- Parser LaTeX sang cây quan hệ chạy được.
- Demo graph trên lưới feature map chạy được.
- Một entrypoint duy nhất để replay toàn bộ kiểm chứng của pha.

## Tiêu chí hoàn tất

- `python run_phase1.py` chạy thành công.
- Ba kịch bản `test_data.py`, `test_tree.py`, `test_gnn_graph.py` đều có log riêng.
- Không cần import code từ `baseline` để hoàn thành demo.

## Ghi chú phạm vi

Phase 1 ưu tiên tính cô lập và khả năng giải thích. Nếu cần train thật hoặc dùng dữ liệu thật, đó là Phase 2 trở đi.
