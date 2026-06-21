# Milestone 01: Chạy thành công UniMERNet Direct Recognition

Hệ thống UniMERNet đã được cài đặt và chạy thành công ở chế độ Direct Recognition. 
Với ảnh đầu vào là một biểu thức toán học viết tay đã được cắt gọn, hệ thống có thể sinh mã LaTeX và render lại công thức tương ứng.

Kết quả thử nghiệm ban đầu cho thấy mô hình nhận diện được các thành phần quan trọng như biến, số mũ, phân số và toán tử. 
Đây là cơ sở để sử dụng UniMERNet làm backbone nhận dạng trong luận văn. 

Các bước tiếp theo gồm: 
1. Chạy batch inference trên tập 50 ảnh để thu được độ chính xác thực tế (Baseline).
2. Ghi nhận lỗi và phân tích các trường hợp thất bại (Error Cases Log).
3. Xây dựng module tiền xử lý ảnh (Image Preprocessing) để tối ưu hóa nét vẽ.
4. Phát triển bộ kiểm chứng dựa trên đồ thị (Graph-based Validator) để phát hiện và sửa các lỗi cấu trúc LaTeX (như thiếu ngoặc, lỗi phân số, căn thức).
