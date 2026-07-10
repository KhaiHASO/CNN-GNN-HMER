# BỘ 140 CÂU HỎI “VÌ SAO” CHUẨN BỊ BẢO VỆ LUẬN VĂN HMER

## Phạm vi sử dụng

- **Nguồn sự thật chính:** repo luận văn hiện tại, bao gồm code, config, dataset loader, script đánh giá, log, checkpoint và demo.
- **Tài liệu chuyên đề cũ:** chỉ dùng để tham khảo bối cảnh và quá trình hình thành hướng nghiên cứu; không mặc định dùng kiến trúc, cấu hình hoặc kết quả cũ cho luận văn mới.
- Mọi câu trả lời kỹ thuật nên truy được về ít nhất một trong bốn loại bằng chứng: **code – dữ liệu – công thức metric – kết quả thực nghiệm**.

## Mức độ ưu tiên

- **ĐỎ — 60 câu:** phải trả lời ngay trong khoảng 30–45 giây.
- **VÀNG — 50 câu:** cần trả lời trong 1–2 phút, kèm số liệu, bảng hoặc dẫn chứng.
- **BẪY — 30 câu:** không nhất thiết có kết luận chắc chắn; phải phân biệt rõ quan sát, giả thuyết và cách kiểm chứng.

## Cấu trúc 12 nhóm

| Nhóm | Nội dung | Số câu |
|---:|---|---:|
| 1 | Bài toán, mục tiêu và phạm vi | 8 |
| 2 | Khoảng trống nghiên cứu và đóng góp mới | 8 |
| 3 | Dataset và phân bố dữ liệu | 14 |
| 4 | Tiền xử lý ảnh, token hóa và nhãn LaTeX | 10 |
| 5 | Kiến trúc tổng thể | 10 |
| 6 | Graph, GNN, GAT và xây dựng đồ thị | 14 |
| 7 | M1–M5 và ablation | 16 |
| 8 | Huấn luyện và hyperparameter | 10 |
| 9 | Metric và quy trình đánh giá | 14 |
| 10 | Phân tích kết quả giữa dataset và mô hình | 16 |
| 11 | Phân tích lỗi và demo | 12 |
| 12 | Hạn chế, tái lập và hướng phát triển | 8 |
|  | **Tổng cộng** | **140** |

## Khung trả lời ba tầng

1. **Tầng 1 — Trả lời thẳng:** kết luận trong một hoặc hai câu.
2. **Tầng 2 — Giải thích cơ chế:** vì sao kiến trúc, dữ liệu hoặc metric tạo ra hiện tượng đó.
3. **Tầng 3 — Bằng chứng và giới hạn:** số liệu, log, script, ví dụ lỗi; nêu rõ điều gì đã chứng minh và điều gì mới là giả thuyết.

---

## Nhóm 1 — Bài toán, mục tiêu và phạm vi nghiên cứu

**Mục tiêu nhóm:** Làm rõ luận văn giải quyết bài toán gì, giải quyết đến đâu và không tuyên bố quá phạm vi.

1. **[ĐỎ]** Bài toán nghiên cứu chính xác của luận văn là gì, đầu vào và đầu ra của hệ thống là gì?
2. **[ĐỎ]** Vì sao nhận dạng biểu thức toán học viết tay khó hơn OCR văn bản thông thường?
3. **[ĐỎ]** Luận văn tập trung vào nhận dạng ký hiệu, phân tích cấu trúc hay sinh chuỗi LaTeX end-to-end?
4. **[ĐỎ]** Mục tiêu tổng quát và các mục tiêu cụ thể của luận văn là gì?
5. **[ĐỎ]** Phạm vi dữ liệu, loại biểu thức và điều kiện thực nghiệm được giới hạn như thế nào?
6. **[VÀNG]** Vì sao chọn đầu ra LaTeX thay vì MathML, cây cú pháp hoặc ảnh đã chuẩn hóa?
7. **[VÀNG]** Luận văn giải quyết bài toán online handwriting hay offline image recognition, và vì sao?
8. **[BẪY]** Trong trường hợp kết quả chưa vượt baseline trên mọi tập kiểm thử, luận văn còn giá trị khoa học ở điểm nào?

---

## Nhóm 2 — Khoảng trống nghiên cứu và đóng góp mới

**Mục tiêu nhóm:** Chứng minh đề tài không chỉ là ghép module, mà có giả thuyết nghiên cứu và đóng góp kiểm chứng được.

9. **[ĐỎ]** Khoảng trống nghiên cứu mà luận văn muốn giải quyết là gì?
10. **[ĐỎ]** Vì sao các mô hình CNN–Transformer hoặc sequence-to-sequence vẫn có hạn chế với cấu trúc 2D?
11. **[ĐỎ]** Đóng góp mới cụ thể của luận văn nằm ở kiến trúc, cách xây dựng graph, thông tin vị trí hay quy trình đánh giá?
12. **[ĐỎ]** Điểm khác biệt giữa công trình hiện tại và cuốn chuyên đề cũ là gì?
13. **[VÀNG]** Vì sao lựa chọn GNN/GAT thay vì chỉ tăng độ sâu CNN hoặc dùng thêm self-attention?
14. **[VÀNG]** Đóng góp nào là đóng góp phương pháp, đóng góp thực nghiệm và đóng góp kỹ thuật triển khai?
15. **[VÀNG]** Ablation M1–M5 giúp kiểm chứng giả thuyết nghiên cứu nào?
16. **[BẪY]** Nếu hội đồng cho rằng đây chỉ là tích hợp các thành phần đã có, em sẽ chứng minh tính mới như thế nào?

---

## Nhóm 3 — Dataset và phân bố dữ liệu

**Mục tiêu nhóm:** Nắm chắc nguồn dữ liệu, cách chia tập, khác biệt giữa các năm và bằng chứng cho mọi nhận định về dữ liệu.

17. **[ĐỎ]** Repo hiện tại sử dụng chính xác những bộ dữ liệu nào cho train, validation và test?
18. **[ĐỎ]** Mỗi tập có bao nhiêu mẫu và số liệu này được lấy từ file hoặc script nào?
19. **[ĐỎ]** Dữ liệu gốc là stroke online hay ảnh raster offline; repo sử dụng dạng nào?
20. **[ĐỎ]** Quy trình chuyển từ dữ liệu gốc sang ảnh đầu vào của mô hình được thực hiện như thế nào?
21. **[ĐỎ]** CROHME 2014, 2016 và 2019 khác nhau ở những điểm nào liên quan đến phân bố dữ liệu?
22. **[ĐỎ]** Vì sao cần đánh giá trên nhiều tập test thay vì chỉ dùng một tập?
23. **[ĐỎ]** Có nguy cơ trùng người viết, trùng biểu thức hoặc rò rỉ dữ liệu giữa train và test không?
24. **[ĐỎ]** Từ điển token có bao nhiêu phần tử và được xây dựng từ tập nào?
25. **[VÀNG]** Phân bố độ dài chuỗi LaTeX trong train, validation và từng test set ra sao?
26. **[VÀNG]** Những loại cấu trúc nào xuất hiện nhiều và những loại nào hiếm trong tập huấn luyện?
27. **[VÀNG]** Dataset có bao nhiêu mẫu chứa tích phân, tích phân có cận dưới, cận trên và có cả hai cận?
28. **[VÀNG]** Các token hiếm hoặc ngoài từ điển được xử lý như thế nào?
29. **[BẪY]** Có thể kết luận một test set khó hơn test set khác chỉ từ ExpRate thấp hơn hay không? Vì sao?
30. **[BẪY]** Nếu demo ngoài đời thất bại nhưng test set tốt, đó là lỗi mô hình hay lỗi phân bố dữ liệu? Cần kiểm chứng thế nào?

---

## Nhóm 4 — Tiền xử lý ảnh, token hóa và nhãn LaTeX

**Mục tiêu nhóm:** Giải thích được mọi biến đổi từ ảnh thô và nhãn thô đến tensor và chuỗi token mà mô hình thật sự nhìn thấy.

31. **[ĐỎ]** Ảnh đầu vào được resize, padding và chuẩn hóa như thế nào?
32. **[ĐỎ]** Quy trình tiền xử lý lúc demo có giống hoàn toàn lúc train và test không?
33. **[ĐỎ]** Việc cố định chiều cao ảnh ảnh hưởng thế nào đến các ký hiệu nhỏ như cận tích phân hoặc chỉ số?
34. **[ĐỎ]** Token hóa LaTeX theo token, ký tự hay quy tắc nào?
35. **[ĐỎ]** Các token đặc biệt như BOS, EOS, PAD và UNK có vai trò gì?
36. **[VÀNG]** Nhãn LaTeX có được normalize trước khi train và đánh giá không?
37. **[VÀNG]** Hai biểu diễn tương đương như x^2 và x^{2} được xem là giống hay khác?
38. **[VÀNG]** Augmentation nào thật sự được áp dụng trong code, với xác suất và tham số bao nhiêu?
39. **[VÀNG]** Vì sao một số augmentation như lật ngang có thể không phù hợp với biểu thức toán học?
40. **[BẪY]** Nếu cận trên hoặc cận dưới biến mất sau resize, lỗi nên được quy cho dataset, preprocessing hay mô hình?

---

## Nhóm 5 — Kiến trúc tổng thể của hệ thống

**Mục tiêu nhóm:** Có thể mô tả đúng pipeline trong repo từ tensor ảnh đến chuỗi LaTeX, không lẫn với symbol graph của chuyên đề cũ.

41. **[ĐỎ]** Hãy mô tả toàn bộ pipeline hiện tại từ ảnh đầu vào đến chuỗi LaTeX đầu ra.
42. **[ĐỎ]** CNN/DenseNet trong repo có nhiệm vụ gì và đầu ra có kích thước như thế nào?
43. **[ĐỎ]** Node trong graph hiện tại là ký hiệu, bounding box, pixel hay một ô trên feature map?
44. **[ĐỎ]** Decoder nhận đầu vào gì từ encoder và sinh token theo cơ chế nào?
45. **[ĐỎ]** Mô hình hiện tại có thật sự phát hiện và phân đoạn từng ký hiệu hay không?
46. **[VÀNG]** Vì sao dùng DenseNet hoặc backbone hiện tại thay vì ResNet, ViT hay CNN đơn giản hơn?
47. **[VÀNG]** Feature map được flatten hoặc sắp xếp thành chuỗi/graph theo thứ tự nào?
48. **[VÀNG]** Thông tin padding và mask được truyền qua encoder và decoder như thế nào?
49. **[VÀNG]** Điểm nghẽn tính toán và bộ nhớ lớn nhất của pipeline nằm ở module nào?
50. **[BẪY]** Nếu không có symbol detector, có được gọi graph hiện tại là Symbol Layout Graph hay không?

---

## Nhóm 6 — Graph, GNN, GAT và thuật toán xây dựng đồ thị

**Mục tiêu nhóm:** Hiểu chính xác graph là gì, được tạo ra thế nào, message passing hoạt động ra sao và giới hạn của thiết kế hiện tại.

51. **[ĐỎ]** Graph trong repo được xây dựng từ feature map theo thuật toán cụ thể nào?
52. **[ĐỎ]** Mỗi node chứa những thành phần đặc trưng nào?
53. **[ĐỎ]** Hai node được nối cạnh khi thỏa điều kiện gì?
54. **[ĐỎ]** Vì sao chọn lưới 8 láng giềng thay vì 4 láng giềng, k-NN hoặc fully connected graph?
55. **[ĐỎ]** Graph là có hướng hay vô hướng; có self-loop hay không?
56. **[ĐỎ]** Một lớp GAT cập nhật node embedding theo công thức và trực giác nào?
57. **[ĐỎ]** Attention coefficient trong GAT biểu diễn điều gì và được chuẩn hóa trên tập láng giềng ra sao?
58. **[VÀNG]** Số head trong GAT ảnh hưởng thế nào đến năng lực biểu diễn và chi phí tính toán?
59. **[VÀNG]** Vì sao dùng GAT thay vì GCN, GraphSAGE hoặc graph transformer?
60. **[VÀNG]** Độ phức tạp của graph thay đổi thế nào theo chiều cao và chiều rộng feature map?
61. **[VÀNG]** Graph grid có mô hình hóa trực tiếp quan hệ superscript, subscript, above và below hay không?
62. **[VÀNG]** Thông tin tọa độ tuyệt đối và vị trí tương đối được đưa vào GAT như thế nào?
63. **[BẪY]** Nếu graph chỉ nối các ô lân cận, làm sao thông tin giữa hai vùng xa nhau có thể tương tác?
64. **[BẪY]** GAT có thật sự 'hiểu cấu trúc toán học' hay chỉ học tương quan cục bộ trên feature map?

---

## Nhóm 7 — M1–M5 và các thí nghiệm ablation

**Mục tiêu nhóm:** Giải thích được từng phiên bản, giả thuyết thay đổi, kết quả và kết luận khoa học mà không phóng đại.

65. **[ĐỎ]** M1 là baseline gì và mục đích tồn tại của M1 trong nghiên cứu là gì?
66. **[ĐỎ]** M2 thay đổi module nào so với M1 và giả thuyết ban đầu là gì?
67. **[ĐỎ]** M3 khác M2 chính xác ở vị trí đưa positional encoding như thế nào?
68. **[ĐỎ]** M4 bổ sung những thành phần nào liên quan đến tọa độ và relative position?
69. **[ĐỎ]** M5 tăng độ sâu hoặc độ rộng ở đâu so với M4?
70. **[ĐỎ]** Bảng cấu hình M1–M5 cần những cột nào để hội đồng nhìn là hiểu ngay?
71. **[ĐỎ]** Vì sao phải giữ nguyên dataset, preprocessing và quy trình đánh giá khi so sánh M1–M5?
72. **[ĐỎ]** Kết luận chính của toàn bộ chuỗi ablation M1–M5 là gì?
73. **[VÀNG]** Vì sao đưa positional encoding trước GAT có thể làm giảm hiệu quả?
74. **[VÀNG]** Vì sao đưa positional encoding sau GAT có thể phục hồi kết quả?
75. **[VÀNG]** Relative position bias trong M4 được định nghĩa theo bao nhiêu trạng thái hoặc khoảng cách?
76. **[VÀNG]** Vì sao M4 có thể giảm edit distance nhưng không tăng exact match?
77. **[VÀNG]** Vì sao mô hình sâu hơn như M5 không nhất thiết tốt hơn?
78. **[VÀNG]** Có thể tách ảnh hưởng của số layer, số head, d_model và dropout bằng thí nghiệm nào?
79. **[BẪY]** Nếu M3 chỉ tốt hơn M1 trên một test set nhưng thấp hơn ở hai test set khác, có được gọi M3 là tốt hơn không?
80. **[BẪY]** Nếu nhiều thay đổi được đưa vào M4 cùng lúc, làm sao biết thành phần nào thật sự tạo ra cải thiện?

---

## Nhóm 8 — Huấn luyện và hyperparameter

**Mục tiêu nhóm:** Nắm rõ cấu hình train thực tế, lý do chọn siêu tham số và cách nhận diện underfitting, overfitting, bất ổn tối ưu.

81. **[ĐỎ]** Optimizer, learning rate, scheduler và weight decay thực tế trong repo là gì?
82. **[ĐỎ]** Batch size train, validation và gradient accumulation được thiết lập ra sao?
83. **[ĐỎ]** Checkpoint tốt nhất được chọn theo metric nào và vì sao?
84. **[ĐỎ]** Số epoch tối đa, early stopping và tần suất đánh giá được cấu hình thế nào?
85. **[VÀNG]** Vì sao chọn optimizer hiện tại thay vì AdamW hoặc SGD?
86. **[VÀNG]** Mixed precision/FP16 được dùng như thế nào và có kiểm soát overflow hay không?
87. **[VÀNG]** Teacher forcing, label smoothing hoặc dropout có được sử dụng không?
88. **[VÀNG]** Random seed được cố định ở đâu và kết quả có lặp lại ổn định không?
89. **[BẪY]** Có thể giải thích mọi kết quả thấp bằng GPU yếu hoặc số epoch ít hay không?
90. **[BẪY]** Train loss tiếp tục giảm nhưng ExpRate không tăng có thể do những nguyên nhân nào?

---

## Nhóm 9 — Metric và quy trình đánh giá

**Mục tiêu nhóm:** Trả lời chính xác 'đo cái gì', 'đo trên đơn vị nào', 'cao hay thấp thì tốt' và 'metric được tính trong code ra sao'.

91. **[ĐỎ]** ExpRate/Exact Match được định nghĩa chính xác như thế nào?
92. **[ĐỎ]** Một mẫu sai một token có được tính là đúng một phần trong ExpRate hay không?
93. **[ĐỎ]** Edit distance trong repo được tính theo token hay theo ký tự?
94. **[ĐỎ]** Mean Edit Distance càng cao hay càng thấp càng tốt, và vì sao?
95. **[ĐỎ]** Các chỉ số ≤1 error và ≤2 errors được tính như thế nào?
96. **[ĐỎ]** Kết quả được đánh giá sau khi normalize LaTeX hay trên chuỗi thô?
97. **[VÀNG]** Symbol Accuracy khác ExpRate ở đâu và khi nào hai chỉ số có thể mâu thuẫn?
98. **[VÀNG]** Loss khác metric đánh giá như thế nào; loss thấp có đồng nghĩa ExpRate cao không?
99. **[VÀNG]** Syntax Error Rate phải được định nghĩa bằng parser, quy tắc ngoặc hay bộ kiểm tra nào?
100. **[VÀNG]** F1-score phù hợp với module nào và có phù hợp với recognizer không có detector hay không?
101. **[VÀNG]** Kết quả dùng greedy decoding hay beam search; beam size bao nhiêu?
102. **[BẪY]** x^2 và x^{2} nên được xem là cùng một dự đoán hay hai dự đoán khác nhau?
103. **[BẪY]** Có nên báo cáo kết quả tốt nhất của một lần chạy hay trung bình nhiều seed?
104. **[BẪY]** Một mô hình có edit distance tốt nhất nhưng ExpRate thấp hơn thì nên kết luận mô hình nào tốt hơn?

---

## Nhóm 10 — Phân tích kết quả giữa các dataset và mô hình

**Mục tiêu nhóm:** Biết đọc bảng kết quả, tách quan sát khỏi giả thuyết và đưa ra cách kiểm chứng nguyên nhân.

105. **[ĐỎ]** Mô hình nào có ExpRate cao nhất trên từng test set và trung bình toàn bộ?
106. **[ĐỎ]** Mô hình nào có Mean Edit Distance thấp nhất?
107. **[ĐỎ]** Kết quả chính mà luận văn có thể khẳng định chắc chắn từ bảng M1–M5 là gì?
108. **[ĐỎ]** Vì sao không được kết luận một mô hình tốt nhất chỉ dựa trên một metric?
109. **[ĐỎ]** Vì sao cùng một mô hình lại cao trên CROHME 2016 nhưng thấp trên 2014 hoặc 2019?
110. **[VÀNG]** Khác biệt độ dài biểu thức giữa các test set có thể ảnh hưởng kết quả thế nào?
111. **[VÀNG]** Phân bố loại cấu trúc như phân số, căn, chỉ số và tích phân có thể gây chênh lệch ra sao?
112. **[VÀNG]** Tần suất token hiếm hoặc ký hiệu dễ nhầm ảnh hưởng thế nào đến từng test set?
113. **[VÀNG]** Làm sao phân biệt cải thiện thật với dao động ngẫu nhiên do seed?
114. **[VÀNG]** Cần báo cáo độ lệch chuẩn hoặc khoảng tin cậy trong trường hợp nào?
115. **[VÀNG]** Nếu M4 giảm lỗi trung bình nhưng Exact Match không tăng, điều đó nói gì về phân bố lỗi?
116. **[VÀNG]** Có thể dùng histogram edit distance để giải thích kết quả như thế nào?
117. **[BẪY]** Có được nói test set 2019 khó hơn chỉ vì ExpRate thấp hơn không?
118. **[BẪY]** Có được quy toàn bộ suy giảm của M5 cho over-smoothing khi chưa có phép đo node similarity không?
119. **[BẪY]** Có được nói GAT tốt hơn CNN/Transformer nếu M1 vẫn cao hơn trên nhiều tập không?
120. **[BẪY]** Khi kết quả không ủng hộ giả thuyết ban đầu, nên trình bày thế nào để vẫn có giá trị khoa học?

---

## Nhóm 11 — Phân tích lỗi, demo, nét viết quằn và tích phân có cận

**Mục tiêu nhóm:** Chuẩn bị cho các câu hỏi thực chiến về demo thất bại, chữ viết xấu, lỗi encoder/decoder và dữ liệu ngoài phân bố.

121. **[ĐỎ]** Vì sao ảnh trong test set nhận dạng tốt nhưng ảnh người dùng tự vẽ trên demo có thể nhận dạng kém?
122. **[ĐỎ]** Tích phân thường nhận đúng nhưng tích phân có cận trên/dưới nhận sai có thể do những nguyên nhân nào?
123. **[VÀNG]** Làm sao xác định mô hình mất ký hiệu tích phân, mất dấu _/^ hay mất nội dung cận?
124. **[VÀNG]** Làm sao kiểm tra cận trên và cận dưới còn nhìn rõ sau preprocessing?
125. **[VÀNG]** Nét viết quá cong, nghiêng, dày, mảnh hoặc đứt đoạn ảnh hưởng đến feature extractor thế nào?
126. **[VÀNG]** Hai ký hiệu viết dính hoặc chồng lên nhau có thể gây lỗi ở đâu trong pipeline?
127. **[BẪY]** Có được trả lời 'dataset không có tích phân có cận' khi chưa thống kê hay không?
128. **[BẪY]** Có được đổ lỗi cho người dùng viết xấu khi preprocessing hoặc giao diện vẽ chưa tương thích với train data không?
129. **[BẪY]** Làm sao phân biệt lỗi thị giác của encoder với lỗi ngôn ngữ/cấu trúc của decoder?
130. **[BẪY]** Nếu ground truth có cấu trúc đúng nhưng output thiếu ngoặc, đó là lỗi cú pháp hay lỗi nhận dạng cấu trúc?
131. **[BẪY]** Cần xây dựng bộ test chuyên biệt nào để đánh giá tích phân có cận một cách thuyết phục?
132. **[BẪY]** Nếu thêm dữ liệu tích phân có cận mà kết quả chung giảm, em sẽ phân tích hiện tượng này thế nào?

---

## Nhóm 12 — Hạn chế, tính tái lập, ứng dụng và hướng phát triển

**Mục tiêu nhóm:** Thể hiện sự trung thực khoa học, khả năng tái lập và kế hoạch phát triển có căn cứ.

133. **[ĐỎ]** Ba hạn chế kỹ thuật quan trọng nhất của mô hình hiện tại là gì?
134. **[VÀNG]** Người khác cần những file, seed, checkpoint và lệnh nào để tái lập kết quả?
135. **[VÀNG]** Ứng dụng thực tế nào phù hợp với mức độ chính xác hiện tại và ứng dụng nào chưa phù hợp?
136. **[BẪY]** Hướng phát triển nào giải quyết trực tiếp hạn chế hiện tại thay vì chỉ nói chung chung 'thêm dữ liệu, thêm GPU'?
137. **[BẪY]** Nếu chuyển từ grid graph sang symbol-level graph, cần bổ sung những module và nhãn nào?
138. **[BẪY]** Nếu sử dụng grammar-constrained decoding, metric nào có thể tăng và rủi ro nào có thể xuất hiện?
139. **[BẪY]** Làm sao đánh giá thời gian suy luận, bộ nhớ và khả năng triển khai thời gian thực?
140. **[BẪY]** Nếu có thêm ba tháng nghiên cứu, thí nghiệm nào nên ưu tiên để tăng sức thuyết phục của luận văn?

---

## Checklist bằng chứng cần chuẩn bị kèm bộ câu hỏi

- Bảng thống kê số mẫu train/validation/test và nguồn script tạo ra số liệu.
- Bảng phân bố độ dài chuỗi, token và loại cấu trúc toán học.
- Bảng cấu hình M1–M5 theo cùng một mẫu: backbone, graph, GAT, positional encoding, relative position, số layer/head, d_model, dropout.
- Công thức và code của ExpRate, edit distance, ≤1, ≤2 và quy trình normalize.
- Bảng kết quả theo từng test set, nhiều metric và nhiều seed nếu có.
- Train/validation curve, tiêu chí chọn checkpoint và lệnh chạy đánh giá.
- Error analysis theo độ dài, loại cấu trúc và nhóm ký hiệu.
- Bộ test riêng cho tích phân có cận, kèm ảnh trước/sau preprocessing và so sánh token-by-token.
- Một sơ đồ pipeline đúng với code hiện tại, không dùng sơ đồ symbol-level graph nếu repo chưa có symbol detector.

## Nguyên tắc trả lời trước hội đồng

- Không nói **“dataset không có”** khi chưa có thống kê.
- Không nói **“mô hình tốt hơn”** nếu chỉ tốt hơn ở một tập hoặc một metric.
- Không dùng **loss thấp** để thay thế kết luận về ExpRate.
- Không gán nguyên nhân duy nhất khi chưa có error analysis.
- Không mô tả node/cạnh khác với code thực tế.
- Không đọc con số mà không biết nó đến từ checkpoint, log hoặc script nào.
- Khi chưa đủ bằng chứng, nói rõ: **đây là giả thuyết và thí nghiệm cần làm để kiểm chứng là gì**.
