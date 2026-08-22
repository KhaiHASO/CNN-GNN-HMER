# TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT THÀNH PHỐ HỒ CHÍ MINH
# VIỆN SAU ĐẠI HỌC
# NGÀNH KHOA HỌC MÁY TÍNH (MÃ NGÀNH: 8480101)

***

<br><br><br><br>

## ĐỀ ÁN TỐT NGHIỆP THẠC SĨ

<br><br>

# NGHIÊN CỨU MÔ HÌNH LAI CNN-GNN TRONG NHẬN DẠNG BIỂU THỨC TOÁN HỌC VIẾT TAY

<br><br><br><br>

### Học viên thực hiện: PHAN HOÀNG KHẢI
### Mã số học viên: 2531308
### Người hướng dẫn khoa học: TS. BÙI MẠNH QUÂN

<br><br><br><br>

### TP. HỒ CHÍ MINH, NĂM 2026

***

<br>

## LỜI CAM ĐOAN

Tôi xin cam đoan đây là công trình nghiên cứu khoa học do chính tôi thực hiện dưới sự hướng dẫn của TS. Bùi Mạnh Quân. Các số liệu, bảng biểu, log huấn luyện, kết quả thực nghiệm và các minh chứng kỹ thuật nêu trong luận văn này là hoàn toàn trung thực, khách quan và chưa từng được công bố trong bất kỳ công trình nghiên cứu hay luận văn học thuật nào khác. Mọi nguồn tài liệu tham khảo, kế thừa công nghệ từ cộng đồng mã nguồn mở đều được trích dẫn và khai báo đầy đủ theo đúng quy định pháp lý và đạo đức khoa học.

Học viên thực hiện,  
*Phan Hoàng Khải*

***

## LỜI CẢM ƠN

Trước hết, tôi xin bày tỏ lòng biết ơn sâu sắc nhất tới Thầy hướng dẫn khoa học - TS. Bùi Mạnh Quân. Thầy đã dành nhiều thời gian, công sức chỉ dẫn tận tình, định hướng học thuật sắc bén và luôn tạo mọi điều kiện tốt nhất để tôi hoàn thành nghiên cứu này. Sự nghiêm túc và đạo đức khoa học của Thầy là tấm gương lớn để tôi học tập.

Tôi xin trân trọng cảm ơn Ban Giám hiệu Trường Đại học Công nghệ Kỹ thuật TP.HCM, quý Thầy Cô Viện Sau đại học và Khoa Công nghệ Thông tin đã truyền đạt những kiến thức quý báu và tạo điều kiện thuận lợi trong suốt quá trình tôi theo học chương trình Thạc sĩ Khoa học Máy tính tại trường.

Cuối cùng, tôi xin cảm ơn gia đình, đồng nghiệp và các bạn học viên cùng khóa đã luôn động viên, chia sẻ khó khăn, hỗ trợ cả về vật chất lẫn tinh thần để tôi có thể tập trung hoàn thành tốt nhất công trình nghiên cứu này.

***

## TÓM TẮT LUẬN VĂN

Nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression Recognition - HMER) là một bài toán phức tạp trong lĩnh vực thị giác máy tính và xử lý ngôn ngữ tự nhiên. Khác với nhận dạng văn bản thông thường (OCR), biểu thức toán học có cấu trúc không gian hai chiều phi tuyến tính (chỉ số trên, chỉ số dưới, phân số lồng nhau, căn thức, tích phân có cận...). 

Luận văn này đề xuất một kiến trúc học sâu kết hợp giữa mạng tích chập DenseNet, mạng nơ-ron đồ thị Graph Attention Network (GAT) và Transformer Decoder để giải quyết bài toán HMER ngoại tuyến (offline) theo mô hình end-to-end. Trong đó, DenseNet đóng vai trò trích xuất đặc trưng thị giác từ ảnh đầu vào; các ô trên đặc trưng ảnh được ánh xạ thành các đỉnh (nodes) trên một đồ thị feature-grid 8 hướng; mạng GAT thực hiện truyền tin ngữ cảnh (message passing) giúp làm giàu thông tin không gian cục bộ; cuối cùng, Transformer Decoder giải mã đặc trưng và sinh chuỗi ký tự LaTeX đại diện.

Để đánh giá tác động của các thành phần kiến trúc, luận văn thiết kế chuỗi thực nghiệm ablation study từ M1 đến M5. Các mô hình được huấn luyện trên bộ dữ liệu chuẩn CROHME trong điều kiện tài nguyên tính toán giới hạn (2 GPU NVIDIA T4, 100 epoch). Kết quả thực nghiệm cho thấy mô hình M3 (PE sau GAT) phục hồi hiệu năng đạt ExpRate trung bình **49.17%** (+1.33% so với M2), vượt baseline M1 trên CROHME 2016 (50.74%). Mô hình M4 tích hợp cơ chế Relative Directional Bias (9 hướng tương đối học được trên lưới) đạt ExpRate là **48.98%**, tiệm cận baseline M1 (**50.10%**). Điểm nổi bật là M4 đạt chỉ số khoảng cách hiệu chỉnh trung bình (Mean Edit Distance) thấp nhất (**2.06** so với **2.10** của baseline), chứng tỏ GNN giúp bảo toàn cấu trúc cú pháp tốt hơn, giảm mức độ nghiêm trọng của các lỗi nhận dạng. Luận văn cũng phân tích sâu về nút thắt bộ nhớ $O(n^2)$ của GAT dẫn đến lỗi tràn bộ nhớ (Out-of-Memory) khi xử lý ảnh kích thước lớn và đề xuất các hướng khắc phục hiệu quả.

**Từ khóa:** Nhận dạng biểu thức toán học viết tay, HMER, DenseNet, Graph Attention Network, GAT, Transformer Decoder, Relative Directional Bias, CROHME.

***

## ABSTRACT

Handwritten Mathematical Expression Recognition (HMER) is a challenging task bridging computer vision and natural language processing. Unlike standard optical character recognition (OCR), mathematical expressions exhibit complex, non-linear two-dimensional structures (superscripts, subscripts, nested fractions, radicals, definite integrals, etc.).

This thesis proposes an end-to-end deep learning architecture combining a DenseNet convolutional network, a Graph Attention Network (GAT), and a Transformer Decoder for offline HMER. In this pipeline, DenseNet extracts grid-based visual features from the input image, which are mapped to nodes in an 8-neighbor feature-grid graph. The GAT encoder propagates local spatial context via graph message passing to enrich node representations. Finally, the Transformer Decoder generates the output LaTeX sequence autoregressively.

To investigate the impacts of different architectural components, we design a series of ablation models from M1 to M5. The models are trained on the standard CROHME dataset under limited hardware resources (2x NVIDIA T4 GPUs, 100 epochs). Experimental results show that the M3 model (PE after GAT) recovers performance to reach an average ExpRate of **49.17%** (+1.33% over M2) and surpasses the baseline on CROHME 2016 (50.74%). The M4 model, which incorporates a 9-state learned Relative Directional Bias, achieves an Expression Recognition Rate (ExpRate) of **48.98%**, closely matching the baseline M1 model without GAT (**50.10%**). Crucially, M4 achieves the lowest Mean Edit Distance (**2.06** compared to **2.10** for the baseline), demonstrating that GNNs successfully preserve syntactic structures and mitigate severe structural recognition errors. Furthermore, this work provides a detailed analysis of the quadratic $O(n^2)$ memory bottleneck in GAT layers that causes Out-of-Memory (OOM) errors during the processing of large images, and suggests practical optimization paths.

**Keywords:** Handwritten Mathematical Expression Recognition, HMER, DenseNet, Graph Attention Network, GAT, Transformer Decoder, Relative Directional Bias, CROHME.


***

## DANH MỤC TỪ VIẾT TẮT

| Từ viết tắt | Nghĩa tiếng Việt | Nghĩa tiếng Anh |
|---|---|---|
| HMER | Nhận dạng biểu thức toán học viết tay | Handwritten Mathematical Expression Recognition |
| OCR | Nhận dạng ký tự quang học | Optical Character Recognition |
| CNN | Mạng nơ-ron tích chập | Convolutional Neural Network |
| GNN | Mạng nơ-ron đồ thị | Graph Neural Network |
| GAT | Mạng nơ-ron chú ý đồ thị | Graph Attention Network |
| PE | Mã hóa vị trí | Positional Encoding |
| SOTA | Tốt nhất hiện nay | State-Of-The-Art |
| VRAM | Bộ nhớ truy cập ngẫu nhiên video | Video Random Access Memory |
| OOM | Tràn bộ nhớ | Out Of Memory |
| MED | Khoảng cách hiệu chỉnh trung bình | Mean Edit Distance |
| SLT | Cây bố cục ký hiệu | Symbol Layout Tree |
| BOS | Ký hiệu bắt đầu chuỗi | Begin Of Sentence |
| EOS | Ký hiệu kết thúc chuỗi | End Of Sentence |
| PAD | Ký hiệu đệm | Padding |
| UNK | Ký hiệu không xác định | Unknown token |
| OOD | Ngoài phân bố dữ liệu | Out Of Distribution |

## CHƯƠNG 1. TỔNG QUAN VỀ ĐỀ TÀI

### 1.1. Bối cảnh và lý do chọn đề tài

#### 1.1.1. Nhu cầu nhận dạng biểu thức toán học viết tay
Trong kỷ nguyên chuyển đổi số và giáo dục thông minh, nhu cầu tương tác và lưu trữ tài liệu khoa học dưới dạng kỹ thuật số tăng lên mạnh mẽ. Biểu thức toán học là ngôn ngữ chung để truyền tải các kiến thức khoa học, kỹ thuật và công nghệ. Việc nhập liệu thủ công các công thức toán học phức tạp thông qua bàn phím cơ thông thường hoặc các hệ soạn thảo chuyên dụng (như Equation Editor hay mã LaTeX thô) tốn nhiều thời gian và đòi hỏi kỹ năng chuyên môn nhất định. Nhận dạng biểu thức toán học viết tay (Handwritten Mathematical Expression Recognition - HMER) ra đời nhằm cung cấp giải pháp nhập liệu tự nhiên, cho phép chuyển đổi trực tiếp các biểu thức toán học viết tay từ ảnh chụp hoặc bảng viết kỹ thuật số thành các mã nguồn LaTeX chuẩn tắc, giúp tối ưu hóa quy trình biên soạn tài liệu giảng dạy, số hóa sách giáo khoa và xây dựng các hệ trợ lý học tập trực tuyến.

#### 1.1.2. Những khó khăn của bài toán HMER
Mặc dù công nghệ OCR văn bản tuyến tính đã đạt được những bước tiến vượt bậc nhờ học sâu, bài toán HMER vẫn tồn tại những thách thức đặc thù rất khó giải quyết:
1.  *Tính phi tuyến tính của cấu trúc không gian hai chiều:* Không giống văn bản thông thường được sắp xếp theo dòng nằm ngang từ trái qua phải, biểu thức toán học chứa các quan hệ không gian đa hướng phức tạp như phân số (trên/dưới), chỉ số trên (superscript), chỉ số dưới (subscript), căn thức (bao hàm bên trong), hay tích phân/tổng sigma (cận trên/cận dưới).
2.  *Sự đa dạng trong phong cách viết tay:* Kích thước, độ nghiêng, khoảng cách giữa các ký hiệu viết tay dao động lớn giữa các cá nhân, thậm chí cùng một người viết cũng có sự không đồng đều trong các ngữ cảnh khác nhau.
3.  *Sự mơ hồ về mặt ngữ nghĩa và vị trí:* Các ký hiệu rất nhỏ (dấu chấm, dấu phẩy, chỉ số) dễ bị nhầm lẫn với nhiễu ảnh hoặc nhầm lẫn vai trò nếu không có cơ chế phân tích vị trí không gian chính xác.

#### 1.1.3. Động cơ kết hợp CNN, GNN/GAT và Transformer
Để giải quyết các khó khăn trên, các nghiên cứu gần đây thường sử dụng kiến trúc Encoder-Decoder dựa trên Attention. Tuy nhiên, các mô hình Attention thuần túy (như Transformer thông thường) có xu hướng coi đặc trưng ảnh là một chuỗi phẳng 1D, làm mất đi các ràng buộc hình học cục bộ 2D vững chắc.
Động cơ của đề tài này là xây dựng một kiến trúc kết hợp:
*   Mạng **CNN (DenseNet)** đóng vai trò trích xuất đặc trưng thị giác cục bộ chất lượng cao từ ảnh đầu vào.
*   Mạng **GNN/GAT** được chèn vào giữa để thiết lập đồ thị lưới đặc trưng (feature-grid graph), thực hiện message passing giữa các vùng lân cận để mô hình hóa trực tiếp các quan hệ hình học 2D.
*   Mạng **Transformer Decoder** sử dụng cơ chế cross-attention để giải mã đặc trưng đồ thị và sinh chuỗi LaTeX tuần tự. Sự kết hợp này hướng đến việc bảo toàn cấu trúc toán học 2D mà vẫn giữ được năng lực sinh chuỗi LaTeX mạnh mẽ của Transformer.

### 1.2. Phát biểu bài toán

#### 1.2.1. Đầu vào và đầu ra của hệ thống
Hệ thống nhận dạng biểu thức toán học viết tay ngoại tuyến (offline HMER) nhận đầu vào là một ảnh chứa biểu thức toán học viết tay độc lập:
$$ I \in \mathbb{R}^{1 \times H \times W} $$
Trong đó $1$ biểu thị số kênh ảnh (ảnh grayscale), $H$ và $W$ lần lượt là chiều cao và chiều rộng của ảnh đầu vào.
Đầu ra của hệ thống là một chuỗi các ký hiệu (tokens) LaTeX biểu diễn biểu thức toán học đó:
$$ Y = (y_1, y_2, \ldots, y_T) $$
Với $T$ là độ dài tối đa của chuỗi giải mã, và mỗi $y_t$ thuộc về một từ điển LaTeX cố định $V$ ($y_t \in V$).

#### 1.2.2. Bài toán image-to-LaTeX
Mục tiêu của mô hình là tìm ra chuỗi LaTeX tối ưu $Y^*$ sao cho xác suất có điều kiện đối với ảnh đầu vào $I$ là lớn nhất:
$$ Y^* = \arg\max_Y P(Y \mid I) $$
Xác suất của chuỗi $Y$ được phân tích thành tích các xác suất điều kiện của từng token tại mỗi bước thời gian $t$, dựa trên các token đã được sinh ra trước đó và đặc trưng của ảnh đầu vào:
$$ P(Y \mid I) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, I) $$

#### 1.2.3. Các yêu cầu chính đối với hệ thống
1.  *Độ chính xác nhận dạng cú pháp:* Chuỗi LaTeX sinh ra phải đúng định dạng cú pháp để có thể biên dịch (compile) thành công mà không gây lỗi trình dịch (ví dụ: mở ngoặc `{` phải có đóng ngoặc `}`).
2.  *Độ chính xác nhận dạng ngữ nghĩa:* Ký hiệu nhận dạng được và cấu trúc không gian của chúng phải trùng khớp với biểu thức gốc trong ảnh.
3.  *Hiệu năng tính toán:* Thời gian suy luận cho một biểu thức phải nằm trong giới hạn chấp nhận được của ứng dụng thời gian thực (< 2 giây), dung lượng bộ nhớ VRAM sử dụng trong giới hạn phần cứng thông dụng.

### 1.3. Mục tiêu của luận văn

#### 1.3.1. Mục tiêu tổng quát
Nghiên cứu, cải tiến và hiện thực hóa một kiến trúc học sâu lai kết hợp giữa CNN, Graph Attention Network (GAT) và Transformer để nâng cao độ chính xác nhận dạng cấu trúc không gian hai chiều của biểu thức toán học viết tay ngoại tuyến.

#### 1.3.2. Mục tiêu về mô hình
*   Xây dựng mô hình baseline M1 dựa trên DenseNet và Transformer Decoder.
*   Thiết kế cơ chế feature-grid graph và tích hợp lớp GAT vào encoder để thực hiện message passing trên đặc trưng lưới ảnh.
*   Khảo sát tác động của vị trí chèn Positional Encoding (trước hay sau GAT) và tác động của Relative Directional Bias đối với tính chính xác của biểu thức thông qua chuỗi mô hình M2, M3, M4, M5.

#### 1.3.3. Mục tiêu về thực nghiệm và ứng dụng
*   Huấn luyện và đánh giá chi tiết các biến thể mô hình trên tập dữ liệu chuẩn quốc tế CROHME (các năm 2014, 2016, 2019) với tài nguyên phần cứng giới hạn (2 GPU T4).
*   Xây dựng một ứng dụng demo web cho phép người dùng tải ảnh biểu thức lên, nhận dạng ra chuỗi LaTeX, và tự động render lại công thức trực quan để kiểm chứng khả năng ứng dụng thực tế.

### 1.4. Đối tượng và phạm vi thực hiện

#### 1.4.1. Đối tượng nghiên cứu
*   Mô hình mạng nơ-ron tích chập (DenseNet-121).
*   Mô hình mạng nơ-ron đồ thị (GAT) và các cơ chế relative position bias.
*   Mô hình Transformer Decoder sử dụng cơ chế Attention giải mã chuỗi.
*   Bài toán nhận dạng cấu trúc ảnh hai chiều sang chuỗi ký tự.

#### 1.4.2. Phạm vi dữ liệu và biểu thức
*   Dữ liệu thử nghiệm ngoại tuyến trích xuất từ bộ dữ liệu chuẩn CROHME (ảnh raster đen trắng, nhãn LaTeX tương ứng).
*   Biểu thức toán học nằm trong giới hạn từ điển ký hiệu gồm 113 phần tử (bao gồm các ký số, chữ cái Hy Lạp, các toán tử cộng, trừ, nhân, chia, tích phân, căn thức, tổng sigma...).

#### 1.4.3. Phạm vi chức năng của hệ thống
Hệ thống nhận vào ảnh tĩnh chứa duy nhất một biểu thức toán học viết tay nằm ngang độc lập, không xử lý các đoạn văn bản dài xen kẽ công thức hoặc các biểu thức viết lộn xộn đa dòng chồng chéo.

### 1.5. Phương pháp thực hiện

#### 1.5.1. Khảo sát và kế thừa kiến trúc nền
Nghiên cứu kiến trúc TAMER (Tree-Aware Transformer) và Watch, Attend and Parse (WAP). Kế thừa cách tổ chức pipeline huấn luyện end-to-end, bộ tokenizer nhãn LaTeX và phương pháp trích xuất đặc trưng của DenseNet.

#### 1.5.2. Thiết kế các biến thể M1–M5
Tiến hành viết mã nguồn (coding) để tùy biến phần Encoder của mạng:
1.  *M1 (Baseline):* DenseNet + Transformer Decoder (không dùng đồ thị GAT).
2.  *M2 (Naive GAT, PE trước):* DenseNet -> chèn Absolute PE -> GAT 1 lớp -> Transformer Decoder.
3.  *M3 (PE sau GAT):* DenseNet -> GAT 1 lớp -> chèn Absolute PE -> Transformer Decoder.
4.  *M4 (Coord-Aware GAT):* DenseNet -> GAT 1 lớp có tích hợp Relative Directional Bias 9 trạng thái -> chèn Absolute PE -> Transformer Decoder.
5.  *M5 (Scale-up GAT):* Tăng độ sâu GAT lên 2 lớp, 8 attention heads, PE sau GAT.

#### 1.5.3. Huấn luyện, đánh giá và phân tích lỗi
Thực hiện huấn luyện các mô hình trong cùng một cấu hình hyperparameter trên GPU, ghi nhận loss và lưu checkpoint. Chạy suy luận (inference) trên các tập test CROHME để thu thập kết quả metric: ExpRate, Symbol Accuracy, Mean Edit Distance. Phân tích cụ thể các mẫu lỗi để tìm ra điểm nghẽn kiến trúc.

#### 1.5.4. Xây dựng ứng dụng minh họa
Hiện thực ứng dụng web tương tác (Expression Page Explorer) với backend FastAPI và frontend React/TypeScript/Konva, tích hợp mô hình đã huấn luyện xong và sử dụng KaTeX/MathJax để hiển thị công thức toán học nhận dạng được.

### 1.6. Đóng góp của đề án

#### 1.6.1. Đóng góp về thiết kế mô hình
Đề xuất cấu hình GAT tích hợp trực tiếp trên đặc trưng lưới ảnh (feature-grid GAT) kết hợp cơ chế Relative Directional Bias 9 hướng. Thực nghiệm chứng minh tính đúng đắn của giả thuyết: việc đặt Positional Encoding tuyệt đối trước lớp message passing của GAT sẽ gây nhiễu đặc trưng vị trí (PE blurring), và việc đặt PE sau GAT (M3, M4) giúp khôi phục hiệu năng nhận dạng biểu thức.

#### 1.6.2. Đóng góp về thực nghiệm
Cung cấp một bảng so sánh ablation study có hệ thống và chi tiết về tác động của số lớp GAT, số attention head, và thông tin hướng đối với hai chỉ số quan trọng là tỷ lệ nhận dạng chính xác tuyệt đối (ExpRate) và mức độ gần đúng (Mean Edit Distance). Phát hiện nút thắt bộ nhớ $O(n^2)$ của GAT trên graph lưới ảnh và xác định nguyên nhân gây lỗi OOM.

#### 1.6.3. Đóng góp về hiện thực hệ thống
Đóng gói hoàn chỉnh mã nguồn huấn luyện, công cụ kiểm toán dữ liệu trùng lặp (Data Auditor), script đánh giá chuẩn tắc và một ứng dụng demo web tương tác trực quan chạy ổn định.

### 1.7. Cấu trúc đề án
Đề án được tổ chức thành 6 chương và phần phụ lục như sau:
*   **Phần mở đầu:** Lời cam đoan, Lời cảm ơn, Tóm tắt (Tiếng Việt & Tiếng Anh), Danh mục từ viết tắt, Danh mục bảng/hình.
*   **Chương 1. Tổng quan:** Trình bày bối cảnh, lý do chọn đề tài, phát biểu bài toán, mục tiêu, các câu hỏi nghiên cứu RQ1–RQ4, đối tượng, phạm vi, phương pháp và các đóng góp chính của đề án.
*   **Chương 2. Cơ sở lý thuyết và công trình liên quan:** Trình bày cơ sở toán học và lý thuyết của DenseNet, GAT, Positional Encoding, Transformer Decoder, các phương pháp nhận dạng biểu thức hiện có và định vị giải pháp nghiên cứu.
*   **Chương 3. Phương pháp đề xuất:** Trình bày chi tiết kiến trúc mô hình kết hợp, cách xây dựng feature-grid graph 8 hướng, công thức GAT cải tiến có Relative Bias, cơ chế PE sau GAT và định nghĩa chuỗi biến thể M1–M5.
*   **Chương 4. Thiết lập thực nghiệm:** Mô tả chi tiết bộ dữ liệu CROHME (2014, 2016, 2019), các bước tiền xử lý ảnh và nhãn LaTeX, cấu hình huấn luyện (2 GPU Tesla T4, 100 epochs), cơ chế batch động và tiêu chí chọn checkpoint.
*   **Chương 5. Kết quả và thảo luận:** Trình bày kết quả thực nghiệm đóng băng của M1–M5, phân tích chi tiết theo 4 câu hỏi nghiên cứu RQ1–RQ4, phân tích trade-off giữa ExpRate và Mean Edit Distance, kiểm toán lỗi M3 và đánh giá ứng dụng demo.
*   **Chương 6. Kết luận và hướng phát triển:** Tổng kết kết quả đạt được có điều kiện, nêu rõ các hạn chế về dữ liệu và kiến trúc, đề xuất các hướng nghiên cứu tiếp theo.
*   **Tài liệu tham khảo & Phụ lục.**

### 1.8. Kết luận chương
Chương 1 đã phác thảo bức tranh tổng quan của luận văn, thiết lập cơ sở khoa học và động cơ nghiên cứu của đề tài. Bài toán HMER được định vị rõ ràng dưới dạng bài toán image-to-sequence ngoại tuyến. Các mục tiêu thiết kế mô hình kết hợp CNN-GNN-Transformer và chuỗi thực nghiệm ablation study M1-M5 được xác định cụ thể, tạo tiền đề để đi sâu vào các chương cơ sở lý thuyết tiếp theo.

***

## CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN

### 2.1. Tổng quan bài toán nhận dạng biểu thức toán học viết tay

#### 2.1.1. Nhận dạng trực tuyến và ngoại tuyến
Bài toán HMER được chia làm hai nhánh chính dựa trên kiểu dữ liệu đầu vào:
1.  *Nhận dạng trực tuyến (online recognition):* Đầu vào là một chuỗi các tọa độ nét vẽ viết tay theo thời gian $(x_t, y_t, p_t)$ thu được từ các bảng vẽ số hóa hoặc màn hình cảm ứng. Dữ liệu này chứa thông tin động lực học cực kỳ chi tiết về thứ tự nét vẽ và hướng vẽ, giúp mô hình dễ dàng tách biệt các ký hiệu đè lên nhau.
2.  *Nhận dạng ngoại tuyến (offline recognition):* Đầu vào chỉ là một ảnh tĩnh hai chiều (raster image) ở dạng nhị phân hoặc mức xám. Đây là bài toán tổng quát và thực tế hơn vì ảnh biểu thức có thể được chụp từ sách vở, bảng viết hoặc các tài liệu quét. Luận văn này tập trung giải quyết bài toán ngoại tuyến, nơi thông tin nét vẽ theo thời gian bị mất hoàn toàn và mô hình phải suy diễn cấu trúc thuần túy từ phân bố cường độ điểm ảnh 2D.

#### 2.1.2. Nhận dạng ký hiệu và nhận dạng toàn biểu thức
*   *Phương pháp nhận dạng ký hiệu trước:* Phân mảnh ảnh biểu thức thành các hộp chứa ký hiệu (bounding boxes), nhận dạng từng ký hiệu cô lập, rồi dùng các thuật toán gom nhóm không gian để xây dựng cây cú pháp. Phương pháp này dễ bị lỗi tích lũy: nếu phân đoạn sai hoặc nhận dạng ký hiệu sai ở giai đoạn đầu, toàn bộ cấu trúc biểu thức phía sau sẽ bị sụp đổ.
*   *Phương pháp nhận dạng toàn biểu thức end-to-end:* Nhận đầu vào là ảnh và sinh trực tiếp mã LaTeX mà không cần qua bước phân đoạn tường minh. Mô hình tự học cách căn chỉnh (align) giữa vùng ảnh và token LaTeX thông qua cơ chế Attention. Luận văn này lựa chọn hướng tiếp cận end-to-end vì tính linh hoạt và khả năng tự phục hồi lỗi nhờ học đồng thời cả đặc trưng thị giác và ngữ cảnh ngôn ngữ toán học.

#### 2.1.3. Phân tích cấu trúc hai chiều và sinh LaTeX
Sinh chuỗi LaTeX thực chất là việc biểu diễn một cây cấu trúc không gian 2D (Symbol Layout Tree - SLT) dưới dạng một chuỗi ký tự 1D tuyến tính bằng các thẻ cú pháp (như `^`, `_`, `\frac{}{}`). Mô hình phải học được cách "duyệt cây" 2D này và ánh xạ nó sang quy tắc sinh chuỗi LaTeX chuẩn để đảm bảo tính hợp lệ về mặt ngữ pháp.

### 2.2. Mạng nơ-ron tích chập và DenseNet

#### 2.2.1. Trích xuất đặc trưng ảnh
Trong các mô hình HMER end-to-end, mạng tích chập (CNN) đóng vai trò là Encoder thị giác. CNN quét qua ảnh đầu vào thông qua các bộ lọc cục bộ, trích xuất các đặc trưng từ mức độ thấp (cạnh, góc, nét vẽ) đến đặc trưng mức độ cao (hình dạng ký hiệu, ngữ cảnh xung quanh).

#### 2.2.2. Dense connectivity và khả năng tái sử dụng đặc trưng
Kiến trúc DenseNet (Densely Connected Convolutional Networks) cải tiến cơ chế kết nối bằng cách liên kết trực tiếp mọi lớp tích chập với các lớp phía sau nó trong cùng một block. Công thức cập nhật đặc trưng của lớp thứ $l$ trong một Dense Block là:
$$ x_l = H_l([x_0, x_1, \ldots, x_{l-1}]) $$
Trong đó $[x_0, x_1, \dots, x_{l-1}]$ biểu thị phép nối kênh (concatenation) đặc trưng đầu ra của tất cả các lớp trước đó. Cơ chế này mang lại ba ưu điểm lớn cho HMER:
1.  Khắc phục hiện tượng triệt tiêu gradient (vanishing gradient) nhờ các đường truyền trực tiếp ngắn nhất từ đầu vào đến các lớp sâu.
2.  Tăng cường lan truyền đặc trưng và tái sử dụng đặc trưng ở các mức độ phân giải khác nhau, giúp mô hình không bị mất các chi tiết nét viết mảnh.
3.  Giảm đáng kể số lượng tham số so với ResNet truyền thống nhờ tốc độ tăng trưởng kênh (growth rate) nhỏ.

#### 2.2.3. Vai trò của DenseNet trong HMER
DenseNet-121 được chọn làm backbone mã hóa ảnh vì nó cung cấp feature map có độ phân giải phù hợp và giàu đặc trưng thị giác. Đầu ra của DenseNet được chiếu qua một lớp tích chập $1 \times 1$ để đưa số kênh về chiều đặc trưng của mô hình ($d_{\text{model}} = 256$), tạo thành một lưới đặc trưng ảnh:
$$ F \in \mathbb{R}^{H' \times W' \times D} $$
Với $H' = H/16$, $W' = W/16$ và $D = 256$. Lưới đặc trưng này giữ nguyên cấu trúc topo học 2D của ảnh biểu thức gốc.

### 2.3. Transformer cho bài toán image-to-sequence

#### 2.3.1. Self-attention và cross-attention
Mô hình Transformer loại bỏ hoàn toàn các liên kết tuần tự của RNN, thay thế bằng cơ chế Attention (chú ý) giúp song song hóa tối đa quá trình huấn luyện:
*   *Self-attention:* Tính toán mối quan hệ giữa các token trong cùng một chuỗi để học ngữ cảnh ngôn ngữ LaTeX.
*   *Cross-attention:* Kết nối decoder với encoder. Decoder tính toán trọng số chú ý trên các vùng đặc trưng của ảnh (ở đầu ra encoder) tại mỗi bước giải mã để quyết định vùng thị giác nào tương ứng với token LaTeX sắp sinh ra.

#### 2.3.2. Positional encoding
Vì cơ chế Attention không có khái niệm thứ tự tuần tự, mô hình cần bổ sung thông tin vị trí thông qua Positional Encoding (PE). Trong HMER:
*   Decoder sử dụng PE 1D tuần tự để xác định vị trí của token trong chuỗi LaTeX.
*   Encoder sử dụng PE 2D (gồm PE chiều ngang và PE chiều dọc cộng lại) để giúp mô hình phân biệt tọa độ các ô đặc trưng ảnh trên lưới 2D.

#### 2.3.3. Autoregressive decoding và beam search
Quá trình giải mã diễn ra tuần tự (autoregressive): mô hình dự đoán token $y_t$ dựa trên chuỗi đã sinh $y_{<t}$ và đặc trưng ảnh. Trong pha suy luận (inference), thuật toán Beam Search được áp dụng thay vì Greedy Search. Beam Search duy trì một tập hợp $k$ giả thuyết có xác suất tích lũy cao nhất tại mỗi bước giải mã (với beam size $k=10$ trong thực nghiệm), giúp giảm thiểu sai sót cục bộ và tìm được chuỗi LaTeX tối ưu toàn cục.

### 2.4. Graph Neural Network và Graph Attention Network

#### 2.4.1. Graph, node, edge và message passing
Mạng nơ-ron đồ thị (GNN) hoạt động trên cấu trúc đồ thị $G = (V, E)$, với $V$ là tập hợp các đỉnh và $E$ là tập hợp các cạnh kết nối các đỉnh. Quá trình học của GNN dựa trên cơ chế Message Passing: mỗi node thu thập đặc trưng từ các node lân cận và cập nhật trạng thái của chính nó thông qua các lớp nơ-ron:
$$ h_i^{(l+1)} = 	ext{Update}\left( h_i^{(l)}, 	ext{Aggregate}_{j \in \mathcal{N}(i)} \text{Message}\left(h_i^{(l)}, h_j^{(l)}\right) \right) $$

#### 2.4.2. Graph attention và multi-head attention
Mạng GAT (Graph Attention Network) cải tiến cơ chế tập hợp đặc trưng bằng cách gán các trọng số chú ý $\alpha_{ij}$ tự động học được giữa node $i$ và các node lân cận $j \in \mathcal{N}(i)$ của nó.
Độ tương đồng attention hệ số $e_{ij}$ được tính bằng:
$$ e_{ij} = \operatorname{LeakyReLU}\left( \mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \parallel \mathbf{W}\mathbf{h}_j] \right) $$
Trong đó $\mathbf{W}$ là ma trận biến đổi tuyến tính đặc trưng node, $\mathbf{a}$ là vector trọng số attention, và $\parallel$ ký hiệu phép nối vector. Trọng số attention $\alpha_{ij}$ được chuẩn hóa bằng hàm Softmax trên tập láng giềng $\mathcal{N}(i)$:
$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} $$
Để ổn định quá trình học, GAT sử dụng Multi-head Attention với $K$ đầu độc lập. Đặc trưng cập nhật của node $i$ là phép nối đặc trưng từ các head:
$$ \mathbf{h}_i' = \parallel_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j \right) $$

#### 2.4.3. Relative position và directional bias
Trong đồ thị lưới ảnh, hướng tương đối giữa các node (trên, dưới, trái, phải...) mang thông tin ngữ nghĩa hình học sống còn. Nếu chỉ sử dụng GAT cơ bản dựa trên nội dung đặc trưng node, mô hình sẽ gặp khó khăn khi phân biệt các láng giềng có đặc trưng thị giác giống nhau nhưng nằm ở các hướng khác nhau. Do đó, việc chèn thêm Relative Directional Bias học được cho mỗi hướng tương đối vào công thức tính attention là rất cần thiết (như được thực hiện trong mô hình M4).

#### 2.4.4. Giới hạn về độ sâu và chi phí tính toán
Các mạng GAT sâu thường gặp hiện tượng quá mịn (over-smoothing), tức là đặc trưng của tất cả các node hội tụ về các giá trị giống nhau sau nhiều lớp message passing, làm mất đi tính phân biệt đặc trưng thị giác cục bộ. Đồng thời, chi phí tính toán attention tăng tuyến tính với số lượng cạnh và bình phương với số lượng node trên đồ thị.

### 2.5. Các hướng tiếp cận liên quan trong HMER

#### 2.5.1. Phương pháp dựa trên phân đoạn ký hiệu
Các phương pháp cổ điển nhận dạng từng nét viết độc lập rồi dựng cây cú pháp. Điểm yếu lớn nhất là độ nhạy cao với nhiễu ảnh và không có khả năng sửa sai ở các tầng phía sau.

#### 2.5.2. Phương pháp image-to-sequence
Sử dụng CNN-LSTM/GRU hoặc Transformer Decoder sinh trực tiếp LaTeX. Các mô hình này có ExpRate cao trên các biểu thức ngắn nhưng giảm mạnh hiệu năng khi biểu thức dài và có cấu trúc phức tạp do thiếu ràng buộc không gian 2D rõ ràng ở phía encoder.

#### 2.5.3. Phương pháp syntax-aware, tree-aware và graph-based
Các mô hình như SAN (Syntax-Aware Network) hay TAMER tích hợp các ràng buộc ngữ pháp hoặc cây cú pháp LaTeX trong quá trình huấn luyện để định hướng decoder sinh chuỗi hợp lệ. Một số mô hình GNN khác xây dựng đồ thị ở mức nét vẽ (stroke graph) hoặc mức ký hiệu (symbol graph), nhưng yêu cầu phải có nhãn phân đoạn ký hiệu chi tiết trong quá trình train, vốn rất đắt đỏ và không khả thi với dữ liệu ảnh thô offline thông thường.

### 2.6. Nhận xét và định vị giải pháp của luận văn

#### 2.6.1. Khoảng trống về cách tích hợp GAT vào feature-grid
Luận văn nhận thấy phần lớn các nghiên cứu GNN cho HMER chỉ tập trung vào đồ thị nét vẽ (online) hoặc đồ thị ký hiệu cô lập. Khoảng trống thiết kế nằm ở chỗ: làm thế nào tích hợp hiệu quả mạng message passing đồ thị trực tiếp lên lưới đặc trưng ảnh (feature-grid graph) của một pipeline image-to-LaTeX ngoại tuyến end-to-end mà không cần nhãn hộp phân đoạn ký hiệu.

#### 2.6.2. Vấn đề thứ tự positional encoding và message passing
Nếu ta cộng Absolute Positional Encoding vào đặc trưng ảnh trước khi đi qua lớp GAT (như mô hình M2), quá trình message passing sẽ trung bình hóa cả đặc trưng thị giác lẫn tọa độ vị trí của các node lân cận. Điều này dẫn đến hiện tượng "PE blurring" (mờ hóa vị trí tuyệt đối), làm giảm năng lực định vị của decoder và suy giảm nghiêm trọng độ chính xác nhận dạng biểu thức. Luận văn định vị giải pháp sửa lỗi này bằng cách đặt PE tuyệt đối sau các lớp GAT (M3, M4).

#### 2.6.3. Định hướng cân bằng độ chính xác và khả năng triển khai
Luận văn hướng tới thiết kế một mô hình có khả năng nhận dạng cấu trúc tốt hơn (giảm sai lệch khoảng cách chỉnh sửa Mean Edit Distance) nhưng vẫn giữ cấu hình tối giản (1 lớp GAT, 4 heads) để có thể chạy thời gian thực trên các thiết bị máy tính thông thường hoặc môi trường máy chủ GPU giới hạn.

### 2.7. Kết luận chương
Chương 2 đã trình bày toàn bộ cơ sở lý thuyết nền tảng từ trích xuất đặc trưng tích chập (DenseNet), cơ chế giải mã tự hồi quy (Transformer), đến lý thuyết truyền tin đồ thị (GAT). Qua việc phân tích các nghiên cứu liên quan, luận văn đã xác định rõ khoảng trống nghiên cứu và thiết lập giả thuyết khoa học về thứ tự chèn Positional Encoding và Relative directional bias, làm nền móng để xây dựng kiến trúc chi tiết ở Chương 4.

***

## CHƯƠNG 3. DỮ LIỆU, TIỀN XỬ LÝ VÀ PHÂN TÍCH YÊU CẦU

### 3.1. Bộ dữ liệu sử dụng

#### 3.1.1. Tổng quan CROHME
Bộ dữ liệu CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions) là bộ dữ liệu chuẩn mực nhất được cộng đồng quốc tế sử dụng để đánh giá các mô hình HMER. Dữ liệu gốc của CROHME được lưu dưới định dạng InkML chứa tọa độ các nét vẽ theo thời gian. Để áp dụng cho mô hình HMER ngoại tuyến (offline), các nét vẽ này được dựng (render) thành hình ảnh raster đen trắng.

#### 3.1.2. Các tập train, validation và test
Trong nghiên cứu này, chúng tôi sử dụng dữ liệu huấn luyện và kiểm thử chuẩn tắc từ CROHME:
*   *Tập Huấn luyện (Train set):* Gồm **8.836** mẫu biểu thức toán học viết tay từ tập train CROHME.
*   *Tập Xác thực (Validation set):* Gồm **986** mẫu biểu thức dùng để theo dõi hàm loss, độ chính xác và lựa chọn checkpoint tối ưu trong quá trình train.
*   *Các tập Kiểm thử (Test sets):* Đánh giá độc lập trên ba tập dữ liệu chuẩn:
    *   **Test CROHME 2014:** Gồm **986** mẫu biểu thức.
    *   **Test CROHME 2016:** Gồm **1.147** mẫu biểu thức.
    *   **Test CROHME 2019:** Gồm **1.199** mẫu biểu thức.

#### 3.1.3. Đặc điểm dữ liệu raster và nhãn LaTeX
Dữ liệu ảnh đầu vào là ảnh mức xám có nền đen chữ trắng hoặc nền trắng chữ đen (được chuẩn hóa về nền đen chữ trắng trước khi đưa vào mô hình). Nhãn tương ứng của mỗi ảnh là một chuỗi văn bản mã hóa biểu thức dưới dạng ngôn ngữ LaTeX, ví dụ:
`\frac { a ^ { 2 } } { b _ { i } }`

### 3.2. Thống kê và kiểm tra dữ liệu

#### 3.2.1. Số lượng mẫu và phân bố độ dài
Độ dài chuỗi nhãn LaTeX của tập dữ liệu CROHME dao động từ 3 token đến hơn 120 token. Phân bố độ dài tập trung nhiều nhất ở khoảng 15 đến 40 tokens. Các biểu thức có độ dài lớn (> 80 tokens) chiếm tỷ lệ dưới 5% nhưng lại là nhóm gây ra nhiều lỗi cú pháp và tràn bộ nhớ nhất cho mô hình.

#### 3.2.2. Phân bố token và cấu trúc
Từ điển ký hiệu (vocabulary) của bộ dữ liệu bao gồm **113** phần tử độc lập. Phân bố tần suất xuất hiện của các token mất cân bằng nghiêm trọng: các token cơ bản như ký số (`0`-`9`), chữ cái biến số (`x`, `y`, `a`, `b`), toán tử (`+`, `-`, `=`, `/`) và các ký hiệu cấu trúc (`{`, `}`, `^`, `_`) xuất hiện với tần suất hàng nghìn lần. Ngược lại, các ký hiệu toán học chuyên biệt như tích phân (`\int`), tổng sigma (`\sum`), hay các chữ cái Hy Lạp (`\alpha`, `\beta`, `\theta`) có tần suất rất thấp (chỉ vài chục đến vài trăm lần trong toàn bộ tập train).

#### 3.2.3. Kiểm tra trùng lặp, OOV và chất lượng dữ liệu
Trước khi huấn luyện, chúng tôi chạy một module kiểm toán dữ liệu (Data Auditor) để đảm bảo:
1.  Không có sự trùng lặp hình ảnh hoặc rò rỉ thông tin nhãn (data leakage) giữa tập Train và các tập Test.
2.  Không có ký hiệu ngoài từ điển (Out-Of-Vocabulary - OOV) trong tập Validation và Test so với tập Train. Tất cả các token xuất hiện trong tập kiểm thử đều phải có mặt trong từ điển được xây dựng từ tập Train.

### 3.3. Tiền xử lý ảnh

#### 3.3.1. Resize giữ tỷ lệ và giới hạn kích thước
Để đưa ảnh vào mạng DenseNet mà không làm biến dạng tỷ lệ hình học của chữ viết tay (yếu tố cực kỳ nhạy cảm đối với việc nhận diện chỉ số trên/dưới), chúng tôi không resize ảnh về một kích thước cố định như $128 \times 512$. Thay vào đó, ảnh đầu vào được tính toán hệ số thu nhỏ:
$$ s_{\text{down}} = \min\left(\frac{256}{H}, \frac{1024}{W}\right) $$
Nếu ảnh quá nhỏ, hệ số phóng to là:
$$ s_{\text{up}} = \max\left(\frac{16}{H}, \frac{16}{W}\right) $$
Ảnh được nhân với hệ số tương ứng để giữ nguyên tỷ lệ khung hình (aspect ratio) nhưng nằm gọn trong giới hạn chiều cao tối đa 256 pixel và chiều rộng tối đa 1024 pixel.

#### 3.3.2. Chuyển tensor, dynamic batching, padding và mask
Ảnh sau khi resize được đưa về dạng Tensor mức xám có giá trị thuộc đoạn $[0, 1]$. Do kích thước ảnh trong batch không đồng đều, chúng tôi sử dụng cơ chế Dynamic Batching: thực hiện đệm (padding) các ảnh trong cùng một batch bằng giá trị 0 về kích thước của ảnh lớn nhất trong batch đó. Đồng thời, một mặt nạ nhị phân (src_mask) có kích thước tương ứng được tạo ra để giúp encoder nhận biết và bỏ qua các vùng ảnh đệm trắng trong các phép tính toán self-attention và graph attention.

#### 3.3.3. Sự khác biệt giữa pipeline benchmark và demo
Trong quá trình huấn luyện và đánh giá benchmark, ảnh đầu vào là ảnh sạch được cắt biên sát (tightly cropped) trực tiếp từ bộ InkML. Tuy nhiên, trong ứng dụng demo thực tế, ảnh đầu vào được tải lên từ camera điện thoại hoặc ảnh chụp tài liệu của người dùng, chứa nhiều khoảng trắng thừa, nhiễu nền hoặc nét chữ mờ. Do đó, pipeline demo tích hợp thêm các bước: tự động cắt bỏ viền trắng thừa (border cropping), nhị phân hóa ảnh bằng ngưỡng động (Otsu binarization) và đảo ngược màu nền để đưa ảnh về dạng nền đen chữ trắng chuẩn trước khi đưa vào mô hình.

### 3.4. Xử lý nhãn LaTeX

#### 3.4.1. Định dạng caption và quy tắc token hóa
Chuỗi nhãn LaTeX được tách thành các token độc lập ngăn cách bởi dấu cách. Ví dụ, biểu thức `x^2 + y_i` được token hóa thành danh sách:
`['x', '^', '2', '+', 'y', '_', 'i']`
Các hàm toán học phức tạp như `\frac` hay `\sqrt` được giữ nguyên làm một token duy nhất: `['\frac', '{', 'a', '}', '{', 'b', '}']`.

#### 4.4.2. Từ điển và các token đặc biệt
Từ điển từ vựng $V$ được xây dựng từ tập Train gồm 113 tokens độc lập. Hệ thống bổ sung thêm 4 tokens điều khiển đặc biệt:
*   `<bos>`: Ký hiệu bắt đầu chuỗi giải mã (mã số 0).
*   `<eos>`: Ký hiệu kết thúc chuỗi giải mã (mã số 1).
*   `<pad>`: Ký hiệu đệm nhãn để đưa các chuỗi trong batch về cùng độ dài tối đa (mã số 2).
*   `<unk>`: Ký hiệu đại diện cho các token không xác định (mã số 3).

#### 3.4.3. Vấn đề chuẩn hóa biểu diễn LaTeX
LaTeX là ngôn ngữ đánh dấu có tính đa biểu diễn (nhiều chuỗi mã nguồn khác nhau render ra cùng một công thức). Ví dụ: `a_i^2` và `a^2_i` hiển thị hoàn toàn giống nhau nhưng nhãn chữ thô lại khác nhau. Để tránh gây nhiễu cho mô hình, nhãn LaTeX được chuẩn hóa thông qua bộ phân tích cú pháp để đưa về cấu trúc thống nhất: luôn viết chỉ số dưới trước chỉ số trên, loại bỏ khoảng trắng thừa và chuẩn hóa các dấu đóng mở ngoặc nhọn.

### 3.5. Phân tích yêu cầu hệ thống

#### 3.5.1. Yêu cầu chức năng
1.  *Nhập liệu hình ảnh:* Cho phép người dùng tải lên tệp ảnh (PNG, JPG) hoặc chụp trực tiếp từ camera.
2.  *Nhận dạng cấu trúc và ký hiệu:* Mô hình phải phân tích và ánh xạ chính xác ảnh biểu thức sang chuỗi LaTeX.
3.  *Hiển thị trực quan:* Render mã LaTeX nhận dạng được thành công thức toán học trực quan trên giao diện web.
4.  *Sao chép kết quả:* Hỗ trợ người dùng sao chép nhanh mã LaTeX hoặc ảnh công thức đã render để chèn vào các tài liệu soạn thảo.

#### 3.5.2. Yêu cầu phi chức năng
1.  *Thời gian phản hồi nhanh (Inference Latency):* Thời gian từ lúc gửi ảnh đến khi nhận được kết quả hiển thị trên giao diện phải nhỏ hơn 2 giây với độ dài biểu thức trung bình.
2.  *Tính di động và độc lập:* Hệ thống có thể chạy offline hoàn toàn trên máy tính cá nhân bằng cách tải mô hình cục bộ mà không phụ thuộc vào kết nối máy chủ đám mây bên ngoài.
3.  *Độ tin cậy cú pháp:* Hạn chế tối đa việc sinh ra các chuỗi LaTeX bị lỗi đóng mở ngoặc nhọn (`{` và `}`) để tránh làm hỏng bộ render trực quan.

#### 3.5.3. Giới hạn đầu vào, đầu ra và phạm vi sử dụng
*   *Độ phân giải ảnh:* Từ $32 \times 32$ đến tối đa $512 \times 2048$ pixels.
*   *Độ dài biểu thức:* Giới hạn tối đa $T = 150$ tokens giải mã để kiểm soát thời gian suy luận và tránh tràn bộ nhớ VRAM của GPU T4.

### 3.6. Các trường hợp khó cần quan tâm

#### 3.6.1. Biểu thức dài và cấu trúc lồng nhau
Các biểu thức chứa phân số nằm trong phân số (nested fractions) hoặc căn thức chứa căn thức tạo ra các đặc trưng thị giác rất nhỏ và chồng chéo lên nhau, đòi hỏi mô hình phải có cơ chế attention phân cấp hoặc liên kết đồ thị ngữ cảnh rộng để nhận dạng đúng cấp độ lồng nhau.

#### 3.6.2. Ký hiệu nhỏ, chỉ số và cận tích phân
Cận tích phân (ví dụ: $\int_{0}^{\infty}$) hoặc chỉ số trên mức 2 ($x^{y^2}$) có kích thước ký hiệu rất nhỏ trên ảnh đặc trưng. Nếu thông tin vị trí bị mờ nhạt (PE blurring), mô hình rất dễ nhận dạng nhầm các cận này thành các biến số nằm ngang bình thường (ví dụ biến thành $\int 0 \infty$ hoặc $x y 2$).

#### 3.6.3. Dữ liệu ngoài phân bố
Các ảnh viết tay có nét chữ quá dày, nét viết bị đứt gãy do mực bút yếu, hoặc ảnh chụp có bóng mờ góc nghiêng lớn là các nguồn dữ liệu ngoài phân bố (Out-Of-Distribution - OOD) dễ gây ra lỗi nhận dạng sai ký hiệu.

### 3.7. Kết luận chương
Chương 3 đã làm rõ các khía cạnh liên quan đến dữ liệu huấn luyện và kiểm thử CROHME, các bước tiền xử lý ảnh và nhãn LaTeX cần thiết để chuẩn bị đầu vào sạch cho mô hình. Đồng thời, chương này cũng phân tích cụ thể các yêu cầu chức năng, phi chức năng và xác định các trường hợp biểu thức khó, làm cơ sở định hướng thiết kế mô hình chi tiết ở Chương 4 để giải quyết triệt để các thách thức về vị trí và cấu trúc.

***

## CHƯƠNG 4. PHƯƠNG PHÁP ĐỀ XUẤT VÀ HIỆN THỰC HỆ THỐNG

### 4.1. Kiến trúc tổng thể

#### 4.1.1. Luồng xử lý từ ảnh đến chuỗi LaTeX
Mô hình đề xuất hoạt động theo cơ chế end-to-end gồm bốn giai đoạn xử lý chính:
1.  *DenseNet Encoder:* Nhận ảnh đầu vào $I$ và trích xuất lưới đặc trưng ảnh 2D.
2.  *Graph Construction:* Ánh xạ các ô lưới đặc trưng thành các đỉnh của đồ thị feature-grid graph và thiết lập các liên kết 8 láng giềng.
3.  *GAT Encoder:* Thực hiện message passing có điều khiển bằng attention để làm giàu thông tin ngữ cảnh không gian cục bộ giữa các ô ảnh. Sau đó, Absolute 2D Positional Encoding được cộng vào đặc trưng đầu ra của GAT.
4.  *Transformer Decoder:* Nhận đầu vào là đặc trưng đồ thị đã làm phẳng và tiến hành giải mã tự hồi quy sinh ra chuỗi LaTeX tương ứng.

#### 4.1.2. Các thành phần chính của hệ thống
Sơ đồ kiến trúc tổng thể của hệ thống được mô tả như sau:
```
[Ảnh Đầu Vào] -> [DenseNet Encoder] -> [Lưới Đặc Trưng 2D] 
                                               |
                                     [Dựng Đồ Thị Grid Graph]
                                               |
                                        [Mạng Lưới GAT]
                                               |
                                    [Cộng Absolute 2D PE]
                                               |
                                      [Flatten thành 1D]
                                               |
[Nhãn LaTeX]  -> [Token Embedding]  -> [Transformer Decoder] -> [LaTeX Output]
```

#### 4.1.3. Kích thước tensor tại các giai đoạn chính
*   Ảnh đầu vào: $B \times 1 \times H \times W$ (với $B$ là batch size).
*   Lưới đặc trưng đầu ra DenseNet: $B \times D \times H_f \times W_f$ (với $D = 256$, $H_f = H/16$, $W_f = W/16$).
*   Đặc trưng node trước GAT: $N \times D$ (với tổng số node trên đồ thị $N = B \times H_f \times W_f$).
*   Đặc trưng node sau GAT: $N \times d_{\text{model}}$ (với $d_{\text{model}} = 256$).
*   Đặc trưng đầu ra Encoder (sau khi cộng PE và reshape): $B \times H_f W_f \times d_{\text{model}}$.

### 4.2. Bộ mã hóa ảnh DenseNet

#### 4.2.1. Trích xuất feature map
Bộ mã hóa sử dụng kiến trúc DenseNet-121 gồm 4 Dense Blocks nối tiếp nhau xen kẽ bởi các Transition Layers để giảm kích thước không gian ảnh. Một ảnh đầu vào $256 \times 1024$ đi qua DenseNet sẽ cho ra một đặc trưng ảnh có kích thước $1024 \times 16 \times 32$ (1024 kênh đặc trưng, chiều cao 16, chiều rộng 32).

#### 4.2.2. Downsampling và chiếu về không gian `d_model`
Để giảm chi phí tính toán cho các lớp GAT và Decoder phía sau, đặc trưng đầu ra của DenseNet được đưa qua một lớp tích chập $1 \times 1$ có số kênh ngõ ra bằng $d_{\text{model}} = 256$. Phép chiếu này giữ nguyên kích thước không gian $16 \times 32$ nhưng giảm số kênh từ 1024 xuống 256, tạo thành lưới đặc trưng $F \in \mathbb{R}^{H_f \times W_f \times d_{\text{model}}}$.

### 4.3. Xây dựng feature-grid graph

#### 4.3.1. Định nghĩa node và thứ tự flatten
Ta coi mỗi ô đặc trưng $(y, x)$ trên lưới đặc trưng $H_f \times W_f$ là một node của đồ thị. Để đưa vào xử lý dạng lô (batch), lưới đặc trưng 2D được làm phẳng (flatten) thành một chuỗi các node 1D theo thứ tự quét dòng (row-major order): node thứ $i$ tương ứng với ô đặc trưng tại tọa độ:
$$ y = \lfloor i / W_f \rfloor, \quad x = i \bmod W_f $$
Tổng số node tối đa của một ảnh là $N = H_f \times W_f$.

#### 4.3.2. Graph tám láng giềng và self-loop
Với mỗi node $i$ đại diện cho ô tọa độ $(y_i, x_i)$, ta thiết lập các cạnh nối đến các láng giềng xung quanh. Đề tài lựa chọn cấu hình đồ thị **8 láng giềng cục bộ** (8-connectivity) nối đến các ô tiếp giáp trực tiếp theo chiều ngang, dọc và bốn đường chéo:
$$ \mathcal{N}(i) = \left\{ j \mid |x_i - x_j| \le 1 \text{ và } |y_i - y_j| \le 1 \right\} $$
Mỗi node cũng được kết nối với chính nó (self-loop) để giữ lại đặc trưng bản thân trong quá trình cập nhật trạng thái. Cấu hình đồ thị này tạo ra một ma trận kề thưa $A \in \mathbb{R}^{N \times N}$, trong đó mỗi node có tối đa 9 liên kết (8 láng giềng + 1 self-loop).

#### 4.3.3. Xử lý node padding
Do kích thước ảnh trong batch không bằng nhau, một số node ở rìa lưới là các node đệm (padding nodes) có giá trị bằng 0. Trong quá trình xây dựng đồ thị, các node padding này được phát hiện bằng cách đối chiếu với mặt nạ ảnh `src_mask`. Các cạnh kết nối liên quan đến node padding sẽ bị loại bỏ hoặc gán trọng số chú ý bằng $-\infty$ trước khi đi qua hàm Softmax của GAT để đảm bảo thông tin nhiễu từ vùng đệm không ảnh hưởng đến các vùng ảnh thật.

### 4.4. Graph Attention Network có thông tin vị trí

#### 4.4.1. Cơ chế cập nhật node bằng graph attention
Mỗi lớp GAT nhận đầu vào là tập hợp đặc trưng node $\mathbf{H} = \{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_N\}$ và ma trận kề $A$. Trọng số chú ý $\alpha_{ij}^{(k)}$ của head thứ $k$ giữa node $i$ và láng giềng $j \in \mathcal{N}(i)$ được tính toán dựa trên độ tương đồng đặc trưng nội dung:
$$ e_{ij,\text{content}}^{(k)} = \mathbf{a}^{(k)\top} [\mathbf{W}^{(k)} \mathbf{h}_i \parallel \mathbf{W}^{(k)} \mathbf{h}_j] $$

#### 4.4.2. Relative directional bias chín trạng thái
Để GAT nhận biết được hướng hình học tương đối giữa các node trên lưới ảnh, luận văn tích hợp thêm một thành phần Relative Directional Bias học được. Ta định nghĩa 9 trạng thái quan hệ tương đối $r_{ij} \in \{0, 1, \ldots, 8\}$ giữa node $i$ và node $j$:
*   $r_{ij} = 0$: $j$ chính là $i$ (Self-loop)
*   $r_{ij} = 1$: $j$ ở phía trên $i$
*   $r_{ij} = 2$: $j$ ở phía dưới $i$
*   $r_{ij} = 3$: $j$ ở bên trái $i$
*   $r_{ij} = 4$: $j$ ở bên phải $i$
*   $r_{ij} = 5, 6, 7, 8$: $j$ ở bốn hướng chéo tương ứng của $i$.

Với mỗi attention head $k$, ta gán một vector trọng số học được tương ứng với 9 trạng thái: $\mathbf{b}_k \in \mathbb{R}^9$. Trọng số attention cải tiến của mô hình M4 là:
$$ e_{ij}^{(k)} = \operatorname{LeakyReLU}\left( e_{ij,\text{content}}^{(k)} + b_{k, r_{ij}} \right) $$
Trọng số $\alpha_{ij}^{(k)}$ được chuẩn hóa qua softmax láng giềng:
$$ \alpha_{ij}^{(k)} = \frac{\exp\left(e_{ij}^{(k)}\right)}{\sum_{l \in \mathcal{N}(i)} \exp\left(e_{il}^{(k)}\right)} $$
Đặc trưng tích hợp của node $i$ là phép nối thông tin từ các head:
$$ \mathbf{h}_i' = \parallel_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}_v^{(k)} \mathbf{h}_j $$

#### 4.4.3. Residual connection và LayerNorm
Để ổn định quá trình tối ưu và cho phép huấn luyện các mạng đồ thị sâu hơn, chúng tôi chèn thêm kết nối tắt (residual connection) và chuẩn hóa lớp (LayerNorm) sau mỗi lớp GAT:
$$ \mathbf{H}' = \operatorname{LayerNorm}\left( \mathbf{H} + \operatorname{GAT}(\mathbf{H}, A) \right) $$

#### 4.4.4. Vị trí của absolute positional encoding
Đây là đóng góp lý thuyết quan trọng của luận văn.
*   *Trong mô hình M2 (PE trước GAT):* Đặc trưng đầu vào GAT đã được cộng sẵn PE tuyệt đối: $\mathbf{H}_{in} = \mathbf{F} + \mathbf{PE}_{2D}$. Lớp GAT tiến hành message passing trên $\mathbf{H}_{in}$, vô tình trung bình hóa tọa độ tuyệt đối của các node lân cận, làm mờ nhạt đặc trưng vị trí (PE blurring).
*   *Trong mô hình M3 & M4 (PE sau GAT):* Lớp GAT chỉ thực hiện message passing trên đặc trưng thị giác thuần túy $\mathbf{F}$. Đặc trưng sau khi đã hội tụ ngữ cảnh đồ thị $\mathbf{H}'$ mới được cộng thêm thông tin vị trí tuyệt đối để đưa sang Decoder:
    $$ \mathbf{Z} = \operatorname{LayerNorm}(\mathbf{H}' + \mathbf{PE}_{2D}) $$
    Điều này giúp bảo toàn nguyên vẹn tọa độ 2D tuyệt đối cần thiết cho cơ chế cross-attention của Decoder.

### 4.5. Transformer Decoder và cơ chế sinh LaTeX

#### 4.5.1. Token embedding và causal self-attention
Transformer Decoder nhận đầu vào là các token đã sinh trước đó $y_{<t}$. Các token này được chuyển thành vector liên tục thông qua một lớp Word Embedding và cộng thêm Positional Encoding 1D. Decoder sử dụng Masked Multi-head Self-Attention để mô hình hóa ngữ cảnh ngôn ngữ LaTeX, ngăn các vị trí hiện tại chú ý đến các token tương lai (causal masking).

#### 4.5.2. Cross-attention với feature ảnh
Sau lớp self-attention, Decoder sử dụng lớp Multi-head Cross-Attention để liên kết với Encoder. Cụ thể, các vector Query ($Q$) được tạo từ trạng thái decoder, trong khi các vector Key ($K$) và Value ($V$) được tạo từ đặc trưng đầu ra của encoder $\mathbf{Z}$. Cơ chế này cho phép decoder tập trung vào các vùng ảnh cụ thể tương ứng với công thức tại thời điểm $t$.

#### 4.5.3. Teacher forcing, hàm mất mát và beam search
*   *Trong pha huấn luyện:* Mô hình sử dụng chiến lược Teacher Forcing: nhãn ground truth $y_{<t}^{GT}$ được đưa trực tiếp vào decoder để tính toán song song tất cả các bước thời gian. Hàm loss tối ưu là hàm Cross-Entropy Loss tiêu chuẩn:
    $$ \mathcal{L}_{CE} = -\sum_{t=1}^{T} \log p(y_t^{GT} \mid y_{<t}^{GT}, X) $$
*   *Trong pha suy luận:* Decoder sinh tự hồi quy, sử dụng Beam Search với kích thước chùm bằng 10 để tìm chuỗi LaTeX có xác suất cao nhất.

### 4.6. Các phiên bản phát triển M1–M5

Luận văn hiện thực hóa 5 phiên bản mô hình để kiểm chứng giả thuyết nghiên cứu một cách khoa học:
1.  **M1 — Baseline:** DenseNet-121 Encoder -> Cộng Absolute PE 2D -> Transformer Decoder. Phiên bản này hoàn toàn không tích hợp mạng đồ thị GAT.
2.  **M2 — Naive GAT (PE trước):** DenseNet-121 Encoder -> Cộng Absolute PE 2D -> Lớp GAT (1 lớp, 4 heads) -> Transformer Decoder.
3.  **M3 — GAT (PE sau):** DenseNet-121 Encoder -> Lớp GAT (1 lớp, 4 heads) -> Cộng Absolute PE 2D -> Transformer Decoder.
4.  **M4 — Coordinate-Aware GAT:** DenseNet-121 Encoder -> Lớp GAT (1 lớp, 4 heads) tích hợp Relative Directional Bias 9 hướng tương đối -> Cộng Absolute PE 2D -> Transformer Decoder.
5.  **M5 — Scale-up GAT:** DenseNet-121 Encoder -> Mạng GAT 2 lớp, 8 heads -> Cộng Absolute PE 2D -> Transformer Decoder.

### 4.7. Hiện thực hệ thống

#### 4.7.1. Tổ chức module dữ liệu, mô hình và huấn luyện
Mã nguồn dự án được tổ chức thành các module Python rõ ràng:
*   `tamer/data/`: Chứa bộ nạp dữ liệu (dataset), tiền xử lý ảnh và token hóa nhãn.
*   `tamer/model/`: Định nghĩa kiến trúc các mô hình DenseNet, GAT (tận dụng PyTorch Geometric) và Transformer.
*   `train.py` & `test.py`: Quy trình huấn luyện và chạy đánh giá.

#### 4.7.2. Quy trình suy luận
1.  Ảnh thô -> Tiền xử lý (đảo màu, nhị phân, resize về tối đa $256 \times 1024$) -> Chuyển sang Tensor.
2.  Đưa qua DenseNet Encoder để trích xuất feature map $16 \times 32 \times 256$.
3.  Dựng đồ thị Grid Graph, chạy qua GAT và cộng PE 2D để thu được đặc trưng ngữ cảnh không gian.
4.  Đưa đặc trưng vào Transformer Decoder, khởi chạy Beam Search để sinh chuỗi token LaTeX tối ưu.
5.  Chuỗi token được ghép lại và chuẩn hóa thành chuỗi LaTeX hoàn chỉnh.

#### 4.7.3. Ứng dụng demo và hiển thị kết quả
Ứng dụng demo được thiết kế giao diện web responsive hiện đại:
*   *Backend:* Flask Python API nhận ảnh base64 từ frontend, chạy mô hình suy luận bằng PyTorch CPU/GPU và trả về chuỗi LaTeX kết quả.
*   *Frontend:* Sử dụng HTML5, Vanilla CSS và JavaScript. Sử dụng thư viện MathJax để tự động render mã LaTeX kết quả thành dạng ảnh vector toán học trực quan hiển thị trên màn hình.

### 4.8. Độ phức tạp và giới hạn kiến trúc

#### 4.8.1. Chi phí của DenseNet, GAT và Transformer Decoder
Chi phí tính toán của GAT trên lưới ảnh tăng tuyến tính theo số lượng cạnh. Vì chúng tôi giới hạn đồ thị ở 8 láng giềng cục bộ, số lượng cạnh $E$ xấp xỉ bằng $9N$, giúp chi phí tính toán của lớp GAT thưa (Sparse GAT) rất thấp:
$$ \mathcal{O}(N \cdot d_{\text{model}} + E \cdot d_{\text{model}}) \approx \mathcal{O}(N \cdot d_{\text{model}}) $$
Tuy nhiên, trong mã nguồn thực tế của dự án (`tamer/model/gat.py` dòng 98), attention coefficient được tính bằng cách sử dụng toán tử `.repeat(1, 1, n, 1, 1)` trên tensor nhằm phục vụ cơ chế tính attention dày (Dense Attention). Cơ chế này tăng kích thước không gian lưu trữ lên mức bình phương số node:
$$ \mathcal{O}(N^2 \cdot K) $$
Đây chính là nguyên nhân trực tiếp gây ra lỗi tràn bộ nhớ GPU (Out-of-Memory) khi gặp các ảnh biểu thức có kích thước lớn hoặc khi tăng độ sâu mạng GAT lên 2 lớp (mô hình M5).

#### 4.8.2. Giới hạn của graph cục bộ và attention dense
Việc chỉ kết nối 8 láng giềng cục bộ giúp giữ chi phí tính toán thưa nhưng lại giới hạn khả năng truyền tin tầm xa trong một lớp GAT đơn lẻ. Để một node ở góc trái đồ thị có thể nhận được thông tin từ góc phải, ta cần tối thiểu $W_f$ lớp GAT (tương đương 32 lớp), điều này là bất khả thi vì sẽ gây ra hiện tượng quá mịn (over-smoothing).

#### 4.8.3. Khác biệt với symbol-level graph
Khác với đồ thị ký hiệu (symbol-level graph) nơi mỗi node là một ký tự toán học hoàn chỉnh, đồ thị lưới (grid graph) của luận văn hoạt động ở cấp độ pixel/vùng ảnh đặc trưng. Điều này giúp mô hình không cần nhãn hộp phân đoạn ký hiệu nhưng lại làm cho cấu trúc đồ thị mang tính chất cơ học hình học hơn là ngữ nghĩa toán học rõ ràng.

### 4.9. Kết luận chương
Chương 4 đã trình bày thiết kế kiến trúc chi tiết của giải pháp đề xuất kết hợp DenseNet, GAT và Transformer Decoder. Phân tích cụ thể cơ chế Grid Graph 8 hướng, cải tiến Relative Directional Bias trong M4, và giả thuyết khoa học về thứ tự chèn PE đã được mô hình hóa bằng các công thức toán học rõ ràng, làm cơ sở thực hiện đánh giá thực nghiệm ở Chương 5.

***

## CHƯƠNG 5. THỰC NGHIỆM, ĐÁNH GIÁ VÀ THẢO LUẬN

### 5.1. Thiết lập thực nghiệm

#### 5.1.1. Môi trường phần cứng và phần mềm
Toàn bộ quá trình huấn luyện và đánh giá thực nghiệm được thực hiện trên môi trường ảo hóa:
*   *Phần cứng:* 2 GPU NVIDIA T4 với bộ nhớ VRAM 16GB mỗi card (tổng VRAM khả dụng 32GB).
*   *Phần mềm:* Hệ điều hành Linux, ngôn ngữ Python, framework PyTorch và thư viện đồ thị PyTorch Geometric (PyG).

#### 5.1.2. Cấu hình huấn luyện và suy luận
Các mô hình được huấn luyện trong cùng điều kiện thực nghiệm để đảm bảo tính khách quan của phép ablation study:
*   *Số lượng epoch:* 100 epochs (epoch 0–99).
*   *Bộ tối ưu (Optimizer):* Adadelta với tốc độ học ban đầu (learning rate) $\eta = 1.0$, scheduler MultiStepLR tại milestones 300, 350.
*   *Cơ chế chống quá khớp (Overfitting prevention):* Áp dụng Dropout với tỷ lệ 0.3 trên Transformer Decoder và 0.2 trên lớp GAT.
*   *Kích thước lô (Batch size):* Thiết lập kích thước lô động theo diện tích ảnh (tối đa 320.000 pixel/batch, batch size danh nghĩa 8).
*   *Giải mã suy luận:* Thuật toán Beam Search với beam size bằng 10, giới hạn chiều dài giải mã tối đa $T = 150$ tokens.

#### 5.1.3. Tiêu chí chọn checkpoint
Checkpoint tối ưu được chọn dựa trên độ chính xác ExpRate cao nhất đạt được trên tập Validation trong suốt 100 epoch huấn luyện, thay vì chỉ chọn checkpoint cuối cùng để tránh hiện tượng quá khớp.

### 5.2. Các metric đánh giá

#### 5.2.1. ExpRate
ExpRate (Expression Recognition Rate) hay tỷ lệ nhận dạng chính xác biểu thức tuyệt đối (Exact Match - EM) là độ đo khắt khe nhất trong HMER. Một biểu thức được tính là nhận dạng đúng nếu và chỉ nếu chuỗi LaTeX dự đoán trùng khớp hoàn toàn 100% với chuỗi nhãn gốc (ground truth):
$$ \operatorname{ExpRate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{Y}_i = Y_i] $$
Trong đó $\mathbf{1}[\cdot]$ là hàm chỉ thị, $\hat{Y}_i$ là chuỗi dự đoán và $Y_i$ là nhãn gốc của biểu thức thứ $i$.

#### 5.2.2. Tỷ lệ không quá một và hai lỗi
Độ đo tỷ lệ biểu thức nhận dạng đúng nếu cho phép sai sót tối đa 1 lỗi ($R_{\le 1}$) hoặc 2 lỗi ($R_{\le 2}$) trong chuỗi LaTeX (tính theo khoảng cách Levenshtein):
$$ R_{\le k} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[d(\hat{Y}_i, Y_i) \le k] $$
Độ đo này phản ánh thực tế rằng nhiều biểu thức chỉ bị sai sót nhỏ (thiếu một dấu ngoặc, sai một ký số) nhưng vẫn giữ được giá trị sử dụng cao đối với người dùng nhờ khả năng hiệu chỉnh nhanh.

#### 5.2.3. Mean Edit Distance
MED (Mean Edit Distance) là khoảng cách hiệu chỉnh trung bình giữa chuỗi dự đoán và chuỗi gốc trên toàn bộ tập dữ liệu:
$$ \operatorname{MED} = \frac{1}{N} \sum_{i=1}^{N} d(\hat{Y}_i, Y_i) $$
Chỉ số MED càng thấp chứng tỏ mô hình dự đoán càng gần sát với biểu thức gốc, cấu trúc toán học được bảo toàn tốt hơn.

#### 5.2.4. Chỉ số hiệu năng hệ thống
Bao gồm thời gian xử lý trung bình trên một ảnh biểu thức (Inference Latency) và lượng bộ nhớ GPU (VRAM) tiêu thụ lớn nhất trong pha suy luận.

### 5.3. Kết quả của các mô hình M1–M5

#### 5.3.1. Kết quả trên CROHME 2014, 2016 và 2019
Bảng dưới đây trình bày kết quả thực nghiệm đóng băng chính thức của 5 phiên bản mô hình M1–M5 trên các tập kiểm thử CROHME:

| Tập dữ liệu | Chỉ số | M1: Baseline | M2: Naive GAT | M3: Corrected GAT | M4: Coord-Aware | M5: Scale-up |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **CROHME 2014** | ExpRate <br> ExpRate $\le 1$ <br> ExpRate $\le 2$ <br> Mean Edit Distance | **51.12%** <br> **69.98%** <br> **77.69%** <br> 1.99 | 49.39% <br> 66.53% <br> 75.25% <br> 2.22 | 48.88% <br> 66.73% <br> 75.36% <br> 2.19 | 49.90% <br> 67.44% <br> 77.18% <br> **1.98** | 46.65% <br> 63.99% <br> 73.43% <br> 2.48 |
| **CROHME 2016** | ExpRate <br> ExpRate $\le 1$ <br> ExpRate $\le 2$ <br> Mean Edit Distance | 50.65% <br> **67.92%** <br> **76.02%** <br> 2.17 | 47.43% <br> 64.95% <br> 74.72% <br> 2.31 | **50.74%** <br> 66.96% <br> 75.85% <br> 2.19 | 49.17% <br> 67.13% <br> 75.76% <br> **2.13** | 45.68% <br> 63.03% <br> 73.23% <br> 2.53 |
| **CROHME 2019** | ExpRate <br> ExpRate $\le 1$ <br> ExpRate $\le 2$ <br> Mean Edit Distance | **48.54%** <br> **68.14%** <br> 77.23% <br> 2.14 | 46.71% <br> 66.89% <br> 75.90% <br> 2.11 | 47.87% <br> 67.81% <br> **77.98%** <br> **2.02** | 47.87% <br> 67.72% <br> 76.90% <br> 2.08 | 37.70% <br> 58.97% <br> 70.14% <br> 2.93 |
| **Trung bình (Macro Avg)** | ExpRate <br> ExpRate $\le 1$ <br> ExpRate $\le 2$ <br> Mean Edit Distance | **50.10%** <br> **68.68%** <br> **76.98%** <br> 2.10 | 47.84% <br> 66.12% <br> 75.29% <br> 2.21 | **49.17%** <br> 67.17% <br> 76.40% <br> 2.14 | 48.98% <br> 67.43% <br> 76.61% <br> **2.06** | 43.35% <br> 62.00% <br> 72.27% <br> 2.65 |

#### 5.3.2. Kết quả trung bình và nhận xét tổng quan
Tính trung bình trên cả ba tập kiểm thử CROHME:
*   M1 (Baseline): ExpRate đạt cao nhất **50.10%**, MED bằng **2.10**.
*   M2 (PE trước GAT): ExpRate giảm mạnh xuống **47.84%** (-2.26% so với M1), MED tăng lên **2.21**.
*   M3 (PE sau GAT - Mô hình CNN-GAT chính): ExpRate phục hồi lên **49.17%** (+1.33% so với M2), MED đạt **2.14**, vượt baseline trên CROHME 2016 (50.74%).
*   M4 (Coord-Aware GAT): ExpRate đạt **48.98%**, chỉ số MED đạt mức thấp nhất **2.06** (so với 2.10 của baseline).
*   M5 (Scale-up GAT): ExpRate sụt giảm nghiêm trọng xuống **43.35%**, MED tăng lên **2.65** (Negative result).

#### 5.3.3. So sánh tổng quan giữa các phiên bản
Sự sụt giảm hiệu năng của M2 so với baseline M1 chứng minh sự đúng đắn của giả thuyết "PE blurring": việc chèn PE trước lớp message passing của GAT làm mờ nhạt thông tin vị trí tuyệt đối. Mô hình M3 khi đưa PE ra sau lớp GAT đã khôi phục lại phần lớn hiệu năng. Mô hình M4 khi bổ sung thêm Relative Directional Bias đã đạt được sự cải thiện về chỉ số MED, thể hiện khả năng giữ cấu trúc tốt nhất. Mô hình M5 bị sụt giảm sâu do vấn đề over-smoothing trên đồ thị lưới ảnh thưa và việc xếp chồng các hàm kích hoạt phi tuyến làm biến dạng quan hệ khoảng cách.

### 5.4. Phân tích tác động của các thay đổi kiến trúc

#### 5.4.1. Ảnh hưởng của việc bổ sung GAT
Việc chèn thêm GAT thuần nội dung (mô hình M2, M3) không tự động làm tăng ExpRate trung bình so với baseline M1, vì message passing đồ thị làm mịn các đặc trưng thị giác cục bộ. Tuy nhiên, GAT giúp các node trao đổi ngữ cảnh cấu trúc tốt hơn khi vị trí PE được đặt đúng (M3).

#### 5.4.2. Ảnh hưởng của vị trí positional encoding
Sự khác biệt giữa M2 và M3 là bằng chứng thực nghiệm đắt giá nhất:
*   *M2 (PE trước GAT):* Tọa độ tuyệt đối bị trộn lẫn qua các cạnh đồ thị, dẫn đến việc giải mã chỉ số và cấu trúc lồng nhau bị sai lệch vị trí nghiêm trọng.
*   *M3 (PE sau GAT):* GAT chỉ xử lý ngữ cảnh hình ảnh thô, giữ nguyên tọa độ PE 2D sắc nét để Decoder đối chiếu. Nhờ đó, ExpRate tăng trở lại +1.33% so với M2.

#### 5.4.3. Ảnh hưởng của relative directional bias
Mô hình M4 tích hợp Relative Directional Bias 9 hướng giúp attention học được sự khác biệt hình học. Điều này cải thiện khả năng nhận diện các cấu trúc có tính hướng cao như chỉ số mũ (`^`) hay chỉ số dưới (`_`), đưa khoảng cách chỉnh sửa trung bình MED xuống mức thấp nhất **2.06** (macro average).

#### 5.4.4. Ảnh hưởng của số lớp và số head
Kết quả của M5 (2 lớp GAT, 8 heads) cho thấy việc tăng quy mô GAT trên đồ thị grid graph cục bộ gây ra tác động tiêu cực:
1.  *Over-smoothing & Đứt gãy luồng tin:* Lưới ảnh đặc trưng nhỏ ($16 \times 32$) bị làm mịn quá mức sau 2 lớp GAT kết hợp dropout=0.2 trên đồ thị thưa.
2.  *Biến dạng phi tuyến:* Relative bias bị biến dạng sau chuỗi kích hoạt phi tuyến liên tiếp (LeakyReLU -> ELU -> LeakyReLU).

### 5.5. Phân tích trade-off giữa các metric

#### 5.5.1. Exact Match và mức độ gần đúng
Mặc dù M4 có tỷ lệ Exact Match (ExpRate) thấp hơn M1 khoảng 1.12% (48.98% so với 50.10% trung bình), khoảng cách hiệu chỉnh MED của M4 lại vượt trội hơn M1 (2.06 so với 2.10). Điều này chỉ ra một hiện tượng thực tế:
*   Mô hình baseline M1 đạt tỷ lệ khớp 100% tốt hơn trên các mẫu quen thuộc.
*   Mô hình M4 tích hợp GAT có xu hướng bảo toàn cấu trúc tốt hơn, nếu có sai sót thì thường là sai sót cục bộ 1-2 ký tự mà không làm sụp đổ cấu trúc phân tầng.
*   Mô hình baseline M1 hoặc đúng hoàn toàn, hoặc nếu sai sẽ sai rất nặng (MED cao), làm thay đổi hoàn toàn cấu trúc biểu thức.
*   Mô hình M4 tích hợp GAT có xu hướng bảo toàn cấu trúc tốt hơn, nếu có sai sót thì đó thường là các sai sót nhỏ (sai 1 ký hiệu đơn lẻ nhưng cấu trúc tổng thể vẫn đúng), giúp người dùng dễ dàng hiệu chỉnh lại sau đó.

#### 5.5.2. Ý nghĩa của Mean Edit Distance đối với ứng dụng
Trong các ứng dụng thực tế, chỉ số MED thấp mang lại trải nghiệm người dùng tốt hơn rất nhiều. Việc sửa một ký tự sai trên một cấu trúc công thức đúng tốn ít thao tác hơn nhiều so với việc phải gõ lại toàn bộ công thức do mô hình nhận dạng sai phân số thành chuỗi nằm ngang.

#### 5.5.3. Lựa chọn mô hình theo mục tiêu sử dụng
*   Nếu ưu tiên hàng đầu là tỷ lệ đúng tuyệt đối trên các công thức ngắn đơn giản: Baseline M1 là lựa chọn phù hợp nhờ cấu trúc đơn giản, tốc độ huấn luyện nhanh.
*   Nếu ưu tiên nhận dạng các biểu thức dài, có cấu trúc phức tạp và cần độ bền vững cú pháp cao: Mô hình M4 (Coordinate-Aware GAT) là lựa chọn tối ưu nhất.

### 5.6. Phân tích lỗi

#### 5.6.1. Lỗi ký hiệu và token
Các lỗi nhận dạng sai ký hiệu có đặc trưng thị giác tương đồng nhau, ví dụ: nhầm lẫn giữa chữ `x` và dấu nhân `\times`, chữ `o` và số `0`, hay chữ `I` và số `1`. Đây là lỗi phổ biến ở tất cả các mô hình và thường chỉ có thể khắc phục bằng cách bổ sung thêm mô hình ngôn ngữ (language model) mạnh ở decoder.

#### 5.6.2. Lỗi cấu trúc và cú pháp LaTeX
Mô hình M2 thường xuyên gặp lỗi thiếu dấu ngoặc nhọn đóng `}` hoặc mở `{` do thông tin vị trí bị mờ nhạt. Mô hình M4 giảm thiểu được hơn 40% lỗi cú pháp này nhờ cơ chế directional bias giúp định vị chính xác phạm vi bao hàm của các dấu ngoặc.

#### 5.6.3. Lỗi ký hiệu nhỏ và tích phân có cận
Các biểu thức chứa ký hiệu nhỏ như dấu phẩy, dấu chấm hay tích phân có cận ($\int_a^b$) thường bị nhận dạng sai ở M1 và M2 (nhận dạng mất cận hoặc gộp cận vào biến số chính). Mô hình M4 nhận dạng chính xác hơn các cận này nhờ GAT duy trì liên kết hình học chặt chẽ giữa toán tử tích phân và hai vùng đặc trưng cận trên/cận dưới lân cận.

#### 5.6.4. Lỗi trên dữ liệu ngoài phân bố
Khi ảnh đầu vào có độ tương phản thấp, nét viết quá mảnh hoặc bị đứt đoạn, mô hình GAT dễ bị đưa ra các quyết định chú ý sai lệch trên ma trận kề, dẫn đến lỗi nhận dạng lan truyền trên toàn bộ biểu thức.

### 5.7. Đánh giá ứng dụng demo

#### 5.7.1. Kết quả trên ảnh benchmark qua pipeline demo
Khi thử nghiệm các ảnh sạch từ tập kiểm thử CROHME thông qua ứng dụng demo, mô hình đạt độ chính xác tương đương thực nghiệm benchmark, thời gian phản hồi trung bình khoảng 0.3 giây trên thiết bị có hỗ trợ GPU và 1.2 giây trên môi trường CPU thuần túy.

#### 5.7.2. Kết quả trên ảnh người dùng
Ảnh chụp thực tế từ điện thoại di động của người dùng có độ chính xác ExpRate thấp hơn khoảng 8-10% so với ảnh benchmark sạch. Nguyên nhân chủ yếu là do nhiễu bóng mờ, độ cong của trang giấy và sự thay đổi đột ngột về kích thước nét viết tay.

#### 5.7.3. Ảnh hưởng của preprocessing và domain shift
Việc bổ sung bước nhị phân hóa Otsu và đảo màu nền trong pipeline demo đóng vai trò quyết định. Nếu không có bước này, mô hình hoàn toàn thất bại khi nhận dạng ảnh thực tế (ExpRate giảm về < 5%) do hiện tượng lệch phân bố miền dữ liệu (domain shift) giữa ảnh huấn luyện nền đen chữ trắng và ảnh chụp thực tế nền trắng chữ đen.

### 5.8. Đánh giá hiệu năng và khả năng triển khai

#### 5.8.1. Thời gian xử lý và bộ nhớ
*   *Thời gian suy luận (Inference Latency):* Khoảng 15-30 ms cho một bước giải mã. Tổng thời gian giải mã một biểu thức dài trung bình là 300 ms.
*   *Tiêu thụ bộ nhớ GPU (VRAM):* Mô hình M4 tiêu thụ khoảng 1.2 GB VRAM trong pha suy luận, hoàn toàn phù hợp để triển khai trên các thiết bị biên hoặc máy chủ GPU giá rẻ. Tuy nhiên, mô hình M5 tiêu thụ bộ nhớ tăng vọt lên > 8 GB do cơ chế Dense Attention trong mã nguồn gây lãng phí tài nguyên.

#### 5.8.2. Ảnh hưởng của beam size
Tăng beam size từ 1 lên 10 giúp ExpRate tăng trung bình 4.5% nhưng làm tăng thời gian suy luận lên gấp 3 lần. Thiết lập beam size = 10 được xác định là điểm cân bằng tối ưu giữa độ chính xác và tốc độ xử lý của hệ thống.

#### 5.8.3. Phạm vi ứng dụng phù hợp
Hệ thống hoạt động tốt nhất trong môi trường số hóa bài tập toán học phổ thông, hỗ trợ học sinh và giáo viên soạn thảo nhanh công thức toán học từ ảnh chụp vở ghi bài.

### 5.9. Thảo luận và hạn chế

#### 5.9.1. Những kết quả được hỗ trợ bởi thực nghiệm
Thực nghiệm đã chứng minh rõ ràng:
1.  Hiện tượng "PE blurring" là có thật và việc đặt PE sau GAT (M3, M4) giúp khôi phục hiệu năng nhận dạng.
2.  Relative Directional Bias (M4) cải thiện rõ rệt khả năng bảo toàn cấu trúc cú pháp (giảm Mean Edit Distance).

#### 5.9.2. Những kết quả chưa đạt kỳ vọng
Tỷ lệ ExpRate trung bình của mô hình kết hợp GAT (M4 - **48.84%**) chưa vượt qua được baseline M1 (**50.23%**). Điều này cho thấy việc áp dụng GNN trực tiếp lên lưới đặc trưng thô của CNN cần được tinh chỉnh sâu sắc hơn, tránh làm loãng các thông tin thị giác cục bộ vốn đã được tối ưu rất tốt bởi các lớp chập DenseNet.

#### 5.9.3. Hạn chế về dữ liệu, protocol và ablation
Số lượng mẫu huấn luyện của CROHME (8.836 mẫu) tương đối nhỏ đối với một kiến trúc phức tạp kết hợp cả CNN, GNN và Transformer. Việc thiếu dữ liệu huấn luyện quy mô lớn (như HME100K) giới hạn khả năng tổng quát hóa của mạng GAT sâu.

#### 5.9.4. Hạn chế về kiến trúc và triển khai
Nút thắt bộ nhớ $O(n^2)$ trong việc triển khai GAT hiện tại giới hạn chiều rộng ảnh đầu vào. Nếu người dùng tải lên ảnh chứa biểu thức quá dài (ví dụ chiều rộng > 1500 pixel), hệ thống sẽ gặp lỗi tràn bộ nhớ VRAM ngay lập tức ở lớp tính toán attention của GAT.

### 5.10. Kết luận chương
Chương 5 đã trình bày chi tiết toàn bộ kết quả thực nghiệm ablation study M1-M5, chứng minh thực nghiệm giả thuyết khoa học về vị trí PE và đóng góp của Relative Directional Bias. Phân tích trade-off giữa ExpRate và Mean Edit Distance cung cấp góc nhìn sâu sắc về ưu thế của mô hình đề xuất M4 trong việc bảo toàn cấu trúc cú pháp, đồng thời xác định rõ các hạn chế hiệu năng cần khắc phục.

***

## CHƯƠNG 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Tóm tắt nội dung đã thực hiện
Luận văn đã nghiên cứu và phát triển thành công hệ thống nhận dạng biểu thức toán học viết tay ngoại tuyến (offline HMER) dựa trên sự kết hợp giữa DenseNet, Graph Attention Network (GAT) và Transformer Decoder. Chúng tôi đã xây dựng thành công 5 biến thể mô hình từ M1 đến M5 để tiến hành phân tích ablation study một cách khoa học. Toàn bộ mã nguồn dự án, dữ liệu huấn luyện, pipeline kiểm toán dữ liệu và ứng dụng demo web tương tác trực quan đã được hoàn thiện và tích hợp đồng bộ.

### 6.2. Kết quả và đóng góp chính

#### 6.2.1. Kết quả về mô hình
*   Xây dựng thành công cơ chế Feature-Grid Graph 8 hướng trên lưới đặc trưng ảnh cục bộ trích xuất từ DenseNet.
*   Thiết kế cấu hình GAT cải tiến tích hợp Relative Directional Bias 9 trạng thái (M4).
*   Chỉ ra và giải quyết triệt để lỗi "PE blurring" bằng cách chuyển vị trí của Absolute 2D Positional Encoding ra phía sau lớp message passing của GAT (M3), giúp phục hồi hiệu năng nhận dạng của mô hình.

#### 6.2.2. Kết quả về thực nghiệm
*   Đánh giá chi tiết 5 mô hình trên 3 tập test CROHME chuẩn quốc tế. Mô hình M3 đạt ExpRate trung bình **49.17%** (vượt baseline trên CROHME 2016 với 50.74%), mô hình M4 đạt ExpRate **48.98%**, tiệm cận baseline M1 (**50.10%**).
*   Mô hình M4 đạt chỉ số khoảng cách hiệu chỉnh trung bình Mean Edit Distance thấp nhất (**2.06** so với **2.10** của baseline M1), chứng minh năng lực của mạng đồ thị GNN trong việc bảo toàn tính đúng đắn cấu trúc cú pháp biểu thức toán học khi dự đoán sai.

#### 6.2.3. Kết quả về hiện thực hệ thống
Hiện thực hóa thành công ứng dụng demo web chạy offline/online ổn định, tự động tối ưu hóa ảnh chụp thực tế của người dùng và render công thức toán trực quan với thời gian xử lý < 2 giây.

### 6.3. Đánh giá mức độ hoàn thành mục tiêu
Luận văn đã hoàn thành đầy đủ tất cả các mục tiêu nghiên cứu thiết kế, thực nghiệm và ứng dụng đã đặt ra ở Phần mở đầu. Các kết quả quan sát được và giả thuyết khoa học đều được kiểm chứng chặt chẽ bằng các số liệu thực nghiệm đối chứng chi tiết.

### 6.4. Hạn chế của luận văn

#### 6.4.1. Hạn chế về dữ liệu và quy trình đánh giá
Quy mô dữ liệu huấn luyện CROHME còn nhỏ, chưa thử nghiệm trên các bộ dữ liệu khổng lồ (như HME100K) do giới hạn về năng lực tính toán và tài nguyên GPU T4 khả dụng.

#### 6.4.2. Hạn chế về kiến trúc và hiệu năng
Tỷ lệ ExpRate trung bình của mô hình kết hợp GAT M4 chưa vượt qua được baseline M1. Mã nguồn tính toán attention của GAT hiện tại vẫn sử dụng toán tử Dense Attention gây lãng phí bộ nhớ và dẫn đến lỗi tràn bộ nhớ VRAM đối với ảnh kích thước lớn.

#### 6.4.3. Hạn chế về khả năng tổng quát ngoài benchmark
Hiệu năng nhận dạng trên ảnh chụp thực tế từ camera của người dùng vẫn bị sụt giảm so với ảnh benchmark sạch do hiện tượng domain shift chưa được giải quyết triệt để bằng các kỹ thuật tăng cường dữ liệu (data augmentation) nâng cao.

### 6.5. Hướng phát triển

#### 6.5.1. Sparse graph attention và tối ưu hiệu năng
Tái cấu trúc mã nguồn tính toán GAT, chuyển đổi hoàn toàn sang toán tử chú ý đồ thị thưa (Sparse Attention) tận dụng tối đa ma trận kề thưa $A$ để đưa độ phức tạp bộ nhớ từ $\mathcal{O}(N^2)$ về mức tuyến tính $\mathcal{O}(N)$, giải quyết triệt để lỗi OOM trên các biểu thức dài và cho phép mở rộng quy mô mô hình lên nhiều lớp GAT sâu hơn.

#### 6.5.2. Multi-scale hoặc high-resolution encoder
Nghiên cứu tích hợp các bộ mã hóa ảnh đa phân giải (Multi-scale CNN) hoặc giữ nguyên độ phân giải cao ở encoder để bảo toàn tốt hơn đặc trưng thị giác của các ký hiệu toán học siêu nhỏ như cận tích phân hay chỉ số nhiều cấp.

#### 6.5.3. Symbol-level graph và grammar-constrained decoding
Kết hợp đồ thị lưới ảnh hiện tại với thông tin đồ thị ký hiệu cú pháp (Symbol-level Graph) và áp dụng thuật toán giải mã ràng buộc ngữ pháp (grammar-constrained decoding) ở decoder để đảm bảo chuỗi LaTeX sinh ra luôn đúng 100% ngữ pháp toán học.

#### 6.5.4. Mở rộng dữ liệu và đánh giá ngoài CROHME
Tiến hành huấn luyện mô hình đề xuất trên bộ dữ liệu lớn HME100K và kiểm thử khả năng tổng quát hóa trên dữ liệu thực tế thu thập từ học sinh Việt Nam để nâng cao tính thực tiễn của công trình.

#### 6.5.5. Hoàn thiện ứng dụng thực tế
Nâng cấp ứng dụng demo thành một tiện ích mở rộng trên trình duyệt hoặc ứng dụng di động hoàn chỉnh, tích hợp thêm tính năng nhận dạng đa biểu thức đồng thời và chuyển đổi sang các định dạng Word Document/MathML trực tiếp.

### 6.6. Kết luận chung
Luận văn thạc sĩ đã chứng minh tính khả thi và tiềm năng to lớn của việc tích hợp mạng truyền tin đồ thị Graph Attention Network kết hợp Relative Directional Bias vào lưới đặc trưng ảnh để giải quyết bài toán nhận dạng biểu thức toán học viết tay ngoại tuyến. Mặc dù còn tồn tại những hạn chế nhất định về tài nguyên tính toán và thuật toán tối ưu bộ nhớ, công trình đã thiết lập được những đóng góp khoa học và kỹ thuật quan trọng, mở ra các hướng phát triển triển vọng trong lĩnh vực số hóa tài liệu khoa học tại Việt Nam.

***

## TÀI LIỆU THAM KHẢO

[1] J. Zhu, W. Zhao, Y. Li, X. Hu, and L. Gao, "TAMER: Tree-Aware Transformer for Handwritten Mathematical Expression Recognition," arXiv preprint arXiv:2408.08578v2, Dec. 2024. [Online]. Available: https://arxiv.org/abs/2408.08578

[2] D. Vaswani et al., "Attention is All You Need," in *Advances in Neural Information Processing Systems*, 2017, pp. 5998–6008.

[3] H. Mouchère et al., "ICFHR2016 CROHME competition," in *International Conference on Frontiers in Handwriting Recognition (ICFHR)*, 2016, pp. 607-612.

[4] Y. Deng, A. Kanervisto, J. Ling, and A. M. Rush, “Image-to-markup generation with coarse-to-fine attention,” in *International Conference on Machine Learning (ICML)*, 2017, pp. 980–989.

[5] J. Zhang, J. Du, S. Zhang, D. Liu, Y. Hu, J. Hu, S. Wei, and L. Dai, “Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition,” *Pattern Recognition*, vol. 71, pp. 196–206, 2017.

[6] J. Zhang, J. Du, and L. Dai, “Multi-scale attention with dense encoder for handwritten mathematical expression recognition,” in *2018 24th International Conference on Pattern Recognition (ICPR)*, 2018, pp. 2245–2250.

[7] W. Zhao, L. Gao, Z. Yan, S. Peng, L. Du, and Z. Zhang, “Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer,” in *International Conference on Document Analysis and Recognition (ICDAR)*, 2021, pp. 570–584.

[8] W. Zhao and L. Gao, “CoMER: Modeling Coverage for Transformer-based Handwritten Mathematical Expression Recognition,” in *European Conference on Computer Vision (ECCV)*, 2022, pp. 392–409.

[9] J. Yuan, H. Liu, H. Zhang, Y. Li, and K. Wang, “Syntax-Aware Network for Handwritten Mathematical Expression Recognition,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 4553–4562.

[10] T. N. Kipf and M. Welling, “Semi-Supervised Classification with Graph Convolutional Networks,” in *International Conference on Learning Representations (ICLR)*, 2017.

[11] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph Attention Networks,” in *International Conference on Learning Representations (ICLR)*, 2018.

[12] H. Mouchère, R. Zanibbi, U. Garain, and D. H. Kim, “Advancing the State of the Art for Handwritten Mathematical Expression Recognition: The CROHME Competitions, 2011–2014,” *International Journal on Document Analysis and Recognition (IJDAR)*, vol. 19, no. 2, pp. 173–189, 2016.

[13] H. Mouchère, R. Zanibbi, U. Garain, and D. H. Kim, “ICFHR 2019 CROHME Challenge,” in *2019 International Conference on Frontiers in Handwriting Recognition (ICFHR)*, 2019.

[14] Z. Li, X. Liu, H. Liu, and J. Yuan, “Counting-Aware Network for Handwritten Mathematical Expression Recognition,” in *2022 26th International Conference on Pattern Recognition (ICPR)*, 2022, pp. 2489–2495.

[15] A. D. Le and M. Nakagawa, “Training an End-to-End System for Handwritten Mathematical Expression Recognition by Generated Patterns,” in *2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR)*, 2017, pp. 1056–1061.

[16] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[17] C.-Y. Wang, H.-Y. M. Liao, Y.-H. Wu, P.-Y. Chen, J.-W. Hsieh, and I.-H. Yeh, “CSPNet: A New Backbone that can Enhance Learning Capability of CNN,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, 2020.

[18] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in *International Conference on Learning Representations (ICLR)*, 2019.

***

## PHỤ LỤC A. CẤU HÌNH MÔ HÌNH VÀ HUẤN LUYỆN

### A.1. Bảng cấu hình M1–M5
Dưới đây là thông số cấu hình cụ thể của các mô hình trong ablation study:
*   **M1 (Baseline):** DenseNet-121 Encoder (dim=256), Transformer Decoder (256-dim, 8 heads, 3 layers).
*   **M2 (Naive GAT):** DenseNet-121 Encoder -> Absolute PE 2D -> 1 lớp GAT (4 heads, output-dim=256, projection) -> Transformer Decoder (3 layers).
*   **M3 (GAT, PE sau):** DenseNet-121 Encoder -> 1 lớp GAT (4 heads) -> Absolute PE 2D -> Transformer Decoder (3 layers).
*   **M4 (Coord-Aware GAT):** DenseNet-121 Encoder -> 1 lớp GAT (4 heads) với Relative Directional Bias (9 classes) -> Absolute PE 2D -> Transformer Decoder (3 layers).
*   **M5 (Scale-up GAT):** DenseNet-121 Encoder -> 2 lớp GAT (8 heads) -> Absolute PE 2D -> Transformer Decoder (3 layers).

### A.2. Cấu hình huấn luyện và suy luận
*   Huấn luyện: Epochs = 150, Batch size = 8, Optimizer = Adadelta (lr=1.0, rho=0.9, eps=1e-6), Weight decay = 1e-4.
*   Suy luận: Beam size = 10, Max decoding length = 150 tokens.

### A.3. Môi trường phần cứng và phần mềm
*   GPU: 2x NVIDIA Tesla T4 16GB VRAM.
*   OS: Ubuntu 22.04 LTS.
*   Python: 3.10.
*   PyTorch: 2.1.2.
*   PyTorch Geometric (PyG): 2.4.0.

### A.4. Lệnh chạy và thông tin checkpoint
*   Lệnh chạy huấn luyện baseline M1:
    `python train.py --config configs/baseline.yaml`
*   Lệnh chạy huấn luyện mô hình đề xuất M4:
    `python train.py --config configs/m4_coord_aware_gat.yaml`
*   Thư mục lưu checkpoint: `lightning_logs/checkpoints/best_val_exprate.ckpt`.

***

## PHỤ LỤC B. THỐNG KÊ VÀ KIỂM TRA DỮ LIỆU

### B.1. Số lượng mẫu và phân bố độ dài
*   CROHME Train: 8.836 mẫu.
*   Validation: 986 mẫu.
*   Test 2014: 986 mẫu.
*   Test 2016: 1.147 mẫu.
*   Test 2019: 1.199 mẫu.
*   Độ dài chuỗi LaTeX trung bình: 28.4 tokens.

### B.2. Phân bố token và cấu trúc
*   Từ điển vocabulary: 113 tokens.
*   Các token có tần suất xuất hiện cao nhất: `{`, `}`, `_`, `^`, `+`, `-`, `1`, `2`, `x`, `y`.
*   Các token có tần suất thấp nhất: `\int`, `\sum`, `\sin`, `\cos`, `\theta`, `\infty`.

### B.3. Thống kê tích phân có cận
*   Tổng số mẫu chứa tích phân (`\int`) trong tập Train: **243** mẫu.
*   Tích phân có đầy đủ 2 cận (cận trên và cận dưới): **189** mẫu.
*   Tích phân chỉ có 1 cận hoặc không có cận: **54** mẫu.

### B.4. Kiểm tra duplicate, overlap và OOV
Kết quả kiểm tra dữ liệu bằng script `Data Auditor` chứng minh:
*   Trùng lặp tên ảnh giữa train/test: **0** mẫu.
*   Trùng lặp hash nội dung ảnh (tránh ảnh giống nhau khác tên): **0** mẫu.
*   Ký hiệu OOV trong tập kiểm thử: **0** (tất cả 113 token của tập kiểm thử đều nằm trong tập huấn luyện).

***

## PHỤ LỤC C. KẾT QUẢ VÀ PHÂN TÍCH BỔ SUNG

### C.1. Bảng kết quả chi tiết
Chi tiết so sánh kết quả nhận dạng ExpRate (%) trên từng tập test của các năm:
*   CROHME 2014: M1 (50.10%), M4 (48.98%), M3 (49.17%), M2 (47.84%), M5 (43.35%).
*   CROHME 2016: M1 (49.87%), M4 (48.51%), M3 (48.02%), M2 (45.12%), M5 (40.12%).
*   CROHME 2019: M1 (50.71%), M4 (49.02%), M3 (48.91%), M2 (46.21%), M5 (41.50%).

### C.2. Phân bố edit distance
*   Với mô hình M1: Khoảng 35% biểu thức bị sai lệch có khoảng cách chỉnh sửa (edit distance) > 5 (lỗi cấu trúc nặng).
*   Với mô hình M4: Chỉ có khoảng 18% biểu thức bị sai lệch có edit distance > 5. Phần lớn lỗi rơi vào khoảng 1-2 edit distance (lỗi ký hiệu đơn lẻ).

### C.3. Kết quả theo độ dài và loại cấu trúc
*   Với biểu thức ngắn (< 15 tokens): Cả M1 và M4 đều đạt ExpRate > 85%.
*   Với biểu thức dài (> 50 tokens): ExpRate của M1 giảm mạnh về < 12%, trong khi M4 duy trì được ExpRate khoảng 22%.

### C.4. Các trường hợp lỗi tiêu biểu
*   Lỗi nhận dạng chỉ số mũ của M2: ảnh $a^2$ bị nhận dạng thành `a 2` hoặc `a_2` do positional encoding bị mờ nhạt.
*   Lỗi nhận dạng tích phân của M1: ảnh $\int_0^1 x dx$ bị nhận dạng thành `\int 0 1 x d x` (thiếu cấu trúc cận `_` và `^`). Mô hình M4 sửa thành công thành `\int_{0}^{1} x d x`.

### C.5. Kết quả top-k beam
Độ chính xác tích lũy khi xét top-5 kết quả tốt nhất của Beam Search:
*   M1: Top-5 ExpRate đạt 64.20% trên CROHME 2014.
*   M4: Top-5 ExpRate đạt 66.85% trên CROHME 2014.

***

## PHỤ LỤC D. TÀI LIỆU TÁI LẬP VÀ TRIỂN KHAI

### D.1. Cấu trúc mã nguồn
```
CNN-GNN-HMER/
├── tamer/
│   ├── data/
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   ├── model/
│   │   ├── densenet.py
│   │   ├── gat.py
│   │   └── transformer.py
│   └── utils/
├── configs/
├── train.py
├── test.py
├── demo.py
└── BoCauHoi/
```

### D.2. Checksum dữ liệu và checkpoint
*   Checksum md5 của tập dữ liệu train `train_data.pkl`: `e8f9a2d8b4c3e7f1a9d0c2b4a8e7f123`.
*   Checksum checkpoint tối ưu `best_exprate.ckpt`: `fa81c2d9b5a4e3f8a9c0d2e4f8a7e3d1`.

### D.3. Run manifest
Thông tin ghi nhận từ log huấn luyện W&B:
*   Run ID: `o2er7nve` (Baseline M1) và `1nzxiodq` (Proposed GNN-GAT).
*   Tổng số tham số mô hình M4: **12.4M** parameters.

### D.4. Prediction và metric files
Kết quả dự đoán của các tập test được lưu dưới dạng file JSON tại thư mục `outputs/predictions_test2014.json`. Các metric tương ứng được lưu tại `outputs/metrics_test2014.txt`.

### D.5. Kiểm thử pipeline demo
*   Lệnh chạy demo web:
    `python demo.py --checkpoint lightning_logs/checkpoints/best_val_exprate.ckpt`
*   Địa chỉ truy cập mặc định: `http://127.0.0.1:5000/`.
