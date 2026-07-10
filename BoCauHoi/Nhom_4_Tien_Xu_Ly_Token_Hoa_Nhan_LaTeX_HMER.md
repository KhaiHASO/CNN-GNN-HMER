# NHÓM 4 — TIỀN XỬ LÝ ẢNH, TOKEN HÓA VÀ NHÃN LATEX

> **Mục tiêu nhóm:** Giải thích được mọi biến đổi từ ảnh thô và nhãn thô đến tensor và chuỗi token mà mô hình thật sự nhìn thấy.

## Quy ước của tài liệu

- Công thức Markdown dùng `$...$` và `$$...$$`.
- Mọi mô tả “đã dùng” phải lấy từ code/config của đúng run.
- Phải tách **preprocessing dataset**, **transform trong dataloader** và **preprocessing của app**.
- Cuốn chuyên đề cũ không phải nguồn sự thật cho pipeline hiện tại.

---

# Câu 1 — Ảnh đầu vào được resize, padding và chuẩn hóa như thế nào?

## 1.1. Bản trả lời nhanh

> **Ảnh được đọc dưới dạng mảng raster một kênh. Nếu `scale_to_limit` bật, code resize giữ nguyên tỷ lệ để chiều cao nằm trong khoảng 16–256 và chiều rộng trong khoảng 16–1024; ảnh đang nằm trong giới hạn thì không bị resize. Trong mỗi batch, ảnh được pad bằng 0 đến chiều cao và chiều rộng lớn nhất của batch, đồng thời tạo mask đánh dấu vùng padding. Sau đó ảnh được chuyển thành tensor một kênh bằng `ToTensor()`.**
>
> **Repo hiện không cho thấy bước chuẩn hóa mean–standard deviation. Vì vậy em chỉ được nói có scale kích thước, chuyển tensor, zero-padding và mask; không được tự thêm các bước grayscale, contrast hoặc normalization nếu không có code/log chứng minh.**

## 1.2. Giải thích chi tiết

### a. Đầu vào mà loader nhận

Gói dữ liệu chứa ảnh trong `images.pkl`. Một sample được lấy bằng image ID trong `caption.txt`. `HMEDataset` áp dụng chuỗi transform lên từng ảnh.

Các giới hạn kích thước trong code:

```python
H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024
```

### b. Resize giữ tỷ lệ

Với ảnh có kích thước $H \times W$, nếu ảnh vượt giới hạn trên, hệ số thu nhỏ là:

$$ s_{\text{down}} = \min\left(\frac{256}{H},\frac{1024}{W}\right) $$

Nếu ảnh nhỏ hơn giới hạn dưới, hệ số phóng to là:

$$ s_{\text{up}} = \max\left(\frac{16}{H},\frac{16}{W}\right) $$

Cả hai chiều đều nhân cùng một hệ số, nên tỷ lệ hình học được giữ. Ảnh không bị ép méo về một khung cố định như $128 \times 512$.

Code dùng `cv2.INTER_LINEAR` cho cả thu nhỏ và phóng to. Đây là điểm có thể ảnh hưởng nét mảnh: interpolation tuyến tính có thể làm các pixel biên trở thành mức xám trung gian.

### c. Padding theo batch

Vì mỗi ảnh có kích thước khác nhau, collate tạo tensor:

$$ X \in \mathbb{R}^{B \times 1 \times H_{\max} \times W_{\max}} $$

Ảnh nhỏ được đặt vào góc trên trái và phần còn lại được điền 0. Một mask có dạng:

$$ M \in \{0,1\}^{B \times H_{\max} \times W_{\max}} $$

đánh dấu vùng thật và vùng padding. Trong code encoder, mask được downsample song song với feature map và được dùng để loại node padding khỏi attention.

### d. “Chuẩn hóa” cần dùng đúng nghĩa

Trong bảo vệ, từ “chuẩn hóa” có thể gây hiểu nhầm. Nên tách:

- chuẩn hóa kích thước: có;
- chuẩn hóa số kênh: đầu vào tensor một kênh;
- chuẩn hóa cường độ bằng `ToTensor()`: chuyển sang tensor và scale theo quy tắc torchvision;
- chuẩn hóa mean/std: chưa thấy;
- binarization, deskew, morphology: thuộc app nếu có, không phải pipeline train mặc định.

### e. Vấn đề cần audit

Cần lấy 20 ảnh train và ghi:

- dtype trước `ToTensor`;
- min, max, mean;
- polarity nền/nét;
- shape trước/sau resize;
- tỷ lệ foreground;
- tensor sau padding.

Không làm audit này thì rất dễ demo đảo màu hoặc scale khác train.

## 1.3. Câu hỏi truy tiếp và cách trả lời

**“Ảnh có bị cố định chiều cao 128 không?”**

> Không trong loader train hiện tại. Loader chỉ đảm bảo $16 \le H \le 256$ và $16 \le W \le 1024$, giữ tỷ lệ. Nếu app demo có resize về chiều cao 128 thì đó là một pipeline khác và phải kiểm tra tính nhất quán.

**“Padding bằng 0 có chắc là nền không?”**

> Chưa chắc nếu chưa kiểm tra polarity. Mask giúp attention bỏ vùng padding, nhưng các lớp CNN ban đầu vẫn nhận pixel pad. Cần xác nhận pixel 0 tương ứng gần nền train hoặc đánh giá ảnh hưởng biên padding.

## 1.4. Không nên nói

- “Mọi ảnh được resize về cùng kích thước.”
- “Ảnh luôn được cố định chiều cao 128.”
- “Có normalization mean/std” khi chưa thấy code.
- “Padding không thể ảnh hưởng vì đã có mask.”
- “Ảnh chắc chắn nền đen nét trắng” nếu chưa audit dữ liệu thật.

## 1.5. Bằng chứng cần chỉ ra

- `tamer/datamodule/dataset.py`: các hằng số 16, 256, 16, 1024 và chuỗi transform.
- `tamer/datamodule/transforms.py`: công thức scale giữ tỷ lệ và `INTER_LINEAR`.
- `tamer/datamodule/datamodule.py`: zero-padding và mask.
- Một file audit ảnh trước/sau transform do luận văn tự xuất.

---

# Câu 2 — Quy trình tiền xử lý lúc demo có giống hoàn toàn lúc train và test không?

## 2.1. Bản trả lời nhanh

> **Không được mặc định là giống. Train/test đọc trực tiếp ảnh đã đóng gói trong `images.pkl`, dùng `ScaleToLimitRange`, `ToTensor`, padding và mask. Demo lại có thể đi qua upload/canvas, crop, grayscale, threshold, đảo màu, deskew hoặc resize riêng. Chỉ cần khác polarity, tỷ lệ, margin hoặc độ dày nét là đã tạo domain gap.**
>
> **Muốn khẳng định giống nhau, em phải đưa cùng một ảnh qua hai đường xử lý và so sánh tensor cuối cùng, không chỉ nhìn ảnh bằng mắt.**

## 2.2. Giải thích chi tiết

### a. Hai pipeline phải được xem là hai hệ thống độc lập cho đến khi kiểm chứng

**Đường benchmark:**

```text
images.pkl
→ ScaleToLimitRange
→ ToTensor
→ dynamic batching
→ zero-padding + mask
→ model
```

**Đường demo có thể là:**

```text
canvas/ảnh chụp
→ decode RGB
→ crop
→ remove background
→ grayscale
→ threshold
→ invert
→ deskew
→ resize
→ pad
→ model
```

Số bước nhiều hơn làm tăng nguy cơ mismatch.

### b. Các mismatch nguy hiểm

1. **Polarity:** train nét sáng trên nền tối nhưng demo ngược lại.
2. **Range:** train tensor $[0,1]$, demo truyền $[0,255]$.
3. **Channel:** train 1 channel, demo 3 channel.
4. **Crop:** demo crop sát làm mất dấu mũ; train có margin.
5. **Aspect ratio:** demo ép khung cố định.
6. **Interpolation:** demo dùng nearest/bicubic khác train.
7. **Foreground ratio:** nét demo dày hơn nhiều.
8. **Antialiasing:** canvas có biên mờ khác ảnh render CROHME.
9. **Padding position:** train đặt ở góc trên trái nhưng demo căn giữa.
10. **Threshold:** dấu chấm hoặc cận nhỏ bị xóa.

### c. Phép kiểm chứng đúng

Tạo một hàm preprocessing dùng chung cho cả offline evaluation và API. Sau đó:

- lấy một ảnh từ `images.pkl`;
- lưu tensor chuẩn từ dataloader;
- xuất ảnh đó thành PNG;
- gửi PNG qua API demo;
- lấy tensor ngay trước model;
- tính:

$$ \Delta = \frac{1}{CHW} \lVert X_{\text{loader}}-X_{\text{demo}} Vert_1 $$

Nếu shape khác, polarity khác hoặc $\Delta$ lớn, hai pipeline chưa tương đương.

### d. Kiểm thử hồi quy

Nên có test:

```python
assert model_input_from_api(test_png)
       == model_input_from_dataset(test_png)
```

Trong thực tế có sai số interpolation, nên dùng `torch.allclose` với tolerance rõ ràng.

### e. Cách trình bày trước hội đồng

> **Hiện em tách core recognizer và app preprocessing. Em chỉ gọi chúng tương thích sau khi có parity test trên tensor. Nếu chưa làm parity test, em ghi đây là hạn chế triển khai, không nói demo dùng đúng preprocessing chỉ vì cùng resize.**

## 2.3. Câu hỏi truy tiếp và cách trả lời

**“Demo nhận ảnh CROHME tốt nhưng ảnh tự vẽ kém thì sao?”**

> Khi ảnh CROHME đi qua chính API vẫn tốt, integration cơ bản đúng; phần còn lại nhiều khả năng là domain shift của nét và bố cục. Tuy vậy vẫn cần controlled perturbation để xác nhận.

**“Có thể chỉ nhìn ảnh normalized để kết luận giống không?”**

> Không. Hai ảnh nhìn giống nhưng range, antialiasing hoặc margin có thể khác. Bằng chứng tốt hơn là tensor statistics và output parity.

## 2.4. Không nên nói

- “Demo chắc chắn giống train.”
- “Cùng kích thước là cùng phân bố.”
- “Chỉ cần grayscale là đủ.”
- “Ảnh render được bằng mắt nên model sẽ nhận được.”

## 2.5. Bằng chứng cần chỉ ra

- Pipeline dataloader trong `dataset.py`, `transforms.py`, `datamodule.py`.
- Pipeline thực tế trong thư mục `App`.
- Bộ parity tests và các ảnh `raw/crop/binary/resized/model_input`.

---

# Câu 3 — Việc cố định chiều cao ảnh ảnh hưởng thế nào đến các ký hiệu nhỏ như cận tích phân hoặc chỉ số?

## 3.1. Bản trả lời nhanh

> **Khi biểu thức dài được co về một chiều cao cố định, toàn bộ ký hiệu cùng bị thu nhỏ. Cận tích phân, số mũ, chỉ số dưới và dấu chấm vốn chiếm ít pixel có thể chỉ còn vài pixel, bị interpolation làm mờ hoặc bị threshold xóa. Sau đó DenseNet còn downsample nhiều lần, nên tín hiệu nhỏ có thể mất trước khi đến GAT.**
>
> **Vì vậy không thể chỉ đổ cho decoder. Cần đo kích thước ký hiệu trước/sau resize và lưu feature hoặc accuracy riêng theo tỷ lệ kích thước.**

## 3.2. Giải thích chi tiết

### a. Tỷ lệ thu nhỏ

Nếu ảnh gốc cao $H_0$ và được đưa về $H_t$, hệ số:

$$ s=\frac{H_t}{H_0} $$

Một cận cao $h_b$ pixel sẽ còn:

$$ h_b'=s h_b $$

Nếu $h_b'$ chỉ còn 2–4 pixel, một lần downsampling stride 2 có thể làm biểu diễn gần biến mất.

### b. Chuỗi downsampling trong DenseNet

Encoder có:

- convolution stride 2;
- max pooling stride 2;
- hai transition dùng average pooling 2.

Kích thước không gian giảm xấp xỉ nhiều lần. Chi tiết nhỏ có thể bị:

- trộn với nền;
- trộn với ký hiệu chính;
- làm yếu contrast;
- mất vị trí tương đối.

GAT chỉ hoạt động sau DenseNet. Nó có thể truyền thông tin còn lại, nhưng không tái tạo chi tiết đã mất hoàn toàn.

### c. Biểu thức dài và rộng đặc biệt nguy hiểm

Giới hạn chiều rộng 1024 cũng có thể buộc ảnh rất dài thu nhỏ. Khi giữ tỷ lệ, chiều cao thực tế cũng giảm, dù chưa chạm giới hạn chiều cao.

Do đó lỗi không chỉ phụ thuộc “fixed height”, mà phụ thuộc:

- aspect ratio;
- scale factor;
- kích thước tuyệt đối của cận;
- số bước downsampling;
- threshold và interpolation.

### d. Thí nghiệm cần làm

1. Tách subset có superscript/subscript/integral bounds.
2. Với mỗi ảnh, tính scale factor.
3. Chia bin theo scale:
   - không resize;
   - $0.75  \lt  s \le 1$;
   - $0.5  \lt  s \le 0.75$;
   - $s \le 0.5$.
4. Tính ExpRate và recall của `_`, `^`.
5. Lưu ảnh trước/sau preprocessing.
6. Thử multi-scale inference hoặc tăng resolution.
7. So sánh feature map của cận trước và sau từng block.

### e. Giải pháp khả dĩ

- giữ chiều cao lớn hơn;
- adaptive resolution theo aspect ratio;
- multi-scale encoder;
- crop hoặc tile cho biểu thức rất dài;
- tránh threshold quá mạnh;
- auxiliary high-resolution branch;
- augment riêng các cận nhỏ;
- super-resolution chỉ khi có thí nghiệm kiểm chứng.

## 3.4. Không nên nói

- “Cận mất chắc chắn do dataset.”
- “GAT sẽ khôi phục ký hiệu nhỏ.”
- “Resize giữ tỷ lệ nên không mất thông tin.”
- “Chỉ cần tăng chiều cao là chắc chắn tốt hơn.”

## 3.5. Bằng chứng cần chỉ ra

- Sơ đồ downsampling trong `encoder.py`.
- Phân bố scale factor của dataset và demo.
- Ảnh trước/sau resize của các trường hợp có cận.
- Accuracy subset theo kích thước cận.

---

# Câu 4 — Token hóa LaTeX theo token, ký tự hay quy tắc nào?

## 4.1. Bản trả lời nhanh

> **Repo không token hóa chuỗi LaTeX thô bằng một tokenizer động trong lúc train. `caption.txt` đã được tách sẵn bằng khoảng trắng; loader lấy token đầu làm image ID và phần còn lại là danh sách token. Vì vậy đơn vị đánh giá và học là token đã chuẩn bị, không phải từng ký tự Unicode.**
>
> **Ví dụ `\frac`, `{`, `x`, `}`, `^` có thể là các token riêng. Muốn mô tả đầy đủ quy tắc token hóa gốc, em phải truy script tạo `caption.txt`; code hiện tại chỉ chứng minh cách đọc token đã tách.**

## 4.2. Giải thích chi tiết

### a. Token list trong loader

Một dòng có dạng:

```text
image_id \frac { x } { y }
```

Sau `split()`:

```python
image_id = parts[0]
tokens = parts[1:]
```

Do đó:

- whitespace là ranh giới token trong file đã chuẩn bị;
- macro như `\frac` là một token;
- `{` và `}` thường là token;
- `^` và `_` là token cấu trúc;
- ký tự nhiều chữ như `\alpha` không bị tách từng chữ.

### b. Không phải character-level thuần túy

Nếu character-level, `\frac` sẽ thành các ký tự `\`, `f`, `r`, `a`, `c`. Repo không làm vậy ở loader.

### c. Không phải subword/BPE

Không có:

- vocabulary learning BPE;
- SentencePiece;
- unigram tokenizer;
- byte tokenizer.

Vocabulary được đọc từ `dictionary.txt`.

### d. Tác động đến metric

Edit distance được tính trên danh sách `pred` và `gt`, nên là token-level nếu hai biến đó vẫn là list token. Sai một macro `\frac` tính như một phép thay thế token, không phải năm ký tự.

### e. Cần audit tokenizer gốc

Những câu hỏi chưa được code loader trả lời:

- `x^2` được canonicalize thành `x ^ { 2 }` hay không;
- macro tương đương được đổi về cùng dạng không;
- khoảng trắng trong `\operatorname` xử lý thế nào;
- alias như `\lt` và `<` có thống nhất không.

Phải truy script tạo data hoặc kiểm tra toàn bộ `caption.txt`.

## 4.3. Câu hỏi truy tiếp và cách trả lời

**“Một token có luôn tương ứng một ký hiệu trên ảnh không?”**

> Không. `\frac`, `{`, `}` hoặc token cấu trúc không nhất thiết có một glyph riêng; ngược lại một vùng ảnh có thể dẫn đến nhiều token cấu trúc.

**“Edit distance là ký tự hay token?”**

> Với code đang truyền list token vào `editdistance.eval`, đó là token-level. Cần xác nhận không có bước join chuỗi trước hàm.

## 4.4. Không nên nói

- “Token là từng ký tự.”
- “Mỗi token là một bounding box.”
- “Repo dùng BPE.”
- “Tokenizer tự normalize mọi LaTeX tương đương.”

## 4.5. Bằng chứng cần chỉ ra

- `caption.txt` trong gói dữ liệu.
- `extract_data()` hoặc loader đọc `split()`.
- `vocab.py`.
- `lit_tamer.py` nơi `pred` và `gt` được chuyển thành danh sách từ.

---

# Câu 5 — Các token đặc biệt như BOS, EOS, PAD và UNK có vai trò gì?

## 5.1. Bản trả lời nhanh

> **Repo định nghĩa `<pad>` index 0, `<sos>` index 1 và `<eos>` index 2. `<sos>` khởi động quá trình giải mã; `<eos>` báo kết thúc; `<pad>` làm các chuỗi trong batch có cùng chiều dài và được mask trong loss/attention. Repo không định nghĩa `<unk>`.**
>
> **Vì không có UNK, token ngoài dictionary gây lỗi khi mã hóa nhãn và model cũng không thể sinh ký hiệu ngoài vocabulary 113 lớp.**

## 5.2. Giải thích chi tiết

### BOS/SOS

BOS thường được gọi là beginning-of-sequence; repo dùng `<sos>`.

Decoder autoregressive bắt đầu với:

$$ y_0=\langle sos angle $$

rồi dự đoán token tiếp theo.

### EOS

Khi model dự đoán `<eos>`, hypothesis có thể dừng. Nếu EOS quá sớm:

- chuỗi bị thiếu;
- biểu thức dài bị cắt.

Nếu EOS quá muộn:

- sinh token thừa;
- chạm `max_len`.

### PAD

PAD dùng để:

- pad target sequence;
- tạo `tgt_key_padding_mask`;
- bỏ vị trí padding khỏi loss nếu `ce_loss` cấu hình đúng;
- không cho attention xem padding như token thật.

### UNK

Repo không có `<unk>` trong `CROHMEVocab`. `words2indices` dùng tra cứu trực tiếp:

```python
self.word2idx[w]
```

Token không tồn tại gây `KeyError`.

### Ranh giới cần phân biệt

- PAD là kỹ thuật batching, không phải token LaTeX.
- SOS/EOS là control token, không render ra công thức.
- UNK nếu có chỉ biểu diễn không biết, không giúp model nhận đúng ký hiệu mới.

## 5.4. Không nên nói

- “BOS và SOS là hai token khác nhau trong repo.”
- “Có UNK.”
- “PAD được tính như token đúng trong ExpRate.”
- “EOS chỉ dùng khi train, không dùng inference.”

## 5.5. Bằng chứng cần chỉ ra

- `tamer/datamodule/vocab.py`.
- `decoder.py`: target padding mask và beam search.
- Hàm tạo target hai chiều và hàm loss.

---

# Câu 6 — Nhãn LaTeX có được normalize trước khi train và đánh giá không?

## 6.1. Bản trả lời nhanh

> **Nhãn trong `caption.txt` đã ở dạng token hóa/canonicalized nào đó, nhưng repo hiện không cho thấy một hàm normalize LaTeX được gọi ngay trước train hoặc trước Exact Match. Loader đọc token trực tiếp; evaluation so sequence index/token trực tiếp.**
>
> **Vì vậy em chỉ được nói model học và đánh giá trên biểu diễn đã có trong dataset. Muốn khẳng định normalize những gì, phải truy script tạo caption hoặc xây một normalizer được version hóa.**

## 6.2. Giải thích chi tiết

### Hai nghĩa của normalization

1. **Dataset preparation normalization:** xảy ra trước khi repo đọc data.
2. **Evaluation-time normalization:** chuyển prediction và GT về canonical form trước khi so.

Code hiện tại chứng minh chắc phần đọc token; chưa chứng minh có normalizer semantic ở evaluation.

### Hậu quả

Hai chuỗi render giống nhau nhưng token khác có thể bị tính sai. Ví dụ:

```latex
x ^ 2
```

và:

```latex
x ^ { 2 }
```

Nếu dataset protocol yêu cầu braces, chuỗi thứ nhất không match chuỗi thứ hai.

### Cần xây normalization contract

Một tài liệu phải nêu rõ:

- bỏ hay giữ whitespace;
- chuẩn hóa braces;
- alias macro;
- `\left`/`\right`;
- `\mathrm`/`\operatorname`;
- optional braces;
- Unicode versus macro;
- cách xử lý `\!`, `\,`;
- thứ tự token.

### Không nên normalize quá mức

Không được biến:

```latex
x + 1
```

thành tương đương đại số với:

```latex
1 + x
```

vì đó không còn là recognition exact match mà là symbolic equivalence.

### Thực nghiệm nên báo cáo hai metric nếu cần

- Protocol Exact Match: theo token benchmark.
- Render/canonical equivalence: dùng parser/canonicalizer được định nghĩa riêng.

Không trộn hai metric.

## 6.4. Không nên nói

- “Mọi LaTeX tương đương đều được xem giống nhau.”
- “Whitespace chắc chắn bị bỏ.”
- “Exact Match đo tương đương toán học.”
- “Có normalize vì caption đã token hóa.”

## 6.5. Bằng chứng cần chỉ ra

- Script tạo `caption.txt`, nếu tìm được.
- Code `ExpRateRecorder`.
- Một tài liệu `normalization_spec.md`.
- Unit test với các cặp biểu diễn tương đương và không tương đương.

---

# Câu 7 — Hai biểu diễn tương đương như x^2 và x^{2} được xem là giống hay khác?

## 7.1. Bản trả lời nhanh

> **Theo Exact Match token-level, chúng chỉ được xem giống nếu bước chuẩn hóa biến cả hai về cùng chuỗi. Với pipeline hiện tại chưa thấy normalizer evaluation, nên nếu token sequence khác thì bị xem là khác, dù render giống.**
>
> **Em phải trả lời theo protocol, không theo cảm giác ngữ nghĩa. Có thể bổ sung canonical metric, nhưng metric benchmark chính vẫn phải giữ nhất quán với các nghiên cứu đối chứng.**

## 7.2. Giải thích chi tiết

Giả sử tokenizer tạo:

```text
x ^ 2
```

và:

```text
x ^ { 2 }
```

Hai dãy có độ dài khác nhau. Exact Match:

$$ \mathbf{1}[\hat{Y}=Y] $$

sẽ bằng 0 nếu không canonicalize.

### Tại sao benchmark thường nghiêm?

- giúp metric đơn giản và tái lập;
- tránh parser khác nhau;
- giữ đúng annotation protocol;
- không cần quyết định tương đương semantic.

### Nhưng có hạn chế

- phạt sự khác biệt presentation không ảnh hưởng render;
- làm score phụ thuộc canonicalization;
- có thể đánh giá thấp khả năng sử dụng thực tế.

### Cách xử lý tốt

Báo cáo:

1. token Exact Match chính thức;
2. normalized Exact Match nếu có;
3. syntax validity;
4. rendered equivalence trên subset.

Normalizer phải:

- deterministic;
- công khai;
- áp dụng đồng đều cho prediction và GT;
- không dùng thông tin ground truth để sửa prediction.

## 7.4. Không nên nói

- “Chắc chắn giống vì toán học giống.”
- “Chắc chắn khác trong mọi benchmark.”
- “Có thể sửa prediction theo GT trước khi chấm.”

## 7.5. Bằng chứng cần chỉ ra

- Ví dụ token thực tế trong `caption.txt`.
- Output của `ExpRateRecorder`.
- Unit test normalizer nếu bổ sung.

---

# Câu 8 — Augmentation nào thật sự được áp dụng trong code, với xác suất và tham số bao nhiêu?

## 8.1. Bản trả lời nhanh

> **Code dataset chỉ có một augmentation được khai báo rõ là `ScaleAugmentation` với hệ số trong khoảng 0,7–1,4, và chỉ áp dụng khi `is_train` cùng `scale_aug=True`. Tuy nhiên cấu hình CROHME hiện tại đặt `scale_aug: false`, nên run theo config đó không dùng scale augmentation.**
>
> **`ScaleToLimitRange` không nên gọi là augmentation ngẫu nhiên; đó là preprocessing giữ kích thước trong giới hạn. Repo không cho thấy random flip, rotation, Gaussian noise hoặc stroke-width augmentation trong pipeline train chính.**

## 8.2. Giải thích chi tiết

### Phân biệt augmentation và preprocessing

**Augmentation:**

- ngẫu nhiên;
- chỉ train;
- tạo biến thể cùng nhãn.

**Preprocessing:**

- deterministic;
- train/val/test;
- đưa dữ liệu vào miền kích thước hợp lệ.

Trong code:

```python
if is_train and scale_aug:
    ScaleAugmentation(0.7, 1.4)
```

sau đó:

```python
if scale_to_limit:
    ScaleToLimitRange(...)
```

### Cần xác nhận distribution của ScaleAugmentation

Phải đọc chính xác implementation để biết:

- lấy uniform hay distribution khác;
- luôn áp dụng hay có xác suất;
- scale isotropic hay anisotropic;
- interpolation.

Không nên tự nói xác suất 50% nếu code không có Bernoulli.

### Config hiện tại

```yaml
scale_aug: false
scale_to_limit: true
```

Vì vậy khi mô tả run M1–M5 phải lấy config lưu cùng run; config main có thể không phản ánh override trên Kaggle.

### Vì sao điều này quan trọng?

Nếu demo có domain shift nét và scale nhưng train không augment, khả năng tổng quát ngoài benchmark có thể thấp.

## 8.4. Không nên nói

- “Có augmentation mạnh.”
- “Dùng lật ngang, noise, rotation.”
- “ScaleToLimitRange là augmentation.”
- “Scale augmentation chắc chắn chạy trong M1–M5” nếu config run chưa đối chiếu.

## 8.5. Bằng chứng cần chỉ ra

- `dataset.py`.
- `transforms.py`.
- `config/crohme.yaml` của từng run trong `KetQua`.
- Log hyperparameter được lưu cùng checkpoint.

---

# Câu 9 — Vì sao một số augmentation như lật ngang có thể không phù hợp với biểu thức toán học?

## 9.1. Bản trả lời nhanh

> **Augmentation chỉ hợp lệ khi phép biến đổi bảo toàn nhãn. Lật ngang thường không bảo toàn biểu thức: thứ tự trái–phải bị đảo, dấu ngoặc đổi hướng, ký hiệu bất đối xứng thay đổi và bố cục của hàm hoặc giới hạn bị phá. Nếu vẫn giữ nguyên LaTeX thì tạo cặp ảnh–nhãn sai.**
>
> **Chỉ nên dùng biến đổi bảo toàn ngữ nghĩa đã được kiểm chứng, như scale nhẹ, affine nhỏ, thay đổi độ dày nét hoặc nhiễu vừa phải.**

## 9.2. Giải thích chi tiết

### Điều kiện của augmentation hợp lệ

Với phép biến đổi $T$:

$$ Y(T(X)) = Y(X) $$

Nếu điều này không đúng, augmentation làm hỏng supervision.

### Lật ngang gây gì?

- chuỗi đọc trái sang phải bị đảo;
- `(` thành hình giống `)`;
- `<` thành `>`;
- vị trí chỉ số có thể chuyển phía;
- các macro không còn tương ứng;
- phân bố không giống cách con người viết.

### Lật dọc còn nguy hiểm hơn

- superscript thành subscript;
- tử và mẫu đổi vị trí;
- cận trên thành cận dưới.

### Những biến đổi có thể phù hợp hơn

- scale isotropic nhẹ;
- translation trong canvas;
- shear/rotation rất nhỏ;
- elastic distortion có giới hạn;
- stroke width;
- blur/noise nhẹ;
- random erasing rất thận trọng;
- synthetic handwriting styles.

Mỗi augmentation cần test label preservation và ablation.

### Không phải augmentation càng mạnh càng tốt

Một augmentation làm ảnh đa dạng nhưng lệch quy luật toán học có thể:

- làm loss khó giảm;
- giảm Exact Match;
- khiến model học invariance sai;
- xóa positional cues quan trọng.

## 9.4. Không nên nói

- “Flip luôn giúp tăng dữ liệu.”
- “Ảnh toán học cũng giống ảnh vật thể nên dùng augmentation ImageNet.”
- “Lật rồi đảo chuỗi LaTeX là đủ” — nhiều cấu trúc không đơn giản như đảo token.

## 9.5. Bằng chứng cần chỉ ra

- Ví dụ trực quan 5 công thức trước/sau flip.
- Ablation từng augmentation.
- Kiểm tra bằng renderer/parser xem nhãn có còn phù hợp.

---

# Câu 10 — Nếu cận trên hoặc cận dưới biến mất sau resize, lỗi nên được quy cho dataset, preprocessing hay mô hình?

## 10.1. Bản trả lời nhanh

> **Không nên quy ngay cho một phía. Nếu cận tồn tại trong ảnh và nhãn gốc nhưng biến mất ở ảnh sau preprocessing, lỗi trực tiếp thuộc preprocessing; dataset chỉ góp phần nếu cấu trúc quá hiếm hoặc ảnh gốc đã quá nhỏ. Nếu cận vẫn rõ trong tensor nhưng model không sinh được, lỗi nghiêng về representation hoặc decoder.**
>
> **Cần chẩn đoán theo chuỗi bằng chứng: ảnh gốc → ảnh sau transform → feature map → beam output → token error.**

## 10.2. Giải thích chi tiết

### Ma trận quy trách nhiệm

| Quan sát | Nguyên nhân chính cần nghi |
|---|---|
| Ảnh gốc không có/nhãn sai | Dataset annotation |
| Ảnh gốc có, sau resize mất | Preprocessing |
| Sau resize rõ, feature không phản ứng | Encoder/downsampling |
| Feature có tín hiệu, decoder bỏ `_`/`^` | Decoder/language bias |
| Train hầu như không có mẫu tương tự | Data imbalance |
| Chỉ demo sai, benchmark subset tốt | Domain shift/integration |

### Quy trình kiểm chứng

1. Mở ảnh và caption gốc.
2. Lưu ảnh sau từng transform.
3. Zoom cận và đo foreground pixels.
4. So sánh activation hoặc saliency vùng cận.
5. Xem top-k beam:
   - có hypothesis đúng cận nhưng score thấp không;
   - hay tất cả hypothesis đều bỏ cận.
6. Tính accuracy subset tích phân hai cận.
7. Thử tăng resolution mà giữ nguyên checkpoint, sau đó retrain nếu cần.

### Phân biệt nguyên nhân trực tiếp và nguyên nhân gốc

Ví dụ preprocessing xóa cận là nguyên nhân trực tiếp. Nhưng nguyên nhân gốc có thể là:

- app resize khác train;
- ảnh quá dài;
- threshold không phù hợp;
- thiết kế resolution không đủ.

Câu trả lời tốt cần nêu cả hai tầng.

## 10.4. Không nên nói

- “Dataset không có.”
- “Model không học.”
- “Người dùng viết nhỏ.”
- “Resize giữ tỷ lệ nên chắc chắn không mất.”

## 10.5. Bằng chứng cần chỉ ra

- Bộ ảnh forensic trước/sau.
- Pixel/foreground statistics.
- Activation map.
- Token-level error và beam candidates.
- Tần suất cấu trúc trong train.

---

# Phụ lục A — Checklist parity train/test/demo

- [ ] Cùng số kênh.
- [ ] Cùng polarity.
- [ ] Cùng range và dtype.
- [ ] Cùng quy tắc resize.
- [ ] Cùng cách giữ tỷ lệ.
- [ ] Cùng padding và vị trí đặt ảnh.
- [ ] Cùng mask.
- [ ] Không có threshold riêng làm mất nét.
- [ ] Test tensor parity bằng code.
- [ ] Lưu ảnh trung gian cho mọi lỗi demo quan trọng.

# Phụ lục B — Nguồn đối chiếu

- `tamer/datamodule/dataset.py`
- `tamer/datamodule/transforms.py`
- `tamer/datamodule/datamodule.py`
- `tamer/datamodule/vocab.py`
- `tamer/model/encoder.py`
- `tamer/model/decoder.py`
- các file preprocessing thực tế trong `App`
