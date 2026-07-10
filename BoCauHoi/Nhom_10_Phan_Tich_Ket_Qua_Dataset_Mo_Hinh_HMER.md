# NHÓM 10 — PHÂN TÍCH KẾT QUẢ GIỮA CÁC DATASET VÀ MÔ HÌNH

> **Mục tiêu nhóm:** Biết đọc bảng kết quả, tách quan sát khỏi giả thuyết và đưa ra cách kiểm chứng nguyên nhân.

---

## 0. Quy tắc ba tầng khi phân tích kết quả

Mọi câu trả lời nên theo cấu trúc:

1. **Quan sát:** con số nào cao/thấp.
2. **Diễn giải giới hạn:** metric đó nói được gì.
3. **Giả thuyết và phép kiểm chứng:** vì sao có thể xảy ra và cần thí nghiệm gì.

Ví dụ:

> **Quan sát:** M4 có Mean Edit Distance trung bình 2,06, thấp hơn M1 là 2,20.  
> **Diễn giải:** prediction của M4 cần ít phép sửa token hơn trung bình.  
> **Giới hạn:** M4 vẫn có Avg ExpRate thấp hơn M1.  
> **Kiểm chứng:** xem histogram distance, cấu trúc lỗi và nhiều seed.

---

# Câu 1 — Mô hình nào có ExpRate cao nhất trên từng test set và trung bình toàn bộ?

## 1.1. Bản trả lời nhanh

> **Trên CROHME 2014, M1 cao nhất với 51,12%. Trên CROHME 2016, M3 cao nhất với 50,74%, chỉ nhỉnh hơn M1 50,65% đúng 0,09 điểm phần trăm. Trên CROHME 2019, M1 cao nhất với 48,54%. Trung bình ba tập, M1 cao nhất với 50,10%.**

## 1.2. Bảng xếp hạng

| Phạm vi | Mô hình cao nhất | ExpRate |
|---|---|---:|
| CROHME 2014 | M1 | 51,12% |
| CROHME 2016 | M3 | 50,74% |
| CROHME 2019 | M1 | 48,54% |
| Trung bình | M1 | 50,10% |

### Diễn giải đúng

- M1 thắng hai trong ba tập và thắng trung bình.
- M3 có một kết quả cục bộ tốt trên 2016.
- Chênh lệch 0,09 điểm phần trăm chưa đủ gọi có ý nghĩa nếu chỉ một seed.
- M4 không phải model có Exact Match tốt nhất.

## 1.3. Không nên nói

- “M3 vượt baseline.”
- “M4 chính xác nhất.”
- “M1 thắng mọi dataset.”
- “0,09% là cải thiện chắc chắn.”

---

# Câu 2 — Mô hình nào có Mean Edit Distance thấp nhất?

## 2.1. Bản trả lời nhanh

> **Trên CROHME 2014 và 2016, M4 thấp nhất lần lượt 1,98 và 2,13. Trên CROHME 2019, M3 thấp nhất với 2,02. Xét trung bình, M4 thấp nhất với 2,06.**

## 2.2. Bảng

| Phạm vi | Model MED thấp nhất | MED |
|---|---|---:|
| 2014 | M4 | 1,98 |
| 2016 | M4 | 2,13 |
| 2019 | M3 | 2,02 |
| Trung bình | M4 | 2,06 |

### Lưu ý

M1 MED 1,99 trên 2014 chỉ kém M4 0,01. Không có nhiều seed thì chênh lệch này có thể chỉ là dao động.

### Kết luận đúng

> **M4 có error severity trung bình tốt nhất, không đồng nghĩa Exact Match tốt nhất.**

---

# Câu 3 — Kết quả chính mà luận văn có thể khẳng định chắc chắn từ bảng M1–M5 là gì?

## 3.1. Bản trả lời nhanh

> **Có bốn quan sát chắc chắn trong các run hiện có: M1 có Avg ExpRate cao nhất; M2 thấp hơn M1; M3 phục hồi so với M2; M4 có Avg MED thấp nhất; M5 giảm mạnh.**
>
> **Từ đó có thể kết luận thứ tự PE và độ sâu GAT ảnh hưởng kết quả. Chưa thể khẳng định GAT tốt hơn baseline, relative bias một mình tạo cải thiện, hoặc nguyên nhân M5 là over-smoothing.**

## 3.2. Phân biệt observation và causal claim

### Quan sát trực tiếp

- metric từng run;
- ranking;
- chênh lệch.

### Suy luận được hỗ trợ

- PE sau tốt hơn PE trước trong M2–M3 nếu các yếu tố khác giống.
- Scale-up M5 không hiệu quả trong cấu hình hiện tại.

### Chưa chứng minh

- PE blurring;
- over-smoothing;
- relative bias là nguyên nhân riêng;
- cải thiện cấu trúc semantic;
- significance.

### Câu chốt

> **Bảng chứng minh sensitivity to design, không chứng minh superiority of GAT.**

---

# Câu 4 — Vì sao không được kết luận một mô hình tốt nhất chỉ dựa trên một metric?

## 4.1. Bản trả lời nhanh

> **Mỗi metric đo một mục tiêu khác nhau. ExpRate đo số biểu thức đúng hoàn toàn; MED đo số token cần sửa trung bình; latency và parameter count đo khả năng triển khai. Một model có thể thắng metric này nhưng thua metric khác.**
>
> **Vì vậy phải nói “tốt nhất theo ExpRate” hoặc “tốt nhất theo MED”, không nói chung chung “tốt nhất”.**

## 4.2. Ví dụ M1 và M4

- M1 Avg ExpRate: 50,10%.
- M4 Avg ExpRate: 48,98%.
- M1 MED: 2,20.
- M4 MED: 2,06.

Nếu ứng dụng yêu cầu output tự động không sửa:

- ưu tiên M1.

Nếu có người hậu kiểm:

- M4 có thể giảm số thao tác sửa.

### Còn thiếu các metric triển khai

- inference time;
- VRAM;
- parameter count;
- syntax validity;
- robustness;
- confidence.

## 4.3. Không nên nói

- “MED là metric quan trọng nhất.”
- “Exact Match là metric duy nhất có ý nghĩa.”
- “M4 tốt nhất.”
- “M1 tốt nhất mọi mặt.”

---

# Câu 5 — Vì sao cùng một mô hình lại cao trên CROHME 2016 nhưng thấp trên 2014 hoặc 2019?

## 5.1. Bản trả lời nhanh

> **Một model tương tác với phân bố của từng test set. Các năm có thể khác về độ dài, loại cấu trúc, token hiếm, người viết, kích thước ảnh và mức giống train. Thiết kế GAT/PE có thể hợp hơn với một phân bố nhưng không ổn định trên phân bố khác.**
>
> **Hiện chưa có thống kê đầy đủ, nên đây là nhóm giả thuyết; không được nói 2016 dễ hơn chỉ vì M3 cao.**

## 5.2. Các cơ chế có thể

### Covariate shift

$$ P_{2014}(X)\ne P_{2016}(X)\ne P_{2019}(X) $$

### Label/structure shift

$$ P_{2014}(Y)\ne P_{2016}(Y)\ne P_{2019}(Y) $$

### Conditional shift

Cùng cấu trúc nhưng cách viết khác:

$$ P(X\mid Y) $$

### Model-selection effect

CROHME 2014 được dùng validation trong config hiện tại, nên score và ranking có thể chịu selection bias khác 2016/2019.

### Seed variance

Một run duy nhất có thể đổi ranking nhỏ.

## 5.3. Cách kiểm chứng

- length distribution;
- structure frequency;
- token frequency;
- image statistics;
- writer metadata;
- matched subsets;
- nhiều seed;
- per-category accuracy.

---

# Câu 6 — Khác biệt độ dài biểu thức giữa các test set có thể ảnh hưởng kết quả thế nào?

## 6.1. Bản trả lời nhanh

> **Biểu thức dài có nhiều cơ hội sai token hơn, nhiều cấu trúc lồng hơn và chịu exposure bias lâu hơn. Vì Exact Match yêu cầu toàn chuỗi đúng, xác suất đúng thường giảm nhanh theo độ dài. Max decode length 150 còn có thể cắt các mẫu rất dài.**

## 6.2. Trực giác xác suất

Nếu xác suất đúng token trung bình là $p$ và giả sử độc lập đơn giản:

$$ P(\text{toàn chuỗi đúng})\approx p^L $$

Khi $L$ tăng, Exact Match giảm ngay cả khi token accuracy không đổi.

Ví dụ với $p=0,98$:

$$ 0,98^{10}\approx0,817 $$

$$ 0,98^{50}\approx0,364 $$

Đây chỉ là minh họa vì token không độc lập.

### Các tác động khác

- nhiều braces;
- nhiều EOS risk;
- beam search space lớn;
- memory cross-attention dài;
- nested context.

### Phép phân tích

Tính ExpRate theo bin độ dài và chuẩn hóa khi so các dataset.

---

# Câu 7 — Phân bố loại cấu trúc như phân số, căn, chỉ số và tích phân có thể gây chênh lệch ra sao?

## 7.1. Bản trả lời nhanh

> **Mỗi cấu trúc đòi hỏi loại quan hệ khác nhau. Phân số cần tách tử–mẫu; căn cần xác định phạm vi; chỉ số và cận cần nhận ký hiệu nhỏ; tích phân có cận kết hợp cả token và vị trí. Nếu một test set có nhiều cấu trúc hiếm hoặc lồng hơn, model có thể giảm.**
>
> **Nhưng phải đếm cấu trúc trước khi dùng nó giải thích chênh lệch.**

## 7.2. Các failure mode

| Cấu trúc | Failure mode |
|---|---|
| Fraction | đảo tử/mẫu, thiếu braces |
| Root | kết thúc phạm vi sai |
| Superscript | mất ký hiệu nhỏ, nhầm `_` |
| Subscript | nhầm `^`, dính baseline |
| Integral bounds | mất cận hoặc nội dung cận |
| Nested | sai closing order |

### Phân tích cần làm

- expression-level subset;
- occurrence count;
- nesting depth;
- matched length;
- M1 versus M4 per structure.

Không được gộp mọi mẫu có `^` thành “số mũ”, vì có thể là cận trên.

---

# Câu 8 — Tần suất token hiếm hoặc ký hiệu dễ nhầm ảnh hưởng thế nào đến từng test set?

## 8.1. Bản trả lời nhanh

> **Token hiếm nhận ít cập nhật gradient nên xác suất và calibration thường kém hơn. Nếu test set có tỷ lệ token hiếm hoặc cặp hình dạng dễ nhầm cao, Exact Match có thể giảm. Một token hiếm sai cũng làm cả biểu thức sai.**
>
> **Cần tính tần suất train của mỗi token và error rate có điều kiện, không chỉ liệt kê ví dụ.**

## 8.2. Phân tích đề xuất

Với token $v$:

$$ f_{\text{train}}(v) $$

và error rate:

$$ E(v) = P(v\text{ bị sai}\mid v\text{ xuất hiện trong GT}) $$

Chia frequency bin:

- 1–5;
- 6–20;
- 21–100;
- >100.

### Confusion cần alignment

Ví dụ:

- `1` / `l`;
- `0` / `O`;
- `x` / `\times`;
- `-` / fraction bar;
- `c` / `(`.

Không được suy confusion chỉ từ string bag; cần Levenshtein alignment hoặc parser.

---

# Câu 9 — Làm sao phân biệt cải thiện thật với dao động ngẫu nhiên do seed?

## 9.1. Bản trả lời nhanh

> **Chạy nhiều seed với cùng protocol, báo cáo mean ± standard deviation và confidence interval. Nếu chênh lệch giữa model nhỏ hơn độ biến thiên run-to-run, không nên gọi là cải thiện ổn định.**
>
> **Chênh 0,09 điểm phần trăm M3–M1 trên 2016 đặc biệt cần kiểm tra.**

## 9.2. Thiết kế

- ít nhất 3 seed, tốt hơn 5;
- cùng seed list cho mọi model;
- paired comparison;
- lưu checkpoint/config.

### Paired difference

Với cùng seed $s$:

$$ \Delta_s = M3_s-M1_s $$

Phân tích mean/std của $\Delta_s$ tốt hơn so hai tập run không ghép.

### Bootstrap trên test samples

Ngoài seed variance, bootstrap estimate uncertainty do finite test set.

### Hai nguồn uncertainty

1. training randomness;
2. sampling uncertainty của test.

Cần phân biệt.

---

# Câu 10 — Cần báo cáo độ lệch chuẩn hoặc khoảng tin cậy trong trường hợp nào?

## 10.1. Bản trả lời nhanh

> **Độ lệch chuẩn cần khi có nhiều lần train với seed khác nhau. Khoảng tin cậy bootstrap có thể báo cáo cho metric trên một test set ngay cả với một checkpoint, nhưng nó chỉ phản ánh uncertainty do sample, không phản ánh training variance.**
>
> **Khi so các model chênh lệch nhỏ, cả hai loại uncertainty đều quan trọng.**

## 10.2. Báo cáo đề xuất

```text
ExpRate = 49,20 ± 0,35 (3 seeds)
95% bootstrap CI = [48,1; 50,3]
```

Không nên nhầm:

- std với CI;
- CI sample với seed variability.

### Khi một run duy nhất

Có thể bootstrap paired prediction, nhưng phải ghi rõ:

> **Single-checkpoint confidence interval; không bao gồm training randomness.**

---

# Câu 11 — Nếu M4 giảm lỗi trung bình nhưng Exact Match không tăng, điều đó nói gì về phân bố lỗi?

## 11.1. Bản trả lời nhanh

> **Nó gợi ý mass của phân bố lỗi dịch từ khoảng cách lớn về khoảng cách nhỏ, nhưng số mẫu ở distance 0 không tăng. Có thể nhiều mẫu từ 3–5 lỗi xuống 1–2 lỗi.**
>
> **Muốn xác nhận phải xem histogram hoặc CDF edit distance, không chỉ mean.**

## 11.2. Các kịch bản

### Kịch bản A

- exact giảm nhẹ;
- distance 1 tăng nhiều;
- tail >5 giảm.

Đây là near-correct trade-off.

### Kịch bản B

- vài outlier cực lớn được sửa;
- phần lớn mẫu không đổi.

Mean giảm nhưng ảnh hưởng application có thể khác.

### Kịch bản C

- sequence ngắn tốt hơn;
- sequence dài tệ hơn;
- mean tổng che subgroup.

### Phép kiểm tra

- count distance 0/1/2/3–5/>5;
- median/P90;
- paired per-sample delta:

$$ \Delta d_i=d_{M4,i}-d_{M1,i} $$

---

# Câu 12 — Có thể dùng histogram edit distance để giải thích kết quả như thế nào?

## 12.1. Bản trả lời nhanh

> **Histogram cho biết metric trung bình được tạo bởi nhóm gần đúng hay outlier. Ta có thể thấy model nào có nhiều exact, nhiều distance 1–2, hoặc tail lỗi lớn. CDF còn cho thấy trực tiếp tỷ lệ ≤k.**

## 12.2. Biểu đồ nên làm

### Histogram

Bins:

- 0;
- 1;
- 2;
- 3;
- 4–5;
- 6–10;
- >10.

### CDF

$$ F(k)=P(d\le k) $$

- $F(0)$ là ExpRate;
- $F(1)$ là ≤1;
- $F(2)$ là ≤2.

### Paired delta plot

- $\Delta d \lt 0$: M4 tốt hơn M1;
- $\Delta d=0$: bằng nhau;
- $\Delta d \gt 0$: M4 tệ hơn.

### Stratify

- theo dataset;
- độ dài;
- cấu trúc;
- scale ảnh.

---

# Câu 13 — Có được nói test set 2019 khó hơn chỉ vì ExpRate thấp hơn không?

## 13.1. Bản trả lời nhanh

> **Không. Chỉ được nói các model hiện tại đạt ExpRate thấp hơn trên 2019. Độ khó là thuộc tính tương tác giữa dataset và model; cần nhiều model, matched analysis và thống kê phân bố trước khi xếp hạng.**

## 13.2. Bằng chứng mạnh hơn cần có

- đa số model cùng giảm;
- độ dài/cấu trúc phức tạp hơn;
- rare token cao;
- image quality khác;
- CI không chồng đáng kể;
- matched subsets vẫn thấp.

### Cách nói đúng

> **CROHME 2019 là tập mà các cấu hình hiện tại cho kết quả thấp hơn, đặc biệt M5. Nguyên nhân chưa được xác định.**

---

# Câu 14 — Có được quy toàn bộ suy giảm của M5 cho over-smoothing khi chưa có phép đo node similarity không?

## 14.1. Bản trả lời nhanh

> **Không. Over-smoothing chỉ là một giả thuyết. M5 đồng thời tăng layer và head; suy giảm còn có thể do optimization, overfitting, dropout, head dimension, scheduler hoặc seed.**
>
> **Muốn chứng minh over-smoothing cần đo mức giống nhau của node embedding theo layer và liên hệ với performance.**

## 14.2. Phép đo

Mean pairwise cosine:

$$ S^{(l)} = \frac{1}{N(N-1)} \sum_{i\ne j} \cos(h_i^{(l)},h_j^{(l)}) $$

Nếu $S^{(l)}$ tăng mạnh và spatial discriminability giảm, có bằng chứng hỗ trợ.

Có thể dùng:

- rank/effective rank;
- feature variance;
- coordinate linear probe;
- edge perturbation;
- compare 2L no-dropout.

### Câu kết luận đúng

> **Kết quả M5 nhất quán với một số cơ chế như over-smoothing, nhưng chưa phân biệt được nguyên nhân.**

---

# Câu 15 — Có được nói GAT tốt hơn CNN/Transformer nếu M1 vẫn cao hơn trên nhiều tập không?

## 15.1. Bản trả lời nhanh

> **Không. Dữ liệu hiện tại không hỗ trợ tuyên bố GAT tốt hơn về Exact Match. M1 cao nhất trung bình và trên hai test set. Chỉ có thể nói GAT thay đổi trade-off; M4 giảm MED và M3 cho thấy PE order quan trọng.**

## 15.2. Tuyên bố hợp lệ

- “M4 có MED thấp hơn M1.”
- “M3 nhỉnh hơn M1 trên 2016.”
- “GAT integration chưa tạo cải thiện Exact Match ổn định.”
- “Design của GAT rất nhạy với PE/depth.”

### Muốn nói GAT tốt hơn cần

- cùng budget;
- nhiều seed;
- significance;
- thắng metric mục tiêu;
- không cherry-pick;
- parameter/latency analysis.

---

# Câu 16 — Khi kết quả không ủng hộ giả thuyết ban đầu, nên trình bày thế nào để vẫn có giá trị khoa học?

## 16.1. Bản trả lời nhanh

> **Trình bày giả thuyết ban đầu, thiết kế kiểm chứng, kết quả thực tế và giới hạn một cách trung thực. Negative result có giá trị nếu nó bác bỏ một giả định hợp lý và rút ra điều kiện thiết kế cụ thể.**
>
> **Không đổi metric sau khi thấy kết quả hoặc bịa nguyên nhân.**

## 16.2. Mẫu trình bày

> **Giả thuyết:** thêm GAT sẽ tăng Exact Match.  
> **Quan sát:** M2 giảm; M3 phục hồi; M4 chỉ cải thiện MED; M5 giảm mạnh.  
> **Kết luận:** GAT không tự động cải thiện và phụ thuộc PE/depth.  
> **Giới hạn:** một seed, M4 confounded.  
> **Thí nghiệm tiếp:** tách bias/layer/head, nhiều seed.

### Vì sao vẫn có giá trị?

- ngăn người khác lặp thiết kế thất bại;
- làm rõ boundary conditions;
- tạo insight;
- tăng tính trung thực;
- hướng thí nghiệm tiếp theo.

### Không biến negative result thành chiến thắng giả

Không nói:

> “M4 tốt nhất vì em chọn MED.”

Mà nói:

> “Giả thuyết Exact Match chưa được xác nhận; phát hiện phụ là error severity giảm.”

---

# Phụ lục A — Bản trả lời tổng hợp khoảng hai phút

> **M1 có ExpRate cao nhất trên 2014, 2019 và trung bình; M3 chỉ cao nhất trên 2016 với chênh lệch 0,09 điểm. M4 có MED thấp nhất trên 2014, 2016 và trung bình, còn M3 thấp nhất trên 2019. Vì vậy không có một model tốt nhất trên mọi metric.**
>
> **Bảng cho phép khẳng định M2 thấp hơn baseline, M3 phục hồi so với M2, M4 giảm mức lỗi trung bình và M5 giảm mạnh. Nó chưa cho phép khẳng định PE blurring, over-smoothing hoặc relative bias là nguyên nhân duy nhất. Chênh lệch theo năm có thể do độ dài, cấu trúc, token hiếm và writer distribution, nhưng phải thống kê trước.**
>
> **Để phân tích thuyết phục, em sẽ chạy nhiều seed, bootstrap CI, histogram/CDF edit distance, paired per-sample analysis và error breakdown theo độ dài/cấu trúc. Khi giả thuyết không được ủng hộ, em báo cáo đó là negative result và rút ra điều kiện thiết kế, không đổi metric để tuyên bố thắng.**

# Phụ lục B — Biểu đồ nên chuẩn bị

- [ ] Bar chart ExpRate từng model × dataset.
- [ ] Bar chart MED từng model × dataset.
- [ ] Histogram/CDF edit distance.
- [ ] Paired delta M4–M1.
- [ ] ExpRate theo length bin.
- [ ] Accuracy theo structure.
- [ ] Token frequency versus error.
- [ ] Mean ± std nhiều seed.
- [ ] Bootstrap CI.
