# Báo cáo Các Trường Hợp Lỗi (Error Cases Log)

Tổng số ảnh lỗi: 26 / 50

Dưới đây là chi tiết các trường hợp UniMERNet dự đoán sai cấu trúc hoặc ký hiệu:

### Image ID: spe_0020448.png
**Ground truth:** `\{ F , G \} =`  
**Prediction:** `\left\{ F , G \right\} =`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: spe_0031600.png
**Ground truth:** `\bar { V } = V`  
**Prediction:** `{ \bar { V } } = V`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: spe_0076260.png
**Ground truth:** `K = - \beta H + { \bf v } { \bf P } .`  
**Prediction:** `K = - \beta H + \mathbf { v } \mathbf { P } .`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: spe_0088793.png
**Ground truth:** `v ( \xi ) \approx - 1 + \kappa \xi ,`  
**Prediction:** `\begin{array} { r } { v ( \xi ) \approx - 1 + \kappa \xi , } \end{array}`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: frac_0000035.png
**Ground truth:** `e ^ { A } = e ^ { A _ { 0 } } \left( t _ { 0 } - \mathrm { s i g n } ( m ) t \right) ^ { - \frac { m } { 2 } } \; , \; \; \; \; \chi = \chi _ { 0 } \left( t _ { 0 } - \mathrm { s i g n } ( m ) t \right) ^ { m } \; ,`  
**Prediction:** `e ^ { A } = e ^ { A _ { 0 } } \left( t _ { 0 } - \mathrm { s i g n } ( m ) t \right) ^ { - \frac { m } { 2 } } \; , \quad \chi = \chi _ { 0 } \left( t _ { 0 } - \mathrm { s i g n } ( m ) t \right) ^ { m } \; ,`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: frac_0000155.png
**Ground truth:** `\frac 1 { \nabla ^ { 2 } } \, \delta _ { \Sigma } ^ { ( 2 ) } ( z - z _ { 0 } ) = - \frac 1 \pi \log { \cal E } ( z , z _ { 0 } )`  
**Prediction:** `\frac { 1 } { \nabla ^ { 2 } } \, \delta _ { \Sigma } ^ { ( 2 ) } ( z - z _ { 0 } ) = - \frac { 1 } { \pi } \log E ( z , z _ { 0 } )`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: frac_0000317.png
**Ground truth:** `\stackrel { \mathrm { G } } { { \mathcal L } } \, : = \frac { 1 } { 2 m } \left[ ( D _ { \alpha } \overline { { { \Psi } } } ) D ^ { \alpha } \Psi - m ^ { 2 } \overline { { { \Psi } } } \Psi \right] \, .`  
**Prediction:** `\stackrel { \mathrm { G } } { \mathcal { L } } : = \frac { 1 } { 2 m } \left[ ( D _ { \alpha } \overline { { { \Psi } } } ) D ^ { \alpha } \Psi - m ^ { 2 } \overline { { { \Psi } } } \Psi \right] \; .`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: frac_0000404.png
**Ground truth:** `\Delta = - D ^ { 2 } - \frac { \mathrm { i } } 2 \sigma _ { \mu \nu } F _ { \mu \nu } .`  
**Prediction:** `\Delta = - D ^ { 2 } - { \frac { \mathrm { i } } { 2 } } \sigma _ { \mu \nu } F _ { \mu \nu } .`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: frac_0000496.png
**Ground truth:** `r = \frac { \alpha } { \beta } \vert \sin \beta \left( \sigma - \sigma _ { 0 } \right) \vert`  
**Prediction:** `r = \frac { \alpha } { \beta } | \sin \beta \left( \sigma - \sigma _ { 0 } \right) |`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: supsub_0000069.png
**Ground truth:** `g _ { J _ { 1 } \, J _ { 2 } } ^ { J } \bigl ( J _ { 1 } , M _ { 1 } ; J _ { 2 } , M _ { 2 } \vert J _ { 1 } , J _ { 2 } ; J , M _ { 1 } + M _ { 2 } \bigr ) \, \Bigl ( \xi _ { M _ { 1 } + M _ { 2 } } ^ { ( J ) } ( \sigma _ { 1 } ) + \mathrm { d e s c e n d a n t s } \Bigr ) \Bigr \} ,`  
**Prediction:** `g _ { J _ { 1 } J _ { 2 } } ^ { J } ( J _ { 1 } , M _ { 1 } ; J _ { 2 } , M _ { 2 } | J _ { 1 } , J _ { 2 } ; J , M _ { 1 } + M _ { 2 } ) \, \Big ( \xi _ { M _ { 1 } + M _ { 2 } } ^ { ( J ) } ( \sigma _ { 1 } ) + \mathrm { d e s c e n d a n t s } \Big ) \Big \} ,`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: supsub_0000160.png
**Ground truth:** `g x ( \alpha \cdot q , \xi ) \to \left\{ \begin{array} { c l } { { \mathrm { f i n i t e } , } } & { { \mathrm { f o r } \quad \pm \alpha _ { i } \in \Pi \quad ( \delta \leq 1 / h ) \quad \mathrm { a n d } \ \pm \alpha _ { h } \quad ( \delta = 1 / h ) , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array} \right.`  
**Prediction:** `g x ( \alpha \cdot q , \xi ) \to \left\{ \begin{array} { c l l } { \mathrm { f i n i t e } , } & { \mathrm { f o r } \quad \pm \alpha _ { i } \in \Pi \quad ( \delta \leq 1 / h ) \quad \mathrm { a n d } \, \, \, \pm \alpha _ { h } } & { ( \delta = 1 / h ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: supsub_0000313.png
**Ground truth:** `{ \cal R } = { \cal R } _ { r } \sin \eta , ~ ~ ~ \tau = { \cal R } _ { r } ( 1 - \cos \eta ) .`  
**Prediction:** `R = R _ { r } \sin \eta , \quad \tau = R _ { r } ( 1 - \cos \eta ) .`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: supsub_0000357.png
**Ground truth:** `\delta _ { { \hat { \xi } } _ { 2 } } L [ { \hat { \xi } } _ { 1 } ] = \left\{ L [ { \hat { \xi } } _ { 2 } ] , L [ { \hat { \xi } } _ { 1 } ] \right\} = L [ \{ { \hat { \xi } } _ { 1 } , { \hat { \xi } } _ { 2 } \} _ { \mathrm { \scriptsize ~ S D } } ] + K [ { \hat { \xi } } _ { 1 } , { \hat { \xi } } _ { 2 } ]`  
**Prediction:** `\delta _ { \hat { \xi } _ { 2 } } L [ \hat { \xi } _ { 1 } ] = \left\{ L [ \hat { \xi } _ { 2 } ] , L [ \hat { \xi } _ { 1 } ] \right\} = L [ \{ \hat { \xi } _ { 1 } , \hat { \xi } _ { 2 } \} _ { \mathrm { \scriptsize ~ S D } } ] + K [ \hat { \xi } _ { 1 } , \hat { \xi } _ { 2 } ]`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: supsub_0000838.png
**Ground truth:** `2 \kappa _ { 1 1 } { } ^ { 2 } T _ { 3 } { \tilde { T } } _ { 6 } = 2 \pi n`  
**Prediction:** `{ 2 \kappa _ { 1 1 } } ^ { 2 } T _ { 3 } \tilde { T } _ { 6 } = 2 \pi n`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: supsub_0000850.png
**Ground truth:** `( c _ { 0 } ^ { ( 1 ) } + c _ { 0 } ^ { ( 2 ) } + c _ { 0 } ^ { ( 3 ) } ) | V _ { 3 } \rangle = 0 , \ \ c _ { 0 } | I \star A \rangle = | I \star ( c _ { 0 } A ) \rangle , \ \ \forall A`  
**Prediction:** `( c _ { 0 } ^ { ( 1 ) } + c _ { 0 } ^ { ( 2 ) } + c _ { 0 } ^ { ( 3 ) } ) | V _ { 3 } \rangle = 0 , \; \; c _ { 0 } | I \star A \rangle = | I \star ( c _ { 0 } A ) \rangle , \; \; \forall A`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: supsub_0001170.png
**Ground truth:** `\mathcal { F } _ { \mu \nu } ^ { a } = \left( \begin{array} { c c c c c c c } { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { \mathcal { F } _ { 1 2 } ^ { a } } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { - \mathcal { F } _ { 1 2 } ^ { a } } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { \mathcal { F } _ { 3 4 } ^ { a } } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { - \mathcal { F } _ { 3 4 } ^ { a } } } & { { 0 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { \mathcal { F } _ { 5 6 } ^ { a } } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { 0 } } & { { - \mathcal { F } _ { 5 6 } ^ { a } } } & { { 0 } } \end{array} \right) .`  
**Prediction:** `\mathcal { F } _ { \mu \nu } ^ { a } = \left( \begin{array} { c c c c c c c } { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { \mathcal { F } _ { 1 2 } ^ { a } } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { - \mathcal { F } _ { 1 2 } ^ { a } } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { \mathcal { F } _ { 3 4 } ^ { a } } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { - \mathcal { F } _ { 3 4 } ^ { a } } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { \mathcal { F } _ { 5 6 } ^ { a } } \\ { 0 } & { 0 } & { 0 } & { 0 } & { 0 } & { - \mathcal { F } _ { 5 6 } ^ { a } } & { 0 } \end{array} \right) .`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: sqrt_0001221.png
**Ground truth:** `w ( m ) = w _ { 0 } ( m \sqrt { \alpha \prime } ) ^ { - a } e ^ { b m \sqrt { \alpha \prime } }`  
**Prediction:** `w ( m ) = w _ { 0 } ( m \sqrt { \alpha ^ { \prime } } ) ^ { - a } e ^ { b m \sqrt { \alpha ^ { \prime } } }`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: sqrt_0019646.png
**Ground truth:** `\mathcal { L } _ { B } I = \sqrt { \operatorname * { d e t } ( \eta + b F ^ { \mu \nu } ) } = \sqrt { \operatorname * { d e t } H _ { \mu \nu } } ,`  
**Prediction:** `{ \mathcal { L } } _ { B } I = { \sqrt { \operatorname* { d e t } ( \eta + b F ^ { \mu \nu } ) } } = { \sqrt { \operatorname* { d e t } H _ { \mu \nu } } } ,`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: sqrt_0033831.png
**Ground truth:** `\begin{array} { r c l c r c l c } { { \left[ K _ { a } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { \quad . \quad } } & { { \left[ K _ { a } , P _ { - } \right] } } & { { = } } & { { - \sqrt { 2 } P _ { a } } } & { { \quad . } } \\ { { \left[ L _ { a b } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { \quad . \quad } } & { { \left[ L _ { a b } , P _ { - } \right] } } & { { = } } & { { 0 } } & { { \quad . } } \\ { { \left[ K _ { \mu } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { \quad . \quad } } & { { \left[ K _ { \mu } , P _ { - } \right] } } & { { = } } & { { - \sqrt { 2 } P _ { \mu } } } & { { \quad . } } \\ { { \left[ L _ { \mu \nu } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { \quad . \quad } } & { { \left[ L _ { \mu \nu } , P _ { - } \right] } } & { { = } } & { { 0 } } & { { \quad . } } \\ { { \left[ \Delta , P _ { + } \right] } } & { { = } } & { { P _ { + } } } & { { \quad . \quad } } & { { \left[ \Delta , P _ { - } \right] } } & { { = } } & { { - P _ { - } } } & { { \quad . } } \end{array}`  
**Prediction:** `\begin{array} { r c l c r c l c l } { { \left[ K _ { a } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { . } } & { { \left[ K _ { a } , P _ { - } \right] } } & { { = } } & { { - \sqrt { 2 } P _ { a } } } & { { . } } \\ { { \left[ L _ { a b } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { . } } & { { \left[ L _ { a b } , P _ { - } \right] } } & { { = } } & { { 0 } } & { { . } } \\ { { \left[ K _ { \mu } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { . } } & { { \left[ K _ { \mu } , P _ { - } \right] } } & { { = } } & { { - \sqrt { 2 } P _ { \mu } } } & { { . } } \\ { { \left[ L _ { \mu \nu } , P _ { + } \right] } } & { { = } } & { { 0 } } & { { . } } & { { \left[ L _ { \mu \nu } , P _ { - } \right] } } & { { = } } & { { 0 } } & { { . } } \\ { { \left[ \Delta , P _ { + } \right] } } & { { = } } & { { P _ { + } } } & { { . } } & { { \left[ \Delta , P _ { - } \right] } } & { { = } } & { { - P _ { - } } } & { { . } } \end{array}`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: sqrt_0043327.png
**Ground truth:** `\left\{ \Delta ( f ^ { - 1 } ) = 0 , ~ \mathrm { f o r } ~ - \omega \le k _ { 0 } \le \omega ~ , ~ \omega = + \sqrt { \vec { k } ^ { 2 } } ~ \mathrm { w h e n } ~ m = 0 \right\}`  
**Prediction:** `\left\{ \Delta ( f ^ { - 1 } ) = 0 , \; \mathrm { f o r } \; - \omega \leq k _ { 0 } \leq \omega \; , \; \omega = + \sqrt { \vec { k } ^ { 2 } } \; \mathrm { w h e n } \; m = 0 \right\}`  
**Loại lỗi:** `other_structure_error`  
**Nguyên nhân đoán & Hướng xử lý:**  
Cần thêm graph validator để lọc và sửa cấu trúc cú pháp LaTeX lỗi.  

---

### Image ID: hwe_0000001.png
**Ground truth:** `N H _ { 4 } C l + N a C H = N a C l + N H _ { 3 } \uparrow + H _ { 2 } O`  
**Prediction:** `N H _ { 4 } C l + N a O H = N a C l + N H _ { 3 } \uparrow + H _ { 2 } O`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: hwe_0000002.png
**Ground truth:** `= \frac { n } { n + 1 } ( n \in N ^ { \ast } )`  
**Prediction:** `C = \frac { n } { n + 1 } ( n \in N ^ { \ast } )`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: hwe_0000012.png
**Ground truth:** `\{ a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } \}`  
**Prediction:** `\sum a _ { 1 } a _ { 2 , a _ { 3 } , a _ { 4 } } ]`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

### Image ID: hwe_0000013.png
**Ground truth:** `3 2 x ^ { 6 } - 4 8 x ^ { 4 } + 1 8 x ^ { 2 } - 1`  
**Prediction:** `3 2 x ^ { 2 } - 4 8 x ^ { 4 } + 1 8 x ^ { 2 } - 1`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: hwe_0000016.png
**Ground truth:** `| a | = | x ^ { 1 } | + \ldots + | x ^ { n } |`  
**Prediction:** `| a | = | x ^ { 1 } | + \cdots + | x ^ { n } |`  
**Loại lỗi:** `symbol_misrecognition`  
**Nguyên nhân đoán & Hướng xử lý:**  
Lỗi nhận dạng ký tự viết tay gần giống nhau. Tiền xử lý tương phản sẽ giúp ích.  

---

### Image ID: hwe_0000019.png
**Ground truth:** `m _ { z n } = M _ { z n } \cdot n _ { z n } = 6 5 g / m o l \times 0 . 2 m o l = 1 3 g`  
**Prediction:** `m _ { z n } = M z \cdot n _ { z n } = 6 5 g / m o l \times 0 . 2 m o l = 1 3 g`  
**Loại lỗi:** `bracket_mismatch`  
**Nguyên nhân đoán & Hướng xử lý:**  
Có thể xử lý bằng graph validator hoặc post-processing đóng ngoặc tự động.  

---

