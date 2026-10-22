# Báo cáo kỹ thuật hai kiến trúc Legacy và Unified của VnECheck

> Phạm vi khảo sát: trạng thái repository ngày 22/10/2026. Báo cáo được lập từ mã nguồn, checkpoint, cấu hình run, `summary.json`, manifest ONNX và báo cáo benchmark đang có. Không mở nội dung ảnh, không huấn luyện lại và không thay đổi checkpoint.

## Quy ước về mức độ bằng chứng

| Ký hiệu | Ý nghĩa trong báo cáo |
|---|---|
| **Đã triển khai** | Có luồng mã nguồn thực thi được. Chưa mặc nhiên có checkpoint hay kết quả. |
| **Đã huấn luyện** | Có checkpoint/run config/summary truy vết được trong repository. |
| **Đã kiểm thử** | Có kết quả được lưu trong repository. Cần đọc đúng phạm vi tập mẫu và cách đo. |
| **Đề xuất** | Hướng nghiên cứu hoặc cải tiến; chưa được coi là chức năng hiện tại. |
| **Thiếu bằng chứng** | **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. |

Một kết luận quan trọng ngay từ đầu: mã hiện tại đã triển khai Router Legacy 7 lớp, nhưng checkpoint Router đang lưu vẫn là Router 5 lớp. Vì vậy mọi số liệu Router Legacy hiện có trong repository là của bài toán 5 lớp, không phải 7 lớp.

---

# I. Tổng quan quá trình phát triển

## 1. Bài toán đầu vào và đầu ra

Đầu vào của cả hai pipeline là một ảnh màu, được quy đổi thành tensor `float32` kích thước `[B, 3, 224, 224]`. Đầu ra nghiệp vụ gồm:

1. trạng thái ảnh: `SUPPORTED`, `UNSUPPORTED` hoặc `NON_FOOD`;
2. nhóm thực phẩm được hỗ trợ: `cu`, `gia_cam`, `gia_suc`, `qua`, `rau`;
3. trạng thái tươi: `FRESH`, `HALF-FRESH`, `SPOILED` với ràng buộc theo nhóm;
4. điểm độ tươi quy ước trên thang 1–10.

Hợp đồng nhãn nằm tại [`food_ai/config.py`](food_ai/config.py#L3-L37). Ba nhóm `cu`, `qua`, `rau` chỉ dùng `FRESH/SPOILED`; `gia_cam`, `gia_suc` dùng đủ ba trạng thái. Danh sách sản phẩm thực tế trong dữ liệu gồm cà rốt, khoai tây; gà, vịt; bò, cừu, heo; táo, chuối, xoài, cam, dâu; ớt chuông, khổ qua, capsicum, dưa leo, đậu bắp, cà chua.

`UNSUPPORTED` nghĩa là ảnh thực phẩm nhưng không thuộc năm nhóm/sản phẩm mà hệ thống chịu trách nhiệm đánh giá. `NON_FOOD` nghĩa là ảnh không phải thực phẩm. Hai lớp này ngăn hệ thống gán ép mọi ảnh đầu vào thành một nhóm thực phẩm hợp lệ.

## 2. Legacy ban đầu: Router 5 lớp và năm Freshness Expert

Legacy phân rã bài toán theo kiểu mixture-of-experts định tuyến cứng:

```text
Ảnh
  │
  ▼
Router 5 lớp: cu | gia_cam | gia_suc | qua | rau
  │ chọn đúng một nhánh bằng argmax
  ├── cu ───────► Expert cu ───────► FRESH/SPOILED
  ├── gia_cam ──► Expert gia_cam ──► FRESH/HALF-FRESH/SPOILED
  ├── gia_suc ──► Expert gia_suc ──► FRESH/HALF-FRESH/SPOILED
  ├── qua ──────► Expert qua ──────► FRESH/SPOILED
  └── rau ──────► Expert rau ──────► FRESH/SPOILED
```

Mỗi ảnh được Router phân nhóm trước, sau đó chỉ Expert của nhóm dự đoán được chạy. Mã suy luận nằm ở [`predict.py`, hàm `predict`](predict.py#L33-L96). Cách thiết kế này hợp lý vì mỗi nhóm có dấu hiệu hư hỏng và tập trạng thái khác nhau; Expert không phải học đồng thời tất cả miền hình ảnh. Đổi lại, sai số Router truyền thẳng xuống cuối chuỗi: nếu Router chọn sai nhóm thì Expert đúng không được gọi. Một ảnh supported cần hai lần forward; bộ đầy đủ gồm sáu model/ONNX.

Hạn chế căn bản của Router 5 lớp là không có phương án “không thuộc phạm vi”. Do softmax luôn có top-1, ảnh phong cảnh, động vật hoặc món ăn ngoài danh mục vẫn bị ép vào một trong năm nhóm. Checkpoint Router hiện tại thực sự có đúng năm lớp; run tương ứng là [`runs/20261018_234532_router_all_mobilenet_v3_large/summary.json`](runs/20261018_234532_router_all_mobilenet_v3_large/summary.json#L1-L42).

## 3. Legacy 7 lớp

Mã hiện tại mở rộng đầu Router thành:

```text
5 category + UNSUPPORTED + NON_FOOD = 7 logits

Ảnh ─► Router 7 lớp ─┬─► category ─► Expert tương ứng ─► freshness + score
                     ├─► UNSUPPORTED ─► từ chối đánh giá độ tươi
                     └─► NON_FOOD ────► từ chối đánh giá độ tươi
```

`ROUTER_CLASSES = CATEGORIES + FOOD_STATUS_CLASSES[1:]` được định nghĩa tại [`food_ai/config.py`](food_ai/config.py#L24-L32); bộ nạp dữ liệu Router ghép ảnh supported và negative tại [`food_ai/data.py`, `build_router_datasets`](food_ai/data.py#L183-L234); `train.py` tạo head theo `ROUTER_CLASSES` và kiểm tra hợp đồng lớp tại [`train.py`](train.py#L200-L223).

Router 5 lớp không thể resume trực tiếp thành 7 lớp. Ngoài kiểm tra từ chối class mismatch tại [`train.py`](train.py#L273-L282), ma trận head cuối cũng đổi từ `[5, d]` thành `[7, d]`, nên kích thước state dictionary không tương thích. Có thể dùng cùng pretrained backbone để train mới, nhưng đó không phải resume đầy đủ optimizer/head.

Trạng thái bằng chứng:

- **Đã triển khai:** loader, hợp đồng 7 lớp, loss weighting và nhánh từ chối.
- **Đã huấn luyện Router 7 lớp:** **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.
- **Đã kiểm thử Router 7 lớp:** **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

## 4. Unified

Unified thay sáu mạng độc lập bằng một backbone dùng chung và ba head:

```text
                         ┌─► Food head: 3 logits
Ảnh ─► shared backbone ──┼─► Category head: 5 logits
                         └─► Freshness head: 3 logits
                                  │
                                  └─ mask HALF-FRESH theo category khi giải mã
```

Mục tiêu thiết kế là giảm số file, số lần forward và trùng lặp feature extraction; đồng thời học food gate, category và freshness trong một biểu diễn chung. Các thành phần đã triển khai gồm manifest dữ liệu, dataset có nhãn thiếu, shared model, masked multi-task loss, script train/evaluate/export/verify và runtime ONNX. Nguồn chính: [`food_ai/unified_data.py`](food_ai/unified_data.py#L13-L117), [`food_ai/unified_model.py`](food_ai/unified_model.py#L16-L96), [`food_ai/unified_loss.py`](food_ai/unified_loss.py#L7-L76), [`train_unified.py`](train_unified.py#L43-L437).

Repository có bốn run Unified cho MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0 và ResNet18; có các ONNX tương ứng. Đây là bằng chứng đã train/test từng head. Tuy nhiên chưa có một lần đánh giá full-test chuẩn hóa, cùng giao thức, so sánh end-to-end Legacy 7 lớp với Unified.

## 5. Trạng thái hiện tại

| Pipeline | Mã | Checkpoint | ONNX | Kết quả lưu | Diễn giải đúng |
|---|---:|---:|---:|---:|---|
| Legacy Router 5 + Experts | Có | Có | Có | Có | Baseline đã train; Router không có negative gate. |
| Legacy Router 7 + Experts | Có | **Không thấy** | **Không thấy** | **Không thấy** | Chỉ mới triển khai mã. |
| Unified 3-head | Có | Có | Có | Có | Đã train bốn backbone; có test từng head và benchmark mẫu. |
| Flutter/Android inference | **Không thấy** | – | – | – | **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. |

---

# II. Dữ liệu và tiền xử lý ảnh

## 1. Cấu trúc và số lượng dữ liệu

### 1.1 Nguồn và cấu trúc

`data_split/{train,valid,test}/{category}/{freshness}/{product}/...` chứa ảnh supported. Script chia lại mặc định lấy từ `data/train` và `data/valid`, loại exact duplicate/xung đột SHA-256, gom pHash trong cùng lớp và không tách một cluster qua nhiều split; xem [`scripts/resplit_dataset.py`](scripts/resplit_dataset.py#L1-L7), [`scripts/resplit_dataset.py`](scripts/resplit_dataset.py#L321-L427), [`scripts/resplit_dataset.py`](scripts/resplit_dataset.py#L511-L562).

`data_negative/{train,valid,test}/{UNSUPPORTED,NON_FOOD}/...` chứa negative. Script [`scripts/split_negative_dataset.py`](scripts/split_negative_dataset.py#L148-L194) chia 70/15/15 và cũng dùng SHA-256/pHash. `UNSUPPORTED` có pipeline trích Food-101 từ parquet tại [`scripts/extract_food101_parquet.py`](scripts/extract_food101_parquet.py#L137-L177). Nguồn gốc chi tiết của toàn bộ ảnh `NON_FOOD`: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

Unified không copy ảnh sang bộ mới. [`scripts/build_unified_manifest.py`](scripts/build_unified_manifest.py#L55-L86) tạo CSV trỏ tới cả `data_split` và `data_negative`. Dòng supported có `category/freshness`; dòng negative để trống hai nhãn đó.

### 1.2 Thống kê thực tế

Thống kê dưới đây được đếm từ cấu trúc file, không giải mã nội dung ảnh.

| Split | Supported | UNSUPPORTED | NON_FOOD | Tổng Unified |
|---|---:|---:|---:|---:|
| train | 66.115 | 3.500 | 3.500 | 73.115 |
| valid | 14.172 | 750 | 750 | 15.672 |
| test | 14.162 | 750 | 750 | 15.662 |
| **Tổng** | **94.449** | **5.000** | **5.000** | **104.449** |

Phân bố supported theo nhóm/trạng thái:

| Nhóm | FRESH | HALF-FRESH | SPOILED | Tổng |
|---|---:|---:|---:|---:|
| cu | 10.200 | 0 | 3.331 | 13.531 |
| gia_cam | 3.491 | 1.000 | 4.605 | 9.096 |
| gia_suc | 11.806 | 5.935 | 9.105 | 26.846 |
| qua | 14.889 | 0 | 10.321 | 25.210 |
| rau | 14.226 | 0 | 5.540 | 19.766 |

Phân bố supported theo split và trạng thái:

| Split/category | FRESH | HALF-FRESH | SPOILED |
|---|---:|---:|---:|
| train/cu | 7.140 | 0 | 2.332 |
| train/gia_cam | 2.444 | 700 | 3.224 |
| train/gia_suc | 8.264 | 4.155 | 6.373 |
| train/qua | 10.422 | 0 | 7.224 |
| train/rau | 9.958 | 0 | 3.879 |
| valid/cu | 1.530 | 0 | 500 |
| valid/gia_cam | 524 | 150 | 691 |
| valid/gia_suc | 1.771 | 890 | 1.366 |
| valid/qua | 2.234 | 0 | 1.550 |
| valid/rau | 2.135 | 0 | 831 |
| test/cu | 1.530 | 0 | 499 |
| test/gia_cam | 523 | 150 | 690 |
| test/gia_suc | 1.771 | 890 | 1.366 |
| test/qua | 2.233 | 0 | 1.547 |
| test/rau | 2.133 | 0 | 830 |

Legacy 5 lớp/Expert dùng `data_split`; mã Legacy 7 lớp ghép thêm `data_negative`. Unified dùng cùng các ảnh qua manifest. Vì vậy có thể so sánh trên cùng nguồn ảnh, nhưng các `summary.json` chỉ so sánh công bằng nếu cùng split, chính sách giải mã và tập mẫu.

### 1.3 Ánh xạ nhãn thiếu

Unified ánh xạ `SUPPORTED/UNSUPPORTED/NON_FOOD` thành 0/1/2. Category và freshness của negative được gán `IGNORE_INDEX=-100`; supported bắt buộc có đủ hai nhãn. Logic nằm tại [`food_ai/unified_data.py`](food_ai/unified_data.py#L13-L20), [`food_ai/unified_data.py`](food_ai/unified_data.py#L57-L90). Nhờ vậy negative huấn luyện food head nhưng không tạo gradient giả cho category/freshness head.

## 2. Tiền xử lý ảnh

### 2.1 Pipeline chính xác

Ảnh được đọc bằng Pillow qua `Image.open`. Bước đầu là `ImageOps.exif_transpose(image).convert("RGB")` trong [`PrepareImage`](food_ai/data.py#L89-L93). EXIF transpose đưa ảnh chụp về đúng chiều hiển thị; RGB bảo đảm đúng ba kênh, loại khác biệt ảnh grayscale/RGBA và khớp pretrained ImageNet.

Train transform tại [`build_transforms`](food_ai/data.py#L110-L132):

1. sửa EXIF và chuyển RGB;
2. `RandomResizedCrop(224, scale=(0.85,1.0), ratio=(0.85,1.15), antialias=True)`;
3. `RandomHorizontalFlip()` — mặc định xác suất 0,5;
4. `ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0)`;
5. `ToTensor()` chuyển HWC uint8 0–255 thành CHW float và chia 255;
6. `Normalize(mean, std)`.

Validation/test tại [`build_transforms`](food_ai/data.py#L133-L139): EXIF/RGB → resize cạnh ngắn 232 → center crop 224×224 → tensor → normalize. Với ảnh PIL, `Resize` mặc định dùng bilinear; code bật antialias. Runtime web ONNX ghi rõ `Image.Resampling.BILINEAR` và tự thực hiện cùng phép biến đổi tại [`local_web/server.py`](local_web/server.py#L127-L148).

Mean/std ImageNet tại [`food_ai/config.py`](food_ai/config.py#L34-L37):

\[
\mu=(0.485,0.456,0.406),\qquad
\sigma=(0.229,0.224,0.225)
\]

Với pixel 8-bit `p_c`, chuẩn hóa kênh `c` là:

\[
x_c=\frac{p_c}{255},\qquad z_c=\frac{x_c-\mu_c}{\sigma_c}
\]

Ví dụ pixel RGB `(200,120,60)` trở thành xấp xỉ `(0,7843; 0,4706; 0,2353)`, rồi thành `(1,307; 0,065; -0,759)`. Đây chỉ là ví dụ toán học của preprocessing, không phải dự đoán model.

Ví dụ toàn ảnh: một ảnh cà chua 800×600 có EXIF xoay được đưa đúng chiều, chuyển RGB; khi test cạnh ngắn 600 được resize thành 232 nên ảnh thành khoảng 309×232, sau đó lấy ô giữa 224×224. Tensor cuối có `[3,224,224]`, được normalize rồi thêm batch dimension thành `[1,3,224,224]`.

### 2.2 Tính nhất quán và nguy cơ sai lệch

- Train cố ý ngẫu nhiên; validation, test và hàm `predict.py` dùng deterministic eval transform.
- ONNX graph nhận tensor đã preprocess; preprocessing không nằm trong graph.
- Runtime web tự mô phỏng eval transform. Có unit test so sánh hai tensor với `atol=1e-6` tại [`tests/test_local_web.py`](tests/test_local_web.py#L43-L50), nhưng log CI/test đã chạy: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.
- `verify_onnx.py` và `verify_unified_onnx.py` có kiểm tra sai khác PyTorch–ONNX, nhưng không thấy log xác minh đã lưu. Do đó không được tuyên bố các artifact hiện tại đã vượt tolerance.
- Pipeline camera điện thoại có áp dụng EXIF/RGB/resize/crop/normalize giống hệt hay không: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**, vì không có mã Flutter/Android.

Center crop có thể cắt mất đối tượng nếu thực phẩm ở sát biên hoặc nhỏ. Random crop train với scale tối thiểu 0,85 tương đối bảo thủ, nhưng vẫn có thể bỏ dấu hiệu hỏng nằm ở rìa. Đây là trade-off giữa bất biến vị trí và bảo toàn toàn bộ vật thể.

## 3. Data augmentation và shortcut learning

| Biến đổi thực sự có code | Cơ chế và mục đích | Nguy cơ |
|---|---|---|
| RandomResizedCrop | Chọn vùng ngẫu nhiên 85–100% diện tích với tỷ lệ 0,85–1,15 rồi resize 224. Tạo bất biến vị trí/tỷ lệ. | Cắt dấu hiệu hỏng hoặc đẩy model học vùng cục bộ. |
| HorizontalFlip | Lật trái–phải xác suất 0,5. | Thường an toàn với thực phẩm; có thể lật chữ/watermark và củng cố shortcut. |
| ColorJitter brightness ±10% | Mô phỏng ánh sáng. | Màu là dấu hiệu tươi quan trọng; jitter quá mạnh sẽ phá nhãn. Mức hiện tại nhỏ nhưng chưa có ablation. |
| ColorJitter contrast ±10% | Đổi chênh lệch sáng tối. | Có thể che/nhấn mạnh vết thâm, mốc. |
| ColorJitter saturation ±5% | Đổi độ bão hòa màu. | Có thể làm thực phẩm trông tươi/hỏng khác đi. |
| Resize/normalize | Chuẩn hóa kích thước/phân phối đầu vào, không phải augmentation ngẫu nhiên ở eval. | Nội suy làm mất chi tiết nhỏ; crop gây mất vật thể. |

Không có rotation, hue jitter, vertical flip, blur, CutMix, MixUp hoặc random erasing trong transform hiện tại. Không nên mô tả chúng là đã dùng.

Nguy cơ shortcut learning vẫn còn: nếu mỗi lớp có nền, khay, ánh sáng, watermark hoặc nguồn dữ liệu đặc trưng, CNN có thể dùng các tín hiệu này thay vì bề mặt thực phẩm. Augmentation hiện tại làm thay đổi crop/màu nhẹ nhưng không tách foreground khỏi background.

Các kiểm tra 3×3, che vùng, thay nền, crop đối tượng, Grad-CAM: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. Chúng là đề xuất ở Phần IX, không phải preprocessing hiện có.

---

# III. Thuật toán và kiến trúc mô hình

## 1. Backbone và CNN

Backbone là phần mạng biến ảnh thành vector đặc trưng; classification head biến vector đó thành logits. Repository cho phép `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`, `resnet18` tại [`food_ai/model.py`](food_ai/model.py#L5-L37) và [`food_ai/unified_model.py`](food_ai/unified_model.py#L8-L65).

Một convolution dùng kernel trượt trên ảnh/feature map để phát hiện cạnh, texture và mẫu cục bộ. Activation tạo phi tuyến để ghép các mẫu đơn giản thành đặc trưng phức tạp. Pooling giảm kích thước không gian và tăng vùng quan sát. Global Average Pooling lấy trung bình mỗi kênh trên toàn bộ `H×W`, biến `[B,C,H,W]` thành `[B,C,1,1]`, giảm số tham số và bớt phụ thuộc vị trí tuyệt đối.

- **MobileNetV3** dùng inverted residual, depthwise convolution, pointwise convolution, squeeze-and-excitation và các activation tối ưu di động như h-swish trong implementation torchvision. Repository không tự viết lại các block này. Depthwise separable convolution giảm chi phí từ xấp xỉ `HW·Cin·Cout·K²` xuống `HW·Cin·K² + HW·Cin·Cout`. Small ít capacity/compute hơn Large.
- **ResNet18** dùng residual connection `y=F(x)+x`; đường tắt giúp gradient truyền qua mạng sâu và học phần dư thay vì toàn ánh xạ.
- **EfficientNet-B0** là cấu hình cơ sở dùng MBConv và compound scaling; phù hợp làm điểm so sánh hiệu quả tham số/độ chính xác.

Các kiến trúc phù hợp vì bao phủ phổ nhẹ–nặng và đều có pretrained ImageNet trong torchvision. Train scripts gọi `pretrained=True`; constructor chọn `*_Weights.DEFAULT`. Legacy thay `classifier[3]`, `classifier[1]` hoặc `fc`; Unified giữ feature extractor/shared classifier rồi thêm ba `Linear` heads.

Đếm tham số trực tiếp từ constructor hiện tại (đây là tính chất cấu trúc, không phải kết quả train):

| Backbone | Legacy 7-head tổng | Legacy head-only trainable | Unified tổng | Unified head-only trainable |
|---|---:|---:|---:|---:|
| MobileNetV3-Small | 1.525.031 | 598.023 | 1.529.131 | 11.275 |
| MobileNetV3-Large | 4.210.999 | 1.239.047 | 4.216.123 | 14.091 |
| EfficientNet-B0 | 4.016.515 | 8.967 | 4.021.639 | 14.091 |
| ResNet18 | 11.180.103 | 3.591 | 11.182.155 | 5.643 |

Legacy `head_only` bỏ freeze chỉ toàn bộ module `classifier`/`fc`, nên với MobileNet cả MLP classifier được học. Unified `head_only` chỉ mở ba head mới; shared classifier vẫn freeze tại [`food_ai/unified_model.py`](food_ai/unified_model.py#L84-L96). `fine_tune` giữ toàn bộ tham số trainable. Không có hàm khởi tạo riêng cho head; `nn.Linear` dùng mặc định PyTorch.

## 2. Transfer learning

Transfer learning bắt đầu từ trọng số ImageNet thay vì ngẫu nhiên. `head_only` nhanh hơn, ít VRAM gradient hơn và giảm overfit khi dữ liệu nhỏ, nhưng feature ImageNet không thích nghi sâu với dấu hiệu hư hỏng. `fine_tune` cập nhật toàn mạng, có tiềm năng tốt hơn nhưng chậm hơn và nhạy learning rate. Cấu hình freeze/unfreeze Legacy nằm tại [`food_ai/model.py`, `configure_training_method`](food_ai/model.py#L46-L55); Unified tại [`food_ai/unified_model.py`](food_ai/unified_model.py#L84-L96). Khi Legacy head-only, BatchNorm bị đặt eval trong epoch tại [`train.py`](train.py#L122-L127).

## 3. Legacy chi tiết

Router tạo logits, softmax thành xác suất, chọn top-1. Với Router 7 lớp, nếu top-1 là negative thì trả về ngay; nếu là category thì chạy đúng một Expert. Mã production CLI không có confidence threshold, chỉ argmax tại [`predict.py`](predict.py#L33-L73). `evaluate_router_uncertainty.py` có phân tích threshold riêng, nhưng đó không phải chính sách trong `predict.py`.

Food gate Legacy 7 lớp được suy ra từ cùng phân phối Router:

\[
P(SUPPORTED)=\sum_{c\in\{cu,gia\_cam,gia\_suc,qua,rau\}} P(c)
\]

và `P(UNSUPPORTED)`, `P(NON_FOOD)` là hai xác suất tương ứng; xem [`predict.py`](predict.py#L44-L63). Router 5 lớp không có gate.

Nếu Router chọn sai category, pipeline deployment dùng Expert sai. Evaluator có “oracle freshness” chạy Expert của nhãn category thật để tách lỗi Expert khỏi lỗi định tuyến tại [`evaluate_end_to_end.py`](evaluate_end_to_end.py#L88-L160). Oracle chỉ là công cụ phân tích, không phải chế độ triển khai thực tế.

## 4. Unified chi tiết

Với batch `[B,3,224,224]`, ví dụ MobileNetV3-Small tạo feature map `[B,576,7,7]`, GAP thành `[B,576,1,1]`, flatten `[B,576]`, shared classifier thành `[B,1024]`, rồi:

```text
shared vector [B,D]
  ├─ Linear(D,3) ─► food_logits       [B,3]
  ├─ Linear(D,5) ─► category_logits   [B,5]
  └─ Linear(D,3) ─► freshness_logits  [B,3]
```

`D=1024` cho MobileNetV3-Small, `D=1280` cho MobileNetV3-Large/EfficientNet-B0, `D=512` cho ResNet18. Luồng thực nằm tại [`food_ai/unified_model.py`, `forward`](food_ai/unified_model.py#L53-L76).

Ba head đều đọc cùng vector nhưng không truyền logits cho nhau trong forward. Sự phụ thuộc category→freshness chỉ xuất hiện khi giải mã: logits `HALF-FRESH` bị đặt `-inf` nếu category dự đoán không thuộc `gia_cam/gia_suc`, rồi mới softmax/argmax tại [`predict_unified.py`](predict_unified.py#L36-L42). Vì category dùng dự đoán, một lỗi category có thể thay đổi không gian freshness.

Masked loss dùng `IGNORE_INDEX=-100`. Food loss tính trên mọi ảnh; category/freshness chỉ lấy phần tử có target hợp lệ. Nếu mini-batch không có supported, hàm trả `logits.sum()*0.0`, giữ graph nhưng gradient bằng 0 cho head đó; xem [`food_ai/unified_loss.py`](food_ai/unified_loss.py#L50-L75). Cross entropy mặc định reduction `mean`, nên từng component được chuẩn hóa theo số mẫu hợp lệ của chính nó trước khi cộng.

Shared backbone nhận gradient từ cả ba nhiệm vụ. Điều này tiết kiệm model và có thể tạo regularization đa nhiệm, nhưng cũng có negative transfer: food gate cần đặc trưng OOD rộng, category cần hình dạng/loài, freshness cần màu/texture tinh tế. Mã hiện tại không có GradNorm, uncertainty weighting, PCGrad hay đo cosine gradient; ảnh hưởng cạnh tranh gradient chưa được kiểm chứng.

---

# IV. Công thức và thuật toán huấn luyện

## 1. Cross Entropy, weighting và label smoothing

Với logits `z_k`, softmax:

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

Cross Entropy một mẫu nhãn thật `y`:

\[
L_{CE}=-\log p_y
\]

Weighted CE nhân đóng góp lớp `y` với `w_y` trước khi reduction, giúp lớp hiếm ảnh hưởng mạnh hơn. Unified tính trọng số từ số mẫu train `n_c` tại [`food_ai/unified_loss.py`](food_ai/unified_loss.py#L10-L28):

\[
w_c\propto n_c^{-1/2}\quad(\text{sqrt\_inverse}),\qquad
w_c\propto n_c^{-1}\quad(\text{inverse})
\]

sau đó chia trung bình trọng số để `mean(w)=1`. Legacy hiện chỉ áp dụng class weight cho Router; Expert dùng CE không weight tại [`train.py`](train.py#L291-L307).

Label smoothing thay target one-hot cứng bằng phân phối có một phần khối lượng cho lớp khác, giảm quá tự tin. Script mặc định `0.05`; implementation chuyển tham số này vào `CrossEntropyLoss` tại [`food_ai/unified_loss.py`](food_ai/unified_loss.py#L50-L56). Với `K` lớp, trực giác là phần lớn mass ở nhãn đúng và `ε` được phân tán; công thức chi tiết tuân theo PyTorch `CrossEntropyLoss`, repository không tự cài lại.

## 2. Multi-task loss Unified

\[
L_{total}=\lambda_fL_{food}+\lambda_cL_{category}+\lambda_rL_{freshness}
\]

- `L_food`: mọi ảnh, gồm supported và hai negative;
- `L_category`: chỉ supported có category;
- `L_freshness`: chỉ supported có freshness;
- mỗi loss là mean trên tập hợp hợp lệ của mini-batch;
- mini-batch không có supported cho component tương ứng nhận zero differentiable.

Mặc định `λ_food=λ_category=λ_freshness=1` trong [`train_unified.py`](train_unified.py#L43-L77) và [`food_ai/unified_loss.py`](food_ai/unified_loss.py#L31-L75). Đây là baseline cân bằng số học, không phải bằng chứng tối ưu. Do scale/độ khó/gradient của từng head khác nhau, một head vẫn có thể lấn át dù lambda bằng nhau.

## 3. Optimizer và scheduler

AdamW dùng moment bậc một/hai của gradient và weight decay tách khỏi bước gradient. Dạng rút gọn:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\theta_t\leftarrow(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Code dùng `torch.optim.AdamW` với learning rate mặc định CLI `5e-4`, weight decay `1e-4`; các beta không được override. LR quá lớn có thể phá pretrained features/dao động; quá nhỏ hội tụ chậm. Weight decay hạn chế trọng số lớn và overfit.

`ReduceLROnPlateau` theo dõi validation loss, giảm LR một nửa sau một epoch không cải thiện (`factor=0.5`, `patience=1`) tại [`train_unified.py`](train_unified.py#L293-L306) và [`train.py`](train.py#L310-L315). Legacy còn hỗ trợ cosine scheduler.

Batch size quyết định số ảnh/gradient step: lớn dùng nhiều VRAM, gradient ổn định nhưng ít nhiễu; nhỏ tiết kiệm VRAM nhưng dao động. Epoch là một lượt qua tập train. Seed 42 cố định Python/NumPy/PyTorch và sampling, nhưng Legacy bật `cudnn.benchmark=True`, nên không bảo đảm bitwise deterministic trên mọi GPU; xem [`train.py`](train.py#L188-L196).

## 4. Chọn checkpoint, overfitting và early stopping

Legacy lưu best theo **validation loss thấp nhất**, stale 3 epoch thì dừng; sau đó reload best và test tại [`train.py`](train.py#L349-L446). Unified mặc định chọn **food macro-F1 cao nhất**, validation loss dùng phá hòa; stale 3 epoch thì dừng tại [`train_unified.py`](train_unified.py#L330-L437).

Overfitting được nhận biết khi train loss tiếp tục giảm/accuracy tăng nhưng validation loss tăng hoặc metric giảm. Code ghi history và early stopping nhưng không tự chẩn đoán nguyên nhân. Cần vẽ learning curves từ `metrics.csv` thay vì chỉ nhìn test cuối.

Một bất thường cần báo cáo: run Unified EfficientNet-B0 lưu `best_valid_loss: NaN` tại [`summary.json`](runs/20261020_222831_efficientnet_b0_v1/summary.json#L1-L5). Checkpoint vẫn được chọn nhờ food macro-F1, nhưng loss NaN làm mất ý nghĩa tie-breaker/scheduler và cần điều tra trước khi coi run hoàn toàn tin cậy.

---

# V. Cơ chế dự đoán và tính điểm

## 1. Logits, softmax, argmax và confidence

Logit là điểm chưa chuẩn hóa. Softmax biến vector logits thành phân phối tổng bằng 1; argmax chọn chỉ số lớn nhất. “Confidence” trong code là xác suất softmax của top-1, không phải xác suất mô hình chắc chắn đúng theo nghĩa thống kê đã hiệu chuẩn. Neural network có thể overconfident khi gặp ảnh lạ.

Legacy Router confidence là max softmax Router. Food confidence với Router 7 lớp dùng tổng năm xác suất category cho `SUPPORTED`; xem [`predict.py`](predict.py#L27-L48). Unified có food head riêng nên `P(SUPPORTED)` là phần tử `SUPPORTED` của softmax ba lớp, không phải tổng category; xem [`predict_unified.py`](predict_unified.py#L31-L81).

## 2. Công thức điểm độ tươi thực tế

Các anchor được định nghĩa tại [`food_ai/config.py`](food_ai/config.py#L18-L22):

\[
a_{FRESH}=10,\qquad a_{HALF}=5.5,\qquad a_{SPOILED}=1
\]

Điểm là kỳ vọng trên **toàn bộ phân phối xác suất hợp lệ**, không chỉ nhãn argmax:

\[
Score=\sum_{s\in S(category)}P(s\mid image)\,a_s
\]

Legacy tính tại [`predict.py`](predict.py#L74-L96); Unified tại [`predict_unified.py`](predict_unified.py#L83-L116). Điểm được làm tròn hai chữ số. Vì là tổ hợp lồi của các anchor nên tự nằm trong `[1,10]`; không có clamp riêng. Router/category confidence không được nhân vào score.

Nhóm hai trạng thái mask/bỏ `HALF-FRESH`, nên:

\[
Score=10P(FRESH)+1P(SPOILED)=1+9P(FRESH)
\]

Nhóm ba trạng thái:

\[
Score=10P(FRESH)+5.5P(HALF)+1P(SPOILED)
\]

Ví dụ tính toán minh họa đúng công thức code:

1. `cu`: `(FRESH=0,90; SPOILED=0,10)` → `0,90×10+0,10×1=9,10`.
2. `gia_suc`: `(0,20; 0,50; 0,30)` → `2+2,75+0,30=5,05`.
3. `rau`: `(0,55; 0,45)` → `5,5+0,45=5,95`; dù top-1 là FRESH, điểm không phải 10 vì phân phối còn bất định.

Điểm thường trên 9 khi model gán `P(FRESH)>0,888...` cho bài toán hai lớp. Đây không phải phép đo hóa-lý hay phần trăm độ tươi; nó là chỉ số quy ước từ xác suất phân loại và ba anchor do hệ thống đặt.

Legacy và Unified dùng cùng anchor/kỳ vọng, nhưng phân phối đến từ model khác nhau. Unified còn mask lớp không hợp lệ theo category dự đoán trước khi tính.

## 3. Chính sách từ chối

Legacy CLI hiện tại: chỉ từ chối khi top-1 Router là `UNSUPPORTED/NON_FOOD`; không threshold. Nếu top-1 supported dù confidence thấp, vẫn chạy Expert. Router 5 lớp không thể từ chối.

Unified CLI có ba threshold mặc định tại [`predict_unified.py`](predict_unified.py#L45-L51), [`predict_unified.py`](predict_unified.py#L67-L98): food `0,70`, category `0,50`, freshness `0,50`. Ảnh bị từ chối nếu food top-1 không phải supported hoặc top-1 food confidence dưới 0,70; sau đó kết quả chỉ `accepted` khi category và freshness đều vượt 0,50.

Tuy nhiên runtime web ONNX giải mã bằng top-1 và không áp các threshold CLI tại [`local_web/server.py`](local_web/server.py#L327-L380). Vì vậy chính sách web và `predict_unified.py` hiện không đồng nhất; cần thống nhất trước khi báo cáo tỷ lệ từ chối/triển khai di động.

---

# VI. Đánh giá và kết quả thực nghiệm

## 1. Định nghĩa và cách code tính

Với một lớp theo one-vs-rest:

\[
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
\]

\[
Accuracy=\frac{\text{số dự đoán đúng}}{N},\qquad
MacroF1=\frac{1}{K}\sum_{k=1}^{K}F1_k
\]

Code đếm TP/FP/FN, xử lý mẫu số 0 và lập confusion matrix tại [`food_ai/experiment.py`, `classification_report`](food_ai/experiment.py#L52-L76). Ví dụ giả định 100 ảnh, TP=40, FP=10, FN=5: precision=0,80; recall≈0,889; F1≈0,842.

- Accuracy dễ bị lớp đông chi phối.
- Precision trả lời “những ảnh model gọi là lớp này, bao nhiêu đúng?”.
- Recall trả lời “ảnh thật của lớp này, model tìm được bao nhiêu?”. Recall negative quan trọng để không lọt ảnh ngoài phạm vi.
- Macro-F1 cho mỗi lớp trọng số bằng nhau, phù hợp dữ liệu mất cân bằng.
- Confusion matrix chỉ rõ nhầm lớp nào sang lớp nào.
- False Acceptance Rate có thể định nghĩa `negative predicted SUPPORTED / total negative`; False Rejection Rate là `supported predicted negative / total supported`. Repository chưa có hàm/báo cáo đặt tên trực tiếp FAR/FRR: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.
- Joint/end-to-end accuracy chỉ đúng khi toàn chuỗi đúng. Web tính supported đúng khi category và freshness cùng đúng (và food đúng nếu gate tồn tại); negative đúng khi food status đúng tại [`local_web/server.py`](local_web/server.py#L1194-L1234), [`local_web/server.py`](local_web/server.py#L1310-L1353).

## 2. Legacy đã huấn luyện

### Router 5 lớp MobileNetV3-Large

Run hiện gắn với checkpoint Router có best epoch 12, test accuracy **99,6328%**, macro-F1 **0,996263** trên 14.162 ảnh supported; nguồn [`summary.json`](runs/20261018_234532_router_all_mobilenet_v3_large/summary.json#L1-L42). Đây chỉ là category Router 5 lớp, không đo negative gate.

### Năm Expert MobileNetV3-Large

| Expert | Trạng thái | Best epoch | Test accuracy | Macro-F1 | Nguồn |
|---|---|---:|---:|---:|---|
| cu | FRESH/SPOILED | 1 | 96,9443% | 0,958526 | [`summary`](runs/20261019_123835_freshness_cu_mobilenet_v3_large/summary.json#L1-L21) |
| gia_cam | 3 lớp | 5 | 98,7528% | 0,989637 | [`summary`](runs/20261019_130433_freshness_gia_cam_mobilenet_v3_large/summary.json#L1-L28) |
| gia_suc | 3 lớp | 8 | 99,7517% | 0,997513 | [`summary`](runs/20261019_133832_freshness_gia_suc_mobilenet_v3_large/summary.json#L1-L28) |
| qua | FRESH/SPOILED | 10 | 99,6825% | 0,996718 | [`summary`](runs/20261019_154807_freshness_qua_mobilenet_v3_large/summary.json#L1-L21) |
| rau | FRESH/SPOILED | 8 | 98,9875% | 0,987429 | [`summary`](runs/20261019_181216_freshness_rau_mobilenet_v3_large/summary.json#L1-L21) |

Không được lấy trung bình tùy ý các accuracy Expert rồi gọi là end-to-end. Sai số Router và phân bố category phải được đưa vào evaluator. Repository có code [`evaluate_end_to_end.py`](evaluate_end_to_end.py#L63-L160), nhưng không thấy báo cáo full-test đã lưu: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

### Router 7 lớp

Accuracy, macro-F1, recall `UNSUPPORTED`, recall `NON_FOOD`, FAR/FRR: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

## 3. Unified đã huấn luyện

Các kết quả dưới đây là test summary của từng run. Category/freshness được tính trên ảnh có nhãn hợp lệ; đây là metric từng head, chưa phải joint deployment.

| Backbone | Best epoch | Food acc. | Food macro-F1 | Category acc. | Freshness acc. | Recall UNSUPPORTED | Recall NON_FOOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 6 | 0,993551 | 0,963574 | 0,994563 | 0,983194 | 0,985333 | 0,916000 |
| MobileNetV3-Large | 12 | 0,996744 | 0,982603 | 0,996540 | 0,990397 | 0,984000 | 0,966667 |
| EfficientNet-B0 | 10 | 0,996680 | 0,981706 | 0,997034 | 0,993363 | 0,980000 | 0,964000 |
| ResNet18 | 4 | 0,991827 | 0,959601 | 0,990256 | 0,971191 | 0,950667 | 0,921333 |

Nguồn: [`MobileNetV3-Small summary`](runs/20261020_100924_unified_v1/summary.json#L1-L160), [`MobileNetV3-Large summary`](runs/20261020_193319_mobilenet_v3_large_v1/summary.json#L1-L160), [`EfficientNet-B0 summary`](runs/20261020_222831_efficientnet_b0_v1/summary.json#L1-L160), [`ResNet18 summary`](runs/20261021_050354_resnet18_v1/summary.json#L1-L160).

Lưu ý: `evaluate_unified.py` tái tạo criterion nhưng không nạp food class weights như lúc train tại [`evaluate_unified.py`](evaluate_unified.py#L20-L48). Metrics argmax không bị đổi, nhưng loss từ script này có thể không trực tiếp so với weighted train/test loss.

## 4. Benchmark web hiện có

Báo cáo [`evaluation_20261021T064242Z_06b293af.json`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L1-L20) lấy ngẫu nhiên **200/15.662** candidates, gồm 184 supported và 16 negative, seed `510420056`; không phải full-test. CPU ONNX Runtime, batch 32, một warmup; thời gian model không gồm đọc/preprocess/session initialization.

| Artifact | Joint acc. trên mẫu | Mean ms/ảnh | p50 | p95 | Throughput ảnh/s | ONNX MiB |
|---|---:|---:|---:|---:|---:|---:|
| Unified EfficientNet-B0 | 0,985 | 20,099 | 20,016 | 21,030 | 49,755 | 15,332 |
| Unified MobileNetV3-Large | 0,985 | 8,001 | 7,909 | 8,642 | 124,988 | 16,074 |
| Unified ResNet18 | 0,980 | 36,115 | 36,452 | 36,803 | 27,689 | 42,650 |
| Unified MobileNetV3-Small | 0,970 | 3,116 | 3,137 | 3,250 | 320,930 | 5,843 |
| Legacy “original” 5 lớp | 0,994565 trên 184 supported | 75,383 | 74,931 | 77,740 | 13,266 | 255,789 |

Nguồn các block metric/speed: [`JSON#L44-L81`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L44-L81), [`#L109-L146`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L109-L146), [`#L174-L211`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L174-L211), [`#L239-L276`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L239-L276), [`#L320-L359`](reports/web_evaluations/evaluation_20261021T064242Z_06b293af.json#L320-L359).

Không thể dùng bảng này kết luận accuracy Unified kém/tốt hơn Legacy về food gate: Legacy 5 lớp không đánh giá 16 negative. Cỡ mẫu negative quá nhỏ để ước lượng chắc chắn recall/FAR. RAM inference và kết quả ảnh camera thật: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

## 5. Vì sao accuracy có thể rất cao?

Các giả thuyết hợp lý gồm tập dữ liệu dễ, background/source artifacts, sản phẩm giữa lớp quá khác nhau, near-duplicate vượt ngoài ngưỡng pHash, cùng phiên chụp xuất hiện ở nhiều split, hoặc augmentation chưa phản ánh camera thật. Script đã loại exact SHA-256 và gom pHash theo thiết kế, nên có biện pháp chống leakage trực tiếp. Tuy nhiên không có audit theo nguồn/phiên chụp/đối tượng vật lý và không có external test; vì vậy **không đủ bằng chứng để khẳng định data leakage**, cũng không đủ để khẳng định tổng quát hóa thực tế.

---

# VII. So sánh Legacy và Unified

| Tiêu chí | Legacy hiện có | Unified hiện có | Kết luận hợp lệ |
|---|---|---|---|
| Kiến trúc | Router + 5 Expert, định tuyến cứng | 1 shared backbone + 3 head | Unified chia sẻ feature; Legacy chuyên biệt theo category. |
| Backbone artifact chính | MobileNetV3-Large | 4 backbone đã thử | Không có một backbone duy nhất cho mọi Unified artifact. |
| Số model/file ONNX | 6 | 1/artifact | Lợi thế thiết kế trực tiếp cho Unified. |
| Forward/ảnh | 2 nếu supported; Router7 negative là 1 | 1 | Unified ít forward hơn. |
| Category accuracy | Router5: 99,6328% full supported test | 99,4563–99,7034% tùy run | Không so trực tiếp thành end-to-end/gate. |
| Freshness accuracy | 96,9443–99,7517% theo Expert | 97,1191–99,3363% từng head | Khác cách phân rã; cần joint full-test. |
| Food macro-F1 | Router5: không áp dụng; Router7 chưa có | 0,959601–0,982603 | Chỉ Unified có bằng chứng. |
| Recall NON_FOOD | Chưa có Router7 | 0,916–0,966667 | Chỉ Unified có bằng chứng. |
| Recall UNSUPPORTED | Chưa có Router7 | 0,950667–0,985333 | Chỉ Unified có bằng chứng. |
| End-to-end/joint full-test | Chưa có báo cáo lưu | Chưa có báo cáo deployment full-test | **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. |
| Tổng ONNX benchmark | 255,789 MiB bundle Legacy | 5,843–42,650 MiB/model | Unified nhỏ hơn trong artifact benchmark hiện có. |
| CPU p50/p95 | 74,931/77,740 ms | 3,137/3,250 đến 36,452/36,803 ms | Chỉ đúng cho sample/hardware/protocol đã lưu. |
| RAM | Chưa đo | Chưa đo | **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. |
| Tổng quát hóa camera | Chưa có external test | Chưa có external test | **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**. |
| Ưu điểm | Expert chuyên biệt; dễ phân tích lỗi theo nhóm; oracle tách lỗi Router/Expert | Một file, một forward, feature dùng chung, negative gate tích hợp | Đây là lợi thế cấu trúc. |
| Nhược điểm | Cascade error, sáu model, dung lượng/latency lớn, Router5 không từ chối | Gradient cạnh tranh; một backbone phải phục vụ ba nhiệm vụ; policy threshold chưa đồng nhất | Cần ablation và full-test để định lượng. |

Unified tốt hơn rõ ràng về số file và số lần forward. Benchmark mẫu cũng cho thấy ONNX nhỏ và nhanh hơn trong môi trường đã đo. Không đủ bằng chứng để kết luận Unified tốt hơn Legacy về accuracy tổng thể hay khả năng tổng quát hóa.

---

# VIII. Triển khai ONNX và Android

## 1. Export và hợp đồng ONNX

Legacy export từng checkpoint với dummy `[1,3,224,224]`, input `image`, output `logits`, dynamic batch axis và opset 18 tại [`export_onnx.py`](export_onnx.py#L39-L63). Bundle gồm một Router và năm Expert; manifest ghi classes, gate readiness, kích thước và SHA-256 tại [`export_onnx.py`](export_onnx.py#L101-L146).

Unified export một graph, input `image`, ba output `food_logits`, `category_logits`, `freshness_logits`, dynamic batch, opset 18 tại [`export_unified_onnx.py`](export_unified_onnx.py#L67-L117). Manifest gắn architecture, class order, preprocessing, checkpoint/ONNX checksum.

`verify_onnx.py` kiểm tra graph, top-1 và max probability difference `<1e-5` cho sáu Legacy model tại [`verify_onnx.py`](verify_onnx.py#L34-L117). Unified so ba tensor output với tolerance mặc định `1e-5` tại [`verify_unified_onnx.py`](verify_unified_onnx.py#L21-L53). Kết quả verify đã lưu cho artifact hiện tại: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

Một vấn đề quản lý artifact: manifest mặc định Unified mô tả ONNX MobileNetV3-Small/checkpoint hash cũ, trong khi `models/unified/food_multitask.pth` hiện là ResNet18. Vì vậy không nên dùng cặp đường dẫn mặc định để verify lẫn nhau nếu chưa chọn đúng checkpoint của run. Snapshot `models/legacy/original` cũng có thời điểm ONNX/PTH khác nhau và không có verify log; cần package artifact bất biến theo run.

## 2. Runtime

Legacy cần quản lý sáu session hoặc lazy-load Router + Expert được chọn; Unified một session trả ba output. Web dùng `CPUExecutionProvider`, warmup và benchmark batch tại [`local_web/server.py`](local_web/server.py#L216-L380), [`local_web/server.py`](local_web/server.py#L384-L628).

Repository không có `pubspec.yaml`, mã Dart hay Android bridge. Vì vậy cách ONNX Runtime được tích hợp vào Flutter, thời gian khởi tạo trên điện thoại, RAM, thermal throttling và camera preprocessing: **CHƯA CÓ BẰNG CHỨNG TRONG REPOSITORY**.

FP16/INT8 chưa có pipeline export/calibration/quantization. FP16 thường hữu ích trên GPU/NPU hỗ trợ; INT8 có thể giảm size và tăng tốc CPU nhưng cần representative calibration data hoặc QAT, đồng thời phải đo suy giảm food recall/freshness. Không được coi đây là tính năng đã hoàn thành.

Giới hạn di động cần kiểm chứng: memory của nhiều session Legacy, thời gian decode/resize ảnh, khác biệt provider, pin/nhiệt, operator support opset 18, dynamic batch không cần thiết cho camera batch 1 và độ chính xác sau quantization.

---

# IX. Hạn chế và định hướng nghiên cứu

| Hướng | Vấn đề/nguyên lý | Thay đổi và thí nghiệm | Chỉ số | Rủi ro | Ưu tiên |
|---|---|---|---|---|---|
| Che vùng 3×3 | Chẩn đoán vùng nào chi phối: lần lượt che 9 ô và đo delta xác suất/score. | Viết evaluator tạo 9 biến thể/ảnh; giữ nguyên model; so foreground/background nếu có mask. | Δ đúng lớp, flip rate, score variance. | Ô lớn có thể che cả vật thể; không tự “bỏ nền”. | **P0**, rẻ và giải thích được. |
| Grad-CAM | Chiếu gradient lớp lên feature map để quan sát vùng chú ý. | Hook layer cuối mỗi backbone; tạo heatmap trên tập lỗi/đúng. | Tỷ lệ năng lượng trong foreground; pointing game nếu có mask. | Heatmap thô, dễ diễn giải quá mức. | **P0**. |
| External/camera test | Kiểm tra domain shift thực tế. | Thu bộ ảnh mới theo thiết bị/ánh sáng/nền, khóa trước khi train. | Joint acc., macro-F1, FAR/FRR, ECE. | Tốn gán nhãn; phải tránh trùng nguồn. | **P0**. |
| Background randomization | Cắt foreground và ghép nhiều nền để phá tương quan nền–nhãn. | Cần mask/segmentation; tạo train variants, giữ test thật. | Worst-group acc., background-swap consistency. | Biên ghép giả tạo thành shortcut mới; màu ánh sáng không khớp. | **P1**. |
| Object cropping | Dùng detector/segmenter crop vùng thực phẩm trước classifier. | Gán box/mask hoặc dùng detector pretrained; đánh giá crop miss. | Detection recall, joint classifier acc., latency. | Detector bỏ sót dấu hỏng/đa vật thể. | **P1**. |
| Confidence calibration | Softmax confidence chưa phải xác suất đúng; temperature scaling học `T` trên validation. | Fit `softmax(z/T)` sau train; không dùng test để fit. | ECE, NLL, Brier, risk–coverage. | Calibration lệch khi domain đổi. | **P1**. |
| OOD detection | Negative hữu hạn không bao phủ mọi ảnh lạ. Dùng energy/MSP/embedding distance. | Xây OOD validation/test độc lập; tune threshold chỉ trên validation. | AUROC, AUPR, FPR@95TPR, FAR. | OOD không có phân phối duy nhất; threshold phụ thuộc domain. | **P1**. |
| Multi-task loss tuning | Ba gradient có thể cạnh tranh. | Grid lambda; ghi norm/cosine gradient; thử uncertainty weighting/GradNorm/PCGrad. | Mỗi head macro-F1, joint acc., Pareto front. | Tăng không gian thí nghiệm và overfit validation. | **P1**. |
| Nhiều thực phẩm/ảnh | Classifier toàn ảnh chỉ trả một category/freshness. | Gán box/mask; detector hoặc instance segmentation, rồi đánh giá từng instance. | mAP/IoU + per-instance freshness/joint. | Chi phí nhãn và latency cao; freshness có thể không đồng nhất trong vật thể. | **P2**. |
| Quantization | Giảm size/latency Android. | PTQ INT8 với calibration đại diện; nếu giảm mạnh thì QAT; benchmark thiết bị thật. | Size, p50/p95, RAM, pin và delta metric. | Operator/provider không hỗ trợ; giảm recall lớp hiếm. | **P1** sau khi khóa baseline. |
| Android optimization | Biết chi phí end-to-end thật, không chỉ ONNX CPU desktop. | Prototype ORT Mobile, batch1, thread tuning, reuse buffers/session, profile decode→render. | cold start, warm p50/p95, peak RSS, energy. | Phân mảnh thiết bị và thermal throttling. | **P1**. |

Tất cả các hướng trong bảng là **đề xuất/chưa kiểm chứng**, ngoại trừ baseline augmentation/evaluator hiện có. Thứ tự nên làm: khóa artifact và full-test → external camera test → chẩn đoán shortcut → calibration/OOD → tối ưu loss → quantization/Android → detection đa vật thể.

---

# X. Kết luận dùng cho thuyết trình

1. Legacy giải bài toán bằng phân rã tuần tự: Router nhận dạng nhóm, một Expert chuyên biệt đánh giá độ tươi. Nó dễ giải thích nhưng có cascade error, sáu model và Router 5 lớp không biết từ chối ảnh ngoài phạm vi.
2. Mã đã mở rộng Legacy thành Router 7 lớp bằng `UNSUPPORTED/NON_FOOD`, nhưng repository chưa có checkpoint/kết quả 7 lớp; không được trình bày phần này như thí nghiệm hoàn tất.
3. Unified dùng một feature vector chung và ba head. Masked loss cho phép negative chỉ huấn luyện food head; category/freshness chỉ học từ supported. Khi inference, freshness được giới hạn theo category.
4. Điểm độ tươi là kỳ vọng xác suất với anchor 10/5,5/1, không phải phép đo vật lý. Confidence softmax cũng chưa được calibration.
5. Unified có lợi thế đã chứng minh về cấu trúc: một file, một forward. Benchmark mẫu CPU cho thấy nhỏ và nhanh hơn bundle Legacy, nhưng chưa đủ để kết luận accuracy/tổng quát hóa tốt hơn.
6. Điểm yếu bằng chứng hiện tại là thiếu Legacy 7-class run, thiếu full-test end-to-end chuẩn hóa, thiếu external camera/Android benchmark, thiếu RAM và thiếu kiểm tra shortcut có định lượng.

## Bộ câu hỏi giảng viên có thể hỏi

- **Vì sao không chỉ dùng accuracy?** Vì supported áp đảo negative; macro-F1 và recall từng negative cho mỗi lớp trọng số rõ ràng hơn.
- **Unified có thật sự luôn tốt hơn?** Không. Nó chắc chắn gọn hơn theo thiết kế và nhanh hơn trong benchmark mẫu; accuracy/tổng quát hóa chưa có thí nghiệm đối chứng đầy đủ.
- **Điểm 9,5 có nghĩa 95% tươi không?** Không. Đó là kỳ vọng từ anchor do hệ thống quy ước.
- **Tại sao cần masked loss?** Negative không có category/freshness hợp lệ; ép nhãn sẽ dạy hai head thông tin sai.
- **Tại sao Router sai làm Expert sai?** Vì định tuyến hard argmax chỉ gọi một Expert; không có weighted mixture hay fallback.
- **Dữ liệu đã hết leakage chưa?** Exact duplicate và pHash cluster đã được xử lý, nhưng chưa có audit nguồn/phiên chụp/external test nên không thể khẳng định tuyệt đối.
- **Tại sao không dùng ảnh 3×3 ngay để train?** Chia cứng có thể tách vật thể/dấu hiệu hỏng khỏi ngữ cảnh; trước tiên nên dùng occlusion 3×3 như một phép chẩn đoán, sau đó mới quyết định augmentation/segmentation.

