# Kiến trúc hệ thống

Chi tiết từng bước xử lý ảnh, forward pass, softmax và tính điểm tươi nằm trong
[ALGORITHM.md](ALGORITHM.md).

## 1. Mô hình phân tầng

Sơ đồ lớp CNN và phương pháp so sánh model được trình bày riêng tại
[MODEL_COMPARISON.md](MODEL_COMPARISON.md). Tài liệu đó phân biệt đánh giá một
classifier với đánh giá toàn bộ router -> expert.

```text
ảnh RGB
  -> Resize(232) + CenterCrop(224) + ImageNet normalization
  -> router
  -> category
  -> freshness expert của category
  -> freshness class
  -> điểm tươi ước tính
```

Router và expert là các model độc lập. Router không truyền feature sang expert; expert nhận lại tensor ảnh đã tiền xử lý.

## 2. Category contract

Thứ tự category canonical hiện tại là:

```text
0 cu
1 gia_cam
2 gia_suc
3 qua
4 rau
```

Thứ tự này khớp `food_ai.config.CATEGORIES`, thứ tự alphabetic của `ImageFolder` và ONNX inference hiện tại. Tuy nhiên checkpoint vẫn lưu `classes` riêng, vì vậy export/release phải kiểm tra metadata thay vì chỉ dựa vào constant.

## 3. Freshness contract thực tế

Contract cấu hình có ba nhãn:

```text
FRESH, HALF-FRESH, SPOILED
```

Nhưng checkpoint hiện tại có trạng thái:

| Expert | Class thực tế |
|---|---|
| `cu` | `FRESH`, `SPOILED` |
| `gia_cam` | `FRESH`, `HALF-FRESH`, `SPOILED` |
| `gia_suc` | `FRESH`, `HALF-FRESH`, `SPOILED` |
| `qua` | `FRESH`, `SPOILED` |
| `rau` | `FRESH`, `SPOILED` |

Không được dùng một danh sách freshness cố định để diễn giải mọi expert. Client phải đọc class order từ metadata model hoặc contract phát hành đã chốt.

## 4. Model

`food_ai/model.py` hỗ trợ:

- `mobilenet_v3_large` mặc định.
- `mobilenet_v3_small`.
- `efficientnet_b0`.
- `resnet18`.

Khi train, model dùng pretrained ImageNet và thay classifier cuối bằng số output của task. `fine_tune` cập nhật toàn mạng; `head_only` đóng băng backbone và chỉ cập nhật classifier.

Khi inference, model được dựng với `pretrained=False` rồi nạp `state_dict` từ checkpoint. Checkpoint nên có `model_name`, `classes`, `task`, `category` và `training_method`; các checkpoint cũ có thể thiếu một số metadata.

## 5. Điểm tươi

Expert chọn class có xác suất cao nhất. Điểm tươi là kỳ vọng theo anchor:

```text
FRESH      = 10.0
HALF-FRESH = 5.5
SPOILED    = 1.0
score = sum(probability[class] * anchor[class])
```

Với expert 2 lớp, score chỉ dùng các class thực tế trong checkpoint. Điểm này không phải phép đo an toàn thực phẩm.

## 6. ONNX

`export_onnx.py` export input `1 x 3 x 224 x 224`, dynamic batch và output logits. `inference.py` dùng preprocessing chung với PyTorch. Một số file ONNX có external data `.onnx.data`; phải đóng gói cả file sidecar.

`verify_onnx.py` so sánh xác suất PyTorch/ONNX trên cùng tensor và ngưỡng sai khác `1e-5`. Trước release cần kiểm tra thêm class order và metadata.

## 7. Giới hạn kiến trúc hiện tại

- Router chọn đúng một expert theo argmax; lỗi router kéo theo expert sai.
- Chưa có reject/`UNKNOWN` trong inference production.
- Không thể coi accuracy hiện tại là accuracy ngoài đời vì dataset có cross-split duplicate và cùng nguồn/product ở nhiều split.
- `evaluate_end_to_end.py` cần được sửa để hỗ trợ expert có số lớp khác nhau trước khi dùng làm báo cáo.
