# Báo cáo triển khai kế hoạch model

## 1. Mục đích

Tài liệu này ghi lại các thay đổi đã thực hiện trong quá trình kiểm tra và chuẩn hóa pipeline model. Báo cáo phân biệt giữa:

- Thay đổi đã thực hiện.
- Kết quả đã kiểm tra.
- Vấn đề đã phát hiện nhưng chưa sửa.
- Artifact hoặc thay đổi có sẵn trong workspace nhưng chưa xác định nguồn.

## 2. Tóm tắt trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Category contract | Đã đồng bộ config và ONNX inference |
| Preprocessing PyTorch/ONNX | Đã dùng chung `build_transforms()` |
| Phát hiện duplicate | Đã có công cụ SHA-256; dataset vẫn còn leakage |
| Đánh giá end-to-end | Đã có công cụ; hiện lỗi với expert 2/3 lớp |
| Router uncertainty | Đã có công cụ đo confidence/entropy |
| Benchmark MobileNetV3 | Đã có runner benchmark, chưa chạy train benchmark mới |
| Manifest và assets | Đã có công cụ tạo manifest/copy sidecar ONNX |
| Regression tests | Đã có test cơ bản |
| Dataset sạch | Chưa đạt |
| Model phát hành mới | Chưa train/chọn lại |

## 3. Các bước đã thực hiện

### Bước 1 - Đồng bộ category contract

Đã thay đổi thứ tự `CATEGORIES` trong `food_ai/config.py` thành:

```text
cu, gia_cam, gia_suc, qua, rau
```

Thứ tự này khớp thứ tự alphabetic của `ImageFolder` và category mapping đang dùng
trong ONNX inference.

`inference.py` cũng đã bỏ danh sách category/freshness tự khai báo và dùng
constant từ `food_ai.config`.

**Phạm vi:** chỉ đồng bộ label order; không thay đổi model architecture hoặc
trọng số.

### Bước 2 - Chuẩn hóa preprocessing

`inference.py` hiện dùng `build_transforms()` từ `food_ai.data`, thay cho
preprocessing riêng resize trực tiếp về `224x224`.

Preprocessing eval/inference chung:

```text
RGB
Resize(232)
CenterCrop(224)
ToTensor
ImageNet normalization
```

**Phạm vi:** PyTorch validation/test và ONNX Python inference dùng cùng helper.
Client Flutter chưa được kiểm tra trong workspace này.

### Bước 3 - Phát hiện duplicate và leakage

Đã thêm `scripts/check_duplicates.py`:

- Quét ảnh trong `data/train` và `data/valid`.
- Tính SHA-256 theo từng file.
- Báo ảnh byte-identical nằm ở nhiều split.
- Trả exit code `1` khi phát hiện cross-split duplicate.

Kết quả dataset hiện tại: có nhiều nhóm ảnh trùng giữa train và valid, kể cả ảnh
khác tên hoặc khác đường dẫn. Vì vậy metric hiện tại có thể bị lạc quan và chưa
chứng minh model tổng quát hóa.

Chưa xóa hoặc di chuyển ảnh vì cần xác định bản gốc, augmentation và nhãn đúng.
Near-duplicate sau resize/crop/nén vẫn chưa được kiểm tra.

### Bước 4 - Đánh giá end-to-end

Đã thêm `evaluate_end_to_end.py`, dự kiến đo:

- Router accuracy/macro-F1/confusion matrix.
- Freshness expert khi category đúng, gọi là oracle evaluation.
- Freshness end-to-end khi router tự chọn expert.

Khi chạy trên dataset hiện tại, công cụ dừng tại expert không có class
`HALF-FRESH`. Lý do là các expert hiện tại không đồng nhất:

```text
cu       : FRESH, SPOILED
gia_cam  : FRESH, HALF-FRESH, SPOILED
gia_suc  : FRESH, HALF-FRESH, SPOILED
qua      : FRESH, SPOILED
rau      : FRESH, SPOILED
```

Vì vậy công cụ chưa được dùng để công bố metric end-to-end chính thức.

### Bước 5 - Đánh giá uncertainty router

Đã thêm `evaluate_router_uncertainty.py`, đo:

- Mean confidence.
- Entropy.
- Accuracy theo confidence bin.
- Coverage.
- Accuracy trong số prediction được chấp nhận theo threshold.

Đã chạy thành công trên `data/valid` với threshold `0.8`:

```text
samples: 13675
accuracy: 0.9961974406
mean_confidence: 0.9583227112
coverage: 0.9962705667
accepted_accuracy: 0.9977980035
rejected_samples: 51
```

Các số này chỉ có giá trị chẩn đoán vì dataset vẫn có leakage.
Threshold chưa được bật trong inference production.

### Bước 6 - Benchmark MobileNetV3

Đã thêm `benchmark_mobilenet.py` với ba trial có cùng cấu hình:

```text
mobilenet_v3_large + fine_tune
mobilenet_v3_small + fine_tune
mobilenet_v3_small + head_only
```

Các trial dùng chung seed, optimizer, scheduler, learning rate, weight decay,
label smoothing, epoch và batch size.

Runner chưa chạy benchmark mới vì dataset chưa sạch; không dùng metric hiện tại
để chọn model phát hành.

### Bước 7 - Manifest và assets

Đã thêm `package_models.py`, tạo manifest chứa:

- Model name.
- Training method.
- Class order.
- Preprocessing contract.
- SHA-256 checkpoint và ONNX.
- SHA-256 file `.onnx.data` nếu có.

Có thể copy assets bằng `--assets-destination`. Sidecar `.onnx.data` được copy
cùng ONNX để tránh package thiếu external weights.

Manifest thực tế chưa được tạo vì chưa chạy lệnh package release.

### Bước 8 - Regression tests

Đã thêm `test_regression.py`, kiểm tra:

- Category/freshness order.
- Duplicate detector.
- Preprocessing shape và dtype.
- Manifest contract nếu manifest tồn tại.

Kết quả gần nhất:

```text
Ran 6 tests
OK (skipped=1)
```

Một test được skip vì `models/manifest.json` chưa tồn tại.

## 4. Tài liệu đã viết lại/bổ sung

Đã viết lại hoặc bổ sung:

- `README.md`.
- `ARCHITECTURE.md`.
- `ALGORITHM.md`.
- `DATA.md`.
- `TRAIN.md`.
- `LIBRARY_PRINCIPLES.md`.
- `MODEL_COMPARISON.md`.
- `PROJECT_FILES.md`.
- `IMPLEMENTATION_REPORT.md` này.

Tài liệu hiện mô tả model, preprocessing, CNN flow, single/multi-stage evaluation,
leakage, benchmark và chức năng từng file.


## 5. Kiểm tra đã thực hiện

Đã chạy thành công:

```text
Regression tests: 6 tests, OK, 1 skipped
Python compileall: PASS
Python pip check: No broken requirements found
Markdown diagnostics: không có lỗi
git diff --check: PASS
Router uncertainty evaluator: chạy thành công
```

End-to-end evaluator chưa đạt vì class contract freshness 2/3 lớp không đồng
nhất; đây là lỗi dữ liệu/contract cần xử lý, không phải lỗi môi trường Python.

## 6. Thay đổi có sẵn nhưng chưa gán nguồn

Tại thời điểm lập báo cáo, workspace có diff trong:

- `export_onnx.py`.
- `food_ai/model.py`.

Các diff này đã tồn tại trong workspace nhưng không được gán vào các bước triển
khai trong báo cáo này. Cần kiểm tra lịch sử hoặc bản backup trước khi quyết định
giữ, sửa hoặc hoàn tác.

## 7. Việc cần làm tiếp theo

1. Chốt freshness contract: tất cả expert 2 lớp hoặc tất cả expert 3 lớp.
2. Tạo metadata group/source/product/session cho dataset.
3. Làm sạch exact duplicate và near-duplicate.
4. Tạo lại train/validation/test theo group.
5. Sửa evaluator end-to-end theo contract đã chốt.
6. Train lại model trên split sạch.
7. Chạy single-model và multi-stage evaluation.
8. Chạy benchmark MobileNetV3 trên cùng split sạch.
9. Tạo manifest release và verify ONNX/Flutter assets.
