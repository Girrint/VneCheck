# Unified model

Unified pipeline tồn tại song song với pipeline legacy. Không file legacy nào
bị đổi đường dẫn hoặc bị ghi đè.

## Kiến trúc

```text
Backbone dùng chung
  ├─ food_logits       : SUPPORTED / UNSUPPORTED / NON_FOOD
  ├─ category_logits   : cu / gia_cam / gia_suc / qua / rau
  └─ freshness_logits  : FRESH / HALF-FRESH / SPOILED
```

Một lần forward tạo cả ba output. Khi train negative image, category và
freshness dùng `ignore_index`; masked loss chỉ cập nhật food head cho mẫu đó.
Backbone có thể là `mobilenet_v3_small`, `mobilenet_v3_large`,
`efficientnet_b0` hoặc `resnet18`.

## Legacy snapshot

Baseline hiện tại đã được đóng băng tại:

```text
models/legacy/original/
```

Thư mục chứa 6 checkpoint PyTorch, 6 ONNX và `manifest.json` checksum. Đây là
baseline trước clean-split retraining, được giữ độc lập để rollback/đối chiếu.

Có thể tạo snapshot khác trước mỗi benchmark:

```powershell
python scripts\snapshot_legacy_models.py --name clean_split
```

Với model mới, cách ưu tiên là export một bundle ONNX có tên bằng
`python export_onnx.py --bundle-name <id>` sau khi train đủ router và 5 expert.

## Manifest dữ liệu

Chỉ khi cần baseline không có food gate, tạo manifest positive-only bằng:

```powershell
python scripts\build_unified_manifest.py
```

Negative data dùng cấu trúc:

```text
data_negative/
  train/NON_FOOD/...
  train/UNSUPPORTED/...
  valid/NON_FOOD/...
  valid/UNSUPPORTED/...
  test/NON_FOOD/...
  test/UNSUPPORTED/...
```

Sau khi có negative data:

```powershell
python scripts\build_unified_manifest.py --negative-root data_negative
```

Manifest hiện tại có 94.449 mẫu `SUPPORTED`, 5.000 `UNSUPPORTED` và 5.000
`NON_FOOD`. Cả ba split đều có đủ ba nhãn nên food gate được bật.

## Train và đánh giá

```powershell
python train_unified.py
python evaluate_unified.py
```

Food head mặc định dùng `sqrt_inverse` class weight tính từ split train. Cách
này giảm thiên lệch do 66.115 ảnh `SUPPORTED` so với 3.500 ảnh của mỗi lớp
negative nhưng không bỏ bớt positive, nên category/freshness head vẫn dùng toàn
bộ dữ liệu. Có thể tắt hoặc dùng inverse weight:

```powershell
python train_unified.py --food-class-balance none
python train_unified.py --food-class-balance inverse
```

Checkpoint tốt nhất mặc định được chọn theo validation food macro-F1, không
theo accuracy tổng. `metrics.csv` ghi precision/recall/F1 của từng food status
ở mỗi epoch; `summary.json` lưu đầy đủ confusion matrix và metric test.

Output không đụng model legacy:

```text
models/unified/food_multitask.pth
runs/<timestamp>_unified_*/
```

Freshness head có ba output nhưng inference tạm mask `HALF-FRESH` với `cu`,
`qua`, `rau` vì dataset các nhóm này chưa có nhãn tương ứng.

## Inference A/B

```powershell
python predict.py image.jpg
python predict_unified.py image.jpg
python predict_pipeline.py image.jpg --pipeline legacy
python predict_pipeline.py image.jpg --pipeline unified
python predict_pipeline.py image.jpg --pipeline both
```

Mặc định `predict_pipeline.py` dùng legacy. Có thể cấu hình:

```powershell
$env:AI_PIPELINE = "unified"
python predict_pipeline.py image.jpg
```

## Một ONNX

```powershell
python export_unified_onnx.py
python verify_unified_onnx.py --image image.jpg
```

Artifact mới:

```text
models/unified/food_multitask.onnx
models/unified/food_multitask.manifest.json
models/unified/manifest.json
```

### Lưu nhiều model để so sánh

Mỗi run giữ checkpoint riêng trong `runs/`. Export từng checkpoint với một tên
artifact duy nhất để không ghi đè model khác:

```powershell
python train_unified.py `
  --model mobilenet_v3_small `
  --run-name mobilenet_v3_small_v1

python train_unified.py `
  --model mobilenet_v3_large `
  --run-name mobilenet_v3_large_v1
```

Sau khi train, lấy `best_model.pth` trong đúng thư mục run tương ứng rồi export:

```powershell
python export_unified_onnx.py `
  --checkpoint runs\<run-small>\best_model.pth `
  --name mobilenet_v3_small_v1 `
  --display-name "MobileNetV3 Small v1"

python export_unified_onnx.py `
  --checkpoint runs\<run-large>\best_model.pth `
  --name mobilenet_v3_large_v1 `
  --display-name "MobileNetV3 Large v1"
```

Mỗi lệnh sinh một cặp độc lập:

```text
models/unified/exports/<name>.onnx
models/unified/exports/<name>.manifest.json
```

Sidecar manifest lưu tên hiển thị, kiến trúc dùng để train, training method,
run name, epoch và hash checkpoint. Web quét đệ quy `models/unified`, đưa mọi
artifact vào mục `Unified ONNX model`, đồng thời ghi model ID/tên/kiến trúc vào
CSV và JSON của mỗi lần đánh giá.

ONNX có một input `image` và ba output logits. Chỉ chuyển app sang unified sau
khi so sánh category accuracy, freshness macro-F1, recall `SPOILED`, dung lượng,
latency và peak memory trên cùng `data_split/test`.

## Web local đánh giá ONNX

Sau khi đã export ONNX:

```powershell
python -m local_web.server
```

Mở `http://127.0.0.1:8000`. Trang web lập chỉ mục ảnh positive từ
`data_split/test` và negative từ `data_negative/test`, cho phép xem một dự đoán hoặc đo accuracy trên
25/100/500/1.000/5.000 hay toàn bộ ảnh phù hợp bộ lọc. Mỗi lần chạy batch sinh
một seed mới để lấy nhóm ảnh khác; seed được lưu trong JSON tổng hợp để truy
lại lần đánh giá. Trong một lượt, cùng ảnh được chạy qua mọi Unified artifact
và/hoặc Legacy bundle được tích chọn. Với Legacy, pipeline thật dùng router → expert,
còn oracle dùng category thật để đo riêng expert 2 lớp hoặc 3 lớp. Unified chưa
có vẫn có thể đánh giá Legacy. Inference dùng ONNX Runtime CPU và tiền xử lý
PIL/NumPy tương đương eval transform; web không cần load PyTorch model.

Sau mỗi batch, web sinh hai report trong `reports/web_evaluations/`:

- CSV: một dòng cho mỗi cặp ảnh–model, gồm định danh model, truth, dự đoán,
  confidence, đúng/sai, điểm và xác suất từng lớp.
- JSON: bộ lọc, số ảnh, tổng số kết quả và metric riêng của từng model.

Hai nút tải xuất hiện ngay dưới kết quả batch.

Bộ lọc `Loại ảnh` cho phép chấm riêng `SUPPORTED`, `UNSUPPORTED` hoặc
`NON_FOOD`. Unified và Legacy router 7 lớp được chấm food accuracy/recall trên
cả ba nhãn; category/freshness chỉ được chấm trên ảnh `SUPPORTED`. CSV có cả
`truth_food_status`, dự đoán, confidence, xác suất và cờ đúng/sai.

Mục `Chế độ đánh giá` cho phép chạy `Chỉ Unified`, `Chỉ Legacy` hoặc cả hai.
Hai danh sách checkbox cho phép chọn một hoặc nhiều Unified artifact và một
hoặc nhiều Legacy bundle hoàn chỉnh gồm router và năm expert. Tất cả hiển thị
tên cùng kiến trúc đã train và chạy trên đúng cùng danh sách ảnh để so sánh.
Model không được tích sẽ không chạy inference. Khi lọc negative, bundle Legacy
router 7 lớp vẫn chọn được; bundle router 5 lớp cũ tự bị vô hiệu hóa.

Giao diện desktop chia hai cột: bộ lọc và điều khiển ở bên trái; metric cùng
kết quả từng ảnh ở bên phải. Mỗi model tạo một thẻ kết quả riêng cho cùng ảnh.
Kết quả batch đọc CSV theo từng trang 20 cặp ảnh–model, nên có thể duyệt toàn
bộ supported và negative test mà không nạp mọi dòng vào trình duyệt cùng lúc.

Có thể đổi cổng hoặc đường dẫn artifact:

```powershell
python -m local_web.server --port 8080 --onnx models/unified/food_multitask.onnx
```

Web mặc định quét mọi `*.onnx` trong `models/unified`. Có thể đổi registry root:

```powershell
python -m local_web.server --onnx-root path/to/unified-models
```

Có thể đổi thư mục negative test:

```powershell
python -m local_web.server --negative-test-root data_negative/test
```

Web mặc định quét các bundle trong `models/legacy`. Có thể đổi registry và
bundle mặc định nếu cần:

```powershell
python -m local_web.server `
  --legacy-registry-root path/to/legacy-bundles `
  --legacy-root path/to/legacy-bundles/clean_split
```
