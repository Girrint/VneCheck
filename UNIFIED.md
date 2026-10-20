# Unified model

Unified pipeline tồn tại song song với pipeline legacy. Không file legacy nào
bị đổi đường dẫn hoặc bị ghi đè.

## Kiến trúc

```text
MobileNetV3-Small backbone
  ├─ food_logits       : SUPPORTED / UNSUPPORTED / NON_FOOD
  ├─ category_logits   : cu / gia_cam / gia_suc / qua / rau
  └─ freshness_logits  : FRESH / HALF-FRESH / SPOILED
```

Một lần forward tạo cả ba output. Khi train negative image, category và
freshness dùng `ignore_index`; masked loss chỉ cập nhật food head cho mẫu đó.

## Legacy snapshot

Baseline hiện tại đã được đóng băng tại:

```text
models/legacy/original/
```

Thư mục chứa 6 checkpoint PyTorch, 6 ONNX và `manifest.json` checksum. Snapshot
dùng bản copy độc lập để việc train hoặc ghi đè model legacy sau này không làm
thay đổi baseline.

Có thể tạo snapshot khác trước mỗi benchmark:

```powershell
python scripts\snapshot_legacy_models.py --name clean_split
```

## Manifest dữ liệu

Tạo manifest positive-only hiện tại:

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

Manifest hiện tại chỉ có 94.449 mẫu `SUPPORTED`. Vì vậy food loss tự tắt và
checkpoint được ghi `food_gate_ready=false`. Pipeline không được phép reject
non-food cho tới khi cả ba split có đủ hai nhóm negative.

## Train và đánh giá

```powershell
python train_unified.py
python evaluate_unified.py
```

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
models/unified/manifest.json
```

ONNX có một input `image` và ba output logits. Chỉ chuyển app sang unified sau
khi so sánh category accuracy, freshness macro-F1, recall `SPOILED`, dung lượng,
latency và peak memory trên cùng `data_split/test`.
