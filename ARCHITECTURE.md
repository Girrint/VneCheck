# Kiến trúc hệ thống

## Luồng suy luận

```text
Ảnh RGB
  │
  ├─ Resize(232) → CenterCrop(224) → ImageNet normalization
  │
  ▼
Router 5 lớp
  │
  ├─ cu ────────► expert cu: FRESH / SPOILED
  ├─ gia_cam ───► expert gia_cam: FRESH / HALF-FRESH / SPOILED
  ├─ gia_suc ───► expert gia_suc: FRESH / HALF-FRESH / SPOILED
  ├─ qua ───────► expert qua: FRESH / SPOILED
  └─ rau ───────► expert rau: FRESH / SPOILED
```

Router và expert là các model độc lập. Expert nhận lại tensor ảnh, không nhận
feature trung gian từ router. `predict.py` đọc class order từ từng checkpoint,
vì vậy expert 2 lớp và 3 lớp có thể cùng tồn tại an toàn.

## Artifact contract

```text
models/
  router.pth
  router.onnx
  freshness/
    cu.pth / cu.onnx
    gia_cam.pth / gia_cam.onnx
    gia_suc.pth / gia_suc.onnx
    qua.pth / qua.onnx
    rau.pth / rau.onnx
  manifest.json
```

Checkpoint PyTorch chứa ít nhất `state_dict`, `classes`, `model_name` và
`training_method`. Không hard-code class order khi đọc checkpoint.

Manifest do `package_models.py` tạo, gồm preprocessing, class list, kiến trúc
và SHA-256 của artifact PyTorch/ONNX.

## Các module chính

- `food_ai/config.py`: category, nhãn, score anchor và đường dẫn.
- `food_ai/data.py`: dataset, transform và ba DataLoader độc lập.
- `food_ai/model.py`: model factory, transfer learning và nạp checkpoint.
- `food_ai/experiment.py`: run directory, metric và báo cáo phân loại.
- `train.py`: vòng lặp train/valid/test.
- `predict.py`: inference PyTorch hai tầng.

## Điểm tươi

Sau softmax của expert, điểm được tính bằng kỳ vọng:

```text
FRESH      = 10.0
HALF-FRESH = 5.5
SPOILED    = 1.0
score = Σ probability(class) × anchor(class)
```

Với expert 2 lớp, tổng chỉ gồm `FRESH` và `SPOILED`. Điểm này là chỉ báo từ
model, không phải phép đo vật lý hoặc chứng nhận an toàn thực phẩm.

## Giới hạn hiện tại

- Router dùng top-1 cứng; chưa có unknown/reject/fallback.
- Softmax confidence chưa được calibration.
- Sai router kéo theo chọn sai expert.
- pHash giảm duplicate leakage nhưng không thay thế group metadata thật.
- Model hiện có cần được train lại trên `data_split`.
