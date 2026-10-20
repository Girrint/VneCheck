# Huấn luyện và đánh giá

## Contract dữ liệu

`train.py` đọc ba thư mục độc lập:

- `data_split/train`: cập nhật trọng số.
- `data_split/valid`: scheduler, chọn checkpoint và early stopping.
- `data_split/test`: chấm đúng một lần sau khi nạp checkpoint tốt nhất.

Class list của ba split phải giống nhau. Loader sẽ dừng nếu contract này sai.

Tiền xử lý được thực hiện online khi DataLoader đọc batch; script không tạo bản
resize và không thay đổi ảnh gốc. Train dùng crop ngẫu nhiên nhẹ, lật ngang và
jitter ánh sáng nhỏ. Validation, test, `predict.py` và ONNX verification dùng
chung pipeline EXIF → RGB → Resize(232) → CenterCrop(224) → ImageNet normalize.

## Lệnh train

Router:

```powershell
python train.py router
```

Freshness experts:

```powershell
python train.py freshness --category cu
python train.py freshness --category gia_cam
python train.py freshness --category gia_suc
python train.py freshness --category qua
python train.py freshness --category rau
```

Ví dụ cấu hình rõ ràng:

```powershell
python train.py --epochs 15 --batch-size 16 --workers 4 `
  --model mobilenet_v3_large --method fine_tune `
  --optimizer adamw --scheduler plateau `
  --learning-rate 5e-4 --weight-decay 1e-4 `
  router
```

Mặc định: 15 epoch, batch 16, MobileNetV3-Large, fine-tune, AdamW, Plateau,
learning rate `5e-4`, weight decay `1e-4`, label smoothing `0.05`, seed `42`.
Early stopping xảy ra sau 3 epoch không cải thiện validation loss.

Các backbone legacy hỗ trợ: `mobilenet_v3_small`, `mobilenet_v3_large`,
`efficientnet_b0`, `resnet18`. Unified hỗ trợ cùng bốn backbone, ví dụ:

```powershell
python train_unified.py --model efficientnet_b0 --run-name efficientnet_b0_v1
python train_unified.py --model resnet18 --run-name resnet18_v1
```

Mỗi lệnh vẫn đọc cùng manifest/split hiện tại. Dùng `--run-name` khác nhau và
export từng `runs/<run>/best_model.pth` bằng tên artifact khác để web so sánh.

## Output

Mỗi run tạo:

```text
runs/<timestamp>_<name>/
  config.json
  metrics.csv
  best_model.pth
  last_checkpoint.pth
  summary.json
```

Checkpoint tốt nhất đồng thời được copy vào:

```text
models/router.pth
models/freshness/<category>.pth
```

Với Unified, checkpoint tốt nhất được copy vào
`models/unified/food_multitask.pth`. Mỗi run vẫn giữ bản riêng tại
`runs/<run>/best_model.pth`; dùng bản trong run khi export nhiều artifact để
không nhầm checkpoint đã bị lần train sau ghi đè.

`summary.json` chứa validation tốt nhất, test loss, test accuracy, macro-F1,
metric từng lớp và confusion matrix.

## Resume

```powershell
python train.py --epochs 30 `
  --resume runs\<run>\last_checkpoint.pth router
```

`--epochs` là tổng epoch mục tiêu, không phải số epoch chạy thêm. Chỉ resume
checkpoint cùng task, category, kiến trúc, method và class order.

## Đánh giá

Router confidence và entropy trên `data_split/test`:

```powershell
python evaluate_router_uncertainty.py --threshold 0.8
```

Toàn pipeline router → expert:

```powershell
python evaluate_end_to_end.py
```

Evaluator end-to-end hỗ trợ expert 2 lớp và 3 lớp bằng cách quy đổi tên lớp về
contract chung `FRESH`, `HALF-FRESH`, `SPOILED`.

## Tuning

```powershell
python tune.py --task router --epochs 3
python tune.py --task freshness --category gia_suc --epochs 3
python benchmark_mobilenet.py router --epochs 15
```

Chọn cấu hình bằng validation loss/F1, không chọn bằng test. Các script tự động
vẫn ghi test của từng trial vì gọi `train.py`; không dùng các test score đó để
tuning hoặc liên tục điều chỉnh model.

Snapshot `models/legacy/original` được train trước khi tạo `data_split`; hãy
train lại đủ router và 5 expert trước khi công bố baseline Legacy trên test
sạch. Unified và các bundle mới phải ghi rõ run/checkpoint được dùng để export.
