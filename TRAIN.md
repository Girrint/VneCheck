# Huấn luyện và đánh giá

## 1. Kiểm tra môi trường và CUDA

Kích hoạt đúng virtual environment trước mọi lệnh:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra driver/GPU:

```powershell
nvidia-smi
```

Kiểm tra bản PyTorch trong chính `.venv` có nhìn thấy CUDA hay không:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Điều kiện để train bằng GPU là `CUDA available: True`. `train.py` và
`train_unified.py` tự chọn `cuda` khi khả dụng, tự bật mixed precision và in
`Thiết bị: <tên GPU>` lúc bắt đầu; dự án không có tham số `--device`. Nếu kết
quả là `False`, kiểm tra đã kích hoạt đúng `.venv` và đã cài bản PyTorch có
CUDA. Không dựa riêng vào `nvidia-smi`, vì driver nhận GPU chưa đồng nghĩa bản
PyTorch đang dùng có CUDA.

Với GPU 4 GB, bắt đầu bằng `--batch-size 8`. Nếu báo CUDA out of memory, giảm
về 4 hoặc 2; nếu còn nhiều VRAM có thể tăng lên 16. Export và verify ONNX chạy
trên CPU, nên CUDA chỉ bắt buộc nếu muốn tăng tốc train.

## 2. Contract dữ liệu

`train.py` đọc ba thư mục độc lập:

- `data_split/train`: cập nhật trọng số.
- `data_split/valid`: scheduler, chọn checkpoint và early stopping.
- `data_split/test`: chấm đúng một lần sau khi nạp checkpoint tốt nhất.

Class list của ba split phải giống nhau. Loader sẽ dừng nếu contract này sai.

Tiền xử lý được thực hiện online khi DataLoader đọc batch; script không tạo bản
resize và không thay đổi ảnh gốc. Train dùng crop ngẫu nhiên nhẹ, lật ngang và
jitter ánh sáng nhỏ. Validation, test, `predict.py` và ONNX verification dùng
chung pipeline EXIF → RGB → Resize(232) → CenterCrop(224) → ImageNet normalize.

## 3. Chuẩn bị dữ liệu

Kiểm tra supported split và tạo lại unified manifest trước khi train:

```powershell
python scripts\dataset_stats.py
python scripts\build_unified_manifest.py --negative-root data_negative
```

Manifest Unified phải có đủ `SUPPORTED`, `UNSUPPORTED`, `NON_FOOD` trong
train/valid/test. Không thay đổi `data_split`, `data_negative` hoặc manifest
giữa các model cần so sánh.

## 4. Train từng Legacy model và export bundle

Một Legacy model hoàn chỉnh không phải một checkpoint: nó gồm **1 router + 5
freshness expert** cùng một backbone. `train.py` copy best checkpoint hiện hành
vào `models/router.pth` và `models/freshness/*.pth`; vì vậy phải train đủ sáu
task rồi export bundle ngay, trước khi bắt đầu backbone kế tiếp.

Dùng helper PowerShell sau để tránh bỏ sót expert. Mỗi task vẫn có run riêng
trong `runs/`; helper dừng ngay nếu một lệnh lỗi:

```powershell
function Train-LegacyBundle {
  param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$BundleName,
    [Parameter(Mandatory=$true)][string]$VerifyImage,
    [int]$BatchSize = 8
  )

  python train.py --model $Model --method fine_tune --epochs 15 `
    --batch-size $BatchSize --workers 4 --seed 42 `
    --run-name "${BundleName}_router" router
  if ($LASTEXITCODE -ne 0) { throw "Train router thất bại: $Model" }

  foreach ($Category in @("cu", "gia_cam", "gia_suc", "qua", "rau")) {
    python train.py --model $Model --method fine_tune --epochs 15 `
      --batch-size $BatchSize --workers 4 --seed 42 `
      --run-name "${BundleName}_${Category}" `
      freshness --category $Category
    if ($LASTEXITCODE -ne 0) {
      throw "Train expert $Category thất bại: $Model"
    }
  }

  python export_onnx.py `
    --checkpoint-root models `
    --bundle-name $BundleName `
    --display-name "$Model Legacy"
  if ($LASTEXITCODE -ne 0) { throw "Export Legacy thất bại: $Model" }

  python verify_onnx.py `
    --image $VerifyImage `
    --checkpoint-root models `
    --onnx-root "models\legacy\$BundleName"
  if ($LASTEXITCODE -ne 0) { throw "Verify Legacy thất bại: $Model" }
}
```

Chạy lần lượt cho từng backbone. Thay `$VerifyImage` bằng một ảnh thật trong
`data_split/test`:

```powershell
$VerifyImage = "data_split\test\cu\FRESH\carrot\<ten-anh>.jpg"

Train-LegacyBundle mobilenet_v3_small mobilenet_v3_small_v1 $VerifyImage 8
Train-LegacyBundle mobilenet_v3_large mobilenet_v3_large_v1 $VerifyImage 8
Train-LegacyBundle efficientnet_b0 efficientnet_b0_v1 $VerifyImage 8
Train-LegacyBundle resnet18 resnet18_v1 $VerifyImage 8
```

Mỗi lời gọi tạo một bundle độc lập:

```text
models/legacy/<bundle-name>/
  router.onnx
  freshness/{cu,gia_cam,gia_suc,qua,rau}.onnx
  manifest.json
```

Không chạy backbone tiếp theo nếu bundle hiện tại chưa export và verify thành
công. Không trộn router của backbone này với expert của backbone khác.

## 5. Train từng Unified model, export và verify ONNX

Mỗi Unified run chỉ train một checkpoint ba head. Helper dưới đây tìm đúng run
vừa tạo theo `--run-name`, export bằng tên riêng và verify chính cặp
checkpoint/ONNX đó:

```powershell
function Train-UnifiedArtifact {
  param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ArtifactName,
    [Parameter(Mandatory=$true)][string]$VerifyImage,
    [int]$BatchSize = 8
  )

  python train_unified.py `
    --manifest reports\unified_manifest.csv `
    --model $Model --method fine_tune --epochs 15 `
    --batch-size $BatchSize --workers 4 --seed 42 `
    --run-name $ArtifactName
  if ($LASTEXITCODE -ne 0) { throw "Train Unified thất bại: $Model" }

  $Run = Get-ChildItem runs -Directory -Filter "*_$ArtifactName" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $Run) { throw "Không tìm thấy run: $ArtifactName" }
  $Checkpoint = Join-Path $Run.FullName "best_model.pth"
  $Onnx = "models\unified\exports\$ArtifactName.onnx"

  python export_unified_onnx.py `
    --checkpoint $Checkpoint `
    --name $ArtifactName `
    --display-name "$Model Unified"
  if ($LASTEXITCODE -ne 0) { throw "Export Unified thất bại: $Model" }

  python verify_unified_onnx.py `
    --image $VerifyImage `
    --checkpoint $Checkpoint `
    --onnx $Onnx
  if ($LASTEXITCODE -ne 0) { throw "Verify Unified thất bại: $Model" }
}
```

Chạy lần lượt bốn backbone:

```powershell
$VerifyImage = "data_split\test\cu\FRESH\carrot\<ten-anh>.jpg"

Train-UnifiedArtifact mobilenet_v3_small mobilenet_v3_small_v1 $VerifyImage 8
Train-UnifiedArtifact mobilenet_v3_large mobilenet_v3_large_v1 $VerifyImage 8
Train-UnifiedArtifact efficientnet_b0 efficientnet_b0_v1 $VerifyImage 8
Train-UnifiedArtifact resnet18 resnet18_v1 $VerifyImage 8
```

Mỗi lời gọi tạo:

```text
runs/<timestamp>_<artifact-name>/best_model.pth
models/unified/exports/<artifact-name>.onnx
models/unified/exports/<artifact-name>.manifest.json
```

Verify hợp lệ khi cả ba output `food`, `category`, `freshness` đều `PASS`.

## 6. Tham số train mặc định

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

Legacy và Unified cùng hỗ trợ `mobilenet_v3_small`, `mobilenet_v3_large`,
`efficientnet_b0`, `resnet18` và hai method `fine_tune`, `head_only`.

## 7. Output

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

## 8. Resume Legacy

```powershell
python train.py --epochs 30 `
  --resume runs\<run>\last_checkpoint.pth router
```

`--epochs` là tổng epoch mục tiêu, không phải số epoch chạy thêm. Chỉ resume
checkpoint cùng task, category, kiến trúc, method và class order.

## 9. Đánh giá và so sánh trên web

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

Sau khi export, web tự quét tất cả artifact/bundle:

```powershell
python -m local_web.server
```

Mở `http://127.0.0.1:8000`, chọn `Legacy`, `Unified` hoặc `Cả hai`, sau đó tích
một hay nhiều bundle/artifact cần test. Mọi model được chọn chạy trên cùng nhóm
ảnh; batch lấy seed mới mỗi lượt, ghi seed vào JSON report, xuất metric riêng
từng model và một dòng CSV cho mỗi cặp ảnh–model.

## 10. Tuning

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
