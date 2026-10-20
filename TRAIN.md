# Hướng dẫn huấn luyện model

Tài liệu này mô tả quy trình train hiện tại của hai pipeline:

- **Legacy:** một router 7 lớp và năm freshness expert.
- **Unified:** một backbone dùng chung với ba head food/category/freshness.

Hai pipeline dùng cùng supported dataset và negative dataset. Không thay đổi
dataset, seed hoặc preprocessing giữa các backbone cần so sánh.

Nếu chỉ cần lệnh để chạy, đọc mục 2–3 để kiểm tra môi trường và dữ liệu, khai
báo hai helper ở mục 6.5 và 7.3, sau đó dùng toàn bộ lời gọi tại mục 8.

## 1. Nguyên tắc đánh giá

- `train` dùng để cập nhật trọng số.
- `valid` dùng để chọn checkpoint, early stopping và lựa chọn cấu hình.
- `test` chỉ dùng để báo cáo kết quả cuối sau khi đã chốt cấu hình.
- Không chọn backbone, learning rate, threshold hoặc epoch dựa trên test.
- Mỗi artifact phải giữ được checkpoint, class order, kiến trúc và seed đã dùng.

Các script hiện vẫn chấm test sau mỗi run. Khi thử nhiều cấu hình, chỉ dùng
metric validation trong `runs/<run>/` để ra quyết định; không dùng test metric
của từng trial để tuning.

## 2. Kích hoạt môi trường và kiểm tra CUDA

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra driver:

```powershell
nvidia-smi
```

Kiểm tra PyTorch trong đúng virtual environment:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Train bằng GPU khi kết quả có:

```text
CUDA available: True
```

`train.py` và `train_unified.py` tự chọn CUDA, mixed precision và
channels-last khi khả dụng. Dự án không có tham số `--device`.

Với GPU 4 GB:

- Bắt đầu bằng `--batch-size 8`.
- Nếu CUDA out of memory, giảm xuống 4 hoặc 2.
- Chỉ tăng lên 16 sau khi kiểm tra VRAM thực tế.
- Export và verify ONNX chạy được trên CPU.

## 3. Contract dữ liệu

### Supported

```text
data_split/
  train/<category>/<freshness>/<product>/*
  valid/<category>/<freshness>/<product>/*
  test/<category>/<freshness>/<product>/*
```

Các category:

```text
cu, gia_cam, gia_suc, qua, rau
```

### Negative

```text
data_negative/
  train/{UNSUPPORTED,NON_FOOD}/*
  valid/{UNSUPPORTED,NON_FOOD}/*
  test/{UNSUPPORTED,NON_FOOD}/*
```

### Số lượng hiện tại

| Dataset | Train | Valid | Test | Tổng |
|---|---:|---:|---:|---:|
| Supported | 66.115 | 14.172 | 14.162 | 94.449 |
| Negative | 7.000 | 1.500 | 1.500 | 10.000 |
| Router Legacy sau khi ghép | 73.115 | 15.672 | 15.662 | 104.449 |

Kiểm tra cấu trúc và số lượng mà không mở nội dung ảnh:

```powershell
python scripts\dataset_stats.py
```

Tạo lại manifest cho Unified:

```powershell
python scripts\build_unified_manifest.py `
  --negative-root data_negative `
  --output reports\unified_manifest.csv
```

Không thay đổi `data_split`, `data_negative` hoặc manifest giữa các model cần
so sánh.

## 4. Preprocessing

Preprocessing chạy online trong DataLoader; không tạo ảnh resize mới và không
thay đổi file gốc.

Train:

```text
EXIF orientation → RGB
RandomResizedCrop(224, scale=0.85..1.0, ratio=0.85..1.15)
RandomHorizontalFlip
ColorJitter nhẹ
ToTensor
ImageNet normalization
```

Validation, test, CLI prediction và ONNX web:

```text
EXIF orientation → RGB
Resize(232)
CenterCrop(224)
ToTensor
ImageNet normalization
```

## 5. Backbone và tham số chung

Legacy và Unified cùng hỗ trợ:

```text
mobilenet_v3_small
mobilenet_v3_large
efficientnet_b0
resnet18
```

Hai phương pháp transfer learning:

- `fine_tune`: cập nhật toàn bộ backbone và classifier.
- `head_only`: đóng băng backbone, chỉ cập nhật classifier/head.

Thiết lập khuyến nghị ban đầu:

```text
epochs          = 15
batch size      = 8 với GPU 4 GB
workers         = 4
learning rate   = 5e-4
weight decay    = 1e-4
label smoothing = 0.05
seed            = 42
method          = fine_tune
```

## 6. Train Legacy

### 6.1 Kiến trúc hiện tại

Legacy gồm sáu model:

```text
router 7 lớp
  ├─ cu          → expert cu
  ├─ gia_cam     → expert gia_cam
  ├─ gia_suc     → expert gia_suc
  ├─ qua         → expert qua
  ├─ rau         → expert rau
  ├─ UNSUPPORTED → dừng
  └─ NON_FOOD    → dừng
```

Router tự ghép:

```text
data_split/<split> + data_negative/<split>
```

Class order cố định:

```text
cu, gia_cam, gia_suc, qua, rau, UNSUPPORTED, NON_FOOD
```

Router mặc định dùng `--class-balance sqrt_inverse`. Có thể chọn:

- `none`: không cân bằng.
- `sqrt_inverse`: mặc định, ít cực đoan hơn inverse.
- `inverse`: cân bằng mạnh hơn.

Freshness expert chỉ đọc `data_split`; negative không được đưa vào expert.

### 6.2 Train router 7 lớp

```powershell
python train.py `
  --model mobilenet_v3_small `
  --method fine_tune `
  --epochs 15 `
  --batch-size 8 `
  --workers 4 `
  --learning-rate 5e-4 `
  --weight-decay 1e-4 `
  --seed 42 `
  --negative-root data_negative `
  --class-balance sqrt_inverse `
  --run-name mobilenet_v3_small_router `
  router
```

Checkpoint router tốt nhất được copy vào:

```text
models/router.pth
```

Router 5 lớp cũ không thể dùng với `--resume` vì classifier shape và class
contract đã thay đổi.

### 6.3 Train năm freshness expert

```powershell
python train.py --model mobilenet_v3_small --method fine_tune `
  --epochs 15 --batch-size 8 --workers 4 --seed 42 `
  --run-name mobilenet_v3_small_cu freshness --category cu

python train.py --model mobilenet_v3_small --method fine_tune `
  --epochs 15 --batch-size 8 --workers 4 --seed 42 `
  --run-name mobilenet_v3_small_gia_cam freshness --category gia_cam

python train.py --model mobilenet_v3_small --method fine_tune `
  --epochs 15 --batch-size 8 --workers 4 --seed 42 `
  --run-name mobilenet_v3_small_gia_suc freshness --category gia_suc

python train.py --model mobilenet_v3_small --method fine_tune `
  --epochs 15 --batch-size 8 --workers 4 --seed 42 `
  --run-name mobilenet_v3_small_qua freshness --category qua

python train.py --model mobilenet_v3_small --method fine_tune `
  --epochs 15 --batch-size 8 --workers 4 --seed 42 `
  --run-name mobilenet_v3_small_rau freshness --category rau
```

Output hiện hành:

```text
models/freshness/cu.pth
models/freshness/gia_cam.pth
models/freshness/gia_suc.pth
models/freshness/qua.pth
models/freshness/rau.pth
```

`cu`, `qua`, `rau` có hai lớp `FRESH/SPOILED`. `gia_cam`, `gia_suc` có ba lớp
`FRESH/HALF-FRESH/SPOILED`.

### 6.4 Export và verify Legacy

Sau khi train router và đủ năm expert của cùng một backbone, export ngay trước
khi train backbone tiếp theo. Các file trong `models/` là checkpoint hiện hành
và sẽ bị lần train sau ghi đè.

```powershell
python export_onnx.py `
  --checkpoint-root models `
  --bundle-name mobilenet_v3_small_v1 `
  --display-name "MobileNetV3 Small Legacy v1"
```

Kết quả:

```text
models/legacy/mobilenet_v3_small_v1/
  router.onnx
  freshness/
    cu.onnx
    gia_cam.onnx
    gia_suc.onnx
    qua.onnx
    rau.onnx
  manifest.json
```

Manifest ghi class order của router, trạng thái negative gate, backbone,
training method và checksum artifact.

Verify PyTorch và ONNX bằng một ảnh có thật:

```powershell
python verify_onnx.py `
  --image data_split\test\cu\FRESH\carrot\<ten-anh>.jpg `
  --checkpoint-root models `
  --onnx-root models\legacy\mobilenet_v3_small_v1
```

Lệnh phải kết thúc với:

```text
ALL MODELS VERIFIED SUCCESSFULLY
```

### 6.5 Helper train và export trọn một Legacy backbone

Helper này train router, năm expert rồi export bundle ONNX. Nó ghi đè checkpoint
hiện hành trong `models/`, nhưng giữ mỗi run trong `runs/` và mỗi ONNX bundle
trong `models/legacy/<bundle-name>`.

```powershell
function Train-LegacyBundle {
  param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$BundleName,
    [int]$BatchSize = 8
  )

  python train.py --model $Model --method fine_tune --epochs 15 `
    --batch-size $BatchSize --workers 4 --seed 42 `
    --negative-root data_negative --class-balance sqrt_inverse `
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

  python export_onnx.py --checkpoint-root models `
    --bundle-name $BundleName --display-name "$Model Legacy"
  if ($LASTEXITCODE -ne 0) { throw "Export Legacy thất bại: $Model" }
}
```

Ví dụ:

```powershell
Train-LegacyBundle `
  -Model mobilenet_v3_small `
  -BundleName mobilenet_v3_small_v1 `
  -BatchSize 8
```

Không trộn router của backbone này với expert của backbone khác khi báo cáo
kết quả chính thức.

## 7. Train Unified

Unified tạo một model với ba output:

```text
food_logits       → SUPPORTED / UNSUPPORTED / NON_FOOD
category_logits   → cu / gia_cam / gia_suc / qua / rau
freshness_logits  → FRESH / HALF-FRESH / SPOILED
```

Trên ảnh negative, category và freshness target được mask. Mặc định food loss
dùng `sqrt_inverse` class weight và checkpoint tốt nhất được chọn bằng
validation food macro-F1.

### 7.1 Train một Unified model

```powershell
python train_unified.py `
  --manifest reports\unified_manifest.csv `
  --model mobilenet_v3_small `
  --method fine_tune `
  --epochs 15 `
  --batch-size 8 `
  --workers 4 `
  --learning-rate 5e-4 `
  --weight-decay 1e-4 `
  --food-class-balance sqrt_inverse `
  --selection-metric food_macro_f1 `
  --seed 42 `
  --run-name mobilenet_v3_small_v1
```

Unified không có tham số `--resume` ở trạng thái hiện tại.

Checkpoint mặc định được copy vào:

```text
models/unified/food_multitask.pth
```

Luôn dùng checkpoint riêng trong run khi export nhiều model, vì file mặc định
trên sẽ bị lần train Unified tiếp theo ghi đè.

### 7.2 Export và verify Unified

Tìm run vừa tạo:

```powershell
$Run = Get-ChildItem runs -Directory -Filter "*_mobilenet_v3_small_v1" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
$Checkpoint = Join-Path $Run.FullName "best_model.pth"
```

Export bằng artifact ID riêng:

```powershell
python export_unified_onnx.py `
  --checkpoint $Checkpoint `
  --name mobilenet_v3_small_v1 `
  --display-name "MobileNetV3 Small Unified v1"
```

Kết quả:

```text
models/unified/exports/mobilenet_v3_small_v1.onnx
models/unified/exports/mobilenet_v3_small_v1.manifest.json
```

Verify:

```powershell
python verify_unified_onnx.py `
  --image data_split\test\cu\FRESH\carrot\<ten-anh>.jpg `
  --checkpoint $Checkpoint `
  --onnx models\unified\exports\mobilenet_v3_small_v1.onnx
```

Kết quả hợp lệ khi `food`, `category` và `freshness` đều báo `PASS`.

### 7.3 Helper train và export Unified

```powershell
function Train-UnifiedArtifact {
  param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$ArtifactName,
    [int]$BatchSize = 8
  )

  python train_unified.py --manifest reports\unified_manifest.csv `
    --model $Model --method fine_tune --epochs 15 `
    --batch-size $BatchSize --workers 4 --seed 42 `
    --food-class-balance sqrt_inverse `
    --selection-metric food_macro_f1 --run-name $ArtifactName
  if ($LASTEXITCODE -ne 0) { throw "Train Unified thất bại: $Model" }

  $Run = Get-ChildItem runs -Directory -Filter "*_$ArtifactName" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $Run) { throw "Không tìm thấy run: $ArtifactName" }

  $Checkpoint = Join-Path $Run.FullName "best_model.pth"

  python export_unified_onnx.py --checkpoint $Checkpoint `
    --name $ArtifactName --display-name "$Model Unified"
  if ($LASTEXITCODE -ne 0) { throw "Export Unified thất bại: $Model" }
}
```

Ví dụ:

```powershell
Train-UnifiedArtifact `
  -Model mobilenet_v3_small `
  -ArtifactName mobilenet_v3_small_v1 `
  -BatchSize 8
```

## 8. Lệnh train và export toàn bộ model

Dùng ID riêng cho từng artifact:

| Backbone | Legacy bundle | Unified artifact |
|---|---|---|
| MobileNetV3-Small | `mobilenet_v3_small_v1` | `mobilenet_v3_small_v1` |
| MobileNetV3-Large | `mobilenet_v3_large_v1` | `mobilenet_v3_large_v1` |
| EfficientNet-B0 | `efficientnet_b0_v1` | `efficientnet_b0_v1` |
| ResNet18 | `resnet18_v1` | `resnet18_v1` |

Hai registry khác nhau nên Legacy và Unified có thể dùng cùng ID. Không dùng
lại ID trong cùng registry nếu muốn giữ artifact cũ; tăng hậu tố `v2`, `v3`.

Trước khi chạy phần này, phải khai báo hai function `Train-LegacyBundle` và
`Train-UnifiedArtifact` ở mục 6.5 và 7.3 trong cùng cửa sổ PowerShell. Mỗi lời
gọi Legacy thực hiện đủ sáu lệnh train rồi export sáu ONNX. Mỗi lời gọi Unified
train một model, tìm đúng checkpoint trong run rồi export một ONNX ba output.

Hai helper không yêu cầu ảnh kiểm tra. Có thể chạy verify riêng sau khi train
nếu cần. Không chạy hai backbone Legacy song song vì chúng cùng ghi vào
`models/router.pth` và `models/freshness/*.pth`.

### 8.1 MobileNetV3-Small

```powershell
Train-LegacyBundle mobilenet_v3_small mobilenet_v3_small_v1 8
Train-UnifiedArtifact mobilenet_v3_small mobilenet_v3_small_v1 8
```

Output ONNX:

```text
models/legacy/mobilenet_v3_small_v1/router.onnx
models/legacy/mobilenet_v3_small_v1/freshness/*.onnx
models/unified/exports/mobilenet_v3_small_v1.onnx
```

### 8.2 MobileNetV3-Large

```powershell
Train-LegacyBundle mobilenet_v3_large mobilenet_v3_large_v1 8
Train-UnifiedArtifact mobilenet_v3_large mobilenet_v3_large_v1 8
```

Output ONNX:

```text
models/legacy/mobilenet_v3_large_v1/router.onnx
models/legacy/mobilenet_v3_large_v1/freshness/*.onnx
models/unified/exports/mobilenet_v3_large_v1.onnx
```

### 8.3 EfficientNet-B0

```powershell
Train-LegacyBundle efficientnet_b0 efficientnet_b0_v1 8
Train-UnifiedArtifact efficientnet_b0 efficientnet_b0_v1 8
```

Output ONNX:

```text
models/legacy/efficientnet_b0_v1/router.onnx
models/legacy/efficientnet_b0_v1/freshness/*.onnx
models/unified/exports/efficientnet_b0_v1.onnx
```

### 8.4 ResNet18

```powershell
Train-LegacyBundle resnet18 resnet18_v1 8
Train-UnifiedArtifact resnet18 resnet18_v1 8
```

Output ONNX:

```text
models/legacy/resnet18_v1/router.onnx
models/legacy/resnet18_v1/freshness/*.onnx
models/unified/exports/resnet18_v1.onnx
```

### 8.5 Chạy tất cả bằng một block

Block dưới đây chạy đủ bốn Legacy bundle và bốn Unified artifact. Tổng cộng
train 28 checkpoint và export 28 file ONNX. Thời gian có thể rất dài; nên chạy
từng cặp backbone ở các mục trên nếu cần kiểm tra kết quả giữa chừng.

```powershell
Train-LegacyBundle mobilenet_v3_small mobilenet_v3_small_v1 8
Train-UnifiedArtifact mobilenet_v3_small mobilenet_v3_small_v1 8

Train-LegacyBundle mobilenet_v3_large mobilenet_v3_large_v1 8
Train-UnifiedArtifact mobilenet_v3_large mobilenet_v3_large_v1 8

Train-LegacyBundle efficientnet_b0 efficientnet_b0_v1 8
Train-UnifiedArtifact efficientnet_b0 efficientnet_b0_v1 8

Train-LegacyBundle resnet18 resnet18_v1 8
Train-UnifiedArtifact resnet18 resnet18_v1 8
```

Sau khi hoàn tất, web sẽ quét được tám cấu hình hoàn chỉnh:

- 4 Legacy bundle, mỗi bundle có 1 router và 5 expert ONNX.
- 4 Unified artifact, mỗi artifact có 1 ONNX ba output.

## 9. Run directory và checkpoint

Mỗi task tạo:

```text
runs/<timestamp>_<run-name>/
  config.json
  metrics.csv
  best_model.pth
  last_checkpoint.pth
  summary.json
```

`config.json` lưu dataset, class order, preprocessing, kiến trúc, method, seed,
thiết bị và class weights. `summary.json` lưu validation tốt nhất, test metric,
macro-F1, metric từng lớp và confusion matrix.

### Resume Legacy

```powershell
python train.py --epochs 30 `
  --resume runs\<run>\last_checkpoint.pth `
  router
```

`--epochs 30` là tổng epoch mục tiêu, không phải chạy thêm 30 epoch. Chỉ resume
checkpoint có cùng task, category, backbone, method và class order. Router 5
lớp cũ sẽ bị từ chối khi resume vào router 7 lớp.

Freshness example:

```powershell
python train.py --epochs 30 `
  --resume runs\<run>\last_checkpoint.pth `
  freshness --category gia_suc
```

## 10. Đánh giá Legacy

Đánh giá confidence và entropy của router 7 lớp trên cả supported và negative
test:

```powershell
python evaluate_router_uncertainty.py `
  --root data_split\test `
  --negative-root data_negative\test `
  --threshold 0.8
```

Đánh giá router, oracle expert và pipeline end-to-end:

```powershell
python evaluate_end_to_end.py `
  --root data_split\test `
  --negative-root data_negative\test
```

Nếu router từ chối nhầm ảnh `SUPPORTED`, end-to-end ghi `REJECTED` và tính là
sai. Expert 2 lớp và 3 lớp được quy về contract freshness chung khi tổng hợp.

## 11. So sánh ONNX trên web

Khởi động:

```powershell
python -m local_web.server
```

Mở:

```text
http://127.0.0.1:8000
```

Web cho phép:

- Chọn `Legacy`, `Unified` hoặc `Cả hai`.
- Tích đồng thời nhiều Unified artifact và nhiều Legacy bundle.
- Chạy một ảnh hoặc batch trên cùng danh sách ảnh và cùng seed.
- Xuất một dòng CSV cho mỗi cặp ảnh–model.
- Xuất JSON metric riêng cho từng model.
- Chấm food accuracy/recall trên `SUPPORTED`, `UNSUPPORTED`, `NON_FOOD`.
- Chấm category/freshness trên ảnh `SUPPORTED`.

Bundle router Legacy 7 lớp chấm được negative. Snapshot
`models/legacy/original` dùng router 5 lớp cũ nên chỉ chấm được `SUPPORTED`.

Web mặc định dùng test set. Không dùng việc chạy test lặp lại để chọn model.
Chọn backbone bằng validation metric trong run; chỉ dùng web test cho báo cáo
cuối hoặc kiểm tra lỗi định tính.

## 12. Quy trình khuyến nghị

1. Kiểm tra CUDA và thống kê dataset.
2. Tạo và cố định Unified manifest.
3. Chọn một backbone và một run/artifact ID.
4. Train Legacy router 7 lớp và đủ năm expert.
5. Export, verify Legacy ngay trước khi train backbone khác.
6. Train Unified bằng cùng backbone, seed và epoch budget.
7. Export checkpoint riêng trong run và verify Unified ONNX.
8. Lặp lại với backbone khác.
9. Chọn model bằng validation metric, kích thước và latency.
10. Chạy test cuối một lần, lưu CSV/JSON và benchmark trên điện thoại thật.

Snapshot `models/legacy/original` được train trước split sạch và dùng router 5
lớp. Không dùng snapshot này làm kết quả Legacy chính thức.
