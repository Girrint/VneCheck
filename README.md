# VnGrocery AI

Pipeline AI phân loại nhóm thực phẩm và đánh giá độ tươi.

## Luồng

```text
ảnh → router → nhóm thực phẩm → freshness expert → FRESH / HALF-FRESH / SPOILED
```

Router hiện có các nhóm:

```text
cu, gia_suc, qua, rau
```

`gia_cam` chỉ hoạt động sau khi có `data/train/gia_cam`, `data/valid/gia_cam` và checkpoint expert tương ứng.

## Cài đặt

PowerShell tại thư mục `vnecheck`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset

```text
data/
  train/
    cu/
      FRESH/
      HALF-FRESH/
      SPOILED/
    gia_suc/
    gia_cam/
    qua/
    rau/
  valid/
    ... cùng cấu trúc với train ...
```

Mỗi ảnh chỉ xuất hiện trong một split. Không để ảnh gốc hoặc augmentation của cùng ảnh trong cả `train` và `valid`.

## Train

Train router:

```powershell
python train.py --epochs 15 --batch-size 8 --workers 0 router
```

Train freshness expert:

```powershell
python train.py --epochs 15 --batch-size 8 --workers 0 freshness --category cu
python train.py --epochs 15 --batch-size 8 --workers 0 freshness --category gia_suc
python train.py --epochs 15 --batch-size 8 --workers 0 freshness --category qua
python train.py --epochs 15 --batch-size 8 --workers 0 freshness --category rau
```

`--epochs` là tổng số epoch mục tiêu. `--workers 0` ổn định hơn trên Windows; tăng lên khi data loading không còn là nút thắt.

## Checkpoint và resume

Mỗi run nằm trong `runs/<run>/`:

```text
last_checkpoint.pth  # trạng thái mới nhất
best_model.pth       # model có validation loss tốt nhất
metrics.csv          # metrics từng epoch
summary.json         # kết quả test cuối
config.json          # cấu hình run
```

`last_checkpoint.pth` được tạo trước epoch đầu và cập nhật sau mỗi epoch. Nó lưu:

- Model weights
- Optimizer
- Scheduler
- AMP scaler
- Epoch hiện tại
- Best metrics và early-stopping state
- Random state CPU/GPU

Resume từ trạng thái mới nhất:

```powershell
python train.py --epochs 15 --resume runs\20260722_071335_router_resume_epoch1_6\last_checkpoint.pth router
```

Resume từ `best_model.pth` cũ vẫn chạy được, nhưng checkpoint cũ chỉ có weights nên không khôi phục được optimizer/scheduler chính xác. Checkpoint mới thì khôi phục đầy đủ.

Nếu process bị kill giữa epoch, mất tối đa epoch đang chạy; checkpoint trước đó không bị ghi hỏng.

## Dự đoán

```powershell
python predict.py path\to\image.jpg
```

Pipeline đầy đủ cần router và freshness expert tương ứng trong `models/`.

## Export ONNX cho Flutter

Export router và các expert hiện có:

```powershell
python export_onnx.py
```

Sinh ra:

```text
models/router.onnx
models/freshness/cu.onnx
models/freshness/gia_suc.onnx
models/freshness/qua.onnx
models/freshness/rau.onnx
```

Các file ONNX được copy vào:

```text
../VnGrocery/assets/models/
```

Flutter dùng preprocessing giống Python: RGB, resize cạnh ngắn `232`, center crop `224x224`, rồi normalize ImageNet.

## Kết quả hiện tại

Router tốt nhất đã đạt khoảng `99.70%` test accuracy. Kết quả không thay thế kiểm nghiệm an toàn thực phẩm.
