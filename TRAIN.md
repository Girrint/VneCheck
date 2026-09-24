# Hướng dẫn huấn luyện

## Menu tương tác

Khởi động menu chính:

```powershell
python main.py
```

Menu hỗ trợ train router, train từng freshness expert, dự đoán ảnh, kiểm tra
dataset và có trang trợ giúp bằng tiếng Anh.

## 1. Kích hoạt môi trường

Chạy trong PowerShell tại thư mục dự án:

```powershell
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu `py` không tồn tại, dùng đường dẫn Python trực tiếp như lệnh tạo venv ở trên.

## 2. Trước khi train

Chạy kiểm tra duplicate:

```powershell
python scripts\check_duplicates.py
```

Nếu công cụ trả exit code `1`, phải xử lý leakage hoặc ghi nhận rõ đây chỉ là train thử nghiệm. Dataset hiện tại đã có cross-split duplicate.

Chốt contract freshness trước khi train lại. Hiện `gia_cam` và `gia_suc` có 3 lớp; `cu`, `qua`, `rau` có 2 lớp.

## 3. Huấn luyện

Router:

```powershell
python train.py --epochs 15 --batch-size 16 --workers 0 router
```

Expert:

```powershell
python train.py --epochs 15 --batch-size 16 --workers 0 freshness --category gia_suc
python train.py --epochs 15 --batch-size 16 --workers 0 freshness --category gia_cam
python train.py --epochs 15 --batch-size 16 --workers 0 freshness --category rau
python train.py --epochs 15 --batch-size 16 --workers 0 freshness --category cu
python train.py --epochs 15 --batch-size 16 --workers 0 freshness --category qua
```

Default thực tế trong `train.py`: epochs `15`, batch size `16`, workers là `min(4, os.cpu_count())`, model `mobilenet_v3_large`, method `fine_tune`, optimizer `adamw`, scheduler `plateau`, learning rate `5e-4`, weight decay `1e-4`, label smoothing `0.05`, seed `42`.

Các lựa chọn model/method:

```powershell
python train.py --model mobilenet_v3_small --method head_only router
python train.py --model mobilenet_v3_large --method fine_tune router
```

## 4. Split và metric

`food_ai.data.create_loaders()` hiện tách ngẫu nhiên 20% từ `data/train` thành test nội bộ bằng seed `42`; `data/valid` được dùng để chọn checkpoint và early stopping. Đây là split theo file, không phải group split, nên không đủ chống leakage.

Train dùng augmentation:

```text
RandomResizedCrop(224, scale=0.8..1.0)
RandomHorizontalFlip
RandomRotation(12)
ToTensor + ImageNet normalization
```

Validation/test dùng `Resize(232) + CenterCrop(224) + ToTensor + ImageNet normalization`.

Checkpoint tốt nhất chọn theo validation loss. Báo cáo gồm accuracy, macro-F1, precision/recall/F1 từng lớp và confusion matrix.

## 5. Resume và output

Mỗi run nằm trong `runs/<run>/`:

```text
config.json
metrics.csv
best_model.pth
last_checkpoint.pth
summary.json
```

Resume:

```powershell
python train.py --epochs 15 --resume runs\<run>\last_checkpoint.pth router
```

`--epochs` là tổng số epoch mục tiêu. Chỉ resume checkpoint cùng task, category, model, class order và training method; code hiện tại chưa tự kiểm tra đầy đủ các metadata này.

## 6. Đánh giá

Router uncertainty:

```powershell
python evaluate_router_uncertainty.py --root data\valid --threshold 0.8
```

End-to-end:

```powershell
python evaluate_end_to_end.py --root data\valid
```

Evaluator end-to-end hiện chưa hỗ trợ dataset có expert 2 lớp và 3 lớp đồng thời; cần sửa contract/evaluator trước khi dùng.

Benchmark MobileNetV3:

```powershell
python benchmark_mobilenet.py router --epochs 15 --batch-size 16 --workers 0
python benchmark_mobilenet.py freshness --category cu --epochs 15 --batch-size 16 --workers 0
```

Không chọn model dựa trên accuracy của split đang leakage.

## 7. Export và kiểm thử

```powershell
python export_onnx.py
python verify_onnx.py --image path\to\image.jpg
python package_models.py --require-onnx
python -m unittest -v test_regression.py
```

`package_models.py` tạo manifest checksum và copy cả `.onnx.data` khi cần.
