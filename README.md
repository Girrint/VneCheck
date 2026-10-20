# VnGrocery AI

Hệ thống phân loại ảnh thực phẩm theo hai tầng:

1. Router nhận diện 5 nhóm: `cu`, `gia_cam`, `gia_suc`, `qua`, `rau`.
2. Expert của nhóm đã chọn dự đoán độ tươi.

Model mặc định là MobileNetV3-Large pretrained từ ImageNet. Pipeline hiện dùng
ba split độc lập trong `data_split/`; không còn lấy một phần train làm test.

Một pipeline unified MobileNetV3-Small đã được thêm song song để thay
6 ONNX bằng 1 ONNX ba output. Pipeline legacy vẫn là mặc định và đã được
snapshot để so sánh/rollback. Xem [UNIFIED.md](UNIFIED.md).

## Trạng thái hiện tại

- Dataset sạch: 94.449 ảnh.
- Train: 66.115 ảnh; validation: 14.172 ảnh; test: 14.162 ảnh.
- Exact duplicate cùng nhãn chỉ giữ một bản.
- File SHA-256 giống nhau nhưng xung đột nhãn bị loại toàn bộ.
- pHash cluster không bị tách qua các split.
- Các cluster quá lớn đối với validation/test được ép vào train.
- Expert `gia_cam` và `gia_suc` có 3 lớp; `cu`, `qua`, `rau` có 2 lớp.

Các model đang có trong `models/` được tạo trước split sạch hiện tại. Cần train
lại router và 5 expert trước khi dùng metric trên `data_split/test` làm kết quả
chính thức.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quy trình chuẩn

Kiểm tra hoặc tạo lại split:

```powershell
python scripts\resplit_dataset.py --threshold 0 --dry-run
python scripts\resplit_dataset.py --threshold 0
```

Thống kê số lượng ảnh mà không mở nội dung ảnh:

```powershell
python scripts\dataset_stats.py
python scripts\dataset_stats.py --products
python scripts\dataset_stats.py --csv reports\dataset_stats.csv
```

Train:

```powershell
python train.py router
python train.py freshness --category cu
python train.py freshness --category gia_cam
python train.py freshness --category gia_suc
python train.py freshness --category qua
python train.py freshness --category rau
```

Dự đoán bằng checkpoint PyTorch:

```powershell
python predict.py path\to\image.jpg
```

Đánh giá trên test độc lập:

```powershell
python evaluate_router_uncertainty.py
python evaluate_end_to_end.py
```

Export và kiểm tra ONNX:

```powershell
python export_onnx.py
python verify_onnx.py --image path\to\image.jpg
python package_models.py --require-onnx
```

Chạy kiểm thử:

```powershell
python -m unittest -v test_regression.py
```

Có thể dùng `python main.py` nếu muốn thao tác qua menu.

Unified positive baseline:

```powershell
python scripts\build_unified_manifest.py
python train_unified.py
python evaluate_unified.py
python export_unified_onnx.py
```

## Tài liệu

- [DATA.md](DATA.md): cấu trúc, số lượng và cách tạo split.
- [TRAIN.md](TRAIN.md): huấn luyện, resume và output.
- [ARCHITECTURE.md](ARCHITECTURE.md): kiến trúc router–expert.
- [ALGORITHM.md](ALGORITHM.md): thuật toán dữ liệu, train và inference.
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md): quy tắc so sánh model.
- [PROJECT_FILES.md](PROJECT_FILES.md): vai trò từng file còn lại.
- [UNIFIED.md](UNIFIED.md): pipeline một model chạy song song với legacy.

Kết quả chỉ là ước lượng từ hình ảnh, không thay thế kiểm nghiệm an toàn thực phẩm.
